
use std::{path::Path, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sqlx::{Sqlite, SqlitePool, Transaction, sqlite::SqlitePoolOptions};

use crate::{
    config::StorageConfig,
    error::{NovaError, NovaResult},
};

pub mod entity;
pub mod fts5;
pub mod memory;
pub mod migration;
pub mod relation;
pub mod tag;
pub mod vector;
#[cfg(feature = "sqlite-vec")]
pub mod vector_vec;

#[derive(Clone)]
pub struct Database {

    pub pool: SqlitePool,

    pub config: Arc<StorageConfig>,
}

impl Database {

    pub async fn open(config: StorageConfig) -> NovaResult<Self> {
        Self::ensure_parent_dir(&config.db_path)?;

        let db_url = format!("sqlite://{}?mode=rwc", config.db_path.display());
        let pool = SqlitePoolOptions::new()
            .max_connections(config.pool_max_connections)
            .min_connections(config.pool_min_connections)
            .acquire_timeout(std::time::Duration::from_millis(config.busy_timeout_ms as u64))
            .connect(&db_url)
            .await
            .map_err(NovaError::storage)?;

        let sync_lit = synchronous_pragma_value(&config.synchronous);
        sqlx::query(&format!(
            "
            PRAGMA journal_mode = {};
            PRAGMA synchronous = {};
            PRAGMA busy_timeout = {};
            PRAGMA cache_size = -{};   -- negative => KiB
            PRAGMA temp_store = MEMORY;
            PRAGMA foreign_keys = ON;
            PRAGMA wal_autocheckpoint = {};  -- pages; page_size=4K => N*4K thresholds
            PRAGMA journal_size_limit = {};  -- bytes
            PRAGMA mmap_size = {};           -- bytes; 0 disables
            PRAGMA soft_heap_limit = {};     -- bytes; 0 disables
            ",
            if config.wal_mode { "WAL" } else { "DELETE" },
            sync_lit,
            config.busy_timeout_ms,
            config.cache_size_kb,

            match config.page_size.max(4096).saturating_div(1024).max(1) {
                kb_per_page if config.wal_autocheckpoint_kb > 0 => {
                    config.wal_autocheckpoint_kb.saturating_div(kb_per_page).max(1)
                },
                _ => 0,
            },

            if config.journal_size_limit_kb > 0 {
                config.journal_size_limit_kb * 1024
            } else {

                -1
            },
            if config.mmap_size_kb > 0 { config.mmap_size_kb * 1024 } else { 0 },
            if config.soft_heap_limit_kb > 0 { config.soft_heap_limit_kb * 1024 } else { 0 },
        ))
        .execute(&pool)
        .await
        .map_err(NovaError::storage)?;

        if config.page_size > 0 {
            sqlx::query(&format!("PRAGMA page_size = {};", config.page_size))
                .execute(&pool)
                .await
                .map_err(NovaError::storage)?;
        }

        migration::Migrator::run(&pool).await?;

        Ok(Self { pool, config: Arc::new(config) })
    }

    pub async fn begin(&self) -> NovaResult<Transaction<'_, Sqlite>> {
        self.pool.begin().await.map_err(NovaError::storage)
    }

    pub async fn close(self) -> NovaResult<()> {

        let _ = sqlx::query("PRAGMA wal_checkpoint(TRUNCATE);").execute(&self.pool).await;
        self.pool.close().await;
        Ok(())
    }

    fn ensure_parent_dir(path: &Path) -> NovaResult<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    NovaError::storage_msg(format!("create db dir {}: {e}", parent.display()))
                })?;
            }
        }
        Ok(())
    }

    pub fn size_on_disk_bytes(&self) -> NovaResult<u64> {
        let main = std::fs::metadata(&self.config.db_path).map(|m| m.len()).unwrap_or(0);
        let wal = std::fs::metadata(self.config.db_path.with_extension("db-wal"))
            .map(|m| m.len())
            .unwrap_or(0);
        let shm = std::fs::metadata(self.config.db_path.with_extension("db-shm"))
            .map(|m| m.len())
            .unwrap_or(0);
        Ok(main + wal + shm)
    }
}

fn synchronous_pragma_value(s: &str) -> String {
    let lowered = s.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "off" | "0" => "0".into(),
        "normal" | "1" => "1".into(),
        "full" | "2" => "2".into(),
        "extra" | "3" => "3".into(),
        other => {

            tracing::warn!(value = other, "unexpected storage.synchronous; falling back to NORMAL");
            "1".into()
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {

    Active,

    Archived,

    Expired,

    Deleted,
}

impl MemoryStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryStatus::Active => "active",
            MemoryStatus::Archived => "archived",
            MemoryStatus::Expired => "expired",
            MemoryStatus::Deleted => "deleted",
        }
    }
}

impl std::fmt::Display for MemoryStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TryFrom<&str> for MemoryStatus {
    type Error = NovaError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(match value {
            "active" => MemoryStatus::Active,
            "archived" => MemoryStatus::Archived,
            "expired" => MemoryStatus::Expired,
            "deleted" => MemoryStatus::Deleted,
            other => return Err(NovaError::validation(format!("unknown status: {other}"))),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemorySource {

    #[default]
    Agent,

    User,

    System,

    Tool,
}

impl MemorySource {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemorySource::Agent => "agent",
            MemorySource::User => "user",
            MemorySource::System => "system",
            MemorySource::Tool => "tool",
        }
    }
}

impl std::fmt::Display for MemorySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TryFrom<&str> for MemorySource {
    type Error = NovaError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(match value {
            "agent" => MemorySource::Agent,
            "user" => MemorySource::User,
            "system" => MemorySource::System,
            "tool" => MemorySource::Tool,
            other => return Err(NovaError::validation(format!("unknown source: {other}"))),
        })
    }
}

#[async_trait]
pub trait Repository<T>: Send + Sync {

    fn name(&self) -> &'static str;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryFilter {

    pub source_in: Option<Vec<MemorySource>>,

    pub created_after: Option<chrono::DateTime<chrono::Utc>>,

    pub created_before: Option<chrono::DateTime<chrono::Utc>>,

    pub importance_min: Option<f32>,

    pub importance_max: Option<f32>,

    pub status_in: Option<Vec<MemoryStatus>>,

    pub access_count_lt: Option<i64>,

    pub last_accessed_before: Option<chrono::DateTime<chrono::Utc>>,

    pub last_accessed_after: Option<chrono::DateTime<chrono::Utc>>,

    pub tags_all: Option<Vec<String>>,

    pub tags_any: Option<Vec<String>>,

    pub metadata_match: Option<Vec<(String, serde_json::Value)>>,
}

pub use entity::{
    Direction, EntityRecord, EntityRepository, SqliteEntityRepository, TraverseNode,
    UpsertEntityInput, UpsertOutcome,
};
pub use memory::{
    InsertMemoryInput, InsertOutcome, MemoryRecord, MemoryRepository, SqliteMemoryRepository,
};
pub use migration::Migrator;
pub use relation::{
    InsertRelationInput, InsertRelationOutcome, RelationRecord, RelationRepository,
    SqliteRelationRepository,
};
pub use tag::{SqliteTagRepository, TagRecord, TagRepository};
pub use vector::{SqliteVectorStore, VectorHit, VectorStore};
#[cfg(feature = "sqlite-vec")]
pub use vector_vec::SqliteVecVectorStore;
