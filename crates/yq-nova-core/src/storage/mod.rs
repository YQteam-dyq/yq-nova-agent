//! Storage layer: SQLite pool + per-domain repositories.
//!
//! Implemented in M2. This placeholder exposes the traits and types we
//! know we'll need so other crates can already depend on them without
//! waiting for the full implementation.

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

// -----------------------------------------------------------------------------
// Connection pool factory
// -----------------------------------------------------------------------------

/// 封装 `SqlitePool` + 运行期配置的数据库句柄。
///
/// 这是所有仓储对象共享的唯一连接池来源，Clone 成本低（共享 Arc 内部配置）。
#[derive(Clone)]
pub struct Database {
    /// sqlx 连接池。
    pub pool: SqlitePool,
    /// 存储层运行时配置（Arc 共享）。
    pub config: Arc<StorageConfig>,
}

impl Database {
    /// 打开（若文件不存在则创建）SQLite 数据库，应用所有 PRAGMA 调优参数
    /// 与未执行的迁移脚本。若 db_path 的父目录不存在会尝试创建。
    pub async fn open(config: StorageConfig) -> NovaResult<Self> {
        Self::ensure_parent_dir(&config.db_path)?;

        // `create=true` is critical: sqlx SQLite (sqlx 0.7 with rusqlite/
        // sqlite3 backends) does NOT auto-create a file unless explicitly
        // told to; otherwise we get SQLITE_CANTOPEN (14).
        let db_url = format!("sqlite://{}?mode=rwc", config.db_path.display());
        let pool = SqlitePoolOptions::new()
            .max_connections(config.pool_max_connections)
            .min_connections(config.pool_min_connections)
            .acquire_timeout(std::time::Duration::from_millis(config.busy_timeout_ms as u64))
            .connect(&db_url)
            .await
            .map_err(NovaError::storage)?;

        // Apply PRAGMAs on every acquired connection.
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
            // wal_autocheckpoint is measured in *pages*. We accept KB from the
            // config user, so divide by (page_size/1024). If page_size is 0
            // (use default), assume 4096.
            match config.page_size.max(4096).saturating_div(1024).max(1) {
                kb_per_page if config.wal_autocheckpoint_kb > 0 => {
                    config.wal_autocheckpoint_kb.saturating_div(kb_per_page).max(1)
                },
                _ => 0,
            },
            // journal_size_limit is *bytes*; convert from KB. 0 disables.
            if config.journal_size_limit_kb > 0 {
                config.journal_size_limit_kb * 1024
            } else {
                // SQLite wants -1 for "no limit", not 0, for this pragma.
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

        // Apply migrations (see storage/migration.rs — M2 fills this in).
        migration::Migrator::run(&pool).await?;

        Ok(Self { pool, config: Arc::new(config) })
    }

    /// 开启一个新的 SQLite 事务。
    pub async fn begin(&self) -> NovaResult<Transaction<'_, Sqlite>> {
        self.pool.begin().await.map_err(NovaError::storage)
    }

    /// 优雅关闭连接池：先做一次 TRUNCATE checkpoint 将 WAL 回写到主文件，
    /// 再关闭 pool。关闭后再调用将返回错误。
    pub async fn close(self) -> NovaResult<()> {
        // Flush WAL back into the main DB file before closing.
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

    /// 当前数据库文件的磁盘占用（字节），包含主文件 + WAL + SHM 三部分。
    /// 用于 `GET /v1/stats` 与 `GET /v1/health` 等端点统计。
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

// ---------------------------------------------------------------------------
// SQLite PRAGMA helpers
// ---------------------------------------------------------------------------

/// Convert the user-facing `storage.synchronous` string ("full" / "normal" /
/// "off" / "0" … "3") into the literal SQLite accepts for the PRAGMA.
fn synchronous_pragma_value(s: &str) -> String {
    let lowered = s.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "off" | "0" => "0".into(),
        "normal" | "1" => "1".into(),
        "full" | "2" => "2".into(),
        "extra" | "3" => "3".into(),
        other => {
            // Config::validate already rejected non-standard values; fall
            // back to "normal" so runtime behavior is never undefined.
            tracing::warn!(value = other, "unexpected storage.synchronous; falling back to NORMAL");
            "1".into()
        },
    }
}

// -----------------------------------------------------------------------------
// Shared enums used across repositories.
// -----------------------------------------------------------------------------

/// 记忆条目的生命周期状态。与 `migrations/001_init.sql` 中的
/// `memory_items.status` 列枚举保持一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    /// 活跃：正常出现在 recall/list 结果中。
    Active,
    /// 已归档：默认不出现，但可通过过滤器显式查询（审计/回溯）。
    Archived,
    /// 已过期：TTL 到期后设置的状态。
    Expired,
    /// 已删除：软删除状态，默认不出现。
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

/// 记忆来源。用于影响遗忘权重、过滤与 UI 分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemorySource {
    /// Agent 自身产生的思考/规划（默认）。
    #[default]
    Agent,
    /// 用户显式输入的偏好、指令或事实。
    User,
    /// 系统级元信息（启动、配置变更、策略等）。
    System,
    /// 工具调用返回的结果或观测。
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

// -----------------------------------------------------------------------------
// Repository trait (placeholder — M2 implements).
// -----------------------------------------------------------------------------

/// Unified repository interface. `Database` holds a pool, and each repo
/// impl takes `&Database` in every method so we can add transactional
/// variants later without changing callers.
#[async_trait]
pub trait Repository<T>: Send + Sync {
    /// Unique repository name, used in logs and error contexts.
    fn name(&self) -> &'static str;
}

// -----------------------------------------------------------------------------
// MemoryFilter — query predicate for `list` / `count`.
// -----------------------------------------------------------------------------

/// 记忆查询/过滤谓词。用于 `list`、`count`、`forget(Filter)` 与 `recall` 的后置过滤。
/// 所有字段均为 Option：None 表示不施加该条件；多个条件为 AND 关系。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryFilter {
    /// 若存在，则仅保留来源在此列表中的记忆。
    pub source_in: Option<Vec<MemorySource>>,
    /// 仅返回严格晚于此时间戳创建的记忆。
    pub created_after: Option<chrono::DateTime<chrono::Utc>>,
    /// 仅返回严格早于此时间戳创建的记忆。
    pub created_before: Option<chrono::DateTime<chrono::Utc>>,
    /// 最小 importance（含），合法范围 [0.0, 1.0]。
    pub importance_min: Option<f32>,
    /// 最大 importance（含），合法范围 [0.0, 1.0]。遗忘 GC 任务用它来挑「低价值」行。
    pub importance_max: Option<f32>,
    /// 仅返回状态在此列表中的行。
    pub status_in: Option<Vec<MemoryStatus>>,
    /// 仅返回 `access_count` 严格小于该值的行。
    pub access_count_lt: Option<i64>,
    /// 仅返回最后访问时间（若从未访问则用 `created_at`）严格早于该时刻的行。
    /// 用于陈旧性 GC。
    pub last_accessed_before: Option<chrono::DateTime<chrono::Utc>>,
    /// 仅返回最后访问时间（若从未访问则用 `created_at`）严格晚于该时刻的行。
    pub last_accessed_after: Option<chrono::DateTime<chrono::Utc>>,
    /// 仅返回同时包含**全部**给定标签的行。
    pub tags_all: Option<Vec<String>>,
    /// 仅返回至少包含**任一**给定标签的行。
    pub tags_any: Option<Vec<String>>,
    /// 保留字段：`metadata_json` 顶层键值对必须全部匹配。
    /// 调用方也可自行在返回的行上做后置过滤。
    pub metadata_match: Option<Vec<(String, serde_json::Value)>>,
}

// ---------------------------------------------------------------------------
// Re-exports: traits + concrete impls so downstream can use them easily.
// ---------------------------------------------------------------------------

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
