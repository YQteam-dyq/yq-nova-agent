//! Tag repository.
//!
//! Tags are short string labels attached to memory items. The underlying
//! tables (`tags` / `memory_tags`) are already populated by
//! `memory::attach_tags` / `memory::detach_tags` during memory CRUD; this
//! module wraps those low-level helpers in a repository trait so higher
//! layers never reach into memory.rs internals. It also exposes listing
//! operations that span memories: "show me all tags used so far", "which
//! memories carry tag X?", etc.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

use crate::{
    error::{NovaError, NovaResult},
    storage::{Database, Repository, memory},
};

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRecord {
    pub id: i64,
    pub name: String,
    pub color: Option<String>,
    pub created_at: DateTime<Utc>,
    /// Count of memories currently carrying this tag (populated by `list_all`
    /// and `get_tag`; set to 0 for raw row decodes).
    pub memory_count: i64,
}

// -----------------------------------------------------------------------------
// Trait
// -----------------------------------------------------------------------------

#[async_trait]
pub trait TagRepository: Repository<TagRecord> {
    /// Add tags to a memory. Creates tag rows lazily if they don't exist yet.
    /// Duplicate tags on the same memory are silently ignored thanks to the
    /// `memory_tags` composite PK.
    async fn attach_tags(
        &self,
        db: &Database,
        memory_uuid: Uuid,
        tags: &[String],
    ) -> NovaResult<()>;

    /// Remove specific tags from a memory. No-op for tags that are absent.
    async fn detach_tags(
        &self,
        db: &Database,
        memory_uuid: Uuid,
        tags: &[String],
    ) -> NovaResult<()>;

    /// Replace the full tag set of a memory: tags not in `new_tags` are
    /// detached, tags in `new_tags` are attached. Empty vector clears all.
    async fn replace_tags(
        &self,
        db: &Database,
        memory_uuid: Uuid,
        new_tags: &[String],
    ) -> NovaResult<()>;

    /// List all tags currently attached to `memory_uuid`.
    async fn list_tags_of_memory(
        &self,
        db: &Database,
        memory_uuid: Uuid,
    ) -> NovaResult<Vec<String>>;

    /// List all tags known to the system, each annotated with how many
    /// memories carry it.
    async fn list_all_tags(
        &self,
        db: &Database,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<TagRecord>>;

    /// Get a single tag by name, or None.
    async fn get_tag_by_name(&self, db: &Database, name: &str) -> NovaResult<Option<TagRecord>>;
}

// -----------------------------------------------------------------------------
// Sqlite impl
// -----------------------------------------------------------------------------

#[derive(Clone)]
pub struct SqliteTagRepository;

impl SqliteTagRepository {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for SqliteTagRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Repository<TagRecord> for SqliteTagRepository {
    fn name(&self) -> &'static str {
        "tag.sqlite"
    }
}

#[async_trait]
impl TagRepository for SqliteTagRepository {
    async fn attach_tags(
        &self,
        db: &Database,
        memory_uuid: Uuid,
        tags: &[String],
    ) -> NovaResult<()> {
        memory::attach_tags(&db.pool, memory_uuid, tags).await
    }

    async fn detach_tags(
        &self,
        db: &Database,
        memory_uuid: Uuid,
        tags: &[String],
    ) -> NovaResult<()> {
        memory::detach_tags(&db.pool, memory_uuid, tags).await
    }

    async fn replace_tags(
        &self,
        db: &Database,
        memory_uuid: Uuid,
        new_tags: &[String],
    ) -> NovaResult<()> {
        // 1. attach all new tags first (idempotent)
        memory::attach_tags(&db.pool, memory_uuid, new_tags).await?;
        // 2. compute the set to detach: current tags - new_tags
        let current = memory::list_tags_of_memory(&db.pool, memory_uuid).await?;
        let new_set: std::collections::HashSet<&str> =
            new_tags.iter().map(|s| s.as_str()).collect();
        let to_remove: Vec<String> =
            current.into_iter().filter(|t| !new_set.contains(t.as_str())).collect();
        if !to_remove.is_empty() {
            memory::detach_tags(&db.pool, memory_uuid, &to_remove).await?;
        }
        Ok(())
    }

    async fn list_tags_of_memory(
        &self,
        db: &Database,
        memory_uuid: Uuid,
    ) -> NovaResult<Vec<String>> {
        memory::list_tags_of_memory(&db.pool, memory_uuid).await
    }

    async fn list_all_tags(
        &self,
        db: &Database,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<TagRecord>> {
        let limit = limit.min(10_000) as i64;
        let offset = offset as i64;
        let rows = sqlx::query(
            "SELECT t.id, t.name, t.color, t.created_at, COUNT(mt.memory_uuid) AS memory_count \
             FROM tags t \
             LEFT JOIN memory_tags mt ON mt.tag_id = t.id \
             GROUP BY t.id, t.name, t.color, t.created_at \
             ORDER BY memory_count DESC, t.name ASC \
             LIMIT ? OFFSET ?",
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(row_to_tag(&row)?);
        }
        Ok(out)
    }

    async fn get_tag_by_name(&self, db: &Database, name: &str) -> NovaResult<Option<TagRecord>> {
        let row = sqlx::query(
            "SELECT t.id, t.name, t.color, t.created_at, COUNT(mt.memory_uuid) AS memory_count \
             FROM tags t \
             LEFT JOIN memory_tags mt ON mt.tag_id = t.id \
             WHERE t.name = ?1 \
             GROUP BY t.id, t.name, t.color, t.created_at",
        )
        .bind(name)
        .fetch_optional(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        match row {
            Some(r) => Ok(Some(row_to_tag(&r)?)),
            None => Ok(None),
        }
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn row_to_tag(row: &sqlx::sqlite::SqliteRow) -> NovaResult<TagRecord> {
    let id: i64 = row.try_get("id").map_err(NovaError::storage)?;
    let name: String = row.try_get("name").map_err(NovaError::storage)?;
    let color: Option<String> = row.try_get("color").map_err(NovaError::storage)?;
    let created_ts: i64 = row.try_get("created_at").map_err(NovaError::storage)?;
    let memory_count: i64 = row.try_get("memory_count").map_err(NovaError::storage)?;
    Ok(TagRecord {
        id,
        name,
        color,
        created_at: DateTime::from_timestamp(created_ts, 0)
            .ok_or_else(|| NovaError::storage_msg(format!("bad ts {created_ts}")))?,
        memory_count,
    })
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageConfig;
    use crate::storage::memory::{InsertMemoryInput, MemoryRepository, SqliteMemoryRepository};

    async fn temp_db() -> Database {
        let dir = std::env::temp_dir().join(format!("yq-nova-m2-tag-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        Database::open(cfg).await.expect("open temp db")
    }

    #[tokio::test]
    async fn attach_detach_replace_and_list_tags() {
        let db = temp_db().await;
        let tags = SqliteTagRepository::new();
        let mem = SqliteMemoryRepository::new();
        let muuid = mem
            .insert(
                &db,
                InsertMemoryInput {
                    content: "hello",
                    tags: &["a".into(), "b".into()],
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .uuid();

        let got = tags.list_tags_of_memory(&db, muuid).await.unwrap();
        assert_eq!(got.len(), 2);

        tags.detach_tags(&db, muuid, &["a".into()]).await.unwrap();
        let got = tags.list_tags_of_memory(&db, muuid).await.unwrap();
        assert_eq!(got, vec!["b"]);

        tags.replace_tags(&db, muuid, &["c".into(), "d".into()]).await.unwrap();
        let mut got = tags.list_tags_of_memory(&db, muuid).await.unwrap();
        got.sort();
        assert_eq!(got, vec!["c", "d"]);
    }

    #[tokio::test]
    async fn list_all_tags_reports_counts() {
        let db = temp_db().await;
        let tags = SqliteTagRepository::new();
        let mem = SqliteMemoryRepository::new();
        for (text, tag_list) in [
            ("m1", vec!["shared".to_string(), "one".to_string()]),
            ("m2", vec!["shared".to_string(), "two".to_string()]),
        ] {
            mem.insert(
                &db,
                InsertMemoryInput { content: text, tags: &tag_list, ..Default::default() },
            )
            .await
            .unwrap();
        }
        let all = tags.list_all_tags(&db, 100, 0).await.unwrap();
        let by_name: std::collections::HashMap<String, i64> =
            all.iter().map(|t| (t.name.clone(), t.memory_count)).collect();
        assert_eq!(by_name.get("shared"), Some(&2));
        assert_eq!(by_name.get("one"), Some(&1));
        assert_eq!(by_name.get("two"), Some(&1));
    }
}
