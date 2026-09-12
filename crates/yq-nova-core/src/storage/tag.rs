use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

use crate::{
    error::{NovaError, NovaResult},
    storage::{Database, Repository, memory},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRecord {
    pub id: i64,
    pub name: String,
    pub color: Option<String>,
    pub created_at: DateTime<Utc>,

    pub memory_count: i64,
}

#[async_trait]
pub trait TagRepository: Repository<TagRecord> {
    async fn attach_tags(
        &self,
        db: &Database,
        memory_uuid: Uuid,
        tags: &[String],
    ) -> NovaResult<()>;

    async fn detach_tags(
        &self,
        db: &Database,
        memory_uuid: Uuid,
        tags: &[String],
    ) -> NovaResult<()>;

    async fn replace_tags(
        &self,
        db: &Database,
        memory_uuid: Uuid,
        new_tags: &[String],
    ) -> NovaResult<()>;

    async fn list_tags_of_memory(
        &self,
        db: &Database,
        memory_uuid: Uuid,
    ) -> NovaResult<Vec<String>>;

    async fn list_all_tags(
        &self,
        db: &Database,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<TagRecord>>;

    async fn get_tag_by_name(&self, db: &Database, name: &str) -> NovaResult<Option<TagRecord>>;

    async fn count_all_tags(&self, db: &Database) -> NovaResult<i64>;

    async fn rename_tag(&self, db: &Database, name: &str, new_name: &str) -> NovaResult<()>;

    async fn delete_tag(&self, db: &Database, name: &str) -> NovaResult<i64>;
}

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
        memory::attach_tags(&db.pool, memory_uuid, new_tags).await?;

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
             FROM tags t LEFT JOIN memory_tags mt ON mt.tag_id = t.id GROUP BY t.id, t.name, \
             t.color, t.created_at ORDER BY memory_count DESC, t.name ASC LIMIT ? OFFSET ?",
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
             FROM tags t LEFT JOIN memory_tags mt ON mt.tag_id = t.id WHERE t.name = ?1 GROUP BY \
             t.id, t.name, t.color, t.created_at",
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

    async fn count_all_tags(&self, db: &Database) -> NovaResult<i64> {
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tags")
            .fetch_one(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        Ok(n)
    }

    async fn rename_tag(&self, db: &Database, name: &str, new_name: &str) -> NovaResult<()> {
        let target = new_name.trim();
        if target.is_empty() {
            return Err(NovaError::validation("tag.new_name must not be empty"));
        }
        if target == name {
            return Ok(());
        }
        let mut tx = db.begin().await?;
        let res = sqlx::query("UPDATE tags SET name = ?1 WHERE name = ?2")
            .bind(target)
            .bind(name)
            .execute(&mut *tx)
            .await
            .map_err(NovaError::from)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("tag {name}")));
        }
        tx.commit().await.map_err(NovaError::from)?;
        Ok(())
    }

    async fn delete_tag(&self, db: &Database, name: &str) -> NovaResult<i64> {
        let affected: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_tags WHERE tag_id = (SELECT id FROM tags WHERE name = ?1)",
        )
        .bind(name)
        .fetch_one(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        let res = sqlx::query("DELETE FROM tags WHERE name = ?1")
            .bind(name)
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("tag {name}")));
        }
        Ok(affected)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::StorageConfig,
        storage::memory::{InsertMemoryInput, MemoryRepository, SqliteMemoryRepository},
    };

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
                InsertMemoryInput {
                    content: text,
                    tags: &tag_list,
                    ..Default::default()
                },
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

    #[tokio::test]
    async fn rename_and_delete_tag_keep_associations_consistent() {
        let db = temp_db().await;
        let tags = SqliteTagRepository::new();
        let mem = SqliteMemoryRepository::new();
        for (text, tag_list) in [
            ("m1", vec!["legacy".to_string(), "keep".to_string()]),
            ("m2", vec!["legacy".to_string()]),
            ("m3", vec!["keep".to_string()]),
        ] {
            mem.insert(
                &db,
                InsertMemoryInput {
                    content: text,
                    tags: &tag_list,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        }

        assert_eq!(tags.count_all_tags(&db).await.unwrap(), 2);

        tags.rename_tag(&db, "legacy", "archive").await.unwrap();
        let renamed = tags.get_tag_by_name(&db, "archive").await.unwrap().unwrap();
        assert_eq!(renamed.memory_count, 2);
        assert!(tags.get_tag_by_name(&db, "legacy").await.unwrap().is_none());

        let conflict = tags.rename_tag(&db, "keep", "archive").await.unwrap_err();
        assert_eq!(conflict.code(), crate::error::ErrorCode::Conflict);

        let affected = tags.delete_tag(&db, "archive").await.unwrap();
        assert_eq!(affected, 2);
        assert_eq!(tags.count_all_tags(&db).await.unwrap(), 1);

        let gone = tags.delete_tag(&db, "archive").await.unwrap_err();
        assert_eq!(gone.code(), crate::error::ErrorCode::NotFound);
    }
}
