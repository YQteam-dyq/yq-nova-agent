//! Memory repository backed by `memory_items` + `memory_tags` + `tags`.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{Row, SqlitePool, sqlite::SqliteRow};
use uuid::Uuid;

use crate::{
    error::{NovaError, NovaResult},
    storage::{Database, MemoryFilter, MemorySource, MemoryStatus, Repository},
};

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: i64,
    pub uuid: Uuid,
    pub content: String,
    pub content_hash: String,
    pub metadata: serde_json::Value,
    pub source: MemorySource,
    pub importance: f32,
    pub access_count: i64,
    pub last_accessed: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub status: MemoryStatus,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct InsertMemoryInput<'a> {
    pub content: &'a str,
    pub source: MemorySource,
    pub importance: f32,
    pub metadata: Option<&'a serde_json::Value>,
    pub expires_at: Option<DateTime<Utc>>,
    pub tags: &'a [String],
}

impl<'a> Default for InsertMemoryInput<'a> {
    fn default() -> Self {
        Self {
            content: "",
            source: MemorySource::Agent,
            importance: 0.5,
            metadata: None,
            expires_at: None,
            tags: &[],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome<T = ()> {
    Inserted(Uuid),
    Duplicate(Uuid, T),
}

impl<T> InsertOutcome<T> {
    pub fn uuid(&self) -> Uuid {
        match self {
            InsertOutcome::Inserted(u) | InsertOutcome::Duplicate(u, _) => *u,
        }
    }

    pub fn is_duplicate(&self) -> bool {
        matches!(self, InsertOutcome::Duplicate(_, _))
    }
}

#[derive(Clone)]
enum BindValue {
    Text(String),
    Int(i64),
    Real(f64),
}

// -----------------------------------------------------------------------------
// Trait
// -----------------------------------------------------------------------------

#[async_trait]
pub trait MemoryRepository: Repository<MemoryRecord> {
    async fn insert(
        &self,
        db: &Database,
        input: InsertMemoryInput<'_>,
    ) -> NovaResult<InsertOutcome>;
    async fn get_by_uuid(&self, db: &Database, uuid: Uuid) -> NovaResult<MemoryRecord>;
    async fn update_status(
        &self,
        db: &Database,
        uuid: Uuid,
        status: MemoryStatus,
    ) -> NovaResult<()>;
    async fn update_metadata(
        &self,
        db: &Database,
        uuid: Uuid,
        metadata: &serde_json::Value,
    ) -> NovaResult<()>;
    async fn update_importance(&self, db: &Database, uuid: Uuid, importance: f32)
    -> NovaResult<()>;
    async fn delete(&self, db: &Database, uuid: Uuid) -> NovaResult<()>;
    async fn mark_accessed(&self, db: &Database, uuid: Uuid) -> NovaResult<()>;
    async fn list(
        &self,
        db: &Database,
        filter: &MemoryFilter,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<MemoryRecord>>;
    async fn count(&self, db: &Database, filter: &MemoryFilter) -> NovaResult<i64>;
}

// -----------------------------------------------------------------------------
// Sqlite impl
// -----------------------------------------------------------------------------

#[derive(Clone)]
pub struct SqliteMemoryRepository;

impl SqliteMemoryRepository {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for SqliteMemoryRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Repository<MemoryRecord> for SqliteMemoryRepository {
    fn name(&self) -> &'static str {
        "memory.sqlite"
    }
}

#[async_trait]
impl MemoryRepository for SqliteMemoryRepository {
    async fn insert(
        &self,
        db: &Database,
        input: InsertMemoryInput<'_>,
    ) -> NovaResult<InsertOutcome> {
        if input.content.is_empty() {
            return Err(NovaError::validation("memory.content must not be empty"));
        }
        if !(0.0..=1.0).contains(&input.importance) {
            return Err(NovaError::validation("memory.importance must be in [0.0, 1.0]"));
        }

        let content_hash = sha256_hex(input.content);
        let pool = &db.pool;

        let existing: Option<(i64, String)> = sqlx::query_as(
            "SELECT id, uuid FROM memory_items WHERE content_hash = ?1 AND status != 'deleted'",
        )
        .bind(&content_hash)
        .fetch_optional(pool)
        .await
        .map_err(NovaError::storage)?;

        if let Some((_id, existing_uuid_s)) = existing {
            let existing_uuid = Uuid::parse_str(&existing_uuid_s)
                .map_err(|e| NovaError::storage_msg(format!("bad existing uuid: {e}")))?;
            if !input.tags.is_empty() {
                attach_tags(pool, existing_uuid, input.tags).await?;
            }
            return Ok(InsertOutcome::Duplicate(existing_uuid, ()));
        }

        let uuid = Uuid::new_v4();
        let metadata_json = input.metadata.cloned().unwrap_or_else(|| serde_json::json!({}));
        let now = Utc::now().timestamp();
        let expires_ts = input.expires_at.map(|t| t.timestamp());

        sqlx::query(
            "INSERT INTO memory_items (
                uuid, content, content_hash, metadata_json, source, importance,
                access_count, last_accessed, created_at, expires_at, status
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        )
        .bind(uuid.to_string())
        .bind(input.content)
        .bind(&content_hash)
        .bind(metadata_json.to_string())
        .bind(input.source.as_str())
        .bind(input.importance as f64)
        .bind(0_i64)
        .bind::<Option<i64>>(None)
        .bind(now)
        .bind(expires_ts)
        .bind(MemoryStatus::Active.as_str())
        .execute(pool)
        .await
        .map_err(NovaError::storage)?;

        if !input.tags.is_empty() {
            attach_tags(pool, uuid, input.tags).await?;
        }

        Ok(InsertOutcome::Inserted(uuid))
    }

    async fn get_by_uuid(&self, db: &Database, uuid: Uuid) -> NovaResult<MemoryRecord> {
        let row = sqlx::query(
            "SELECT id, uuid, content, content_hash, metadata_json, source, importance,
                    access_count, last_accessed, created_at, expires_at, status
             FROM memory_items WHERE uuid = ?1",
        )
        .bind(uuid.to_string())
        .fetch_optional(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        let row = row.ok_or_else(|| NovaError::not_found(format!("memory {uuid}")))?;
        let mut rec = row_to_memory(&row)?;
        rec.tags = load_tags_for(&db.pool, uuid).await?;
        Ok(rec)
    }

    async fn update_status(
        &self,
        db: &Database,
        uuid: Uuid,
        status: MemoryStatus,
    ) -> NovaResult<()> {
        let res = sqlx::query("UPDATE memory_items SET status = ?1 WHERE uuid = ?2")
            .bind(status.as_str())
            .bind(uuid.to_string())
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("memory {uuid}")));
        }
        Ok(())
    }

    async fn update_metadata(
        &self,
        db: &Database,
        uuid: Uuid,
        metadata: &serde_json::Value,
    ) -> NovaResult<()> {
        let res = sqlx::query("UPDATE memory_items SET metadata_json = ?1 WHERE uuid = ?2")
            .bind(metadata.to_string())
            .bind(uuid.to_string())
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("memory {uuid}")));
        }
        Ok(())
    }

    async fn update_importance(
        &self,
        db: &Database,
        uuid: Uuid,
        importance: f32,
    ) -> NovaResult<()> {
        if !(0.0..=1.0).contains(&importance) {
            return Err(NovaError::validation("memory.importance must be in [0.0, 1.0]"));
        }
        let res = sqlx::query("UPDATE memory_items SET importance = ?1 WHERE uuid = ?2")
            .bind(importance as f64)
            .bind(uuid.to_string())
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("memory {uuid}")));
        }
        Ok(())
    }

    async fn delete(&self, db: &Database, uuid: Uuid) -> NovaResult<()> {
        let res = sqlx::query("DELETE FROM memory_items WHERE uuid = ?1")
            .bind(uuid.to_string())
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("memory {uuid}")));
        }
        Ok(())
    }

    async fn mark_accessed(&self, db: &Database, uuid: Uuid) -> NovaResult<()> {
        let now = Utc::now().timestamp();
        sqlx::query(
            "UPDATE memory_items
                SET access_count = access_count + 1,
                    last_accessed = ?1
              WHERE uuid = ?2",
        )
        .bind(now)
        .bind(uuid.to_string())
        .execute(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        Ok(())
    }

    async fn list(
        &self,
        db: &Database,
        filter: &MemoryFilter,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<MemoryRecord>> {
        let built = build_filter(filter, false);
        let limit = limit.min(10_000) as i64;
        let offset = offset as i64;
        let sql = format!("{} ORDER BY m.created_at DESC LIMIT ? OFFSET ?", built.sql);

        let mut q = sqlx::query(&sql);
        for b in &built.binds {
            q = match b {
                BindValue::Text(s) => q.bind(s),
                BindValue::Int(i) => q.bind(i),
                BindValue::Real(f) => q.bind(f),
            };
        }
        let rows =
            q.bind(limit).bind(offset).fetch_all(&db.pool).await.map_err(NovaError::storage)?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let mut rec = row_to_memory(&row)?;
            rec.tags = load_tags_for(&db.pool, rec.uuid).await?;
            out.push(rec);
        }
        Ok(out)
    }

    async fn count(&self, db: &Database, filter: &MemoryFilter) -> NovaResult<i64> {
        let built = build_filter(filter, true);
        let mut q = sqlx::query_scalar::<_, i64>(&built.sql);
        for b in &built.binds {
            q = match b {
                BindValue::Text(s) => q.bind(s),
                BindValue::Int(i) => q.bind(i),
                BindValue::Real(f) => q.bind(f),
            };
        }
        let n = q.fetch_one(&db.pool).await.map_err(NovaError::storage)?;
        Ok(n)
    }
}

struct BuiltQuery {
    sql: String,
    binds: Vec<BindValue>,
}

fn build_filter(filter: &MemoryFilter, count_only: bool) -> BuiltQuery {
    // Strategy for `tags_all`:
    //   JOIN + WHERE-IN + GROUP BY + HAVING(COUNT DISTINCT = N).
    // All placeholders use anonymous `?` so binds order trivially maps.

    let cols_list: &str = "m.id, m.uuid, m.content, m.content_hash, m.metadata_json, m.source, \
         m.importance, m.access_count, m.last_accessed, m.created_at, m.expires_at, m.status";

    let mut joins: String = String::new();
    let mut where_clauses: Vec<String> = Vec::new();
    let mut group_by_having: Option<String> = None;
    let mut binds: Vec<BindValue> = Vec::new();

    fn qmarks(n: usize) -> String {
        let mut out = String::with_capacity(2 * n);
        for i in 0..n {
            if i > 0 {
                out.push(',');
            }
            out.push('?');
        }
        out
    }

    if let Some(statuses) = &filter.status_in {
        if !statuses.is_empty() {
            for s in statuses {
                binds.push(BindValue::Text(s.as_str().to_string()));
            }
            where_clauses.push(format!("m.status IN ({})", qmarks(statuses.len())));
        }
    } else {
        where_clauses.push("m.status != 'deleted'".into());
    }

    if let Some(sources) = &filter.source_in {
        if !sources.is_empty() {
            for s in sources {
                binds.push(BindValue::Text(s.as_str().to_string()));
            }
            where_clauses.push(format!("m.source IN ({})", qmarks(sources.len())));
        }
    }
    if let Some(after) = filter.created_after {
        binds.push(BindValue::Int(after.timestamp()));
        where_clauses.push("m.created_at > ?".into());
    }
    if let Some(before) = filter.created_before {
        binds.push(BindValue::Int(before.timestamp()));
        where_clauses.push("m.created_at < ?".into());
    }
    if let Some(min) = filter.importance_min {
        binds.push(BindValue::Real(min as f64));
        where_clauses.push("m.importance >= ?".into());
    }
    if let Some(max) = filter.importance_max {
        binds.push(BindValue::Real(max as f64));
        where_clauses.push("m.importance <= ?".into());
    }
    if let Some(lt) = filter.access_count_lt {
        binds.push(BindValue::Int(lt));
        where_clauses.push("m.access_count < ?".into());
    }
    // "last_accessed" staleness checks: fall back to created_at when NULL so
    // rows that have never been accessed still age out.
    if let Some(before) = filter.last_accessed_before {
        binds.push(BindValue::Int(before.timestamp()));
        where_clauses.push("COALESCE(m.last_accessed, m.created_at) < ?".into());
    }
    if let Some(after) = filter.last_accessed_after {
        binds.push(BindValue::Int(after.timestamp()));
        where_clauses.push("COALESCE(m.last_accessed, m.created_at) > ?".into());
    }

    if let Some(tags) = &filter.tags_all {
        if !tags.is_empty() {
            for t in tags {
                binds.push(BindValue::Text(t.clone()));
            }
            joins.push_str(&format!(
                " INNER JOIN memory_tags mta ON mta.memory_uuid = m.uuid \
                 INNER JOIN tags ta ON ta.id = mta.tag_id AND ta.name IN ({})",
                qmarks(tags.len())
            ));
            group_by_having =
                Some(format!("GROUP BY m.id HAVING COUNT(DISTINCT ta.name) = {}", tags.len()));
        }
    }

    if let Some(tags) = &filter.tags_any {
        if !tags.is_empty() {
            for t in tags {
                binds.push(BindValue::Text(t.clone()));
            }
            where_clauses.push(format!(
                "EXISTS (SELECT 1 FROM memory_tags mte \
                 INNER JOIN tags te ON te.id = mte.tag_id \
                 WHERE mte.memory_uuid = m.uuid AND te.name IN ({}))",
                qmarks(tags.len())
            ));
        }
    }

    let wc: String = where_clauses.join(" AND ");
    let from_with_joins: String = format!("FROM memory_items m{joins}");

    let sql: String = if count_only {
        if group_by_having.is_some() {
            let gbh = group_by_having.as_deref().unwrap();
            let inner = format!("SELECT m.id {from_with_joins} WHERE {wc} {gbh}");
            format!("SELECT COUNT(*) FROM ({inner})")
        } else {
            format!("SELECT COUNT(*) {from_with_joins} WHERE {wc}")
        }
    } else if let Some(gbh) = group_by_having {
        format!("SELECT {cols_list} {from_with_joins} WHERE {wc} {gbh}")
    } else {
        format!("SELECT {cols_list} {from_with_joins} WHERE {wc}")
    };

    BuiltQuery { sql, binds }
}

fn sha256_hex(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn row_to_memory(row: &SqliteRow) -> NovaResult<MemoryRecord> {
    let id: i64 = row.try_get("id").map_err(NovaError::storage)?;
    let uuid_s: String = row.try_get("uuid").map_err(NovaError::storage)?;
    let uuid =
        Uuid::parse_str(&uuid_s).map_err(|e| NovaError::storage_msg(format!("bad uuid: {e}")))?;
    let content: String = row.try_get("content").map_err(NovaError::storage)?;
    let content_hash: String = row.try_get("content_hash").map_err(NovaError::storage)?;
    let meta_s: String = row.try_get("metadata_json").map_err(NovaError::storage)?;
    let metadata: serde_json::Value = serde_json::from_str(&meta_s)
        .map_err(|e| NovaError::storage_msg(format!("metadata json parse: {e}")))?;
    let source_s: String = row.try_get("source").map_err(NovaError::storage)?;
    let source = MemorySource::try_from(source_s.as_str())?;
    let importance: f64 = row.try_get("importance").map_err(NovaError::storage)?;
    let access_count: i64 = row.try_get("access_count").map_err(NovaError::storage)?;
    let last_acc: Option<i64> = row.try_get("last_accessed").map_err(NovaError::storage)?;
    let created_ts: i64 = row.try_get("created_at").map_err(NovaError::storage)?;
    let expires_ts: Option<i64> = row.try_get("expires_at").map_err(NovaError::storage)?;
    let status_s: String = row.try_get("status").map_err(NovaError::storage)?;
    let status = MemoryStatus::try_from(status_s.as_str())?;

    Ok(MemoryRecord {
        id,
        uuid,
        content,
        content_hash,
        metadata,
        source,
        importance: importance as f32,
        access_count,
        last_accessed: last_acc.map(ts_to_dt).transpose()?,
        created_at: ts_to_dt(created_ts)?,
        expires_at: expires_ts.map(ts_to_dt).transpose()?,
        status,
        tags: Vec::new(),
    })
}

fn ts_to_dt(ts: i64) -> NovaResult<DateTime<Utc>> {
    DateTime::from_timestamp(ts, 0)
        .ok_or_else(|| NovaError::storage_msg(format!("bad timestamp {ts}")))
}

async fn load_tags_for(pool: &SqlitePool, uuid: Uuid) -> NovaResult<Vec<String>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT t.name FROM tags t
            INNER JOIN memory_tags mt ON mt.tag_id = t.id
           WHERE mt.memory_uuid = ?1
        ORDER BY t.name",
    )
    .bind(uuid.to_string())
    .fetch_all(pool)
    .await
    .map_err(NovaError::storage)?;
    Ok(rows.into_iter().map(|(n,)| n).collect())
}

pub(crate) async fn attach_tags(
    pool: &SqlitePool,
    memory_uuid: Uuid,
    tags: &[String],
) -> NovaResult<()> {
    let mem = memory_uuid.to_string();
    for tag in tags {
        let tag = tag.trim();
        if tag.is_empty() {
            continue;
        }
        sqlx::query("INSERT OR IGNORE INTO tags (name, created_at) VALUES (?1, ?2)")
            .bind(tag)
            .bind(Utc::now().timestamp())
            .execute(pool)
            .await
            .map_err(NovaError::storage)?;
        let (tag_id,): (i64,) = sqlx::query_as("SELECT id FROM tags WHERE name = ?1")
            .bind(tag)
            .fetch_one(pool)
            .await
            .map_err(NovaError::storage)?;
        sqlx::query("INSERT OR IGNORE INTO memory_tags (memory_uuid, tag_id) VALUES (?1, ?2)")
            .bind(&mem)
            .bind(tag_id)
            .execute(pool)
            .await
            .map_err(NovaError::storage)?;
    }
    Ok(())
}

pub(crate) async fn detach_tags(
    pool: &SqlitePool,
    memory_uuid: Uuid,
    tags: &[String],
) -> NovaResult<()> {
    let mem = memory_uuid.to_string();
    for tag in tags {
        let tag = tag.trim();
        if tag.is_empty() {
            continue;
        }
        sqlx::query(
            "DELETE FROM memory_tags
                WHERE memory_uuid = ?1
                  AND tag_id = (SELECT id FROM tags WHERE name = ?2)",
        )
        .bind(&mem)
        .bind(tag)
        .execute(pool)
        .await
        .map_err(NovaError::storage)?;
    }
    Ok(())
}

pub(crate) async fn list_tags_of_memory(
    pool: &SqlitePool,
    memory_uuid: Uuid,
) -> NovaResult<Vec<String>> {
    load_tags_for(pool, memory_uuid).await
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageConfig;

    async fn temp_db() -> Database {
        let dir = std::env::temp_dir().join(format!("yq-nova-m2-mem-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("test.db");
        let cfg = StorageConfig {
            db_path,
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        Database::open(cfg).await.expect("open temp db")
    }

    #[tokio::test]
    async fn insert_unique_then_duplicate_with_tags() {
        let db = temp_db().await;
        let repo = SqliteMemoryRepository::new();
        let tags = vec!["a".into(), "b".into()];
        let input = InsertMemoryInput {
            content: "hello world",
            tags: &tags,
            importance: 0.8,
            ..Default::default()
        };

        let first = repo.insert(&db, input.clone()).await.unwrap();
        assert!(!first.is_duplicate());

        let second = repo.insert(&db, input).await.unwrap();
        assert!(second.is_duplicate());
        assert_eq!(first.uuid(), second.uuid());

        let got = repo.get_by_uuid(&db, first.uuid()).await.unwrap();
        assert_eq!(got.tags.len(), 2);
        assert_eq!(got.importance, 0.8);
    }

    #[tokio::test]
    async fn missing_uuid_returns_not_found() {
        let db = temp_db().await;
        let repo = SqliteMemoryRepository::new();
        let err = repo.get_by_uuid(&db, Uuid::new_v4()).await.expect_err("NotFound");
        assert_eq!(err.code(), crate::error::ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn update_status_metadata_importance_and_mark_accessed() {
        let db = temp_db().await;
        let repo = SqliteMemoryRepository::new();
        let uuid = repo
            .insert(&db, InsertMemoryInput { content: "x", ..Default::default() })
            .await
            .unwrap()
            .uuid();

        repo.update_status(&db, uuid, MemoryStatus::Archived).await.unwrap();
        repo.update_metadata(&db, uuid, &serde_json::json!({"k":"v"})).await.unwrap();
        repo.update_importance(&db, uuid, 0.99).await.unwrap();
        repo.mark_accessed(&db, uuid).await.unwrap();

        let got = repo.get_by_uuid(&db, uuid).await.unwrap();
        assert_eq!(got.status, MemoryStatus::Archived);
        assert_eq!(got.metadata, serde_json::json!({"k":"v"}));
        assert_eq!(got.importance, 0.99);
        assert_eq!(got.access_count, 1);
        assert!(got.last_accessed.is_some());
    }

    #[tokio::test]
    async fn delete_hard_removes_row() {
        let db = temp_db().await;
        let repo = SqliteMemoryRepository::new();
        let uuid = repo
            .insert(&db, InsertMemoryInput { content: "x", ..Default::default() })
            .await
            .unwrap()
            .uuid();
        repo.delete(&db, uuid).await.unwrap();
        let err = repo.get_by_uuid(&db, uuid).await.expect_err("gone");
        assert_eq!(err.code(), crate::error::ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn list_and_count_support_source_time_importance_tags_filters() {
        let db = temp_db().await;
        let repo = SqliteMemoryRepository::new();
        for (name, src, imp, tag) in [
            ("a", MemorySource::Agent, 0.5, "tag-a"),
            ("b", MemorySource::User, 0.9, "tag-a"),
            ("c", MemorySource::Tool, 0.1, "tag-b"),
        ] {
            repo.insert(
                &db,
                InsertMemoryInput {
                    content: name,
                    source: src,
                    importance: imp,
                    tags: &[tag.to_string()],
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        }

        assert_eq!(repo.count(&db, &MemoryFilter::default()).await.unwrap(), 3);

        let f = MemoryFilter { tags_all: Some(vec!["tag-a".into()]), ..Default::default() };
        assert_eq!(repo.list(&db, &f, 100, 0).await.unwrap().len(), 2);

        let f2 = MemoryFilter { importance_min: Some(0.8), ..Default::default() };
        let hi = repo.list(&db, &f2, 100, 0).await.unwrap();
        assert_eq!(hi.len(), 1);
        assert_eq!(hi[0].content, "b");

        let f3 = MemoryFilter { tags_any: Some(vec!["tag-b".into()]), ..Default::default() };
        assert_eq!(repo.list(&db, &f3, 100, 0).await.unwrap().len(), 1);

        let f4 = MemoryFilter {
            source_in: Some(vec![MemorySource::User, MemorySource::Tool]),
            ..Default::default()
        };
        assert_eq!(repo.list(&db, &f4, 100, 0).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn validation_rejects_bad_input() {
        let db = temp_db().await;
        let repo = SqliteMemoryRepository::new();
        let bad_empty = InsertMemoryInput { content: "", ..Default::default() };
        assert_eq!(
            repo.insert(&db, bad_empty).await.unwrap_err().code(),
            crate::error::ErrorCode::Validation
        );
        let bad_imp = InsertMemoryInput { content: "x", importance: 1.5, ..Default::default() };
        assert_eq!(
            repo.insert(&db, bad_imp).await.unwrap_err().code(),
            crate::error::ErrorCode::Validation
        );
        let uuid = repo
            .insert(&db, InsertMemoryInput { content: "ok", ..Default::default() })
            .await
            .unwrap()
            .uuid();
        assert_eq!(
            repo.update_importance(&db, uuid, -0.1).await.unwrap_err().code(),
            crate::error::ErrorCode::Validation
        );
    }
}
