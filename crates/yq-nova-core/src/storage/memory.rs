use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{Row, SqlitePool, sqlite::SqliteRow};
use uuid::Uuid;

use crate::{
    error::{NovaError, NovaResult},
    storage::{Database, MemoryFilter, MemorySortOrder, MemorySource, MemoryStatus, Repository},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: i64,
    pub uuid: Uuid,
    pub namespace_id: i64,
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
    pub namespace_id: i64,
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
            namespace_id: crate::storage::namespace::DEFAULT_NAMESPACE_ID,
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

#[async_trait]
pub trait MemoryRepository: Repository<MemoryRecord> {
    async fn insert(
        &self,
        db: &Database,
        input: InsertMemoryInput<'_>,
    ) -> NovaResult<InsertOutcome>;
    async fn get_by_uuid(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
    ) -> NovaResult<MemoryRecord>;
    async fn update_status(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
        status: MemoryStatus,
    ) -> NovaResult<()>;
    async fn update_metadata(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
        metadata: &serde_json::Value,
    ) -> NovaResult<()>;
    async fn update_importance(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
        importance: f32,
    ) -> NovaResult<()>;
    async fn update_content(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
        content: &str,
        content_hash: &str,
    ) -> NovaResult<()>;
    async fn update_expires_at(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
        expires_at: Option<i64>,
    ) -> NovaResult<()>;
    async fn delete(&self, db: &Database, namespace_id: i64, uuid: Uuid) -> NovaResult<()>;
    async fn mark_accessed(&self, db: &Database, namespace_id: i64, uuid: Uuid) -> NovaResult<()>;
    async fn list(
        &self,
        db: &Database,
        filter: &MemoryFilter,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<MemoryRecord>>;
    async fn list_ordered(
        &self,
        db: &Database,
        filter: &MemoryFilter,
        limit: usize,
        offset: usize,
        order: MemorySortOrder,
    ) -> NovaResult<Vec<MemoryRecord>>;
    async fn count(&self, db: &Database, filter: &MemoryFilter) -> NovaResult<i64>;
}

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
            "SELECT id, uuid FROM memory_items WHERE content_hash = ?1 AND namespace_id = ?2 AND \
             status != 'deleted'",
        )
        .bind(&content_hash)
        .bind(input.namespace_id)
        .fetch_optional(pool)
        .await
        .map_err(NovaError::storage)?;

        if let Some((_id, existing_uuid_s)) = existing {
            let existing_uuid = Uuid::parse_str(&existing_uuid_s)
                .map_err(|e| NovaError::storage_msg(format!("bad existing uuid: {e}")))?;
            if !input.tags.is_empty() {
                attach_tags(pool, input.namespace_id, existing_uuid, input.tags).await?;
            }
            return Ok(InsertOutcome::Duplicate(existing_uuid, ()));
        }

        let uuid = Uuid::new_v4();
        let metadata_json = input.metadata.cloned().unwrap_or_else(|| serde_json::json!({}));
        let now = Utc::now().timestamp();
        let expires_ts = input.expires_at.map(|t| t.timestamp());

        sqlx::query(
            "INSERT INTO memory_items (
                uuid, namespace_id, content, content_hash, metadata_json, source, importance,
                access_count, last_accessed, created_at, expires_at, status
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        )
        .bind(uuid.to_string())
        .bind(input.namespace_id)
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
            attach_tags(pool, input.namespace_id, uuid, input.tags).await?;
        }

        Ok(InsertOutcome::Inserted(uuid))
    }

    async fn get_by_uuid(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
    ) -> NovaResult<MemoryRecord> {
        let row = sqlx::query(
            "SELECT id, uuid, namespace_id, content, content_hash, metadata_json, source, \
             importance, access_count, last_accessed, created_at, expires_at, status
             FROM memory_items WHERE uuid = ?1 AND namespace_id = ?2",
        )
        .bind(uuid.to_string())
        .bind(namespace_id)
        .fetch_optional(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        let row = row.ok_or_else(|| NovaError::not_found(format!("memory {uuid}")))?;
        let mut rec = row_to_memory(&row)?;
        rec.tags = load_tags_for(&db.pool, namespace_id, uuid).await?;
        Ok(rec)
    }

    async fn update_status(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
        status: MemoryStatus,
    ) -> NovaResult<()> {
        let res = sqlx::query(
            "UPDATE memory_items SET status = ?1 WHERE uuid = ?2 AND namespace_id = ?3",
        )
        .bind(status.as_str())
        .bind(uuid.to_string())
        .bind(namespace_id)
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
        namespace_id: i64,
        uuid: Uuid,
        metadata: &serde_json::Value,
    ) -> NovaResult<()> {
        let res = sqlx::query(
            "UPDATE memory_items SET metadata_json = ?1 WHERE uuid = ?2 AND namespace_id = ?3",
        )
        .bind(metadata.to_string())
        .bind(uuid.to_string())
        .bind(namespace_id)
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
        namespace_id: i64,
        uuid: Uuid,
        importance: f32,
    ) -> NovaResult<()> {
        if !(0.0..=1.0).contains(&importance) {
            return Err(NovaError::validation("memory.importance must be in [0.0, 1.0]"));
        }
        let res = sqlx::query(
            "UPDATE memory_items SET importance = ?1 WHERE uuid = ?2 AND namespace_id = ?3",
        )
        .bind(importance as f64)
        .bind(uuid.to_string())
        .bind(namespace_id)
        .execute(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("memory {uuid}")));
        }
        Ok(())
    }

    async fn update_content(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
        content: &str,
        content_hash: &str,
    ) -> NovaResult<()> {
        let res = sqlx::query(
            "UPDATE memory_items SET content = ?1, content_hash = ?2 WHERE uuid = ?3 AND \
             namespace_id = ?4",
        )
        .bind(content)
        .bind(content_hash)
        .bind(uuid.to_string())
        .bind(namespace_id)
        .execute(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("memory {uuid}")));
        }
        Ok(())
    }

    async fn update_expires_at(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
        expires_at: Option<i64>,
    ) -> NovaResult<()> {
        let res = sqlx::query(
            "UPDATE memory_items SET expires_at = ?1 WHERE uuid = ?2 AND namespace_id = ?3",
        )
        .bind(expires_at)
        .bind(uuid.to_string())
        .bind(namespace_id)
        .execute(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("memory {uuid}")));
        }
        Ok(())
    }

    async fn delete(&self, db: &Database, namespace_id: i64, uuid: Uuid) -> NovaResult<()> {
        let res = sqlx::query("DELETE FROM memory_items WHERE uuid = ?1 AND namespace_id = ?2")
            .bind(uuid.to_string())
            .bind(namespace_id)
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("memory {uuid}")));
        }
        Ok(())
    }

    async fn mark_accessed(&self, db: &Database, namespace_id: i64, uuid: Uuid) -> NovaResult<()> {
        let now = Utc::now().timestamp();
        sqlx::query(
            "UPDATE memory_items
                SET access_count = access_count + 1,
                    last_accessed = ?1
              WHERE uuid = ?2 AND namespace_id = ?3",
        )
        .bind(now)
        .bind(uuid.to_string())
        .bind(namespace_id)
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
        self.list_ordered(db, filter, limit, offset, MemorySortOrder::CreatedDesc).await
    }

    async fn list_ordered(
        &self,
        db: &Database,
        filter: &MemoryFilter,
        limit: usize,
        offset: usize,
        order: MemorySortOrder,
    ) -> NovaResult<Vec<MemoryRecord>> {
        let built = build_filter(filter, false);
        let limit = limit.min(10_000) as i64;
        let offset = offset as i64;
        let sql = format!("{} ORDER BY {} LIMIT ? OFFSET ?", built.sql, order.order_by_sql());

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
        let ns = filter.namespace_id.unwrap_or(crate::storage::namespace::DEFAULT_NAMESPACE_ID);
        for row in rows {
            let mut rec = row_to_memory(&row)?;
            rec.tags = load_tags_for(&db.pool, ns, rec.uuid).await?;
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

impl SqliteMemoryRepository {
    pub async fn check_content_hash_conflict(
        &self,
        db: &Database,
        namespace_id: i64,
        content_hash: &str,
        exclude_uuid: Uuid,
    ) -> NovaResult<Option<Uuid>> {
        let row: Option<(String,)> = sqlx::query_as(
            "SELECT uuid FROM memory_items WHERE content_hash = ?1 AND namespace_id = ?2 AND uuid \
             != ?3 AND status != 'deleted'",
        )
        .bind(content_hash)
        .bind(namespace_id)
        .bind(exclude_uuid.to_string())
        .fetch_optional(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        match row {
            Some((uuid_s,)) => {
                let uuid = Uuid::parse_str(&uuid_s)
                    .map_err(|e| NovaError::storage_msg(format!("bad uuid: {e}")))?;
                Ok(Some(uuid))
            },
            None => Ok(None),
        }
    }
}

struct BuiltQuery {
    sql: String,
    binds: Vec<BindValue>,
}

fn build_filter(filter: &MemoryFilter, count_only: bool) -> BuiltQuery {
    let cols_list: &str = "m.id, m.uuid, m.namespace_id, m.content, m.content_hash, \
                           m.metadata_json, m.source, m.importance, m.access_count, \
                           m.last_accessed, m.created_at, m.expires_at, m.status";

    let mut joins: String = String::new();
    let mut where_clauses: Vec<String> = Vec::new();
    let mut group_by_having: Option<String> = None;
    let mut join_binds: Vec<BindValue> = Vec::new();
    let mut where_binds: Vec<BindValue> = Vec::new();

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

    if let Some(ns) = filter.namespace_id {
        where_binds.push(BindValue::Int(ns));
        where_clauses.push("m.namespace_id = ?".into());
    }

    if let Some(tags) = &filter.tags_all {
        if !tags.is_empty() {
            for t in tags {
                join_binds.push(BindValue::Text(t.clone()));
            }
            joins.push_str(&format!(
                " INNER JOIN memory_tags mta ON mta.memory_uuid = m.uuid INNER JOIN tags ta ON \
                 ta.id = mta.tag_id AND ta.name IN ({})",
                qmarks(tags.len())
            ));
            group_by_having =
                Some(format!("GROUP BY m.id HAVING COUNT(DISTINCT ta.name) = {}", tags.len()));
        }
    }

    if let Some(statuses) = &filter.status_in {
        if !statuses.is_empty() {
            for s in statuses {
                where_binds.push(BindValue::Text(s.as_str().to_string()));
            }
            where_clauses.push(format!("m.status IN ({})", qmarks(statuses.len())));
        }
    } else {
        where_clauses.push("m.status != 'deleted'".into());
    }

    if let Some(sources) = &filter.source_in {
        if !sources.is_empty() {
            for s in sources {
                where_binds.push(BindValue::Text(s.as_str().to_string()));
            }
            where_clauses.push(format!("m.source IN ({})", qmarks(sources.len())));
        }
    }
    if let Some(after) = filter.created_after {
        where_binds.push(BindValue::Int(after.timestamp()));
        where_clauses.push("m.created_at > ?".into());
    }
    if let Some(before) = filter.created_before {
        where_binds.push(BindValue::Int(before.timestamp()));
        where_clauses.push("m.created_at < ?".into());
    }
    if let Some(min) = filter.importance_min {
        where_binds.push(BindValue::Real(min as f64));
        where_clauses.push("m.importance >= ?".into());
    }
    if let Some(max) = filter.importance_max {
        where_binds.push(BindValue::Real(max as f64));
        where_clauses.push("m.importance <= ?".into());
    }
    if let Some(lt) = filter.access_count_lt {
        where_binds.push(BindValue::Int(lt));
        where_clauses.push("m.access_count < ?".into());
    }

    if let Some(before) = filter.last_accessed_before {
        where_binds.push(BindValue::Int(before.timestamp()));
        where_clauses.push("COALESCE(m.last_accessed, m.created_at) < ?".into());
    }
    if let Some(after) = filter.last_accessed_after {
        where_binds.push(BindValue::Int(after.timestamp()));
        where_clauses.push("COALESCE(m.last_accessed, m.created_at) > ?".into());
    }

    if let Some(tags) = &filter.tags_any {
        if !tags.is_empty() {
            for t in tags {
                where_binds.push(BindValue::Text(t.clone()));
            }
            where_clauses.push(format!(
                "EXISTS (SELECT 1 FROM memory_tags mte INNER JOIN tags te ON te.id = mte.tag_id \
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

    join_binds.extend(where_binds);
    BuiltQuery {
        sql,
        binds: join_binds,
    }
}

pub(crate) fn sha256_hex(s: &str) -> String {
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
    let namespace_id: i64 = row.try_get("namespace_id").map_err(NovaError::storage)?;
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
        namespace_id,
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

async fn load_tags_for(
    pool: &SqlitePool,
    namespace_id: i64,
    uuid: Uuid,
) -> NovaResult<Vec<String>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT t.name FROM tags t
            INNER JOIN memory_tags mt ON mt.tag_id = t.id
           WHERE mt.memory_uuid = ?1 AND mt.namespace_id = ?2
        ORDER BY t.name",
    )
    .bind(uuid.to_string())
    .bind(namespace_id)
    .fetch_all(pool)
    .await
    .map_err(NovaError::storage)?;
    Ok(rows.into_iter().map(|(n,)| n).collect())
}

pub(crate) async fn attach_tags(
    pool: &SqlitePool,
    namespace_id: i64,
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
            "INSERT OR IGNORE INTO tags (name, created_at, namespace_id) VALUES (?1, ?2, ?3)",
        )
        .bind(tag)
        .bind(Utc::now().timestamp())
        .bind(namespace_id)
        .execute(pool)
        .await
        .map_err(NovaError::storage)?;
        let (tag_id,): (i64,) =
            sqlx::query_as("SELECT id FROM tags WHERE name = ?1 AND namespace_id = ?2")
                .bind(tag)
                .bind(namespace_id)
                .fetch_one(pool)
                .await
                .map_err(NovaError::storage)?;
        sqlx::query(
            "INSERT OR IGNORE INTO memory_tags (memory_uuid, tag_id, namespace_id) VALUES (?1, \
             ?2, ?3)",
        )
        .bind(&mem)
        .bind(tag_id)
        .bind(namespace_id)
        .execute(pool)
        .await
        .map_err(NovaError::storage)?;
    }
    Ok(())
}

pub(crate) async fn detach_tags(
    pool: &SqlitePool,
    namespace_id: i64,
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
                  AND namespace_id = ?2
                  AND tag_id = (SELECT id FROM tags WHERE name = ?3 AND namespace_id = ?2)",
        )
        .bind(&mem)
        .bind(namespace_id)
        .bind(tag)
        .execute(pool)
        .await
        .map_err(NovaError::storage)?;
    }
    Ok(())
}

pub(crate) async fn list_tags_of_memory(
    pool: &SqlitePool,
    namespace_id: i64,
    memory_uuid: Uuid,
) -> NovaResult<Vec<String>> {
    load_tags_for(pool, namespace_id, memory_uuid).await
}

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

    const NS: i64 = crate::storage::namespace::DEFAULT_NAMESPACE_ID;

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

        let got = repo.get_by_uuid(&db, NS, first.uuid()).await.unwrap();
        assert_eq!(got.tags.len(), 2);
        assert_eq!(got.importance, 0.8);
    }

    #[tokio::test]
    async fn missing_uuid_returns_not_found() {
        let db = temp_db().await;
        let repo = SqliteMemoryRepository::new();
        let err = repo.get_by_uuid(&db, NS, Uuid::new_v4()).await.expect_err("NotFound");
        assert_eq!(err.code(), crate::error::ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn update_status_metadata_importance_and_mark_accessed() {
        let db = temp_db().await;
        let repo = SqliteMemoryRepository::new();
        let uuid = repo
            .insert(
                &db,
                InsertMemoryInput {
                    content: "x",
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .uuid();

        repo.update_status(&db, NS, uuid, MemoryStatus::Archived).await.unwrap();
        repo.update_metadata(&db, NS, uuid, &serde_json::json!({"k":"v"})).await.unwrap();
        repo.update_importance(&db, NS, uuid, 0.99).await.unwrap();
        repo.mark_accessed(&db, NS, uuid).await.unwrap();

        let got = repo.get_by_uuid(&db, NS, uuid).await.unwrap();
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
            .insert(
                &db,
                InsertMemoryInput {
                    content: "x",
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .uuid();
        repo.delete(&db, NS, uuid).await.unwrap();
        let err = repo.get_by_uuid(&db, NS, uuid).await.expect_err("gone");
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

        let f = MemoryFilter {
            tags_all: Some(vec!["tag-a".into()]),
            ..Default::default()
        };
        assert_eq!(repo.list(&db, &f, 100, 0).await.unwrap().len(), 2);

        let f2 = MemoryFilter {
            importance_min: Some(0.8),
            ..Default::default()
        };
        let hi = repo.list(&db, &f2, 100, 0).await.unwrap();
        assert_eq!(hi.len(), 1);
        assert_eq!(hi[0].content, "b");

        let f3 = MemoryFilter {
            tags_any: Some(vec!["tag-b".into()]),
            ..Default::default()
        };
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
        let bad_empty = InsertMemoryInput {
            content: "",
            ..Default::default()
        };
        assert_eq!(
            repo.insert(&db, bad_empty).await.unwrap_err().code(),
            crate::error::ErrorCode::Validation
        );
        let bad_imp = InsertMemoryInput {
            content: "x",
            importance: 1.5,
            ..Default::default()
        };
        assert_eq!(
            repo.insert(&db, bad_imp).await.unwrap_err().code(),
            crate::error::ErrorCode::Validation
        );
        let uuid = repo
            .insert(
                &db,
                InsertMemoryInput {
                    content: "ok",
                    ..Default::default()
                },
            )
            .await
            .unwrap()
            .uuid();
        assert_eq!(
            repo.update_importance(&db, NS, uuid, -0.1).await.unwrap_err().code(),
            crate::error::ErrorCode::Validation
        );
    }

    #[tokio::test]
    async fn list_and_count_bind_tag_joins_ahead_of_status_filters() {
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

        let f = MemoryFilter {
            status_in: Some(vec![MemoryStatus::Active, MemoryStatus::Archived]),
            tags_all: Some(vec!["tag-a".into()]),
            ..Default::default()
        };
        assert_eq!(repo.count(&db, &f).await.unwrap(), 2);
        assert_eq!(repo.list(&db, &f, 100, 0).await.unwrap().len(), 2);

        let f2 = MemoryFilter {
            status_in: Some(vec![MemoryStatus::Active]),
            tags_all: Some(vec!["tag-a".into(), "tag-b".into()]),
            ..Default::default()
        };
        assert_eq!(repo.count(&db, &f2).await.unwrap(), 0);
        assert_eq!(repo.list(&db, &f2, 100, 0).await.unwrap().len(), 0);
    }
}
