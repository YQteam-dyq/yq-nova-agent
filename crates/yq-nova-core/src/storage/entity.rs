//! Entity repository backed by the `entities` table.
//!
//! Entities are (name, type)-unique nodes in the knowledge graph. The same
//! real-world concept with different capitalizations maps to a single row,
//! but the store itself does NOT do normalization — callers are expected to
//! normalize names before insert (e.g. lowercase + strip) if that behaviour
//! is desired. Upsert semantics allow safely re-inserting the same pair
//! repeatedly without creating duplicate rows.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

use crate::{
    error::{NovaError, NovaResult},
    storage::{Database, Repository},
};

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/// 实体记录：知识图谱中的节点，由 (name, type) 唯一标识。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    /// 内部自增 ID。
    pub id: i64,
    /// 对外暴露的实体 UUID。
    pub uuid: Uuid,
    /// 实体名称（如 "Alice Smith"）。
    pub name: String,
    /// 实体类型（如 "person"、"company"、"unknown"）。
    pub r#type: String,
    /// 可选的自由文本描述。
    pub description: Option<String>,
    /// 自定义元数据 JSON。
    pub metadata: serde_json::Value,
    /// 创建时间（UTC）。
    pub created_at: DateTime<Utc>,
    /// 最近一次 upsert 更新时间（UTC）。
    pub updated_at: DateTime<Utc>,
}

/// 实体 upsert 输入，以 (name, type) 为幂等键。
#[derive(Debug, Clone)]
pub struct UpsertEntityInput<'a> {
    /// 实体名称（去除首尾空白后不能为空）。
    pub name: &'a str,
    /// 实体类型；空字符串在下游会被回退为 "unknown"。
    pub r#type: &'a str,
    /// 可选描述，None 表示不覆盖现有值。
    pub description: Option<&'a str>,
    /// 可选元数据 JSON，None 表示不覆盖现有值。
    pub metadata: Option<&'a serde_json::Value>,
}

/// Result of an `upsert` call — tells the caller whether a new row was
/// created or an existing one was (potentially) updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", content = "uuid", rename_all = "snake_case")]
pub enum UpsertOutcome {
    /// 新创建了一条记录；内部携带该记录的 UUID。
    Created(Uuid),
    /// 更新了已存在的记录；内部携带该记录的 UUID。
    Updated(Uuid),
}

impl UpsertOutcome {
    pub fn uuid(&self) -> Uuid {
        match self {
            UpsertOutcome::Created(u) | UpsertOutcome::Updated(u) => *u,
        }
    }
}

// -----------------------------------------------------------------------------
// Trait
// -----------------------------------------------------------------------------

#[async_trait]
pub trait EntityRepository: Repository<EntityRecord> {
    /// Idempotent upsert keyed on `(name, type)`. Creates a new row if the
    /// pair is absent; otherwise updates `description` and `metadata` only
    /// when the input carries `Some(...)` values (pass `None` to keep the
    /// existing value). `updated_at` is always bumped on UPDATE.
    async fn upsert(
        &self,
        db: &Database,
        input: UpsertEntityInput<'_>,
    ) -> NovaResult<UpsertOutcome>;

    /// Fetch by UUID.
    async fn get_by_uuid(&self, db: &Database, uuid: Uuid) -> NovaResult<EntityRecord>;

    /// Fetch a single entity by the unique `(name, type)` pair.
    async fn get_by_name_type(
        &self,
        db: &Database,
        name: &str,
        r#type: &str,
    ) -> NovaResult<Option<EntityRecord>>;

    /// Delete an entity. Relations referencing it are deleted by FK CASCADE.
    async fn delete(&self, db: &Database, uuid: Uuid) -> NovaResult<()>;

    /// List entities matching a simple prefix-name / type filter.
    async fn list(
        &self,
        db: &Database,
        name_prefix: Option<&str>,
        type_filter: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<EntityRecord>>;
}

// -----------------------------------------------------------------------------
// Sqlite impl
// -----------------------------------------------------------------------------

#[derive(Clone)]
pub struct SqliteEntityRepository;

impl SqliteEntityRepository {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for SqliteEntityRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Repository<EntityRecord> for SqliteEntityRepository {
    fn name(&self) -> &'static str {
        "entity.sqlite"
    }
}

#[async_trait]
impl EntityRepository for SqliteEntityRepository {
    async fn upsert(
        &self,
        db: &Database,
        input: UpsertEntityInput<'_>,
    ) -> NovaResult<UpsertOutcome> {
        let name = input.name.trim();
        let r#type = input.r#type.trim();
        if name.is_empty() {
            return Err(NovaError::validation("entity.name must not be empty"));
        }
        if r#type.is_empty() {
            return Err(NovaError::validation("entity.type must not be empty"));
        }

        let pool = &db.pool;
        let now = Utc::now().timestamp();
        let metadata_json = input.metadata.cloned().unwrap_or_else(|| serde_json::json!({}));

        // Try existing lookup first so we can correctly report Created vs
        // Updated (SQLite's ON CONFLICT DO UPDATE doesn't easily tell us
        // which branch happened without RETURNING + pre-state comparison).
        let existing: Option<(i64, String, Option<String>, String)> = sqlx::query_as(
            "SELECT id, uuid, description, metadata_json FROM entities \
             WHERE name = ?1 AND type = ?2",
        )
        .bind(name)
        .bind(r#type)
        .fetch_optional(pool)
        .await
        .map_err(NovaError::storage)?;

        if let Some((_id, existing_uuid_s, existing_desc, existing_meta_s)) = existing {
            let existing_uuid = Uuid::parse_str(&existing_uuid_s)
                .map_err(|e| NovaError::storage_msg(format!("bad entity uuid: {e}")))?;
            let new_desc = input.description.or(existing_desc.as_deref());
            let new_meta = if input.metadata.is_some() {
                &metadata_json
            } else {
                // keep existing; re-parse because we want a Value for bind
                &serde_json::from_str::<serde_json::Value>(&existing_meta_s)
                    .map_err(|e| NovaError::storage_msg(format!("bad meta: {e}")))?
            };
            sqlx::query(
                "UPDATE entities SET description = ?1, metadata_json = ?2, updated_at = ?3 \
                 WHERE uuid = ?4",
            )
            .bind(new_desc)
            .bind(new_meta.to_string())
            .bind(now)
            .bind(existing_uuid.to_string())
            .execute(pool)
            .await
            .map_err(NovaError::storage)?;
            return Ok(UpsertOutcome::Updated(existing_uuid));
        }

        let uuid = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO entities (uuid, name, type, description, metadata_json, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )
        .bind(uuid.to_string())
        .bind(name)
        .bind(r#type)
        .bind(input.description)
        .bind(metadata_json.to_string())
        .bind(now)
        .bind(now)
        .execute(pool)
        .await
        .map_err(NovaError::storage)?;
        Ok(UpsertOutcome::Created(uuid))
    }

    async fn get_by_uuid(&self, db: &Database, uuid: Uuid) -> NovaResult<EntityRecord> {
        let row = sqlx::query(
            "SELECT id, uuid, name, type, description, metadata_json, created_at, updated_at \
             FROM entities WHERE uuid = ?1",
        )
        .bind(uuid.to_string())
        .fetch_optional(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        let row = row.ok_or_else(|| NovaError::not_found(format!("entity {uuid}")))?;
        row_to_entity(&row)
    }

    async fn get_by_name_type(
        &self,
        db: &Database,
        name: &str,
        r#type: &str,
    ) -> NovaResult<Option<EntityRecord>> {
        let row = sqlx::query(
            "SELECT id, uuid, name, type, description, metadata_json, created_at, updated_at \
             FROM entities WHERE name = ?1 AND type = ?2",
        )
        .bind(name)
        .bind(r#type)
        .fetch_optional(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        match row {
            Some(r) => Ok(Some(row_to_entity(&r)?)),
            None => Ok(None),
        }
    }

    async fn delete(&self, db: &Database, uuid: Uuid) -> NovaResult<()> {
        let res = sqlx::query("DELETE FROM entities WHERE uuid = ?1")
            .bind(uuid.to_string())
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("entity {uuid}")));
        }
        Ok(())
    }

    async fn list(
        &self,
        db: &Database,
        name_prefix: Option<&str>,
        type_filter: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<EntityRecord>> {
        let mut where_clauses: Vec<&str> = Vec::new();
        if name_prefix.is_some() {
            where_clauses.push("name LIKE ?");
        }
        if type_filter.is_some() {
            where_clauses.push("type = ?");
        }
        let wc = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };
        let limit = limit.min(10_000) as i64;
        let offset = offset as i64;
        let sql = format!(
            "SELECT id, uuid, name, type, description, metadata_json, created_at, updated_at \
             FROM entities {wc} ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        );
        let mut q = sqlx::query(&sql);
        if let Some(np) = name_prefix {
            q = q.bind(format!("{}%", np.replace('%', "\\%")));
        }
        if let Some(t) = type_filter {
            q = q.bind(t);
        }
        let rows =
            q.bind(limit).bind(offset).fetch_all(&db.pool).await.map_err(NovaError::storage)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(row_to_entity(&row)?);
        }
        Ok(out)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn row_to_entity(row: &sqlx::sqlite::SqliteRow) -> NovaResult<EntityRecord> {
    let id: i64 = row.try_get("id").map_err(NovaError::storage)?;
    let uuid_s: String = row.try_get("uuid").map_err(NovaError::storage)?;
    let uuid =
        Uuid::parse_str(&uuid_s).map_err(|e| NovaError::storage_msg(format!("bad uuid: {e}")))?;
    let name: String = row.try_get("name").map_err(NovaError::storage)?;
    let r#type: String = row.try_get("type").map_err(NovaError::storage)?;
    let description: Option<String> = row.try_get("description").map_err(NovaError::storage)?;
    let meta_s: String = row.try_get("metadata_json").map_err(NovaError::storage)?;
    let metadata: serde_json::Value = serde_json::from_str(&meta_s)
        .map_err(|e| NovaError::storage_msg(format!("entity meta json: {e}")))?;
    let created_ts: i64 = row.try_get("created_at").map_err(NovaError::storage)?;
    let updated_ts: i64 = row.try_get("updated_at").map_err(NovaError::storage)?;
    Ok(EntityRecord {
        id,
        uuid,
        name,
        r#type,
        description,
        metadata,
        created_at: ts_to_dt(created_ts)?,
        updated_at: ts_to_dt(updated_ts)?,
    })
}

fn ts_to_dt(ts: i64) -> NovaResult<DateTime<Utc>> {
    DateTime::from_timestamp(ts, 0).ok_or_else(|| NovaError::storage_msg(format!("bad ts {ts}")))
}

// -----------------------------------------------------------------------------
// Graph traversal types (used by RelationRepository; re-exported for brevity).
// -----------------------------------------------------------------------------

/// Graph traversal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Out,
    In,
    #[default]
    Both,
}

/// BFS 图谱遍历结果中的单个节点。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraverseNode {
    /// 该节点对应的完整实体记录（name / type / description / metadata 等）。
    pub entity: EntityRecord,
    /// 从遍历起点到该节点的最短跳数（起点深度为 0）。
    pub depth: u8,
    /// 从起点到该节点的 UUID 路径（含起点与终点），用于溯源。
    pub path: Vec<Uuid>,
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageConfig;

    async fn temp_db() -> Database {
        let dir = std::env::temp_dir().join(format!("yq-nova-m2-ent-{}", Uuid::new_v4()));
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
    async fn upsert_creates_then_updates_metadata_preserves_uuid() {
        let db = temp_db().await;
        let repo = SqliteEntityRepository::new();
        let first = repo
            .upsert(
                &db,
                UpsertEntityInput {
                    name: "Alice",
                    r#type: "person",
                    description: Some("desc 1"),
                    metadata: Some(&serde_json::json!({"role":"ceo"})),
                },
            )
            .await
            .unwrap();
        assert!(matches!(first, UpsertOutcome::Created(_)));
        let uuid = first.uuid();

        let second = repo
            .upsert(
                &db,
                UpsertEntityInput {
                    name: "Alice",
                    r#type: "person",
                    description: None, // keep existing
                    metadata: None,    // keep existing
                },
            )
            .await
            .unwrap();
        assert!(matches!(second, UpsertOutcome::Updated(_)));
        assert_eq!(second.uuid(), uuid);

        let got = repo.get_by_uuid(&db, uuid).await.unwrap();
        assert_eq!(got.description.as_deref(), Some("desc 1"));
        assert_eq!(got.metadata, serde_json::json!({"role":"ceo"}));
    }

    #[tokio::test]
    async fn upsert_rejects_blank_name_or_type() {
        let db = temp_db().await;
        let repo = SqliteEntityRepository::new();
        let err = repo
            .upsert(
                &db,
                UpsertEntityInput { name: "  ", r#type: "x", description: None, metadata: None },
            )
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
        let err2 = repo
            .upsert(
                &db,
                UpsertEntityInput { name: "x", r#type: "", description: None, metadata: None },
            )
            .await
            .unwrap_err();
        assert_eq!(err2.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn delete_returns_not_found_for_missing() {
        let db = temp_db().await;
        let repo = SqliteEntityRepository::new();
        let err = repo.delete(&db, Uuid::new_v4()).await.unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn list_supports_name_prefix_and_type_filter() {
        let db = temp_db().await;
        let repo = SqliteEntityRepository::new();
        for (n, t) in [("Alice", "person"), ("Alex", "person"), ("Bob", "org")] {
            repo.upsert(
                &db,
                UpsertEntityInput { name: n, r#type: t, description: None, metadata: None },
            )
            .await
            .unwrap();
        }
        let all = repo.list(&db, None, None, 100, 0).await.unwrap();
        assert_eq!(all.len(), 3);
        let al = repo.list(&db, Some("Al"), None, 100, 0).await.unwrap();
        assert_eq!(al.len(), 2);
        let org = repo.list(&db, None, Some("org"), 100, 0).await.unwrap();
        assert_eq!(org.len(), 1);
        assert_eq!(org[0].name, "Bob");
    }
}
