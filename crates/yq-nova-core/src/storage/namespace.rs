use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

use crate::{
    error::{NovaError, NovaResult},
    storage::{Database, Repository},
};

pub const DEFAULT_NAMESPACE_ID: i64 = 1;
pub const DEFAULT_NAMESPACE_NAME: &str = "default";
pub const DEFAULT_NAMESPACE_UUID: &str = "00000000-0000-0000-0000-000000000001";

pub fn default_namespace_id() -> i64 {
    DEFAULT_NAMESPACE_ID
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceRecord {
    pub id: i64,
    pub uuid: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub config: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CreateNamespaceInput<'a> {
    pub name: &'a str,
    pub description: Option<&'a str>,
    pub config: Option<&'a serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct UpdateNamespaceInput<'a> {
    pub name: &'a str,
    pub description: Option<&'a str>,
    pub config: Option<&'a serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteNamespaceOutcome {
    Deleted(i64),
    Protected(&'static str),
    NotFound,
}

#[async_trait]
pub trait NamespaceRepository: Repository<NamespaceRecord> {
    async fn create(
        &self,
        db: &Database,
        input: CreateNamespaceInput<'_>,
    ) -> NovaResult<NamespaceRecord>;
    async fn get_by_id(&self, db: &Database, id: i64) -> NovaResult<NamespaceRecord>;
    async fn get_by_name(&self, db: &Database, name: &str) -> NovaResult<Option<NamespaceRecord>>;
    async fn list(
        &self,
        db: &Database,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<NamespaceRecord>>;
    async fn count_all(&self, db: &Database) -> NovaResult<i64>;
    async fn update(
        &self,
        db: &Database,
        input: UpdateNamespaceInput<'_>,
    ) -> NovaResult<NamespaceRecord>;
    async fn delete(&self, db: &Database, name: &str) -> NovaResult<DeleteNamespaceOutcome>;
    async fn resolve(&self, db: &Database, name: &str) -> NovaResult<Option<NamespaceRecord>>;
}

#[derive(Clone)]
pub struct SqliteNamespaceRepository;

impl SqliteNamespaceRepository {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for SqliteNamespaceRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Repository<NamespaceRecord> for SqliteNamespaceRepository {
    fn name(&self) -> &'static str {
        "namespace.sqlite"
    }
}

fn row_to_namespace(row: &sqlx::sqlite::SqliteRow) -> NovaResult<NamespaceRecord> {
    let id: i64 = row.try_get("id").map_err(NovaError::storage)?;
    let uuid_s: String = row.try_get("uuid").map_err(NovaError::storage)?;
    let uuid =
        Uuid::parse_str(&uuid_s).map_err(|e| NovaError::storage_msg(format!("bad uuid: {e}")))?;
    let name: String = row.try_get("name").map_err(NovaError::storage)?;
    let description: Option<String> = row.try_get("description").map_err(NovaError::storage)?;
    let config_s: String = row.try_get("config_json").map_err(NovaError::storage)?;
    let config: serde_json::Value = serde_json::from_str(&config_s)
        .map_err(|e| NovaError::storage_msg(format!("config json parse: {e}")))?;
    let created_ts: i64 = row.try_get("created_at").map_err(NovaError::storage)?;
    let updated_ts: i64 = row.try_get("updated_at").map_err(NovaError::storage)?;
    Ok(NamespaceRecord {
        id,
        uuid,
        name,
        description,
        config,
        created_at: DateTime::from_timestamp(created_ts, 0)
            .ok_or_else(|| NovaError::storage_msg(format!("bad ts {created_ts}")))?,
        updated_at: DateTime::from_timestamp(updated_ts, 0)
            .ok_or_else(|| NovaError::storage_msg(format!("bad ts {updated_ts}")))?,
    })
}

const NS_COLUMNS: &str = "id, uuid, name, description, config_json, created_at, updated_at";

#[async_trait]
impl NamespaceRepository for SqliteNamespaceRepository {
    async fn create(
        &self,
        db: &Database,
        input: CreateNamespaceInput<'_>,
    ) -> NovaResult<NamespaceRecord> {
        let name = input.name.trim();
        if name.is_empty() {
            return Err(NovaError::validation("namespace.name must not be empty"));
        }
        if name != input.name {
            return Err(NovaError::validation(
                "namespace.name must not contain leading/trailing whitespace",
            ));
        }
        if name.eq_ignore_ascii_case(DEFAULT_NAMESPACE_NAME) {
            return Err(NovaError::conflict(format!(
                "namespace name '{DEFAULT_NAMESPACE_NAME}' is reserved"
            )));
        }
        let pool = &db.pool;
        let exists: Option<(String,)> =
            sqlx::query_as("SELECT name FROM namespaces WHERE LOWER(name) = LOWER(?1)")
                .bind(name)
                .fetch_optional(pool)
                .await
                .map_err(NovaError::storage)?;
        if exists.is_some() {
            return Err(NovaError::conflict(format!("namespace '{name}' already exists")));
        }
        let uuid = Uuid::new_v4();
        let now = Utc::now().timestamp();
        let config_json =
            input.config.cloned().unwrap_or_else(|| serde_json::json!({})).to_string();
        let mut tx = db.begin().await?;
        sqlx::query(
            "INSERT INTO namespaces (uuid, name, description, config_json, created_at, \
             updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .bind(uuid.to_string())
        .bind(name)
        .bind(input.description)
        .bind(config_json)
        .bind(now)
        .bind(now)
        .execute(&mut *tx)
        .await
        .map_err(NovaError::storage)?;
        let id: i64 = sqlx::query_scalar("SELECT last_insert_rowid()")
            .fetch_one(&mut *tx)
            .await
            .map_err(NovaError::storage)?;
        tx.commit().await.map_err(NovaError::from)?;
        self.get_by_id(db, id).await
    }

    async fn get_by_id(&self, db: &Database, id: i64) -> NovaResult<NamespaceRecord> {
        let row = sqlx::query(&format!("SELECT {NS_COLUMNS} FROM namespaces WHERE id = ?1"))
            .bind(id)
            .fetch_optional(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        let row = row.ok_or_else(|| NovaError::not_found(format!("namespace id {id}")))?;
        row_to_namespace(&row)
    }

    async fn get_by_name(&self, db: &Database, name: &str) -> NovaResult<Option<NamespaceRecord>> {
        let row = sqlx::query(&format!("SELECT {NS_COLUMNS} FROM namespaces WHERE name = ?1"))
            .bind(name)
            .fetch_optional(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        match row {
            Some(r) => Ok(Some(row_to_namespace(&r)?)),
            None => Ok(None),
        }
    }

    async fn list(
        &self,
        db: &Database,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<NamespaceRecord>> {
        let limit = limit.min(10_000) as i64;
        let offset = offset as i64;
        let rows = sqlx::query(&format!(
            "SELECT {NS_COLUMNS} FROM namespaces ORDER BY id ASC LIMIT ? OFFSET ?"
        ))
        .bind(limit)
        .bind(offset)
        .fetch_all(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(row_to_namespace(&row)?);
        }
        Ok(out)
    }

    async fn count_all(&self, db: &Database) -> NovaResult<i64> {
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM namespaces")
            .fetch_one(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        Ok(n)
    }

    async fn update(
        &self,
        db: &Database,
        input: UpdateNamespaceInput<'_>,
    ) -> NovaResult<NamespaceRecord> {
        let name = input.name.trim();
        if name.is_empty() {
            return Err(NovaError::validation("namespace.name must not be empty"));
        }
        if name != input.name {
            return Err(NovaError::validation(
                "namespace.name must not contain leading/trailing whitespace",
            ));
        }
        let pool = &db.pool;
        let current = self
            .get_by_name(db, name)
            .await?
            .ok_or_else(|| NovaError::not_found(format!("namespace {name}")))?;
        let now = Utc::now().timestamp();
        let config_json =
            input.config.cloned().unwrap_or_else(|| current.config.clone()).to_string();
        let desc: Option<&str> = match input.description {
            Some(d) if d.trim().is_empty() => None,
            Some(d) => Some(d.trim()),
            None => current.description.as_deref(),
        };
        sqlx::query(
            "UPDATE namespaces SET description = ?1, config_json = ?2, updated_at = ?3 WHERE id = \
             ?4",
        )
        .bind(desc)
        .bind(config_json)
        .bind(now)
        .bind(current.id)
        .execute(pool)
        .await
        .map_err(NovaError::storage)?;
        self.get_by_id(db, current.id).await
    }

    async fn delete(&self, db: &Database, name: &str) -> NovaResult<DeleteNamespaceOutcome> {
        let name = name.trim();
        if name.is_empty() {
            return Err(NovaError::validation("namespace.name must not be empty"));
        }
        if name.eq_ignore_ascii_case(DEFAULT_NAMESPACE_NAME) {
            return Ok(DeleteNamespaceOutcome::Protected("default"));
        }
        let pool = &db.pool;
        let exists: Option<(i64,)> = sqlx::query_as("SELECT id FROM namespaces WHERE name = ?1")
            .bind(name)
            .fetch_optional(pool)
            .await
            .map_err(NovaError::storage)?;
        let Some((id,)) = exists else {
            return Ok(DeleteNamespaceOutcome::NotFound);
        };
        let mut tx = db.begin().await?;
        // Clean up sqlite-vec vectors that belong to this namespace before
        // deleting the tenant rows; otherwise the virtual table keeps orphaned
        // vectors on disk even though the parent memories are gone. The virtual
        // table is only created lazily when the vector store is initialized, so
        // skip the cleanup when it has never been created.
        #[cfg(feature = "sqlite-vec")]
        {
            let has_vec_table: Option<(i64,)> =
                sqlx::query_as("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1")
                    .bind(crate::storage::vector_vec::SqliteVecVectorStore::TABLE)
                    .fetch_optional(&mut *tx)
                    .await
                    .map_err(NovaError::storage)?;
            if has_vec_table.is_some() {
                sqlx::query(&format!(
                    "DELETE FROM {} WHERE memory_uuid IN (SELECT uuid FROM memory_items WHERE \
                     namespace_id = ?1)",
                    crate::storage::vector_vec::SqliteVecVectorStore::TABLE
                ))
                .bind(id)
                .execute(&mut *tx)
                .await
                .map_err(NovaError::storage)?;
            }
        }
        sqlx::query("DELETE FROM entities WHERE namespace_id = ?1")
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(NovaError::storage)?;
        sqlx::query("DELETE FROM tags WHERE namespace_id = ?1")
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(NovaError::storage)?;
        sqlx::query("DELETE FROM memory_items WHERE namespace_id = ?1")
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(NovaError::storage)?;
        sqlx::query("DELETE FROM namespaces WHERE id = ?1")
            .bind(id)
            .execute(&mut *tx)
            .await
            .map_err(NovaError::storage)?;
        tx.commit().await.map_err(NovaError::from)?;
        Ok(DeleteNamespaceOutcome::Deleted(id))
    }

    async fn resolve(&self, db: &Database, name: &str) -> NovaResult<Option<NamespaceRecord>> {
        let name = name.trim();
        if name.is_empty() || name.eq_ignore_ascii_case(DEFAULT_NAMESPACE_NAME) {
            return self.get_by_name(db, DEFAULT_NAMESPACE_NAME).await;
        }
        self.get_by_name(db, name).await
    }
}

#[inline]
pub fn resolve_namespace_name(map: &Option<serde_json::Value>, aliases: &[(&str, &str)]) -> String {
    let value = map.as_ref();
    for (key, default) in aliases {
        if let Some(v) = value.and_then(|m| m.get(*key)) {
            if let Some(s) = v.as_str() {
                if !s.trim().is_empty() {
                    return s.trim().to_string();
                }
            }
        }
        if !default.is_empty() {
            return default.to_string();
        }
    }
    DEFAULT_NAMESPACE_NAME.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::StorageConfig;

    async fn temp_db() -> Database {
        let dir = std::env::temp_dir().join(format!("yq-nova-m2-ns-{}", Uuid::new_v4()));
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
    async fn delete_removes_tenant_data() {
        let db = temp_db().await;
        let repo = SqliteNamespaceRepository::new();
        let created = repo
            .create(
                &db,
                CreateNamespaceInput {
                    name: "team-a".trim(),
                    description: None,
                    config: None,
                },
            )
            .await
            .unwrap();
        let ns_id = created.id;
        sqlx::query(
            "INSERT INTO memory_items (uuid, content, content_hash, namespace_id, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )
        .bind("00000000-0000-0000-0000-00000000000a")
        .bind("data")
        .bind("h")
        .bind(ns_id)
        .bind(1)
        .execute(&db.pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO entities (uuid, namespace_id, name, type, created_at, updated_at) VALUES \
             (?1, ?2, ?3, ?4, ?5, ?5)",
        )
        .bind("00000000-0000-0000-0000-0000000000ab")
        .bind(ns_id)
        .bind("ent")
        .bind("person")
        .bind(1)
        .execute(&db.pool)
        .await
        .unwrap();
        sqlx::query("INSERT INTO tags (namespace_id, name, created_at) VALUES (?1, ?2, ?3)")
            .bind(ns_id)
            .bind("t")
            .bind(1)
            .execute(&db.pool)
            .await
            .unwrap();

        let out = repo.delete(&db, "team-a").await.unwrap();
        assert!(matches!(out, DeleteNamespaceOutcome::Deleted(id) if id == ns_id));

        let ns: Option<(i64,)> = sqlx::query_as("SELECT id FROM namespaces WHERE id = ?1")
            .bind(ns_id)
            .fetch_optional(&db.pool)
            .await
            .unwrap();
        assert!(ns.is_none());
        let mem: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM memory_items WHERE namespace_id = ?1")
                .bind(ns_id)
                .fetch_one(&db.pool)
                .await
                .unwrap();
        assert_eq!(mem, 0);
        let ent: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM entities WHERE namespace_id = ?1")
            .bind(ns_id)
            .fetch_one(&db.pool)
            .await
            .unwrap();
        assert_eq!(ent, 0);
        let tag: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tags WHERE namespace_id = ?1")
            .bind(ns_id)
            .fetch_one(&db.pool)
            .await
            .unwrap();
        assert_eq!(tag, 0);
    }

    #[tokio::test]
    async fn default_namespace_accepts_config_update() {
        let db = temp_db().await;
        let repo = SqliteNamespaceRepository::new();
        let upd = repo
            .update(
                &db,
                UpdateNamespaceInput {
                    name: "default",
                    description: Some("updated"),
                    config: Some(&serde_json::json!({"retention_days": 30})),
                },
            )
            .await
            .unwrap();
        assert_eq!(upd.description.as_deref(), Some("updated"));
        assert_eq!(upd.id, DEFAULT_NAMESPACE_ID);
    }

    #[tokio::test]
    async fn update_preserves_omitted_fields() {
        let db = temp_db().await;
        let repo = SqliteNamespaceRepository::new();
        let created = repo
            .create(
                &db,
                CreateNamespaceInput {
                    name: "team-b",
                    description: Some("original desc"),
                    config: Some(&serde_json::json!({"retention_days": 7, "qps": 10})),
                },
            )
            .await
            .unwrap();

        // Update only the description; the config must be preserved.
        let only_desc = repo
            .update(
                &db,
                UpdateNamespaceInput {
                    name: "team-b",
                    description: Some("new desc"),
                    config: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(only_desc.description.as_deref(), Some("new desc"));
        assert_eq!(only_desc.config, created.config);

        // Update only the config; the description must be preserved.
        let only_cfg = repo
            .update(
                &db,
                UpdateNamespaceInput {
                    name: "team-b",
                    description: None,
                    config: Some(&serde_json::json!({"retention_days": 30})),
                },
            )
            .await
            .unwrap();
        assert_eq!(only_cfg.config, serde_json::json!({"retention_days": 30}));
        assert_eq!(only_cfg.description.as_deref(), Some("new desc"));
    }
}
