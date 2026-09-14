use std::collections::{HashMap, HashSet, VecDeque};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

use crate::{
    error::{NovaError, NovaResult},
    storage::{
        Database, Repository,
        entity::{Direction, EntityRecord, EntityRepository, TraverseNode},
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationRecord {
    pub id: i64,

    pub uuid: Uuid,

    pub source_uuid: Uuid,

    pub target_uuid: Uuid,

    pub predicate: String,

    pub confidence: f32,

    pub memory_uuid: Option<Uuid>,

    pub metadata: serde_json::Value,

    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct InsertRelationInput<'a> {
    pub namespace_id: i64,

    pub source_uuid: Uuid,

    pub target_uuid: Uuid,

    pub predicate: &'a str,

    pub confidence: f32,

    pub memory_uuid: Option<Uuid>,

    pub metadata: Option<&'a serde_json::Value>,

    pub idempotent: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertRelationOutcome {
    Inserted(Uuid),

    Updated(Uuid),

    Unchanged(Uuid),
}

impl InsertRelationOutcome {
    pub fn uuid(&self) -> Uuid {
        match self {
            InsertRelationOutcome::Inserted(u)
            | InsertRelationOutcome::Updated(u)
            | InsertRelationOutcome::Unchanged(u) => *u,
        }
    }
    pub fn is_inserted(&self) -> bool {
        matches!(self, InsertRelationOutcome::Inserted(_) | InsertRelationOutcome::Updated(_))
    }
}

#[async_trait]
pub trait RelationRepository: Repository<RelationRecord> {
    async fn insert(
        &self,
        db: &Database,
        input: InsertRelationInput<'_>,
    ) -> NovaResult<InsertRelationOutcome>;

    async fn get_by_uuid(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
    ) -> NovaResult<RelationRecord>;
    async fn delete(&self, db: &Database, namespace_id: i64, uuid: Uuid) -> NovaResult<()>;
    async fn delete_by_memory(
        &self,
        db: &Database,
        namespace_id: i64,
        memory_uuid: Uuid,
    ) -> NovaResult<usize>;

    async fn list_outgoing(
        &self,
        db: &Database,
        namespace_id: i64,
        entity_uuid: Uuid,
        predicate: Option<&str>,
        limit: usize,
    ) -> NovaResult<Vec<RelationRecord>>;

    async fn list_incoming(
        &self,
        db: &Database,
        namespace_id: i64,
        entity_uuid: Uuid,
        predicate: Option<&str>,
        limit: usize,
    ) -> NovaResult<Vec<RelationRecord>>;

    #[allow(clippy::too_many_arguments)]
    async fn bfs_traverse(
        &self,
        db: &Database,
        namespace_id: i64,
        start_entity: Uuid,
        direction: Direction,
        max_depth: u8,
        max_nodes: usize,
        predicate_whitelist: &[String],
        min_confidence: f32,
    ) -> NovaResult<Vec<TraverseNode>>;
}

#[derive(Clone)]
pub struct SqliteRelationRepository;

impl SqliteRelationRepository {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for SqliteRelationRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Repository<RelationRecord> for SqliteRelationRepository {
    fn name(&self) -> &'static str {
        "relation.sqlite"
    }
}

#[async_trait]
impl RelationRepository for SqliteRelationRepository {
    async fn insert(
        &self,
        db: &Database,
        input: InsertRelationInput<'_>,
    ) -> NovaResult<InsertRelationOutcome> {
        let pred = input.predicate.trim();
        if pred.is_empty() {
            return Err(NovaError::validation("relation.predicate must not be empty"));
        }
        if !(0.0..=1.0).contains(&input.confidence) {
            return Err(NovaError::validation("relation.confidence must be in [0.0, 1.0]"));
        }
        if input.source_uuid == input.target_uuid {
            return Err(NovaError::validation(
                "relation.source and relation.target must be different entities",
            ));
        }

        let pool = &db.pool;
        let ns = input.namespace_id;

        let src_exists: Option<(String,)> =
            sqlx::query_as("SELECT uuid FROM entities WHERE uuid = ?1 AND namespace_id = ?2")
                .bind(input.source_uuid.to_string())
                .bind(ns)
                .fetch_optional(pool)
                .await
                .map_err(NovaError::storage)?;
        if src_exists.is_none() {
            return Err(NovaError::validation(format!(
                "relation.source_uuid {0} does not exist; upsert entity {0} first",
                input.source_uuid
            )));
        }
        let tgt_exists: Option<(String,)> =
            sqlx::query_as("SELECT uuid FROM entities WHERE uuid = ?1 AND namespace_id = ?2")
                .bind(input.target_uuid.to_string())
                .bind(ns)
                .fetch_optional(pool)
                .await
                .map_err(NovaError::storage)?;
        if tgt_exists.is_none() {
            return Err(NovaError::validation(format!(
                "relation.target_uuid {0} does not exist; upsert entity {0} first",
                input.target_uuid
            )));
        }

        let metadata_json = input.metadata.cloned().unwrap_or_else(|| serde_json::json!({}));
        let memory_uuid_s: Option<String> = input.memory_uuid.map(|u| u.to_string());

        if input.idempotent {
            let existing: Option<(i64, String, f64)> = sqlx::query_as(
                "SELECT id, uuid, confidence FROM relations WHERE source_uuid = ?1 AND predicate \
                 = ?2 AND target_uuid = ?3 AND namespace_id = ?4",
            )
            .bind(input.source_uuid.to_string())
            .bind(pred)
            .bind(input.target_uuid.to_string())
            .bind(ns)
            .fetch_optional(pool)
            .await
            .map_err(NovaError::storage)?;
            if let Some((_id, uuid_s, existing_conf)) = existing {
                let existing_uuid = Uuid::parse_str(&uuid_s)
                    .map_err(|e| NovaError::storage_msg(format!("bad rel uuid: {e}")))?;
                if (existing_conf as f32) >= input.confidence {
                    return Ok(InsertRelationOutcome::Unchanged(existing_uuid));
                }
                sqlx::query(
                    "UPDATE relations SET confidence = ?1, memory_uuid = ?2, metadata_json = ?3 \
                     WHERE uuid = ?4 AND namespace_id = ?5",
                )
                .bind(input.confidence as f64)
                .bind(&memory_uuid_s)
                .bind(metadata_json.to_string())
                .bind(uuid_s)
                .bind(ns)
                .execute(pool)
                .await
                .map_err(NovaError::storage)?;
                return Ok(InsertRelationOutcome::Updated(existing_uuid));
            }
        }

        let uuid = Uuid::new_v4();
        let now = Utc::now().timestamp();
        sqlx::query(
            "INSERT INTO relations (uuid, namespace_id, source_uuid, target_uuid, predicate, \
             confidence, memory_uuid, metadata_json, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, \
             ?7, ?8, ?9)",
        )
        .bind(uuid.to_string())
        .bind(ns)
        .bind(input.source_uuid.to_string())
        .bind(input.target_uuid.to_string())
        .bind(pred)
        .bind(input.confidence as f64)
        .bind(&memory_uuid_s)
        .bind(metadata_json.to_string())
        .bind(now)
        .execute(pool)
        .await
        .map_err(NovaError::storage)?;
        Ok(InsertRelationOutcome::Inserted(uuid))
    }

    async fn get_by_uuid(
        &self,
        db: &Database,
        namespace_id: i64,
        uuid: Uuid,
    ) -> NovaResult<RelationRecord> {
        let row = sqlx::query(
            "SELECT id, uuid, source_uuid, target_uuid, predicate, confidence, memory_uuid, \
             metadata_json, created_at FROM relations WHERE uuid = ?1 AND namespace_id = ?2",
        )
        .bind(uuid.to_string())
        .bind(namespace_id)
        .fetch_optional(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        let row = row.ok_or_else(|| NovaError::not_found(format!("relation {uuid}")))?;
        row_to_relation(&row)
    }

    async fn delete(&self, db: &Database, namespace_id: i64, uuid: Uuid) -> NovaResult<()> {
        let res = sqlx::query("DELETE FROM relations WHERE uuid = ?1 AND namespace_id = ?2")
            .bind(uuid.to_string())
            .bind(namespace_id)
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        if res.rows_affected() == 0 {
            return Err(NovaError::not_found(format!("relation {uuid}")));
        }
        Ok(())
    }

    async fn delete_by_memory(
        &self,
        db: &Database,
        namespace_id: i64,
        memory_uuid: Uuid,
    ) -> NovaResult<usize> {
        let res = sqlx::query("DELETE FROM relations WHERE memory_uuid = ?1 AND namespace_id = ?2")
            .bind(memory_uuid.to_string())
            .bind(namespace_id)
            .execute(&db.pool)
            .await
            .map_err(NovaError::storage)?;
        Ok(res.rows_affected() as usize)
    }

    async fn list_outgoing(
        &self,
        db: &Database,
        namespace_id: i64,
        entity_uuid: Uuid,
        predicate: Option<&str>,
        limit: usize,
    ) -> NovaResult<Vec<RelationRecord>> {
        let (sql, bind_pred): (String, bool) = match predicate {
            Some(p) if !p.is_empty() => (
                "SELECT id, uuid, source_uuid, target_uuid, predicate, confidence, memory_uuid, \
                 metadata_json, created_at FROM relations WHERE source_uuid = ?1 AND predicate = \
                 ?2 AND namespace_id = ?3 ORDER BY confidence DESC LIMIT ?4"
                    .into(),
                true,
            ),
            _ => (
                "SELECT id, uuid, source_uuid, target_uuid, predicate, confidence, memory_uuid, \
                 metadata_json, created_at FROM relations WHERE source_uuid = ?1 AND namespace_id \
                 = ?2 ORDER BY confidence DESC LIMIT ?3"
                    .into(),
                false,
            ),
        };
        let mut q = sqlx::query(&sql).bind(entity_uuid.to_string());
        if bind_pred {
            q = q.bind(predicate.unwrap());
        }
        q = q.bind(namespace_id);
        q = q.bind(limit.min(10_000) as i64);
        let rows = q.fetch_all(&db.pool).await.map_err(NovaError::storage)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(row_to_relation(&row)?);
        }
        Ok(out)
    }

    async fn list_incoming(
        &self,
        db: &Database,
        namespace_id: i64,
        entity_uuid: Uuid,
        predicate: Option<&str>,
        limit: usize,
    ) -> NovaResult<Vec<RelationRecord>> {
        let (sql, bind_pred): (String, bool) = match predicate {
            Some(p) if !p.is_empty() => (
                "SELECT id, uuid, source_uuid, target_uuid, predicate, confidence, memory_uuid, \
                 metadata_json, created_at FROM relations WHERE target_uuid = ?1 AND predicate = \
                 ?2 AND namespace_id = ?3 ORDER BY confidence DESC LIMIT ?4"
                    .into(),
                true,
            ),
            _ => (
                "SELECT id, uuid, source_uuid, target_uuid, predicate, confidence, memory_uuid, \
                 metadata_json, created_at FROM relations WHERE target_uuid = ?1 AND namespace_id \
                 = ?2 ORDER BY confidence DESC LIMIT ?3"
                    .into(),
                false,
            ),
        };
        let mut q = sqlx::query(&sql).bind(entity_uuid.to_string());
        if bind_pred {
            q = q.bind(predicate.unwrap());
        }
        q = q.bind(namespace_id);
        q = q.bind(limit.min(10_000) as i64);
        let rows = q.fetch_all(&db.pool).await.map_err(NovaError::storage)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(row_to_relation(&row)?);
        }
        Ok(out)
    }

    async fn bfs_traverse(
        &self,
        db: &Database,
        namespace_id: i64,
        start_entity: Uuid,
        direction: Direction,
        max_depth: u8,
        max_nodes: usize,
        predicate_whitelist: &[String],
        min_confidence: f32,
    ) -> NovaResult<Vec<TraverseNode>> {
        let max_depth = max_depth.min(8);
        let max_nodes = max_nodes.min(1024);

        let entity_repo = crate::storage::entity::SqliteEntityRepository::new();
        let start =
            entity_repo.get_by_uuid(db, namespace_id, start_entity).await.map_err(|_| {
                NovaError::validation(format!("bfs start_entity {start_entity} does not exist"))
            })?;

        let all_edges: Vec<(String, String, String, f64)> = sqlx::query_as(
            "SELECT source_uuid, target_uuid, predicate, confidence FROM relations WHERE \
             namespace_id = ?1",
        )
        .bind(namespace_id)
        .fetch_all(&db.pool)
        .await
        .map_err(NovaError::storage)?;
        let mut out_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut in_map: HashMap<String, Vec<String>> = HashMap::new();
        for (s, t, p, c) in all_edges {
            if !predicate_whitelist.is_empty() && !predicate_whitelist.contains(&p) {
                continue;
            }
            let confidence = c as f32;
            if confidence < min_confidence {
                continue;
            }
            out_map.entry(s.clone()).or_default().push(t.clone());
            in_map.entry(t).or_default().push(s);
        }

        let start_s = start_entity.to_string();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start_s.clone());
        let mut queue: VecDeque<(String, u8, Vec<Uuid>)> = VecDeque::new();
        queue.push_back((start_s, 0, vec![start_entity]));

        let mut results: Vec<TraverseNode> = vec![TraverseNode {
            entity: start,
            depth: 0,
            path: vec![start_entity],
        }];

        while let Some((node_s, depth, path)) = queue.pop_front() {
            if depth >= max_depth || results.len() >= max_nodes {
                continue;
            }
            let mut neighbours: Vec<String> = Vec::new();
            if matches!(direction, Direction::Out | Direction::Both) {
                if let Some(outs) = out_map.get(&node_s) {
                    neighbours.extend(outs.iter().cloned());
                }
            }
            if matches!(direction, Direction::In | Direction::Both) {
                if let Some(ins) = in_map.get(&node_s) {
                    neighbours.extend(ins.iter().cloned());
                }
            }
            for nb_s in neighbours {
                if visited.contains(&nb_s) {
                    continue;
                }
                visited.insert(nb_s.clone());
                if results.len() >= max_nodes {
                    break;
                }
                let nb_uuid = Uuid::parse_str(&nb_s)
                    .map_err(|e| NovaError::storage_msg(format!("bad nb uuid: {e}")))?;
                let nb_ent: EntityRecord =
                    match entity_repo.get_by_uuid(db, namespace_id, nb_uuid).await {
                        Ok(e) => e,
                        Err(_) => continue,
                    };
                let mut nb_path = path.clone();
                nb_path.push(nb_uuid);
                results.push(TraverseNode {
                    entity: nb_ent,
                    depth: depth + 1,
                    path: nb_path.clone(),
                });
                queue.push_back((nb_s, depth + 1, nb_path));
            }
        }
        Ok(results)
    }
}

fn row_to_relation(row: &sqlx::sqlite::SqliteRow) -> NovaResult<RelationRecord> {
    let id: i64 = row.try_get("id").map_err(NovaError::storage)?;
    let uuid_s: String = row.try_get("uuid").map_err(NovaError::storage)?;
    let uuid = Uuid::parse_str(&uuid_s)
        .map_err(|e| NovaError::storage_msg(format!("bad rel uuid: {e}")))?;
    let src_s: String = row.try_get("source_uuid").map_err(NovaError::storage)?;
    let source_uuid = Uuid::parse_str(&src_s)
        .map_err(|e| NovaError::storage_msg(format!("bad src uuid: {e}")))?;
    let tgt_s: String = row.try_get("target_uuid").map_err(NovaError::storage)?;
    let target_uuid = Uuid::parse_str(&tgt_s)
        .map_err(|e| NovaError::storage_msg(format!("bad tgt uuid: {e}")))?;
    let predicate: String = row.try_get("predicate").map_err(NovaError::storage)?;
    let confidence: f64 = row.try_get("confidence").map_err(NovaError::storage)?;
    let memory_s: Option<String> = row.try_get("memory_uuid").map_err(NovaError::storage)?;
    let memory_uuid: Option<Uuid> = match memory_s {
        Some(s) => Some(
            Uuid::parse_str(&s)
                .map_err(|e| NovaError::storage_msg(format!("bad mem uuid: {e}")))?,
        ),
        None => None,
    };
    let meta_s: String = row.try_get("metadata_json").map_err(NovaError::storage)?;
    let metadata: serde_json::Value = serde_json::from_str(&meta_s)
        .map_err(|e| NovaError::storage_msg(format!("rel meta json: {e}")))?;
    let created_ts: i64 = row.try_get("created_at").map_err(NovaError::storage)?;
    Ok(RelationRecord {
        id,
        uuid,
        source_uuid,
        target_uuid,
        predicate,
        confidence: confidence as f32,
        memory_uuid,
        metadata,
        created_at: ts_to_dt(created_ts)?,
    })
}

fn ts_to_dt(ts: i64) -> NovaResult<DateTime<Utc>> {
    DateTime::from_timestamp(ts, 0).ok_or_else(|| NovaError::storage_msg(format!("bad ts {ts}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::StorageConfig,
        storage::entity::{EntityRepository, SqliteEntityRepository, UpsertEntityInput},
    };

    const NS: i64 = crate::storage::namespace::DEFAULT_NAMESPACE_ID;

    async fn temp_db() -> Database {
        let dir = std::env::temp_dir().join(format!("yq-nova-m2-rel-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        Database::open(cfg).await.expect("open temp db")
    }

    async fn upsert_pair(db: &Database, a: (&str, &str), b: (&str, &str)) -> (Uuid, Uuid) {
        let er = SqliteEntityRepository::new();
        let au = er
            .upsert(
                db,
                UpsertEntityInput {
                    namespace_id: NS,
                    name: a.0,
                    r#type: a.1,
                    description: None,
                    metadata: None,
                },
            )
            .await
            .unwrap()
            .uuid();
        let bu = er
            .upsert(
                db,
                UpsertEntityInput {
                    namespace_id: NS,
                    name: b.0,
                    r#type: b.1,
                    description: None,
                    metadata: None,
                },
            )
            .await
            .unwrap()
            .uuid();
        (au, bu)
    }

    #[tokio::test]
    async fn insert_requires_existing_endpoints_and_validates_confidence() {
        let db = temp_db().await;
        let rr = SqliteRelationRepository::new();
        let bad_pred = rr
            .insert(
                &db,
                InsertRelationInput {
                    namespace_id: NS,
                    source_uuid: Uuid::new_v4(),
                    target_uuid: Uuid::new_v4(),
                    predicate: "",
                    confidence: 1.0,
                    memory_uuid: None,
                    metadata: None,
                    idempotent: false,
                },
            )
            .await
            .unwrap_err();
        assert_eq!(bad_pred.code(), crate::error::ErrorCode::Validation);

        let (a, b) = upsert_pair(&db, ("Alice", "person"), ("ACME", "org")).await;
        let bad_src = rr
            .insert(
                &db,
                InsertRelationInput {
                    namespace_id: NS,
                    source_uuid: Uuid::new_v4(),
                    target_uuid: b,
                    predicate: "works_at",
                    confidence: 1.5,
                    memory_uuid: None,
                    metadata: None,
                    idempotent: false,
                },
            )
            .await
            .unwrap_err();
        assert_eq!(bad_src.code(), crate::error::ErrorCode::Validation);

        let bad_conf = rr
            .insert(
                &db,
                InsertRelationInput {
                    namespace_id: NS,
                    source_uuid: a,
                    target_uuid: b,
                    predicate: "works_at",
                    confidence: 1.5,
                    memory_uuid: None,
                    metadata: None,
                    idempotent: false,
                },
            )
            .await
            .unwrap_err();
        assert_eq!(bad_conf.code(), crate::error::ErrorCode::Validation);

        let ok = rr
            .insert(
                &db,
                InsertRelationInput {
                    namespace_id: NS,
                    source_uuid: a,
                    target_uuid: b,
                    predicate: "works_at",
                    confidence: 0.9,
                    memory_uuid: None,
                    metadata: None,
                    idempotent: false,
                },
            )
            .await
            .unwrap();
        rr.get_by_uuid(&db, NS, ok.uuid()).await.unwrap();
    }

    #[tokio::test]
    async fn idempotent_insert_updates_existing_edge() {
        let db = temp_db().await;
        let rr = SqliteRelationRepository::new();
        let (a, b) = upsert_pair(&db, ("X", "p"), ("Y", "p")).await;
        let u1 = rr
            .insert(
                &db,
                InsertRelationInput {
                    namespace_id: NS,
                    source_uuid: a,
                    target_uuid: b,
                    predicate: "knows",
                    confidence: 0.4,
                    memory_uuid: None,
                    metadata: None,
                    idempotent: true,
                },
            )
            .await
            .unwrap();
        let u2 = rr
            .insert(
                &db,
                InsertRelationInput {
                    namespace_id: NS,
                    source_uuid: a,
                    target_uuid: b,
                    predicate: "knows",
                    confidence: 0.9,
                    memory_uuid: None,
                    metadata: Some(&serde_json::json!({"since":2024})),
                    idempotent: true,
                },
            )
            .await
            .unwrap();
        assert_eq!(u1.uuid(), u2.uuid());
        let got = rr.get_by_uuid(&db, NS, u1.uuid()).await.unwrap();
        assert!((got.confidence - 0.9).abs() < 1e-6);
        assert_eq!(got.metadata, serde_json::json!({"since":2024}));
    }

    #[tokio::test]
    async fn list_outgoing_and_incoming_and_bfs() {
        let db = temp_db().await;
        let rr = SqliteRelationRepository::new();
        let er = SqliteEntityRepository::new();
        let (a, b) = upsert_pair(&db, ("A", "x"), ("B", "x")).await;
        let (c, _) = {
            let cu = er
                .upsert(
                    &db,
                    UpsertEntityInput {
                        namespace_id: NS,
                        name: "C",
                        r#type: "x",
                        description: None,
                        metadata: None,
                    },
                )
                .await
                .unwrap()
                .uuid();
            (cu, a)
        };
        for (src, tgt, pred) in [(a, b, "a→b"), (a, c, "a→c"), (c, b, "c→b")] {
            rr.insert(
                &db,
                InsertRelationInput {
                    namespace_id: NS,
                    source_uuid: src,
                    target_uuid: tgt,
                    predicate: pred,
                    confidence: 1.0,
                    memory_uuid: None,
                    metadata: None,
                    idempotent: false,
                },
            )
            .await
            .unwrap();
        }
        let outs = rr.list_outgoing(&db, NS, a, None, 100).await.unwrap();
        assert_eq!(outs.len(), 2);
        let incs = rr.list_incoming(&db, NS, b, None, 100).await.unwrap();
        assert_eq!(incs.len(), 2);

        let nodes = rr.bfs_traverse(&db, NS, a, Direction::Out, 3, 100, &[], 0.0).await.unwrap();

        let ids: HashSet<Uuid> = nodes.iter().map(|n| n.entity.uuid).collect();
        assert_eq!(ids.len(), 3);
    }
}
