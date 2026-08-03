//! HTTP handlers for `/v1/graph` routes.
//!
//! M4.3: entities (POST upsert / GET list)
//!       relations (POST upsert / GET list)
//!       traverse (POST — BFS from entity)

use axum::{
    Json,
    extract::{Query, State},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use yq_nova_core::{
    graph::{GraphExtractOpts, GraphService, LinkResult, TraverseOpts},
    storage::{
        EntityRecord, EntityRepository, InsertRelationInput, InsertRelationOutcome, RelationRecord,
        RelationRepository, SqliteEntityRepository, SqliteRelationRepository, TraverseNode,
        UpsertEntityInput, UpsertOutcome,
    },
};

use crate::http::{AppError, AppState, Result};

// ---------------------------------------------------------------------------
// Request / response types.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UpsertEntityRequest {
    pub name: String,
    #[serde(rename = "type")]
    pub entity_type: String,
    pub description: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertEntityResponse {
    #[serde(flatten)]
    pub outcome: UpsertOutcome,
    pub entity: EntityRecord,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListEntitiesQuery {
    pub name_prefix: Option<String>,
    pub entity_type: Option<String>,
    #[serde(default = "default_limit_200")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_limit_200() -> usize {
    200
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UpsertRelationRequest {
    pub source_uuid: Uuid,
    pub target_uuid: Uuid,
    pub predicate: String,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    pub metadata: Option<serde_json::Value>,
    #[serde(default = "default_true")]
    pub idempotent: bool,
    pub memory_uuid: Option<Uuid>,
}
fn default_confidence() -> f32 {
    1.0
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize)]
pub struct UpsertRelationResponse {
    pub inserted: bool,
    pub updated: bool,
    pub relation_uuid: Uuid,
    pub relation: RelationRecord,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListRelationsQuery {
    pub source: Option<Uuid>,
    pub target: Option<Uuid>,
    pub predicate: Option<String>,
    #[serde(default = "default_limit_200")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TraverseRequest {
    pub start: Uuid,
    #[serde(default = "default_max_depth")]
    pub max_depth: u8,
    #[serde(default = "default_max_nodes")]
    pub max_nodes: usize,
    #[serde(default)]
    pub predicate_whitelist: Vec<String>,
    #[serde(default)]
    pub min_confidence: f32,
}
fn default_max_depth() -> u8 {
    3
}
fn default_max_nodes() -> usize {
    100
}

impl Default for TraverseRequest {
    fn default() -> Self {
        Self {
            start: Uuid::nil(),
            max_depth: default_max_depth(),
            max_nodes: default_max_nodes(),
            predicate_whitelist: vec![],
            min_confidence: 0.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default)]
pub struct ExtractLinkRequest {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub opts: GraphExtractOpts,
}

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

pub async fn upsert_entity(
    State(state): State<AppState>,
    Json(req): Json<UpsertEntityRequest>,
) -> Result<Json<UpsertEntityResponse>> {
    let repo = SqliteEntityRepository::new();
    let db = &state.db;
    let input = UpsertEntityInput {
        name: req.name.trim(),
        r#type: req.entity_type.trim(),
        description: req.description.as_deref(),
        metadata: req.metadata.as_ref(),
    };
    let outcome = repo.upsert(db, input).await?;
    let entity = repo.get_by_uuid(db, outcome.uuid()).await?;
    Ok(Json(UpsertEntityResponse { outcome, entity }))
}

pub async fn list_entities(
    State(state): State<AppState>,
    Query(q): Query<ListEntitiesQuery>,
) -> Result<Json<Vec<EntityRecord>>> {
    let repo = SqliteEntityRepository::new();
    let db = &state.db;
    let limit = q.limit.min(500);
    // 简单 offset 分页：list 本身没 offset，这里我们 limit+offset 就用 offset 跳过。
    let rows = repo
        .list(
            db,
            q.name_prefix.as_deref(),
            q.entity_type.as_deref(),
            limit.saturating_add(q.offset),
            0,
        )
        .await?
        .into_iter()
        .skip(q.offset)
        .take(limit)
        .collect();
    Ok(Json(rows))
}

pub async fn upsert_relation(
    State(state): State<AppState>,
    Json(req): Json<UpsertRelationRequest>,
) -> Result<Json<UpsertRelationResponse>> {
    let repo = SqliteRelationRepository::new();
    let db = &state.db;
    let predicate = req.predicate.trim().to_string();
    let input = InsertRelationInput {
        source_uuid: req.source_uuid,
        target_uuid: req.target_uuid,
        predicate: &predicate,
        confidence: req.confidence,
        memory_uuid: req.memory_uuid,
        metadata: req.metadata.as_ref(),
        idempotent: req.idempotent,
    };
    let outcome = repo.insert(db, input).await?;
    let relation_uuid = outcome.uuid();

    // Fetch the relation: try listing outgoing with predicate filter.
    let list = repo.list_outgoing(db, req.source_uuid, Some(predicate.as_str()), 50).await?;
    let relation = list
        .into_iter()
        .find(|r| r.target_uuid == req.target_uuid && r.predicate.trim() == predicate)
        .ok_or_else(|| {
            AppError::from(yq_nova_core::error::NovaError::internal(
                "relation was inserted but could not be re-read",
            ))
        })?;

    let inserted = matches!(outcome, InsertRelationOutcome::Inserted(_));
    let updated = matches!(outcome, InsertRelationOutcome::Updated(_));
    Ok(Json(UpsertRelationResponse { inserted, updated, relation_uuid, relation }))
}

pub async fn list_relations(
    State(state): State<AppState>,
    Query(q): Query<ListRelationsQuery>,
) -> Result<Json<Vec<RelationRecord>>> {
    let repo = SqliteRelationRepository::new();
    let db = &state.db;
    let limit = q.limit.min(500);
    let total_limit = limit.saturating_add(q.offset);

    let mut rows: Vec<RelationRecord> = Vec::new();

    if let Some(src) = q.source {
        let got = repo.list_outgoing(db, src, q.predicate.as_deref(), total_limit).await?;
        if let Some(tgt) = q.target {
            rows.extend(got.into_iter().filter(|r| r.target_uuid == tgt));
        } else {
            rows = got;
        }
    } else if let Some(tgt) = q.target {
        let got = repo.list_incoming(db, tgt, q.predicate.as_deref(), total_limit).await?;
        rows = got;
    } else {
        // 无 source/target，列出前 N 个已知实体的 outgoing，避免全表扫描需要 FromRow。
        let ent_repo = SqliteEntityRepository::new();
        let ents = ent_repo.list(db, None, None, 50, 0).await?;
        for e in ents {
            if rows.len() >= total_limit {
                break;
            }
            let mut got =
                repo.list_outgoing(db, e.uuid, q.predicate.as_deref(), total_limit).await?;
            rows.append(&mut got);
        }
        rows.sort_by_key(|r| std::cmp::Reverse(r.id));
        rows.dedup_by_key(|r| r.id);
    }

    let trimmed: Vec<RelationRecord> = rows.into_iter().skip(q.offset).take(limit).collect();
    Ok(Json(trimmed))
}

pub async fn traverse(
    State(state): State<AppState>,
    Json(req): Json<TraverseRequest>,
) -> Result<Json<Vec<TraverseNode>>> {
    let svc: &GraphService = &state.graph;
    let opts = TraverseOpts {
        max_depth: req.max_depth,
        max_nodes: req.max_nodes,
        predicate_whitelist: req.predicate_whitelist,
        min_confidence: req.min_confidence,
    };
    let nodes = svc.traverse_graph(req.start, opts).await?;
    Ok(Json(nodes))
}

pub async fn extract_and_link(
    State(state): State<AppState>,
    Json(req): Json<ExtractLinkRequest>,
) -> Result<Json<LinkResult>> {
    let svc: &GraphService = &state.graph;
    let out = svc.extract_and_link(&req.text, &req.opts).await?;
    Ok(Json(out))
}
