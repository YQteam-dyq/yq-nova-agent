//! HTTP handlers for the `/v1/memory` routes.
//!
//! M4.2: POST remember / POST recall / POST forget / GET :uuid / DELETE :uuid

use axum::{
    Json,
    extract::{Path, State},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use yq_nova_core::{
    memory::{
        MemoryService,
        ops_forget::ForgetInput,
        ops_recall::{RecallInput, RecallOutput},
        ops_remember::{RememberInput, RememberOutput},
    },
    storage::{MemoryFilter, MemorySource},
};

use crate::http::{AppError, AppState, Result};

// ---------------------------------------------------------------------------
// Request DTOs (owned variants — HTTP bodies cannot have borrows).
// ---------------------------------------------------------------------------

/// Body for `POST /v1/memory/remember`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct RememberRequest {
    pub content: String,
    pub source: MemorySource,
    pub importance: f32,
    pub metadata: Option<serde_json::Value>,
    pub expires_at: Option<DateTime<Utc>>,
    pub tags: Vec<String>,
    pub embed: bool,
    pub extract_graph: bool,
}

impl Default for RememberRequest {
    fn default() -> Self {
        Self {
            content: String::new(),
            source: MemorySource::Agent,
            importance: 0.5,
            metadata: None,
            expires_at: None,
            tags: vec![],
            embed: true,
            extract_graph: false, // MVP 默认不抽取图，避免误报
        }
    }
}

/// Body for `POST /v1/memory/recall`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct RecallRequest {
    pub query: String,
    pub top_k: usize,
    pub score_threshold: f32,
    pub similarity_threshold: f32,
    pub mode: yq_nova_core::memory::SearchMode,
    pub graph: yq_nova_core::memory::GraphTraversalOpts,
    pub hybrid_weights: Option<yq_nova_core::memory::HybridWeights>,
    pub rrf_k: Option<u32>,
    pub rank_weights: Option<yq_nova_core::memory::rank::RankWeights>,
    pub filter: MemoryFilter,
}

impl Default for RecallRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            top_k: 20,
            score_threshold: 0.0,
            similarity_threshold: -1.0,
            mode: Default::default(),
            graph: Default::default(),
            hybrid_weights: None,
            rrf_k: None,
            rank_weights: None,
            filter: MemoryFilter::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

pub async fn remember(
    State(state): State<AppState>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<RememberOutput>> {
    let svc: &MemoryService = &state.memory;
    let input = RememberInput {
        content: &req.content,
        source: req.source,
        importance: req.importance,
        metadata: req.metadata.as_ref(),
        expires_at: req.expires_at,
        tags: &req.tags,
        embed: req.embed,
        extract_graph: req.extract_graph,
    };
    let out = svc.remember(input).await?;
    Ok(Json(out))
}

pub async fn recall(
    State(state): State<AppState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<RecallOutput>> {
    let svc: &MemoryService = &state.memory;
    let input = RecallInput {
        query: &req.query,
        top_k: req.top_k,
        score_threshold: req.score_threshold,
        similarity_threshold: req.similarity_threshold,
        mode: req.mode,
        graph: req.graph.clone(),
        hybrid_weights: req.hybrid_weights,
        rrf_k: req.rrf_k,
        rank_weights: req.rank_weights,
        filter: req.filter.clone(),
    };
    let out: RecallOutput = svc.recall(input).await?;
    Ok(Json(out))
}

pub async fn forget(
    State(state): State<AppState>,
    Json(req): Json<ForgetInput>,
) -> Result<Json<yq_nova_core::memory::ops_forget::ForgetOutput>> {
    let svc: &MemoryService = &state.memory;
    let out = svc.forget(req).await?;
    Ok(Json(out))
}

pub async fn get_memory(
    State(state): State<AppState>,
    Path(uuid): Path<Uuid>,
) -> Result<Json<yq_nova_core::storage::MemoryRecord>> {
    let svc: &MemoryService = &state.memory;
    let mem = svc.get_memory(uuid).await.map_err(AppError::from)?;
    Ok(Json(mem))
}

pub async fn delete_memory(
    State(state): State<AppState>,
    Path(uuid): Path<Uuid>,
) -> Result<Json<yq_nova_core::memory::ops_forget::ForgetOutput>> {
    use yq_nova_core::memory::ops_forget::{ForgetInput, ForgetMode, ForgetTarget};

    let svc: &MemoryService = &state.memory;
    let input = ForgetInput {
        target: ForgetTarget::One(uuid),
        mode: ForgetMode::Hard, // DELETE /memory/:uuid => hard delete (no trace)
        gc_graph: false,
        batch_limit: 1,
    };
    let out = svc.forget(input).await?;
    Ok(Json(out))
}

// ---------------------------------------------------------------------------
// Tests (P8 集成测试 — 用 axum 的 tower ServiceExt::oneshot 直接测试路由).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::graph::UpsertEntityResponse;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use yq_nova_core::{
        config::StorageConfig,
        graph::extractor::RegexWikiExtractor,
        memory::MemoryService,
        storage::{Database, Migrator},
    };

    async fn make_router() -> (AppState, axum::Router) {
        let dir = std::env::temp_dir().join(format!(
            "yq-nova-test-api-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let db_path = dir.join("nova.sqlite");
        let storage_cfg = StorageConfig { db_path, ..Default::default() };
        let db = Database::open(storage_cfg.clone()).await.expect("open db");
        Migrator::run(&db.pool).await.expect("migrations");

        let embed = yq_nova_core::embedding::MockEmbeddingProvider::new(16);
        let shared_embed: std::sync::Arc<dyn yq_nova_core::embedding::EmbeddingProvider> =
            std::sync::Arc::new(embed);
        let memory = MemoryService::new(db.clone(), shared_embed);
        let graph = yq_nova_core::graph::GraphService::with_parts(
            db.clone(),
            std::sync::Arc::new(RegexWikiExtractor::new()),
        );

        let srv_cfg = yq_nova_core::config::ServerConfig::default();
        let state = AppState::new(srv_cfg, db, memory, graph);
        let router = crate::http::build_router(state.clone());
        (state, router)
    }

    #[tokio::test]
    async fn remember_accepts_and_returns_uuid() {
        let (_state, router) = make_router().await;

        let body = serde_json::json!({
            "content": "Alice loves apples and Python",
            "importance": 0.8,
            "tags": ["python", "alice"],
            "embed": true,
            "extract_graph": false,
        });
        let req = Request::builder()
            .uri("/v1/memory/remember")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "remember 200");

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let out: RememberOutput = serde_json::from_slice(&bytes).expect("remember output");
        assert!(!out.uuid.is_nil());
        assert!(!out.duplicate);
    }

    #[tokio::test]
    async fn remember_duplicate_on_same_content_returns_true() {
        let (_state, router) = make_router().await;

        let body = serde_json::json!({
            "content": "dedup-me",
            "importance": 0.5,
            "embed": true,
            "extract_graph": false,
        });

        let do_post = || async {
            let req = Request::builder()
                .uri("/v1/memory/remember")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let r = router.clone().oneshot(req).await.unwrap();
            let bytes = r.into_body().collect().await.unwrap().to_bytes();
            let out: RememberOutput = serde_json::from_slice(&bytes).unwrap();
            out
        };

        let first = do_post().await;
        let second = do_post().await;
        assert!(!first.duplicate);
        assert!(second.duplicate);
        assert_eq!(first.uuid, second.uuid);
    }

    #[tokio::test]
    async fn recall_without_graph_returns_memory_ranked_by_importance() {
        let (_state, router) = make_router().await;

        // 先插入两条，importance 不一样
        for (content, imp) in [("common task", 0.1), ("important user profile", 0.95)] {
            let body = serde_json::json!({
                "content": content,
                "importance": imp,
                "tags": ["recall_test"],
                "embed": true,
                "extract_graph": false,
            });
            let req = Request::builder()
                .uri("/v1/memory/remember")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            router.clone().oneshot(req).await.unwrap();
        }

        let body = serde_json::json!({
            "query": "anything",
            "top_k": 10,
            "graph": {"enabled": false},
        });
        let req = Request::builder()
            .uri("/v1/memory/recall")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let out: RecallOutput = serde_json::from_slice(&bytes).expect("recall output");
        assert!(!out.hits.is_empty(), "expect at least one recall hit");
        // 第一个应该是 imp=0.95 的那条
        let first_mem = &out.hits[0].memory;
        assert!(first_mem.importance > 0.8, "higher importance should rank first");
    }

    #[tokio::test]
    async fn get_memory_by_uuid_returns_record() {
        let (_state, router) = make_router().await;

        // insert
        let body = serde_json::json!({
            "content": "get-by-uuid content",
            "importance": 0.3,
            "tags": ["get_test"],
            "embed": true,
            "extract_graph": false,
        });
        let req = Request::builder()
            .uri("/v1/memory/remember")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let out: RememberOutput = serde_json::from_slice(&bytes).unwrap();

        let req = Request::builder()
            .uri(format!("/v1/memory/{}", out.uuid))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let mem: yq_nova_core::storage::MemoryRecord =
            serde_json::from_slice(&bytes).expect("memory record");
        assert_eq!(mem.uuid, out.uuid);
    }

    #[tokio::test]
    async fn forget_one_and_then_get_returns_404() {
        let (_state, router) = make_router().await;

        let body = serde_json::json!({
            "content": "forget-me",
            "importance": 0.5,
            "embed": true,
            "extract_graph": false,
        });
        let req = Request::builder()
            .uri("/v1/memory/remember")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let out: RememberOutput = serde_json::from_slice(&bytes).unwrap();

        let forget_body = serde_json::json!({
            "target": { "type": "one", "value": out.uuid.to_string() },
            "mode": "hard",
            "gc_graph": false,
            "batch_limit": 1,
        });
        let req = Request::builder()
            .uri("/v1/memory/forget")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(forget_body.to_string()))
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let forget_out: yq_nova_core::memory::ops_forget::ForgetOutput =
            serde_json::from_slice(&bytes).unwrap();
        assert_eq!(forget_out.affected_memories, 1);

        // GET now returns 404 (status = not_found)
        let req = Request::builder()
            .uri(format!("/v1/memory/{}", out.uuid))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn route_missing_returns_404_json() {
        let (_state, router) = make_router().await;
        let req =
            Request::builder().uri("/v1/does-not-exist").method("GET").body(Body::empty()).unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], "not_found");
    }

    // =========================================================================
    // M9: Keyword / Hybrid / Graph expansion recall.
    // =========================================================================

    async fn remember(router: &axum::Router, content: &str, tags: &[&str]) -> RememberOutput {
        let body = serde_json::json!({
            "content": content,
            "tags": tags,
            "importance": 0.5,
            "embed": true,
            "extract_graph": false,
        });
        let req = Request::builder()
            .uri("/v1/memory/remember")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice::<RememberOutput>(&bytes).unwrap()
    }

    async fn recall_req(router: &axum::Router, body: serde_json::Value) -> RecallOutput {
        let req = Request::builder()
            .uri("/v1/memory/recall")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice::<RecallOutput>(&bytes).unwrap()
    }

    #[tokio::test]
    async fn keyword_mode_returns_only_matching_docs() {
        let (_state, router) = make_router().await;
        remember(&router, "banana smoothie with coconut flakes", &["food"]).await;
        remember(&router, "rust sqlx async lifetime details", &["rust"]).await;
        remember(&router, "banana bread recipe from grandma", &["food"]).await;

        let out = recall_req(
            &router,
            serde_json::json!({
                "query": "banana",
                "top_k": 10,
                "mode": "keyword",
            }),
        )
        .await;
        let returned: Vec<String> = out.hits.iter().map(|h| h.memory.content.clone()).collect();
        assert_eq!(out.hits.len(), 2, "keyword 'banana' should match only 2 docs");
        assert!(returned.iter().any(|c| c.contains("smoothie")));
        assert!(returned.iter().any(|c| c.contains("bread")));
        assert!(returned.iter().all(|c| !c.contains("rust")), "rust doc must not appear");
    }

    #[tokio::test]
    async fn hybrid_mode_merges_keyword_and_semantic_sets() {
        let (_state, router) = make_router().await;
        // Only keyword matches "xyzzy_phrase":
        let k = remember(&router, "xyzzy_phrase unique token", &["k"]).await;
        // Only semantic matches (mock embedding returns small random distances
        // so "anything" often returns everything within fetch_k):
        let s = remember(&router, "different topic not containing magic word", &["s"]).await;

        // Keyword-only mode: only k.
        let kw = recall_req(
            &router,
            serde_json::json!({"query":"xyzzy_phrase","top_k":10,"mode":"keyword"}),
        )
        .await;
        let kw_uuids: std::collections::HashSet<_> =
            kw.hits.iter().map(|h| h.memory.uuid).collect();
        assert!(kw_uuids.contains(&k.uuid));
        assert!(!kw_uuids.contains(&s.uuid));

        // Hybrid with generous thresholds → union.
        let out = recall_req(
            &router,
            serde_json::json!({
                "query":"xyzzy_phrase",
                "top_k":10,
                "mode":"hybrid",
                "similarity_threshold": -1.0,
                "score_threshold": 0.0,
                "hybrid_weights": {"semantic": 0.5, "keyword": 0.5, "graph": 0.0},
            }),
        )
        .await;
        let hy_uuids: std::collections::HashSet<_> =
            out.hits.iter().map(|h| h.memory.uuid).collect();
        // Keyword must always be there (hard FTS match).
        assert!(hy_uuids.contains(&k.uuid), "hybrid must include keyword hit");
        // And hybrid total_candidates should be >= the keyword-only total on
        // the same query because semantic adds candidates.
        assert!(
            out.total_candidates >= kw.total_candidates,
            "hybrid >= keyword candidates (got {} vs {})",
            out.total_candidates,
            kw.total_candidates
        );
        let _ = (k, s);
    }

    #[tokio::test]
    async fn graph_expansion_pulls_in_related_memory_via_entity_edge() {
        let (state, router) = make_router().await;

        // 1. Upsert two entities.
        async fn upsert_ent(router: &axum::Router, name: &str, ty: &str) -> UpsertEntityResponse {
            let body = serde_json::json!({"name": name, "type": ty});
            let req = Request::builder()
                .uri("/v1/graph/entities")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let resp = router.clone().oneshot(req).await.unwrap();
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            serde_json::from_slice(&bytes).unwrap()
        }
        let alice = upsert_ent(&router, "Alice", "person").await.entity.uuid;
        let bob = upsert_ent(&router, "Bob", "person").await.entity.uuid;

        // 2. Remember memory #1 about Alice; remember memory #2 about Bob (no
        // shared keywords).
        let m1 = remember(&router, "alice_foo works on rust code", &["alice"]).await;
        let m2 = remember(&router, "bob_bar likes hiking on weekends", &["bob"]).await;

        // 3. Create TWO relations so m1 references Alice (seed) and m2
        // references Bob (target-for-expansion). Each row's memory_uuid is
        // what the graph expansion step scans to find "memories related to
        // entity X".
        for (src, tgt, mem, pred) in
            [(alice, bob, m1.uuid, "reports_to"), (alice, bob, m2.uuid, "shares_project_with")]
        {
            let rel_body = serde_json::json!({
                "source_uuid": src.to_string(),
                "target_uuid": tgt.to_string(),
                "predicate": pred,
                "confidence": 0.9,
                "memory_uuid": mem.to_string(),
                "idempotent": true,
            });
            let req = Request::builder()
                .uri("/v1/graph/relations")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(rel_body.to_string()))
                .unwrap();
            let r = router.clone().oneshot(req).await.unwrap();
            assert!(r.status().is_success(), "upsert relation should succeed, got {}", r.status());
        }
        let _ = state;

        // 4. Recall with mode=semantic + graph.enabled = true, querying
        // something that only matches m1's keywords.  Graph expansion follows
        // the Alice--Bob edge and returns memories that reference Bob (m2) as
        // `from_graph=true`.
        let out = recall_req(
            &router,
            serde_json::json!({
                "query": "alice_foo",
                "top_k": 10,
                "mode": "hybrid",
                "score_threshold": 0.0,
                "similarity_threshold": -1.0,
                "hybrid_weights": {"semantic": 0.4, "keyword": 0.4, "graph": 0.2},
                "graph": {"enabled": true, "max_depth": 2, "predicate_whitelist": []},
            }),
        )
        .await;
        let ids: std::collections::HashSet<_> = out.hits.iter().map(|h| h.memory.uuid).collect();
        assert!(ids.contains(&m1.uuid), "seed memory m1 must be present");
        // m2 should be pulled in via graph expansion.
        assert!(
            ids.contains(&m2.uuid),
            "m2 should be pulled in via graph expansion. Recall ids: {ids:?}. total_candidates={}",
            out.total_candidates
        );
        assert!(
            out.hits.iter().any(|h| h.from_graph),
            "at least one result should be marked from_graph"
        );
    }
}
