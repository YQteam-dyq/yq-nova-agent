use axum::{
    Json,
    extract::{Extension, Path, Query, State},
};
use serde::{Deserialize, Serialize};
use yq_nova_core::memory::ops_tag::{
    TagDeleteInput, TagDeleteOutput, TagListInput, TagListOutput, TagRenameInput, TagRenameOutput,
};

use crate::http::{AppState, Result, auth::NamespaceContext};

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ListTagsQuery {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RenameTagRequest {
    pub new_name: String,
}

pub async fn list_tags(
    State(state): State<AppState>,
    Extension(ns): Extension<NamespaceContext>,
    Query(query): Query<ListTagsQuery>,
) -> Result<Json<TagListOutput>> {
    let input = TagListInput {
        namespace_id: ns.id,
        limit: query.limit.unwrap_or(100),
        offset: query.offset.unwrap_or(0),
    };
    let out = state.memory.list_tags(input).await?;
    Ok(Json(out))
}

pub async fn rename_tag(
    State(state): State<AppState>,
    Extension(ns): Extension<NamespaceContext>,
    Path(name): Path<String>,
    Json(req): Json<RenameTagRequest>,
) -> Result<Json<TagRenameOutput>> {
    let input = TagRenameInput {
        namespace_id: ns.id,
        name,
        new_name: req.new_name,
    };
    let out = state.memory.rename_tag(input).await?;
    Ok(Json(out))
}

pub async fn delete_tag(
    State(state): State<AppState>,
    Extension(ns): Extension<NamespaceContext>,
    Path(name): Path<String>,
) -> Result<Json<TagDeleteOutput>> {
    let input = TagDeleteInput {
        namespace_id: ns.id,
        name,
    };
    let out = state.memory.delete_tag(input).await?;
    Ok(Json(out))
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use yq_nova_core::{
        config::{ServerConfig, StorageConfig},
        graph::extractor::RegexWikiExtractor,
        memory::MemoryService,
        storage::{Database, Migrator},
    };

    use super::*;

    async fn make_router() -> axum::Router {
        let dir = std::env::temp_dir().join(format!(
            "yq-nova-test-tags-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let db_path = dir.join("nova.sqlite");
        let storage_cfg = StorageConfig {
            db_path,
            ..Default::default()
        };
        let db = Database::open(storage_cfg).await.expect("open db");
        Migrator::run(&db.pool).await.expect("migrations");

        let embed = yq_nova_core::embedding::MockEmbeddingProvider::new(16);
        let shared_embed: std::sync::Arc<dyn yq_nova_core::embedding::EmbeddingProvider> =
            std::sync::Arc::new(embed);
        let memory = MemoryService::new(db.clone(), shared_embed);
        let graph = yq_nova_core::graph::GraphService::with_parts(
            db.clone(),
            std::sync::Arc::new(RegexWikiExtractor::new()),
        );

        let state = AppState::new(ServerConfig::default(), db, memory, graph);
        crate::http::build_router(state)
    }

    async fn remember_with_tags(router: &axum::Router, content: &str, tags: &[&str]) {
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
        assert_eq!(resp.status(), StatusCode::OK, "remember should return 200");
    }

    async fn get_tags(router: &axum::Router, query: &str) -> TagListOutput {
        let req = Request::builder()
            .uri(format!("/v1/tags{query}"))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "list tags should return 200");
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice::<TagListOutput>(&bytes).unwrap()
    }

    #[tokio::test]
    async fn list_tags_endpoint_reports_counts() {
        let router = make_router().await;
        remember_with_tags(&router, "tags alpha entry", &["shared", "one"]).await;
        remember_with_tags(&router, "tags beta entry", &["shared", "two"]).await;

        let out = get_tags(&router, "?limit=10").await;
        assert_eq!(out.total, 3);
        assert_eq!(out.count, 3);

        let shared = out.items.iter().find(|t| t.name == "shared").expect("shared tag present");
        assert_eq!(shared.memory_count, 2);
    }

    #[tokio::test]
    async fn rename_tag_endpoint_updates_associations() {
        let router = make_router().await;
        remember_with_tags(&router, "rename alpha entry", &["old"]).await;
        remember_with_tags(&router, "rename beta entry", &["old"]).await;

        let body = serde_json::json!({"new_name": "new"});
        let req = Request::builder()
            .uri("/v1/tags/old")
            .method("PATCH")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let out: TagRenameOutput = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(out.name, "old");
        assert_eq!(out.new_name, "new");

        let listed = get_tags(&router, "").await;
        assert!(listed.items.iter().any(|t| t.name == "new" && t.memory_count == 2));
        assert!(!listed.items.iter().any(|t| t.name == "old"));

        let missing = Request::builder()
            .uri("/v1/tags/absent")
            .method("PATCH")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::json!({"new_name": "other"}).to_string()))
            .unwrap();
        let resp = router.oneshot(missing).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn delete_tag_endpoint_detaches_memories() {
        let router = make_router().await;
        remember_with_tags(&router, "delete alpha entry", &["doomed"]).await;
        remember_with_tags(&router, "delete beta entry", &["doomed"]).await;

        let req =
            Request::builder().uri("/v1/tags/doomed").method("DELETE").body(Body::empty()).unwrap();
        let resp = router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let out: TagDeleteOutput = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(out.name, "doomed");
        assert!(out.deleted);
        assert_eq!(out.affected_memories, 2);

        let listed = get_tags(&router, "").await;
        assert_eq!(listed.total, 0);
    }
}
