//! API Token 鉴权中间件。
//!
//! 当 `ServerConfig::auth_token` 为空时不做任何鉴权（透传）；
//! 非空时要求每个请求的 `Authorization` 头为 `Bearer <token>`（或原始 token），
//! 否则返回 401 JSON 错误体。比较使用常量时间算法以降低时序攻击风险。

use axum::{
    Json,
    body::Body,
    extract::State,
    http::{Request, StatusCode, header},
    response::{IntoResponse, Response},
    middleware::Next,
};

use crate::http::{AppState, error::ErrorBody};

/// 常量时间比较两个字节切片是否相等（长度不同直接返回 false）。
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// 鉴权中间件：通过校验放行，否则返回 401。
pub async fn auth_middleware(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let configured = state.server_cfg.auth_token.clone();

    // 未配置 token：不做鉴权，直接放行。
    if configured.is_empty() {
        return next.run(req).await;
    }

    let provided = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(ToOwned::to_owned);

    // 提取待比较的 token：优先 `Bearer <token>`，否则把整个值当作 raw token。
    let candidate = provided
        .as_deref()
        .and_then(|v| {
            let trimmed = v.trim();
            if let Some(rest) = trimmed.strip_prefix("Bearer ") {
                Some(rest.trim())
            } else if let Some(rest) = trimmed.strip_prefix("bearer ") {
                Some(rest.trim())
            } else {
                Some(trimmed)
            }
        })
        .unwrap_or("");

    let ok = constant_time_eq(configured.as_bytes(), candidate.as_bytes());

    if ok {
        next.run(req).await
    } else {
        let body = ErrorBody {
            code: "forbidden",
            message: "unauthorized: invalid or missing API token".to_string(),
            trace_id: None,
        };
        (StatusCode::UNAUTHORIZED, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use yq_nova_core::{
        config::{ServerConfig, StorageConfig},
        graph::extractor::RegexWikiExtractor,
        memory::MemoryService,
        storage::{Database, Migrator},
    };

    /// 构造带指定 auth_token 的测试 router。
    async fn make_router(auth_token: &str) -> axum::Router {
        let dir = std::env::temp_dir().join(format!(
            "yq-nova-test-auth-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let db_path = dir.join("nova.sqlite");
        let storage_cfg = StorageConfig { db_path, ..Default::default() };
        let db = Database::open(storage_cfg.clone()).await.expect("open db");
        Migrator::run(&db.pool).await.expect("migrations");

        let embed = std::sync::Arc::new(yq_nova_core::embedding::MockEmbeddingProvider::new(16));
        let memory = MemoryService::new(db.clone(), embed);
        let graph = yq_nova_core::graph::GraphService::with_parts(
            db.clone(),
            std::sync::Arc::new(RegexWikiExtractor::new()),
        );

        let srv_cfg = ServerConfig { auth_token: auth_token.into(), ..Default::default() };
        let state = AppState::new(srv_cfg, db, memory, graph);
        crate::http::build_router(state)
    }

    const HEALTH: &str = "/v1/health";

    #[tokio::test]
    async fn missing_auth_header_returns_401() {
        let router = make_router("secret123").await;
        let req =
            Request::builder().uri(HEALTH).method("GET").body(Body::empty()).unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], "forbidden");
        assert_eq!(body["message"], "unauthorized: invalid or missing API token");
    }

    #[tokio::test]
    async fn wrong_token_returns_401() {
        let router = make_router("secret123").await;
        let req = Request::builder()
            .uri(HEALTH)
            .method("GET")
            .header("authorization", "Bearer wrong-token")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn correct_bearer_token_passes() {
        let router = make_router("secret123").await;
        let req = Request::builder()
            .uri(HEALTH)
            .method("GET")
            .header("authorization", "Bearer secret123")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn empty_token_disables_auth() {
        let router = make_router("").await;
        let req =
            Request::builder().uri(HEALTH).method("GET").body(Body::empty()).unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}