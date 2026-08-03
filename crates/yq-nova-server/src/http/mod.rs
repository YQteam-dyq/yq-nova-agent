//! HTTP Server: axum 路由 + 状态 + 中间件。
//!
//! 顶层 router 结构：
//! ```text
//! /v1/health                    (P6-1)
//! /v1/stats                     (P6-2)
//! /v1/memory :remember /recall /forget /:uuid
//! /v1/graph  :entities /relations /traverse
//! ```

use axum::{
    Router,
    http::Method,
    routing::{get, post},
};
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    limit::RequestBodyLimitLayer,
    trace::TraceLayer,
};

pub mod error;
pub mod graph;
pub mod memory;
pub mod meta;
pub mod state;

pub use error::{AppError, fallback_404};
pub use state::AppState;

/// 统一 result 别名，让 handler 写起来更短。
/// NovaError 会通过 `From<NovaError> for AppError` 自动转换。
pub type Result<T> = std::result::Result<T, AppError>;

/// 组装 axum Router：中间件 + 路由 + fallback 404。
pub fn build_router(state: AppState) -> Router {
    let methods =
        [Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS, Method::PATCH];

    let api_v1 = Router::new()
        .route("/health", get(meta::health))
        .route("/stats", get(meta::stats))
        // M4.2 memory routes
        .route("/memory/remember", post(memory::remember))
        .route("/memory/recall", post(memory::recall))
        .route("/memory/forget", post(memory::forget))
        .route("/memory/:uuid", get(memory::get_memory).delete(memory::delete_memory))
        // M4.3 graph routes
        .route("/graph/extract-and-link", post(graph::extract_and_link))
        .route("/graph/entities", post(graph::upsert_entity).get(graph::list_entities))
        .route("/graph/relations", post(graph::upsert_relation).get(graph::list_relations))
        .route("/graph/traverse", post(graph::traverse));

    Router::new()
        .nest("/v1", api_v1)
        .with_state(state)
        // ---- 通用中间件（从外到内执行） ----
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(methods).allow_headers(Any))
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // P8-4: 10MB body limit
        .layer(CompressionLayer::new().gzip(true).no_deflate().no_zstd())
        .layer(TraceLayer::new_for_http())
        .fallback(fallback_404)
}
