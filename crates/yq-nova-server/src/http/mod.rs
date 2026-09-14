use axum::{
    Router,
    http::Method,
    routing::{get, patch, post},
};
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    limit::RequestBodyLimitLayer,
    trace::TraceLayer,
};

pub mod auth;
pub mod error;
pub mod graph;
pub mod memory;
pub mod meta;
pub mod namespaces;
pub mod state;
pub mod tags;

pub use error::{AppError, fallback_404};
pub use state::AppState;

pub type Result<T> = std::result::Result<T, AppError>;

pub fn build_router(state: AppState) -> Router {
    let methods =
        [Method::GET, Method::POST, Method::PUT, Method::DELETE, Method::OPTIONS, Method::PATCH];

    let api_v1 = Router::new()
        .route("/health", get(meta::health))
        .route("/stats", get(meta::stats))
        .route("/memory/list", post(memory::list_memories))
        .route("/memory/remember", post(memory::remember))
        .route("/memory/remember-batch", post(memory::remember_batch))
        .route("/memory/recall", post(memory::recall))
        .route("/memory/forget", post(memory::forget))
        .route("/memory/export", post(memory::export_memories))
        .route("/memory/import", post(memory::import_memories))
        .route("/memory/merge", post(memory::merge_memories))
        .route(
            "/memory/:uuid",
            get(memory::get_memory).delete(memory::delete_memory).patch(memory::update_memory),
        )
        .route("/tags", get(tags::list_tags))
        .route("/tags/:name", patch(tags::rename_tag).delete(tags::delete_tag))
        .route("/graph/extract-and-link", post(graph::extract_and_link))
        .route("/graph/entities", post(graph::upsert_entity).get(graph::list_entities))
        .route("/graph/entities/merge", post(graph::merge_entities))
        .route("/graph/relations", post(graph::upsert_relation).get(graph::list_relations))
        .route("/graph/traverse", post(graph::traverse))
        .route("/namespaces", post(namespaces::create_namespace).get(namespaces::list_namespaces))
        .route(
            "/namespaces/:name",
            get(namespaces::get_namespace)
                .patch(namespaces::update_namespace)
                .delete(namespaces::delete_namespace),
        );

    let auth_state = state.clone();

    Router::new()
        .nest("/v1", api_v1)
        .with_state(state)
        .layer(axum::middleware::from_fn_with_state(auth_state, auth::auth_middleware))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(methods).allow_headers(Any))
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024))
        .layer(CompressionLayer::new().gzip(true).no_deflate().no_zstd())
        .layer(TraceLayer::new_for_http())
        .fallback(fallback_404)
}
