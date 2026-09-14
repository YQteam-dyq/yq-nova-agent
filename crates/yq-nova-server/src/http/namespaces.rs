use axum::{
    Json,
    extract::{Extension, Path, Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use yq_nova_core::storage::{
    NamespaceRepository, SqliteNamespaceRepository, namespace::DeleteNamespaceOutcome,
};

use crate::http::{
    AppError, AppState, Result,
    auth::{ClientScope, NamespaceContext, tenant_ns},
};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct NamespaceListQuery {
    pub limit: usize,
    pub offset: usize,
}

impl Default for NamespaceListQuery {
    fn default() -> Self {
        Self {
            limit: 100,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreateNamespaceRequest {
    pub name: String,
    pub description: Option<String>,
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpdateNamespaceRequest {
    pub description: Option<String>,
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NamespaceListOutput {
    pub total: i64,
    pub count: usize,
    pub limit: usize,
    pub offset: usize,
    pub items: Vec<yq_nova_core::storage::namespace::NamespaceRecord>,
}

pub async fn list_namespaces(
    State(state): State<AppState>,
    Extension(scope): Extension<ClientScope>,
    Query(q): Query<NamespaceListQuery>,
) -> Result<Json<NamespaceListOutput>> {
    if scope == ClientScope::Tenant {
        return Err(AppError(yq_nova_core::NovaError::forbidden(
            "tenant scoped clients may not list all namespaces",
        )));
    }
    let repo = SqliteNamespaceRepository::new();
    let limit = q.limit.clamp(1, 10_000);
    let offset = q.offset;
    let total = repo.count_all(&state.db).await?;
    let items = repo.list(&state.db, limit, offset).await?;
    Ok(Json(NamespaceListOutput {
        total,
        count: items.len(),
        limit,
        offset,
        items,
    }))
}

pub async fn get_namespace(
    State(state): State<AppState>,
    Extension(scope): Extension<ClientScope>,
    Extension(ns): Extension<NamespaceContext>,
    Path(name): Path<String>,
) -> Result<Json<yq_nova_core::storage::namespace::NamespaceRecord>> {
    let repo = SqliteNamespaceRepository::new();
    let name = name.trim();
    if scope == ClientScope::Tenant && tenant_ns(&state.db, ns.id).await?.name != name {
        return Err(AppError(yq_nova_core::NovaError::forbidden(
            "tenant scoped clients may only access their own namespace",
        )));
    }
    let ns = repo
        .get_by_name(&state.db, name)
        .await?
        .ok_or_else(|| AppError(yq_nova_core::NovaError::not_found(format!("namespace {name}"))))?;
    Ok(Json(ns))
}

pub async fn create_namespace(
    State(state): State<AppState>,
    Extension(scope): Extension<ClientScope>,
    Json(req): Json<CreateNamespaceRequest>,
) -> Result<(StatusCode, Json<yq_nova_core::storage::namespace::NamespaceRecord>)> {
    if scope == ClientScope::Tenant {
        return Err(AppError(yq_nova_core::NovaError::forbidden(
            "tenant scoped clients may not create namespaces",
        )));
    }
    let repo = SqliteNamespaceRepository::new();
    let ns = repo
        .create(
            &state.db,
            yq_nova_core::storage::namespace::CreateNamespaceInput {
                name: req.name.trim(),
                description: req.description.as_deref(),
                config: req.config.as_ref(),
            },
        )
        .await?;
    Ok((StatusCode::CREATED, Json(ns)))
}

pub async fn update_namespace(
    State(state): State<AppState>,
    Extension(scope): Extension<ClientScope>,
    Extension(ns): Extension<NamespaceContext>,
    Path(name): Path<String>,
    Json(req): Json<UpdateNamespaceRequest>,
) -> Result<Json<yq_nova_core::storage::namespace::NamespaceRecord>> {
    let repo = SqliteNamespaceRepository::new();
    let name = name.trim();
    if scope == ClientScope::Tenant && tenant_ns(&state.db, ns.id).await?.name != name {
        return Err(AppError(yq_nova_core::NovaError::forbidden(
            "tenant scoped clients may only update their own namespace",
        )));
    }
    let ns = repo
        .update(
            &state.db,
            yq_nova_core::storage::namespace::UpdateNamespaceInput {
                name,
                description: req
                    .description
                    .as_deref()
                    .map(|s| if s.trim().is_empty() { "" } else { s.trim() }),
                config: req.config.as_ref(),
            },
        )
        .await?;
    Ok(Json(ns))
}

#[derive(Debug, Clone, Serialize)]
pub struct DeleteNamespaceOutput {
    pub name: String,
    pub deleted: bool,
    pub protected: bool,
}

pub async fn delete_namespace(
    State(state): State<AppState>,
    Extension(scope): Extension<ClientScope>,
    Path(name): Path<String>,
) -> Result<Json<DeleteNamespaceOutput>> {
    if scope == ClientScope::Tenant {
        return Err(AppError(yq_nova_core::NovaError::forbidden(
            "tenant scoped clients may not delete namespaces",
        )));
    }
    let repo = SqliteNamespaceRepository::new();
    let name = name.trim().to_string();
    match repo.delete(&state.db, &name).await? {
        DeleteNamespaceOutcome::Deleted(_) => Ok(Json(DeleteNamespaceOutput {
            name,
            deleted: true,
            protected: false,
        })),
        DeleteNamespaceOutcome::Protected(_protected_name) => Ok(Json(DeleteNamespaceOutput {
            name,
            deleted: false,
            protected: true,
        })),
        DeleteNamespaceOutcome::NotFound => Ok(Json(DeleteNamespaceOutput {
            name,
            deleted: false,
            protected: false,
        })),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use uuid::Uuid;
    use yq_nova_core::{
        config::{ServerConfig, StorageConfig},
        graph::extractor::RegexWikiExtractor,
        memory::MemoryService,
        storage::{Database, Migrator, NamespaceRepository, SqliteNamespaceRepository},
    };

    use super::*;

    async fn make_router() -> (AppState, Router) {
        let dir = std::env::temp_dir().join(format!(
            "yq-nova-test-ns-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let db_path = dir.join("nova.sqlite");
        let storage_cfg = StorageConfig {
            db_path,
            ..Default::default()
        };
        let db = Database::open(storage_cfg.clone()).await.expect("open db");
        Migrator::run(&db.pool).await.expect("migrations");

        let repo = SqliteNamespaceRepository::new();
        for name in ["tenant-a", "tenant-b"] {
            repo.create(
                &db,
                yq_nova_core::storage::namespace::CreateNamespaceInput {
                    name,
                    description: None,
                    config: None,
                },
            )
            .await
            .unwrap();
        }

        let mut namespace_keys = BTreeMap::new();
        namespace_keys.insert("tenant-a".to_string(), "key-a".to_string());
        namespace_keys.insert("tenant-b".to_string(), "key-b".to_string());

        let embed = std::sync::Arc::new(yq_nova_core::embedding::MockEmbeddingProvider::new(16));
        let memory = MemoryService::new(db.clone(), embed);
        let graph = yq_nova_core::graph::GraphService::with_parts(
            db.clone(),
            std::sync::Arc::new(RegexWikiExtractor::new()),
        );

        let srv_cfg = ServerConfig {
            auth_token: "global-secret".into(),
            namespace_keys,
            ..Default::default()
        };
        let state = AppState::new(srv_cfg, db, memory, graph);
        let router = crate::http::build_router(state.clone());
        (state, router)
    }

    async fn send(router: &Router, method: &str, uri: &str, auth: &str, body: Option<&str>) -> StatusCode {
        let mut b = Request::builder().uri(uri).method(method);
        b = b.header("authorization", &format!("Bearer {auth}"));
        let b = if let Some(payload) = body {
            b.header("content-type", "application/json").body(Body::from(payload.to_string()))
        } else {
            b.body(Body::empty())
        };
        let resp = router.clone().oneshot(b.unwrap()).await.unwrap();
        resp.status()
    }

    #[tokio::test]
    async fn admin_global_can_fully_manage_namespaces() {
        let (_state, router) = make_router().await;
        let auth = "global-secret";

        assert_eq!(
            send(&router, "GET", "/v1/namespaces", auth, None).await,
            StatusCode::OK
        );
        assert_eq!(
            send(
                &router,
                "POST",
                "/v1/namespaces",
                auth,
                Some(r#"{"name":"newns"}"#)
            )
            .await,
            StatusCode::CREATED
        );
        assert_eq!(
            send(&router, "DELETE", "/v1/namespaces/newns", auth, None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn tenant_key_cannot_administer_namespaces() {
        let (_state, router) = make_router().await;

        assert_eq!(
            send(&router, "GET", "/v1/namespaces", "key-a", None).await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            send(
                &router,
                "POST",
                "/v1/namespaces",
                "key-a",
                Some(r#"{"name":"rogue"}"#)
            )
            .await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            send(&router, "DELETE", "/v1/namespaces/tenant-a", "key-a", None).await,
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            send(&router, "GET", "/v1/namespaces/tenant-b", "key-a", None).await,
            StatusCode::FORBIDDEN
        );
    }

    #[tokio::test]
    async fn tenant_key_can_read_and_update_only_own_namespace() {
        let (_state, router) = make_router().await;

        assert_eq!(
            send(&router, "GET", "/v1/namespaces/tenant-a", "key-a", None).await,
            StatusCode::OK
        );
        assert_eq!(
            send(
                &router,
                "PATCH",
                "/v1/namespaces/tenant-a",
                "key-a",
                Some(r#"{"description":"mine"}"#)
            )
            .await,
            StatusCode::OK
        );
        assert_eq!(
            send(
                &router,
                "PATCH",
                "/v1/namespaces/tenant-b",
                "key-a",
                Some(r#"{"description":"yours"}"#)
            )
            .await,
            StatusCode::FORBIDDEN
        );
    }
}
