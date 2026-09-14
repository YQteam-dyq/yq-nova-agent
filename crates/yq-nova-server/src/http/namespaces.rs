use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use yq_nova_core::storage::{
    NamespaceRepository, SqliteNamespaceRepository, namespace::DeleteNamespaceOutcome,
};

use crate::http::{AppError, AppState, Result};

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
    Query(q): Query<NamespaceListQuery>,
) -> Result<Json<NamespaceListOutput>> {
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
    Path(name): Path<String>,
) -> Result<Json<yq_nova_core::storage::namespace::NamespaceRecord>> {
    let repo = SqliteNamespaceRepository::new();
    let ns = repo
        .get_by_name(&state.db, name.trim())
        .await?
        .ok_or_else(|| AppError(yq_nova_core::NovaError::not_found(format!("namespace {name}"))))?;
    Ok(Json(ns))
}

pub async fn create_namespace(
    State(state): State<AppState>,
    Json(req): Json<CreateNamespaceRequest>,
) -> Result<(StatusCode, Json<yq_nova_core::storage::namespace::NamespaceRecord>)> {
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
    Path(name): Path<String>,
    Json(req): Json<UpdateNamespaceRequest>,
) -> Result<Json<yq_nova_core::storage::namespace::NamespaceRecord>> {
    let repo = SqliteNamespaceRepository::new();
    let ns = repo
        .update(
            &state.db,
            yq_nova_core::storage::namespace::UpdateNamespaceInput {
                name: name.trim(),
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
    Path(name): Path<String>,
) -> Result<Json<DeleteNamespaceOutput>> {
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
