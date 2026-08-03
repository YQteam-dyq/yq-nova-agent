//! 元信息 endpoints：health / stats。
//!
//! P6-1: `/v1/health` 快速探活（返回 ok + uptime）。
//! P6-2: `/v1/stats` 返回库大小、记忆数量、实体数量、tag 统计等。
//!
//! 两个 handler 都禁止 panic，错误统一转为 500 JSON（通过 AppError/NovaError）。

use axum::{Json, extract::State};
use chrono::Utc;
use serde::Serialize;
use yq_nova_core::storage::{
    MemoryFilter, MemoryRepository, MemoryStatus, SqliteEntityRepository, SqliteMemoryRepository,
    SqliteTagRepository,
};

use crate::http::{AppError, AppState, Result};

// ---------------------------------------------------------------------------
// /v1/health
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct HealthOut {
    pub status: &'static str,
    pub version: &'static str,
    pub git_sha: &'static str,
    pub uptime_secs: i64,
}

pub async fn health(State(state): State<AppState>) -> Result<Json<HealthOut>> {
    let uptime_secs = Utc::now().timestamp().saturating_sub(state.started_at_epoch_secs);
    Ok(Json(HealthOut {
        status: "ok",
        version: yq_nova_core::VERSION,
        git_sha: yq_nova_core::git_sha(),
        uptime_secs: uptime_secs.max(0),
    }))
}

// ---------------------------------------------------------------------------
// /v1/stats
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Default)]
pub struct StatsOut {
    pub uptime_secs: i64,
    pub database_size_bytes: Option<i64>,
    pub memory_active: i64,
    pub memory_archived: i64,
    pub memory_total: i64,
    pub entity_count: i64,
    pub relation_count: i64,
    pub tag_count: i64,
}

async fn db_size_bytes(db: &yq_nova_core::storage::Database) -> Option<i64> {
    // 对于 SQLite，我们直接从 DB 句柄做一个简单查询，获取 page_size * page_count。
    let sql_page_size: std::result::Result<i64, sqlx::Error> =
        sqlx::query_scalar::<_, i64>("PRAGMA page_size").fetch_one(&db.pool).await;
    let sql_page_count: std::result::Result<i64, sqlx::Error> =
        sqlx::query_scalar::<_, i64>("PRAGMA page_count").fetch_one(&db.pool).await;
    match (sql_page_size, sql_page_count) {
        (Ok(ps), Ok(pc)) => Some(ps.saturating_mul(pc)),
        _ => None,
    }
}

pub async fn stats(State(state): State<AppState>) -> Result<Json<StatsOut>> {
    let db = &state.db;
    let mem_repo = SqliteMemoryRepository::new();
    let tag_repo = SqliteTagRepository::new();
    let entity_repo = SqliteEntityRepository::new();

    // 计数：status=Active
    let active_count = mem_repo
        .count(
            db,
            &MemoryFilter { status_in: Some(vec![MemoryStatus::Active]), ..Default::default() },
        )
        .await?;

    let archived_count = mem_repo
        .count(
            db,
            &MemoryFilter { status_in: Some(vec![MemoryStatus::Archived]), ..Default::default() },
        )
        .await?;

    // 直接查 entities / relations 计数，避免复杂 join，保持快：
    let entity_count: i64 = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM entities")
        .fetch_one(&db.pool)
        .await
        .map_err(|e| AppError::from(yq_nova_core::error::NovaError::storage(e)))?;

    let relation_count: i64 = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM relations")
        .fetch_one(&db.pool)
        .await
        .map_err(|e| AppError::from(yq_nova_core::error::NovaError::storage(e)))?;

    let tag_count: i64 = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM tags")
        .fetch_one(&db.pool)
        .await
        .map_err(|e| AppError::from(yq_nova_core::error::NovaError::storage(e)))?;

    let _ = (tag_repo, entity_repo); // suppress unused warnings; repos used later in M4.3

    // total: 所有非 deleted（等价于默认 filter）
    let total_count = mem_repo.count(db, &MemoryFilter::default()).await?;

    let database_size_bytes = db_size_bytes(db).await;
    let uptime_secs = Utc::now().timestamp().saturating_sub(state.started_at_epoch_secs);

    Ok(Json(StatsOut {
        uptime_secs: uptime_secs.max(0),
        database_size_bytes,
        memory_active: active_count,
        memory_archived: archived_count,
        memory_total: total_count,
        entity_count,
        relation_count,
        tag_count,
    }))
}
