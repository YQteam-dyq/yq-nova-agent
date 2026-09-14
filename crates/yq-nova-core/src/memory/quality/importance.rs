use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::super::MemoryService;
use crate::{
    error::{NovaError, NovaResult},
    storage::{MemoryFilter, MemorySortOrder, MemoryStatus, memory::MemoryRepository},
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct ImportanceWeights {
    pub access: f32,
    pub recency: f32,
    pub reference: f32,
    pub half_life_days: f32,
}

impl Default for ImportanceWeights {
    fn default() -> Self {
        Self {
            access: 0.2,
            recency: 0.2,
            reference: 0.2,
            half_life_days: 30.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct RebalanceOptions {
    pub enabled: bool,
    pub weights: ImportanceWeights,
    pub batch_limit: usize,
    pub persist: bool,
}

impl Default for RebalanceOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            weights: ImportanceWeights::default(),
            batch_limit: 5000,
            persist: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct RebalanceOutput {
    pub scanned: usize,
    pub updated: usize,
    pub unchanged: usize,
}

pub fn access_signal(count: i64) -> f32 {
    let n = count.clamp(0, 50) as f32;
    (n / 50.0).sqrt()
}

pub fn reference_signal(count: usize) -> f32 {
    let n = count.min(20) as f32;
    n / 20.0
}

pub fn score(
    base: f32,
    access_count: i64,
    last_seen: DateTime<Utc>,
    now: DateTime<Utc>,
    references: usize,
    weights: &ImportanceWeights,
) -> f32 {
    let base = base.clamp(0.0, 1.0);
    let seconds = now.signed_duration_since(last_seen).num_seconds().max(0) as f32;
    let age_days = seconds / 86400.0;
    let recency = if weights.half_life_days > 0.0 {
        (-(std::f32::consts::LN_2) * age_days / weights.half_life_days).exp()
    } else {
        1.0
    };
    let raw = base * 0.4
        + weights.access.max(0.0) * access_signal(access_count)
        + weights.recency.max(0.0) * recency
        + weights.reference.max(0.0) * reference_signal(references);
    raw.clamp(0.0, 1.0)
}

async fn count_references(svc: &MemoryService, uuid: Uuid) -> NovaResult<usize> {
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM relations WHERE memory_uuid = ?1")
        .bind(uuid.to_string())
        .fetch_one(&svc.database.pool)
        .await
        .map_err(NovaError::storage)?;
    Ok(count.max(0) as usize)
}

async fn record_log(svc: &MemoryService, uuid: Uuid, importance: f32) -> NovaResult<()> {
    let now = Utc::now().timestamp();
    sqlx::query(
        "INSERT OR REPLACE INTO importance_log (memory_uuid, importance, recorded_at) VALUES (?1, \
         ?2, ?3)",
    )
    .bind(uuid.to_string())
    .bind(importance as f64)
    .bind(now)
    .execute(&svc.database.pool)
    .await
    .map_err(NovaError::storage)?;
    Ok(())
}

pub async fn maintain_one(svc: &MemoryService, uuid: Uuid) -> NovaResult<f32> {
    let record = svc.memory_repo.get_by_uuid(&svc.database, uuid).await?;
    if record.status != MemoryStatus::Active {
        return Ok(record.importance);
    }
    let now = Utc::now();
    let last_seen = record.last_accessed.unwrap_or(record.created_at);
    let references = count_references(svc, uuid).await?;
    let updated = score(
        record.importance,
        record.access_count,
        last_seen,
        now,
        references,
        &ImportanceWeights::default(),
    );
    if (updated - record.importance).abs() > 0.001 {
        svc.memory_repo.update_importance(&svc.database, uuid, updated).await?;
        record_log(svc, uuid, updated).await?;
    }
    Ok(updated)
}

pub async fn rebalance(svc: &MemoryService, opts: RebalanceOptions) -> NovaResult<RebalanceOutput> {
    if !opts.enabled {
        return Ok(RebalanceOutput::default());
    }
    let filter = MemoryFilter {
        status_in: Some(vec![MemoryStatus::Active]),
        ..Default::default()
    };
    let limit = opts.batch_limit.clamp(1, 50_000);
    let records = svc
        .memory_repo
        .list_ordered(&svc.database, &filter, limit, 0, MemorySortOrder::ImportanceDesc)
        .await?;
    let now = Utc::now();
    let weights = opts.weights;
    let mut seen: HashSet<Uuid> = HashSet::with_capacity(records.len());
    let mut out = RebalanceOutput {
        scanned: records.len(),
        ..Default::default()
    };
    for record in records {
        if !seen.insert(record.uuid) {
            continue;
        }
        let references = count_references(svc, record.uuid).await?;
        let last_seen = record.last_accessed.unwrap_or(record.created_at);
        let updated =
            score(record.importance, record.access_count, last_seen, now, references, &weights);
        if (updated - record.importance).abs() <= 0.001 {
            out.unchanged += 1;
            continue;
        }
        if opts.persist {
            svc.memory_repo.update_importance(&svc.database, record.uuid, updated).await?;
            record_log(svc, record.uuid, updated).await?;
        }
        out.updated += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::StorageConfig,
        memory::ops_remember::{RememberInput, service_for_tests},
        storage::Database,
    };

    async fn temp_svc() -> crate::memory::MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-q3-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..Default::default()
        };
        let db = Database::open(cfg).await.unwrap();
        service_for_tests(db, 8, None)
    }

    #[test]
    fn score_is_bounded_and_signal_increases() {
        let now = Utc::now();
        let weights = ImportanceWeights::default();
        let low = score(0.1, 0, now, now, 0, &weights);
        let high = score(0.1, 50, now, now, 20, &weights);
        assert!(high > low);
        assert!((0.0..=1.0).contains(&high));
    }

    #[test]
    fn recency_decays_with_age() {
        let now = Utc::now();
        let weights = ImportanceWeights::default();
        let fresh = score(0.1, 0, now, now, 0, &weights);
        let old = score(0.1, 0, now - chrono::Duration::days(365), now, 0, &weights);
        assert!(fresh > old);
    }

    #[tokio::test]
    async fn rebalance_updates_access_heavy_memory() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "frequently recalled memory",
                importance: 0.1,
                ..Default::default()
            })
            .await
            .unwrap();
        svc.recall(crate::memory::ops_recall::RecallInput {
            query: "frequently recalled memory",
            top_k: 5,
            ..Default::default()
        })
        .await
        .unwrap();
        let out = rebalance(
            &svc,
            RebalanceOptions {
                enabled: true,
                persist: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(out.scanned, 1);
        let after = svc.get_memory(a.uuid).await.unwrap();
        assert!(after.importance > 0.1);
        assert!(out.updated >= 1);
    }
}
