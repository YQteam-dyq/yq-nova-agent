use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::super::MemoryService;
use crate::{
    error::NovaResult,
    storage::{
        MemoryStatus,
        memory::{MemoryRepository, sha256_hex},
        vector::VectorStore,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DedupMode {
    #[default]
    Report,
    Merge,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct DedupOptions {
    pub enabled: bool,
    pub threshold: f32,
    pub mode: DedupMode,
    pub max_candidates: usize,
}

impl Default for DedupOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 0.92,
            mode: DedupMode::Report,
            max_candidates: 12,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarHit {
    pub memory_uuid: Uuid,
    pub similarity: f32,
    pub content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DedupDecision {
    pub blocked_by: Option<Uuid>,
    pub merged: bool,
}

pub async fn detect(
    svc: &MemoryService,
    content: &str,
    opts: &DedupOptions,
) -> NovaResult<Vec<SimilarHit>> {
    if !opts.enabled {
        return Ok(Vec::new());
    }
    let threshold = opts.threshold.clamp(0.0, 1.0);
    let k = opts.max_candidates.clamp(1, 100);
    let query = svc.embedding.embed_one(content).await?;
    let hits = svc.vector_store.knn_search(&query, k, threshold).await?;
    let mut out = Vec::with_capacity(hits.len());
    for hit in hits {
        let record = match svc.memory_repo.get_by_uuid(&svc.database, hit.memory_uuid).await {
            Ok(r) => r,
            Err(_) => continue,
        };
        if record.status != MemoryStatus::Active {
            continue;
        }
        out.push(SimilarHit {
            memory_uuid: hit.memory_uuid,
            similarity: hit.similarity,
            content: record.content,
        });
    }
    out.sort_by(|a, b| {
        b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

pub(crate) async fn decide(
    svc: &MemoryService,
    content: &str,
    opts: &DedupOptions,
) -> NovaResult<DedupDecision> {
    let hits = detect(svc, content, opts).await?;
    let Some(best) = hits.first() else {
        return Ok(DedupDecision {
            blocked_by: None,
            merged: false,
        });
    };
    if opts.threshold > best.similarity {
        return Ok(DedupDecision {
            blocked_by: None,
            merged: false,
        });
    }
    let decision = DedupDecision {
        blocked_by: Some(best.memory_uuid),
        merged: false,
    };
    if opts.mode == DedupMode::Merge {
        merge_into(svc, best.memory_uuid, content).await?;
        return Ok(DedupDecision {
            blocked_by: Some(best.memory_uuid),
            merged: true,
        });
    }
    Ok(decision)
}

async fn merge_into(svc: &MemoryService, existing_uuid: Uuid, content: &str) -> NovaResult<()> {
    let existing = svc.memory_repo.get_by_uuid(&svc.database, existing_uuid).await?;
    let combined = if existing.content.is_empty() {
        content.to_string()
    } else {
        format!("{}\n{}", existing.content, content)
    };
    let hash = sha256_hex(&combined);
    svc.memory_repo.update_content(&svc.database, existing_uuid, &combined, &hash).await?;
    Ok(())
}
