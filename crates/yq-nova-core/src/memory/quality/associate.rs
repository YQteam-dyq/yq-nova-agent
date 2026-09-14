use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::super::MemoryService;
use crate::{
    error::{NovaError, NovaResult},
    storage::{MemoryFilter, MemoryStatus, memory::MemoryRepository, vector::cosine_similarity},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AssociateOptions {
    pub enabled: bool,
    pub similarity_threshold: f32,
    pub max_memories: usize,
    pub max_links_per_memory: usize,
}

impl Default for AssociateOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            similarity_threshold: 0.82,
            max_memories: 500,
            max_links_per_memory: 6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLink {
    pub source_memory_uuid: Uuid,
    pub target_memory_uuid: Uuid,
    pub kind: String,
    pub weight: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssociateOutput {
    pub scanned: usize,
    pub created: usize,
    pub updated: usize,
}

fn canonical(a: Uuid, b: Uuid) -> (Uuid, Uuid) {
    if a.as_bytes() <= b.as_bytes() { (a, b) } else { (b, a) }
}

async fn upsert_link(
    svc: &MemoryService,
    a: Uuid,
    b: Uuid,
    kind: &str,
    weight: f32,
) -> NovaResult<bool> {
    let (src, tgt) = canonical(a, b);
    let exists = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM memory_links WHERE source_memory_uuid = ?1 AND target_memory_uuid = ?2 AND \
         kind = ?3",
    )
    .bind(src.to_string())
    .bind(tgt.to_string())
    .bind(kind)
    .fetch_optional(&svc.database.pool)
    .await
    .map_err(NovaError::storage)?
    .is_some();
    let now = Utc::now().timestamp();
    sqlx::query(
        "INSERT INTO memory_links (source_memory_uuid, target_memory_uuid, kind, weight, \
         metadata_json, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6) ON \
         CONFLICT(source_memory_uuid, target_memory_uuid, kind) DO UPDATE SET weight = \
         excluded.weight, updated_at = excluded.updated_at",
    )
    .bind(src.to_string())
    .bind(tgt.to_string())
    .bind(kind)
    .bind(weight.clamp(0.0, 1.0) as f64)
    .bind("{}")
    .bind(now)
    .execute(&svc.database.pool)
    .await
    .map_err(NovaError::storage)?;
    Ok(!exists)
}

pub async fn discover(
    svc: &MemoryService,
    namespace_id: i64,
    opts: AssociateOptions,
) -> NovaResult<AssociateOutput> {
    if !opts.enabled {
        return Ok(AssociateOutput::default());
    }
    let filter = MemoryFilter {
        namespace_id: Some(namespace_id),
        status_in: Some(vec![MemoryStatus::Active]),
        ..Default::default()
    };
    let limit = opts.max_memories.clamp(2, 2000);
    let records = svc.memory_repo.list(&svc.database, &filter, limit, 0).await?;
    let mut out = AssociateOutput {
        scanned: records.len(),
        ..Default::default()
    };
    if records.len() < 2 {
        return Ok(out);
    }
    let max_links = opts.max_links_per_memory.clamp(1, 50).max(1);

    let text_refs: Vec<&str> = records.iter().map(|r| r.content.as_str()).collect();
    let vectors = svc.embedding.embed_batch(&text_refs).await?;
    let threshold = opts.similarity_threshold.clamp(-1.0, 1.0);
    let n = records.len();
    for i in 0..n {
        let mut scored: Vec<(f32, usize)> = Vec::new();
        for j in 0..n {
            if i == j {
                continue;
            }
            let sim = cosine_similarity(&vectors[i], &vectors[j]);
            if sim >= threshold {
                scored.push((sim, j));
            }
        }
        scored.sort_by(|x, y| {
            y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| x.1.cmp(&y.1))
        });
        scored.truncate(max_links);
        for (sim, j) in scored {
            let created =
                upsert_link(svc, records[i].uuid, records[j].uuid, "semantic", sim).await?;
            if created {
                out.created += 1;
            } else {
                out.updated += 1;
            }
        }
    }

    let mut tag_index: HashMap<String, Vec<Uuid>> = HashMap::new();
    for record in &records {
        for tag in &record.tags {
            tag_index.entry(tag.clone()).or_default().push(record.uuid);
        }
    }
    let mut tag_pairs: HashSet<(Uuid, Uuid)> = HashSet::new();
    for members in tag_index.values() {
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                tag_pairs.insert(canonical(members[i], members[j]));
            }
        }
    }
    for (a, b) in tag_pairs {
        let created = upsert_link(svc, a, b, "shared_tag", 1.0).await?;
        if created {
            out.created += 1;
        } else {
            out.updated += 1;
        }
    }
    Ok(out)
}

pub async fn links_for(svc: &MemoryService, memory_uuid: Uuid) -> NovaResult<Vec<MemoryLink>> {
    let sub = memory_uuid.to_string();
    let rows: Vec<(String, String, String, f64)> = sqlx::query_as(
        "SELECT source_memory_uuid, target_memory_uuid, kind, weight FROM memory_links WHERE \
         source_memory_uuid = ?1 OR target_memory_uuid = ?1 ORDER BY weight DESC",
    )
    .bind(&sub)
    .fetch_all(&svc.database.pool)
    .await
    .map_err(NovaError::storage)?;
    let mut links = Vec::with_capacity(rows.len());
    for (src, tgt, kind, weight) in rows {
        let source = Uuid::parse_str(&src)
            .map_err(|e| NovaError::storage_msg(format!("bad source uuid: {e}")))?;
        let target = Uuid::parse_str(&tgt)
            .map_err(|e| NovaError::storage_msg(format!("bad target uuid: {e}")))?;
        links.push(MemoryLink {
            source_memory_uuid: source,
            target_memory_uuid: target,
            kind,
            weight: weight as f32,
        });
    }
    Ok(links)
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
        let dir = std::env::temp_dir().join(format!("yq-nova-q4-{}", Uuid::new_v4()));
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

    #[tokio::test]
    async fn discover_creates_semantic_and_tag_links() {
        let svc = temp_svc().await;
        let a = svc
            .remember(RememberInput {
                content: "alpha project status shared",
                tags: &["team".to_string()],
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "beta project status shared",
                tags: &["team".to_string()],
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_ne!(a.uuid, b.uuid);
        let out = discover(
            &svc,
            crate::storage::namespace::DEFAULT_NAMESPACE_ID,
            AssociateOptions {
                enabled: true,
                similarity_threshold: -1.0,
                max_memories: 100,
                max_links_per_memory: 5,
            },
        )
        .await
        .unwrap();
        assert_eq!(out.scanned, 2);
        assert!(out.created >= 1);
        let links = links_for(&svc, a.uuid).await.unwrap();
        assert!(!links.is_empty());
    }

    #[tokio::test]
    async fn discover_disabled_returns_empty() {
        let svc = temp_svc().await;
        let _ = svc
            .remember(RememberInput {
                content: "some memory",
                ..Default::default()
            })
            .await
            .unwrap();
        let out = discover(
            &svc,
            crate::storage::namespace::DEFAULT_NAMESPACE_ID,
            AssociateOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(out.created, 0);
        assert_eq!(out.updated, 0);
    }
}
