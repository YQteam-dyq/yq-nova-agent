//! `remember` — persist a memory, embed it for recall, extract entities/
//! relations for the graph, and attach tags. This is the **write path**; it
//! must be idempotent (the underlying MemoryRepository already dedupes by
//! content hash so repeated identical `remember` calls are free).

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::MemoryService;
use crate::{
    error::{NovaError, NovaResult},
    graph::extractor::RelationCandidate,
    storage::{
        MemorySource,
        entity::{EntityRepository, SqliteEntityRepository, UpsertEntityInput},
        memory::MemoryRepository,
        relation::{InsertRelationInput, RelationRepository, SqliteRelationRepository},
        vector::VectorStore,
    },
};

/// Input to [`remember`](crate::memory::MemoryService::remember).
#[derive(Debug, Clone)]
pub struct RememberInput<'a> {
    /// Raw text to remember. The core of the memory; used for both the
    /// content hash and semantic embedding.
    pub content: &'a str,
    /// Where this memory came from. Controls how the background forgetting
    /// policy weights it later.
    pub source: MemorySource,
    /// Caller-assigned importance ∈ [0.0, 1.0]. 0.0 = ephemeral scratch;
    /// 1.0 = never-forget user profile info. Default `0.5` is fine for most
    /// memories.
    pub importance: f32,
    /// Free-form metadata. Stored verbatim as JSON; queryable in `MemoryFilter`.
    pub metadata: Option<&'a serde_json::Value>,
    /// Optional hard TTL. `None` = the memory lives until the forgetting
    /// policy cleans it up based on access/importance.
    pub expires_at: Option<DateTime<Utc>>,
    /// Caller-provided tags, merged with any tags extracted from `content`
    /// (e.g. `#hashtags` from the RegexWikiExtractor).
    pub tags: &'a [String],
    /// If false, skip the embedding step. Useful when:
    ///   - the caller already has an embedding and will store it via
    ///     `insert_vector` directly; or
    ///   - the memory is a small structured metadata row not meant for
    ///     semantic search.
    pub embed: bool,
    /// If false, skip entity/relation extraction + graph storage. Use for
    /// non-narrative content (raw log lines, etc.) where graph pollution is
    /// a real risk.
    pub extract_graph: bool,
}

impl<'a> Default for RememberInput<'a> {
    fn default() -> Self {
        Self {
            content: "",
            source: MemorySource::Agent,
            importance: 0.5,
            metadata: None,
            expires_at: None,
            tags: &[],
            embed: true,
            extract_graph: true,
        }
    }
}

/// RememberOutput — describes what actually happened on the write side so
/// callers can log / notify observers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberOutput {
    pub uuid: Uuid,
    /// True when this `remember` call found an identical (by content hash)
    /// existing memory and returned the previous record instead of inserting
    /// a new one. Tags, graph edges, and embeddings are still attached for
    /// the new call, so a `true` here is informational only — not an error.
    pub duplicate: bool,
    /// True if a new embedding was stored for this memory. False on dupes
    /// where the vector already existed, or when `embed = false`.
    pub embedding_stored: bool,
    /// Number of entities upserted to the graph table from this memory.
    pub entities_extracted: usize,
    /// Number of relation edges upserted from this memory.
    pub relations_extracted: usize,
    /// Final set of tags attached after merging caller-tags + extractor tags.
    pub tags: Vec<String>,
}

pub async fn remember(svc: &MemoryService, input: RememberInput<'_>) -> NovaResult<RememberOutput> {
    // --- 1. Validation -------------------------------------------------------
    let content = input.content.trim();
    if content.is_empty() {
        return Err(NovaError::validation("remember: content must not be empty"));
    }
    if !(0.0..=1.0).contains(&input.importance) || !input.importance.is_finite() {
        return Err(NovaError::validation(format!(
            "remember: importance must be in [0.0, 1.0], got {}",
            input.importance
        )));
    }

    // --- 2. Extraction (before insert so we have merged tags + candidates) -
    let extraction = if input.extract_graph {
        svc.extractor.extract(content).await.unwrap_or_default()
    } else {
        Default::default()
    };

    // Merge caller tags + extractor tags, preserving order, dedup.
    let merged = merge_tags(input.tags, &extraction.tags);

    // --- 3. Insert (or dedupe) the core memory row -------------------------
    let insert_in = crate::storage::memory::InsertMemoryInput {
        content,
        source: input.source,
        importance: input.importance,
        metadata: input.metadata,
        expires_at: input.expires_at,
        tags: &merged,
    };
    let outcome = svc.memory_repo.insert(&svc.database, insert_in).await?;
    let uuid = outcome.uuid();
    let duplicate = outcome.is_duplicate();

    // --- 4. Embedding -------------------------------------------------------
    // If the memory was a dupe the vector store already has a vector for the
    // same memory_uuid from a prior call. Skip the embed call in that case
    // to avoid wasting provider quota.
    let mut embedding_stored = false;
    if input.embed && !duplicate {
        let meta = svc.embedding.meta();
        let vec = svc.embedding.embed_one(content).await?;
        if vec.len() != meta.dims {
            return Err(NovaError::embedding_msg(format!(
                "embed_one returned dims={} expected dims={} for provider {}",
                vec.len(),
                meta.dims,
                meta.provider
            )));
        }
        svc.vector_store.insert_vector(uuid, &meta.provider, &meta.model, &vec).await?;
        embedding_stored = true;
    }

    // --- 5. Graph: upsert entities + relations -----------------------------
    let mut entities_extracted = 0usize;
    let mut relations_extracted = 0usize;
    if !extraction.entities.is_empty() {
        // Upsert each entity. SqliteEntityRepository::upsert handles name+type
        // collisions; failures are per-entity so we count successes.
        let mut entity_names: std::collections::HashMap<(String, String), Uuid> =
            std::collections::HashMap::new();
        for ent in &extraction.entities {
            let r = upsert_one_entity(&svc.entity_repo, svc, ent).await;
            match r {
                Ok(ent_uuid) => {
                    entity_names.insert((ent.name.clone(), ent.entity_type.clone()), ent_uuid);
                    entities_extracted += 1;
                },
                Err(e) => {
                    // Skip single bad entities; log at warn but don't fail remember.
                    tracing::warn!(entity = %ent.name, error = %e, "skip entity upsert");
                },
            }
        }

        // Then create relations between the entities we actually succeeded on.
        if !extraction.relations.is_empty() && entity_names.len() >= 2 {
            for rel in dedupe_relations(&extraction.relations) {
                let r = insert_one_relation(&svc.relation_repo, svc, &entity_names, &rel).await;
                match r {
                    Ok(true) => relations_extracted += 1,
                    Ok(false) => {}, // idempotent no-op
                    Err(e) => {
                        tracing::warn!(source = %rel.source_name, target = %rel.target_name, error = %e, "skip relation");
                    },
                }
            }
        }
    }

    Ok(RememberOutput {
        uuid,
        duplicate,
        embedding_stored,
        entities_extracted,
        relations_extracted,
        tags: merged,
    })
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn merge_tags(caller: &[String], extractor: &[String]) -> Vec<String> {
    use std::collections::BTreeSet;
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out = Vec::with_capacity(caller.len() + extractor.len());
    for t in caller.iter().chain(extractor.iter()) {
        let trimmed = t.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.clone()) {
            out.push(trimmed);
        }
    }
    out
}

async fn upsert_one_entity(
    repo: &SqliteEntityRepository,
    svc: &MemoryService,
    ent: &crate::graph::extractor::EntityCandidate,
) -> NovaResult<Uuid> {
    let name = ent.name.trim().to_string();
    if name.is_empty() {
        return Err(NovaError::validation("entity name empty"));
    }
    let etype = if ent.entity_type.trim().is_empty() {
        "unknown".to_string()
    } else {
        ent.entity_type.trim().to_string()
    };
    let r = repo
        .upsert(
            &svc.database,
            UpsertEntityInput {
                name: &name,
                r#type: &etype,
                description: ent.description.as_deref(),
                metadata: None,
            },
        )
        .await?;
    Ok(r.uuid())
}

async fn insert_one_relation(
    repo: &SqliteRelationRepository,
    svc: &MemoryService,
    entity_uuids: &std::collections::HashMap<(String, String), Uuid>,
    rel: &RelationCandidate,
) -> NovaResult<bool> {
    // Best-effort: look up the entity by exact (name, type) first with
    // `(name, "unknown")` fallback so capitalised-proper-noun entities can
    // still be matched against the extractor's "unknown" type.
    let lookup = |n: &str| -> Option<Uuid> {
        entity_uuids
            .get(&(n.to_string(), "unknown".to_string()))
            .copied()
            .or_else(|| entity_uuids.iter().find(|((name, _), _)| name == n).map(|(_, u)| *u))
    };
    let Some(src) = lookup(&rel.source_name) else {
        return Ok(false);
    };
    let Some(tgt) = lookup(&rel.target_name) else {
        return Ok(false);
    };
    if src == tgt {
        return Ok(false); // skip self-loops
    }
    let conf = rel.confidence.clamp(0.0, 1.0);
    let pred = if rel.predicate.trim().is_empty() {
        "mentions".to_string()
    } else {
        rel.predicate.trim().to_string()
    };
    let inserted = repo
        .insert(
            &svc.database,
            InsertRelationInput {
                source_uuid: src,
                target_uuid: tgt,
                predicate: &pred,
                confidence: conf,
                memory_uuid: None,
                metadata: None,
                idempotent: true,
            },
        )
        .await?;
    Ok(inserted.is_inserted())
}

// InsertRelationOutcome::is_inserted() is defined directly on the enum in
// crate::storage::relation — no local trait shim needed here.

fn dedupe_relations(rels: &[RelationCandidate]) -> Vec<RelationCandidate> {
    use std::collections::BTreeMap;
    // (src, pred, tgt) → highest confidence candidate.
    let mut best: BTreeMap<(String, String, String), RelationCandidate> = BTreeMap::new();
    for r in rels {
        let key = (r.source_name.clone(), r.predicate.clone(), r.target_name.clone());
        match best.entry(key) {
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(r.clone());
            },
            std::collections::btree_map::Entry::Occupied(mut e) => {
                if r.confidence > e.get().confidence {
                    e.insert(r.clone());
                }
            },
        }
    }
    best.into_values().collect()
}

/// Extra helpers for tests: create a service with a mock extractor. Exposed so
/// the integration tests in M3.x don't have to repeat this boilerplate.
pub fn service_for_tests(
    database: crate::storage::Database,
    embed_dims: usize,
    extractor: Option<Arc<dyn crate::graph::extractor::EntityExtractor>>,
) -> MemoryService {
    use crate::embedding::MockEmbeddingProvider;
    let provider: crate::embedding::SharedEmbeddingProvider =
        Arc::new(MockEmbeddingProvider::new(embed_dims));
    match extractor {
        Some(e) => MemoryService::with_parts(database, provider, e),
        None => MemoryService::new(database, provider),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Uuid;
    use crate::config::StorageConfig;
    use crate::graph::extractor::{EntityExtractor, RegexWikiExtractor};
    use crate::storage::{Database, MemoryStatus};
    use std::sync::Arc;

    async fn temp_svc(extractor: bool) -> MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-m3-svc-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        let db = Database::open(cfg).await.unwrap();
        let ext = if extractor {
            Arc::new(RegexWikiExtractor::new()) as Arc<dyn EntityExtractor>
        } else {
            Arc::new(crate::graph::extractor::NoopExtractor) as Arc<dyn EntityExtractor>
        };
        service_for_tests(db, 8, Some(ext))
    }

    #[tokio::test]
    async fn empty_content_is_rejected() {
        let svc = temp_svc(false).await;
        let bad = RememberInput { content: "   ", ..Default::default() };
        let err = svc.remember(bad).await.unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn importance_out_of_range_rejected() {
        let svc = temp_svc(false).await;
        for bad in [-0.1_f32, 1.5, f32::NAN, f32::INFINITY] {
            let inp = RememberInput { content: "hi", importance: bad, ..Default::default() };
            let err = svc.remember(inp).await.unwrap_err();
            assert_eq!(err.code(), crate::error::ErrorCode::Validation);
        }
    }

    #[tokio::test]
    async fn remember_basic_creates_row_and_embedding() {
        let svc = temp_svc(false).await;
        let out = svc
            .remember(RememberInput {
                content: "I like strawberries.",
                importance: 0.7,
                tags: &["fruit".to_string()],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!out.duplicate);
        assert!(out.embedding_stored);
        assert_eq!(out.tags, vec!["fruit".to_string()]);

        // Make sure the MemoryRepository returns it.
        let mem = svc.get_memory(out.uuid).await.unwrap();
        assert_eq!(mem.content, "I like strawberries.");
        assert_eq!(mem.importance, 0.7);
        assert_eq!(mem.status, MemoryStatus::Active);
        assert!(mem.access_count == 0);
    }

    #[tokio::test]
    async fn remember_duplicate_dedupes_same_content() {
        let svc = temp_svc(false).await;
        let a = svc
            .remember(RememberInput { content: "shared-content", ..Default::default() })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput { content: "  shared-content  ", ..Default::default() })
            .await
            .unwrap();
        assert_eq!(a.uuid, b.uuid);
        assert!(!a.duplicate);
        assert!(b.duplicate);
        assert!(!b.embedding_stored, "dupe should not embed again");
    }

    #[tokio::test]
    async fn remember_extracts_entities_and_hashtags_with_regex_extractor() {
        let svc = temp_svc(true).await;
        let out = svc
            .remember(RememberInput {
                content: "#docs [[Trae]] told Alice Smith to deploy Kubernetes.",
                tags: &["v1".to_string()],
                ..Default::default()
            })
            .await
            .unwrap();
        // Tags from extractor (#docs) + caller (#v1) both appear.
        assert!(out.tags.iter().any(|t| t == "docs"), "tags = {:?}", out.tags);
        assert!(out.tags.iter().any(|t| t == "v1"));
        assert!(out.entities_extracted >= 3, "entities = {}", out.entities_extracted);
        // Trae wiki entity, Alice Smith, Kubernetes.
        assert!(out.relations_extracted >= 1);
    }

    #[tokio::test]
    async fn remember_embed_false_skips_storage() {
        let svc = temp_svc(false).await;
        let out = svc
            .remember(RememberInput {
                content: "metadata-only row",
                embed: false,
                extract_graph: false,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!out.embedding_stored);
        assert_eq!(out.entities_extracted, 0);
        assert_eq!(out.relations_extracted, 0);
    }
}
