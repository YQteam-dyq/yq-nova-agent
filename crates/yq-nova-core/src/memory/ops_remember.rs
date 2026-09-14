use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    MemoryService,
    chunk::{ChunkInfo, ChunkOptions, chunk_text},
    quality::dedup::{DedupOptions, SimilarHit},
};
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

#[derive(Debug, Clone)]
pub struct RememberInput<'a> {
    pub content: &'a str,
    pub source: MemorySource,
    pub importance: f32,
    pub metadata: Option<&'a serde_json::Value>,
    pub expires_at: Option<DateTime<Utc>>,
    pub tags: &'a [String],
    pub embed: bool,
    pub extract_graph: bool,
    pub chunk_options: Option<ChunkOptions>,
    pub dedup: Option<DedupOptions>,
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
            chunk_options: None,
            dedup: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberOutput {
    pub uuid: Uuid,
    pub duplicate: bool,
    pub embedding_stored: bool,
    pub entities_extracted: usize,
    pub relations_extracted: usize,
    pub tags: Vec<String>,
    pub chunks: Vec<ChunkInfo>,
    pub near_duplicates: Vec<SimilarHit>,
    pub merged: bool,
}

pub async fn remember(svc: &MemoryService, input: RememberInput<'_>) -> NovaResult<RememberOutput> {
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

    if let Some(ref opts) = input.chunk_options {
        if opts.enabled {
            if let Err(msg) = opts.validate() {
                return Err(NovaError::validation(format!("chunk_options: {msg}")));
            }
            let chunk_texts = chunk_text(content, opts);
            if chunk_texts.len() > 1 {
                let group_uuid = Uuid::new_v4();
                let total = chunk_texts.len();
                let mut chunks = Vec::with_capacity(total);
                let mut total_entities = 0usize;
                let mut total_relations = 0usize;
                let mut any_embedding_stored = false;
                let mut merged_tags = Vec::new();

                for (i, ct) in chunk_texts.iter().enumerate() {
                    let chunk_meta = build_chunk_metadata(input.metadata, group_uuid, i, total);
                    let out = remember_one(
                        svc,
                        ct,
                        input.tags,
                        Some(&chunk_meta),
                        input.source,
                        input.importance,
                        input.expires_at,
                        input.embed,
                        input.extract_graph,
                        input.dedup,
                    )
                    .await?;
                    if i == 0 {
                        merged_tags = out.tags;
                    }
                    total_entities += out.entities_extracted;
                    total_relations += out.relations_extracted;
                    if out.embedding_stored {
                        any_embedding_stored = true;
                    }
                    chunks.push(ChunkInfo {
                        uuid: out.uuid,
                        chunk_index: i,
                        chunk_total: total,
                    });
                }

                return Ok(RememberOutput {
                    uuid: group_uuid,
                    duplicate: false,
                    embedding_stored: any_embedding_stored,
                    entities_extracted: total_entities,
                    relations_extracted: total_relations,
                    tags: merged_tags,
                    chunks,
                    near_duplicates: Vec::new(),
                    merged: false,
                });
            }
        }
    }

    remember_one(
        svc,
        content,
        input.tags,
        input.metadata,
        input.source,
        input.importance,
        input.expires_at,
        input.embed,
        input.extract_graph,
        input.dedup,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn remember_one(
    svc: &MemoryService,
    content: &str,
    tags: &[String],
    metadata: Option<&serde_json::Value>,
    source: MemorySource,
    importance: f32,
    expires_at: Option<DateTime<Utc>>,
    embed: bool,
    extract_graph: bool,
    dedup: Option<DedupOptions>,
) -> NovaResult<RememberOutput> {
    if let Some(opts) = dedup {
        if opts.enabled {
            let decision = super::quality::dedup::decide(svc, content, &opts).await?;
            if let Some(existing_uuid) = decision.blocked_by {
                let existing = svc.memory_repo.get_by_uuid(&svc.database, existing_uuid).await?;
                let hits = super::quality::dedup::detect(svc, content, &opts).await?;
                return Ok(RememberOutput {
                    uuid: existing_uuid,
                    duplicate: true,
                    embedding_stored: false,
                    entities_extracted: 0,
                    relations_extracted: 0,
                    tags: existing.tags.clone(),
                    chunks: Vec::new(),
                    near_duplicates: hits,
                    merged: decision.merged,
                });
            }
        }
    }

    let extraction = if extract_graph {
        svc.extractor.extract(content).await.unwrap_or_default()
    } else {
        Default::default()
    };

    let merged = merge_tags(tags, &extraction.tags);

    let insert_in = crate::storage::memory::InsertMemoryInput {
        content,
        source,
        importance,
        metadata,
        expires_at,
        tags: &merged,
    };
    let outcome = svc.memory_repo.insert(&svc.database, insert_in).await?;
    let uuid = outcome.uuid();
    let duplicate = outcome.is_duplicate();

    let mut embedding_stored = false;
    if embed && !duplicate {
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

    let mut entities_extracted = 0usize;
    let mut relations_extracted = 0usize;
    if !extraction.entities.is_empty() {
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
                    tracing::warn!(entity = %ent.name, error = %e, "skip entity upsert");
                },
            }
        }

        if !extraction.relations.is_empty() && entity_names.len() >= 2 {
            for rel in dedupe_relations(&extraction.relations) {
                let r = insert_one_relation(&svc.relation_repo, svc, &entity_names, &rel).await;
                match r {
                    Ok(true) => relations_extracted += 1,
                    Ok(false) => {},
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
        chunks: Vec::new(),
        near_duplicates: Vec::new(),
        merged: false,
    })
}

fn build_chunk_metadata(
    base: Option<&serde_json::Value>,
    group_uuid: Uuid,
    chunk_index: usize,
    chunk_total: usize,
) -> serde_json::Value {
    let mut meta = base.cloned().unwrap_or_else(|| serde_json::json!({}));
    if let serde_json::Value::Object(ref mut map) = meta {
        map.insert("chunk_group".to_string(), serde_json::json!(group_uuid.to_string()));
        map.insert("chunk_index".to_string(), serde_json::json!(chunk_index));
        map.insert("chunk_total".to_string(), serde_json::json!(chunk_total));
    }
    meta
}

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
        return Ok(false);
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

fn dedupe_relations(rels: &[RelationCandidate]) -> Vec<RelationCandidate> {
    use std::collections::BTreeMap;
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
    use std::sync::Arc;

    use super::*;
    use crate::{
        Uuid,
        config::StorageConfig,
        graph::extractor::{EntityExtractor, RegexWikiExtractor},
        memory::chunk::SplitBy,
        storage::{Database, MemoryStatus},
    };

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
        let bad = RememberInput {
            content: "   ",
            ..Default::default()
        };
        let err = svc.remember(bad).await.unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn importance_out_of_range_rejected() {
        let svc = temp_svc(false).await;
        for bad in [-0.1_f32, 1.5, f32::NAN, f32::INFINITY] {
            let inp = RememberInput {
                content: "hi",
                importance: bad,
                ..Default::default()
            };
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
        assert!(out.chunks.is_empty());

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
            .remember(RememberInput {
                content: "shared-content",
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "  shared-content  ",
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(a.uuid, b.uuid);
        assert!(!a.duplicate);
        assert!(b.duplicate);
        assert!(!b.embedding_stored, "dupe should not embed again");
    }

    #[tokio::test]
    async fn remember_semantic_dedup_merge_blocks_and_merges() {
        let svc = temp_svc(false).await;
        let a = svc
            .remember(RememberInput {
                content: "shared content alpha project",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let opts = crate::memory::quality::dedup::DedupOptions {
            enabled: true,
            threshold: 0.999,
            mode: crate::memory::quality::dedup::DedupMode::Merge,
            max_candidates: 10,
        };
        let b = svc
            .remember(RememberInput {
                content: "shared content alpha project",
                dedup: Some(opts),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(a.uuid, b.uuid);
        assert!(b.duplicate);
        assert!(b.merged);
        assert!(!b.near_duplicates.is_empty());
        let combined = svc.get_memory(a.uuid).await.unwrap();
        assert_eq!(combined.content, "shared content alpha project\nshared content alpha project");
    }

    #[tokio::test]
    async fn remember_semantic_dedup_report_flags_conflict() {
        let svc = temp_svc(false).await;
        let a = svc
            .remember(RememberInput {
                content: "shared content beta project",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let opts = crate::memory::quality::dedup::DedupOptions {
            enabled: true,
            threshold: 0.999,
            mode: crate::memory::quality::dedup::DedupMode::Report,
            max_candidates: 10,
        };
        let b = svc
            .remember(RememberInput {
                content: "shared content beta project",
                dedup: Some(opts),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(a.uuid, b.uuid);
        assert!(b.duplicate);
        assert!(!b.merged);
        assert!(!b.near_duplicates.is_empty());
        let unchanged = svc.get_memory(a.uuid).await.unwrap();
        assert_eq!(unchanged.content, "shared content beta project");
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
        assert!(out.tags.iter().any(|t| t == "docs"), "tags = {:?}", out.tags);
        assert!(out.tags.iter().any(|t| t == "v1"));
        assert!(out.entities_extracted >= 3, "entities = {}", out.entities_extracted);
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

    #[tokio::test]
    async fn chunking_disabled_behavior_unchanged() {
        let svc = temp_svc(false).await;
        let out = svc
            .remember(RememberInput {
                content: "regular remember without chunking enabled",
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(out.chunks.is_empty());
        let mem = svc.get_memory(out.uuid).await.unwrap();
        assert_eq!(mem.content, "regular remember without chunking enabled");
    }

    #[tokio::test]
    async fn chunking_enabled_single_chunk_falls_back() {
        let svc = temp_svc(false).await;
        let opts = ChunkOptions {
            enabled: true,
            split_by: SplitBy::Paragraph,
            max_chars: 2000,
            overlap_chars: 150,
        };
        let out = svc
            .remember(RememberInput {
                content: "short content that fits in one chunk",
                chunk_options: Some(opts),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(out.chunks.is_empty(), "single chunk should fall back to regular remember");
        let mem = svc.get_memory(out.uuid).await.unwrap();
        assert_eq!(mem.content, "short content that fits in one chunk");
    }

    #[tokio::test]
    async fn chunking_enabled_multi_chunk_stores_chunks() {
        let svc = temp_svc(false).await;
        let mut text = String::new();
        for i in 0..10 {
            if i > 0 {
                text.push_str("\n\n");
            }
            text.push_str(&format!(
                "This is paragraph {} in the chunking test for the remember pipeline.",
                i
            ));
        }
        let opts = ChunkOptions {
            enabled: true,
            split_by: SplitBy::Paragraph,
            max_chars: 200,
            overlap_chars: 30,
        };
        let out = svc
            .remember(RememberInput {
                content: &text,
                chunk_options: Some(opts),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!out.chunks.is_empty(), "should produce chunks");
        assert!(out.chunks.len() > 1, "should produce multiple chunks, got {}", out.chunks.len());

        for ci in &out.chunks {
            assert!(ci.chunk_total == out.chunks.len());
            let mem = svc.get_memory(ci.uuid).await.unwrap();
            let chunk_group =
                mem.metadata.get("chunk_group").and_then(|v| v.as_str()).unwrap().to_string();
            assert_eq!(chunk_group, out.uuid.to_string(), "chunk_group should match group_uuid");
            assert_eq!(
                mem.metadata.get("chunk_index").and_then(|v| v.as_u64()).unwrap() as usize,
                ci.chunk_index
            );
            assert_eq!(
                mem.metadata.get("chunk_total").and_then(|v| v.as_u64()).unwrap() as usize,
                ci.chunk_total
            );
        }

        let original = svc.memory_repo.get_by_uuid(&svc.database, out.uuid).await;
        assert!(original.is_err(), "original group uuid should not be a stored memory");
    }

    #[tokio::test]
    async fn chunking_invalid_options_rejected() {
        let svc = temp_svc(false).await;
        let opts = ChunkOptions {
            enabled: true,
            max_chars: 100,
            ..Default::default()
        };
        let err = svc
            .remember(RememberInput {
                content: "some text here",
                chunk_options: Some(opts),
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }
}
