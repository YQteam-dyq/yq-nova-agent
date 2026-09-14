use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use crate::storage::{
    entity::{Direction, EntityRecord, TraverseNode},
    relation::RelationRecord,
};

pub mod batch;
pub mod extractor;
pub mod llm_extractor;
pub mod policy;
pub mod quality;

pub use batch::{
    BatchExtractOptions, BatchExtractQueue, BatchExtractSummary, BatchJobState, BatchJobView,
};
use extractor::{
    EntityCandidate, EntityExtractor, Extraction, NoopExtractor, RegexWikiExtractor,
    RelationCandidate,
};
pub use llm_extractor::{DEFAULT_EXTRACT_PROMPT, LLMEntityExtractor, LlmExtractorConfig};
pub use policy::{ExtractionFrequency, ExtractionPolicy, FilteringExtractor};
pub use quality::{
    ExtractionFeedbackInput, FeedbackRecord, FeedbackTargetKind, FeedbackVerdict, KindStats,
    QualityStats, SqliteQualityStore,
};

use crate::{
    config::GraphConfig,
    error::{NovaError, NovaResult},
    storage::{
        Database,
        entity::{EntityRepository, SqliteEntityRepository, UpsertEntityInput, UpsertOutcome},
        relation::{InsertRelationInput, RelationRepository, SqliteRelationRepository},
    },
};

pub fn extractor_from_config(config: &GraphConfig) -> NovaResult<Arc<dyn EntityExtractor>> {
    if config.extract_llm.trim().is_empty() {
        return Ok(Arc::new(RegexWikiExtractor::new()));
    }
    let chat = config.openai_compatible_chat.get(&config.extract_llm).ok_or_else(|| {
        NovaError::config_msg(format!(
            "graph.extract_llm references unknown chat provider '{}'",
            config.extract_llm
        ))
    })?;
    let llm_config = LlmExtractorConfig {
        base_url: chat.base_url.clone(),
        api_key: chat.api_key.clone(),
        model: chat.model.clone(),
        timeout: chat.timeout,
        ..Default::default()
    };
    let prompt = match &config.extract_prompt_file {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| NovaError::config_msg(format!("read graph.extract_prompt_file: {e}")))?,
        None => DEFAULT_EXTRACT_PROMPT.to_string(),
    };
    Ok(Arc::new(LLMEntityExtractor::new_with_prompt(llm_config, prompt)?))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphExtractOpts {
    pub enabled: bool,

    pub upsert_entities: bool,

    pub create_relations: bool,

    pub min_confidence: f32,
}

impl Default for GraphExtractOpts {
    fn default() -> Self {
        Self {
            enabled: false,
            upsert_entities: true,
            create_relations: true,
            min_confidence: 0.0,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LinkResult {
    pub entities: Vec<(EntityCandidate, Uuid)>,

    pub entities_upserted: usize,

    pub relations_created: usize,

    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEntitiesInput {
    pub keep_uuid: Uuid,

    pub discard_uuids: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEntitiesOutput {
    pub kept_uuid: Uuid,

    pub merged: Vec<Uuid>,

    pub remapped_relations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraverseOpts {
    pub max_depth: u8,

    pub max_nodes: usize,

    pub predicate_whitelist: Vec<String>,

    pub min_confidence: f32,
}

impl Default for TraverseOpts {
    fn default() -> Self {
        Self {
            max_depth: 2,
            max_nodes: 200,
            predicate_whitelist: Vec::new(),
            min_confidence: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct GraphService {
    pub database: Database,

    pub extractor: Arc<dyn EntityExtractor>,

    pub entity_repo: SqliteEntityRepository,

    pub relation_repo: SqliteRelationRepository,
}

impl std::fmt::Debug for GraphService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphService").finish_non_exhaustive()
    }
}

impl GraphService {
    pub fn with_parts(database: Database, extractor: Arc<dyn EntityExtractor>) -> Self {
        Self {
            database,
            extractor,
            entity_repo: SqliteEntityRepository::new(),
            relation_repo: SqliteRelationRepository::new(),
        }
    }

    pub fn new(database: Database) -> Self {
        Self::with_parts(database, Arc::new(NoopExtractor))
    }

    pub async fn extract_and_link(
        &self,
        namespace_id: i64,
        text: &str,
        opts: &GraphExtractOpts,
    ) -> NovaResult<LinkResult> {
        if !opts.enabled || text.trim().is_empty() {
            return Ok(LinkResult::default());
        }
        let extraction: Extraction = self.extractor.extract(text).await?;

        let mut entity_uuids: std::collections::HashMap<(String, String), Uuid> =
            std::collections::HashMap::new();
        let mut entities: Vec<(EntityCandidate, Uuid)> = Vec::new();
        let mut upserted = 0usize;
        if opts.upsert_entities {
            for ent in &extraction.entities {
                let r = upsert_one_entity(self, namespace_id, ent).await;
                if let Some((outcome, uuid)) = r {
                    if matches!(outcome, UpsertOutcome::Created(_)) {
                        upserted += 1;
                    }
                    entity_uuids.insert((ent.name.clone(), ent.entity_type.clone()), uuid);
                    entities.push((ent.clone(), uuid));
                }
            }
        } else if opts.create_relations {
            for ent in &extraction.entities {
                let name = ent.name.trim().to_string();
                if name.is_empty() {
                    continue;
                }
                let existing = match self
                    .entity_repo
                    .find_by_name(&self.database, namespace_id, &name)
                    .await
                {
                    Ok(rows) => rows.into_iter().next(),
                    Err(_) => None,
                };
                if let Some(rec) = existing {
                    entity_uuids.insert((name.clone(), ent.entity_type.clone()), rec.uuid);
                    entities.push((ent.clone(), rec.uuid));
                }
            }
        }

        let mut created = 0usize;
        if opts.create_relations && entity_uuids.len() >= 2 {
            for rel in dedupe_by_confidence(&extraction.relations) {
                if rel.confidence < opts.min_confidence {
                    continue;
                }
                if let Some(true) = link_one_relation(self, namespace_id, &entity_uuids, &rel).await
                {
                    created += 1;
                }
            }
        }

        Ok(LinkResult {
            entities,
            entities_upserted: upserted,
            relations_created: created,
            tags: extraction.tags,
        })
    }

    pub async fn traverse_graph(
        &self,
        namespace_id: i64,
        start: Uuid,
        opts: TraverseOpts,
    ) -> NovaResult<Vec<TraverseNode>> {
        let _start_ent = self.entity_repo.get_by_uuid(&self.database, namespace_id, start).await?;
        let nodes = self
            .relation_repo
            .bfs_traverse(
                &self.database,
                namespace_id,
                start,
                Direction::Both,
                opts.max_depth,
                opts.max_nodes,
                &opts.predicate_whitelist,
                opts.min_confidence,
            )
            .await?;
        Ok(nodes)
    }

    pub async fn list_entities(
        &self,
        namespace_id: i64,
        name_prefix: Option<&str>,
        entity_type: Option<&str>,
        limit: usize,
    ) -> NovaResult<Vec<EntityRecord>> {
        self.entity_repo
            .list(&self.database, namespace_id, name_prefix, entity_type, limit, 0)
            .await
    }

    pub async fn merge_entities(
        &self,
        namespace_id: i64,
        input: MergeEntitiesInput,
    ) -> NovaResult<MergeEntitiesOutput> {
        let keep = input.keep_uuid;
        let discards = input.discard_uuids;

        if discards.is_empty() || discards.len() > 50 {
            return Err(NovaError::validation("discard_uuids must contain between 1 and 50 UUIDs"));
        }
        if discards.contains(&keep) {
            return Err(NovaError::validation("keep_uuid must not be in discard_uuids"));
        }

        let _keep_entity = self.entity_repo.get_by_uuid(&self.database, namespace_id, keep).await?;
        for &d in &discards {
            self.entity_repo.get_by_uuid(&self.database, namespace_id, d).await?;
        }

        let pool = &self.database.pool;
        let keep_str = keep.to_string();
        let discard_set: std::collections::HashSet<Uuid> = discards.iter().copied().collect();
        let mut remapped = 0usize;

        for &d in &discards {
            let d_str = d.to_string();

            let outgoing: Vec<(String, String)> = sqlx::query_as(
                "SELECT uuid, target_uuid FROM relations WHERE source_uuid = ?1 AND namespace_id \
                 = ?2",
            )
            .bind(&d_str)
            .bind(namespace_id)
            .fetch_all(pool)
            .await
            .map_err(NovaError::storage)?;

            for (rel_uuid, tgt_str) in &outgoing {
                let tgt = match Uuid::parse_str(tgt_str) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                if tgt == keep || discard_set.contains(&tgt) {
                    sqlx::query("DELETE FROM relations WHERE uuid = ?1 AND namespace_id = ?2")
                        .bind(rel_uuid)
                        .bind(namespace_id)
                        .execute(pool)
                        .await
                        .map_err(NovaError::storage)?;
                } else {
                    sqlx::query(
                        "UPDATE relations SET source_uuid = ?1 WHERE uuid = ?2 AND namespace_id = \
                         ?3",
                    )
                    .bind(&keep_str)
                    .bind(rel_uuid)
                    .bind(namespace_id)
                    .execute(pool)
                    .await
                    .map_err(NovaError::storage)?;
                    remapped += 1;
                }
            }

            let incoming: Vec<(String, String)> = sqlx::query_as(
                "SELECT uuid, source_uuid FROM relations WHERE target_uuid = ?1 AND namespace_id \
                 = ?2",
            )
            .bind(&d_str)
            .bind(namespace_id)
            .fetch_all(pool)
            .await
            .map_err(NovaError::storage)?;

            for (rel_uuid, src_str) in &incoming {
                let src = match Uuid::parse_str(src_str) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                if discard_set.contains(&src) {
                    continue;
                }
                if src == keep {
                    sqlx::query("DELETE FROM relations WHERE uuid = ?1 AND namespace_id = ?2")
                        .bind(rel_uuid)
                        .bind(namespace_id)
                        .execute(pool)
                        .await
                        .map_err(NovaError::storage)?;
                } else {
                    sqlx::query(
                        "UPDATE relations SET target_uuid = ?1 WHERE uuid = ?2 AND namespace_id = \
                         ?3",
                    )
                    .bind(&keep_str)
                    .bind(rel_uuid)
                    .bind(namespace_id)
                    .execute(pool)
                    .await
                    .map_err(NovaError::storage)?;
                    remapped += 1;
                }
            }

            self.entity_repo.delete(&self.database, namespace_id, d).await?;
        }

        sqlx::query("DELETE FROM relations WHERE source_uuid = target_uuid AND namespace_id = ?1")
            .bind(namespace_id)
            .execute(pool)
            .await
            .map_err(NovaError::storage)?;

        Ok(MergeEntitiesOutput {
            kept_uuid: keep,
            merged: discards,
            remapped_relations: remapped,
        })
    }
}

async fn upsert_one_entity(
    svc: &GraphService,
    namespace_id: i64,
    ent: &EntityCandidate,
) -> Option<(UpsertOutcome, Uuid)> {
    let name = ent.name.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let etype = if ent.entity_type.trim().is_empty() {
        "unknown".to_string()
    } else {
        ent.entity_type.trim().to_string()
    };
    let outcome = svc
        .entity_repo
        .upsert(
            &svc.database,
            UpsertEntityInput {
                namespace_id,
                name: &name,
                r#type: &etype,
                description: ent.description.as_deref(),
                metadata: None,
            },
        )
        .await
        .ok()?;
    let uuid = outcome.uuid();
    Some((outcome, uuid))
}

async fn link_one_relation(
    svc: &GraphService,
    namespace_id: i64,
    entity_uuids: &std::collections::HashMap<(String, String), Uuid>,
    rel: &RelationCandidate,
) -> Option<bool> {
    let lookup = |n: &str| -> Option<Uuid> {
        entity_uuids
            .get(&(n.to_string(), "unknown".to_string()))
            .copied()
            .or_else(|| entity_uuids.iter().find(|((name, _), _)| name == n).map(|(_, u)| *u))
    };
    let src = lookup(&rel.source_name)?;
    let tgt = lookup(&rel.target_name)?;
    if src == tgt {
        return Some(false);
    }
    let pred = if rel.predicate.trim().is_empty() {
        "mentions".to_string()
    } else {
        rel.predicate.trim().to_string()
    };
    let conf = rel.confidence.clamp(0.0, 1.0);
    let outcome = svc
        .relation_repo
        .insert(
            &svc.database,
            InsertRelationInput {
                namespace_id,
                source_uuid: src,
                target_uuid: tgt,
                predicate: &pred,
                confidence: conf,
                memory_uuid: None,
                metadata: None,
                idempotent: true,
            },
        )
        .await
        .ok()?;
    Some(outcome.is_inserted())
}

fn dedupe_by_confidence(rels: &[RelationCandidate]) -> Vec<RelationCandidate> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Uuid, config::StorageConfig, graph::extractor::RegexWikiExtractor, storage::Database,
    };

    const NS: i64 = crate::storage::namespace::DEFAULT_NAMESPACE_ID;

    async fn temp_svc() -> GraphService {
        let dir = std::env::temp_dir().join(format!("yq-nova-m3-graph-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        let db = Database::open(cfg).await.unwrap();
        GraphService::with_parts(db, Arc::new(RegexWikiExtractor::new()))
    }

    #[tokio::test]
    async fn disabled_extract_is_noop() {
        let svc = temp_svc().await;
        let r = svc
            .extract_and_link(
                NS,
                "[[A]] and [[B]]",
                &GraphExtractOpts {
                    enabled: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(r.entities_upserted, 0);
        assert_eq!(r.relations_created, 0);
    }

    #[tokio::test]
    async fn wikilinks_create_entities_and_mentions_edges() {
        let svc = temp_svc().await;
        let r = svc
            .extract_and_link(
                NS,
                "#todo [[Alice Smith]] had a meeting with [[Bob Jones]] at [[Acme Corp]].",
                &GraphExtractOpts {
                    enabled: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(r.tags.iter().any(|t| t == "todo"));

        assert!(r.entities_upserted >= 3, "entities upserted: {}", r.entities_upserted);

        assert!(r.relations_created >= 3);
    }

    async fn upsert(db: &Database, repo: &SqliteEntityRepository, n: &str, t: &str) -> Uuid {
        let out = repo
            .upsert(
                db,
                UpsertEntityInput {
                    namespace_id: NS,
                    name: n,
                    r#type: t,
                    description: None,
                    metadata: None,
                },
            )
            .await
            .unwrap();
        out.uuid()
    }

    async fn insert_edge(
        db: &Database,
        repo: &SqliteRelationRepository,
        src: Uuid,
        tgt: Uuid,
        p: &str,
        c: f32,
    ) {
        repo.insert(
            db,
            InsertRelationInput {
                namespace_id: NS,
                source_uuid: src,
                target_uuid: tgt,
                predicate: p,
                confidence: c,
                memory_uuid: None,
                metadata: None,
                idempotent: false,
            },
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn bfs_traverse_on_small_diamond_graph() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let b = upsert(&svc.database, &svc.entity_repo, "B", "t").await;
        let c = upsert(&svc.database, &svc.entity_repo, "C", "t").await;
        let d = upsert(&svc.database, &svc.entity_repo, "D", "t").await;
        insert_edge(&svc.database, &svc.relation_repo, a, b, "knows", 1.0).await;
        insert_edge(&svc.database, &svc.relation_repo, a, c, "knows", 1.0).await;
        insert_edge(&svc.database, &svc.relation_repo, b, d, "knows", 1.0).await;
        insert_edge(&svc.database, &svc.relation_repo, c, d, "knows", 1.0).await;

        let nodes = svc
            .traverse_graph(
                NS,
                a,
                TraverseOpts {
                    max_depth: 2,
                    max_nodes: 50,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let visited: std::collections::HashSet<Uuid> =
            nodes.iter().map(|n| n.entity.uuid).collect();
        assert!(visited.contains(&a), "A must be in visited set: {visited:?}");
        assert!(visited.contains(&b));
        assert!(visited.contains(&c));
        assert!(visited.contains(&d));
    }

    #[tokio::test]
    async fn traverse_with_predicate_whitelist_filters_out_edges() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let b = upsert(&svc.database, &svc.entity_repo, "B", "t").await;
        let c = upsert(&svc.database, &svc.entity_repo, "C", "t").await;
        let d = upsert(&svc.database, &svc.entity_repo, "D", "t").await;
        insert_edge(&svc.database, &svc.relation_repo, a, b, "knows", 1.0).await;
        insert_edge(&svc.database, &svc.relation_repo, a, c, "likes", 1.0).await;
        insert_edge(&svc.database, &svc.relation_repo, b, d, "knows", 1.0).await;
        insert_edge(&svc.database, &svc.relation_repo, c, d, "likes", 1.0).await;

        let nodes = svc
            .traverse_graph(
                NS,
                a,
                TraverseOpts {
                    max_depth: 2,
                    max_nodes: 50,
                    predicate_whitelist: vec!["knows".to_string()],
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let visited: std::collections::HashSet<Uuid> =
            nodes.iter().map(|n| n.entity.uuid).collect();
        assert!(visited.contains(&a));
        assert!(visited.contains(&b));
        assert!(visited.contains(&d));
        assert!(!visited.contains(&c));
    }

    #[tokio::test]
    async fn traverse_with_confidence_threshold_filters_low_confidence_edges() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let b = upsert(&svc.database, &svc.entity_repo, "B", "t").await;
        let c = upsert(&svc.database, &svc.entity_repo, "C", "t").await;
        insert_edge(&svc.database, &svc.relation_repo, a, b, "knows", 0.9).await;
        insert_edge(&svc.database, &svc.relation_repo, a, c, "knows", 0.5).await;

        let nodes = svc
            .traverse_graph(
                NS,
                a,
                TraverseOpts {
                    max_depth: 1,
                    max_nodes: 50,
                    min_confidence: 0.8,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let visited: std::collections::HashSet<Uuid> =
            nodes.iter().map(|n| n.entity.uuid).collect();
        assert!(visited.contains(&a));
        assert!(visited.contains(&b));
        assert!(!visited.contains(&c));
    }

    #[tokio::test]
    async fn traverse_with_empty_whitelist_returns_all_nodes() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let b = upsert(&svc.database, &svc.entity_repo, "B", "t").await;
        let c = upsert(&svc.database, &svc.entity_repo, "C", "t").await;
        insert_edge(&svc.database, &svc.relation_repo, a, b, "knows", 1.0).await;
        insert_edge(&svc.database, &svc.relation_repo, a, c, "likes", 1.0).await;

        let nodes = svc
            .traverse_graph(
                NS,
                a,
                TraverseOpts {
                    max_depth: 1,
                    max_nodes: 50,
                    predicate_whitelist: vec![],
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let visited: std::collections::HashSet<Uuid> =
            nodes.iter().map(|n| n.entity.uuid).collect();
        assert!(visited.contains(&a));
        assert!(visited.contains(&b));
        assert!(visited.contains(&c));
    }

    #[tokio::test]
    async fn merge_entities_remaps_relations_and_deletes_discards() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let b = upsert(&svc.database, &svc.entity_repo, "B", "t").await;
        let c = upsert(&svc.database, &svc.entity_repo, "C", "t").await;
        insert_edge(&svc.database, &svc.relation_repo, a, b, "knows", 1.0).await;
        insert_edge(&svc.database, &svc.relation_repo, b, c, "knows", 1.0).await;
        insert_edge(&svc.database, &svc.relation_repo, c, a, "knows", 1.0).await;

        let out = svc
            .merge_entities(
                NS,
                MergeEntitiesInput {
                    keep_uuid: a,
                    discard_uuids: vec![b],
                },
            )
            .await
            .unwrap();
        assert_eq!(out.kept_uuid, a);
        assert_eq!(out.merged, vec![b]);
        assert_eq!(out.remapped_relations, 1);

        assert!(svc.entity_repo.get_by_uuid(&svc.database, NS, a).await.is_ok());
        assert!(svc.entity_repo.get_by_uuid(&svc.database, NS, b).await.is_err());

        let outgoing =
            svc.relation_repo.list_outgoing(&svc.database, NS, a, None, 100).await.unwrap();
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].target_uuid, c);
    }

    #[tokio::test]
    async fn merge_entities_rejects_keep_in_discard() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let err = svc
            .merge_entities(
                NS,
                MergeEntitiesInput {
                    keep_uuid: a,
                    discard_uuids: vec![a],
                },
            )
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn merge_entities_rejects_empty_discard() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let err = svc
            .merge_entities(
                NS,
                MergeEntitiesInput {
                    keep_uuid: a,
                    discard_uuids: vec![],
                },
            )
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn merge_entities_rejects_nonexistent_entity() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let missing = Uuid::new_v4();
        let err = svc
            .merge_entities(
                NS,
                MergeEntitiesInput {
                    keep_uuid: a,
                    discard_uuids: vec![missing],
                },
            )
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn merge_entities_removes_self_loop_after_merge() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let b = upsert(&svc.database, &svc.entity_repo, "B", "t").await;
        insert_edge(&svc.database, &svc.relation_repo, a, b, "knows", 1.0).await;

        let out = svc
            .merge_entities(
                NS,
                MergeEntitiesInput {
                    keep_uuid: a,
                    discard_uuids: vec![b],
                },
            )
            .await
            .unwrap();
        assert_eq!(out.remapped_relations, 0);

        let outgoing =
            svc.relation_repo.list_outgoing(&svc.database, NS, a, None, 100).await.unwrap();
        assert!(outgoing.is_empty());
    }

    #[tokio::test]
    async fn merge_entities_handles_two_discard_entities() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let b = upsert(&svc.database, &svc.entity_repo, "B", "t").await;
        let c = upsert(&svc.database, &svc.entity_repo, "C", "t").await;
        let d = upsert(&svc.database, &svc.entity_repo, "D", "t").await;
        insert_edge(&svc.database, &svc.relation_repo, b, d, "knows", 1.0).await;
        insert_edge(&svc.database, &svc.relation_repo, c, d, "likes", 1.0).await;

        let out = svc
            .merge_entities(
                NS,
                MergeEntitiesInput {
                    keep_uuid: a,
                    discard_uuids: vec![b, c],
                },
            )
            .await
            .unwrap();
        assert_eq!(out.remapped_relations, 2);

        let outgoing =
            svc.relation_repo.list_outgoing(&svc.database, NS, a, None, 100).await.unwrap();
        assert_eq!(outgoing.len(), 2);
        for rel in &outgoing {
            assert_eq!(rel.target_uuid, d);
        }
    }
}
