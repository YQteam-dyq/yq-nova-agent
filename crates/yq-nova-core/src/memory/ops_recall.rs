use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{
    GraphTraversalOpts, HybridWeights, MemoryService, SearchMode,
    rank::{RankCandidate, RankWeights, rank},
};
use crate::{
    Uuid,
    error::{NovaError, NovaResult},
    storage::{
        Direction, MemoryFilter, MemoryStatus,
        entity::EntityRepository,
        fts5::Fts5Store,
        memory::{MemoryRecord, MemoryRepository},
        relation::RelationRepository,
        vector::VectorStore,
    },
};

#[derive(Debug, Clone)]
pub struct RecallInput<'a> {
    pub query: &'a str,

    pub top_k: usize,

    pub score_threshold: f32,

    pub similarity_threshold: f32,

    pub mode: SearchMode,

    pub graph: GraphTraversalOpts,

    pub hybrid_weights: Option<HybridWeights>,

    pub rrf_k: Option<u32>,

    pub rank_weights: Option<RankWeights>,

    pub filter: MemoryFilter,

    pub group_chunks: bool,

    pub entity_focus: Vec<String>,

    pub rebalance_importance: bool,
}

impl<'a> Default for RecallInput<'a> {
    fn default() -> Self {
        Self {
            query: "",
            top_k: 20,
            score_threshold: 0.0,
            similarity_threshold: -1.0,
            mode: SearchMode::Semantic,
            graph: GraphTraversalOpts::default(),
            hybrid_weights: None,
            rrf_k: None,
            rank_weights: None,
            filter: MemoryFilter::default(),
            group_chunks: false,
            entity_focus: Vec::new(),
            rebalance_importance: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallHit {
    pub memory: MemoryRecord,
    pub final_score: f32,
    pub raw_similarity: Option<f32>,
    pub from_graph: bool,
    pub components: super::rank::ScoreComponents,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecallOutput {
    pub hits: Vec<RecallHit>,

    pub total_candidates: usize,

    pub query: String,
}

pub async fn recall(svc: &MemoryService, input: RecallInput<'_>) -> NovaResult<RecallOutput> {
    let query = input.query.trim();
    if query.is_empty() {
        return Err(NovaError::validation("recall: query must not be empty"));
    }
    let top_k = input.top_k.min(200);
    if top_k == 0 {
        return Err(NovaError::validation("recall: top_k must be >= 1"));
    }
    if !input.similarity_threshold.is_finite()
        || input.similarity_threshold < -1.0
        || input.similarity_threshold > 1.0
    {
        return Err(NovaError::validation(format!(
            "recall: similarity_threshold must be in [-1.0, 1.0], got {}",
            input.similarity_threshold
        )));
    }
    if !input.score_threshold.is_finite() {
        return Err(NovaError::validation("recall: score_threshold must be finite"));
    }

    let fetch_k = (top_k * 4).min(200).max(top_k.max(10));
    let hybrid_weights = input.hybrid_weights.unwrap_or_default();
    let namespace_id =
        input.filter.namespace_id.unwrap_or(crate::storage::namespace::DEFAULT_NAMESPACE_ID);
    let statuses: Vec<MemoryStatus> =
        input.filter.status_in.clone().unwrap_or_else(|| vec![MemoryStatus::Active]);
    let mut semantic_hits: Vec<(Uuid, f32)> = Vec::new();
    if matches!(input.mode, SearchMode::Semantic | SearchMode::Hybrid) {
        let q_vec = svc.embedding.embed_one(query).await?;
        let expected_dims = svc.embedding.meta().dims;
        if q_vec.len() != expected_dims {
            return Err(NovaError::embedding_msg(format!(
                "query embedding returned dims={} expected {}",
                q_vec.len(),
                expected_dims
            )));
        }
        let vhits = svc
            .vector_store
            .knn_search(namespace_id, &q_vec, fetch_k, input.similarity_threshold)
            .await?;
        semantic_hits = vhits.into_iter().map(|h| (h.memory_uuid, h.similarity)).collect();
    }

    let mut keyword_hits: Vec<(Uuid, f32)> = Vec::new();
    if matches!(input.mode, SearchMode::Keyword | SearchMode::Hybrid) {
        let khits = svc
            .fts5_store
            .keyword_search(&svc.database, namespace_id, query, fetch_k, &statuses)
            .await?;

        keyword_hits = khits.into_iter().map(|h| (h.uuid, h.score * 2.0 - 1.0)).collect();
    }

    let mut graph_memories: Vec<Uuid> = Vec::new();
    if input.graph.enabled {
        let mut seed_uuids: Vec<Uuid> = semantic_hits
            .iter()
            .map(|(u, _)| *u)
            .chain(keyword_hits.iter().map(|(u, _)| *u))
            .collect();
        seed_uuids.sort_unstable();
        seed_uuids.dedup();
        if !seed_uuids.is_empty() {
            let (expanded_mem, _seed_ents) =
                expand_graph_memories(svc, namespace_id, &seed_uuids, &input.graph, fetch_k)
                    .await?;
            graph_memories = expanded_mem;
        }
    }

    if !input.entity_focus.is_empty() {
        let mut focus_entity_uuids: BTreeSet<Uuid> = BTreeSet::new();
        for name in &input.entity_focus {
            match svc.entity_repo.find_by_name(&svc.database, namespace_id, name).await {
                Ok(entities) => {
                    for ent in entities {
                        focus_entity_uuids.insert(ent.uuid);
                    }
                },
                Err(_) => continue,
            }
        }
        if !focus_entity_uuids.is_empty() {
            let mut visited_entities: BTreeSet<Uuid> = focus_entity_uuids.clone();
            for ent_uuid in &focus_entity_uuids {
                if let Ok(nodes) = svc
                    .relation_repo
                    .bfs_traverse(
                        &svc.database,
                        namespace_id,
                        *ent_uuid,
                        Direction::Both,
                        2,
                        500,
                        &[],
                        0.0,
                    )
                    .await
                {
                    for n in nodes {
                        visited_entities.insert(n.entity.uuid);
                    }
                }
            }
            let entity_strs: Vec<String> = visited_entities.iter().map(|u| u.to_string()).collect();
            let ph: Vec<&str> = entity_strs.iter().map(|_| "?").collect();
            let phs = ph.join(",");
            let sql = format!(
                r#"
                SELECT DISTINCT memory_uuid FROM relations
                WHERE memory_uuid IS NOT NULL
                  AND namespace_id = ?1
                  AND (source_uuid IN ({phs}) OR target_uuid IN ({phs}))
                "#
            );
            let mut q = sqlx::query_scalar::<_, String>(&sql).bind(namespace_id);
            for s in &entity_strs {
                q = q.bind(s);
            }
            for s in &entity_strs {
                q = q.bind(s);
            }
            if let Ok(mem_strs) = q.fetch_all(&svc.database.pool).await.map_err(NovaError::storage)
            {
                for s in mem_strs {
                    if let Ok(u) = Uuid::parse_str(&s) {
                        graph_memories.push(u);
                    }
                }
            }
        }
    }

    struct CollectedCandidates {
        ordered: Vec<Uuid>,
        sim_by_uuid: BTreeMap<Uuid, f32>,
        from_graph: BTreeSet<Uuid>,
        keyword_scores: BTreeMap<Uuid, f32>,
    }
    let collected: CollectedCandidates = match input.mode {
        SearchMode::Semantic => {
            let sim_by_uuid: BTreeMap<Uuid, f32> = semantic_hits.iter().copied().collect();
            let mut ordered: Vec<Uuid> = semantic_hits.iter().map(|(u, _)| *u).collect();

            let mut from_graph: BTreeSet<Uuid> = BTreeSet::new();
            for u in &graph_memories {
                from_graph.insert(*u);
                if !sim_by_uuid.contains_key(u) {
                    ordered.push(*u);
                }
            }
            CollectedCandidates {
                ordered,
                sim_by_uuid,
                from_graph,
                keyword_scores: BTreeMap::new(),
            }
        },
        SearchMode::Keyword => {
            let sim_by_uuid: BTreeMap<Uuid, f32> = keyword_hits.iter().copied().collect();
            let ordered: Vec<Uuid> = keyword_hits.iter().map(|(u, _)| *u).collect();
            let mut from_graph: BTreeSet<Uuid> = BTreeSet::new();
            for u in &graph_memories {
                from_graph.insert(*u);
            }
            CollectedCandidates {
                ordered,
                sim_by_uuid,
                from_graph,
                keyword_scores: keyword_hits.into_iter().collect(),
            }
        },
        SearchMode::Hybrid => {
            use crate::memory::rank::{RrfSource, reciprocal_rank_fusion};

            let semantic_ranked: Vec<Uuid> = semantic_hits.iter().map(|(u, _)| *u).collect();
            let keyword_ranked: Vec<Uuid> = keyword_hits.iter().map(|(u, _)| *u).collect();
            let graph_ranked: Vec<Uuid> = graph_memories.clone();

            let mut sources: Vec<RrfSource> = Vec::with_capacity(3);
            if hybrid_weights.semantic > 0.0 && !semantic_ranked.is_empty() {
                sources.push(RrfSource {
                    items: semantic_ranked,
                    weight: hybrid_weights.semantic,
                    label: "semantic",
                });
            }
            if hybrid_weights.keyword > 0.0 && !keyword_ranked.is_empty() {
                sources.push(RrfSource {
                    items: keyword_ranked,
                    weight: hybrid_weights.keyword,
                    label: "keyword",
                });
            }
            if hybrid_weights.graph > 0.0 && !graph_ranked.is_empty() {
                sources.push(RrfSource {
                    items: graph_ranked,
                    weight: hybrid_weights.graph,
                    label: "graph",
                });
            }

            let mut sim_by_uuid: BTreeMap<Uuid, f32> = semantic_hits.into_iter().collect();
            let keyword_scores: BTreeMap<Uuid, f32> = keyword_hits.iter().copied().collect();

            for (u, s) in &keyword_hits {
                sim_by_uuid.entry(*u).or_insert_with(|| *s);
            }
            let mut from_graph: BTreeSet<Uuid> = BTreeSet::new();
            for u in &graph_memories {
                from_graph.insert(*u);
            }

            let rrf = reciprocal_rank_fusion(sources, input.rrf_k);

            let rrf_limit = (top_k * 3).min(400);
            let ordered: Vec<Uuid> = rrf.into_iter().take(rrf_limit).map(|h| h.uuid).collect();
            CollectedCandidates {
                ordered,
                sim_by_uuid,
                from_graph,
                keyword_scores,
            }
        },
    };

    let _ = collected.keyword_scores;

    let mut records: Vec<MemoryRecord> = Vec::with_capacity(collected.ordered.len());
    for uuid in &collected.ordered {
        match svc.memory_repo.get_by_uuid(&svc.database, namespace_id, *uuid).await {
            Ok(r) => records.push(r),
            Err(e) if matches!(e.code(), crate::error::ErrorCode::NotFound) => continue,
            Err(e) => return Err(e),
        }
    }
    let filter = input.filter.clone();
    let filtered: Vec<MemoryRecord> =
        records.into_iter().filter(|r| passes_filter(r, &filter)).collect();
    let total_candidates = filtered.len();

    let weights = input.rank_weights.unwrap_or_default();
    let threshold = input.score_threshold;
    let candidates: Vec<RankCandidate<'_>> = filtered
        .iter()
        .map(|r| RankCandidate {
            memory: r,
            raw_similarity: collected.sim_by_uuid.get(&r.uuid).copied(),
            from_graph: collected.from_graph.contains(&r.uuid),
        })
        .collect();
    let ranked = rank(candidates, weights, threshold);

    let collapsed: Vec<super::rank::RankedHit<'_>> = if input.group_chunks {
        let mut best_per_group: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut deduped: Vec<super::rank::RankedHit<'_>> = Vec::with_capacity(ranked.len());
        for rh in &ranked {
            let group = rh
                .memory
                .metadata
                .get("chunk_group")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            if let Some(ref g) = group {
                let entry = best_per_group.entry(g.clone()).or_insert(usize::MAX);
                if *entry == usize::MAX {
                    *entry = deduped.len();
                    deduped.push(rh.clone());
                } else {
                    let existing = &deduped[*entry];
                    if rh.final_score > existing.final_score {
                        deduped[*entry] = rh.clone();
                    }
                }
            } else {
                deduped.push(rh.clone());
            }
        }
        deduped
    } else {
        ranked
    };

    let mut hits: Vec<RecallHit> = Vec::with_capacity(collapsed.len().min(top_k));
    for rh in collapsed.into_iter().take(top_k) {
let _ = svc.memory_repo.mark_accessed(&svc.database, namespace_id, rh.memory.uuid).await;
        if input.rebalance_importance {
            let _ = super::quality::importance::maintain_one(svc, rh.memory.uuid).await;
        }
        hits.push(RecallHit {
            memory: rh.memory.clone(),
            final_score: rh.final_score,
            raw_similarity: collected.sim_by_uuid.get(&rh.memory.uuid).copied(),
            from_graph: collected.from_graph.contains(&rh.memory.uuid),
            components: rh.components,
        });
    }

    Ok(RecallOutput {
        hits,
        total_candidates,
        query: query.to_string(),
    })
}

async fn expand_graph_memories(
    svc: &MemoryService,
    namespace_id: i64,
    seed_memory_uuids: &[Uuid],
    opts: &GraphTraversalOpts,
    limit: usize,
) -> NovaResult<(Vec<Uuid>, BTreeSet<Uuid>)> {
    if seed_memory_uuids.is_empty() {
        return Ok((Vec::new(), BTreeSet::new()));
    }
    let memory_strs: Vec<String> = seed_memory_uuids.iter().map(|u| u.to_string()).collect();
    let first_ph: Vec<String> = (2..=memory_strs.len() + 1).map(|i| format!("?{i}")).collect();
    let second_ph: Vec<String> =
        (memory_strs.len() + 2..=2 * memory_strs.len() + 1).map(|i| format!("?{i}")).collect();
    let first = first_ph.join(",");
    let second = second_ph.join(",");
    let sql = format!(
        r#"
        SELECT DISTINCT source_uuid FROM relations
        WHERE memory_uuid IN ({first}) AND source_uuid IS NOT NULL AND namespace_id = ?1
        UNION
        SELECT DISTINCT target_uuid FROM relations
        WHERE memory_uuid IN ({second}) AND target_uuid IS NOT NULL AND namespace_id = ?1
        "#
    );
    let mut q = sqlx::query_scalar::<_, String>(&sql).bind(namespace_id);
    for s in &memory_strs {
        q = q.bind(s);
    }
    for s in &memory_strs {
        q = q.bind(s);
    }
    let entity_strs: Vec<String> =
        q.fetch_all(&svc.database.pool).await.map_err(NovaError::storage)?;
    let mut start_entities: BTreeSet<Uuid> = BTreeSet::new();
    for s in entity_strs {
        if let Ok(u) = Uuid::parse_str(&s) {
            start_entities.insert(u);
        }
    }

    let max_depth = opts.max_depth.min(6);
    let mut visited_entities: BTreeSet<Uuid> = start_entities.clone();
    for ent in &start_entities {
        let Ok(nodes) = svc
            .relation_repo
            .bfs_traverse(
                &svc.database,
                namespace_id,
                *ent,
                Direction::Both,
                max_depth,
                500,
                &[],
                0.0,
            )
            .await
        else {
            continue;
        };
        for n in nodes {
            visited_entities.insert(n.entity.uuid);
        }
    }

    let entity_strs2: Vec<String> = visited_entities.iter().map(|u| u.to_string()).collect();
    if entity_strs2.is_empty() {
        return Ok((Vec::new(), start_entities));
    }
    let ph2: Vec<&str> = entity_strs2.iter().map(|_| "?").collect();
    let ph2s = ph2.join(",");
    let sql2 = format!(
        r#"
        SELECT DISTINCT memory_uuid FROM relations
        WHERE memory_uuid IS NOT NULL
          AND (source_uuid IN ({ph2s}) OR target_uuid IN ({ph2s}))
        "#
    );
    let mut q2 = sqlx::query_scalar::<_, String>(&sql2);
    for s in &entity_strs2 {
        q2 = q2.bind(s);
    }
    for s in &entity_strs2 {
        q2 = q2.bind(s);
    }
    let mem_strs: Vec<String> =
        q2.fetch_all(&svc.database.pool).await.map_err(NovaError::storage)?;
    let mut out: Vec<Uuid> = Vec::new();
    for s in mem_strs {
        let Ok(u) = Uuid::parse_str(&s) else { continue };
        out.push(u);
    }
    out.sort_unstable();
    out.dedup();
    out.truncate(limit);
    Ok((out, start_entities))
}

pub(crate) fn passes_filter(r: &MemoryRecord, f: &MemoryFilter) -> bool {
    if let Some(ref statuses) = f.status_in {
        if !statuses.contains(&r.status) {
            return false;
        }
    }
    if let Some(ref srcs) = f.source_in {
        if !srcs.contains(&r.source) {
            return false;
        }
    }
    if let Some(after) = f.created_after {
        if r.created_at <= after {
            return false;
        }
    }
    if let Some(before) = f.created_before {
        if r.created_at >= before {
            return false;
        }
    }
    if let Some(imp_min) = f.importance_min {
        if r.importance < imp_min {
            return false;
        }
    }
    if let Some(imp_max) = f.importance_max {
        if r.importance > imp_max {
            return false;
        }
    }
    if let Some(acc_lt) = f.access_count_lt {
        if r.access_count >= acc_lt {
            return false;
        }
    }
    if let Some(la_before) = f.last_accessed_before {
        let effective = r.last_accessed.unwrap_or(r.created_at);
        if effective >= la_before {
            return false;
        }
    }
    if let Some(la_after) = f.last_accessed_after {
        let effective = r.last_accessed.unwrap_or(r.created_at);
        if effective <= la_after {
            return false;
        }
    }
    if let Some(ref tags_all) = f.tags_all {
        if tags_all.is_empty() {
            return false;
        }
        for t in tags_all {
            if !r.tags.contains(t) {
                return false;
            }
        }
    }
    if let Some(ref tags_any) = f.tags_any {
        if !tags_any.is_empty() && !r.tags.iter().any(|t| tags_any.contains(t)) {
            return false;
        }
    }
    if let Some(ref metadata_match) = f.metadata_match {
        for (key, val) in metadata_match {
            match r.metadata.get(key) {
                Some(actual) => {
                    if actual != val {
                        return false;
                    }
                },
                None => return false,
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Uuid,
        config::StorageConfig,
        memory::{
            chunk::{ChunkOptions, SplitBy},
            ops_remember::{RememberInput, service_for_tests},
        },
        storage::{Database, MemorySource},
    };

    async fn temp_svc() -> crate::memory::MemoryService {
        let dir = std::env::temp_dir().join(format!("yq-nova-m3-recall-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = StorageConfig {
            db_path: dir.join("test.db"),
            pool_max_connections: 2,
            pool_min_connections: 0,
            ..StorageConfig::default()
        };
        let db = Database::open(cfg).await.unwrap();
        service_for_tests(db, 8, None)
    }

    #[tokio::test]
    async fn empty_query_and_bad_top_k_are_rejected() {
        let svc = temp_svc().await;
        let err = svc
            .recall(RecallInput {
                query: "  ",
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);

        let err = svc
            .recall(RecallInput {
                query: "q",
                top_k: 0,
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn recall_returns_both_memories_and_marks_accessed() {
        let svc = temp_svc().await;

        let a = svc
            .remember(RememberInput {
                content: "aaaaaaaaaaaaaa one",
                tags: &["a".into()],
                importance: 1.0,
                ..Default::default()
            })
            .await
            .unwrap();
        let b = svc
            .remember(RememberInput {
                content: "bbbbbbbbbbbbbb two",
                tags: &["b".into()],
                importance: 0.0,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_ne!(a.uuid, b.uuid, "different content → different uuids");

        let out = svc
            .recall(RecallInput {
                query: "anything really",
                top_k: 5,
                score_threshold: 0.0,
                similarity_threshold: -1.0,
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(
            out.hits.len(),
            2,
            "both memories should be returned (mock dims=8 with very low thresholds)"
        );

        let returned: std::collections::HashSet<Uuid> =
            out.hits.iter().map(|h| h.memory.uuid).collect();
        assert!(returned.contains(&a.uuid));
        assert!(returned.contains(&b.uuid));

        let a_after = svc.get_memory(a.uuid).await.unwrap();
        let b_after = svc.get_memory(b.uuid).await.unwrap();
        assert_eq!(a_after.access_count, 1, "memory A access_count should be 1 (within top_k)");
        assert_eq!(b_after.access_count, 1, "memory B access_count should be 1 (within top_k)");
    }

    #[tokio::test]
    async fn tag_filter_restricts_results() {
        let svc = temp_svc().await;
        let _x = svc
            .remember(RememberInput {
                content: "alpha alpha alpha",
                tags: &["x".into()],
                ..Default::default()
            })
            .await
            .unwrap();
        let _y = svc
            .remember(RememberInput {
                content: "alpha alpha alpha",
                tags: &["y".into()],
                importance: 0.99,
                ..Default::default()
            })
            .await
            .unwrap();

        let zz = svc
            .remember(RememberInput {
                content: "beta beta beta beta",
                tags: &["z".into()],
                ..Default::default()
            })
            .await
            .unwrap();

        let f = MemoryFilter {
            tags_any: Some(vec!["z".to_string()]),
            ..Default::default()
        };
        let out = svc
            .recall(RecallInput {
                query: "beta",
                top_k: 5,
                filter: f,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(out.hits.iter().all(|h| h.memory.uuid == zz.uuid));
    }

    #[tokio::test]
    async fn importance_min_filter_works() {
        let svc = temp_svc().await;
        let low = svc
            .remember(RememberInput {
                content: "low importance filler text",
                importance: 0.05,
                ..Default::default()
            })
            .await
            .unwrap();
        let hi = svc
            .remember(RememberInput {
                content: "hi importance critical note",
                importance: 0.95,
                ..Default::default()
            })
            .await
            .unwrap();

        let f = MemoryFilter {
            importance_min: Some(0.9),
            ..Default::default()
        };
        let out = svc
            .recall(RecallInput {
                query: "critical note",
                top_k: 10,
                filter: f,
                ..Default::default()
            })
            .await
            .unwrap();
        let ids: std::collections::HashSet<_> = out.hits.iter().map(|h| h.memory.uuid).collect();
        assert!(ids.contains(&hi.uuid), "high imp should pass filter");
        assert!(!ids.contains(&low.uuid), "low imp must be filtered out");

        let low_mem = svc.get_memory(low.uuid).await.unwrap();
        assert_eq!(low_mem.access_count, 0);
        let _ = (low,);
    }

    #[tokio::test]
    async fn source_filter_agent_only() {
        let svc = temp_svc().await;
        let user = svc
            .remember(RememberInput {
                content: "preference: vegan dark chocolate",
                source: MemorySource::User,
                importance: 0.9,
                ..Default::default()
            })
            .await
            .unwrap();
        let agent = svc
            .remember(RememberInput {
                content: "thought: suggested oat milk latte",
                source: MemorySource::Agent,
                importance: 0.9,
                ..Default::default()
            })
            .await
            .unwrap();

        let f = MemoryFilter {
            source_in: Some(vec![MemorySource::User]),
            ..Default::default()
        };
        let out = svc
            .recall(RecallInput {
                query: "chocolate preference",
                top_k: 10,
                filter: f,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(out.hits.iter().all(|h| h.memory.source == MemorySource::User));
        let ids: std::collections::HashSet<_> = out.hits.iter().map(|h| h.memory.uuid).collect();
        assert!(ids.contains(&user.uuid));
        assert!(!ids.contains(&agent.uuid));
    }

    #[tokio::test]
    async fn group_chunks_collapses_duplicate_groups() {
        let svc = temp_svc().await;
        let mut text = String::new();
        for i in 0..6 {
            if i > 0 {
                text.push_str("\n\n");
            }
            text.push_str(&format!("This is paragraph {} for the chunk group recall test.", i));
        }
        let opts = ChunkOptions {
            enabled: true,
            split_by: SplitBy::Paragraph,
            max_chars: 200,
            overlap_chars: 30,
        };
        let chunked = svc
            .remember(RememberInput {
                content: &text,
                chunk_options: Some(opts),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(chunked.chunks.len() > 1, "need at least 2 chunks for this test");

        let standalone = svc
            .remember(RememberInput {
                content: "completely unrelated standalone memory for comparison",
                importance: 0.9,
                ..Default::default()
            })
            .await
            .unwrap();

        let out_no_group = svc
            .recall(RecallInput {
                query: "paragraph",
                top_k: 20,
                score_threshold: 0.0,
                similarity_threshold: -1.0,
                group_chunks: false,
                ..Default::default()
            })
            .await
            .unwrap();
        let chunk_uuids_no_group: std::collections::HashSet<Uuid> =
            chunked.chunks.iter().map(|c| c.uuid).collect();
        let returned_uuids_no_group: std::collections::HashSet<Uuid> =
            out_no_group.hits.iter().map(|h| h.memory.uuid).collect();
        let overlap_no_group: Vec<&Uuid> =
            chunk_uuids_no_group.iter().filter(|u| returned_uuids_no_group.contains(u)).collect();
        assert!(
            overlap_no_group.len() > 1,
            "without group_chunks, multiple chunks should be returned, got {}",
            overlap_no_group.len()
        );

        let out_grouped = svc
            .recall(RecallInput {
                query: "paragraph",
                top_k: 20,
                score_threshold: 0.0,
                similarity_threshold: -1.0,
                group_chunks: true,
                ..Default::default()
            })
            .await
            .unwrap();
        let chunk_uuids_grouped: std::collections::HashSet<Uuid> =
            chunked.chunks.iter().map(|c| c.uuid).collect();
        let returned_uuids_grouped: std::collections::HashSet<Uuid> =
            out_grouped.hits.iter().map(|h| h.memory.uuid).collect();
        let overlap_grouped: Vec<&Uuid> =
            chunk_uuids_grouped.iter().filter(|u| returned_uuids_grouped.contains(u)).collect();
        assert!(
            overlap_grouped.len() <= 1,
            "with group_chunks, at most one chunk per group should be returned, got {}",
            overlap_grouped.len()
        );

        let standalone_in_grouped =
            out_grouped.hits.iter().any(|h| h.memory.uuid == standalone.uuid);
        let standalone_in_no_group =
            out_no_group.hits.iter().any(|h| h.memory.uuid == standalone.uuid);
        assert_eq!(
            standalone_in_grouped, standalone_in_no_group,
            "standalone memory should be equally visible in both modes"
        );
    }

    #[test]
    fn passes_filter_metadata_match_subset() {
        let rec = MemoryRecord {
            metadata: serde_json::json!({"key1": "val1", "key2": "val2", "key3": 42}),
            ..dummy_memory_record()
        };
        let f = MemoryFilter {
            metadata_match: Some(vec![("key1".into(), serde_json::json!("val1"))]),
            ..Default::default()
        };
        assert!(passes_filter(&rec, &f));
    }

    #[test]
    fn passes_filter_metadata_match_multiple() {
        let rec = MemoryRecord {
            metadata: serde_json::json!({"key1": "val1", "key2": "val2", "key3": 42}),
            ..dummy_memory_record()
        };
        let f = MemoryFilter {
            metadata_match: Some(vec![
                ("key1".into(), serde_json::json!("val1")),
                ("key3".into(), serde_json::json!(42)),
            ]),
            ..Default::default()
        };
        assert!(passes_filter(&rec, &f));
    }

    #[test]
    fn passes_filter_metadata_match_not_subset_wrong_value() {
        let rec = MemoryRecord {
            metadata: serde_json::json!({"key1": "val1", "key2": "val2"}),
            ..dummy_memory_record()
        };
        let f = MemoryFilter {
            metadata_match: Some(vec![("key1".into(), serde_json::json!("wrong"))]),
            ..Default::default()
        };
        assert!(!passes_filter(&rec, &f));
    }

    #[test]
    fn passes_filter_metadata_match_not_subset_missing_key() {
        let rec = MemoryRecord {
            metadata: serde_json::json!({"key1": "val1"}),
            ..dummy_memory_record()
        };
        let f = MemoryFilter {
            metadata_match: Some(vec![("missing_key".into(), serde_json::json!("val"))]),
            ..Default::default()
        };
        assert!(!passes_filter(&rec, &f));
    }

    #[test]
    fn passes_filter_metadata_match_none_does_not_filter() {
        let rec = MemoryRecord {
            metadata: serde_json::json!({"key1": "val1"}),
            ..dummy_memory_record()
        };
        let f = MemoryFilter {
            metadata_match: None,
            ..Default::default()
        };
        assert!(passes_filter(&rec, &f));
    }

    fn dummy_memory_record() -> MemoryRecord {
        MemoryRecord {
            id: 0,
            uuid: Uuid::nil(),
            namespace_id: crate::storage::namespace::DEFAULT_NAMESPACE_ID,
            content: String::new(),
            content_hash: String::new(),
            metadata: serde_json::json!({}),
            source: MemorySource::Agent,
            importance: 0.5,
            access_count: 0,
            last_accessed: None,
            created_at: chrono::Utc::now(),
            expires_at: None,
            status: MemoryStatus::Active,
            tags: Vec::new(),
        }
    }

    #[tokio::test]
    async fn entity_focus_empty_does_not_change_behavior() {
        let svc = temp_svc().await;
        let m = svc
            .remember(RememberInput {
                content: "entity focus empty test content",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let out = svc
            .recall(RecallInput {
                query: "entity focus empty test content",
                top_k: 10,
                score_threshold: 0.0,
                similarity_threshold: -1.0,
                entity_focus: vec![],
                ..Default::default()
            })
            .await
            .unwrap();
        let ids: std::collections::HashSet<_> = out.hits.iter().map(|h| h.memory.uuid).collect();
        assert!(ids.contains(&m.uuid), "memory should be recalled without entity_focus");
    }

    #[tokio::test]
    async fn entity_focus_unknown_name_is_ignored() {
        let svc = temp_svc().await;
        let m = svc
            .remember(RememberInput {
                content: "unknown entity focus test",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let out = svc
            .recall(RecallInput {
                query: "unknown entity focus test",
                top_k: 10,
                score_threshold: 0.0,
                similarity_threshold: -1.0,
                entity_focus: vec!["NonExistentEntity".to_string()],
                ..Default::default()
            })
            .await
            .unwrap();
        let ids: std::collections::HashSet<_> = out.hits.iter().map(|h| h.memory.uuid).collect();
        assert!(
            ids.contains(&m.uuid),
            "memory should still be recalled when entity_focus name is unknown"
        );
    }

    #[tokio::test]
    async fn entity_focus_anchors_memories_via_graph() {
        let svc = temp_svc().await;
        use crate::storage::{entity::UpsertEntityInput, relation::InsertRelationInput};

        let alice = svc
            .entity_repo
            .upsert(
                &svc.database,
                UpsertEntityInput {
                    namespace_id: crate::storage::namespace::DEFAULT_NAMESPACE_ID,
                    name: "Alice",
                    r#type: "person",
                    description: None,
                    metadata: None,
                },
            )
            .await
            .unwrap()
            .uuid();
        let bob = svc
            .entity_repo
            .upsert(
                &svc.database,
                UpsertEntityInput {
                    namespace_id: crate::storage::namespace::DEFAULT_NAMESPACE_ID,
                    name: "Bob",
                    r#type: "person",
                    description: None,
                    metadata: None,
                },
            )
            .await
            .unwrap()
            .uuid();

        let m1 = svc
            .remember(RememberInput {
                content: "alice_work related content",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();
        let m2 = svc
            .remember(RememberInput {
                content: "bob_hobby completely different topic",
                importance: 0.5,
                ..Default::default()
            })
            .await
            .unwrap();

        for (src, tgt, mem, pred) in
            [(alice, bob, m1.uuid, "works_with"), (alice, bob, m2.uuid, "knows")]
        {
            svc.relation_repo
                .insert(
                    &svc.database,
                    InsertRelationInput {
                        namespace_id: crate::storage::namespace::DEFAULT_NAMESPACE_ID,
                        source_uuid: src,
                        target_uuid: tgt,
                        predicate: pred,
                        confidence: 1.0,
                        memory_uuid: Some(mem),
                        metadata: None,
                        idempotent: true,
                    },
                )
                .await
                .unwrap();
        }

        let out = svc
            .recall(RecallInput {
                query: "alice_work",
                top_k: 10,
                score_threshold: 0.0,
                similarity_threshold: -1.0,
                entity_focus: vec!["Alice".to_string()],
                ..Default::default()
            })
            .await
            .unwrap();
        let ids: std::collections::HashSet<_> = out.hits.iter().map(|h| h.memory.uuid).collect();
        assert!(ids.contains(&m1.uuid), "m1 should be recalled (semantic match)");
        assert!(ids.contains(&m2.uuid), "m2 should be recalled via entity_focus graph expansion");
        assert!(
            out.hits.iter().any(|h| h.from_graph && h.memory.uuid == m2.uuid),
            "m2 should be marked as from_graph"
        );
    }
}
