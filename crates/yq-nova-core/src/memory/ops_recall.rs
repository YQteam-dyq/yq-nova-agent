//! `recall` — given a natural-language query, retrieve semantically +
//! graph-relevant memories, rank them, return a sorted list of hits.
//!
//! Pipeline (MVP semantic-only mode, fully local):
//!
//! 1. Validate inputs (query non-empty, top_k sane, threshold bounded).
//! 2. Embed the query using the service's default provider.
//! 3. KNN against the vector store for initial candidates (`vector_candidates`).
//! 4. (Optional, if graph traversal is enabled) BFS outward from entities
//!    attached to the top K hits; pull in memories attached to those
//!    entities as `graph_candidates`.
//! 5. Dedupe the union by `memory_uuid`.
//! 6. Pass all candidates through `rank::rank` with weighted components;
//!    apply the final score threshold; sort desc.
//! 7. Mark every returned hit as `accessed` (access_count + 1, last_accessed).
//! 8. Return the top `top_k`.

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
        fts5::Fts5Store,
        memory::{MemoryRecord, MemoryRepository},
        relation::RelationRepository,
        vector::VectorStore,
    },
};

/// Input to [`MemoryService::recall`].
#[derive(Debug, Clone)]
pub struct RecallInput<'a> {
    /// Natural-language query. For `semantic` mode this gets embedded and used
    /// for KNN; for `keyword` (v0.2) this becomes a SQLite LIKE query.
    pub query: &'a str,
    /// Max number of results to return. Hard-capped at 200 to avoid blowing
    /// up the caller's RAM on accidentally-wide queries.
    pub top_k: usize,
    /// Minimum *final* weighted score. Applied after the ranker, so it
    /// applies to semantic-similarity + importance + access combined.
    pub score_threshold: f32,
    /// Minimum *raw cosine similarity* for a direct vector hit to count.
    /// Kept independent so callers can enforce a "the embedding must at
    /// least match this well" floor without also penalising graph-adjacent
    /// memories (which have no raw cosine).
    pub similarity_threshold: f32,
    /// Retrieval mode. MVP only implements `Semantic`; the others fall back
    /// to semantic with a debug-level log.
    pub mode: SearchMode,
    /// Graph-traversal options. `enabled = false` (default) means no BFS
    /// expansion.
    pub graph: GraphTraversalOpts,
    /// Per-source weights for hybrid-mode RRF fusion (keyword/semantic/graph).
    /// `None` uses `HybridWeights::default()`.
    pub hybrid_weights: Option<HybridWeights>,
    /// Optional smoothing constant `k` for RRF. Default = 25 (see
    /// `rank::reciprocal_rank_fusion`).
    pub rrf_k: Option<u32>,
    /// Optional per-call rank weight overrides; `None` uses `RankWeights::default()`.
    pub rank_weights: Option<RankWeights>,
    /// Optional filter applied on all retrieved MemoryRecords before ranking.
    /// Use to restrict recall to a subset of sources / tags / time windows.
    pub filter: MemoryFilter,
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
        }
    }
}

/// A single recall hit. The caller gets a full `MemoryRecord` + the scoring
/// breakdown so they can build explainability UI or debug poor ranking.
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
    /// Total candidate count considered before top_k cutoff (useful for
    /// pagination / "hits shown X of Y" UI).
    pub total_candidates: usize,
    /// Query string echoed back (useful for async pipelines that lose context).
    pub query: String,
}

pub async fn recall(svc: &MemoryService, input: RecallInput<'_>) -> NovaResult<RecallOutput> {
    // --- 1. Validation -------------------------------------------------------
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

    // Candidate fetch size for each source. Fetch more than we need so RRF
    // + the linear ranker have enough material to work with.
    let fetch_k = (top_k * 4).min(200).max(top_k.max(10));
    let hybrid_weights = input.hybrid_weights.unwrap_or_default();

    // Determine which statuses are acceptable (for both FTS5 and vector
    // post-filtering). If filter.status_in is set, use it; otherwise default
    // to [Active] so archived/expired rows never come back unless explicitly
    // requested.
    let statuses: Vec<MemoryStatus> =
        input.filter.status_in.clone().unwrap_or_else(|| vec![MemoryStatus::Active]);

    // --- 2. Semantic source (if the mode wants it) --------------------------
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
        let vhits =
            svc.vector_store.knn_search(&q_vec, fetch_k, input.similarity_threshold).await?;
        semantic_hits = vhits.into_iter().map(|h| (h.memory_uuid, h.similarity)).collect();
    }

    // --- 3. Keyword source (if the mode wants it) ---------------------------
    let mut keyword_hits: Vec<(Uuid, f32)> = Vec::new();
    if matches!(input.mode, SearchMode::Keyword | SearchMode::Hybrid) {
        let khits = svc.fts5_store.keyword_search(&svc.database, query, fetch_k, &statuses).await?;
        // `score` is already normalised to [0, 1] with 1.0 = best. For
        // downstream ranker compatibility we map to a pseudo raw_similarity
        // value in [-1, 1] via `keyword_score * 2 - 1`.
        keyword_hits = khits.into_iter().map(|h| (h.uuid, h.score * 2.0 - 1.0)).collect();
    }

    // --- 4. Graph expansion (if enabled) ------------------------------------
    // Graph candidates are seeded from the entities that are already attached
    // to semantic + keyword hits via relation.memory_uuid, then BFS expanded.
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
                expand_graph_memories(svc, &seed_uuids, &input.graph, fetch_k).await?;
            graph_memories = expanded_mem;
        }
    }

    // --- 5. Per-mode: collect ordered uuid list + build "best sim" map ------
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
            // Flag ALL graph-expanded memories, even those that already appeared
            // in semantic hits, so the caller can see "graph also agrees".
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

            // Sim maps: preserve best available per source.
            let mut sim_by_uuid: BTreeMap<Uuid, f32> = semantic_hits.into_iter().collect();
            let keyword_scores: BTreeMap<Uuid, f32> = keyword_hits.iter().copied().collect();
            // Keyword also contributes to "best sim" if semantic didn't have it.
            for (u, s) in &keyword_hits {
                sim_by_uuid.entry(*u).or_insert_with(|| *s);
            }
            let mut from_graph: BTreeSet<Uuid> = BTreeSet::new();
            for u in &graph_memories {
                from_graph.insert(*u);
            }

            let rrf = reciprocal_rank_fusion(sources, input.rrf_k);
            // Take top (top_k * 3) from RRF as candidates to rank linearly.
            let rrf_limit = (top_k * 3).min(400);
            let ordered: Vec<Uuid> = rrf.into_iter().take(rrf_limit).map(|h| h.uuid).collect();
            CollectedCandidates { ordered, sim_by_uuid, from_graph, keyword_scores }
        },
    };

    let _ = collected.keyword_scores; // field reserved for future explainability

    // --- 6. Load MemoryRecords + apply post-filter -------------------------
    let mut records: Vec<MemoryRecord> = Vec::with_capacity(collected.ordered.len());
    for uuid in &collected.ordered {
        match svc.memory_repo.get_by_uuid(&svc.database, *uuid).await {
            Ok(r) => records.push(r),
            Err(e) if matches!(e.code(), crate::error::ErrorCode::NotFound) => continue,
            Err(e) => return Err(e),
        }
    }
    let filter = input.filter.clone();
    let filtered: Vec<MemoryRecord> =
        records.into_iter().filter(|r| passes_filter(r, &filter)).collect();
    let total_candidates = filtered.len();

    // --- 7. Linear ranker ---------------------------------------------------
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

    // --- 8. Mark accessed for returned hits only ---------------------------
    let mut hits: Vec<RecallHit> = Vec::with_capacity(ranked.len().min(top_k));
    for rh in ranked.into_iter().take(top_k) {
        let _ = svc.memory_repo.mark_accessed(&svc.database, rh.memory.uuid).await;
        hits.push(RecallHit {
            memory: rh.memory.clone(),
            final_score: rh.final_score,
            raw_similarity: collected.sim_by_uuid.get(&rh.memory.uuid).copied(),
            from_graph: collected.from_graph.contains(&rh.memory.uuid),
            components: rh.components,
        });
    }

    Ok(RecallOutput { hits, total_candidates, query: query.to_string() })
}

// ---------------------------------------------------------------------------
// Graph expansion helper — seed memories → related memories via entities.
// ---------------------------------------------------------------------------
async fn expand_graph_memories(
    svc: &MemoryService,
    seed_memory_uuids: &[Uuid],
    opts: &GraphTraversalOpts,
    limit: usize,
) -> NovaResult<(Vec<Uuid>, BTreeSet<Uuid>)> {
    // Step 1: entities attached directly to seed memories via relations that
    // have `memory_uuid` set.
    if seed_memory_uuids.is_empty() {
        return Ok((Vec::new(), BTreeSet::new()));
    }
    let memory_strs: Vec<String> = seed_memory_uuids.iter().map(|u| u.to_string()).collect();
    let placeholders: Vec<&str> = memory_strs.iter().map(|_| "?").collect();
    let ph = placeholders.join(",");
    let sql = format!(
        r#"
        SELECT DISTINCT source_uuid FROM relations
        WHERE memory_uuid IN ({ph}) AND source_uuid IS NOT NULL
        UNION
        SELECT DISTINCT target_uuid FROM relations
        WHERE memory_uuid IN ({ph}) AND target_uuid IS NOT NULL
        "#
    );
    let mut q = sqlx::query_scalar::<_, String>(&sql);
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

    // Step 2: BFS each start entity up to `max_depth`, accumulating entities.
    let max_depth = opts.max_depth.min(6);
    let mut visited_entities: BTreeSet<Uuid> = start_entities.clone();
    for ent in &start_entities {
        let Ok(nodes) = svc
            .relation_repo
            .bfs_traverse(&svc.database, *ent, Direction::Both, max_depth, 500)
            .await
        else {
            continue;
        };
        for n in nodes {
            visited_entities.insert(n.entity.uuid);
        }
    }

    // Step 3: find memories that reference any visited entity (via any
    // relation row that has memory_uuid != NULL). We keep seed memories in the
    // output so the caller can mark them as "graph also confirmed" via the
    // from_graph flag; downstream callers de-dup them when merging ordered
    // lists.
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

// ---------------------------------------------------------------------------
// filter helper
// ---------------------------------------------------------------------------

fn passes_filter(r: &MemoryRecord, f: &MemoryFilter) -> bool {
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
        // Use created_at as a proxy when NULL (same as the SQL side).
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
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Uuid;
    use crate::config::StorageConfig;
    use crate::memory::ops_remember::{RememberInput, service_for_tests};
    use crate::storage::{Database, MemorySource};

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
        let err = svc.recall(RecallInput { query: "  ", ..Default::default() }).await.unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);

        let err = svc
            .recall(RecallInput { query: "q", top_k: 0, ..Default::default() })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn recall_returns_both_memories_and_marks_accessed() {
        let svc = temp_svc().await;

        // Insert two memories with clearly distinct content (hashes differ).
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

        // Recall with very permissive thresholds to pull in everything.
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
        // Both returned uuids should be the ones we inserted.
        let returned: std::collections::HashSet<Uuid> =
            out.hits.iter().map(|h| h.memory.uuid).collect();
        assert!(returned.contains(&a.uuid));
        assert!(returned.contains(&b.uuid));

        // Every returned hit (within top_k) gets its access_count bumped once.
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
                content: "alpha alpha alpha", // same content hash? no, trim—yes same.
                tags: &["y".into()],
                importance: 0.99,
                ..Default::default()
            })
            .await
            .unwrap();

        // Second call was content-duplicate. So both x.uuid and y.uuid are the same.
        // Tags are merged from x+y remember calls. We can still test tags_any filter that requires tag.
        // Use a *different* content so we actually get a different uuid:
        let zz = svc
            .remember(RememberInput {
                content: "beta beta beta beta",
                tags: &["z".into()],
                ..Default::default()
            })
            .await
            .unwrap();

        let f = MemoryFilter { tags_any: Some(vec!["z".to_string()]), ..Default::default() };
        let out = svc
            .recall(RecallInput { query: "beta", top_k: 5, filter: f, ..Default::default() })
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

        let f = MemoryFilter { importance_min: Some(0.9), ..Default::default() };
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
        // Check low has access_count == 0 still since it wasn't returned.
        let low_mem = svc.get_memory(low.uuid).await.unwrap();
        assert_eq!(low_mem.access_count, 0);
        let _ = (low,); // silence unused for non-test
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

        let f = MemoryFilter { source_in: Some(vec![MemorySource::User]), ..Default::default() };
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
}
