//! Hybrid ranker — combine vector similarity, memory-importance,
//! access-frequency/recency, and optional graph boost into a single final
//! score per recall result.
//!
//! The MVP ranker is intentionally simple and fully linear:
//!
//! ```text
//! final =  w_sim      * (raw_cosine ∊ [-1, 1] → mapped to [0, 1])
//!        + w_imp      * importance ∊ [0, 1]
//!        + w_access   * access_signal ∊ [0, 1]
//!        + w_graph    * graph_hit_boost_bonus
//! ```
//!
//! Weights default to something sensible: heavy on semantic similarity, a bit
//! of importance for long-term memory, a touch of access recency for session
//! continuity, and a graph bonus when a record was pulled in via graph
//! traversal (without similarity to anchor). Tuned empirically, and all
//! weights are exposed to callers so they can tweak.
//!
//! Every weight is normalised internally to sum to 1 so recall thresholds
//! remain comparable across weight configs.

use serde::{Deserialize, Serialize};

use crate::storage::memory::MemoryRecord;

// -----------------------------------------------------------------------------
// RankWeights + linear ranker (single-source, used in Semantic mode MVP)
// -----------------------------------------------------------------------------

/// Weights for the linear ranker. All values are non-negative; they're
/// normalised to sum to 1 internally before use.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct RankWeights {
    pub similarity: f32,
    pub importance: f32,
    pub access: f32,
    /// Extra bump when a memory was pulled in through graph expansion alone
    /// (no direct vector hit). Acts as a soft tie-breaker so graph-adjacent
    /// memories still show up near the bottom of relevant recalls.
    pub graph_boost: f32,
}

impl Default for RankWeights {
    fn default() -> Self {
        Self { similarity: 0.6, importance: 0.15, access: 0.1, graph_boost: 0.15 }
    }
}

/// Input to the ranker — one candidate memory with its retrieval signals.
#[derive(Debug, Clone)]
pub struct RankCandidate<'a> {
    pub memory: &'a MemoryRecord,
    /// Raw cosine similarity from vector search; None if this candidate
    /// arrived purely via graph traversal (use 0.0 as the base in that case
    /// but we also flip the graph_boost flag so it still gets some score).
    pub raw_similarity: Option<f32>,
    /// If true, this row was added through graph expansion and should get
    /// the `graph_boost` additive bonus.
    pub from_graph: bool,
}

/// Output of the ranker — one memory with its final score.
#[derive(Debug, Clone)]
pub struct RankedHit<'a> {
    pub memory: &'a MemoryRecord,
    /// Final score after weighting + normalisation, strictly in [0, 1].
    pub final_score: f32,
    /// Normalised components (after weight normalisation). Kept for debug
    /// and explainability surfaces.
    pub components: ScoreComponents,
}

#[derive(Debug, Clone, Default, Copy, Serialize, Deserialize)]
pub struct ScoreComponents {
    pub similarity: f32,
    pub importance: f32,
    pub access: f32,
    pub graph_boost: f32,
}

fn normalise_weights(w: RankWeights) -> RankWeights {
    let mut sim = w.similarity.max(0.0);
    let mut imp = w.importance.max(0.0);
    let mut acc = w.access.max(0.0);
    let mut gb = w.graph_boost.max(0.0);
    let total = sim + imp + acc + gb;
    if total <= 0.0 || !total.is_finite() {
        return RankWeights::default();
    }
    sim /= total;
    imp /= total;
    acc /= total;
    gb /= total;
    RankWeights { similarity: sim, importance: imp, access: acc, graph_boost: gb }
}

fn map_cosine_to_unit(raw: Option<f32>) -> f32 {
    // Cosine sim ∈ [-1, 1]. Shift+scale to [0, 1] so negative similarities
    // become 0 and identical vectors score 1. None → 0 (no signal).
    match raw {
        None => 0.0,
        Some(v) if v <= -1.0 => 0.0,
        Some(v) if v >= 1.0 => 1.0,
        Some(v) => (v + 1.0) * 0.5,
    }
}

fn access_signal(mem: &MemoryRecord) -> f32 {
    // access_count capped at 50, then sqrt to sub-linearise.
    // A memory that's been touched 50+ times gets full access-score.
    // A never-accessed memory gets 0.
    let n = mem.access_count.max(0) as f32;
    let capped = n.min(50.0);
    (capped / 50.0).sqrt()
}

/// Rank a set of candidates. Callers pass in an already-deduplicated list;
/// this function does NOT dedupe by uuid (it's the caller's responsibility).
///
/// Returns hits sorted descending by `final_score`, and also drops any hit
/// whose score is strictly below `score_threshold` (∈ [0, 1]). Pass 0.0 to
/// keep everything.
pub fn rank<'a>(
    candidates: Vec<RankCandidate<'a>>,
    weights: RankWeights,
    score_threshold: f32,
) -> Vec<RankedHit<'a>> {
    let w = normalise_weights(weights);
    let threshold = score_threshold.clamp(0.0, 1.0);

    let mut hits: Vec<RankedHit<'a>> = candidates
        .into_iter()
        .map(|c| {
            let s_sim = map_cosine_to_unit(c.raw_similarity);
            let s_imp = c.memory.importance.clamp(0.0, 1.0);
            let s_acc = access_signal(c.memory);
            let s_gb = if c.from_graph { 1.0 } else { 0.0 };

            let components = ScoreComponents {
                similarity: s_sim * w.similarity,
                importance: s_imp * w.importance,
                access: s_acc * w.access,
                graph_boost: s_gb * w.graph_boost,
            };
            let final_score = (components.similarity
                + components.importance
                + components.access
                + components.graph_boost)
                .clamp(0.0, 1.0);

            RankedHit { memory: c.memory, final_score, components }
        })
        .filter(|h| h.final_score >= threshold)
        .collect();

    hits.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            // stable secondary: most-recent memory first
            .then_with(|| b.memory.created_at.cmp(&a.memory.created_at))
    });
    hits
}

// =============================================================================
// Reciprocal Rank Fusion (RRF) — merge multiple ranked lists into one.
// =============================================================================
//
// Classic RRF formula per item:
//
//   score(uuid) = Σ over sources  weight_source / (k + rank_pos)
//
// Where `rank_pos` starts at 1 (NOT 0) and `k` is a smoothing constant that
// controls how aggressively we prefer top-ranked items. Literature commonly
// uses k=60; we use 25 by default so small, local lists (top-50) have a
// gentler fall-off — a doc that is #1 in only ONE source still beats a doc
// that's #10 in three sources.
//
// Each source is just an *ordered* list of UUIDs. Any UUID not present in a
// source is treated as having infinite rank (contributes 0 to its RRF score).

use std::collections::HashMap;

use crate::Uuid;

/// One ranked list to feed into the RRF fusion step.
pub struct RrfSource {
    /// UUIDs *sorted best-to-worst* (position 0 = rank 1).
    pub items: Vec<Uuid>,
    /// Weight for this source (0..=1 typically, all weights normalised inside).
    pub weight: f32,
    /// Name of the source, only used for debug logs ("semantic"/"keyword"/"graph").
    pub label: &'static str,
}

/// Output from RRF: one per UUID that appears in any source, with a higher
/// `score` meaning a better final ranking after fusion.
#[derive(Debug, Clone)]
pub struct RrfHit {
    pub uuid: Uuid,
    pub score: f32,
    /// Which of the inputs had this UUID present. Useful for debugging why a
    /// result showed up in hybrid output.
    pub from_sources: Vec<&'static str>,
}

/// Merge multiple ordered lists via RRF.
///
/// `smoothing_k` defaults to 25 if 0. Returns hits sorted descending by RRF
/// score. Empty input → empty output. If all weights are zero we fall back
/// to equal weight per source.
pub fn reciprocal_rank_fusion(sources: Vec<RrfSource>, smoothing_k: Option<u32>) -> Vec<RrfHit> {
    let k: f32 = {
        let raw = smoothing_k.unwrap_or(25).max(1) as f32;
        if raw <= 0.0 { 25.0 } else { raw }
    };

    // Normalise source weights so they sum to 1. If zero/non-finite → equal.
    let n_sources = sources.len() as f32;
    let mut weights: Vec<f32> = sources.iter().map(|s| s.weight.max(0.0)).collect();
    let total: f32 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        weights = std::iter::repeat(1.0 / n_sources.max(1.0)).take(sources.len()).collect();
    } else {
        for w in &mut weights {
            *w /= total;
        }
    }

    let mut accum: HashMap<Uuid, (f32, Vec<&'static str>)> = HashMap::new();
    for (i, src) in sources.iter().enumerate() {
        let w = weights[i];
        if w <= 0.0 {
            continue;
        }
        for (pos_0, uuid) in src.items.iter().enumerate() {
            let rank = (pos_0 + 1) as f32;
            let contribution = w / (k + rank);
            let entry = accum.entry(*uuid).or_insert_with(|| (0.0, Vec::new()));
            entry.0 += contribution;
            if !entry.1.contains(&src.label) {
                entry.1.push(src.label);
            }
        }
    }

    let mut out: Vec<RrfHit> = accum
        .into_iter()
        .map(|(uuid, (score, from_sources))| RrfHit { uuid, score, from_sources })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.uuid.cmp(&b.uuid))
    });
    out
}

#[cfg(test)]
mod tests_rrf {
    use super::*;

    fn u(n: u8) -> Uuid {
        let mut bytes = [0u8; 16];
        bytes[0] = n;
        bytes[1] = n;
        bytes[2] = n;
        bytes[3] = n;
        Uuid::from_bytes(bytes)
    }

    #[test]
    fn rrf_empty_sources_empty_output() {
        assert!(reciprocal_rank_fusion(vec![], None).is_empty());
    }

    #[test]
    fn rrf_one_source_preserves_order() {
        let src = vec![RrfSource { items: vec![u(1), u(2), u(3)], weight: 1.0, label: "s" }];
        let out = reciprocal_rank_fusion(src, None);
        let order: Vec<u8> = out.iter().map(|h| h.uuid.as_fields().0 as u8).collect();
        assert_eq!(order, vec![1, 2, 3]);
    }

    #[test]
    fn rrf_fusion_prefers_items_in_multiple_sources() {
        // Item A = #1 in source1 + #1 in source2 → highest.
        // Item B = #1 in source1 only → next.
        // Item C = #5 in source2 only → lowest.
        let a = u(1);
        let b = u(2);
        let c = u(3);
        let sources = vec![
            RrfSource { items: vec![a, b], weight: 1.0, label: "s1" },
            RrfSource { items: vec![a, c], weight: 1.0, label: "s2" },
        ];
        let out = reciprocal_rank_fusion(sources, None);
        assert_eq!(out[0].uuid, a);
        // b and c each appear once; since both are #2 in opposite sources
        // they get equal score, broken by uuid ordering (b=2 < c=3 → b first).
        assert_eq!(out[1].uuid, b);
        assert_eq!(out[2].uuid, c);
        assert_eq!(out[0].from_sources, vec!["s1", "s2"]);
    }

    #[test]
    fn rrf_zero_weights_fall_back_equal() {
        let a = u(1);
        let b = u(2);
        let sources = vec![
            RrfSource { items: vec![a], weight: 0.0, label: "a" },
            RrfSource { items: vec![b], weight: 0.0, label: "b" },
        ];
        let out = reciprocal_rank_fusion(sources, None);
        assert_eq!(out.len(), 2);
        assert!(out[0].score == out[1].score, "equal weight + equal rank → equal score");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{MemorySource, MemoryStatus};
    use chrono::Utc;
    use uuid::Uuid;

    fn mk_mem(access_count: i64, importance: f32) -> MemoryRecord {
        MemoryRecord {
            id: 0,
            uuid: Uuid::new_v4(),
            content: String::new(),
            content_hash: String::new(),
            metadata: serde_json::Value::Null,
            source: MemorySource::Agent,
            importance,
            access_count,
            last_accessed: None,
            created_at: Utc::now(),
            expires_at: None,
            status: MemoryStatus::Active,
            tags: Vec::new(),
        }
    }

    #[test]
    fn normalise_weights_sums_to_one() {
        let w = normalise_weights(RankWeights {
            similarity: 6.0,
            importance: 1.5,
            access: 1.0,
            graph_boost: 1.5,
        });
        let s = w.similarity + w.importance + w.access + w.graph_boost;
        assert!((s - 1.0).abs() < 1e-5);
        // 6 : 1.5 : 1 : 1.5 → 0.6 / 0.15 / 0.1 / 0.15
        assert!((w.similarity - 0.6).abs() < 1e-5);
        assert!((w.graph_boost - 0.15).abs() < 1e-5);
    }

    #[test]
    fn cosine_mapping_edges() {
        assert_eq!(map_cosine_to_unit(Some(1.0)), 1.0);
        assert_eq!(map_cosine_to_unit(Some(-1.0)), 0.0);
        assert_eq!(map_cosine_to_unit(Some(0.0)), 0.5);
        assert_eq!(map_cosine_to_unit(None), 0.0);
        // clamping beyond bounds
        assert_eq!(map_cosine_to_unit(Some(1.5)), 1.0);
        assert_eq!(map_cosine_to_unit(Some(-2.0)), 0.0);
    }

    #[test]
    fn access_signal_mono_and_capped() {
        assert_eq!(access_signal(&mk_mem(0, 0.0)), 0.0);
        assert!(access_signal(&mk_mem(5, 0.0)) > 0.0);
        let a = access_signal(&mk_mem(50, 0.0));
        let b = access_signal(&mk_mem(5000, 0.0));
        assert_eq!(a, b, "values over 50 should be capped");
        assert!((a - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rank_sorts_and_drops_below_threshold() {
        let hi = mk_mem(0, 0.9);
        let lo = mk_mem(0, 0.2);
        let cands = vec![
            RankCandidate { memory: &lo, raw_similarity: Some(0.2), from_graph: false },
            RankCandidate { memory: &hi, raw_similarity: Some(0.9), from_graph: false },
        ];
        let hits = rank(cands, RankWeights::default(), 0.0);
        assert_eq!(hits.len(), 2);
        assert!(hits[0].final_score >= hits[1].final_score);
        assert_eq!(hits[0].memory.uuid, hi.uuid);

        // With a high threshold only the top one survives.
        let cands2 = vec![
            RankCandidate { memory: &lo, raw_similarity: Some(0.2), from_graph: false },
            RankCandidate { memory: &hi, raw_similarity: Some(0.9), from_graph: false },
        ];
        let hits = rank(cands2, RankWeights::default(), 0.8);
        assert!(hits.iter().all(|h| h.final_score >= 0.8));
    }

    #[test]
    fn graph_boost_pulls_in_graph_only_rows() {
        // A graph-only hit (no vector sim) but high importance still beats
        // a very-low-similarity low-importance direct hit.
        let graph_mem = mk_mem(0, 1.0);
        let direct_mem = mk_mem(0, 0.0);
        // Use negative cosine similarity so the mapped unit-similarity is very low.
        // Some(-0.6) maps to (1 - 0.6) * 0.5 = 0.2. With w_sim=0.6 that's 0.12.
        // Graph-only score = imp(1.0)*0.15 + gb(1)*0.15 = 0.30 > 0.12.
        let cands = vec![
            RankCandidate { memory: &direct_mem, raw_similarity: Some(-0.6), from_graph: false },
            RankCandidate { memory: &graph_mem, raw_similarity: None, from_graph: true },
        ];
        let hits = rank(cands, RankWeights::default(), 0.0);
        assert_eq!(
            hits[0].memory.uuid, graph_mem.uuid,
            "graph+high-imp should outrank low-sim+low-imp"
        );
        assert!(hits[0].components.graph_boost > 0.0);
    }
}
