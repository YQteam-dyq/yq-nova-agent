//! Graph operations + optional LLM extractor. The MVP service exposes two
//! main entry points:
//!
//! 1. `extract_and_link(text, memory_uuid)` — run the configured EntityExtractor
//!    over the text, upsert any entities, create "mentions" (or more specific)
//!    edges between the entities mentioned by a given memory, and optionally
//!    create a per-memory link table (when we add it in M9). For now the
//!    entity → memory join is done via name matching in recall's graph
//!    expansion phase.
//!
//! 2. `traverse_graph(start_entity_uuid, opts)` — BFS from a given entity with
//!    depth limits + predicate whitelist + confidence threshold.
//!
//! The graph service is intentionally lightweight; the heavy lifting of
//! persistence lives in the SqliteEntityRepository / SqliteRelationRepository
//! storage modules. GraphService here is the *orchestrator*.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use crate::storage::entity::{Direction, EntityRecord, TraverseNode};
pub use crate::storage::relation::RelationRecord;

pub mod extractor;

use crate::{
    error::{NovaError, NovaResult},
    storage::{
        Database,
        entity::{EntityRepository, SqliteEntityRepository, UpsertEntityInput, UpsertOutcome},
        relation::{InsertRelationInput, RelationRepository, SqliteRelationRepository},
    },
};
use extractor::{EntityCandidate, EntityExtractor, Extraction, NoopExtractor, RelationCandidate};

/// 图谱抽取选项：控制 `extract_and_link` 的行为。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphExtractOpts {
    /// 若为 false，`extract_and_link` 直接短路为空操作。便于共享同一
    /// GraphService 实例但对部分内容类型跳过抽取。
    pub enabled: bool,
    /// 是否将抽取到的实体 upsert 到 entities 表。默认 true。
    pub upsert_entities: bool,
    /// 是否在实体间创建关系边。默认 true。
    pub create_relations: bool,
    /// 置信度严格低于该阈值的实体/关系会被跳过。合法范围 [0.0, 1.0]。
    /// 默认 0.0（接受所有，包括 RegexWikiExtractor 的 0.3 共现「mentions」边）。
    pub min_confidence: f32,
}

impl Default for GraphExtractOpts {
    fn default() -> Self {
        Self { enabled: false, upsert_entities: true, create_relations: true, min_confidence: 0.0 }
    }
}

/// extract-and-link 接口的返回结果（也即 SDK 中 ExtractAndLinkResponse）。
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LinkResult {
    /// 抽取出的实体候选列表：每个元素是（候选信息、对应图谱实体 UUID）对。
    pub entities: Vec<(EntityCandidate, Uuid)>,
    /// 实际 upsert 到数据库中的实体个数。
    pub entities_upserted: usize,
    /// 实际新创建的关系边条数。
    pub relations_created: usize,
    /// 文本中识别出的标签集合。
    pub tags: Vec<String>,
}

/// 实体合并输入。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEntitiesInput {
    /// 合并后保留的实体 UUID。
    pub keep_uuid: Uuid,
    /// 被合并的实体 UUID 列表（1~50 个）。
    pub discard_uuids: Vec<Uuid>,
}

/// 实体合并输出。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEntitiesOutput {
    /// 保留的实体 UUID。
    pub kept_uuid: Uuid,
    /// 实际被合并的实体 UUID 列表。
    pub merged: Vec<Uuid>,
    /// 被重映射的关系边数量。
    pub remapped_relations: usize,
}

/// 图谱 BFS 遍历参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraverseOpts {
    /// 最大遍历深度，默认 2；推荐 1~3。
    pub max_depth: u8,
    /// 最大访问节点数上限，避免大图爆炸。默认 200。
    pub max_nodes: usize,
    /// 仅沿谓词在白名单中的边遍历；空列表表示不限制。
    pub predicate_whitelist: Vec<String>,
    /// 置信度阈值，仅沿置信度 ≥ 该值的边遍历。合法范围 [0.0, 1.0]。
    pub min_confidence: f32,
}

impl Default for TraverseOpts {
    fn default() -> Self {
        Self { max_depth: 2, max_nodes: 200, predicate_whitelist: Vec::new(), min_confidence: 0.0 }
    }
}

/// 图谱服务：封装实体/关系仓储 + 抽取器，对外提供抽取（extract_and_link）
/// 与遍历（traverse_graph）两大入口。
///
/// 实际持久化逻辑位于 SqliteEntityRepository / SqliteRelationRepository，
/// GraphService 是轻量编排层。
#[derive(Clone)]
pub struct GraphService {
    /// SQLite 数据库句柄。
    pub database: Database,
    /// 实体/关系抽取器（共享 Arc）。
    pub extractor: Arc<dyn EntityExtractor>,
    /// 实体仓储实现。
    pub entity_repo: SqliteEntityRepository,
    /// 关系仓储实现。
    pub relation_repo: SqliteRelationRepository,
}

impl std::fmt::Debug for GraphService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphService").finish_non_exhaustive()
    }
}

impl GraphService {
    /// 使用显式抽取器构造服务。可接入 RegexWikiExtractor 或 LLM 抽取器。
    pub fn with_parts(database: Database, extractor: Arc<dyn EntityExtractor>) -> Self {
        Self {
            database,
            extractor,
            entity_repo: SqliteEntityRepository::new(),
            relation_repo: SqliteRelationRepository::new(),
        }
    }

    /// 便捷构造器：使用 Noop 抽取器（不做任何抽取）。
    pub fn new(database: Database) -> Self {
        Self::with_parts(database, Arc::new(NoopExtractor))
    }

    /// 对一段文本执行实体抽取 → 实体 upsert → 关系边创建。
    ///
    /// 通常作为 `remember()` 后的后置步骤，但也可被高级调用方直接用于
    /// 向图谱喂入任意文本。空文本或 opts.enabled=false 时直接短路返回空结果。
    pub async fn extract_and_link(
        &self,
        text: &str,
        opts: &GraphExtractOpts,
    ) -> NovaResult<LinkResult> {
        if !opts.enabled || text.trim().is_empty() {
            return Ok(LinkResult::default());
        }
        let extraction: Extraction = self.extractor.extract(text).await.unwrap_or_default();

        // --- step 1: upsert entities ---------------------------------------
        let mut entity_uuids: std::collections::HashMap<(String, String), Uuid> =
            std::collections::HashMap::new();
        let mut entities: Vec<(EntityCandidate, Uuid)> = Vec::new();
        let mut upserted = 0usize;
        if opts.upsert_entities {
            for ent in &extraction.entities {
                let r = upsert_one_entity(self, ent).await;
                if let Some((outcome, uuid)) = r {
                    if matches!(outcome, UpsertOutcome::Created(_)) {
                        upserted += 1;
                    }
                    entity_uuids.insert((ent.name.clone(), ent.entity_type.clone()), uuid);
                    entities.push((ent.clone(), uuid));
                }
            }
        }

        // --- step 2: create relation edges ---------------------------------
        let mut created = 0usize;
        if opts.create_relations && entity_uuids.len() >= 2 {
            for rel in dedupe_by_confidence(&extraction.relations) {
                if rel.confidence < opts.min_confidence {
                    continue;
                }
                if let Some(true) = link_one_relation(self, &entity_uuids, &rel).await {
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

    /// 从指定实体出发进行 BFS 图谱遍历。
    ///
    /// 是 `RelationRepository::bfs_traverse` 的简单封装，提供深度、节点数、
    /// 置信度等参数控制；会先校验起始实体是否存在，不存在则返回 NotFound。
    pub async fn traverse_graph(
        &self,
        start: Uuid,
        opts: TraverseOpts,
    ) -> NovaResult<Vec<TraverseNode>> {
        // Validate start exists.
        let _start_ent = self.entity_repo.get_by_uuid(&self.database, start).await?;
        let nodes = self
            .relation_repo
            .bfs_traverse(
                &self.database,
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

    /// 列举实体，支持按名称前缀与实体类型过滤。
    ///
    /// 是实体仓储的薄封装，方便 HTTP 层无需直接 import Repository 即可调用。
    pub async fn list_entities(
        &self,
        name_prefix: Option<&str>,
        entity_type: Option<&str>,
        limit: usize,
    ) -> NovaResult<Vec<EntityRecord>> {
        self.entity_repo.list(&self.database, name_prefix, entity_type, limit, 0).await
    }

    /// 合并实体：将 discard_uuids 中的实体合并到 keep_uuid 实体。
    pub async fn merge_entities(&self, input: MergeEntitiesInput) -> NovaResult<MergeEntitiesOutput> {
        let keep = input.keep_uuid;
        let discards = input.discard_uuids;

        if discards.is_empty() || discards.len() > 50 {
            return Err(NovaError::validation(
                "discard_uuids must contain between 1 and 50 UUIDs",
            ));
        }
        if discards.contains(&keep) {
            return Err(NovaError::validation(
                "keep_uuid must not be in discard_uuids",
            ));
        }

        let _keep_entity = self.entity_repo.get_by_uuid(&self.database, keep).await?;
        for &d in &discards {
            self.entity_repo.get_by_uuid(&self.database, d).await?;
        }

        let pool = &self.database.pool;
        let keep_str = keep.to_string();
        let discard_set: std::collections::HashSet<Uuid> = discards.iter().copied().collect();
        let mut remapped = 0usize;

        for &d in &discards {
            let d_str = d.to_string();

            let outgoing: Vec<(String, String)> = sqlx::query_as(
                "SELECT uuid, target_uuid FROM relations WHERE source_uuid = ?1",
            )
            .bind(&d_str)
            .fetch_all(pool)
            .await
            .map_err(NovaError::storage)?;

            for (rel_uuid, tgt_str) in &outgoing {
                let tgt = match Uuid::parse_str(tgt_str) {
                    Ok(u) => u,
                    Err(_) => continue,
                };
                if tgt == keep || discard_set.contains(&tgt) {
                    sqlx::query("DELETE FROM relations WHERE uuid = ?1")
                        .bind(rel_uuid)
                        .execute(pool)
                        .await
                        .map_err(NovaError::storage)?;
                } else {
                    sqlx::query("UPDATE relations SET source_uuid = ?1 WHERE uuid = ?2")
                        .bind(&keep_str)
                        .bind(rel_uuid)
                        .execute(pool)
                        .await
                        .map_err(NovaError::storage)?;
                    remapped += 1;
                }
            }

            let incoming: Vec<(String, String)> = sqlx::query_as(
                "SELECT uuid, source_uuid FROM relations WHERE target_uuid = ?1",
            )
            .bind(&d_str)
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
                    sqlx::query("DELETE FROM relations WHERE uuid = ?1")
                        .bind(rel_uuid)
                        .execute(pool)
                        .await
                        .map_err(NovaError::storage)?;
                } else {
                    sqlx::query("UPDATE relations SET target_uuid = ?1 WHERE uuid = ?2")
                        .bind(&keep_str)
                        .bind(rel_uuid)
                        .execute(pool)
                        .await
                        .map_err(NovaError::storage)?;
                    remapped += 1;
                }
            }

            self.entity_repo.delete(&self.database, d).await?;
        }

        sqlx::query("DELETE FROM relations WHERE source_uuid = target_uuid")
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

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

async fn upsert_one_entity(
    svc: &GraphService,
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
    use crate::Uuid;
    use crate::config::StorageConfig;
    use crate::graph::extractor::RegexWikiExtractor;
    use crate::storage::Database;

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
                "[[A]] and [[B]]",
                &GraphExtractOpts { enabled: false, ..Default::default() },
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
                "#todo [[Alice Smith]] had a meeting with [[Bob Jones]] at [[Acme Corp]].",
                &GraphExtractOpts { enabled: true, ..Default::default() },
            )
            .await
            .unwrap();
        // #todo tag comes through
        assert!(r.tags.iter().any(|t| t == "todo"));
        // Three wiki entities (Alice, Bob, Acme)
        assert!(r.entities_upserted >= 3, "entities upserted: {}", r.entities_upserted);
        // C(3,2)=3 mentions edges for co-occurring entities
        assert!(r.relations_created >= 3);
    }

    async fn upsert(db: &Database, repo: &SqliteEntityRepository, n: &str, t: &str) -> Uuid {
        let out = repo
            .upsert(
                db,
                UpsertEntityInput { name: n, r#type: t, description: None, metadata: None },
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
            .traverse_graph(a, TraverseOpts { max_depth: 2, max_nodes: 50, ..Default::default() })
            .await
            .unwrap();
        // BFS from A with depth 2 should visit all 4 nodes.
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
            .merge_entities(MergeEntitiesInput {
                keep_uuid: a,
                discard_uuids: vec![b],
            })
            .await
            .unwrap();
        assert_eq!(out.kept_uuid, a);
        assert_eq!(out.merged, vec![b]);
        assert_eq!(out.remapped_relations, 1);

        assert!(svc.entity_repo.get_by_uuid(&svc.database, a).await.is_ok());
        assert!(svc.entity_repo.get_by_uuid(&svc.database, b).await.is_err());

        let outgoing = svc.relation_repo.list_outgoing(&svc.database, a, None, 100).await.unwrap();
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].target_uuid, c);
    }

    #[tokio::test]
    async fn merge_entities_rejects_keep_in_discard() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let err = svc
            .merge_entities(MergeEntitiesInput {
                keep_uuid: a,
                discard_uuids: vec![a],
            })
            .await
            .unwrap_err();
        assert_eq!(err.code(), crate::error::ErrorCode::Validation);
    }

    #[tokio::test]
    async fn merge_entities_rejects_empty_discard() {
        let svc = temp_svc().await;
        let a = upsert(&svc.database, &svc.entity_repo, "A", "t").await;
        let err = svc
            .merge_entities(MergeEntitiesInput {
                keep_uuid: a,
                discard_uuids: vec![],
            })
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
            .merge_entities(MergeEntitiesInput {
                keep_uuid: a,
                discard_uuids: vec![missing],
            })
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
            .merge_entities(MergeEntitiesInput {
                keep_uuid: a,
                discard_uuids: vec![b],
            })
            .await
            .unwrap();
        assert_eq!(out.remapped_relations, 0);

        let outgoing = svc.relation_repo.list_outgoing(&svc.database, a, None, 100).await.unwrap();
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
            .merge_entities(MergeEntitiesInput {
                keep_uuid: a,
                discard_uuids: vec![b, c],
            })
            .await
            .unwrap();
        assert_eq!(out.remapped_relations, 2);

        let outgoing = svc.relation_repo.list_outgoing(&svc.database, a, None, 100).await.unwrap();
        assert_eq!(outgoing.len(), 2);
        for rel in &outgoing {
            assert_eq!(rel.target_uuid, d);
        }
    }
}
