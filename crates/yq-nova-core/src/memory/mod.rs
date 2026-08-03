//! Business operations: `remember` / `recall` / `forget`.
//!
//! Each sub-module owns one op (input + output types + implementation), and
//! [`MemoryService`] here exposes thin wrappers so callers get the ergonomic
//! `svc.remember(input)` style. The service *owns* its dependencies so the
//! HTTP layer (or SDK embedded direct callers) can build one and reuse it.

pub mod ops_forget;
pub mod ops_recall;
pub mod ops_remember;
pub mod rank;

pub use ops_forget::{ForgetInput, ForgetMode, ForgetOutput};
pub use ops_recall::{RecallHit, RecallInput, RecallOutput};
pub use ops_remember::{RememberInput, RememberOutput};
pub use rank::RankWeights;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    embedding::{EmbeddingMeta, SharedEmbeddingProvider},
    error::NovaResult,
    graph::extractor::{EntityExtractor, NoopExtractor},
    storage::{
        Database,
        entity::SqliteEntityRepository,
        fts5::SqliteFts5Store,
        memory::{
            InsertMemoryInput, InsertOutcome, MemoryRecord, MemoryRepository,
            SqliteMemoryRepository,
        },
        relation::SqliteRelationRepository,
        tag::SqliteTagRepository,
        vector::SqliteVectorStore,
    },
};

/// recall 支持的检索模式。MVP 已实现 `Semantic`（向量语义）、
/// `Keyword`（FTS5 关键词）与 `Hybrid`（RRF 混合）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    /// 纯语义向量检索（KNN + 余弦相似度）。
    #[default]
    Semantic,
    /// 纯关键词 FTS5 检索。
    Keyword,
    /// 语义 + 关键词 + 图扩展三路 RRF 融合检索。
    Hybrid,
}

/// 混合检索模式下三路来源（语义/关键词/图）的 RRF 融合权重。
/// 每项应 ≥ 0；总和无需归一化，RRF 内部按权重缩放。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct HybridWeights {
    /// 语义检索源权重，默认 0.6。
    pub semantic: f32,
    /// 关键词检索源权重，默认 0.3。
    pub keyword: f32,
    /// 图扩展候选源权重，默认 0.1。
    pub graph: f32,
}

impl Default for HybridWeights {
    fn default() -> Self {
        Self { semantic: 0.6, keyword: 0.3, graph: 0.1 }
    }
}

/// recall 时的图谱扩展选项。
///
/// 推荐取值：`max_depth` 1~3；默认 2。深度过大会显著拉取过多候选。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct GraphTraversalOpts {
    /// 是否启用图扩展。默认 false。
    pub enabled: bool,
    /// BFS 最大遍历深度。默认 2；硬上限 6。
    pub max_depth: u8,
    /// 仅沿谓词在白名单中的边遍历；空列表表示不限制谓词。
    pub predicate_whitelist: Vec<String>,
}

/// 记忆服务：封装仓储层、Embedding 提供者与图谱抽取器。
///
/// 所有字段均 `Clone + Send + Sync`，可廉价克隆并分发到请求处理线程。
/// 对外暴露三大入口：`remember` / `recall` / `forget`。
#[derive(Clone)]
pub struct MemoryService {
    /// SQLite 数据库句柄（池 + 配置）。
    pub database: Database,
    /// 默认 Embedding 提供者（共享 Arc）。
    pub embedding: SharedEmbeddingProvider,
    /// 实体/关系抽取器（共享 Arc）。
    pub extractor: Arc<dyn EntityExtractor>,

    // Stateless repositories. Kept as fields so callers don't need to know
    // which concrete impl is in use; if we ever add feature-gated backends
    // (e.g. sqlite-vec for vectors) the swap happens here.
    /// 记忆仓储实现。
    pub memory_repo: SqliteMemoryRepository,
    /// 向量存储实现。
    pub vector_store: SqliteVectorStore,
    /// 实体仓储实现。
    pub entity_repo: SqliteEntityRepository,
    /// 关系仓储实现。
    pub relation_repo: SqliteRelationRepository,
    /// 标签仓储实现。
    pub tag_repo: SqliteTagRepository,
    /// FTS5 全文检索存储实现。
    pub fts5_store: SqliteFts5Store,
}

impl std::fmt::Debug for MemoryService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryService")
            .field("embedding_meta", self.embedding.meta())
            .finish_non_exhaustive()
    }
}

impl MemoryService {
    /// 使用显式依赖构造服务。调用方需保证 `embedding.meta()` 的维度
    /// 与向量表中已存储的向量维度一致。
    pub fn with_parts(
        database: Database,
        embedding: SharedEmbeddingProvider,
        extractor: Arc<dyn EntityExtractor>,
    ) -> Self {
        let dims = embedding.meta().dims;
        Self {
            database: database.clone(),
            embedding,
            extractor,
            memory_repo: SqliteMemoryRepository::new(),
            vector_store: SqliteVectorStore::with_db(&database, dims),
            entity_repo: SqliteEntityRepository::new(),
            relation_repo: SqliteRelationRepository::new(),
            tag_repo: SqliteTagRepository::new(),
            fts5_store: SqliteFts5Store::new(),
        }
    }

    /// 便捷构造器：使用 Noop 抽取器（不做图谱抽取）。
    /// 若需要 RegexWikiExtractor 或 LLM 抽取器，请用 [`Self::with_parts`]。
    pub fn new(database: Database, embedding: SharedEmbeddingProvider) -> Self {
        Self::with_parts(database, embedding, Arc::new(NoopExtractor))
    }

    /// 当前服务使用的 Embedding 元数据。便于调用方对齐外部向量的提供者/模型/维度。
    pub fn embedding_meta(&self) -> &EmbeddingMeta {
        self.embedding.meta()
    }

    // --- Delegating public methods. Each one lives in its own sub-module  ---
    // --- so the codebase stays readable even as we add extraction logic. ---

    /// **写入记忆**：持久化一段文本，生成语义 Embedding，可选抽取实体/关系
    /// 到图谱，并合并调用方与抽取器提供的标签。对相同内容（去空白后）的
    /// 重复调用是幂等的，会返回已存在记录的 UUID。
    pub async fn remember(
        &self,
        input: ops_remember::RememberInput<'_>,
    ) -> NovaResult<ops_remember::RememberOutput> {
        ops_remember::remember(self, input).await
    }

    /// **召回记忆**：按自然语言查询从向量、关键词、图扩展等来源检索候选，
    /// 经线性加权排序与阈值过滤后返回 top_k 命中，并对每条命中记一次
    /// `access_count + 1`。
    pub async fn recall(
        &self,
        input: ops_recall::RecallInput<'_>,
    ) -> NovaResult<ops_recall::RecallOutput> {
        ops_recall::recall(self, input).await
    }

    /// **遗忘记忆**：按 UUID 或 MemoryFilter 批量软/硬删除或归档记忆。
    /// 可选启用孤儿实体/关系 GC。
    pub async fn forget(
        &self,
        input: ops_forget::ForgetInput,
    ) -> NovaResult<ops_forget::ForgetOutput> {
        ops_forget::forget(self, input).await
    }

    // --- Direct repository access helpers for tests / advanced callers  ---

    /// Thin pass-through so the HTTP layer doesn't have to import repos.
    pub async fn get_memory(&self, uuid: uuid::Uuid) -> NovaResult<MemoryRecord> {
        self.memory_repo.get_by_uuid(&self.database, uuid).await
    }

    pub async fn insert_memory_raw<'a>(
        &self,
        input: InsertMemoryInput<'a>,
    ) -> NovaResult<InsertOutcome> {
        self.memory_repo.insert(&self.database, input).await
    }
}
