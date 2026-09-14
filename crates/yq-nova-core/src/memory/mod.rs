pub mod chunk;
pub mod ops_batch;
pub mod ops_export;
pub mod ops_forget;
pub mod ops_import;
pub mod ops_list;
pub mod ops_merge;
pub mod ops_recall;
pub mod ops_remember;
pub mod ops_tag;
pub mod ops_update;
pub mod quality;
pub mod rank;

use std::sync::Arc;

pub use chunk::{ChunkInfo, ChunkOptions, SplitBy};
pub use ops_batch::{
    BatchRememberInput, BatchRememberItem, BatchRememberOutput, BatchRememberResult,
};
pub use ops_export::{ExportInput, ExportOutput};
pub use ops_forget::{ForgetInput, ForgetMode, ForgetOutput};
pub use ops_import::{ConflictStrategy, ImportError, ImportInput, ImportItem, ImportOutput};
pub use ops_list::{ListInput, ListOutput};
pub use ops_merge::{MergeInput, MergeOutput};
pub use ops_recall::{RecallHit, RecallInput, RecallOutput};
pub use ops_remember::{RememberInput, RememberOutput};
pub use ops_tag::{
    TagDeleteInput, TagDeleteOutput, TagListInput, TagListOutput, TagRenameInput, TagRenameOutput,
};
pub use ops_update::UpdateInput;
pub use rank::RankWeights;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    #[default]
    Semantic,

    Keyword,

    Hybrid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct HybridWeights {
    pub semantic: f32,

    pub keyword: f32,

    pub graph: f32,
}

impl Default for HybridWeights {
    fn default() -> Self {
        Self {
            semantic: 0.6,
            keyword: 0.3,
            graph: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct GraphTraversalOpts {
    pub enabled: bool,

    pub max_depth: u8,

    pub predicate_whitelist: Vec<String>,
}

#[derive(Clone)]
pub struct MemoryService {
    pub database: Database,

    pub embedding: SharedEmbeddingProvider,

    pub extractor: Arc<dyn EntityExtractor>,

    pub memory_repo: SqliteMemoryRepository,

    pub vector_store: SqliteVectorStore,

    pub entity_repo: SqliteEntityRepository,

    pub relation_repo: SqliteRelationRepository,

    pub tag_repo: SqliteTagRepository,

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

    pub fn new(database: Database, embedding: SharedEmbeddingProvider) -> Self {
        Self::with_parts(database, embedding, Arc::new(NoopExtractor))
    }

    pub fn embedding_meta(&self) -> &EmbeddingMeta {
        self.embedding.meta()
    }

    pub async fn remember(
        &self,
        input: ops_remember::RememberInput<'_>,
    ) -> NovaResult<ops_remember::RememberOutput> {
        ops_remember::remember(self, input).await
    }

    pub async fn remember_batch(
        &self,
        input: ops_batch::BatchRememberInput,
    ) -> NovaResult<ops_batch::BatchRememberOutput> {
        ops_batch::remember_batch(self, input).await
    }

    pub async fn recall(
        &self,
        input: ops_recall::RecallInput<'_>,
    ) -> NovaResult<ops_recall::RecallOutput> {
        ops_recall::recall(self, input).await
    }

    pub async fn list_memories(
        &self,
        input: ops_list::ListInput,
    ) -> NovaResult<ops_list::ListOutput> {
        ops_list::list_memories(self, input).await
    }

    pub async fn forget(
        &self,
        input: ops_forget::ForgetInput,
    ) -> NovaResult<ops_forget::ForgetOutput> {
        ops_forget::forget(self, input).await
    }

    pub async fn export(
        &self,
        input: ops_export::ExportInput,
    ) -> NovaResult<ops_export::ExportOutput> {
        ops_export::export_memories(self, input).await
    }

    pub async fn import(
        &self,
        input: ops_import::ImportInput,
    ) -> NovaResult<ops_import::ImportOutput> {
        ops_import::import_memories(self, input).await
    }

    pub async fn merge(&self, input: ops_merge::MergeInput) -> NovaResult<ops_merge::MergeOutput> {
        ops_merge::merge_memories(self, input).await
    }

    pub async fn update(&self, uuid: uuid::Uuid, input: UpdateInput) -> NovaResult<MemoryRecord> {
        ops_update::update_memory(self, uuid, input).await
    }

    pub async fn list_tags(
        &self,
        input: ops_tag::TagListInput,
    ) -> NovaResult<ops_tag::TagListOutput> {
        ops_tag::list_tags(self, input).await
    }

    pub async fn rename_tag(
        &self,
        input: ops_tag::TagRenameInput,
    ) -> NovaResult<ops_tag::TagRenameOutput> {
        ops_tag::rename_tag(self, input).await
    }

    pub async fn delete_tag(
        &self,
        input: ops_tag::TagDeleteInput,
    ) -> NovaResult<ops_tag::TagDeleteOutput> {
        ops_tag::delete_tag(self, input).await
    }

    pub async fn get_memory(&self, uuid: uuid::Uuid) -> NovaResult<MemoryRecord> {
        self.get_memory_in(crate::storage::namespace::DEFAULT_NAMESPACE_ID, uuid).await
    }

    pub async fn get_memory_in(
        &self,
        namespace_id: i64,
        uuid: uuid::Uuid,
    ) -> NovaResult<MemoryRecord> {
        self.memory_repo.get_by_uuid(&self.database, namespace_id, uuid).await
    }

    pub async fn find_near_duplicates(
        &self,
        namespace_id: i64,
        content: &str,
        opts: quality::dedup::DedupOptions,
    ) -> NovaResult<Vec<quality::dedup::SimilarHit>> {
        quality::dedup::detect(self, namespace_id, content, &opts).await
    }

    pub async fn summarize_memories(
        &self,
        namespace_id: i64,
        member_uuids: &[uuid::Uuid],
        opts: quality::summary::SummaryOptions,
    ) -> NovaResult<quality::summary::SummaryEntry> {
        quality::summary::summarize_group(self, namespace_id, member_uuids, &opts).await
    }

    pub async fn rebalance_importance(
        &self,
        namespace_id: i64,
        opts: quality::importance::RebalanceOptions,
    ) -> NovaResult<quality::importance::RebalanceOutput> {
        quality::importance::rebalance(self, namespace_id, opts).await
    }

    pub async fn discover_associations(
        &self,
        namespace_id: i64,
        opts: quality::associate::AssociateOptions,
    ) -> NovaResult<quality::associate::AssociateOutput> {
        quality::associate::discover(self, namespace_id, opts).await
    }

    pub async fn insert_memory_raw<'a>(
        &self,
        input: InsertMemoryInput<'a>,
    ) -> NovaResult<InsertOutcome> {
        self.memory_repo.insert(&self.database, input).await
    }
}
