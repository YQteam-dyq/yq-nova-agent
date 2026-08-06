//! Embedded-mode SDK impl.
//!
//! Compiles `yq-nova-core` directly into the host process. Zero network
//! calls, best performance, sharing the same single SQLite database file.
//! This is the recommended mode for Rust-native callers who want the
//! lowest latency and no locally-running HTTP server.

use std::{path::PathBuf, sync::Arc};

use yq_nova_core::{
    Uuid,
    config::StorageConfig,
    embedding::{MockEmbeddingProvider, SharedEmbeddingProvider},
    error::{NovaError, NovaResult},
    graph::{GraphService, LinkResult, TraverseNode, TraverseOpts},
    memory::{
        ForgetInput, ForgetOutput, MemoryService, RecallOutput, RememberOutput, ops_forget,
        ops_recall, ops_remember,
    },
    storage::{Database, entity::EntityRepository},
};

use crate::http_client;

/// Embedded-mode yq-nova client.
///
/// Owns the concrete [`Database`], [`MemoryService`] and [`GraphService`]
/// instances so callers can call `remember` / `recall` / `forget` / graph
/// ops directly in-process, without any HTTP hop. All components are
/// `Clone` (sharing the same SQLite pool under the hood), so an
/// `EmbeddedNova` can be cheaply cloned across threads.
///
/// # Constructing
///
/// - [`EmbeddedNova::open`] — open (or create) a SQLite DB at a path and
///   wire up a local [`MockEmbeddingProvider`]. Zero network, best for local
///   use / tests.
/// - [`EmbeddedNova::from_services`] — for advanced callers who already
///   built their own providers / services (e.g. a real OpenAI embedder via
///   `yq_nova_core::embedding::openai_compat`).
#[derive(Clone)]
pub struct EmbeddedNova {
    /// The shared SQLite database handle (pool + config).
    pub database: Database,
    /// Memory service (remember / recall / forget).
    pub memory: MemoryService,
    /// Graph service (extract-and-link / traverse / list entities).
    pub graph: GraphService,
}

impl EmbeddedNova {
    /// Open (or create) a SQLite database at `db_path` and build a fully
    /// local, zero-network embedded client.
    ///
    /// Uses `Database::open` with default storage tuning, a
    /// [`MockEmbeddingProvider`] (deterministic pseudo-vectors, no network),
    /// and `MemoryService::new` / `GraphService::new` (noop extractors).
    pub async fn open(db_path: impl Into<PathBuf>) -> NovaResult<Self> {
        let database = Database::open(StorageConfig { db_path: db_path.into(), ..Default::default() })
            .await?;
        let embedding: SharedEmbeddingProvider = Arc::new(MockEmbeddingProvider::new(64));
        let memory = MemoryService::new(database.clone(), embedding);
        let graph = GraphService::new(database.clone());
        Ok(Self { database, memory, graph })
    }

    /// Construct an embedded client from already-built services.
    ///
    /// Advanced callers who need a real embedder / extractor (e.g. an
    /// OpenAI-compatible provider) should build [`MemoryService`] /
    /// [`GraphService`] themselves and wire them in here.
    pub fn from_services(
        database: Database,
        memory: MemoryService,
        graph: GraphService,
    ) -> Self {
        Self { database, memory, graph }
    }

    // --- memory -----------------------------------------------------------

    /// Persist a memory (text + importance + tags + optional embedding).
    pub async fn remember(&self, req: http_client::RememberRequest) -> NovaResult<RememberOutput> {
        let input = ops_remember::RememberInput {
            content: &req.content,
            source: req.source,
            importance: req.importance,
            metadata: req.metadata.as_ref(),
            expires_at: req.expires_at,
            tags: req.tags.as_ref(),
            embed: req.embed,
            extract_graph: req.extract_graph,
        };
        self.memory.remember(input).await
    }

    /// Retrieve memories matching a natural-language query.
    pub async fn recall(&self, req: http_client::RecallRequest) -> NovaResult<RecallOutput> {
        let input = ops_recall::RecallInput {
            query: &req.query,
            top_k: req.top_k,
            score_threshold: req.score_threshold,
            similarity_threshold: req.similarity_threshold,
            mode: req.mode,
            graph: req.graph,
            hybrid_weights: req.hybrid_weights,
            rrf_k: req.rrf_k,
            rank_weights: req.rank_weights,
            filter: req.filter,
        };
        self.memory.recall(input).await
    }

    /// Forget / archive memories (by UUID or by filter).
    pub async fn forget(&self, input: ForgetInput) -> NovaResult<ForgetOutput> {
        self.memory.forget(input).await
    }

    /// Fetch a single memory record by UUID.
    pub async fn get_memory(&self, uuid: Uuid) -> NovaResult<yq_nova_core::storage::MemoryRecord> {
        self.memory.get_memory(uuid).await
    }

    /// Hard-delete a single memory by UUID (not recoverable).
    pub async fn delete_memory(&self, uuid: Uuid) -> NovaResult<ForgetOutput> {
        self.memory
            .forget(ops_forget::ForgetInput {
                target: ops_forget::ForgetTarget::One(uuid),
                mode: ops_forget::ForgetMode::Hard,
                gc_graph: false,
                batch_limit: 1,
            })
            .await
    }

    // --- graph ------------------------------------------------------------

    /// Create or update a knowledge-graph entity keyed on `(name, type)`.
    pub async fn upsert_entity(
        &self,
        req: http_client::UpsertEntityRequest,
    ) -> NovaResult<http_client::UpsertEntityResponse> {
        let outcome = self
            .graph
            .entity_repo
            .upsert(
                &self.graph.database,
                yq_nova_core::storage::entity::UpsertEntityInput {
                    name: &req.name,
                    r#type: &req.entity_type,
                    description: req.description.as_deref(),
                    metadata: req.metadata.as_ref(),
                },
            )
            .await?;
        let uuid = outcome.uuid();
        let entity = self.graph.entity_repo.get_by_uuid(&self.graph.database, uuid).await?;
        Ok(http_client::UpsertEntityResponse { outcome, entity })
    }

    /// BFS-traverse the graph from a start entity.
    pub async fn traverse(
        &self,
        req: http_client::TraverseRequest,
    ) -> NovaResult<Vec<TraverseNode>> {
        let opts = TraverseOpts {
            max_depth: req.max_depth,
            max_nodes: req.max_nodes,
            predicate_whitelist: req.predicate_whitelist,
            min_confidence: req.min_confidence,
        };
        self.graph.traverse_graph(req.start, opts).await
    }

    /// Extract entities/relations from free text and (optionally) write
    /// them into the graph.
    pub async fn extract_and_link(
        &self,
        req: http_client::ExtractAndLinkRequest,
    ) -> NovaResult<LinkResult> {
        self.graph.extract_and_link(&req.text, &req.opts).await
    }

    // --- meta -------------------------------------------------------------

    /// Synthetic health check for the embedded client.
    pub async fn health(&self) -> NovaResult<http_client::HealthResponse> {
        Ok(http_client::HealthResponse {
            status: "ok".into(),
            version: yq_nova_core::VERSION.into(),
            git_sha: yq_nova_core::git_sha().into(),
            uptime_secs: 0,
        })
    }

    /// Compute coarse-grained statistics from the database directly.
    pub async fn stats(&self) -> NovaResult<http_client::StatsResponse> {
        Ok(http_client::StatsResponse {
            uptime_secs: 0,
            database_size_bytes: self.database.size_on_disk_bytes().unwrap_or(0),
            memory_active: self.count("SELECT COUNT(*) FROM memory_items WHERE status = 'active'").await?,
            memory_archived: self
                .count("SELECT COUNT(*) FROM memory_items WHERE status = 'archived'")
                .await?,
            memory_total: self.count("SELECT COUNT(*) FROM memory_items").await?,
            entity_count: self.count("SELECT COUNT(*) FROM entities").await?,
            relation_count: self.count("SELECT COUNT(*) FROM relations").await?,
            tag_count: self.count("SELECT COUNT(*) FROM tags").await?,
        })
    }

    async fn count(&self, sql: &str) -> NovaResult<u64> {
        let n: i64 = sqlx::query_scalar(sql)
            .fetch_one(&self.database.pool)
            .await
            .map_err(NovaError::storage)?;
        Ok(n.max(0) as u64)
    }
}