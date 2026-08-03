//! Synchronous-ish HTTP client for the yq-nova server.
//!
//! Thin async wrapper around `reqwest::Client` with:
//! - Auto-JSON request/response bodies
//! - `{code, message, trace_id?}` error mapping to `NovaError`
//! - Optional per-call trace-id propagation via `x-trace-id` header
//! - Builder methods (`remember()`, `recall()`, `forget()`) so callers don't
//!   have to type out the long DTO names.
//!
//! The DTOs here live alongside the client (rather than being re-exported from
//! `yq-nova-server`) so downstream users of `yq-nova-sdk` don't need to pull
//! in the axum/sqlx/tracing heavy server crate.

use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::{Client as ReqwestClient, Response, StatusCode, header};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use yq_nova_core::{
    Uuid,
    error::{ErrorCode, NovaError, NovaResult},
    graph::{GraphExtractOpts, LinkResult},
    memory::{
        ForgetInput, ForgetOutput, GraphTraversalOpts, HybridWeights, RankWeights, RecallOutput,
        RememberOutput, SearchMode,
    },
    storage::{
        EntityRecord, MemoryFilter, MemoryRecord, MemorySource, RelationRecord, TraverseNode,
        UpsertOutcome,
    },
};

// ---------- Memory DTOs (mirrors of yq-nova-server/http/memory.rs) ---------

/// POST /v1/memory/remember 的请求体。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RememberRequest {
    /// 要记住的原始文本内容。
    pub content: String,
    /// 这条记忆的来源渠道（Agent / User / System 等）。
    pub source: MemorySource,
    /// 重要性评分，范围 0.0 ~ 1.0，越高越不容易被遗忘。
    pub importance: f32,
    /// 任意结构化元数据，会原样存储并在 recall 时返回。
    pub metadata: Option<serde_json::Value>,
    /// 过期时间（UTC），到点后自动归档；None 表示永不过期。
    pub expires_at: Option<DateTime<Utc>>,
    /// 用户自定义标签列表，可用于检索过滤。
    pub tags: Vec<String>,
    /// 是否对 content 生成向量嵌入以支持语义检索。
    pub embed: bool,
    /// 是否从 content 中自动抽取实体与关系并写入知识图谱。
    pub extract_graph: bool,
}

impl Default for RememberRequest {
    fn default() -> Self {
        Self {
            content: String::new(),
            source: MemorySource::Agent,
            importance: 0.5,
            metadata: None,
            expires_at: None,
            tags: vec![],
            embed: true,
            extract_graph: false,
        }
    }
}

/// POST /v1/memory/recall 的请求体。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RecallRequest {
    /// 用于语义 / 关键词匹配的查询文本。
    pub query: String,
    /// 返回结果的最大条数（至少 1）。
    pub top_k: usize,
    /// 综合得分阈值，低于该值的结果会被过滤掉。
    pub score_threshold: f32,
    /// 纯向量相似度阈值（仅用于 Hybrid / Semantic 模式）。
    pub similarity_threshold: f32,
    /// 检索模式：关键词、语义、混合或图谱增强等。
    pub mode: SearchMode,
    /// 图谱遍历扩展选项（启用后会在召回阶段展开相关实体）。
    pub graph: GraphTraversalOpts,
    /// Hybrid 模式下关键词 vs 语义分的权重；None 使用服务端默认值。
    pub hybrid_weights: Option<HybridWeights>,
    /// RRF 融合算法的 k 参数；None 使用服务端默认值（通常 60）。
    pub rrf_k: Option<u32>,
    /// 最终重排阶段，各维度（相似度 / 重要性 / 新鲜度）的权重。
    pub rank_weights: Option<RankWeights>,
    /// 记忆过滤条件（按状态、标签、重要性范围等过滤候选集）。
    pub filter: MemoryFilter,
}

// ---------- Graph DTOs (mirrors yq-nova-server/http/graph.rs) --------------

/// POST /v1/graph/entities 的请求体（基于 `(name, type)` 唯一键做 upsert）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UpsertEntityRequest {
    /// 实体名称，例如 "Alice"、"Rust"。与 `entity_type` 联合作为唯一键。
    pub name: String,
    /// 实体类型 / 分类，例如 "person"、"programming_language"。
    #[serde(rename = "type")]
    pub entity_type: String,
    /// 自由文本描述；None 时保留已有记录的 description 不变。
    pub description: Option<String>,
    /// 任意结构化元数据；None 时保留已有记录的 metadata 不变。
    pub metadata: Option<serde_json::Value>,
}

impl Default for UpsertEntityRequest {
    fn default() -> Self {
        Self {
            name: String::new(),
            entity_type: "generic".into(),
            description: None,
            metadata: None,
        }
    }
}

/// POST /v1/graph/entities 的响应体。
///
/// 包含完整的创建/更新后实体记录以及一个 `UpsertOutcome` 字段，
/// 用于区分是新创建（Created）还是对已有记录的更新（Updated）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertEntityResponse {
    /// 操作结果：Created(uuid) 或 Updated(uuid)。
    #[serde(flatten)]
    pub outcome: UpsertOutcome,
    /// 创建/更新后的实体完整记录。
    pub entity: EntityRecord,
}

/// POST /v1/graph/relations 的请求体。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UpsertRelationRequest {
    /// 起点实体的 UUID（出边）。
    pub source_uuid: Uuid,
    /// 终点实体的 UUID（入边）。
    pub target_uuid: Uuid,
    /// 关系谓词，例如 "reports_to"、"written_in"。
    pub predicate: String,
    /// 置信度，0.0 ~ 1.0，越高表示关系越可信。
    pub confidence: f32,
    /// 任意结构化元数据；None 时保留已有值（仅 update 路径）。
    pub metadata: Option<serde_json::Value>,
    /// 是否幂等：true 时若 `(source, predicate, target)` 已存在则跳过。
    pub idempotent: bool,
    /// 可选，关联到哪条记忆（用于从 remember 自动抽取的关系溯源）。
    pub memory_uuid: Option<Uuid>,
}

impl Default for UpsertRelationRequest {
    fn default() -> Self {
        Self {
            source_uuid: Uuid::nil(),
            target_uuid: Uuid::nil(),
            predicate: String::new(),
            confidence: 1.0,
            metadata: None,
            idempotent: true,
            memory_uuid: None,
        }
    }
}

/// POST /v1/graph/relations 的响应体。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRelationResponse {
    /// 是否为全新插入（即此前该三元组不存在）。
    pub inserted: bool,
    /// 是否对已有记录执行了字段更新（confidence / metadata 等变更）。
    pub updated: bool,
    /// 该关系行的 UUID。
    pub relation_uuid: Uuid,
    /// 创建/更新后的完整关系记录。
    pub relation: RelationRecord,
}

/// POST /v1/graph/traverse 的请求体（BFS 遍历）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TraverseRequest {
    /// 遍历起点实体的 UUID。
    pub start: Uuid,
    /// BFS 最大深度（跳数），例如 3 表示最多走 3 步关系。
    pub max_depth: u8,
    /// 返回节点数上限，用于限制超大子图的结果规模。
    pub max_nodes: usize,
    /// 关系谓词白名单；空列表表示不做过滤，所有谓词均可走。
    pub predicate_whitelist: Vec<String>,
    /// 最小置信度阈值，低于该值的边不会被遍历。
    pub min_confidence: f32,
}

impl Default for TraverseRequest {
    fn default() -> Self {
        Self {
            start: Uuid::nil(),
            max_depth: 3,
            max_nodes: 100,
            predicate_whitelist: vec![],
            min_confidence: 0.0,
        }
    }
}

/// POST /v1/graph/extract-and-link 的请求体。
///
/// 从自由文本中自动抽取实体候选并（可选）写入实体库/建立关系，
/// 返回抽取到的实体、关系、标签及统计信息。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ExtractAndLinkRequest {
    /// 待分析的原始文本。
    pub text: String,
    /// 抽取选项：是否启用、是否 upsert 实体、是否创建关系、最小置信度。
    pub opts: GraphExtractOpts,
}

// ---------- Meta DTOs -------------------------------------------------------

/// GET /v1/health 的响应体。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// 服务健康状态，正常情况下固定为 "ok"。
    pub status: String,
    /// 服务端语义化版本号（与 `yq_nova_core::VERSION` 一致）。
    pub version: String,
    /// 构建时的 Git SHA，便于追踪部署版本。
    pub git_sha: String,
    /// 进程已启动秒数。
    pub uptime_secs: u64,
}

/// GET /v1/stats 的响应体（粗粒度计数器）。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct StatsResponse {
    /// 进程已启动秒数。
    pub uptime_secs: u64,
    /// SQLite 数据库文件占用字节数。
    pub database_size_bytes: u64,
    /// 当前状态为 Active（可检索）的记忆条数。
    pub memory_active: u64,
    /// 当前状态为 Archived（已归档）的记忆条数。
    pub memory_archived: u64,
    /// 记忆总条数（active + archived + any other）。
    pub memory_total: u64,
    /// 图谱实体总数。
    pub entity_count: u64,
    /// 图谱关系总数。
    pub relation_count: u64,
    /// 去重后的标签总数。
    pub tag_count: u64,
}

// ---------- Error shape returned by the server ------------------------------

#[derive(Debug, Clone, Deserialize)]
struct ServerErrorBody {
    code: String,
    message: String,
    #[serde(default)]
    trace_id: Option<String>,
}

// ---------- Client ----------------------------------------------------------

/// yq-nova HTTP 服务的同步风格客户端。
///
/// 内部基于 `reqwest::Client` 的轻量异步封装，提供：
///
/// - JSON 请求/响应体自动序列化
/// - 服务端 `{code, message, trace_id?}` 错误自动映射到 [`NovaError`]
/// - 可选的 trace-id 传播（通过 `x-trace-id` header）
/// - 两套 builder 方法（[`remember_builder`](Self::remember_builder) /
///   [`recall_builder`](Self::recall_builder)），避免手写冗长的 DTO 字面量
///
/// # 构造
///
/// - 推荐 [`HttpClient::new`]：使用默认 30s 超时；支持环境变量
///   `YQ_NOVA_BASE_URL` 覆盖传入的 `base_url`（如果环境变量非空）。
/// - 需要自定义超时使用 [`HttpClient::with_timeout`]。
///
/// ```no_run
/// # use yq_nova_sdk::http_client::HttpClient;
/// # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let client = HttpClient::new("http://127.0.0.1:7999")?;
/// let h = client.health().await?;
/// println!("server version = {}", h.version);
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct HttpClient {
    client: ReqwestClient,
    base_url: String,
}

impl HttpClient {
    /// 使用默认 30 秒请求超时创建客户端。
    ///
    /// `base_url` 尾部多余的 `/` 会被自动剥除；空字符串将返回
    /// [`ErrorCode::Validation`] 错误。
    ///
    /// 若需要自定义超时或连接池参数，使用 [`with_timeout`](Self::with_timeout)。
    pub fn new(base_url: impl Into<String>) -> NovaResult<Self> {
        Self::with_timeout(base_url, Duration::from_secs(30))
    }

    /// 使用自定义请求超时创建客户端。
    ///
    /// 除超时时间外，行为与 [`new`](Self::new) 一致：
    /// - 自动去除 `base_url` 尾部斜杠；
    /// - `base_url` 为空返回验证错误；
    /// - 默认 `Content-Type: application/json` 与 `Accept: application/json`。
    pub fn with_timeout(base_url: impl Into<String>, timeout: Duration) -> NovaResult<Self> {
        let mut base = base_url.into();
        while base.ends_with('/') {
            base.pop();
        }
        if base.is_empty() {
            return Err(NovaError::validation("http client: base_url must not be empty"));
        }
        let mut headers = header::HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, header::HeaderValue::from_static("application/json"));
        headers.insert(header::ACCEPT, header::HeaderValue::from_static("application/json"));
        let client = ReqwestClient::builder()
            .default_headers(headers)
            .timeout(timeout)
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| NovaError::internal_with_ctx("build reqwest client", e))?;
        Ok(Self { client, base_url: base })
    }

    /// 返回当前配置的服务端 base_url（不含末尾斜杠）。
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    // --- low-level generic helpers ------------------------------------------

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> NovaResult<T> {
        let url = self.url(path);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| NovaError::internal_with_ctx(format!("GET {url}"), e))?;
        self.map_response(resp, &url).await
    }

    async fn post_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> NovaResult<T> {
        let url = self.url(path);
        let resp = self
            .client
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| NovaError::internal_with_ctx(format!("POST {url}"), e))?;
        self.map_response(resp, &url).await
    }

    async fn delete_json<T: DeserializeOwned>(&self, path: &str) -> NovaResult<T> {
        let url = self.url(path);
        let resp = self
            .client
            .delete(&url)
            .send()
            .await
            .map_err(|e| NovaError::internal_with_ctx(format!("DELETE {url}"), e))?;
        self.map_response(resp, &url).await
    }

    async fn map_response<T: DeserializeOwned>(&self, resp: Response, url: &str) -> NovaResult<T> {
        let status = resp.status();
        if status.is_success() {
            return resp
                .json::<T>()
                .await
                .map_err(|e| NovaError::internal_with_ctx(format!("decode {url}"), e));
        }
        // Non-2xx: try to decode the structured server error; fall back to
        // status-line if the body isn't JSON.
        let bytes = resp.bytes().await.unwrap_or_default();
        let server_err = serde_json::from_slice::<ServerErrorBody>(&bytes).ok();
        let code = match server_err.as_ref().map(|e| e.code.as_str()) {
            Some("validation") => ErrorCode::Validation,
            Some("not_found") => ErrorCode::NotFound,
            Some("conflict") => ErrorCode::Conflict,
            Some("forbidden") => ErrorCode::Forbidden,
            Some("config") => ErrorCode::Config,
            Some("storage") => ErrorCode::Storage,
            Some("embedding") => ErrorCode::Embedding,
            Some("graph") => ErrorCode::Graph,
            Some(_) | None => status_to_generic_code(status),
        };
        let message = if let Some(se) = &server_err {
            format!("{} (url={url})", se.message)
        } else {
            let text = String::from_utf8_lossy(&bytes).to_string();
            format!(
                "HTTP {} {} (url={url}): {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("error"),
                if text.is_empty() { "(empty body)".to_string() } else { text }
            )
        };
        // Use the typed constructors so source-anchoring is preserved.
        let mut err = match code {
            ErrorCode::Validation => NovaError::validation(message),
            ErrorCode::NotFound => NovaError::not_found(message),
            ErrorCode::Conflict => NovaError::conflict(message),
            ErrorCode::Forbidden => NovaError::validation(message),
            ErrorCode::Config => NovaError::config_msg(message),
            ErrorCode::Storage => NovaError::storage_msg(message),
            ErrorCode::Embedding => NovaError::embedding_msg(message),
            ErrorCode::Graph => NovaError::graph_msg(message),
            ErrorCode::Internal => NovaError::internal(message),
        };
        if let Some(se) = server_err {
            if let Some(tid) = se.trace_id {
                err = err.with_trace_id(tid);
            }
        }
        Err(err)
    }

    // --- meta endpoints ------------------------------------------------------

    /// 健康检查端点：返回服务状态、版本、Git SHA 与已启动秒数。
    ///
    /// 对 `GET /v1/health` 的薄封装；正常返回 `status = "ok"`。
    pub async fn health(&self) -> NovaResult<HealthResponse> {
        self.get_json("/v1/health").await
    }

    /// 运行统计端点：返回数据库大小、活跃/归档记忆数、实体/关系/标签计数。
    ///
    /// 对应 `GET /v1/stats`；用于快速观察实例资源占用与规模。
    pub async fn stats(&self) -> NovaResult<StatsResponse> {
        self.get_json("/v1/stats").await
    }

    // --- memory endpoints ----------------------------------------------------

    /// 写入一条记忆（文本 + 元信息 + 标签）。
    ///
    /// # 参数
    /// - `req`：记忆内容、重要性、来源、过期时间、是否向量化、是否抽图谱等。
    ///
    /// # 返回
    /// - `RememberOutput { uuid, content_hash, embedded, entities_extracted }`
    ///   —— 新记忆的 UUID 等写入信息。
    ///
    /// 大多数调用场景建议使用 [`remember_builder`](Self::remember_builder)
    /// 链式 API，避免手写冗长的 DTO。
    pub async fn remember(&self, req: RememberRequest) -> NovaResult<RememberOutput> {
        self.post_json("/v1/memory/remember", &req).await
    }

    /// 按查询文本检索记忆（语义 / 关键词 / 混合 / 图谱增强）。
    ///
    /// # 参数
    /// - `req`：查询语句、top_k、阈值、检索模式、图谱扩展、重排权重、过滤条件等。
    ///
    /// # 返回
    /// - `RecallOutput { hits: Vec<RecallHit>, total_candidates, ... }`
    ///   每条 hit 包含匹配记忆本身、相似度得分与重排后综合得分。
    ///
    /// 推荐优先使用 [`recall_builder`](Self::recall_builder)。
    pub async fn recall(&self, req: RecallRequest) -> NovaResult<RecallOutput> {
        self.post_json("/v1/memory/recall", &req).await
    }

    /// 主动遗忘 / 归档记忆（按 UUID、按过滤条件、或按时间批量）。
    ///
    /// # 参数
    /// - `req`：遗忘目标（单条 UUID / 过滤条件）、模式（Archive vs Delete）、
    ///   是否级联清理关联的孤点关系、批次上限等。
    ///
    /// # 返回
    /// - `ForgetOutput { affected_memories, affected_relations, mode }`。
    pub async fn forget(&self, req: ForgetInput) -> NovaResult<ForgetOutput> {
        self.post_json("/v1/memory/forget", &req).await
    }

    /// 按 UUID 获取单条记忆的完整记录（含内容、向量、标签、访问计数等）。
    ///
    /// 若 UUID 不存在，返回 [`ErrorCode::NotFound`]。
    pub async fn get_memory(&self, uuid: Uuid) -> NovaResult<MemoryRecord> {
        self.get_json(&format!("/v1/memory/{uuid}")).await
    }

    /// 按 UUID 物理删除单条记忆（相比 [`forget`](Self::forget) 的 Archive
    /// 模式，此操作是硬删除，不可恢复）。
    ///
    /// 返回 `ForgetOutput` 以描述实际受影响的行数。
    pub async fn delete_memory(&self, uuid: Uuid) -> NovaResult<ForgetOutput> {
        self.delete_json(&format!("/v1/memory/{uuid}")).await
    }

    // --- builders ------------------------------------------------------------

    /// 构造 remember 请求的 ergonomic builder：
    ///
    /// ```no_run
    /// # use yq_nova_sdk::http_client::HttpClient;
    /// # async fn demo(c: &HttpClient) -> Result<(), Box<dyn std::error::Error>> {
    /// let out = c.remember_builder()
    ///     .content("Rust: impl Deref for MyBox")
    ///     .importance(0.9)
    ///     .tag("rust")
    ///     .send().await?;
    /// # Ok(()) }
    /// ```
    pub fn remember_builder(&self) -> RememberReqBuilder<'_> {
        RememberReqBuilder { client: self, req: RememberRequest::default() }
    }

    /// 构造 recall 请求的 ergonomic builder：
    ///
    /// ```no_run
    /// # use yq_nova_sdk::http_client::HttpClient;
    /// # async fn demo(c: &HttpClient) -> Result<(), Box<dyn std::error::Error>> {
    /// let hits = c.recall_builder()
    ///     .query("rust deref coercion")
    ///     .top_k(10)
    ///     .score_threshold(0.3)
    ///     .send().await?;
    /// # Ok(()) }
    /// ```
    pub fn recall_builder(&self) -> RecallReqBuilder<'_> {
        RecallReqBuilder { client: self, req: RecallRequest::default() }
    }

    // --- graph endpoints -----------------------------------------------------

    /// 创建或更新一个图谱实体（唯一键为 `(name, entity_type)` 组合）。
    ///
    /// # 参数
    /// - `req.name`：非空字符串，实体名称；
    /// - `req.entity_type`：实体分类；
    /// - `req.description` / `req.metadata`：None 表示不覆盖该字段（保留旧值）。
    ///
    /// # 返回
    /// - `UpsertEntityResponse { outcome, entity }`，其中 `outcome` 区分
    ///   [`UpsertOutcome::Created`] 与 [`UpsertOutcome::Updated`]。
    ///
    /// 若传入空 `name`，直接返回 [`ErrorCode::Validation`]（不发请求）。
    pub async fn upsert_entity(
        &self,
        req: UpsertEntityRequest,
    ) -> NovaResult<UpsertEntityResponse> {
        if req.name.trim().is_empty() {
            return Err(NovaError::validation("upsert_entity: name must be non-empty"));
        }
        self.post_json("/v1/graph/entities", &req).await
    }

    /// 列出实体（支持按名称前缀 / 实体类型过滤 + 分页）。
    ///
    /// # 参数
    /// - `name_prefix`：按 `name LIKE "prefix%"` 模糊匹配，None / 空串不过滤；
    /// - `entity_type`：按类型精确匹配，None / 空串不过滤；
    /// - `limit` / `offset`：SQL 风格分页。
    ///
    /// # 返回
    /// - 匹配的 [`EntityRecord`] 列表（按 `name` 字典序升序）。
    pub async fn list_entities(
        &self,
        name_prefix: Option<&str>,
        entity_type: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<EntityRecord>> {
        let mut url = format!("/v1/graph/entities?limit={limit}&offset={offset}");
        if let Some(p) = name_prefix {
            if !p.is_empty() {
                url.push_str("&name_prefix=");
                url.push_str(&urlencoding(p));
            }
        }
        if let Some(t) = entity_type {
            if !t.is_empty() {
                url.push_str("&entity_type=");
                url.push_str(&urlencoding(t));
            }
        }
        self.get_json(&url).await
    }

    /// 创建或更新一条有向关系边 `source —[predicate]→ target`。
    ///
    /// # 参数
    /// - `req.source_uuid` / `req.target_uuid`：两端实体 UUID，缺一不可；
    /// - `req.predicate`：关系类型，非空；
    /// - `req.confidence`：0.0 ~ 1.0；
    /// - `req.idempotent`：true 时若 `(source, predicate, target)` 已存在则跳过。
    ///
    /// # 返回
    /// - `UpsertRelationResponse { inserted, updated, relation_uuid, relation }`。
    ///
    /// 入参非法时（空 UUID / 空 predicate）直接在客户端返回
    /// [`ErrorCode::Validation`]。
    pub async fn upsert_relation(
        &self,
        req: UpsertRelationRequest,
    ) -> NovaResult<UpsertRelationResponse> {
        if req.source_uuid.is_nil() || req.target_uuid.is_nil() {
            return Err(NovaError::validation("upsert_relation: source_uuid/target_uuid required"));
        }
        if req.predicate.trim().is_empty() {
            return Err(NovaError::validation("upsert_relation: predicate required"));
        }
        self.post_json("/v1/graph/relations", &req).await
    }

    /// 列出关系边（按 source / target / predicate 过滤 + 分页）。
    ///
    /// # 参数
    /// - `source`：仅列出从某实体出发的边；
    /// - `target`：仅列出指向某实体的边；
    /// - `predicate`：按谓词精确匹配；
    /// - `limit` / `offset`：分页参数。
    ///
    /// # 返回
    /// - 匹配的 [`RelationRecord`] 列表。
    pub async fn list_relations(
        &self,
        source: Option<Uuid>,
        target: Option<Uuid>,
        predicate: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> NovaResult<Vec<RelationRecord>> {
        let mut url = format!("/v1/graph/relations?limit={limit}&offset={offset}");
        if let Some(s) = source {
            url.push_str("&source=");
            url.push_str(&s.to_string());
        }
        if let Some(t) = target {
            url.push_str("&target=");
            url.push_str(&t.to_string());
        }
        if let Some(p) = predicate {
            if !p.is_empty() {
                url.push_str("&predicate=");
                url.push_str(&urlencoding(p));
            }
        }
        self.get_json(&url).await
    }

    /// 从指定起点做 BFS 图谱遍历，返回可达实体及各节点深度与路径。
    ///
    /// # 参数
    /// - `req.start`：起点实体 UUID（客户端非空校验）；
    /// - `req.max_depth`：最大跳数（建议 1~5，过深可能触发 `max_nodes` 截断）；
    /// - `req.max_nodes`：最大返回节点数，避免大图爆炸；
    /// - `req.predicate_whitelist`：只走白名单里的谓词；空列表表示不限制；
    /// - `req.min_confidence`：忽略置信度低于该阈值的边。
    ///
    /// # 返回
    /// - `Vec<TraverseNode>`，每个元素含实体记录、深度、从起点到该节点的
    ///   UUID 路径（含起点与终点）。
    pub async fn traverse(&self, req: TraverseRequest) -> NovaResult<Vec<TraverseNode>> {
        if req.start.is_nil() {
            return Err(NovaError::validation("traverse: start uuid required"));
        }
        self.post_json("/v1/graph/traverse", &req).await
    }

    /// 从自由文本中抽取实体候选并（可选）自动 upsert 到图谱 / 建立关系。
    ///
    /// # 参数
    /// - `req.text`：待分析文本；空文本直接返回空结果（不发请求）；
    /// - `req.opts`：是否启用抽取、是否 upsert 实体、是否创建关系、
    ///   以及候选最小置信度。
    ///
    /// # 返回
    /// - [`LinkResult`]：包含 `(EntityCandidate, Uuid)` 对列表、实际 upsert 的实体数、
    ///   实际创建的关系数、以及文本中识别出的标签。
    pub async fn extract_and_link(&self, req: ExtractAndLinkRequest) -> NovaResult<LinkResult> {
        if req.text.trim().is_empty() {
            return Ok(LinkResult::default());
        }
        self.post_json("/v1/graph/extract-and-link", &req).await
    }
}

fn status_to_generic_code(status: StatusCode) -> ErrorCode {
    match status {
        StatusCode::BAD_REQUEST => ErrorCode::Validation,
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => ErrorCode::Forbidden,
        StatusCode::NOT_FOUND => ErrorCode::NotFound,
        StatusCode::CONFLICT => ErrorCode::Conflict,
        StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS => ErrorCode::Internal,
        StatusCode::BAD_GATEWAY | StatusCode::GATEWAY_TIMEOUT => ErrorCode::Embedding,
        s if s.is_server_error() => ErrorCode::Storage,
        _ => ErrorCode::Internal,
    }
}

fn urlencoding(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '0'..='9' | 'a'..='z' | 'A'..='Z' | '-' | '_' | '.' | '~' => out.push(c),
            _ => {
                for byte in c.to_string().as_bytes() {
                    out.push_str(&format!("%{byte:02X}"));
                }
            },
        }
    }
    out
}

// ---------- Builder helpers -------------------------------------------------

/// 围绕 [`RememberRequest`] 的链式构造器。
///
/// 典型用法：
///
/// ```no_run
/// # use yq_nova_sdk::http_client::HttpClient;
/// # async fn demo(c: &HttpClient) -> Result<(), Box<dyn std::error::Error>> {
/// c.remember_builder()
///     .content("Rust memory layout notes")
///     .importance(0.85)
///     .tag("rust")
///     .tag("memory")
///     .embed(true)
///     .extract_graph(true)
///     .send().await?;
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct RememberReqBuilder<'a> {
    client: &'a HttpClient,
    req: RememberRequest,
}

impl<'a> RememberReqBuilder<'a> {
    /// 设置要记住的原始文本内容（必填，send 前会做非空校验）。
    pub fn content(mut self, s: impl Into<String>) -> Self {
        self.req.content = s.into();
        self
    }
    /// 设置记忆来源渠道（Agent / User / System 等）。
    pub fn source(mut self, s: MemorySource) -> Self {
        self.req.source = s;
        self
    }
    /// 设置重要性评分 0.0 ~ 1.0；超出范围会被自动 clamp。
    pub fn importance(mut self, v: f32) -> Self {
        self.req.importance = v.clamp(0.0, 1.0);
        self
    }
    /// 设置过期时间（UTC）；None 表示永不过期。
    pub fn expires_at(mut self, t: DateTime<Utc>) -> Self {
        self.req.expires_at = Some(t);
        self
    }
    /// 批量覆盖标签列表。
    pub fn tags(mut self, t: impl IntoIterator<Item = String>) -> Self {
        self.req.tags = t.into_iter().collect();
        self
    }
    /// 追加单个标签（可多次调用）。
    pub fn tag(mut self, t: impl Into<String>) -> Self {
        self.req.tags.push(t.into());
        self
    }
    /// 设置结构化元数据；会通过 serde_json 转成 Value，转换失败返回
    /// [`ErrorCode::Validation`]。
    pub fn metadata(mut self, v: impl Serialize) -> NovaResult<Self> {
        self.req.metadata = Some(
            serde_json::to_value(v).map_err(|e| NovaError::validation(format!("metadata: {e}")))?,
        );
        Ok(self)
    }
    /// 是否为 content 生成向量嵌入以支持语义检索（默认 true）。
    pub fn embed(mut self, v: bool) -> Self {
        self.req.embed = v;
        self
    }
    /// 是否从 content 中自动抽取实体 / 关系并写入知识图谱（默认 false）。
    pub fn extract_graph(mut self, v: bool) -> Self {
        self.req.extract_graph = v;
        self
    }
    /// 发送 remember 请求。
    ///
    /// - 若 `content` 为空，直接返回客户端侧
    ///   [`ErrorCode::Validation`]，不发请求。
    /// - 成功返回 `RememberOutput { uuid, ... }`。
    pub async fn send(self) -> NovaResult<RememberOutput> {
        if self.req.content.trim().is_empty() {
            return Err(NovaError::validation("remember: content must be non-empty"));
        }
        self.client.remember(self.req).await
    }
}

/// 围绕 [`RecallRequest`] 的链式构造器。
///
/// 典型用法：
///
/// ```no_run
/// # use yq_nova_sdk::http_client::HttpClient;
/// # use yq_nova_core::memory::SearchMode;
/// # async fn demo(c: &HttpClient) -> Result<(), Box<dyn std::error::Error>> {
/// let out = c.recall_builder()
///     .query("rust memory layout deref")
///     .top_k(8)
///     .score_threshold(0.25)
///     .mode(SearchMode::Hybrid)
///     .graph_enable(2)
///     .send().await?;
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct RecallReqBuilder<'a> {
    client: &'a HttpClient,
    req: RecallRequest,
}

impl<'a> RecallReqBuilder<'a> {
    /// 设置查询文本（必填，用于语义 / 关键词匹配）。
    pub fn query(mut self, q: impl Into<String>) -> Self {
        self.req.query = q.into();
        self
    }
    /// 设置返回结果的最大条数（至少 1，默认 10）。
    pub fn top_k(mut self, k: usize) -> Self {
        self.req.top_k = k;
        self
    }
    /// 设置综合得分阈值（最终重排后得分低于该值的结果会被过滤）。
    pub fn score_threshold(mut self, v: f32) -> Self {
        self.req.score_threshold = v;
        self
    }
    /// 设置纯向量相似度阈值（仅对 Hybrid / Semantic 模式生效）。
    pub fn similarity_threshold(mut self, v: f32) -> Self {
        self.req.similarity_threshold = v;
        self
    }
    /// 设置检索模式：Keyword / Semantic / Hybrid / Graph 等。
    pub fn mode(mut self, m: SearchMode) -> Self {
        self.req.mode = m;
        self
    }
    /// 直接传入完整的图谱遍历扩展配置。
    pub fn graph(mut self, g: GraphTraversalOpts) -> Self {
        self.req.graph = g;
        self
    }
    /// 快捷开关：启用图谱扩展并指定 BFS 最大深度（默认谓词白名单为空）。
    pub fn graph_enable(mut self, max_depth: u8) -> Self {
        self.req.graph =
            GraphTraversalOpts { enabled: true, max_depth, predicate_whitelist: vec![] };
        self
    }
    /// 设置 Hybrid 模式下关键词 vs 语义分的权重。
    pub fn hybrid_weights(mut self, w: HybridWeights) -> Self {
        self.req.hybrid_weights = Some(w);
        self
    }
    /// 设置 RRF 融合算法的 k 参数（通常 20~100）。
    pub fn rrf_k(mut self, k: u32) -> Self {
        self.req.rrf_k = Some(k);
        self
    }
    /// 设置重排阶段各维度（相似度 / 重要性 / 新鲜度）的权重。
    pub fn rank_weights(mut self, w: RankWeights) -> Self {
        self.req.rank_weights = Some(w);
        self
    }
    /// 设置记忆过滤条件（按状态、标签、重要性范围、时间范围等）。
    pub fn filter(mut self, f: MemoryFilter) -> Self {
        self.req.filter = f;
        self
    }
    /// 发送 recall 请求。
    ///
    /// - 若 `query` 为空或 `top_k == 0`，返回客户端侧
    ///   [`ErrorCode::Validation`]。
    /// - 成功返回 `RecallOutput { hits, total_candidates, ... }`。
    pub async fn send(self) -> NovaResult<RecallOutput> {
        if self.req.query.trim().is_empty() {
            return Err(NovaError::validation("recall: query must be non-empty"));
        }
        if self.req.top_k == 0 {
            return Err(NovaError::validation("recall: top_k must be >= 1"));
        }
        self.client.recall(self.req).await
    }
}

// ---------- Integration tests against the axum server directly -------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use yq_nova_core::{
        Uuid,
        config::{ServerConfig, StorageConfig},
        embedding::MockEmbeddingProvider,
        graph::extractor::RegexWikiExtractor,
        graph::{GraphExtractOpts, GraphService},
        memory::{ForgetMode, MemoryService, ops_forget},
        storage::{Database, MemoryStatus},
    };

    fn tmp_db(tag: &str) -> StorageConfig {
        use std::time::{SystemTime, UNIX_EPOCH};
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        let mut p = std::env::temp_dir();
        p.push(format!("yqnova-sdk-{tag}-{}.db", n ^ 0x9e3779b97f4a7c15));
        StorageConfig {
            db_path: p,
            wal_mode: true,
            page_size: 4096,
            cache_size_kb: 32_000,
            busy_timeout_ms: 5000,
            pool_max_connections: 4,
            pool_min_connections: 0,
            ..Default::default()
        }
    }

    async fn spawn_server(tag: &str) -> HttpClient {
        let db = Database::open(tmp_db(tag)).await.expect("open db");
        let provider = Arc::new(MockEmbeddingProvider::new(64));
        let memory = MemoryService::new(db.clone(), provider);
        let graph = GraphService::with_parts(db.clone(), Arc::new(RegexWikiExtractor::new()));

        let state = yq_nova_server::http::AppState::new(ServerConfig::default(), db, memory, graph);
        let router = yq_nova_server::http::build_router(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        let client = HttpClient::new(format!("http://{addr}")).expect("client");
        // wait up to 1s for bind.
        for _ in 0..20 {
            if client.health().await.is_ok() {
                return client;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("server never became reachable on {addr}");
    }

    #[tokio::test]
    async fn health_returns_version_and_ok_status() {
        let client = spawn_server("health").await;
        let h = client.health().await.unwrap();
        assert_eq!(h.status, "ok");
        assert_eq!(h.version, yq_nova_core::VERSION);
    }

    #[tokio::test]
    async fn remember_and_recall_end_to_end() {
        let client = spawn_server("rr").await;

        let uuid = client
            .remember_builder()
            .content("Rust memory: impl Deref for MyBox via Box-like layout")
            .importance(0.9)
            .tag("rust")
            .tag("memory")
            .send()
            .await
            .unwrap()
            .uuid;

        let got = client.get_memory(uuid).await.unwrap();
        assert_eq!(got.tags, vec!["memory".to_string(), "rust".to_string()]);
        assert!((got.importance - 0.9).abs() < 1e-4);

        let recall = client.recall_builder().query("rust deref").top_k(5).send().await.unwrap();
        assert!(!recall.hits.is_empty(), "recall should hit the single memory");
        assert_eq!(recall.hits[0].memory.uuid, uuid);
    }

    #[tokio::test]
    async fn not_found_returns_validation_free_error_with_right_code() {
        let client = spawn_server("nf").await;
        let err = client.get_memory(Uuid::new_v4()).await.unwrap_err();
        assert_eq!(err.code(), ErrorCode::NotFound);
    }

    #[tokio::test]
    async fn forget_then_stats_reflect_active_and_archived() {
        let client = spawn_server("forget").await;

        for i in 0..5 {
            client
                .remember_builder()
                .content(format!("note-{i}: some low priority note"))
                .importance(0.1)
                .send()
                .await
                .unwrap();
        }
        let stats_before = client.stats().await.unwrap();
        assert_eq!(stats_before.memory_active, 5);

        let f = MemoryFilter {
            status_in: Some(vec![MemoryStatus::Active]),
            access_count_lt: Some(2),
            importance_max: Some(0.2),
            ..Default::default()
        };
        let forgotten = client
            .forget(ops_forget::ForgetInput {
                target: ops_forget::ForgetTarget::Filter(f),
                mode: ForgetMode::Archive,
                gc_graph: false,
                batch_limit: 3,
            })
            .await
            .unwrap();
        assert_eq!(forgotten.affected_memories, 3);

        let stats_after = client.stats().await.unwrap();
        assert_eq!(stats_after.memory_active, 2);
        assert_eq!(stats_after.memory_archived, 3);
    }

    #[tokio::test]
    async fn validation_error_from_server_propagates_message_and_code() {
        let client = spawn_server("val").await;
        let err = client.remember_builder().content("").send().await.unwrap_err();
        assert_eq!(err.code(), ErrorCode::Validation);
        let msg = format!("{err}");
        assert!(
            msg.contains("content") || msg.contains("empty"),
            "expected content/empty mention in: {msg}"
        );
    }

    #[tokio::test]
    async fn graph_upsert_list_traverse_and_extract_work() {
        let client = spawn_server("graph").await;

        let alice = client
            .upsert_entity(UpsertEntityRequest {
                name: "Alice".into(),
                entity_type: "person".into(),
                description: Some("Alice, the Rust engineer".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        let bob = client
            .upsert_entity(UpsertEntityRequest {
                name: "Bob".into(),
                entity_type: "person".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(matches!(alice.outcome, UpsertOutcome::Created(_)));
        assert!(matches!(bob.outcome, UpsertOutcome::Created(_)));

        // --- upsert_relation ---
        let rel = client
            .upsert_relation(UpsertRelationRequest {
                source_uuid: alice.outcome.uuid(),
                target_uuid: bob.outcome.uuid(),
                predicate: "reports_to".into(),
                confidence: 0.9,
                idempotent: true,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(rel.inserted);
        assert_eq!(rel.relation.predicate, "reports_to");

        // --- list_entities by prefix ---
        let by_prefix = client.list_entities(Some("A"), None, 10, 0).await.unwrap();
        assert_eq!(by_prefix.len(), 1);
        assert_eq!(by_prefix[0].name, "Alice");

        // --- list_relations with predicate filter ---
        let rels = client.list_relations(None, None, Some("reports_to"), 10, 0).await.unwrap();
        assert_eq!(rels.len(), 1);

        // --- BFS traverse from Alice ---
        let nodes = client
            .traverse(TraverseRequest {
                start: alice.outcome.uuid(),
                max_depth: 2,
                max_nodes: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        // Alice + Bob = 2 nodes
        assert_eq!(nodes.len(), 2);
        assert!(nodes.iter().any(|n| n.entity.name == "Alice"));
        assert!(nodes.iter().any(|n| n.entity.name == "Bob"));

        // --- extract_and_link: wikilinks must fire; enable opts explicitly ---
        let opts = GraphExtractOpts {
            enabled: true,
            upsert_entities: true,
            create_relations: false,
            min_confidence: 0.2,
        };
        let out = client
            .extract_and_link(ExtractAndLinkRequest {
                text: "I love [[Rust]] and [[Tokio]] async runtime".into(),
                opts,
            })
            .await
            .unwrap();
        assert_eq!(out.entities_upserted, 2);
        assert!(out.entities.iter().any(|(e, _)| e.name == "Rust"));
        assert!(out.entities.iter().any(|(e, _)| e.name == "Tokio"));
    }
}
