use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::{Client as ReqwestClient, Response, StatusCode, header};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use yq_nova_core::{
    Uuid,
    error::{ErrorCode, NovaError, NovaResult},
    graph::{GraphExtractOpts, LinkResult},
    memory::{
        ChunkOptions, ForgetInput, ForgetOutput, GraphTraversalOpts, HybridWeights, RankWeights,
        RecallOutput, RememberOutput, SearchMode,
        ops_batch::{BatchRememberInput, BatchRememberOutput},
        ops_list::{ListInput, ListOutput},
        ops_tag::{TagDeleteOutput, TagListOutput, TagRenameOutput},
    },
    storage::{
        EntityRecord, MemoryFilter, MemoryRecord, MemorySource, RelationRecord, TraverseNode,
        UpsertOutcome, namespace::NamespaceRecord,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RememberRequest {
    pub content: String,

    pub source: MemorySource,

    pub importance: f32,

    pub metadata: Option<serde_json::Value>,

    pub expires_at: Option<DateTime<Utc>>,

    pub tags: Vec<String>,

    pub embed: bool,

    pub extract_graph: bool,

    pub chunk_options: Option<ChunkOptions>,
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
            chunk_options: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RecallRequest {
    pub query: String,

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

    #[serde(default)]
    pub entity_focus: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UpsertEntityRequest {
    pub name: String,

    #[serde(rename = "type")]
    pub entity_type: String,

    pub description: Option<String>,

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertEntityResponse {
    #[serde(flatten)]
    pub outcome: UpsertOutcome,

    pub entity: EntityRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UpsertRelationRequest {
    pub source_uuid: Uuid,

    pub target_uuid: Uuid,

    pub predicate: String,

    pub confidence: f32,

    pub metadata: Option<serde_json::Value>,

    pub idempotent: bool,

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRelationResponse {
    pub inserted: bool,

    pub updated: bool,

    pub relation_uuid: Uuid,

    pub relation: RelationRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TraverseRequest {
    pub start: Uuid,

    pub max_depth: u8,

    pub max_nodes: usize,

    pub predicate_whitelist: Vec<String>,

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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ExtractAndLinkRequest {
    pub text: String,

    pub opts: GraphExtractOpts,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct UpdateMemoryRequest {
    pub content: Option<String>,
    pub importance: Option<f32>,
    pub metadata: Option<serde_json::Value>,
    pub tags: Option<Vec<String>>,
    pub expires_at: Option<Option<DateTime<Utc>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeMemoriesRequest {
    pub uuids: Vec<Uuid>,
    pub keep_uuid: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeMemoriesResponse {
    pub kept_uuid: Uuid,
    pub merged: Vec<Uuid>,
    pub remapped_relations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportMemoriesRequest {
    pub filter: Option<MemoryFilter>,
    pub limit: u32,
    pub offset: u32,
}

impl Default for ExportMemoriesRequest {
    fn default() -> Self {
        Self {
            filter: None,
            limit: 500,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMemoriesResponse {
    pub count: usize,
    pub offset: u32,
    pub items: Vec<MemoryRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImportItem {
    pub content: String,
    pub uuid: Option<Uuid>,
    pub metadata: Option<serde_json::Value>,
    pub importance: Option<f32>,
    pub source: Option<String>,
    pub tags: Option<Vec<String>>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImportMemoriesRequest {
    pub items: Vec<ImportItem>,
    pub embed: bool,
    pub on_conflict: String,
}

impl Default for ImportMemoriesRequest {
    fn default() -> Self {
        Self {
            items: vec![],
            embed: true,
            on_conflict: "skip".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportError {
    pub index: usize,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportMemoriesResponse {
    pub received: usize,
    pub imported: usize,
    pub duplicates: usize,
    pub errors: Vec<ImportError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEntitiesRequest {
    pub keep_uuid: Uuid,
    pub discard_uuids: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEntitiesResponse {
    pub kept_uuid: Uuid,
    pub merged: Vec<Uuid>,
    pub remapped_relations: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct NamespaceListQuery {
    pub limit: usize,
    pub offset: usize,
}

impl Default for NamespaceListQuery {
    fn default() -> Self {
        Self {
            limit: 100,
            offset: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceListOutput {
    pub total: i64,
    pub count: usize,
    pub limit: usize,
    pub offset: usize,
    pub items: Vec<NamespaceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceDeleteOutput {
    pub name: String,
    pub deleted: bool,
    pub protected: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateNamespaceRequest {
    pub description: Option<String>,
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,

    pub version: String,

    pub git_sha: String,

    pub uptime_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct StatsResponse {
    pub uptime_secs: u64,

    pub database_size_bytes: u64,

    pub memory_active: u64,

    pub memory_archived: u64,

    pub memory_total: u64,

    pub entity_count: u64,

    pub relation_count: u64,

    pub tag_count: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct ServerErrorBody {
    code: String,
    message: String,
    #[serde(default)]
    trace_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct HttpClient {
    client: ReqwestClient,
    base_url: String,
    namespace: Option<String>,
}

impl HttpClient {
    pub fn new(base_url: impl Into<String>) -> NovaResult<Self> {
        Self::with_timeout(base_url, Duration::from_secs(30))
    }

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
        Ok(Self {
            client,
            base_url: base,
            namespace: None,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn with_namespace(mut self, name: impl Into<String>) -> Self {
        self.namespace = Some(name.into());
        self
    }

    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> NovaResult<T> {
        let url = self.url(path);
        let resp = self
            .client
            .get(&url)
            .headers(self.scoped_headers())
            .send()
            .await
            .map_err(|e| NovaError::internal_with_ctx(format!("GET {url}"), e))?;
        self.map_response(resp, &url).await
    }

    fn scoped_headers(&self) -> header::HeaderMap {
        let mut h = header::HeaderMap::new();
        if let Some(ns) = &self.namespace {
            if !ns.is_empty() {
                if let Ok(v) = header::HeaderValue::from_str(ns) {
                    h.insert(header::HeaderName::from_static("x-namespace"), v);
                }
            }
        }
        h
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
            .headers(self.scoped_headers())
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
            .headers(self.scoped_headers())
            .send()
            .await
            .map_err(|e| NovaError::internal_with_ctx(format!("DELETE {url}"), e))?;
        self.map_response(resp, &url).await
    }

    async fn patch_json<B: Serialize, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> NovaResult<T> {
        let url = self.url(path);
        let resp = self
            .client
            .patch(&url)
            .headers(self.scoped_headers())
            .json(body)
            .send()
            .await
            .map_err(|e| NovaError::internal_with_ctx(format!("PATCH {url}"), e))?;
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

    pub async fn health(&self) -> NovaResult<HealthResponse> {
        self.get_json("/v1/health").await
    }

    pub async fn stats(&self) -> NovaResult<StatsResponse> {
        self.get_json("/v1/stats").await
    }

    pub async fn remember(&self, req: RememberRequest) -> NovaResult<RememberOutput> {
        self.post_json("/v1/memory/remember", &req).await
    }

    pub async fn recall(&self, req: RecallRequest) -> NovaResult<RecallOutput> {
        self.post_json("/v1/memory/recall", &req).await
    }

    pub async fn forget(&self, req: ForgetInput) -> NovaResult<ForgetOutput> {
        self.post_json("/v1/memory/forget", &req).await
    }

    pub async fn get_memory(&self, uuid: Uuid) -> NovaResult<MemoryRecord> {
        self.get_json(&format!("/v1/memory/{uuid}")).await
    }

    pub async fn delete_memory(&self, uuid: Uuid) -> NovaResult<ForgetOutput> {
        self.delete_json(&format!("/v1/memory/{uuid}")).await
    }

    pub async fn update_memory(
        &self,
        uuid: Uuid,
        req: UpdateMemoryRequest,
    ) -> NovaResult<MemoryRecord> {
        self.patch_json(&format!("/v1/memory/{uuid}"), &req).await
    }

    pub async fn merge_memories(
        &self,
        req: MergeMemoriesRequest,
    ) -> NovaResult<MergeMemoriesResponse> {
        self.post_json("/v1/memory/merge", &req).await
    }

    pub async fn export_memories(
        &self,
        req: ExportMemoriesRequest,
    ) -> NovaResult<ExportMemoriesResponse> {
        self.post_json("/v1/memory/export", &req).await
    }

    pub async fn import_memories(
        &self,
        req: ImportMemoriesRequest,
    ) -> NovaResult<ImportMemoriesResponse> {
        self.post_json("/v1/memory/import", &req).await
    }

    pub async fn list_memories(&self, req: ListInput) -> NovaResult<ListOutput> {
        self.post_json("/v1/memory/list", &req).await
    }

    pub async fn remember_batch(&self, req: BatchRememberInput) -> NovaResult<BatchRememberOutput> {
        self.post_json("/v1/memory/remember-batch", &req).await
    }

    pub async fn list_tags(&self, limit: usize, offset: usize) -> NovaResult<TagListOutput> {
        self.get_json(&format!("/v1/tags?limit={limit}&offset={offset}")).await
    }

    pub async fn rename_tag(&self, name: &str, new_name: &str) -> NovaResult<TagRenameOutput> {
        if name.trim().is_empty() {
            return Err(NovaError::validation("rename_tag: name must not be empty"));
        }
        let path = format!("/v1/tags/{}", urlencoding(name));
        let body = serde_json::json!({ "new_name": new_name });
        self.patch_json(&path, &body).await
    }

    pub async fn delete_tag(&self, name: &str) -> NovaResult<TagDeleteOutput> {
        if name.trim().is_empty() {
            return Err(NovaError::validation("delete_tag: name must not be empty"));
        }
        self.delete_json(&format!("/v1/tags/{}", urlencoding(name))).await
    }

    pub async fn merge_entities(
        &self,
        req: MergeEntitiesRequest,
    ) -> NovaResult<MergeEntitiesResponse> {
        self.post_json("/v1/graph/entities/merge", &req).await
    }

    pub fn remember_builder(&self) -> RememberReqBuilder<'_> {
        RememberReqBuilder {
            client: self,
            req: RememberRequest::default(),
        }
    }

    pub fn recall_builder(&self) -> RecallReqBuilder<'_> {
        RecallReqBuilder {
            client: self,
            req: RecallRequest::default(),
        }
    }

    pub async fn upsert_entity(
        &self,
        req: UpsertEntityRequest,
    ) -> NovaResult<UpsertEntityResponse> {
        if req.name.trim().is_empty() {
            return Err(NovaError::validation("upsert_entity: name must be non-empty"));
        }
        self.post_json("/v1/graph/entities", &req).await
    }

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

    pub async fn traverse(&self, req: TraverseRequest) -> NovaResult<Vec<TraverseNode>> {
        if req.start.is_nil() {
            return Err(NovaError::validation("traverse: start uuid required"));
        }
        self.post_json("/v1/graph/traverse", &req).await
    }

    pub async fn extract_and_link(&self, req: ExtractAndLinkRequest) -> NovaResult<LinkResult> {
        if req.text.trim().is_empty() {
            return Ok(LinkResult::default());
        }
        self.post_json("/v1/graph/extract-and-link", &req).await
    }

    pub async fn list_namespaces(
        &self,
        limit: usize,
        offset: usize,
    ) -> NovaResult<NamespaceListOutput> {
        let path = format!("/v1/namespaces?limit={limit}&offset={offset}");
        self.get_json(&path).await
    }

    pub async fn get_namespace(&self, name: &str) -> NovaResult<NamespaceRecord> {
        if name.trim().is_empty() {
            return Err(NovaError::validation("get_namespace: name must not be empty"));
        }
        self.get_json(&format!("/v1/namespaces/{}", urlencoding(name))).await
    }

    pub async fn create_namespace(
        &self,
        name: &str,
        description: Option<&str>,
        config: Option<serde_json::Value>,
    ) -> NovaResult<NamespaceRecord> {
        let name = name.trim();
        if name.is_empty() {
            return Err(NovaError::validation("create_namespace: name must not be empty"));
        }
        if name.eq_ignore_ascii_case("default") {
            return Err(NovaError::validation("create_namespace: 'default' is reserved"));
        }
        let body = serde_json::json!({
            "name": name,
            "description": description,
            "config": config.unwrap_or(serde_json::json!({})),
        });
        self.post_json("/v1/namespaces", &body).await
    }

    pub async fn update_namespace(
        &self,
        name: &str,
        description: Option<String>,
        config: Option<serde_json::Value>,
    ) -> NovaResult<NamespaceRecord> {
        if name.trim().is_empty() {
            return Err(NovaError::validation("update_namespace: name must not be empty"));
        }
        let body = UpdateNamespaceRequest {
            description,
            config,
        };
        self.patch_json(&format!("/v1/namespaces/{}", urlencoding(name)), &body).await
    }

    pub async fn delete_namespace(&self, name: &str) -> NovaResult<NamespaceDeleteOutput> {
        if name.trim().is_empty() {
            return Err(NovaError::validation("delete_namespace: name must not be empty"));
        }
        self.delete_json(&format!("/v1/namespaces/{}", urlencoding(name))).await
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

#[derive(Debug)]
pub struct RememberReqBuilder<'a> {
    client: &'a HttpClient,
    req: RememberRequest,
}

impl<'a> RememberReqBuilder<'a> {
    pub fn content(mut self, s: impl Into<String>) -> Self {
        self.req.content = s.into();
        self
    }

    pub fn source(mut self, s: MemorySource) -> Self {
        self.req.source = s;
        self
    }

    pub fn importance(mut self, v: f32) -> Self {
        self.req.importance = v.clamp(0.0, 1.0);
        self
    }

    pub fn expires_at(mut self, t: DateTime<Utc>) -> Self {
        self.req.expires_at = Some(t);
        self
    }

    pub fn tags(mut self, t: impl IntoIterator<Item = String>) -> Self {
        self.req.tags = t.into_iter().collect();
        self
    }

    pub fn tag(mut self, t: impl Into<String>) -> Self {
        self.req.tags.push(t.into());
        self
    }

    pub fn metadata(mut self, v: impl Serialize) -> NovaResult<Self> {
        self.req.metadata = Some(
            serde_json::to_value(v).map_err(|e| NovaError::validation(format!("metadata: {e}")))?,
        );
        Ok(self)
    }

    pub fn embed(mut self, v: bool) -> Self {
        self.req.embed = v;
        self
    }

    pub fn extract_graph(mut self, v: bool) -> Self {
        self.req.extract_graph = v;
        self
    }

    pub fn chunk_options(mut self, opts: ChunkOptions) -> Self {
        self.req.chunk_options = Some(opts);
        self
    }

    pub async fn send(self) -> NovaResult<RememberOutput> {
        if self.req.content.trim().is_empty() {
            return Err(NovaError::validation("remember: content must be non-empty"));
        }
        self.client.remember(self.req).await
    }
}

#[derive(Debug)]
pub struct RecallReqBuilder<'a> {
    client: &'a HttpClient,
    req: RecallRequest,
}

impl<'a> RecallReqBuilder<'a> {
    pub fn query(mut self, q: impl Into<String>) -> Self {
        self.req.query = q.into();
        self
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.req.top_k = k;
        self
    }

    pub fn score_threshold(mut self, v: f32) -> Self {
        self.req.score_threshold = v;
        self
    }

    pub fn similarity_threshold(mut self, v: f32) -> Self {
        self.req.similarity_threshold = v;
        self
    }

    pub fn mode(mut self, m: SearchMode) -> Self {
        self.req.mode = m;
        self
    }

    pub fn graph(mut self, g: GraphTraversalOpts) -> Self {
        self.req.graph = g;
        self
    }

    pub fn graph_enable(mut self, max_depth: u8) -> Self {
        self.req.graph = GraphTraversalOpts {
            enabled: true,
            max_depth,
            predicate_whitelist: vec![],
        };
        self
    }

    pub fn hybrid_weights(mut self, w: HybridWeights) -> Self {
        self.req.hybrid_weights = Some(w);
        self
    }

    pub fn rrf_k(mut self, k: u32) -> Self {
        self.req.rrf_k = Some(k);
        self
    }

    pub fn rank_weights(mut self, w: RankWeights) -> Self {
        self.req.rank_weights = Some(w);
        self
    }

    pub fn filter(mut self, f: MemoryFilter) -> Self {
        self.req.filter = f;
        self
    }

    pub fn group_chunks(mut self, v: bool) -> Self {
        self.req.group_chunks = v;
        self
    }

    pub fn entity_focus(mut self, v: Vec<String>) -> Self {
        self.req.entity_focus = v;
        self
    }

    pub fn metadata_match(mut self, v: serde_json::Value) -> Self {
        if let serde_json::Value::Object(map) = v {
            self.req.filter.metadata_match = Some(map.into_iter().collect());
        }
        self
    }

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use yq_nova_core::{
        Uuid,
        config::{ServerConfig, StorageConfig},
        embedding::MockEmbeddingProvider,
        graph::{GraphExtractOpts, GraphService, extractor::RegexWikiExtractor},
        memory::{ForgetMode, MemoryService, ops_forget},
        storage::{Database, MemoryStatus},
    };

    use super::*;

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
                namespace_id: yq_nova_core::storage::namespace::DEFAULT_NAMESPACE_ID,
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

        let by_prefix = client.list_entities(Some("A"), None, 10, 0).await.unwrap();
        assert_eq!(by_prefix.len(), 1);
        assert_eq!(by_prefix[0].name, "Alice");

        let rels = client.list_relations(None, None, Some("reports_to"), 10, 0).await.unwrap();
        assert_eq!(rels.len(), 1);

        let nodes = client
            .traverse(TraverseRequest {
                start: alice.outcome.uuid(),
                max_depth: 2,
                max_nodes: 10,
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(nodes.len(), 2);
        assert!(nodes.iter().any(|n| n.entity.name == "Alice"));
        assert!(nodes.iter().any(|n| n.entity.name == "Bob"));

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

    #[tokio::test]
    async fn namespace_crud_and_tenant_isolation() {
        let client = spawn_server("ns").await;

        let ns_a = client.create_namespace("nsA", Some("tenant A"), None).await.unwrap();
        let ns_b = client
            .create_namespace("nsB", Some("tenant B"), Some(serde_json::json!({ "quota": 100 })))
            .await
            .unwrap();
        assert_eq!(ns_a.name, "nsA");
        assert_eq!(ns_b.config["quota"], 100);

        let listed = client.list_namespaces(100, 0).await.unwrap();
        let names: std::collections::BTreeSet<String> =
            listed.items.iter().map(|n| n.name.clone()).collect();
        assert!(names.contains("default"));
        assert!(names.contains("nsA"));
        assert!(names.contains("nsB"));

        let got = client.get_namespace("nsA").await.unwrap();
        assert_eq!(got.description.as_deref(), Some("tenant A"));

        let c_a = client.clone().with_namespace("nsA");
        let c_b = client.clone().with_namespace("nsB");
        c_a.remember_builder().content("secret of A").send().await.unwrap();
        c_b.remember_builder().content("secret of B").send().await.unwrap();
        c_a.remember_builder().content("another A note").send().await.unwrap();

        let list_a = c_a.list_memories(Default::default()).await.unwrap();
        let list_b = c_b.list_memories(Default::default()).await.unwrap();
        assert_eq!(list_a.total, 2);
        assert_eq!(list_b.total, 1);

        let updated =
            client.update_namespace("nsA", Some("renamed desc A".into()), None).await.unwrap();
        assert_eq!(updated.description.as_deref(), Some("renamed desc A"));

        let del = client.delete_namespace("nsA").await.unwrap();
        assert!(del.deleted);
        let not_found = client.get_namespace("nsA").await.unwrap_err();
        assert_eq!(not_found.code(), ErrorCode::NotFound);
    }
}
