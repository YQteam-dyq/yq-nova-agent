//! OpenAI-compatible `/v1/embeddings` HTTP embedding provider.
//!
//! Works with any upstream that speaks the same contract:
//!   - OpenAI official (`https://api.openai.com/v1`)
//!   - Azure OpenAI (pass a custom base_url that includes the deployment path)
//!   - Local Ollama `/v1` proxy
//!   - vLLM / LM Studio / text-embeddings-inference (when running in
//!     OpenAI-compat mode)
//!
//! The provider:
//!   * Splits `embed_batch` inputs into sub-batches of at most `batch_size`
//!     (OpenAI has a hard 2048-inputs-per-request cap; many self-hosted
//!     proxies are lower).
//!   * Retries transient failures (429 / 5xx) via [`super::retry`].
//!   * Validates returned vector dims match [`EmbeddingMeta::dims`] so a
//!     misconfigured provider can't silently poison the vector store.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use super::{
    EmbeddingMeta, EmbeddingProvider,
    retry::{RetryAction, RetryConfig, classify_http_status, with_retry},
};
use crate::error::NovaResult;

/// Config for [`OpenAiCompatProvider`]. Kept `Serialize/Deserialize` so
/// callers can round-trip it through `Config` TOML files.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenAiCompatConfig {
    /// Base URL without the trailing `/embeddings`. Defaults to the public
    /// OpenAI endpoint; point at `http://localhost:11434/v1` for Ollama, etc.
    pub base_url: String,
    /// `Authorization: Bearer <api_key>`. May be empty for local proxies that
    /// don't require auth.
    #[serde(default)]
    pub api_key: String,
    /// Model name, e.g. `text-embedding-3-small`. Passed verbatim to the
    /// upstream in the request body.
    pub model: String,
    /// Expected output dimensionality. Used both for the EmbeddingMeta tag
    /// stored alongside vectors AND for response validation.
    pub dims: usize,
    /// Max number of `input` entries per HTTP call. Upstreams vary wildly;
    /// OpenAI caps at 2048 but self-hosted proxies often default to 32-256.
    pub batch_size: usize,
    /// Per-request timeout (applied inside every retry attempt).
    #[serde(with = "crate::config::duration_seconds")]
    pub request_timeout: Duration,
    /// Retry policy (429/5xx) before propagating an error.
    #[serde(flatten)]
    pub retry: RetryConfig,
}

impl Default for OpenAiCompatConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            model: "text-embedding-3-small".to_string(),
            dims: 1536,
            batch_size: 128,
            request_timeout: Duration::from_secs(15),
            retry: RetryConfig::default(),
        }
    }
}

// ---- request / response DTOs ------------------------------------------------

#[derive(Debug, Serialize)]
struct EmbeddingReqBody<'a> {
    model: &'a str,
    input: &'a [&'a str],
    encoding_format: &'static str,
}

#[serde_as]
#[derive(Debug, Deserialize)]
struct EmbeddingRespDataItem {
    #[serde_as(as = "serde_with::VecSkipError<_>")]
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResp {
    data: Vec<EmbeddingRespDataItem>,
}

// ---- provider impl ---------------------------------------------------------

/// OpenAI 兼容 `/v1/embeddings` HTTP Embedding 提供者。
///
/// 支持任意兼容该接口的上游：OpenAI 官方、Azure OpenAI、Ollama（/v1 代理）、
/// vLLM、LM Studio、text-embeddings-inference 等。
///
/// 核心特性：
/// - 将 `embed_batch` 按 `batch_size` 拆分子批次
/// - 对 429/5xx 等临时性错误通过 `retry` 模块重试
/// - 校验返回向量维度与配置一致，避免毒化向量库
pub struct OpenAiCompatProvider {
    client: reqwest::Client,
    meta: EmbeddingMeta,
    config: OpenAiCompatConfig,
    endpoint: String,
}

impl std::fmt::Debug for OpenAiCompatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatProvider")
            .field("meta", &self.meta)
            .field("base_url", &self.config.base_url)
            .field("model", &self.config.model)
            .field("batch_size", &self.config.batch_size)
            .finish_non_exhaustive()
    }
}

impl OpenAiCompatProvider {
    /// 使用指定配置构造 OpenAI 兼容 Embedding 提供者。
    ///
    /// 核心入口：调用方通常先从配置层拿到 `OpenAiCompatConfig`，再调用本函数
    /// 构造实例，最后 `Arc::new(provider)` 作为 `SharedEmbeddingProvider` 注入
    /// 到 `MemoryService` / `EmbeddingRegistry`。
    ///
    /// 校验项：`model` 非空、`dims > 0`、`batch_size > 0`；
    /// 同时会构建带超时与 User-Agent 的 reqwest 客户端。
    pub fn new(config: OpenAiCompatConfig) -> NovaResult<Self> {
        if config.model.trim().is_empty() {
            return Err(crate::error::NovaError::validation(
                "openai_compat: model must not be empty",
            ));
        }
        if config.dims == 0 {
            return Err(crate::error::NovaError::validation("openai_compat: dims must be > 0"));
        }
        if config.batch_size == 0 {
            return Err(crate::error::NovaError::validation(
                "openai_compat: batch_size must be > 0",
            ));
        }

        let base = config.base_url.trim_end_matches('/').to_string();
        let endpoint = format!("{base}/embeddings");

        let client = reqwest::Client::builder()
            .user_agent(concat!("yq-nova-agent/", env!("CARGO_PKG_VERSION")))
            .timeout(config.request_timeout)
            .build()
            .map_err(|e| crate::error::NovaError::embedding_msg(format!("build client: {e}")))?;

        let meta = EmbeddingMeta {
            provider: "openai_compat".into(),
            model: config.model.clone(),
            dims: config.dims,
        };
        Ok(Self { client, meta, config, endpoint })
    }

    /// Run a single sub-batch. Used by the batching loop in `embed_batch`.
    /// Public (crate) only for tests.
    async fn run_once(&self, texts: &[&str]) -> NovaResult<Vec<Vec<f32>>> {
        let body =
            EmbeddingReqBody { model: &self.config.model, input: texts, encoding_format: "float" };

        let resp = with_retry(&self.config.retry, |_attempt| async {
            let mut req = self.client.post(self.endpoint.as_str()).json(&body);
            if !self.config.api_key.is_empty() {
                req = req.bearer_auth(&self.config.api_key);
            }
            let res = req.send().await.map_err(|e| {
                let action = if e.is_timeout() || e.is_connect() || e.is_request() {
                    RetryAction::Retry
                } else {
                    RetryAction::Fail
                };
                (action, anyhow::anyhow!("{e}"))
            })?;

            let status = res.status();
            if !status.is_success() {
                let text = res.text().await.unwrap_or_default();
                let snippet = if text.len() > 400 { &text[..400] } else { text.as_str() };
                return Err((
                    classify_http_status(status),
                    anyhow::anyhow!("HTTP {}: {}", status.as_u16(), snippet),
                ));
            }

            let parsed: EmbeddingResp = res
                .json()
                .await
                .map_err(|e| (RetryAction::Fail, anyhow::anyhow!("parse response: {e}")))?;
            Ok(parsed)
        })
        .await?;

        if resp.data.len() != texts.len() {
            return Err(crate::error::NovaError::embedding_msg(format!(
                "upstream returned {} embeddings for {} inputs",
                resp.data.len(),
                texts.len()
            )));
        }

        let expected_dims = self.config.dims;
        let mut out = Vec::with_capacity(resp.data.len());
        for item in resp.data {
            if item.embedding.len() != expected_dims {
                return Err(crate::error::NovaError::embedding_msg(format!(
                    "upstream returned dims={} expected {expected_dims}",
                    item.embedding.len()
                )));
            }
            out.push(item.embedding);
        }
        Ok(out)
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiCompatProvider {
    fn meta(&self) -> &EmbeddingMeta {
        &self.meta
    }

    async fn embed_batch(&self, texts: &[&str]) -> NovaResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() <= self.config.batch_size {
            return self.run_once(texts).await;
        }
        // Split into chunks and run sequentially (keeps impl simple, avoids
        // overwhelming small upstreams). Callers that need parallelism can
        // pre-split; the retry layer still works per-chunk.
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.config.batch_size) {
            // Lifetime dance: `run_once` wants &[&str] but chunk is &&[&str].
            // Collect into a tiny vec of refs — negligible allocation because
            // each chunk is at most batch_size (~128).
            let refs: Vec<&str> = chunk.to_vec();
            let sub = self.run_once(&refs).await?;
            out.extend(sub);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_sane() {
        let c = OpenAiCompatConfig::default();
        assert_eq!(c.model, "text-embedding-3-small");
        assert_eq!(c.dims, 1536);
        assert_eq!(c.batch_size, 128);
        assert!(c.api_key.is_empty());
    }

    #[test]
    fn new_validates_required_fields() {
        let bad = OpenAiCompatConfig { model: "  ".into(), ..OpenAiCompatConfig::default() };
        assert!(matches!(
            OpenAiCompatProvider::new(bad).unwrap_err().code(),
            crate::error::ErrorCode::Validation
        ));

        let zero_dim = OpenAiCompatConfig { dims: 0, ..OpenAiCompatConfig::default() };
        assert!(matches!(
            OpenAiCompatProvider::new(zero_dim).unwrap_err().code(),
            crate::error::ErrorCode::Validation
        ));

        let zero_batch = OpenAiCompatConfig { batch_size: 0, ..OpenAiCompatConfig::default() };
        assert!(matches!(
            OpenAiCompatProvider::new(zero_batch).unwrap_err().code(),
            crate::error::ErrorCode::Validation
        ));
    }

    #[test]
    fn endpoint_strips_trailing_slash() {
        let cfg = OpenAiCompatConfig {
            base_url: "https://example.com/v1///".into(),
            model: "m".into(),
            dims: 8,
            batch_size: 4,
            ..OpenAiCompatConfig::default()
        };
        let p = OpenAiCompatProvider::new(cfg).unwrap();
        assert_eq!(p.endpoint, "https://example.com/v1/embeddings");
        assert_eq!(p.meta().provider, "openai_compat");
    }

    // NOTE: full end-to-end tests against a real / mocked server are not in
    // unit tests because they require either network access or a helper
    // like `mockito`. Integration tests in the `tests/` directory (M5)
    // validate the happy path using axum::test server.
}
