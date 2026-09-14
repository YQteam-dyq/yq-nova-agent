use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use super::{
    EmbeddingMeta, EmbeddingProvider,
    retry::{RetryAction, RetryConfig, classify_http_status, with_retry},
    truncate_utf8,
};
use crate::error::NovaResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct JinaConfig {
    pub base_url: String,

    pub api_key: String,

    pub model: String,

    pub dims: usize,

    pub task: String,

    pub batch_size: usize,

    #[serde(with = "crate::config::duration_seconds")]
    pub request_timeout: Duration,

    #[serde(flatten)]
    pub retry: RetryConfig,
}

impl Default for JinaConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.jina.ai/v1".to_string(),
            api_key: String::new(),
            model: "jina-embeddings-v3".to_string(),
            dims: 0,
            task: String::new(),
            batch_size: 64,
            request_timeout: Duration::from_secs(15),
            retry: RetryConfig::default(),
        }
    }
}

fn default_dims(model: &str) -> Option<usize> {
    let name = model.trim();
    if name.starts_with("jina-embeddings-v3") {
        Some(1024)
    } else if name.starts_with("jina-embeddings-v2") {
        Some(768)
    } else {
        None
    }
}

#[derive(Debug, Serialize)]
struct JinaReqBody<'a> {
    model: &'a str,

    input: &'a [&'a str],

    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,

    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<&'a str>,
}

#[serde_as]
#[derive(Debug, Deserialize)]
struct JinaRespItem {
    #[serde_as(as = "serde_with::VecSkipError<_>")]
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct JinaResp {
    data: Vec<JinaRespItem>,
}

pub struct JinaProvider {
    client: reqwest::Client,

    meta: EmbeddingMeta,

    config: JinaConfig,

    endpoint: String,
}

impl std::fmt::Debug for JinaProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JinaProvider")
            .field("meta", &self.meta)
            .field("base_url", &self.config.base_url)
            .field("model", &self.config.model)
            .field("task", &self.config.task)
            .field("batch_size", &self.config.batch_size)
            .finish_non_exhaustive()
    }
}

impl JinaProvider {
    pub fn new(config: JinaConfig) -> NovaResult<Self> {
        let model = config.model.trim().to_string();
        if model.is_empty() {
            return Err(crate::error::NovaError::validation("jina: model must not be empty"));
        }
        let dims = if config.dims > 0 {
            config.dims
        } else {
            default_dims(&model).ok_or_else(|| {
                crate::error::NovaError::validation(
                    "jina: dims must be set for unknown model, supported defaults are ".to_owned()
                        + "jina-embeddings-v2 (768) and jina-embeddings-v3 (1024)",
                )
            })?
        };
        if config.batch_size == 0 {
            return Err(crate::error::NovaError::validation("jina: batch_size must be > 0"));
        }

        let base = config.base_url.trim_end_matches('/').to_string();
        let endpoint = format!("{base}/embeddings");

        let client = reqwest::Client::builder()
            .user_agent(concat!("yq-nova-agent/", env!("CARGO_PKG_VERSION")))
            .timeout(config.request_timeout)
            .build()
            .map_err(|e| {
                crate::error::NovaError::embedding_msg(format!("jina: build client: {e}"))
            })?;

        let meta = EmbeddingMeta {
            provider: "jina".into(),
            model: model.clone(),
            dims,
        };
        Ok(Self {
            client,
            meta,
            config: JinaConfig {
                model,
                ..config
            },
            endpoint,
        })
    }

    async fn run_once(&self, texts: &[&str]) -> NovaResult<Vec<Vec<f32>>> {
        let task =
            if self.config.task.trim().is_empty() { None } else { Some(self.config.task.as_str()) };
        let body = JinaReqBody {
            model: &self.config.model,
            input: texts,
            dimensions: Some(self.meta.dims),
            task,
        };

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
                let snippet = truncate_utf8(&text, 400);
                return Err((
                    classify_http_status(status),
                    anyhow::anyhow!("HTTP {}: {}", status.as_u16(), snippet),
                ));
            }

            let parsed: JinaResp = res
                .json()
                .await
                .map_err(|e| (RetryAction::Fail, anyhow::anyhow!("jina: parse response: {e}")))?;
            Ok(parsed)
        })
        .await?;

        if resp.data.len() != texts.len() {
            return Err(crate::error::NovaError::embedding_msg(format!(
                "jina: upstream returned {} embeddings for {} inputs",
                resp.data.len(),
                texts.len()
            )));
        }

        let expected_dims = self.meta.dims;
        let mut out = Vec::with_capacity(resp.data.len());
        for item in resp.data {
            if item.embedding.len() != expected_dims {
                return Err(crate::error::NovaError::embedding_msg(format!(
                    "jina: upstream returned dims={} expected {expected_dims}",
                    item.embedding.len()
                )));
            }
            out.push(item.embedding);
        }
        Ok(out)
    }
}

#[async_trait]
impl EmbeddingProvider for JinaProvider {
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

        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.config.batch_size) {
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
        let c = JinaConfig::default();
        assert_eq!(c.model, "jina-embeddings-v3");
        assert_eq!(c.batch_size, 64);
        assert_eq!(c.dims, 0);
        assert!(c.task.is_empty());
        assert!(c.api_key.is_empty());
    }

    #[test]
    fn new_derives_dims_for_known_models() {
        let p = JinaProvider::new(JinaConfig {
            model: "jina-embeddings-v3".into(),
            ..JinaConfig::default()
        })
        .unwrap();
        assert_eq!(p.meta().dims, 1024);
        assert_eq!(p.meta().provider, "jina");

        let p = JinaProvider::new(JinaConfig {
            model: "jina-embeddings-v2-base-en".into(),
            ..JinaConfig::default()
        })
        .unwrap();
        assert_eq!(p.meta().dims, 768);
    }

    #[test]
    fn new_rejects_unknown_model_without_dims() {
        let r = JinaProvider::new(JinaConfig {
            model: "unknown".into(),
            ..JinaConfig::default()
        });
        assert!(matches!(r.unwrap_err().code(), crate::error::ErrorCode::Validation));
    }

    #[test]
    fn new_validates_required_fields() {
        let empty = JinaConfig {
            model: "  ".into(),
            ..JinaConfig::default()
        };
        assert!(JinaProvider::new(empty).is_err());

        let zero_batch = JinaConfig {
            batch_size: 0,
            ..JinaConfig::default()
        };
        assert!(JinaProvider::new(zero_batch).is_err());
    }

    #[test]
    fn endpoint_strips_trailing_slash() {
        let cfg = JinaConfig {
            base_url: "https://api.jina.ai/v1/".into(),
            ..JinaConfig::default()
        };
        let p = JinaProvider::new(cfg).unwrap();
        assert_eq!(p.endpoint, "https://api.jina.ai/v1/embeddings");
    }
}
