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
pub struct QwenConfig {
    pub base_url: String,

    pub api_key: String,

    pub model: String,

    pub dims: usize,

    pub batch_size: usize,

    #[serde(with = "crate::config::duration_seconds")]
    pub request_timeout: Duration,

    #[serde(flatten)]
    pub retry: RetryConfig,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            base_url: "https://dashscope.aliyuncs.com/api/v1".to_string(),
            api_key: String::new(),
            model: "text-embedding-v3".to_string(),
            dims: 0,
            batch_size: 16,
            request_timeout: Duration::from_secs(15),
            retry: RetryConfig::default(),
        }
    }
}

fn default_dims(model: &str) -> Option<usize> {
    match model.trim() {
        "text-embedding-v1" => Some(1536),
        "text-embedding-v2" => Some(1536),
        "text-embedding-v3" => Some(1024),
        _ => None,
    }
}

#[derive(Debug, Serialize)]
struct QwenInput<'a> {
    texts: &'a [&'a str],
}

#[derive(Debug, Serialize)]
struct QwenParameters {
    dimension: usize,
}

#[derive(Debug, Serialize)]
struct QwenReqBody<'a> {
    model: &'a str,

    input: QwenInput<'a>,

    parameters: QwenParameters,
}

#[serde_as]
#[derive(Debug, Deserialize)]
struct QwenOutputItem {
    #[serde_as(as = "serde_with::VecSkipError<_>")]
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct QwenOutput {
    embeddings: Vec<QwenOutputItem>,
}

#[derive(Debug, Deserialize)]
struct QwenResp {
    output: QwenOutput,
}

pub struct QwenProvider {
    client: reqwest::Client,

    meta: EmbeddingMeta,

    config: QwenConfig,

    endpoint: String,
}

impl std::fmt::Debug for QwenProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenProvider")
            .field("meta", &self.meta)
            .field("base_url", &self.config.base_url)
            .field("model", &self.config.model)
            .field("batch_size", &self.config.batch_size)
            .finish_non_exhaustive()
    }
}

impl QwenProvider {
    pub fn new(config: QwenConfig) -> NovaResult<Self> {
        let model = config.model.trim().to_string();
        if model.is_empty() {
            return Err(crate::error::NovaError::validation("qwen: model must not be empty"));
        }
        let dims = if config.dims > 0 {
            config.dims
        } else {
            default_dims(&model).ok_or_else(|| {
                crate::error::NovaError::validation(
                    "qwen: dims must be set for unknown model, supported defaults are ".to_owned()
                        + "text-embedding-v1 (1536), text-embedding-v2 (1536) and \
                           text-embedding-v3 (1024)",
                )
            })?
        };
        if config.batch_size == 0 {
            return Err(crate::error::NovaError::validation("qwen: batch_size must be > 0"));
        }

        let base = config.base_url.trim_end_matches('/').to_string();
        let endpoint = format!("{base}/services/embeddings/text-embedding/text-embedding");

        let client = reqwest::Client::builder()
            .user_agent(concat!("yq-nova-agent/", env!("CARGO_PKG_VERSION")))
            .timeout(config.request_timeout)
            .build()
            .map_err(|e| {
                crate::error::NovaError::embedding_msg(format!("qwen: build client: {e}"))
            })?;

        let meta = EmbeddingMeta {
            provider: "qwen".into(),
            model: model.clone(),
            dims,
        };
        Ok(Self {
            client,
            meta,
            config: QwenConfig {
                model,
                ..config
            },
            endpoint,
        })
    }

    async fn run_once(&self, texts: &[&str]) -> NovaResult<Vec<Vec<f32>>> {
        let body = QwenReqBody {
            model: &self.config.model,
            input: QwenInput {
                texts,
            },
            parameters: QwenParameters {
                dimension: self.meta.dims,
            },
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

            let parsed: QwenResp = res
                .json()
                .await
                .map_err(|e| (RetryAction::Fail, anyhow::anyhow!("qwen: parse response: {e}")))?;
            Ok(parsed)
        })
        .await?;

        let embeddings = resp.output.embeddings;
        if embeddings.len() != texts.len() {
            return Err(crate::error::NovaError::embedding_msg(format!(
                "qwen: upstream returned {} embeddings for {} inputs",
                embeddings.len(),
                texts.len()
            )));
        }

        let expected_dims = self.meta.dims;
        let mut out = Vec::with_capacity(embeddings.len());
        for item in embeddings {
            if item.embedding.len() != expected_dims {
                return Err(crate::error::NovaError::embedding_msg(format!(
                    "qwen: upstream returned dims={} expected {expected_dims}",
                    item.embedding.len()
                )));
            }
            out.push(item.embedding);
        }
        Ok(out)
    }
}

#[async_trait]
impl EmbeddingProvider for QwenProvider {
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
        let c = QwenConfig::default();
        assert_eq!(c.model, "text-embedding-v3");
        assert_eq!(c.batch_size, 16);
        assert_eq!(c.dims, 0);
        assert!(c.api_key.is_empty());
    }

    #[test]
    fn new_derives_dims_for_known_models() {
        let p = QwenProvider::new(QwenConfig {
            model: "text-embedding-v2".into(),
            ..QwenConfig::default()
        })
        .unwrap();
        assert_eq!(p.meta().dims, 1536);
        assert_eq!(p.meta().provider, "qwen");

        let p = QwenProvider::new(QwenConfig {
            model: "text-embedding-v3".into(),
            ..QwenConfig::default()
        })
        .unwrap();
        assert_eq!(p.meta().dims, 1024);
    }

    #[test]
    fn new_rejects_unknown_model_without_dims() {
        let r = QwenProvider::new(QwenConfig {
            model: "unknown".into(),
            ..QwenConfig::default()
        });
        assert!(matches!(r.unwrap_err().code(), crate::error::ErrorCode::Validation));
    }

    #[test]
    fn new_validates_required_fields() {
        let empty = QwenConfig {
            model: "  ".into(),
            ..QwenConfig::default()
        };
        assert!(QwenProvider::new(empty).is_err());

        let zero_batch = QwenConfig {
            batch_size: 0,
            ..QwenConfig::default()
        };
        assert!(QwenProvider::new(zero_batch).is_err());
    }

    #[test]
    fn endpoint_strips_trailing_slash() {
        let cfg = QwenConfig {
            base_url: "https://dashscope.aliyuncs.com/api/v1/".into(),
            model: "text-embedding-v3".into(),
            ..QwenConfig::default()
        };
        let p = QwenProvider::new(cfg).unwrap();
        assert_eq!(
            p.endpoint,
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        );
    }
}
