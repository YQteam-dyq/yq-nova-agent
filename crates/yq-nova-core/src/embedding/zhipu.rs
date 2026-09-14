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
pub struct ZhipuConfig {
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

impl Default for ZhipuConfig {
    fn default() -> Self {
        Self {
            base_url: "https://open.bigmodel.cn/api/paas/v4".to_string(),
            api_key: String::new(),
            model: "embedding-3".to_string(),
            dims: 0,
            batch_size: 16,
            request_timeout: Duration::from_secs(15),
            retry: RetryConfig::default(),
        }
    }
}

fn resolve_dims(model: &str, dims: usize) -> NovaResult<(usize, bool)> {
    let model = model.trim();
    match model {
        "embedding-2" => {
            if dims != 0 && dims != 1024 {
                return Err(crate::error::NovaError::validation(
                    "zhipu: embedding-2 returns a fixed 1024 dimensions",
                ));
            }
            Ok((1024, false))
        },
        "embedding-3" => {
            let dim = if dims == 0 { 2048 } else { dims };
            if !(256..=2048).contains(&dim) {
                return Err(crate::error::NovaError::validation(format!(
                    "zhipu: embedding-3 supports 256..=2048 dimensions, got {}",
                    dim
                )));
            }
            Ok((dim, true))
        },
        _ => {
            if dims == 0 {
                return Err(crate::error::NovaError::validation(
                    "zhipu: dims must be set for unknown model",
                ));
            }
            Ok((dims, true))
        },
    }
}

#[derive(Debug, Serialize)]
struct ZhipuReqBody<'a> {
    model: &'a str,

    input: &'a [&'a str],

    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,

    encoding_format: &'static str,
}

#[serde_as]
#[derive(Debug, Deserialize)]
struct ZhipuRespItem {
    #[serde_as(as = "serde_with::VecSkipError<_>")]
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ZhipuResp {
    data: Vec<ZhipuRespItem>,
}

pub struct ZhipuProvider {
    client: reqwest::Client,

    meta: EmbeddingMeta,

    config: ZhipuConfig,

    endpoint: String,

    send_dimensions: bool,
}

impl std::fmt::Debug for ZhipuProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZhipuProvider")
            .field("meta", &self.meta)
            .field("base_url", &self.config.base_url)
            .field("model", &self.config.model)
            .field("batch_size", &self.config.batch_size)
            .finish_non_exhaustive()
    }
}

impl ZhipuProvider {
    pub fn new(config: ZhipuConfig) -> NovaResult<Self> {
        let model = config.model.trim().to_string();
        if model.is_empty() {
            return Err(crate::error::NovaError::validation("zhipu: model must not be empty"));
        }
        let (dims, send_dimensions) = resolve_dims(&model, config.dims)?;
        if config.batch_size == 0 {
            return Err(crate::error::NovaError::validation("zhipu: batch_size must be > 0"));
        }

        let base = config.base_url.trim_end_matches('/').to_string();
        let endpoint = format!("{base}/embeddings");

        let client = reqwest::Client::builder()
            .user_agent(concat!("yq-nova-agent/", env!("CARGO_PKG_VERSION")))
            .timeout(config.request_timeout)
            .build()
            .map_err(|e| {
                crate::error::NovaError::embedding_msg(format!("zhipu: build client: {e}"))
            })?;

        let meta = EmbeddingMeta {
            provider: "zhipu".into(),
            model: model.clone(),
            dims,
        };
        Ok(Self {
            client,
            meta,
            config: ZhipuConfig {
                model,
                ..config
            },
            endpoint,
            send_dimensions,
        })
    }

    async fn run_once(&self, texts: &[&str]) -> NovaResult<Vec<Vec<f32>>> {
        let dimensions = self.send_dimensions.then_some(self.meta.dims);
        let body = ZhipuReqBody {
            model: &self.config.model,
            input: texts,
            dimensions,
            encoding_format: "float",
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

            let parsed: ZhipuResp = res
                .json()
                .await
                .map_err(|e| (RetryAction::Fail, anyhow::anyhow!("zhipu: parse response: {e}")))?;
            Ok(parsed)
        })
        .await?;

        if resp.data.len() != texts.len() {
            return Err(crate::error::NovaError::embedding_msg(format!(
                "zhipu: upstream returned {} embeddings for {} inputs",
                resp.data.len(),
                texts.len()
            )));
        }

        let expected_dims = self.meta.dims;
        let mut out = Vec::with_capacity(resp.data.len());
        for item in resp.data {
            if item.embedding.len() != expected_dims {
                return Err(crate::error::NovaError::embedding_msg(format!(
                    "zhipu: upstream returned dims={} expected {expected_dims}",
                    item.embedding.len()
                )));
            }
            out.push(item.embedding);
        }
        Ok(out)
    }
}

#[async_trait]
impl EmbeddingProvider for ZhipuProvider {
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
        let c = ZhipuConfig::default();
        assert_eq!(c.model, "embedding-3");
        assert_eq!(c.batch_size, 16);
        assert_eq!(c.dims, 0);
        assert!(c.api_key.is_empty());
    }

    #[test]
    fn new_derives_dims_for_known_models() {
        let p = ZhipuProvider::new(ZhipuConfig {
            model: "embedding-2".into(),
            ..ZhipuConfig::default()
        })
        .unwrap();
        assert_eq!(p.meta().dims, 1024);
        assert_eq!(p.meta().provider, "zhipu");
        assert!(!p.send_dimensions);

        let p = ZhipuProvider::new(ZhipuConfig {
            model: "embedding-3".into(),
            ..ZhipuConfig::default()
        })
        .unwrap();
        assert_eq!(p.meta().dims, 2048);
        assert!(p.send_dimensions);

        let p = ZhipuProvider::new(ZhipuConfig {
            model: "embedding-3".into(),
            dims: 512,
            ..ZhipuConfig::default()
        })
        .unwrap();
        assert_eq!(p.meta().dims, 512);
    }

    #[test]
    fn new_rejects_dims_outside_model_range() {
        let too_big = ZhipuConfig {
            model: "embedding-3".into(),
            dims: 4096,
            ..ZhipuConfig::default()
        };
        assert!(matches!(
            ZhipuProvider::new(too_big).unwrap_err().code(),
            crate::error::ErrorCode::Validation
        ));

        let fixed = ZhipuConfig {
            model: "embedding-2".into(),
            dims: 512,
            ..ZhipuConfig::default()
        };
        assert!(matches!(
            ZhipuProvider::new(fixed).unwrap_err().code(),
            crate::error::ErrorCode::Validation
        ));
    }

    #[test]
    fn new_rejects_unknown_model_without_dims() {
        let r = ZhipuProvider::new(ZhipuConfig {
            model: "unknown".into(),
            ..ZhipuConfig::default()
        });
        assert!(matches!(r.unwrap_err().code(), crate::error::ErrorCode::Validation));
    }

    #[test]
    fn new_validates_required_fields() {
        let empty = ZhipuConfig {
            model: "  ".into(),
            ..ZhipuConfig::default()
        };
        assert!(ZhipuProvider::new(empty).is_err());

        let zero_batch = ZhipuConfig {
            batch_size: 0,
            ..ZhipuConfig::default()
        };
        assert!(ZhipuProvider::new(zero_batch).is_err());
    }

    #[test]
    fn endpoint_strips_trailing_slash() {
        let cfg = ZhipuConfig {
            base_url: "https://open.bigmodel.cn/api/paas/v4/".into(),
            model: "embedding-3".into(),
            ..ZhipuConfig::default()
        };
        let p = ZhipuProvider::new(cfg).unwrap();
        assert_eq!(p.endpoint, "https://open.bigmodel.cn/api/paas/v4/embeddings");
    }
}
