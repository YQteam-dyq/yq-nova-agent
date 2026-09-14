use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use super::{
    EmbeddingMeta, EmbeddingProvider,
    retry::{RetryAction, RetryConfig, classify_http_status, with_retry},
};
use crate::error::NovaResult;

const DEFAULT_MODEL: &str = "Baichuan-Text-Embedding";
const OUTPUT_DIMS: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BaichuanConfig {
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

impl Default for BaichuanConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.baichuan-ai.com/v1".to_string(),
            api_key: String::new(),
            model: DEFAULT_MODEL.to_string(),
            dims: 0,
            batch_size: 16,
            request_timeout: Duration::from_secs(15),
            retry: RetryConfig::default(),
        }
    }
}

#[derive(Debug, Serialize)]
struct BaichuanReqBody<'a> {
    model: &'a str,

    input: &'a [&'a str],
}

#[serde_as]
#[derive(Debug, Deserialize)]
struct BaichuanRespItem {
    #[serde_as(as = "serde_with::VecSkipError<_>")]
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct BaichuanResp {
    data: Vec<BaichuanRespItem>,
}

pub struct BaichuanProvider {
    client: reqwest::Client,

    meta: EmbeddingMeta,

    config: BaichuanConfig,

    endpoint: String,
}

impl std::fmt::Debug for BaichuanProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaichuanProvider")
            .field("meta", &self.meta)
            .field("base_url", &self.config.base_url)
            .field("model", &self.config.model)
            .field("batch_size", &self.config.batch_size)
            .finish_non_exhaustive()
    }
}

impl BaichuanProvider {
    pub fn new(config: BaichuanConfig) -> NovaResult<Self> {
        let model = config.model.trim().to_string();
        if model.is_empty() {
            return Err(crate::error::NovaError::validation("baichuan: model must not be empty"));
        }
        let dims = if config.dims > 0 { config.dims } else { OUTPUT_DIMS };
        if config.batch_size == 0 {
            return Err(crate::error::NovaError::validation("baichuan: batch_size must be > 0"));
        }

        let base = config.base_url.trim_end_matches('/').to_string();
        let endpoint = format!("{base}/embeddings");

        let client = reqwest::Client::builder()
            .user_agent(concat!("yq-nova-agent/", env!("CARGO_PKG_VERSION")))
            .timeout(config.request_timeout)
            .build()
            .map_err(|e| {
                crate::error::NovaError::embedding_msg(format!("baichuan: build client: {e}"))
            })?;

        let meta = EmbeddingMeta {
            provider: "baichuan".into(),
            model: model.clone(),
            dims,
        };
        Ok(Self {
            client,
            meta,
            config: BaichuanConfig {
                model,
                ..config
            },
            endpoint,
        })
    }

    async fn run_once(&self, texts: &[&str]) -> NovaResult<Vec<Vec<f32>>> {
        let body = BaichuanReqBody {
            model: &self.config.model,
            input: texts,
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
                let snippet = if text.len() > 400 { &text[..400] } else { text.as_str() };
                return Err((
                    classify_http_status(status),
                    anyhow::anyhow!("HTTP {}: {}", status.as_u16(), snippet),
                ));
            }

            let parsed: BaichuanResp = res.json().await.map_err(|e| {
                (RetryAction::Fail, anyhow::anyhow!("baichuan: parse response: {e}"))
            })?;
            Ok(parsed)
        })
        .await?;

        if resp.data.len() != texts.len() {
            return Err(crate::error::NovaError::embedding_msg(format!(
                "baichuan: upstream returned {} embeddings for {} inputs",
                resp.data.len(),
                texts.len()
            )));
        }

        let expected_dims = self.meta.dims;
        let mut out = Vec::with_capacity(resp.data.len());
        for item in resp.data {
            if item.embedding.len() != expected_dims {
                return Err(crate::error::NovaError::embedding_msg(format!(
                    "baichuan: upstream returned dims={} expected {expected_dims}",
                    item.embedding.len()
                )));
            }
            out.push(item.embedding);
        }
        Ok(out)
    }
}

#[async_trait]
impl EmbeddingProvider for BaichuanProvider {
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
        let c = BaichuanConfig::default();
        assert_eq!(c.model, DEFAULT_MODEL);
        assert_eq!(c.batch_size, 16);
        assert_eq!(c.dims, 0);
        assert!(c.api_key.is_empty());
    }

    #[test]
    fn new_uses_fixed_output_dims() {
        let p = BaichuanProvider::new(BaichuanConfig::default()).unwrap();
        assert_eq!(p.meta().dims, OUTPUT_DIMS);
        assert_eq!(p.meta().provider, "baichuan");
        assert_eq!(p.meta().model, DEFAULT_MODEL);
    }

    #[test]
    fn new_respects_explicit_dims() {
        let p = BaichuanProvider::new(BaichuanConfig {
            dims: 512,
            ..BaichuanConfig::default()
        })
        .unwrap();
        assert_eq!(p.meta().dims, 512);
    }

    #[test]
    fn new_validates_required_fields() {
        let empty = BaichuanConfig {
            model: "  ".into(),
            ..BaichuanConfig::default()
        };
        assert!(BaichuanProvider::new(empty).is_err());

        let zero_batch = BaichuanConfig {
            batch_size: 0,
            ..BaichuanConfig::default()
        };
        assert!(BaichuanProvider::new(zero_batch).is_err());
    }

    #[test]
    fn endpoint_strips_trailing_slash() {
        let cfg = BaichuanConfig {
            base_url: "https://api.baichuan-ai.com/v1/".into(),
            ..BaichuanConfig::default()
        };
        let p = BaichuanProvider::new(cfg).unwrap();
        assert_eq!(p.endpoint, "https://api.baichuan-ai.com/v1/embeddings");
    }
}
