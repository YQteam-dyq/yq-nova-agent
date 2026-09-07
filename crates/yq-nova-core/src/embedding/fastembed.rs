
use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use tokio::sync::Mutex;

use super::{EmbeddingMeta, EmbeddingProvider};
use crate::error::{NovaError, NovaResult};

#[derive(Debug, Clone)]
pub struct FastEmbedProviderConfig {

    pub model_name: String,

    pub dimensions: usize,

    pub cache_dir: PathBuf,
}

impl Default for FastEmbedProviderConfig {
    fn default() -> Self {
        Self {
            model_name: "BAAI/bge-small-en-v1.5".into(),
            dimensions: 0,
            cache_dir: PathBuf::new(),
        }
    }
}

pub struct FastEmbedProvider {
    model: Arc<Mutex<TextEmbedding>>,
    meta: EmbeddingMeta,
}

impl std::fmt::Debug for FastEmbedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedProvider").field("meta", &self.meta).finish_non_exhaustive()
    }
}

const SUPPORTED_MODELS: &[&str] = &[
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "jinaai/jina-embeddings-v2-base-en",
];

fn resolve_model(name: &str) -> NovaResult<EmbeddingModel> {
    match name.trim() {
        "BAAI/bge-small-en-v1.5" => Ok(EmbeddingModel::BGESmallENV15),
        "BAAI/bge-base-en-v1.5" => Ok(EmbeddingModel::BGEBaseENV15),
        "sentence-transformers/all-MiniLM-L6-v2" => Ok(EmbeddingModel::AllMiniLML6V2),
        "jinaai/jina-embeddings-v2-base-en" => Ok(EmbeddingModel::JinaEmbeddingsV2BaseEN),
        _ => Err(NovaError::config_msg(format!(
            "fastembed: unsupported model_name='{}'. Supported: {}",
            name,
            SUPPORTED_MODELS.join(", "),
        ))),
    }
}

fn known_dims(model: &EmbeddingModel) -> usize {
    match model {
        EmbeddingModel::BGESmallENV15 => 384,
        EmbeddingModel::BGEBaseENV15 => 768,
        EmbeddingModel::AllMiniLML6V2 => 384,
        EmbeddingModel::JinaEmbeddingsV2BaseEN => 768,

        _ => 384,
    }
}

impl FastEmbedProvider {

    pub fn new(config: FastEmbedProviderConfig) -> NovaResult<Self> {
        let model_kind = resolve_model(&config.model_name)?;
        let dims = if config.dimensions > 0 {
            config.dimensions
        } else {
            known_dims(&model_kind)
        };

        let mut opts = TextInitOptions::new(model_kind);
        if !config.cache_dir.as_os_str().is_empty() {
            opts = opts.with_cache_dir(config.cache_dir.clone());
        }
        let model = TextEmbedding::try_new(opts)
            .map_err(|e| NovaError::embedding(format!("fastembed: init model: {e}"), e))?;

        let meta = EmbeddingMeta {
            provider: "fastembed_local".into(),
            model: config.model_name.clone(),
            dims,
        };
        Ok(Self { model: Arc::new(Mutex::new(model)), meta })
    }
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    fn meta(&self) -> &EmbeddingMeta {
        &self.meta
    }

    async fn embed_batch(&self, texts: &[&str]) -> NovaResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let input: Vec<&str> = texts.to_vec();
        let embeddings = {
            let mut model = self.model.lock().await;
            model
                .embed(input, None)
                .map_err(|e| NovaError::embedding(format!("fastembed: embed failed: {e}"), e))?
        };

        for v in &embeddings {
            if v.len() != self.meta.dims {
                return Err(NovaError::embedding_msg(format!(
                    "fastembed: got dims={} expected {}",
                    v.len(),
                    self.meta.dims
                )));
            }
        }
        Ok(embeddings)
    }
}