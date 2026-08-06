//! 本地 ONNX FastEmbed 本地推理提供者（feature-gated：`fastembed`）。
//!
//! FastEmbed 通过 ONNX Runtime 在本地 CPU 上加载句子嵌入模型并做推理，
//! 无需任何外部 HTTP 服务。核心 API 是同步阻塞的（`TextEmbedding::embed`），
//! 因此把模型实例包在 `tokio::sync::Mutex` 中，在 `embed_batch` 里加锁调用，
//! 从而满足 `EmbeddingProvider` 的 `Send + Sync` 约束。

use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use tokio::sync::Mutex;

use super::{EmbeddingMeta, EmbeddingProvider};
use crate::error::{NovaError, NovaResult};

/// 本地 FastEmbed 提供者的构造配置（可从 `config::FastEmbedConfig` 转换而来）。
#[derive(Debug, Clone)]
pub struct FastEmbedProviderConfig {
    /// FastEmbed 模型名称，如 `BAAI/bge-small-en-v1.5`。
    pub model_name: String,
    /// 向量维度。`0` 表示按模型自动推断（见 [`known_dims`]）。
    pub dimensions: usize,
    /// 模型文件缓存目录。空表示使用 fastembed 的全局默认缓存。
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

/// 本地 ONNX FastEmbed 提供者。
///
/// 模型在 `new` 时同步加载（首次会下载 ONNX 权重到缓存目录），推理阶段
/// 通过内部 `Mutex` 串行化阻塞调用。
pub struct FastEmbedProvider {
    model: Arc<Mutex<TextEmbedding>>,
    meta: EmbeddingMeta,
}

impl std::fmt::Debug for FastEmbedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedProvider").field("meta", &self.meta).finish_non_exhaustive()
    }
}

/// 支持的模型名列表（用于错误提示）。
const SUPPORTED_MODELS: &[&str] = &[
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "jinaai/jina-embeddings-v2-base-en",
];

/// 将配置中的模型名解析为 fastembed 的 [`EmbeddingModel`] 枚举。
///
/// 对无法识别的名称返回 `Config` 错误并列出支持列表。
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

/// 已知模型的默认输出维度（当 `dimensions == 0` 时用于自动推断）。
fn known_dims(model: &EmbeddingModel) -> usize {
    match model {
        EmbeddingModel::BGESmallENV15 => 384,
        EmbeddingModel::BGEBaseENV15 => 768,
        EmbeddingModel::AllMiniLML6V2 => 384,
        EmbeddingModel::JinaEmbeddingsV2BaseEN => 768,
        // 其它未显式列出的模型统一回退到 384（与 AllMiniLM 一致）。
        _ => 384,
    }
}

impl FastEmbedProvider {
    /// 构造本地 ONNX FastEmbed 提供者。
    ///
    /// 注意：`try_new` 会同步加载（必要时下载）模型权重，可能在首次调用
    /// 时阻塞较长时间。调用方通常在启动阶段调用本函数。
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

        // `TextEmbedding::embed` 需要 `Vec<&str>`（非切片），且为同步阻塞调用，
        // 因此持锁执行以避免并发推理。
        let input: Vec<&str> = texts.to_vec();
        let embeddings = {
            let mut model = self.model.lock().await;
            model
                .embed(input, None)
                .map_err(|e| NovaError::embedding(format!("fastembed: embed failed: {e}"), e))?
        };

        // 校验返回维度与配置一致，避免毒化向量库。
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