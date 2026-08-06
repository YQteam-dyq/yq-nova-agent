//! Embedding 配置 → Provider 实例化的 glue 层。
//!
//! 这里把 `yq_nova_core::config::EmbeddingConfig`（用户 TOML/env 配置）映射为
//! `yq_nova_core::embedding::EmbeddingProvider`（真实可用的 trait object）。
//!
//! 规则：
//! - `provider = mock` / 全部 key 都没有 api_key + base_url 走通：回退 MockEmbeddingProvider（开发环境/测试用）。
//! - 其它走 `config.openai_compatible["default_provider"]` → OpenAiCompatProvider。
//! - FastEmbed 等本地模型：MVP 占位（feature-gated），检测到就报错提示 --features。

use std::sync::Arc;

use yq_nova_core::{
    config::{EmbeddingConfig, FastEmbedConfig},
    embedding::{
        EmbeddingRegistry, MockEmbeddingProvider, OpenAiCompatConfig, OpenAiCompatProvider,
        SharedEmbeddingProvider, retry::RetryConfig,
    },
    error::{NovaError, NovaResult},
};
#[cfg(feature = "fastembed")]
use yq_nova_core::embedding::{EmbeddingProvider, FastEmbedProvider, FastEmbedProviderConfig};

/// 给定一个 EmbeddingConfig，解析成 SharedEmbeddingProvider（Arc<dyn ...>）。
///
/// 输出 provider 名称 + 实际 provider 句柄 + 可选 embedding 维度（用于初始化 vector_store）。
pub fn build_default_provider(
    cfg: &EmbeddingConfig,
) -> NovaResult<(String, SharedEmbeddingProvider, usize)> {
    let key = cfg.default_provider.trim().to_string();
    if key.is_empty() {
        return Err(NovaError::config_msg("embedding.default_provider must be non-empty"));
    }

    // --- 1. mock provider: 显式绕过任何网络请求 ---
    if key.eq_ignore_ascii_case("mock") {
        let dims = cfg
            .openai_compatible
            .get("default")
            .map(|p| p.dimensions)
            .filter(|d| *d > 0)
            .unwrap_or(1536);
        let mock = MockEmbeddingProvider::new(dims);
        return Ok(("mock".into(), Arc::new(mock), dims));
    }

    // --- 2. OpenAI-compatible provider ---
    if let Some(openai_cfg) = cfg.openai_compatible.get(&key) {
        let dims = openai_cfg.dimensions.max(1);
        let embed_config = OpenAiCompatConfig {
            base_url: openai_cfg.base_url.trim_end_matches('/').to_string(),
            api_key: openai_cfg.api_key.clone(),
            model: openai_cfg.model.clone(),
            dims,
            batch_size: openai_cfg.batch_size.max(1),
            request_timeout: openai_cfg.timeout,
            retry: RetryConfig {
                max_attempts: openai_cfg.max_retries.max(1),
                ..Default::default()
            },
        };
        let provider = OpenAiCompatProvider::new(embed_config)?;
        return Ok((key, Arc::new(provider), dims));
    }

    // --- 3. FastEmbed provider (local ONNX) ---
    if let Some(fastembed_cfg) = cfg.fastembed_local.get(&key) {
        return build_fastembed(fastembed_cfg, &key);
    }

    Err(NovaError::config_msg(format!(
        "embedding.default_provider='{key}' not found. \
         Available keys: openai_compatible keys: [{}], fastembed keys: [{}], special: 'mock'",
        cfg.openai_compatible.keys().cloned().collect::<Vec<_>>().join(", "),
        cfg.fastembed_local.keys().cloned().collect::<Vec<_>>().join(", "),
    )))
}

/// 构造 FastEmbed 本地 ONNX 提供者。
///
/// 启用 `fastembed` feature 时真正加载模型；未启用时返回配置错误，提示
/// 需要 `--features fastembed`。
fn build_fastembed(
    cfg: &FastEmbedConfig,
    key: &str,
) -> NovaResult<(String, SharedEmbeddingProvider, usize)> {
    build_fastembed_impl(cfg, key)
}

#[cfg(feature = "fastembed")]
fn build_fastembed_impl(
    cfg: &FastEmbedConfig,
    key: &str,
) -> NovaResult<(String, SharedEmbeddingProvider, usize)> {
    let provider_cfg = FastEmbedProviderConfig {
        model_name: cfg.model_name.clone(),
        dimensions: cfg.dimensions,
        cache_dir: cfg.cache_dir.clone(),
    };
    let provider = FastEmbedProvider::new(provider_cfg)?;
    let dims = provider.meta().dims;
    Ok((key.to_string(), Arc::new(provider), dims))
}

#[cfg(not(feature = "fastembed"))]
fn build_fastembed_impl(
    _cfg: &FastEmbedConfig,
    key: &str,
) -> NovaResult<(String, SharedEmbeddingProvider, usize)> {
    Err(NovaError::config_msg(format!(
        "fastembed-local provider '{key}' requires --features fastembed; \
         current binary was built without local-onnx support. \
         Tip: use 'mock' or an openai_compatible provider for now.",
    )))
}

/// 把所有在配置里定义的 provider 都注册到 EmbeddingRegistry，方便 M10 多 provider 切换。
/// 默认 provider 单独返回（最常用）。
pub fn build_registry(
    cfg: &EmbeddingConfig,
) -> NovaResult<(String, SharedEmbeddingProvider, usize, EmbeddingRegistry)> {
    let mut reg = EmbeddingRegistry::new();
    for (name, oai) in &cfg.openai_compatible {
        let embed_config = OpenAiCompatConfig {
            base_url: oai.base_url.trim_end_matches('/').to_string(),
            api_key: oai.api_key.clone(),
            model: oai.model.clone(),
            dims: oai.dimensions.max(1),
            batch_size: oai.batch_size.max(1),
            request_timeout: oai.timeout,
            retry: RetryConfig { max_attempts: oai.max_retries.max(1), ..Default::default() },
        };
        match OpenAiCompatProvider::new(embed_config) {
            Ok(p) => {
                reg.insert(name.clone(), Arc::new(p));
            },
            Err(e) => {
                tracing::warn!(provider = %name, error = %e, "skip registering invalid openai_compatible provider");
            },
        }
    }
    // Always register mock provider so callers can fall back to it.
    let mock_dims = cfg.openai_compatible.get("default").map(|p| p.dimensions).unwrap_or(1536);
    reg.insert(String::from("mock"), Arc::new(MockEmbeddingProvider::new(mock_dims)));

    let (default_name, default_provider, dims) = build_default_provider(cfg)?;
    Ok((default_name, default_provider, dims, reg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_picks_default_openai_key() {
        let cfg = EmbeddingConfig::default();
        let (name, _p, dims) = build_default_provider(&cfg).unwrap();
        assert_eq!(name, "default");
        assert_eq!(dims, 1536);
    }

    #[test]
    fn mock_provider_returns_mock_name_and_dims() {
        let cfg = EmbeddingConfig { default_provider: "mock".into(), ..Default::default() };
        let (name, _p, dims) = build_default_provider(&cfg).unwrap();
        assert_eq!(name, "mock");
        assert!(dims >= 1);
    }

    #[test]
    fn unknown_provider_returns_config_error() {
        let cfg =
            EmbeddingConfig { default_provider: "does-not-exist".into(), ..Default::default() };
        let err = build_default_provider(&cfg).unwrap_err();
        assert_eq!(err.code(), yq_nova_core::error::ErrorCode::Config);
    }
}
