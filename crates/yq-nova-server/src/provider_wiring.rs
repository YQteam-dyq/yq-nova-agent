use std::sync::Arc;

#[cfg(feature = "fastembed")]
use yq_nova_core::embedding::{EmbeddingProvider, FastEmbedProvider, FastEmbedProviderConfig};
use yq_nova_core::{
    config::{EmbeddingConfig, FastEmbedConfig},
    embedding::{
        BaichuanConfig, BaichuanProvider, EmbeddingProvider, EmbeddingRegistry, JinaConfig,
        JinaProvider, MockEmbeddingProvider, OpenAiCompatConfig, OpenAiCompatProvider, QwenConfig,
        QwenProvider, SharedEmbeddingProvider, ZhipuConfig, ZhipuProvider, retry::RetryConfig,
    },
    error::{NovaError, NovaResult},
};

pub fn build_default_provider(
    cfg: &EmbeddingConfig,
) -> NovaResult<(String, SharedEmbeddingProvider, usize)> {
    let key = cfg.default_provider.trim().to_string();
    if key.is_empty() {
        return Err(NovaError::config_msg("embedding.default_provider must be non-empty"));
    }

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

    if let Some(fastembed_cfg) = cfg.fastembed_local.get(&key) {
        return build_fastembed(fastembed_cfg, &key);
    }

    if let Some(zhipu_cfg) = cfg.zhipu.get(&key) {
        let embed_config = ZhipuConfig {
            base_url: zhipu_cfg.base_url.trim_end_matches('/').to_string(),
            api_key: zhipu_cfg.api_key.clone(),
            model: zhipu_cfg.model.clone(),
            dims: zhipu_cfg.dims,
            batch_size: zhipu_cfg.batch_size.max(1),
            request_timeout: zhipu_cfg.request_timeout,
            retry: RetryConfig {
                max_attempts: zhipu_cfg.retry.max_attempts,
                ..Default::default()
            },
        };
        let provider = ZhipuProvider::new(embed_config)?;
        let dims = provider.meta().dims;
        return Ok((key.to_string(), Arc::new(provider), dims));
    }

    if let Some(qwen_cfg) = cfg.qwen.get(&key) {
        let embed_config = QwenConfig {
            base_url: qwen_cfg.base_url.trim_end_matches('/').to_string(),
            api_key: qwen_cfg.api_key.clone(),
            model: qwen_cfg.model.clone(),
            dims: qwen_cfg.dims,
            batch_size: qwen_cfg.batch_size.max(1),
            request_timeout: qwen_cfg.request_timeout,
            retry: RetryConfig {
                max_attempts: qwen_cfg.retry.max_attempts,
                ..Default::default()
            },
        };
        let provider = QwenProvider::new(embed_config)?;
        let dims = provider.meta().dims;
        return Ok((key.to_string(), Arc::new(provider), dims));
    }

    if let Some(baichuan_cfg) = cfg.baichuan.get(&key) {
        let embed_config = BaichuanConfig {
            base_url: baichuan_cfg.base_url.trim_end_matches('/').to_string(),
            api_key: baichuan_cfg.api_key.clone(),
            model: baichuan_cfg.model.clone(),
            dims: baichuan_cfg.dims,
            batch_size: baichuan_cfg.batch_size.max(1),
            request_timeout: baichuan_cfg.request_timeout,
            retry: RetryConfig {
                max_attempts: baichuan_cfg.retry.max_attempts,
                ..Default::default()
            },
        };
        let provider = BaichuanProvider::new(embed_config)?;
        let dims = provider.meta().dims;
        return Ok((key.to_string(), Arc::new(provider), dims));
    }

    if let Some(jina_cfg) = cfg.jina.get(&key) {
        let embed_config = JinaConfig {
            base_url: jina_cfg.base_url.trim_end_matches('/').to_string(),
            api_key: jina_cfg.api_key.clone(),
            model: jina_cfg.model.clone(),
            dims: jina_cfg.dims,
            task: jina_cfg.task.clone(),
            batch_size: jina_cfg.batch_size.max(1),
            request_timeout: jina_cfg.request_timeout,
            retry: RetryConfig {
                max_attempts: jina_cfg.retry.max_attempts,
                ..Default::default()
            },
        };
        let provider = JinaProvider::new(embed_config)?;
        let dims = provider.meta().dims;
        return Ok((key.to_string(), Arc::new(provider), dims));
    }

    Err(NovaError::config_msg(format!(
        "embedding.default_provider='{key}' not found. Available keys: openai_compatible [{}], \
         zhipu [{}], qwen [{}], baichuan [{}], jina [{}], fastembed_local [{}], special 'mock'",
        cfg.openai_compatible.keys().cloned().collect::<Vec<_>>().join(", "),
        cfg.zhipu.keys().cloned().collect::<Vec<_>>().join(", "),
        cfg.qwen.keys().cloned().collect::<Vec<_>>().join(", "),
        cfg.baichuan.keys().cloned().collect::<Vec<_>>().join(", "),
        cfg.jina.keys().cloned().collect::<Vec<_>>().join(", "),
        cfg.fastembed_local.keys().cloned().collect::<Vec<_>>().join(", "),
    )))
}

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
        "fastembed-local provider '{key}' requires --features fastembed; current binary was built \
         without local-onnx support. Tip: use 'mock' or an openai_compatible provider for now.",
    )))
}

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
            retry: RetryConfig {
                max_attempts: oai.max_retries.max(1),
                ..Default::default()
            },
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

    for (name, z) in &cfg.zhipu {
        let embed_config = ZhipuConfig {
            base_url: z.base_url.trim_end_matches('/').to_string(),
            api_key: z.api_key.clone(),
            model: z.model.clone(),
            dims: z.dims,
            batch_size: z.batch_size.max(1),
            request_timeout: z.request_timeout,
            retry: RetryConfig {
                max_attempts: z.retry.max_attempts,
                ..Default::default()
            },
        };
        match ZhipuProvider::new(embed_config) {
            Ok(p) => {
                reg.insert(name.clone(), Arc::new(p));
            },
            Err(e) => {
                tracing::warn!(provider = %name, error = %e, "skip registering invalid zhipu provider");
            },
        }
    }

    for (name, q) in &cfg.qwen {
        let embed_config = QwenConfig {
            base_url: q.base_url.trim_end_matches('/').to_string(),
            api_key: q.api_key.clone(),
            model: q.model.clone(),
            dims: q.dims,
            batch_size: q.batch_size.max(1),
            request_timeout: q.request_timeout,
            retry: RetryConfig {
                max_attempts: q.retry.max_attempts,
                ..Default::default()
            },
        };
        match QwenProvider::new(embed_config) {
            Ok(p) => {
                reg.insert(name.clone(), Arc::new(p));
            },
            Err(e) => {
                tracing::warn!(provider = %name, error = %e, "skip registering invalid qwen provider");
            },
        }
    }

    for (name, b) in &cfg.baichuan {
        let embed_config = BaichuanConfig {
            base_url: b.base_url.trim_end_matches('/').to_string(),
            api_key: b.api_key.clone(),
            model: b.model.clone(),
            dims: b.dims,
            batch_size: b.batch_size.max(1),
            request_timeout: b.request_timeout,
            retry: RetryConfig {
                max_attempts: b.retry.max_attempts,
                ..Default::default()
            },
        };
        match BaichuanProvider::new(embed_config) {
            Ok(p) => {
                reg.insert(name.clone(), Arc::new(p));
            },
            Err(e) => {
                tracing::warn!(provider = %name, error = %e, "skip registering invalid baichuan provider");
            },
        }
    }

    for (name, j) in &cfg.jina {
        let embed_config = JinaConfig {
            base_url: j.base_url.trim_end_matches('/').to_string(),
            api_key: j.api_key.clone(),
            model: j.model.clone(),
            dims: j.dims,
            task: j.task.clone(),
            batch_size: j.batch_size.max(1),
            request_timeout: j.request_timeout,
            retry: RetryConfig {
                max_attempts: j.retry.max_attempts,
                ..Default::default()
            },
        };
        match JinaProvider::new(embed_config) {
            Ok(p) => {
                reg.insert(name.clone(), Arc::new(p));
            },
            Err(e) => {
                tracing::warn!(provider = %name, error = %e, "skip registering invalid jina provider");
            },
        }
    }

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
        let cfg = EmbeddingConfig {
            default_provider: "mock".into(),
            ..Default::default()
        };
        let (name, _p, dims) = build_default_provider(&cfg).unwrap();
        assert_eq!(name, "mock");
        assert!(dims >= 1);
    }

    #[test]
    fn unknown_provider_returns_config_error() {
        let cfg = EmbeddingConfig {
            default_provider: "does-not-exist".into(),
            ..Default::default()
        };
        let err = build_default_provider(&cfg).unwrap_err();
        assert_eq!(err.code(), yq_nova_core::error::ErrorCode::Config);
    }

    #[test]
    fn zhipu_provider_is_selectable() {
        let mut zhipu = std::collections::BTreeMap::new();
        zhipu.insert(
            "zh".into(),
            ZhipuConfig {
                model: "embedding-3".into(),
                ..Default::default()
            },
        );
        let cfg = EmbeddingConfig {
            default_provider: "zh".into(),
            zhipu,
            ..Default::default()
        };
        let (name, p, dims) = build_default_provider(&cfg).unwrap();
        assert_eq!(name, "zh");
        assert_eq!(dims, 2048);
        assert_eq!(p.meta().provider, "zhipu");
    }

    #[test]
    fn qwen_provider_is_selectable() {
        let mut qwen = std::collections::BTreeMap::new();
        qwen.insert(
            "qw".into(),
            QwenConfig {
                model: "text-embedding-v3".into(),
                ..Default::default()
            },
        );
        let cfg = EmbeddingConfig {
            default_provider: "qw".into(),
            qwen,
            ..Default::default()
        };
        let (name, p, dims) = build_default_provider(&cfg).unwrap();
        assert_eq!(name, "qw");
        assert_eq!(dims, 1024);
        assert_eq!(p.meta().provider, "qwen");
    }

    #[test]
    fn registry_registers_provider_families() {
        let mut zhipu = std::collections::BTreeMap::new();
        zhipu.insert(
            "zh".into(),
            ZhipuConfig {
                model: "embedding-3".into(),
                ..Default::default()
            },
        );
        let cfg = EmbeddingConfig {
            default_provider: "zh".into(),
            zhipu,
            ..Default::default()
        };
        let (name, _p, _dims, reg) = build_registry(&cfg).unwrap();
        assert_eq!(name, "zh");
        assert!(reg.get("zh").is_some());
        assert_eq!(reg.get("zh").unwrap().meta().provider, "zhipu");
    }
}
