use std::{collections::BTreeMap, fmt::Debug, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::NovaResult;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EmbeddingMeta {
    pub provider: String,

    pub model: String,

    pub dims: usize,
}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync + Debug {
    fn meta(&self) -> &EmbeddingMeta;

    async fn embed_batch(&self, texts: &[&str]) -> NovaResult<Vec<Vec<f32>>>;

    async fn embed_one(&self, text: &str) -> NovaResult<Vec<f32>> {
        let mut v = self.embed_batch(&[text]).await?;
        v.pop().ok_or_else(|| {
            crate::error::NovaError::embedding_msg(
                "embed_batch returned an empty list for a single input",
            )
        })
    }
}

pub type SharedEmbeddingProvider = Arc<dyn EmbeddingProvider>;

#[derive(Debug, Default)]
pub struct EmbeddingRegistry {
    providers: BTreeMap<String, SharedEmbeddingProvider>,
}

impl EmbeddingRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, provider: SharedEmbeddingProvider) {
        self.providers.insert(name.into(), provider);
    }

    pub fn get(&self, name: &str) -> Option<SharedEmbeddingProvider> {
        self.providers.get(name).cloned()
    }

    pub fn names(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }
}

#[derive(Debug)]
pub struct MockEmbeddingProvider {
    meta: EmbeddingMeta,

    pub return_zero: bool,
}

impl MockEmbeddingProvider {
    pub fn new(dims: usize) -> Self {
        Self {
            meta: EmbeddingMeta {
                provider: "mock".into(),
                model: format!("mock-{dims}d"),
                dims,
            },
            return_zero: false,
        }
    }

    pub fn zero(dims: usize) -> Self {
        let mut s = Self::new(dims);
        s.return_zero = true;
        s
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    fn meta(&self) -> &EmbeddingMeta {
        &self.meta
    }

    async fn embed_batch(&self, texts: &[&str]) -> NovaResult<Vec<Vec<f32>>> {
        let dims = self.meta.dims;
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            if self.return_zero {
                out.push(vec![0.0f32; dims]);
            } else {
                out.push(deterministic_pseudo_embedding(t, dims));
            }
        }
        Ok(out)
    }
}

pub fn deterministic_pseudo_embedding(text: &str, dims: usize) -> Vec<f32> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(dims);
    for i in 0..dims {
        let mut h: u64 = 0xcbf29ce484222325;
        h ^= i as u64;
        h = h.wrapping_mul(0x100000001b3);
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }

        let val = (h as u32) as f32 / (u32::MAX as f32 / 2.0) - 1.0;
        out.push(val.clamp(-1.0, 1.0));
    }
    out
}

pub mod openai_compat;
pub use openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};

#[cfg(feature = "fastembed")]
pub mod fastembed;
#[cfg(feature = "fastembed")]
pub use fastembed::{FastEmbedProvider, FastEmbedProviderConfig};

pub mod retry;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_embedding_is_deterministic() {
        let p = MockEmbeddingProvider::new(8);
        let a = p.embed_one("hello world").await.unwrap();
        let b = p.embed_one("hello world").await.unwrap();
        let c = p.embed_one("hello galaxy").await.unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 8);
    }

    #[tokio::test]
    async fn zero_embedding_all_zeros() {
        let p = MockEmbeddingProvider::zero(4);
        let v = p.embed_one("any").await.unwrap();
        assert_eq!(v, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn registry_insert_and_get() {
        let mut r = EmbeddingRegistry::new();
        let p: SharedEmbeddingProvider = Arc::new(MockEmbeddingProvider::new(4));
        r.insert("default", p);
        assert!(r.get("default").is_some());
        assert!(r.get("nope").is_none());
    }
}
