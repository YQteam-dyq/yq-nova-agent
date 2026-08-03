//! Embedding provider abstraction + mock impl (M3 fills in real providers).

use std::{collections::BTreeMap, fmt::Debug, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::NovaResult;

/// Embedding 提供者的元信息。用于在存储中打标向量，防止不同模型/提供者
/// 的向量混在同一张表内导致余弦相似度无意义。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EmbeddingMeta {
    /// 提供者名称，与 EmbeddingConfig 中的 key 对应。
    pub provider: String,
    /// 模型名称，如 text-embedding-3-small。
    pub model: String,
    /// 向量维度，如 1536 / 768 / 3072。
    pub dims: usize,
}

/// Embedding 提供者 trait：将文本批次转换为固定维度的向量。
///
/// 实现包括：
/// - `MockEmbeddingProvider`：测试用，确定性伪向量
/// - `OpenAiCompatProvider`：OpenAI 兼容 HTTP 接口
/// - （v0.3）`FastEmbedProvider`：本地 ONNX FastEmbed
#[async_trait]
pub trait EmbeddingProvider: Send + Sync + Debug {
    /// 返回该提供者的元信息（名称/模型/维度）。
    fn meta(&self) -> &EmbeddingMeta;

    /// 对一批文本进行 embedding。返回的向量顺序必须与输入顺序一致。
    ///
    /// 实现可根据自身限制（如 OpenAI 的 batch_size）进一步拆分子批次。
    async fn embed_batch(&self, texts: &[&str]) -> NovaResult<Vec<Vec<f32>>>;

    /// 便捷方法：单条文本 embedding。默认实现调用 `embed_batch(&[text])`。
    async fn embed_one(&self, text: &str) -> NovaResult<Vec<f32>> {
        let mut v = self.embed_batch(&[text]).await?;
        v.pop().ok_or_else(|| {
            crate::error::NovaError::embedding_msg(
                "embed_batch returned an empty list for a single input",
            )
        })
    }
}

/// 共享 Embedding 提供者别名：`Arc<dyn EmbeddingProvider>`，跨线程安全共享。
pub type SharedEmbeddingProvider = Arc<dyn EmbeddingProvider>;

// -----------------------------------------------------------------------------
// Registry
// -----------------------------------------------------------------------------

/// Embedding 提供者注册表：按自定义名称管理一组提供者，供配置层按
/// `default_provider` 名称选择。
#[derive(Debug, Default)]
pub struct EmbeddingRegistry {
    providers: BTreeMap<String, SharedEmbeddingProvider>,
}

impl EmbeddingRegistry {
    /// 创建空注册表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 插入一个命名提供者；同名会被覆盖。
    pub fn insert(&mut self, name: impl Into<String>, provider: SharedEmbeddingProvider) {
        self.providers.insert(name.into(), provider);
    }

    /// 按名称查询提供者，不存在返回 None。
    pub fn get(&self, name: &str) -> Option<SharedEmbeddingProvider> {
        self.providers.get(name).cloned()
    }

    /// 列出所有已注册名称（统计/调试用）。
    pub fn names(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }
}

// -----------------------------------------------------------------------------
// Mock provider: returns a deterministic, cheap-to-compute pseudo-embedding.
// Used in unit tests and as a fallback until the user configures a real one.
// -----------------------------------------------------------------------------

/// Mock Embedding 提供者：返回基于滚动哈希的确定性伪向量，用于单元测试
/// 与未配置真实提供者时的兜底。相同输入产生相同向量，成本极低。
#[derive(Debug)]
pub struct MockEmbeddingProvider {
    meta: EmbeddingMeta,
    /// 若为 true，每次调用返回全零向量；否则使用基于哈希的确定性签名。
    pub return_zero: bool,
}

impl MockEmbeddingProvider {
    /// 创建指定维度的 Mock 提供者（默认使用哈希签名模式）。
    pub fn new(dims: usize) -> Self {
        Self {
            meta: EmbeddingMeta { provider: "mock".into(), model: format!("mock-{dims}d"), dims },
            return_zero: false,
        }
    }

    /// 创建指定维度的 Mock 提供者（全零向量模式），用于「不关心语义，仅测流程」的测试。
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

/// 确定性伪向量生成函数：基于 FNV-1a 滚动哈希，将每维哈希归一化到 [-1, 1]。
///
/// 无语义价值，但相同输入产生相同向量、差异大的输入产生差异大的向量，
/// 足够用于无网络调用的 recall 管线集成测试。
pub fn deterministic_pseudo_embedding(text: &str, dims: usize) -> Vec<f32> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(dims);
    for i in 0..dims {
        // FNV-1a each dimension independently.
        let mut h: u64 = 0xcbf29ce484222325;
        h ^= i as u64;
        h = h.wrapping_mul(0x100000001b3);
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        // normalise the low 32 bits to [-1, 1]
        let val = (h as u32) as f32 / (u32::MAX as f32 / 2.0) - 1.0;
        out.push(val.clamp(-1.0, 1.0));
    }
    out
}

// Real OpenAI-compatible provider lives in M3.
pub mod openai_compat;
pub use openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};

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
