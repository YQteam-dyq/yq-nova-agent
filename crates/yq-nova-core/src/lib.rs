//! yq-nova-core: Agent 记忆 / 状态层核心库
//!
//! 为 yq-nova-agent 提供轻量级、单文件（SQLite）的 Agent 记忆与状态持久化能力。
//! 本 crate **与 HTTP / Web 框架无关**；服务端逻辑在 `yq-nova-server`，
//! Rust SDK 在 `yq-nova-sdk`。
//!
//! # 核心能力
//!
//! - **SQLite 持久化**：记忆、向量、实体、关系、标签全部落在单一 SQLite 文件，
//!   支持 WAL、mmap、连接池等生产级调优参数。
//! - **语义检索**：基于 EmbeddingProvider 的向量 KNN + 可选 FTS5 关键词搜索，
//!   并支持 RRF 混合排序。
//! - **图遍历**：实体-关系图谱的 BFS 遍历，支持深度、谓词白名单、置信度控制。
//! - **三大入口**：
//!   - [`remember()`](memory::MemoryService::remember) — 写入记忆（去重 + Embedding + 图谱抽取）
//!   - [`recall()`](memory::MemoryService::recall) — 按查询召回相关记忆（语义/关键词/图扩展/混合）
//!   - [`forget()`](memory::MemoryService::forget) — 归档或删除记忆，可选孤儿节点 GC
//!
//! # 公开模块（按生命周期阶段）
//!
//! | Module          | 阶段   | 用途
//! |-----------------|--------|---------------------------------------
//! | [`error`]       | M1     | 统一 `NovaError` + `NovaResult`
//! | [`config`]      | M1     | 基于 Figment 的分层配置加载
//! | [`logging`]     | M1     | Tracing subscriber 初始化 + trace_id
//! | [`storage`]     | M2     | SQLite + 向量库仓储层
//! | [`embedding`]   | M3     | `EmbeddingProvider` trait + 注册表
//! | [`memory`]      | M4     | remember / recall / forget 业务操作
//! | [`background`]  | M7     | TTL / 遗忘后台任务调度
//! | [`graph`]       | M8/M10 | 图谱操作 + 可选 LLM 抽取器

pub mod background;
pub mod config;
pub mod embedding;
pub mod error;
pub mod graph;
pub mod logging;
pub mod memory;
pub mod storage;

// Re-export the most commonly used types at the crate root so downstream
// crates can write `use yq_nova_core::NovaResult;` without nesting.
pub use config::Config;
pub use error::{ErrorCode, NovaError, NovaResult};
pub use uuid::Uuid;

/// Crate version (filled from Cargo.toml by `env!("CARGO_PKG_VERSION")`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// SHA of the Git commit built (or `"dev"` for local builds). Returned from
/// `GET /health` for debugging.
pub fn git_sha() -> &'static str {
    option_env!("YQ_NOVA_GIT_SHA").unwrap_or("dev")
}
