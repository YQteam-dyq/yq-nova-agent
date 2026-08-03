//! # yq-nova-sdk
//!
//! yq-nova-agent 官方 Rust HTTP SDK。
//!
//! 通过 [`http_client::HttpClient`] 与本地（或远程）运行的 `yq-nova` HTTP
//! 服务进行通信，提供记忆存储/检索、知识图谱构建/遍历等能力。SDK
//! 内置了两套请求 [`builder`](http_client::RememberReqBuilder)
//! 用于流畅地构造 remember / recall 请求，并复用
//! [`yq_nova_core::NovaError`] 统一的结构化错误类型与
//! [`yq_nova_core::error::ErrorCode`] 机器可读错误码。
//!
//! 两种传输模式：
//!
//! * **embedded** — 将 `yq-nova-core` 直接编译进宿主进程。零网络调用，
//!   性能最佳，共享同一个 SQLite 数据库文件（M10+ 预留占位）。
//! * **http** — 与本地运行的 `yq-nova` HTTP 服务器（默认端口 7999）通信。
//!   语言无关（Python / Go / JS 使用同一服务），当前主模式。

pub mod embedded;
pub mod http_client;

pub use yq_nova_core::{Config, NovaError, NovaResult, Uuid, VERSION};
