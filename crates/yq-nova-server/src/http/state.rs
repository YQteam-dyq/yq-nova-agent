//! 共享 HTTP 服务状态：配置 + 核心 service 引用。
//!
//! 被所有 handler 克隆使用（内部是 Arc 或 Clone，成本很低）。

use std::sync::Arc;

use yq_nova_core::{
    config::ServerConfig, graph::GraphService, memory::MemoryService, storage::Database,
};

#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
    pub server_cfg: ServerConfig,
    pub db: Database,
    pub memory: MemoryService,
    pub graph: GraphService,
    /// build / start time for uptime reporting in /v1/health.
    pub started_at_epoch_secs: i64,
}

impl AppState {
    pub fn new(
        server_cfg: ServerConfig,
        db: Database,
        memory: MemoryService,
        graph: GraphService,
    ) -> Self {
        let started_at_epoch_secs = chrono::Utc::now().timestamp();
        Self { server_cfg, db, memory, graph, started_at_epoch_secs }
    }
}

// Silence unused Arc warning (Arc will be used when we add shared registry in M6.x).
#[allow(dead_code)]
fn _unused_arc(_: Arc<()>) {}
