
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

#[allow(dead_code)]
fn _unused_arc(_: Arc<()>) {}
