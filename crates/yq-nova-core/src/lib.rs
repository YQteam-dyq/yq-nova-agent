
pub mod background;
pub mod config;
pub mod embedding;
pub mod error;
pub mod graph;
pub mod logging;
pub mod memory;
pub mod storage;

pub use config::Config;
pub use error::{ErrorCode, NovaError, NovaResult};
pub use uuid::Uuid;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn git_sha() -> &'static str {
    option_env!("YQ_NOVA_GIT_SHA").unwrap_or("dev")
}
