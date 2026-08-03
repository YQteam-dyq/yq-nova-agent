//! Background job runner (TTL expire job + forgetting job). M7 implements.

use serde::{Deserialize, Serialize};

/// Outcome of a single job execution, logged for observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRun {
    pub job: &'static str,
    pub affected_rows: u64,
    pub took_ms: u64,
    pub started_at_unix_ms: i64,
}

/// Action taken by the `forgetting` job when records match the stale criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ForgettingAction {
    #[default]
    Archive,
    Delete,
}

impl TryFrom<&str> for ForgettingAction {
    type Error = crate::error::NovaError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(match value {
            "archive" => ForgettingAction::Archive,
            "delete" => ForgettingAction::Delete,
            other => {
                return Err(crate::error::NovaError::config_msg(format!(
                    "unknown forgetting action: {other}"
                )));
            },
        })
    }
}

// M7 adds: BackgroundRunner struct with spawn/shutdown, TTL expire job,
// forgetting job, SIGHUP/TERM graceful integration.
