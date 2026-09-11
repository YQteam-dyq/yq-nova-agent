use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRun {
    pub job: &'static str,
    pub affected_rows: u64,
    pub took_ms: u64,
    pub started_at_unix_ms: i64,
}

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
