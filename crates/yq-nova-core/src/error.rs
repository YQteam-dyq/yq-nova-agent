
use std::fmt;

use serde::Serialize;

pub type NovaResult<T> = Result<T, NovaError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ErrorCode {

    Validation = 100,

    NotFound = 101,

    Conflict = 102,

    Forbidden = 103,

    Storage = 200,

    Embedding = 201,

    Graph = 202,

    Config = 203,

    Internal = 204,
}

impl ErrorCode {

    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::Validation => "validation",
            ErrorCode::NotFound => "not_found",
            ErrorCode::Conflict => "conflict",
            ErrorCode::Forbidden => "forbidden",
            ErrorCode::Storage => "storage",
            ErrorCode::Embedding => "embedding",
            ErrorCode::Graph => "graph",
            ErrorCode::Config => "config",
            ErrorCode::Internal => "internal",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub struct NovaError {
    code: ErrorCode,
    message: String,
    #[source]
    source: Option<anyhow::Error>,

    trace_id: Option<String>,
}

impl NovaError {

    pub fn validation<M: Into<String>>(msg: M) -> Self {
        Self::new(ErrorCode::Validation, msg.into(), None)
    }

    pub fn validation_with_ctx<M: Into<String>, E: Into<anyhow::Error>>(msg: M, src: E) -> Self {
        Self::new(ErrorCode::Validation, msg.into(), Some(src.into()))
    }

    pub fn not_found<M: Into<String>>(msg: M) -> Self {
        Self::new(ErrorCode::NotFound, msg.into(), None)
    }

    pub fn validation_msg<M: Into<String>>(msg: M) -> Self {
        Self::new(ErrorCode::Validation, msg.into(), None)
    }

    pub fn conflict<M: Into<String>>(msg: M) -> Self {
        Self::new(ErrorCode::Conflict, msg.into(), None)
    }

    pub fn storage<E: Into<anyhow::Error>>(src: E) -> Self {
        let src = src.into();
        Self::new(ErrorCode::Storage, src.to_string(), Some(src))
    }

    pub fn storage_msg<M: Into<String>>(msg: M) -> Self {
        Self::new(ErrorCode::Storage, msg.into(), None)
    }

    pub fn storage_with_ctx<M: Into<String>, E: Into<anyhow::Error>>(msg: M, src: E) -> Self {
        Self::new(ErrorCode::Storage, msg.into(), Some(src.into()))
    }

    pub fn embedding<M: Into<String>, E: Into<anyhow::Error>>(msg: M, src: E) -> Self {
        Self::new(ErrorCode::Embedding, msg.into(), Some(src.into()))
    }

    pub fn embedding_msg<M: Into<String>>(msg: M) -> Self {
        Self::new(ErrorCode::Embedding, msg.into(), None)
    }

    pub fn graph<M: Into<String>, E: Into<anyhow::Error>>(msg: M, src: E) -> Self {
        Self::new(ErrorCode::Graph, msg.into(), Some(src.into()))
    }

    pub fn graph_msg<M: Into<String>>(msg: M) -> Self {
        Self::new(ErrorCode::Graph, msg.into(), None)
    }

    pub fn config<E: Into<anyhow::Error>>(src: E) -> Self {
        let src = src.into();
        Self::new(ErrorCode::Config, src.to_string(), Some(src))
    }

    pub fn config_msg<M: Into<String>>(msg: M) -> Self {
        Self::new(ErrorCode::Config, msg.into(), None)
    }

    pub fn internal<M: Into<String>>(msg: M) -> Self {
        Self::new(ErrorCode::Internal, msg.into(), None)
    }

    pub fn internal_with_ctx<M: Into<String>, E: Into<anyhow::Error>>(msg: M, src: E) -> Self {
        Self::new(ErrorCode::Internal, msg.into(), Some(src.into()))
    }

    pub fn new(code: ErrorCode, message: String, source: Option<anyhow::Error>) -> Self {
        Self { code, message, source, trace_id: None }
    }

    #[inline]
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    #[inline]
    pub fn message(&self) -> &str {
        &self.message
    }

    #[inline]
    pub fn trace_id(&self) -> Option<&str> {
        self.trace_id.as_deref()
    }

    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    pub fn http_status(&self) -> u16 {
        match self.code {
            ErrorCode::Validation | ErrorCode::Conflict => 400,
            ErrorCode::NotFound => 404,
            ErrorCode::Forbidden => 403,
            ErrorCode::Storage | ErrorCode::Graph | ErrorCode::Config | ErrorCode::Internal => 500,
            ErrorCode::Embedding => 502,
        }
    }
}

impl fmt::Display for NovaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}", self.code, self.message)?;
        if let Some(src) = &self.source {
            write!(f, ": {src:#}")?;
        }
        Ok(())
    }
}

impl From<sqlx::Error> for NovaError {
    fn from(value: sqlx::Error) -> Self {
        match &value {
            sqlx::Error::RowNotFound => {
                Self::new(ErrorCode::NotFound, "record not found".into(), Some(value.into()))
            },
            sqlx::Error::Database(db) if db.is_unique_violation() || db.is_check_violation() => {
                Self::new(
                    ErrorCode::Conflict,
                    format!("database constraint: {}", db.message()),
                    Some(value.into()),
                )
            },
            _ => Self::storage(value),
        }
    }
}

impl From<reqwest::Error> for NovaError {
    fn from(value: reqwest::Error) -> Self {
        if value.is_timeout()
            || value.is_connect()
            || value.status().is_some_and(|s| s.is_server_error() || s.as_u16() == 429)
        {
            Self::embedding(format!("request failed: {value}"), value)
        } else {
            Self::embedding_msg(format!("request failed: {value}"))
        }
    }
}

impl From<anyhow::Error> for NovaError {
    fn from(value: anyhow::Error) -> Self {
        Self::new(ErrorCode::Internal, value.to_string(), Some(value))
    }
}

impl From<figment::Error> for NovaError {
    fn from(value: figment::Error) -> Self {
        Self::config(value)
    }
}

impl From<std::io::Error> for NovaError {
    fn from(value: std::io::Error) -> Self {
        Self::new(ErrorCode::Storage, value.to_string(), Some(value.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_stable() {
        assert_eq!(ErrorCode::Validation as i32, 100);
        assert_eq!(ErrorCode::NotFound as i32, 101);
        assert_eq!(ErrorCode::Conflict as i32, 102);
        assert_eq!(ErrorCode::Storage as i32, 200);
        assert_eq!(ErrorCode::Embedding as i32, 201);
        assert_eq!(ErrorCode::Internal as i32, 204);
    }

    #[test]
    fn display_includes_source() {
        let inner = anyhow::anyhow!("boom");
        let err = NovaError::storage(inner);
        let s = format!("{err}");
        assert!(s.contains("Storage"));
        assert!(s.contains("boom"));
    }

    #[test]
    fn http_status_mapping() {
        assert_eq!(NovaError::validation("bad").http_status(), 400);
        assert_eq!(NovaError::not_found("missing").http_status(), 404);
        assert_eq!(NovaError::internal("oops").http_status(), 500);
        assert_eq!(NovaError::embedding_msg("upstream").http_status(), 502);
    }
}
