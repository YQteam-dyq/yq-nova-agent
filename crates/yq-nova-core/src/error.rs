//! Unified error type for yq-nova-core.
//!
//! All fallible public APIs return [`NovaResult<T>`], which is a thin
//! alias for `Result<T, NovaError>`. [`NovaError`] is intentionally
//! cheap-to-clone and structured so that HTTP endpoints can easily
//! map each variant to an HTTP status code + error code.

use std::fmt;

use serde::Serialize;

/// Short-hand result alias used across the whole crate.
pub type NovaResult<T> = Result<T, NovaError>;

/// Stable machine-readable error codes.
///
/// HTTP handlers map these to status codes, so new variants should
/// be added sparingly and with a clear mapping in mind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ErrorCode {
    /// Input failed validation (HTTP 400).
    Validation = 100,
    /// Requested record does not exist (HTTP 404).
    NotFound = 101,
    /// Conflict, e.g. duplicate on an unique field when the caller
    /// did not ask for upsert (HTTP 409).
    Conflict = 102,
    /// Missing or invalid credentials, permission denied (HTTP 403).
    Forbidden = 103,
    /// Storage/SQLite error (HTTP 500).
    Storage = 200,
    /// Error from embedding provider (HTTP 502/500 depending on cause).
    Embedding = 201,
    /// Graph traversal error.
    Graph = 202,
    /// Bad configuration (at startup; HTTP 500 if caught at runtime).
    Config = 203,
    /// Anything else / internal invariant violation (HTTP 500).
    Internal = 204,
}

impl ErrorCode {
    /// Stable machine-readable string, e.g. "validation" / "not_found".
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

/// The main error type.
///
/// The source error is stored as an `anyhow::Error` so we retain a
/// full chain of context, but [`ErrorCode`] and the top-level message
/// are enough for stable matching.
#[derive(Debug, thiserror::Error)]
pub struct NovaError {
    code: ErrorCode,
    message: String,
    #[source]
    source: Option<anyhow::Error>,
    /// Populated by the HTTP layer; not serialised to clients.
    trace_id: Option<String>,
}

impl NovaError {
    // ----------------------------------------------------------------
    // Constructors — one per ErrorCode.
    // ----------------------------------------------------------------

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

    // ----------------------------------------------------------------
    // Helpers.
    // ----------------------------------------------------------------

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

    /// Suggested HTTP status code for this error.
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

// --------------------------------------------------------------------
// Convenience From impls for common error types. These map to the
// closest semantic variant; callers that need precise mapping should
// use the explicit constructors above.
// --------------------------------------------------------------------

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
