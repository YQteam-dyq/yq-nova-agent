//! HTTP 层统一错误格式。
//!
//! 所有 API 错误都会序列化为 JSON：
//! ```json
//! {"code": "validation", "message": "...", "trace_id": "abc123"}
//! ```
//!
//! 非 2xx 状态码由 NovaError 的 error_code 映射而来。
//!
//! 注意：`NovaError` 定义在 yq-nova-core 里，`IntoResponse` 定义在 axum 里。
//! 按 Rust 孤儿规则，server crate 不能直接为外部类型 impl 外部 trait。
//! 我们用一个轻量 newtype `AppError`（包装 `NovaError`）来实现转换。

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use tracing::warn;
use yq_nova_core::error::{ErrorCode, NovaError};

pub struct AppError(pub NovaError);

impl From<NovaError> for AppError {
    fn from(e: NovaError) -> Self {
        Self(e)
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorBody<'a> {
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<&'a str>,
}

fn code_to_status(code: ErrorCode) -> StatusCode {
    match code {
        ErrorCode::Validation => StatusCode::BAD_REQUEST,
        ErrorCode::NotFound => StatusCode::NOT_FOUND,
        ErrorCode::Conflict => StatusCode::CONFLICT,
        ErrorCode::Forbidden => StatusCode::FORBIDDEN,
        ErrorCode::Storage
        | ErrorCode::Embedding
        | ErrorCode::Graph
        | ErrorCode::Config
        | ErrorCode::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let AppError(inner) = self;
        let status = code_to_status(inner.code());
        let body = ErrorBody {
            code: inner.code().as_str(),
            message: inner.to_string(),
            trace_id: inner.trace_id(),
        };
        if status.is_server_error() {
            warn!(error = %inner, %status, "http 5xx response");
        }
        (status, Json(body)).into_response()
    }
}

/// 404 / 405 fallback: never return HTML (P6-4 统一 JSON 错误格式)。
pub async fn fallback_404() -> impl IntoResponse {
    let body = ErrorBody {
        code: ErrorCode::NotFound.as_str(),
        message: "route not found (check method + path)".to_string(),
        trace_id: None,
    };
    (StatusCode::NOT_FOUND, Json(body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_mapping_edges() {
        assert_eq!(code_to_status(ErrorCode::Validation), StatusCode::BAD_REQUEST);
        assert_eq!(code_to_status(ErrorCode::NotFound), StatusCode::NOT_FOUND);
        assert_eq!(code_to_status(ErrorCode::Conflict), StatusCode::CONFLICT);
        assert_eq!(code_to_status(ErrorCode::Forbidden), StatusCode::FORBIDDEN);
        assert_eq!(code_to_status(ErrorCode::Internal), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(code_to_status(ErrorCode::Storage), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(code_to_status(ErrorCode::Embedding), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(code_to_status(ErrorCode::Graph), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(code_to_status(ErrorCode::Config), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn app_error_converts_nova_error_into_json() {
        let e = NovaError::validation("bad input");
        let resp: Response = AppError(e).into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
