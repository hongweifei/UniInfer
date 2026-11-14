use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Error reply with a stable machine-readable code.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    pub fallback_reason: Option<String>,
}

impl ApiError {
    pub fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
            fallback_reason: None,
        }
    }

    pub fn not_found(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code,
            message: message.into(),
            fallback_reason: None,
        }
    }

    pub fn busy(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            code: "busy",
            message: message.into(),
            fallback_reason: None,
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal",
            message: message.into(),
            fallback_reason: None,
        }
    }
}

impl From<uniinfer_core::EngineError> for ApiError {
    fn from(error: uniinfer_core::EngineError) -> Self {
        use uniinfer_core::EngineError;
        let status = match &error {
            EngineError::UnsupportedFormat(_)
            | EngineError::UnsupportedQuantization(_)
            | EngineError::UnsupportedEngine(_)
            | EngineError::UnsupportedModality { .. }
            | EngineError::ManifestInvalid(_) => StatusCode::BAD_REQUEST,
            EngineError::InputTooLong { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            EngineError::MissingEp(_) => StatusCode::NOT_IMPLEMENTED,
            EngineError::InsufficientVram { .. } | EngineError::PlacementUnsatisfiable(_) => {
                StatusCode::CONFLICT
            }
            EngineError::Busy => StatusCode::TOO_MANY_REQUESTS,
            EngineError::Cancelled => StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            EngineError::Io(_) | EngineError::LoadFailed(_) | EngineError::Other(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        Self {
            status,
            code: error.code(),
            message: error.to_string(),
            fallback_reason: None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({
            "code": self.code,
            "message": self.message,
            "fallback_reason": self.fallback_reason,
        });
        (self.status, Json(body)).into_response()
    }
}
