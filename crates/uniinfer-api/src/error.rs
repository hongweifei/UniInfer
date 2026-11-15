use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use uniinfer_config::ConfigError;
use uniinfer_model_inspect::InspectError;
use uniinfer_router::RouterError;

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

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized",
            message: message.into(),
            fallback_reason: None,
        }
    }

    pub fn forbidden(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code,
            message: message.into(),
            fallback_reason: None,
        }
    }

    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            code: "not_implemented",
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
            | EngineError::UnsupportedArchitecture { .. }
            | EngineError::UnsupportedModality { .. }
            | EngineError::ManifestInvalid(_)
            | EngineError::InvalidAudio(_)
            | EngineError::InvalidSampling(_) => StatusCode::BAD_REQUEST,
            EngineError::InputTooLong { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            EngineError::MissingEp(_) => StatusCode::NOT_IMPLEMENTED,
            EngineError::InsufficientVram { .. } | EngineError::PlacementUnsatisfiable(_) => {
                StatusCode::CONFLICT
            }
            EngineError::Busy => StatusCode::TOO_MANY_REQUESTS,
            EngineError::Cancelled => StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            EngineError::Timeout => StatusCode::GATEWAY_TIMEOUT,
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

impl From<ConfigError> for ApiError {
    fn from(error: ConfigError) -> Self {
        let status = match &error {
            ConfigError::Invalid(_) | ConfigError::Parse { .. } => StatusCode::BAD_REQUEST,
            ConfigError::Read { .. }
            | ConfigError::Write { .. }
            | ConfigError::Serialize(_)
            | ConfigError::NoPath
            | ConfigError::LockPoisoned => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            code: "config_error",
            message: error.to_string(),
            fallback_reason: None,
        }
    }
}

impl From<InspectError> for ApiError {
    fn from(error: InspectError) -> Self {
        let status = match &error {
            InspectError::NotFound(_) | InspectError::Unsupported(_) => StatusCode::NOT_FOUND,
            InspectError::Config { .. } | InspectError::Manifest(_) | InspectError::Gguf(_) => {
                StatusCode::BAD_REQUEST
            }
            InspectError::Io { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            code: "model_inspect_error",
            message: error.to_string(),
            fallback_reason: None,
        }
    }
}

impl From<RouterError> for ApiError {
    fn from(error: RouterError) -> Self {
        match error {
            RouterError::ModelNotFound(model) => {
                Self::not_found("model_not_found", format!("model {model} is not loaded"))
            }
            RouterError::ContextOverflow { tokens, limit } => Self::bad_request(
                "context_overflow",
                format!("prompt uses {tokens} tokens, context_len is {limit}"),
            ),
            RouterError::InvalidMaxTokens => {
                Self::bad_request("invalid_max_tokens", "max_tokens resolves to zero")
            }
            RouterError::Busy => Self::busy("model is serving another request"),
            RouterError::Engine(error) => error.into(),
        }
    }
}
