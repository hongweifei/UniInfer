use crate::error::{ConfigError, Result};
use crate::schema::UniInferConfig;

pub(crate) fn apply(config: &mut UniInferConfig) -> Result<()> {
    if let Ok(value) = std::env::var("UNIINFER_ADDR") {
        config.server.addr = value
            .parse()
            .map_err(|error| ConfigError::Invalid(format!("UNIINFER_ADDR: {error}")))?;
    }
    if let Ok(value) = std::env::var("UNIINFER_QUEUE_TIMEOUT_MS") {
        config.server.queue_timeout_ms = value
            .parse()
            .map_err(|error| ConfigError::Invalid(format!("UNIINFER_QUEUE_TIMEOUT_MS: {error}")))?;
    }
    if let Ok(value) = std::env::var("UNIINFER_MAX_BODY_BYTES") {
        config.server.max_body_bytes = value
            .parse()
            .map_err(|error| ConfigError::Invalid(format!("UNIINFER_MAX_BODY_BYTES: {error}")))?;
    }
    if let Ok(value) = std::env::var("UNIINFER_MAX_LOADED_MODELS") {
        config.server.max_loaded_models = value.parse().map_err(|error| {
            ConfigError::Invalid(format!("UNIINFER_MAX_LOADED_MODELS: {error}"))
        })?;
    }
    if let Ok(value) = std::env::var("UNIINFER_TOKEN") {
        config.server.token = if value.is_empty() { None } else { Some(value) };
    }
    config.validate()
}
