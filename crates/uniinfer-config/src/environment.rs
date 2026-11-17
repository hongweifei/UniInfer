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
    if let Ok(value) = std::env::var("UNIINFER_REQUEST_TIMEOUT_MS") {
        config.server.request_timeout_ms = value.parse().map_err(|error| {
            ConfigError::Invalid(format!("UNIINFER_REQUEST_TIMEOUT_MS: {error}"))
        })?;
    }
    if let Ok(value) = std::env::var("UNIINFER_MODEL_LOAD_TIMEOUT_MS") {
        config.server.model_load_timeout_ms = value.parse().map_err(|error| {
            ConfigError::Invalid(format!("UNIINFER_MODEL_LOAD_TIMEOUT_MS: {error}"))
        })?;
    }
    if let Ok(value) = std::env::var("UNIINFER_VRAM_RESERVE_MB") {
        config.server.vram_reserve_mb = value
            .parse()
            .map_err(|error| ConfigError::Invalid(format!("UNIINFER_VRAM_RESERVE_MB: {error}")))?;
    }
    if let Ok(value) = std::env::var("UNIINFER_HOST_RESERVE_MB") {
        config.server.host_reserve_mb = value
            .parse()
            .map_err(|error| ConfigError::Invalid(format!("UNIINFER_HOST_RESERVE_MB: {error}")))?;
    }
    if let Ok(value) = std::env::var("UNIINFER_BACKEND_DIR") {
        config.server.backend_dir = if value.is_empty() {
            None
        } else {
            Some(std::path::PathBuf::from(value))
        };
    }
    if let Ok(value) = std::env::var("UNIINFER_BACKEND_CATALOG_URL") {
        config.server.backend_catalog_url = if value.is_empty() { None } else { Some(value) };
    }
    if let Ok(value) = std::env::var("UNIINFER_ISOLATE_MODELS") {
        config.server.isolate_models = matches!(value.as_str(), "1" | "true" | "yes");
    }
    if let Ok(value) = std::env::var("UNIINFER_TOKEN") {
        config.server.token = if value.is_empty() { None } else { Some(value) };
    }
    config.validate()
}
