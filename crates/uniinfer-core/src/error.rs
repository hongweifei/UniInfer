use thiserror::Error;

use crate::manifest::{Modality, ModelFormat, Quantization};

/// Errors surfaced by engines and the registry.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("unsupported format: {0}")]
    UnsupportedFormat(ModelFormat),
    #[error("unsupported quantization: {0}")]
    UnsupportedQuantization(Quantization),
    #[error("unsupported engine: {0}")]
    UnsupportedEngine(String),
    #[error("unsupported architecture {architecture} for format {format}")]
    UnsupportedArchitecture {
        format: ModelFormat,
        architecture: String,
    },
    #[error("unsupported modality {modality:?} for format {format}")]
    UnsupportedModality {
        format: ModelFormat,
        modality: Modality,
    },
    #[error("input too long: {input} exceeds limit {limit}")]
    InputTooLong { input: usize, limit: usize },
    #[error("missing execution provider: {0}")]
    MissingEp(String),
    #[error("insufficient memory: required {required}, available {available}")]
    InsufficientVram { required: String, available: String },
    #[error("invalid manifest: {0}")]
    ManifestInvalid(String),
    #[error("model load failed: {0}")]
    LoadFailed(String),
    #[error("cancelled")]
    Cancelled,
    #[error("request timed out")]
    Timeout,
    #[error("resource busy")]
    Busy,
    #[error("placement unsatisfiable: {0}")]
    PlacementUnsatisfiable(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid audio: {0}")]
    InvalidAudio(String),
    #[error("invalid sampling: {0}")]
    InvalidSampling(String),
    #[error("engine error: {0}")]
    Other(String),
}

impl EngineError {
    /// Stable machine-readable code used by the API error envelope.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedFormat(_) => "unsupported_format",
            Self::UnsupportedQuantization(_) => "unsupported_quantization",
            Self::UnsupportedEngine(_) => "unsupported_engine",
            Self::UnsupportedArchitecture { .. } => "unsupported_architecture",
            Self::UnsupportedModality { .. } => "unsupported_modality",
            Self::InputTooLong { .. } => "input_too_long",
            Self::MissingEp(_) => "missing_ep",
            Self::InsufficientVram { .. } => "insufficient_vram",
            Self::ManifestInvalid(_) => "manifest_invalid",
            Self::LoadFailed(_) => "load_failed",
            Self::Cancelled => "cancelled",
            Self::Timeout => "timeout",
            Self::Busy => "busy",
            Self::PlacementUnsatisfiable(_) => "placement_unsatisfiable",
            Self::Io(_) => "io_error",
            Self::InvalidAudio(_) => "invalid_audio",
            Self::InvalidSampling(_) => "invalid_sampling",
            Self::Other(_) => "engine_error",
        }
    }
}

pub type Result<T, E = EngineError> = std::result::Result<T, E>;
