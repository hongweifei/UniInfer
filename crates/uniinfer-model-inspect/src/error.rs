use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InspectError {
    #[error("model directory {0} does not exist")]
    NotFound(PathBuf),
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot parse {path}: {message}")]
    Config { path: PathBuf, message: String },
    #[error("no supported model files found in {0}")]
    Unsupported(PathBuf),
    #[error("invalid manifest: {0}")]
    Manifest(String),
}

pub type Result<T, E = InspectError> = std::result::Result<T, E>;
