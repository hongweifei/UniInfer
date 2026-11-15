use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot read config {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot write config {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot parse config {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("cannot serialize config: {0}")]
    Serialize(String),
    #[error("invalid config: {0}")]
    Invalid(String),
    #[error("config handle has no file path")]
    NoPath,
    #[error("config lock poisoned")]
    LockPoisoned,
}

pub type Result<T, E = ConfigError> = std::result::Result<T, E>;
