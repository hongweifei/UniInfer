use std::path::Path;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{ConfigError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigFormat {
    Toml,
    Json,
}

impl ConfigFormat {
    pub(crate) fn for_path(path: &Path) -> Self {
        match path.extension().and_then(|value| value.to_str()) {
            Some("json") => Self::Json,
            _ => Self::Toml,
        }
    }

    pub(crate) fn parse<T: DeserializeOwned>(self, raw: &str, path: &Path) -> Result<T> {
        let result = match self {
            Self::Toml => toml::from_str(raw).map_err(|error| error.to_string()),
            Self::Json => serde_json::from_str(raw).map_err(|error| error.to_string()),
        };
        result.map_err(|message| ConfigError::Parse {
            path: path.to_path_buf(),
            message,
        })
    }

    pub(crate) fn serialize<T: Serialize>(self, value: &T) -> Result<String> {
        let result = match self {
            Self::Toml => toml::to_string_pretty(value).map_err(|error| error.to_string()),
            Self::Json => serde_json::to_string_pretty(value).map_err(|error| error.to_string()),
        };
        result.map_err(ConfigError::Serialize)
    }
}
