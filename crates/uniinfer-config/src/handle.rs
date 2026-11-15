use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::error::{ConfigError, Result};
use crate::schema::UniInferConfig;

/// Thread-safe current config plus its optional source file.
pub struct ConfigHandle {
    path: Option<PathBuf>,
    config: RwLock<UniInferConfig>,
}

impl fmt::Debug for ConfigHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConfigHandle")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl ConfigHandle {
    #[must_use]
    pub fn new(config: UniInferConfig, path: Option<PathBuf>) -> Self {
        Self {
            path,
            config: RwLock::new(config),
        }
    }

    pub fn load(path: PathBuf) -> Result<Self> {
        let config = UniInferConfig::load(&path)?;
        Ok(Self::new(config, Some(path)))
    }

    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    #[must_use]
    pub fn base_dir(&self) -> PathBuf {
        self.path
            .as_deref()
            .and_then(Path::parent)
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    }

    pub fn snapshot(&self) -> Result<UniInferConfig> {
        self.config
            .read()
            .map(|config| config.clone())
            .map_err(|_| ConfigError::LockPoisoned)
    }

    /// Validate, persist to the source file, then publish the new snapshot.
    pub fn replace(&self, next: UniInferConfig) -> Result<()> {
        next.validate()?;
        let path = self.path.as_deref().ok_or(ConfigError::NoPath)?;
        let mut current = self.config.write().map_err(|_| ConfigError::LockPoisoned)?;
        next.save(path)?;
        *current = next;
        Ok(())
    }
}
