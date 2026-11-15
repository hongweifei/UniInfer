//! Typed application configuration shared by the CLI, API and future GUI.
//!
//! The schema contains only durable user choices. Runtime objects (loaded
//! models, ORT sessions, devices) are deliberately not serialized here.

mod environment;
mod error;
mod format;
mod handle;
mod schema;

pub use error::{ConfigError, Result};
pub use handle::ConfigHandle;
pub use schema::{ModelEntry, SCHEMA_VERSION, ServerConfig, UniInferConfig};

impl UniInferConfig {
    /// Apply `UNIINFER_*` server overrides after file/CLI values are loaded.
    pub fn apply_env(&mut self) -> Result<()> {
        environment::apply(self)
    }
}
