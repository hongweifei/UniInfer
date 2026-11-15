//! Whisper generation config parsed from config.json.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use uniinfer_core::{AudioConfig, EngineError, Result};
#[derive(Debug, Clone, Deserialize)]
pub(super) struct WhisperConfig {
    pub(super) decoder_start_token_id: u32,
    pub(super) eos_token_id: u32,
    #[serde(default)]
    pub(super) forced_decoder_ids: Vec<[u32; 2]>,
    #[serde(default)]
    pub(super) max_length: Option<usize>,
}

impl WhisperConfig {
    pub(super) fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|error| {
            EngineError::ManifestInvalid(format!("cannot read {}: {error}", path.display()))
        })?;
        let config: Self = serde_json::from_str(&raw).map_err(|error| {
            EngineError::ManifestInvalid(format!("{}: {error}", path.display()))
        })?;
        Ok(config)
    }

    pub(super) fn initial_ids(&self) -> Vec<u32> {
        let mut ids = vec![self.decoder_start_token_id];
        ids.extend(self.forced_decoder_ids.iter().map(|pair| pair[1]));
        ids
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WhisperOptions {
    pub(super) encoder_file: PathBuf,
    pub(super) decoder_file: PathBuf,
    #[serde(default)]
    pub(super) decoder_with_past_file: Option<PathBuf>,
    pub(super) tokenizer_file: PathBuf,
    pub(super) config_file: PathBuf,
    #[serde(default)]
    pub(super) max_tokens: Option<usize>,
}

pub(super) fn options(audio: &AudioConfig) -> Result<WhisperOptions> {
    super::super::options::parse(audio)
}
