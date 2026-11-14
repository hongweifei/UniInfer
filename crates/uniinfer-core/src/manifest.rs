use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::device::{DeviceClass, KvCachePlacement, PlacementPolicy};
use crate::error::{EngineError, Result};

/// Model container formats understood by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Onnx,
    Gguf,
    Safetensors,
}

impl ModelFormat {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Gguf => "gguf",
            Self::Safetensors => "safetensors",
        }
    }
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Quantization schemes; engines reject the ones they cannot execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Quantization {
    #[serde(rename = "fp32")]
    Fp32,
    #[serde(rename = "fp16")]
    Fp16,
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "int8")]
    Int8,
    #[serde(rename = "fp8")]
    Fp8,
    #[serde(rename = "q4_0")]
    Q4_0,
    #[serde(rename = "q4_k_m")]
    Q4KM,
    #[serde(rename = "q5_k_m")]
    Q5KM,
    #[serde(rename = "q6_k")]
    Q6K,
    #[serde(rename = "q8_0")]
    Q8_0,
}

impl Quantization {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
            Self::Fp8 => "fp8",
            Self::Q4_0 => "q4_0",
            Self::Q4KM => "q4_k_m",
            Self::Q5KM => "q5_k_m",
            Self::Q6K => "q6_k",
            Self::Q8_0 => "q8_0",
        }
    }
}

impl fmt::Display for Quantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Input or output channel of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Audio,
    Image,
    Video,
}

/// One weight file with an integrity digest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelFile {
    pub path: PathBuf,
    pub sha256: String,
}

impl ModelFile {
    fn verify(&self, base_dir: &Path) -> Result<()> {
        let path = base_dir.join(&self.path);
        let mut file = File::open(&path).map_err(|error| {
            EngineError::ManifestInvalid(format!("cannot open {}: {error}", path.display()))
        })?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual = hex::encode(hasher.finalize());
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(EngineError::ManifestInvalid(format!(
                "sha256 mismatch for {}: expected {}, got {actual}",
                path.display(),
                self.sha256
            )));
        }
        Ok(())
    }
}

/// Text-generation specific model parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextConfig {
    pub vocab_size: usize,
    #[serde(default)]
    pub eos_token: Option<u32>,
}

/// Placement directives copied from the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementConfig {
    #[serde(default)]
    pub policy: PlacementPolicy,
    #[serde(default = "default_allow_cpu_fallback")]
    pub allow_cpu_fallback: bool,
    #[serde(default)]
    pub prefer: Vec<DeviceClass>,
    #[serde(default)]
    pub stages: BTreeMap<String, DeviceClass>,
    #[serde(default)]
    pub offload_layers: Option<String>,
    #[serde(default)]
    pub kv_cache: Option<KvCachePlacement>,
}

fn default_allow_cpu_fallback() -> bool {
    true
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            policy: PlacementPolicy::Auto,
            allow_cpu_fallback: true,
            prefer: Vec::new(),
            stages: BTreeMap::new(),
            offload_layers: None,
            kv_cache: None,
        }
    }
}

/// Model manifest, schema version 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub id: String,
    pub format: ModelFormat,
    pub quantization: Quantization,
    pub modalities: Vec<Modality>,
    #[serde(default)]
    pub streaming_capable: bool,
    pub context_len: usize,
    #[serde(default)]
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub min_vram: Option<String>,
    #[serde(default)]
    pub placement: PlacementConfig,
    pub files: Vec<ModelFile>,
    #[serde(default)]
    pub text: Option<TextConfig>,
}

impl ModelManifest {
    /// Parse and validate a manifest from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|error| {
            EngineError::ManifestInvalid(format!("cannot read {}: {error}", path.display()))
        })?;
        let manifest: Self = toml::from_str(&raw).map_err(|error| {
            EngineError::ManifestInvalid(format!("{}: {error}", path.display()))
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate schema invariants independent of the filesystem.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(EngineError::ManifestInvalid(format!(
                "unsupported schema_version {}, expected 1",
                self.schema_version
            )));
        }
        if self.id.is_empty()
            || !self
                .id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        {
            return Err(EngineError::ManifestInvalid(
                "id must be a non-empty [A-Za-z0-9._-] string".into(),
            ));
        }
        if self.modalities.is_empty() {
            return Err(EngineError::ManifestInvalid("modalities is empty".into()));
        }
        if self.context_len == 0 || self.context_len > 1_048_576 {
            return Err(EngineError::ManifestInvalid(
                "context_len must be within 1..=1048576".into(),
            ));
        }
        if self.files.is_empty() {
            return Err(EngineError::ManifestInvalid("files is empty".into()));
        }
        for file in &self.files {
            if file.path.is_absolute()
                || file.path.components().any(|c| {
                    matches!(
                        c,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
            {
                return Err(EngineError::ManifestInvalid(format!(
                    "file path must stay inside the model directory: {}",
                    file.path.display()
                )));
            }
            if file.sha256.len() != 64 || !file.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(EngineError::ManifestInvalid(format!(
                    "sha256 must be 64 hex characters: {}",
                    file.path.display()
                )));
            }
        }
        if let Some(text) = &self.text {
            if text.vocab_size == 0 || text.vocab_size > 1_048_576 {
                return Err(EngineError::ManifestInvalid(
                    "text.vocab_size must be within 1..=1048576".into(),
                ));
            }
        }
        Ok(())
    }

    /// Verify every listed weight file against its digest.
    pub fn verify_files(&self, base_dir: &Path) -> Result<()> {
        for file in &self.files {
            file.verify(base_dir)?;
        }
        Ok(())
    }

    /// Directory containing the manifest's weight files.
    #[must_use]
    pub fn base_dir(manifest_path: &Path) -> PathBuf {
        manifest_path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> String {
        r#"
schema_version = 1
id = "tiny-fixture"
format = "onnx"
quantization = "fp32"
modalities = ["text"]
context_len = 64
tokenizer = "bytes"
[placement]
policy = "auto"
stages.llm = "cpu"

[text]
vocab_size = 256

[[files]]
path = "model.onnx"
sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
"#
        .into()
    }

    fn parse(raw: &str) -> ModelManifest {
        toml::from_str(raw).expect("parse")
    }

    #[test]
    fn accepts_valid_manifest() {
        let manifest = parse(&base());
        manifest.validate().expect("valid");
        assert_eq!(manifest.format, ModelFormat::Onnx);
        assert_eq!(manifest.placement.stages["llm"], DeviceClass::Cpu);
    }

    #[test]
    fn rejects_parent_directory_paths() {
        let raw = base().replace("path = \"model.onnx\"", "path = \"../model.onnx\"");
        let error = parse(&raw).validate().expect_err("must reject");
        assert!(error.to_string().contains("model directory"));
    }

    #[test]
    fn rejects_bad_digest_length() {
        let raw = base().replace(
            "sha256 = \"0000000000000000000000000000000000000000000000000000000000000000\"",
            "sha256 = \"abcd\"",
        );
        let error = parse(&raw).validate().expect_err("must reject");
        assert!(error.to_string().contains("64 hex"));
    }

    #[test]
    fn rejects_unknown_quantization_at_parse_time() {
        let raw = base().replace("quantization = \"fp32\"", "quantization = \"q9_k\"");
        assert!(toml::from_str::<ModelManifest>(&raw).is_err());
    }

    #[test]
    fn rejects_empty_modalities() {
        let raw = base().replace("modalities = [\"text\"]", "modalities = []");
        let error = parse(&raw).validate().expect_err("must reject");
        assert!(error.to_string().contains("modalities"));
    }
}
