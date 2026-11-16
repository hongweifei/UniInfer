use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::device::{DeviceClass, KvCachePlacement, PlacementPolicy};
use crate::error::{EngineError, Result};

/// Execution runtime selected declaratively by the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    Onnx,
    Candle,
    Llama,
    Whisper,
    Process,
}

/// Operation a capability performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Generate,
    Understand,
    Transcribe,
    Synthesize,
    Edit,
    Embed,
}

/// Execution mode of a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Batch,
    Stream,
    Duplex,
}

/// One model capability: operation, input contract, output contract and mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capability {
    pub name: String,
    pub operation: Operation,
    /// Inputs that must always be present.
    #[serde(default)]
    pub required_inputs: Vec<Modality>,
    /// Inputs that may accompany the required ones.
    #[serde(default)]
    pub optional_inputs: Vec<Modality>,
    pub outputs: Vec<Modality>,
    pub mode: Mode,
}

impl Capability {
    /// Match a concrete request against this capability.
    #[must_use]
    pub fn matches(&self, request: &CapabilityRequest) -> bool {
        if self.operation != request.operation || !self.mode.accepts(request.mode) {
            return false;
        }
        if self
            .required_inputs
            .iter()
            .any(|modality| !request.present_inputs.contains(modality))
        {
            return false;
        }
        if request.present_inputs.iter().any(|modality| {
            !self.required_inputs.contains(modality) && !self.optional_inputs.contains(modality)
        }) {
            return false;
        }
        if unique_len(&request.present_inputs) == 0 {
            return false;
        }
        let requested_outputs = unique_len(&request.requested_outputs);
        if requested_outputs == 0 {
            return false;
        }
        request
            .requested_outputs
            .iter()
            .all(|modality| self.outputs.contains(modality))
    }
}

fn unique_len(modalities: &[Modality]) -> usize {
    modalities.iter().copied().collect::<HashSet<_>>().len()
}

impl Mode {
    /// Batch requests may use a streaming model; streaming requests may not
    /// use a batch-only model. Duplex is exact.
    #[must_use]
    pub fn accepts(self, requested: Self) -> bool {
        match requested {
            Self::Batch => matches!(self, Self::Batch | Self::Stream | Self::Duplex),
            Self::Stream => matches!(self, Self::Stream | Self::Duplex),
            Self::Duplex => self == Self::Duplex,
        }
    }
}

/// Concrete request signature used for capability matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRequest {
    pub operation: Operation,
    pub present_inputs: Vec<Modality>,
    pub requested_outputs: Vec<Modality>,
    pub mode: Mode,
}

/// Model container formats understood by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Onnx,
    Gguf,
    Safetensors,
    /// Raw GGML/GGUF-style tensor bundles; the engine, not the format,
    /// decides which model families it can execute.
    Ggml,
}

impl ModelFormat {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Gguf => "gguf",
            Self::Safetensors => "safetensors",
            Self::Ggml => "ggml",
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
    #[serde(rename = "int4")]
    Int4,
    #[serde(rename = "int8")]
    Int8,
    #[serde(rename = "fp8")]
    Fp8,
    #[serde(rename = "q4_0")]
    Q4_0,
    #[serde(rename = "q4_1")]
    Q4_1,
    #[serde(rename = "q5_0")]
    Q5_0,
    #[serde(rename = "q5_1")]
    Q5_1,
    #[serde(rename = "q8_0")]
    Q8_0,
    #[serde(rename = "q2_k")]
    Q2K,
    #[serde(rename = "q3_k_s")]
    Q3KS,
    #[serde(rename = "q3_k_m")]
    Q3KM,
    #[serde(rename = "q3_k_l")]
    Q3KL,
    #[serde(rename = "q4_k_s")]
    Q4KS,
    #[serde(rename = "q4_k_m")]
    Q4KM,
    #[serde(rename = "q5_k_s")]
    Q5KS,
    #[serde(rename = "q5_k_m")]
    Q5KM,
    #[serde(rename = "q6_k")]
    Q6K,
    #[serde(rename = "q8_k")]
    Q8K,
    #[serde(rename = "q2_k_s")]
    Q2KS,
    #[serde(rename = "iq1_s")]
    Iq1S,
    #[serde(rename = "iq1_m")]
    Iq1M,
    #[serde(rename = "iq2_xxs")]
    Iq2Xxs,
    #[serde(rename = "iq2_xs")]
    Iq2Xs,
    #[serde(rename = "iq2_s")]
    Iq2S,
    #[serde(rename = "iq2_m")]
    Iq2M,
    #[serde(rename = "iq3_xxs")]
    Iq3Xxs,
    #[serde(rename = "iq3_xs")]
    Iq3Xs,
    #[serde(rename = "iq3_s")]
    Iq3S,
    #[serde(rename = "iq3_m")]
    Iq3M,
    #[serde(rename = "iq4_nl")]
    Iq4Nl,
    #[serde(rename = "iq4_xs")]
    Iq4Xs,
    #[serde(rename = "mxfp4")]
    Mxfp4,
    #[serde(rename = "nvfp4")]
    Nvfp4,
    #[serde(rename = "tq1_0")]
    Tq1_0,
    #[serde(rename = "tq2_0")]
    Tq2_0,
    #[serde(rename = "q1_0")]
    Q1_0,
    #[serde(rename = "q2_0")]
    Q2_0,
}

impl Quantization {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Int4 => "int4",
            Self::Int8 => "int8",
            Self::Fp8 => "fp8",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q2K => "q2_k",
            Self::Q3KS => "q3_k_s",
            Self::Q3KM => "q3_k_m",
            Self::Q3KL => "q3_k_l",
            Self::Q4KS => "q4_k_s",
            Self::Q4KM => "q4_k_m",
            Self::Q5KS => "q5_k_s",
            Self::Q5KM => "q5_k_m",
            Self::Q6K => "q6_k",
            Self::Q8K => "q8_k",
            Self::Q2KS => "q2_k_s",
            Self::Iq1S => "iq1_s",
            Self::Iq1M => "iq1_m",
            Self::Iq2Xxs => "iq2_xxs",
            Self::Iq2Xs => "iq2_xs",
            Self::Iq2S => "iq2_s",
            Self::Iq2M => "iq2_m",
            Self::Iq3Xxs => "iq3_xxs",
            Self::Iq3Xs => "iq3_xs",
            Self::Iq3S => "iq3_s",
            Self::Iq3M => "iq3_m",
            Self::Iq4Nl => "iq4_nl",
            Self::Iq4Xs => "iq4_xs",
            Self::Mxfp4 => "mxfp4",
            Self::Nvfp4 => "nvfp4",
            Self::Tq1_0 => "tq1_0",
            Self::Tq2_0 => "tq2_0",
            Self::Q1_0 => "q1_0",
            Self::Q2_0 => "q2_0",
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

impl Modality {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Audio => "audio",
            Self::Image => "image",
            Self::Video => "video",
        }
    }
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
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
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub config_file: Option<PathBuf>,
    #[serde(default)]
    pub tokenizer_file: Option<PathBuf>,
}

/// Vision / multimodal model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisionConfig {
    /// Vision adapter/family name.
    pub family: String,
    #[serde(default)]
    pub mmproj_file: Option<PathBuf>,
    #[serde(default)]
    pub image_token: Option<String>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub image_seq_len: Option<usize>,
    #[serde(default)]
    pub global_image_token_id: Option<u32>,
    #[serde(default)]
    pub fake_image_token_id: Option<u32>,
}

/// One step of an ONNX graph pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PipelineStep {
    Graph {
        name: String,
        file: PathBuf,
        #[serde(default)]
        inputs: BTreeMap<String, String>,
        #[serde(default)]
        auto: Vec<String>,
    },
    Merge {
        source: String,
        target: String,
        token: String,
    },
    Adapter {
        adapter: String,
    },
}

/// Declarative ONNX pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineConfig {
    #[serde(default)]
    pub image_preprocessor: Option<String>,
    #[serde(default)]
    pub preprocessor_file: Option<PathBuf>,
    #[serde(default)]
    pub token_layout: Option<String>,
    pub output: String,
    pub steps: Vec<PipelineStep>,
}

/// Image generation output contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageConfig {
    pub width: u32,
    pub height: u32,
}

/// Audio preprocessing contract shared by every speech adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_channels")]
    pub channels: u16,
    #[serde(default = "default_audio_input")]
    pub input: String,
    #[serde(default = "default_audio_output")]
    pub output: String,
    /// Audio adapter/family name.
    #[serde(default)]
    pub family: Option<String>,
    /// Adapter-owned configuration. Core does not interpret these keys.
    #[serde(default)]
    pub options: Option<toml::Value>,
}

fn default_sample_rate() -> u32 {
    16_000
}

fn default_channels() -> u16 {
    1
}

fn default_audio_input() -> String {
    "audio".into()
}

fn default_audio_output() -> String {
    "text".into()
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            channels: default_channels(),
            input: default_audio_input(),
            output: default_audio_output(),
            family: None,
            options: None,
        }
    }
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

/// Model manifest: capability, runtime, implementation and resource contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    pub id: String,
    pub runtime: Runtime,
    pub implementation: String,
    pub capabilities: Vec<Capability>,
    pub format: ModelFormat,
    pub quantization: Quantization,
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
    #[serde(default)]
    pub audio: Option<AudioConfig>,
    #[serde(default)]
    pub vision: Option<VisionConfig>,
    #[serde(default)]
    pub pipeline: Option<PipelineConfig>,
    #[serde(default)]
    pub image: Option<ImageConfig>,
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

    /// Render the manifest as canonical TOML.
    pub fn to_toml(&self) -> Result<String> {
        self.validate()?;
        toml::to_string_pretty(self)
            .map_err(|error| EngineError::ManifestInvalid(format!("serialize manifest: {error}")))
    }

    /// Validate and persist the manifest to its original path.
    pub fn save(&self, path: &Path) -> Result<()> {
        let raw = self.to_toml()?;
        std::fs::write(path, raw).map_err(|error| {
            EngineError::ManifestInvalid(format!("write {}: {error}", path.display()))
        })
    }

    /// Validate manifest invariants independent of the filesystem.
    pub fn validate(&self) -> Result<()> {
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
        if self.implementation.is_empty()
            || !self
                .implementation
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        {
            return Err(EngineError::ManifestInvalid(
                "implementation must be a non-empty [A-Za-z0-9._-] string".into(),
            ));
        }
        if self.capabilities.is_empty() {
            return Err(EngineError::ManifestInvalid("capabilities is empty".into()));
        }
        let mut names = std::collections::BTreeSet::new();
        for capability in &self.capabilities {
            if capability.name.is_empty()
                || !capability
                    .name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            {
                return Err(EngineError::ManifestInvalid(
                    "capability.name must be a non-empty [A-Za-z0-9._-] string".into(),
                ));
            }
            if !names.insert(&capability.name) {
                return Err(EngineError::ManifestInvalid(format!(
                    "duplicate capability {}",
                    capability.name
                )));
            }
            let mut inputs = HashSet::new();
            for modality in capability
                .required_inputs
                .iter()
                .chain(capability.optional_inputs.iter())
            {
                if !inputs.insert(*modality) {
                    return Err(EngineError::ManifestInvalid(format!(
                        "capability {} declares input {modality} twice",
                        capability.name
                    )));
                }
            }
            if inputs.is_empty() {
                return Err(EngineError::ManifestInvalid(format!(
                    "capability {} has no inputs",
                    capability.name
                )));
            }
            if capability.outputs.is_empty() {
                return Err(EngineError::ManifestInvalid(format!(
                    "capability {} has no outputs",
                    capability.name
                )));
            }
            let mut outputs = HashSet::new();
            for modality in &capability.outputs {
                if !outputs.insert(*modality) {
                    return Err(EngineError::ManifestInvalid(format!(
                        "capability {} declares output {modality} twice",
                        capability.name
                    )));
                }
            }
        }
        if self.implementation == "onnx.pipeline" {
            let pipeline = self.pipeline.as_ref().ok_or_else(|| {
                EngineError::ManifestInvalid("onnx.pipeline requires [pipeline]".into())
            })?;
            if pipeline.steps.is_empty() {
                return Err(EngineError::ManifestInvalid(
                    "pipeline.steps is empty".into(),
                ));
            }
            let mut names = std::collections::BTreeSet::new();
            for step in &pipeline.steps {
                match step {
                    PipelineStep::Graph { name, file, .. } => {
                        if name.is_empty() || !names.insert(name) {
                            return Err(EngineError::ManifestInvalid(
                                "pipeline graph step names must be unique and non-empty".into(),
                            ));
                        }
                        if file.is_absolute() {
                            return Err(EngineError::ManifestInvalid(
                                "pipeline graph path must be relative".into(),
                            ));
                        }
                    }
                    PipelineStep::Merge { source, target, .. } => {
                        if source == target {
                            return Err(EngineError::ManifestInvalid(
                                "pipeline merge source and target must differ".into(),
                            ));
                        }
                    }
                    PipelineStep::Adapter { adapter } => {
                        if adapter.is_empty() {
                            return Err(EngineError::ManifestInvalid(
                                "pipeline adapter step requires adapter".into(),
                            ));
                        }
                    }
                }
            }
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
        if let Some(image) = &self.image
            && (image.width == 0 || image.height == 0)
        {
            return Err(EngineError::ManifestInvalid(
                "image dimensions must be positive".into(),
            ));
        }
        if let Some(audio) = &self.audio {
            if !(8_000..=192_000).contains(&audio.sample_rate) {
                return Err(EngineError::ManifestInvalid(
                    "audio.sample_rate must be within 8000..=192000".into(),
                ));
            }
            if audio.channels != 1 {
                return Err(EngineError::ManifestInvalid(
                    "audio.channels currently must be 1".into(),
                ));
            }
            if audio.input.is_empty() || audio.output.is_empty() {
                return Err(EngineError::ManifestInvalid(
                    "audio tensor names must not be empty".into(),
                ));
            }
            if audio.family.as_deref() == Some("") {
                return Err(EngineError::ManifestInvalid(
                    "audio.family must not be empty".into(),
                ));
            }
        }
        Ok(())
    }

    /// Whether any declared capability satisfies the request.
    #[must_use]
    pub fn supports(&self, request: &CapabilityRequest) -> bool {
        self.capabilities
            .iter()
            .any(|capability| capability.matches(request))
    }

    /// Union of required and optional input modalities.
    #[must_use]
    pub fn input_modalities(&self) -> Vec<Modality> {
        let mut out = Vec::new();
        for capability in &self.capabilities {
            for modality in capability
                .required_inputs
                .iter()
                .chain(capability.optional_inputs.iter())
            {
                if !out.contains(modality) {
                    out.push(*modality);
                }
            }
        }
        out
    }

    /// Union of output modalities.
    #[must_use]
    pub fn output_modalities(&self) -> Vec<Modality> {
        let mut out = Vec::new();
        for capability in &self.capabilities {
            for modality in &capability.outputs {
                if !out.contains(modality) {
                    out.push(*modality);
                }
            }
        }
        out
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
id = "tiny-fixture"
runtime = "onnx"
implementation = "onnx.graph"
format = "onnx"
quantization = "fp32"
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

[[capabilities]]
name = "text_chat"
operation = "generate"
required_inputs = ["text"]
outputs = ["text"]
mode = "stream"
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
    fn parses_all_supported_quantizations() {
        for value in [
            "fp32", "fp16", "bf16", "int4", "int8", "fp8", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
            "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_k_s", "q4_k_m", "q5_k_s", "q5_k_m", "q6_k",
        ] {
            let raw = base().replace(
                "quantization = \"fp32\"",
                &format!("quantization = \"{value}\""),
            );
            let manifest = parse(&raw);
            assert_eq!(manifest.quantization.as_str(), value);
        }
    }

    #[test]
    fn rejects_unknown_quantization_at_parse_time() {
        let raw = base().replace("quantization = \"fp32\"", "quantization = \"q9_k\"");
        assert!(toml::from_str::<ModelManifest>(&raw).is_err());
    }

    #[test]
    fn parses_audio_contract() {
        let raw = base().replace(
            "required_inputs = [\"text\"]",
            "required_inputs = [\"audio\"]",
        ) + "\n[audio]\nsample_rate = 16000\nchannels = 1\n";
        let manifest = parse(&raw);
        manifest.validate().expect("valid");
        assert_eq!(manifest.format, ModelFormat::Onnx);
        let audio = manifest.audio.expect("audio");
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.input, "audio");
        assert_eq!(audio.output, "text");
    }

    #[test]
    fn rejects_stereo_audio_contract() {
        let raw = base().replace(
            "required_inputs = [\"text\"]",
            "required_inputs = [\"audio\"]",
        ) + "\n[audio]\nchannels = 2\n";
        let error = parse(&raw).validate().expect_err("must reject");
        assert!(error.to_string().contains("channels"));
    }

    #[test]
    fn rejects_empty_capabilities() {
        let raw = base()
            .split("[[capabilities]]")
            .next()
            .expect("base")
            .replace("[placement]", "capabilities = []\n[placement]");
        let error = parse(&raw).validate().expect_err("must reject");
        assert!(error.to_string().contains("capabilities"));
    }

    #[test]
    fn rejects_empty_implementation() {
        let raw = base().replace("implementation = \"onnx.graph\"", "implementation = \"\"");
        let error = parse(&raw).validate().expect_err("must reject");
        assert!(error.to_string().contains("implementation"));
    }

    #[test]
    fn capability_keeps_optional_text_separate_from_required_audio() {
        let capability = Capability {
            name: "speech_to_speech".into(),
            operation: Operation::Generate,
            required_inputs: vec![Modality::Audio],
            optional_inputs: vec![Modality::Text],
            outputs: vec![Modality::Audio],
            mode: Mode::Batch,
        };
        let audio_only = CapabilityRequest {
            operation: Operation::Generate,
            present_inputs: vec![Modality::Audio],
            requested_outputs: vec![Modality::Audio],
            mode: Mode::Batch,
        };
        let with_text = CapabilityRequest {
            present_inputs: vec![Modality::Audio, Modality::Text],
            ..audio_only.clone()
        };
        let text_only = CapabilityRequest {
            present_inputs: vec![Modality::Text],
            ..audio_only.clone()
        };
        assert!(capability.matches(&audio_only));
        assert!(capability.matches(&with_text));
        assert!(!capability.matches(&text_only));
    }

    #[test]
    fn image_to_video_capability_matches_image_and_optional_text() {
        let capability = Capability {
            name: "image_to_video".into(),
            operation: Operation::Generate,
            required_inputs: vec![Modality::Image],
            optional_inputs: vec![Modality::Text],
            outputs: vec![Modality::Video],
            mode: Mode::Batch,
        };
        let request = CapabilityRequest {
            operation: Operation::Generate,
            present_inputs: vec![Modality::Image, Modality::Text],
            requested_outputs: vec![Modality::Video],
            mode: Mode::Batch,
        };
        assert!(capability.matches(&request));
    }

    #[test]
    fn accepts_optional_only_inputs_with_output_subsets() {
        let raw = base()
            .replace(
                "required_inputs = [\"text\"]",
                "required_inputs = []\noptional_inputs = [\"text\", \"audio\", \"image\"]",
            )
            .replace(
                "outputs = [\"text\"]",
                "outputs = [\"text\", \"audio\", \"image\"]\n",
            );
        let manifest = parse(&raw);
        manifest.validate().expect("optional-only capability");
        assert!(manifest.supports(&CapabilityRequest {
            operation: Operation::Generate,
            present_inputs: vec![Modality::Audio],
            requested_outputs: vec![Modality::Image, Modality::Text],
            mode: Mode::Batch,
        }));
        assert!(manifest.supports(&CapabilityRequest {
            operation: Operation::Generate,
            present_inputs: vec![Modality::Audio],
            requested_outputs: vec![Modality::Text],
            mode: Mode::Batch,
        }));
        assert!(!manifest.supports(&CapabilityRequest {
            operation: Operation::Generate,
            present_inputs: vec![Modality::Video],
            requested_outputs: vec![Modality::Text],
            mode: Mode::Batch,
        }));
        assert!(!manifest.supports(&CapabilityRequest {
            operation: Operation::Generate,
            present_inputs: vec![],
            requested_outputs: vec![Modality::Text],
            mode: Mode::Batch,
        }));
    }

    #[test]
    fn rejects_empty_inputs() {
        let raw = base().replace("required_inputs = [\"text\"]", "required_inputs = []");
        let error = parse(&raw).validate().expect_err("must reject");
        assert!(error.to_string().contains("no inputs"));
    }

    #[test]
    fn rejects_duplicate_inputs() {
        let raw = base().replace(
            "required_inputs = [\"text\"]",
            "required_inputs = [\"text\", \"text\"]",
        );
        let error = parse(&raw).validate().expect_err("must reject");
        assert!(error.to_string().contains("twice"));
    }

    #[test]
    fn rejects_duplicate_outputs() {
        let raw = base().replace("outputs = [\"text\"]", "outputs = [\"text\", \"text\"]");
        let error = parse(&raw).validate().expect_err("must reject");
        assert!(error.to_string().contains("twice"));
    }
}
