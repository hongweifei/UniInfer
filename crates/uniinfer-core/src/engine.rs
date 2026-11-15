use std::path::PathBuf;
use std::time::Instant;

use crate::cancel::CancelToken;
use crate::device::{DeviceBudget, DeviceClass, PlacementPlan};
use crate::error::Result;
use crate::manifest::{Modality, ModelFormat, ModelManifest, Quantization};
use crate::tensor::{Tensor, TensorSpec};

/// Static capability matrix reported by an engine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineCapabilities {
    pub formats: Vec<ModelFormat>,
    pub modalities: Vec<(ModelFormat, Vec<Modality>)>,
    pub quantizations: Vec<(ModelFormat, Vec<Quantization>)>,
    pub devices: Vec<DeviceClass>,
}

/// Everything an engine needs to load one model.
#[derive(Debug, Clone)]
pub struct EngineLoadSpec {
    pub manifest: ModelManifest,
    pub base_dir: PathBuf,
    /// On-disk manifest path, when the load originated from one.
    pub manifest_path: Option<PathBuf>,
    pub placement: PlacementPlan,
    pub budget: DeviceBudget,
    /// Cooperative cancellation checked by engines during load.
    pub cancel: CancelToken,
}

/// Stop reason reported by an engine that decodes tokens internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeFinish {
    Eos,
    Length,
    SinkClosed,
}

/// Request for engines that own tokenization and decoding (llama.cpp, etc.).
#[derive(Debug, Clone, Copy)]
pub struct NativeTextRequest<'a> {
    pub prompt: &'a str,
    pub images: &'a [Vec<u8>],
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: Option<u64>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub deadline: Option<Instant>,
    pub cancel: &'a CancelToken,
}

/// Bytes and accounting returned by native text generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeTextOutput {
    pub bytes: Vec<u8>,
    pub tokens: usize,
    pub finish: NativeFinish,
}

/// A loaded model session.
pub trait EngineSession: Send + Sync {
    fn inputs(&self) -> &[TensorSpec];
    fn outputs(&self) -> &[TensorSpec];

    /// Run one forward pass.
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>>;

    /// Run with an optional wall-clock deadline. Engines that cannot interrupt
    /// a forward pass keep the default behavior and report elapsed deadlines
    /// at the next generation-token boundary.
    fn run_with_deadline(
        &self,
        inputs: Vec<Tensor>,
        _deadline: Option<Instant>,
    ) -> Result<Vec<Tensor>> {
        self.run(inputs)
    }

    /// Streaming entry point. The default yields the full result as one chunk;
    /// engines that can emit partial results override it.
    fn run_streaming(
        &self,
        inputs: Vec<Tensor>,
        sink: &mut dyn FnMut(Vec<Tensor>) -> bool,
    ) -> Result<()> {
        let outputs = self.run(inputs)?;
        sink(outputs);
        Ok(())
    }

    /// Whether this session implements `generate_text`.
    fn supports_native_generation(&self) -> bool {
        false
    }

    /// Generate text inside the engine when it owns its own tokenizer/decoder.
    ///
    /// The default `None` keeps logits-based sessions on `TextGenerator`.
    /// Returning `Some` opts into the native path, including streaming through
    /// `on_bytes`.
    fn generate_text(
        &self,
        _request: NativeTextRequest<'_>,
        _on_bytes: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<Option<NativeTextOutput>> {
        Ok(None)
    }

    /// Idempotent cancellation. Pending and subsequent runs return `Cancelled`.
    fn cancel(&self);
}

/// Engine contract.
pub trait InferenceEngine: Send + Sync {
    fn engine_name(&self) -> &'static str;

    fn supported_formats(&self) -> &[ModelFormat];

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality];

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization];

    fn supported_devices(&self) -> &[DeviceClass];

    /// Whether this engine implements the manifest's model family/architecture.
    ///
    /// Engines with a generic graph contract return true; model-family engines
    /// override this so routing fails before loading weights.
    fn supports_architecture(&self, _manifest: &ModelManifest) -> bool {
        true
    }

    fn capabilities(&self) -> EngineCapabilities {
        let formats = self.supported_formats().to_vec();
        let modalities = formats
            .iter()
            .map(|format| (*format, self.supported_modalities(*format).to_vec()))
            .collect();
        let quantizations = formats
            .iter()
            .map(|format| (*format, self.supported_quantizations(*format).to_vec()))
            .collect();
        EngineCapabilities {
            formats,
            modalities,
            quantizations,
            devices: self.supported_devices().to_vec(),
        }
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>>;
}
