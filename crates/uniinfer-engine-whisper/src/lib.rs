//! whisper.cpp engine for speech transcription.
//!
//! The engine exposes Whisper as an `InferenceEngine` session: one `audio`
//! input tensor (16 kHz mono F32 samples) and one UTF-8 `text` output tensor.
//! The session adapter keeps whisper-specific state behind the engine boundary.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use uniinfer_core::{
    Device, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, Quantization, Result, Tensor, TensorData, TensorDtype, TensorSpec,
};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperError,
};

const FORMATS: &[ModelFormat] = &[ModelFormat::Whisper];
const MODALITIES: &[Modality] = &[Modality::Audio];
const QUANTIZATIONS: &[Quantization] = &[Quantization::Fp16, Quantization::Fp32];
const DEVICES: &[DeviceClass] = &[DeviceClass::Cpu];

/// whisper.cpp-backed engine.
pub struct WhisperEngine;

impl WhisperEngine {
    #[must_use]
    pub fn new() -> Self {
        whisper_rs::install_logging_hooks();
        Self
    }

    /// whisper.cpp is CPU-only in this slice.
    #[must_use]
    pub fn probe_devices(&self) -> Vec<Device> {
        vec![Device {
            id: "cpu:whisper".into(),
            class: DeviceClass::Cpu,
            backend: "whisper.cpp".into(),
            memory_bytes: 0,
        }]
    }
}

impl Default for WhisperEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceEngine for WhisperEngine {
    fn engine_name(&self) -> &'static str {
        "whisper"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        FORMATS
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Whisper => MODALITIES,
            _ => &[],
        }
    }

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization] {
        match format {
            ModelFormat::Whisper => QUANTIZATIONS,
            _ => &[],
        }
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        DEVICES
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if spec.manifest.format != ModelFormat::Whisper {
            return Err(EngineError::UnsupportedFormat(spec.manifest.format));
        }
        let file = spec
            .manifest
            .files
            .iter()
            .find(|file| {
                file.path
                    .extension()
                    .is_some_and(|extension| extension == "bin" || extension == "ggml")
            })
            .or_else(|| spec.manifest.files.first())
            .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
        let path = spec.base_dir.join(&file.path);
        let context = WhisperContext::new_with_params(&path, WhisperContextParameters::default())
            .map_err(|error| load_error(&path, error))?;
        Ok(Box::new(WhisperSession::new(Arc::new(context))))
    }
}

fn load_error(path: &Path, error: WhisperError) -> EngineError {
    EngineError::LoadFailed(format!("{}: {error}", path.display()))
}

struct WhisperSession {
    context: Arc<WhisperContext>,
    cancel: Arc<AtomicBool>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl WhisperSession {
    fn new(context: Arc<WhisperContext>) -> Self {
        Self {
            context,
            cancel: Arc::new(AtomicBool::new(false)),
            inputs: vec![TensorSpec {
                name: "audio".into(),
                dtype: TensorDtype::F32,
                shape: vec![None],
            }],
            outputs: vec![TensorSpec {
                name: "text".into(),
                dtype: TensorDtype::U8,
                shape: vec![None],
            }],
        }
    }
}

impl EngineSession for WhisperSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        if self.cancel.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let samples = inputs
            .iter()
            .find(|tensor| tensor.name == "audio")
            .ok_or_else(|| EngineError::InvalidAudio("missing audio input".into()))?;
        let samples = match &samples.data {
            TensorData::F32(values) => values,
            _ => return Err(EngineError::InvalidAudio("audio must be f32".into())),
        };
        if samples.is_empty() {
            return Err(EngineError::InvalidAudio("audio is empty".into()));
        }

        let mut state = self
            .context
            .create_state()
            .map_err(|error| EngineError::LoadFailed(format!("whisper state: {error}")))?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_translate(false);
        params.set_no_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        let cancel = Arc::clone(&self.cancel);
        let abort: Box<dyn FnMut() -> bool> = Box::new(move || cancel.load(Ordering::SeqCst));
        params
            .set_abort_callback_safe::<Option<Box<dyn FnMut() -> bool>>, Box<dyn FnMut() -> bool>>(
                Some(abort),
            );
        state
            .full(params, samples)
            .map_err(|error| transcription_error(&self.cancel, error))?;

        let mut text = String::new();
        for segment in state.as_iter() {
            let part = segment.to_str_lossy().map_err(|error| {
                EngineError::Other(format!("whisper segment decoding: {error}"))
            })?;
            text.push_str(&part);
        }
        let bytes = text.trim().as_bytes().to_vec();
        let text = Tensor::new("text", vec![bytes.len()], TensorData::U8(bytes))?;
        Ok(vec![text])
    }

    fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

fn transcription_error(cancel: &AtomicBool, error: WhisperError) -> EngineError {
    if cancel.load(Ordering::SeqCst) {
        EngineError::Cancelled
    } else {
        EngineError::LoadFailed(format!("whisper transcription: {error}"))
    }
}
