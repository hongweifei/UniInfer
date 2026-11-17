//! Adapter for the official whisper.cpp `whisper-server` executable.
//!
//! The sidecar owns the GGML Whisper context. UniInfer sends one WAV request
//! to `/inference` per transcription and parses the JSON `text` field. Killing
//! the child is the cancellation path.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use hound::{SampleFormat, WavSpec, WavWriter};
use reqwest::blocking::multipart::{Form, Part};
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, Quantization, Residency, Result, Runtime, Tensor, TensorData, TensorDtype,
    TensorSpec,
};

mod server;

use server::ServerProcess;

const FORMATS: &[ModelFormat] = &[ModelFormat::Ggml];
const QUANTIZATIONS: &[Quantization] = &[
    Quantization::Fp32,
    Quantization::Fp16,
    Quantization::Q4_0,
    Quantization::Q4_1,
    Quantization::Q5_0,
    Quantization::Q5_1,
    Quantization::Q8_0,
];

/// whisper.cpp `whisper-server`-backed ASR engine.
pub struct WhisperServerEngine {
    executable: PathBuf,
    devices: Vec<DeviceClass>,
    backend_id: String,
    variant: String,
}

impl WhisperServerEngine {
    #[must_use]
    pub fn new(executable: PathBuf, devices: Vec<DeviceClass>) -> Self {
        Self::with_identity(executable, devices, "whisper.cpp", "auto")
    }

    #[must_use]
    pub fn with_identity(
        executable: PathBuf,
        devices: Vec<DeviceClass>,
        backend_id: impl Into<String>,
        variant: impl Into<String>,
    ) -> Self {
        Self {
            executable,
            devices: if devices.is_empty() {
                vec![DeviceClass::Cpu]
            } else {
                devices
            },
            backend_id: backend_id.into(),
            variant: variant.into(),
        }
    }
}

impl InferenceEngine for WhisperServerEngine {
    fn engine_name(&self) -> &'static str {
        "whisper-server"
    }

    fn runtime(&self) -> Runtime {
        Runtime::Whisper
    }

    fn backend_id(&self) -> &str {
        &self.backend_id
    }

    fn backend_variant(&self) -> &str {
        &self.variant
    }

    fn implementations(&self) -> &[&'static str] {
        &["whisper.ggml"]
    }

    fn is_transport(&self) -> bool {
        true
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        FORMATS
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Ggml => &[Modality::Audio],
            _ => &[],
        }
    }

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization] {
        match format {
            ModelFormat::Ggml => QUANTIZATIONS,
            _ => &[],
        }
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        &self.devices
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if spec.manifest.format != ModelFormat::Ggml {
            return Err(EngineError::UnsupportedFormat(spec.manifest.format));
        }
        if !self.executable.is_file() {
            return Err(EngineError::LoadFailed(format!(
                "whisper-server executable is missing: {}",
                self.executable.display()
            )));
        }
        let model = model_path(&spec)?;
        let gpu = spec
            .placement
            .devices
            .iter()
            .any(|device| device.class == DeviceClass::Gpu);
        let process = server::spawn(&self.executable, &model, gpu, &spec.cancel)?;
        let audio = spec.manifest.audio.clone().unwrap_or_default();
        let bytes = model_bytes(&spec);
        let residency = Residency {
            vram_bytes: if gpu { bytes } else { 0 },
            host_bytes: if gpu { 0 } else { bytes },
            disk_bytes: bytes,
            offload_layers: 0,
            total_layers: None,
            estimated: true,
        };
        let inputs = vec![TensorSpec {
            name: audio.input.clone(),
            dtype: TensorDtype::F32,
            shape: vec![None],
        }];
        let outputs = vec![TensorSpec {
            name: audio.output.clone(),
            dtype: TensorDtype::U8,
            shape: vec![None],
        }];
        Ok(Box::new(WhisperServerSession {
            process,
            input_name: audio.input,
            output_name: audio.output,
            inputs,
            outputs,
            sample_rate: audio.sample_rate,
            residency,
            cancelled: AtomicBool::new(false),
        }))
    }
}

fn model_path(spec: &EngineLoadSpec) -> Result<PathBuf> {
    let mut candidates = spec
        .manifest
        .files
        .iter()
        .filter(|file| {
            file.path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    extension.eq_ignore_ascii_case("bin") || extension.eq_ignore_ascii_case("ggml")
                })
        })
        .map(|file| spec.base_dir.join(&file.path))
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
        .into_iter()
        .max_by_key(|path| std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0))
        .ok_or_else(|| EngineError::ManifestInvalid("no GGML model file for whisper-server".into()))
}

fn model_bytes(spec: &EngineLoadSpec) -> u64 {
    spec.manifest
        .files
        .iter()
        .filter(|file| {
            file.path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    extension.eq_ignore_ascii_case("bin") || extension.eq_ignore_ascii_case("ggml")
                })
        })
        .filter_map(|file| std::fs::metadata(spec.base_dir.join(&file.path)).ok())
        .map(|metadata| metadata.len())
        .sum()
}

struct WhisperServerSession {
    process: Arc<ServerProcess>,
    input_name: String,
    output_name: String,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    sample_rate: u32,
    residency: Residency,
    cancelled: AtomicBool,
}

impl EngineSession for WhisperServerSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn residency(&self) -> Option<Residency> {
        Some(self.residency)
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        self.transcribe(inputs, None)
    }

    fn run_with_deadline(
        &self,
        inputs: Vec<Tensor>,
        deadline: Option<Instant>,
    ) -> Result<Vec<Tensor>> {
        self.transcribe(inputs, deadline)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.process.kill();
    }
}

impl WhisperServerSession {
    fn transcribe(&self, inputs: Vec<Tensor>, deadline: Option<Instant>) -> Result<Vec<Tensor>> {
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let timeout = deadline.map(|deadline| {
            deadline
                .checked_duration_since(Instant::now())
                .unwrap_or(Duration::ZERO)
        });
        if timeout.is_some_and(|timeout| timeout.is_zero()) {
            return Err(EngineError::Timeout);
        }
        let samples = inputs
            .iter()
            .find(|tensor| tensor.name == self.input_name)
            .ok_or_else(|| EngineError::InvalidAudio("missing audio input".into()))?;
        let samples = match &samples.data {
            TensorData::F32(values) => values,
            _ => return Err(EngineError::InvalidAudio("audio must be f32".into())),
        };
        if samples.is_empty() {
            return Err(EngineError::InvalidAudio("audio is empty".into()));
        }
        let wav = encode_wav(samples, self.sample_rate)?;
        let part = Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|error| EngineError::InvalidAudio(format!("wav part: {error}")))?;
        let form = Form::new()
            .part("file", part)
            .text("response_format", "json")
            .text("temperature", "0.0");
        let response = self
            .process
            .client
            .post(format!("{}/inference", self.process.base))
            .multipart(form)
            .timeout(timeout.unwrap_or_else(|| Duration::from_secs(600)))
            .send()
            .map_err(|error| {
                if self.cancelled.load(Ordering::SeqCst) {
                    EngineError::Cancelled
                } else {
                    EngineError::LoadFailed(format!("whisper-server inference: {error}"))
                }
            })?;
        let status = response.status();
        let body = response
            .text()
            .map_err(|error| EngineError::Other(format!("whisper-server response: {error}")))?;
        if !status.is_success() {
            return Err(EngineError::Other(format!(
                "whisper-server HTTP {status}: {}",
                body.trim()
            )));
        }
        let text = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .get("text")
                    .and_then(|text| text.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or(body);
        let bytes = text.trim().as_bytes().to_vec();
        Tensor::new(
            self.output_name.clone(),
            vec![bytes.len()],
            TensorData::U8(bytes),
        )
        .map(|tensor| vec![tensor])
    }
}

impl Drop for WhisperServerSession {
    fn drop(&mut self) {
        self.process.kill();
    }
}

fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate.max(1),
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut output = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut output, spec)
            .map_err(|error| EngineError::InvalidAudio(format!("wav writer: {error}")))?;
        for sample in samples {
            let value = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
            writer
                .write_sample(value)
                .map_err(|error| EngineError::InvalidAudio(format!("wav sample: {error}")))?;
        }
        writer
            .finalize()
            .map_err(|error| EngineError::InvalidAudio(format!("wav finalize: {error}")))?;
    }
    Ok(output.into_inner())
}
