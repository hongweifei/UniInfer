//! Adapter for the official `stable-diffusion.cpp` `sd-server` executable.
//!
//! UniInfer starts the server with a full model and uses its A1111-compatible
//! `txt2img` endpoint. The server owns prompting and image decoding; the
//! process is killed for cancellation or session teardown.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, NativeImageGenerator, NativeImageOutput, NativeImageRequest, Quantization,
    Residency, Result, Runtime, Tensor, TensorSpec,
};
use uniinfer_engine_sidecar::{ServerProcess, SpawnOptions};

const FORMATS: &[ModelFormat] = &[ModelFormat::Gguf, ModelFormat::Safetensors];
const DEVICES: &[DeviceClass] = &[DeviceClass::Cpu, DeviceClass::Gpu];
const QUANTIZATIONS: &[Quantization] = &[
    Quantization::Fp32,
    Quantization::Fp16,
    Quantization::Bf16,
    Quantization::Q4_0,
    Quantization::Q4_1,
    Quantization::Q5_0,
    Quantization::Q5_1,
    Quantization::Q8_0,
    Quantization::Q2K,
    Quantization::Q3KS,
    Quantization::Q3KM,
    Quantization::Q3KL,
    Quantization::Q4KS,
    Quantization::Q4KM,
    Quantization::Q5KS,
    Quantization::Q5KM,
    Quantization::Q6K,
    Quantization::Q8K,
];

pub struct StableDiffusionServerEngine {
    executable: PathBuf,
    devices: Vec<DeviceClass>,
    backend_id: String,
    variant: String,
}

impl StableDiffusionServerEngine {
    #[must_use]
    pub fn new(executable: PathBuf, devices: Vec<DeviceClass>) -> Self {
        Self::with_identity(executable, devices, "stable-diffusion.cpp", "auto")
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
                DEVICES.to_vec()
            } else {
                devices
            },
            backend_id: backend_id.into(),
            variant: variant.into(),
        }
    }
}

impl InferenceEngine for StableDiffusionServerEngine {
    fn engine_name(&self) -> &'static str {
        "sd-server"
    }

    fn runtime(&self) -> Runtime {
        Runtime::Onnx
    }

    fn backend_id(&self) -> &str {
        &self.backend_id
    }

    fn backend_variant(&self) -> &str {
        &self.variant
    }

    fn implementations(&self) -> &[&'static str] {
        &["sd.txt2img"]
    }

    fn is_transport(&self) -> bool {
        true
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        FORMATS
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Gguf | ModelFormat::Safetensors => &[Modality::Text],
            _ => &[],
        }
    }

    fn supported_output_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Gguf | ModelFormat::Safetensors => &[Modality::Image],
            _ => &[],
        }
    }

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization] {
        match format {
            ModelFormat::Gguf | ModelFormat::Safetensors => QUANTIZATIONS,
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
        if !self.executable.is_file() {
            return Err(EngineError::LoadFailed(format!(
                "sd-server executable is missing: {}",
                self.executable.display()
            )));
        }
        let model = model_path(&spec)?;
        let process = uniinfer_engine_sidecar::spawn(SpawnOptions {
            executable: &self.executable,
            readiness_path: "/sdapi/v1/options",
            startup_timeout: Duration::from_secs(600),
            request_timeout: Duration::from_secs(900),
            cancel: &spec.cancel,
            label: "sd-server",
            args: |port: u16| {
                vec![
                    "-m".into(),
                    model.as_os_str().to_owned(),
                    "--listen-ip".into(),
                    "127.0.0.1".into(),
                    "--listen-port".into(),
                    port.to_string().into(),
                ]
            },
        })?;
        let bytes = std::fs::metadata(&model)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        tracing::info!(model = %spec.manifest.id, "official sd-server ready");
        Ok(Box::new(StableDiffusionSession {
            process,
            residency: Some(Residency {
                vram_bytes: bytes,
                disk_bytes: bytes,
                estimated: true,
                ..Residency::default()
            }),
        }))
    }
}

struct StableDiffusionSession {
    process: Arc<ServerProcess>,
    residency: Option<Residency>,
}

impl EngineSession for StableDiffusionSession {
    fn inputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn outputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn residency(&self) -> Option<Residency> {
        self.residency
    }

    fn run(&self, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        Err(EngineError::UnsupportedEngine(
            "sd-server uses native image generation".into(),
        ))
    }

    fn native_image_generator(&self) -> Option<&dyn NativeImageGenerator> {
        Some(self)
    }

    fn cancel(&self) {
        self.process.kill();
    }
}

impl Drop for StableDiffusionSession {
    fn drop(&mut self) {
        self.process.kill();
    }
}

#[derive(Serialize)]
struct Txt2Img<'a> {
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    negative_prompt: Option<&'a str>,
    width: u32,
    height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    steps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cfg_scale: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct Txt2ImgResponse {
    #[serde(default)]
    images: Vec<String>,
}

impl NativeImageGenerator for StableDiffusionSession {
    fn generate_image(&self, request: &mut NativeImageRequest<'_>) -> Result<NativeImageOutput> {
        if request.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let payload = Txt2Img {
            prompt: request.prompt,
            negative_prompt: request.negative_prompt,
            width: request.width,
            height: request.height,
            steps: (request.steps > 0).then_some(request.steps),
            cfg_scale: (request.guidance > 0.0).then_some(request.guidance),
            seed: request.seed,
        };
        let mut call = self
            .process
            .client
            .post(format!("{}/sdapi/v1/txt2img", self.process.base))
            .json(&payload);
        if let Some(deadline) = request.deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(EngineError::Timeout);
            }
            call = call.timeout(remaining);
        }
        let response = call.send().map_err(map_request_error)?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(EngineError::LoadFailed(format!(
                "sd-server txt2img HTTP {status}: {body}"
            )));
        }
        let response: Txt2ImgResponse = response
            .json()
            .map_err(|error| EngineError::Other(format!("sd-server response: {error}")))?;
        let image = response
            .images
            .first()
            .ok_or_else(|| EngineError::Other("sd-server returned no image".into()))?;
        let encoded = image.strip_prefix("data:").map_or(image.as_str(), |value| {
            value.split_once(',').map_or(value, |(_, data)| data)
        });
        let bytes = BASE64
            .decode(encoded)
            .map_err(|error| EngineError::Other(format!("sd-server image base64: {error}")))?;
        (request.progress)(1, 1);
        Ok(NativeImageOutput {
            mime: "image/png".into(),
            bytes,
            width: request.width,
            height: request.height,
        })
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
                    matches!(
                        extension.to_ascii_lowercase().as_str(),
                        "safetensors" | "gguf" | "ckpt"
                    )
                })
        })
        .map(|file| spec.base_dir.join(&file.path))
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
        .into_iter()
        .max_by_key(|path| {
            std::fs::metadata(path)
                .map(|metadata| metadata.len())
                .unwrap_or(0)
        })
        .ok_or_else(|| EngineError::ManifestInvalid("no diffusion model file for sd-server".into()))
}

fn map_request_error(error: reqwest::Error) -> EngineError {
    if error.is_timeout() {
        EngineError::Timeout
    } else {
        EngineError::Other(format!("sd-server request: {error}"))
    }
}
