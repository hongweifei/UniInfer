//! Adapter for the official `llama.cpp` `llama-server` executable.
//!
//! The server owns the GGUF model and tokenizer. UniInfer sends the rendered
//! chat prompt to `/completion` and streams OpenAI-style SSE chunks back into
//! the native decoder contract. Killing the child is the cancellation path.

use std::path::PathBuf;
use std::sync::Arc;

use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, NativeTextDecoder, NativeTextRequest, Quantization, Residency, Result, Runtime,
    Tensor, TensorSpec,
};

mod completion;
mod model;
mod server;

use server::ServerProcess;

const FORMATS: &[ModelFormat] = &[ModelFormat::Gguf];
const DEVICES: &[DeviceClass] = &[DeviceClass::Cpu, DeviceClass::Gpu];
const QUANTIZATIONS: &[Quantization] = &[
    Quantization::Fp32,
    Quantization::Fp16,
    Quantization::Bf16,
    Quantization::Q2K,
    Quantization::Q3KS,
    Quantization::Q3KM,
    Quantization::Q3KL,
    Quantization::Q4_0,
    Quantization::Q4_1,
    Quantization::Q4KS,
    Quantization::Q4KM,
    Quantization::Q5_0,
    Quantization::Q5_1,
    Quantization::Q5KS,
    Quantization::Q5KM,
    Quantization::Q6K,
    Quantization::Q8_0,
    Quantization::Q8K,
    Quantization::Q2KS,
    Quantization::Iq1S,
    Quantization::Iq1M,
    Quantization::Iq2Xxs,
    Quantization::Iq2Xs,
    Quantization::Iq2S,
    Quantization::Iq2M,
    Quantization::Iq3Xxs,
    Quantization::Iq3Xs,
    Quantization::Iq3S,
    Quantization::Iq3M,
    Quantization::Iq4Nl,
    Quantization::Iq4Xs,
    Quantization::Mxfp4,
    Quantization::Nvfp4,
    Quantization::Tq1_0,
    Quantization::Tq2_0,
    Quantization::Q1_0,
    Quantization::Q2_0,
];

pub struct LlamaServerEngine {
    executable: PathBuf,
    devices: Vec<DeviceClass>,
    backend_id: String,
    variant: String,
}

impl LlamaServerEngine {
    #[must_use]
    pub fn new(executable: PathBuf, devices: Vec<DeviceClass>) -> Self {
        Self::with_identity(executable, devices, "llama.cpp", "auto")
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

impl InferenceEngine for LlamaServerEngine {
    fn engine_name(&self) -> &'static str {
        "llama-server"
    }

    fn runtime(&self) -> Runtime {
        Runtime::Llama
    }

    fn backend_id(&self) -> &str {
        &self.backend_id
    }

    fn backend_variant(&self) -> &str {
        &self.variant
    }

    fn implementations(&self) -> &[&'static str] {
        &["llama.text"]
    }

    fn is_transport(&self) -> bool {
        true
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        FORMATS
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Gguf => &[Modality::Text],
            _ => &[],
        }
    }

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization] {
        match format {
            ModelFormat::Gguf => QUANTIZATIONS,
            _ => &[],
        }
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        &self.devices
    }

    fn supports_layer_offload(&self) -> bool {
        true
    }

    fn supports_kv_cache_offload(&self) -> bool {
        true
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if !self.executable.is_file() {
            return Err(EngineError::LoadFailed(format!(
                "llama-server executable is missing: {}",
                self.executable.display()
            )));
        }
        let model = model::model_path(&spec)?;
        let n_gpu_layers = model::offload_layers(&spec);
        let process = server::spawn(
            &self.executable,
            &model,
            spec.manifest.context_len.max(1),
            n_gpu_layers,
            &spec.cancel,
        )?;
        let bytes = model::model_bytes(&spec);
        let residency = Residency {
            vram_bytes: if n_gpu_layers > 0 { bytes } else { 0 },
            host_bytes: if n_gpu_layers > 0 { 0 } else { bytes },
            disk_bytes: bytes,
            offload_layers: usize::try_from(n_gpu_layers).unwrap_or(usize::MAX),
            total_layers: None,
            estimated: true,
        };
        tracing::info!(
            model = %spec.manifest.id,
            n_gpu_layers,
            "official llama-server ready"
        );
        Ok(Box::new(LlamaServerSession {
            process,
            residency: Some(residency),
        }))
    }
}

struct LlamaServerSession {
    process: Arc<ServerProcess>,
    residency: Option<Residency>,
}

impl EngineSession for LlamaServerSession {
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
            "llama-server uses native text decoding".into(),
        ))
    }

    fn native_decoder(&self) -> Option<&dyn NativeTextDecoder> {
        Some(self)
    }

    fn cancel(&self) {
        self.process.kill();
    }
}

impl Drop for LlamaServerSession {
    fn drop(&mut self) {
        self.process.kill();
    }
}

impl NativeTextDecoder for LlamaServerSession {
    fn generate_text(
        &self,
        request: NativeTextRequest<'_>,
        on_bytes: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<uniinfer_core::NativeTextOutput> {
        completion::stream_completion(&self.process, request, on_bytes)
    }
}
