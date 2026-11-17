#![cfg(feature = "llama")]
//! llama.cpp engine for GGUF models.
//!
//! llama.cpp is compiled from the vendored sources pulled in by
//! `llama-cpp-2`, so there is no runtime DLL to locate. GPU support is opt-in:
//! `vulkan` (AMD/Intel), `cuda`, `rocm`, `metal`, or `opencl`.

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use std::sync::Arc;
use uniinfer_core::{
    Device as CoreDevice, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine,
    Modality, ModelFormat, Quantization, Result, Runtime, TensorDtype, TensorSpec,
};

#[cfg(feature = "mtmd")]
mod mtmd;
mod offload;
mod session;

#[cfg(feature = "mtmd")]
pub use mtmd::LlamaMtmdEngine;
use session::LlamaSession;

const BATCH_TOKENS: usize = 512;
const FORMATS: &[ModelFormat] = &[ModelFormat::Gguf];
const MODALITIES: &[Modality] = &[Modality::Text];
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

/// llama.cpp-backed GGUF engine.
pub struct LlamaEngine {
    backend: LlamaBackend,
    devices: Vec<DeviceClass>,
    device_ids: Vec<CoreDevice>,
    gpu: bool,
}

impl LlamaEngine {
    /// Initialise the llama.cpp backend and probe its devices.
    pub fn new() -> Result<Self> {
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(true),
        );
        let backend = LlamaBackend::init()
            .map_err(|error| EngineError::LoadFailed(format!("llama backend: {error}")))?;
        let probed = llama_cpp_2::list_llama_ggml_backend_devices();
        let mut devices = vec![DeviceClass::Cpu];
        let mut device_ids = vec![CoreDevice {
            id: "cpu:llama:0".into(),
            class: DeviceClass::Cpu,
            backend: "cpu".into(),
            memory_bytes: 0,
            available_memory_bytes: 0,
        }];
        let mut gpu = false;
        for device in &probed {
            let class = match device.device_type {
                llama_cpp_2::LlamaBackendDeviceType::Gpu
                | llama_cpp_2::LlamaBackendDeviceType::IntegratedGpu
                | llama_cpp_2::LlamaBackendDeviceType::Accelerator => DeviceClass::Gpu,
                _ => continue,
            };
            gpu = true;
            if !devices.contains(&class) {
                devices.push(class);
            }
            device_ids.push(CoreDevice {
                id: format!("gpu:llama:{index}", index = device.index),
                class,
                backend: device.backend.to_lowercase(),
                memory_bytes: u64::try_from(device.memory_total).unwrap_or(0),
                available_memory_bytes: u64::try_from(device.memory_free).unwrap_or(0),
            });
        }
        tracing::info!(?devices, gpu, "llama.cpp devices probed");
        Ok(Self {
            backend,
            devices,
            device_ids,
            gpu,
        })
    }

    #[must_use]
    pub fn probe_devices(&self) -> Vec<CoreDevice> {
        self.device_ids.clone()
    }
}

impl InferenceEngine for LlamaEngine {
    fn engine_name(&self) -> &'static str {
        "llama.cpp"
    }

    fn runtime(&self) -> Runtime {
        Runtime::Llama
    }

    fn implementations(&self) -> &[&'static str] {
        &["llama.text"]
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        FORMATS
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Gguf => MODALITIES,
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
        if spec.manifest.format != ModelFormat::Gguf {
            return Err(EngineError::UnsupportedFormat(spec.manifest.format));
        }
        if !QUANTIZATIONS.contains(&spec.manifest.quantization) {
            return Err(EngineError::UnsupportedQuantization(
                spec.manifest.quantization,
            ));
        }
        let class = spec
            .placement
            .devices
            .first()
            .map(|device| device.class)
            .ok_or_else(|| EngineError::PlacementUnsatisfiable("empty placement".into()))?;
        match class {
            DeviceClass::Cpu => {}
            DeviceClass::Gpu if self.gpu => {}
            DeviceClass::Gpu => {
                return Err(EngineError::PlacementUnsatisfiable(
                    "llama.cpp was built without a GPU backend".into(),
                ));
            }
            DeviceClass::Npu => {
                return Err(EngineError::PlacementUnsatisfiable(
                    "llama.cpp has no NPU backend".into(),
                ));
            }
        }
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        spec.manifest.verify_files(&spec.base_dir)?;
        let text = spec
            .manifest
            .text
            .as_ref()
            .ok_or_else(|| EngineError::ManifestInvalid("text config required".into()))?;
        let model_file = spec
            .manifest
            .files
            .first()
            .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
        let path = spec.base_dir.join(&model_file.path);

        let offload = offload::plan(&spec, &path, &self.backend)?;
        let model = Arc::new(
            LlamaModel::load_from_file(&self.backend, &path, &offload.params).map_err(|error| {
                if spec.cancel.is_cancelled() {
                    EngineError::Cancelled
                } else {
                    EngineError::LoadFailed(format!("llama model {}: {error}", path.display()))
                }
            })?,
        );
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let vocab = match usize::try_from(model.n_vocab()) {
            Ok(0) | Err(_) => {
                return Err(EngineError::LoadFailed(
                    "llama model reported no vocabulary".into(),
                ));
            }
            Ok(vocab) => vocab,
        };
        if text.vocab_size != 0 && text.vocab_size != vocab {
            tracing::warn!(
                manifest = text.vocab_size,
                model = vocab,
                "manifest vocab_size differs from GGUF vocabulary"
            );
        }
        let inputs = vec![TensorSpec {
            name: "input_ids".into(),
            dtype: TensorDtype::I64,
            shape: vec![Some(1), None],
        }];
        let outputs = vec![TensorSpec {
            name: "logits".into(),
            dtype: TensorDtype::F32,
            shape: vec![Some(1), None, Some(vocab)],
        }];
        Ok(Box::new(LlamaSession::new(
            model,
            &self.backend,
            spec.manifest.context_len,
            offload.offload_kqv,
            inputs,
            outputs,
        )?))
    }
}
