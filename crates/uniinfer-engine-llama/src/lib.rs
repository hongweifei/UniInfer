#![cfg(feature = "llama")]
//! llama.cpp engine for GGUF models.
//!
//! llama.cpp is compiled from the vendored sources pulled in by
//! `llama-cpp-2`, so there is no runtime DLL to locate. GPU support is opt-in:
//! `vulkan` (AMD/Intel), `cuda`, `rocm`, `metal`, or `opencl`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::token::LlamaToken;
use self_cell::self_cell;
use uniinfer_core::{
    Device as CoreDevice, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine,
    Modality, ModelFormat, Quantization, Result, Tensor as CoreTensor, TensorData, TensorDtype,
    TensorSpec,
};

const FORMATS: &[ModelFormat] = &[ModelFormat::Gguf];
const MODALITIES: &[Modality] = &[Modality::Text];
const QUANTIZATIONS: &[Quantization] = &[
    Quantization::Fp16,
    Quantization::Bf16,
    Quantization::Q4_0,
    Quantization::Q4KM,
    Quantization::Q5KM,
    Quantization::Q6K,
    Quantization::Q8_0,
];
const BATCH_TOKENS: usize = 512;

self_cell!(
    struct ModelContext {
        owner: Arc<LlamaModel>,
        #[covariant]
        dependent: LlamaContext,
    }
);

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
            llama_cpp_2::LogOptions::default().with_logs_enabled(false),
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

        let mut params = LlamaModelParams::default();
        if class == DeviceClass::Gpu {
            params = params.with_n_gpu_layers(999);
        }
        let model = Arc::new(
            LlamaModel::load_from_file(&self.backend, &path, &params).map_err(|error| {
                EngineError::LoadFailed(format!("llama model {}: {error}", path.display()))
            })?,
        );
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
        let context_len = spec.manifest.context_len.max(1);
        let n_ctx = u32::try_from(context_len).unwrap_or(u32::MAX);
        let batch = u32::try_from(BATCH_TOKENS.min(context_len)).unwrap_or(u32::MAX);
        let context_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_batch(batch);
        let context = ModelContext::try_new(model, |model| {
            model
                .new_context(&self.backend, context_params)
                .map_err(|error| EngineError::LoadFailed(format!("llama context: {error}")))
        })?;

        Ok(Box::new(LlamaSession {
            state: Mutex::new(LlamaState {
                context,
                cached_ids: Vec::new(),
            }),
            cancelled: AtomicBool::new(false),
            inputs: vec![TensorSpec {
                name: "input_ids".into(),
                dtype: TensorDtype::I64,
                shape: vec![Some(1), None],
            }],
            outputs: vec![TensorSpec {
                name: "logits".into(),
                dtype: TensorDtype::F32,
                shape: vec![Some(1), None, Some(vocab)],
            }],
        }))
    }
}

struct LlamaState {
    context: ModelContext,
    cached_ids: Vec<u32>,
}

struct LlamaSession {
    state: Mutex<LlamaState>,
    cancelled: AtomicBool,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl EngineSession for LlamaSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<CoreTensor>) -> Result<Vec<CoreTensor>> {
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let ids = inputs
            .iter()
            .find(|tensor| tensor.name == "input_ids")
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) => Some(values.clone()),
                _ => None,
            })
            .ok_or_else(|| EngineError::Other("missing int64 input_ids".into()))?;
        if ids.is_empty() {
            return Err(EngineError::Other("input_ids is empty".into()));
        }
        let ids = ids
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| EngineError::Other("id overflow".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("llama state lock poisoned".into()))?;
        let incremental = state.cached_ids.len() + 1 == ids.len()
            && ids.starts_with(&state.cached_ids)
            && !state.cached_ids.is_empty();
        let (tokens, offset) = if incremental {
            (&ids[ids.len() - 1..], state.cached_ids.len())
        } else {
            state.cached_ids.clear();
            state
                .context
                .with_dependent_mut(|_, context| context.clear_kv_cache());
            (ids.as_slice(), 0)
        };
        let logits = state.context.with_dependent_mut(|_, context| {
            decode_chunked(
                context,
                tokens,
                offset,
                usize::try_from(context.n_batch()).unwrap_or(BATCH_TOKENS),
            )
        })?;
        state.cached_ids = ids;
        let vocab = logits.len();
        Ok(vec![CoreTensor::new(
            "logits",
            vec![1, 1, vocab],
            TensorData::F32(logits),
        )?])
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

fn decode_chunked(
    context: &mut LlamaContext<'_>,
    tokens: &[u32],
    offset: usize,
    chunk: usize,
) -> Result<Vec<f32>> {
    let chunk = chunk.max(1);
    let last = tokens.len().saturating_sub(1);
    for (start, window) in tokens.chunks(chunk).enumerate() {
        let base = offset + start * chunk;
        let mut batch = LlamaBatch::new(window.len(), 1);
        for (index, token) in window.iter().enumerate() {
            let token = i32::try_from(*token)
                .map_err(|_| EngineError::Other("token id overflow".into()))?;
            let position = i32::try_from(base + index)
                .map_err(|_| EngineError::Other("position overflow".into()))?;
            batch
                .add(
                    LlamaToken(token),
                    position,
                    &[0],
                    base + index == offset + last,
                )
                .map_err(|error| EngineError::Other(format!("llama batch: {error}")))?;
        }
        context
            .decode(&mut batch)
            .map_err(|error| EngineError::Other(format!("llama decode: {error}")))?;
    }
    Ok(context.get_logits().to_vec())
}
