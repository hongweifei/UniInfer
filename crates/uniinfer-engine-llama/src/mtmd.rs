//! llama.cpp mtmd multimodal engine (GGUF + mmproj).

use std::sync::{Arc, Mutex};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::sampling::LlamaSampler;
use self_cell::self_cell;
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, NativeFinish, NativeTextDecoder, NativeTextOutput, NativeTextRequest,
    Quantization, Result, Runtime, Tensor as CoreTensor, TensorData, TensorSpec,
};

const FORMATS: &[ModelFormat] = &[ModelFormat::Gguf];
const MODALITIES: &[Modality] = &[Modality::Text, Modality::Image];
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
];
const DEVICES: &[DeviceClass] = &[DeviceClass::Cpu, DeviceClass::Gpu];

self_cell!(
    struct ModelContext {
        owner: Arc<LlamaModel>,
        #[covariant]
        dependent: LlamaContext,
    }
);

pub struct LlamaMtmdEngine {
    backend: LlamaBackend,
}

impl LlamaMtmdEngine {
    pub fn new() -> Result<Self> {
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(true),
        );
        let backend = LlamaBackend::init()
            .map_err(|error| EngineError::LoadFailed(format!("llama backend: {error}")))?;
        Ok(Self { backend })
    }

    #[must_use]
    pub fn probe_devices(&self) -> Vec<uniinfer_core::Device> {
        llama_cpp_2::list_llama_ggml_backend_devices()
            .into_iter()
            .filter_map(|device| {
                let class = match device.device_type {
                    llama_cpp_2::LlamaBackendDeviceType::Gpu
                    | llama_cpp_2::LlamaBackendDeviceType::IntegratedGpu
                    | llama_cpp_2::LlamaBackendDeviceType::Accelerator => DeviceClass::Gpu,
                    _ => return None,
                };
                Some(uniinfer_core::Device {
                    id: format!("gpu:llama-mtmd:{}", device.index),
                    class,
                    backend: device.backend.to_lowercase(),
                    memory_bytes: u64::try_from(device.memory_total).unwrap_or(0),
                })
            })
            .collect()
    }
}

impl InferenceEngine for LlamaMtmdEngine {
    fn engine_name(&self) -> &'static str {
        "llama-mtmd"
    }

    fn runtime(&self) -> Runtime {
        Runtime::Llama
    }

    fn implementations(&self) -> &[&'static str] {
        &["llama.mtmd"]
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
        DEVICES
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        let vision = spec
            .manifest
            .vision
            .as_ref()
            .ok_or_else(|| EngineError::ManifestInvalid("vision config required".into()))?;
        let mmproj = vision
            .mmproj_file
            .as_ref()
            .ok_or_else(|| EngineError::ManifestInvalid("vision.mmproj_file required".into()))?;
        let model_file = spec
            .manifest
            .files
            .iter()
            .find(|file| {
                file.path
                    .extension()
                    .is_some_and(|extension| extension == "gguf")
            })
            .ok_or_else(|| EngineError::ManifestInvalid("no GGUF model file".into()))?;
        let model_path = spec.base_dir.join(&model_file.path);
        let mmproj_path = spec.base_dir.join(mmproj);
        spec.manifest.verify_files(&spec.base_dir)?;

        let class = spec
            .placement
            .devices
            .first()
            .map(|device| device.class)
            .ok_or_else(|| EngineError::PlacementUnsatisfiable("empty placement".into()))?;
        let mut params = LlamaModelParams::default();
        if class == DeviceClass::Gpu {
            params = params.with_n_gpu_layers(999);
        }
        let model = Arc::new(
            LlamaModel::load_from_file(&self.backend, &model_path, &params)
                .map_err(|error| EngineError::LoadFailed(format!("llama mtmd model: {error}")))?,
        );
        let mtmd = MtmdContext::init_from_file(
            mmproj_path
                .to_str()
                .ok_or_else(|| EngineError::ManifestInvalid("mmproj path is not UTF-8".into()))?,
            &model,
            &MtmdContextParams::default(),
        )
        .map_err(|error| EngineError::LoadFailed(format!("mtmd init: {error}")))?;
        let context_len = spec.manifest.context_len.max(1);
        let context_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(
                u32::try_from(context_len).unwrap_or(u32::MAX),
            ))
            .with_n_batch(u32::try_from(512.min(context_len)).unwrap_or(512));
        let context = ModelContext::try_new(model.clone(), |model| {
            model
                .new_context(&self.backend, context_params)
                .map_err(|error| EngineError::LoadFailed(format!("llama context: {error}")))
        })?;
        Ok(Box::new(LlamaMtmdSession {
            state: Mutex::new(MtmdState {
                model,
                context,
                mtmd,
            }),
            inputs: vec![
                TensorSpec {
                    name: "text".into(),
                    dtype: uniinfer_core::TensorDtype::U8,
                    shape: vec![None],
                },
                TensorSpec {
                    name: "image".into(),
                    dtype: uniinfer_core::TensorDtype::U8,
                    shape: vec![None],
                },
            ],
            outputs: vec![TensorSpec {
                name: "text".into(),
                dtype: uniinfer_core::TensorDtype::U8,
                shape: vec![None],
            }],
        }))
    }
}

struct MtmdState {
    model: Arc<LlamaModel>,
    context: ModelContext,
    mtmd: MtmdContext,
}

struct LlamaMtmdSession {
    state: Mutex<MtmdState>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl EngineSession for LlamaMtmdSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<CoreTensor>) -> Result<Vec<CoreTensor>> {
        let text = inputs
            .iter()
            .find(|tensor| tensor.name == "text")
            .and_then(|tensor| match &tensor.data {
                TensorData::U8(bytes) => String::from_utf8(bytes.clone()).ok(),
                _ => None,
            })
            .ok_or_else(|| EngineError::Other("missing UTF-8 text input".into()))?;
        let image = inputs
            .iter()
            .find(|tensor| tensor.name == "image")
            .and_then(|tensor| match &tensor.data {
                TensorData::U8(bytes) => Some(bytes.clone()),
                _ => None,
            });
        let images = image.into_iter().collect::<Vec<_>>();
        let output = self.generate_text(
            NativeTextRequest {
                prompt: &text,
                images: &images,
                max_tokens: 64,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                deadline: None,
                cancel: &uniinfer_core::CancelToken::new(),
            },
            &mut |_| true,
        )?;
        Ok(vec![CoreTensor::new(
            "text",
            vec![output.bytes.len()],
            TensorData::U8(output.bytes),
        )?])
    }

    fn native_decoder(&self) -> Option<&dyn NativeTextDecoder> {
        Some(self)
    }

    fn cancel(&self) {}
}

impl NativeTextDecoder for LlamaMtmdSession {
    fn generate_text(
        &self,
        request: NativeTextRequest<'_>,
        on_bytes: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<NativeTextOutput> {
        if request.images.len() > 1 {
            return Err(EngineError::Other(
                "only one image per generation is supported".into(),
            ));
        }
        if request.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if request.max_tokens == 0 {
            return Ok(NativeTextOutput {
                bytes: Vec::new(),
                tokens: 0,
                finish: NativeFinish::Length,
            });
        }

        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("mtmd state lock poisoned".into()))?;
        let bitmap = request
            .images
            .first()
            .map(|image| {
                MtmdBitmap::from_buffer(&state.mtmd, image, false)
                    .map_err(|error| EngineError::Other(format!("mtmd bitmap: {error}")))
            })
            .transpose()?;
        let bitmaps = bitmap.iter().collect::<Vec<_>>();
        let chunks = state
            .mtmd
            .tokenize(
                MtmdInputText {
                    text: request.prompt.to_owned(),
                    add_special: true,
                    parse_special: true,
                },
                &bitmaps,
            )
            .map_err(|error| EngineError::Other(format!("mtmd tokenize: {error}")))?;

        let MtmdState {
            model,
            context,
            mtmd,
        } = &mut *state;
        let seed = request
            .seed
            .unwrap_or_else(clock_seed)
            .try_into()
            .unwrap_or(u32::MAX);
        let mut samplers = Vec::new();
        if request.frequency_penalty > 0.0 || request.presence_penalty > 0.0 {
            samplers.push(LlamaSampler::penalties(
                model.n_vocab(),
                64,
                1.0,
                request.frequency_penalty,
                request.presence_penalty,
            ));
        }
        if request.temperature > 0.0 {
            samplers.push(LlamaSampler::temp(request.temperature));
            if request.top_k > 0 {
                samplers.push(LlamaSampler::top_k(
                    i32::try_from(request.top_k).unwrap_or(i32::MAX),
                ));
            }
            if request.top_p < 1.0 {
                samplers.push(LlamaSampler::top_p(request.top_p, 1));
            }
            samplers.push(LlamaSampler::dist(seed));
        } else {
            samplers.push(LlamaSampler::greedy());
        }
        let mut sampler = LlamaSampler::chain_simple(samplers);

        let mut bytes = Vec::new();
        let mut tokens = 0usize;
        let mut finish = NativeFinish::Length;
        context.with_dependent_mut(|_, context| {
            context.clear_kv_cache();
            let n_batch = i32::try_from(context.n_batch()).unwrap_or(512);
            let n_past = chunks
                .eval_chunks(mtmd, context, 0, 0, n_batch, true)
                .map_err(|error| EngineError::Other(format!("mtmd eval: {error}")))?;
            let vocab = model.vocab();
            let max_tokens = i32::try_from(request.max_tokens).unwrap_or(i32::MAX);
            for position in n_past..n_past.saturating_add(max_tokens) {
                if request
                    .deadline
                    .is_some_and(|deadline| std::time::Instant::now() >= deadline)
                {
                    return Err(EngineError::Timeout);
                }
                if request.cancel.is_cancelled() {
                    return Err(EngineError::Cancelled);
                }
                let token = sampler.sample(context, -1);
                if vocab.is_eog(token) {
                    finish = NativeFinish::Eos;
                    break;
                }
                let piece = vocab.token_to_piece(token, false, None);
                tokens += 1;
                bytes.extend_from_slice(&piece);
                if !on_bytes(&piece) {
                    finish = NativeFinish::SinkClosed;
                    break;
                }
                sampler.accept(token);
                let mut batch = LlamaBatch::new(1, 1);
                batch
                    .add(token, position, &[0], true)
                    .map_err(|error| EngineError::Other(format!("mtmd batch: {error}")))?;
                context
                    .decode(&mut batch)
                    .map_err(|error| EngineError::Other(format!("mtmd decode: {error}")))?;
            }
            Ok::<(), EngineError>(())
        })?;
        Ok(NativeTextOutput {
            bytes,
            tokens,
            finish,
        })
    }
}

fn clock_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0x9e37_79b9_7f4a_7c15, |duration| {
            duration.as_nanos() as u64 ^ 0x9e37_79b9_7f4a_7c15
        })
}
