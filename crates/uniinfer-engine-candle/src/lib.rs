//! Candle engine for quantized Qwen3 GGUF and Safetensors models.
#![allow(unsafe_code)] // mmap of trusted local weight files is the only unsafe boundary.

use std::sync::{LazyLock, Mutex};

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM};
use uniinfer_core::{
    Device as CoreDevice, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine,
    Modality, ModelFormat, Quantization, Result, Tensor as CoreTensor, TensorData, TensorDtype,
    TensorSpec,
};

const FORMATS: &[ModelFormat] = &[ModelFormat::Gguf, ModelFormat::Safetensors];
const MODALITIES: &[Modality] = &[Modality::Text];
const GGUF_QUANTIZATIONS: &[Quantization] = &[
    Quantization::Q4KM,
    Quantization::Q4_0,
    Quantization::Q5KM,
    Quantization::Q6K,
    Quantization::Q8_0,
];
const SAFETENSORS_QUANTIZATIONS: &[Quantization] =
    &[Quantization::Bf16, Quantization::Fp16, Quantization::Fp32];

static DEVICES: LazyLock<Vec<DeviceClass>> = LazyLock::new(|| {
    let mut devices = vec![DeviceClass::Cpu];
    if cfg!(any(feature = "cuda", feature = "metal")) {
        devices.push(DeviceClass::Gpu);
    }
    devices
});

pub struct CandleEngine {
    device: Device,
}

impl CandleEngine {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Ok(device) = Device::new_cuda(0) {
            return Ok(Self { device });
        }
        #[cfg(feature = "metal")]
        if let Ok(device) = Device::new_metal(0) {
            return Ok(Self { device });
        }
        Ok(Self {
            device: Device::Cpu,
        })
    }

    #[must_use]
    pub fn probe_devices(&self) -> Vec<CoreDevice> {
        let mut devices = vec![CoreDevice {
            id: "cpu:candle:0".into(),
            class: DeviceClass::Cpu,
            backend: "candle".into(),
            memory_bytes: 0,
        }];
        if self.device.is_cuda() || self.device.is_metal() {
            devices.push(CoreDevice {
                id: "gpu:candle:0".into(),
                class: DeviceClass::Gpu,
                backend: "candle".into(),
                memory_bytes: 0,
            });
        }
        devices
    }
}

impl InferenceEngine for CandleEngine {
    fn engine_name(&self) -> &'static str {
        "candle"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        FORMATS
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Gguf | ModelFormat::Safetensors => MODALITIES,
            _ => &[],
        }
    }

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization] {
        match format {
            ModelFormat::Gguf => GGUF_QUANTIZATIONS,
            ModelFormat::Safetensors => SAFETENSORS_QUANTIZATIONS,
            _ => &[],
        }
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        &DEVICES
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if !FORMATS.contains(&spec.manifest.format) {
            return Err(EngineError::UnsupportedFormat(spec.manifest.format));
        }
        let supported_quantizations = match spec.manifest.format {
            ModelFormat::Gguf => GGUF_QUANTIZATIONS,
            ModelFormat::Safetensors => SAFETENSORS_QUANTIZATIONS,
            _ => &[],
        };
        if !supported_quantizations.contains(&spec.manifest.quantization) {
            return Err(EngineError::UnsupportedQuantization(
                spec.manifest.quantization,
            ));
        }
        if spec
            .placement
            .devices
            .iter()
            .any(|device| !DEVICES.contains(&device.class))
        {
            return Err(EngineError::PlacementUnsatisfiable(format!(
                "candle engine has no device for {:?}",
                spec.placement.devices
            )));
        }
        spec.manifest.verify_files(&spec.base_dir)?;

        let model_file = spec
            .manifest
            .files
            .first()
            .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
        let path = spec.base_dir.join(&model_file.path);
        let text = spec
            .manifest
            .text
            .as_ref()
            .ok_or_else(|| EngineError::ManifestInvalid("text config required".into()))?;

        let model = match spec.manifest.format {
            ModelFormat::Gguf => {
                let mut file = std::fs::File::open(&path).map_err(|error| {
                    EngineError::LoadFailed(format!("open {}: {error}", path.display()))
                })?;
                let content = gguf_file::Content::read(&mut file)
                    .map_err(|error| EngineError::LoadFailed(format!("gguf metadata: {error}")))?;
                CandleModel::Quantized(
                    ModelWeights::from_gguf(content, &mut file, &self.device).map_err(|error| {
                        EngineError::LoadFailed(format!("gguf weights: {error}"))
                    })?,
                )
            }
            ModelFormat::Safetensors => {
                let config_file = text.config_file.as_ref().ok_or_else(|| {
                    EngineError::ManifestInvalid("text.config_file is required".into())
                })?;
                let raw = std::fs::read_to_string(spec.base_dir.join(config_file))
                    .map_err(|error| EngineError::LoadFailed(format!("config: {error}")))?;
                let config: Qwen3Config = serde_json::from_str(&raw)
                    .map_err(|error| EngineError::LoadFailed(format!("config json: {error}")))?;
                let dtype = match spec.manifest.quantization {
                    Quantization::Bf16 => DType::BF16,
                    Quantization::Fp16 => DType::F16,
                    _ => DType::F32,
                };
                let vb = unsafe {
                    candle_nn::VarBuilder::from_mmaped_safetensors(
                        &[path.as_path()],
                        dtype,
                        &self.device,
                    )
                }
                .map_err(|error| EngineError::LoadFailed(format!("safetensors: {error}")))?;
                CandleModel::Dense(
                    ModelForCausalLM::new(&config, vb)
                        .map_err(|error| EngineError::LoadFailed(format!("qwen3: {error}")))?,
                )
            }
            _ => unreachable!(),
        };

        let vocab = text.vocab_size;
        if vocab == 0 {
            return Err(EngineError::ManifestInvalid(
                "text.vocab_size is required".into(),
            ));
        }

        Ok(Box::new(CandleSession {
            model: Mutex::new(model),
            device: self.device.clone(),
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

enum CandleModel {
    Quantized(ModelWeights),
    Dense(ModelForCausalLM),
}

impl CandleModel {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Quantized(model) => {
                model.clear_kv_cache();
                model.forward(input, offset)
            }
            Self::Dense(model) => {
                model.clear_kv_cache();
                model.forward(input, offset)
            }
        }
    }
}

struct CandleSession {
    model: Mutex<CandleModel>,
    device: Device,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl EngineSession for CandleSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<CoreTensor>) -> Result<Vec<CoreTensor>> {
        let ids = inputs
            .iter()
            .find(|tensor| tensor.name == "input_ids")
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) => Some(values.clone()),
                _ => None,
            })
            .ok_or_else(|| EngineError::Other("missing int64 input_ids".into()))?;
        let ids = ids
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| EngineError::Other("id overflow".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let input = Tensor::new(ids.as_slice(), &self.device)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|error| EngineError::Other(format!("input tensor: {error}")))?;

        let mut model = self
            .model
            .lock()
            .map_err(|_| EngineError::Other("candle model lock poisoned".into()))?;
        let logits = model
            .forward(&input, 0)
            .map_err(|error| EngineError::Other(format!("candle forward: {error}")))?;
        let logits = logits
            .to_dtype(DType::F32)
            .map_err(|error| EngineError::Other(format!("logits dtype: {error}")))?;
        let shape = logits.dims().to_vec();
        let values = logits
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|error| EngineError::Other(format!("logits: {error}")))?;
        Ok(vec![CoreTensor::new(
            "logits",
            shape,
            TensorData::F32(values),
        )?])
    }

    fn cancel(&self) {}
}
