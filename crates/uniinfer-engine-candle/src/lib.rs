//! Candle engine for Qwen3 GGUF and Safetensors models.

mod qwen3;

use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use uniinfer_core::{
    Device as CoreDevice, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine,
    Modality, ModelFormat, Quantization, Result, Tensor as CoreTensor, TensorData, TensorDtype,
    TensorSpec,
};

const FORMATS: &[ModelFormat] = &[ModelFormat::Gguf, ModelFormat::Safetensors];
const MODALITIES: &[Modality] = &[Modality::Text];
const FAMILY: &str = "qwen3";

/// Candle engine; the concrete device is chosen per session from the plan.
pub struct CandleEngine {
    devices: Vec<DeviceClass>,
}

impl CandleEngine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            devices: probe_devices(),
        })
    }

    #[must_use]
    pub fn probe_devices(&self) -> Vec<CoreDevice> {
        self.devices
            .iter()
            .map(|class| CoreDevice {
                id: format!("{class}:candle:0"),
                class: *class,
                backend: "candle".into(),
                memory_bytes: 0,
            })
            .collect()
    }
}

fn probe_devices() -> Vec<DeviceClass> {
    let mut devices = vec![DeviceClass::Cpu];
    if gpu_available() {
        devices.push(DeviceClass::Gpu);
    }
    devices
}

fn gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    if Device::new_cuda(0).is_ok() {
        return true;
    }
    #[cfg(feature = "metal")]
    if Device::new_metal(0).is_ok() {
        return true;
    }
    false
}

fn device_for(class: DeviceClass) -> Result<Device> {
    match class {
        DeviceClass::Cpu => Ok(Device::Cpu),
        DeviceClass::Gpu => {
            #[cfg(feature = "cuda")]
            if let Ok(device) = Device::new_cuda(0) {
                return Ok(device);
            }
            #[cfg(feature = "metal")]
            if let Ok(device) = Device::new_metal(0) {
                return Ok(device);
            }
            Err(EngineError::PlacementUnsatisfiable(
                "candle GPU backend is unavailable".into(),
            ))
        }
        DeviceClass::Npu => Err(EngineError::PlacementUnsatisfiable(
            "candle has no NPU backend".into(),
        )),
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
            ModelFormat::Gguf => qwen3::GGUF_QUANTIZATIONS,
            ModelFormat::Safetensors => qwen3::SAFETENSORS_QUANTIZATIONS,
            _ => &[],
        }
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        &self.devices
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if !FORMATS.contains(&spec.manifest.format) {
            return Err(EngineError::UnsupportedFormat(spec.manifest.format));
        }
        if !self
            .supported_quantizations(spec.manifest.format)
            .contains(&spec.manifest.quantization)
        {
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
        if !self.devices.contains(&class) {
            return Err(EngineError::PlacementUnsatisfiable(format!(
                "candle has no {class} device"
            )));
        }
        let device = device_for(class)?;
        spec.manifest.verify_files(&spec.base_dir)?;

        let text = spec
            .manifest
            .text
            .as_ref()
            .ok_or_else(|| EngineError::ManifestInvalid("text config required".into()))?;
        if text.family.as_deref() != Some(FAMILY) {
            return Err(EngineError::UnsupportedEngine(format!(
                "candle family {:?} is not implemented",
                text.family
            )));
        }
        let model_file = spec
            .manifest
            .files
            .first()
            .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
        let path = spec.base_dir.join(&model_file.path);

        let model = match spec.manifest.format {
            ModelFormat::Gguf => qwen3::load_gguf(&path, &device)?,
            ModelFormat::Safetensors => {
                let config_file = text.config_file.as_ref().ok_or_else(|| {
                    EngineError::ManifestInvalid("text.config_file is required".into())
                })?;
                let dtype = match spec.manifest.quantization {
                    Quantization::Bf16 => DType::BF16,
                    Quantization::Fp16 => DType::F16,
                    _ => DType::F32,
                };
                qwen3::load_safetensors(&path, &spec.base_dir.join(config_file), dtype, &device)?
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
            device,
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

struct CandleSession {
    model: Mutex<qwen3::Model>,
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
