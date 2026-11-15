//! Candle engine with a manifest-driven text family registry.

mod cancel_io;
mod qwen3;
mod text;

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use uniinfer_core::{
    Device as CoreDevice, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine,
    Modality, ModelFormat, Quantization, Result, Tensor as CoreTensor, TensorData, TensorDtype,
    TensorSpec,
};

const FORMATS: &[ModelFormat] = &[ModelFormat::Gguf, ModelFormat::Safetensors];
const MODALITIES: &[Modality] = &[Modality::Text];

/// Candle engine; the concrete device is chosen per session from the plan.
pub struct CandleEngine {
    devices: Vec<DeviceClass>,
    text: text::TextRegistry,
}

impl CandleEngine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            devices: probe_devices(),
            text: text::TextRegistry::new(),
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
            ModelFormat::Gguf => text::GGUF_QUANTIZATIONS,
            ModelFormat::Safetensors => text::SAFETENSORS_QUANTIZATIONS,
            _ => &[],
        }
    }

    fn supports_architecture(&self, manifest: &uniinfer_core::ModelManifest) -> bool {
        manifest
            .text
            .as_ref()
            .and_then(|text| text.family.as_deref())
            .is_some_and(|family| self.text.supported(family))
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
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        spec.manifest.verify_files(&spec.base_dir)?;

        let text = spec
            .manifest
            .text
            .as_ref()
            .ok_or_else(|| EngineError::ManifestInvalid("text config required".into()))?;
        let family_name = text
            .family
            .as_deref()
            .ok_or_else(|| EngineError::ManifestInvalid("text.family is required".into()))?;
        let family =
            self.text
                .get(family_name)
                .ok_or_else(|| EngineError::UnsupportedArchitecture {
                    format: spec.manifest.format,
                    architecture: family_name.to_owned(),
                })?;
        let model_paths = match spec.manifest.format {
            ModelFormat::Gguf => spec
                .manifest
                .files
                .first()
                .map(|file| vec![spec.base_dir.join(&file.path)])
                .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?,
            ModelFormat::Safetensors => {
                let shards = spec
                    .manifest
                    .files
                    .iter()
                    .map(|file| spec.base_dir.join(&file.path))
                    .filter(|path| {
                        path.extension()
                            .is_some_and(|extension| extension == "safetensors")
                    })
                    .collect::<Vec<_>>();
                if shards.is_empty() {
                    return Err(EngineError::ManifestInvalid(
                        "no .safetensors shards in files".into(),
                    ));
                }
                shards
            }
            _ => unreachable!(),
        };
        let dtype = match spec.manifest.quantization {
            Quantization::Bf16 => DType::BF16,
            Quantization::Fp16 => DType::F16,
            _ => DType::F32,
        };
        let config_path = spec.base_dir.join(
            text.config_file
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new("")),
        );
        let model = family.load(&text::LoadContext {
            format: spec.manifest.format,
            model_paths: &model_paths,
            config_path: &config_path,
            dtype,
            device: &device,
            cancel: &spec.cancel,
        })?;

        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let vocab = text.vocab_size;
        if vocab == 0 {
            return Err(EngineError::ManifestInvalid(
                "text.vocab_size is required".into(),
            ));
        }

        Ok(Box::new(CandleSession {
            state: Mutex::new(CandleState {
                model,
                cached_ids: Vec::new(),
            }),
            cancelled: AtomicBool::new(false),
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

struct CandleState {
    model: Box<dyn text::TextModel>,
    cached_ids: Vec<u32>,
}

struct CandleSession {
    state: Mutex<CandleState>,
    cancelled: AtomicBool,
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
        let ids = ids
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| EngineError::Other("id overflow".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("candle state lock poisoned".into()))?;
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let incremental = state.cached_ids.len() + 1 == ids.len()
            && ids.starts_with(&state.cached_ids)
            && !state.cached_ids.is_empty();
        let (tokens, offset) = if incremental {
            (&ids[ids.len() - 1..], state.cached_ids.len())
        } else {
            state.model.clear();
            state.cached_ids.clear();
            (ids.as_slice(), 0)
        };
        let input = Tensor::new(tokens, &self.device)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|error| EngineError::Other(format!("input tensor: {error}")))?;
        let logits = state
            .model
            .forward(&input, offset)
            .map_err(|error| EngineError::Other(format!("candle forward: {error}")))?;
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let logits = logits
            .to_dtype(DType::F32)
            .map_err(|error| EngineError::Other(format!("logits dtype: {error}")))?;
        let shape = logits.dims().to_vec();
        let values = logits
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|error| EngineError::Other(format!("logits: {error}")))?;
        state.cached_ids = ids;
        Ok(vec![CoreTensor::new(
            "logits",
            shape,
            TensorData::F32(values),
        )?])
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}
