//! ONNX Runtime engine.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use ort::environment::Environment;
use ort::session::Session;
use ort::value::{DynValue, Outlet, Tensor as OrtTensor, TensorElementType, ValueType};
use uniinfer_core::{
    DeviceClass, EngineCapabilities, EngineError, EngineLoadSpec, EngineSession, InferenceEngine,
    Modality, ModelFormat, Quantization, Result, Tensor, TensorData, TensorDtype, TensorSpec,
};

const FORMATS: &[ModelFormat] = &[ModelFormat::Onnx];
const DEVICES: &[DeviceClass] = &[DeviceClass::Cpu];
const MODALITIES: &[Modality] = &[
    Modality::Text,
    Modality::Audio,
    Modality::Image,
    Modality::Video,
];
const QUANTIZATIONS: &[Quantization] =
    &[Quantization::Fp32, Quantization::Fp16, Quantization::Int8];

/// ONNX Runtime engine with the CPU execution provider.
pub struct OnnxEngine {
    environment: Environment,
}

impl OnnxEngine {
    pub fn new() -> Result<Self> {
        let environment = ort::init()
            .with_name("uniinfer")
            .build()
            .map_err(|error| EngineError::LoadFailed(format!("ort environment: {error}")))?;
        Ok(Self { environment })
    }
}

impl InferenceEngine for OnnxEngine {
    fn engine_name(&self) -> &'static str {
        "onnx"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        FORMATS
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Onnx => MODALITIES,
            _ => &[],
        }
    }

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization] {
        match format {
            ModelFormat::Onnx => QUANTIZATIONS,
            _ => &[],
        }
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        DEVICES
    }

    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities {
            formats: FORMATS.to_vec(),
            modalities: vec![(ModelFormat::Onnx, MODALITIES.to_vec())],
            quantizations: vec![(ModelFormat::Onnx, QUANTIZATIONS.to_vec())],
            devices: DEVICES.to_vec(),
        }
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if spec.manifest.format != ModelFormat::Onnx {
            return Err(EngineError::UnsupportedFormat(spec.manifest.format));
        }
        if !QUANTIZATIONS.contains(&spec.manifest.quantization) {
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
                "onnx engine has no device for {:?}",
                spec.placement.devices
            )));
        }
        spec.manifest.verify_files(&spec.base_dir)?;

        let model_file = spec
            .manifest
            .files
            .first()
            .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
        let model_path = spec.base_dir.join(&model_file.path);

        let session = Session::builder(&self.environment)
            .map_err(|error| EngineError::LoadFailed(format!("session builder: {error}")))?
            .commit_from_file(&model_path)
            .map_err(|error| {
                EngineError::LoadFailed(format!("cannot load {}: {error}", model_path.display()))
            })?;

        let inputs = convert_specs(session.inputs(), "input")?;
        let outputs = convert_specs(session.outputs(), "output")?;

        tracing::debug!(
            model = %spec.manifest.id,
            path = %model_path.display(),
            inputs = inputs.len(),
            outputs = outputs.len(),
            "onnx session loaded"
        );

        Ok(Box::new(OnnxSession {
            session: Mutex::new(session),
            inputs,
            outputs,
            cancelled: AtomicBool::new(false),
        }))
    }
}

fn convert_specs(outlets: &[Outlet], kind: &str) -> Result<Vec<TensorSpec>> {
    outlets
        .iter()
        .map(|outlet| {
            let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
                return Err(EngineError::LoadFailed(format!(
                    "{kind} {} is not a tensor",
                    outlet.name()
                )));
            };
            let dtype = match *ty {
                TensorElementType::Float32 => TensorDtype::F32,
                TensorElementType::Int64 => TensorDtype::I64,
                TensorElementType::Uint8 => TensorDtype::U8,
                other => {
                    return Err(EngineError::LoadFailed(format!(
                        "{kind} {} has unsupported dtype {other:?}",
                        outlet.name()
                    )));
                }
            };
            let shape = shape
                .iter()
                .map(|dim| if *dim >= 0 { Some(*dim as usize) } else { None })
                .collect();
            Ok(TensorSpec {
                name: outlet.name().to_owned(),
                dtype,
                shape,
            })
        })
        .collect()
}

struct OnnxSession {
    session: Mutex<Session>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    cancelled: AtomicBool,
}

impl EngineSession for OnnxSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }

        let mut values: Vec<(String, DynValue)> = Vec::with_capacity(inputs.len());
        for input in inputs {
            if !self.inputs.iter().any(|spec| spec.name == input.name) {
                return Err(EngineError::Other(format!(
                    "unknown input tensor {}",
                    input.name
                )));
            }
            values.push((input.name.clone(), to_ort(input)?));
        }

        for spec in &self.inputs {
            if !values.iter().any(|(name, _)| name == &spec.name) {
                return Err(EngineError::Other(format!(
                    "missing input tensor {}",
                    spec.name
                )));
            }
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| EngineError::Other("session lock poisoned".into()))?;
        let outputs = session
            .run(values)
            .map_err(|error| EngineError::Other(format!("onnx run: {error}")))?;

        let mut result = Vec::with_capacity(self.outputs.len());
        for spec in &self.outputs {
            let value = &outputs[spec.name.as_str()];
            result.push(from_ort(spec, value)?);
        }
        Ok(result)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

fn to_ort(tensor: Tensor) -> Result<DynValue> {
    let shape = tensor.shape.clone();
    let value = match tensor.data {
        TensorData::F32(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
        TensorData::I64(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
        TensorData::U8(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
    }
    .map_err(|error| EngineError::Other(format!("tensor {}: {error}", tensor.name)))?;
    Ok(value)
}

fn from_ort(spec: &TensorSpec, value: &DynValue) -> Result<Tensor> {
    match spec.dtype {
        TensorDtype::F32 => {
            let (shape, slice) = value
                .try_extract_tensor::<f32>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::F32(slice.to_vec()))
        }
        TensorDtype::I64 => {
            let (shape, slice) = value
                .try_extract_tensor::<i64>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::I64(slice.to_vec()))
        }
        TensorDtype::U8 => {
            let (shape, slice) = value
                .try_extract_tensor::<u8>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::U8(slice.to_vec()))
        }
    }
}
