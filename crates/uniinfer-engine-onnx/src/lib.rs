//! ONNX Runtime engine.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};

use ort::environment::Environment;
use ort::session::Session;
use ort::value::{DynValue, Outlet, Tensor as OrtTensor, TensorElementType, ValueType};
use uniinfer_core::{
    Device, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, Quantization, Result, Tensor, TensorData, TensorDtype, TensorSpec,
};

const FORMATS: &[ModelFormat] = &[ModelFormat::Onnx];
static DEVICES: LazyLock<Vec<DeviceClass>> = LazyLock::new(|| {
    let mut devices = vec![DeviceClass::Cpu];
    if cfg!(any(
        feature = "ep-cuda",
        feature = "ep-tensorrt",
        feature = "ep-rocm",
        feature = "ep-coreml",
        feature = "ep-directml"
    )) {
        devices.push(DeviceClass::Gpu);
    }
    if cfg!(any(
        feature = "ep-nnapi",
        feature = "ep-qnn",
        feature = "ep-openvino",
        feature = "ep-vitis",
        feature = "ep-cann"
    )) {
        devices.push(DeviceClass::Npu);
    }
    devices
});
const MODALITIES: &[Modality] = &[
    Modality::Text,
    Modality::Audio,
    Modality::Image,
    Modality::Video,
];
const QUANTIZATIONS: &[Quantization] =
    &[Quantization::Fp32, Quantization::Fp16, Quantization::Int8];

static ENVIRONMENT: LazyLock<std::result::Result<Environment, String>> =
    LazyLock::new(build_environment);

fn build_environment() -> std::result::Result<Environment, String> {
    #[allow(unused_mut)]
    let mut providers: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
    #[cfg(feature = "ep-tensorrt")]
    providers.push(ort::ep::TensorRT::default().build());
    #[cfg(feature = "ep-cuda")]
    providers.push(ort::ep::CUDA::default().build());
    #[cfg(feature = "ep-rocm")]
    providers.push(ort::ep::ROCm::default().build());
    #[cfg(feature = "ep-coreml")]
    providers.push(ort::ep::CoreML::default().build());
    #[cfg(feature = "ep-directml")]
    providers.push(ort::ep::DirectML::default().build());
    #[cfg(feature = "ep-qnn")]
    providers.push(ort::ep::QNN::default().build());
    #[cfg(feature = "ep-nnapi")]
    providers.push(ort::ep::NNAPI::default().build());
    #[cfg(feature = "ep-openvino")]
    providers.push(ort::ep::OpenVINO::default().build());
    #[cfg(feature = "ep-vitis")]
    providers.push(ort::ep::Vitis::default().build());
    #[cfg(feature = "ep-cann")]
    providers.push(ort::ep::CANN::default().build());

    ort::init()
        .with_name("uniinfer")
        .with_execution_providers(providers)
        .build()
        .map_err(|error| error.to_string())
}

/// ONNX Runtime engine with the configured execution providers.
pub struct OnnxEngine {
    environment: Environment,
}

impl OnnxEngine {
    pub fn new() -> Result<Self> {
        let environment = ENVIRONMENT
            .as_ref()
            .map_err(|error| EngineError::LoadFailed(format!("ort environment: {error}")))?
            .clone();
        Ok(Self { environment })
    }

    /// Enumerate hardware devices visible to the configured execution providers.
    #[must_use]
    pub fn probe_devices(&self) -> Vec<Device> {
        self.environment
            .devices()
            .filter_map(|device| {
                let hardware = device.hardware_device();
                let class = match hardware.ty() {
                    ort::memory::DeviceType::CPU => DeviceClass::Cpu,
                    ort::memory::DeviceType::GPU => DeviceClass::Gpu,
                    ort::memory::DeviceType::NPU => DeviceClass::Npu,
                    _ => return None,
                };
                let backend = device.ep().unwrap_or("unknown").to_owned();
                if !ep_enabled(&backend) {
                    return None;
                }
                Some(Device {
                    id: format!("{class}:{backend}:{}", hardware.id()),
                    class,
                    backend,
                    memory_bytes: 0,
                })
            })
            .collect()
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
        &DEVICES
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

fn ep_enabled(ep: &str) -> bool {
    let ep = ep.to_ascii_lowercase();
    if ep.contains("cpu") {
        return true;
    }
    (cfg!(feature = "ep-cuda") && ep.contains("cuda"))
        || (cfg!(feature = "ep-tensorrt") && ep.contains("tensorrt"))
        || (cfg!(feature = "ep-rocm") && ep.contains("rocm"))
        || (cfg!(feature = "ep-coreml") && ep.contains("coreml"))
        || (cfg!(feature = "ep-directml") && (ep.contains("dml") || ep.contains("directml")))
        || (cfg!(feature = "ep-qnn") && ep.contains("qnn"))
        || (cfg!(feature = "ep-nnapi") && ep.contains("nnapi"))
        || (cfg!(feature = "ep-openvino") && ep.contains("openvino"))
        || (cfg!(feature = "ep-vitis") && ep.contains("vitis"))
        || (cfg!(feature = "ep-cann") && ep.contains("cann"))
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
