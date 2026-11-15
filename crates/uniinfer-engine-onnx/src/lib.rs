//! ONNX Runtime engine.

use std::path::Path;
use std::sync::LazyLock;

use ort::environment::Environment;
use ort::session::Session;
use uniinfer_core::{
    Device, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, Quantization, Result,
};

mod asr;
mod genai;
mod graph;
mod io;
mod session;

use genai::{GenaiSpec, load_genai_config};
use graph::Graph;
use session::{GenaiOnnxSession, PlainOnnxSession};

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
const QUANTIZATIONS: &[Quantization] = &[
    Quantization::Fp32,
    Quantization::Fp16,
    Quantization::Int8,
    Quantization::Int4,
];

static ENVIRONMENT: LazyLock<std::result::Result<Environment, String>> =
    LazyLock::new(build_environment);

fn build_environment() -> std::result::Result<Environment, String> {
    #[allow(unused_mut)]
    let mut providers: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
    // NPU providers are registered first so ORT prefers them when available.
    #[cfg(feature = "ep-qnn")]
    providers.push(ort::ep::QNN::default().build());
    #[cfg(feature = "ep-vitis")]
    providers.push(ort::ep::Vitis::default().build());
    #[cfg(feature = "ep-nnapi")]
    providers.push(ort::ep::NNAPI::default().build());
    #[cfg(feature = "ep-cann")]
    providers.push(ort::ep::CANN::default().build());
    #[cfg(feature = "ep-coreml")]
    providers.push(ort::ep::CoreML::default().build());
    #[cfg(feature = "ep-openvino")]
    providers.push(ort::ep::OpenVINO::default().build());
    #[cfg(feature = "ep-tensorrt")]
    providers.push(ort::ep::TensorRT::default().build());
    #[cfg(feature = "ep-cuda")]
    providers.push(ort::ep::CUDA::default().build());
    #[cfg(feature = "ep-rocm")]
    providers.push(ort::ep::ROCm::default().build());
    #[cfg(feature = "ep-directml")]
    providers.push(ort::ep::DirectML::default().build());

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
        let class = spec
            .placement
            .devices
            .first()
            .map(|device| device.class)
            .ok_or_else(|| EngineError::PlacementUnsatisfiable("empty placement".into()))?;
        spec.manifest.verify_files(&spec.base_dir)?;

        let model_file = spec
            .manifest
            .files
            .first()
            .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
        let model_path = spec.base_dir.join(&model_file.path);

        // Encoder-decoder ASR manifests carry their own graph files. Keep the
        // dispatch here so ONNX stays the first choice for every ONNX model.
        if let Some(audio) = &spec.manifest.audio
            && audio.architecture.is_some()
        {
            return asr::load(&self.environment, &spec, class);
        }

        let session = build_session(&self.environment, class, &model_path)?;

        let genai_config = load_genai_config(&spec.base_dir)?;
        if let Some(config) = genai_config {
            let graph = Graph::from_session(session, "onnx genai")?;
            let genai = GenaiSpec::from_config(&config, &graph)?;
            let inputs = genai.input_specs(&graph)?;
            let outputs = genai.output_specs(&graph)?;
            tracing::debug!(
                model = %spec.manifest.id,
                path = %model_path.display(),
                inputs = inputs.len(),
                outputs = outputs.len(),
                genai = true,
                "onnx genai session loaded"
            );
            return Ok(Box::new(GenaiOnnxSession::new(graph, genai)?));
        }

        let graph = Graph::from_session(session, "onnx")?;
        tracing::debug!(
            model = %spec.manifest.id,
            path = %model_path.display(),
            inputs = graph.inputs().len(),
            outputs = graph.outputs().len(),
            genai = false,
            "onnx session loaded"
        );
        Ok(Box::new(PlainOnnxSession::new(graph)))
    }
}

pub(crate) fn build_session(
    environment: &Environment,
    class: DeviceClass,
    path: &Path,
) -> Result<Session> {
    let builder = Session::builder(environment)
        .map_err(|error| EngineError::LoadFailed(format!("session builder: {error}")))?;
    // Honor a CPU placement even when GPU EPs are registered in the
    // process-wide ORT environment.
    let mut builder = if class == DeviceClass::Cpu {
        builder
            .with_no_environment_execution_providers()
            .map_err(|error| {
                EngineError::LoadFailed(format!("cpu-only session options: {error}"))
            })?
    } else {
        builder
    };
    builder.commit_from_file(path).map_err(|error| {
        EngineError::LoadFailed(format!("cannot load {}: {error}", path.display()))
    })
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
