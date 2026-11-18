//! ONNX Runtime engine.

use std::path::Path;
#[cfg(feature = "load-dynamic")]
use std::path::PathBuf;
use std::sync::LazyLock;
#[cfg(feature = "load-dynamic")]
use std::sync::OnceLock;

use ort::environment::Environment;
use ort::session::Session;
use uniinfer_core::{
    Device, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, Quantization, Result, Runtime,
};

mod asr;
mod diffusion;
mod genai;
mod graph;
mod io;
mod layered;
mod partition;
mod pipeline;
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
        feature = "ep-directml",
        feature = "ep-openvino",
        feature = "ep-onednn",
        feature = "ep-webgpu"
    )) {
        devices.push(DeviceClass::Gpu);
    }
    if cfg!(any(
        feature = "ep-nnapi",
        feature = "ep-qnn",
        feature = "ep-openvino",
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

#[cfg(feature = "load-dynamic")]
static DYNAMIC_LIBRARY: OnceLock<PathBuf> = OnceLock::new();

/// Select the ONNX Runtime shared library used by the process-wide engine.
///
/// This must be called before the first [`OnnxEngine`] is constructed. The
/// desktop backend store calls it when an installed ONNX Runtime package is
/// selected; rebuilds that use the default static ORT do not call it.
#[cfg(feature = "load-dynamic")]
pub fn configure_dynamic_library(path: impl Into<PathBuf>) -> Result<()> {
    DYNAMIC_LIBRARY
        .set(path.into())
        .map_err(|_| EngineError::LoadFailed("ONNX Runtime library already configured".into()))
}

fn build_environment() -> std::result::Result<Environment, String> {
    #[allow(unused_mut)]
    #[allow(unused_mut, unused_variables)]
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
    #[cfg(feature = "ep-nvrtx")]
    providers.push(ort::ep::NVRTX::default().build());
    #[cfg(feature = "ep-cuda")]
    providers.push(ort::ep::CUDA::default().build());
    #[cfg(feature = "ep-rocm")]
    providers.push(ort::ep::ROCm::default().build());
    #[cfg(feature = "ep-migraphx")]
    providers.push(ort::ep::MIGraphX::default().build());
    #[cfg(feature = "ep-directml")]
    providers.push(ort::ep::DirectML::default().build());
    #[cfg(feature = "ep-webgpu")]
    providers.push(ort::ep::WebGPU::default().build());
    #[cfg(feature = "ep-onednn")]
    providers.push(ort::ep::OneDNN::default().build());
    #[cfg(feature = "ep-xnnpack")]
    providers.push(ort::ep::XNNPACK::default().build());
    #[cfg(feature = "ep-acl")]
    providers.push(ort::ep::ACL::default().build());

    #[cfg(feature = "load-dynamic")]
    let builder = {
        let path = DYNAMIC_LIBRARY
            .get()
            .cloned()
            .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from))
            .unwrap_or_else(|| {
                PathBuf::from(format!(
                    "{}onnxruntime{}",
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                ))
            });
        ort::init_from(path).map_err(|error| error.to_string())?
    };
    #[cfg(not(feature = "load-dynamic"))]
    let builder = ort::init();
    builder
        .with_name("uniinfer")
        .with_execution_providers(providers)
        .build()
        .map_err(|error| error.to_string())
}

/// ONNX Runtime engine with the configured execution providers.
pub struct OnnxEngine {
    environment: Environment,
    backend_id: String,
    variant: String,
}

impl OnnxEngine {
    pub fn new() -> Result<Self> {
        Self::with_identity("onnx", "builtin")
    }

    /// Create an engine that advertises which installed backend selected it.
    pub fn with_identity(
        backend_id: impl Into<String>,
        variant: impl Into<String>,
    ) -> Result<Self> {
        let environment = ENVIRONMENT
            .as_ref()
            .map_err(|error| EngineError::LoadFailed(format!("ort environment: {error}")))?
            .clone();
        Ok(Self {
            environment,
            backend_id: backend_id.into(),
            variant: variant.into(),
        })
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
                let unified_memory =
                    matches!(backend.to_ascii_lowercase().as_str(), "coreml" | "metal");
                Some(Device {
                    id: format!("{class}:{backend}:{}", hardware.id()),
                    class,
                    backend,
                    memory_bytes: 0,
                    available_memory_bytes: 0,
                    unified_memory,
                })
            })
            .collect()
    }
}

impl InferenceEngine for OnnxEngine {
    fn engine_name(&self) -> &'static str {
        "onnx"
    }

    fn runtime(&self) -> Runtime {
        Runtime::Onnx
    }

    fn backend_id(&self) -> &str {
        &self.backend_id
    }

    fn backend_variant(&self) -> &str {
        &self.variant
    }

    fn implementations(&self) -> &[&'static str] {
        &[
            "onnx.graph",
            "onnx.pipeline",
            "onnx.diffusion",
            "onnx.layered",
        ]
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

    fn supports_layer_offload(&self) -> bool {
        true
    }

    fn supports_layer_offload_for(&self, implementation: &str) -> bool {
        implementation == "onnx.layered"
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
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

        if spec.manifest.implementation == "onnx.pipeline" {
            return pipeline::load(&self.environment, &spec, class);
        }
        if spec.manifest.implementation == "onnx.diffusion" {
            return diffusion::load(&self.environment, &spec, class);
        }
        if spec.manifest.implementation == "onnx.layered" {
            return layered::load(&self.environment, &spec, class);
        }

        let model_file = spec
            .manifest
            .files
            .first()
            .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
        let model_path = spec.base_dir.join(&model_file.path);

        // Encoder-decoder ASR manifests carry their own graph files. Keep the
        // dispatch here so ONNX stays the first choice for every ONNX model.
        let session = build_session(
            &self.environment,
            class,
            &model_path,
            backend_variant(&spec),
        )?;
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }

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
            if spec.cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
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
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        Ok(Box::new(PlainOnnxSession::new(graph)))
    }
}

pub(crate) fn backend_variant(spec: &EngineLoadSpec) -> Option<&str> {
    spec.manifest
        .backend
        .as_ref()
        .and_then(|backend| backend.variant.as_deref())
        .map(str::trim)
        .filter(|variant| !variant.is_empty())
}

fn selected_providers(
    variant: Option<&str>,
) -> Result<Option<Vec<ort::ep::ExecutionProviderDispatch>>> {
    let Some(variant) = variant else {
        return Ok(None);
    };
    let variant = variant.to_ascii_lowercase();
    if matches!(variant.as_str(), "auto" | "builtin" | "gpu" | "npu") {
        return Ok(None);
    }
    if variant == "cpu" {
        return Ok(Some(Vec::new()));
    }
    #[allow(unused_mut, unused_variables)]
    let mut providers: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
    macro_rules! select {
        ($name:literal, $feature:literal, $provider:expr) => {
            #[cfg(feature = $feature)]
            if variant == $name {
                providers.push($provider);
                return Ok(Some(providers));
            }
        };
    }
    select!("cuda", "ep-cuda", ort::ep::CUDA::default().build());
    select!(
        "tensorrt",
        "ep-tensorrt",
        ort::ep::TensorRT::default().build()
    );
    select!(
        "directml",
        "ep-directml",
        ort::ep::DirectML::default().build()
    );
    select!(
        "openvino",
        "ep-openvino",
        ort::ep::OpenVINO::default().build()
    );
    select!("rocm", "ep-rocm", ort::ep::ROCm::default().build());
    select!("coreml", "ep-coreml", ort::ep::CoreML::default().build());
    select!("qnn", "ep-qnn", ort::ep::QNN::default().build());
    select!("nnapi", "ep-nnapi", ort::ep::NNAPI::default().build());
    select!("cann", "ep-cann", ort::ep::CANN::default().build());
    select!("onednn", "ep-onednn", ort::ep::OneDNN::default().build());
    select!("xnnpack", "ep-xnnpack", ort::ep::XNNPACK::default().build());
    select!("acl", "ep-acl", ort::ep::ACL::default().build());
    select!(
        "migraphx",
        "ep-migraphx",
        ort::ep::MIGraphX::default().build()
    );
    select!("nvrtx", "ep-nvrtx", ort::ep::NVRTX::default().build());
    select!("webgpu", "ep-webgpu", ort::ep::WebGPU::default().build());
    select!("vitis", "ep-vitis", ort::ep::Vitis::default().build());
    select!("vitis ai", "ep-vitis", ort::ep::Vitis::default().build());
    Err(EngineError::UnsupportedEngine(format!(
        "ONNX execution provider {variant:?} is not compiled into this build"
    )))
}

pub(crate) fn build_session_from_memory(
    environment: &Environment,
    class: DeviceClass,
    bytes: &[u8],
    variant: Option<&str>,
) -> Result<Session> {
    let builder = Session::builder(environment)
        .map_err(|error| EngineError::LoadFailed(format!("session builder: {error}")))?;
    let selected = selected_providers(variant)?;
    let mut builder = match selected {
        Some(providers) if providers.is_empty() => builder
            .with_no_environment_execution_providers()
            .map_err(|error| {
                EngineError::LoadFailed(format!("cpu-only session options: {error}"))
            })?,
        Some(providers) => builder
            .with_execution_providers(providers)
            .map_err(|error| EngineError::LoadFailed(format!("execution providers: {error}")))?,
        None if class == DeviceClass::Cpu => builder
            .with_no_environment_execution_providers()
            .map_err(|error| {
                EngineError::LoadFailed(format!("cpu-only session options: {error}"))
            })?,
        None => builder,
    };
    builder
        .commit_from_memory(bytes)
        .map_err(|error| EngineError::LoadFailed(format!("cannot load ONNX stage: {error}")))
}

pub(crate) fn build_session(
    environment: &Environment,
    class: DeviceClass,
    path: &Path,
    variant: Option<&str>,
) -> Result<Session> {
    let builder = Session::builder(environment)
        .map_err(|error| EngineError::LoadFailed(format!("session builder: {error}")))?;
    let selected = selected_providers(variant)?;
    let mut builder = match selected {
        Some(providers) if providers.is_empty() => builder
            .with_no_environment_execution_providers()
            .map_err(|error| {
                EngineError::LoadFailed(format!("cpu-only session options: {error}"))
            })?,
        Some(providers) => builder
            .with_execution_providers(providers)
            .map_err(|error| EngineError::LoadFailed(format!("execution providers: {error}")))?,
        None if class == DeviceClass::Cpu => builder
            .with_no_environment_execution_providers()
            .map_err(|error| {
                EngineError::LoadFailed(format!("cpu-only session options: {error}"))
            })?,
        None => builder,
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
        || (cfg!(feature = "ep-acl") && ep.contains("acl"))
        || (cfg!(feature = "ep-xnnpack") && ep.contains("xnnpack"))
        || (cfg!(feature = "ep-nvrtx") && ep.contains("nvrtx"))
        || (cfg!(feature = "ep-migraphx") && ep.contains("migraphx"))
        || (cfg!(feature = "ep-webgpu") && ep.contains("webgpu"))
        || (cfg!(feature = "ep-onednn") && (ep.contains("onednn") || ep.contains("dnnl")))
}
