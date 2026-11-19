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

mod adapter;
mod asr;
mod genai;
mod generic;
mod graph;
mod image;
mod io;
mod layered;
mod partition;
mod pipeline;
mod session;

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
        feature = "ep-vitis",
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

#[allow(clippy::vec_init_then_push)]
fn build_environment() -> std::result::Result<Environment, String> {
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
        #[allow(unused_mut)]
        let mut devices: Vec<Device> = self
            .environment
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
            .collect();
        // Vitis AI EP does not expose its NPU through ORT device enumeration.
        // A build with the provider compiled in targets the local Ryzen AI NPU,
        // so advertise it explicitly for placement.
        #[cfg(feature = "ep-vitis")]
        if !devices
            .iter()
            .any(|device| device.class == DeviceClass::Npu)
        {
            devices.push(Device {
                id: "npu:vitis:0".into(),
                class: DeviceClass::Npu,
                backend: "vitis".into(),
                memory_bytes: 0,
                available_memory_bytes: 0,
                unified_memory: false,
            });
        }
        devices
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

        adapter::load(&self.environment, &spec, class)
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
    class: DeviceClass,
) -> Result<Vec<ort::ep::ExecutionProviderDispatch>> {
    let variant = variant
        .map(str::trim)
        .filter(|variant| !variant.is_empty())
        .map(str::to_ascii_lowercase);
    if let Some(variant) = variant.as_deref() {
        match variant {
            "cpu" => return Ok(Vec::new()),
            "auto" | "builtin" | "gpu" | "npu" => {}
            _ => {
                #[allow(unused_mut, unused_variables)]
                let mut providers: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
                macro_rules! select {
                    ($name:literal, $feature:literal, $provider:expr) => {
                        #[cfg(feature = $feature)]
                        if variant == $name {
                            providers.push($provider);
                            return Ok(providers);
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
                return Err(EngineError::UnsupportedEngine(format!(
                    "ONNX execution provider {variant:?} is not compiled into this build"
                )));
            }
        }
    }

    let providers = providers_for_class(class);
    if providers.is_empty() && class != DeviceClass::Cpu {
        return Err(EngineError::UnsupportedEngine(format!(
            "no {class:?} execution provider is compiled into this build"
        )));
    }
    Ok(providers)
}

/// Providers that can actually satisfy a placement class. The environment
/// provider list deliberately contains every compiled EP so device probing can
/// see them, but sessions must not inherit an unrelated accelerator: an NPU
/// placement that silently falls through to DirectML or CUDA is a bug.
#[allow(unused_mut)]
fn providers_for_class(class: DeviceClass) -> Vec<ort::ep::ExecutionProviderDispatch> {
    let mut providers = Vec::new();
    match class {
        DeviceClass::Cpu => {}
        DeviceClass::Npu => {
            #[cfg(feature = "ep-qnn")]
            providers.push(ort::ep::QNN::default().build());
            #[cfg(feature = "ep-vitis")]
            providers.push(ort::ep::Vitis::default().build());
            #[cfg(feature = "ep-nnapi")]
            providers.push(ort::ep::NNAPI::default().build());
            #[cfg(feature = "ep-cann")]
            providers.push(ort::ep::CANN::default().build());
            #[cfg(feature = "ep-openvino")]
            providers.push(ort::ep::OpenVINO::default().build());
        }
        DeviceClass::Gpu => {
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
        }
    }
    providers
}

pub(crate) fn build_session_from_memory(
    environment: &Environment,
    class: DeviceClass,
    bytes: &[u8],
    variant: Option<&str>,
) -> Result<Session> {
    let builder = Session::builder(environment)
        .map_err(|error| EngineError::LoadFailed(format!("session builder: {error}")))?;
    let providers = selected_providers(variant, class)?;
    let mut builder = if providers.is_empty() {
        builder
            .with_no_environment_execution_providers()
            .map_err(|error| {
                EngineError::LoadFailed(format!("cpu-only session options: {error}"))
            })?
    } else {
        let names = providers
            .iter()
            .map(|provider| provider.name())
            .collect::<Vec<_>>()
            .join(", ");
        tracing::debug!(class = ?class, variant = variant.unwrap_or("auto"), %names, "onnx session providers");
        let builder = builder
            .with_no_environment_execution_providers()
            .map_err(|error| {
                EngineError::LoadFailed(format!("execution provider options: {error}"))
            })?;
        builder
            .with_execution_providers(providers)
            .map_err(|error| EngineError::LoadFailed(format!("execution providers: {error}")))?
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
    let providers = selected_providers(variant, class)?;
    let mut builder = if providers.is_empty() {
        builder
            .with_no_environment_execution_providers()
            .map_err(|error| {
                EngineError::LoadFailed(format!("cpu-only session options: {error}"))
            })?
    } else {
        let names = providers
            .iter()
            .map(|provider| provider.name())
            .collect::<Vec<_>>()
            .join(", ");
        tracing::debug!(class = ?class, variant = variant.unwrap_or("auto"), %names, "onnx session providers");
        let builder = builder
            .with_no_environment_execution_providers()
            .map_err(|error| {
                EngineError::LoadFailed(format!("execution provider options: {error}"))
            })?;
        builder
            .with_execution_providers(providers)
            .map_err(|error| EngineError::LoadFailed(format!("execution providers: {error}")))?
    };
    #[cfg(feature = "ep-vitis")]
    {
        // Vitis AI needs its dynamic-dispatch cache and, for graphs with
        // weights in a sidecar file, the external-data file name.
        if let Some(parent) = path.parent()
            && parent
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case("dd"))
        {
            let cache = parent.join("cache");
            if cache.is_dir() {
                builder = builder
                    .with_config_entry("dd_cache", cache.to_string_lossy())
                    .map_err(|error| {
                        EngineError::LoadFailed(format!("Vitis dd_cache option: {error}"))
                    })?;
            }
            builder = builder
                .with_config_entry("onnx_custom_ops_const_key", path.to_string_lossy())
                .map_err(|error| {
                    EngineError::LoadFailed(format!("Vitis const key option: {error}"))
                })?;
        }
        let external = path.with_extension("onnx.data");
        if external.is_file()
            && let Some(name) = external.file_name().and_then(|name| name.to_str())
        {
            builder = builder
                .with_config_entry("external_data_file", name)
                .map_err(|error| {
                    EngineError::LoadFailed(format!("Vitis external data option: {error}"))
                })?;
        }
    }
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

#[cfg(test)]
mod provider_tests {
    use super::*;

    #[test]
    fn cpu_selection_has_no_accelerator_provider() {
        let selected = selected_providers(Some("cpu"), DeviceClass::Cpu).expect("cpu selection");
        assert!(selected.is_empty());
    }

    #[cfg(feature = "ep-vitis")]
    #[test]
    fn npu_class_selects_only_npu_providers() {
        let providers = providers_for_class(DeviceClass::Npu);
        assert!(
            providers
                .iter()
                .any(|provider| provider.name().contains("Vitis"))
        );
        assert!(
            !providers
                .iter()
                .any(|provider| provider.name().contains("Dml"))
        );
        assert!(
            !providers
                .iter()
                .any(|provider| provider.name().contains("CUDA"))
        );
    }

    #[test]
    fn unknown_named_provider_is_rejected() {
        let error = match selected_providers(Some("not-a-provider"), DeviceClass::Cpu) {
            Err(error) => error,
            Ok(_) => panic!("unknown provider must not fall back"),
        };
        assert!(error.to_string().contains("not-a-provider"));
    }
}
