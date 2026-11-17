use uniinfer_core::{
    BackendConfig, Capability, Device, DeviceClass, EngineLoadSpec, EngineSession, InferenceEngine,
    Modality, Mode, ModelFormat, ModelManifest, Operation, PlacementConfig, PlacementPolicy,
    Quantization, Result, Runtime, TensorSpec,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_registry::EngineRegistry;

struct FakeEngine {
    name: &'static str,
    backend: &'static str,
    variant: &'static str,
    transport: bool,
}

impl InferenceEngine for FakeEngine {
    fn engine_name(&self) -> &'static str {
        self.name
    }

    fn runtime(&self) -> Runtime {
        Runtime::Onnx
    }

    fn backend_id(&self) -> &str {
        self.backend
    }

    fn backend_variant(&self) -> &str {
        self.variant
    }

    fn implementations(&self) -> &[&'static str] {
        &["onnx.fake"]
    }

    fn is_transport(&self) -> bool {
        self.transport
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Onnx]
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Onnx => &[Modality::Text],
            _ => &[],
        }
    }

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization] {
        match format {
            ModelFormat::Onnx => &[Quantization::Fp32],
            _ => &[],
        }
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        &[DeviceClass::Cpu, DeviceClass::Gpu]
    }

    fn load_session(&self, _spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        Ok(Box::new(Noop))
    }
}

struct Noop;

impl EngineSession for Noop {
    fn inputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn outputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn run(&self, _inputs: Vec<uniinfer_core::Tensor>) -> Result<Vec<uniinfer_core::Tensor>> {
        Ok(Vec::new())
    }

    fn cancel(&self) {}
}

fn device(id: &str, class: DeviceClass, backend: &str) -> Device {
    Device {
        id: id.into(),
        class,
        backend: backend.into(),
        memory_bytes: 8_000,
        available_memory_bytes: 8_000,
        unified_memory: false,
    }
}

fn inventory() -> DeviceInventory {
    DeviceInventory::from(vec![
        device("cpu:0", DeviceClass::Cpu, "cpu"),
        device("gpu:0", DeviceClass::Gpu, "cuda"),
    ])
}

fn manifest() -> ModelManifest {
    ModelManifest {
        id: "routing".into(),
        runtime: Runtime::Onnx,
        implementation: "onnx.fake".into(),
        capabilities: vec![Capability {
            name: "text_chat".into(),
            operation: Operation::Generate,
            required_inputs: vec![Modality::Text],
            optional_inputs: Vec::new(),
            outputs: vec![Modality::Text],
            mode: Mode::Stream,
        }],
        format: ModelFormat::Onnx,
        quantization: Quantization::Fp32,
        streaming_capable: false,
        context_len: 16,
        tokenizer: Some("bytes".into()),
        min_vram: None,
        placement: PlacementConfig {
            policy: PlacementPolicy::Auto,
            allow_cpu_fallback: true,
            prefer: vec![DeviceClass::Gpu],
            stages: std::collections::BTreeMap::new(),
            offload_layers: None,
            kv_cache: None,
        },
        backend: None,
        files: vec![uniinfer_core::ModelFile {
            path: "model.onnx".into(),
            sha256: "0".repeat(64),
        }],
        text: None,
        audio: None,
        vision: None,
        pipeline: None,
        image: None,
    }
}

fn registry() -> EngineRegistry {
    let registry = EngineRegistry::new(inventory());
    registry.register(std::sync::Arc::new(FakeEngine {
        name: "builtin-onnx",
        backend: "builtin-onnx",
        variant: "builtin",
        transport: false,
    }));
    registry.register(std::sync::Arc::new(FakeEngine {
        name: "alpha-cuda",
        backend: "alpha",
        variant: "CUDA",
        transport: true,
    }));
    registry.register(std::sync::Arc::new(FakeEngine {
        name: "alpha-vulkan",
        backend: "alpha",
        variant: "Vulkan",
        transport: true,
    }));
    registry
}

#[test]
fn explicit_variant_selects_the_transport() {
    let mut manifest = manifest();
    manifest.backend = Some(BackendConfig {
        id: Some("alpha".into()),
        variant: Some("Vulkan".into()),
    });
    let route = registry().route(&manifest).expect("route");
    assert_eq!(route.engine_name, "alpha-vulkan");
}

#[test]
fn automatic_variant_prefers_the_cuda_device() {
    let mut manifest = manifest();
    manifest.backend = Some(BackendConfig {
        id: Some("alpha".into()),
        variant: Some("auto".into()),
    });
    let route = registry().route(&manifest).expect("route");
    assert_eq!(route.engine_name, "alpha-cuda");
}

#[test]
fn sidecars_can_be_registered_and_removed_at_runtime() {
    let registry = EngineRegistry::new(inventory());
    registry.register(std::sync::Arc::new(FakeEngine {
        name: "builtin-onnx",
        backend: "builtin-onnx",
        variant: "builtin",
        transport: false,
    }));
    let mut manifest = manifest();
    manifest.backend = Some(BackendConfig {
        id: Some("alpha".into()),
        variant: Some("Vulkan".into()),
    });
    assert!(registry.route(&manifest).is_err());

    registry.register_sidecar(std::sync::Arc::new(FakeEngine {
        name: "alpha-vulkan",
        backend: "alpha",
        variant: "Vulkan",
        transport: true,
    }));
    assert_eq!(
        registry.route(&manifest).expect("route").engine_name,
        "alpha-vulkan"
    );

    registry.unregister_sidecar("alpha", "Vulkan");
    assert!(registry.route(&manifest).is_err());
}

#[test]
fn automatic_variant_prefers_a_rocm_device() {
    let inventory = DeviceInventory::from(vec![
        device("cpu:0", DeviceClass::Cpu, "cpu"),
        device("gpu:0", DeviceClass::Gpu, "rocm"),
    ]);
    let registry = EngineRegistry::new(inventory);
    for (name, variant) in [
        ("alpha-cuda", "CUDA"),
        ("alpha-vulkan", "Vulkan"),
        ("alpha-rocm", "ROCm"),
    ] {
        registry.register_sidecar(std::sync::Arc::new(FakeEngine {
            name,
            backend: "alpha",
            variant,
            transport: true,
        }));
    }
    let mut manifest = manifest();
    manifest.backend = Some(BackendConfig {
        id: Some("alpha".into()),
        variant: Some("auto".into()),
    });
    assert_eq!(
        registry.route(&manifest).expect("route").engine_name,
        "alpha-rocm"
    );
}

#[test]
fn builtin_selection_rejects_transports() {
    let mut manifest = manifest();
    manifest.backend = Some(BackendConfig {
        id: Some("builtin".into()),
        variant: None,
    });
    let route = registry().route(&manifest).expect("route");
    assert_eq!(route.engine_name, "builtin-onnx");
}
