use std::path::{Path, PathBuf};

use uniinfer_core::{
    Capability, Device, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine,
    Modality, Mode, ModelFormat, ModelManifest, Operation, PlacementConfig, PlacementPolicy,
    Quantization, Residency, Result, Runtime, Tensor, TensorData, TensorSpec,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_registry::EngineRegistry;

struct FakeEngine;

impl InferenceEngine for FakeEngine {
    fn engine_name(&self) -> &'static str {
        "fake"
    }

    fn runtime(&self) -> Runtime {
        Runtime::Onnx
    }

    fn implementations(&self) -> &[&'static str] {
        &["onnx.fake"]
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
        &[DeviceClass::Cpu]
    }

    fn load_session(&self, _spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        Ok(Box::new(Noop))
    }
}

struct GpuOffloadEngine;

impl InferenceEngine for GpuOffloadEngine {
    fn engine_name(&self) -> &'static str {
        "fake-offload"
    }

    fn runtime(&self) -> Runtime {
        Runtime::Onnx
    }

    fn implementations(&self) -> &[&'static str] {
        &["onnx.offload"]
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Onnx]
    }

    fn supported_modalities(&self, _format: ModelFormat) -> &[Modality] {
        &[Modality::Text]
    }

    fn supported_quantizations(&self, _format: ModelFormat) -> &[Quantization] {
        &[Quantization::Fp32]
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        &[DeviceClass::Cpu, DeviceClass::Gpu]
    }

    fn supports_layer_offload(&self) -> bool {
        true
    }

    fn supports_kv_cache_offload(&self) -> bool {
        true
    }

    fn load_session(&self, _spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        Ok(Box::new(Resident))
    }
}

struct Resident;

impl EngineSession for Resident {
    fn inputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn outputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn residency(&self) -> Option<Residency> {
        Some(Residency {
            vram_bytes: 600,
            host_bytes: 10,
            disk_bytes: 5,
            offload_layers: 2,
            total_layers: Some(4),
            estimated: false,
        })
    }

    fn run(&self, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        Ok(vec![Tensor::new(
            "out",
            vec![1],
            TensorData::F32(vec![0.0]),
        )?])
    }

    fn cancel(&self) {}
}

struct Noop;

impl EngineSession for Noop {
    fn inputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn outputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn run(&self, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        Ok(vec![Tensor::new(
            "out",
            vec![1],
            TensorData::F32(vec![0.0]),
        )?])
    }

    fn cancel(&self) {}
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
            prefer: Vec::new(),
            stages: std::collections::BTreeMap::new(),
            offload_layers: None,
            kv_cache: None,
        },
        files: vec![uniinfer_core::ModelFile {
            path: PathBuf::from("weights.bin"),
            sha256: "0".repeat(64),
        }],
        text: None,
        audio: None,
        vision: None,
        pipeline: None,
        image: None,
    }
}

fn write_manifest(dir: &Path, manifest: &ModelManifest) -> PathBuf {
    std::fs::write(dir.join("weights.bin"), b"weights").expect("weights");
    let digest = sha256(dir.join("weights.bin"));
    let mut manifest = manifest.clone();
    manifest.files[0].sha256 = digest;
    let path = dir.join("manifest.toml");
    std::fs::write(&path, toml::to_string(&manifest).expect("toml")).expect("manifest");
    path
}

fn sha256(path: PathBuf) -> String {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).expect("read");
    hex::encode(Sha256::digest(bytes))
}

fn registry() -> EngineRegistry {
    let mut registry = EngineRegistry::new(DeviceInventory::probe());
    registry.register(std::sync::Arc::new(FakeEngine));
    registry
}

#[test]
fn loads_matching_manifest() {
    let temp = tempfile::tempdir().expect("tempdir");
    let path = write_manifest(temp.path(), &manifest());
    let lease = registry().load(&path).expect("load");
    assert_eq!(lease.engine_name(), "fake");
    drop(lease);
}

#[test]
fn rejects_unsupported_quantization() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut manifest = manifest();
    manifest.quantization = Quantization::Fp8;
    let path = write_manifest(temp.path(), &manifest);
    let error = registry().load(&path).err().expect("must reject");
    assert!(matches!(error, EngineError::UnsupportedQuantization(_)));
}

#[test]
fn rejects_unsupported_implementation() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut manifest = manifest();
    manifest.implementation = "onnx.unsupported".into();
    let path = write_manifest(temp.path(), &manifest);
    let error = registry().load(&path).err().expect("must reject");
    assert!(matches!(error, EngineError::UnsupportedEngine(_)));
}

#[test]
fn rejects_unsupported_modality() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut manifest = manifest();
    manifest.capabilities[0].required_inputs = vec![Modality::Audio];
    let path = write_manifest(temp.path(), &manifest);
    let error = registry().load(&path).err().expect("must reject");
    assert!(matches!(error, EngineError::UnsupportedModality { .. }));
}

#[test]
fn rejects_unsupported_format() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut manifest = manifest();
    manifest.format = ModelFormat::Gguf;
    let path = write_manifest(temp.path(), &manifest);
    let error = registry().load(&path).err().expect("must reject");
    assert!(matches!(error, EngineError::UnsupportedFormat(_)));
}

#[test]
fn rejects_layer_offload_on_engine_without_support() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut manifest = manifest();
    manifest.placement.offload_layers = Some("4".into());
    let path = write_manifest(temp.path(), &manifest);
    let error = registry().load(&path).err().expect("must reject");
    assert!(matches!(error, EngineError::PlacementUnsatisfiable(_)));
}

#[test]
fn rejects_pinned_device_unavailable_to_engine() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut manifest = manifest();
    manifest.placement.policy = PlacementPolicy::Pinned;
    manifest
        .placement
        .stages
        .insert("llm".into(), DeviceClass::Gpu);
    let path = write_manifest(temp.path(), &manifest);
    let error = registry().load(&path).err().expect("must reject");
    assert!(matches!(error, EngineError::PlacementUnsatisfiable(_)));
}

#[test]
fn defaults_to_auto_offload_for_capable_gpu_engine() {
    let inventory = DeviceInventory::from(vec![
        Device {
            id: "cpu:0".into(),
            class: DeviceClass::Cpu,
            backend: "cpu".into(),
            memory_bytes: 16 * 1024 * 1024 * 1024,
            available_memory_bytes: 8 * 1024 * 1024 * 1024,
        },
        Device {
            id: "gpu:0".into(),
            class: DeviceClass::Gpu,
            backend: "fake".into(),
            memory_bytes: 8 * 1024 * 1024 * 1024,
            available_memory_bytes: 6 * 1024 * 1024 * 1024,
        },
    ]);
    let mut registry = EngineRegistry::new(inventory);
    registry.register(std::sync::Arc::new(GpuOffloadEngine));
    let mut manifest = manifest();
    manifest.implementation = "onnx.offload".into();
    let route = registry.route(&manifest).expect("route");
    assert!(route.placement.auto_offload);
    assert_eq!(route.placement.devices[0].class, DeviceClass::Gpu);
}

#[test]
fn rejects_min_vram_when_full_gpu_load_does_not_fit() {
    let inventory = DeviceInventory::from(vec![
        Device {
            id: "cpu:0".into(),
            class: DeviceClass::Cpu,
            backend: "cpu".into(),
            memory_bytes: 16 * 1024 * 1024 * 1024,
            available_memory_bytes: 8 * 1024 * 1024 * 1024,
        },
        Device {
            id: "gpu:0".into(),
            class: DeviceClass::Gpu,
            backend: "fake".into(),
            memory_bytes: 8 * 1024 * 1024 * 1024,
            available_memory_bytes: 4 * 1024 * 1024 * 1024,
        },
    ]);
    let mut registry = EngineRegistry::new(inventory);
    registry.register(std::sync::Arc::new(GpuOffloadEngine));
    let mut manifest = manifest();
    manifest.implementation = "onnx.offload".into();
    manifest.placement.offload_layers = Some("0".into());
    manifest.min_vram = Some("6GiB".into());
    let Err(error) = registry.route(&manifest) else {
        panic!("must reject");
    };
    assert!(matches!(error, EngineError::PlacementUnsatisfiable(_)));
}

fn gpu_budget() -> DeviceInventory {
    DeviceInventory::from(vec![
        Device {
            id: "cpu:0".into(),
            class: DeviceClass::Cpu,
            backend: "cpu".into(),
            memory_bytes: 16 * 1024,
            available_memory_bytes: 16 * 1024,
        },
        Device {
            id: "gpu:0".into(),
            class: DeviceClass::Gpu,
            backend: "fake".into(),
            memory_bytes: 1_000,
            available_memory_bytes: 1_000,
        },
    ])
}

#[test]
fn ledger_reserves_and_releases_session_residency() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut manifest = manifest();
    manifest.implementation = "onnx.offload".into();
    let path = write_manifest(temp.path(), &manifest);
    let mut registry = EngineRegistry::new(gpu_budget());
    registry.register(std::sync::Arc::new(GpuOffloadEngine));

    let lease = registry.load(&path).expect("first load");
    let summary = registry.residency();
    assert_eq!(summary.sessions, 1);
    assert_eq!(summary.vram_bytes, 600);
    assert_eq!(summary.host_bytes, 10);
    assert_eq!(summary.disk_bytes, 5);

    let error = match registry.load(&path) {
        Ok(_) => panic!("second load must exceed budget"),
        Err(error) => error,
    };
    assert!(matches!(error, EngineError::InsufficientVram { .. }));

    drop(lease);
    let summary = registry.residency();
    assert_eq!(summary.sessions, 0);
    assert_eq!(summary.vram_bytes, 0);

    let lease = registry.load(&path).expect("reload after release");
    drop(lease);
}
