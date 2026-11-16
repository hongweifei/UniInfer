use std::path::{Path, PathBuf};

use uniinfer_core::{
    Capability, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, ModelManifest, PlacementConfig, PlacementPolicy, Quantization, Result, Runtime,
    Tensor, TensorData, TensorSpec,
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
        capabilities: vec![Capability::TextGeneration],
        format: ModelFormat::Onnx,
        quantization: Quantization::Fp32,
        modalities: vec![Modality::Text],
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
    manifest.modalities = vec![Modality::Audio];
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
