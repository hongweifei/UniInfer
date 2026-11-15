use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use uniinfer_core::{
    CancelToken, DeviceBudget, EngineError, EngineLoadSpec, InferenceEngine, ModelManifest, Tensor,
    TensorData,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_onnx::OnnxEngine;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx")
}

fn engine() -> &'static OnnxEngine {
    static ENGINE: LazyLock<OnnxEngine> = LazyLock::new(|| OnnxEngine::new().expect("engine"));
    &ENGINE
}

fn spec(manifest: ModelManifest, base_dir: &Path) -> EngineLoadSpec {
    let inventory = DeviceInventory::probe();
    let placement = inventory
        .plan(&manifest, &[uniinfer_core::DeviceClass::Cpu])
        .expect("placement");
    EngineLoadSpec {
        manifest,
        base_dir: base_dir.to_path_buf(),
        placement,
        budget: DeviceBudget::default(),
        cancel: CancelToken::new(),
    }
}

#[test]
fn capabilities_cover_onnx_text_on_cpu() {
    let engine = engine();
    assert_eq!(engine.engine_name(), "onnx");
    assert!(
        engine
            .supported_formats()
            .contains(&uniinfer_core::ModelFormat::Onnx)
    );
    assert!(
        engine
            .supported_modalities(uniinfer_core::ModelFormat::Onnx)
            .contains(&uniinfer_core::Modality::Text)
    );
    assert!(
        engine
            .supported_devices()
            .contains(&uniinfer_core::DeviceClass::Cpu)
    );
}

#[test]
fn loads_fixture_and_runs_one_step() {
    let dir = fixture_dir();
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
    let session = engine().load_session(spec(manifest, &dir)).expect("load");

    assert_eq!(session.inputs()[0].name, "input_ids");
    let input = Tensor::new("input_ids", vec![1, 2], TensorData::I64(vec![1, 2])).expect("input");
    let outputs = session.run(vec![input]).expect("run");
    let logits = &outputs[0];
    assert_eq!(logits.shape, vec![1, 2, 256]);

    let TensorData::F32(values) = &logits.data else {
        panic!("expected f32 logits");
    };
    let last = &values[256..];
    let best = last
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .expect("argmax");
    assert_eq!(best, 3);
    session.cancel();
}

#[test]
fn cancelled_session_refuses_to_run() {
    let dir = fixture_dir();
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
    let session = engine().load_session(spec(manifest, &dir)).expect("load");
    session.cancel();

    let input = Tensor::new("input_ids", vec![1, 1], TensorData::I64(vec![1])).expect("input");
    let error = session.run(vec![input]).expect_err("must be cancelled");
    assert!(matches!(error, EngineError::Cancelled));
}

#[test]
fn unsupported_quantization_is_rejected_at_load() {
    let dir = fixture_dir();
    let mut manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
    manifest.quantization = uniinfer_core::Quantization::Fp8;

    let error = engine()
        .load_session(spec(manifest, &dir))
        .err()
        .expect("must reject");
    assert!(matches!(error, EngineError::UnsupportedQuantization(_)));
}

#[test]
fn checksum_mismatch_is_rejected() {
    let dir = fixture_dir();
    let temp = tempfile::tempdir().expect("tempdir");
    std::fs::copy(dir.join("manifest.toml"), temp.path().join("manifest.toml")).expect("manifest");
    std::fs::copy(dir.join("model.onnx"), temp.path().join("model.onnx")).expect("model");

    let mut bytes = std::fs::read(temp.path().join("model.onnx")).expect("read");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(temp.path().join("model.onnx"), bytes).expect("write");

    let manifest = ModelManifest::load(&temp.path().join("manifest.toml")).expect("manifest");
    let error = engine()
        .load_session(spec(manifest, temp.path()))
        .err()
        .expect("must reject");
    assert!(matches!(error, EngineError::ManifestInvalid(_)));
}
