use std::path::Path;

use uniinfer_core::{
    CancelToken, DeviceBudget, DeviceClass, EngineLoadSpec, InferenceEngine, ModelManifest, Tensor,
    TensorData,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_onnx::OnnxEngine;

mod common;

#[test]
fn whisper_manifest_loads_through_pipeline_adapter() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/whisper-tiny.en-onnx");
    let manifest_path = dir.join("manifest.toml");
    if !common::fixture_ready(&manifest_path) {
        eprintln!("skip: Whisper ONNX fixture is not downloaded");
        return;
    }
    let manifest = ModelManifest::load(&manifest_path).expect("manifest");
    let inventory = DeviceInventory::probe();
    let placement = inventory
        .plan(&manifest, &[DeviceClass::Cpu])
        .expect("placement");
    let session = OnnxEngine::new()
        .expect("engine")
        .load_session(EngineLoadSpec {
            manifest,
            base_dir: dir.clone(),
            manifest_path: Some(dir.join("manifest.toml")),
            placement,
            budget: DeviceBudget::default(),
            cancel: CancelToken::new(),
        })
        .expect("load");
    let audio =
        Tensor::new("audio", vec![16_000], TensorData::F32(vec![0.0; 16_000])).expect("audio");
    let output = session.run(vec![audio]).expect("transcribe");
    assert_eq!(output[0].name, "text");
}
