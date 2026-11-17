use std::path::Path;

use uniinfer_core::{
    CancelToken, DeviceBudget, DeviceClass, EngineLoadSpec, InferenceEngine, ModelManifest, Tensor,
    TensorData,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_onnx::OnnxEngine;

#[test]
#[ignore = "loads the 500 MiB Qwen3 ONNX fixture and the layered CPU path"]
fn qwen3_layered_cpu_generates_logits() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3-0.6b-onnx");
    let mut manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
    manifest.implementation = "onnx.layered".into();
    let placement = DeviceInventory::probe()
        .plan(&manifest, &[DeviceClass::Cpu])
        .expect("placement");
    let session = OnnxEngine::new()
        .expect("engine")
        .load_session(EngineLoadSpec {
            manifest,
            base_dir: dir,
            manifest_path: None,
            placement,
            budget: DeviceBudget::default(),
            cancel: CancelToken::new(),
        })
        .expect("load layered");
    let ids = Tensor::new("input_ids", vec![1, 3], TensorData::I64(vec![1, 2, 3])).expect("ids");
    let mask =
        Tensor::new("attention_mask", vec![1, 3], TensorData::I64(vec![1, 1, 1])).expect("mask");
    let output = session.run(vec![ids, mask]).expect("run");
    assert!(output.iter().any(|tensor| tensor.name == "logits"));

    let ids = Tensor::new("input_ids", vec![1, 4], TensorData::I64(vec![1, 2, 3, 4])).expect("ids");
    let mask = Tensor::new(
        "attention_mask",
        vec![1, 4],
        TensorData::I64(vec![1, 1, 1, 1]),
    )
    .expect("mask");
    let output = session.run(vec![ids, mask]).expect("incremental run");
    assert!(output.iter().any(|tensor| tensor.name == "logits"));
}
