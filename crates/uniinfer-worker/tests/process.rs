use std::path::{Path, PathBuf};

use uniinfer_core::{
    CancelToken, DeviceBudget, DeviceClass, EngineLoadSpec, InferenceEngine, ModelManifest, Tensor,
    TensorData,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_process::ProcessEngine;

#[test]
fn isolated_worker_runs_tiny_onnx() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx");
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
    let inventory = DeviceInventory::probe();
    let placement = inventory
        .plan(&manifest, &[DeviceClass::Cpu])
        .expect("placement");
    let worker = PathBuf::from(env!("CARGO_BIN_EXE_uniinfer-worker"));
    let session = ProcessEngine::with_worker(worker)
        .load_session(EngineLoadSpec {
            manifest,
            base_dir: dir,
            manifest_path: Some(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml"),
            ),
            placement,
            budget: DeviceBudget::default(),
            cancel: CancelToken::new(),
        })
        .expect("load worker");

    let input = Tensor::new("input_ids", vec![1, 2], TensorData::I64(vec![1, 2])).expect("input");
    let outputs = session.run(vec![input]).expect("run");
    assert_eq!(outputs[0].name, "logits");
    assert_eq!(outputs[0].shape, vec![1, 2, 256]);
}
