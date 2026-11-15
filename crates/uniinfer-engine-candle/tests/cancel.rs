//! Real-weight cancellation smoke; opt in with `cargo test -p uniinfer-engine-candle -- --ignored`.

use std::path::Path;

use uniinfer_core::{
    DeviceBudget, DeviceClass, EngineError, EngineLoadSpec, InferenceEngine, ModelManifest, Tensor,
    TensorData,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_candle::CandleEngine;

#[test]
#[ignore = "loads the 379 MiB Qwen3 GGUF fixture"]
fn cancelled_session_refuses_to_run() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3-0.6b");
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
    let inventory = DeviceInventory::probe();
    let placement = inventory
        .plan(&manifest, &[DeviceClass::Cpu])
        .expect("placement");
    let session = CandleEngine::new()
        .expect("engine")
        .load_session(EngineLoadSpec {
            manifest,
            base_dir: dir,
            placement,
            budget: DeviceBudget::default(),
        })
        .expect("load");

    session.cancel();
    let input = Tensor::new("input_ids", vec![1, 1], TensorData::I64(vec![1])).expect("input");
    let error = session.run(vec![input]).expect_err("must be cancelled");
    assert!(matches!(error, EngineError::Cancelled));
}
