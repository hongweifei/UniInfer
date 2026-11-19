use std::path::Path;

use uniinfer_core::{
    CancelToken, DeviceBudget, DeviceClass, EngineLoadSpec, ImageGenerationRequest,
    InferenceEngine, ModelManifest, TensorData,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_onnx::OnnxEngine;

#[test]
fn tiny_diffusion_generates_an_image() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-diffusion");
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
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
        .expect("load");

    let prompt = vec![1i64; 77];
    let image = session
        .generate_image(&mut ImageGenerationRequest {
            prompt_ids: &prompt,
            negative_ids: &[],
            width: 64,
            height: 64,
            steps: 1,
            guidance: 1.0,
            seed: Some(7),
            cancel: &CancelToken::new(),
            preview: false,
            progress: &mut |_, _, _| {},
        })
        .expect("generate");
    assert_eq!(image.shape, vec![1, 3, 64, 64]);
    let TensorData::F32(values) = image.data else {
        panic!("expected f32 image");
    };
    assert_eq!(values.len(), 3 * 64 * 64);
    assert!(values.iter().all(|value| value.is_finite()));
}
