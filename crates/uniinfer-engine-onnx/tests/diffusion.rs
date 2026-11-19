use std::path::Path;

use uniinfer_core::{
    CancelToken, DeviceBudget, DeviceClass, EngineLoadSpec, EngineSession, ImageGenerationRequest,
    InferenceEngine, ModelManifest, TensorData,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_onnx::OnnxEngine;

fn load_session() -> Box<dyn EngineSession> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-diffusion");
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
    let placement = DeviceInventory::probe()
        .plan(&manifest, &[DeviceClass::Cpu])
        .expect("placement");
    OnnxEngine::new()
        .expect("engine")
        .load_session(EngineLoadSpec {
            manifest,
            base_dir: dir,
            manifest_path: None,
            placement,
            budget: DeviceBudget::default(),
            cancel: CancelToken::new(),
        })
        .expect("load")
}

fn generate(session: &dyn EngineSession, seed: Option<u64>) -> Vec<f32> {
    let prompt = vec![1i64; 77];
    let image = session
        .generate_image(&mut ImageGenerationRequest {
            prompt_ids: &prompt,
            negative_ids: &[],
            width: 64,
            height: 64,
            steps: 1,
            guidance: 1.0,
            seed,
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
    values
}

#[test]
fn tiny_diffusion_generates_an_image() {
    let session = load_session();
    generate(&*session, Some(7));
}

#[test]
fn tiny_diffusion_randomizes_missing_seed() {
    let session = load_session();
    let first = generate(&*session, None);
    let second = generate(&*session, None);
    assert_ne!(first, second, "missing seed must randomize latent noise");
}

#[test]
fn tiny_diffusion_reuses_explicit_seed() {
    let session = load_session();
    assert_eq!(generate(&*session, Some(7)), generate(&*session, Some(7)));
}
