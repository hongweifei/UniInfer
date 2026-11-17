#![cfg(feature = "mtmd")]

use std::path::Path;

use uniinfer_core::{
    CancelToken, DeviceBudget, DeviceClass, EngineLoadSpec, InferenceEngine, ModelManifest, Tensor,
    TensorData,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_llama::LlamaMtmdEngine;

#[test]
#[ignore = "loads the 5 GiB Qwen2.5-VL Q4 GGUF + mmproj"]
fn qwen2_5_vl_gguf_describes_image() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen2.5-vl-7b-gguf");
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
    let inventory = DeviceInventory::probe();
    let placement = inventory
        .plan(&manifest, &[DeviceClass::Cpu])
        .expect("placement");
    let session = LlamaMtmdEngine::new()
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
    let residency = session.residency().expect("residency");
    assert!(residency.disk_bytes > 0);

    let mut image = image::RgbImage::new(64, 64);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgb([220, 40, 40]);
    }
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("png");
    let prompt = "<|im_start|>user\n<__media__>\nDescribe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n";
    let text = Tensor::new(
        "text",
        vec![prompt.len()],
        TensorData::U8(prompt.as_bytes().to_vec()),
    )
    .expect("text");
    let encoded = uniinfer_core::ImageBatch::encode(&[png]).expect("encode");
    let image = Tensor::new("image", vec![encoded.len()], TensorData::U8(encoded)).expect("image");
    let outputs = session.run(vec![text, image]).expect("run");
    let TensorData::U8(text) = &outputs[0].data else {
        panic!("expected text");
    };
    assert!(!text.is_empty());
}

#[test]
#[ignore = "loads the 256 MiB SmolVLM GGUF fixture"]
fn smolvlm_gguf_describes_image() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smolvlm2-256m-q4");
    if !dir.join("manifest.toml").is_file() {
        eprintln!("skip: SmolVLM fixture is not downloaded");
        return;
    }
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
    let inventory = DeviceInventory::probe();
    let placement = inventory
        .plan(&manifest, &[DeviceClass::Cpu])
        .expect("placement");
    let session = LlamaMtmdEngine::new()
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
    let residency = session.residency().expect("residency");
    assert!(residency.disk_bytes > 0);

    let mut image = image::RgbImage::new(64, 64);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgb([220, 40, 40]);
    }
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("png");
    let prompt =
        "<|im_start|>User:<__media__>What color is this image?<end_of_utterance>\nAssistant:";
    let mut bytes = Vec::new();
    let decoder = session.native_decoder().expect("native decoder");
    let output = decoder
        .generate_text(
            uniinfer_core::NativeTextRequest {
                prompt,
                images: &[png],
                max_tokens: 16,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                deadline: None,
                cancel: &CancelToken::new(),
            },
            &mut |chunk| {
                bytes.extend_from_slice(chunk);
                true
            },
        )
        .expect("generate");
    let text = String::from_utf8_lossy(&bytes);
    eprintln!("smolvlm: {text}");
    assert!(!output.bytes.is_empty());
    assert!(!text.trim().is_empty());
}
