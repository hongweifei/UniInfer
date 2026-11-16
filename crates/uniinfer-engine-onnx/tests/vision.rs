use std::path::Path;

use uniinfer_core::{
    CancelToken, DeviceBudget, DeviceClass, EngineLoadSpec, InferenceEngine, ModelManifest, Tensor,
    TensorData,
};
use uniinfer_devices::DeviceInventory;
use uniinfer_engine_onnx::OnnxEngine;

#[test]
fn smolvlm_onnx_returns_logits_for_image() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smolvlm2-256m-onnx-q4");
    if !dir.join("manifest.toml").is_file() {
        eprintln!("skip: SmolVLM ONNX fixture is not downloaded");
        return;
    }
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
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

    let mut image = image::RgbImage::new(64, 64);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgb([220, 40, 40]);
    }
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("png");
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let prompt = "<|im_start|>User:<image>What color is this image?<end_of_utterance>\nAssistant:";
    let ids = tokenizer
        .encode(prompt, true)
        .expect("encode")
        .get_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>();
    assert!(ids.contains(&49190), "prompt must contain the image token");
    let outputs = session
        .run(vec![
            Tensor::new("input_ids", vec![1, ids.len()], TensorData::I64(ids)).expect("ids"),
            {
                let encoded = uniinfer_core::ImageBatch::encode(&[png]).expect("encode");
                Tensor::new("image", vec![encoded.len()], TensorData::U8(encoded)).expect("image")
            },
        ])
        .expect("run");
    let logits = &outputs[0];
    assert_eq!(logits.name, "logits");
    assert_eq!(logits.shape[0], 1);
    assert_eq!(logits.shape[2], 49_280);
    let TensorData::F32(values) = &logits.data else {
        panic!("expected f32 logits");
    };
    assert!(values.iter().any(|value| value.is_finite()));
}

#[test]
fn smolvlm_onnx_returns_logits_for_two_images() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smolvlm2-256m-onnx-q4");
    if !dir.join("manifest.toml").is_file() {
        eprintln!("skip: SmolVLM ONNX fixture is not downloaded");
        return;
    }
    let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
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

    let mut images = Vec::new();
    for color in [[220, 40, 40], [40, 220, 40]] {
        let mut image = image::RgbImage::new(64, 64);
        for pixel in image.pixels_mut() {
            *pixel = image::Rgb(color);
        }
        let mut png = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("png");
        images.push(png);
    }
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let prompt =
        "<|im_start|>User:<image><image>Are these colors different?<end_of_utterance>\nAssistant:";
    let ids = tokenizer
        .encode(prompt, true)
        .expect("encode")
        .get_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>();
    assert_eq!(
        ids.iter().filter(|id| **id == 49190).count(),
        2,
        "prompt must contain two image tokens"
    );
    let outputs = session
        .run(vec![
            Tensor::new("input_ids", vec![1, ids.len()], TensorData::I64(ids)).expect("ids"),
            {
                let encoded = uniinfer_core::ImageBatch::encode(&images).expect("encode");
                Tensor::new("image", vec![encoded.len()], TensorData::U8(encoded)).expect("images")
            },
        ])
        .expect("run");
    let logits = &outputs[0];
    assert_eq!(logits.name, "logits");
    assert_eq!(logits.shape[0], 1);
    assert_eq!(logits.shape[2], 49_280);
    let TensorData::F32(values) = &logits.data else {
        panic!("expected f32 logits");
    };
    assert!(values.iter().any(|value| value.is_finite()));
}
