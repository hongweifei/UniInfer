use std::fs;

use uniinfer_core::{Modality, ModelFormat};
use uniinfer_model_inspect::{InspectOptions, inspect};

#[test]
fn detects_single_onnx_text_model() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::write(dir.path().join("model.onnx"), b"onnx").expect("model");
    fs::write(
        dir.path().join("config.json"),
        r#"{"architectures":["Qwen3ForCausalLM"],"vocab_size":100,"eos_token_id":2,"max_position_embeddings":128}"#,
    )
    .expect("config");
    fs::write(dir.path().join("tokenizer.json"), b"{}").expect("tokenizer");
    let inspection = inspect(dir.path(), &InspectOptions::default()).expect("inspect");
    assert_eq!(inspection.manifest.format, ModelFormat::Onnx);
    assert_eq!(inspection.manifest.modalities, vec![Modality::Text]);
    assert_eq!(
        inspection.manifest.text.as_ref().expect("text").vocab_size,
        100
    );
    assert!(inspection.manifest.audio.is_none());
}

#[test]
fn detects_onnx_encoder_decoder_asr() {
    let dir = tempfile::tempdir().expect("tempdir");
    fs::create_dir(dir.path().join("onnx")).expect("onnx dir");
    fs::write(dir.path().join("onnx/encoder_model.onnx"), b"encoder").expect("encoder");
    fs::write(dir.path().join("onnx/decoder_model.onnx"), b"decoder").expect("decoder");
    fs::write(dir.path().join("tokenizer.json"), b"{}").expect("tokenizer");
    fs::write(
        dir.path().join("config.json"),
        r#"{"architectures":["WhisperForConditionalGeneration"],"max_source_positions":1500,"max_length":448}"#,
    )
    .expect("config");
    let inspection = inspect(dir.path(), &InspectOptions::default()).expect("inspect");
    assert_eq!(inspection.manifest.modalities, vec![Modality::Audio]);
    assert_eq!(inspection.manifest.context_len, 448);
    let audio = inspection.manifest.audio.expect("audio");
    assert_eq!(audio.architecture.as_deref(), Some("whisper"));
    assert!(audio.options.is_some());
}
