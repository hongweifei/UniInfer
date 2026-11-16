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
    assert_eq!(audio.family.as_deref(), Some("whisper"));
    assert!(audio.options.is_some());
}

fn gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn gguf_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
    gguf_string(bytes, key);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    gguf_string(bytes, value);
}

#[test]
fn reads_gguf_metadata() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&4u64.to_le_bytes());
    gguf_kv_string(&mut bytes, "general.architecture", "qwen3");
    gguf_string(&mut bytes, "general.file_type");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&15u32.to_le_bytes());
    gguf_string(&mut bytes, "general.context_length");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&40960u32.to_le_bytes());
    gguf_string(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&3u64.to_le_bytes());
    for token in ["a", "b", "c"] {
        gguf_string(&mut bytes, token);
    }
    fs::write(dir.path().join("model.gguf"), bytes).expect("gguf");
    fs::write(dir.path().join("tokenizer.json"), b"{}").expect("tokenizer");
    let inspection = inspect(dir.path(), &InspectOptions::default()).expect("inspect");
    assert_eq!(inspection.manifest.format, ModelFormat::Gguf);
    assert_eq!(inspection.manifest.quantization.as_str(), "q4_k_m");
    assert_eq!(inspection.manifest.context_len, 40960);
    let text = inspection.manifest.text.expect("text");
    assert_eq!(text.vocab_size, 3);
    assert_eq!(text.family.as_deref(), Some("qwen3"));
}
