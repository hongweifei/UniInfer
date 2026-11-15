use uniinfer_core::{DeviceClass, InferenceEngine, Modality, ModelFormat, Quantization};
use uniinfer_engine_whisper::WhisperEngine;

#[test]
fn advertises_only_whisper_audio_cpu() {
    let engine = WhisperEngine::new();
    assert_eq!(engine.engine_name(), "whisper");
    assert_eq!(engine.supported_formats(), &[ModelFormat::Whisper]);
    assert_eq!(
        engine.supported_modalities(ModelFormat::Whisper),
        &[Modality::Audio]
    );
    assert!(
        engine
            .supported_quantizations(ModelFormat::Whisper)
            .contains(&Quantization::Fp16)
    );
    assert_eq!(engine.supported_devices(), &[DeviceClass::Cpu]);
    assert!(engine.supported_modalities(ModelFormat::Onnx).is_empty());
}
