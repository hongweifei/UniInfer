use uniinfer_core::{DeviceClass, InferenceEngine, Modality, ModelFormat};
use uniinfer_engine_whisper_server::WhisperServerEngine;

#[test]
fn advertises_whisper_ggml_audio() {
    let engine = WhisperServerEngine::new("whisper-server.exe".into(), vec![DeviceClass::Cpu]);
    assert_eq!(engine.engine_name(), "whisper-server");
    assert_eq!(engine.implementations(), &["whisper.ggml"]);
    assert_eq!(engine.supported_formats(), &[ModelFormat::Ggml]);
    assert_eq!(
        engine.supported_modalities(ModelFormat::Ggml),
        &[Modality::Audio]
    );
    assert_eq!(engine.supported_devices(), &[DeviceClass::Cpu]);
    assert!(engine.is_transport());
}

#[test]
#[ignore = "requires WHISPER_SERVER_BIN and WHISPER_MODEL_DIR"]
fn transcribes_with_official_whisper_server() {
    use std::path::PathBuf;
    use std::time::Duration;

    use uniinfer_core::{
        CancelToken, DeviceBudget, EngineLoadSpec, ModelManifest, PlacementPlan, Tensor, TensorData,
    };

    let executable =
        PathBuf::from(std::env::var("WHISPER_SERVER_BIN").expect("WHISPER_SERVER_BIN"));
    let directory = PathBuf::from(std::env::var("WHISPER_MODEL_DIR").expect("WHISPER_MODEL_DIR"));
    let manifest_path = directory.join("manifest.toml");
    let manifest = ModelManifest::load(&manifest_path).expect("manifest");
    let session = WhisperServerEngine::new(executable, vec![DeviceClass::Cpu])
        .load_session(EngineLoadSpec {
            manifest,
            base_dir: directory,
            manifest_path: Some(manifest_path),
            placement: PlacementPlan::default(),
            budget: DeviceBudget::default(),
            cancel: CancelToken::new(),
        })
        .expect("load whisper-server");
    let audio =
        Tensor::new("audio", vec![16_000], TensorData::F32(vec![0.0; 16_000])).expect("audio");
    let output = session
        .run_with_deadline(
            vec![audio],
            Some(std::time::Instant::now() + Duration::from_secs(60)),
        )
        .expect("transcribe");
    assert_eq!(output[0].name, "text");
}
