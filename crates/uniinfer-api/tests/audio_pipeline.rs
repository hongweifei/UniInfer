use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};

#[tokio::test]
async fn audio_endpoint_runs_through_model_executor() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/whisper-tiny.en-onnx/manifest.toml");
    if !manifest.is_file() {
        eprintln!("skip: Whisper ONNX fixture is not downloaded");
        return;
    }
    let state = AppState::new(&ApiConfig::new(manifest)).expect("state");
    let app = routes::router(state);

    let mut wav = Vec::new();
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer =
            hound::WavWriter::new(std::io::Cursor::new(&mut wav), spec).expect("wav writer");
        for _ in 0..16_000 {
            writer.write_sample(0i16).expect("sample");
        }
        writer.finalize().expect("wav");
    }
    let boundary = "uniinfer-audio-boundary";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-tiny.en-onnx\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(&wav);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let request = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert!(body["text"].is_string(), "{body}");
}
