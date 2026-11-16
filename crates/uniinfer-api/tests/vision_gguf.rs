#![cfg(feature = "llama-mtmd")]

use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};

mod common;

#[tokio::test]
async fn gguf_vision_chat_returns_text() {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smolvlm2-256m-q4/manifest.toml");
    if !common::fixture_ready(&manifest) {
        eprintln!("skip: SmolVLM fixture is not downloaded");
        return;
    }
    let state = AppState::new(&ApiConfig::new(manifest)).expect("state");
    let app = routes::router(state);

    let mut image = image::RgbImage::new(64, 64);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgb([220, 40, 40]);
    }
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("png");
    let url = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&png)
    );
    let body = json!({
        "model": "smolvlm2-256m-q4",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url.clone()}},
                {"type": "text", "text": "What color is this image?"}
            ]
        }],
        "max_tokens": 16,
        "temperature": 0
    })
    .to_string();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    eprintln!("gguf vision: {content}");
    assert!(!content.trim().is_empty(), "{body}");
    assert!(
        body["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "{body}"
    );

    let body = json!({
        "model": "smolvlm2-256m-q4",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url.clone()}},
                {"type": "text", "text": "What color is this image?"}
            ]
        }],
        "max_tokens": 8,
        "temperature": 0,
        "stream": true
    })
    .to_string();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let stream = String::from_utf8_lossy(&bytes);
    assert!(stream.contains("data: {"), "{stream}");
    assert!(stream.contains("[DONE]"), "{stream}");
}
