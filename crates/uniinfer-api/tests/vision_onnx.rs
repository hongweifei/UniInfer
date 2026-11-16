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
async fn onnx_vision_chat_returns_text() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/smolvlm2-256m-onnx-q4/manifest.toml");
    if !common::fixture_ready(&manifest) {
        eprintln!("skip: SmolVLM ONNX fixture is not downloaded");
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
        "model": "smolvlm2-256m-onnx-q4",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "What color is this image?"}
            ]
        }],
        "max_tokens": 8,
        "temperature": 0
    })
    .to_string();
    let response = app
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
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    eprintln!("onnx vision: {content}");
    assert!(!content.trim().is_empty(), "{body}");
    assert!(
        body["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "{body}"
    );
}

#[tokio::test]
async fn onnx_vision_chat_accepts_two_images() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/smolvlm2-256m-onnx-q4/manifest.toml");
    if !common::fixture_ready(&manifest) {
        eprintln!("skip: SmolVLM ONNX fixture is not downloaded");
        return;
    }
    let state = AppState::new(&ApiConfig::new(manifest)).expect("state");
    let app = routes::router(state);
    let mut urls = Vec::new();
    for color in [[220, 40, 40], [40, 40, 220]] {
        let mut image = image::RgbImage::new(64, 64);
        for pixel in image.pixels_mut() {
            *pixel = image::Rgb(color);
        }
        let mut png = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("png");
        urls.push(format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&png)
        ));
    }
    let body = json!({
        "model": "smolvlm2-256m-onnx-q4",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": urls[0]}},
                {"type": "text", "text": " and "},
                {"type": "image_url", "image_url": {"url": urls[1]}},
                {"type": "text", "text": "Are these colors different?"}
            ]
        }],
        "max_tokens": 8,
        "temperature": 0
    })
    .to_string();
    let response = app
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
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    eprintln!("onnx vision two images: {content}");
    assert!(!content.trim().is_empty(), "{body}");
}
