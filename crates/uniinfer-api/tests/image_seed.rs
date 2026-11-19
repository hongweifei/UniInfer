use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};

async fn generate(payload: Value) -> String {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-diffusion/manifest.toml");
    let app = routes::router(AppState::new(&ApiConfig::new(manifest)).expect("state"));
    let body = payload.to_string();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
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
    String::from_utf8(bytes.to_vec()).expect("utf8")
}

#[tokio::test]
async fn implicit_seed_changes_between_requests() {
    let first: Value = serde_json::from_str(
        &generate(json!({"model": "tiny-diffusion-onnx", "prompt": "red cube", "size": "64x64", "steps": 1}))
            .await,
    )
    .expect("json");
    let second: Value = serde_json::from_str(
        &generate(json!({"model": "tiny-diffusion-onnx", "prompt": "red cube", "size": "64x64", "steps": 1}))
            .await,
    )
    .expect("json");
    let first_seed = first["data"][0]["seed"].as_u64().expect("seed");
    let second_seed = second["data"][0]["seed"].as_u64().expect("seed");
    assert_ne!(first_seed, second_seed);
}

#[tokio::test]
async fn explicit_seed_is_echoed() {
    let body: Value = serde_json::from_str(
        &generate(json!({
            "model": "tiny-diffusion-onnx",
            "prompt": "red cube",
            "size": "64x64",
            "steps": 1,
            "seed": 42
        }))
        .await,
    )
    .expect("json");
    assert_eq!(body["data"][0]["seed"], 42);
}

#[tokio::test]
async fn stream_done_carries_seed() {
    let text = generate(json!({
        "model": "tiny-diffusion-onnx",
        "prompt": "red cube",
        "size": "64x64",
        "steps": 1,
        "stream": true
    }))
    .await;
    assert!(text.contains("event: done"), "{text}");
    assert!(text.contains("\"seed\":"), "{text}");
}
