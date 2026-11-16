use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};

#[tokio::test]
async fn tiny_vision_accepts_two_images() {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest-vision.toml");
    let state = AppState::new(&ApiConfig::new(manifest)).expect("state");
    let app = routes::router(state);
    let body = json!({
        "model": "tiny-vision-onnx",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                {"type": "text", "text": " and "},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,enk="}},
                {"type": "text", "text": "compare"}
            ]
        }],
        "max_tokens": 4,
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
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
}
