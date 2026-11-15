use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};

#[tokio::test]
async fn chat_completion_accepts_data_url_image() {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest-vision.toml");
    let state = AppState::new(&ApiConfig::new(manifest)).expect("state");
    let app = routes::router(state);

    let url = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(b"not-a-real-png")
    );
    let body = json!({
        "model": "tiny-vision-onnx",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "ab"}
            ]
        }],
        "max_tokens": 3,
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
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "cde", "{body}");
    assert_eq!(body["usage"]["prompt_tokens"], 9, "{body}");
    assert_eq!(body["usage"]["completion_tokens"], 3, "{body}");
}
