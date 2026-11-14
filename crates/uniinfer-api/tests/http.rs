use std::path::Path;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};

fn app() -> Router {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml");
    let state = AppState::new(&ApiConfig::new(manifest)).expect("state");
    routes::router(state)
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("json")
}

fn post(uri: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .expect("request")
}

#[tokio::test]
async fn health_and_models_are_available() {
    let response = app()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let body = body_json(response).await;
    assert_eq!(body["data"][0]["id"], "tiny-byte-onnx");
}

#[tokio::test]
async fn chat_completion_returns_generated_bytes() {
    let body = json!({
        "model": "tiny-byte-onnx",
        "messages": [{"role": "user", "content": "ab"}],
        "max_tokens": 3
    })
    .to_string();
    let response = app()
        .oneshot(post("/v1/chat/completions", &body))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["choices"][0]["message"]["content"], "cde");
}

#[tokio::test]
async fn streaming_emits_deltas_and_done() {
    let body = json!({
        "model": "tiny-byte-onnx",
        "messages": [{"role": "user", "content": "ab"}],
        "max_tokens": 3,
        "stream": true
    })
    .to_string();
    let response = app()
        .oneshot(post("/v1/chat/completions", &body))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("\"content\":\"c\""));
    assert!(text.contains("\"finish_reason\":\"stop\""));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn malformed_json_returns_error_envelope() {
    let response = app()
        .oneshot(post("/v1/chat/completions", "{\"model\":"))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "invalid_json");
}

#[tokio::test]
async fn unknown_model_is_rejected() {
    let body = json!({
        "model": "missing",
        "messages": [{"role": "user", "content": "ab"}],
        "max_tokens": 3
    })
    .to_string();
    let response = app()
        .oneshot(post("/v1/chat/completions", &body))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_json(response).await;
    assert_eq!(body["code"], "model_not_found");
}

#[tokio::test]
async fn context_overflow_is_rejected() {
    let body = json!({
        "model": "tiny-byte-onnx",
        "messages": [{"role": "user", "content": "x".repeat(64)}],
        "max_tokens": 3
    })
    .to_string();
    let response = app()
        .oneshot(post("/v1/chat/completions", &body))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "context_overflow");
}
