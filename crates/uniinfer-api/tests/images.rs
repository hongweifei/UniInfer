use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};

#[tokio::test]
async fn images_generations_returns_png() {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest-image.toml");
    let state = AppState::new(&ApiConfig::new(manifest)).expect("state");
    let app = routes::router(state);
    let body = json!({
        "model": "auto",
        "prompt": "ab",
        "n": 1,
        "response_format": "b64_json",
        "size": "64x64"
    })
    .to_string();
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
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert!(body["data"][0]["seed"].as_u64().is_some(), "seed");
    let encoded = body["data"][0]["b64_json"].as_str().expect("b64_json");
    let png = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .expect("base64");
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(body["data"][0]["width"], 64);
    assert_eq!(body["data"][0]["height"], 64);
}
