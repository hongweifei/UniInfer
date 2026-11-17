use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};

#[tokio::test]
async fn diffusion_image_endpoint_returns_png() {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-diffusion/manifest.toml");
    let state = AppState::new(&ApiConfig::new(manifest)).expect("state");
    let app = routes::router(state);
    let body = json!({
        "model": "tiny-diffusion-onnx",
        "prompt": "a red cube",
        "size": "64x64",
        "steps": 1
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
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let encoded = value["data"][0]["b64_json"].as_str().expect("b64_json");
    use base64::Engine as _;
    let png = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .expect("base64");
    assert!(png.starts_with(b"\x89PNG"));
}
