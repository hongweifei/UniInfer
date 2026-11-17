use std::path::Path;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};
use uniinfer_backend_store::{BackendEngine, BackendManifest, BackendStore};

fn app(config: &ApiConfig) -> Router {
    let state = AppState::new(config).expect("state");
    routes::router(state)
}

fn install_fake(store: &BackendStore, id: &str, version: &str) {
    let dir = store.package_dir(id, version);
    std::fs::create_dir_all(dir.join("bin")).expect("dir");
    std::fs::write(dir.join("bin/worker.exe"), b"worker").expect("worker");
    let manifest = BackendManifest {
        id: id.into(),
        name: "Test Vulkan".into(),
        version: version.into(),
        description: "fake sidecar".into(),
        homepage: None,
        license: None,
        platform: "test".into(),
        worker: "bin/worker.exe".into(),
        devices: vec![uniinfer_core::DeviceClass::Gpu],
        engines: vec![BackendEngine {
            id: "llama-vulkan".into(),
            name: "llama.cpp Vulkan".into(),
            runtime: "llama".into(),
            description: "Vulkan".into(),
            vendor: "llama.cpp".into(),
            variant: "Vulkan".into(),
            category: "language".into(),
            homepage: None,
            implementations: vec!["llama.text".into()],
            formats: vec!["gguf".into()],
            modalities: vec!["text".into()],
            devices: vec![uniinfer_core::DeviceClass::Gpu],
        }],
        files: Vec::new(),
    };
    store.write_manifest(&dir, &manifest).expect("manifest");
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

#[tokio::test]
async fn lists_and_activates_installed_backends() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = BackendStore::new(temp.path());
    install_fake(&store, "llama-vulkan", "1.0.0");

    let mut config = ApiConfig::new(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml"),
    );
    config.backend_dir = temp.path().to_path_buf();
    let app = app(&config);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/backends")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert!(body["backends"].as_array().is_some_and(|items| {
        items.iter().any(|item| {
            item["kind"] == "builtin"
                && item["id"]
                    .as_str()
                    .is_some_and(|id| id.starts_with("builtin:"))
                && item["engines"]
                    .as_array()
                    .is_some_and(|engines| !engines.is_empty())
        })
    }));
    assert!(body["backends"].as_array().is_some_and(|items| {
        items
            .iter()
            .any(|item| item["id"] == "llama-vulkan" && item["kind"] == "sidecar")
    }));
    assert_eq!(body["active"]["id"], "builtin");
    assert_eq!(body["running"]["id"], "builtin");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/backends/llama-vulkan/activate")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["restart_required"], true);
    assert_eq!(body["active"]["version"], "1.0.0");
    assert_eq!(body["active"]["kind"], "sidecar");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/backends")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let body = body_json(response).await;
    assert_eq!(body["active"]["id"], "llama-vulkan");
    assert_eq!(body["running"]["id"], "builtin");
}

#[tokio::test]
async fn builtin_backend_is_selectable_and_clears_sidecar_marker() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = BackendStore::new(temp.path());
    install_fake(&store, "llama-vulkan", "1.0.0");
    store
        .set_active("llama-vulkan", "1.0.0")
        .expect("activate sidecar");

    let mut config = ApiConfig::new(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml"),
    );
    config.backend_dir = temp.path().to_path_buf();
    let app = app(&config);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/backends/builtin/activate")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["active"]["id"], "builtin");
    assert_eq!(body["active"]["kind"], "builtin");
    assert!(store.active().expect("active").is_none());
}
