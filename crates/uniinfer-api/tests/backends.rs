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
        launcher: uniinfer_backend_store::BackendLauncher::Worker,
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
async fn lists_installed_backends_without_runtime_state() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = BackendStore::new(temp.path());
    install_fake(&store, "llama-vulkan", "1.0.0");

    let mut config = ApiConfig::new(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml"),
    );
    config.backend_dir = temp.path().to_path_buf();
    let app = app(&config);

    let response = app
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
    assert!(body.get("active").is_none());
    assert!(body.get("running").is_none());
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
}

#[tokio::test]
async fn removes_an_installed_backend() {
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
                .method("DELETE")
                .uri("/v1/backends/llama-vulkan")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["removed"], true);
    assert!(!temp.path().join("llama-vulkan").exists());

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
    assert!(
        body["backends"]
            .as_array()
            .is_some_and(|items| { items.iter().all(|item| item["id"] != "llama-vulkan") })
    );
}

#[tokio::test]
async fn active_marker_does_not_change_the_backend_list() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = BackendStore::new(temp.path());
    install_fake(&store, "llama-vulkan", "1.0.0");
    std::fs::write(temp.path().join("active"), "llama-vulkan/1.0.0\n").expect("legacy marker");

    let mut config = ApiConfig::new(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml"),
    );
    config.backend_dir = temp.path().to_path_buf();
    let app = app(&config);

    let response = app
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
    assert!(body.get("active").is_none());
    assert!(body.get("running").is_none());
    assert!(body["backends"].as_array().is_some_and(|items| {
        items
            .iter()
            .any(|item| item["id"] == "llama-vulkan" && item["kind"] == "sidecar")
    }));
}
