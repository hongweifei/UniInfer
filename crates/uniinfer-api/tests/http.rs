use std::path::Path;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};

fn app() -> Router {
    app_with_token(None)
}

fn app_with_token(token: Option<&str>) -> Router {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml");
    let mut config = ApiConfig::new(manifest);
    if let Some(token) = token {
        config = config.with_token(token);
    }
    let state = AppState::new(&config).expect("state");
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
    assert_eq!(body["usage"]["completion_tokens"], 3);
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

#[tokio::test]
async fn token_is_required_when_configured() {
    let request = Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .expect("request");
    let response = app_with_token(Some("secret"))
        .oneshot(request)
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let request = Request::builder()
        .uri("/v1/models")
        .header("authorization", "Bearer secret")
        .body(Body::empty())
        .expect("request");
    let response = app_with_token(Some("secret"))
        .oneshot(request)
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn lists_and_routes_multiple_models() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx");
    let config = ApiConfig::new(fixture.join("manifest.toml")).with_manifests(vec![
        fixture.join("manifest.toml"),
        fixture.join("manifest-2.toml"),
    ]);
    let state = AppState::new(&config).expect("state");
    let app = routes::router(state);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let body = body_json(response).await;
    assert_eq!(body["data"].as_array().expect("array").len(), 2);

    let request = json!({
        "model": "tiny-byte-onnx-2",
        "messages": [{"role": "user", "content": "ab"}],
        "max_tokens": 3
    })
    .to_string();
    let response = app
        .oneshot(post("/v1/chat/completions", &request))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["choices"][0]["message"]["content"], "cde");
}

#[tokio::test]
async fn devices_endpoint_reports_runtime_plan() {
    let response = app()
        .oneshot(
            Request::builder()
                .uri("/v1/devices")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert!(!body["devices"].as_array().expect("devices").is_empty());
    assert_eq!(body["models"][0]["fallback_reason"], "engine_no_npu");
}

#[tokio::test]
async fn stop_string_truncates_completion() {
    let body = json!({
        "model": "tiny-byte-onnx",
        "messages": [{"role": "user", "content": "ab"}],
        "max_tokens": 3,
        "stop": "d"
    })
    .to_string();
    let response = app()
        .oneshot(post("/v1/chat/completions", &body))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["choices"][0]["message"]["content"], "c");
}

#[tokio::test]
async fn stop_array_truncates_streaming() {
    let body = json!({
        "model": "tiny-byte-onnx",
        "messages": [{"role": "user", "content": "ab"}],
        "max_tokens": 3,
        "stream": true,
        "stop": ["e", "zzz"]
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
    assert!(text.contains("\"content\":\"d\""));
    assert!(!text.contains("\"content\":\"e\""));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn audio_endpoint_reports_missing_model() {
    let boundary = "uniinfer-test-boundary";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n\r\nabc\r\n--{boundary}--\r\n"
    );
    let request = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("request");
    let response = app().oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "missing_model");
}

#[tokio::test]
async fn config_endpoint_round_trips_and_saves() {
    use std::sync::Arc;
    use uniinfer_config::{ConfigHandle, ModelEntry, ServerConfig, UniInferConfig};

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("uniinfer.toml");
    let mut config = UniInferConfig {
        server: ServerConfig::default(),
        ..UniInferConfig::default()
    };
    config.models = vec![ModelEntry::new("manifest.toml".into())];
    let handle = Arc::new(ConfigHandle::new(config, Some(path.clone())));

    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx");
    let mut api = ApiConfig::new(fixture.join("manifest.toml"));
    api = api.with_config(handle);
    let app = routes::router(AppState::new(&api).expect("state"));

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/config")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["config"]["models"][0]["manifest"], "manifest.toml");

    let next = json!({
        "schema_version": 1,
        "server": {"addr": "127.0.0.1:9091"},
        "models": [{"manifest": "manifest.toml", "enabled": true}]
    });
    let request = Request::builder()
        .method("PUT")
        .uri("/v1/config")
        .header("content-type", "application/json")
        .body(Body::from(next.to_string()))
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["saved"], true);
    assert!(path.exists());
    let saved = UniInferConfig::load(&path).expect("saved config");
    assert_eq!(saved.server.addr.port(), 9091);
}

#[tokio::test]
async fn model_inspect_and_import_endpoints_work() {
    use std::sync::Arc;
    use uniinfer_config::{ConfigHandle, UniInferConfig};

    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("model");
    std::fs::create_dir(&model_dir).expect("model dir");
    std::fs::write(model_dir.join("model.onnx"), b"onnx").expect("model");
    std::fs::write(
        model_dir.join("config.json"),
        r#"{"architectures":["Qwen3ForCausalLM"],"vocab_size":16,"max_position_embeddings":64}"#,
    )
    .expect("config");
    std::fs::write(model_dir.join("tokenizer.json"), b"{}").expect("tokenizer");

    let config_path = dir.path().join("uniinfer.toml");
    let handle = Arc::new(ConfigHandle::new(
        UniInferConfig::default(),
        Some(config_path.clone()),
    ));
    std::fs::write(&config_path, "schema_version = 1\n").expect("config file");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx");
    let api = ApiConfig::new(fixture.join("manifest.toml")).with_config(handle);
    let app = routes::router(AppState::new(&api).expect("state"));

    let body = json!({"path": model_dir.clone(), "id": "imported"}).to_string();
    let response = app
        .clone()
        .oneshot(post("/v1/models/inspect", &body))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["manifest"]["id"], "imported");
    assert_eq!(body["manifest"]["format"], "onnx");
    assert_eq!(body["route"]["supported"], true);

    let response = app
        .oneshot(post(
            "/v1/models/import",
            &json!({"path": model_dir, "id": "imported"}).to_string(),
        ))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    assert!(model_dir.join("manifest.toml").is_file());
    let saved = UniInferConfig::load(&config_path).expect("saved config");
    assert_eq!(saved.models.len(), 1);
}

#[tokio::test]
async fn planned_generation_endpoints_return_not_implemented() {
    for (method, uri) in [
        ("POST", "/v1/audio/speech"),
        ("POST", "/v1/images/generations"),
        ("POST", "/v1/images/edits"),
        ("POST", "/v1/videos/generations"),
        ("GET", "/v1/videos/generations/job"),
    ] {
        let response = app()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED, "{uri}");
        let body = body_json(response).await;
        assert_eq!(body["code"], "not_implemented", "{uri}");
    }
}

#[tokio::test]
async fn cors_only_allows_local_and_tauri_origins() {
    let request = |origin: &str| {
        Request::builder()
            .method("OPTIONS")
            .uri("/v1/models")
            .header("origin", origin)
            .header("access-control-request-method", "GET")
            .body(Body::empty())
            .expect("request")
    };
    let allowed = app()
        .oneshot(request("http://localhost:5173"))
        .await
        .expect("response");
    assert_eq!(allowed.status(), StatusCode::OK);
    assert!(
        allowed
            .headers()
            .contains_key("access-control-allow-origin")
    );

    let denied = app()
        .oneshot(request("https://evil.example"))
        .await
        .expect("response");
    assert!(!denied.headers().contains_key("access-control-allow-origin"));
}

#[tokio::test]
async fn config_get_redacts_token_and_put_preserves_it() {
    use std::sync::Arc;
    use uniinfer_config::{ConfigHandle, ModelEntry, ServerConfig, UniInferConfig};

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("uniinfer.toml");
    let mut durable = UniInferConfig {
        server: ServerConfig {
            token: Some("secret".into()),
            ..ServerConfig::default()
        },
        ..UniInferConfig::default()
    };
    durable.models = vec![ModelEntry::new("manifest.toml".into())];
    durable.save(&path).expect("save");
    let handle = Arc::new(ConfigHandle::load(path.clone()).expect("load"));
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx");
    let api = ApiConfig::new(fixture.join("manifest.toml")).with_config(handle);
    let app = routes::router(AppState::new(&api).expect("state"));

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/config")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let body = body_json(response).await;
    assert_eq!(body["token_configured"], true);
    assert_eq!(body["config"]["server"]["token"], "<redacted>");

    let mut next = body["config"].clone();
    next["server"]["addr"] = json!("127.0.0.1:9092");
    next["server"]["token"] = json!("<redacted>");
    let request = Request::builder()
        .method("PUT")
        .uri("/v1/config")
        .header("content-type", "application/json")
        .body(Body::from(next.to_string()))
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let saved = UniInferConfig::load(&path).expect("saved");
    assert_eq!(saved.server.token.as_deref(), Some("secret"));
}

#[tokio::test]
async fn model_inspect_rejects_paths_outside_config_root() {
    use std::sync::Arc;
    use uniinfer_config::ConfigHandle;

    let root = tempfile::tempdir().expect("root");
    let outside = tempfile::tempdir().expect("outside");
    std::fs::write(outside.path().join("model.onnx"), b"onnx").expect("model");
    let config_path = root.path().join("uniinfer.toml");
    std::fs::write(&config_path, "schema_version = 1\n").expect("config");
    let handle = Arc::new(ConfigHandle::load(config_path).expect("load"));
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx");
    let api = ApiConfig::new(fixture.join("manifest.toml")).with_config(handle);
    let app = routes::router(AppState::new(&api).expect("state"));

    let body = json!({"path": outside.path()}).to_string();
    let response = app
        .oneshot(post("/v1/models/inspect", &body))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let body = body_json(response).await;
    assert_eq!(body["code"], "path_not_allowed");
}

#[tokio::test]
async fn sampling_parameters_are_validated() {
    let body = json!({
        "model": "tiny-byte-onnx",
        "messages": [{"role": "user", "content": "ab"}],
        "max_tokens": 1,
        "temperature": 3.0
    })
    .to_string();
    let response = app()
        .oneshot(post("/v1/chat/completions", &body))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "invalid_sampling");

    let body = json!({
        "model": "tiny-byte-onnx",
        "messages": [{"role": "user", "content": "ab"}],
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 4,
        "seed": 7
    })
    .to_string();
    let response = app()
        .oneshot(post("/v1/chat/completions", &body))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
}
