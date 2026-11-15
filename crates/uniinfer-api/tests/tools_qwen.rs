//! Opt-in real-model tool-call test: `cargo test -p uniinfer-api --test tools_qwen -- --ignored`.

use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;
use uniinfer_api::{ApiConfig, AppState, routes};

#[tokio::test]
#[ignore = "loads the 379 MiB Qwen3 GGUF fixture"]
async fn qwen3_emits_structured_tool_call() {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3-0.6b/manifest.toml");
    let app = routes::router(AppState::new(&ApiConfig::new(manifest)).expect("state"));
    let body = json!({
        "model": "qwen3-0.6b-q4_k_m",
        "messages": [{"role": "user", "content": "Use the lookup function to find the capital of France."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a fact",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }],
        "max_tokens": 100
    })
    .to_string();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
}
