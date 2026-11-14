use std::path::Path;

use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;
use uniinfer_api::{ApiConfig, AppState, routes};

async fn start() -> (String, tokio::task::JoinHandle<()>) {
    let manifest =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-onnx/manifest.toml");
    let config = ApiConfig::new(manifest).with_token("secret");
    let state = AppState::new(&config).expect("state");
    let app = routes::router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let handle = tokio::spawn(async move {
        uniinfer_api::serve_listener(listener, app)
            .await
            .expect("server");
    });
    (format!("ws://{addr}/v1/realtime?token=secret"), handle)
}

async fn next_json(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> Value {
    loop {
        let message = socket.next().await.expect("message").expect("ws");
        if let Message::Text(text) = message {
            return serde_json::from_str(&text).expect("json");
        }
    }
}

#[tokio::test]
async fn streams_text_and_finishes() {
    let (url, server) = start().await;
    let (mut socket, _) = tokio_tungstenite::connect_async(url)
        .await
        .expect("connect");

    socket
        .send(Message::Text(
            json!({"type": "session.start", "model": "tiny-byte-onnx", "channels": ["text"]})
                .to_string()
                .into(),
        ))
        .await
        .expect("start");
    let started = next_json(&mut socket).await;
    assert_eq!(started["type"], "session.started");

    socket
        .send(Message::Text(
            json!({"type": "input.text", "text": "ab"})
                .to_string()
                .into(),
        ))
        .await
        .expect("input");
    assert_eq!(next_json(&mut socket).await["type"], "input.text.accepted");

    socket
        .send(Message::Text(
            json!({"type": "response.create", "max_tokens": 3})
                .to_string()
                .into(),
        ))
        .await
        .expect("create");

    let mut text = String::new();
    loop {
        let event = next_json(&mut socket).await;
        match event["type"].as_str() {
            Some("output.text.delta") => text.push_str(event["delta"].as_str().unwrap_or("")),
            Some("response.done") => break,
            Some("error") => panic!("unexpected error: {event}"),
            _ => {}
        }
    }
    assert_eq!(text, "cde");
    server.abort();
}

#[tokio::test]
async fn rejects_unsupported_channel() {
    let (url, server) = start().await;
    let (mut socket, _) = tokio_tungstenite::connect_async(url)
        .await
        .expect("connect");
    socket
        .send(Message::Text(
            json!({"type": "session.start", "model": "tiny-byte-onnx", "channels": ["image"]})
                .to_string()
                .into(),
        ))
        .await
        .expect("start");
    let error = next_json(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["code"], "unsupported_channel");
    server.abort();
}

fn pcm_frame(value: i16) -> Vec<u8> {
    value.to_le_bytes().repeat(320)
}

#[tokio::test]
async fn audio_frames_emit_vad_events() {
    let (url, server) = start().await;
    let (mut socket, _) = tokio_tungstenite::connect_async(url)
        .await
        .expect("connect");
    socket
        .send(Message::Text(
            json!({"type": "session.start", "model": "tiny-byte-onnx", "channels": ["text", "audio"]})
                .to_string()
                .into(),
        ))
        .await
        .expect("start");
    let started = next_json(&mut socket).await;
    assert_eq!(started["channels"], json!(["text", "audio"]));

    socket
        .send(Message::Binary(pcm_frame(2_000).into()))
        .await
        .expect("speech");
    let speech = next_json(&mut socket).await;
    assert_eq!(speech["type"], "input.speech.started");

    for _ in 0..10 {
        socket
            .send(Message::Binary(pcm_frame(0).into()))
            .await
            .expect("silence");
    }
    let stopped = next_json(&mut socket).await;
    assert_eq!(stopped["type"], "input.speech.stopped");
    server.abort();
}
