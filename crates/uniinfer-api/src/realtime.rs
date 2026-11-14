use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use uniinfer_core::{CancelToken, EngineError};

use crate::state::{ActiveGuard, AppState};
use crate::utf8::Utf8Decoder;

/// WebSocket entry point for the realtime session protocol.
pub(crate) async fn realtime(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if let Some(expected) = &state.token
        && params.get("token") != Some(expected)
    {
        return (StatusCode::UNAUTHORIZED, "invalid realtime token").into_response();
    }
    ws.on_upgrade(move |socket| session(socket, state))
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClientEvent {
    #[serde(rename = "session.start")]
    SessionStart {
        model: String,
        #[serde(default)]
        channels: Vec<String>,
    },
    #[serde(rename = "input.text")]
    InputText { text: String },
    #[serde(rename = "response.create")]
    ResponseCreate {
        #[serde(default)]
        max_tokens: Option<usize>,
    },
    #[serde(rename = "response.cancel")]
    ResponseCancel,
}

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
static NEXT_RESPONSE: AtomicU64 = AtomicU64::new(1);

async fn session(socket: WebSocket, state: AppState) {
    let (mut sink, mut incoming) = socket.split();
    let (tx, mut rx) = mpsc::channel::<String>(32);
    let mut session_id = String::new();
    let mut prompt: Option<String> = None;
    let mut active: Option<CancelToken> = None;

    loop {
        tokio::select! {
            message = incoming.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        match handle_client(
                            &state,
                            &tx,
                            &mut session_id,
                            &mut prompt,
                            &mut active,
                            &text,
                        )
                        .await
                        {
                            Ok(()) => {}
                            Err(error) => {
                                let _ = tx.send(error_json(&error)).await;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        if let Some(token) = active.take() {
                            token.cancel();
                        }
                        break;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(_)) => break,
                }
            }
            event = rx.recv() => {
                let Some(event) = event else { break };
                if event.contains("\"response.done\"") {
                    active = None;
                }
                if sink.send(Message::Text(event.into())).await.is_err() {
                    break;
                }
            }
        }
    }
}

async fn handle_client(
    state: &AppState,
    tx: &mpsc::Sender<String>,
    session_id: &mut String,
    prompt: &mut Option<String>,
    active: &mut Option<CancelToken>,
    text: &str,
) -> Result<(), String> {
    let event: ClientEvent = serde_json::from_str(text).map_err(|error| error.to_string())?;
    match event {
        ClientEvent::SessionStart { model, channels } => {
            let manifest = state.slot.manifest();
            if model != manifest.id {
                return tx
                    .send(error_event(
                        "model_not_found",
                        "requested model is not loaded",
                    ))
                    .await
                    .map_err(|_| "channel closed".to_string());
            }
            if let Some(channel) = channels.iter().find(|channel| channel.as_str() != "text") {
                return tx
                    .send(error_event(
                        "unsupported_channel",
                        &format!("channel {channel} is not available yet"),
                    ))
                    .await
                    .map_err(|_| "channel closed".to_string());
            }
            *session_id = new_session_id();
            *prompt = None;
            let payload = json!({
                "type": "session.started",
                "session_id": session_id.clone(),
                "model": manifest.id,
                "channels": ["text"],
            });
            tx.send(payload.to_string())
                .await
                .map_err(|_| "channel closed".to_string())
        }
        ClientEvent::InputText { text } => {
            *prompt = Some(text.clone());
            let payload = json!({"type": "input.text.accepted", "bytes": text.len()});
            tx.send(payload.to_string())
                .await
                .map_err(|_| "channel closed".to_string())
        }
        ClientEvent::ResponseCreate { max_tokens } => {
            if session_id.is_empty() {
                return tx
                    .send(error_event(
                        "session_not_started",
                        "send session.start first",
                    ))
                    .await
                    .map_err(|_| "channel closed".to_string());
            }
            if active.is_some() {
                return tx
                    .send(error_event("busy", "a response is already running"))
                    .await
                    .map_err(|_| "channel closed".to_string());
            }
            let Some(prompt) = prompt.clone() else {
                return tx
                    .send(error_event("invalid_request", "send input.text first"))
                    .await
                    .map_err(|_| "channel closed".to_string());
            };
            let Some(permit) = state.slot.try_acquire() else {
                return tx
                    .send(error_event("busy", "model is serving another request"))
                    .await
                    .map_err(|_| "channel closed".to_string());
            };
            let token = CancelToken::new();
            *active = Some(token.clone());
            spawn_response(state.clone(), tx.clone(), permit, token, prompt, max_tokens);
            Ok(())
        }
        ClientEvent::ResponseCancel => {
            if let Some(token) = active.take() {
                token.cancel();
                let payload = json!({"type": "response.cancelled"});
                tx.send(payload.to_string())
                    .await
                    .map_err(|_| "channel closed".to_string())?;
            }
            Ok(())
        }
    }
}

fn spawn_response(
    state: AppState,
    tx: mpsc::Sender<String>,
    permit: tokio::sync::OwnedSemaphorePermit,
    cancel: CancelToken,
    prompt: String,
    max_tokens: Option<usize>,
) {
    let response_id = format!("resp-{:x}", NEXT_RESPONSE.fetch_add(1, Ordering::Relaxed));
    let max_tokens = max_tokens.unwrap_or(16);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = ActiveGuard::new(state.active.clone());
        let mut decoder = Utf8Decoder::new();
        let send = |event: Value| -> bool { tx.blocking_send(event.to_string()).is_ok() };
        let result = state
            .slot
            .generate(&prompt, max_tokens, &cancel, &mut |chunk| {
                for &byte in chunk {
                    if let Some(delta) = decoder.push(byte) {
                        let event = json!({
                            "type": "output.text.delta",
                            "response_id": response_id,
                            "delta": delta,
                        });
                        if !send(event) {
                            return false;
                        }
                    }
                }
                true
            });
        let tail = decoder.finish();
        if !tail.is_empty() {
            let _ = send(json!({
                "type": "output.text.delta",
                "response_id": response_id,
                "delta": tail,
            }));
        }
        match result {
            Ok(_) => {
                let _ = send(json!({"type": "response.done", "response_id": response_id}));
            }
            Err(EngineError::Cancelled) => {
                let _ = send(
                    json!({"type": "response.done", "response_id": response_id, "cancelled": true}),
                );
            }
            Err(error) => {
                let _ = send(json!({
                    "type": "error",
                    "code": error.code(),
                    "message": error.to_string(),
                    "response_id": response_id,
                }));
                let _ = send(json!({"type": "response.done", "response_id": response_id}));
            }
        }
    });
}

fn error_json(error: &str) -> String {
    error_event("invalid_event", error)
}

fn error_event(code: &str, message: &str) -> String {
    json!({"type": "error", "code": code, "message": message}).to_string()
}

fn new_session_id() -> String {
    format!("rt-{:x}", NEXT_SESSION.fetch_add(1, Ordering::Relaxed))
}
