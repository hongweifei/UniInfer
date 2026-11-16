use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use uniinfer_core::CancelToken;
use uniinfer_service::{AudioSlot, ModelSlot};

use crate::state::AppState;

mod audio;
mod generation;

use audio::handle_audio;
use generation::spawn_response;
use uniinfer_backends::audio::EnergyVad;

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
        #[serde(default)]
        asr_model: Option<String>,
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

pub(super) struct SessionState {
    id: String,
    prompt: Option<String>,
    slot: Option<Arc<ModelSlot>>,
    pub(super) asr_slot: Option<Arc<AudioSlot>>,
    pub(super) active: Option<CancelToken>,
    pub(super) audio_enabled: bool,
    pub(super) speaking: bool,
    pub(super) audio_buffer: Vec<f32>,
    pub(super) vad: EnergyVad,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            id: String::new(),
            prompt: None,
            slot: None,
            asr_slot: None,
            active: None,
            audio_enabled: false,
            speaking: false,
            audio_buffer: Vec::new(),
            vad: EnergyVad::new(500.0),
        }
    }
}

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
pub(super) static NEXT_RESPONSE: AtomicU64 = AtomicU64::new(1);

async fn session(socket: WebSocket, state: AppState) {
    let (mut sink, mut incoming) = socket.split();
    let (tx, mut rx) = mpsc::channel::<String>(32);
    let mut session = SessionState::default();

    loop {
        tokio::select! {
            message = incoming.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        if let Err(error) = handle_client(&state, &tx, &mut session, &text).await {
                            let _ = tx.send(error_json(&error)).await;
                        }
                    }
                    Some(Ok(Message::Binary(frame))) => {
                        if let Err(error) = handle_audio(&state, &tx, &mut session, &frame).await {
                            let _ = tx.send(error_json(&error)).await;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        if let Some(token) = session.active.take() {
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
                    session.active = None;
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
    session: &mut SessionState,
    text: &str,
) -> Result<(), String> {
    let event: ClientEvent = serde_json::from_str(text).map_err(|error| error.to_string())?;
    match event {
        ClientEvent::SessionStart {
            model,
            channels,
            asr_model,
        } => {
            let Some(entry) = state.catalog_entry(&model) else {
                return tx
                    .send(error_event(
                        "model_not_found",
                        "requested model is not configured",
                    ))
                    .await
                    .map_err(|_| "channel closed".to_string());
            };
            if entry.audio_only {
                return tx
                    .send(error_event(
                        "model_not_text",
                        "realtime text model is audio-only",
                    ))
                    .await
                    .map_err(|_| "channel closed".to_string());
            }
            let selected = match state.text_slot(&model).await {
                Ok(slot) => slot,
                Err(error) => {
                    return tx
                        .send(error_event(error.code(), &error.to_string()))
                        .await
                        .map_err(|_| "channel closed".to_string());
                }
            };
            let model_id = selected.manifest().id.clone();
            if let Some(channel) = channels
                .iter()
                .find(|channel| !matches!(channel.as_str(), "text" | "audio"))
            {
                return tx
                    .send(error_event(
                        "unsupported_channel",
                        &format!("channel {channel} is not available yet"),
                    ))
                    .await
                    .map_err(|_| "channel closed".to_string());
            }
            let audio = channels.iter().any(|channel| channel == "audio");
            let asr_slot = if audio {
                match select_asr(state, asr_model.as_deref()).await {
                    Ok(slot) => slot,
                    Err(message) => {
                        return tx
                            .send(error_event("asr_model_unavailable", &message))
                            .await
                            .map_err(|_| "channel closed".to_string());
                    }
                }
            } else {
                None
            };
            session.id = new_session_id();
            session.prompt = None;
            session.slot = Some(selected);
            session.asr_slot = asr_slot;
            session.audio_enabled = audio;
            session.speaking = false;
            session.audio_buffer.clear();
            session.vad = EnergyVad::new(500.0);
            let payload = json!({
                "type": "session.started",
                "session_id": session.id,
                "model": model_id,
                "channels": if audio { vec!["text", "audio"] } else { vec!["text"] },
                "asr": session.asr_slot.as_ref().map(|slot| slot.manifest().id.clone()),
            });
            tx.send(payload.to_string())
                .await
                .map_err(|_| "channel closed".to_string())
        }
        ClientEvent::InputText { text } => {
            session.prompt = Some(text.clone());
            let payload = json!({"type": "input.text.accepted", "bytes": text.len()});
            tx.send(payload.to_string())
                .await
                .map_err(|_| "channel closed".to_string())
        }
        ClientEvent::ResponseCreate { max_tokens } => {
            start_response(state, tx, session, max_tokens).await
        }
        ClientEvent::ResponseCancel => {
            if let Some(token) = session.active.take() {
                token.cancel();
                tx.send(json!({"type": "response.cancelled"}).to_string())
                    .await
                    .map_err(|_| "channel closed".to_string())?;
            }
            Ok(())
        }
    }
}

async fn select_asr(
    state: &AppState,
    requested: Option<&str>,
) -> Result<Option<Arc<AudioSlot>>, String> {
    if let Some(model) = requested {
        if !state.has_model(model) {
            return Err(format!("ASR model {model} is not configured"));
        }
        return state
            .audio_slot(model)
            .await
            .map(Some)
            .map_err(|error| error.to_string());
    }
    match state.audio_catalog_ids().as_slice() {
        [] => Ok(None),
        [model] => state
            .audio_slot(model)
            .await
            .map(Some)
            .map_err(|error| error.to_string()),
        _ => Err("multiple ASR models are configured; set asr_model explicitly".into()),
    }
}

async fn start_response(
    state: &AppState,
    tx: &mpsc::Sender<String>,
    session: &mut SessionState,
    max_tokens: Option<usize>,
) -> Result<(), String> {
    if session.id.is_empty() {
        return tx
            .send(error_event(
                "session_not_started",
                "send session.start first",
            ))
            .await
            .map_err(|_| "channel closed".to_string());
    }
    if session.active.is_some() {
        return tx
            .send(error_event("busy", "a response is already running"))
            .await
            .map_err(|_| "channel closed".to_string());
    }
    let Some(selected) = session.slot.clone() else {
        return tx
            .send(error_event(
                "session_not_started",
                "send session.start first",
            ))
            .await
            .map_err(|_| "channel closed".to_string());
    };
    let Some(prompt) = session.prompt.clone() else {
        return tx
            .send(error_event("invalid_request", "send input.text first"))
            .await
            .map_err(|_| "channel closed".to_string());
    };
    let model = selected.manifest().id.clone();
    let prepared = match state
        .router
        .prepare_text_now(&model, prompt, max_tokens)
        .await
    {
        Ok(prepared) => prepared,
        Err(error) => {
            return tx
                .send(error_event(error.code(), &error.to_string()))
                .await
                .map_err(|_| "channel closed".to_string());
        }
    };
    session.active = Some(prepared.cancel.clone());
    let deadline = state
        .request_timeout()
        .map(|timeout| std::time::Instant::now() + timeout);
    spawn_response(prepared, tx.clone(), Arc::clone(&state.router), deadline);
    Ok(())
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
