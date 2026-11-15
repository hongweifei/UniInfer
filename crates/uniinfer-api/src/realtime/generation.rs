//! Realtime text response task.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde_json::{Value, json};
use tokio::sync::mpsc;
use uniinfer_backends::utf8::Utf8Decoder;
use uniinfer_core::{CancelToken, EngineError};
use uniinfer_service::ModelSlot;

use super::NEXT_RESPONSE;

pub(super) fn spawn_response(
    slot: Arc<ModelSlot>,
    tx: mpsc::Sender<String>,
    permit: tokio::sync::OwnedSemaphorePermit,
    cancel: CancelToken,
    prompt: String,
    max_tokens: Option<usize>,
    active: uniinfer_router::ActiveGuard,
) {
    let response_id = format!("resp-{:x}", NEXT_RESPONSE.fetch_add(1, Ordering::Relaxed));
    let max_tokens = max_tokens.unwrap_or(16);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = active;
        let mut decoder = Utf8Decoder::new();
        let send = |event: Value| -> bool { tx.blocking_send(event.to_string()).is_ok() };
        let result = slot.generate(&prompt, max_tokens, &cancel, &mut |chunk| {
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
