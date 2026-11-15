//! Realtime text response task.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde_json::{Value, json};
use tokio::sync::mpsc;
use uniinfer_backends::utf8::Utf8Decoder;
use uniinfer_core::EngineError;
use uniinfer_router::{ModelRouter, PreparedText};

use super::NEXT_RESPONSE;

pub(super) fn spawn_response(
    prepared: PreparedText,
    tx: mpsc::Sender<String>,
    router: Arc<ModelRouter>,
) {
    let response_id = format!("resp-{:x}", NEXT_RESPONSE.fetch_add(1, Ordering::Relaxed));
    tokio::task::spawn_blocking(move || {
        let PreparedText {
            slot,
            permit,
            prompt,
            max_tokens,
            cancel,
            guard,
            started,
            ..
        } = prepared;
        let _permit = permit;
        let _guard = guard;
        let _active = router.active_guard();
        let mut decoder = Utf8Decoder::new();
        let metric_model = slot.manifest().id.clone();
        let mut measured_ttft = false;
        let send = |event: Value| -> bool { tx.blocking_send(event.to_string()).is_ok() };
        let result = slot.generate(&prompt, max_tokens, &cancel, &mut |chunk| {
            if !measured_ttft {
                measured_ttft = true;
                router.record_ttft(&metric_model, started.elapsed());
            }
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
        router.record_completion(
            &metric_model,
            result.as_ref().map_or(0, |output| output.tokens),
        );
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
