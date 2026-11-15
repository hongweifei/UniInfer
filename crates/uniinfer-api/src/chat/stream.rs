//! Streaming chat completion response.

use axum::http::HeaderValue;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::response::sse::{KeepAlive, Sse};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use uniinfer_backends::stop::StopMatcher;
use uniinfer_backends::utf8::Utf8Decoder;
use uniinfer_core::EngineError;

use crate::error::ApiError;
use crate::sse::{self, StreamEvent};

use super::{Generation, unix_seconds};

fn send_text(tx: &mpsc::Sender<StreamEvent>, decoder: &mut Utf8Decoder, bytes: &[u8]) -> bool {
    for &byte in bytes {
        if let Some(text) = decoder.push(byte)
            && tx.blocking_send(StreamEvent::Delta(text)).is_err()
        {
            return false;
        }
    }
    true
}

pub(super) async fn stream_response(generation: Generation) -> Result<Response, ApiError> {
    let Generation {
        state,
        slot,
        permit,
        session_id,
        prompt,
        max_tokens,
        stop,
        sampling,
        cancel,
        ..
    } = generation;
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);
    let model = slot.manifest().id.clone();
    let created = unix_seconds();
    let task_state = state.clone();
    let task_session = session_id.clone();
    let task_slot = Arc::clone(&slot);

    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = task_state.router.active_guard();
        let mut matcher = StopMatcher::new(stop);
        let mut decoder = Utf8Decoder::new();
        let mut open = true;
        let result =
            task_slot.generate_with(&prompt, max_tokens, &cancel, &sampling, &mut |chunk| {
                matcher.push(chunk, &mut |safe| {
                    if open {
                        open = send_text(&tx, &mut decoder, safe);
                    }
                });
                open && !matcher.is_matched()
            });
        matcher.finish(&mut |safe| {
            if open {
                open = send_text(&tx, &mut decoder, safe);
            }
        });
        let tail = decoder.finish();
        if open && !tail.is_empty() {
            let _ = tx.blocking_send(StreamEvent::Delta(tail));
        }
        match result {
            Ok(_) | Err(EngineError::Cancelled) => {}
            Err(error) => {
                let _ = tx.blocking_send(StreamEvent::Error(error.into()));
            }
        }
        let _ = tx.blocking_send(StreamEvent::Finish);
        let _ = tx.blocking_send(StreamEvent::Done);
        task_state.router.unregister(&task_session);
    });

    let stream = sse::stream(rx, session_id.clone(), model, created);
    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response();
    response.headers_mut().insert(
        "x-uniinfer-session-id",
        HeaderValue::from_str(&session_id).map_err(|_| ApiError::internal("invalid session id"))?,
    );
    Ok(response)
}
