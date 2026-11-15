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
use uniinfer_backends::{FinishReason, GenerationSpec};
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
        prompt_tokens,
        include_usage,
        started,
        deadline,
        cancel,
        ..
    } = generation;
    let (tx, rx) = mpsc::channel::<StreamEvent>(32);
    let model = slot.manifest().id.clone();
    let process_isolated = slot.engine_name() == "process";
    let metric_model = model.clone();
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
        let mut measured_ttft = false;
        let result = task_slot.generate_with(
            &prompt,
            &cancel,
            GenerationSpec {
                max_tokens,
                sampling,
                deadline,
            },
            &mut |chunk| {
                if !measured_ttft {
                    measured_ttft = true;
                    task_state
                        .router
                        .record_ttft(&metric_model, started.elapsed());
                }
                matcher.push(chunk, &mut |safe| {
                    if open {
                        open = send_text(&tx, &mut decoder, safe);
                        if !open {
                            cancel.cancel();
                        }
                    }
                });
                open && !matcher.is_matched()
            },
        );
        matcher.finish(&mut |safe| {
            if open {
                open = send_text(&tx, &mut decoder, safe);
                if !open {
                    cancel.cancel();
                }
            }
        });
        let tail = decoder.finish();
        if open && !tail.is_empty() {
            let _ = tx.blocking_send(StreamEvent::Delta(tail));
        }
        let (finish, completion_tokens) = match result {
            Ok(output) => (reason(output.finish), output.tokens),
            Err(EngineError::Cancelled) => ("stop", 0),
            Err(error) => {
                if matches!(&error, EngineError::Timeout) {
                    task_state.router.record_timeout(&metric_model);
                    if process_isolated {
                        task_state.unload_text(&metric_model);
                    }
                }
                let _ = tx.blocking_send(StreamEvent::Error(error.into()));
                ("stop", 0)
            }
        };
        task_state
            .router
            .record_generation(&metric_model, started.elapsed());
        task_state
            .router
            .record_completion(&metric_model, completion_tokens);
        let _ = tx.blocking_send(StreamEvent::Finish(finish));
        if include_usage {
            let _ = tx.blocking_send(StreamEvent::Usage {
                prompt_tokens,
                completion_tokens,
            });
        }
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

fn reason(finish: FinishReason) -> &'static str {
    match finish {
        FinishReason::Length => "length",
        FinishReason::Eos | FinishReason::SinkClosed => "stop",
    }
}
