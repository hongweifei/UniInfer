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
use uniinfer_backends::{
    FinishReason, GenerationSpec, ReasoningEvent, ReasoningSplitter, ToolStream, ToolStreamEvent,
};
use uniinfer_core::EngineError;

use crate::error::ApiError;
use crate::sse::{self, StreamEvent};

use super::{Generation, unix_seconds};

fn send_split(
    tx: &mpsc::Sender<StreamEvent>,
    splitter: &mut ReasoningSplitter,
    text: &str,
) -> bool {
    for event in splitter.push(text) {
        let event = match event {
            ReasoningEvent::Reasoning(text) => StreamEvent::Reasoning(text),
            ReasoningEvent::Content(text) => StreamEvent::Delta(text),
        };
        if tx.blocking_send(event).is_err() {
            return false;
        }
    }
    true
}

fn send_text(
    tx: &mpsc::Sender<StreamEvent>,
    decoder: &mut Utf8Decoder,
    splitter: &mut ReasoningSplitter,
    bytes: &[u8],
) -> bool {
    for &byte in bytes {
        if let Some(text) = decoder.push(byte)
            && !send_split(tx, splitter, &text)
        {
            return false;
        }
    }
    true
}

fn send_tool_text(
    tx: &mpsc::Sender<StreamEvent>,
    decoder: &mut Utf8Decoder,
    splitter: &mut ReasoningSplitter,
    stream: &mut ToolStream,
    bytes: &[u8],
) -> bool {
    for &byte in bytes {
        if let Some(text) = decoder.push(byte) {
            for event in stream.push(&text) {
                match event {
                    ToolStreamEvent::Text(text) => {
                        if !send_split(tx, splitter, &text) {
                            return false;
                        }
                    }
                    ToolStreamEvent::Calls(calls) => {
                        if tx.blocking_send(StreamEvent::ToolCalls(calls)).is_err() {
                            return false;
                        }
                    }
                }
            }
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
        images,
        max_tokens,
        stop,
        sampling,
        prompt_tokens,
        include_usage,
        parse_tool_calls: tool_calls_enabled,
        implicit_reasoning,
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
        let mut reasoning = ReasoningSplitter::new(implicit_reasoning);
        let mut tool_stream = tool_calls_enabled.then(|| ToolStream::new(task_session.clone()));
        let mut open = true;
        let mut measured_ttft = false;
        let result = task_slot.generate_with_images(
            &prompt,
            &images,
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
                        open = match tool_stream.as_mut() {
                            Some(stream) => {
                                send_tool_text(&tx, &mut decoder, &mut reasoning, stream, safe)
                            }
                            None => send_text(&tx, &mut decoder, &mut reasoning, safe),
                        };
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
                open = match tool_stream.as_mut() {
                    Some(stream) => send_tool_text(&tx, &mut decoder, &mut reasoning, stream, safe),
                    None => send_text(&tx, &mut decoder, &mut reasoning, safe),
                };
                if !open {
                    cancel.cancel();
                }
            }
        });
        let tail = decoder.finish();
        if open && !tail.is_empty() {
            if let Some(stream) = tool_stream.as_mut() {
                for event in stream.push(&tail) {
                    match event {
                        ToolStreamEvent::Text(text) => {
                            if !send_split(&tx, &mut reasoning, &text) {
                                cancel.cancel();
                                break;
                            }
                        }
                        ToolStreamEvent::Calls(calls) => {
                            if tx.blocking_send(StreamEvent::ToolCalls(calls)).is_err() {
                                cancel.cancel();
                                break;
                            }
                        }
                    }
                }
            } else if !send_split(&tx, &mut reasoning, &tail) {
                cancel.cancel();
            }
        }
        let (mut finish, completion_tokens) = match result {
            Ok(output) => (reason(output.finish), output.tokens),
            Err(EngineError::Cancelled) => ("stop", 0),
            Err(error) => {
                if matches!(&error, EngineError::Timeout) {
                    task_state.router.record_timeout(&metric_model);
                }
                if process_isolated
                    && matches!(&error, EngineError::Timeout | EngineError::ProcessExited)
                {
                    task_state.unload_text(&metric_model);
                }
                let _ = tx.blocking_send(StreamEvent::Error(error.into()));
                ("stop", 0)
            }
        };
        if open && let Some(stream) = tool_stream {
            let mut emitted_calls = false;
            for event in stream.finish() {
                match event {
                    ToolStreamEvent::Text(text) => {
                        if !send_split(&tx, &mut reasoning, &text) {
                            cancel.cancel();
                            break;
                        }
                    }
                    ToolStreamEvent::Calls(calls) => {
                        emitted_calls = true;
                        if tx.blocking_send(StreamEvent::ToolCalls(calls)).is_err() {
                            cancel.cancel();
                            break;
                        }
                    }
                }
            }
            if emitted_calls {
                finish = "tool_calls";
            }
        }
        if open {
            for event in reasoning.finish() {
                let event = match event {
                    ReasoningEvent::Reasoning(text) => StreamEvent::Reasoning(text),
                    ReasoningEvent::Content(text) => StreamEvent::Delta(text),
                };
                if tx.blocking_send(event).is_err() {
                    break;
                }
            }
        }
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
