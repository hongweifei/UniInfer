use std::convert::Infallible;

use axum::response::sse::Event;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::error::ApiError;

pub(crate) enum StreamEvent {
    Delta(String),
    Finish,
    Error(ApiError),
    Done,
}

/// Convert generation events into OpenAI-compatible SSE frames.
pub(crate) fn stream(
    rx: mpsc::Receiver<StreamEvent>,
    id: String,
    model: String,
    created: u64,
) -> impl tokio_stream::Stream<Item = Result<Event, Infallible>> {
    ReceiverStream::new(rx).map(move |event| {
        Ok(match event {
            StreamEvent::Delta(text) => {
                Event::default().data(chunk_json(&id, &model, created, &text))
            }
            StreamEvent::Finish => Event::default().data(finish_json(&id, &model, created)),
            StreamEvent::Error(error) => Event::default()
                .event("error")
                .data(json!({"code": error.code, "message": error.message}).to_string()),
            StreamEvent::Done => Event::default().data("[DONE]"),
        })
    })
}

fn finish_json(id: &str, model: &str, created: u64) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    })
    .to_string()
}

fn chunk_json(id: &str, model: &str, created: u64, text: &str) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": null
        }]
    })
    .to_string()
}
