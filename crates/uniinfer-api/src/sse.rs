use std::convert::Infallible;

use axum::response::sse::Event;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use uniinfer_backends::ToolCall;

use crate::error::ApiError;

pub(crate) enum StreamEvent {
    Delta(String),
    Reasoning(String),
    ToolCalls(Vec<ToolCall>),
    Finish(&'static str),
    Usage {
        prompt_tokens: usize,
        completion_tokens: usize,
    },
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
            StreamEvent::Reasoning(text) => {
                Event::default().data(reasoning_json(&id, &model, created, &text))
            }
            StreamEvent::ToolCalls(calls) => {
                Event::default().data(tool_calls_json(&id, &model, created, &calls))
            }
            StreamEvent::Finish(reason) => {
                Event::default().data(finish_json(&id, &model, created, reason))
            }
            StreamEvent::Usage {
                prompt_tokens,
                completion_tokens,
            } => Event::default().data(usage_json(
                &id,
                &model,
                created,
                prompt_tokens,
                completion_tokens,
            )),
            StreamEvent::Error(error) => Event::default()
                .event("error")
                .data(json!({"code": error.code, "message": error.message}).to_string()),
            StreamEvent::Done => Event::default().data("[DONE]"),
        })
    })
}

fn finish_json(id: &str, model: &str, created: u64, reason: &str) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": reason
        }]
    })
    .to_string()
}

fn tool_calls_json(id: &str, model: &str, created: u64, calls: &[ToolCall]) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": calls},
            "finish_reason": null
        }]
    })
    .to_string()
}

fn reasoning_json(id: &str, model: &str, created: u64, text: &str) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"reasoning_content": text},
            "finish_reason": null
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

fn usage_json(
    id: &str,
    model: &str,
    created: u64,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_chunk_uses_reasoning_content() {
        let frame = reasoning_json("chatcmpl-1", "model", 1, "step");
        let value: serde_json::Value = serde_json::from_str(&frame).expect("json");
        assert_eq!(value["choices"][0]["delta"]["reasoning_content"], "step");
    }
}
