//! Non-streaming chat completion response.

use axum::Json;
use axum::response::{IntoResponse, Response};
use uniinfer_backends::FinishReason;
use uniinfer_backends::stop::StopMatcher;

use crate::error::ApiError;

use super::{Generation, unix_seconds};

pub(super) async fn complete_response(generation: Generation) -> Result<Response, ApiError> {
    let Generation {
        state,
        slot,
        permit,
        session_id,
        prompt,
        prompt_tokens,
        max_tokens,
        stop,
        sampling,
        include_usage: _,
        cancel,
    } = generation;
    let task_state = state.clone();
    let task_session = session_id.clone();
    let model = slot.manifest().id.clone();
    let (result, bytes, finish, completion_tokens) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = task_state.router.active_guard();
        let mut bytes = Vec::new();
        let mut matcher = StopMatcher::new(stop);
        let output = slot.generate_with(&prompt, max_tokens, &cancel, &sampling, &mut |chunk| {
            matcher.push(chunk, &mut |safe| bytes.extend_from_slice(safe))
        });
        matcher.finish(&mut |safe| bytes.extend_from_slice(safe));
        task_state.router.unregister(&task_session);
        let finish = output
            .as_ref()
            .map_or(FinishReason::SinkClosed, |output| output.finish);
        let tokens = output.as_ref().map_or(0, |output| output.tokens);
        (output.map(|_| ()), bytes, finish, tokens)
    })
    .await
    .map_err(|_| ApiError::internal("generation task panicked"))?;

    if let Err(error) = result {
        return Err(error.into());
    }

    let content = String::from_utf8_lossy(&bytes).into_owned();
    Ok(Json(serde_json::json!({
        "id": format!("chatcmpl-{session_id}"),
        "object": "chat.completion",
        "created": unix_seconds(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason(finish)
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }))
    .into_response())
}

fn finish_reason(finish: FinishReason) -> &'static str {
    match finish {
        FinishReason::Length => "length",
        FinishReason::Eos | FinishReason::SinkClosed => "stop",
    }
}
