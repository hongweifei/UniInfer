//! Non-streaming chat completion response.

use axum::Json;
use axum::response::{IntoResponse, Response};

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
        cancel,
    } = generation;
    let task_state = state.clone();
    let task_session = session_id.clone();
    let model = slot.manifest().id.clone();
    let (result, bytes) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = task_state.router.active_guard();
        let mut bytes = Vec::new();
        let result = slot.generate(&prompt, max_tokens, &cancel, &mut |chunk| {
            bytes.extend_from_slice(chunk);
            true
        });
        task_state.router.unregister(&task_session);
        (result, bytes)
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
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": bytes.len(),
            "total_tokens": prompt_tokens + bytes.len()
        }
    }))
    .into_response())
}
