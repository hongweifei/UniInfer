//! Non-streaming chat completion response.

use axum::Json;
use axum::response::{IntoResponse, Response};
use uniinfer_backends::stop::StopMatcher;
use uniinfer_backends::{FinishReason, GenerationSpec};

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
        started,
        deadline,
        cancel,
    } = generation;
    let task_state = state.clone();
    let task_session = session_id.clone();
    let model = slot.manifest().id.clone();
    let process_isolated = slot.engine_name() == "process";
    let metric_model = model.clone();
    let task_model = model.clone();
    let (result, bytes, finish, completion_tokens) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = task_state.router.active_guard();
        let mut bytes = Vec::new();
        let mut matcher = StopMatcher::new(stop);
        let mut measured_ttft = false;
        let output = slot.generate_with(
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
                        .record_ttft(&task_model, started.elapsed());
                }
                matcher.push(chunk, &mut |safe| bytes.extend_from_slice(safe))
            },
        );
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
        if matches!(error, uniinfer_core::EngineError::Timeout) {
            state.router.record_timeout(&metric_model);
        }
        if process_isolated
            && matches!(
                error,
                uniinfer_core::EngineError::Timeout | uniinfer_core::EngineError::ProcessExited
            )
        {
            state.unload_text(&metric_model);
        }
        return Err(error.into());
    }
    state
        .router
        .record_completion(&metric_model, completion_tokens);

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
