//! SSE translation between `llama-server` `/completion` and the native decoder.

use std::io::{BufRead, BufReader};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use uniinfer_core::{
    CancelToken, EngineError, Modality, ModelFormat, NativeFinish, NativeTextOutput,
    NativeTextRequest, Result,
};

use uniinfer_engine_sidecar::ServerProcess;

#[derive(Serialize)]
struct CompletionRequest<'a> {
    prompt: &'a str,
    n_predict: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    seed: u64,
    stream: bool,
}

#[derive(Deserialize)]
struct CompletionChunk {
    #[serde(default)]
    content: String,
    #[serde(default)]
    stop: bool,
    #[serde(default)]
    tokens_predicted: Option<usize>,
}

pub(crate) fn stream_completion(
    process: &ServerProcess,
    request: NativeTextRequest<'_>,
    on_bytes: &mut dyn FnMut(&[u8]) -> bool,
) -> Result<NativeTextOutput> {
    check_cancel(request.cancel)?;
    if !request.images.is_empty() {
        return Err(EngineError::UnsupportedModality {
            format: ModelFormat::Gguf,
            modality: Modality::Image,
        });
    }
    let payload = CompletionRequest {
        prompt: request.prompt,
        n_predict: request.max_tokens.max(1),
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        seed: request.seed.unwrap_or(0),
        stream: true,
    };
    let mut call = process
        .client
        .post(format!("{}/completion", process.base))
        .json(&payload);
    if let Some(deadline) = request.deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(EngineError::Timeout);
        }
        call = call.timeout(remaining);
    }
    let response = call.send().map_err(map_request_error)?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(EngineError::LoadFailed(format!(
            "llama-server /completion HTTP {status}: {body}"
        )));
    }
    read_stream(BufReader::new(response), request.cancel, on_bytes)
}

fn read_stream<R: BufRead>(
    mut reader: R,
    cancel: &CancelToken,
    on_bytes: &mut dyn FnMut(&[u8]) -> bool,
) -> Result<NativeTextOutput> {
    let mut line = String::new();
    let mut bytes = Vec::new();
    let mut tokens = 0;
    let mut finish = NativeFinish::Eos;
    loop {
        check_cancel(cancel)?;
        line.clear();
        let read = reader
            .read_line(&mut line)
            .map_err(|error| EngineError::Other(format!("llama-server stream: {error}")))?;
        if read == 0 {
            break;
        }
        let Some(data) = line.trim_end().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let Ok(chunk) = serde_json::from_str::<CompletionChunk>(data) else {
            continue;
        };
        if !chunk.content.is_empty() {
            if !on_bytes(chunk.content.as_bytes()) {
                finish = NativeFinish::SinkClosed;
                break;
            }
            bytes.extend_from_slice(chunk.content.as_bytes());
        }
        if let Some(predicted) = chunk.tokens_predicted {
            tokens = predicted;
        }
        if chunk.stop {
            break;
        }
    }
    if tokens == 0 {
        tokens = bytes.len().max(1);
    }
    Ok(NativeTextOutput {
        bytes,
        tokens,
        finish,
    })
}

fn check_cancel(cancel: &CancelToken) -> Result<()> {
    if cancel.is_cancelled() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

fn map_request_error(error: reqwest::Error) -> EngineError {
    if error.is_timeout() {
        EngineError::Timeout
    } else {
        EngineError::Other(format!("llama-server request: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_completion_sse_chunks() {
        let stream = concat!(
            "data: {\"content\":\"he\",\"tokens_predicted\":1}\n\n",
            "data: {\"content\":\"llo\",\"tokens_predicted\":2}\n\n",
            "data: [DONE]\n\n",
        );
        let mut bytes = Vec::new();
        let output = read_stream(
            Cursor::new(stream.as_bytes()),
            &CancelToken::new(),
            &mut |chunk| {
                bytes.extend_from_slice(chunk);
                true
            },
        )
        .expect("stream");
        assert_eq!(bytes, b"hello");
        assert_eq!(output.bytes, b"hello");
        assert_eq!(output.tokens, 2);
        assert_eq!(output.finish, NativeFinish::Eos);
    }

    #[test]
    fn stops_when_sink_closes() {
        let stream = r#"data: {"content":"hello","tokens_predicted":1}

"#;
        let output = read_stream(
            Cursor::new(stream.as_bytes()),
            &CancelToken::new(),
            &mut |_| false,
        )
        .expect("stream");
        assert_eq!(output.finish, NativeFinish::SinkClosed);
    }
}
