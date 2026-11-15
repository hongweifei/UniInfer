//! OpenAI-compatible transcription response formats.

use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseFormat {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

impl ResponseFormat {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "json" => Some(Self::Json),
            "text" => Some(Self::Text),
            "verbose_json" => Some(Self::VerboseJson),
            "srt" => Some(Self::Srt),
            "vtt" => Some(Self::Vtt),
            _ => None,
        }
    }
}

pub(crate) fn render(format: ResponseFormat, text: &str, duration_seconds: f64) -> Response {
    match format {
        ResponseFormat::Json => (
            StatusCode::OK,
            [(CONTENT_TYPE, "application/json")],
            json!({"text": text}).to_string(),
        )
            .into_response(),
        ResponseFormat::Text => (
            StatusCode::OK,
            [(CONTENT_TYPE, "text/plain; charset=utf-8")],
            text.to_owned(),
        )
            .into_response(),
        ResponseFormat::VerboseJson => (
            StatusCode::OK,
            [(CONTENT_TYPE, "application/json")],
            json!({
                "task": "transcribe",
                "duration": duration_seconds,
                "text": text,
                "segments": [{
                    "id": 0,
                    "start": 0.0,
                    "end": duration_seconds,
                    "text": text
                }]
            })
            .to_string(),
        )
            .into_response(),
        ResponseFormat::Srt => (
            StatusCode::OK,
            [(CONTENT_TYPE, "application/x-subrip; charset=utf-8")],
            format!(
                "1\n{} --> {}\n{text}\n",
                timestamp(0.0, true),
                timestamp(duration_seconds, true)
            ),
        )
            .into_response(),
        ResponseFormat::Vtt => (
            StatusCode::OK,
            [(CONTENT_TYPE, "text/vtt; charset=utf-8")],
            format!(
                "WEBVTT\n\n{} --> {}\n{text}\n",
                timestamp(0.0, false),
                timestamp(duration_seconds, false)
            ),
        )
            .into_response(),
    }
}

fn timestamp(seconds: f64, comma: bool) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let millis = total_ms % 1000;
    let total_seconds = total_ms / 1000;
    let secs = total_seconds % 60;
    let minutes = (total_seconds / 60) % 60;
    let hours = total_seconds / 3600;
    let separator = if comma { ',' } else { '.' };
    format!("{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_timestamps() {
        assert_eq!(timestamp(0.0, true), "00:00:00,000");
        assert_eq!(timestamp(3661.234, false), "01:01:01.234");
    }

    #[test]
    fn parses_formats() {
        assert_eq!(
            ResponseFormat::parse("verbose_json"),
            Some(ResponseFormat::VerboseJson)
        );
        assert_eq!(ResponseFormat::parse("bogus"), None);
    }
}
