//! Realtime audio frame handling.

use serde_json::json;
use tokio::sync::mpsc;
use uniinfer_backends::audio::{FRAME_BYTES, VadEvent};

use crate::state::AppState;

use super::{SessionState, error_event, start_response};

pub(super) async fn handle_audio(
    state: &AppState,
    tx: &mpsc::Sender<String>,
    session: &mut SessionState,
    frame: &[u8],
) -> Result<(), String> {
    if !session.audio_enabled {
        return tx
            .send(error_event(
                "unsupported_channel",
                "session.start did not enable audio",
            ))
            .await
            .map_err(|_| "channel closed".to_string());
    }
    if frame.len() != FRAME_BYTES {
        return tx
            .send(error_event(
                "invalid_audio_frame",
                &format!("expected {FRAME_BYTES} bytes of 16 kHz PCM16"),
            ))
            .await
            .map_err(|_| "channel closed".to_string());
    }

    match session.vad.push(frame) {
        Some(VadEvent::SpeechStarted) => {
            session.speaking = true;
            session.audio_buffer.clear();
            append_pcm16(&mut session.audio_buffer, frame);
            tx.send(json!({"type": "input.speech.started"}).to_string())
                .await
                .map_err(|_| "channel closed".to_string())?;
            if let Some(token) = session.active.take() {
                token.cancel();
                tx.send(json!({"type": "response.cancelled", "reason": "barge_in"}).to_string())
                    .await
                    .map_err(|_| "channel closed".to_string())?;
            }
            Ok(())
        }
        Some(VadEvent::SpeechStopped) => {
            session.speaking = false;
            let samples = std::mem::take(&mut session.audio_buffer);
            tx.send(json!({"type": "input.speech.stopped"}).to_string())
                .await
                .map_err(|_| "channel closed".to_string())?;
            transcribe_and_respond(state, tx, session, samples).await
        }
        None => {
            if session.speaking {
                append_pcm16(&mut session.audio_buffer, frame);
            }
            Ok(())
        }
    }
}

async fn transcribe_and_respond(
    state: &AppState,
    tx: &mpsc::Sender<String>,
    session: &mut SessionState,
    samples: Vec<f32>,
) -> Result<(), String> {
    let Some(asr) = session.asr_slot.clone() else {
        return Ok(());
    };
    if samples.is_empty() {
        return Ok(());
    }
    let permit = match asr.acquire(state.router.queue_timeout()).await {
        Ok(permit) => permit,
        Err(error) => {
            return tx
                .send(error_event(error.code(), &error.to_string()))
                .await
                .map_err(|_| "channel closed".to_string());
        }
    };
    let text = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        asr.transcribe(samples)
    })
    .await
    .map_err(|_| "transcription task panicked".to_string())?;
    let text = match text {
        Ok(text) => text,
        Err(error) => {
            return tx
                .send(error_event(error.code(), &error.to_string()))
                .await
                .map_err(|_| "channel closed".to_string());
        }
    };
    tx.send(
        json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": text,
        })
        .to_string(),
    )
    .await
    .map_err(|_| "channel closed".to_string())?;
    if text.trim().is_empty() {
        return Ok(());
    }
    session.prompt = Some(text);
    start_response(state, tx, session, Some(64)).await
}

fn append_pcm16(buffer: &mut Vec<f32>, frame: &[u8]) {
    buffer.extend(
        frame
            .chunks_exact(2)
            .map(|sample| f32::from(i16::from_le_bytes([sample[0], sample[1]])) / 32768.0),
    );
}
