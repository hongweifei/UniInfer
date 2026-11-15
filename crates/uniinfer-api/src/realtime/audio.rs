//! Realtime audio frame handling.

use serde_json::json;
use tokio::sync::mpsc;
use uniinfer_backends::audio::{FRAME_BYTES, VadEvent};

use super::{SessionState, error_event};

pub(super) async fn handle_audio(
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
        Some(VadEvent::SpeechStopped) => tx
            .send(json!({"type": "input.speech.stopped"}).to_string())
            .await
            .map_err(|_| "channel closed".to_string()),
        None => Ok(()),
    }
}
