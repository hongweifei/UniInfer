//! Isolated text model worker.
//!
//! The parent owns the process; this process owns the native model session.
//! Requests and replies are length-prefixed MessagePack frames on stdin/stdout.

use std::sync::Arc;

use uniinfer_core::{EngineSession, ModelManifest};
use uniinfer_engine_process::{
    WireSpec, WireTensor, WorkerRequest, WorkerResponse, decode_frame, read_frame, write_frame,
};
use uniinfer_service::{AudioSlot, ModelSlot, Runtime};

enum WorkerModel {
    Text(Arc<ModelSlot>),
    Audio(Arc<AudioSlot>),
}

impl WorkerModel {
    fn session(&self) -> &dyn EngineSession {
        match self {
            Self::Text(slot) => slot.session(),
            Self::Audio(slot) => slot.session(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: uniinfer-worker <manifest.toml>"))?;
    let manifest = ModelManifest::load(std::path::Path::new(&path))?;
    let runtime = Runtime::new()?;
    let slot = if manifest.text.is_none() {
        WorkerModel::Audio(runtime.load_audio(std::path::Path::new(&path))?)
    } else {
        WorkerModel::Text(runtime.load(std::path::Path::new(&path))?)
    };
    let session = slot.session();

    let mut stdout = std::io::stdout().lock();
    let ready = WorkerResponse::Ready {
        inputs: session.inputs().iter().map(WireSpec::from).collect(),
        outputs: session.outputs().iter().map(WireSpec::from).collect(),
    };
    write_frame(&mut stdout, &ready)?;

    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    while let Some(frame) = read_frame(&mut input)? {
        let request: WorkerRequest = decode_frame(&frame)?;
        match request {
            WorkerRequest::Shutdown => break,
            WorkerRequest::Run { inputs } => {
                let inputs = inputs
                    .into_iter()
                    .map(WireTensor::into_tensor)
                    .collect::<uniinfer_core::Result<Vec<_>>>();
                let response = match inputs.and_then(|inputs| session.run(inputs)) {
                    Ok(outputs) => WorkerResponse::Outputs {
                        outputs: outputs.into_iter().map(WireTensor::from_tensor).collect(),
                    },
                    Err(error) => WorkerResponse::Error {
                        code: error.code().to_string(),
                        message: error.to_string(),
                    },
                };
                write_frame(&mut stdout, &response)?;
            }
        }
    }
    Ok(())
}
