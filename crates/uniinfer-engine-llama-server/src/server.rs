//! `llama-server` process arguments and readiness policy.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use uniinfer_core::{CancelToken, Result};
use uniinfer_engine_sidecar::{ServerProcess, SpawnOptions};

pub(crate) fn spawn(
    executable: &Path,
    model: &Path,
    context: usize,
    n_gpu_layers: u32,
    cancel: &CancelToken,
) -> Result<Arc<ServerProcess>> {
    uniinfer_engine_sidecar::spawn(SpawnOptions {
        executable,
        readiness_path: "/health",
        startup_timeout: Duration::from_secs(300),
        request_timeout: Duration::from_secs(600),
        cancel,
        label: "llama-server",
        args: |port: u16| {
            vec![
                "--model".into(),
                model.as_os_str().to_owned(),
                "--host".into(),
                "127.0.0.1".into(),
                "--port".into(),
                port.to_string().into(),
                "--ctx-size".into(),
                context.to_string().into(),
                "--n-gpu-layers".into(),
                n_gpu_layers.to_string().into(),
                "--parallel".into(),
                "1".into(),
            ]
        },
    })
}
