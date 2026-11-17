//! `whisper-server` process arguments and readiness policy.

use std::ffi::OsString;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use uniinfer_core::{CancelToken, Result};
use uniinfer_engine_sidecar::{ServerProcess, SpawnOptions};

pub(crate) fn spawn(
    executable: &Path,
    model: &Path,
    gpu: bool,
    cancel: &CancelToken,
) -> Result<Arc<ServerProcess>> {
    uniinfer_engine_sidecar::spawn(SpawnOptions {
        executable,
        readiness_path: "/health",
        startup_timeout: Duration::from_secs(300),
        request_timeout: Duration::from_secs(600),
        cancel,
        label: "whisper-server",
        args: |port: u16| {
            let mut args = vec![
                OsString::from("--model"),
                model.as_os_str().to_owned(),
                OsString::from("--host"),
                OsString::from("127.0.0.1"),
                OsString::from("--port"),
                port.to_string().into(),
                OsString::from("--no-timestamps"),
            ];
            if !gpu {
                args.push(OsString::from("--no-gpu"));
            }
            args
        },
    })
}
