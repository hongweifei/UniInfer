//! Child process lifecycle and HTTP transport for `whisper-server`.

use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use reqwest::blocking::Client;
use uniinfer_core::{CancelToken, EngineError, Result};

pub(crate) struct ServerProcess {
    child: Mutex<Child>,
    pub(crate) base: String,
    pub(crate) client: Client,
}

impl ServerProcess {
    pub(crate) fn kill(&self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn is_alive(&self) -> bool {
        self.child
            .lock()
            .ok()
            .and_then(|mut child| child.try_wait().ok().flatten())
            .is_none()
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

pub(crate) fn spawn(
    executable: &Path,
    model: &Path,
    gpu: bool,
    cancel: &CancelToken,
) -> Result<Arc<ServerProcess>> {
    let port = free_port()?;
    let mut command = Command::new(executable);
    if let Some(parent) = executable.parent() {
        command.current_dir(parent);
        #[cfg(windows)]
        {
            // The official Windows archive keeps ggml/whisper DLLs next to the
            // server executable.
            let path = std::env::var_os("PATH").unwrap_or_default();
            let mut paths = vec![parent.to_path_buf()];
            paths.extend(std::env::split_paths(&path));
            if let Ok(joined) = std::env::join_paths(paths) {
                command.env("PATH", joined);
            }
        }
    }
    command
        .arg("--model")
        .arg(model)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--no-timestamps");
    if !gpu {
        command.arg("--no-gpu");
    }
    let child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| {
            EngineError::LoadFailed(format!("spawn {}: {error}", executable.display()))
        })?;
    let process = Arc::new(ServerProcess {
        child: Mutex::new(child),
        base: format!("http://127.0.0.1:{port}"),
        client: Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .map_err(|error| EngineError::LoadFailed(format!("HTTP client: {error}")))?,
    });
    wait_ready(&process, cancel)?;
    tracing::info!(port, gpu, "official whisper-server ready");
    Ok(process)
}

fn free_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| {
        EngineError::LoadFailed(format!("reserve whisper-server port: {error}"))
    })?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| EngineError::LoadFailed(format!("whisper-server port: {error}")))
}

fn wait_ready(process: &ServerProcess, cancel: &CancelToken) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if cancel.is_cancelled() {
            process.kill();
            return Err(EngineError::Cancelled);
        }
        if !process.is_alive() {
            return Err(EngineError::LoadFailed(
                "whisper-server exited during startup".into(),
            ));
        }
        if let Ok(response) = process
            .client
            .get(format!("{}/health", process.base))
            .timeout(Duration::from_secs(2))
            .send()
            && response.status().is_success()
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            process.kill();
            return Err(EngineError::Timeout);
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}
