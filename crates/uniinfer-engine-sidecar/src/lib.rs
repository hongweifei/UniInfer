//! Child-process lifecycle shared by official HTTP sidecar engines.
//!
//! The engine crates supply the executable, port-dependent arguments and the
//! readiness endpoint. This crate owns port reservation, working-directory /
//! DLL lookup setup, readiness polling, cancellation and process cleanup.

use std::ffi::OsString;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use reqwest::blocking::Client;
use uniinfer_core::{CancelToken, EngineError, Result};

pub struct ServerProcess {
    child: Mutex<Child>,
    pub base: String,
    pub client: Client,
}

impl ServerProcess {
    pub fn kill(&self) {
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

pub struct SpawnOptions<'a, F> {
    pub executable: &'a Path,
    pub readiness_path: &'a str,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
    pub cancel: &'a CancelToken,
    pub label: &'a str,
    pub args: F,
}

pub fn spawn<F>(options: SpawnOptions<'_, F>) -> Result<Arc<ServerProcess>>
where
    F: FnOnce(u16) -> Vec<OsString>,
{
    let port = free_port(options.label)?;
    let mut command = Command::new(options.executable);
    if let Some(parent) = options.executable.parent() {
        command.current_dir(parent);
        #[cfg(windows)]
        {
            // Official Windows archives keep their runtime DLLs next to the
            // server executable.
            let path = std::env::var_os("PATH").unwrap_or_default();
            let mut paths = vec![parent.to_path_buf()];
            paths.extend(std::env::split_paths(&path));
            if let Ok(joined) = std::env::join_paths(paths) {
                command.env("PATH", joined);
            }
        }
    }
    command.args((options.args)(port));
    let child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| {
            EngineError::LoadFailed(format!("spawn {}: {error}", options.executable.display()))
        })?;
    let process = Arc::new(ServerProcess {
        child: Mutex::new(child),
        base: format!("http://127.0.0.1:{port}"),
        client: Client::builder()
            .timeout(options.request_timeout)
            .build()
            .map_err(|error| EngineError::LoadFailed(format!("HTTP client: {error}")))?,
    });
    wait_ready(
        &process,
        options.readiness_path,
        options.startup_timeout,
        options.cancel,
        options.label,
    )?;
    tracing::info!(port, label = options.label, "official sidecar server ready");
    Ok(process)
}

fn free_port(label: &str) -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .map_err(|error| EngineError::LoadFailed(format!("reserve {label} port: {error}")))?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| EngineError::LoadFailed(format!("{label} port: {error}")))
}

fn wait_ready(
    process: &ServerProcess,
    path: &str,
    timeout: Duration,
    cancel: &CancelToken,
    label: &str,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        if cancel.is_cancelled() {
            process.kill();
            return Err(EngineError::Cancelled);
        }
        if !process.is_alive() {
            return Err(EngineError::LoadFailed(format!(
                "{label} exited during startup"
            )));
        }
        if let Ok(response) = process
            .client
            .get(format!("{}{path}", process.base))
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
