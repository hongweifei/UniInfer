//! Adapter for the official `llama.cpp` `llama-server` executable.
//!
//! The server owns the GGUF model and tokenizer. UniInfer sends the rendered
//! chat prompt to `/completion` and streams OpenAI-style SSE chunks back into
//! the native decoder contract. Killing the child is the cancellation path.

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use uniinfer_core::{
    CancelToken, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine,
    Modality, ModelFormat, NativeFinish, NativeTextDecoder, NativeTextOutput, NativeTextRequest,
    Quantization, Residency, Result, Runtime, Tensor, TensorSpec,
};

const FORMATS: &[ModelFormat] = &[ModelFormat::Gguf];
const DEVICES: &[DeviceClass] = &[DeviceClass::Cpu, DeviceClass::Gpu];
const QUANTIZATIONS: &[Quantization] = &[
    Quantization::Fp32,
    Quantization::Fp16,
    Quantization::Bf16,
    Quantization::Q2K,
    Quantization::Q3KS,
    Quantization::Q3KM,
    Quantization::Q3KL,
    Quantization::Q4_0,
    Quantization::Q4_1,
    Quantization::Q4KS,
    Quantization::Q4KM,
    Quantization::Q5_0,
    Quantization::Q5_1,
    Quantization::Q5KS,
    Quantization::Q5KM,
    Quantization::Q6K,
    Quantization::Q8_0,
    Quantization::Q8K,
    Quantization::Q2KS,
    Quantization::Iq1S,
    Quantization::Iq1M,
    Quantization::Iq2Xxs,
    Quantization::Iq2Xs,
    Quantization::Iq2S,
    Quantization::Iq2M,
    Quantization::Iq3Xxs,
    Quantization::Iq3Xs,
    Quantization::Iq3S,
    Quantization::Iq3M,
    Quantization::Iq4Nl,
    Quantization::Iq4Xs,
    Quantization::Mxfp4,
    Quantization::Nvfp4,
    Quantization::Tq1_0,
    Quantization::Tq2_0,
    Quantization::Q1_0,
    Quantization::Q2_0,
];

pub struct LlamaServerEngine {
    executable: PathBuf,
    devices: Vec<DeviceClass>,
}

impl LlamaServerEngine {
    #[must_use]
    pub fn new(executable: PathBuf, devices: Vec<DeviceClass>) -> Self {
        Self {
            executable,
            devices: if devices.is_empty() {
                DEVICES.to_vec()
            } else {
                devices
            },
        }
    }
}

impl InferenceEngine for LlamaServerEngine {
    fn engine_name(&self) -> &'static str {
        "llama-server"
    }

    fn runtime(&self) -> Runtime {
        Runtime::Llama
    }

    fn implementations(&self) -> &[&'static str] {
        &["llama.text"]
    }

    fn is_transport(&self) -> bool {
        true
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        FORMATS
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Gguf => &[Modality::Text],
            _ => &[],
        }
    }

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization] {
        match format {
            ModelFormat::Gguf => QUANTIZATIONS,
            _ => &[],
        }
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        &self.devices
    }

    fn supports_layer_offload(&self) -> bool {
        true
    }

    fn supports_kv_cache_offload(&self) -> bool {
        true
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if !self.executable.is_file() {
            return Err(EngineError::LoadFailed(format!(
                "llama-server executable is missing: {}",
                self.executable.display()
            )));
        }
        let model = model_path(&spec)?;
        let port = free_port()?;
        let base = format!("http://127.0.0.1:{port}");
        let n_gpu_layers = offload_layers(&spec);
        let context = spec.manifest.context_len.max(1);
        let mut command = Command::new(&self.executable);
        if let Some(parent) = self.executable.parent() {
            command.current_dir(parent);
            #[cfg(windows)]
            {
                // The official Windows archive keeps DLLs next to the executable.
                let path = std::env::var_os("PATH").unwrap_or_default();
                let mut paths = vec![parent.to_path_buf()];
                paths.extend(std::env::split_paths(&path));
                if let Ok(joined) = std::env::join_paths(paths) {
                    command.env("PATH", joined);
                }
            }
        }
        let child = command
            .arg("--model")
            .arg(&model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg(context.to_string())
            .arg("--n-gpu-layers")
            .arg(n_gpu_layers.to_string())
            .arg("--parallel")
            .arg("1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| {
                EngineError::LoadFailed(format!("spawn {}: {error}", self.executable.display()))
            })?;
        let process = Arc::new(ServerProcess {
            child: Mutex::new(child),
            base,
            client: Client::builder()
                .timeout(Duration::from_secs(600))
                .build()
                .map_err(|error| EngineError::LoadFailed(format!("HTTP client: {error}")))?,
        });
        wait_ready(&process, &spec.cancel)?;
        let bytes = model_bytes(&spec);
        let residency = Residency {
            vram_bytes: if n_gpu_layers > 0 { bytes } else { 0 },
            host_bytes: if n_gpu_layers > 0 { 0 } else { bytes },
            disk_bytes: bytes,
            offload_layers: usize::try_from(n_gpu_layers).unwrap_or(usize::MAX),
            total_layers: None,
            estimated: true,
        };
        tracing::info!(
            model = %spec.manifest.id,
            port,
            n_gpu_layers,
            "official llama-server ready"
        );
        Ok(Box::new(LlamaServerSession {
            process,
            residency: Some(residency),
        }))
    }
}

struct ServerProcess {
    child: Mutex<Child>,
    base: String,
    client: Client,
}

impl ServerProcess {
    fn kill(&self) {
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

struct LlamaServerSession {
    process: Arc<ServerProcess>,
    residency: Option<Residency>,
}

impl EngineSession for LlamaServerSession {
    fn inputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn outputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn residency(&self) -> Option<Residency> {
        self.residency
    }

    fn run(&self, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        Err(EngineError::UnsupportedEngine(
            "llama-server uses native text decoding".into(),
        ))
    }

    fn native_decoder(&self) -> Option<&dyn NativeTextDecoder> {
        Some(self)
    }

    fn cancel(&self) {
        self.process.kill();
    }
}

impl Drop for LlamaServerSession {
    fn drop(&mut self) {
        self.process.kill();
    }
}

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

impl NativeTextDecoder for LlamaServerSession {
    fn generate_text(
        &self,
        request: NativeTextRequest<'_>,
        on_bytes: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<NativeTextOutput> {
        if request.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
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
        let mut call = self
            .process
            .client
            .post(format!("{}/completion", self.process.base))
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
        let mut reader = BufReader::new(response);
        let mut line = String::new();
        let mut bytes = Vec::new();
        let mut tokens = 0;
        let mut finish = NativeFinish::Eos;
        loop {
            if request.cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
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
            if data.is_empty() || data == "[DONE]" {
                if data == "[DONE]" {
                    break;
                }
                continue;
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
}

fn model_path(spec: &EngineLoadSpec) -> Result<PathBuf> {
    let mut candidates = spec
        .manifest
        .files
        .iter()
        .filter(|file| {
            file.path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .filter(|file| {
            !file
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.to_ascii_lowercase().contains("mmproj"))
        })
        .map(|file| spec.base_dir.join(&file.path))
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
        .into_iter()
        .max_by_key(|path| {
            std::fs::metadata(path)
                .map(|metadata| metadata.len())
                .unwrap_or(0)
        })
        .ok_or_else(|| EngineError::ManifestInvalid("no GGUF model file for llama-server".into()))
}

fn model_bytes(spec: &EngineLoadSpec) -> u64 {
    spec.manifest
        .files
        .iter()
        .filter(|file| {
            file.path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .filter_map(|file| std::fs::metadata(spec.base_dir.join(&file.path)).ok())
        .map(|metadata| metadata.len())
        .sum()
}

fn offload_layers(spec: &EngineLoadSpec) -> u32 {
    let gpu = spec
        .placement
        .devices
        .iter()
        .any(|device| device.class == DeviceClass::Gpu);
    if !gpu {
        return 0;
    }
    if spec.placement.auto_offload {
        return 999;
    }
    u32::try_from(spec.placement.offload_layers).unwrap_or(u32::MAX)
}

fn free_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .map_err(|error| EngineError::LoadFailed(format!("reserve llama-server port: {error}")))?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| EngineError::LoadFailed(format!("llama-server port: {error}")))
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
                "llama-server exited during startup".into(),
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

fn map_request_error(error: reqwest::Error) -> EngineError {
    if error.is_timeout() {
        EngineError::Timeout
    } else {
        EngineError::Other(format!("llama-server request: {error}"))
    }
}
