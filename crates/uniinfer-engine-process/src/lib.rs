//! Text inference in a child process.
//!
//! The parent owns the process lifecycle; the child owns the native engine and
//! its model. Killing the child is therefore a hard cancellation path even
//! when the underlying ONNX/Candle loader has no cancellation callback.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, Quantization, Result, Tensor, TensorData, TensorDtype, TensorSpec,
};

#[derive(Debug, Serialize, Deserialize)]
pub enum WireData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WireTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: WireData,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WireSpec {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<Option<usize>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum WorkerRequest {
    Run { inputs: Vec<WireTensor> },
    Shutdown,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum WorkerResponse {
    Ready {
        inputs: Vec<WireSpec>,
        outputs: Vec<WireSpec>,
    },
    Outputs {
        outputs: Vec<WireTensor>,
    },
    Error {
        code: String,
        message: String,
    },
}

impl WireTensor {
    pub fn from_tensor(tensor: Tensor) -> Self {
        let data = match tensor.data {
            TensorData::F32(values) => WireData::F32(values),
            TensorData::F16(values) => {
                WireData::F16(values.into_iter().map(|v| v.to_bits()).collect())
            }
            TensorData::I64(values) => WireData::I64(values),
            TensorData::U8(values) => WireData::U8(values),
        };
        Self {
            name: tensor.name,
            shape: tensor.shape,
            data,
        }
    }

    pub fn into_tensor(self) -> Result<Tensor> {
        let data = match self.data {
            WireData::F32(values) => TensorData::F32(values),
            WireData::F16(values) => {
                TensorData::F16(values.into_iter().map(half::f16::from_bits).collect())
            }
            WireData::I64(values) => TensorData::I64(values),
            WireData::U8(values) => TensorData::U8(values),
        };
        Tensor::new(self.name, self.shape, data)
    }
}

impl From<&TensorSpec> for WireSpec {
    fn from(spec: &TensorSpec) -> Self {
        Self {
            name: spec.name.clone(),
            dtype: spec.dtype,
            shape: spec.shape.clone(),
        }
    }
}

impl WireSpec {
    pub fn into_spec(self) -> TensorSpec {
        TensorSpec {
            name: self.name,
            dtype: self.dtype,
            shape: self.shape,
        }
    }
}

pub struct ProcessEngine {
    worker: PathBuf,
}

impl ProcessEngine {
    pub fn new() -> Self {
        let worker = std::env::var_os("UNIINFER_WORKER")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::current_exe()
                    .ok()
                    .map(|path| path.with_file_name(worker_name()))
            })
            .unwrap_or_else(|| PathBuf::from(worker_name()));
        Self { worker }
    }
}

impl ProcessEngine {
    #[must_use]
    pub fn with_worker(worker: PathBuf) -> Self {
        Self { worker }
    }
}

impl Default for ProcessEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn worker_name() -> &'static str {
    if cfg!(windows) {
        "uniinfer-worker.exe"
    } else {
        "uniinfer-worker"
    }
}

impl InferenceEngine for ProcessEngine {
    fn engine_name(&self) -> &'static str {
        "process"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[
            ModelFormat::Onnx,
            ModelFormat::Gguf,
            ModelFormat::Safetensors,
        ]
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Onnx => &[Modality::Text, Modality::Audio],
            ModelFormat::Gguf | ModelFormat::Safetensors => &[Modality::Text],
            _ => &[],
        }
    }

    fn supported_quantizations(&self, _format: ModelFormat) -> &[Quantization] {
        &[
            Quantization::Fp32,
            Quantization::Fp16,
            Quantization::Bf16,
            Quantization::Int4,
            Quantization::Int8,
            Quantization::Q4_0,
            Quantization::Q4_1,
            Quantization::Q5_0,
            Quantization::Q5_1,
            Quantization::Q8_0,
            Quantization::Q2K,
            Quantization::Q3KS,
            Quantization::Q3KM,
            Quantization::Q3KL,
            Quantization::Q4KS,
            Quantization::Q4KM,
            Quantization::Q5KS,
            Quantization::Q5KM,
            Quantization::Q6K,
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
        ]
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        &[DeviceClass::Cpu]
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let manifest_path = spec.manifest_path.as_ref().ok_or_else(|| {
            EngineError::LoadFailed("process isolation requires a manifest path".into())
        })?;
        let mut child = Command::new(&self.worker)
            .arg(manifest_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                EngineError::LoadFailed(format!("spawn {}: {error}", self.worker.display()))
            })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| EngineError::LoadFailed("worker stdout unavailable".into()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| EngineError::LoadFailed("worker stdin unavailable".into()))?;
        let (ready_tx, ready_rx) = mpsc::channel();
        let (line_tx, line_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) > 0 {
                let _ = ready_tx.send(line.clone());
            }
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if line_tx.send(line.clone()).is_err() {
                            break;
                        }
                    }
                }
            }
        });

        let ready = loop {
            match ready_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(line) => break line,
                Err(RecvTimeoutError::Timeout) => {
                    if spec.cancel.is_cancelled() {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(EngineError::Cancelled);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(EngineError::LoadFailed("worker exited before ready".into()));
                }
            }
        };
        let ready: WorkerResponse = serde_json::from_str(&ready).map_err(|error| {
            let _ = child.kill();
            let _ = child.wait();
            EngineError::LoadFailed(format!("worker ready: {error}"))
        })?;
        let WorkerResponse::Ready { inputs, outputs } = ready else {
            let _ = child.kill();
            return Err(EngineError::LoadFailed(
                "worker sent an invalid ready".into(),
            ));
        };
        Ok(Box::new(ProcessSession {
            io: std::sync::Mutex::new(ProcessIo {
                child,
                stdin,
                reply: line_rx,
            }),
            inputs: inputs.into_iter().map(WireSpec::into_spec).collect(),
            outputs: outputs.into_iter().map(WireSpec::into_spec).collect(),
        }))
    }
}

struct ProcessIo {
    child: Child,
    stdin: ChildStdin,
    reply: Receiver<String>,
}

struct ProcessSession {
    io: std::sync::Mutex<ProcessIo>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl ProcessSession {
    fn exchange(&self, request: &WorkerRequest, timeout: Option<Duration>) -> Result<Vec<Tensor>> {
        let mut io = self
            .io
            .lock()
            .map_err(|_| EngineError::Other("process session lock poisoned".into()))?;
        let line = serde_json::to_string(request)
            .map_err(|error| EngineError::Other(format!("worker request: {error}")))?;
        writeln!(io.stdin, "{line}")
            .and_then(|_| io.stdin.flush())
            .map_err(|error| EngineError::LoadFailed(format!("worker stdin: {error}")))?;
        let line = match timeout {
            Some(timeout) => match io.reply.recv_timeout(timeout) {
                Ok(line) => line,
                Err(_) => {
                    let _ = io.child.kill();
                    let _ = io.child.wait();
                    return Err(EngineError::Timeout);
                }
            },
            None => io
                .reply
                .recv()
                .map_err(|_| EngineError::Other("worker exited".into()))?,
        };
        match serde_json::from_str::<WorkerResponse>(&line)
            .map_err(|error| EngineError::Other(format!("worker reply: {error}")))?
        {
            WorkerResponse::Outputs { outputs } => {
                outputs.into_iter().map(WireTensor::into_tensor).collect()
            }
            WorkerResponse::Error { code, message } => {
                Err(EngineError::Other(format!("worker {code}: {message}")))
            }
            WorkerResponse::Ready { .. } => Err(EngineError::Other("unexpected ready".into())),
        }
    }
}

impl EngineSession for ProcessSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        self.exchange(
            &WorkerRequest::Run {
                inputs: inputs.into_iter().map(WireTensor::from_tensor).collect(),
            },
            None,
        )
    }

    fn run_with_deadline(
        &self,
        inputs: Vec<Tensor>,
        deadline: Option<Instant>,
    ) -> Result<Vec<Tensor>> {
        let timeout = deadline.map(|deadline| {
            deadline
                .saturating_duration_since(Instant::now())
                .max(Duration::from_millis(1))
        });
        self.exchange(
            &WorkerRequest::Run {
                inputs: inputs.into_iter().map(WireTensor::from_tensor).collect(),
            },
            timeout,
        )
    }

    fn cancel(&self) {
        if let Ok(mut io) = self.io.lock() {
            let _ = io.child.kill();
            let _ = io.child.wait();
        }
    }
}

impl Drop for ProcessSession {
    fn drop(&mut self) {
        if let Ok(mut io) = self.io.lock() {
            let _ = io.child.kill();
            let _ = io.child.wait();
        }
    }
}
