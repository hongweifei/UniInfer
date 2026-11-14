//! ONNX Runtime engine.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};

use half::f16;
use ort::environment::Environment;
use ort::session::{NoSelectedOutputs, RunOptions, Session, SessionInputValue};
use ort::value::{DynValue, Outlet, Tensor as OrtTensor, TensorElementType, ValueType};
use serde::Deserialize;
use uniinfer_core::{
    Device, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine, Modality,
    ModelFormat, Quantization, Result, Tensor, TensorData, TensorDtype, TensorSpec,
};

const FORMATS: &[ModelFormat] = &[ModelFormat::Onnx];
static DEVICES: LazyLock<Vec<DeviceClass>> = LazyLock::new(|| {
    let mut devices = vec![DeviceClass::Cpu];
    if cfg!(any(
        feature = "ep-cuda",
        feature = "ep-tensorrt",
        feature = "ep-rocm",
        feature = "ep-coreml",
        feature = "ep-directml"
    )) {
        devices.push(DeviceClass::Gpu);
    }
    if cfg!(any(
        feature = "ep-nnapi",
        feature = "ep-qnn",
        feature = "ep-openvino",
        feature = "ep-vitis",
        feature = "ep-cann"
    )) {
        devices.push(DeviceClass::Npu);
    }
    devices
});
const MODALITIES: &[Modality] = &[
    Modality::Text,
    Modality::Audio,
    Modality::Image,
    Modality::Video,
];
const QUANTIZATIONS: &[Quantization] = &[
    Quantization::Fp32,
    Quantization::Fp16,
    Quantization::Int8,
    Quantization::Int4,
];

static ENVIRONMENT: LazyLock<std::result::Result<Environment, String>> =
    LazyLock::new(build_environment);

fn build_environment() -> std::result::Result<Environment, String> {
    #[allow(unused_mut)]
    let mut providers: Vec<ort::ep::ExecutionProviderDispatch> = Vec::new();
    // NPU providers are registered first so ORT prefers them when available.
    #[cfg(feature = "ep-qnn")]
    providers.push(ort::ep::QNN::default().build());
    #[cfg(feature = "ep-vitis")]
    providers.push(ort::ep::Vitis::default().build());
    #[cfg(feature = "ep-nnapi")]
    providers.push(ort::ep::NNAPI::default().build());
    #[cfg(feature = "ep-cann")]
    providers.push(ort::ep::CANN::default().build());
    #[cfg(feature = "ep-coreml")]
    providers.push(ort::ep::CoreML::default().build());
    #[cfg(feature = "ep-openvino")]
    providers.push(ort::ep::OpenVINO::default().build());
    #[cfg(feature = "ep-tensorrt")]
    providers.push(ort::ep::TensorRT::default().build());
    #[cfg(feature = "ep-cuda")]
    providers.push(ort::ep::CUDA::default().build());
    #[cfg(feature = "ep-rocm")]
    providers.push(ort::ep::ROCm::default().build());
    #[cfg(feature = "ep-directml")]
    providers.push(ort::ep::DirectML::default().build());

    ort::init()
        .with_name("uniinfer")
        .with_execution_providers(providers)
        .build()
        .map_err(|error| error.to_string())
}

/// ONNX Runtime engine with the configured execution providers.
pub struct OnnxEngine {
    environment: Environment,
}

impl OnnxEngine {
    pub fn new() -> Result<Self> {
        let environment = ENVIRONMENT
            .as_ref()
            .map_err(|error| EngineError::LoadFailed(format!("ort environment: {error}")))?
            .clone();
        Ok(Self { environment })
    }

    /// Enumerate hardware devices visible to the configured execution providers.
    #[must_use]
    pub fn probe_devices(&self) -> Vec<Device> {
        self.environment
            .devices()
            .filter_map(|device| {
                let hardware = device.hardware_device();
                let class = match hardware.ty() {
                    ort::memory::DeviceType::CPU => DeviceClass::Cpu,
                    ort::memory::DeviceType::GPU => DeviceClass::Gpu,
                    ort::memory::DeviceType::NPU => DeviceClass::Npu,
                    _ => return None,
                };
                let backend = device.ep().unwrap_or("unknown").to_owned();
                if !ep_enabled(&backend) {
                    return None;
                }
                Some(Device {
                    id: format!("{class}:{backend}:{}", hardware.id()),
                    class,
                    backend,
                    memory_bytes: 0,
                })
            })
            .collect()
    }
}

impl InferenceEngine for OnnxEngine {
    fn engine_name(&self) -> &'static str {
        "onnx"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        FORMATS
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Onnx => MODALITIES,
            _ => &[],
        }
    }

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization] {
        match format {
            ModelFormat::Onnx => QUANTIZATIONS,
            _ => &[],
        }
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        &DEVICES
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if spec.manifest.format != ModelFormat::Onnx {
            return Err(EngineError::UnsupportedFormat(spec.manifest.format));
        }
        if !QUANTIZATIONS.contains(&spec.manifest.quantization) {
            return Err(EngineError::UnsupportedQuantization(
                spec.manifest.quantization,
            ));
        }
        if spec
            .placement
            .devices
            .iter()
            .any(|device| !DEVICES.contains(&device.class))
        {
            return Err(EngineError::PlacementUnsatisfiable(format!(
                "onnx engine has no device for {:?}",
                spec.placement.devices
            )));
        }
        let class = spec
            .placement
            .devices
            .first()
            .map(|device| device.class)
            .ok_or_else(|| EngineError::PlacementUnsatisfiable("empty placement".into()))?;
        spec.manifest.verify_files(&spec.base_dir)?;

        let model_file = spec
            .manifest
            .files
            .first()
            .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?;
        let model_path = spec.base_dir.join(&model_file.path);

        let builder = Session::builder(&self.environment)
            .map_err(|error| EngineError::LoadFailed(format!("session builder: {error}")))?;
        // Honor a CPU placement even when GPU EPs are registered in the
        // process-wide ORT environment.
        let mut builder = if class == DeviceClass::Cpu {
            builder
                .with_no_environment_execution_providers()
                .map_err(|error| {
                    EngineError::LoadFailed(format!("cpu-only session options: {error}"))
                })?
        } else {
            builder
        };
        let session = builder.commit_from_file(&model_path).map_err(|error| {
            EngineError::LoadFailed(format!("cannot load {}: {error}", model_path.display()))
        })?;

        let genai_config = load_genai_config(&spec.base_dir)?;
        let (inputs, outputs, genai) = match genai_config {
            Some(config) => {
                let genai = GenaiSpec::from_config(&config, &session)?;
                let inputs = genai.input_specs(&session)?;
                let outputs = genai.output_specs(&session)?;
                (inputs, outputs, Some(genai))
            }
            None => (
                convert_specs(session.inputs(), "input")?,
                convert_specs(session.outputs(), "output")?,
                None,
            ),
        };

        tracing::debug!(
            model = %spec.manifest.id,
            path = %model_path.display(),
            inputs = inputs.len(),
            outputs = outputs.len(),
            genai = genai.is_some(),
            "onnx session loaded"
        );

        Ok(Box::new(OnnxSession {
            state: Mutex::new(OnnxState {
                session,
                genai: genai.map(GenaiState::new),
            }),
            inputs,
            outputs,
            cancelled: AtomicBool::new(false),
            active: Mutex::new(None),
        }))
    }
}

fn ep_enabled(ep: &str) -> bool {
    let ep = ep.to_ascii_lowercase();
    if ep.contains("cpu") {
        return true;
    }
    (cfg!(feature = "ep-cuda") && ep.contains("cuda"))
        || (cfg!(feature = "ep-tensorrt") && ep.contains("tensorrt"))
        || (cfg!(feature = "ep-rocm") && ep.contains("rocm"))
        || (cfg!(feature = "ep-coreml") && ep.contains("coreml"))
        || (cfg!(feature = "ep-directml") && (ep.contains("dml") || ep.contains("directml")))
        || (cfg!(feature = "ep-qnn") && ep.contains("qnn"))
        || (cfg!(feature = "ep-nnapi") && ep.contains("nnapi"))
        || (cfg!(feature = "ep-openvino") && ep.contains("openvino"))
        || (cfg!(feature = "ep-vitis") && ep.contains("vitis"))
        || (cfg!(feature = "ep-cann") && ep.contains("cann"))
}

fn convert_specs(outlets: &[Outlet], kind: &str) -> Result<Vec<TensorSpec>> {
    outlets
        .iter()
        .map(|outlet| convert_spec(outlet, kind))
        .collect()
}

fn convert_spec(outlet: &Outlet, kind: &str) -> Result<TensorSpec> {
    let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
        return Err(EngineError::LoadFailed(format!(
            "{kind} {} is not a tensor",
            outlet.name()
        )));
    };
    let dtype = match *ty {
        TensorElementType::Float32 => TensorDtype::F32,
        TensorElementType::Int64 => TensorDtype::I64,
        TensorElementType::Uint8 => TensorDtype::U8,
        other => {
            return Err(EngineError::LoadFailed(format!(
                "{kind} {} has unsupported dtype {other:?}",
                outlet.name()
            )));
        }
    };
    let shape = shape
        .iter()
        .map(|dim| if *dim >= 0 { Some(*dim as usize) } else { None })
        .collect();
    Ok(TensorSpec {
        name: outlet.name().to_owned(),
        dtype,
        shape,
    })
}

#[derive(Debug, Deserialize)]
struct GenaiConfig {
    model: GenaiModel,
}

#[derive(Debug, Deserialize)]
struct GenaiModel {
    decoder: GenaiDecoder,
}

#[derive(Debug, Deserialize)]
struct GenaiDecoder {
    inputs: GenaiInputs,
    outputs: GenaiOutputs,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_size: usize,
}

#[derive(Debug, Deserialize)]
struct GenaiInputs {
    input_ids: String,
    attention_mask: String,
    past_key_names: String,
    past_value_names: String,
}

#[derive(Debug, Deserialize)]
struct GenaiOutputs {
    logits: String,
    present_key_names: String,
    present_value_names: String,
}

#[derive(Debug, Clone)]
struct GenaiSpec {
    input_ids: String,
    attention_mask: String,
    logits: String,
    past_order: Vec<String>,
    present_order: Vec<String>,
    kv_heads: usize,
    head_size: usize,
    past_dtype: TensorElementType,
}

impl GenaiSpec {
    fn from_config(config: &GenaiConfig, session: &Session) -> Result<Self> {
        let decoder = &config.model.decoder;
        if decoder.num_hidden_layers == 0
            || decoder.num_key_value_heads == 0
            || decoder.head_size == 0
        {
            return Err(EngineError::ManifestInvalid(
                "genai_config has zero layers/heads/head_size".into(),
            ));
        }
        let names = |pattern: &str| {
            (0..decoder.num_hidden_layers)
                .map(|layer| pattern.replace("%d", &layer.to_string()))
                .collect::<Vec<_>>()
        };
        let past_keys = names(&decoder.inputs.past_key_names);
        let past_values = names(&decoder.inputs.past_value_names);
        let present_keys = names(&decoder.outputs.present_key_names);
        let present_values = names(&decoder.outputs.present_value_names);
        let past_dtype = session
            .inputs()
            .iter()
            .find(|outlet| outlet.name() == past_keys[0])
            .and_then(|outlet| match outlet.dtype() {
                ValueType::Tensor { ty, .. } => Some(*ty),
                _ => None,
            })
            .ok_or_else(|| {
                EngineError::ManifestInvalid(format!(
                    "genai past input {} is missing",
                    past_keys[0]
                ))
            })?;
        if !matches!(
            past_dtype,
            TensorElementType::Float32 | TensorElementType::Float16
        ) {
            return Err(EngineError::LoadFailed(format!(
                "unsupported genai KV dtype {past_dtype:?}"
            )));
        }
        let past_order = session
            .inputs()
            .iter()
            .map(|outlet| outlet.name().to_owned())
            .filter(|name| past_keys.contains(name) || past_values.contains(name))
            .collect::<Vec<_>>();
        let present_order = session
            .outputs()
            .iter()
            .map(|outlet| outlet.name().to_owned())
            .filter(|name| present_keys.contains(name) || present_values.contains(name))
            .collect::<Vec<_>>();
        if past_order.len() != decoder.num_hidden_layers * 2
            || present_order.len() != decoder.num_hidden_layers * 2
        {
            return Err(EngineError::ManifestInvalid(
                "genai KV input/output names do not match layer count".into(),
            ));
        }
        Ok(Self {
            input_ids: decoder.inputs.input_ids.clone(),
            attention_mask: decoder.inputs.attention_mask.clone(),
            logits: decoder.outputs.logits.clone(),
            past_order,
            present_order,
            kv_heads: decoder.num_key_value_heads,
            head_size: decoder.head_size,
            past_dtype,
        })
    }

    fn input_specs(&self, session: &Session) -> Result<Vec<TensorSpec>> {
        let wanted = [&self.input_ids, &self.attention_mask];
        let filtered = session
            .inputs()
            .iter()
            .filter(|outlet| wanted.contains(&&outlet.name().to_owned()))
            .map(|outlet| convert_spec(outlet, "input"))
            .collect::<Result<Vec<_>>>()?;
        if filtered.len() != 2 {
            return Err(EngineError::ManifestInvalid(
                "genai input_ids/attention_mask are missing".into(),
            ));
        }
        Ok(filtered)
    }

    fn output_specs(&self, session: &Session) -> Result<Vec<TensorSpec>> {
        let filtered = session
            .outputs()
            .iter()
            .filter(|outlet| outlet.name() == self.logits)
            .map(|outlet| convert_spec(outlet, "output"))
            .collect::<Result<Vec<_>>>()?;
        if filtered.len() != 1 {
            return Err(EngineError::ManifestInvalid(
                "genai logits output is missing".into(),
            ));
        }
        Ok(filtered)
    }
}

struct GenaiState {
    spec: GenaiSpec,
    past: Vec<DynValue>,
    cached_ids: Vec<i64>,
}

impl GenaiState {
    fn new(spec: GenaiSpec) -> Self {
        Self {
            spec,
            past: Vec::new(),
            cached_ids: Vec::new(),
        }
    }
}

struct OnnxState {
    session: Session,
    genai: Option<GenaiState>,
}

fn load_genai_config(base_dir: &Path) -> Result<Option<GenaiConfig>> {
    let path = base_dir.join("genai_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path).map_err(|error| {
        EngineError::ManifestInvalid(format!("cannot read {}: {error}", path.display()))
    })?;
    let config = serde_json::from_str(&raw)
        .map_err(|error| EngineError::ManifestInvalid(format!("{}: {error}", path.display())))?;
    Ok(Some(config))
}

struct OnnxSession {
    state: Mutex<OnnxState>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    cancelled: AtomicBool,
    active: Mutex<Option<std::sync::Arc<RunOptions<NoSelectedOutputs>>>>,
}

struct ActiveRun<'a> {
    slot: &'a Mutex<Option<std::sync::Arc<RunOptions<NoSelectedOutputs>>>>,
    options: std::sync::Arc<RunOptions<NoSelectedOutputs>>,
}

impl Drop for ActiveRun<'_> {
    fn drop(&mut self) {
        if let Ok(mut slot) = self.slot.lock() {
            *slot = None;
        }
    }
}

impl OnnxSession {
    fn begin_run(&self) -> Result<ActiveRun<'_>> {
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let options = std::sync::Arc::new(
            RunOptions::new()
                .map_err(|error| EngineError::Other(format!("onnx run options: {error}")))?,
        );
        *self
            .active
            .lock()
            .map_err(|_| EngineError::Other("run options lock poisoned".into()))? =
            Some(std::sync::Arc::clone(&options));
        Ok(ActiveRun {
            slot: &self.active,
            options,
        })
    }
}

impl EngineSession for OnnxSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let active = self.begin_run()?;

        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("session lock poisoned".into()))?;
        if state.genai.is_some() {
            return run_genai(&mut state, &inputs, &active.options, &self.cancelled);
        }

        let mut values: Vec<(String, DynValue)> = Vec::with_capacity(inputs.len());
        for input in inputs {
            if !self.inputs.iter().any(|spec| spec.name == input.name) {
                return Err(EngineError::Other(format!(
                    "unknown input tensor {}",
                    input.name
                )));
            }
            values.push((input.name.clone(), to_ort(input)?));
        }

        for spec in &self.inputs {
            if !values.iter().any(|(name, _)| name == &spec.name) {
                return Err(EngineError::Other(format!(
                    "missing input tensor {}",
                    spec.name
                )));
            }
        }

        let OnnxState { session, .. } = &mut *state;
        let outputs = session
            .run_with_options(values, &active.options)
            .map_err(|error| {
                if self.cancelled.load(Ordering::SeqCst) {
                    EngineError::Cancelled
                } else {
                    EngineError::Other(format!("onnx run: {error}"))
                }
            })?;

        let mut result = Vec::with_capacity(self.outputs.len());
        for spec in &self.outputs {
            let value = &outputs[spec.name.as_str()];
            result.push(from_ort(spec, value)?);
        }
        Ok(result)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        if let Ok(slot) = self.active.lock()
            && let Some(options) = slot.as_ref()
        {
            let _ = options.terminate();
        }
    }
}

fn run_genai(
    state: &mut OnnxState,
    inputs: &[Tensor],
    options: &RunOptions<NoSelectedOutputs>,
    cancelled: &AtomicBool,
) -> Result<Vec<Tensor>> {
    let mut genai = state
        .genai
        .take()
        .ok_or_else(|| EngineError::Other("genai state missing".into()))?;
    let result = genai.forward(&mut state.session, inputs, options, cancelled);
    state.genai = Some(genai);
    result
}

impl GenaiState {
    fn forward(
        &mut self,
        session: &mut Session,
        inputs: &[Tensor],
        options: &RunOptions<NoSelectedOutputs>,
        cancelled: &AtomicBool,
    ) -> Result<Vec<Tensor>> {
        let ids = inputs
            .iter()
            .find(|tensor| tensor.name == self.spec.input_ids)
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) => Some(values.clone()),
                _ => None,
            })
            .ok_or_else(|| EngineError::Other("genai input_ids missing".into()))?;
        if ids.is_empty() {
            return Err(EngineError::Other("genai input_ids is empty".into()));
        }
        let mask = inputs
            .iter()
            .find(|tensor| tensor.name == self.spec.attention_mask)
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) if values.len() == ids.len() => Some(values.clone()),
                _ => None,
            })
            .unwrap_or_else(|| vec![1; ids.len()]);

        let reuse = !self.cached_ids.is_empty()
            && ids.len() > self.cached_ids.len()
            && ids.starts_with(&self.cached_ids);
        let mut past = if reuse {
            std::mem::take(&mut self.past)
        } else {
            Vec::new()
        };
        if past.is_empty() {
            past = zero_past(&self.spec)?;
        }
        let delta_start = if reuse { self.cached_ids.len() } else { 0 };
        let delta = &ids[delta_start..];
        if delta.is_empty() {
            return Err(EngineError::Other("genai input has no new tokens".into()));
        }

        let ids_value = to_ort(Tensor::new(
            self.spec.input_ids.clone(),
            vec![1, delta.len()],
            TensorData::I64(delta.to_vec()),
        )?)?;
        let mask_value = to_ort(Tensor::new(
            self.spec.attention_mask.clone(),
            vec![1, mask.len()],
            TensorData::I64(mask),
        )?)?;

        let mut values: Vec<(String, SessionInputValue)> = Vec::with_capacity(2 + past.len());
        values.push((
            self.spec.input_ids.clone(),
            SessionInputValue::from(&ids_value),
        ));
        values.push((
            self.spec.attention_mask.clone(),
            SessionInputValue::from(&mask_value),
        ));
        for (name, value) in self.spec.past_order.iter().zip(past.iter()) {
            values.push((name.clone(), SessionInputValue::from(value)));
        }

        let mut outputs = session.run_with_options(values, options).map_err(|error| {
            if cancelled.load(Ordering::SeqCst) {
                EngineError::Cancelled
            } else {
                EngineError::Other(format!("onnx genai run: {error}"))
            }
        })?;
        let logits = from_ort(
            &TensorSpec {
                name: self.spec.logits.clone(),
                dtype: TensorDtype::F32,
                shape: Vec::new(),
            },
            outputs
                .get(self.spec.logits.as_str())
                .ok_or_else(|| EngineError::Other("genai logits missing".into()))?,
        )?;
        let mut presents = Vec::with_capacity(self.spec.present_order.len());
        for name in &self.spec.present_order {
            presents.push(
                outputs
                    .remove(name)
                    .ok_or_else(|| EngineError::Other(format!("genai output {name} missing")))?,
            );
        }
        drop(outputs);
        drop(past);
        self.past = presents;
        self.cached_ids = ids;
        Ok(vec![logits])
    }
}

fn zero_past(spec: &GenaiSpec) -> Result<Vec<DynValue>> {
    let mut values = Vec::with_capacity(spec.past_order.len());
    for _ in &spec.past_order {
        let shape = vec![1usize, spec.kv_heads, 0, spec.head_size];
        let value =
            match spec.past_dtype {
                TensorElementType::Float32 => OrtTensor::from_array((shape, Vec::<f32>::new()))
                    .map(ort::value::Value::into_dyn),
                TensorElementType::Float16 => OrtTensor::from_array((shape, Vec::<f16>::new()))
                    .map(ort::value::Value::into_dyn),
                other => {
                    return Err(EngineError::LoadFailed(format!(
                        "unsupported genai KV dtype {other:?}"
                    )));
                }
            }
            .map_err(|error| EngineError::Other(format!("genai zero past: {error}")))?;
        values.push(value);
    }
    Ok(values)
}

fn to_ort(tensor: Tensor) -> Result<DynValue> {
    let shape = tensor.shape.clone();
    let value = match tensor.data {
        TensorData::F32(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
        TensorData::I64(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
        TensorData::U8(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
    }
    .map_err(|error| EngineError::Other(format!("tensor {}: {error}", tensor.name)))?;
    Ok(value)
}

fn from_ort(spec: &TensorSpec, value: &DynValue) -> Result<Tensor> {
    match spec.dtype {
        TensorDtype::F32 => {
            let (shape, slice) = value
                .try_extract_tensor::<f32>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::F32(slice.to_vec()))
        }
        TensorDtype::I64 => {
            let (shape, slice) = value
                .try_extract_tensor::<i64>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::I64(slice.to_vec()))
        }
        TensorDtype::U8 => {
            let (shape, slice) = value
                .try_extract_tensor::<u8>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::U8(slice.to_vec()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_genai_config_from_model_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("genai_config.json"),
            r#"{
                "model": {
                    "vocab_size": 151936,
                    "eos_token_id": 151645,
                    "decoder": {
                        "inputs": {
                            "input_ids": "input_ids",
                            "attention_mask": "attention_mask",
                            "past_key_names": "past_key_values.%d.key",
                            "past_value_names": "past_key_values.%d.value"
                        },
                        "outputs": {
                            "logits": "logits",
                            "present_key_names": "present.%d.key",
                            "present_value_names": "present.%d.value"
                        },
                        "num_hidden_layers": 28,
                        "num_key_value_heads": 8,
                        "head_size": 128
                    }
                }
            }"#,
        )
        .expect("write config");
        let config = load_genai_config(dir.path())
            .expect("parse config")
            .expect("config present");
        assert_eq!(config.model.decoder.num_hidden_layers, 28);
        assert_eq!(config.model.decoder.num_key_value_heads, 8);
        assert_eq!(config.model.decoder.head_size, 128);
    }
}
