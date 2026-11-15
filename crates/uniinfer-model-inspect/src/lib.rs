//! Model layout detection and manifest generation.
//!
//! This is intentionally engine-neutral: it reads files and metadata, then
//! emits a `ModelManifest` draft. GUI/CLI can show the draft before writing.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;
use uniinfer_core::{
    AudioConfig, Modality, ModelFile, ModelFormat, ModelManifest, PlacementConfig, Quantization,
    TextConfig,
};

#[derive(Debug, Error)]
pub enum InspectError {
    #[error("model directory {0} does not exist")]
    NotFound(PathBuf),
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot parse {path}: {message}")]
    Config { path: PathBuf, message: String },
    #[error("no supported model files found in {0}")]
    Unsupported(PathBuf),
    #[error("invalid manifest: {0}")]
    Manifest(String),
}

pub type Result<T, E = InspectError> = std::result::Result<T, E>;

#[derive(Debug, Clone, Default)]
pub struct InspectOptions {
    pub id: Option<String>,
    pub architecture: Option<String>,
    pub format: Option<ModelFormat>,
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone)]
pub struct Inspection {
    pub root: PathBuf,
    pub manifest: ModelManifest,
    pub warnings: Vec<String>,
}

impl Inspection {
    pub fn to_toml(&self) -> Result<String, InspectError> {
        self.manifest
            .validate()
            .map_err(|error| InspectError::Manifest(error.to_string()))?;
        toml::to_string_pretty(&self.manifest)
            .map_err(|error| InspectError::Manifest(error.to_string()))
    }

    /// Write `manifest.toml` next to the model files.
    pub fn write_manifest(&self) -> Result<PathBuf, InspectError> {
        let path = self.root.join("manifest.toml");
        fs::write(&path, self.to_toml()?).map_err(|source| InspectError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(path)
    }
}

#[derive(Debug, Deserialize)]
struct ConfigJson {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    max_source_positions: Option<usize>,
    #[serde(default)]
    max_length: Option<usize>,
    #[serde(default)]
    torch_dtype: Option<String>,
    #[serde(default)]
    quantization_config: Option<QuantizationConfig>,
}

#[derive(Debug, Deserialize)]
struct QuantizationConfig {
    #[serde(default)]
    bits: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct PreprocessorConfig {
    #[serde(default)]
    sampling_rate: Option<u32>,
}

pub fn inspect(root: &Path, options: &InspectOptions) -> Result<Inspection> {
    if !root.is_dir() {
        return Err(InspectError::NotFound(root.to_path_buf()));
    }
    let mut files = Vec::new();
    collect_files(root, &mut files, 0)?;
    let by_name = |needle: &str| {
        files
            .iter()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name == needle)
            })
            .cloned()
    };
    let config_path = by_name("config.json");
    let config = config_path
        .as_deref()
        .map(read_json::<ConfigJson>)
        .transpose()?;
    let preprocessor = by_name("preprocessor_config.json")
        .as_deref()
        .map(read_json::<PreprocessorConfig>)
        .transpose()?;
    let tokenizer = by_name("tokenizer.json");
    let genai = by_name("genai_config.json");

    let onnx = files
        .iter()
        .filter(|path| extension(path) == Some("onnx"))
        .cloned()
        .collect::<Vec<_>>();
    let gguf = files
        .iter()
        .filter(|path| extension(path) == Some("gguf"))
        .cloned()
        .collect::<Vec<_>>();
    let safetensors = files
        .iter()
        .filter(|path| extension(path) == Some("safetensors"))
        .cloned()
        .collect::<Vec<_>>();
    let ggml = files
        .iter()
        .filter(|path| extension(path) == Some("bin"))
        .cloned()
        .collect::<Vec<_>>();

    let detected_format = options.format.or(if !onnx.is_empty() {
        Some(ModelFormat::Onnx)
    } else if !gguf.is_empty() {
        Some(ModelFormat::Gguf)
    } else if !safetensors.is_empty() {
        Some(ModelFormat::Safetensors)
    } else if !ggml.is_empty() {
        Some(ModelFormat::Ggml)
    } else {
        None
    });
    let format = detected_format.ok_or_else(|| InspectError::Unsupported(root.to_path_buf()))?;
    let architecture = options
        .architecture
        .clone()
        .or_else(|| config.as_ref().and_then(architecture_name))
        .or_else(|| {
            if format == ModelFormat::Ggml && looks_like_whisper(root) {
                Some("whisper".into())
            } else {
                None
            }
        });
    let quantization = options
        .quantization
        .or_else(|| detect_quantization(&files, config.as_ref()))
        .unwrap_or(Quantization::Fp32);
    let mut warnings = Vec::new();

    let (modalities, text, audio, selected) = match format {
        ModelFormat::Onnx => {
            let layout = onnx_layout(
                &onnx,
                OnnxInputs {
                    root,
                    config: config.as_ref(),
                    tokenizer: tokenizer.as_deref(),
                    config_path: config_path.as_deref(),
                    genai: genai.as_deref(),
                    architecture: architecture.as_deref(),
                    preprocessor: preprocessor.as_ref(),
                },
                &mut warnings,
            )?;
            (
                layout.modalities,
                layout.text,
                layout.audio,
                layout.selected,
            )
        }
        ModelFormat::Gguf => {
            let primary = choose_primary(&gguf, "model.gguf");
            let selected = with_metadata(primary, tokenizer.as_deref(), config_path.as_deref());
            let text = text_config(
                root,
                config.as_ref(),
                tokenizer.as_deref(),
                config_path.as_deref(),
            );
            (vec![Modality::Text], text, None, selected)
        }
        ModelFormat::Safetensors => {
            let primary = choose_primary(&safetensors, "model.safetensors");
            let selected = with_metadata(primary, tokenizer.as_deref(), config_path.as_deref());
            let text = text_config(
                root,
                config.as_ref(),
                tokenizer.as_deref(),
                config_path.as_deref(),
            );
            (vec![Modality::Text], text, None, selected)
        }
        ModelFormat::Ggml => {
            let primary = choose_primary(&ggml, "ggml-model.bin");
            let selected = with_metadata(primary, None, config_path.as_deref());
            let audio = AudioConfig {
                architecture: Some("whisper".into()),
                sample_rate: preprocessor
                    .as_ref()
                    .and_then(|config| config.sampling_rate)
                    .unwrap_or(16_000),
                channels: 1,
                ..AudioConfig::default()
            };
            (vec![Modality::Audio], None, Some(audio), selected)
        }
    };

    let context_len = context_len(config.as_ref(), &modalities);
    let id = options.id.clone().unwrap_or_else(|| model_id(root));
    let files = selected
        .into_iter()
        .map(|path| {
            Ok(ModelFile {
                path: relative(root, &path),
                sha256: sha256(&path)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let manifest = ModelManifest {
        schema_version: 1,
        id,
        format,
        quantization,
        modalities,
        streaming_capable: false,
        context_len,
        tokenizer: tokenizer.as_ref().map(|_| "hf".into()),
        min_vram: None,
        placement: PlacementConfig::default(),
        files,
        text,
        audio,
    };
    manifest
        .validate()
        .map_err(|error| InspectError::Manifest(error.to_string()))?;
    Ok(Inspection {
        root: root.to_path_buf(),
        manifest,
        warnings,
    })
}

struct OnnxInputs<'a> {
    root: &'a Path,
    config: Option<&'a ConfigJson>,
    tokenizer: Option<&'a Path>,
    config_path: Option<&'a Path>,
    genai: Option<&'a Path>,
    architecture: Option<&'a str>,
    preprocessor: Option<&'a PreprocessorConfig>,
}

struct OnnxLayout {
    modalities: Vec<Modality>,
    text: Option<TextConfig>,
    audio: Option<AudioConfig>,
    selected: Vec<PathBuf>,
}

fn onnx_layout(
    onnx: &[PathBuf],
    input: OnnxInputs<'_>,
    warnings: &mut Vec<String>,
) -> Result<OnnxLayout> {
    let OnnxInputs {
        root,
        config,
        tokenizer,
        config_path,
        genai,
        architecture,
        preprocessor,
    } = input;
    let encoder = onnx
        .iter()
        .find(|path| file_name(path).contains("encoder_model"))
        .cloned();
    let decoder = onnx
        .iter()
        .find(|path| {
            file_name(path).contains("decoder_model") && !file_name(path).contains("with_past")
        })
        .cloned();
    let decoder_with_past = onnx
        .iter()
        .find(|path| file_name(path).contains("decoder_with_past_model"))
        .cloned();
    if let (Some(encoder), Some(decoder)) = (encoder, decoder) {
        let tokenizer = tokenizer.ok_or_else(|| {
            InspectError::Manifest("ASR ONNX model requires tokenizer.json".into())
        })?;
        let config_path = config_path
            .ok_or_else(|| InspectError::Manifest("ASR ONNX model requires config.json".into()))?;
        let mut options = toml::Table::new();
        options.insert(
            "encoder_file".into(),
            toml::Value::String(relative_string(root, &encoder)),
        );
        options.insert(
            "decoder_file".into(),
            toml::Value::String(relative_string(root, &decoder)),
        );
        if let Some(path) = &decoder_with_past {
            options.insert(
                "decoder_with_past_file".into(),
                toml::Value::String(relative_string(root, path)),
            );
        }
        options.insert(
            "tokenizer_file".into(),
            toml::Value::String(relative_string(root, tokenizer)),
        );
        options.insert(
            "config_file".into(),
            toml::Value::String(relative_string(root, config_path)),
        );
        options.insert("max_tokens".into(), toml::Value::Integer(128));
        let audio = AudioConfig {
            architecture: Some(architecture.unwrap_or("whisper").into()),
            sample_rate: preprocessor
                .and_then(|config| config.sampling_rate)
                .unwrap_or(16_000),
            channels: 1,
            options: Some(toml::Value::Table(options)),
            ..AudioConfig::default()
        };
        let mut selected = vec![encoder, decoder];
        selected.extend(decoder_with_past);
        selected.extend(with_metadata(
            Vec::new(),
            Some(tokenizer),
            Some(config_path),
        ));
        return Ok(OnnxLayout {
            modalities: vec![Modality::Audio],
            text: None,
            audio: Some(audio),
            selected,
        });
    }

    let selected = choose_primary(onnx, "model.onnx");
    if selected.is_empty() {
        return Err(InspectError::Unsupported(root.to_path_buf()));
    }
    if tokenizer.is_none() {
        warnings.push("no tokenizer.json; text generation may not be runnable".into());
    }
    let text = text_config(root, config, tokenizer, config_path);
    let selected = with_metadata(selected, tokenizer, config_path);
    if genai.is_none() {
        warnings.push("no genai_config.json; assuming a standard ONNX graph".into());
    }
    Ok(OnnxLayout {
        modalities: vec![Modality::Text],
        text,
        audio: None,
        selected,
    })
}

fn text_config(
    root: &Path,
    config: Option<&ConfigJson>,
    tokenizer: Option<&Path>,
    config_path: Option<&Path>,
) -> Option<TextConfig> {
    let config = config?;
    let vocab_size = config.vocab_size?;
    Some(TextConfig {
        vocab_size,
        eos_token: config.eos_token_id,
        family: architecture_name(config),
        config_file: config_path.map(|path| relative(root, path)),
        tokenizer_file: tokenizer.map(|path| relative(root, path)),
    })
}

fn architecture_name(config: &ConfigJson) -> Option<String> {
    config
        .architectures
        .first()
        .cloned()
        .or_else(|| config.model_type.clone())
        .map(|name| {
            name.trim_end_matches("ForCausalLM")
                .trim_end_matches("ForConditionalGeneration")
                .to_ascii_lowercase()
        })
}

fn context_len(config: Option<&ConfigJson>, modalities: &[Modality]) -> usize {
    if modalities.contains(&Modality::Audio) {
        return config.and_then(|config| config.max_length).unwrap_or(448);
    }
    config
        .and_then(|config| {
            config
                .max_position_embeddings
                .or(config.max_source_positions)
        })
        .unwrap_or(2048)
}

fn detect_quantization(files: &[PathBuf], config: Option<&ConfigJson>) -> Option<Quantization> {
    let names = files
        .iter()
        .map(|path| path.to_string_lossy().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    for (needle, quantization) in [
        ("q4_k_m", Quantization::Q4KM),
        ("q5_k_m", Quantization::Q5KM),
        ("q6_k", Quantization::Q6K),
        ("q8_0", Quantization::Q8_0),
        ("q4_0", Quantization::Q4_0),
        ("int4", Quantization::Int4),
        ("int8", Quantization::Int8),
        ("fp32", Quantization::Fp32),
        ("fp16", Quantization::Fp16),
        ("bf16", Quantization::Bf16),
    ] {
        if names.contains(needle) {
            return Some(quantization);
        }
    }
    if let Some(bits) = config
        .and_then(|config| config.quantization_config.as_ref())
        .and_then(|config| config.bits)
    {
        return match bits {
            4 => Some(Quantization::Int4),
            8 => Some(Quantization::Int8),
            _ => None,
        };
    }
    config
        .and_then(|config| config.torch_dtype.as_deref())
        .and_then(|dtype| match dtype {
            "float32" => Some(Quantization::Fp32),
            "float16" => Some(Quantization::Fp16),
            "bfloat16" => Some(Quantization::Bf16),
            _ => None,
        })
}

fn choose_primary(files: &[PathBuf], preferred: &str) -> Vec<PathBuf> {
    if files.is_empty() {
        return Vec::new();
    }
    let mut files = files.to_vec();
    files.sort_by_key(|path| {
        let name = file_name(path);
        if name == preferred {
            0
        } else if name.contains("model") {
            1
        } else {
            2
        }
    });
    files
}

fn with_metadata(
    mut files: Vec<PathBuf>,
    tokenizer: Option<&Path>,
    config: Option<&Path>,
) -> Vec<PathBuf> {
    if let Some(tokenizer) = tokenizer {
        files.push(tokenizer.to_path_buf());
    }
    if let Some(config) = config {
        files.push(config.to_path_buf());
    }
    files
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let raw = fs::read_to_string(path).map_err(|source| InspectError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|error| InspectError::Config {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) -> Result<()> {
    if depth > 4 {
        return Ok(());
    }
    let entries = fs::read_dir(dir).map_err(|source| InspectError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| InspectError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let name = file_name(&path);
        if name.starts_with('.')
            || matches!(name.as_str(), "__pycache__" | "target" | "onnxruntime")
        {
            continue;
        }
        if path.is_dir() {
            collect_files(&path, out, depth + 1)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn model_id(root: &Path) -> String {
    let name = root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model");
    let mut id = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    if id.is_empty() {
        id = "model".into();
    }
    id
}

fn looks_like_whisper(root: &Path) -> bool {
    root.components()
        .filter_map(|component| component.as_os_str().to_str())
        .any(|component| component.to_ascii_lowercase().contains("whisper"))
}

fn extension(path: &Path) -> Option<&str> {
    path.extension().and_then(|extension| extension.to_str())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn relative(root: &Path, path: &Path) -> PathBuf {
    let relative = path
        .strip_prefix(root)
        .map_or_else(|_| path.to_path_buf(), Path::to_path_buf);
    let normalized = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    PathBuf::from(normalized)
}

fn relative_string(root: &Path, path: &Path) -> String {
    relative(root, path).to_string_lossy().into_owned()
}

fn sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|source| InspectError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(hex::encode(hasher.finalize()))
}

pub fn write_manifest(root: &Path, manifest: &ModelManifest) -> Result<PathBuf> {
    manifest
        .validate()
        .map_err(|error| InspectError::Manifest(error.to_string()))?;
    let path = root.join("manifest.toml");
    let raw = toml::to_string_pretty(manifest)
        .map_err(|error| InspectError::Manifest(error.to_string()))?;
    fs::write(&path, raw).map_err(|source| InspectError::Io {
        path: path.clone(),
        source,
    })?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_single_onnx_text_model() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("model.onnx"), b"onnx").expect("model");
        fs::write(dir.path().join("config.json"), r#"{"architectures":["Qwen3ForCausalLM"],"vocab_size":100,"eos_token_id":2,"max_position_embeddings":128}"#).expect("config");
        fs::write(dir.path().join("tokenizer.json"), b"{}").expect("tokenizer");
        let inspection = inspect(dir.path(), &InspectOptions::default()).expect("inspect");
        assert_eq!(inspection.manifest.format, ModelFormat::Onnx);
        assert_eq!(inspection.manifest.modalities, vec![Modality::Text]);
        assert_eq!(
            inspection.manifest.text.as_ref().expect("text").vocab_size,
            100
        );
        assert!(inspection.manifest.audio.is_none());
    }

    #[test]
    fn detects_onnx_encoder_decoder_asr() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::create_dir(dir.path().join("onnx")).expect("onnx dir");
        fs::write(dir.path().join("onnx/encoder_model.onnx"), b"encoder").expect("encoder");
        fs::write(dir.path().join("onnx/decoder_model.onnx"), b"decoder").expect("decoder");
        fs::write(dir.path().join("tokenizer.json"), b"{}").expect("tokenizer");
        fs::write(
            dir.path().join("config.json"),
            r#"{"architectures":["WhisperForConditionalGeneration"],"max_source_positions":1500}"#,
        )
        .expect("config");
        let inspection = inspect(dir.path(), &InspectOptions::default()).expect("inspect");
        assert_eq!(inspection.manifest.modalities, vec![Modality::Audio]);
        let audio = inspection.manifest.audio.expect("audio");
        assert_eq!(audio.architecture.as_deref(), Some("whisper"));
        assert!(audio.options.is_some());
    }
}
