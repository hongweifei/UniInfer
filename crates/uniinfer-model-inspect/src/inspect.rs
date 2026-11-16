use std::path::Path;

use uniinfer_core::{
    AudioConfig, Capability, Modality, Mode, ModelFile, ModelFormat, ModelManifest, Operation,
    PlacementConfig, Quantization, Runtime,
};

use crate::error::{InspectError, Result};
use crate::files::{
    choose_primary, collect_files, extension, file_name, looks_like_whisper, model_id, relative,
    sha256, with_metadata,
};
use crate::gguf::read as read_gguf;
use crate::inspection::Inspection;
use crate::metadata::{
    ConfigJson, PreprocessorConfig, architecture_name, context_len, detect_quantization, read_json,
};
use crate::onnx::{OnnxInputs, onnx_layout};

#[derive(Debug, Clone, Default)]
pub struct InspectOptions {
    pub id: Option<String>,
    pub architecture: Option<String>,
    pub format: Option<ModelFormat>,
    pub quantization: Option<Quantization>,
}

pub fn inspect(root: &Path, options: &InspectOptions) -> Result<Inspection> {
    if !root.is_dir() {
        return Err(InspectError::NotFound(root.to_path_buf()));
    }
    let mut files = Vec::new();
    collect_files(root, &mut files, 0)?;
    let config_path = files
        .iter()
        .find(|path| file_name(path) == "config.json")
        .cloned();
    let config = config_path
        .as_deref()
        .map(read_json::<ConfigJson>)
        .transpose()?;
    let preprocessor = files
        .iter()
        .find(|path| file_name(path) == "preprocessor_config.json")
        .cloned()
        .map(|path| read_json::<PreprocessorConfig>(&path))
        .transpose()?;
    let tokenizer = files
        .iter()
        .find(|path| file_name(path) == "tokenizer.json")
        .cloned();
    let genai = files
        .iter()
        .find(|path| file_name(path) == "genai_config.json")
        .cloned();

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
    let gguf_meta = if format == ModelFormat::Gguf {
        gguf.first().map(|path| read_gguf(path)).transpose()?
    } else {
        None
    };
    let architecture = options
        .architecture
        .clone()
        .or_else(|| {
            gguf_meta
                .as_ref()
                .and_then(|meta| meta.architecture.clone())
        })
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
        .or_else(|| gguf_meta.as_ref().and_then(|meta| meta.quantization))
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
            let vocab_size = gguf_meta
                .as_ref()
                .and_then(|meta| meta.vocab_size)
                .or_else(|| config.as_ref().and_then(|config| config.vocab_size));
            let text = match (vocab_size, tokenizer.as_deref()) {
                (Some(vocab_size), Some(tokenizer)) => Some(uniinfer_core::TextConfig {
                    vocab_size,
                    eos_token: config.as_ref().and_then(|config| config.eos_token_id),
                    family: architecture.clone(),
                    config_file: config_path.as_deref().map(|path| relative(root, path)),
                    tokenizer_file: Some(relative(root, tokenizer)),
                }),
                _ => {
                    warnings.push(
                        "GGUF model lacks tokenizer.json or vocab metadata; text loading may fail"
                            .into(),
                    );
                    None
                }
            };
            (vec![Modality::Text], text, None, selected)
        }
        ModelFormat::Safetensors => {
            let primary = choose_primary(&safetensors, "model.safetensors");
            let selected = with_metadata(primary, tokenizer.as_deref(), config_path.as_deref());
            let text = crate::metadata::text_config(
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
                family: Some("whisper".into()),
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

    let context_len = gguf_meta
        .as_ref()
        .and_then(|meta| meta.context_len)
        .unwrap_or_else(|| context_len(config.as_ref(), &modalities));
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
    let (runtime, implementation) = match format {
        ModelFormat::Onnx if audio.is_some() => ("onnx", "onnx.asr.whisper"),
        ModelFormat::Onnx => ("onnx", "onnx.graph"),
        ModelFormat::Gguf | ModelFormat::Safetensors => ("candle", "candle.text"),
        ModelFormat::Ggml => ("whisper", "whisper.ggml"),
    };
    let mut capabilities = Vec::new();
    if modalities.contains(&Modality::Audio) {
        capabilities.push(Capability {
            name: "transcribe".into(),
            operation: Operation::Transcribe,
            required_inputs: vec![Modality::Audio],
            optional_inputs: Vec::new(),
            outputs: vec![Modality::Text],
            mode: Mode::Batch,
        });
    }
    if modalities.contains(&Modality::Text) {
        capabilities.push(Capability {
            name: "text_chat".into(),
            operation: Operation::Generate,
            required_inputs: vec![Modality::Text],
            optional_inputs: modalities
                .iter()
                .copied()
                .filter(|modality| *modality != Modality::Text)
                .collect(),
            outputs: vec![Modality::Text],
            mode: Mode::Stream,
        });
    }
    if modalities.contains(&Modality::Image) && !modalities.contains(&Modality::Text) {
        capabilities.push(Capability {
            name: "image_understand".into(),
            operation: Operation::Understand,
            required_inputs: vec![Modality::Image],
            optional_inputs: vec![Modality::Text],
            outputs: vec![Modality::Text],
            mode: Mode::Batch,
        });
    }
    let runtime = match runtime {
        "onnx" => Runtime::Onnx,
        "candle" => Runtime::Candle,
        "whisper" => Runtime::Whisper,
        _ => Runtime::Onnx,
    };
    let manifest = ModelManifest {
        id,
        runtime,
        implementation: implementation.into(),
        capabilities,
        format,
        quantization,
        streaming_capable: false,
        context_len,
        tokenizer: tokenizer.as_ref().map(|_| "hf".into()),
        min_vram: None,
        placement: PlacementConfig::default(),
        files,
        text,
        audio,
        vision: None,
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
