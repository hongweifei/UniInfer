use std::path::{Path, PathBuf};

use uniinfer_core::{AudioConfig, Modality, TextConfig};

use crate::error::{InspectError, Result};
use crate::files::{choose_primary, file_name, relative_string, with_metadata};
use crate::metadata::{ConfigJson, PreprocessorConfig, text_config};

pub(crate) struct OnnxInputs<'a> {
    pub(crate) root: &'a Path,
    pub(crate) config: Option<&'a ConfigJson>,
    pub(crate) tokenizer: Option<&'a Path>,
    pub(crate) config_path: Option<&'a Path>,
    pub(crate) genai: Option<&'a Path>,
    pub(crate) architecture: Option<&'a str>,
    pub(crate) preprocessor: Option<&'a PreprocessorConfig>,
}

pub(crate) struct OnnxLayout {
    pub(crate) modalities: Vec<Modality>,
    pub(crate) text: Option<TextConfig>,
    pub(crate) audio: Option<AudioConfig>,
    pub(crate) selected: Vec<PathBuf>,
}

pub(crate) fn onnx_layout(
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
            family: Some(architecture.unwrap_or("whisper").into()),
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
