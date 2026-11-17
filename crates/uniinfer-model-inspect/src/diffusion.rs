//! Detection of standard multi-component diffusers ONNX layouts.

use std::path::{Path, PathBuf};

use uniinfer_core::{DiffusionComponent, DiffusionConfig, DiffusionScheduler, ImageConfig};

use crate::files::{file_name, relative_string};

pub(crate) struct DiffusionLayout {
    pub(crate) image: ImageConfig,
    pub(crate) selected: Vec<PathBuf>,
}

pub(crate) fn detect(root: &Path, onnx: &[PathBuf]) -> Option<DiffusionLayout> {
    let text_encoder = component(onnx, root, "text_encoder")?;
    let unet = component(onnx, root, "unet").or_else(|| component(onnx, root, "transformer"))?;
    let vae_decoder =
        component(onnx, root, "vae_decoder").or_else(|| component(onnx, root, "vae"))?;

    let mut selected = vec![text_encoder.clone(), unet.clone(), vae_decoder.clone()];
    selected.extend(component_metadata(root, "text_encoder"));
    selected.extend(component_metadata(root, "unet"));
    selected.extend(component_metadata(root, "vae_decoder"));
    selected.extend(component_metadata(root, "transformer"));
    selected.extend(component_metadata(root, "vae"));
    selected.extend(files_under(root, "scheduler"));
    selected.extend(files_under(root, "tokenizer"));
    selected.extend(files_under(root, "tokenizer_2"));
    for path in onnx {
        if is_external_data(&path.to_string_lossy()) {
            selected.push(path.clone());
        }
    }

    selected.sort();
    selected.dedup();

    let image = ImageConfig {
        width: 512,
        height: 512,
        diffusion: Some(DiffusionConfig {
            text_encoder: DiffusionComponent {
                path: relative_string(root, &text_encoder),
                output: "last_hidden_state".into(),
                device: None,
            },
            unet: DiffusionComponent {
                path: relative_string(root, &unet),
                output: "out_sample".into(),
                device: None,
            },
            vae_decoder: DiffusionComponent {
                path: relative_string(root, &vae_decoder),
                output: "sample".into(),
                device: None,
            },
            text_encoder_input: "input_ids".into(),
            unet_sample_input: "sample".into(),
            unet_timestep_input: "timestep".into(),
            unet_encoder_input: "encoder_hidden_states".into(),
            unet_timestep_dtype: "f32".into(),
            vae_input: "latent_sample".into(),
            latent_channels: 4,
            max_length: 77,
            pad_token: 0,
            vae_scale: 0.18215,
            steps: 1,
            guidance: 1.0,
            scheduler: DiffusionScheduler::default(),
        }),
    };
    Some(DiffusionLayout { image, selected })
}

fn component(onnx: &[PathBuf], root: &Path, directory: &str) -> Option<PathBuf> {
    let mut candidates = onnx
        .iter()
        .filter(|path| {
            relative_string(root, path)
                .split('/')
                .next()
                .is_some_and(|top| top == directory)
        })
        .cloned()
        .collect::<Vec<_>>();
    candidates.sort_by_key(|path| {
        let name = file_name(path);
        if name == "model.onnx" {
            0
        } else if name == "replaced.onnx" {
            1
        } else {
            2
        }
    });
    candidates.into_iter().next()
}

fn component_metadata(root: &Path, directory: &str) -> Vec<PathBuf> {
    files_under(root, directory)
        .into_iter()
        .filter(|path| {
            matches!(
                file_name(path).as_str(),
                "config.json" | "preprocessor_config.json" | "tokenizer.json"
            )
        })
        .collect()
}

fn files_under(root: &Path, directory: &str) -> Vec<PathBuf> {
    let dir = root.join(directory);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files.extend(files_under(root, &relative_string(root, &path)));
        } else if path.is_file() {
            files.push(path);
        }
    }
    files
}

fn is_external_data(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    name.contains(".onnx_data") || name.ends_with(".onnx.data")
}
