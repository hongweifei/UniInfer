//! Detection of standard multi-component diffusers ONNX layouts.

use std::path::{Path, PathBuf};

use uniinfer_core::{DiffusionComponent, DiffusionConfig, DiffusionScheduler, ImageConfig};

use crate::files::{file_name, relative_string};
use crate::onnx_graph::{GraphInfo, read};

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

    selected.sort();
    selected.dedup();

    let text_info = read(&text_encoder).ok();
    let unet_info = read(&unet).ok();
    let vae_info = read(&vae_decoder).ok();
    let (width, height) = image_size(vae_info.as_ref(), unet_info.as_ref());
    let scheduler = scheduler_config(root);
    let image = ImageConfig {
        width,
        height,
        diffusion: Some(DiffusionConfig {
            text_encoder: DiffusionComponent {
                path: relative_string(root, &text_encoder),
                output: output_name(text_info.as_ref(), "last_hidden_state"),
                device: None,
            },
            unet: DiffusionComponent {
                path: relative_string(root, &unet),
                output: output_name(unet_info.as_ref(), "out_sample"),
                device: None,
            },
            vae_decoder: DiffusionComponent {
                path: relative_string(root, &vae_decoder),
                output: output_name(vae_info.as_ref(), "sample"),
                device: None,
            },
            text_encoder_input: input_name(text_info.as_ref(), "input_ids"),
            unet_sample_input: input_name(unet_info.as_ref(), "sample"),
            unet_timestep_input: input_at(unet_info.as_ref(), 1, "timestep"),
            unet_encoder_input: input_at(unet_info.as_ref(), 2, "encoder_hidden_states"),
            unet_timestep_dtype: input_dtype(unet_info.as_ref(), "timestep"),
            vae_input: input_at(vae_info.as_ref(), 0, "latent_sample"),
            latent_channels: 4,
            max_length: 77,
            pad_token: 0,
            vae_scale: 0.18215,
            steps: 1,
            guidance: 1.0,
            scheduler,
        }),
    };
    Some(DiffusionLayout { image, selected })
}

fn input_name(graph: Option<&GraphInfo>, fallback: &str) -> String {
    graph
        .and_then(|graph| {
            graph
                .inputs
                .iter()
                .find(|tensor| tensor.name == fallback)
                .or_else(|| graph.inputs.first())
        })
        .map_or_else(|| fallback.to_owned(), |tensor| tensor.name.clone())
}

fn input_at(graph: Option<&GraphInfo>, index: usize, fallback: &str) -> String {
    graph
        .and_then(|graph| graph.inputs.get(index))
        .map_or_else(|| fallback.to_owned(), |tensor| tensor.name.clone())
}

fn input_dtype(graph: Option<&GraphInfo>, fallback: &str) -> String {
    graph
        .and_then(|graph| {
            graph
                .inputs
                .iter()
                .find(|tensor| tensor.name == fallback)
                .or_else(|| graph.inputs.get(1))
        })
        .and_then(|tensor| tensor.dtype)
        .map_or_else(
            || "f32".into(),
            |dtype| match dtype {
                7 => "i64".into(),
                6 => "i32".into(),
                _ => "f32".into(),
            },
        )
}

fn output_name(graph: Option<&GraphInfo>, fallback: &str) -> String {
    graph
        .and_then(|graph| graph.outputs.first())
        .map_or_else(|| fallback.to_owned(), |tensor| tensor.name.clone())
}

fn image_size(vae: Option<&GraphInfo>, unet: Option<&GraphInfo>) -> (u32, u32) {
    if let Some(shape) = vae
        .and_then(|graph| graph.outputs.first())
        .map(|tensor| &tensor.shape)
        && shape.len() >= 4
        && shape[2] > 0
        && shape[3] > 0
    {
        return (shape[3] as u32, shape[2] as u32);
    }
    if let Some(shape) = unet
        .and_then(|graph| graph.inputs.iter().find(|tensor| tensor.shape.len() == 4))
        .map(|tensor| &tensor.shape)
        && shape[2] > 0
        && shape[3] > 0
    {
        return ((shape[3] * 8) as u32, (shape[2] * 8) as u32);
    }
    (512, 512)
}

fn scheduler_config(root: &Path) -> DiffusionScheduler {
    #[derive(Default, serde::Deserialize)]
    struct Raw {
        num_train_timesteps: Option<u32>,
        beta_start: Option<f32>,
        beta_end: Option<f32>,
    }

    let path = root.join("scheduler/scheduler_config.json");
    let Ok(raw) = std::fs::read_to_string(path) else {
        return DiffusionScheduler::default();
    };
    let Ok(raw) = serde_json::from_str::<Raw>(&raw) else {
        return DiffusionScheduler::default();
    };
    DiffusionScheduler {
        num_train_timesteps: raw.num_train_timesteps.unwrap_or(1000),
        beta_start: raw.beta_start.unwrap_or(0.00085),
        beta_end: raw.beta_end.unwrap_or(0.012),
    }
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
