//! Multi-component ONNX diffusion (text encoder + UNet + VAE).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use ort::environment::Environment;
use uniinfer_core::{
    CancelToken, DiffusionConfig, EngineError, EngineLoadSpec, EngineSession,
    ImageGenerationRequest, Result, Tensor, TensorData, TensorDtype, TensorSpec,
};

use super::scheduler;
use crate::graph::Graph;

const LATENT_SCALE: u32 = 8;

pub(super) fn load(
    environment: &Environment,
    spec: &EngineLoadSpec,
    class: uniinfer_core::DeviceClass,
) -> Result<Box<dyn EngineSession>> {
    let config = spec
        .manifest
        .image
        .as_ref()
        .and_then(|image| image.diffusion.clone())
        .ok_or_else(|| EngineError::ManifestInvalid("image.diffusion config required".into()))?;
    let load_graph = |component: &uniinfer_core::DiffusionComponent, label: &str| {
        let path = spec.base_dir.join(&component.path);
        Graph::load(
            environment,
            component.device.unwrap_or(class),
            &path,
            label,
            crate::backend_variant(spec),
        )
    };
    let text = load_graph(&config.text_encoder, "diffusion text encoder")?;
    let unet = load_graph(&config.unet, "diffusion unet")?;
    let vae = load_graph(&config.vae_decoder, "diffusion vae")?;
    if spec.cancel.is_cancelled() {
        return Err(EngineError::Cancelled);
    }
    Ok(Box::new(DiffusionSession {
        text: Mutex::new(text),
        unet: Mutex::new(unet),
        vae: Mutex::new(vae),
        config,
        cancelled: AtomicBool::new(false),
    }))
}

struct DiffusionSession {
    text: Mutex<Graph>,
    unet: Mutex<Graph>,
    vae: Mutex<Graph>,
    config: DiffusionConfig,
    cancelled: AtomicBool,
}

impl EngineSession for DiffusionSession {
    fn inputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn outputs(&self) -> &[TensorSpec] {
        &[]
    }

    fn run(&self, _inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        Err(EngineError::UnsupportedEngine(
            "diffusion sessions use generate_image".into(),
        ))
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn generate_image(&self, request: &mut ImageGenerationRequest<'_>) -> Result<Tensor> {
        self.check_cancelled()?;
        let encoder = self.encode(request.prompt_ids, request.cancel)?;
        let negative = if request.guidance > 1.0 && !request.negative_ids.is_empty() {
            Some(self.encode(request.negative_ids, request.cancel)?)
        } else {
            None
        };
        let width = request.width.max(64);
        let height = request.height.max(64);
        let latent_h = height.div_ceil(LATENT_SCALE) as usize;
        let latent_w = width.div_ceil(LATENT_SCALE) as usize;
        let steps = request.steps.clamp(1, 100);
        let kind = self.config.scheduler.kind;
        let use_ddim = kind == uniinfer_core::DiffusionSchedulerKind::Ddim;
        let use_ddpm = kind == uniinfer_core::DiffusionSchedulerKind::Ddpm;
        let use_ancestral = kind == uniinfer_core::DiffusionSchedulerKind::EulerAncestral;
        let (sigmas, timesteps, ddim_alphas, ddpm_alphas) = if use_ddim {
            let (alphas, timesteps) = scheduler::ddim(steps, &self.config.scheduler);
            (Vec::new(), timesteps, alphas, Vec::new())
        } else if use_ddpm {
            let (alphas, timesteps) = scheduler::ddpm(steps, &self.config.scheduler);
            (Vec::new(), timesteps, Vec::new(), alphas)
        } else {
            let (sigmas, timesteps) = scheduler::schedule(steps, &self.config.scheduler);
            (sigmas, timesteps, Vec::new(), Vec::new())
        };
        let mut sample = if use_ddim || use_ddpm {
            normal_latent(
                self.config.latent_channels,
                latent_h,
                latent_w,
                request.seed,
            )
        } else {
            let mut sample = random_latent(
                self.config.latent_channels,
                latent_h,
                latent_w,
                request.seed,
            );
            for value in &mut sample {
                *value *= sigmas[0];
            }
            sample
        };
        let mut last_preview = None;
        for step in 0..steps {
            self.check_cancelled()?;
            let sigma = if use_ddim || use_ddpm {
                1.0
            } else {
                sigmas[step]
            };
            let next_sigma = if use_ddim || use_ddpm {
                1.0
            } else {
                sigmas[step + 1]
            };
            let timestep = Tensor::new(
                self.config.unet_timestep_input.clone(),
                vec![1],
                if self.config.unet_timestep_dtype.eq_ignore_ascii_case("i64") {
                    TensorData::I64(vec![timesteps[step] as i64])
                } else if self.config.unet_timestep_dtype.eq_ignore_ascii_case("f64") {
                    TensorData::F64(vec![f64::from(timesteps[step])])
                } else {
                    TensorData::F32(vec![timesteps[step]])
                },
            )?;
            let input_scale = if matches!(
                self.config.scheduler.kind,
                uniinfer_core::DiffusionSchedulerKind::Euler
                    | uniinfer_core::DiffusionSchedulerKind::EulerAncestral
            ) {
                (sigma * sigma + 1.0).sqrt()
            } else {
                1.0
            };
            let sample_tensor = Tensor::new(
                self.config.unet_sample_input.clone(),
                vec![1, self.config.latent_channels, latent_h, latent_w],
                TensorData::F32(sample.iter().map(|value| value / input_scale).collect()),
            )?;
            let encoder_tensor = encoder.clone();
            let conditional = self.unet(&sample_tensor, &timestep, &encoder_tensor)?;
            let predicted = if let Some(negative) = &negative {
                let unconditional = self.unet(&sample_tensor, &timestep, negative)?;
                combine_cfg(&conditional, &unconditional, request.guidance)?
            } else {
                conditional
            };
            if use_ddim {
                let TensorData::F32(eps) = &predicted.data else {
                    return Err(EngineError::Other("DDIM output is not f32".into()));
                };
                let alpha = ddim_alphas[step];
                let alpha_next = ddim_alphas[step + 1];
                let pred_x0 = sample
                    .iter()
                    .zip(eps)
                    .map(|(sample, eps)| (sample - (1.0 - alpha).sqrt() * eps) / alpha.sqrt())
                    .collect::<Vec<_>>();
                sample = pred_x0
                    .iter()
                    .zip(eps)
                    .map(|(x0, eps)| alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * eps)
                    .collect();
            } else if use_ddpm {
                let TensorData::F32(eps) = &predicted.data else {
                    return Err(EngineError::Other("DDPM output is not f32".into()));
                };
                let alpha = ddpm_alphas[step];
                let alpha_prev = ddpm_alphas.get(step + 1).copied().unwrap_or(1.0);
                let beta = 1.0 - alpha / alpha_prev;
                let pred_x0 = sample
                    .iter()
                    .zip(eps)
                    .map(|(sample, eps)| (sample - (1.0 - alpha).sqrt() * eps) / alpha.sqrt())
                    .collect::<Vec<_>>();
                let denom = (1.0 - alpha).max(f32::MIN_POSITIVE);
                let mean = pred_x0
                    .iter()
                    .zip(sample.iter())
                    .map(|(x0, sample)| {
                        alpha_prev.sqrt() * beta / denom * x0
                            + alpha.sqrt() * (1.0 - alpha_prev) / denom * sample
                    })
                    .collect::<Vec<_>>();
                let variance = ((1.0 - alpha_prev) / denom * beta).max(0.0);
                let noise = normal_latent(
                    self.config.latent_channels,
                    latent_h,
                    latent_w,
                    request.seed.map(|seed| seed.wrapping_add(step as u64 + 1)),
                );
                sample = mean
                    .iter()
                    .zip(noise)
                    .map(|(mean, noise)| mean + variance.sqrt() * noise)
                    .collect();
            } else if use_ancestral && sigma > 0.0 {
                let sigma_sq = sigma * sigma;
                let next_sq = next_sigma * next_sigma;
                let sigma_up = (next_sq * (sigma_sq - next_sq) / sigma_sq).max(0.0).sqrt();
                let sigma_down = (next_sq - sigma_up * sigma_up).max(0.0).sqrt();
                apply_euler(&mut sample, &predicted, sigma_down - sigma)?;
                let noise = normal_latent(
                    self.config.latent_channels,
                    latent_h,
                    latent_w,
                    request.seed.map(|seed| seed.wrapping_add(step as u64 + 1)),
                );
                for (value, noise) in sample.iter_mut().zip(noise) {
                    *value += sigma_up * noise;
                }
            } else {
                apply_euler(&mut sample, &predicted, next_sigma - sigma)?;
            }
            let preview = if request.preview {
                match self.decode_latent(&sample, latent_h, latent_w) {
                    Ok(image) => {
                        last_preview = Some(image.clone());
                        Some(image)
                    }
                    Err(_) => None,
                }
            } else {
                None
            };
            (request.progress)(step + 1, steps, preview);
        }
        if let Some(image) = last_preview {
            return Ok(image);
        }
        self.decode_latent(&sample, latent_h, latent_w)
    }
}

impl DiffusionSession {
    fn decode_latent(&self, sample: &[f32], latent_h: usize, latent_w: usize) -> Result<Tensor> {
        let latent = Tensor::new(
            self.config.vae_input.clone(),
            vec![1, self.config.latent_channels, latent_h, latent_w],
            TensorData::F32(
                sample
                    .iter()
                    .map(|value| value / self.config.vae_scale)
                    .collect(),
            ),
        )?;
        let mut vae = self
            .vae
            .lock()
            .map_err(|_| EngineError::Other("diffusion VAE lock poisoned".into()))?;
        let output = vae.run(vec![latent])?;
        output
            .into_iter()
            .find(|tensor| tensor.name == self.config.vae_decoder.output)
            .ok_or_else(|| EngineError::Other("diffusion VAE output missing".into()))
    }

    fn check_cancelled(&self) -> Result<()> {
        if self.cancelled.load(Ordering::SeqCst) {
            Err(EngineError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn encode(&self, ids: &[i64], cancel: &CancelToken) -> Result<Tensor> {
        check_cancel(cancel)?;
        self.check_cancelled()?;
        let mut text = self
            .text
            .lock()
            .map_err(|_| EngineError::Other("diffusion text encoder lock poisoned".into()))?;
        let dtype = text
            .inputs()
            .iter()
            .find(|spec| spec.name == self.config.text_encoder_input)
            .map_or(TensorDtype::I64, |spec| spec.dtype);
        let data = match dtype {
            TensorDtype::I32 => TensorData::I32(
                ids.iter()
                    .map(|id| i32::try_from(*id))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|_| {
                        EngineError::Other("diffusion token id does not fit in i32".into())
                    })?,
            ),
            TensorDtype::I64 => TensorData::I64(ids.to_vec()),
            other => {
                return Err(EngineError::Other(format!(
                    "diffusion text encoder input_ids has unsupported dtype {other:?}"
                )));
            }
        };
        let input = Tensor::new(
            self.config.text_encoder_input.clone(),
            vec![1, ids.len()],
            data,
        )?;
        let mut encoder = text
            .run(vec![input])?
            .into_iter()
            .find(|tensor| tensor.name == self.config.text_encoder.output)
            .ok_or_else(|| EngineError::Other("diffusion text encoder output missing".into()))?;
        encoder.name = self.config.unet_encoder_input.clone();
        if let TensorData::F32(values) = &encoder.data {
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean = values.iter().copied().sum::<f32>() / values.len().max(1) as f32;
            tracing::debug!(min, max, mean, shape = ?encoder.shape, "text encoder output");
        }
        Ok(encoder)
    }

    fn unet(&self, sample: &Tensor, timestep: &Tensor, encoder: &Tensor) -> Result<Tensor> {
        let mut unet = self
            .unet
            .lock()
            .map_err(|_| EngineError::Other("diffusion UNet lock poisoned".into()))?;
        unet.run(vec![sample.clone(), timestep.clone(), encoder.clone()])?
            .into_iter()
            .find(|tensor| tensor.name == self.config.unet.output)
            .ok_or_else(|| EngineError::Other("diffusion UNet output missing".into()))
    }
}

fn check_cancel(cancel: &CancelToken) -> Result<()> {
    if cancel.is_cancelled() {
        Err(EngineError::Cancelled)
    } else {
        Ok(())
    }
}

fn normal_latent(channels: usize, height: usize, width: usize, seed: Option<u64>) -> Vec<f32> {
    let mut state = seed.unwrap_or(0x5eed_1234_abcd_ef01) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f32 / (1u64 << 53) as f32
    };
    let count = channels * height * width;
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let u1 = next().max(f32::MIN_POSITIVE);
        let u2 = next();
        let radius = (-2.0 * u1.ln()).sqrt();
        values.push(radius * (std::f32::consts::TAU * u2).cos());
        if values.len() < count {
            values.push(radius * (std::f32::consts::TAU * u2).sin());
        }
    }
    values
}

fn random_latent(channels: usize, height: usize, width: usize, seed: Option<u64>) -> Vec<f32> {
    let mut state = seed.unwrap_or(0x5eed_1234_abcd_ef01) | 1;
    let count = channels * height * width;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0);
    }
    values
}

fn combine_cfg(conditional: &Tensor, unconditional: &Tensor, guidance: f32) -> Result<Tensor> {
    let TensorData::F32(cond) = &conditional.data else {
        return Err(EngineError::Other(
            "conditional UNet output is not f32".into(),
        ));
    };
    let TensorData::F32(uncond) = &unconditional.data else {
        return Err(EngineError::Other(
            "unconditional UNet output is not f32".into(),
        ));
    };
    if cond.len() != uncond.len() {
        return Err(EngineError::Other("CFG tensor size mismatch".into()));
    }
    let values = cond
        .iter()
        .zip(uncond)
        .map(|(cond, uncond)| uncond + guidance * (cond - uncond))
        .collect();
    Tensor::new(
        conditional.name.clone(),
        conditional.shape.clone(),
        TensorData::F32(values),
    )
}

fn apply_euler(sample: &mut [f32], predicted: &Tensor, delta: f32) -> Result<()> {
    let TensorData::F32(predicted) = &predicted.data else {
        return Err(EngineError::Other("UNet output is not f32".into()));
    };
    if sample.len() != predicted.len() {
        return Err(EngineError::Other("Euler sample size mismatch".into()));
    }
    for (sample, predicted) in sample.iter_mut().zip(predicted) {
        *sample += delta * predicted;
    }
    Ok(())
}
