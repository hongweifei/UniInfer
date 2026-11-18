//! Multi-component ONNX diffusion (text encoder + UNet + VAE).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use ort::environment::Environment;
use uniinfer_core::{
    CancelToken, DiffusionConfig, EngineError, EngineLoadSpec, EngineSession,
    ImageGenerationRequest, Result, Tensor, TensorData, TensorSpec,
};

use crate::graph::Graph;

const LATENT_SCALE: u32 = 8;

pub(crate) fn load(
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
        let (sigmas, timesteps) = euler_schedule(steps, &self.config.scheduler);
        let mut sample = random_latent(
            self.config.latent_channels,
            latent_h,
            latent_w,
            request.seed,
        );
        for value in &mut sample {
            *value *= sigmas[0];
        }
        for step in 0..steps {
            self.check_cancelled()?;
            let sigma = sigmas[step];
            let next_sigma = sigmas[step + 1];
            let timestep = Tensor::new(
                self.config.unet_timestep_input.clone(),
                vec![1],
                if self.config.unet_timestep_dtype.eq_ignore_ascii_case("i64") {
                    TensorData::I64(vec![timesteps[step] as i64])
                } else {
                    TensorData::F32(vec![timesteps[step]])
                },
            )?;
            let sample_tensor = Tensor::new(
                self.config.unet_sample_input.clone(),
                vec![1, self.config.latent_channels, latent_h, latent_w],
                TensorData::F32(sample.clone()),
            )?;
            let encoder_tensor = encoder.clone();
            let conditional = self.unet(&sample_tensor, &timestep, &encoder_tensor)?;
            let predicted = if let Some(negative) = &negative {
                let unconditional = self.unet(&sample_tensor, &timestep, negative)?;
                combine_cfg(&conditional, &unconditional, request.guidance)?
            } else {
                conditional
            };
            apply_euler(&mut sample, &predicted, next_sigma - sigma)?;
            (request.progress)(step + 1, steps);
        }
        let latent = Tensor::new(
            self.config.vae_input.clone(),
            vec![1, self.config.latent_channels, latent_h, latent_w],
            TensorData::F32(
                sample
                    .into_iter()
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
}

impl DiffusionSession {
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
        let input = Tensor::new(
            self.config.text_encoder_input.clone(),
            vec![1, ids.len()],
            TensorData::I64(ids.to_vec()),
        )?;
        let mut encoder = text
            .run(vec![input])?
            .into_iter()
            .find(|tensor| tensor.name == self.config.text_encoder.output)
            .ok_or_else(|| EngineError::Other("diffusion text encoder output missing".into()))?;
        encoder.name = self.config.unet_encoder_input.clone();
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

fn euler_schedule(
    steps: usize,
    config: &uniinfer_core::DiffusionScheduler,
) -> (Vec<f32>, Vec<f32>) {
    let train = config.num_train_timesteps.max(1) as usize;
    let mut alpha_cumprod = 1.0f32;
    let mut train_sigmas = Vec::with_capacity(train);
    for index in 0..train {
        let ratio = index as f32 / (train.saturating_sub(1)).max(1) as f32;
        let beta = config.beta_start + (config.beta_end - config.beta_start) * ratio;
        alpha_cumprod *= 1.0 - beta;
        train_sigmas.push(((1.0 - alpha_cumprod) / alpha_cumprod).sqrt());
    }
    let mut sigmas = Vec::with_capacity(steps + 1);
    let mut timesteps = Vec::with_capacity(steps);
    for step in 0..steps {
        let index = ((train - 1) as f32 * (steps - step) as f32 / steps as f32).round() as usize;
        let index = index.min(train - 1);
        sigmas.push(train_sigmas[index]);
        timesteps.push(index as f32);
    }
    sigmas.push(0.0);
    (sigmas, timesteps)
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
