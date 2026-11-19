//! Sampling schedules shared by image architecture adapters.

use uniinfer_core::{DiffusionScheduler, DiffusionSchedulerKind};

/// Returns `(sigmas, timesteps)`; `sigmas` has `steps + 1` entries.
pub(crate) fn schedule(steps: usize, config: &DiffusionScheduler) -> (Vec<f32>, Vec<f32>) {
    match config.kind {
        DiffusionSchedulerKind::Euler => euler_schedule(steps, config),
        DiffusionSchedulerKind::FlowMatch => flow_schedule(steps),
        DiffusionSchedulerKind::Ddim => euler_schedule(steps, config),
        DiffusionSchedulerKind::EulerAncestral => euler_schedule(steps, config),
        DiffusionSchedulerKind::Ddpm => euler_schedule(steps, config),
    }
}

fn euler_schedule(steps: usize, config: &DiffusionScheduler) -> (Vec<f32>, Vec<f32>) {
    let train = config.num_train_timesteps.max(1) as usize;
    let mut alpha_cumprod = 1.0f32;
    let mut train_sigmas = Vec::with_capacity(train);
    // Stable Diffusion uses scaled_linear betas: interpolate the square roots
    // of beta_start/beta_end and square the result.
    let beta_start = config.beta_start.max(0.0).sqrt();
    let beta_end = config.beta_end.max(0.0).sqrt();
    for index in 0..train {
        let ratio = index as f32 / (train.saturating_sub(1)).max(1) as f32;
        let beta = beta_start + (beta_end - beta_start) * ratio;
        let beta = beta * beta;
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

fn flow_schedule(steps: usize) -> (Vec<f32>, Vec<f32>) {
    let steps = steps.max(1);
    let mut sigmas = Vec::with_capacity(steps + 1);
    let mut timesteps = Vec::with_capacity(steps);
    for step in 0..steps {
        let sigma = 1.0 - step as f32 / steps as f32;
        sigmas.push(sigma);
        timesteps.push(sigma * 1000.0);
    }
    sigmas.push(0.0);
    (sigmas, timesteps)
}

pub(crate) fn ddim(steps: usize, config: &DiffusionScheduler) -> (Vec<f32>, Vec<f32>) {
    let train = config.num_train_timesteps.max(1) as usize;
    let beta_start = config.beta_start.max(0.0).sqrt();
    let beta_end = config.beta_end.max(0.0).sqrt();
    let mut alpha_cumprod = 1.0f32;
    let mut alphas = Vec::with_capacity(train);
    for index in 0..train {
        let ratio = index as f32 / (train.saturating_sub(1)).max(1) as f32;
        let beta = beta_start + (beta_end - beta_start) * ratio;
        alpha_cumprod *= 1.0 - beta * beta;
        alphas.push(alpha_cumprod);
    }
    let steps = steps.max(1);
    let timesteps = (0..steps)
        .map(|step| (train - 1) as f32 * (steps - step) as f32 / steps as f32)
        .collect::<Vec<_>>();
    let alpha_at = |timestep: f32| alphas[timestep.round() as usize % train];
    let mut schedule_alphas = timesteps
        .iter()
        .map(|timestep| alpha_at(*timestep))
        .collect::<Vec<_>>();
    schedule_alphas.push(1.0);
    (schedule_alphas, timesteps)
}

pub(crate) fn ddpm(steps: usize, config: &DiffusionScheduler) -> (Vec<f32>, Vec<f32>) {
    let train = config.num_train_timesteps.max(1) as usize;
    let beta_start = config.beta_start.max(0.0).sqrt();
    let beta_end = config.beta_end.max(0.0).sqrt();
    let mut alpha_cumprod = 1.0f32;
    let mut alphas = Vec::with_capacity(train);
    for index in 0..train {
        let ratio = index as f32 / (train.saturating_sub(1)).max(1) as f32;
        let beta = beta_start + (beta_end - beta_start) * ratio;
        alpha_cumprod *= 1.0 - beta * beta;
        alphas.push(alpha_cumprod);
    }
    let alpha_at = |timestep: f32| alphas[timestep.round() as usize % train];
    let steps = steps.max(1);
    let timesteps = (0..steps)
        .map(|step| (train - 1) as f32 * (steps - step) as f32 / steps as f32)
        .collect::<Vec<_>>();
    let schedule_alphas = timesteps.iter().map(|t| alpha_at(*t)).collect::<Vec<_>>();
    (schedule_alphas, timesteps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniinfer_core::{DiffusionScheduler, DiffusionSchedulerKind};

    #[test]
    fn flow_match_has_monotonic_sigmas() {
        let config = DiffusionScheduler {
            kind: DiffusionSchedulerKind::FlowMatch,
            ..Default::default()
        };
        let (sigmas, timesteps) = schedule(4, &config);
        assert_eq!(sigmas.len(), 5);
        assert_eq!(timesteps.len(), 4);
        assert!(sigmas.windows(2).all(|pair| pair[0] >= pair[1]));
        assert_eq!(sigmas.last(), Some(&0.0));
    }

    #[test]
    fn euler_schedule_ends_at_zero() {
        let config = DiffusionScheduler {
            kind: DiffusionSchedulerKind::Euler,
            ..Default::default()
        };
        let (sigmas, _) = schedule(3, &config);
        assert_eq!(sigmas.len(), 4);
        assert_eq!(sigmas.last(), Some(&0.0));
    }
}
