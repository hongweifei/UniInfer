//! Per-session residency accounting and admission.
//!
//! The ledger lives beside engine routing because it must serialize loads and
//! reserve bytes for the lifetime of a `SessionLease`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use uniinfer_core::{
    Device, DeviceClass, EngineError, ModelManifest, PlacementPlan, Residency, ResidencySummary,
    Result,
};
use uniinfer_devices::DeviceInventory;

#[derive(Default)]
struct LedgerState {
    vram: HashMap<String, u64>,
    host_bytes: u64,
    disk_bytes: u64,
    sessions: usize,
}

pub(super) struct Reservation {
    ledger: Arc<Mutex<LedgerState>>,
    device: Option<String>,
    vram_bytes: u64,
    host_bytes: u64,
    disk_bytes: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if let Ok(mut state) = self.ledger.lock() {
            if let Some(device) = &self.device
                && self.vram_bytes > 0
                && let Some(used) = state.vram.get_mut(device)
            {
                *used = used.saturating_sub(self.vram_bytes);
                if *used == 0 {
                    state.vram.remove(device);
                }
            }
            state.host_bytes = state.host_bytes.saturating_sub(self.host_bytes);
            state.disk_bytes = state.disk_bytes.saturating_sub(self.disk_bytes);
            state.sessions = state.sessions.saturating_sub(1);
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct ResidencyLedger {
    inner: Arc<Mutex<LedgerState>>,
    vram_reserve: Arc<AtomicU64>,
    host_reserve: Arc<AtomicU64>,
}

impl ResidencyLedger {
    pub(super) fn check(
        &self,
        inventory: &DeviceInventory,
        placement: &PlacementPlan,
        residency: Residency,
    ) -> Result<()> {
        let state = self
            .inner
            .lock()
            .map_err(|_| EngineError::Other("residency ledger lock poisoned".into()))?;
        check_limits(
            &state,
            inventory,
            placement,
            residency,
            self.vram_reserve.load(Ordering::Relaxed),
            self.host_reserve.load(Ordering::Relaxed),
        )
    }

    pub(super) fn set_reserves(&self, vram: u64, host: u64) {
        self.vram_reserve.store(vram, Ordering::Relaxed);
        self.host_reserve.store(host, Ordering::Relaxed);
    }

    pub(super) fn reserves(&self) -> (u64, u64) {
        (
            self.vram_reserve.load(Ordering::Relaxed),
            self.host_reserve.load(Ordering::Relaxed),
        )
    }

    pub(super) fn reserved(&self) -> (HashMap<String, u64>, u64) {
        self.inner.lock().map_or_else(
            |_| (HashMap::new(), 0),
            |state| (state.vram.clone(), state.host_bytes),
        )
    }

    pub(super) fn reserve(
        &self,
        inventory: &DeviceInventory,
        placement: &PlacementPlan,
        residency: Residency,
    ) -> Result<Reservation> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| EngineError::Other("residency ledger lock poisoned".into()))?;
        let gpu = placement
            .devices
            .iter()
            .find(|device| device.class == DeviceClass::Gpu);
        let effective = effective_residency(residency, gpu);
        check_limits(
            &state,
            inventory,
            placement,
            effective,
            self.vram_reserve.load(Ordering::Relaxed),
            self.host_reserve.load(Ordering::Relaxed),
        )?;
        let device = if effective.vram_bytes > 0 {
            gpu.map(|device| device.id.clone())
        } else {
            None
        };
        if effective.vram_bytes > 0
            && let Some(device) = &device
        {
            let used = state.vram.entry(device.clone()).or_default();
            *used = used.saturating_add(effective.vram_bytes);
        }
        state.host_bytes = state.host_bytes.saturating_add(effective.host_bytes);
        state.disk_bytes = state.disk_bytes.saturating_add(effective.disk_bytes);
        state.sessions += 1;
        Ok(Reservation {
            ledger: Arc::clone(&self.inner),
            device,
            vram_bytes: effective.vram_bytes,
            host_bytes: effective.host_bytes,
            disk_bytes: effective.disk_bytes,
        })
    }

    pub(super) fn summary(&self) -> ResidencySummary {
        let Ok(state) = self.inner.lock() else {
            return ResidencySummary::default();
        };
        ResidencySummary {
            sessions: state.sessions,
            vram_bytes: state.vram.values().copied().sum(),
            host_bytes: state.host_bytes,
            disk_bytes: state.disk_bytes,
        }
    }
}

/// Unified-memory accelerators spend host RAM, not a separate VRAM tier.
/// Move their reported VRAM bytes into host accounting before admission so
/// hybrid CPU+GPU sessions on an iGPU cannot double-count the same pool.
pub(super) fn effective_residency(residency: Residency, gpu: Option<&Device>) -> Residency {
    if gpu.is_some_and(|device| device.unified_memory) {
        Residency {
            vram_bytes: 0,
            host_bytes: residency.host_bytes.saturating_add(residency.vram_bytes),
            ..residency
        }
    } else {
        residency
    }
}

fn check_limits(
    state: &LedgerState,
    inventory: &DeviceInventory,
    placement: &PlacementPlan,
    residency: Residency,
    vram_reserve: u64,
    host_reserve: u64,
) -> Result<()> {
    let gpu = placement
        .devices
        .iter()
        .find(|device| device.class == DeviceClass::Gpu);
    let residency = effective_residency(residency, gpu);
    if residency.vram_bytes > 0 {
        let Some(gpu) = gpu else {
            return Err(EngineError::InsufficientVram {
                required: format!("{} bytes", residency.vram_bytes),
                available: "no GPU selected".into(),
            });
        };
        let used = state.vram.get(&gpu.id).copied().unwrap_or(0);
        let available = gpu.available_memory_bytes.saturating_sub(vram_reserve);
        if available > 0 && used.saturating_add(residency.vram_bytes) > available {
            return Err(EngineError::InsufficientVram {
                required: format!("{} bytes", residency.vram_bytes),
                available: format!("{} bytes", available.saturating_sub(used)),
            });
        }
    }
    if residency.host_bytes > 0
        && let Some(cpu) = inventory
            .devices()
            .iter()
            .find(|device| device.class == DeviceClass::Cpu)
    {
        let available = cpu.available_memory_bytes.saturating_sub(host_reserve);
        if available > 0 && state.host_bytes.saturating_add(residency.host_bytes) > available {
            return Err(EngineError::InsufficientMemory {
                required: format!("{} bytes", residency.host_bytes),
                available: format!("{} bytes", available.saturating_sub(state.host_bytes)),
            });
        }
    }
    Ok(())
}

/// Estimate a footprint when the engine cannot report exact residency.
pub(super) fn estimate_residency(
    manifest: &ModelManifest,
    base_dir: &Path,
    placement: &PlacementPlan,
) -> Option<Residency> {
    let bytes = manifest
        .files
        .iter()
        .filter_map(|file| std::fs::metadata(base_dir.join(&file.path)).ok())
        .map(|metadata| metadata.len())
        .sum::<u64>();
    if bytes == 0 {
        return None;
    }
    let full_gpu = placement
        .devices
        .first()
        .is_some_and(|device| device.class == DeviceClass::Gpu)
        && placement.offload_layers == 0
        && !placement.auto_offload;
    Some(Residency {
        vram_bytes: if full_gpu { bytes } else { 0 },
        // Mmap-backed weights are accounted as disk, not resident host bytes.
        host_bytes: 0,
        disk_bytes: bytes,
        offload_layers: placement.offload_layers,
        total_layers: None,
        estimated: true,
    })
}
