//! Per-session residency accounting and admission.
//!
//! The ledger lives beside engine routing because it must serialize loads and
//! reserve bytes for the lifetime of a `SessionLease`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use uniinfer_core::{
    DeviceClass, EngineError, ModelManifest, PlacementPlan, Residency, ResidencySummary, Result,
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
        check_limits(&state, inventory, placement, residency)
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
        check_limits(&state, inventory, placement, residency)?;
        let gpu = placement
            .devices
            .iter()
            .find(|device| device.class == DeviceClass::Gpu);
        let device = gpu.map(|device| device.id.clone());
        if residency.vram_bytes > 0
            && let Some(device) = &device
        {
            let used = state.vram.entry(device.clone()).or_default();
            *used = used.saturating_add(residency.vram_bytes);
        }
        state.host_bytes = state.host_bytes.saturating_add(residency.host_bytes);
        state.disk_bytes = state.disk_bytes.saturating_add(residency.disk_bytes);
        state.sessions += 1;
        Ok(Reservation {
            ledger: Arc::clone(&self.inner),
            device,
            vram_bytes: residency.vram_bytes,
            host_bytes: residency.host_bytes,
            disk_bytes: residency.disk_bytes,
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

fn check_limits(
    state: &LedgerState,
    inventory: &DeviceInventory,
    placement: &PlacementPlan,
    residency: Residency,
) -> Result<()> {
    let gpu = placement
        .devices
        .iter()
        .find(|device| device.class == DeviceClass::Gpu);
    if residency.vram_bytes > 0 {
        let Some(gpu) = gpu else {
            return Err(EngineError::InsufficientVram {
                required: format!("{} bytes", residency.vram_bytes),
                available: "no GPU selected".into(),
            });
        };
        let used = state.vram.get(&gpu.id).copied().unwrap_or(0);
        let available = gpu.available_memory_bytes;
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
        let available = cpu.available_memory_bytes;
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
