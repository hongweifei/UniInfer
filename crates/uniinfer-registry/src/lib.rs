//! Engine routing, preflight checks and session leases.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use uniinfer_core::{
    CancelToken, DeviceBudget, DeviceClass, EngineError, EngineLoadSpec, EngineSession,
    InferenceEngine, ModelManifest, PlacementPlan, Residency, ResidencySummary, Result,
};
use uniinfer_devices::DeviceInventory;

mod residency;

use residency::{Reservation, ResidencyLedger, estimate_residency};

/// Engine selection and placement without loading model weights.
#[derive(Debug, Clone)]
pub struct EngineRoute {
    pub engine_name: String,
    pub placement: PlacementPlan,
}

/// Owns registered engines and hands out leases.
pub struct EngineRegistry {
    engines: Vec<Arc<dyn InferenceEngine>>,
    inventory: DeviceInventory,
    active: Arc<AtomicUsize>,
    ledger: ResidencyLedger,
    load_gate: Mutex<()>,
}

impl EngineRegistry {
    #[must_use]
    pub fn new(inventory: DeviceInventory) -> Self {
        Self {
            engines: Vec::new(),
            inventory,
            active: Arc::new(AtomicUsize::new(0)),
            ledger: ResidencyLedger::default(),
            load_gate: Mutex::new(()),
        }
    }

    pub fn register(&mut self, engine: Arc<dyn InferenceEngine>) {
        self.engines.push(engine);
    }

    #[must_use]
    pub fn inventory(&self) -> &DeviceInventory {
        &self.inventory
    }

    #[must_use]
    pub fn active_sessions(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    /// Aggregate bytes currently reserved by loaded sessions.
    #[must_use]
    pub fn residency(&self) -> ResidencySummary {
        self.ledger.summary()
    }

    fn budget(&self) -> DeviceBudget {
        let (reserved_vram, reserved_host) = self.ledger.reserved();
        let mut devices = self.inventory.devices().to_vec();
        for device in &mut devices {
            let used = match device.class {
                DeviceClass::Cpu => reserved_host,
                DeviceClass::Gpu => reserved_vram.get(&device.id).copied().unwrap_or(0),
                DeviceClass::Npu => 0,
            };
            device.available_memory_bytes = device.available_memory_bytes.saturating_sub(used);
        }
        DeviceBudget { devices }
    }

    /// Resolve engine and placement without allocating or loading weights.
    pub fn route(&self, manifest: &ModelManifest) -> Result<EngineRoute> {
        let (engine, placement) = self.resolve(manifest)?;
        Ok(EngineRoute {
            engine_name: engine.engine_name().to_owned(),
            placement,
        })
    }

    /// Resolve the manifest to an engine, preflight the capability matrix and
    /// load a session held under a lease.
    pub fn load(&self, manifest_path: &Path) -> Result<SessionLease> {
        self.load_with_cancel(manifest_path, CancelToken::new())
    }

    /// Load a manifest path with cooperative cancellation.
    pub fn load_with_cancel(
        &self,
        manifest_path: &Path,
        cancel: CancelToken,
    ) -> Result<SessionLease> {
        let manifest = ModelManifest::load(manifest_path)?;
        let base_dir = ModelManifest::base_dir(manifest_path);
        self.load_manifest_inner(
            manifest,
            &base_dir,
            Some(manifest_path.to_path_buf()),
            cancel,
        )
    }

    /// Load a manifest that was produced in memory (for example by the
    /// inspector) without requiring a manifest file on disk.
    pub fn load_manifest(&self, manifest: ModelManifest, base_dir: &Path) -> Result<SessionLease> {
        self.load_manifest_with_cancel(manifest, base_dir, CancelToken::new())
    }

    /// Load with a cooperative cancellation token shared with the engine.
    pub fn load_manifest_with_cancel(
        &self,
        manifest: ModelManifest,
        base_dir: &Path,
        cancel: CancelToken,
    ) -> Result<SessionLease> {
        self.load_manifest_inner(manifest, base_dir, None, cancel)
    }

    fn load_manifest_inner(
        &self,
        manifest: ModelManifest,
        base_dir: &Path,
        manifest_path: Option<std::path::PathBuf>,
        cancel: CancelToken,
    ) -> Result<SessionLease> {
        let (engine, placement) = self.resolve(&manifest)?;
        let lease_placement = placement.clone();
        let estimated = estimate_residency(&manifest, base_dir, &lease_placement);
        let _load_gate = self
            .load_gate
            .lock()
            .map_err(|_| EngineError::Other("load gate poisoned".into()))?;
        if let Some(estimate) = estimated
            && !lease_placement.auto_offload
            && lease_placement.offload_layers == 0
        {
            self.ledger
                .check(&self.inventory, &lease_placement, estimate)?;
        }
        let spec = EngineLoadSpec {
            manifest,
            base_dir: base_dir.to_path_buf(),
            manifest_path,
            placement,
            budget: self.budget(),
            cancel,
        };
        let engine_name = engine.engine_name().to_owned();
        let manifest = spec.manifest.clone();
        let session = engine.load_session(spec)?;
        let residency = session.residency().or(estimated).unwrap_or_default();
        let reservation = self
            .ledger
            .reserve(&self.inventory, &lease_placement, residency)?;

        self.active.fetch_add(1, Ordering::SeqCst);
        Ok(SessionLease {
            session,
            engine_name,
            manifest,
            placement: lease_placement,
            residency: Some(residency),
            _reservation: reservation,
            active: Arc::clone(&self.active),
        })
    }

    fn resolve(
        &self,
        manifest: &ModelManifest,
    ) -> Result<(Arc<dyn InferenceEngine>, PlacementPlan)> {
        let mut saw_format = false;
        let mut saw_runtime = false;
        let mut saw_implementation = false;
        let mut saw_quantization = false;
        let mut saw_device = false;
        let mut saw_modality = false;

        for engine in &self.engines {
            if !engine.is_transport() {
                if engine.runtime() != manifest.runtime {
                    continue;
                }
                saw_runtime = true;
                if !engine
                    .implementations()
                    .iter()
                    .any(|name| *name == manifest.implementation)
                {
                    saw_implementation = true;
                    tracing::debug!(
                        implementation = %manifest.implementation,
                        "engine does not implement manifest"
                    );
                    continue;
                }
            }
            if !engine.supported_formats().contains(&manifest.format) {
                continue;
            }
            saw_format = true;

            let modalities = engine.supported_modalities(manifest.format);
            if let Some(missing) = manifest
                .input_modalities()
                .iter()
                .find(|modality| !modalities.contains(modality))
            {
                saw_modality = true;
                tracing::debug!(?missing, "engine does not support modality");
                continue;
            }

            if !engine
                .supported_quantizations(manifest.format)
                .contains(&manifest.quantization)
            {
                saw_quantization = true;
                continue;
            }

            match self.inventory.plan(manifest, engine.supported_devices()) {
                Ok(mut placement) => {
                    if (placement.offload_layers > 0 || placement.auto_offload)
                        && !engine.supports_layer_offload_for(&manifest.implementation)
                    {
                        saw_device = true;
                        tracing::debug!(
                            engine = engine.engine_name(),
                            "engine does not support layer offload"
                        );
                        continue;
                    }
                    if placement.kv_cache == uniinfer_core::KvCachePlacement::Ram
                        && !engine.supports_kv_cache_offload()
                    {
                        saw_device = true;
                        tracing::debug!(
                            engine = engine.engine_name(),
                            "engine does not support KV cache offload"
                        );
                        continue;
                    }
                    // Offload-capable engines use auto tiering when the manifest
                    // does not force a fixed layer count or CPU-only execution.
                    if engine.supports_layer_offload_for(&manifest.implementation)
                        && manifest.placement.offload_layers.is_none()
                        && placement
                            .devices
                            .first()
                            .is_some_and(|device| device.class == uniinfer_core::DeviceClass::Gpu)
                    {
                        placement.auto_offload = true;
                    }
                    let min_vram = if let Some(value) = manifest.min_vram.as_deref() {
                        Some(uniinfer_core::parse_memory_size(value).ok_or_else(|| {
                            EngineError::ManifestInvalid(format!("invalid min_vram: {value}"))
                        })?)
                    } else {
                        None
                    };
                    if placement.offload_layers == 0
                        && !placement.auto_offload
                        && let Some(required) = min_vram
                        && let Some(gpu) = placement
                            .devices
                            .iter()
                            .find(|device| device.class == uniinfer_core::DeviceClass::Gpu)
                        && gpu.available_memory_bytes > 0
                        && required > gpu.available_memory_bytes
                    {
                        saw_device = true;
                        tracing::debug!(
                            engine = engine.engine_name(),
                            required,
                            available = gpu.available_memory_bytes,
                            "min_vram does not fit free GPU memory"
                        );
                        continue;
                    }
                    return Ok((Arc::clone(engine), placement));
                }
                Err(EngineError::PlacementUnsatisfiable(reason)) => {
                    saw_device = true;
                    tracing::debug!(%reason, "engine placement unavailable");
                }
                Err(error) => return Err(error),
            }
        }

        if !saw_runtime {
            return Err(EngineError::UnsupportedEngine(format!(
                "no engine for runtime {:?}",
                manifest.runtime
            )));
        }
        if saw_implementation {
            return Err(EngineError::UnsupportedEngine(format!(
                "no engine implements {}",
                manifest.implementation
            )));
        }
        if !saw_format {
            return Err(EngineError::UnsupportedFormat(manifest.format));
        }
        if saw_modality {
            return Err(EngineError::UnsupportedModality {
                format: manifest.format,
                modality: manifest
                    .input_modalities()
                    .first()
                    .copied()
                    .unwrap_or(uniinfer_core::Modality::Text),
            });
        }
        if saw_quantization {
            return Err(EngineError::UnsupportedQuantization(manifest.quantization));
        }
        if saw_device {
            return Err(EngineError::PlacementUnsatisfiable(
                "no engine has an available device".into(),
            ));
        }
        Err(EngineError::UnsupportedEngine(
            "no engine can serve this manifest".into(),
        ))
    }
}

/// RAII session holder; releasing it updates the active-session gauge.
pub struct SessionLease {
    session: Box<dyn EngineSession>,
    engine_name: String,
    manifest: ModelManifest,
    placement: PlacementPlan,
    residency: Option<Residency>,
    _reservation: Reservation,
    active: Arc<AtomicUsize>,
}

impl SessionLease {
    #[must_use]
    pub fn session(&self) -> &dyn EngineSession {
        self.session.as_ref()
    }

    #[must_use]
    pub fn engine_name(&self) -> &str {
        &self.engine_name
    }

    #[must_use]
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    #[must_use]
    pub fn placement(&self) -> &PlacementPlan {
        &self.placement
    }

    #[must_use]
    pub fn residency(&self) -> Option<Residency> {
        self.residency
    }
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
        tracing::debug!(engine = %self.engine_name, "session lease released");
    }
}
