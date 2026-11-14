//! Engine routing, preflight checks and session leases.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use uniinfer_core::{
    EngineError, EngineLoadSpec, EngineSession, InferenceEngine, ModelManifest, PlacementPlan,
    Result,
};
use uniinfer_devices::DeviceInventory;

/// Owns registered engines and hands out leases.
pub struct EngineRegistry {
    engines: Vec<Arc<dyn InferenceEngine>>,
    inventory: DeviceInventory,
    active: Arc<AtomicUsize>,
}

impl EngineRegistry {
    #[must_use]
    pub fn new(inventory: DeviceInventory) -> Self {
        Self {
            engines: Vec::new(),
            inventory,
            active: Arc::new(AtomicUsize::new(0)),
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

    /// Resolve the manifest to an engine, preflight the capability matrix and
    /// load a session held under a lease.
    pub fn load(&self, manifest_path: &Path) -> Result<SessionLease> {
        let manifest = ModelManifest::load(manifest_path)?;
        let base_dir = ModelManifest::base_dir(manifest_path);

        let placement = self.inventory.plan(&manifest)?;
        let engine = self.resolve(&manifest, &placement)?;

        let lease_placement = placement.clone();
        let spec = EngineLoadSpec {
            manifest,
            base_dir,
            placement,
            budget: self.inventory.budget(),
        };
        let engine_name = engine.engine_name().to_owned();
        let manifest = spec.manifest.clone();
        let session = engine.load_session(spec)?;

        self.active.fetch_add(1, Ordering::SeqCst);
        Ok(SessionLease {
            session,
            engine_name,
            manifest,
            placement: lease_placement,
            active: Arc::clone(&self.active),
        })
    }

    fn resolve(
        &self,
        manifest: &ModelManifest,
        placement: &PlacementPlan,
    ) -> Result<Arc<dyn InferenceEngine>> {
        let mut saw_format = false;
        let mut saw_quantization = false;
        let mut saw_device = false;
        let mut saw_modality = false;
        let mut saw_engine = false;

        for engine in &self.engines {
            if !manifest.supported_engines.is_empty()
                && !manifest
                    .supported_engines
                    .iter()
                    .any(|name| name == engine.engine_name())
            {
                continue;
            }
            saw_engine = true;

            if !engine.supported_formats().contains(&manifest.format) {
                continue;
            }
            saw_format = true;

            let modalities = engine.supported_modalities(manifest.format);
            if let Some(missing) = manifest
                .modalities
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

            let devices = engine.supported_devices();
            if !placement
                .devices
                .iter()
                .all(|device| devices.contains(&device.class))
            {
                saw_device = true;
                continue;
            }

            return Ok(Arc::clone(engine));
        }

        if !saw_engine {
            return Err(EngineError::UnsupportedEngine(
                manifest.supported_engines.join(","),
            ));
        }
        if !saw_format {
            return Err(EngineError::UnsupportedFormat(manifest.format));
        }
        if saw_quantization {
            return Err(EngineError::UnsupportedQuantization(manifest.quantization));
        }
        if saw_modality {
            return Err(EngineError::UnsupportedModality {
                format: manifest.format,
                modality: manifest.modalities[0],
            });
        }
        if saw_device {
            return Err(EngineError::PlacementUnsatisfiable(format!(
                "engine devices do not cover {:?}",
                placement.devices
            )));
        }
        Err(EngineError::UnsupportedEngine(format!(
            "no engine can serve {:?} {:?}",
            manifest.format, manifest.modalities
        )))
    }
}

/// RAII session holder; releasing it updates the active-session gauge.
pub struct SessionLease {
    session: Box<dyn EngineSession>,
    engine_name: String,
    manifest: ModelManifest,
    placement: PlacementPlan,
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
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
        tracing::debug!(engine = %self.engine_name, "session lease released");
    }
}
