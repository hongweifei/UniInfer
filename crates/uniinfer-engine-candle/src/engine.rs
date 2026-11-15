//! Candle engine entry point and manifest-driven family dispatch.

use candle_core::DType;
use uniinfer_core::{
    Device as CoreDevice, DeviceClass, EngineError, EngineLoadSpec, EngineSession, InferenceEngine,
    Modality, ModelFormat, Quantization, Result,
};

use crate::device::{device_for, probe_device_ids, probe_devices};
use crate::session::CandleSession;
use crate::text::{self, LoadContext, TextRegistry};
use crate::vision;

const FORMATS: &[ModelFormat] = &[ModelFormat::Gguf, ModelFormat::Safetensors];
const MODALITIES: &[Modality] = &[Modality::Text];

pub struct CandleEngine {
    devices: Vec<DeviceClass>,
    text: TextRegistry,
}

impl CandleEngine {
    pub fn new() -> Result<Self> {
        Ok(Self {
            devices: probe_devices(),
            text: TextRegistry::new(),
        })
    }

    #[must_use]
    pub fn probe_devices(&self) -> Vec<CoreDevice> {
        probe_device_ids(&self.devices)
    }
}

impl InferenceEngine for CandleEngine {
    fn engine_name(&self) -> &'static str {
        "candle"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        FORMATS
    }

    fn supported_modalities(&self, format: ModelFormat) -> &[Modality] {
        match format {
            ModelFormat::Gguf => MODALITIES,
            ModelFormat::Safetensors => &[Modality::Text, Modality::Image],
            _ => &[],
        }
    }

    fn supported_quantizations(&self, format: ModelFormat) -> &[Quantization] {
        match format {
            ModelFormat::Gguf => text::GGUF_QUANTIZATIONS,
            ModelFormat::Safetensors => text::SAFETENSORS_QUANTIZATIONS,
            _ => &[],
        }
    }

    fn supports_architecture(&self, manifest: &uniinfer_core::ModelManifest) -> bool {
        let Some(family) = manifest
            .text
            .as_ref()
            .and_then(|text| text.family.as_deref())
        else {
            return false;
        };
        if manifest.modalities.contains(&Modality::Image) {
            vision::supports(family)
        } else {
            self.text.supported(family)
        }
    }

    fn supported_devices(&self) -> &[DeviceClass] {
        &self.devices
    }

    fn load_session(&self, spec: EngineLoadSpec) -> Result<Box<dyn EngineSession>> {
        if !FORMATS.contains(&spec.manifest.format) {
            return Err(EngineError::UnsupportedFormat(spec.manifest.format));
        }
        if !self
            .supported_quantizations(spec.manifest.format)
            .contains(&spec.manifest.quantization)
        {
            return Err(EngineError::UnsupportedQuantization(
                spec.manifest.quantization,
            ));
        }
        let class = spec
            .placement
            .devices
            .first()
            .map(|device| device.class)
            .ok_or_else(|| EngineError::PlacementUnsatisfiable("empty placement".into()))?;
        if !self.devices.contains(&class) {
            return Err(EngineError::PlacementUnsatisfiable(format!(
                "candle has no {class} device"
            )));
        }
        let device = device_for(class)?;
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        spec.manifest.verify_files(&spec.base_dir)?;

        let text = spec
            .manifest
            .text
            .as_ref()
            .ok_or_else(|| EngineError::ManifestInvalid("text config required".into()))?;
        let family_name = text
            .family
            .as_deref()
            .ok_or_else(|| EngineError::ManifestInvalid("text.family is required".into()))?;
        let model_paths = match spec.manifest.format {
            ModelFormat::Gguf => spec
                .manifest
                .files
                .first()
                .map(|file| vec![spec.base_dir.join(&file.path)])
                .ok_or_else(|| EngineError::ManifestInvalid("files is empty".into()))?,
            ModelFormat::Safetensors => {
                let shards = spec
                    .manifest
                    .files
                    .iter()
                    .map(|file| spec.base_dir.join(&file.path))
                    .filter(|path| {
                        path.extension()
                            .is_some_and(|extension| extension == "safetensors")
                    })
                    .collect::<Vec<_>>();
                if shards.is_empty() {
                    return Err(EngineError::ManifestInvalid(
                        "no .safetensors shards in files".into(),
                    ));
                }
                shards
            }
            _ => unreachable!(),
        };
        let dtype = match spec.manifest.quantization {
            Quantization::Bf16 => DType::BF16,
            Quantization::Fp16 => DType::F16,
            _ => DType::F32,
        };
        let config_path = spec.base_dir.join(
            text.config_file
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new("")),
        );
        if vision::supports(family_name) {
            vision::verify_safetensors(spec.manifest.format)?;
            return vision::load(
                &spec,
                family_name,
                dtype,
                &device,
                &config_path,
                &model_paths,
            );
        }
        let family =
            self.text
                .get(family_name)
                .ok_or_else(|| EngineError::UnsupportedArchitecture {
                    format: spec.manifest.format,
                    architecture: family_name.to_owned(),
                })?;
        let model = family.load(&LoadContext {
            format: spec.manifest.format,
            model_paths: &model_paths,
            config_path: &config_path,
            dtype,
            device: &device,
            cancel: &spec.cancel,
        })?;
        if spec.cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if text.vocab_size == 0 {
            return Err(EngineError::ManifestInvalid(
                "text.vocab_size is required".into(),
            ));
        }
        Ok(Box::new(CandleSession::new(model, device, text.vocab_size)))
    }
}
