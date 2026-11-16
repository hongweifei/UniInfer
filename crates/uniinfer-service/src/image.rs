//! Image-generation slot: one loaded session behind a single permit.

use std::sync::Arc;
use std::time::Duration;

use image::{ImageFormat, RgbImage};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_backends::{ByteTokenizer, HuggingFaceTokenizer, ModelEvent, ModelOutput, Tokenizer};
use uniinfer_core::{
    EngineError, EngineSession, ImageConfig, ModelManifest, PlacementPlan, Result, Tensor,
    TensorData, TensorDtype,
};
use uniinfer_registry::SessionLease;

use crate::{ImageGenerationSpec, ModelExecutor, ModelRequest, ModelView};

pub struct ImageSlot {
    lease: Arc<SessionLease>,
    tokenizer: Arc<dyn Tokenizer>,
    image: ImageConfig,
    permit: Arc<Semaphore>,
}

impl ModelView for ImageSlot {
    fn manifest(&self) -> &ModelManifest {
        ImageSlot::manifest(self)
    }

    fn placement(&self) -> &PlacementPlan {
        ImageSlot::placement(self)
    }

    fn engine_name(&self) -> &str {
        ImageSlot::engine_name(self)
    }
}

impl ImageSlot {
    pub(crate) fn new(lease: Arc<SessionLease>, base_dir: &std::path::Path) -> Result<Self> {
        let manifest = lease.manifest();
        let text = manifest
            .text
            .as_ref()
            .ok_or_else(|| EngineError::ManifestInvalid("image model requires [text]".into()))?;
        let tokenizer: Arc<dyn Tokenizer> = match manifest.tokenizer.as_deref().unwrap_or_default()
        {
            "bytes" => Arc::new(ByteTokenizer::new(text.vocab_size)?),
            "hf" => {
                let file = text.tokenizer_file.as_ref().ok_or_else(|| {
                    EngineError::ManifestInvalid("text.tokenizer_file is required".into())
                })?;
                Arc::new(HuggingFaceTokenizer::from_file(base_dir.join(file))?)
            }
            other => {
                return Err(EngineError::UnsupportedEngine(format!(
                    "tokenizer {other:?}"
                )));
            }
        };
        let image = manifest
            .image
            .clone()
            .ok_or_else(|| EngineError::ManifestInvalid("image config required".into()))?;
        let session = lease.session();
        if !session
            .inputs()
            .iter()
            .any(|spec| spec.name == "input_ids" && spec.dtype == TensorDtype::I64)
        {
            return Err(EngineError::UnsupportedEngine(
                "image model requires int64 input_ids".into(),
            ));
        }
        if !session
            .outputs()
            .iter()
            .any(|spec| spec.name == "image" && spec.dtype == TensorDtype::F32)
        {
            return Err(EngineError::UnsupportedEngine(
                "image model requires f32 image output".into(),
            ));
        }
        Ok(Self {
            lease,
            tokenizer,
            image,
            permit: Arc::new(Semaphore::new(1)),
        })
    }

    #[must_use]
    pub fn image_config(&self) -> &ImageConfig {
        &self.image
    }

    #[must_use]
    pub fn manifest(&self) -> &ModelManifest {
        self.lease.manifest()
    }

    #[must_use]
    pub fn placement(&self) -> &PlacementPlan {
        self.lease.placement()
    }

    #[must_use]
    pub fn engine_name(&self) -> &str {
        self.lease.engine_name()
    }

    pub async fn acquire(&self, wait: Duration) -> Result<OwnedSemaphorePermit> {
        crate::acquire_permit(&self.permit, wait).await
    }

    fn generate(&self, spec: &ImageGenerationSpec) -> Result<ModelOutput> {
        let prompt = if spec.prompt.is_empty() {
            " "
        } else {
            &spec.prompt
        };
        let ids = self
            .tokenizer
            .encode(prompt)?
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        let output = self.session().run(vec![Tensor::new(
            "input_ids",
            vec![1, ids.len()],
            TensorData::I64(ids),
        )?])?;
        let image = output
            .into_iter()
            .find(|tensor| tensor.name == "image")
            .ok_or_else(|| EngineError::Other("image output missing".into()))?;
        let TensorData::F32(values) = image.data else {
            return Err(EngineError::Other("image output must be f32".into()));
        };
        let (channels, height, width) = match image.shape.as_slice() {
            [1, channels, height, width] => (*channels, *height, *width),
            [channels, height, width] => (*channels, *height, *width),
            _ => {
                return Err(EngineError::Other(format!(
                    "unexpected image shape {:?}",
                    image.shape
                )));
            }
        };
        if channels != 3 || height == 0 || width == 0 {
            return Err(EngineError::Other(format!(
                "unsupported image shape {:?}",
                image.shape
            )));
        }
        let mut rgb = RgbImage::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                let pixel = [
                    to_byte(values[index]),
                    to_byte(values[height * width + index]),
                    to_byte(values[2 * height * width + index]),
                ];
                rgb.put_pixel(x as u32, y as u32, image::Rgb(pixel));
            }
        }
        let mut png = Vec::new();
        rgb.write_to(&mut std::io::Cursor::new(&mut png), ImageFormat::Png)
            .map_err(|error| EngineError::Other(format!("png encode: {error}")))?;
        Ok(ModelOutput::Image(uniinfer_backends::ImageOutput {
            mime: "image/png".into(),
            bytes: png,
            width: width as u32,
            height: height as u32,
        }))
    }

    fn session(&self) -> &dyn EngineSession {
        self.lease.session()
    }
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl ModelExecutor for ImageSlot {
    fn execute(
        &self,
        request: ModelRequest,
        _cancel: &uniinfer_core::CancelToken,
        _on_event: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput> {
        match request {
            ModelRequest::Image { spec } => self.generate(&spec),
            _ => Err(EngineError::UnsupportedModality {
                format: self.manifest().format,
                modality: uniinfer_core::Modality::Text,
            }),
        }
    }
}
