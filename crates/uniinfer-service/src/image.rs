//! Image-generation slot: one loaded session behind a single permit.

use std::sync::Arc;
use std::time::Duration;

use image::{ImageFormat, RgbImage};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use uniinfer_backends::{ByteTokenizer, HuggingFaceTokenizer, ModelEvent, ModelOutput, Tokenizer};
use uniinfer_core::{
    DiffusionConfig, EngineError, EngineSession, ImageConfig, ImageGenerationRequest,
    ModelManifest, NativeImageRequest, PlacementPlan, Residency, Result, Tensor, TensorData,
    TensorDtype,
};
use uniinfer_registry::SessionLease;

use crate::{ImageGenerationSpec, ModelExecutor, ModelRequest, ModelView};

pub struct ImageSlot {
    lease: Arc<SessionLease>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    image: ImageConfig,
    diffusion: Option<DiffusionConfig>,
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

    fn residency(&self) -> Option<Residency> {
        self.lease.residency()
    }
}

fn clip_tokenizer(base_dir: &std::path::Path) -> Result<Arc<dyn Tokenizer>> {
    let json = base_dir.join("tokenizer").join("tokenizer.json");
    if json.is_file() {
        return Ok(Arc::new(HuggingFaceTokenizer::from_file(json)?));
    }
    let vocab = base_dir.join("tokenizer").join("vocab.json");
    let merges = base_dir.join("tokenizer").join("merges.txt");
    if vocab.is_file() && merges.is_file() {
        return Ok(Arc::new(HuggingFaceTokenizer::from_vocab_merges(
            vocab, merges,
        )?));
    }
    Err(EngineError::ManifestInvalid(
        "image model requires tokenizer/tokenizer.json or tokenizer/vocab.json + merges.txt".into(),
    ))
}

impl ImageSlot {
    pub(crate) fn new(lease: Arc<SessionLease>, base_dir: &std::path::Path) -> Result<Self> {
        let manifest = lease.manifest();
        let image = manifest
            .image
            .clone()
            .ok_or_else(|| EngineError::ManifestInvalid("image config required".into()))?;
        let diffusion = image.diffusion.clone();
        let session = lease.session();
        let native_image = session.native_image_generator().is_some();
        let tokenizer = if native_image {
            None
        } else {
            let text = manifest.text.as_ref();
            let configured = manifest.tokenizer.as_deref().unwrap_or_default();
            Some(match configured {
                "bytes" => {
                    let text = text.ok_or_else(|| {
                        EngineError::ManifestInvalid("image model requires [text]".into())
                    })?;
                    Arc::new(ByteTokenizer::new(text.vocab_size)?) as Arc<dyn Tokenizer>
                }
                "hf" => {
                    let file = text.and_then(|text| text.tokenizer_file.as_ref());
                    if let Some(file) = file {
                        Arc::new(HuggingFaceTokenizer::from_file(base_dir.join(file))?)
                            as Arc<dyn Tokenizer>
                    } else if diffusion.is_some() {
                        clip_tokenizer(base_dir)?
                    } else {
                        return Err(EngineError::ManifestInvalid(
                            "text.tokenizer_file is required for hf".into(),
                        ));
                    }
                }
                "" if diffusion.is_some() => clip_tokenizer(base_dir)?,
                other => {
                    return Err(EngineError::UnsupportedEngine(format!(
                        "tokenizer {other:?}"
                    )));
                }
            })
        };
        if !native_image && diffusion.is_none() {
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
        }
        Ok(Self {
            lease,
            tokenizer,
            image,
            diffusion,
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

    fn generate(
        &self,
        spec: &ImageGenerationSpec,
        cancel: &uniinfer_core::CancelToken,
        on_event: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput> {
        if let Some(generator) = self.session().native_image_generator() {
            let mut request = NativeImageRequest {
                prompt: if spec.prompt.is_empty() {
                    " "
                } else {
                    &spec.prompt
                },
                negative_prompt: spec.negative_prompt.as_deref(),
                width: spec.width.max(64),
                height: spec.height.max(64),
                steps: spec.steps,
                guidance: spec.guidance,
                seed: spec.seed,
                deadline: None,
                cancel,
                progress: &mut |step, total| on_event(ModelEvent::Progress { step, total }),
            };
            let image = generator.generate_image(&mut request)?;
            return Ok(ModelOutput::Image(uniinfer_backends::ImageOutput {
                mime: image.mime,
                bytes: image.bytes,
                width: image.width,
                height: image.height,
            }));
        }
        if let Some(diffusion) = &self.diffusion {
            let prompt = if spec.prompt.is_empty() {
                " "
            } else {
                &spec.prompt
            };
            let negative = spec.negative_prompt.as_deref().unwrap_or("");
            let prompt_ids = self.padded(prompt, diffusion.max_length, diffusion.pad_token)?;
            let negative_ids = self.padded(negative, diffusion.max_length, diffusion.pad_token)?;
            let mut progress = |step, total, preview: Option<Tensor>| {
                on_event(ModelEvent::Progress { step, total });
                if let Some(tensor) = preview
                    && let Ok(ModelOutput::Image(image)) = encode_image(tensor)
                {
                    on_event(ModelEvent::Preview { step, total, image });
                }
            };
            let mut request = ImageGenerationRequest {
                prompt_ids: &prompt_ids,
                negative_ids: &negative_ids,
                width: spec.width.max(64),
                height: spec.height.max(64),
                steps: if spec.steps == 0 {
                    diffusion.steps
                } else {
                    spec.steps
                },
                guidance: spec.guidance,
                seed: spec.seed,
                cancel,
                preview: spec.preview,
                progress: &mut progress,
            };
            let image = self.session().generate_image(&mut request)?;
            return encode_image(image);
        }
        let prompt = if spec.prompt.is_empty() {
            " "
        } else {
            &spec.prompt
        };
        let ids = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Other("image tokenizer unavailable".into()))?
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
        encode_image(image)
    }

    fn padded(&self, prompt: &str, max_length: usize, pad: u32) -> Result<Vec<i64>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Other("image tokenizer unavailable".into()))?;
        let pad = tokenizer.padding_token().unwrap_or(pad);
        let mut ids = tokenizer
            .encode(prompt)?
            .into_iter()
            .map(i64::from)
            .take(max_length)
            .collect::<Vec<_>>();
        ids.resize(max_length, i64::from(pad));
        Ok(ids)
    }

    fn session(&self) -> &dyn EngineSession {
        self.lease.session()
    }
}

fn encode_image(image: Tensor) -> Result<ModelOutput> {
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
    // SD-family VAEs return [-1, 1]; other image models commonly return
    // [0, 1]. Detect the range instead of assuming one convention, otherwise
    // a perfectly valid VAE output is clipped to near-black.
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let rescale = min < -0.001;
    let mut rgb = RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let pixel = [
                to_byte(values[index], rescale),
                to_byte(values[height * width + index], rescale),
                to_byte(values[2 * height * width + index], rescale),
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

fn to_byte(value: f32, rescale: bool) -> u8 {
    let value = if rescale { value * 0.5 + 0.5 } else { value };
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl ModelExecutor for ImageSlot {
    fn execute(
        &self,
        request: ModelRequest,
        cancel: &uniinfer_core::CancelToken,
        on_event: &mut dyn FnMut(ModelEvent),
    ) -> Result<ModelOutput> {
        match request {
            ModelRequest::Image { spec } => self.generate(&spec, cancel, on_event),
            _ => Err(EngineError::UnsupportedModality {
                format: self.manifest().format,
                modality: uniinfer_core::Modality::Text,
            }),
        }
    }
}
