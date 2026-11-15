//! Qwen3-VL dense vision-language session.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use uniinfer_core::{
    EngineError, EngineLoadSpec, EngineSession, Result, Tensor as CoreTensor, TensorData,
    TensorDtype, TensorSpec,
};

use super::ImageProcessor;

#[allow(unsafe_code)] // mmap of checksum-verified weights.
pub(crate) fn load(
    spec: &EngineLoadSpec,
    dtype: DType,
    device: &Device,
    config_path: &Path,
    model_paths: &[std::path::PathBuf],
) -> Result<Box<dyn EngineSession>> {
    if spec.manifest.format != uniinfer_core::ModelFormat::Safetensors {
        return Err(EngineError::UnsupportedFormat(spec.manifest.format));
    }
    let raw = std::fs::read_to_string(config_path)
        .map_err(|error| EngineError::LoadFailed(format!("vlm config: {error}")))?;
    let config: Config = serde_json::from_str(&raw)
        .map_err(|error| EngineError::LoadFailed(format!("vlm config: {error}")))?;
    let dtype = if device.is_cpu() && dtype == DType::BF16 {
        DType::F16
    } else {
        dtype
    };
    let paths = model_paths
        .iter()
        .map(std::path::PathBuf::as_path)
        .collect::<Vec<_>>();
    let builder = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device) }
        .map_err(|error| EngineError::LoadFailed(format!("vlm weights: {error}")))?;
    let model = Qwen3VLModel::new(&config, builder)
        .map_err(|error| EngineError::LoadFailed(format!("qwen3-vl: {error}")))?;
    let preprocessor =
        ImageProcessor::qwen3_vl(&config_path.with_file_name("preprocessor_config.json"))?;
    Ok(Box::new(Qwen3VlSession {
        model: Mutex::new(model),
        image_token_id: i64::from(config.image_token_id),
        preprocessor,
        device: device.clone(),
        inputs: vec![
            TensorSpec {
                name: "input_ids".into(),
                dtype: TensorDtype::I64,
                shape: vec![Some(1), None],
            },
            TensorSpec {
                name: "pixel_values".into(),
                dtype: TensorDtype::F32,
                shape: vec![None, None],
            },
            TensorSpec {
                name: "image_grid_thw".into(),
                dtype: TensorDtype::I64,
                shape: vec![Some(1), Some(3)],
            },
            TensorSpec {
                name: "image".into(),
                dtype: TensorDtype::U8,
                shape: vec![None],
            },
        ],
        outputs: vec![TensorSpec {
            name: "logits".into(),
            dtype: TensorDtype::F32,
            shape: vec![Some(1), None, Some(151_936)],
        }],
    }))
}

struct Qwen3VlSession {
    model: Mutex<Qwen3VLModel>,
    image_token_id: i64,
    preprocessor: ImageProcessor,
    device: Device,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl Qwen3VlSession {
    fn run_inner(&self, inputs: &[CoreTensor]) -> Result<Vec<CoreTensor>> {
        let ids = inputs
            .iter()
            .find(|tensor| tensor.name == "input_ids")
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) => Some(values.clone()),
                _ => None,
            })
            .ok_or_else(|| EngineError::Other("missing int64 input_ids".into()))?;
        let mut shapes = inputs
            .iter()
            .find(|tensor| tensor.name == "pixel_values")
            .and_then(|tensor| match &tensor.data {
                TensorData::F32(values) => Some(values.clone()),
                _ => None,
            });
        let mut grid = inputs
            .iter()
            .find(|tensor| tensor.name == "image_grid_thw")
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) => Some(values.clone()),
                _ => None,
            });
        let mut image_tokens = None;
        if shapes.is_none()
            && let Some(bytes) = inputs
                .iter()
                .find(|tensor| tensor.name == "image")
                .and_then(|tensor| match &tensor.data {
                    TensorData::U8(values) => Some(values.clone()),
                    _ => None,
                })
        {
            let image = self.preprocessor.preprocess(&bytes)?;
            image_tokens = Some(image.image_tokens);
            shapes = Some(image.pixel_values);
            grid = Some(image.grid_thw.to_vec());
        }
        if let Some(pixel_values) = shapes {
            let grid = grid.ok_or_else(|| EngineError::Other("missing image_grid_thw".into()))?;
            if grid.len() != 3 {
                return Err(EngineError::Other(
                    "image_grid_thw must have 3 values".into(),
                ));
            }
            let merge = self.preprocessor.merge_size() as i64;
            let tokens =
                image_tokens.unwrap_or_else(|| ((grid[1] / merge) * (grid[2] / merge)) as usize);
            let spans = image_spans(&ids, self.image_token_id).map_err(EngineError::Other)?;
            if spans.iter().map(|(s, e)| e - s).sum::<usize>() != tokens {
                return Err(EngineError::Other(
                    "image placeholder count does not match image grid".into(),
                ));
            }
            let rows = pixel_values.len() / (3 * 2 * 16 * 16);
            let pixel_values = Tensor::from_vec(pixel_values, (rows, 1536), &self.device)
                .map_err(|error| EngineError::Other(format!("pixel_values: {error}")))?;
            let grid = grid
                .into_iter()
                .map(|value| u32::try_from(value).unwrap_or(0))
                .collect::<Vec<_>>();
            let grid = Tensor::from_vec(grid, (1, 3), &self.device)
                .map_err(|error| EngineError::Other(format!("image_grid_thw: {error}")))?;
            let ids = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)
                .map_err(|error| EngineError::Other(format!("input_ids: {error}")))?;
            let model = self
                .model
                .lock()
                .map_err(|_| EngineError::Other("vlm state lock poisoned".into()))?;
            let logits = model
                .forward(
                    &ids,
                    Some(pixel_values),
                    None,
                    Some(grid),
                    None,
                    vec![ids.dim(1).unwrap_or(0)],
                    vec![spans],
                    vec![],
                    &[0],
                )
                .map_err(|error| EngineError::Other(format!("qwen3-vl forward: {error}")))?;
            let logits = logits
                .to_dtype(DType::F32)
                .map_err(|error| EngineError::Other(format!("vlm logits dtype: {error}")))?;
            let shape = logits.dims().to_vec();
            let values = logits
                .flatten_all()
                .and_then(|tensor| tensor.to_vec1::<f32>())
                .map_err(|error| EngineError::Other(format!("vlm logits: {error}")))?;
            return Ok(vec![CoreTensor::new(
                "logits",
                shape,
                TensorData::F32(values),
            )?]);
        }
        let ids_len = ids.len();
        let ids = Tensor::from_vec(ids, (1, ids_len), &self.device)
            .map_err(|error| EngineError::Other(format!("input_ids: {error}")))?;
        let model = self
            .model
            .lock()
            .map_err(|_| EngineError::Other("vlm state lock poisoned".into()))?;
        let logits = model
            .forward(
                &ids,
                None,
                None,
                None,
                None,
                vec![ids_len],
                vec![vec![]],
                vec![],
                &[0],
            )
            .map_err(|error| EngineError::Other(format!("qwen3-vl forward: {error}")))?;
        let logits = logits
            .to_dtype(DType::F32)
            .map_err(|error| EngineError::Other(format!("vlm logits dtype: {error}")))?;
        let shape = logits.dims().to_vec();
        let values = logits
            .flatten_all()
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|error| EngineError::Other(format!("vlm logits: {error}")))?;
        Ok(vec![CoreTensor::new(
            "logits",
            shape,
            TensorData::F32(values),
        )?])
    }
}

impl EngineSession for Qwen3VlSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<CoreTensor>) -> Result<Vec<CoreTensor>> {
        self.run_inner(&inputs)
    }

    fn cancel(&self) {}
}

fn image_spans(ids: &[i64], image_token: i64) -> std::result::Result<Vec<(usize, usize)>, String> {
    let mut spans = Vec::new();
    let mut start = None;
    for (index, id) in ids.iter().enumerate() {
        if *id == image_token {
            start.get_or_insert(index);
        } else if let Some(start) = start.take() {
            spans.push((start, index));
        }
    }
    if let Some(start) = start {
        spans.push((start, ids.len()));
    }
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::ImageProcessor;
    use uniinfer_core::{
        DeviceBudget, DeviceClass, EngineLoadSpec, InferenceEngine, ModelManifest,
    };

    #[test]
    #[ignore = "loads the 4 GiB Qwen3-VL-2B fixture"]
    fn qwen3_vl_accepts_image_and_returns_logits() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3-vl-2b");
        let manifest = ModelManifest::load(&dir.join("manifest.toml")).expect("manifest");
        let inventory = uniinfer_devices::DeviceInventory::probe();
        let placement = inventory
            .plan(&manifest, &[DeviceClass::Cpu])
            .expect("placement");
        let engine = crate::CandleEngine::new().expect("engine");
        let session = engine
            .load_session(EngineLoadSpec {
                manifest,
                base_dir: dir.clone(),
                manifest_path: Some(dir.join("manifest.toml")),
                placement,
                budget: DeviceBudget::default(),
                cancel: uniinfer_core::CancelToken::new(),
            })
            .expect("load");

        let processor =
            ImageProcessor::qwen3_vl(&dir.join("preprocessor_config.json")).expect("processor");
        let mut png = Vec::new();
        let mut image = image::RgbImage::new(64, 64);
        for pixel in image.pixels_mut() {
            *pixel = image::Rgb([200, 80, 30]);
        }
        image
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("png");
        let pre = processor.preprocess(&png).expect("preprocess");

        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
        let prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n";
        let mut ids = tokenizer
            .encode(prompt, true)
            .expect("encode")
            .get_ids()
            .iter()
            .map(|id| i64::from(*id))
            .collect::<Vec<_>>();
        let position = ids
            .iter()
            .position(|id| *id == 151655)
            .expect("image token");
        ids.splice(
            position..=position,
            std::iter::repeat_n(151655, pre.image_tokens),
        );
        let input_ids =
            CoreTensor::new("input_ids", vec![1, ids.len()], TensorData::I64(ids)).expect("ids");
        let image_tensor =
            CoreTensor::new("image", vec![png.len()], TensorData::U8(png)).expect("image");
        let outputs = session.run(vec![input_ids, image_tensor]).expect("run");
        assert_eq!(outputs[0].name, "logits");
        let TensorData::F32(logits) = &outputs[0].data else {
            panic!("expected logits");
        };
        assert!(logits.iter().any(|value| value.is_finite()));
    }
}
