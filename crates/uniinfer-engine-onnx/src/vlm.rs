//! ONNX vision-language adapter for split graph exports.
//!
//! The export contract is:
//! - vision graph: `pixel_values` + `pixel_attention_mask` -> image features
//! - embedding graph: `input_ids` -> `inputs_embeds`
//! - decoder graph: `inputs_embeds` + masks + empty KV cache -> logits

use std::path::Path;
use std::sync::Mutex;

use image::GenericImageView;
use image::imageops::FilterType;
use ort::environment::Environment;
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, Result, Tensor, TensorData,
    TensorDtype, TensorSpec,
};

use crate::graph::Graph;

const HIDDEN: usize = 576;

pub(crate) fn load(
    environment: &Environment,
    spec: &EngineLoadSpec,
    class: DeviceClass,
) -> Result<Box<dyn EngineSession>> {
    let vision = spec
        .manifest
        .vision
        .as_ref()
        .ok_or_else(|| EngineError::ManifestInvalid("vision config required".into()))?;
    let vision_graph = required(&vision.vision_graph, "vision.vision_graph")?;
    let embed_graph = required(&vision.embed_graph, "vision.embed_graph")?;
    let text_graph = required(&vision.text_graph, "vision.text_graph")?;
    let preprocessor = required(&vision.preprocessor_file, "vision.preprocessor_file")?;
    let image_token_id = vision
        .image_token_id
        .ok_or_else(|| EngineError::ManifestInvalid("vision.image_token_id required".into()))?;
    let image_seq_len = vision
        .image_seq_len
        .ok_or_else(|| EngineError::ManifestInvalid("vision.image_seq_len required".into()))?;
    let global_id = vision.global_image_token_id.unwrap_or(0);
    let fake_id = vision.fake_image_token_id.unwrap_or(0);
    if image_seq_len == 0 {
        return Err(EngineError::ManifestInvalid(
            "vision.image_seq_len must be positive".into(),
        ));
    }

    let pixel_input = vision
        .pixel_input
        .clone()
        .unwrap_or_else(|| "pixel_values".into());
    let mask_input = vision
        .grid_input
        .clone()
        .unwrap_or_else(|| "pixel_attention_mask".into());
    let processor = Preprocessor::load(&spec.base_dir.join(preprocessor))?;
    let vision_graph = Graph::load(
        environment,
        class,
        &spec.base_dir.join(vision_graph),
        "onnx-vlm-vision",
    )?;
    let embed_graph = Graph::load(
        environment,
        class,
        &spec.base_dir.join(embed_graph),
        "onnx-vlm-embed",
    )?;
    let decoder_graph = Graph::load(
        environment,
        class,
        &spec.base_dir.join(text_graph),
        "onnx-vlm-decoder",
    )?;
    let vocab_size = decoder_graph
        .outputs()
        .iter()
        .find(|spec| spec.name == "logits")
        .and_then(|spec| spec.shape.last().copied().flatten())
        .unwrap_or(49_280);
    Ok(Box::new(VlmSession {
        state: Mutex::new(VlmState {
            vision_graph,
            embed_graph,
            decoder_graph,
            processor,
            pixel_input,
            mask_input,
            image_token_id: i64::from(image_token_id),
            image_seq_len,
            global_id: i64::from(global_id),
            fake_id: i64::from(fake_id),
            cached_image: None,
        }),
        inputs: vec![
            TensorSpec {
                name: "input_ids".into(),
                dtype: TensorDtype::I64,
                shape: vec![Some(1), None],
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
            shape: vec![Some(1), None, Some(vocab_size)],
        }],
    }))
}

fn required<'a>(value: &'a Option<std::path::PathBuf>, field: &str) -> Result<&'a Path> {
    value
        .as_deref()
        .ok_or_else(|| EngineError::ManifestInvalid(format!("{field} required")))
}

struct VlmState {
    vision_graph: Graph,
    embed_graph: Graph,
    decoder_graph: Graph,
    processor: Preprocessor,
    pixel_input: String,
    mask_input: String,
    image_token_id: i64,
    image_seq_len: usize,
    global_id: i64,
    fake_id: i64,
    cached_image: Option<(Vec<u8>, Vec<f32>)>,
}

struct VlmSession {
    state: Mutex<VlmState>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl EngineSession for VlmSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let ids = inputs
            .iter()
            .find(|tensor| tensor.name == "input_ids")
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) => Some(values.clone()),
                _ => None,
            })
            .ok_or_else(|| EngineError::Other("missing int64 input_ids".into()))?;
        let image = inputs.iter().find_map(|tensor| match &tensor.data {
            TensorData::U8(bytes) if tensor.name == "image" => Some(bytes.clone()),
            _ => None,
        });
        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("onnx vlm state poisoned".into()))?;
        let features = match image {
            Some(bytes) => Some(state.image_features(&bytes)?),
            None => None,
        };
        let VlmState {
            embed_graph,
            decoder_graph,
            image_token_id,
            image_seq_len,
            global_id,
            fake_id,
            ..
        } = &mut *state;
        let expanded =
            expand_image_tokens(ids, *image_token_id, *image_seq_len, *global_id, *fake_id)?;
        let seq = expanded.len();
        let embed = embed_graph.run(vec![Tensor::new(
            "input_ids",
            vec![1, seq],
            TensorData::I64(expanded.clone()),
        )?])?;
        let mut embeds = embed
            .into_iter()
            .find(|tensor| tensor.name == "inputs_embeds")
            .ok_or_else(|| EngineError::Other("embed graph has no inputs_embeds".into()))?;
        if let Some(features) = features {
            splice_image_features(&mut embeds, &expanded, *image_token_id, &features)?;
        }
        let seq = expanded.len();
        let mut values = vec![
            Tensor::new("inputs_embeds", vec![1, seq, HIDDEN], embeds.data.clone())?,
            Tensor::new(
                "attention_mask",
                vec![1, seq],
                TensorData::I64(vec![1; seq]),
            )?,
            Tensor::new(
                "position_ids",
                vec![1, seq],
                TensorData::I64((0..seq as i64).collect()),
            )?,
        ];
        values.extend(empty_kv_cache(decoder_graph)?);
        let outputs = decoder_graph.run(values)?;
        let logits = outputs
            .into_iter()
            .find(|tensor| tensor.name == "logits")
            .ok_or_else(|| EngineError::Other("decoder graph has no logits".into()))?;
        Ok(vec![logits])
    }

    fn cancel(&self) {}
}

impl VlmState {
    fn image_features(&mut self, bytes: &[u8]) -> Result<Vec<f32>> {
        if let Some((cached, features)) = &self.cached_image
            && cached == bytes
        {
            return Ok(features.clone());
        }
        let pre = self.processor.preprocess(bytes)?;
        let (height, width) = (pre.height, pre.width);
        let outputs = self.vision_graph.run(vec![
            Tensor::new(
                self.pixel_input.clone(),
                vec![1, 1, 3, height, width],
                TensorData::F32(pre.pixels),
            )?,
            Tensor::new(
                self.mask_input.clone(),
                vec![1, 1, height, width],
                TensorData::Bool(pre.mask),
            )?,
        ])?;
        let features = outputs
            .into_iter()
            .find(|tensor| tensor.name == "image_features")
            .ok_or_else(|| EngineError::Other("vision graph has no image_features".into()))?;
        let TensorData::F32(values) = features.data else {
            return Err(EngineError::Other("image_features is not f32".into()));
        };
        self.cached_image = Some((bytes.to_vec(), values.clone()));
        Ok(values)
    }
}

fn expand_image_tokens(
    ids: Vec<i64>,
    image_token: i64,
    image_tokens: usize,
    global_id: i64,
    fake_id: i64,
) -> Result<Vec<i64>> {
    let markers = ids.iter().filter(|id| **id == image_token).count();
    if markers == 0 {
        return Ok(ids);
    }
    if markers > 1 {
        return Err(EngineError::Other(
            "only one image per prompt is supported".into(),
        ));
    }
    let mut expanded = Vec::with_capacity(ids.len() + image_tokens + 2);
    for id in ids {
        if id == image_token {
            expanded.push(fake_id);
            expanded.push(global_id);
            expanded.extend(std::iter::repeat_n(image_token, image_tokens));
            expanded.push(fake_id);
        } else {
            expanded.push(id);
        }
    }
    Ok(expanded)
}

fn splice_image_features(
    embeds: &mut Tensor,
    ids: &[i64],
    image_token: i64,
    features: &[f32],
) -> Result<()> {
    let TensorData::F32(values) = &mut embeds.data else {
        return Err(EngineError::Other("inputs_embeds is not f32".into()));
    };
    let rows = features.len() / HIDDEN;
    let positions = ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| (*id == image_token).then_some(index))
        .collect::<Vec<_>>();
    if positions.len() != rows {
        return Err(EngineError::Other(format!(
            "image token count {} does not match feature rows {rows}",
            positions.len()
        )));
    }
    for (row, position) in positions.iter().enumerate() {
        let target = position * HIDDEN..(position + 1) * HIDDEN;
        values[target].copy_from_slice(&features[row * HIDDEN..(row + 1) * HIDDEN]);
    }
    Ok(())
}

fn empty_kv_cache(graph: &Graph) -> Result<Vec<Tensor>> {
    graph
        .inputs()
        .iter()
        .filter(|spec| spec.name.starts_with("past_key_values."))
        .map(|spec| {
            let shape = spec
                .shape
                .iter()
                .map(|dim| dim.unwrap_or(0))
                .collect::<Vec<_>>();
            Tensor::new(spec.name.clone(), shape, TensorData::F32(Vec::new()))
        })
        .collect()
}

struct Preprocessor {
    size: u32,
    mean: [f32; 3],
    std: [f32; 3],
}

struct Preprocessed {
    pixels: Vec<f32>,
    mask: Vec<bool>,
    height: usize,
    width: usize,
}

impl Preprocessor {
    fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|error| {
            EngineError::ManifestInvalid(format!("preprocessor config: {error}"))
        })?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|error| EngineError::ManifestInvalid(format!("preprocessor JSON: {error}")))?;
        if value
            .get("do_image_splitting")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return Err(EngineError::UnsupportedEngine(
                "onnx_smolvlm image splitting is not implemented; set do_image_splitting=false"
                    .into(),
            ));
        }
        let size = value
            .get("max_image_size")
            .and_then(|size| size.get("longest_edge"))
            .or_else(|| value.get("size").and_then(|size| size.get("longest_edge")))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(512);
        let size = u32::try_from(size)
            .map_err(|_| EngineError::ManifestInvalid("image size overflow".into()))?;
        Ok(Self {
            size: size.max(1),
            mean: array(&value["image_mean"], 0.5),
            std: array(&value["image_std"], 0.5),
        })
    }

    fn preprocess(&self, bytes: &[u8]) -> Result<Preprocessed> {
        let image = image::load_from_memory(bytes)
            .map_err(|error| EngineError::InvalidAudio(format!("image: {error}")))?;
        let (width, height) = image.dimensions();
        let scale = f64::from(self.size) / f64::from(width.max(height));
        let new_width = ((f64::from(width) * scale).round() as u32).max(1);
        let new_height = ((f64::from(height) * scale).round() as u32).max(1);
        let image = image
            .resize_exact(new_width, new_height, FilterType::Triangle)
            .to_rgb8();
        let (canvas_h, canvas_w) = (self.size as usize, self.size as usize);
        let mut pixels = vec![0.0f32; 3 * canvas_h * canvas_w];
        let mut mask = vec![false; canvas_h * canvas_w];
        for y in 0..canvas_h {
            for x in 0..canvas_w {
                mask[y * canvas_w + x] = x < new_width as usize && y < new_height as usize;
            }
        }
        for y in 0..new_height as usize {
            for x in 0..new_width as usize {
                let pixel = image.get_pixel(x as u32, y as u32);
                for channel in 0..3 {
                    let value = f32::from(pixel[channel]) / 255.0;
                    pixels[channel * canvas_h * canvas_w + y * canvas_w + x] =
                        (value - self.mean[channel]) / self.std[channel];
                }
            }
        }
        Ok(Preprocessed {
            pixels,
            mask,
            height: canvas_h,
            width: canvas_w,
        })
    }
}

fn array(value: &serde_json::Value, default: f32) -> [f32; 3] {
    let values = value.as_array();
    std::array::from_fn(|index| {
        values
            .and_then(|values| values.get(index))
            .and_then(serde_json::Value::as_f64)
            .map_or(default, |value| value as f32)
    })
}
