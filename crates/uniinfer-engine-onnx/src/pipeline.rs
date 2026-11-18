//! Declarative ONNX graph pipeline.
//!
//! A manifest declares graph steps, tensor bindings, merge points and one
//! output. The engine does not know model families or fixed graph roles.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Mutex;

use image::GenericImageView;
use image::imageops::FilterType;
use ort::environment::Environment;
use uniinfer_core::{
    DeviceClass, EngineError, EngineLoadSpec, EngineSession, ImageBatch, PipelineStep, Result,
    Tensor, TensorData, TensorDtype, TensorSpec,
};

use crate::graph::Graph;

pub(crate) fn load(
    environment: &Environment,
    spec: &EngineLoadSpec,
    class: DeviceClass,
) -> Result<Box<dyn EngineSession>> {
    let pipeline = spec
        .manifest
        .pipeline
        .as_ref()
        .ok_or_else(|| EngineError::ManifestInvalid("pipeline config required".into()))?;
    if let [PipelineStep::Adapter { adapter }] = pipeline.steps.as_slice() {
        return crate::asr::load_named(environment, spec, class, adapter);
    }
    let mut steps = Vec::with_capacity(pipeline.steps.len());
    for step in &pipeline.steps {
        match step {
            PipelineStep::Graph {
                name,
                file,
                device,
                inputs,
                auto,
            } => steps.push(RuntimeStep::Graph(RuntimeGraph {
                name: name.clone(),
                inputs: inputs.clone(),
                auto: auto.clone(),
                graph: Graph::load(
                    environment,
                    device.unwrap_or(class),
                    &spec.base_dir.join(file),
                    &format!("onnx-pipeline-{name}"),
                    crate::backend_variant(spec),
                )?,
            })),
            PipelineStep::Merge { source, target, .. } => {
                steps.push(RuntimeStep::Merge {
                    source: source.clone(),
                    target: target.clone(),
                });
            }
            PipelineStep::Adapter { .. } => {
                return Err(EngineError::UnsupportedEngine(
                    "pipeline adapter steps cannot be mixed with graph steps".into(),
                ));
            }
        }
    }
    let preprocessor = match (
        pipeline.image_preprocessor.as_deref(),
        pipeline.preprocessor_file.as_deref(),
    ) {
        (Some(name), Some(file)) => Some(create_preprocessor(name, &spec.base_dir.join(file))?),
        (Some(_), None) => {
            return Err(EngineError::ManifestInvalid(
                "pipeline.preprocessor_file required".into(),
            ));
        }
        (None, _) => None,
    };
    let vision = spec.manifest.vision.as_ref();
    let image_token_id = vision
        .and_then(|vision| vision.image_token_id)
        .map(i64::from)
        .unwrap_or(0);
    let image_seq_len = vision.and_then(|vision| vision.image_seq_len).unwrap_or(0);
    let global_id = vision
        .and_then(|vision| vision.global_image_token_id)
        .map(i64::from)
        .unwrap_or(0);
    let fake_id = vision
        .and_then(|vision| vision.fake_image_token_id)
        .map(i64::from)
        .unwrap_or(0);
    let output = pipeline.output.clone();
    let inputs = vec![
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
    ];
    let outputs = vec![TensorSpec {
        name: "logits".into(),
        dtype: TensorDtype::F32,
        shape: vec![Some(1), None, None],
    }];
    Ok(Box::new(PipelineSession {
        state: Mutex::new(PipelineState {
            steps,
            preprocessor,
            token_layout: pipeline.token_layout.clone(),
            image_token_id,
            image_seq_len,
            global_id,
            fake_id,
            output,
            cached_images: None,
        }),
        inputs,
        outputs,
    }))
}

enum RuntimeStep {
    Graph(RuntimeGraph),
    Merge { source: String, target: String },
}

struct RuntimeGraph {
    name: String,
    inputs: BTreeMap<String, String>,
    auto: Vec<String>,
    graph: Graph,
}

struct PipelineState {
    steps: Vec<RuntimeStep>,
    preprocessor: Option<Box<dyn ImagePreprocessor>>,
    token_layout: Option<String>,
    image_token_id: i64,
    image_seq_len: usize,
    global_id: i64,
    fake_id: i64,
    output: String,
    cached_images: Option<(Vec<Vec<u8>>, Vec<Preprocessed>)>,
}

struct PipelineSession {
    state: Mutex<PipelineState>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl EngineSession for PipelineSession {
    fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }

    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let mut values = HashMap::<String, Tensor>::new();
        for tensor in inputs {
            values.insert(tensor.name.clone(), tensor);
        }
        let ids = values
            .get("input_ids")
            .and_then(|tensor| match &tensor.data {
                TensorData::I64(values) => Some(values.clone()),
                _ => None,
            })
            .ok_or_else(|| EngineError::Other("pipeline requires int64 input_ids".into()))?;
        let image = values.get("image").and_then(|tensor| match &tensor.data {
            TensorData::U8(bytes) => Some(bytes.clone()),
            _ => None,
        });

        let mut state = self
            .state
            .lock()
            .map_err(|_| EngineError::Other("onnx pipeline state poisoned".into()))?;
        if let Some(image) = image {
            let images = ImageBatch::decode(&image)?;
            let pre = state.preprocess(images.as_slice())?;
            let batch = pre.len();
            let height = pre[0].height;
            let width = pre[0].width;
            if pre
                .iter()
                .any(|image| image.height != height || image.width != width)
            {
                return Err(EngineError::Other(
                    "preprocessed images have different sizes".into(),
                ));
            }
            let ids = state.expand_ids(ids)?;
            values.insert(
                "input_ids".into(),
                Tensor::new("input_ids", vec![1, ids.len()], TensorData::I64(ids))?,
            );
            let mut pixels = Vec::with_capacity(batch * 3 * height * width);
            let mut mask = Vec::with_capacity(batch * height * width);
            for image in &pre {
                pixels.extend_from_slice(&image.pixels);
                mask.extend_from_slice(&image.mask);
            }
            values.insert(
                "preprocess.pixel_values".into(),
                Tensor::new(
                    "preprocess.pixel_values",
                    vec![batch, 1, 3, height, width],
                    TensorData::F32(pixels),
                )?,
            );
            values.insert(
                "preprocess.pixel_attention_mask".into(),
                Tensor::new(
                    "preprocess.pixel_attention_mask",
                    vec![batch, 1, height, width],
                    TensorData::Bool(mask),
                )?,
            );
        }

        let PipelineState {
            steps,
            image_token_id,
            output,
            ..
        } = &mut *state;
        for step in steps.iter_mut() {
            match step {
                RuntimeStep::Graph(graph) => {
                    let seq = sequence_len(&values)?;
                    let mut tensors = Vec::with_capacity(graph.graph.inputs().len());
                    for spec in graph.graph.inputs() {
                        if let Some(source) = graph.inputs.get(&spec.name) {
                            let mut tensor = values.get(source).cloned().ok_or_else(|| {
                                EngineError::Other(format!(
                                    "pipeline step {} missing binding {source}",
                                    graph.name
                                ))
                            })?;
                            tensor.name = spec.name.clone();
                            tensors.push(tensor);
                        } else if spec.name == "attention_mask"
                            && graph.auto.iter().any(|name| name == "attention_mask")
                        {
                            tensors.push(Tensor::new(
                                "attention_mask",
                                vec![1, seq],
                                TensorData::I64(vec![1; seq]),
                            )?);
                        } else if spec.name == "position_ids"
                            && graph.auto.iter().any(|name| name == "position_ids")
                        {
                            tensors.push(Tensor::new(
                                "position_ids",
                                vec![1, seq],
                                TensorData::I64((0..seq as i64).collect()),
                            )?);
                        } else if spec.name.starts_with("past_key_values.")
                            && graph.auto.iter().any(|name| name == "past_key_values")
                        {
                            let shape = spec
                                .shape
                                .iter()
                                .map(|dim| dim.unwrap_or(0))
                                .collect::<Vec<_>>();
                            tensors.push(Tensor::new(
                                spec.name.clone(),
                                shape,
                                TensorData::F32(Vec::new()),
                            )?);
                        } else {
                            return Err(EngineError::Other(format!(
                                "pipeline step {} has no binding for {}",
                                graph.name, spec.name
                            )));
                        }
                    }
                    for tensor in graph.graph.run(tensors)? {
                        values.insert(format!("{}.{}", graph.name, tensor.name), tensor);
                    }
                }
                RuntimeStep::Merge { source, target } => {
                    let features = values
                        .get(source)
                        .cloned()
                        .ok_or_else(|| EngineError::Other(format!("merge missing {source}")))?;
                    let mut target_tensor = values
                        .remove(target)
                        .ok_or_else(|| EngineError::Other(format!("merge missing {target}")))?;
                    let ids = values
                        .get("input_ids")
                        .and_then(|tensor| match &tensor.data {
                            TensorData::I64(values) => Some(values.clone()),
                            _ => None,
                        })
                        .ok_or_else(|| EngineError::Other("merge requires input_ids".into()))?;
                    splice_image_features(&mut target_tensor, &ids, *image_token_id, &features)?;
                    values.insert(target.clone(), target_tensor);
                }
            }
        }
        let mut logits = values
            .remove(output)
            .ok_or_else(|| EngineError::Other(format!("pipeline output {output} missing")))?;
        logits.name = "logits".into();
        Ok(vec![logits])
    }

    fn cancel(&self) {}
}

impl PipelineState {
    fn preprocess(&mut self, images: &[Vec<u8>]) -> Result<Vec<Preprocessed>> {
        if let Some((cached, value)) = &self.cached_images
            && cached == images
        {
            return Ok(value.clone());
        }
        let preprocessor = self.preprocessor.as_ref().ok_or_else(|| {
            EngineError::ManifestInvalid("pipeline has no image preprocessor".into())
        })?;
        let value = images
            .iter()
            .map(|image| preprocessor.preprocess(image))
            .collect::<Result<Vec<_>>>()?;
        self.cached_images = Some((images.to_vec(), value.clone()));
        Ok(value)
    }

    fn expand_ids(&self, ids: Vec<i64>) -> Result<Vec<i64>> {
        match self.token_layout.as_deref() {
            Some("smolvlm") => expand_image_tokens(
                ids,
                self.image_token_id,
                self.image_seq_len,
                self.global_id,
                self.fake_id,
            ),
            Some(other) => Err(EngineError::UnsupportedEngine(format!(
                "pipeline token layout {other:?}"
            ))),
            None => Ok(ids),
        }
    }
}

fn sequence_len(values: &HashMap<String, Tensor>) -> Result<usize> {
    if let Some(embeds) = values
        .iter()
        .find(|(name, tensor)| name.ends_with(".inputs_embeds") && tensor.shape.len() == 3)
        .map(|(_, tensor)| tensor)
    {
        return Ok(embeds.shape[1]);
    }
    values
        .get("input_ids")
        .and_then(|tensor| tensor.shape.get(1).copied())
        .ok_or_else(|| EngineError::Other("pipeline cannot determine sequence length".into()))
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
    if image_tokens == 0 {
        return Err(EngineError::ManifestInvalid(
            "vision.image_seq_len required".into(),
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
    features: &Tensor,
) -> Result<()> {
    let TensorData::F32(values) = &mut embeds.data else {
        return Err(EngineError::Other(
            "pipeline merge target is not f32".into(),
        ));
    };
    let hidden = embeds.shape.last().copied().unwrap_or(0);
    let TensorData::F32(features) = &features.data else {
        return Err(EngineError::Other(
            "pipeline merge source is not f32".into(),
        ));
    };
    let rows = features.len() / hidden.max(1);
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
        let target = position * hidden..(position + 1) * hidden;
        values[target].copy_from_slice(&features[row * hidden..(row + 1) * hidden]);
    }
    Ok(())
}

trait ImagePreprocessor: Send + Sync {
    fn preprocess(&self, bytes: &[u8]) -> Result<Preprocessed>;
}

fn create_preprocessor(name: &str, path: &Path) -> Result<Box<dyn ImagePreprocessor>> {
    match name {
        "smolvlm" => Ok(Box::new(SmolVlmPreprocessor::load(path)?)),
        other => Err(EngineError::UnsupportedEngine(format!(
            "image preprocessor {other:?}"
        ))),
    }
}

#[derive(Clone)]
struct Preprocessed {
    pixels: Vec<f32>,
    mask: Vec<bool>,
    height: usize,
    width: usize,
}

struct SmolVlmPreprocessor {
    size: u32,
    mean: [f32; 3],
    std: [f32; 3],
}

impl SmolVlmPreprocessor {
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
                "smolvlm image splitting is not implemented; set do_image_splitting=false".into(),
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
}

impl ImagePreprocessor for SmolVlmPreprocessor {
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
