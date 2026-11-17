//! Minimal ONNX graph metadata reader used by layout detection.

use std::path::Path;

use uniinfer_onnx_protobuf::{Message, ModelProto};

use crate::error::{InspectError, Result};

#[derive(Debug, Clone)]
pub(crate) struct TensorInfo {
    pub(crate) name: String,
    /// ONNX TensorProto element type, when declared.
    pub(crate) dtype: Option<i32>,
    /// Static dimensions; 0 marks a dynamic dimension.
    pub(crate) shape: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GraphInfo {
    pub(crate) inputs: Vec<TensorInfo>,
    pub(crate) outputs: Vec<TensorInfo>,
}

pub(crate) fn read(path: &Path) -> Result<GraphInfo> {
    let bytes = std::fs::read(path).map_err(|source| InspectError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let model = ModelProto::parse_from_bytes(&bytes).map_err(|error| InspectError::Config {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let graph = model.graph.as_ref().ok_or_else(|| InspectError::Config {
        path: path.to_path_buf(),
        message: "ONNX model has no graph".into(),
    })?;
    Ok(GraphInfo {
        inputs: graph.input.iter().map(tensor_info).collect(),
        outputs: graph.output.iter().map(tensor_info).collect(),
    })
}

fn tensor_info(value: &uniinfer_onnx_protobuf::ValueInfoProto) -> TensorInfo {
    let tensor = value
        .type_
        .as_ref()
        .and_then(|ty| ty.value.as_ref())
        .and_then(|value| match value {
            uniinfer_onnx_protobuf::type_proto::Value::TensorType(tensor) => Some(tensor),
            _ => None,
        });
    let dtype = tensor.map(|tensor| tensor.elem_type);
    let shape = tensor
        .and_then(|tensor| tensor.shape.as_ref())
        .map(|shape| {
            shape
                .dim
                .iter()
                .map(|dim| match dim.value.as_ref() {
                    Some(
                        uniinfer_onnx_protobuf::tensor_shape_proto::dimension::Value::DimValue(
                            value,
                        ),
                    ) => usize::try_from(*value).unwrap_or(0),
                    _ => 0,
                })
                .collect()
        })
        .unwrap_or_default();
    TensorInfo {
        name: value.name.clone(),
        dtype,
        shape,
    }
}
