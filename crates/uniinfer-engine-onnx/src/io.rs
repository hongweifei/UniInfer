//! ONNX tensor conversion helpers.

use half::f16;
use ort::value::{DynValue, Outlet, Tensor as OrtTensor, TensorElementType, ValueType};
use uniinfer_core::{EngineError, Result, Tensor, TensorData, TensorDtype, TensorSpec};

pub(crate) fn convert_specs(outlets: &[Outlet], kind: &str) -> Result<Vec<TensorSpec>> {
    outlets
        .iter()
        .map(|outlet| convert_spec(outlet, kind))
        .collect()
}

pub(crate) fn convert_spec(outlet: &Outlet, kind: &str) -> Result<TensorSpec> {
    let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
        return Err(EngineError::LoadFailed(format!(
            "{kind} {} is not a tensor",
            outlet.name()
        )));
    };
    let dtype = match *ty {
        TensorElementType::Float32 => TensorDtype::F32,
        TensorElementType::Float16 => TensorDtype::F16,
        TensorElementType::Int32 => TensorDtype::I32,
        TensorElementType::Int64 => TensorDtype::I64,
        TensorElementType::Uint8 => TensorDtype::U8,
        TensorElementType::Bool => TensorDtype::Bool,
        other => {
            return Err(EngineError::LoadFailed(format!(
                "{kind} {} has unsupported dtype {other:?}",
                outlet.name()
            )));
        }
    };
    let shape = shape
        .iter()
        .map(|dim| if *dim >= 0 { Some(*dim as usize) } else { None })
        .collect();
    Ok(TensorSpec {
        name: outlet.name().to_owned(),
        dtype,
        shape,
    })
}

pub(crate) fn to_ort(tensor: Tensor) -> Result<DynValue> {
    let shape = tensor.shape.clone();
    let value = match tensor.data {
        TensorData::F32(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
        TensorData::F16(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
        TensorData::I32(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
        TensorData::I64(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
        TensorData::U8(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
        TensorData::Bool(values) => {
            OrtTensor::from_array((shape, values)).map(ort::value::Value::into_dyn)
        }
    }
    .map_err(|error| EngineError::Other(format!("tensor {}: {error}", tensor.name)))?;
    Ok(value)
}

pub(crate) fn from_ort(spec: &TensorSpec, value: &DynValue) -> Result<Tensor> {
    match spec.dtype {
        TensorDtype::F32 => {
            let (shape, slice) = value
                .try_extract_tensor::<f32>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::F32(slice.to_vec()))
        }
        TensorDtype::F16 => {
            let (shape, slice) = value
                .try_extract_tensor::<f16>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::F16(slice.to_vec()))
        }
        TensorDtype::I32 => {
            let (shape, slice) = value
                .try_extract_tensor::<i32>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::I32(slice.to_vec()))
        }
        TensorDtype::I64 => {
            let (shape, slice) = value
                .try_extract_tensor::<i64>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::I64(slice.to_vec()))
        }
        TensorDtype::U8 => {
            let (shape, slice) = value
                .try_extract_tensor::<u8>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::U8(slice.to_vec()))
        }
        TensorDtype::Bool => {
            let (shape, slice) = value
                .try_extract_tensor::<bool>()
                .map_err(|error| EngineError::Other(format!("extract {}: {error}", spec.name)))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Tensor::new(spec.name.clone(), shape, TensorData::Bool(slice.to_vec()))
        }
    }
}
