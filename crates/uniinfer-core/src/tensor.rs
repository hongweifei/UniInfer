use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

/// Element types carried across the engine boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TensorDtype {
    F32,
    I64,
    U8,
}

/// Contiguous tensor payload.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

impl TensorData {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U8(values) => values.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn dtype(&self) -> TensorDtype {
        match self {
            Self::F32(_) => TensorDtype::F32,
            Self::I64(_) => TensorDtype::I64,
            Self::U8(_) => TensorDtype::U8,
        }
    }
}

/// Named tensor passed to and from engines.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: TensorData,
}

impl Tensor {
    /// Build a tensor, checking that the shape matches the element count.
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data: TensorData) -> Result<Self> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(EngineError::Other(format!(
                "tensor {}: shape {:?} needs {expected} elements, got {}",
                name.into(),
                shape,
                data.len()
            )));
        }
        Ok(Self {
            name: name.into(),
            shape,
            data,
        })
    }
}

/// Static description of an engine input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: TensorDtype,
    /// `None` marks a dynamic dimension.
    pub shape: Vec<Option<usize>>,
}
