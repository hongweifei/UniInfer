#![allow(mismatched_lifetime_syntaxes)]

mod helpers;
mod proto;

pub use crate::{helpers::AttributeValue, proto::onnx::*};
pub use protobuf::Message;
