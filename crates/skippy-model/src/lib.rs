//! Validated source checkpoint metadata and tensor access for Skippy.

mod float_convert;
pub mod gguf_catalog;
pub mod gguf_metadata;
pub mod gguf_template;
pub mod gguf_writer;
pub mod hf_checkpoint;
pub mod imatrix;
mod inkling_metadata;
pub mod tensor_map;
pub mod tokenizer_metadata;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ConvertOutputType {
    F32,
    F16,
    Bf16,
    Q8_0,
    TQ1_0,
    TQ2_0,
    Auto,
}

impl ConvertOutputType {
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::TQ1_0 => "tq1_0",
            Self::TQ2_0 => "tq2_0",
            Self::Auto => "auto",
        }
    }
}
