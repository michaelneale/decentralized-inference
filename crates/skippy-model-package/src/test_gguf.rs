use std::fs;
use std::path::Path;

use crate::package::ExplicitSourceIdentity;

#[derive(Clone)]
pub(crate) struct FixtureTensor<'a> {
    pub(crate) name: &'a str,
    pub(crate) dimensions: Vec<u64>,
    pub(crate) dtype: u32,
    pub(crate) offset: u64,
}

pub(crate) fn tensor(name: &str, offset: u64) -> FixtureTensor<'_> {
    FixtureTensor {
        name,
        dimensions: vec![2, 2],
        dtype: 0,
        offset,
    }
}

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

pub(crate) fn fixture(path: &Path, tensors: &[FixtureTensor<'_>], split: Option<(u16, u16, u64)>) {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(if split.is_some() { 7_u64 } else { 4 }).to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    string(&mut bytes, "llama");
    string(&mut bytes, "llama.block_count");
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    string(&mut bytes, "general.alignment");
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&32_u32.to_le_bytes());
    string(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9_u32.to_le_bytes());
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    string(&mut bytes, "fixture-token");
    if let Some((number, count, total)) = split {
        for (key, value) in [("split.no", number), ("split.count", count)] {
            string(&mut bytes, key);
            bytes.extend_from_slice(&2_u32.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        string(&mut bytes, "split.tensors.count");
        bytes.extend_from_slice(&10_u32.to_le_bytes());
        bytes.extend_from_slice(&total.to_le_bytes());
    }
    for t in tensors {
        string(&mut bytes, t.name);
        bytes.extend_from_slice(&(t.dimensions.len() as u32).to_le_bytes());
        for dimension in &t.dimensions {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&t.dtype.to_le_bytes());
        bytes.extend_from_slice(&t.offset.to_le_bytes());
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    // All fixtures use a 32-byte padded extent per tensor, including quantized Q8_0
    // fixtures below which explicitly extend their payload.
    bytes.resize(bytes.len() + tensors.len() * 32, 0x3f);
    fs::write(path, bytes).unwrap();
}

pub(crate) fn explicit(source: &Path) -> ExplicitSourceIdentity {
    ExplicitSourceIdentity {
        model_id: Some("fixture/model:Q8_0".to_string()),
        source_revision: Some("immutable-source".to_string()),
        source_file: Some(source.file_name().unwrap().to_str().unwrap().to_string()),
        ..ExplicitSourceIdentity::default()
    }
}
