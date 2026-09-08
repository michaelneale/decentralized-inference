use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Number, Value};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_MIN_VERSION: u32 = 2;
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const MAX_GGUF_STRING_BYTES: u64 = 1_000_000;
const MAX_GGUF_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_GGUF_METADATA_COUNT: u64 = 1_000_000;
const MAX_GGUF_TENSOR_COUNT: u64 = 1_000_000;
const MAX_GGUF_TENSOR_DIMS: u32 = 8;
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;
const GGUF_GENERAL_ALIGNMENT: &str = "general.alignment";

#[derive(Debug, Clone, PartialEq)]
pub struct GgufCatalog {
    pub version: u32,
    pub artifact_bytes: u64,
    pub alignment: u64,
    pub data_start: u64,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<GgufTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub data_offset: u64,
}

pub fn read_gguf_catalog(path: impl AsRef<Path>) -> Result<GgufCatalog> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("open GGUF catalog {}", path.display()))?;
    let artifact_bytes = file
        .metadata()
        .with_context(|| format!("read GGUF size {}", path.display()))?
        .len();
    let mut reader = CatalogReader {
        reader: file,
        artifact_bytes,
    };

    let magic = reader.read_array::<4>()?;
    ensure!(
        &magic == GGUF_MAGIC,
        "{} is not a GGUF file",
        path.display()
    );
    let version = reader.read_u32()?;
    ensure!(
        version >= GGUF_MIN_VERSION,
        "unsupported GGUF version {version} in {}",
        path.display()
    );
    let tensor_count = reader.read_count("tensor", MAX_GGUF_TENSOR_COUNT)?;
    let metadata_count = reader.read_count("metadata", MAX_GGUF_METADATA_COUNT)?;

    let mut metadata = BTreeMap::new();
    for index in 0..metadata_count {
        let key = reader
            .read_string()
            .with_context(|| format!("read GGUF metadata key {index}"))?;
        ensure!(!key.is_empty(), "GGUF metadata key {index} is empty");
        let value_type = reader.read_u32()?;
        let value = reader
            .read_value(value_type)
            .with_context(|| format!("read GGUF metadata value {key:?}"))?;
        ensure!(
            metadata.insert(key.clone(), value).is_none(),
            "duplicate GGUF metadata key {key:?}"
        );
    }

    let alignment = metadata
        .get(GGUF_GENERAL_ALIGNMENT)
        .and_then(Value::as_u64)
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    ensure!(
        alignment.is_power_of_two(),
        "GGUF alignment {alignment} is not a non-zero power of two"
    );

    let mut tensors = Vec::with_capacity(tensor_count);
    for index in 0..tensor_count {
        tensors.push(
            reader
                .read_tensor()
                .with_context(|| format!("read GGUF tensor {index}"))?,
        );
    }
    let tensor_table_end = reader.position()?;
    let data_start = align_to(tensor_table_end, alignment).context("GGUF data offset overflow")?;
    ensure!(
        data_start <= artifact_bytes,
        "GGUF tensor data starts beyond the artifact"
    );

    let mut names = std::collections::BTreeSet::new();
    for tensor in &mut tensors {
        ensure!(
            names.insert(tensor.name.as_str()),
            "duplicate GGUF tensor name {:?}",
            tensor.name
        );
        ensure!(
            tensor.data_offset % alignment == 0,
            "GGUF tensor {:?} offset {} is not aligned to {alignment}",
            tensor.name,
            tensor.data_offset
        );
        tensor.data_offset = data_start
            .checked_add(tensor.data_offset)
            .with_context(|| format!("GGUF tensor {:?} data offset overflow", tensor.name))?;
        ensure!(
            tensor.data_offset < artifact_bytes,
            "GGUF tensor {:?} starts beyond the artifact",
            tensor.name
        );
    }

    Ok(GgufCatalog {
        version,
        artifact_bytes,
        alignment,
        data_start,
        metadata,
        tensors,
    })
}

struct CatalogReader {
    reader: File,
    artifact_bytes: u64,
}

impl CatalogReader {
    fn position(&mut self) -> Result<u64> {
        self.reader
            .stream_position()
            .context("read GGUF stream position")
    }

    fn remaining(&mut self) -> Result<u64> {
        let position = self.position()?;
        self.artifact_bytes
            .checked_sub(position)
            .context("GGUF reader is beyond end of file")
    }

    fn read_count(&mut self, kind: &str, maximum: u64) -> Result<usize> {
        let count = self.read_u64()?;
        ensure!(
            count <= maximum,
            "GGUF {kind} count {count} exceeds safety limit {maximum}"
        );
        ensure!(
            count <= self.remaining()?,
            "GGUF {kind} count {count} exceeds remaining file bytes"
        );
        usize::try_from(count).with_context(|| format!("GGUF {kind} count exceeds usize"))
    }

    fn read_tensor(&mut self) -> Result<GgufTensor> {
        let name = self.read_string()?;
        ensure!(!name.is_empty(), "GGUF tensor name is empty");
        let dimension_count = self.read_u32()?;
        ensure!(
            (1..=MAX_GGUF_TENSOR_DIMS).contains(&dimension_count),
            "GGUF tensor {name:?} has invalid dimension count {dimension_count}"
        );
        ensure!(
            u64::from(dimension_count) <= self.remaining()? / 8,
            "GGUF tensor {name:?} dimensions extend past end of file"
        );
        let dimensions = (0..dimension_count)
            .map(|_| self.read_u64())
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            dimensions.iter().all(|dimension| *dimension > 0),
            "GGUF tensor {name:?} has a zero dimension"
        );
        let ggml_type = self.read_u32()?;
        let data_offset = self.read_u64()?;
        Ok(GgufTensor {
            name,
            dimensions,
            ggml_type,
            data_offset,
        })
    }

    fn read_value(&mut self, value_type: u32) -> Result<Value> {
        match value_type {
            GGUF_TYPE_UINT8 => Ok(Value::from(self.read_u8()?)),
            GGUF_TYPE_INT8 => Ok(Value::from(self.read_i8()?)),
            GGUF_TYPE_UINT16 => Ok(Value::from(self.read_u16()?)),
            GGUF_TYPE_INT16 => Ok(Value::from(self.read_i16()?)),
            GGUF_TYPE_UINT32 => Ok(Value::from(self.read_u32()?)),
            GGUF_TYPE_INT32 => Ok(Value::from(self.read_i32()?)),
            GGUF_TYPE_FLOAT32 => float_value(self.read_f32()? as f64),
            GGUF_TYPE_BOOL => match self.read_u8()? {
                0 => Ok(Value::Bool(false)),
                1 => Ok(Value::Bool(true)),
                value => bail!("invalid GGUF boolean value {value}"),
            },
            GGUF_TYPE_STRING => Ok(Value::String(self.read_string()?)),
            GGUF_TYPE_ARRAY => self.read_array_value(),
            GGUF_TYPE_UINT64 => Ok(Value::from(self.read_u64()?)),
            GGUF_TYPE_INT64 => Ok(Value::from(self.read_i64()?)),
            GGUF_TYPE_FLOAT64 => float_value(self.read_f64()?),
            _ => bail!("unsupported GGUF metadata type {value_type}"),
        }
    }

    fn read_array_value(&mut self) -> Result<Value> {
        let element_type = self.read_u32()?;
        ensure!(
            element_type != GGUF_TYPE_ARRAY,
            "nested GGUF metadata arrays are unsupported"
        );
        let count = self.read_count("array element", MAX_GGUF_ARRAY_ELEMENTS)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_value(element_type)?);
        }
        Ok(Value::Array(values))
    }

    fn read_string(&mut self) -> Result<String> {
        let length = self.read_u64()?;
        ensure!(
            length <= MAX_GGUF_STRING_BYTES,
            "GGUF string length {length} exceeds safety limit {MAX_GGUF_STRING_BYTES}"
        );
        ensure!(
            length <= self.remaining()?,
            "GGUF string length {length} exceeds remaining file bytes"
        );
        let length = usize::try_from(length).context("GGUF string length exceeds usize")?;
        let mut bytes = vec![0_u8; length];
        self.reader
            .read_exact(&mut bytes)
            .context("read GGUF string bytes")?;
        String::from_utf8(bytes).context("GGUF string is not UTF-8")
    }
    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0_u8; N];
        self.reader
            .read_exact(&mut bytes)
            .context("read GGUF bytes")?;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(i8::from_le_bytes(self.read_array()?))
    }

    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read_array()?))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read_array()?))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read_array()?))
    }
}

fn align_to(value: u64, alignment: u64) -> Option<u64> {
    value.div_ceil(alignment).checked_mul(alignment)
}

fn float_value(value: f64) -> Result<Value> {
    Number::from_f64(value)
        .map(Value::Number)
        .context("GGUF metadata contains a non-finite float")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;

    use super::*;

    #[test]
    fn reads_metadata_and_absolute_tensor_offsets_without_payload_reads() {
        let path = temp_path("catalog");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        write_string(&mut bytes, GGUF_GENERAL_ALIGNMENT);
        bytes.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        bytes.extend_from_slice(&64_u32.to_le_bytes());
        write_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        write_string(&mut bytes, "llama");
        write_string(&mut bytes, "blk.0.attn_q.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&16_u64.to_le_bytes());
        bytes.extend_from_slice(&32_u64.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        while !bytes.len().is_multiple_of(64) {
            bytes.push(0);
        }
        let data_start = bytes.len() as u64;
        bytes.extend_from_slice(&[0_u8; 32]);
        fs::write(&path, bytes).unwrap();

        let catalog = read_gguf_catalog(&path).unwrap();
        assert_eq!(catalog.version, 3);
        assert_eq!(catalog.alignment, 64);
        assert_eq!(catalog.data_start, data_start);
        assert_eq!(catalog.tensors.len(), 1);
        assert_eq!(catalog.tensors[0].data_offset, data_start);
        assert_eq!(catalog.tensors[0].dimensions, vec![16, 32]);
        assert_eq!(
            catalog.metadata["general.architecture"],
            Value::String("llama".to_string())
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_tensor_offsets_past_the_artifact() {
        let path = temp_path("offset");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        write_string(&mut bytes, "weight");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u64.to_le_bytes());
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        fs::write(&path, bytes).unwrap();

        let error = read_gguf_catalog(&path).unwrap_err();
        assert!(error.to_string().contains("starts beyond the artifact"));
        fs::remove_file(path).unwrap();
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        bytes
            .write_all(&(value.len() as u64).to_le_bytes())
            .unwrap();
        bytes.write_all(value.as_bytes()).unwrap();
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("skippy-gguf-{name}-{unique}.gguf"))
    }
}
