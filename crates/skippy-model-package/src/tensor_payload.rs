use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use skippy_package_format::{Tensor, TensorStorage};

#[derive(Clone)]
pub(crate) struct TensorLocation {
    pub(crate) path: PathBuf,
    pub(crate) tensor: Tensor,
}

fn owned_extent<'a>(
    id: &str,
    tensors: &'a BTreeMap<String, TensorLocation>,
) -> Result<(&'a Path, u64, u64)> {
    let location = tensors
        .get(id)
        .with_context(|| format!("tensor {id:?} is absent from storage inventory"))?;
    match &location.tensor.storage {
        TensorStorage::Owned {
            data_offset,
            stored_length,
            ..
        } => Ok((&location.path, *data_offset, *stored_length)),
        TensorStorage::Alias { target_tensor_id } => {
            let target = tensors.get(target_tensor_id).with_context(|| {
                format!("tensor {id:?} aliases missing target {target_tensor_id:?}")
            })?;
            let TensorStorage::Owned {
                data_offset,
                stored_length,
                ..
            } = &target.tensor.storage
            else {
                anyhow::bail!("tensor {id:?} has an alias chain")
            };
            Ok((&target.path, *data_offset, *stored_length))
        }
    }
}

pub(crate) fn compare_tensor_payload(
    id: &str,
    source: &BTreeMap<String, TensorLocation>,
    emitted: &BTreeMap<String, TensorLocation>,
) -> Result<()> {
    let (source_path, source_offset, source_length) = owned_extent(id, source)?;
    let (emitted_path, emitted_offset, emitted_length) = owned_extent(id, emitted)?;
    ensure!(
        source_length == emitted_length,
        "tensor {id:?} stored length differs from independent source"
    );
    let mut source_file = File::open(source_path)?;
    let mut emitted_file = File::open(emitted_path)?;
    source_file.seek(SeekFrom::Start(source_offset))?;
    emitted_file.seek(SeekFrom::Start(emitted_offset))?;
    let mut source_buffer = [0_u8; 64 * 1024];
    let mut emitted_buffer = [0_u8; 64 * 1024];
    let mut remaining = source_length;
    while remaining > 0 {
        let length = usize::try_from(remaining.min(source_buffer.len() as u64))?;
        source_file.read_exact(&mut source_buffer[..length])?;
        emitted_file.read_exact(&mut emitted_buffer[..length])?;
        ensure!(
            source_buffer[..length] == emitted_buffer[..length],
            "tensor {id:?} payload differs from independent source"
        );
        remaining -= length as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_package_format::TensorIntegrity;

    fn owned(id: &str, path: PathBuf, offset: u64, length: u64) -> TensorLocation {
        TensorLocation {
            path,
            tensor: Tensor {
                id: id.to_string(),
                name: id.to_string(),
                ggml_type: 0,
                dimensions: vec![length / 4],
                layer_ordinal: None,
                storage: TensorStorage::Owned {
                    artifact_id: "artifact".to_string(),
                    data_offset: offset,
                    stored_length: length,
                    alignment: 1,
                    integrity: TensorIntegrity::ArtifactSha256,
                },
            },
        }
    }

    fn alias(id: &str, target: &str, path: PathBuf) -> TensorLocation {
        TensorLocation {
            path,
            tensor: Tensor {
                id: id.to_string(),
                name: id.to_string(),
                ggml_type: 0,
                dimensions: vec![2],
                layer_ordinal: None,
                storage: TensorStorage::Alias {
                    target_tensor_id: target.to_string(),
                },
            },
        }
    }

    #[test]
    fn compares_aliases_through_their_owned_extents() {
        let temp = tempfile::tempdir().unwrap();
        let source_path = temp.path().join("source.bin");
        let emitted_path = temp.path().join("emitted.bin");
        std::fs::write(&source_path, b"padding-payload").unwrap();
        std::fs::write(&emitted_path, b"other---payload").unwrap();

        let source = BTreeMap::from([
            ("base".to_string(), owned("base", source_path.clone(), 8, 7)),
            ("view".to_string(), alias("view", "base", source_path)),
        ]);
        let emitted = BTreeMap::from([
            (
                "base".to_string(),
                owned("base", emitted_path.clone(), 8, 7),
            ),
            ("view".to_string(), alias("view", "base", emitted_path)),
        ]);

        compare_tensor_payload("view", &source, &emitted).unwrap();
    }
}
