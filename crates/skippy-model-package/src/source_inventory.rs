//! Independent source evidence, captured before any package artifacts are written.
//! Native inspection supplies authoritative stored sizes; the GGUF directory supplies
//! exact names, dimensions and absolute offsets. No role or layer-name selection.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use skippy_model::gguf_catalog::{GgufCatalog, read_gguf_catalog};
use skippy_package_format::{SourceFile, Tensor, TensorCatalog, TensorIntegrity, TensorStorage};
use skippy_runtime::{ModelInfo, TensorInfo};

use crate::hash::file_sha256;
use crate::package::PackageInput;
use crate::write::{local_artifact_files, resolve_gguf_shard_paths};

pub(crate) struct SourceInventory {
    pub(crate) shards: Vec<SourceShard>,
    pub(crate) layer_count: u32,
}

pub(crate) struct SourceShard {
    pub(crate) path: PathBuf,
    pub(crate) source_file: SourceFile,
    pub(crate) artifact_id: String,
    pub(crate) directory: GgufCatalog,
    pub(crate) tensors: TensorCatalog,
}

impl SourceInventory {
    pub(crate) fn read(input: &PackageInput) -> Result<Self> {
        let primary = input
            .source_identity
            .primary_file
            .as_deref()
            .context("source primary file is required")?;
        let paths = resolve_gguf_shard_paths(&input.model_path)?;
        let files = local_artifact_files(&input.model_path, primary)?;
        let expected: BTreeSet<_> = input
            .source_identity
            .files
            .iter()
            .map(|f| &f.path)
            .collect();
        ensure!(
            expected.len() == input.source_identity.files.len(),
            "duplicate source file identity"
        );
        ensure!(
            files.iter().map(|f| &f.path).collect::<BTreeSet<_>>() == expected,
            "resolved source file inventory does not match complete GGUF shard set"
        );
        let mut shards = Vec::new();
        let mut names = BTreeSet::new();
        for (index, (path, file)) in paths.into_iter().zip(files).enumerate() {
            let artifact_id = format!("source-{index:05}");
            let (directory, tensors) = inspect(&path, &artifact_id)?;
            for tensor in &tensors.entries {
                ensure!(
                    names.insert(tensor.name.clone()),
                    "duplicate source tensor {:?}",
                    tensor.name
                );
            }
            let sha256 = file_sha256(&path)?;
            let declared = input
                .source_identity
                .files
                .iter()
                .find(|f| f.path == file.path)
                .context("source identity missing a resolved file")?;
            ensure!(
                declared
                    .size_bytes
                    .is_none_or(|size| size == directory.artifact_bytes),
                "source size mismatch for {:?}",
                file.path
            );
            ensure!(
                declared
                    .sha256
                    .as_ref()
                    .is_none_or(|digest| digest == &sha256),
                "source checksum mismatch for {:?}",
                file.path
            );
            shards.push(SourceShard {
                path,
                artifact_id,
                tensors,
                source_file: SourceFile {
                    path: file.path,
                    byte_size: directory.artifact_bytes,
                    sha256,
                },
                directory,
            });
        }
        ensure!(!names.is_empty(), "source tensor inventory is empty");
        validate_shards(&shards)?;
        let metadata = &shards[0].directory.metadata;
        let architecture = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .context("source requires general.architecture")?;
        let layer_count = metadata
            .get(&format!("{architecture}.block_count"))
            .and_then(|v| v.as_u64())
            .context("source requires architecture block_count metadata")?;
        let layer_count = u32::try_from(layer_count).context("source block_count exceeds u32")?;
        ensure!(layer_count > 0, "source block_count must be positive");
        Ok(Self {
            shards,
            layer_count,
        })
    }

    pub(crate) fn tensor_catalog(&self) -> TensorCatalog {
        TensorCatalog {
            entries: self
                .shards
                .iter()
                .flat_map(|s| s.tensors.entries.clone())
                .collect(),
        }
    }
}

fn validate_shards(shards: &[SourceShard]) -> Result<()> {
    let first = shards.first().context("source shard inventory is empty")?;
    let total_tensors: usize = shards.iter().map(|s| s.tensors.entries.len()).sum();
    for (index, shard) in shards.iter().enumerate() {
        let metadata = &shard.directory.metadata;
        if shards.len() > 1 || metadata.contains_key("split.count") {
            ensure!(
                metadata.get("split.count").and_then(|v| v.as_u64()) == Some(shards.len() as u64),
                "incomplete source shard set: split.count mismatch"
            );
            ensure!(
                metadata.get("split.no").and_then(|v| v.as_u64()) == Some(index as u64),
                "source split.no mismatch"
            );
            ensure!(
                metadata.get("split.tensors.count").and_then(|v| v.as_u64())
                    == Some(total_tensors as u64),
                "incomplete source tensor inventory: split.tensors.count mismatch"
            );
        }
        // Later shards may carry a subset, but may not contradict the primary
        // shard's model metadata or introduce metadata absent from that shard.
        for (key, value) in metadata {
            if key != "split.no" && key != "split.count" && key != "split.tensors.count" {
                ensure!(
                    first.directory.metadata.get(key) == Some(value),
                    "inconsistent source metadata {key:?} across shards"
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn inspect(path: &Path, artifact_id: &str) -> Result<(GgufCatalog, TensorCatalog)> {
    let directory = read_gguf_catalog(path)?;
    let native = ModelInfo::open(path)
        .with_context(|| format!("native GGUF inspection failed for {}; shared-offset aliases/non-contiguous storage require a native inspection extension", path.display()))?
        .tensors()?;
    let tensors = catalog_from_inspection(&directory, &native, artifact_id)?;
    Ok((directory, tensors))
}

fn catalog_from_inspection(
    directory: &GgufCatalog,
    native: &[TensorInfo],
    artifact_id: &str,
) -> Result<TensorCatalog> {
    let by_name: BTreeMap<_, _> = native.iter().map(|t| (t.name.as_str(), t)).collect();
    ensure!(
        by_name.len() == native.len() && native.len() == directory.tensors.len(),
        "native and GGUF tensor inventories disagree"
    );
    let mut source = directory.tensors.iter().collect::<Vec<_>>();
    source.sort_by(|a, b| (a.data_offset, &a.name).cmp(&(b.data_offset, &b.name)));
    let mut entries: Vec<Tensor> = Vec::new();
    let mut last_owned: Option<(u64, u64, usize)> = None;
    for tensor in source {
        let info = by_name
            .get(tensor.name.as_str())
            .context("tensor missing from native inventory")?;
        let elements = tensor
            .dimensions
            .iter()
            .try_fold(1_u64, |n, d| n.checked_mul(*d))
            .context("tensor element count overflow")?;
        ensure!(
            info.ggml_type == tensor.ggml_type && info.element_count == elements,
            "native and GGUF metadata disagree for {:?}",
            tensor.name
        );
        ensure!(
            info.byte_size > 0,
            "empty tensor storage for {:?}",
            tensor.name
        );
        let end = tensor
            .data_offset
            .checked_add(info.byte_size)
            .context("tensor extent overflow")?;
        ensure!(
            end <= directory.artifact_bytes,
            "tensor {:?} storage exceeds artifact bounds",
            tensor.name
        );
        let storage = if let Some((start, previous_end, owner)) =
            last_owned.filter(|(_, e, _)| tensor.data_offset < *e)
        {
            let target = &entries[owner];
            ensure!(
                tensor.data_offset == start
                    && end == previous_end
                    && tensor.ggml_type == target.ggml_type
                    && tensor.dimensions == target.dimensions,
                "overlapping or mismatched alias storage for {:?}",
                tensor.name
            );
            TensorStorage::Alias {
                target_tensor_id: target.id.clone(),
            }
        } else {
            last_owned = Some((tensor.data_offset, end, entries.len()));
            TensorStorage::Owned {
                artifact_id: artifact_id.to_string(),
                data_offset: tensor.data_offset,
                stored_length: info.byte_size,
                alignment: directory.alignment,
                integrity: TensorIntegrity::ArtifactSha256,
            }
        };
        entries.push(Tensor {
            id: tensor.name.clone(),
            name: tensor.name.clone(),
            ggml_type: tensor.ggml_type,
            dimensions: tensor.dimensions.clone(),
            // GGUF has no structural per-tensor layer field. The legacy native
            // layer_index is name-derived, so deliberately do not use it here.
            layer_ordinal: None,
            storage,
        });
    }
    entries.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(TensorCatalog { entries })
}

impl SourceShard {
    pub(crate) fn verify_copy(&self, path: &Path) -> Result<()> {
        let (directory, tensors) = inspect(path, &self.artifact_id)?;
        ensure!(
            directory == self.directory && tensors == self.tensors,
            "written artifact {:?} does not match independent source tensor inventory",
            path
        );
        ensure!(
            file_sha256(path)? == self.source_file.sha256,
            "written artifact {:?} checksum differs from independent source",
            path
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;
