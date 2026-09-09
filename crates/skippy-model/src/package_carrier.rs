use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use skippy_package_format::{
    Artifact, PackageManifest, Tensor, TensorCatalog, TensorIntegrity, TensorStorage,
};

use crate::gguf_catalog::read_gguf_metadata_catalog;

const METADATA_ONLY_KEY: &str = "skippy.package.metadata_only";
const PART_COUNT_KEY: &str = "skippy.package.part_count";
const LOCATOR_SCHEMA_KEY: &str = "skippy.package.locator_schema";
const TENSOR_PART_KEY: &str = "skippy.package.tensor_part";
const TENSOR_OFFSET_KEY: &str = "skippy.package.tensor_offset";
const TENSOR_SIZE_KEY: &str = "skippy.package.tensor_size";
const TENSOR_ALIGNMENT_KEY: &str = "skippy.package.tensor_alignment";
const LOCATOR_SCHEMA: u64 = 1;

pub fn resolve_package_carrier(
    manifest: PackageManifest,
    carrier_path: impl AsRef<Path>,
) -> Result<PackageManifest> {
    manifest
        .validate_root()
        .context("validate package root before carrier resolution")?;
    let carrier = read_gguf_metadata_catalog(carrier_path)?;
    ensure!(
        carrier.artifact_bytes == carrier.data_start,
        "metadata carrier contains tensor payload bytes"
    );
    ensure!(
        carrier.metadata.get(METADATA_ONLY_KEY) == Some(&Value::Bool(true)),
        "metadata carrier is missing its typed metadata-only marker"
    );
    ensure!(
        carrier
            .metadata
            .get(LOCATOR_SCHEMA_KEY)
            .and_then(Value::as_u64)
            == Some(LOCATOR_SCHEMA),
        "metadata carrier has an unsupported locator schema"
    );

    let payload_artifacts = payload_artifacts(&manifest)?;
    ensure!(
        carrier.metadata.get(PART_COUNT_KEY).and_then(Value::as_u64)
            == u64::try_from(payload_artifacts.len()).ok(),
        "metadata carrier payload part count differs from the package root"
    );

    let part_indices = u64_array(&carrier.metadata, TENSOR_PART_KEY)?;
    let data_offsets = u64_array(&carrier.metadata, TENSOR_OFFSET_KEY)?;
    let stored_lengths = u64_array(&carrier.metadata, TENSOR_SIZE_KEY)?;
    let alignments = u64_array(&carrier.metadata, TENSOR_ALIGNMENT_KEY)?;
    let tensor_count = carrier.tensors.len();
    ensure!(
        [
            part_indices.len(),
            data_offsets.len(),
            stored_lengths.len(),
            alignments.len(),
        ]
        .into_iter()
        .all(|count| count == tensor_count),
        "metadata carrier tensor locator arrays differ from its descriptor count"
    );

    let mut tensors = carrier
        .tensors
        .into_iter()
        .enumerate()
        .map(|(index, descriptor)| {
            let part_index = usize::try_from(part_indices[index])
                .context("metadata carrier tensor part index exceeds usize")?;
            let artifact = payload_artifacts.get(part_index).with_context(|| {
                format!(
                    "metadata carrier tensor {:?} references missing payload part {part_index}",
                    descriptor.name
                )
            })?;
            let alignment = alignments[index];
            let data_offset = data_offsets[index];
            let stored_length = stored_lengths[index];
            ensure!(
                alignment.is_power_of_two() && data_offset % alignment == 0,
                "metadata carrier tensor {:?} has an invalid payload alignment",
                descriptor.name
            );
            ensure!(
                stored_length > 0
                    && data_offset
                        .checked_add(stored_length)
                        .is_some_and(|end| end <= artifact.byte_size),
                "metadata carrier tensor {:?} payload extent exceeds artifact {:?}",
                descriptor.name,
                artifact.id
            );
            Ok(Tensor {
                id: descriptor.name.clone(),
                name: descriptor.name,
                ggml_type: descriptor.ggml_type,
                dimensions: descriptor.dimensions,
                layer_ordinal: layer_ordinal(artifact),
                storage: TensorStorage::Owned {
                    artifact_id: artifact.id.clone(),
                    data_offset,
                    stored_length,
                    alignment,
                    integrity: TensorIntegrity::ArtifactSha256,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    tensors.sort_by(|left, right| left.id.cmp(&right.id));

    let mut model_metadata = carrier.metadata;
    for key in [
        METADATA_ONLY_KEY,
        PART_COUNT_KEY,
        LOCATOR_SCHEMA_KEY,
        TENSOR_PART_KEY,
        TENSOR_OFFSET_KEY,
        TENSOR_SIZE_KEY,
        TENSOR_ALIGNMENT_KEY,
    ] {
        model_metadata.remove(key);
    }
    let resolved = manifest.resolve_inventory(model_metadata, TensorCatalog { entries: tensors });
    resolved
        .validate()
        .context("validate metadata-carrier package inventory")?;
    Ok(resolved)
}

/// Resolve a root-only package manifest from its declared metadata carrier.
pub fn resolve_package_carrier_from_dir(
    manifest: PackageManifest,
    package_dir: impl AsRef<Path>,
) -> Result<PackageManifest> {
    manifest
        .validate_root()
        .context("validate package root before locating metadata carrier")?;
    let artifact = manifest
        .artifact_catalog
        .entries
        .iter()
        .find(|artifact| artifact.id == manifest.source_model.metadata_artifact_id)
        .context("package root metadata artifact is absent")?;
    let relative = Path::new(&artifact.path).to_path_buf();
    ensure!(
        !relative.as_os_str().is_empty()
            && !relative.is_absolute()
            && relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "package root metadata artifact path is not a safe relative path"
    );
    resolve_package_carrier(manifest, package_dir.as_ref().join(relative))
}

fn payload_artifacts(manifest: &PackageManifest) -> Result<Vec<&Artifact>> {
    let sidecars = manifest
        .sidecars
        .iter()
        .map(|sidecar| sidecar.artifact_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut artifacts = manifest
        .artifact_catalog
        .entries
        .iter()
        .filter(|artifact| {
            artifact.id != manifest.source_model.metadata_artifact_id
                && !sidecars.contains(artifact.id.as_str())
        })
        .collect::<Vec<_>>();
    artifacts.sort_by(|left, right| left.id.cmp(&right.id));
    ensure!(
        !artifacts.is_empty(),
        "package root has no payload artifacts"
    );
    Ok(artifacts)
}

fn u64_array(metadata: &BTreeMap<String, Value>, key: &str) -> Result<Vec<u64>> {
    metadata
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("metadata carrier is missing typed array {key:?}"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .with_context(|| format!("metadata carrier {key:?}[{index}] is not unsigned"))
        })
        .collect()
}

fn layer_ordinal(artifact: &Artifact) -> Option<u32> {
    artifact
        .path
        .strip_prefix("layers/layer-")
        .and_then(|value| value.strip_suffix(".gguf"))
        .and_then(|value| value.parse().ok())
}
