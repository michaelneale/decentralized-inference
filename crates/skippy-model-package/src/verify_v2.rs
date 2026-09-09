//! Read-only verification against independently supplied source evidence.
//! This unit verifies repacked model artifacts tensor by tensor, not graph readiness.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use skippy_model::package_carrier::resolve_package_carrier;
use skippy_package_format::{Artifact, PackageManifest, SourceFile, Tensor, TensorStorage};

use crate::hash::file_sha256;
use crate::source_inventory::{SourceInventory, SourceShard, inspect};
use crate::tensor_payload::{TensorLocation, compare_tensor_payload};

#[derive(Debug, Serialize)]
pub(crate) struct VerificationReport {
    pub(crate) package_id: String,
    pub(crate) source_completeness_verified: bool,
    pub(crate) checked_source_files: usize,
    pub(crate) checked_artifacts: usize,
    pub(crate) checked_tensors: usize,
    pub(crate) checked_projectors: usize,
}

pub(crate) fn verify_package(
    package: &Path,
    source: &Path,
    source_file: Option<&str>,
    source_projectors: &[PathBuf],
) -> Result<VerificationReport> {
    let root = fs::canonicalize(package).context("open package directory")?;
    ensure!(root.is_dir(), "package must be a directory");
    let manifest_path = contained_file(&root, "model-package.json")?;
    let manifest: PackageManifest = serde_json::from_slice(&fs::read(manifest_path)?)
        .context("read v2 manifest (v1 input is not supported)")?;
    manifest.validate_root()?;
    ensure!(
        manifest.format == "gguf",
        "only GGUF v2 packages are supported"
    );
    // The expected primary filename comes from the caller, never the manifest.
    let primary = source_file
        .or_else(|| source.file_name()?.to_str())
        .context("source primary filename is required")?;
    let inventory = SourceInventory::read_local(source, primary)?;
    let projectors = read_projectors(source_projectors)?;
    let sources: Vec<_> = inventory.shards.iter().chain(&projectors).collect();
    let artifacts = verify_artifact_files(&root, &manifest, &sources)?;
    let metadata_artifact_id = manifest.source_model.metadata_artifact_id.clone();
    let metadata_path = artifacts
        .get(metadata_artifact_id.as_str())
        .context("validated metadata artifact is absent")?;
    let manifest = resolve_package_carrier(manifest, metadata_path)?;
    verify_source_identity(&manifest, &inventory, primary)?;
    ensure!(
        manifest
            .tensor_catalog
            .entries
            .iter()
            .all(|t| matches!(t.storage, TensorStorage::Owned { .. })),
        "alias claims cannot be proven by the pinned native GGUF inspector"
    );
    let expected = source_tensor_locations(&inventory)?;
    let declared: BTreeMap<_, _> = manifest
        .tensor_catalog
        .entries
        .iter()
        .map(|tensor| (tensor.id.as_str(), tensor))
        .collect();
    ensure!(
        declared.keys().copied().collect::<BTreeSet<_>>()
            == expected.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        "manifest tensor catalog differs from independent source inventory"
    );
    for (id, tensor) in &declared {
        ensure_tensor_metadata_matches(tensor, &expected[*id].tensor)?;
    }

    let sidecar_ids = manifest
        .sidecars
        .iter()
        .map(|sidecar| sidecar.artifact_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut written = BTreeMap::new();
    let mut written_layer_ordinals = BTreeMap::new();
    let mut used = BTreeSet::new();
    used.insert(metadata_artifact_id.clone());
    for artifact in &manifest.artifact_catalog.entries {
        if artifact.id == metadata_artifact_id || sidecar_ids.contains(artifact.id.as_str()) {
            continue;
        }
        used.insert(artifact.id.clone());
        let path = &artifacts[&artifact.id];
        let (_, catalog) = inspect(path, &artifact.id)?;
        let locations = tensor_locations(path, &catalog.entries)?;
        let layer_ordinals = skippy_runtime::ModelInfo::open(path)?
            .tensors()?
            .into_iter()
            .map(|tensor| (tensor.name, tensor.layer_index))
            .collect::<BTreeMap<_, _>>();
        ensure!(
            layer_ordinals.len() == locations.len(),
            "native layer inventory differs from emitted artifact directory"
        );
        for (id, emitted) in &locations {
            let source = expected.get(id).with_context(|| {
                format!(
                    "artifact {:?} contains unexpected tensor {id:?}",
                    artifact.id
                )
            })?;
            ensure_tensor_metadata_matches(&emitted.tensor, &source.tensor)?;
            compare_tensor_payload(id, &expected, &locations)?;
        }
        written.insert(artifact.id.as_str(), locations);
        written_layer_ordinals.insert(artifact.id.as_str(), layer_ordinals);
    }

    for tensor in manifest.tensor_catalog.entries.iter() {
        let TensorStorage::Owned { artifact_id, .. } = &tensor.storage else {
            unreachable!("alias declarations were rejected above");
        };
        let artifact_record = manifest
            .artifact_catalog
            .entries
            .iter()
            .find(|artifact| artifact.id == *artifact_id)
            .context("validated tensor artifact is absent")?;
        ensure!(
            tensor.layer_ordinal == layer_ordinal_from_artifact_path(&artifact_record.path),
            "tensor {:?} layer ordinal disagrees with its physical artifact",
            tensor.id
        );
        let artifact = written
            .get(artifact_id.as_str())
            .with_context(|| format!("tensor {:?} references a sidecar artifact", tensor.id))?;
        let emitted = artifact.get(&tensor.id).with_context(|| {
            format!(
                "tensor {:?} is absent from declared artifact {artifact_id:?}",
                tensor.id
            )
        })?;
        if tensor.layer_ordinal.is_some() {
            ensure!(
                written_layer_ordinals[artifact_id.as_str()][&tensor.id] == tensor.layer_ordinal,
                "tensor {:?} layer ordinal differs from native artifact inspection",
                tensor.id
            );
        }
        ensure!(
            tensor.storage == emitted.tensor.storage,
            "tensor {:?} storage differs from emitted artifact directory",
            tensor.id
        );
    }
    verify_projectors(&manifest, &projectors, &artifacts, &mut used)?;
    ensure!(
        used.len() == artifacts.len(),
        "artifact catalog contains unaccounted artifacts"
    );
    Ok(VerificationReport {
        package_id: manifest.package_id,
        source_completeness_verified: true,
        checked_source_files: inventory.shards.len(),
        checked_artifacts: artifacts.len(),
        checked_tensors: expected.len(),
        checked_projectors: projectors.len(),
    })
}

fn layer_ordinal_from_artifact_path(path: &str) -> Option<u32> {
    path.strip_prefix("layers/layer-")
        .and_then(|value| value.strip_suffix(".gguf"))
        .and_then(|value| value.parse().ok())
}

fn source_tensor_locations(
    inventory: &SourceInventory,
) -> Result<BTreeMap<String, TensorLocation>> {
    let mut locations = BTreeMap::new();
    for shard in &inventory.shards {
        for tensor in &shard.tensors.entries {
            ensure!(
                locations
                    .insert(
                        tensor.id.clone(),
                        TensorLocation {
                            path: shard.path.clone(),
                            tensor: tensor.clone(),
                        },
                    )
                    .is_none(),
                "duplicate source tensor {:?}",
                tensor.id
            );
        }
    }
    Ok(locations)
}

fn tensor_locations(path: &Path, tensors: &[Tensor]) -> Result<BTreeMap<String, TensorLocation>> {
    let mut locations = BTreeMap::new();
    for tensor in tensors {
        ensure!(
            locations
                .insert(
                    tensor.id.clone(),
                    TensorLocation {
                        path: path.to_path_buf(),
                        tensor: tensor.clone(),
                    },
                )
                .is_none(),
            "duplicate tensor {:?} in {}",
            tensor.id,
            path.display()
        );
    }
    Ok(locations)
}

fn ensure_tensor_metadata_matches(actual: &Tensor, expected: &Tensor) -> Result<()> {
    ensure!(
        actual.id == expected.id
            && actual.name == expected.name
            && actual.ggml_type == expected.ggml_type
            && actual.dimensions == expected.dimensions,
        "tensor {:?} metadata differs from independent source inventory",
        actual.id
    );
    Ok(())
}

fn verify_source_identity(
    manifest: &PackageManifest,
    source: &SourceInventory,
    primary: &str,
) -> Result<()> {
    let expected: BTreeMap<_, _> = source
        .shards
        .iter()
        .map(|s| (s.source_file.path.as_str(), &s.source_file))
        .collect();
    let declared: BTreeMap<_, _> = manifest
        .source_model
        .files
        .iter()
        .map(|f| (f.path.as_str(), f))
        .collect();
    ensure!(
        declared == expected,
        "source file identity differs from independent source inventory"
    );
    ensure!(
        manifest.source_model.primary_file.as_deref() == Some(primary),
        "source primary file mismatch"
    );
    let primary_file = expected
        .get(primary)
        .context("independent primary source missing")?;
    ensure!(
        manifest.source_model.sha256 == primary_file.sha256,
        "source primary digest mismatch"
    );
    ensure!(
        manifest.layer_count == source.layer_count,
        "source block_count mismatch"
    );
    let mut source_metadata = source.shards[0].directory.metadata.clone();
    source_metadata.remove("split.no");
    source_metadata.remove("split.count");
    source_metadata.remove("split.tensors.count");
    ensure!(
        manifest.model_metadata == source_metadata,
        "source model metadata mismatch"
    );
    Ok(())
}

fn read_projectors(paths: &[PathBuf]) -> Result<Vec<SourceShard>> {
    paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let artifact_id = format!("projector-{index:05}");
            let (directory, tensors) = inspect(path, &artifact_id)?;
            Ok(SourceShard {
                path: path.clone(),
                source_file: SourceFile {
                    path: path.display().to_string(),
                    byte_size: directory.artifact_bytes,
                    sha256: file_sha256(path)?,
                },
                artifact_id,
                directory,
                tensors,
            })
        })
        .collect()
}

fn contained_file(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = fs::canonicalize(root.join(relative))
        .with_context(|| format!("open package file {relative:?}"))?;
    ensure!(
        path.starts_with(root) && path.is_file(),
        "package file {relative:?} escapes package directory or is not a file"
    );
    Ok(path)
}

fn verify_artifact_files(
    root: &Path,
    manifest: &PackageManifest,
    sources: &[&SourceShard],
) -> Result<BTreeMap<String, PathBuf>> {
    let source_paths = sources
        .iter()
        .map(|s| fs::canonicalize(&s.path))
        .collect::<std::io::Result<Vec<_>>>()?;
    for path in &source_paths {
        ensure!(
            !path.starts_with(root),
            "independent source must be outside package directory"
        );
    }
    let mut resolved = BTreeSet::new();
    let mut artifacts = BTreeMap::new();
    for artifact in &manifest.artifact_catalog.entries {
        let path = contained_file(root, &artifact.path)?;
        ensure!(
            resolved.insert(path.clone()),
            "duplicate resolved artifact path"
        );
        for source in &source_paths {
            ensure!(
                !same_file::is_same_file(source, &path)?,
                "package artifact cannot be its own independent source"
            );
        }
        ensure!(
            fs::metadata(&path)?.len() == artifact.byte_size,
            "artifact {:?} size mismatch",
            artifact.id
        );
        ensure!(
            file_sha256(&path)? == artifact.sha256,
            "artifact {:?} SHA-256 mismatch",
            artifact.id
        );
        artifacts.insert(artifact.id.clone(), path);
    }
    Ok(artifacts)
}

fn matching_artifact<'a>(
    manifest: &'a PackageManifest,
    source: &SourceShard,
    used: &mut BTreeSet<String>,
) -> Result<&'a Artifact> {
    let mut candidates = manifest.artifact_catalog.entries.iter().filter(|a| {
        a.sha256 == source.source_file.sha256 && a.byte_size == source.source_file.byte_size
    });
    let artifact = candidates.next().with_context(|| {
        format!(
            "no byte-preserving artifact for source {}",
            source.path.display()
        )
    })?;
    ensure!(
        candidates.next().is_none(),
        "ambiguous duplicate artifact for source {}",
        source.path.display()
    );
    ensure!(
        used.insert(artifact.id.clone()),
        "duplicate source/artifact correspondence"
    );
    Ok(artifact)
}

fn verify_projectors(
    manifest: &PackageManifest,
    projectors: &[SourceShard],
    artifacts: &BTreeMap<String, PathBuf>,
    used: &mut BTreeSet<String>,
) -> Result<()> {
    ensure!(
        projectors.len() == manifest.sidecars.len(),
        "independent source required for every projector sidecar"
    );
    let mut sidecars = BTreeSet::new();
    for sidecar in &manifest.sidecars {
        ensure!(
            sidecar.kind == skippy_package_format::SidecarKind::Mmproj,
            "unsupported sidecar kind {:?}",
            sidecar.kind
        );
        ensure!(
            !used.contains(&sidecar.artifact_id),
            "model artifact cannot also be a projector sidecar"
        );
        ensure!(
            sidecars.insert(&sidecar.artifact_id),
            "duplicate sidecar identity"
        );
    }
    for projector in projectors {
        let artifact = matching_artifact(manifest, projector, used)?;
        ensure!(
            sidecars.contains(&artifact.id),
            "projector source does not match a declared sidecar"
        );
        projector.verify_copy(&artifacts[&artifact.id])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
