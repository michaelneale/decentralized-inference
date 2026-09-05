//! Read-only verification against independently supplied source evidence.
//! This unit verifies byte-preserving whole-shard packages, not graph readiness.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use skippy_package_format::{Artifact, PackageManifest, SourceFile, TensorStorage};

use crate::hash::file_sha256;
use crate::source_inventory::{SourceInventory, SourceShard, inspect};

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
    manifest.validate()?;
    ensure!(
        manifest.format == "gguf",
        "only whole-shard GGUF v2 packages are supported"
    );
    ensure!(
        manifest
            .tensor_catalog
            .entries
            .iter()
            .all(|t| matches!(t.storage, TensorStorage::Owned { .. })),
        "alias claims cannot be proven by the pinned native GGUF inspector"
    );
    // The expected primary filename comes from the caller, never the manifest.
    let primary = source_file
        .or_else(|| source.file_name()?.to_str())
        .context("source primary filename is required")?;
    let inventory = SourceInventory::read_local(source, primary)?;
    verify_source_identity(&manifest, &inventory, primary)?;
    let projectors = read_projectors(source_projectors)?;
    let sources: Vec<_> = inventory.shards.iter().chain(&projectors).collect();
    let artifacts = verify_artifact_files(&root, &manifest, &sources)?;
    let mut used = BTreeSet::new();
    let mut expected = BTreeMap::new();
    for shard in &inventory.shards {
        let artifact = matching_artifact(&manifest, shard, &mut used)?;
        shard.verify_copy(&artifacts[&artifact.id])?;
        for tensor in &shard.tensors.entries {
            let mut tensor = tensor.clone();
            if let TensorStorage::Owned { artifact_id, .. } = &mut tensor.storage {
                *artifact_id = artifact.id.clone();
            }
            expected.insert(tensor.id.clone(), tensor);
        }
    }
    let declared: BTreeMap<_, _> = manifest
        .tensor_catalog
        .entries
        .iter()
        .map(|t| (t.id.clone(), t.clone()))
        .collect();
    ensure!(
        declared == expected,
        "manifest tensor catalog differs from independent source inventory"
    );
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
    ensure!(
        manifest.model_metadata == source.shards[0].directory.metadata,
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
