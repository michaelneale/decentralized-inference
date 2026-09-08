//! Source-complete v2 creation. Shards are physical containers, not stage owners.
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use skippy_package_format::{
    Artifact, ArtifactCatalog, PACKAGE_SCHEMA_VERSION, PackageManifest, Sidecar, SidecarKind,
    SourceModel,
};

use crate::hash::file_sha256;
use crate::package::{
    ArtifactHook, ExplicitSourceIdentity, PackageInput, resolve_package_input, run_artifact_hook,
};
use crate::progress::{PackageProgress, format_bytes};
use crate::source_inventory::{SourceInventory, inspect};
use crate::write::write_json_file;

pub(crate) fn write_package(
    model: String,
    out_dir: PathBuf,
    projectors: Vec<PathBuf>,
    artifact_hook: ArtifactHook,
    artifact_transform: ArtifactHook,
    explicit: ExplicitSourceIdentity,
    resume_existing_artifacts: bool,
) -> Result<()> {
    ensure!(
        artifact_transform.command.is_none(),
        "v2 creation preserves source bytes; transform the independent source before packaging, not package artifacts"
    );
    let input = resolve_package_input(model, explicit)?;
    let inventory = SourceInventory::read(&input)?;
    let mut manifest = manifest_from_source(&input, &inventory)?;
    // Validate source extents/aliases/identity before producing any output.
    manifest.package_id = manifest.computed_package_id()?;
    manifest.validate()?;
    fs::create_dir_all(&out_dir)?;
    ensure!(
        !out_dir.join("model-package.json").exists(),
        "output already contains model-package.json; use a new directory for v2 creation"
    );
    let mut progress = PackageProgress::new(inventory.shards.len() + projectors.len() + 1);
    for (shard, artifact) in inventory
        .shards
        .iter()
        .zip(&manifest.artifact_catalog.entries)
    {
        progress.start_step(&artifact.path)?;
        let path = out_dir.join(&artifact.path);
        copy_artifact(&shard.path, &path, resume_existing_artifacts)?;
        // Never construct the expected set from this output. Compare with the
        // frozen independent source directory, native extents, and digest.
        shard.verify_copy(&path)?;
        run_artifact_hook(&artifact_hook, &path, &artifact.path)?;
        if artifact_hook.command.is_some() && path.exists() {
            shard.verify_copy(&path)?;
        }
        progress.finish_step(&format!(
            "{} {}",
            artifact.path,
            format_bytes(artifact.byte_size)
        ))?;
    }
    for (index, projector) in projectors.iter().enumerate() {
        let artifact = copy_projector(projector, index, &out_dir, resume_existing_artifacts)?;
        progress.start_step(&artifact.path)?;
        run_artifact_hook(
            &artifact_hook,
            &out_dir.join(&artifact.path),
            &artifact.path,
        )?;
        if artifact_hook.command.is_some() && out_dir.join(&artifact.path).exists() {
            ensure!(
                file_sha256(&out_dir.join(&artifact.path))? == artifact.sha256,
                "projector changed after artifact hook"
            );
        }
        progress.finish_step(&format!(
            "{} {}",
            artifact.path,
            format_bytes(artifact.byte_size)
        ))?;
        manifest.sidecars.push(Sidecar {
            kind: SidecarKind::Mmproj,
            artifact_id: artifact.id.clone(),
            name: Some(artifact.id.clone()),
        });
        manifest.artifact_catalog.entries.push(artifact);
    }
    manifest.package_id = manifest.computed_package_id()?;
    manifest.validate()?;
    progress.start_step("model-package.json")?;
    // The completion marker is published only after all exact-coverage checks.
    let temporary = out_dir.join(".model-package-v2.json.tmp");
    ensure!(
        !temporary.exists(),
        "stale v2 manifest temporary file exists"
    );
    write_json_file(&temporary, &manifest)?;
    fs::rename(&temporary, out_dir.join("model-package.json"))?;
    progress.finish_step("model-package.json")?;
    progress.finish()?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn manifest_from_source(
    input: &PackageInput,
    inventory: &SourceInventory,
) -> Result<PackageManifest> {
    let identity = &input.source_identity;
    let primary = identity
        .primary_file
        .as_ref()
        .context("missing primary source identity")?;
    let primary_shard = inventory
        .shards
        .iter()
        .find(|s| &s.source_file.path == primary)
        .context("primary source absent from independent inventory")?;
    Ok(PackageManifest {
        schema_version: PACKAGE_SCHEMA_VERSION,
        package_id: String::new(),
        model_id: input.model_id.clone(),
        source_model: SourceModel {
            sha256: primary_shard.source_file.sha256.clone(),
            metadata_artifact_id: primary_shard.artifact_id.clone(),
            repo: identity.repo.clone(),
            revision: identity.revision.clone(),
            primary_file: identity.primary_file.clone(),
            canonical_ref: identity.canonical_ref.clone(),
            distribution_id: identity.distribution_id.clone(),
            files: inventory
                .shards
                .iter()
                .map(|s| s.source_file.clone())
                .collect(),
        },
        format: "gguf".to_string(),
        layer_count: inventory.layer_count,
        model_metadata: inventory.shards[0].directory.metadata.clone(),
        artifact_catalog: ArtifactCatalog {
            entries: inventory
                .shards
                .iter()
                .enumerate()
                .map(|(index, shard)| Artifact {
                    id: shard.artifact_id.clone(),
                    // llama.cpp resolves a sharded GGUF set from the primary
                    // shard's conventional `-00001-of-000NN` filename. Keep
                    // source names for multi-shard packages so runtime slices
                    // can open the package artifacts directly.
                    path: if inventory.shards.len() > 1 {
                        format!("artifacts/{}", shard.source_file.path)
                    } else {
                        format!("artifacts/source-{index:05}.gguf")
                    },
                    byte_size: shard.source_file.byte_size,
                    sha256: shard.source_file.sha256.clone(),
                })
                .collect(),
        },
        tensor_catalog: inventory.tensor_catalog(),
        sidecars: Vec::new(),
        generation: None,
        native_abi_version: format!(
            "{}.{}.{}",
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR,
            skippy_ffi::ABI_VERSION_PATCH
        ),
        generator_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before Unix epoch")?
            .as_secs(),
    })
}

fn copy_artifact(source: &Path, output: &Path, resume: bool) -> Result<()> {
    if resume && output.is_file() {
        ensure!(
            !same_file::is_same_file(source, output)?,
            "package artifact must not be the independent source file"
        );
        return Ok(());
    }
    fs::create_dir_all(
        output
            .parent()
            .context("artifact has no parent directory")?,
    )?;
    let mut input = File::open(source)?;
    let mut destination = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .with_context(|| {
            format!(
                "create artifact {}; use --resume-existing-artifacts to verify an existing copy",
                output.display()
            )
        })?;
    io::copy(&mut input, &mut destination)?;
    destination.sync_all()?;
    Ok(())
}

fn copy_projector(source: &Path, index: usize, out_dir: &Path, resume: bool) -> Result<Artifact> {
    let id = format!("projector-{index:05}");
    let (directory, tensors) = inspect(source, &id)?;
    let sha256 = file_sha256(source)?;
    let relative = format!("projectors/{id}.gguf");
    let path = out_dir.join(&relative);
    copy_artifact(source, &path, resume)?;
    let (written_directory, written_tensors) = inspect(&path, &id)?;
    ensure!(
        written_directory == directory
            && written_tensors == tensors
            && file_sha256(&path)? == sha256,
        "written projector differs from source"
    );
    Ok(Artifact {
        id,
        path: relative,
        byte_size: directory.artifact_bytes,
        sha256,
    })
}

#[cfg(test)]
mod tests;
