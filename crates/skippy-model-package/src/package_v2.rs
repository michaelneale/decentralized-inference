//! Source-complete v2 creation. Shards are physical containers, not stage owners.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use skippy_package_format::{
    Artifact, ArtifactCatalog, PACKAGE_SCHEMA_VERSION, PackageManifest, Sidecar, SidecarKind,
    SourceModel, Tensor, TensorCatalog,
};
use skippy_runtime::{ModelInfo, write_gguf_metadata_from_parts};

use crate::hash::file_sha256;
use crate::package::{
    ArtifactHook, ExplicitSourceIdentity, PackageInput, resolve_package_input, run_artifact_hook,
};
use crate::plan::StagePlan;
use crate::progress::{PackageProgress, format_bytes};
use crate::source_inventory::{SourceInventory, inspect};
use crate::write::{ModelSource, create_parent_dir, write_json_file, write_stage_artifact};

mod layout;

use layout::{PlannedArtifact, PlannedArtifactKind, plan_artifacts};

struct SourceTensor {
    path: PathBuf,
    tensor: Tensor,
}

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
    let source = ModelSource::open(&input.model_path)?;
    ensure_native_inventory_matches(&inventory, &source)?;
    let planned = plan_artifacts(&source.tensors)?;
    let mut manifest = manifest_from_source(&input, &inventory)?;
    fs::create_dir_all(&out_dir)?;
    ensure!(
        !out_dir.join("model-package.json").exists(),
        "output already contains model-package.json; use a new directory for v2 creation"
    );
    let mut progress = PackageProgress::new(planned.len() + projectors.len() + 2);
    progress.start_step("shared/metadata.gguf")?;
    let metadata_artifact = emit_metadata_artifact(
        &source,
        &inventory,
        &out_dir,
        &artifact_hook,
        resume_existing_artifacts,
    )?;
    progress.finish_step(&format!(
        "{} {}",
        metadata_artifact.path,
        format_bytes(metadata_artifact.byte_size)
    ))?;
    manifest.artifact_catalog.entries.push(metadata_artifact);

    let source_tensors = source_tensors_by_name(&inventory)?;
    let common_names = planned
        .iter()
        .find(|artifact| artifact.kind == PlannedArtifactKind::Common)
        .map(|artifact| artifact.tensor_names.iter().cloned().collect())
        .unwrap_or_default();
    let mut catalog = Vec::with_capacity(source_tensors.len());
    for (stage_index, artifact_plan) in planned.iter().enumerate() {
        progress.start_step(&artifact_plan.path)?;
        let (artifact, mut tensors) = emit_payload_artifact(
            &source,
            &source_tensors,
            artifact_plan,
            &common_names,
            inventory.layer_count,
            stage_index,
            &out_dir,
            &artifact_hook,
            resume_existing_artifacts,
        )?;
        progress.finish_step(&format!(
            "{} {}",
            artifact.path,
            format_bytes(artifact.byte_size)
        ))?;
        manifest.artifact_catalog.entries.push(artifact);
        catalog.append(&mut tensors);
    }
    catalog.sort_by(|left, right| left.id.cmp(&right.id));
    ensure!(
        catalog
            .iter()
            .map(|tensor| tensor.id.as_str())
            .collect::<BTreeSet<_>>()
            == source_tensors
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
        "emitted payload tensor catalog differs from independent source inventory"
    );
    manifest.tensor_catalog = TensorCatalog { entries: catalog };
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
            metadata_artifact_id: "metadata".to_string(),
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
            entries: Vec::new(),
        },
        tensor_catalog: TensorCatalog {
            entries: Vec::new(),
        },
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

fn ensure_native_inventory_matches(
    inventory: &SourceInventory,
    source: &ModelSource,
) -> Result<()> {
    let expected = inventory
        .shards
        .iter()
        .flat_map(|shard| {
            shard
                .tensors
                .entries
                .iter()
                .map(|tensor| tensor.name.as_str())
        })
        .collect::<BTreeSet<_>>();
    let actual = source
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        actual.len() == source.tensors.len() && actual == expected,
        "native tensor inventory differs from independent source inventory"
    );
    Ok(())
}

fn source_tensors_by_name(inventory: &SourceInventory) -> Result<BTreeMap<String, SourceTensor>> {
    let mut tensors = BTreeMap::new();
    for shard in &inventory.shards {
        for tensor in &shard.tensors.entries {
            ensure!(
                tensors
                    .insert(
                        tensor.name.clone(),
                        SourceTensor {
                            path: shard.path.clone(),
                            tensor: tensor.clone(),
                        },
                    )
                    .is_none(),
                "duplicate source tensor {:?}",
                tensor.name
            );
        }
    }
    Ok(tensors)
}

fn emit_metadata_artifact(
    source: &ModelSource,
    inventory: &SourceInventory,
    out_dir: &Path,
    artifact_hook: &ArtifactHook,
    resume: bool,
) -> Result<Artifact> {
    let relative = "shared/metadata.gguf";
    let path = out_dir.join(relative);
    create_parent_dir(&path)?;
    ensure_not_source_file(source, &path)?;
    if !path.exists() {
        write_gguf_metadata_from_parts(&source.paths, &path)
            .with_context(|| format!("write GGUF metadata carrier {}", path.display()))?;
    } else {
        ensure!(
            resume,
            "artifact {} already exists; use --resume-existing-artifacts to verify it",
            path.display()
        );
    }
    let info = ModelInfo::open(&path)
        .with_context(|| format!("open metadata carrier {}", path.display()))?;
    let tensors = info.tensors()?;
    let expected = inventory
        .shards
        .iter()
        .flat_map(|shard| shard.tensors.entries.iter())
        .map(|tensor| (&tensor.name, tensor))
        .collect::<BTreeMap<_, _>>();
    let descriptors_match = tensors.iter().all(|tensor| {
        expected.get(&tensor.name).is_some_and(|source| {
            source
                .dimensions
                .iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
                .is_some_and(|element_count| {
                    tensor.ggml_type == source.ggml_type && tensor.element_count == element_count
                })
        })
    });
    ensure!(
        tensors.len() == expected.len() && descriptors_match,
        "metadata carrier descriptor table differs from independent source inventory"
    );
    let artifact = artifact_record("metadata", relative, &path)?;
    run_artifact_hook(artifact_hook, &path, relative)?;
    verify_hook_result(&artifact, &path, artifact_hook)?;
    Ok(artifact)
}

#[allow(clippy::too_many_arguments)]
fn emit_payload_artifact(
    source: &ModelSource,
    source_tensors: &BTreeMap<String, SourceTensor>,
    planned: &PlannedArtifact,
    common_names: &BTreeSet<String>,
    layer_count: u32,
    stage_index: usize,
    out_dir: &Path,
    artifact_hook: &ArtifactHook,
    resume: bool,
) -> Result<(Artifact, Vec<Tensor>)> {
    let path = out_dir.join(&planned.path);
    ensure_not_source_file(source, &path)?;
    if !path.exists() {
        let stage = stage_plan(planned, stage_index, layer_count);
        write_stage_artifact(source, &stage, &path)?;
    } else {
        ensure!(
            resume,
            "artifact {} already exists; use --resume-existing-artifacts to verify it",
            path.display()
        );
    }
    let (_, emitted) = inspect(&path, &planned.id)?;
    let emitted_by_name = emitted
        .entries
        .into_iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect::<BTreeMap<_, _>>();
    let expected_physical = planned
        .tensor_names
        .iter()
        .chain(common_names)
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    ensure!(
        emitted_by_name
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == expected_physical,
        "written artifact {:?} differs from its native role plan",
        planned.id
    );
    let layer_ordinal = match planned.kind {
        PlannedArtifactKind::Layer { ordinal } => Some(ordinal),
        PlannedArtifactKind::Common
        | PlannedArtifactKind::Embeddings
        | PlannedArtifactKind::Output => None,
    };
    let mut bound = Vec::with_capacity(planned.tensor_names.len());
    for name in &planned.tensor_names {
        let mut tensor = emitted_by_name
            .get(name)
            .with_context(|| format!("written artifact {:?} omitted tensor {name:?}", planned.id))?
            .clone();
        let expected = source_tensors
            .get(name)
            .context("planned tensor is absent from independent source inventory")?;
        ensure!(
            tensor.name == expected.tensor.name
                && tensor.ggml_type == expected.tensor.ggml_type
                && tensor.dimensions == expected.tensor.dimensions,
            "written tensor {name:?} metadata differs from independent source inventory"
        );
        compare_tensor_payload(name, expected, &path, &tensor)?;
        tensor.layer_ordinal = layer_ordinal;
        bound.push(tensor);
    }
    let artifact = artifact_record(&planned.id, &planned.path, &path)?;
    run_artifact_hook(artifact_hook, &path, &planned.path)?;
    verify_hook_result(&artifact, &path, artifact_hook)?;
    Ok((artifact, bound))
}

fn stage_plan(planned: &PlannedArtifact, stage_index: usize, layer_count: u32) -> StagePlan {
    let (layer_start, layer_end, includes_embeddings, includes_output) = match planned.kind {
        PlannedArtifactKind::Common => (0, 0, false, false),
        PlannedArtifactKind::Embeddings => (0, 0, true, false),
        PlannedArtifactKind::Output => (layer_count, layer_count, false, true),
        PlannedArtifactKind::Layer { ordinal } => (ordinal, ordinal + 1, false, false),
    };
    StagePlan {
        stage_index,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        includes_per_layer_token_embd: planned
            .tensor_names
            .iter()
            .any(|name| name == crate::plan::PER_LAYER_TOKEN_EMBD),
        tensor_count: planned.tensor_names.len(),
        tensor_bytes: 0,
    }
}

fn artifact_record(id: &str, relative: &str, path: &Path) -> Result<Artifact> {
    Ok(Artifact {
        id: id.to_string(),
        path: relative.to_string(),
        byte_size: fs::metadata(path)?.len(),
        sha256: file_sha256(path)?,
    })
}

fn compare_tensor_payload(
    name: &str,
    source: &SourceTensor,
    emitted_path: &Path,
    emitted: &Tensor,
) -> Result<()> {
    let (
        skippy_package_format::TensorStorage::Owned {
            data_offset: source_offset,
            stored_length: source_length,
            ..
        },
        skippy_package_format::TensorStorage::Owned {
            data_offset: emitted_offset,
            stored_length: emitted_length,
            ..
        },
    ) = (&source.tensor.storage, &emitted.storage)
    else {
        anyhow::bail!("tensor {name:?} has unsupported alias storage")
    };
    ensure!(
        source_length == emitted_length,
        "written tensor {name:?} stored length differs from independent source"
    );
    let mut source_file = File::open(&source.path)?;
    let mut emitted_file = File::open(emitted_path)?;
    source_file.seek(SeekFrom::Start(*source_offset))?;
    emitted_file.seek(SeekFrom::Start(*emitted_offset))?;
    let mut source_buffer = [0_u8; 64 * 1024];
    let mut emitted_buffer = [0_u8; 64 * 1024];
    let mut remaining = *source_length;
    while remaining > 0 {
        let length = usize::try_from(remaining.min(source_buffer.len() as u64))?;
        source_file.read_exact(&mut source_buffer[..length])?;
        emitted_file.read_exact(&mut emitted_buffer[..length])?;
        ensure!(
            source_buffer[..length] == emitted_buffer[..length],
            "written tensor {name:?} payload differs from independent source"
        );
        remaining -= length as u64;
    }
    Ok(())
}

fn ensure_not_source_file(source: &ModelSource, path: &Path) -> Result<()> {
    if path.exists() {
        for source_path in &source.paths {
            ensure!(
                !same_file::is_same_file(source_path, path)?,
                "package artifact must not be the independent source file"
            );
        }
    }
    Ok(())
}

fn verify_hook_result(artifact: &Artifact, path: &Path, hook: &ArtifactHook) -> Result<()> {
    if hook.command.is_some() && path.exists() {
        ensure!(
            fs::metadata(path)?.len() == artifact.byte_size
                && file_sha256(path)? == artifact.sha256,
            "artifact {:?} changed after artifact hook",
            artifact.id
        );
    }
    Ok(())
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
