use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use anyhow::{Context, Result, bail};
use model_artifact::{ModelArtifactFile, ResolvedModelArtifact};
use model_hf::HfModelRepository;
use model_ref::{format_canonical_ref, normalize_gguf_distribution_id, parse_model_ref};
use serde::{Deserialize, Serialize};

use crate::write::local_artifact_files;

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageManifest {
    pub(crate) schema_version: u32,
    pub(crate) model_id: String,
    pub(crate) source_model: PackageSourceModel,
    pub(crate) format: String,
    pub(crate) layer_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) generation: Option<PackageGeneration>,
    pub(crate) shared: PackageShared,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) projectors: Vec<PackageProjector>,
    pub(crate) layers: Vec<PackageLayer>,
    pub(crate) skippy_abi_version: String,
    pub(crate) created_at_unix_secs: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageSourceModel {
    pub(crate) path: String,
    pub(crate) sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) repo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) primary_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) canonical_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) distribution_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) files: Vec<ModelArtifactFile>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageShared {
    pub(crate) metadata: PackageArtifact,
    pub(crate) embeddings: PackageArtifact,
    pub(crate) output: PackageArtifact,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageGeneration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) speculative_decoding: Option<PackageSpeculativeDecoding>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageSpeculativeDecoding {
    pub(crate) default: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) proposers: BTreeMap<String, PackageSpeculativeProposer>,
    pub(crate) strategies: BTreeMap<String, PackageSpeculativeStrategy>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageSpeculativeProposer {
    #[serde(rename = "type")]
    pub(crate) proposer_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) prediction_depth: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) layer_indices: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) ngram_min: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) ngram_max: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max_proposal_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) history_scope: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageSpeculativeStrategy {
    #[serde(rename = "type")]
    pub(crate) strategy_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) prediction_depth: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) layer_indices: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) window_policy: Option<PackageWindowPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) proposer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) primary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) extender: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) extension_policy: Option<PackageExtensionPolicy>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageExtensionPolicy {
    pub(crate) max_tokens: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageWindowPolicy {
    pub(crate) default: String,
    pub(crate) initial_window: u32,
    pub(crate) min_window: u32,
    pub(crate) max_window: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) pipeline_depth: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageLayer {
    pub(crate) layer_index: u32,
    pub(crate) path: String,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageArtifact {
    pub(crate) path: String,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct PackageProjector {
    pub(crate) kind: String,
    pub(crate) path: String,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
    pub(crate) artifact_bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ArtifactHook {
    pub(crate) command: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub(crate) struct ExplicitSourceIdentity {
    pub(crate) model_id: Option<String>,
    pub(crate) source_repo: Option<String>,
    pub(crate) source_revision: Option<String>,
    pub(crate) source_file: Option<String>,
}

#[derive(Debug)]
pub(crate) struct PackageInput {
    pub(crate) model_path: PathBuf,
    pub(crate) model_id: String,
    pub(crate) source_identity: PackageSourceIdentity,
}

#[derive(Debug)]
pub(crate) struct PackageSourceIdentity {
    pub(crate) repo: Option<String>,
    pub(crate) revision: Option<String>,
    pub(crate) primary_file: Option<String>,
    pub(crate) canonical_ref: Option<String>,
    pub(crate) distribution_id: Option<String>,
    pub(crate) files: Vec<ModelArtifactFile>,
}

pub(crate) fn resolve_package_input(
    model: String,
    explicit: ExplicitSourceIdentity,
) -> Result<PackageInput> {
    let path = PathBuf::from(&model);
    if path.exists() {
        return resolve_local_package_input(path, explicit);
    }

    if explicit.model_id.is_some()
        || explicit.source_repo.is_some()
        || explicit.source_revision.is_some()
        || explicit.source_file.is_some()
    {
        bail!(
            "explicit source identity flags are only valid when write-package input is a local path"
        );
    }

    parse_model_ref(&model).with_context(|| {
        format!(
            "write-package input must be a model coordinate like org/repo:Q4_K_M, not {model:?}"
        )
    })?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build async runtime for Hugging Face model resolution")?;

    runtime.block_on(async {
        let repository = HfModelRepository::from_env()?;
        let artifact = model_artifact::resolve_model_artifact_ref(&model, &repository).await?;
        let paths = repository.download_artifact_files(&artifact).await?;
        let primary_index = artifact
            .files
            .iter()
            .position(|file| file.path == artifact.primary_file)
            .context("resolved artifact file list did not include primary file")?;
        let model_path = paths
            .get(primary_index)
            .cloned()
            .context("downloaded artifact path list did not include primary file")?;
        Ok(package_input_from_resolved_artifact(model_path, artifact))
    })
}

pub(crate) fn resolve_local_package_input(
    model_path: PathBuf,
    explicit: ExplicitSourceIdentity,
) -> Result<PackageInput> {
    let model_id = explicit.model_id.context(
        "local write-package input requires --model-id; prefer passing a coordinate like org/repo:Q4_K_M",
    )?;
    let parsed_model_id = parse_model_ref(&model_id)
        .with_context(|| format!("--model-id must be a model coordinate, got {model_id:?}"))?;
    let cache_identity = if explicit.source_revision.is_none() || explicit.source_file.is_none() {
        HfModelRepository::from_env()
            .ok()
            .and_then(|repository| repository.identity_for_path(&model_path))
    } else {
        None
    };

    let repo = explicit
        .source_repo
        .or_else(|| {
            cache_identity
                .as_ref()
                .map(|identity| identity.repo_id.clone())
        })
        .unwrap_or_else(|| parsed_model_id.repo.clone());
    let revision = explicit
        .source_revision
        .or_else(|| cache_identity.as_ref().map(|identity| identity.revision.clone()))
        .context("local write-package input requires --source-revision for paths outside the Hugging Face cache")?;
    let primary_file = explicit
        .source_file
        .or_else(|| cache_identity.as_ref().map(|identity| identity.file.clone()))
        .context("local write-package input requires --source-file for paths outside the Hugging Face cache")?;
    let canonical_ref = format_canonical_ref(&repo, &revision, &primary_file);
    let distribution_id = normalize_gguf_distribution_id(&primary_file);
    let files = local_artifact_files(&model_path, &primary_file)?;

    Ok(PackageInput {
        model_path,
        model_id: parsed_model_id.display_id(),
        source_identity: PackageSourceIdentity {
            repo: Some(repo),
            revision: Some(revision),
            primary_file: Some(primary_file.clone()),
            canonical_ref: Some(canonical_ref),
            distribution_id,
            files,
        },
    })
}

fn package_input_from_resolved_artifact(
    model_path: PathBuf,
    artifact: ResolvedModelArtifact,
) -> PackageInput {
    PackageInput {
        model_path,
        model_id: artifact.model_id,
        source_identity: PackageSourceIdentity {
            repo: Some(artifact.source_repo),
            revision: Some(artifact.source_revision),
            primary_file: Some(artifact.primary_file),
            canonical_ref: Some(artifact.canonical_ref),
            distribution_id: Some(artifact.distribution_id),
            files: artifact.files,
        },
    }
}

#[cfg(test)]
pub(crate) fn model_distribution_id(model: &Path) -> Option<String> {
    model
        .to_str()
        .and_then(normalize_gguf_distribution_id)
        .or_else(|| {
            model
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(normalize_gguf_distribution_id)
        })
}

pub(crate) fn run_artifact_hook(
    artifact_hook: &ArtifactHook,
    absolute_path: &Path,
    relative_path: &str,
) -> Result<()> {
    let Some(command) = &artifact_hook.command else {
        return Ok(());
    };
    let status = ProcessCommand::new(command)
        .env("SKIPPY_PACKAGE_ARTIFACT_PATH", absolute_path)
        .env("SKIPPY_PACKAGE_ARTIFACT_RELATIVE_PATH", relative_path)
        .status()
        .with_context(|| format!("run artifact hook {}", command.display()))?;
    if !status.success() {
        bail!(
            "artifact hook {} failed for {} with status {status}",
            command.display(),
            relative_path
        );
    }
    Ok(())
}
