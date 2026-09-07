use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use skippy_ffi::TensorRole;
use skippy_package_format::{
    Artifact, ArtifactCatalog, PACKAGE_SCHEMA_VERSION, PackageManifest as PackageManifestV2,
    ProposerKind, SourceFile, SourceModel, StrategyKind, Tensor, TensorCatalog, TensorIntegrity,
    TensorStorage,
};
use skippy_runtime::package::PackageGenerationInfo;

use super::hash_cache::{self, SidecarDigestCache};

mod content_addressed;
mod legacy_identity;

pub use content_addressed::synthetic_content_addressed_gguf_package;

const PACKAGE_V2_MANIFEST: &str = "model-package.json";

pub(crate) fn is_package_v2_ref(package_ref: &str) -> bool {
    let manifest_path = Path::new(package_ref).join(PACKAGE_V2_MANIFEST);
    std::fs::read(&manifest_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|manifest| {
            manifest
                .get("schema_version")
                .and_then(serde_json::Value::as_u64)
        })
        == Some(u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SkippyPackageIdentity {
    pub package_ref: String,
    pub manifest_sha256: String,
    pub source_model_path: PathBuf,
    pub source_model_sha256: String,
    pub source_model_bytes: u64,
    pub source_files: Vec<SkippyPackageSourceFile>,
    pub layer_weight_bytes: Vec<u64>,
    pub layer_count: u32,
    pub activation_width: u32,
    pub tensor_count: u64,
    pub generation: Option<PackageGenerationInfo>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SkippyPackageSourceFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

/// Resolve a validated package-v2 directory into the existing host planning
/// identity. The content-derived package ID remains in the v2 manifest and is
/// carried into split control by the generation-8 admission descriptor.
pub fn identity_from_package_v2(package_dir: &Path) -> Result<SkippyPackageIdentity> {
    let package_dir = package_dir.canonicalize().with_context(|| {
        format!(
            "canonicalize package-v2 directory {}",
            package_dir.display()
        )
    })?;
    anyhow::ensure!(
        package_dir.is_dir(),
        "package-v2 path is not a directory: {}",
        package_dir.display()
    );
    let manifest_path = package_dir.join(PACKAGE_V2_MANIFEST);
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?;
    let schema_version = serde_json::from_slice::<serde_json::Value>(&manifest_bytes)
        .context("parse package manifest envelope")?
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .context("package manifest is missing integer schema_version")?;
    anyhow::ensure!(
        schema_version == u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION),
        "split serving requires package schema {}; found schema {schema_version}",
        skippy_package_format::PACKAGE_SCHEMA_VERSION
    );
    let manifest: PackageManifestV2 =
        serde_json::from_slice(&manifest_bytes).context("parse package-v2 manifest")?;
    manifest
        .validate()
        .context("validate package-v2 manifest")?;
    let computed_package_id = manifest
        .computed_package_id()
        .context("compute package-v2 identity")?;
    anyhow::ensure!(
        manifest.package_id == computed_package_id,
        "package-v2 manifest package_id does not match its content"
    );
    let required_native_abi = format!(
        "{}.{}.{}",
        skippy_ffi::ABI_VERSION_MAJOR,
        skippy_ffi::ABI_VERSION_MINOR,
        skippy_ffi::ABI_VERSION_PATCH
    );
    anyhow::ensure!(
        manifest.native_abi_version == required_native_abi,
        "package-v2 native ABI {} differs from runtime ABI {required_native_abi}",
        manifest.native_abi_version
    );

    let metadata_artifact = manifest
        .artifact_catalog
        .entries
        .iter()
        .find(|artifact| artifact.id == manifest.source_model.metadata_artifact_id)
        .context("package-v2 metadata artifact is absent")?;
    let source_model_path = safe_package_v2_artifact_path(&package_dir, &metadata_artifact.path)?;
    let source_metadata = source_model_path
        .metadata()
        .with_context(|| format!("stat package-v2 source {}", source_model_path.display()))?;
    anyhow::ensure!(
        source_metadata.is_file(),
        "package-v2 metadata artifact is not a file: {}",
        source_model_path.display()
    );
    anyhow::ensure!(
        source_metadata.len() == metadata_artifact.byte_size,
        "package-v2 metadata artifact size {} differs from manifest {}",
        source_metadata.len(),
        metadata_artifact.byte_size
    );
    let source_sha256 = source_file_sha256(
        &source_model_path,
        &source_metadata,
        SidecarDigestCache::open_default().as_ref(),
    )?;
    anyhow::ensure!(
        source_sha256 == metadata_artifact.sha256,
        "package-v2 metadata artifact SHA-256 differs from manifest"
    );
    anyhow::ensure!(
        source_sha256 == manifest.source_model.sha256,
        "package-v2 metadata artifact SHA-256 differs from source model identity"
    );

    let architecture = manifest
        .model_metadata
        .get("general.architecture")
        .and_then(serde_json::Value::as_str)
        .context("package-v2 model metadata is missing general.architecture")?;
    let activation_width_key = format!("{architecture}.embedding_length");
    let activation_width = manifest
        .model_metadata
        .get(&activation_width_key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .with_context(|| {
            format!("package-v2 model metadata is missing positive {activation_width_key}")
        })?;

    let source_model_bytes = manifest
        .source_model
        .files
        .iter()
        .try_fold(0_u64, |total, file| total.checked_add(file.byte_size))
        .context("package-v2 source byte count overflow")?;
    anyhow::ensure!(
        source_model_bytes > 0,
        "package-v2 source model byte count must be positive"
    );
    let mut source_files = package_v2_source_files(&package_dir, &manifest)?;
    let layer_weight_bytes = package_v2_layer_weight_bytes(&manifest)?;
    let tensor_count = u64::try_from(manifest.tensor_catalog.entries.len())
        .context("package-v2 tensor count exceeds u64")?;
    let manifest_sha256 = hex_lower(&Sha256::digest(&manifest_bytes));
    source_files.push(SkippyPackageSourceFile {
        path: manifest_path,
        bytes: u64::try_from(manifest_bytes.len())
            .context("package-v2 manifest byte count exceeds u64")?,
        sha256: manifest_sha256.clone(),
    });
    let generation = manifest.generation.as_ref().map(package_v2_generation_info);

    Ok(SkippyPackageIdentity {
        package_ref: package_dir.to_string_lossy().into_owned(),
        manifest_sha256,
        source_model_path,
        source_model_sha256: manifest.source_model.sha256,
        source_model_bytes,
        source_files,
        layer_weight_bytes,
        layer_count: manifest.layer_count,
        activation_width,
        tensor_count,
        generation,
    })
}

fn safe_package_v2_artifact_path(package_dir: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    anyhow::ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_))),
        "package-v2 artifact path is not a safe relative path: {relative:?}"
    );
    let resolved = package_dir
        .join(relative)
        .canonicalize()
        .with_context(|| format!("canonicalize package-v2 artifact {relative:?}"))?;
    anyhow::ensure!(
        resolved.starts_with(package_dir),
        "package-v2 artifact escapes its package directory: {relative:?}"
    );
    Ok(resolved)
}

fn package_v2_source_files(
    package_dir: &Path,
    manifest: &PackageManifestV2,
) -> Result<Vec<SkippyPackageSourceFile>> {
    let referenced_artifact_ids = manifest
        .tensor_catalog
        .entries
        .iter()
        .filter_map(|tensor| match &tensor.storage {
            TensorStorage::Owned { artifact_id, .. } => Some(artifact_id.as_str()),
            TensorStorage::Alias { .. } => None,
        })
        .chain(std::iter::once(
            manifest.source_model.metadata_artifact_id.as_str(),
        ))
        .collect::<std::collections::BTreeSet<_>>();
    let mut artifacts = manifest
        .artifact_catalog
        .entries
        .iter()
        .filter(|artifact| referenced_artifact_ids.contains(artifact.id.as_str()))
        .collect::<Vec<_>>();
    artifacts.sort_by(|left, right| left.id.cmp(&right.id));
    let digest_cache = SidecarDigestCache::open_default();
    artifacts
        .into_iter()
        .map(|artifact| {
            let path = safe_package_v2_artifact_path(package_dir, &artifact.path)?;
            let source = source_file(&path, digest_cache.as_ref())?;
            anyhow::ensure!(
                source.bytes == artifact.byte_size,
                "package-v2 artifact {:?} size differs from manifest",
                artifact.id
            );
            anyhow::ensure!(
                source.sha256 == artifact.sha256,
                "package-v2 artifact {:?} SHA-256 differs from manifest",
                artifact.id
            );
            Ok(source)
        })
        .collect()
}

fn package_v2_layer_weight_bytes(manifest: &PackageManifestV2) -> Result<Vec<u64>> {
    if !manifest
        .tensor_catalog
        .entries
        .iter()
        .any(|tensor| tensor.layer_ordinal.is_some())
    {
        return Ok(Vec::new());
    }
    let mut layer_bytes = vec![0_u64; manifest.layer_count as usize];
    for tensor in &manifest.tensor_catalog.entries {
        let Some(layer) = tensor.layer_ordinal else {
            continue;
        };
        let TensorStorage::Owned { stored_length, .. } = tensor.storage else {
            continue;
        };
        let slot = layer_bytes
            .get_mut(layer as usize)
            .with_context(|| format!("package-v2 tensor layer {layer} exceeds layer count"))?;
        *slot = slot
            .checked_add(stored_length)
            .context("package-v2 layer byte count overflow")?;
    }
    if layer_bytes.contains(&0) {
        return Ok(Vec::new());
    }
    Ok(layer_bytes)
}

fn package_v2_generation_info(
    generation: &skippy_package_format::Generation,
) -> PackageGenerationInfo {
    PackageGenerationInfo {
        speculative_decoding: generation.speculative_decoding.as_ref().map(|speculative| {
            skippy_runtime::package::PackageSpeculativeDecodingInfo {
                default: speculative.default.clone(),
                proposers: speculative
                    .proposers
                    .iter()
                    .map(|(name, proposer)| {
                        let info = match &proposer.kind {
                            ProposerKind::NativeMtp {
                                prediction_depth,
                                layer_indices,
                            } => skippy_runtime::package::PackageSpeculativeProposerInfo {
                                proposer_type: "native-mtp".to_string(),
                                prediction_depth: Some(*prediction_depth),
                                layer_indices: layer_indices.clone(),
                                ngram_min: None,
                                ngram_max: None,
                                max_proposal_tokens: None,
                                history_scope: None,
                            },
                            ProposerKind::NgramCache {
                                ngram_min,
                                ngram_max,
                                max_proposal_tokens,
                                history_scope,
                            }
                            | ProposerKind::NgramSuffix {
                                ngram_min,
                                ngram_max,
                                max_proposal_tokens,
                                history_scope,
                            } => skippy_runtime::package::PackageSpeculativeProposerInfo {
                                proposer_type: match &proposer.kind {
                                    ProposerKind::NgramCache { .. } => "ngram-cache",
                                    ProposerKind::NgramSuffix { .. } => "ngram-suffix",
                                    ProposerKind::NativeMtp { .. } => unreachable!(),
                                }
                                .to_string(),
                                prediction_depth: None,
                                layer_indices: Vec::new(),
                                ngram_min: Some(*ngram_min),
                                ngram_max: Some(*ngram_max),
                                max_proposal_tokens: Some(*max_proposal_tokens),
                                history_scope: Some(history_scope.clone()),
                            },
                        };
                        (name.clone(), info)
                    })
                    .collect(),
                strategies: speculative
                    .strategies
                    .iter()
                    .map(|(name, strategy)| {
                        let info = match &strategy.kind {
                            StrategyKind::NativeMtp {
                                proposer,
                                prediction_depth,
                                layer_indices,
                                window_policy,
                            } => skippy_runtime::package::PackageSpeculativeStrategyInfo {
                                strategy_type: "native-mtp".to_string(),
                                prediction_depth: *prediction_depth,
                                layer_indices: layer_indices.clone(),
                                window_policy: window_policy
                                    .as_ref()
                                    .map(package_v2_window_policy_info),
                                proposer: proposer.clone(),
                                primary: None,
                                extender: None,
                                extension_policy: None,
                            },
                            StrategyKind::NgramCache {
                                proposer,
                                window_policy,
                            }
                            | StrategyKind::NgramSuffix {
                                proposer,
                                window_policy,
                            } => skippy_runtime::package::PackageSpeculativeStrategyInfo {
                                strategy_type: match &strategy.kind {
                                    StrategyKind::NgramCache { .. } => "ngram-cache",
                                    StrategyKind::NgramSuffix { .. } => "ngram-suffix",
                                    StrategyKind::NativeMtp { .. }
                                    | StrategyKind::Composite { .. } => unreachable!(),
                                }
                                .to_string(),
                                prediction_depth: None,
                                layer_indices: Vec::new(),
                                window_policy: window_policy
                                    .as_ref()
                                    .map(package_v2_window_policy_info),
                                proposer: Some(proposer.clone()),
                                primary: None,
                                extender: None,
                                extension_policy: None,
                            },
                            StrategyKind::Composite {
                                primary,
                                extender,
                                extension_policy,
                                window_policy,
                            } => skippy_runtime::package::PackageSpeculativeStrategyInfo {
                                strategy_type: "composite".to_string(),
                                prediction_depth: None,
                                layer_indices: Vec::new(),
                                window_policy: window_policy
                                    .as_ref()
                                    .map(package_v2_window_policy_info),
                                proposer: None,
                                primary: Some(primary.clone()),
                                extender: Some(extender.clone()),
                                extension_policy: Some(
                                    skippy_runtime::package::PackageExtensionPolicyInfo {
                                        max_tokens: extension_policy.max_tokens,
                                    },
                                ),
                            },
                        };
                        (name.clone(), info)
                    })
                    .collect(),
            }
        }),
    }
}

fn package_v2_window_policy_info(
    policy: &skippy_package_format::WindowPolicy,
) -> skippy_runtime::package::PackageWindowPolicyInfo {
    skippy_runtime::package::PackageWindowPolicyInfo {
        default: policy.default.clone(),
        initial_window: policy.initial_window,
        min_window: policy.min_window,
        max_window: policy.max_window,
        pipeline_depth: policy.pipeline_depth,
    }
}

#[derive(Serialize)]
struct SyntheticGgufManifest<'a> {
    schema_version: u32,
    package_kind: &'a str,
    model_id: &'a str,
    package_ref: &'a str,
    source_model_path: &'a str,
    source_model_sha256: &'a str,
    source_model_bytes: u64,
    source_files: &'a [SyntheticGgufManifestFile],
    architecture: &'a str,
    context_length: u32,
    layer_count: u32,
    activation_width: u32,
    tensor_count: u64,
}

#[derive(Serialize)]
struct SyntheticGgufManifestFile {
    path: String,
    bytes: u64,
    sha256: String,
}

pub fn synthetic_direct_gguf_package(
    model_id: &str,
    model_path: &Path,
) -> Result<SkippyPackageIdentity> {
    if let Some(root) = safetensors_checkpoint_root(model_path) {
        return synthetic_safetensors_package(model_id, &root);
    }
    synthetic_gguf_package(model_id, model_path)
}

fn safetensors_checkpoint_root(model_path: &Path) -> Option<PathBuf> {
    let root = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent()?
    };
    (root.join("config.json").is_file()
        && (root.join("model.safetensors").is_file()
            || root.join("model.safetensors.index.json").is_file()))
    .then(|| root.to_path_buf())
}

fn synthetic_safetensors_package(
    model_id: &str,
    checkpoint_root: &Path,
) -> Result<SkippyPackageIdentity> {
    let checkpoint_root = checkpoint_root
        .canonicalize()
        .with_context(|| format!("canonicalize checkpoint {}", checkpoint_root.display()))?;
    let config_path = checkpoint_root.join("config.json");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path)
            .with_context(|| format!("read checkpoint config {}", config_path.display()))?,
    )
    .with_context(|| format!("parse checkpoint config {}", config_path.display()))?;
    let config_u32 = |key: &str| -> Result<u32> {
        let value = config
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .with_context(|| format!("checkpoint config missing integer {key}"))?;
        u32::try_from(value).with_context(|| format!("checkpoint config {key} exceeds u32"))
    };
    let layer_count = config_u32("num_hidden_layers")?;
    let activation_width = config_u32("hidden_size")?;
    anyhow::ensure!(layer_count > 0, "checkpoint layer count must be positive");
    anyhow::ensure!(
        activation_width > 0,
        "checkpoint hidden size must be positive"
    );
    let architecture = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("safetensors");
    let context_length = config
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or_else(|| {
            if architecture == "granitemoehybrid" {
                1 << 20
            } else {
                0
            }
        });
    let plan = skippy_model::hf_checkpoint::inspect_hf_checkpoint(&checkpoint_root, None, 1.0)?;
    let tensor_count = u64::try_from(plan.tensor_count).context("tensor count exceeds u64")?;

    let mut source_paths = skippy_model::hf_checkpoint::discover_safetensors(&checkpoint_root)?;
    for name in [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "chat_template.jinja",
    ] {
        let path = checkpoint_root.join(name);
        if path.is_file() {
            source_paths.push(path);
        }
    }
    source_paths.sort();
    source_paths.dedup();
    let digest_cache = SidecarDigestCache::open_default();
    let source_files = direct_gguf_source_files_from_paths(source_paths, digest_cache.as_ref())?;
    let source_model_bytes = source_files.iter().map(|file| file.bytes).sum();
    let source_model_sha256 = legacy_identity::aggregate_source_sha256(&source_files);
    let package_ref = format!("safetensors://{}", checkpoint_root.display());
    let manifest_sha256 = synthetic_manifest_sha256(SyntheticManifestInput {
        model_id,
        package_kind: "direct-safetensors",
        package_ref: &package_ref,
        source_model_path: &checkpoint_root.to_string_lossy(),
        source_model_sha256: &source_model_sha256,
        source_model_bytes,
        source_files: &source_files,
        architecture,
        context_length,
        layer_count,
        activation_width,
        tensor_count,
    })?;
    Ok(SkippyPackageIdentity {
        package_ref,
        manifest_sha256,
        source_model_path: checkpoint_root,
        source_model_sha256,
        source_model_bytes,
        source_files,
        layer_weight_bytes: Vec::new(),
        layer_count,
        activation_width,
        tensor_count,
        generation: None,
    })
}

fn synthetic_gguf_package(_model_id: &str, model_path: &Path) -> Result<SkippyPackageIdentity> {
    // Direct local GGUF identity crosses the mesh and is therefore always
    // content addressed. Absolute paths and filenames are node-local locators;
    // they must never participate in model, package, or admission identity.
    content_addressed::validate_source_set(model_path)?;
    let source_paths = direct_gguf_source_paths(model_path)?;
    for path in &source_paths {
        anyhow::ensure!(
            path.to_str().is_some(),
            "canonical content-addressed GGUF path must be valid UTF-8: {}",
            path.display()
        );
    }
    let verified_fingerprint = super::local_source::verified_path_fingerprint(&source_paths);
    let source_files = direct_gguf_source_files_from_paths(source_paths, None)?;
    content_addressed::ensure_fingerprint_unchanged(
        &source_files,
        verified_fingerprint.as_deref(),
    )?;

    let source_model_path = source_files
        .first()
        .map(|file| file.path.clone())
        .context("direct GGUF source file list is empty")?;

    let compact = crate::models::gguf::scan_gguf_compact_meta(&source_model_path)
        .with_context(|| format!("read GGUF metadata {}", source_model_path.display()))?;

    let tensor_count = gguf_tensor_count(&source_model_path)
        .with_context(|| format!("read GGUF tensor count {}", source_model_path.display()))?;

    anyhow::ensure!(
        compact.layer_count > 0,
        "GGUF metadata for {} does not contain a positive layer count",
        source_model_path.display()
    );
    anyhow::ensure!(
        compact.embedding_size > 0,
        "GGUF metadata for {} does not contain a positive embedding size",
        source_model_path.display()
    );
    let source_model_bytes = source_files.iter().map(|file| file.bytes).sum();
    let layer_weight_bytes = direct_gguf_layer_weight_bytes(&source_files, compact.layer_count)
        .with_context(|| {
            format!(
                "inspect GGUF tensor weights {}",
                source_model_path.display()
            )
        })?;
    content_addressed::ensure_fingerprint_unchanged(
        &source_files,
        verified_fingerprint.as_deref(),
    )?;

    let source_model_sha256 = content_addressed::aggregate_source_sha256(&source_files);
    let package_ref = super::local_source::content_addressed_package_ref(&source_model_sha256)?;
    let manifest_sha256 = content_addressed::manifest_sha256(
        &source_model_sha256,
        source_model_bytes,
        &source_files,
        &compact.architecture,
        compact.context_length,
        compact.layer_count,
        compact.embedding_size,
        tensor_count,
    )?;

    let identity = SkippyPackageIdentity {
        package_ref,
        manifest_sha256,
        source_model_path,
        source_model_sha256,
        source_model_bytes,
        source_files,
        layer_weight_bytes,
        layer_count: compact.layer_count,
        activation_width: compact.embedding_size,
        tensor_count,
        generation: None,
    };
    super::local_source::register_content_addressed_identity(&identity, verified_fingerprint);
    Ok(identity)
}

struct SyntheticManifestInput<'a> {
    model_id: &'a str,
    package_kind: &'a str,
    package_ref: &'a str,
    source_model_path: &'a str,
    source_model_sha256: &'a str,
    source_model_bytes: u64,
    source_files: &'a [SkippyPackageSourceFile],
    architecture: &'a str,
    context_length: u32,
    layer_count: u32,
    activation_width: u32,
    tensor_count: u64,
}

fn synthetic_manifest_sha256(input: SyntheticManifestInput<'_>) -> Result<String> {
    let files = input
        .source_files
        .iter()
        .map(|file| SyntheticGgufManifestFile {
            path: file.path.to_string_lossy().to_string(),
            bytes: file.bytes,
            sha256: file.sha256.clone(),
        })
        .collect::<Vec<_>>();
    let manifest = SyntheticGgufManifest {
        schema_version: 1,
        package_kind: input.package_kind,
        model_id: input.model_id,
        package_ref: input.package_ref,
        source_model_path: input.source_model_path,
        source_model_sha256: input.source_model_sha256,
        source_model_bytes: input.source_model_bytes,
        source_files: &files,
        architecture: input.architecture,
        context_length: input.context_length,
        layer_count: input.layer_count,
        activation_width: input.activation_width,
        tensor_count: input.tensor_count,
    };
    let bytes = serde_json::to_vec(&manifest).context("serialize synthetic GGUF manifest")?;
    Ok(hex_lower(&Sha256::digest(bytes)))
}

pub(crate) fn direct_gguf_source_paths(model_path: &Path) -> Result<Vec<PathBuf>> {
    let canonical = model_path
        .canonicalize()
        .with_context(|| format!("canonicalize GGUF path {}", model_path.display()))?;
    let Some(file_name) = canonical.file_name().and_then(|name| name.to_str()) else {
        anyhow::bail!("GGUF path has no UTF-8 filename: {}", canonical.display());
    };
    let Some(shard) = model_ref::split_gguf_shard_info(file_name) else {
        return Ok(vec![canonical]);
    };
    anyhow::ensure!(
        shard.part == "00001",
        "split GGUF inputs must point at the first shard, got {}",
        canonical.display()
    );
    let total = shard
        .total
        .parse::<u32>()
        .with_context(|| format!("parse split GGUF shard total in {file_name}"))?;
    anyhow::ensure!(
        total > 0,
        "split GGUF shard total must be greater than zero"
    );
    let parent = canonical
        .parent()
        .with_context(|| format!("split GGUF shard has no parent: {}", canonical.display()))?;
    let mut files = Vec::with_capacity(total as usize);
    for index in 1..=total {
        let shard_name = format!("{}-{index:05}-of-{:05}.gguf", shard.prefix, total);
        let path = parent.join(shard_name);
        files.push(path.canonicalize().with_context(|| {
            format!(
                "read split GGUF shard {index}/{total} for {}",
                canonical.display()
            )
        })?);
    }
    Ok(files)
}

/// Build the source-complete metadata envelope required by generation-8
/// planning directly from local GGUF shards. The envelope is in memory: the
/// source files remain at their original paths and no package or layer shard
/// is written.
pub(crate) fn direct_gguf_planning_manifest_from_identity(
    model_id: &str,
    identity: &SkippyPackageIdentity,
) -> Result<(PackageManifestV2, Vec<PathBuf>)> {
    let shard_paths = identity
        .source_files
        .iter()
        .map(|source| source.path.clone())
        .collect::<Vec<_>>();
    let mut artifacts = Vec::with_capacity(shard_paths.len());
    let mut source_files = Vec::with_capacity(shard_paths.len());
    let mut tensors = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    let mut primary_metadata = None;
    let mut shard_metadata = Vec::with_capacity(shard_paths.len());
    let mut total_tensors = 0_usize;

    for (index, (path, source)) in shard_paths.iter().zip(&identity.source_files).enumerate() {
        let artifact_id = format!("source-{index:05}");
        let logical_path = format!("source-{index:05}.gguf");
        let directory = skippy_model::gguf_catalog::read_gguf_catalog(path)
            .with_context(|| format!("read direct GGUF catalog {}", path.display()))?;
        anyhow::ensure!(
            directory.artifact_bytes == source.bytes,
            "direct GGUF size changed while planning: {}",
            path.display()
        );
        if index == 0 {
            primary_metadata = Some(directory.metadata.clone());
        }
        let native = skippy_runtime::ModelInfo::open(path)
            .with_context(|| format!("open direct GGUF metadata {}", path.display()))?
            .tensors()
            .with_context(|| format!("read direct GGUF tensors {}", path.display()))?;
        let shard_tensors = direct_planning_tensor_catalog(&directory, &native, &artifact_id)?;
        total_tensors = total_tensors
            .checked_add(shard_tensors.entries.len())
            .context("direct GGUF tensor count overflow")?;
        shard_metadata.push(directory.metadata.clone());
        for tensor in shard_tensors.entries {
            anyhow::ensure!(
                names.insert(tensor.name.clone()),
                "duplicate direct GGUF tensor {:?}",
                tensor.name
            );
            tensors.push(tensor);
        }
        source_files.push(SourceFile {
            path: logical_path.clone(),
            byte_size: source.bytes,
            sha256: source.sha256.clone(),
        });
        artifacts.push(Artifact {
            id: artifact_id,
            path: logical_path,
            byte_size: source.bytes,
            sha256: source.sha256.clone(),
        });
    }
    anyhow::ensure!(!tensors.is_empty(), "direct GGUF tensor inventory is empty");
    validate_direct_gguf_shards(&shard_metadata, total_tensors)?;

    let primary = source_files
        .first()
        .context("direct GGUF source file list is empty")?;
    let mut manifest = PackageManifestV2 {
        schema_version: PACKAGE_SCHEMA_VERSION,
        package_id: String::new(),
        model_id: model_id.to_string(),
        source_model: SourceModel {
            sha256: primary.sha256.clone(),
            metadata_artifact_id: "source-00000".to_string(),
            repo: None,
            revision: None,
            primary_file: Some(primary.path.clone()),
            canonical_ref: None,
            distribution_id: Some(content_addressed::aggregate_source_sha256(
                &identity.source_files,
            )),
            files: source_files,
        },
        format: "gguf".to_string(),
        layer_count: identity.layer_count,
        model_metadata: primary_metadata.context("direct GGUF metadata is empty")?,
        artifact_catalog: ArtifactCatalog { entries: artifacts },
        tensor_catalog: TensorCatalog { entries: tensors },
        sidecars: Vec::new(),
        generation: None,
        native_abi_version: format!(
            "{}.{}.{}",
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR,
            skippy_ffi::ABI_VERSION_PATCH
        ),
        generator_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at_unix_secs: 0,
    };
    manifest.package_id = manifest
        .computed_package_id()
        .context("compute direct GGUF planning identity")?;
    manifest
        .validate()
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("validate direct GGUF planning manifest")?;
    Ok((manifest, shard_paths))
}

fn validate_direct_gguf_shards(
    shard_metadata: &[std::collections::BTreeMap<String, serde_json::Value>],
    total_tensors: usize,
) -> Result<()> {
    let first = shard_metadata
        .first()
        .context("direct GGUF shard inventory is empty")?;
    for (index, metadata) in shard_metadata.iter().enumerate() {
        if shard_metadata.len() > 1 || metadata.contains_key("split.count") {
            anyhow::ensure!(
                metadata
                    .get("split.count")
                    .and_then(serde_json::Value::as_u64)
                    == Some(shard_metadata.len() as u64),
                "incomplete direct GGUF shard set: split.count mismatch"
            );
            anyhow::ensure!(
                metadata.get("split.no").and_then(serde_json::Value::as_u64) == Some(index as u64),
                "direct GGUF split.no mismatch"
            );
        }
        for (key, value) in metadata {
            if key != "split.no" && key != "split.count" && key != "split.tensors.count" {
                anyhow::ensure!(
                    first.get(key) == Some(value),
                    "inconsistent direct GGUF metadata {key:?} across shards"
                );
            }
        }
    }
    if shard_metadata.len() > 1 || first.contains_key("split.count") {
        anyhow::ensure!(
            first
                .get("split.tensors.count")
                .and_then(serde_json::Value::as_u64)
                == Some(total_tensors as u64),
            "incomplete direct GGUF tensor inventory: split.tensors.count mismatch"
        );
    }
    Ok(())
}

fn direct_planning_tensor_catalog(
    directory: &skippy_model::gguf_catalog::GgufCatalog,
    native: &[skippy_runtime::TensorInfo],
    artifact_id: &str,
) -> Result<TensorCatalog> {
    let by_name = native
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<std::collections::BTreeMap<_, _>>();
    anyhow::ensure!(
        by_name.len() == native.len() && native.len() == directory.tensors.len(),
        "native and direct GGUF tensor inventories disagree"
    );
    let mut source = directory.tensors.iter().collect::<Vec<_>>();
    source.sort_by(|left, right| {
        (left.data_offset, &left.name).cmp(&(right.data_offset, &right.name))
    });
    let mut entries: Vec<Tensor> = Vec::with_capacity(source.len());
    let mut last_owned: Option<(u64, u64, usize)> = None;
    for tensor in source {
        let info = by_name
            .get(tensor.name.as_str())
            .context("direct GGUF tensor is missing from native inventory")?;
        let elements = tensor
            .dimensions
            .iter()
            .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
            .context("direct GGUF tensor element count overflow")?;
        anyhow::ensure!(
            info.ggml_type == tensor.ggml_type && info.element_count == elements,
            "native and direct GGUF metadata disagree for {:?}",
            tensor.name
        );
        anyhow::ensure!(
            info.byte_size > 0,
            "empty direct GGUF tensor storage for {:?}",
            tensor.name
        );
        let end = tensor
            .data_offset
            .checked_add(info.byte_size)
            .context("direct GGUF tensor extent overflow")?;
        anyhow::ensure!(
            end <= directory.artifact_bytes,
            "direct GGUF tensor {:?} storage exceeds artifact bounds",
            tensor.name
        );
        let storage = if let Some((start, previous_end, owner)) =
            last_owned.filter(|(_, previous_end, _)| tensor.data_offset < *previous_end)
        {
            let target = &entries[owner];
            anyhow::ensure!(
                tensor.data_offset == start
                    && end == previous_end
                    && tensor.ggml_type == target.ggml_type
                    && tensor.dimensions == target.dimensions,
                "overlapping or mismatched direct GGUF alias storage for {:?}",
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
            layer_ordinal: None,
            storage,
        });
    }
    Ok(TensorCatalog { entries })
}

fn direct_gguf_source_files(
    model_path: &Path,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<Vec<SkippyPackageSourceFile>> {
    direct_gguf_source_files_from_paths(direct_gguf_source_paths(model_path)?, digest_cache)
}

fn direct_gguf_source_files_from_paths(
    source_paths: Vec<PathBuf>,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<Vec<SkippyPackageSourceFile>> {
    source_paths
        .into_iter()
        .map(|path| source_file(&path, digest_cache))
        .collect()
}

fn source_file(
    path: &Path,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<SkippyPackageSourceFile> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("canonicalize GGUF source {}", path.display()))?;
    let metadata = canonical
        .metadata()
        .with_context(|| format!("stat GGUF source {}", canonical.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "GGUF source is not a file: {}",
        canonical.display()
    );
    let sha256 = source_file_sha256(&canonical, &metadata, digest_cache)?;
    Ok(SkippyPackageSourceFile {
        path: canonical.clone(),
        bytes: metadata.len(),
        sha256,
    })
}

/// SHA-256 of a source file, served from the sidecar cache when the file's
/// `(size, mtime, ctime)` is unchanged and recomputed (then cached) otherwise.
fn source_file_sha256(
    path: &Path,
    metadata: &std::fs::Metadata,
    digest_cache: Option<&SidecarDigestCache>,
) -> Result<String> {
    let mtime_nanos = hash_cache::file_mtime_nanos(metadata);
    let ctime_nanos = hash_cache::file_ctime_nanos(metadata);
    if let (Some(cache), Some(mtime_nanos)) = (digest_cache, mtime_nanos)
        && let Some(sha256) = cache.lookup(path, metadata.len(), mtime_nanos, ctime_nanos)
    {
        tracing::debug!(path = %path.display(), "GGUF source sha256 cache hit");
        return Ok(sha256);
    }
    let started = Instant::now();
    let sha256 = file_sha256(path)?;
    tracing::debug!(
        path = %path.display(),
        bytes = metadata.len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "computed GGUF source sha256"
    );
    if let (Some(cache), Some(mtime_nanos)) = (digest_cache, mtime_nanos) {
        cache.store(path, metadata.len(), mtime_nanos, ctime_nanos, &sha256);
    }
    Ok(sha256)
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("open GGUF source {}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash GGUF source {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn gguf_tensor_count(path: &Path) -> Result<u64> {
    let mut reader =
        BufReader::new(File::open(path).with_context(|| format!("open GGUF {}", path.display()))?);
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .with_context(|| format!("read GGUF magic {}", path.display()))?;
    anyhow::ensure!(&magic == b"GGUF", "not a GGUF file: {}", path.display());
    let version = read_u32_le(&mut reader)?;
    anyhow::ensure!(
        version >= 2,
        "unsupported GGUF version {version} in {}",
        path.display()
    );
    read_gguf_count(&mut reader, version)
}

fn direct_gguf_layer_weight_bytes(
    source_files: &[SkippyPackageSourceFile],
    layer_count: u32,
) -> Result<Vec<u64>> {
    if !skippy_runtime::native_runtime_loaded() {
        tracing::debug!(
            "GGUF tensor layout unavailable because the native runtime is not loaded; \
             using capacity-based split planning"
        );
        return Ok(Vec::new());
    }

    let mut tensors = Vec::new();
    for source_file in source_files {
        let info = match skippy_runtime::ModelInfo::open(&source_file.path) {
            Ok(info) => info,
            Err(error) => {
                tracing::debug!(
                    path = %source_file.path.display(),
                    error = %error,
                    "GGUF tensor layout unavailable; using capacity-based split planning"
                );
                return Ok(Vec::new());
            }
        };
        tensors.extend(
            info.tensors()
                .with_context(|| format!("read GGUF tensors {}", source_file.path.display()))?,
        );
    }
    Ok(layer_weight_bytes_from_tensors(&tensors, layer_count))
}

fn layer_weight_bytes_from_tensors(
    tensors: &[skippy_runtime::TensorInfo],
    layer_count: u32,
) -> Vec<u64> {
    let Ok(layer_count) = usize::try_from(layer_count) else {
        return Vec::new();
    };
    if layer_count == 0 {
        return Vec::new();
    }

    let mut weights = vec![0_u64; layer_count];
    let mut shared_bytes = 0_u64;
    let mut seen = std::collections::BTreeSet::new();

    for tensor in tensors {
        if !seen.insert(tensor.name.as_str()) {
            continue;
        }
        let bytes = tensor.byte_size;
        match tensor.layer_index {
            Some(layer) if (layer as usize) < layer_count => {
                weights[layer as usize] = weights[layer as usize].saturating_add(bytes);
            }
            // Native MTP blocks are appended after the trunk's declared layer
            // count and must stay with the final stage that owns logits.
            Some(_) => {
                let last = weights.len() - 1;
                weights[last] = weights[last].saturating_add(bytes);
            }
            None => match tensor.role {
                TensorRole::Embedding => {
                    weights[0] = weights[0].saturating_add(bytes);
                }
                TensorRole::FinalNorm | TensorRole::Output => {
                    let last = weights.len() - 1;
                    weights[last] = weights[last].saturating_add(bytes);
                }
                TensorRole::Unknown
                | TensorRole::Metadata
                | TensorRole::Tokenizer
                | TensorRole::Layer => {
                    shared_bytes = shared_bytes.saturating_add(bytes);
                }
            },
        }
    }

    // Metadata is loaded at every stage but is normally tiny. Split it between
    // endpoints so total model weight stays conserved without biasing a middle
    // stage in multi-node plans.
    weights[0] = weights[0].saturating_add(shared_bytes.div_ceil(2));
    let last = weights.len() - 1;
    weights[last] = weights[last].saturating_add(shared_bytes / 2);
    weights
}

fn read_u32_le(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).context("read u32")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i64_le(reader: &mut impl Read) -> Result<i64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).context("read i64")?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_gguf_count(reader: &mut impl Read, _version: u32) -> Result<u64> {
    let value = read_i64_le(reader)?;
    u64::try_from(value).context("GGUF count is negative")
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Build a `SkippyPackageIdentity` from a remote HF layer package.
///
/// Resolves the package into the local HF cache for inspection, downloading
/// the manifest and shared metadata that the resolver requires, but not layer
/// files. Layer artifacts are fetched later by the node that materializes or
/// loads its assigned stage.
pub fn identity_from_layer_package(package_ref: &str) -> Result<SkippyPackageIdentity> {
    // Resolve hf:// to a local package dir for lightweight package inspection.
    let local_ref =
        super::materialization::resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    let info = skippy_runtime::package::inspect_layer_package(&local_ref)
        .with_context(|| format!("inspect layer package {package_ref}"))?;

    let source_model_bytes = info
        .source_model_bytes
        .unwrap_or_else(|| info.layers.iter().map(|l| l.artifact_bytes).sum::<u64>());
    let layer_weight_bytes = layer_weight_bytes_from_info(&info);

    // For local paths inside an HF cache, convert to an exact hf:// ref so all
    // nodes resolve the same snapshot independently. HF cache dirs look like:
    // .../models--owner--name/snapshots/<hash>/
    let canonical_package_ref = canonical_layer_package_ref(package_ref, &local_ref);

    Ok(SkippyPackageIdentity {
        package_ref: canonical_package_ref,
        manifest_sha256: info.manifest_sha256,
        source_model_path: PathBuf::from(&info.source_model_path),
        source_model_sha256: info.source_model_sha256,
        source_model_bytes,
        source_files: Vec::new(),
        layer_weight_bytes,
        layer_count: info.layer_count,
        activation_width: 0,
        tensor_count: info.layers.iter().map(|l| l.tensor_count as u64).sum(),
        generation: info.generation,
    })
}

fn layer_weight_bytes_from_info(info: &skippy_runtime::package::LayerPackageInfo) -> Vec<u64> {
    let mut layers = info.layers.clone();
    layers.sort_by_key(|layer| layer.layer_index);
    if layers.len() != info.layer_count as usize
        || layers
            .iter()
            .enumerate()
            .any(|(index, layer)| layer.layer_index as usize != index)
    {
        return Vec::new();
    }
    let mut weights = layers
        .into_iter()
        .map(|layer| layer.tensor_bytes.max(layer.artifact_bytes))
        .collect::<Vec<_>>();
    let accounted = weights.iter().copied().sum::<u64>();
    let unaccounted = info
        .source_model_bytes
        .unwrap_or_default()
        .saturating_sub(accounted);
    if let Some((first, rest)) = weights.split_first_mut() {
        *first = first.saturating_add(unaccounted.div_ceil(2));
        if let Some(last) = rest.last_mut() {
            *last = last.saturating_add(unaccounted / 2);
        } else {
            *first = first.saturating_add(unaccounted / 2);
        }
    }
    weights
}

/// Detect if a local path is inside an HF cache directory and convert to `hf://` ref.
///
/// HF cache paths look like:
///   `.../hub/models--owner--name/snapshots/<hash>/`
///
/// Returns `Some("hf://owner/name@hash")` if detected, `None` otherwise.
fn hf_ref_from_cache_path(path: &str) -> Option<String> {
    // Walk path components looking for "models--*" followed by "snapshots"
    let path = std::path::Path::new(path);
    let components: Vec<&std::ffi::OsStr> = path
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(s) => Some(s),
            _ => None,
        })
        .collect();
    for (i, comp) in components.iter().enumerate() {
        let s = comp.to_str()?;
        if let Some(repo_part) = s.strip_prefix("models--") {
            // Verify next component is "snapshots" and preserve the exact
            // snapshot revision/hash so peers fetch identical package content.
            if components.get(i + 1).and_then(|c| c.to_str()) == Some("snapshots") {
                let revision = components.get(i + 2)?.to_str()?;
                // repo_part is "owner--name", convert to "owner/name"
                let repo = repo_part.replacen("--", "/", 1);
                if repo.contains('/') {
                    return Some(format!("hf://{repo}@{revision}"));
                }
            }
        }
    }
    None
}

fn canonical_layer_package_ref(package_ref: &str, local_ref: &str) -> String {
    hf_ref_from_cache_path(local_ref)
        .or_else(|| hf_ref_from_cache_path(package_ref))
        .unwrap_or_else(|| package_ref.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_package_format::{
        ArtifactCatalog, SourceModel, Tensor, TensorCatalog, TensorIntegrity,
    };
    use skippy_runtime::TensorInfo;

    fn split_metadata(
        split_no: u64,
        split_count: u64,
        tensor_count: u64,
    ) -> std::collections::BTreeMap<String, serde_json::Value> {
        [
            ("split.no".to_string(), split_no.into()),
            ("split.count".to_string(), split_count.into()),
            ("split.tensors.count".to_string(), tensor_count.into()),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn direct_gguf_inventory_uses_primary_global_tensor_count() {
        let metadata = vec![split_metadata(0, 2, 3), split_metadata(1, 2, 2)];

        validate_direct_gguf_shards(&metadata, 3).unwrap();

        let stale_primary = vec![split_metadata(0, 2, 2), split_metadata(1, 2, 3)];
        let error = validate_direct_gguf_shards(&stale_primary, 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("split.tensors.count mismatch"), "{error}");
    }

    #[test]
    fn package_v2_identity_rejects_v1_without_fallback() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join(PACKAGE_V2_MANIFEST),
            br#"{"schema_version":1}"#,
        )
        .unwrap();

        let error = identity_from_package_v2(root.path())
            .unwrap_err()
            .to_string();

        assert!(error.contains("requires package schema 2"), "{error}");
    }

    #[test]
    fn package_v2_ref_requires_the_v2_schema_marker() {
        let root = tempfile::tempdir().unwrap();
        let manifest = root.path().join(PACKAGE_V2_MANIFEST);

        std::fs::write(&manifest, br#"{"schema_version":1}"#).unwrap();
        assert!(!is_package_v2_ref(&root.path().to_string_lossy()));

        std::fs::write(&manifest, br#"{"schema_version":2}"#).unwrap();
        assert!(is_package_v2_ref(&root.path().to_string_lossy()));
    }

    #[test]
    fn package_v2_layer_weights_fall_back_when_ordinals_are_unavailable() {
        let tensor = |id: &str, layer_ordinal, stored_length| Tensor {
            id: id.to_string(),
            name: id.to_string(),
            ggml_type: 0,
            dimensions: vec![1],
            layer_ordinal,
            storage: TensorStorage::Owned {
                artifact_id: "source".to_string(),
                data_offset: 0,
                stored_length,
                alignment: 1,
                integrity: TensorIntegrity::ArtifactSha256,
            },
        };
        let mut manifest = PackageManifestV2 {
            schema_version: skippy_package_format::PACKAGE_SCHEMA_VERSION,
            package_id: String::new(),
            model_id: "fixture/model".to_string(),
            source_model: SourceModel {
                sha256: String::new(),
                metadata_artifact_id: "source".to_string(),
                repo: None,
                revision: None,
                primary_file: None,
                canonical_ref: None,
                distribution_id: None,
                files: Vec::new(),
            },
            format: "gguf".to_string(),
            layer_count: 2,
            model_metadata: Default::default(),
            artifact_catalog: ArtifactCatalog {
                entries: Vec::new(),
            },
            tensor_catalog: TensorCatalog {
                entries: vec![tensor("first", None, 10), tensor("second", None, 20)],
            },
            sidecars: Vec::new(),
            generation: None,
            native_abi_version: String::new(),
            generator_version: String::new(),
            created_at_unix_secs: 0,
        };

        assert!(package_v2_layer_weight_bytes(&manifest).unwrap().is_empty());

        manifest.tensor_catalog.entries[0].layer_ordinal = Some(0);
        assert!(package_v2_layer_weight_bytes(&manifest).unwrap().is_empty());

        manifest.tensor_catalog.entries[1].layer_ordinal = Some(1);
        assert_eq!(
            package_v2_layer_weight_bytes(&manifest).unwrap(),
            vec![10, 20]
        );
    }

    #[test]
    #[ignore = "requires SKIPPY_PACKAGE_V2_TEST_DIR"]
    fn package_v2_identity_reads_a_real_package() {
        let package_dir = std::env::var_os("SKIPPY_PACKAGE_V2_TEST_DIR")
            .map(PathBuf::from)
            .expect("SKIPPY_PACKAGE_V2_TEST_DIR is required");

        let identity = identity_from_package_v2(&package_dir).unwrap();

        assert_eq!(identity.layer_count, 32);
        assert!(identity.activation_width > 0);
        assert_eq!(identity.layer_weight_bytes.len(), 32);
        assert!(identity.source_model_path.is_file());
        assert_eq!(identity.manifest_sha256.len(), 64);

        let expected_manifest_sha256 = identity.manifest_sha256.clone();
        let expected_source_sha256 = identity.source_model_sha256.clone();
        let strict = super::super::local_source::into_content_addressed_identity(identity)
            .expect("index package-v2 source for strict-local loading");
        let verified = super::super::local_source::verify_registered_content_source(
            "granite-v2-test",
            &strict.package_ref,
            &expected_manifest_sha256,
            &expected_source_sha256,
        )
        .expect("resolve indexed package-v2 source");
        assert_eq!(verified, strict);
    }

    #[test]
    fn synthetic_direct_identity_accepts_safetensors_checkpoint_directory() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            r#"{
              "model_type": "qwen2",
              "num_hidden_layers": 1,
              "hidden_size": 4,
              "max_position_embeddings": 128
            }"#,
        )
        .unwrap();
        let header = serde_json::json!({
            "model.layers.0.input_layernorm.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4]
            }
        })
        .to_string();
        let mut safetensors = Vec::new();
        safetensors.extend_from_slice(&(header.len() as u64).to_le_bytes());
        safetensors.extend_from_slice(header.as_bytes());
        safetensors.extend_from_slice(&1.0_f32.to_le_bytes());
        std::fs::write(root.path().join("model.safetensors"), safetensors).unwrap();

        let identity = synthetic_direct_gguf_package("test", root.path()).unwrap();

        assert!(identity.package_ref.starts_with("safetensors://"));
        assert_eq!(
            identity.source_model_path,
            root.path().canonicalize().unwrap()
        );
        assert_eq!(identity.layer_count, 1);
        assert_eq!(identity.activation_width, 4);
        assert_eq!(identity.tensor_count, 1);
        assert_eq!(identity.source_files.len(), 2);
    }

    #[test]
    fn synthetic_manifest_identity_is_stable_and_metadata_sensitive() {
        let source_files = vec![SkippyPackageSourceFile {
            path: PathBuf::from("/models/model.gguf"),
            bytes: 12,
            sha256: "abc123".to_string(),
        }];
        let first = synthetic_manifest_sha256(SyntheticManifestInput {
            model_id: "model-a",
            package_kind: "direct-gguf",
            package_ref: "gguf:///models/model.gguf",
            source_model_path: "/models/model.gguf",
            source_model_sha256: "abc123",
            source_model_bytes: 12,
            source_files: &source_files,
            architecture: "llama",
            context_length: 4096,
            layer_count: 32,
            activation_width: 4096,
            tensor_count: 100,
        })
        .unwrap();
        let second = synthetic_manifest_sha256(SyntheticManifestInput {
            model_id: "model-a",
            package_kind: "direct-gguf",
            package_ref: "gguf:///models/model.gguf",
            source_model_path: "/models/model.gguf",
            source_model_sha256: "abc123",
            source_model_bytes: 12,
            source_files: &source_files,
            architecture: "llama",
            context_length: 4096,
            layer_count: 32,
            activation_width: 4096,
            tensor_count: 100,
        })
        .unwrap();
        let changed = synthetic_manifest_sha256(SyntheticManifestInput {
            model_id: "model-a",
            package_kind: "direct-gguf",
            package_ref: "gguf:///models/model.gguf",
            source_model_path: "/models/model.gguf",
            source_model_sha256: "abc123",
            source_model_bytes: 12,
            source_files: &source_files,
            architecture: "llama",
            context_length: 4096,
            layer_count: 33,
            activation_width: 4096,
            tensor_count: 100,
        })
        .unwrap();

        assert_eq!(first, second);
        assert_ne!(first, changed);
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn direct_gguf_source_files_expand_split_shards() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("Model-Q4_K_M-00001-of-00003.gguf");
        std::fs::write(&first, b"one").unwrap();
        std::fs::write(dir.path().join("Model-Q4_K_M-00002-of-00003.gguf"), b"two").unwrap();
        std::fs::write(
            dir.path().join("Model-Q4_K_M-00003-of-00003.gguf"),
            b"three",
        )
        .unwrap();

        let files = direct_gguf_source_files(&first, None).unwrap();

        assert_eq!(files.len(), 3);
        assert_eq!(
            files.iter().map(|file| file.bytes).collect::<Vec<_>>(),
            vec![3, 3, 5]
        );
        assert!(files[0].path.ends_with("Model-Q4_K_M-00001-of-00003.gguf"));
        assert!(files[2].path.ends_with("Model-Q4_K_M-00003-of-00003.gguf"));
    }

    #[test]
    fn direct_gguf_source_files_report_missing_split_shard() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("Model-Q4_K_M-00001-of-00002.gguf");
        std::fs::write(&first, b"one").unwrap();

        let error = direct_gguf_source_files(&first, None)
            .unwrap_err()
            .to_string();

        assert!(error.contains("split GGUF shard 2/2"));
    }

    #[test]
    fn direct_gguf_source_files_reject_non_primary_split_shard() {
        let dir = tempfile::tempdir().unwrap();
        let second = dir.path().join("Model-Q4_K_M-00002-of-00002.gguf");
        std::fs::write(&second, b"two").unwrap();

        let error = direct_gguf_source_files(&second, None)
            .unwrap_err()
            .to_string();

        assert!(error.contains("first shard"));
    }

    #[test]
    fn source_file_sha256_is_stable_and_content_sensitive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"content-a").unwrap();

        let first = source_file(&path, None).unwrap();
        let second = source_file(&path, None).unwrap();
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.sha256.len(), 64);
        assert!(first.sha256.chars().all(|c| c.is_ascii_hexdigit()));

        std::fs::write(&path, b"content-b-longer").unwrap();
        let changed = source_file(&path, None).unwrap();
        assert_ne!(first.sha256, changed.sha256);
    }

    #[test]
    fn source_file_reuses_cached_sha256_while_metadata_matches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"content").unwrap();
        let cache = SidecarDigestCache::open_in(dir.path().join("hashes"));

        let computed = source_file(&path, Some(&cache)).unwrap();

        // A matching (size, mtime, ctime) record now exists; prove the second
        // call serves it by making the cached value observably distinct while
        // still shaped like a real SHA-256.
        let distinct_sha256 = "f".repeat(64);
        let canonical = path.canonicalize().unwrap();
        let metadata = canonical.metadata().unwrap();
        let mtime_nanos = hash_cache::file_mtime_nanos(&metadata).unwrap();
        let ctime_nanos = hash_cache::file_ctime_nanos(&metadata);
        cache.store(
            &canonical,
            metadata.len(),
            mtime_nanos,
            ctime_nanos,
            &distinct_sha256,
        );

        let cached = source_file(&path, Some(&cache)).unwrap();
        assert_eq!(cached.sha256, distinct_sha256);
        assert_ne!(computed.sha256, cached.sha256);
    }

    /// Regression test for the review concern on the sidecar cache: a GGUF
    /// replaced with different same-size content while tooling restores its
    /// mtime must not reuse the stale hash. The inode ctime moves on any
    /// rewrite and cannot be restored from userspace, so the cache misses.
    #[cfg(unix)]
    #[test]
    fn source_file_recomputes_when_same_size_content_replaced_with_restored_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"content-a").unwrap();
        let cache = SidecarDigestCache::open_in(dir.path().join("hashes"));

        let first = source_file(&path, Some(&cache)).unwrap();
        let original_mtime = path.metadata().unwrap().modified().unwrap();

        // Inode timestamps use a coarse clock; wait long enough that the
        // rewrite lands on a later ctime tick than the original write.
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Replace with different content of identical size, then restore the
        // original mtime the way rsync/tar/cp --preserve style tooling does.
        std::fs::write(&path, b"content-b").unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(original_mtime)
            .unwrap();
        let metadata = path.metadata().unwrap();
        assert_eq!(metadata.len(), first.bytes);
        assert_eq!(metadata.modified().unwrap(), original_mtime);

        let replaced = source_file(&path, Some(&cache)).unwrap();
        assert_ne!(first.sha256, replaced.sha256);
        assert_eq!(replaced.sha256, file_sha256(&path).unwrap());
    }

    #[test]
    fn source_file_recomputes_when_size_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"content-a").unwrap();
        let cache = SidecarDigestCache::open_in(dir.path().join("hashes"));

        let first = source_file(&path, Some(&cache)).unwrap();

        std::fs::write(&path, b"content-b-longer").unwrap();
        let changed = source_file(&path, Some(&cache)).unwrap();
        assert_ne!(first.sha256, changed.sha256);
        assert_eq!(changed.sha256, file_sha256(&path).unwrap());
    }

    #[test]
    fn hf_ref_from_cache_path_preserves_snapshot_revision() {
        let package_ref =
            "/cache/hub/models--meshllm--Qwen3-layers/snapshots/abc123/model-package.json";

        assert_eq!(
            hf_ref_from_cache_path(package_ref),
            Some("hf://meshllm/Qwen3-layers@abc123".to_string())
        );
    }

    #[test]
    fn canonical_layer_package_ref_prefers_resolved_snapshot() {
        let local_ref = "/cache/hub/models--meshllm--Qwen3-layers/snapshots/abc123";

        assert_eq!(
            canonical_layer_package_ref("hf://meshllm/Qwen3-layers@main", local_ref),
            "hf://meshllm/Qwen3-layers@abc123"
        );
    }

    #[test]
    fn package_layer_weights_include_shared_model_bytes_at_endpoints() {
        let info = skippy_runtime::package::LayerPackageInfo {
            package_dir: PathBuf::from("/models/package"),
            manifest_sha256: "manifest".to_string(),
            model_id: "org/model".to_string(),
            source_model_path: "model.gguf".to_string(),
            source_model_sha256: "source".to_string(),
            source_model_bytes: Some(120),
            layer_count: 2,
            generation: None,
            projectors: Vec::new(),
            layers: vec![
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 0,
                    tensor_count: 1,
                    tensor_bytes: 30,
                    artifact_bytes: 30,
                },
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 1,
                    tensor_count: 1,
                    tensor_bytes: 40,
                    artifact_bytes: 40,
                },
            ],
        };

        assert_eq!(layer_weight_bytes_from_info(&info), vec![55, 65]);
    }

    #[test]
    fn package_layer_weights_require_contiguous_indices() {
        let mut info = skippy_runtime::package::LayerPackageInfo {
            package_dir: PathBuf::from("/models/package"),
            manifest_sha256: "manifest".to_string(),
            model_id: "org/model".to_string(),
            source_model_path: "model.gguf".to_string(),
            source_model_sha256: "source".to_string(),
            source_model_bytes: Some(70),
            layer_count: 2,
            generation: None,
            projectors: Vec::new(),
            layers: vec![
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 0,
                    tensor_count: 1,
                    tensor_bytes: 30,
                    artifact_bytes: 30,
                },
                skippy_runtime::package::LayerPackageLayerInfo {
                    layer_index: 2,
                    tensor_count: 1,
                    tensor_bytes: 40,
                    artifact_bytes: 40,
                },
            ],
        };

        assert!(layer_weight_bytes_from_info(&info).is_empty());
        info.layers[1].layer_index = 1;
        assert_eq!(layer_weight_bytes_from_info(&info), vec![30, 40]);
    }

    #[test]
    fn direct_gguf_weights_charge_native_mtp_block_to_final_stage() {
        let tensors = vec![
            tensor("token_embd.weight", None, TensorRole::Embedding, 5),
            tensor("blk.0.attn_norm.weight", Some(0), TensorRole::Layer, 10),
            tensor("blk.1.attn_norm.weight", Some(1), TensorRole::Layer, 10),
            tensor("blk.2.nextn.eh_proj.weight", Some(2), TensorRole::Layer, 7),
            tensor("output_norm.weight", None, TensorRole::FinalNorm, 1),
            tensor("output.weight", None, TensorRole::Output, 9),
            tensor("general.alignment", None, TensorRole::Metadata, 3),
        ];

        assert_eq!(layer_weight_bytes_from_tensors(&tensors, 2), vec![17, 28]);
    }

    #[cfg(feature = "dynamic-native-runtime")]
    #[test]
    fn direct_gguf_weights_fall_back_when_dynamic_runtime_is_unloaded() {
        assert!(!skippy_runtime::native_runtime_loaded());
        let source_files = vec![SkippyPackageSourceFile {
            path: PathBuf::from("/models/not-opened.gguf"),
            bytes: 12,
            sha256: "abc123".to_string(),
        }];

        assert!(
            direct_gguf_layer_weight_bytes(&source_files, 2)
                .unwrap()
                .is_empty()
        );
    }

    fn tensor(
        name: &str,
        layer_index: Option<u32>,
        role: TensorRole,
        byte_size: u64,
    ) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            layer_index,
            role,
            ggml_type: 0,
            byte_size,
            element_count: byte_size,
        }
    }
}
