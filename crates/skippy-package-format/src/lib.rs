use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub mod materialization;
pub mod stage_admission;

pub const PACKAGE_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageManifest {
    pub schema_version: u32,
    pub package_id: String,
    pub model_id: String,
    pub source_model: SourceModel,
    pub format: String,
    pub layer_count: u32,
    pub model_metadata: BTreeMap<String, Value>,
    pub artifact_catalog: ArtifactCatalog,
    pub tensor_catalog: TensorCatalog,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sidecars: Vec<Sidecar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<Generation>,
    pub native_abi_version: String,
    pub generator_version: String,
    pub created_at_unix_secs: u64,
}

impl PackageManifest {
    /// Validate manifest structure and self-consistency, not source completeness.
    /// Creation/certification must separately compare with independent source
    /// directories and verify the artifact bytes; a manifest cannot certify itself.
    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();
        validate_manifest_fields(self, &mut issues);

        let artifacts = collect_artifacts(&self.artifact_catalog.entries, &mut issues);
        let tensors = collect_tensors(self, &mut issues);

        validate_metadata_artifact_binding(self, &artifacts, &mut issues);
        validate_owned_tensor_storage(&self.tensor_catalog.entries, &artifacts, &mut issues);
        validate_aliases(&self.tensor_catalog.entries, &tensors, &mut issues);
        validate_sidecars(&self.sidecars, &artifacts, &mut issues);
        if let Some(generation) = &self.generation {
            validate_generation(generation, self.layer_count, &mut issues);
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(ValidationErrors { issues })
        }
    }

    pub fn computed_package_id(&self) -> Result<String, serde_json::Error> {
        let mut normalized = self.clone();
        normalized.package_id.clear();
        normalized.created_at_unix_secs = 0;
        normalized
            .source_model
            .files
            .sort_by(|left, right| left.path.cmp(&right.path));
        normalized
            .artifact_catalog
            .entries
            .sort_by(|left, right| left.id.cmp(&right.id));
        normalized
            .tensor_catalog
            .entries
            .sort_by(|left, right| left.id.cmp(&right.id));
        normalized.sidecars.sort();
        let digest = Sha256::digest(serde_json::to_vec(&normalized)?);
        let hex = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Ok(format!("sha256:{hex}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceModel {
    pub sha256: String,
    pub metadata_artifact_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distribution_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<SourceFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFile {
    pub path: String,
    pub byte_size: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactCatalog {
    pub entries: Vec<Artifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub id: String,
    pub path: String,
    pub byte_size: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorCatalog {
    pub entries: Vec<Tensor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub id: String,
    pub name: String,
    pub ggml_type: u32,
    pub dimensions: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_ordinal: Option<u32>,
    pub storage: TensorStorage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TensorStorage {
    Owned {
        artifact_id: String,
        data_offset: u64,
        stored_length: u64,
        alignment: u64,
        integrity: TensorIntegrity,
    },
    Alias {
        target_tensor_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TensorIntegrity {
    ArtifactSha256,
    TensorSha256 { sha256: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sidecar {
    pub kind: SidecarKind,
    pub artifact_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Ord for Sidecar {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (&self.kind, &self.name, &self.artifact_id).cmp(&(
            &other.kind,
            &other.name,
            &other.artifact_id,
        ))
    }
}

impl PartialOrd for Sidecar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SidecarKind {
    Mmproj,
}

/// Generation capability declarations carried as package data.
///
/// Describes speculative-decoding capabilities of the source model only.
/// It is never stage-selection or loader policy: which strategy actually runs
/// is resolved from runtime configuration, and layer indices here must never
/// be consulted by a stage selector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Generation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_decoding: Option<SpeculativeDecoding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeDecoding {
    pub default: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub proposers: BTreeMap<String, ProposerSpec>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub strategies: BTreeMap<String, StrategySpec>,
}

/// Strategy wrapper: `#[serde(flatten)]` of the tagged enum forbids
/// `deny_unknown_fields` here, so unknown-field rejection happens inside the
/// tagged enum variants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposerSpec {
    #[serde(flatten)]
    pub kind: ProposerKind,
}

/// See [`ProposerSpec`] for the flatten/deny_unknown_fields tradeoff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrategySpec {
    #[serde(flatten)]
    pub kind: StrategyKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ProposerKind {
    #[serde(rename = "native-mtp")]
    NativeMtp {
        prediction_depth: u32,
        layer_indices: Vec<u32>,
    },
    NgramCache {
        ngram_min: u32,
        ngram_max: u32,
        max_proposal_tokens: u32,
        history_scope: String,
    },
    NgramSuffix {
        ngram_min: u32,
        ngram_max: u32,
        max_proposal_tokens: u32,
        history_scope: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowPolicy {
    pub default: String,
    pub initial_window: u32,
    pub min_window: u32,
    pub max_window: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline_depth: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum StrategyKind {
    /// Native-MTP strategies carry either a proposer reference or the inline
    /// `prediction_depth`/`layer_indices` form — never both, never neither.
    #[serde(rename = "native-mtp")]
    NativeMtp {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proposer: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prediction_depth: Option<u32>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        layer_indices: Vec<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_policy: Option<WindowPolicy>,
    },
    NgramCache {
        proposer: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_policy: Option<WindowPolicy>,
    },
    NgramSuffix {
        proposer: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_policy: Option<WindowPolicy>,
    },
    Composite {
        primary: String,
        extender: String,
        extension_policy: ExtensionPolicy,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_policy: Option<WindowPolicy>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtensionPolicy {
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_backoff_proposals: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub code: ValidationCode,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationCode {
    UnsupportedSchemaVersion,
    MissingValue,
    InvalidDigest,
    InvalidIdentifier,
    InvalidPath,
    UnknownSourceFile,
    InvalidMetadataArtifact,
    SourceIdentityMismatch,
    DuplicateArtifact,
    DuplicateArtifactPath,
    DuplicateSidecar,
    DuplicateTensor,
    LayerOutOfBounds,
    InvalidDimension,
    UnknownArtifact,
    InvalidAlignment,
    MisalignedOffset,
    StorageOutOfBounds,
    InvalidStoredLength,
    OverlappingStorage,
    UnknownAliasTarget,
    AliasTargetIsAlias,
    AliasMetadataMismatch,
    PackageIdentityMismatch,
    UnknownStrategy,
    UnknownProposer,
    InvalidGenerationValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationErrors {
    issues: Vec<ValidationIssue>,
}

impl ValidationErrors {
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }
}

impl fmt::Display for ValidationErrors {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid package manifest")?;
        for issue in &self.issues {
            write!(formatter, "; {}: {}", issue.path, issue.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationErrors {}

fn validate_manifest_fields(manifest: &PackageManifest, issues: &mut Vec<ValidationIssue>) {
    if manifest.schema_version != PACKAGE_SCHEMA_VERSION {
        push_issue(
            issues,
            ValidationCode::UnsupportedSchemaVersion,
            "schema_version",
            format!(
                "expected schema version {PACKAGE_SCHEMA_VERSION}, got {}",
                manifest.schema_version
            ),
        );
    }
    validate_nonempty("model_id", &manifest.model_id, issues);
    validate_nonempty("format", &manifest.format, issues);
    validate_nonempty("native_abi_version", &manifest.native_abi_version, issues);
    validate_nonempty("generator_version", &manifest.generator_version, issues);
    if manifest.layer_count == 0 {
        push_issue(
            issues,
            ValidationCode::MissingValue,
            "layer_count",
            "layer count must be greater than zero",
        );
    }
    validate_prefixed_sha256("package_id", &manifest.package_id, issues);
    validate_sha256("source_model.sha256", &manifest.source_model.sha256, issues);
    validate_nonempty(
        "source_model.metadata_artifact_id",
        &manifest.source_model.metadata_artifact_id,
        issues,
    );
    if manifest.model_metadata.is_empty() {
        push_issue(
            issues,
            ValidationCode::MissingValue,
            "model_metadata",
            "model metadata must not be empty",
        );
    }
    if !manifest.model_metadata.contains_key("general.architecture") {
        push_issue(
            issues,
            ValidationCode::MissingValue,
            "model_metadata.general.architecture",
            "model metadata must contain general.architecture",
        );
    }
    if manifest.source_model.files.is_empty() {
        push_issue(
            issues,
            ValidationCode::MissingValue,
            "source_model.files",
            "source file inventory must not be empty",
        );
    }
    if manifest.artifact_catalog.entries.is_empty() {
        push_issue(
            issues,
            ValidationCode::MissingValue,
            "artifact_catalog.entries",
            "artifact catalog must not be empty",
        );
    }
    if manifest.tensor_catalog.entries.is_empty() {
        push_issue(
            issues,
            ValidationCode::MissingValue,
            "tensor_catalog.entries",
            "tensor catalog must not be empty",
        );
    }
    match manifest.computed_package_id() {
        Ok(expected) if manifest.package_id != expected => push_issue(
            issues,
            ValidationCode::PackageIdentityMismatch,
            "package_id",
            format!(
                "package identity {:?} does not match canonical identity {:?}",
                manifest.package_id, expected
            ),
        ),
        Ok(_) => {}
        Err(error) => push_issue(
            issues,
            ValidationCode::PackageIdentityMismatch,
            "package_id",
            format!("cannot compute canonical package identity: {error}"),
        ),
    }

    let mut source_paths = BTreeSet::new();
    for (index, file) in manifest.source_model.files.iter().enumerate() {
        validate_relative_path(
            &format!("source_model.files[{index}].path"),
            &file.path,
            issues,
        );
        validate_sha256(
            &format!("source_model.files[{index}].sha256"),
            &file.sha256,
            issues,
        );
        if !source_paths.insert(file.path.as_str()) {
            push_issue(
                issues,
                ValidationCode::InvalidPath,
                format!("source_model.files[{index}].path"),
                format!("source file path {:?} appears more than once", file.path),
            );
        }
    }
    match &manifest.source_model.primary_file {
        Some(primary_file) => {
            validate_relative_path("source_model.primary_file", primary_file, issues);
            if !source_paths.contains(primary_file.as_str()) {
                push_issue(
                    issues,
                    ValidationCode::UnknownSourceFile,
                    "source_model.primary_file",
                    format!(
                        "primary file {:?} is absent from source files",
                        primary_file
                    ),
                );
            }
        }
        None => {
            push_issue(
                issues,
                ValidationCode::MissingValue,
                "source_model.primary_file",
                "primary source file is required",
            );
        }
    }
}

fn validate_metadata_artifact_binding(
    manifest: &PackageManifest,
    artifacts: &BTreeMap<&str, &Artifact>,
    issues: &mut Vec<ValidationIssue>,
) {
    let artifact_id = manifest.source_model.metadata_artifact_id.as_str();
    if artifact_id.trim().is_empty() {
        return;
    }
    let Some(artifact) = artifacts.get(artifact_id) else {
        push_issue(
            issues,
            ValidationCode::UnknownArtifact,
            "source_model.metadata_artifact_id",
            format!("metadata artifact {artifact_id:?} is absent from artifact catalog"),
        );
        return;
    };
    if Path::new(&artifact.path)
        .extension()
        .and_then(|value| value.to_str())
        != Some("gguf")
    {
        push_issue(
            issues,
            ValidationCode::InvalidMetadataArtifact,
            "source_model.metadata_artifact_id",
            format!("metadata artifact {:?} is not a GGUF artifact", artifact.id),
        );
    }
    if manifest
        .sidecars
        .iter()
        .any(|sidecar| sidecar.artifact_id == artifact.id)
    {
        push_issue(
            issues,
            ValidationCode::InvalidMetadataArtifact,
            "source_model.metadata_artifact_id",
            format!("metadata artifact {:?} has a sidecar role", artifact.id),
        );
    }
    if artifact.sha256 != manifest.source_model.sha256 {
        push_issue(
            issues,
            ValidationCode::SourceIdentityMismatch,
            "source_model.metadata_artifact_id",
            format!(
                "metadata artifact {:?} SHA-256 does not match source model digest",
                artifact.id
            ),
        );
    }
    let Some(primary_path) = manifest.source_model.primary_file.as_deref() else {
        return;
    };
    let Some(primary_file) = manifest
        .source_model
        .files
        .iter()
        .find(|file| file.path == primary_path)
    else {
        return;
    };
    if manifest.source_model.sha256 != primary_file.sha256 {
        push_issue(
            issues,
            ValidationCode::SourceIdentityMismatch,
            "source_model.sha256",
            "source model digest does not match the primary source file",
        );
    }
    if artifact.sha256 != primary_file.sha256 {
        push_issue(
            issues,
            ValidationCode::SourceIdentityMismatch,
            "source_model.metadata_artifact_id",
            format!(
                "metadata artifact {:?} SHA-256 does not match primary source file {:?}",
                artifact.id, primary_file.path
            ),
        );
    }
    if artifact.byte_size != primary_file.byte_size {
        push_issue(
            issues,
            ValidationCode::SourceIdentityMismatch,
            "source_model.metadata_artifact_id",
            format!(
                "metadata artifact {:?} byte size does not match primary source file {:?}",
                artifact.id, primary_file.path
            ),
        );
    }
}

fn collect_artifacts<'a>(
    artifacts: &'a [Artifact],
    issues: &mut Vec<ValidationIssue>,
) -> BTreeMap<&'a str, &'a Artifact> {
    let mut by_id = BTreeMap::new();
    let mut paths = BTreeSet::new();
    for (index, artifact) in artifacts.iter().enumerate() {
        let prefix = format!("artifacts[{index}]");
        validate_identifier(&format!("{prefix}.id"), &artifact.id, issues);
        validate_relative_path(&format!("{prefix}.path"), &artifact.path, issues);
        validate_sha256(&format!("{prefix}.sha256"), &artifact.sha256, issues);
        if by_id.insert(artifact.id.as_str(), artifact).is_some() {
            push_issue(
                issues,
                ValidationCode::DuplicateArtifact,
                format!("{prefix}.id"),
                format!("artifact id {:?} appears more than once", artifact.id),
            );
        }
        if !paths.insert(artifact.path.as_str()) {
            push_issue(
                issues,
                ValidationCode::DuplicateArtifactPath,
                format!("{prefix}.path"),
                format!("artifact path {:?} appears more than once", artifact.path),
            );
        }
    }
    by_id
}

fn collect_tensors<'a>(
    manifest: &'a PackageManifest,
    issues: &mut Vec<ValidationIssue>,
) -> BTreeMap<&'a str, &'a Tensor> {
    let tensors = &manifest.tensor_catalog.entries;
    let mut by_id = BTreeMap::new();
    let mut names = BTreeSet::new();
    for (index, tensor) in tensors.iter().enumerate() {
        let prefix = format!("tensors[{index}]");
        validate_nonempty(&format!("{prefix}.id"), &tensor.id, issues);
        validate_nonempty(&format!("{prefix}.name"), &tensor.name, issues);
        if tensor.dimensions.is_empty() || tensor.dimensions.contains(&0) {
            push_issue(
                issues,
                ValidationCode::InvalidDimension,
                format!("{prefix}.dimensions"),
                "tensor dimensions must be non-empty and non-zero",
            );
        }
        if let Some(layer_ordinal) = tensor.layer_ordinal
            && layer_ordinal >= manifest.layer_count
        {
            push_issue(
                issues,
                ValidationCode::LayerOutOfBounds,
                format!("{prefix}.layer_ordinal"),
                format!(
                    "layer ordinal {layer_ordinal} is outside model layer count {}",
                    manifest.layer_count
                ),
            );
        }
        if by_id.insert(tensor.id.as_str(), tensor).is_some() {
            push_issue(
                issues,
                ValidationCode::DuplicateTensor,
                format!("{prefix}.id"),
                format!("tensor id {:?} appears more than once", tensor.id),
            );
        }
        if !names.insert(tensor.name.as_str()) {
            push_issue(
                issues,
                ValidationCode::DuplicateTensor,
                format!("{prefix}.name"),
                format!("tensor name {:?} appears more than once", tensor.name),
            );
        }
    }
    by_id
}

fn validate_owned_tensor_storage(
    tensors: &[Tensor],
    artifacts: &BTreeMap<&str, &Artifact>,
    issues: &mut Vec<ValidationIssue>,
) {
    let mut ranges: BTreeMap<&str, Vec<(u64, u64, usize)>> = BTreeMap::new();
    for (index, tensor) in tensors.iter().enumerate() {
        let TensorStorage::Owned {
            artifact_id,
            data_offset,
            stored_length,
            alignment,
            integrity,
        } = &tensor.storage
        else {
            continue;
        };
        let prefix = format!("tensors[{index}].storage");
        let Some(artifact) = artifacts.get(artifact_id.as_str()) else {
            push_issue(
                issues,
                ValidationCode::UnknownArtifact,
                format!("{prefix}.artifact_id"),
                format!("artifact {:?} does not exist", artifact_id),
            );
            continue;
        };
        if !alignment.is_power_of_two() {
            push_issue(
                issues,
                ValidationCode::InvalidAlignment,
                format!("{prefix}.alignment"),
                format!("alignment {alignment} is not a non-zero power of two"),
            );
        } else if data_offset % alignment != 0 {
            push_issue(
                issues,
                ValidationCode::MisalignedOffset,
                format!("{prefix}.data_offset"),
                format!("offset {data_offset} is not aligned to {alignment} bytes"),
            );
        }
        if *stored_length == 0 {
            push_issue(
                issues,
                ValidationCode::InvalidStoredLength,
                format!("{prefix}.stored_length"),
                "owned tensor storage must not be empty",
            );
        }
        match data_offset.checked_add(*stored_length) {
            Some(end) if end <= artifact.byte_size => {
                ranges
                    .entry(artifact_id)
                    .or_default()
                    .push((*data_offset, end, index));
            }
            _ => push_issue(
                issues,
                ValidationCode::StorageOutOfBounds,
                prefix.clone(),
                format!(
                    "range {data_offset}+{stored_length} exceeds artifact size {}",
                    artifact.byte_size
                ),
            ),
        }
        if let TensorIntegrity::TensorSha256 { sha256 } = integrity {
            validate_sha256(&format!("{prefix}.integrity.sha256"), sha256, issues);
        }
    }

    for (artifact_id, artifact_ranges) in &mut ranges {
        artifact_ranges.sort_unstable_by_key(|range| (range.0, range.1, range.2));
        for pair in artifact_ranges.windows(2) {
            let previous = pair[0];
            let current = pair[1];
            if current.0 < previous.1 {
                push_issue(
                    issues,
                    ValidationCode::OverlappingStorage,
                    format!("tensors[{}].storage", current.2),
                    format!(
                        "storage overlaps tensors[{}] in artifact {:?}",
                        previous.2, artifact_id
                    ),
                );
            }
        }
    }
}

fn validate_aliases(
    tensors: &[Tensor],
    by_id: &BTreeMap<&str, &Tensor>,
    issues: &mut Vec<ValidationIssue>,
) {
    for (index, tensor) in tensors.iter().enumerate() {
        let TensorStorage::Alias { target_tensor_id } = &tensor.storage else {
            continue;
        };
        let path = format!("tensors[{index}].storage.target_tensor_id");
        let Some(target) = by_id.get(target_tensor_id.as_str()) else {
            push_issue(
                issues,
                ValidationCode::UnknownAliasTarget,
                path,
                format!("tensor {:?} does not exist", target_tensor_id),
            );
            continue;
        };
        if matches!(target.storage, TensorStorage::Alias { .. }) {
            push_issue(
                issues,
                ValidationCode::AliasTargetIsAlias,
                path,
                format!("alias target {:?} is not owned storage", target_tensor_id),
            );
        }
        if tensor.ggml_type != target.ggml_type || tensor.dimensions != target.dimensions {
            push_issue(
                issues,
                ValidationCode::AliasMetadataMismatch,
                format!("tensors[{index}].storage"),
                format!(
                    "alias metadata does not match target tensor {:?}",
                    target_tensor_id
                ),
            );
        }
    }
}

fn validate_sidecars(
    sidecars: &[Sidecar],
    artifacts: &BTreeMap<&str, &Artifact>,
    issues: &mut Vec<ValidationIssue>,
) {
    let mut seen = BTreeSet::new();
    for (index, sidecar) in sidecars.iter().enumerate() {
        let prefix = format!("sidecars[{index}]");
        if !seen.insert((sidecar.kind, sidecar.name.as_deref())) {
            push_issue(
                issues,
                ValidationCode::DuplicateSidecar,
                prefix.clone(),
                format!(
                    "sidecar semantic identity ({:?}, {:?}) appears more than once",
                    sidecar.kind, sidecar.name
                ),
            );
        }
        if !artifacts.contains_key(sidecar.artifact_id.as_str()) {
            push_issue(
                issues,
                ValidationCode::UnknownArtifact,
                format!("{prefix}.artifact_id"),
                format!("artifact {:?} does not exist", sidecar.artifact_id),
            );
        }
        if let Some(name) = &sidecar.name {
            validate_nonempty(&format!("{prefix}.name"), name, issues);
        }
    }
}

fn validate_nonempty(path: &str, value: &str, issues: &mut Vec<ValidationIssue>) {
    if value.trim().is_empty() {
        push_issue(
            issues,
            ValidationCode::MissingValue,
            path,
            "value must not be empty",
        );
    }
}

fn validate_identifier(path: &str, value: &str, issues: &mut Vec<ValidationIssue>) {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        push_issue(
            issues,
            ValidationCode::InvalidIdentifier,
            path,
            format!("identifier {:?} contains unsupported characters", value),
        );
    }
}

fn validate_relative_path(path: &str, value: &str, issues: &mut Vec<ValidationIssue>) {
    let candidate = Path::new(value);
    if value.is_empty()
        || value.contains('\\')
        || value.contains("//")
        || value.ends_with('/')
        || candidate.is_absolute()
        || candidate
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        push_issue(
            issues,
            ValidationCode::InvalidPath,
            path,
            format!(
                "path {:?} must be a normalized relative package path",
                value
            ),
        );
    }
}

fn validate_prefixed_sha256(path: &str, value: &str, issues: &mut Vec<ValidationIssue>) {
    let Some(digest) = value.strip_prefix("sha256:") else {
        push_issue(
            issues,
            ValidationCode::InvalidDigest,
            path,
            "digest must use the sha256:<hex> form",
        );
        return;
    };
    validate_sha256(path, digest, issues);
}

fn validate_sha256(path: &str, value: &str, issues: &mut Vec<ValidationIssue>) {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        push_issue(
            issues,
            ValidationCode::InvalidDigest,
            path,
            "digest must contain 64 lowercase hexadecimal characters",
        );
    }
}

fn push_issue(
    issues: &mut Vec<ValidationIssue>,
    code: ValidationCode,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issues.push(ValidationIssue {
        code,
        path: path.into(),
        message: message.into(),
    });
}

/// Schema-level (graph-checkable) generation validation only.
///
/// Deliberately excludes runtime-capability bounds such as the native-MTP
/// prediction-depth value, n-gram window limits, or history-scope vocabulary:
/// those are enforced by the runtime/preflight, not by the package format.
fn validate_generation(
    generation: &Generation,
    layer_count: u32,
    issues: &mut Vec<ValidationIssue>,
) {
    let Some(speculative) = &generation.speculative_decoding else {
        return;
    };
    if speculative.default.trim().is_empty() {
        push_issue(
            issues,
            ValidationCode::InvalidGenerationValue,
            "generation.speculative_decoding.default",
            "default must not be empty",
        );
    }
    for name in speculative.proposers.keys() {
        if name.trim().is_empty() {
            push_issue(
                issues,
                ValidationCode::InvalidGenerationValue,
                "generation.speculative_decoding.proposers",
                "proposer names must not be empty",
            );
        }
        validate_proposer(name, &speculative.proposers[name], layer_count, issues);
    }
    for name in speculative.strategies.keys() {
        if name.trim().is_empty() {
            push_issue(
                issues,
                ValidationCode::InvalidGenerationValue,
                "generation.speculative_decoding.strategies",
                "strategy names must not be empty",
            );
        }
        validate_strategy(
            name,
            &speculative.strategies[name],
            &speculative.proposers,
            layer_count,
            issues,
        );
    }
    if !speculative.default.trim().is_empty()
        && !speculative.strategies.contains_key(&speculative.default)
    {
        push_issue(
            issues,
            ValidationCode::UnknownStrategy,
            "generation.speculative_decoding.default",
            format!(
                "default strategy {:?} is not declared under strategies",
                speculative.default
            ),
        );
    }
}

fn validate_proposer(
    name: &str,
    proposer: &ProposerSpec,
    layer_count: u32,
    issues: &mut Vec<ValidationIssue>,
) {
    match &proposer.kind {
        ProposerKind::NativeMtp {
            prediction_depth,
            layer_indices,
        } => {
            if *prediction_depth == 0 {
                push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("generation.speculative_decoding.proposers[{name}].prediction_depth"),
                    "prediction_depth must be greater than zero",
                );
            }
            if layer_indices.is_empty() {
                push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("generation.speculative_decoding.proposers[{name}].layer_indices"),
                    "layer_indices must not be empty",
                );
            }
            for (index, layer_index) in layer_indices.iter().enumerate() {
                if *layer_index >= layer_count {
                    push_issue(
                        issues,
                        ValidationCode::LayerOutOfBounds,
                        format!(
                            "generation.speculative_decoding.proposers[{name}].layer_indices[{index}]"
                        ),
                        format!(
                            "layer index {layer_index} is out of range for layer_count {layer_count}"
                        ),
                    );
                }
            }
        }
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
        } => {
            if *ngram_min == 0 || *ngram_max < *ngram_min {
                push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("generation.speculative_decoding.proposers[{name}]"),
                    "ngram_min and ngram_max must satisfy 1 <= ngram_min <= ngram_max",
                );
            }
            if *max_proposal_tokens == 0 {
                push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!(
                        "generation.speculative_decoding.proposers[{name}].max_proposal_tokens"
                    ),
                    "max_proposal_tokens must be greater than zero",
                );
            }
            if history_scope.trim().is_empty() {
                push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("generation.speculative_decoding.proposers[{name}].history_scope"),
                    "history_scope must not be empty",
                );
            }
        }
    }
}

fn validate_strategy(
    name: &str,
    strategy: &StrategySpec,
    proposers: &BTreeMap<String, ProposerSpec>,
    layer_count: u32,
    issues: &mut Vec<ValidationIssue>,
) {
    let require_proposer = |field: &str,
                            proposer_name: &str,
                            allowed: &[&str],
                            issues: &mut Vec<ValidationIssue>| {
        if proposer_name.trim().is_empty() {
            push_issue(
                issues,
                ValidationCode::InvalidGenerationValue,
                format!("generation.speculative_decoding.strategies[{name}].{field}"),
                "proposer reference must not be empty",
            );
        } else if !proposers.contains_key(proposer_name) {
            push_issue(
                issues,
                ValidationCode::UnknownProposer,
                format!("generation.speculative_decoding.strategies[{name}].{field}"),
                format!("references undeclared proposer {proposer_name:?}"),
            );
        } else {
            let actual = match &proposers[proposer_name].kind {
                ProposerKind::NativeMtp { .. } => "native-mtp",
                ProposerKind::NgramCache { .. } => "ngram-cache",
                ProposerKind::NgramSuffix { .. } => "ngram-suffix",
            };
            if !allowed.contains(&actual) {
                push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("generation.speculative_decoding.strategies[{name}].{field}"),
                    format!(
                        "proposer {proposer_name:?} has kind {actual:?}, which is not compatible with this strategy (expected one of {allowed:?})"
                    ),
                );
            }
        }
    };
    let validate_window_policy =
        |window_policy: &Option<WindowPolicy>, issues: &mut Vec<ValidationIssue>| {
            if let Some(window) = window_policy {
                if window.default.trim().is_empty() {
                    push_issue(
                        issues,
                        ValidationCode::InvalidGenerationValue,
                        format!(
                            "generation.speculative_decoding.strategies[{name}].window_policy.default"
                        ),
                        "window_policy.default must not be empty",
                    );
                }
                if window.initial_window == 0 || window.min_window == 0 || window.max_window == 0 {
                    push_issue(
                        issues,
                        ValidationCode::InvalidGenerationValue,
                        format!("generation.speculative_decoding.strategies[{name}].window_policy"),
                        "window_policy windows must be greater than zero",
                    );
                }
                if window.pipeline_depth == Some(0) {
                    push_issue(
                        issues,
                        ValidationCode::InvalidGenerationValue,
                        format!(
                            "generation.speculative_decoding.strategies[{name}].window_policy.pipeline_depth"
                        ),
                        "window_policy.pipeline_depth must be greater than zero",
                    );
                }
                if window.min_window > window.max_window {
                    push_issue(
                        issues,
                        ValidationCode::InvalidGenerationValue,
                        format!("generation.speculative_decoding.strategies[{name}].window_policy"),
                        "window_policy min_window must not exceed max_window",
                    );
                }
                if window.initial_window < window.min_window
                    || window.initial_window > window.max_window
                {
                    push_issue(
                        issues,
                        ValidationCode::InvalidGenerationValue,
                        format!("generation.speculative_decoding.strategies[{name}].window_policy"),
                        "window_policy initial_window must lie within min_window..max_window",
                    );
                }
            }
        };
    match &strategy.kind {
        StrategyKind::NativeMtp {
            proposer,
            prediction_depth,
            layer_indices,
            window_policy,
        } => {
            validate_window_policy(window_policy, issues);
            match (
                proposer,
                prediction_depth.is_some(),
                !layer_indices.is_empty(),
            ) {
                (Some(proposer_name), false, false) => {
                    require_proposer("proposer", proposer_name, &["native-mtp"], issues);
                }
                (None, true, true) => {
                    if *prediction_depth == Some(0) {
                        push_issue(
                            issues,
                            ValidationCode::InvalidGenerationValue,
                            format!(
                                "generation.speculative_decoding.strategies[{name}].prediction_depth"
                            ),
                            "prediction_depth must be greater than zero",
                        );
                    }
                    for (index, layer_index) in layer_indices.iter().enumerate() {
                        if *layer_index >= layer_count {
                            push_issue(
                                issues,
                                ValidationCode::LayerOutOfBounds,
                                format!(
                                    "generation.speculative_decoding.strategies[{name}].layer_indices[{index}]"
                                ),
                                format!(
                                    "layer index {layer_index} is out of range for layer_count {layer_count}"
                                ),
                            );
                        }
                    }
                }
                (Some(_), _, _) => push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("generation.speculative_decoding.strategies[{name}]"),
                    "native-mtp strategy must set either proposer or the inline prediction_depth/layer_indices form, not both",
                ),
                (None, true, false) => push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("generation.speculative_decoding.strategies[{name}].layer_indices"),
                    "inline native-mtp strategy must declare layer_indices alongside prediction_depth",
                ),
                (None, false, true) => push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("generation.speculative_decoding.strategies[{name}].prediction_depth"),
                    "inline native-mtp strategy must declare prediction_depth alongside layer_indices",
                ),
                (None, false, false) => push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!("generation.speculative_decoding.strategies[{name}]"),
                    "native-mtp strategy must set either proposer or the inline prediction_depth/layer_indices form",
                ),
            }
        }
        StrategyKind::NgramCache {
            proposer,
            window_policy,
        } => {
            validate_window_policy(window_policy, issues);
            require_proposer("proposer", proposer, &["ngram-cache"], issues);
        }
        StrategyKind::NgramSuffix {
            proposer,
            window_policy,
        } => {
            validate_window_policy(window_policy, issues);
            require_proposer("proposer", proposer, &["ngram-suffix"], issues);
        }
        StrategyKind::Composite {
            primary,
            extender,
            extension_policy,
            window_policy,
        } => {
            validate_window_policy(window_policy, issues);
            if extension_policy.max_tokens == 0 {
                push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!(
                        "generation.speculative_decoding.strategies[{name}].extension_policy.max_tokens"
                    ),
                    "extension_policy.max_tokens must be greater than zero",
                );
            }
            if extension_policy.initial_tokens == Some(0) {
                push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!(
                        "generation.speculative_decoding.strategies[{name}].extension_policy.initial_tokens"
                    ),
                    "extension_policy.initial_tokens must be greater than zero",
                );
            }
            if let Some(initial_tokens) = extension_policy.initial_tokens
                && initial_tokens > extension_policy.max_tokens
            {
                push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!(
                        "generation.speculative_decoding.strategies[{name}].extension_policy.initial_tokens"
                    ),
                    "extension_policy.initial_tokens must not exceed max_tokens",
                );
            }
            if extension_policy.tail_backoff_proposals == Some(0) {
                push_issue(
                    issues,
                    ValidationCode::InvalidGenerationValue,
                    format!(
                        "generation.speculative_decoding.strategies[{name}].extension_policy.tail_backoff_proposals"
                    ),
                    "extension_policy.tail_backoff_proposals must be greater than zero",
                );
            }
            require_proposer("primary", primary, &["native-mtp"], issues);
            require_proposer(
                "extender",
                extender,
                &["ngram-cache", "ngram-suffix"],
                issues,
            );
        }
    }
}

#[cfg(test)]
mod tests;
