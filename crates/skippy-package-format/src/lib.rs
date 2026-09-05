use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

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
        normalized.sidecars.sort_by(|left, right| {
            (&left.kind, &left.name, &left.artifact_id).cmp(&(
                &right.kind,
                &right.name,
                &right.artifact_id,
            ))
        });
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
    pub kind: String,
    pub artifact_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
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
    DuplicateArtifact,
    DuplicateArtifactPath,
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
    if let Some(primary_file) = &manifest.source_model.primary_file {
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
    for (index, sidecar) in sidecars.iter().enumerate() {
        let prefix = format!("sidecars[{index}]");
        validate_identifier(&format!("{prefix}.kind"), &sidecar.kind, issues);
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
mod tests {
    use super::*;

    const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn valid_manifest_passes() {
        fixture().validate().unwrap();
    }

    #[test]
    fn rejects_unknown_and_overlapping_storage() {
        let mut manifest = fixture();
        manifest.tensor_catalog.entries.push(Tensor {
            id: "tensor-2".to_string(),
            name: "blk.0.attn_k.weight".to_string(),
            ggml_type: 8,
            dimensions: vec![16, 16],
            layer_ordinal: Some(0),
            storage: TensorStorage::Owned {
                artifact_id: "layer-0".to_string(),
                data_offset: 96,
                stored_length: 64,
                alignment: 32,
                integrity: TensorIntegrity::ArtifactSha256,
            },
        });
        manifest.tensor_catalog.entries.push(Tensor {
            id: "tensor-3".to_string(),
            name: "blk.0.attn_v.weight".to_string(),
            ggml_type: 8,
            dimensions: vec![16, 16],
            layer_ordinal: Some(0),
            storage: TensorStorage::Owned {
                artifact_id: "missing".to_string(),
                data_offset: 0,
                stored_length: 32,
                alignment: 32,
                integrity: TensorIntegrity::ArtifactSha256,
            },
        });

        let error = manifest.validate().unwrap_err();
        let codes = error
            .issues()
            .iter()
            .map(|issue| issue.code)
            .collect::<BTreeSet<_>>();
        assert!(codes.contains(&ValidationCode::OverlappingStorage));
        assert!(codes.contains(&ValidationCode::UnknownArtifact));
    }

    #[test]
    fn aliases_must_reference_matching_owned_tensors() {
        let mut manifest = fixture();
        manifest.tensor_catalog.entries.push(Tensor {
            id: "tensor-alias".to_string(),
            name: "output.weight".to_string(),
            ggml_type: 1,
            dimensions: vec![16, 16],
            layer_ordinal: None,
            storage: TensorStorage::Alias {
                target_tensor_id: "tensor-1".to_string(),
            },
        });

        let error = manifest.validate().unwrap_err();
        assert!(
            error
                .issues()
                .iter()
                .any(|issue| issue.code == ValidationCode::AliasMetadataMismatch)
        );
    }

    #[test]
    fn manifest_round_trip_is_stable() {
        let manifest = fixture();
        let encoded = serde_json::to_string_pretty(&manifest).unwrap();
        let decoded: PackageManifest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, manifest);
        decoded.validate().unwrap();
    }

    #[test]
    fn package_identity_ignores_catalog_enumeration_order() {
        let mut manifest = fixture();
        manifest.source_model.files.push(SourceFile {
            path: "tokenizer.json".to_string(),
            byte_size: 32,
            sha256: DIGEST.to_string(),
        });
        manifest.artifact_catalog.entries.push(Artifact {
            id: "tokenizer".to_string(),
            path: "sidecars/tokenizer.json".to_string(),
            byte_size: 32,
            sha256: DIGEST.to_string(),
        });
        manifest.tensor_catalog.entries.push(Tensor {
            id: "tensor-2".to_string(),
            name: "blk.0.attn_k.weight".to_string(),
            ggml_type: 8,
            dimensions: vec![16, 16],
            layer_ordinal: Some(0),
            storage: TensorStorage::Owned {
                artifact_id: "layer-0".to_string(),
                data_offset: 128,
                stored_length: 64,
                alignment: 32,
                integrity: TensorIntegrity::ArtifactSha256,
            },
        });
        manifest.sidecars.push(Sidecar {
            kind: "tokenizer".to_string(),
            artifact_id: "tokenizer".to_string(),
            name: Some("default".to_string()),
        });
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest.validate().unwrap();

        let mut reordered = manifest.clone();
        reordered.source_model.files.reverse();
        reordered.artifact_catalog.entries.reverse();
        reordered.tensor_catalog.entries.reverse();
        reordered.sidecars.reverse();
        assert_eq!(
            reordered.computed_package_id().unwrap(),
            manifest.package_id
        );
    }

    #[test]
    fn package_identity_binds_tensor_metadata() {
        let manifest = fixture();
        let mut changed = manifest.clone();
        changed.tensor_catalog.entries[0].dimensions = vec![32, 16];

        let error = changed.validate().unwrap_err();
        assert!(
            error
                .issues()
                .iter()
                .any(|issue| issue.code == ValidationCode::PackageIdentityMismatch)
        );
    }

    #[test]
    fn unknown_manifest_fields_are_rejected() {
        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded
            .as_object_mut()
            .unwrap()
            .insert("future_policy".to_string(), Value::Bool(true));
        assert!(serde_json::from_value::<PackageManifest>(encoded).is_err());
    }

    #[test]
    fn rejects_empty_dimensions_and_empty_storage() {
        let mut manifest = fixture();
        manifest.tensor_catalog.entries[0].dimensions.clear();
        if let TensorStorage::Owned { stored_length, .. } =
            &mut manifest.tensor_catalog.entries[0].storage
        {
            *stored_length = 0;
        }
        manifest.package_id = manifest.computed_package_id().unwrap();
        let error = manifest.validate().unwrap_err();
        let codes = error
            .issues()
            .iter()
            .map(|issue| issue.code)
            .collect::<BTreeSet<_>>();
        assert!(codes.contains(&ValidationCode::InvalidDimension));
        assert!(codes.contains(&ValidationCode::InvalidStoredLength));
    }

    #[test]
    fn absent_generation_passes_validation() {
        let manifest = fixture();
        assert!(manifest.generation.is_none());
        manifest.validate().unwrap();
    }

    #[test]
    fn generation_with_all_variants_round_trips_and_passes() {
        let mut manifest = fixture();
        manifest.generation = Some(Generation {
            speculative_decoding: Some(SpeculativeDecoding {
                default: "mtp".to_string(),
                proposers: BTreeMap::from([
                    (
                        "mtp-head".to_string(),
                        ProposerSpec {
                            kind: ProposerKind::NativeMtp {
                                prediction_depth: 1,
                                layer_indices: vec![0],
                            },
                        },
                    ),
                    (
                        "suffix".to_string(),
                        ProposerSpec {
                            kind: ProposerKind::NgramSuffix {
                                ngram_min: 3,
                                ngram_max: 8,
                                max_proposal_tokens: 4,
                                history_scope: "request".to_string(),
                            },
                        },
                    ),
                    (
                        "cache-head".to_string(),
                        ProposerSpec {
                            kind: ProposerKind::NgramCache {
                                ngram_min: 2,
                                ngram_max: 6,
                                max_proposal_tokens: 8,
                                history_scope: "session".to_string(),
                            },
                        },
                    ),
                ]),
                strategies: BTreeMap::from([
                    (
                        "mtp".to_string(),
                        StrategySpec {
                            kind: StrategyKind::NativeMtp {
                                proposer: Some("mtp-head".to_string()),
                                prediction_depth: None,
                                layer_indices: Vec::new(),
                                window_policy: Some(WindowPolicy {
                                    default: "fixed".to_string(),
                                    initial_window: 1,
                                    min_window: 1,
                                    max_window: 4,
                                    pipeline_depth: Some(2),
                                }),
                            },
                        },
                    ),
                    (
                        "cache".to_string(),
                        StrategySpec {
                            kind: StrategyKind::NgramCache {
                                proposer: "cache-head".to_string(),
                                window_policy: None,
                            },
                        },
                    ),
                    (
                        "suffix-strategy".to_string(),
                        StrategySpec {
                            kind: StrategyKind::NgramSuffix {
                                proposer: "suffix".to_string(),
                                window_policy: None,
                            },
                        },
                    ),
                    (
                        "combined".to_string(),
                        StrategySpec {
                            kind: StrategyKind::Composite {
                                primary: "mtp-head".to_string(),
                                extender: "suffix".to_string(),
                                extension_policy: ExtensionPolicy {
                                    max_tokens: 8,
                                    initial_tokens: Some(4),
                                    tail_backoff_proposals: Some(2),
                                },
                                window_policy: None,
                            },
                        },
                    ),
                ]),
            }),
        });
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest.validate().unwrap();

        let encoded = serde_json::to_string_pretty(&manifest).unwrap();
        let decoded: PackageManifest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, manifest);
        decoded.validate().unwrap();

        let value = serde_json::to_value(&manifest).unwrap();
        let strategy = &value["generation"]["speculative_decoding"]["strategies"]["mtp"];
        assert_eq!(strategy["type"], "native-mtp");
        assert_eq!(strategy["proposer"], "mtp-head");
        assert_eq!(strategy["window_policy"]["default"], "fixed");
        assert_eq!(strategy["window_policy"]["max_window"], 4);
        assert_eq!(strategy["window_policy"]["pipeline_depth"], 2);
        let proposer = &value["generation"]["speculative_decoding"]["proposers"]["mtp-head"];
        assert_eq!(proposer["type"], "native-mtp");
        assert_eq!(proposer["prediction_depth"], 1);
        assert_eq!(proposer["layer_indices"], serde_json::json!([0]));
        let cache_proposer =
            &value["generation"]["speculative_decoding"]["proposers"]["cache-head"];
        assert_eq!(cache_proposer["type"], "ngram-cache");
        let suffix_proposer = &value["generation"]["speculative_decoding"]["proposers"]["suffix"];
        assert_eq!(suffix_proposer["type"], "ngram-suffix");
        let suffix_strategy =
            &value["generation"]["speculative_decoding"]["strategies"]["suffix-strategy"];
        assert_eq!(suffix_strategy["type"], "ngram-suffix");
        let combined = &value["generation"]["speculative_decoding"]["strategies"]["combined"];
        assert_eq!(combined["type"], "composite");
        assert_eq!(combined["extension_policy"]["max_tokens"], 8);
        assert_eq!(combined["extension_policy"]["initial_tokens"], 4);
        assert_eq!(combined["extension_policy"]["tail_backoff_proposals"], 2);
        assert!(strategy.get("kind").is_none());
        assert!(proposer.get("kind").is_none());
    }

    #[test]
    fn rejects_unknown_strategy_type_at_parse() {
        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "s",
                "strategies": { "s": { "type": "draft-token" } }
            }
        });
        assert!(serde_json::from_value::<PackageManifest>(encoded).is_err());
    }

    #[test]
    fn rejects_unknown_proposer_type_at_parse() {
        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "s",
                "proposers": { "p": { "type": "lookup-table" } }
            }
        });
        assert!(serde_json::from_value::<PackageManifest>(encoded).is_err());
    }

    #[test]
    fn rejects_unknown_fields_inside_generation_at_parse() {
        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "s",
                "strategies": { "s": { "type": "native-mtp", "policy": "aggressive" } }
            }
        });
        assert!(serde_json::from_value::<PackageManifest>(encoded).is_err());
    }

    #[test]
    fn rejects_unknown_nested_window_policy_field_at_parse() {
        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "mtp",
                "strategies": {
                    "mtp": {
                        "type": "native-mtp",
                        "prediction_depth": 1,
                        "layer_indices": [0],
                        "window_policy": {
                            "default": "fixed",
                            "initial_window": 1,
                            "min_window": 1,
                            "max_window": 4,
                            "cooldown": 3
                        }
                    }
                }
            }
        });
        let error = serde_json::from_value::<PackageManifest>(encoded).unwrap_err();
        assert!(error.to_string().contains("cooldown"));
    }

    #[test]
    fn accepts_inline_native_mtp_strategy_with_window_policy() {
        let manifest = fixture();
        let encoded_manifest = manifest.clone();
        let mut encoded = serde_json::to_value(&manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "mtp",
                "strategies": {
                    "mtp": {
                        "type": "native-mtp",
                        "prediction_depth": 1,
                        "layer_indices": [0],
                        "window_policy": {
                            "default": "fixed",
                            "initial_window": 1,
                            "min_window": 1,
                            "max_window": 4,
                            "pipeline_depth": 2
                        }
                    }
                }
            }
        });
        let decoded: PackageManifest = serde_json::from_value(encoded).unwrap();
        assert!(decoded.generation.is_some());
        let mut decoded = decoded;
        decoded.package_id = decoded.computed_package_id().unwrap();
        decoded.validate().unwrap();
        let redecoded: PackageManifest =
            serde_json::from_str(&serde_json::to_string(&decoded).unwrap()).unwrap();
        assert_eq!(redecoded, decoded);
        let mut with_generation = encoded_manifest;
        with_generation.generation = decoded.generation;
        with_generation.package_id = with_generation.computed_package_id().unwrap();
        with_generation.validate().unwrap();

        let value = serde_json::to_value(&with_generation).unwrap();
        let strategy = &value["generation"]["speculative_decoding"]["strategies"]["mtp"];
        assert_eq!(strategy["prediction_depth"], 1);
        assert_eq!(strategy["layer_indices"], serde_json::json!([0]));
        assert!(strategy.get("proposer").is_none());
    }

    #[test]
    fn rejects_native_mtp_strategy_with_neither_nor_both_forms() {
        let build = |strategies: serde_json::Value| {
            let manifest = fixture();
            let mut encoded = serde_json::to_value(manifest).unwrap();
            encoded["generation"] = serde_json::json!({
                "speculative_decoding": {
                    "default": "mtp",
                    "strategies": strategies
                }
            });
            let decoded: PackageManifest = serde_json::from_value(encoded).unwrap();
            let mut decoded = decoded;
            decoded.package_id = decoded.computed_package_id().unwrap();
            decoded.validate().unwrap_err()
        };

        let neither = build(serde_json::json!({
            "mtp": { "type": "native-mtp" }
        }));
        assert!(neither.issues().iter().any(|issue| {
            issue.code == ValidationCode::InvalidGenerationValue
                && issue.path == "generation.speculative_decoding.strategies[mtp]"
        }));

        let both = build(serde_json::json!({
            "mtp": {
                "type": "native-mtp",
                "proposer": "ghost",
                "prediction_depth": 1,
                "layer_indices": [0]
            }
        }));
        assert!(both.issues().iter().any(|issue| {
            issue.code == ValidationCode::InvalidGenerationValue
                && issue.path == "generation.speculative_decoding.strategies[mtp]"
        }));

        let depth_without_layers = build(serde_json::json!({
            "mtp": { "type": "native-mtp", "prediction_depth": 1 }
        }));
        assert!(
            depth_without_layers
                .issues()
                .iter()
                .any(|issue| issue.code == ValidationCode::InvalidGenerationValue
                    && issue.path.ends_with("strategies[mtp].layer_indices"))
        );

        let layers_without_depth = build(serde_json::json!({
            "mtp": { "type": "native-mtp", "layer_indices": [0] }
        }));
        assert!(
            layers_without_depth
                .issues()
                .iter()
                .any(|issue| issue.code == ValidationCode::InvalidGenerationValue
                    && issue.path.ends_with("strategies[mtp].prediction_depth"))
        );
    }

    #[test]
    fn rejects_proposer_kind_mismatches() {
        let build = |strategies: serde_json::Value, proposers: serde_json::Value| {
            let manifest = fixture();
            let mut encoded = serde_json::to_value(manifest).unwrap();
            encoded["generation"] = serde_json::json!({
                "speculative_decoding": {
                    "default": "mismatch",
                    "proposers": proposers,
                    "strategies": strategies
                }
            });
            let decoded: PackageManifest = serde_json::from_value(encoded).unwrap();
            let mut decoded = decoded;
            decoded.package_id = decoded.computed_package_id().unwrap();
            decoded.validate().unwrap_err()
        };
        let ngram_proposers = serde_json::json!({
            "cache-head": {
                "type": "ngram-cache",
                "ngram_min": 2,
                "ngram_max": 6,
                "max_proposal_tokens": 8,
                "history_scope": "session"
            },
            "suffix-head": {
                "type": "ngram-suffix",
                "ngram_min": 2,
                "ngram_max": 6,
                "max_proposal_tokens": 8,
                "history_scope": "session"
            }
        });
        let native_proposer = serde_json::json!({
            "mtp-head": { "type": "native-mtp", "prediction_depth": 1, "layer_indices": [0] }
        });

        let cases = [
            (
                serde_json::json!({
                    "mismatch": { "type": "native-mtp", "proposer": "cache-head" }
                }),
                &ngram_proposers,
                "strategies[mismatch].proposer",
            ),
            (
                serde_json::json!({
                    "mismatch": { "type": "ngram-cache", "proposer": "suffix-head" }
                }),
                &ngram_proposers,
                "strategies[mismatch].proposer",
            ),
            (
                serde_json::json!({
                    "mismatch": { "type": "ngram-suffix", "proposer": "cache-head" }
                }),
                &ngram_proposers,
                "strategies[mismatch].proposer",
            ),
            (
                serde_json::json!({
                    "mismatch": { "type": "composite", "primary": "cache-head", "extender": "mtp-head", "extension_policy": { "max_tokens": 4 } }
                }),
                &serde_json::json!({
                    "cache-head": { "type": "ngram-cache", "ngram_min": 2, "ngram_max": 6, "max_proposal_tokens": 8, "history_scope": "session" },
                    "mtp-head": { "type": "native-mtp", "prediction_depth": 1, "layer_indices": [0] }
                }),
                "strategies[mismatch].extender",
            ),
            (
                serde_json::json!({
                    "mismatch": { "type": "composite", "primary": "suffix-head", "extender": "cache-head", "extension_policy": { "max_tokens": 4 } }
                }),
                &ngram_proposers,
                "strategies[mismatch].primary",
            ),
        ];

        for (strategies, proposers, field) in cases {
            let error = build(strategies, proposers.clone());
            let issue = error
                .issues()
                .iter()
                .find(|issue| {
                    issue.code == ValidationCode::InvalidGenerationValue
                        && issue.path.ends_with(field)
                })
                .unwrap_or_else(|| {
                    panic!(
                        "expected an invalid-generation issue at {field}, got {:?}",
                        error.issues()
                    )
                });
            assert!(
                issue.message.contains("not compatible"),
                "unexpected message: {}",
                issue.message
            );
        }

        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "combined",
                "proposers": native_proposer,
                "strategies": {
                    "combined": {
                        "type": "composite",
                        "primary": "mtp-head",
                        "extender": "cache-head",
                        "extension_policy": { "max_tokens": 4 }
                    }
                }
            }
        });
        let mut cache_json = serde_json::json!({
            "cache-head": { "type": "ngram-cache", "ngram_min": 2, "ngram_max": 6, "max_proposal_tokens": 8, "history_scope": "session" }
        });
        encoded["generation"]["speculative_decoding"]["proposers"]["cache-head"] =
            cache_json["cache-head"].take();
        let decoded: PackageManifest = serde_json::from_value(encoded).unwrap();
        let mut decoded = decoded;
        decoded.package_id = decoded.computed_package_id().unwrap();
        decoded.validate().unwrap();
    }

    #[test]
    fn rejects_inline_native_mtp_zero_prediction_depth() {
        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "mtp",
                "strategies": {
                    "mtp": {
                        "type": "native-mtp",
                        "prediction_depth": 0,
                        "layer_indices": [0]
                    }
                }
            }
        });
        let decoded: PackageManifest = serde_json::from_value(encoded).unwrap();
        let mut decoded = decoded;
        decoded.package_id = decoded.computed_package_id().unwrap();

        let error = decoded.validate().unwrap_err();
        assert!(error.issues().iter().any(|issue| {
            issue.code == ValidationCode::InvalidGenerationValue
                && issue.path.ends_with("strategies[mtp].prediction_depth")
                && issue.message.contains("greater than zero")
        }));
    }

    #[test]
    fn rejects_invalid_extension_policy_values() {
        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "combined",
                "proposers": {
                    "mtp-head": { "type": "native-mtp", "prediction_depth": 1, "layer_indices": [0] },
                    "cache-head": { "type": "ngram-cache", "ngram_min": 2, "ngram_max": 6, "max_proposal_tokens": 8, "history_scope": "session" }
                },
                "strategies": {
                    "combined": {
                        "type": "composite",
                        "primary": "mtp-head",
                        "extender": "cache-head",
                        "extension_policy": {
                            "max_tokens": 4,
                            "initial_tokens": 8,
                            "tail_backoff_proposals": 0
                        }
                    }
                }
            }
        });
        let decoded: PackageManifest = serde_json::from_value(encoded).unwrap();
        let mut decoded = decoded;
        decoded.package_id = decoded.computed_package_id().unwrap();

        let error = decoded.validate().unwrap_err();
        let policy_issues: Vec<_> = error
            .issues()
            .iter()
            .filter(|issue| {
                issue.code == ValidationCode::InvalidGenerationValue
                    && issue.path.contains("extension_policy")
            })
            .collect();
        assert!(policy_issues.iter().any(|issue| {
            issue.path.ends_with("extension_policy.initial_tokens")
                && issue.message.contains("must not exceed")
        }));
        assert!(policy_issues.iter().any(|issue| {
            issue
                .path
                .ends_with("extension_policy.tail_backoff_proposals")
                && issue.message.contains("greater than zero")
        }));
    }

    #[test]
    fn rejects_unknown_extension_policy_field_at_parse() {
        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "combined",
                "proposers": {
                    "mtp-head": { "type": "native-mtp", "prediction_depth": 1, "layer_indices": [0] },
                    "cache-head": { "type": "ngram-cache", "ngram_min": 2, "ngram_max": 6, "max_proposal_tokens": 8, "history_scope": "session" }
                },
                "strategies": {
                    "combined": {
                        "type": "composite",
                        "primary": "mtp-head",
                        "extender": "cache-head",
                        "extension_policy": { "max_tokens": 4, "boost": 2 }
                    }
                }
            }
        });
        assert!(serde_json::from_value::<PackageManifest>(encoded).is_err());
    }

    #[test]
    fn rejects_inline_native_mtp_layer_index_out_of_range() {
        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "mtp",
                "strategies": {
                    "mtp": {
                        "type": "native-mtp",
                        "prediction_depth": 1,
                        "layer_indices": [0, 1]
                    }
                }
            }
        });
        let decoded: PackageManifest = serde_json::from_value(encoded).unwrap();
        let mut decoded = decoded;
        decoded.package_id = decoded.computed_package_id().unwrap();

        let error = decoded.validate().unwrap_err();
        assert!(
            error
                .issues()
                .iter()
                .any(|issue| issue.code == ValidationCode::LayerOutOfBounds
                    && issue.path.contains("strategies[mtp].layer_indices[1]"))
        );
    }

    #[test]
    fn rejects_invalid_window_policy() {
        let manifest = fixture();
        let mut encoded = serde_json::to_value(manifest).unwrap();
        encoded["generation"] = serde_json::json!({
            "speculative_decoding": {
                "default": "cache",
                "strategies": {
                    "cache": {
                        "type": "ngram-cache",
                        "proposer": "p",
                        "window_policy": {
                            "default": "",
                            "initial_window": 8,
                            "min_window": 4,
                            "max_window": 4,
                            "pipeline_depth": 0
                        }
                    }
                }
            }
        });
        let decoded: PackageManifest = serde_json::from_value(encoded).unwrap();
        let mut decoded = decoded;
        decoded.package_id = decoded.computed_package_id().unwrap();

        let error = decoded.validate().unwrap_err();
        let window_issues: Vec<_> = error
            .issues()
            .iter()
            .filter(|issue| issue.path.contains("window_policy"))
            .collect();
        assert_eq!(window_issues.len(), 3);
        assert!(window_issues.iter().all(|issue| {
            issue.code == ValidationCode::InvalidGenerationValue
                && issue
                    .path
                    .starts_with("generation.speculative_decoding.strategies[cache]")
        }));
    }

    #[test]
    fn rejects_default_not_declared_as_strategy() {
        let mut manifest = fixture();
        manifest.generation = Some(Generation {
            speculative_decoding: Some(SpeculativeDecoding {
                default: "missing".to_string(),
                proposers: BTreeMap::new(),
                strategies: BTreeMap::new(),
            }),
        });
        manifest.package_id = manifest.computed_package_id().unwrap();

        let error = manifest.validate().unwrap_err();
        assert!(
            error
                .issues()
                .iter()
                .any(|issue| issue.code == ValidationCode::UnknownStrategy)
        );
    }

    #[test]
    fn rejects_strategy_referencing_missing_proposer() {
        let mut manifest = fixture();
        manifest.generation = Some(Generation {
            speculative_decoding: Some(SpeculativeDecoding {
                default: "cache".to_string(),
                proposers: BTreeMap::new(),
                strategies: BTreeMap::from([(
                    "cache".to_string(),
                    StrategySpec {
                        kind: StrategyKind::NgramCache {
                            proposer: "ghost".to_string(),
                            window_policy: None,
                        },
                    },
                )]),
            }),
        });
        manifest.package_id = manifest.computed_package_id().unwrap();

        let error = manifest.validate().unwrap_err();
        assert!(
            error
                .issues()
                .iter()
                .any(|issue| issue.code == ValidationCode::UnknownProposer)
        );
    }

    #[test]
    fn rejects_generation_layer_index_out_of_range() {
        let mut manifest = fixture();
        manifest.generation = Some(Generation {
            speculative_decoding: Some(SpeculativeDecoding {
                default: "mtp".to_string(),
                proposers: BTreeMap::from([(
                    "mtp-head".to_string(),
                    ProposerSpec {
                        kind: ProposerKind::NativeMtp {
                            prediction_depth: 1,
                            layer_indices: vec![0, 1],
                        },
                    },
                )]),
                strategies: BTreeMap::from([(
                    "mtp".to_string(),
                    StrategySpec {
                        kind: StrategyKind::NativeMtp {
                            proposer: Some("mtp-head".to_string()),
                            prediction_depth: None,
                            layer_indices: Vec::new(),
                            window_policy: None,
                        },
                    },
                )]),
            }),
        });
        manifest.package_id = manifest.computed_package_id().unwrap();

        let error = manifest.validate().unwrap_err();
        assert!(
            error
                .issues()
                .iter()
                .any(|issue| issue.code == ValidationCode::LayerOutOfBounds)
        );
    }

    #[test]
    fn rejects_invalid_generation_values() {
        let mut manifest = fixture();
        manifest.generation = Some(Generation {
            speculative_decoding: Some(SpeculativeDecoding {
                default: "combined".to_string(),
                proposers: BTreeMap::from([
                    (
                        "mtp-head".to_string(),
                        ProposerSpec {
                            kind: ProposerKind::NativeMtp {
                                prediction_depth: 0,
                                layer_indices: Vec::new(),
                            },
                        },
                    ),
                    (
                        "suffix".to_string(),
                        ProposerSpec {
                            kind: ProposerKind::NgramSuffix {
                                ngram_min: 0,
                                ngram_max: 0,
                                max_proposal_tokens: 0,
                                history_scope: String::new(),
                            },
                        },
                    ),
                ]),
                strategies: BTreeMap::from([(
                    "combined".to_string(),
                    StrategySpec {
                        kind: StrategyKind::Composite {
                            primary: "mtp-head".to_string(),
                            extender: "suffix".to_string(),
                            extension_policy: ExtensionPolicy {
                                max_tokens: 0,
                                initial_tokens: None,
                                tail_backoff_proposals: None,
                            },
                            window_policy: None,
                        },
                    },
                )]),
            }),
        });
        manifest.package_id = manifest.computed_package_id().unwrap();

        let error = manifest.validate().unwrap_err();
        let messages = error
            .issues()
            .iter()
            .filter(|issue| issue.code == ValidationCode::InvalidGenerationValue)
            .count();
        assert!(
            messages >= 6,
            "expected the zero/empty value rules to fire, got {messages}"
        );
    }

    fn fixture() -> PackageManifest {
        let mut manifest = PackageManifest {
            schema_version: PACKAGE_SCHEMA_VERSION,
            package_id: String::new(),
            model_id: "fixture/model".to_string(),
            source_model: SourceModel {
                sha256: DIGEST.to_string(),
                repo: Some("fixture/model".to_string()),
                revision: Some("revision".to_string()),
                primary_file: Some("model.gguf".to_string()),
                canonical_ref: Some("hf://fixture/model@revision/model.gguf".to_string()),
                distribution_id: Some("fixture-model".to_string()),
                files: vec![SourceFile {
                    path: "model.gguf".to_string(),
                    byte_size: 256,
                    sha256: DIGEST.to_string(),
                }],
            },
            format: "gguf".to_string(),
            layer_count: 1,
            model_metadata: BTreeMap::from([(
                "general.architecture".to_string(),
                Value::String("llama".to_string()),
            )]),
            artifact_catalog: ArtifactCatalog {
                entries: vec![Artifact {
                    id: "layer-0".to_string(),
                    path: "layers/layer-00000.gguf".to_string(),
                    byte_size: 256,
                    sha256: DIGEST.to_string(),
                }],
            },
            tensor_catalog: TensorCatalog {
                entries: vec![Tensor {
                    id: "tensor-1".to_string(),
                    name: "blk.0.attn_q.weight".to_string(),
                    ggml_type: 8,
                    dimensions: vec![16, 16],
                    layer_ordinal: Some(0),
                    storage: TensorStorage::Owned {
                        artifact_id: "layer-0".to_string(),
                        data_offset: 64,
                        stored_length: 64,
                        alignment: 32,
                        integrity: TensorIntegrity::ArtifactSha256,
                    },
                }],
            },
            sidecars: Vec::new(),
            generation: None,
            native_abi_version: "7".to_string(),
            generator_version: "0.76.0-rc9".to_string(),
            created_at_unix_secs: 0,
        };
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest
    }
}
