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
    pub native_abi_version: String,
    pub generator_version: String,
    pub created_at_unix_secs: u64,
}

impl PackageManifest {
    pub fn validate(&self) -> Result<(), ValidationErrors> {
        let mut issues = Vec::new();
        validate_manifest_fields(self, &mut issues);

        let artifacts = collect_artifacts(&self.artifact_catalog.entries, &mut issues);
        let tensors = collect_tensors(self, &mut issues);

        validate_owned_tensor_storage(&self.tensor_catalog.entries, &artifacts, &mut issues);
        validate_aliases(&self.tensor_catalog.entries, &tensors, &mut issues);
        validate_sidecars(&self.sidecars, &artifacts, &mut issues);

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
    OverlappingStorage,
    UnknownAliasTarget,
    AliasTargetIsAlias,
    AliasMetadataMismatch,
    PackageIdentityMismatch,
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
        if tensor.dimensions.contains(&0) {
            push_issue(
                issues,
                ValidationCode::InvalidDimension,
                format!("{prefix}.dimensions"),
                "tensor dimensions must be non-zero",
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
            native_abi_version: "7".to_string(),
            generator_version: "0.76.0-rc9".to_string(),
            created_at_unix_secs: 0,
        };
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest
    }
}
