use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::Serialize;
use skippy_package_format::{
    Artifact, Generation, PACKAGE_SCHEMA_VERSION, PackageManifest, TensorStorage,
};

use crate::hash::file_sha256;

mod artifact_io;

use artifact_io::{safe_relative_path, sha256_bytes};

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PackagePreflightOptions {
    pub stages: Option<usize>,
    pub verify_sha256: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct PackagePreflightReport {
    pub schema_version: u32,
    pub package_path: String,
    pub valid: bool,
    pub package_id: Option<String>,
    pub model_id: Option<String>,
    pub layer_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<Generation>,
    pub manifest_sha256: Option<String>,
    pub checked_artifact_count: usize,
    pub missing_artifact_count: usize,
    pub issue_count: usize,
    pub issues: Vec<PreflightIssue>,
    pub artifacts: Vec<PreflightArtifact>,
    pub stages: Vec<PreflightStage>,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PreflightSeverity {
    Error,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub(crate) struct PreflightIssue {
    pub severity: PreflightSeverity,
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub remediation: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PreflightArtifact {
    pub id: String,
    pub role: String,
    pub path: String,
    pub present: bool,
    pub declared_artifact_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_artifact_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_matches_manifest: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256_matches_manifest: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PreflightStage {
    pub stage_index: usize,
    pub layer_start: u32,
    pub layer_end: u32,
    pub includes_embeddings: bool,
    pub includes_output: bool,
    pub selection_basis: &'static str,
    pub part_count: usize,
    pub artifact_bytes: u64,
    pub parts: Vec<String>,
    pub missing_parts: Vec<String>,
}

pub(crate) fn preflight_package(
    package: &Path,
    options: &PackagePreflightOptions,
) -> PackagePreflightReport {
    let mut report = PackagePreflightReport::new(package);
    let manifest_contents = match fs::read(package.join("model-package.json")) {
        Ok(contents) => contents,
        Err(error) => {
            report.error(
                "missing_manifest",
                format!("cannot read package manifest: {error}"),
                Some("model-package.json".to_string()),
                "ensure the package directory contains model-package.json",
            );
            return report.finalize();
        }
    };
    report.manifest_sha256 = Some(sha256_bytes(&manifest_contents));

    let envelope = match serde_json::from_slice::<serde_json::Value>(&manifest_contents) {
        Ok(envelope) => envelope,
        Err(error) => {
            report.error(
                "invalid_manifest_json",
                format!("cannot parse package manifest JSON: {error}"),
                Some("model-package.json".to_string()),
                "rebuild the package with skippy-model-package write-package",
            );
            return report.finalize();
        }
    };
    let schema_version = envelope
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    if let Some(schema_version) = schema_version {
        report.schema_version = schema_version;
    }
    if schema_version != Some(PACKAGE_SCHEMA_VERSION) {
        report.error(
            "unsupported_schema_version",
            format!(
                "split package preflight requires schema version {PACKAGE_SCHEMA_VERSION}; found {}",
                schema_version
                    .map(|version| version.to_string())
                    .unwrap_or_else(|| "a missing or non-integer value".to_string())
            ),
            Some("model-package.json".to_string()),
            "rebuild the package with the current skippy-model-package write-package command",
        );
        return report.finalize();
    }

    let manifest = match serde_json::from_value::<PackageManifest>(envelope) {
        Ok(manifest) => manifest,
        Err(error) => {
            report.error(
                "invalid_manifest_shape",
                format!("cannot parse package-v2 manifest: {error}"),
                Some("model-package.json".to_string()),
                "rebuild the package with the current skippy-model-package write-package command",
            );
            return report.finalize();
        }
    };

    report.package_id = Some(manifest.package_id.clone());
    report.model_id = Some(manifest.model_id.clone());
    report.layer_count = Some(manifest.layer_count);
    report.generation = manifest.generation.clone();
    if let Err(errors) = manifest.validate() {
        for issue in errors.issues() {
            let code = serde_json::to_value(issue.code)
                .ok()
                .and_then(|value| value.as_str().map(str::to_owned))
                .unwrap_or_else(|| "invalid_manifest".to_string());
            report.error(
                code,
                issue.message.clone(),
                Some(format!("model-package.json: {}", issue.path)),
                "rebuild the package from an independent, complete source model",
            );
        }
    }

    validate_artifacts(package, &manifest, options.verify_sha256, &mut report);
    build_stage_reports(&manifest, options.stages, &mut report);
    report.finalize()
}

impl PackagePreflightReport {
    fn new(package: &Path) -> Self {
        Self {
            schema_version: 0,
            package_path: package.display().to_string(),
            valid: true,
            package_id: None,
            model_id: None,
            layer_count: None,
            generation: None,
            manifest_sha256: None,
            checked_artifact_count: 0,
            missing_artifact_count: 0,
            issue_count: 0,
            issues: Vec::new(),
            artifacts: Vec::new(),
            stages: Vec::new(),
        }
    }

    fn error(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
        path: Option<String>,
        remediation: impl Into<String>,
    ) {
        self.issues.push(PreflightIssue {
            severity: PreflightSeverity::Error,
            code: code.into(),
            message: message.into(),
            path,
            remediation: remediation.into(),
        });
    }

    fn finalize(mut self) -> Self {
        self.issue_count = self.issues.len();
        self.checked_artifact_count = self.artifacts.len();
        self.missing_artifact_count = self
            .artifacts
            .iter()
            .filter(|artifact| !artifact.present)
            .count();
        self.valid = self.issues.is_empty();
        self
    }
}

fn validate_artifacts(
    package: &Path,
    manifest: &PackageManifest,
    verify_sha256: bool,
    report: &mut PackagePreflightReport,
) {
    for artifact in &manifest.artifact_catalog.entries {
        let role = artifact_role(manifest, artifact);
        let relative = match safe_relative_path(&artifact.path) {
            Ok(path) => path,
            Err(message) => {
                report.error(
                    "unsafe_artifact_path",
                    format!("artifact {} path is unsafe: {message}", artifact.id),
                    Some(artifact.path.clone()),
                    "rebuild the package so artifact paths stay inside the package directory",
                );
                report
                    .artifacts
                    .push(artifact_report(artifact, role, false, None, None, None));
                continue;
            }
        };
        let path = package.join(relative);
        let metadata = match fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => {
                report.error(
                    "artifact_not_file",
                    format!("package artifact {} is not a regular file", artifact.path),
                    Some(artifact.path.clone()),
                    "replace the artifact with the immutable file declared by the manifest",
                );
                report
                    .artifacts
                    .push(artifact_report(artifact, role, false, None, None, None));
                continue;
            }
            Err(error) => {
                report.error(
                    "missing_artifact",
                    format!("package artifact {} is missing: {error}", artifact.path),
                    Some(artifact.path.clone()),
                    "download or rebuild the package artifact before split serving",
                );
                report
                    .artifacts
                    .push(artifact_report(artifact, role, false, None, None, None));
                continue;
            }
        };
        let actual_bytes = metadata.len();
        let size_matches = actual_bytes == artifact.byte_size;
        if !size_matches {
            report.error(
                "artifact_size_mismatch",
                format!(
                    "package artifact {} has {actual_bytes} bytes; manifest declares {}",
                    artifact.path, artifact.byte_size
                ),
                Some(artifact.path.clone()),
                "redownload or rebuild the package artifact",
            );
        }
        let sha_matches = verify_sha256.then(|| match file_sha256(&path) {
            Ok(actual) if actual.eq_ignore_ascii_case(&artifact.sha256) => true,
            Ok(actual) => {
                report.error(
                    "artifact_sha256_mismatch",
                    format!(
                        "package artifact {} checksum mismatch: expected {}, got {actual}",
                        artifact.path, artifact.sha256
                    ),
                    Some(artifact.path.clone()),
                    "redownload or rebuild the package artifact",
                );
                false
            }
            Err(error) => {
                report.error(
                    "artifact_sha256_unreadable",
                    format!("cannot hash package artifact {}: {error}", artifact.path),
                    Some(artifact.path.clone()),
                    "ensure the artifact is readable before checksum verification",
                );
                false
            }
        });
        report.artifacts.push(artifact_report(
            artifact,
            role,
            true,
            Some(actual_bytes),
            Some(size_matches),
            sha_matches,
        ));
    }
}

fn artifact_role(manifest: &PackageManifest, artifact: &Artifact) -> String {
    if manifest.source_model.metadata_artifact_id == artifact.id {
        "model_metadata".to_string()
    } else if let Some(sidecar) = manifest
        .sidecars
        .iter()
        .find(|sidecar| sidecar.artifact_id == artifact.id)
    {
        format!("sidecar:{:?}", sidecar.kind).to_ascii_lowercase()
    } else {
        "model".to_string()
    }
}

fn artifact_report(
    artifact: &Artifact,
    role: String,
    present: bool,
    actual_artifact_bytes: Option<u64>,
    size_matches_manifest: Option<bool>,
    sha256_matches_manifest: Option<bool>,
) -> PreflightArtifact {
    PreflightArtifact {
        id: artifact.id.clone(),
        role,
        path: artifact.path.clone(),
        present,
        declared_artifact_bytes: artifact.byte_size,
        actual_artifact_bytes,
        size_matches_manifest,
        sha256_matches_manifest,
    }
}

fn build_stage_reports(
    manifest: &PackageManifest,
    stages: Option<usize>,
    report: &mut PackagePreflightReport,
) {
    let Some(stage_count) = stages else {
        return;
    };
    if stage_count == 0 {
        report.error(
            "invalid_stage_count",
            "--stages must be greater than zero",
            Some("model-package.json".to_string()),
            "choose a positive diagnostic stage count",
        );
        return;
    }
    let Ok(stage_count_u32) = u32::try_from(stage_count) else {
        report.error(
            "stage_count_exceeds_layer_count",
            format!(
                "--stages {stage_count} exceeds package layer_count {}",
                manifest.layer_count
            ),
            Some("model-package.json".to_string()),
            "use at most one diagnostic stage per transformer layer",
        );
        return;
    };
    if stage_count_u32 > manifest.layer_count {
        report.error(
            "stage_count_exceeds_layer_count",
            format!(
                "--stages {stage_count} exceeds package layer_count {}",
                manifest.layer_count
            ),
            Some("model-package.json".to_string()),
            "use at most one diagnostic stage per transformer layer",
        );
        return;
    }

    let artifacts = manifest
        .artifact_catalog
        .entries
        .iter()
        .map(|artifact| (artifact.id.as_str(), artifact))
        .collect::<BTreeMap<_, _>>();
    let tensors = manifest
        .tensor_catalog
        .entries
        .iter()
        .map(|tensor| (tensor.id.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let presence = report
        .artifacts
        .iter()
        .map(|artifact| (artifact.id.as_str(), artifact.present))
        .collect::<BTreeMap<_, _>>();

    for (stage_index, (layer_start, layer_end)) in
        partition_layers(manifest.layer_count, stage_count_u32)
            .into_iter()
            .enumerate()
    {
        let mut artifact_ids = BTreeSet::new();
        for tensor in &manifest.tensor_catalog.entries {
            if tensor
                .layer_ordinal
                .is_some_and(|layer| layer < layer_start || layer >= layer_end)
            {
                continue;
            }
            if let Some(artifact_id) = tensor_artifact_id(tensor, &tensors) {
                artifact_ids.insert(artifact_id);
            }
        }
        let parts = artifact_ids
            .iter()
            .filter_map(|id| artifacts.get(id).map(|artifact| artifact.path.clone()))
            .collect::<Vec<_>>();
        let missing_parts = artifact_ids
            .iter()
            .filter(|id| !presence.get(*id).copied().unwrap_or(false))
            .filter_map(|id| artifacts.get(id).map(|artifact| artifact.path.clone()))
            .collect::<Vec<_>>();
        let artifact_bytes = artifact_ids
            .iter()
            .filter_map(|id| artifacts.get(id))
            .map(|artifact| artifact.byte_size)
            .sum();
        report.stages.push(PreflightStage {
            stage_index,
            layer_start,
            layer_end,
            includes_embeddings: stage_index == 0,
            includes_output: stage_index + 1 == stage_count,
            selection_basis: "diagnostic_layer_ordinals; executable closure requires graph admission",
            part_count: parts.len(),
            artifact_bytes,
            parts,
            missing_parts,
        });
    }
}

fn tensor_artifact_id<'a>(
    tensor: &'a skippy_package_format::Tensor,
    tensors: &'a BTreeMap<&str, &'a skippy_package_format::Tensor>,
) -> Option<&'a str> {
    match &tensor.storage {
        TensorStorage::Owned { artifact_id, .. } => Some(artifact_id),
        TensorStorage::Alias { target_tensor_id } => tensors
            .get(target_tensor_id.as_str())
            .and_then(|target| match &target.storage {
                TensorStorage::Owned { artifact_id, .. } => Some(artifact_id.as_str()),
                TensorStorage::Alias { .. } => None,
            }),
    }
}

fn partition_layers(layer_count: u32, stages: u32) -> Vec<(u32, u32)> {
    (0..stages)
        .map(|stage| {
            let layer_count = u64::from(layer_count);
            let stages = u64::from(stages);
            let stage = u64::from(stage);
            let start = (layer_count * stage / stages) as u32;
            let end = (layer_count * (stage + 1) / stages) as u32;
            (start, end)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::package::ArtifactHook;
    use crate::package_v2::write_package;
    use crate::test_gguf::{explicit, fixture, tensor};

    fn write_v2_fixture(root: &Path) -> std::path::PathBuf {
        let source = root.join("model.gguf");
        fixture(
            &source,
            &[
                tensor("token_embd.weight", 0),
                tensor("blk.0.attn_q.weight", 32),
            ],
            None,
        );
        let package = root.join("package");
        write_package(
            source.display().to_string(),
            package.clone(),
            Vec::new(),
            ArtifactHook { command: None },
            ArtifactHook { command: None },
            explicit(&source),
            false,
        )
        .unwrap();
        package
    }

    #[test]
    fn writer_output_passes_preflight_with_diagnostic_stages() {
        let temp = tempfile::tempdir().unwrap();
        let package = write_v2_fixture(temp.path());
        let report = preflight_package(
            &package,
            &PackagePreflightOptions {
                stages: Some(2),
                verify_sha256: true,
            },
        );
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.schema_version, PACKAGE_SCHEMA_VERSION);
        assert_eq!(report.checked_artifact_count, 1);
        assert_eq!(report.stages.len(), 2);
        assert!(report.stages.iter().all(|stage| !stage.parts.is_empty()));
    }

    #[test]
    fn missing_and_corrupt_artifacts_fail_preflight() {
        let temp = tempfile::tempdir().unwrap();
        let package = write_v2_fixture(temp.path());
        let manifest: PackageManifest =
            serde_json::from_slice(&fs::read(package.join("model-package.json")).unwrap()).unwrap();
        let artifact = package.join(&manifest.artifact_catalog.entries[0].path);
        fs::write(&artifact, b"corrupt").unwrap();
        let report = preflight_package(
            &package,
            &PackagePreflightOptions {
                verify_sha256: true,
                ..PackagePreflightOptions::default()
            },
        );
        assert!(!report.valid);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.code == "artifact_size_mismatch")
        );
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.code == "artifact_sha256_mismatch")
        );
        fs::remove_file(artifact).unwrap();
        let report = preflight_package(&package, &PackagePreflightOptions::default());
        assert!(!report.valid);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.code == "missing_artifact")
        );
    }

    #[test]
    fn incompatible_schema_fails_before_legacy_shape_parsing() {
        let temp = tempfile::tempdir().unwrap();
        let package = temp.path().join("package");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("model-package.json"),
            br#"{"schema_version":1,"format":"layer-package"}"#,
        )
        .unwrap();
        let report = preflight_package(&package, &PackagePreflightOptions::default());
        assert!(!report.valid);
        assert_eq!(report.issues[0].code, "unsupported_schema_version");
    }

    #[test]
    fn malformed_v2_manifest_uses_shared_schema() {
        let temp = tempfile::tempdir().unwrap();
        let package = temp.path().join("package");
        fs::create_dir_all(&package).unwrap();
        fs::write(
            package.join("model-package.json"),
            br#"{"schema_version":2,"unexpected":true}"#,
        )
        .unwrap();
        let report = preflight_package(&package, &PackagePreflightOptions::default());
        assert!(!report.valid);
        assert_eq!(report.issues[0].code, "invalid_manifest_shape");
    }

    #[test]
    fn invalid_stage_counts_fail() {
        let temp = tempfile::tempdir().unwrap();
        let package = write_v2_fixture(temp.path());
        for stages in [0, 3, usize::MAX] {
            let report = preflight_package(
                &package,
                &PackagePreflightOptions {
                    stages: Some(stages),
                    ..PackagePreflightOptions::default()
                },
            );
            assert!(!report.valid);
        }
    }
}
