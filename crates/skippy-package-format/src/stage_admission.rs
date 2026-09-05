use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::materialization::{MaterializationBindingError, MaterializationTensor};
use crate::{Artifact, PackageManifest, Sidecar, SidecarKind, ValidationErrors};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageAdmissionDescriptor {
    pub package_id: String,
    pub resident_tensor_ids: Vec<String>,
    pub sidecars: Vec<Sidecar>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSidecar<'a> {
    pub kind: SidecarKind,
    pub artifact: &'a Artifact,
    pub name: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedStageAdmission<'a> {
    pub package_id: &'a str,
    pub metadata_artifact: &'a Artifact,
    pub tensor_bindings: Vec<MaterializationTensor<'a>>,
    pub required_artifacts: Vec<&'a Artifact>,
    pub sidecars: Vec<ResolvedSidecar<'a>>,
    pub distinct_owned_stored_bytes: u64,
}

#[derive(Debug)]
pub enum StageAdmissionError {
    InvalidManifest(ValidationErrors),
    PackageIdentityComputation(serde_json::Error),
    PackageIdMismatch {
        expected: String,
        actual: String,
    },
    TensorBinding(MaterializationBindingError),
    SidecarsNotStrictlySorted {
        index: usize,
        previous: Sidecar,
        current: Sidecar,
    },
    UnknownSidecar {
        sidecar: Sidecar,
    },
    MissingArtifact {
        artifact_id: String,
    },
    StoredLengthOverflow,
}

impl fmt::Display for StageAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidManifest(error) => write!(formatter, "{error}"),
            Self::PackageIdentityComputation(error) => {
                write!(formatter, "cannot compute package identity: {error}")
            }
            Self::PackageIdMismatch { expected, actual } => write!(
                formatter,
                "stage admission package id {actual:?} does not match manifest package id {expected:?}"
            ),
            Self::TensorBinding(error) => write!(formatter, "{error}"),
            Self::SidecarsNotStrictlySorted {
                index,
                previous,
                current,
            } => write!(
                formatter,
                "stage admission sidecars are not strictly sorted at index {index}: {previous:?} then {current:?}"
            ),
            Self::UnknownSidecar { sidecar } => {
                write!(formatter, "unknown stage admission sidecar {sidecar:?}")
            }
            Self::MissingArtifact { artifact_id } => {
                write!(formatter, "missing required artifact {artifact_id:?}")
            }
            Self::StoredLengthOverflow => {
                write!(formatter, "distinct owned tensor stored length exceeds u64")
            }
        }
    }
}

impl Error for StageAdmissionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidManifest(error) => Some(error),
            Self::PackageIdentityComputation(error) => Some(error),
            Self::TensorBinding(error) => Some(error),
            _ => None,
        }
    }
}

impl PackageManifest {
    /// Resolve exact package identities into immutable admission inputs.
    ///
    /// Tensor names, layer ordinals, roles, and sidecar kinds are never
    /// alternate selectors. Every requested tensor ID and sidecar reference
    /// must be strictly sorted, unique, and present in this manifest.
    pub fn resolve_stage_admission<'a>(
        &'a self,
        descriptor: &StageAdmissionDescriptor,
    ) -> Result<ResolvedStageAdmission<'a>, StageAdmissionError> {
        self.validate()
            .map_err(StageAdmissionError::InvalidManifest)?;
        let package_id = self
            .computed_package_id()
            .map_err(StageAdmissionError::PackageIdentityComputation)?;
        if descriptor.package_id != package_id {
            return Err(StageAdmissionError::PackageIdMismatch {
                expected: package_id,
                actual: descriptor.package_id.clone(),
            });
        }
        validate_requested_sidecars(&descriptor.sidecars)?;

        let tensor_bindings = self
            .materialization_tensors(&descriptor.resident_tensor_ids)
            .map_err(StageAdmissionError::TensorBinding)?;
        let artifacts = self
            .artifact_catalog
            .entries
            .iter()
            .map(|artifact| (artifact.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();
        let metadata_artifact =
            required_artifact(&artifacts, self.source_model.metadata_artifact_id.as_str())?;

        let mut required = BTreeMap::from([(metadata_artifact.id.as_str(), metadata_artifact)]);
        for binding in &tensor_bindings {
            required.insert(binding.artifact.id.as_str(), binding.artifact);
        }

        let mut resolved_sidecars = Vec::with_capacity(descriptor.sidecars.len());
        for requested in &descriptor.sidecars {
            let sidecar = self
                .sidecars
                .iter()
                .find(|sidecar| *sidecar == requested)
                .ok_or_else(|| StageAdmissionError::UnknownSidecar {
                    sidecar: requested.clone(),
                })?;
            let artifact = required_artifact(&artifacts, sidecar.artifact_id.as_str())?;
            required.insert(artifact.id.as_str(), artifact);
            resolved_sidecars.push(ResolvedSidecar {
                kind: sidecar.kind,
                artifact,
                name: sidecar.name.as_deref(),
            });
        }

        let distinct_owned_stored_bytes = checked_distinct_stored_length(&tensor_bindings)?;

        Ok(ResolvedStageAdmission {
            package_id: &self.package_id,
            metadata_artifact,
            tensor_bindings,
            required_artifacts: required.into_values().collect(),
            sidecars: resolved_sidecars,
            distinct_owned_stored_bytes,
        })
    }
}

fn validate_requested_sidecars(sidecars: &[Sidecar]) -> Result<(), StageAdmissionError> {
    for (index, pair) in sidecars.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(StageAdmissionError::SidecarsNotStrictlySorted {
                index: index + 1,
                previous: pair[0].clone(),
                current: pair[1].clone(),
            });
        }
    }
    Ok(())
}

fn required_artifact<'a>(
    artifacts: &BTreeMap<&str, &'a Artifact>,
    artifact_id: &str,
) -> Result<&'a Artifact, StageAdmissionError> {
    artifacts
        .get(artifact_id)
        .copied()
        .ok_or_else(|| StageAdmissionError::MissingArtifact {
            artifact_id: artifact_id.to_string(),
        })
}

fn checked_distinct_stored_length(
    bindings: &[MaterializationTensor<'_>],
) -> Result<u64, StageAdmissionError> {
    let mut storage = BTreeSet::new();
    let mut total = 0_u64;
    for binding in bindings {
        if storage.insert((
            binding.artifact.id.as_str(),
            binding.data_offset,
            binding.stored_length,
        )) {
            total = total
                .checked_add(binding.stored_length)
                .ok_or(StageAdmissionError::StoredLengthOverflow)?;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::Value;

    use super::*;
    use crate::materialization::MaterializationBindingError;
    use crate::{
        ArtifactCatalog, PACKAGE_SCHEMA_VERSION, SourceFile, SourceModel, Tensor, TensorCatalog,
        TensorIntegrity, TensorStorage,
    };

    const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn resolves_exact_artifact_closure_and_distinct_owned_bytes() {
        let manifest = fixture();
        let descriptor = descriptor(&manifest);
        let resolved = manifest.resolve_stage_admission(&descriptor).unwrap();

        assert_eq!(resolved.package_id, manifest.package_id);
        assert_eq!(resolved.metadata_artifact.id, "weights-a");
        assert_eq!(
            resolved
                .tensor_bindings
                .iter()
                .map(|binding| binding.tensor_id)
                .collect::<Vec<_>>(),
            ["tensor-a", "tensor-b"]
        );
        assert_eq!(
            resolved
                .required_artifacts
                .iter()
                .map(|artifact| artifact.id.as_str())
                .collect::<Vec<_>>(),
            ["mmproj-a", "mmproj-b", "weights-a", "weights-b"]
        );
        assert_eq!(resolved.distinct_owned_stored_bytes, 96);
        assert_eq!(
            resolved
                .sidecars
                .iter()
                .map(|sidecar| (sidecar.kind, sidecar.name, sidecar.artifact.id.as_str()))
                .collect::<Vec<_>>(),
            [
                (SidecarKind::Mmproj, None, "mmproj-a"),
                (SidecarKind::Mmproj, Some("vision"), "mmproj-b"),
            ]
        );
    }

    #[test]
    fn rejects_malformed_and_unknown_sidecar_kinds_at_parse() {
        let manifest = fixture();
        for invalid in [Value::String("future".to_string()), Value::from(7)] {
            let mut encoded = serde_json::to_value(&manifest).unwrap();
            encoded["sidecars"][0]["kind"] = invalid;
            assert!(serde_json::from_value::<PackageManifest>(encoded).is_err());
        }
    }

    #[test]
    fn rejects_package_id_mismatch() {
        let manifest = fixture();
        let mut descriptor = descriptor(&manifest);
        descriptor.package_id = format!("sha256:{}", "1".repeat(64));

        assert!(matches!(
            manifest.resolve_stage_admission(&descriptor),
            Err(StageAdmissionError::PackageIdMismatch { expected, actual })
                if expected == manifest.package_id && actual == descriptor.package_id
        ));
    }

    #[test]
    fn sidecar_references_must_be_sorted_unique_and_exact() {
        let manifest = fixture();
        let mut reversed = descriptor(&manifest);
        reversed.sidecars.reverse();
        assert!(matches!(
            manifest.resolve_stage_admission(&reversed),
            Err(StageAdmissionError::SidecarsNotStrictlySorted { .. })
        ));

        let mut duplicate = descriptor(&manifest);
        duplicate.sidecars[1] = duplicate.sidecars[0].clone();
        assert!(matches!(
            manifest.resolve_stage_admission(&duplicate),
            Err(StageAdmissionError::SidecarsNotStrictlySorted { .. })
        ));

        let mut unknown = descriptor(&manifest);
        unknown.sidecars = vec![Sidecar {
            kind: SidecarKind::Mmproj,
            artifact_id: "unused".to_string(),
            name: None,
        }];
        assert!(matches!(
            manifest.resolve_stage_admission(&unknown),
            Err(StageAdmissionError::UnknownSidecar { .. })
        ));
    }

    #[test]
    fn ordinary_sidecar_sort_matches_admission_order() {
        let mut manifest = fixture();
        manifest.sidecars = vec![
            Sidecar {
                kind: SidecarKind::Mmproj,
                artifact_id: "mmproj-b".to_string(),
                name: None,
            },
            Sidecar {
                kind: SidecarKind::Mmproj,
                artifact_id: "mmproj-a".to_string(),
                name: Some("vision".to_string()),
            },
        ];
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest.validate().unwrap();

        let mut descriptor = descriptor(&manifest);
        descriptor.sidecars.reverse();
        assert!(matches!(
            manifest.resolve_stage_admission(&descriptor),
            Err(StageAdmissionError::SidecarsNotStrictlySorted { .. })
        ));

        descriptor.sidecars.sort();
        manifest.resolve_stage_admission(&descriptor).unwrap();
        assert_eq!(descriptor.sidecars, manifest.sidecars);

        let mut tied_names = [
            Sidecar {
                kind: SidecarKind::Mmproj,
                artifact_id: "mmproj-b".to_string(),
                name: Some("vision".to_string()),
            },
            Sidecar {
                kind: SidecarKind::Mmproj,
                artifact_id: "mmproj-a".to_string(),
                name: Some("vision".to_string()),
            },
        ];
        tied_names.sort();
        assert_eq!(tied_names[0].artifact_id, "mmproj-a");
        assert_eq!(tied_names[1].artifact_id, "mmproj-b");

        assert_eq!(
            serde_json::to_string(&tied_names[0]).unwrap(),
            r#"{"kind":"mmproj","artifact_id":"mmproj-a","name":"vision"}"#
        );
    }

    #[test]
    fn manifest_sidecar_declarations_must_be_unique() {
        let mut manifest = fixture();
        manifest.sidecars.push(manifest.sidecars[0].clone());
        manifest.package_id = manifest.computed_package_id().unwrap();

        assert!(
            manifest
                .validate()
                .unwrap_err()
                .issues()
                .iter()
                .any(|issue| issue.code == crate::ValidationCode::DuplicateSidecar)
        );

        let mut conflicting_mapping = fixture();
        conflicting_mapping.sidecars.push(Sidecar {
            kind: SidecarKind::Mmproj,
            artifact_id: "unused".to_string(),
            name: None,
        });
        conflicting_mapping.package_id = conflicting_mapping.computed_package_id().unwrap();
        assert!(
            conflicting_mapping
                .validate()
                .unwrap_err()
                .issues()
                .iter()
                .any(|issue| issue.code == crate::ValidationCode::DuplicateSidecar)
        );
    }

    #[test]
    fn duplicate_storage_is_accounted_once_but_duplicate_ids_are_rejected() {
        let manifest = fixture();
        let descriptor = descriptor(&manifest);
        let resolved = manifest.resolve_stage_admission(&descriptor).unwrap();
        let duplicate_bindings = vec![
            resolved.tensor_bindings[0].clone(),
            resolved.tensor_bindings[0].clone(),
            resolved.tensor_bindings[1].clone(),
        ];
        assert_eq!(
            checked_distinct_stored_length(&duplicate_bindings).unwrap(),
            resolved.distinct_owned_stored_bytes
        );

        let mut duplicate_ids = descriptor;
        duplicate_ids.resident_tensor_ids = vec!["tensor-a".to_string(), "tensor-a".to_string()];
        assert!(matches!(
            manifest.resolve_stage_admission(&duplicate_ids),
            Err(StageAdmissionError::TensorBinding(
                MaterializationBindingError::TensorIdsNotStrictlySorted { .. }
            ))
        ));

        let unsorted_ids = StageAdmissionDescriptor {
            package_id: manifest.package_id.clone(),
            resident_tensor_ids: vec!["tensor-b".to_string(), "tensor-a".to_string()],
            sidecars: Vec::new(),
        };
        assert!(matches!(
            manifest.resolve_stage_admission(&unsorted_ids),
            Err(StageAdmissionError::TensorBinding(
                MaterializationBindingError::TensorIdsNotStrictlySorted { .. }
            ))
        ));
    }

    #[test]
    fn stored_length_sum_fails_closed_on_overflow() {
        let first_artifact = artifact("first", "artifacts/first.gguf", u64::MAX);
        let second_artifact = artifact("second", "artifacts/second.gguf", 1);
        let dimensions = [1];
        let integrity = TensorIntegrity::ArtifactSha256;
        let bindings = vec![
            MaterializationTensor {
                tensor_id: "first",
                native_name: "first",
                ggml_type: 1,
                dimensions: &dimensions,
                artifact: &first_artifact,
                data_offset: 0,
                stored_length: u64::MAX,
                alignment: 1,
                integrity: &integrity,
            },
            MaterializationTensor {
                tensor_id: "second",
                native_name: "second",
                ggml_type: 1,
                dimensions: &dimensions,
                artifact: &second_artifact,
                data_offset: 0,
                stored_length: 1,
                alignment: 1,
                integrity: &integrity,
            },
        ];

        assert!(matches!(
            checked_distinct_stored_length(&bindings),
            Err(StageAdmissionError::StoredLengthOverflow)
        ));
    }

    #[test]
    fn selected_aliases_are_not_admission_bindings() {
        let mut manifest = fixture();
        manifest.tensor_catalog.entries.push(Tensor {
            id: "tensor-c".to_string(),
            name: "alias".to_string(),
            ggml_type: 1,
            dimensions: vec![4, 4],
            layer_ordinal: Some(0),
            storage: TensorStorage::Alias {
                target_tensor_id: "tensor-a".to_string(),
            },
        });
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest.validate().unwrap();
        let request = StageAdmissionDescriptor {
            package_id: manifest.package_id.clone(),
            resident_tensor_ids: vec!["tensor-c".to_string()],
            sidecars: Vec::new(),
        };

        assert!(matches!(
            manifest.resolve_stage_admission(&request),
            Err(StageAdmissionError::TensorBinding(
                MaterializationBindingError::AliasNotMaterializable { .. }
            ))
        ));
    }

    #[test]
    fn tensor_selection_never_falls_back_to_name_layer_or_role() {
        let manifest = fixture();
        for candidate in ["native.layer.0", "0", "shared"] {
            let request = StageAdmissionDescriptor {
                package_id: manifest.package_id.clone(),
                resident_tensor_ids: vec![candidate.to_string()],
                sidecars: Vec::new(),
            };
            assert!(matches!(
                manifest.resolve_stage_admission(&request),
                Err(StageAdmissionError::TensorBinding(
                    MaterializationBindingError::UnknownTensorId { tensor_id }
                )) if tensor_id == candidate
            ));
        }
    }

    fn descriptor(manifest: &PackageManifest) -> StageAdmissionDescriptor {
        StageAdmissionDescriptor {
            package_id: manifest.package_id.clone(),
            resident_tensor_ids: vec!["tensor-a".to_string(), "tensor-b".to_string()],
            sidecars: manifest.sidecars.clone(),
        }
    }

    fn fixture() -> PackageManifest {
        let mut manifest = PackageManifest {
            schema_version: PACKAGE_SCHEMA_VERSION,
            package_id: String::new(),
            model_id: "fixture/model".to_string(),
            source_model: SourceModel {
                sha256: DIGEST.to_string(),
                metadata_artifact_id: "weights-a".to_string(),
                repo: None,
                revision: None,
                primary_file: Some("model.gguf".to_string()),
                canonical_ref: None,
                distribution_id: None,
                files: vec![SourceFile {
                    path: "model.gguf".to_string(),
                    byte_size: 256,
                    sha256: DIGEST.to_string(),
                }],
            },
            format: "gguf".to_string(),
            layer_count: 2,
            model_metadata: BTreeMap::from([(
                "general.architecture".to_string(),
                Value::String("llama".to_string()),
            )]),
            artifact_catalog: ArtifactCatalog {
                entries: vec![
                    artifact("weights-a", "artifacts/source-00000.gguf", 256),
                    artifact("weights-b", "artifacts/source-00001.gguf", 128),
                    artifact("mmproj-a", "sidecars/a.gguf", 64),
                    artifact("mmproj-b", "sidecars/b.gguf", 64),
                    artifact("unused", "artifacts/unused.gguf", 64),
                ],
            },
            tensor_catalog: TensorCatalog {
                entries: vec![
                    Tensor {
                        id: "tensor-a".to_string(),
                        name: "native.layer.0".to_string(),
                        ggml_type: 1,
                        dimensions: vec![4, 4],
                        layer_ordinal: Some(0),
                        storage: TensorStorage::Owned {
                            artifact_id: "weights-a".to_string(),
                            data_offset: 0,
                            stored_length: 32,
                            alignment: 32,
                            integrity: TensorIntegrity::ArtifactSha256,
                        },
                    },
                    Tensor {
                        id: "tensor-b".to_string(),
                        name: "shared".to_string(),
                        ggml_type: 1,
                        dimensions: vec![8, 4],
                        layer_ordinal: None,
                        storage: TensorStorage::Owned {
                            artifact_id: "weights-b".to_string(),
                            data_offset: 64,
                            stored_length: 64,
                            alignment: 32,
                            integrity: TensorIntegrity::ArtifactSha256,
                        },
                    },
                ],
            },
            sidecars: vec![
                Sidecar {
                    kind: SidecarKind::Mmproj,
                    artifact_id: "mmproj-a".to_string(),
                    name: None,
                },
                Sidecar {
                    kind: SidecarKind::Mmproj,
                    artifact_id: "mmproj-b".to_string(),
                    name: Some("vision".to_string()),
                },
            ],
            generation: None,
            native_abi_version: "0.1.49".to_string(),
            generator_version: "test".to_string(),
            created_at_unix_secs: 1,
        };
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest.validate().unwrap();
        manifest
    }

    fn artifact(id: &str, path: &str, byte_size: u64) -> Artifact {
        Artifact {
            id: id.to_string(),
            path: path.to_string(),
            byte_size,
            sha256: DIGEST.to_string(),
        }
    }
}
