use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::{Artifact, PackageManifest, TensorIntegrity, TensorStorage, ValidationErrors};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializationTensor<'a> {
    pub tensor_id: &'a str,
    pub native_name: &'a str,
    pub ggml_type: u32,
    pub dimensions: &'a [u64],
    pub artifact: &'a Artifact,
    pub data_offset: u64,
    pub stored_length: u64,
    pub alignment: u64,
    pub integrity: &'a TensorIntegrity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterializationBindingError {
    InvalidManifest(ValidationErrors),
    EmptyTensorId {
        index: usize,
    },
    TensorIdsNotStrictlySorted {
        index: usize,
        previous: String,
        current: String,
    },
    UnknownTensorId {
        tensor_id: String,
    },
    AliasNotMaterializable {
        tensor_id: String,
        native_name: String,
        target_tensor_id: String,
    },
    MissingArtifact {
        tensor_id: String,
        artifact_id: String,
    },
}

impl fmt::Display for MaterializationBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidManifest(error) => write!(formatter, "{error}"),
            Self::EmptyTensorId { index } => {
                write!(
                    formatter,
                    "materialization tensor id at index {index} is empty"
                )
            }
            Self::TensorIdsNotStrictlySorted {
                index,
                previous,
                current,
            } => write!(
                formatter,
                "materialization tensor ids are not strictly sorted at index {index}: {previous:?} then {current:?}"
            ),
            Self::UnknownTensorId { tensor_id } => {
                write!(formatter, "unknown materialization tensor id {tensor_id:?}")
            }
            Self::AliasNotMaterializable {
                tensor_id,
                native_name,
                target_tensor_id,
            } => write!(
                formatter,
                "selected tensor {tensor_id:?} ({native_name:?}) aliases {target_tensor_id:?}, but alias materialization is unsupported"
            ),
            Self::MissingArtifact {
                tensor_id,
                artifact_id,
            } => write!(
                formatter,
                "selected tensor {tensor_id:?} references missing artifact {artifact_id:?}"
            ),
        }
    }
}

impl Error for MaterializationBindingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidManifest(error) => Some(error),
            _ => None,
        }
    }
}

impl PackageManifest {
    /// Resolve an exact normalized plan closure by package tensor ID.
    ///
    /// Tensor names, layer ordinals, roles, and aliases are never alternate
    /// selectors. The returned native name is used only after its package ID
    /// has been resolved exactly.
    pub fn materialization_tensors<'a>(
        &'a self,
        exact_tensor_ids: &[String],
    ) -> Result<Vec<MaterializationTensor<'a>>, MaterializationBindingError> {
        self.validate()
            .map_err(MaterializationBindingError::InvalidManifest)?;
        validate_requested_ids(exact_tensor_ids)?;

        let tensors = self
            .tensor_catalog
            .entries
            .iter()
            .map(|tensor| (tensor.id.as_str(), tensor))
            .collect::<BTreeMap<_, _>>();
        let artifacts = self
            .artifact_catalog
            .entries
            .iter()
            .map(|artifact| (artifact.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();

        exact_tensor_ids
            .iter()
            .map(|tensor_id| {
                let tensor = tensors.get(tensor_id.as_str()).ok_or_else(|| {
                    MaterializationBindingError::UnknownTensorId {
                        tensor_id: tensor_id.clone(),
                    }
                })?;
                let TensorStorage::Owned {
                    artifact_id,
                    data_offset,
                    stored_length,
                    alignment,
                    integrity,
                } = &tensor.storage
                else {
                    let TensorStorage::Alias { target_tensor_id } = &tensor.storage else {
                        unreachable!()
                    };
                    return Err(MaterializationBindingError::AliasNotMaterializable {
                        tensor_id: tensor.id.clone(),
                        native_name: tensor.name.clone(),
                        target_tensor_id: target_tensor_id.clone(),
                    });
                };
                let artifact = artifacts.get(artifact_id.as_str()).ok_or_else(|| {
                    MaterializationBindingError::MissingArtifact {
                        tensor_id: tensor.id.clone(),
                        artifact_id: artifact_id.clone(),
                    }
                })?;
                Ok(MaterializationTensor {
                    tensor_id: &tensor.id,
                    native_name: &tensor.name,
                    ggml_type: tensor.ggml_type,
                    dimensions: &tensor.dimensions,
                    artifact,
                    data_offset: *data_offset,
                    stored_length: *stored_length,
                    alignment: *alignment,
                    integrity,
                })
            })
            .collect()
    }
}

fn validate_requested_ids(exact_tensor_ids: &[String]) -> Result<(), MaterializationBindingError> {
    for (index, tensor_id) in exact_tensor_ids.iter().enumerate() {
        if tensor_id.is_empty() {
            return Err(MaterializationBindingError::EmptyTensorId { index });
        }
    }
    for (index, pair) in exact_tensor_ids.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(MaterializationBindingError::TensorIdsNotStrictlySorted {
                index: index + 1,
                previous: pair[0].clone(),
                current: pair[1].clone(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::Value;

    use super::*;
    use crate::{
        ArtifactCatalog, PACKAGE_SCHEMA_VERSION, SourceFile, SourceModel, Tensor, TensorCatalog,
    };

    const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn binds_exact_id_to_distinct_native_name() {
        let manifest = fixture();
        let bindings = manifest
            .materialization_tensors(&["a".to_string()])
            .unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].tensor_id, "a");
        assert_eq!(bindings[0].native_name, "native-only");
        assert_eq!(bindings[0].artifact.id, "weights");
        assert_eq!(bindings[0].artifact.path, "artifacts/source-00000.gguf");
        assert_eq!(bindings[0].artifact.sha256, DIGEST);
        assert_eq!(bindings[0].data_offset, 0);
        assert_eq!(bindings[0].stored_length, 32);
        assert_eq!(bindings[0].alignment, 32);
        assert_eq!(bindings[0].integrity, &TensorIntegrity::ArtifactSha256);

        assert!(matches!(
            manifest.materialization_tensors(&["native-only".to_string()]),
            Err(MaterializationBindingError::UnknownTensorId { tensor_id })
                if tensor_id == "native-only"
        ));
        assert!(matches!(
            manifest.materialization_tensors(&["b".to_string()]),
            Ok(bindings) if bindings[0].native_name == "a"
        ));
    }

    #[test]
    fn rejects_non_normalized_or_missing_ids() {
        let manifest = fixture();
        assert!(matches!(
            manifest.materialization_tensors(&["b".to_string(), "a".to_string()]),
            Err(MaterializationBindingError::TensorIdsNotStrictlySorted { .. })
        ));
        assert!(matches!(
            manifest.materialization_tensors(&["a".to_string(), "a".to_string()]),
            Err(MaterializationBindingError::TensorIdsNotStrictlySorted { .. })
        ));
        assert!(matches!(
            manifest.materialization_tensors(&["missing".to_string()]),
            Err(MaterializationBindingError::UnknownTensorId { .. })
        ));

        let bindings = manifest
            .materialization_tensors(&["a".to_string(), "b".to_string()])
            .unwrap();
        assert_eq!(
            bindings
                .iter()
                .map(|binding| binding.tensor_id)
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn rejects_selected_alias_without_substitution() {
        let mut manifest = fixture();
        manifest.tensor_catalog.entries.push(Tensor {
            id: "c".to_string(),
            name: "alias-native".to_string(),
            ggml_type: 1,
            dimensions: vec![4, 4],
            layer_ordinal: None,
            storage: TensorStorage::Alias {
                target_tensor_id: "a".to_string(),
            },
        });
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest.validate().unwrap();

        assert!(manifest.materialization_tensors(&["a".to_string()]).is_ok());
        assert!(matches!(
            manifest.materialization_tensors(&["c".to_string()]),
            Err(MaterializationBindingError::AliasNotMaterializable {
                tensor_id,
                native_name,
                target_tensor_id,
            }) if tensor_id == "c" && native_name == "alias-native" && target_tensor_id == "a"
        ));
        assert!(matches!(
            manifest.materialization_tensors(&["a".to_string(), "c".to_string()]),
            Err(MaterializationBindingError::AliasNotMaterializable { .. })
        ));
    }

    #[test]
    fn rejects_invalid_manifest_before_binding() {
        let mut manifest = fixture();
        manifest.package_id = "sha256:invalid".to_string();

        assert!(matches!(
            manifest.materialization_tensors(&["a".to_string()]),
            Err(MaterializationBindingError::InvalidManifest(_))
        ));
    }

    fn fixture() -> PackageManifest {
        let mut manifest = PackageManifest {
            schema_version: PACKAGE_SCHEMA_VERSION,
            package_id: String::new(),
            model_id: "model".to_string(),
            source_model: SourceModel {
                sha256: DIGEST.to_string(),
                repo: None,
                revision: None,
                primary_file: Some("model.gguf".to_string()),
                canonical_ref: None,
                distribution_id: None,
                files: vec![SourceFile {
                    path: "model.gguf".to_string(),
                    byte_size: 128,
                    sha256: DIGEST.to_string(),
                }],
            },
            format: "gguf-v2".to_string(),
            layer_count: 2,
            model_metadata: BTreeMap::from([(
                "general.architecture".to_string(),
                Value::String("llama".to_string()),
            )]),
            artifact_catalog: ArtifactCatalog {
                entries: vec![Artifact {
                    id: "weights".to_string(),
                    path: "artifacts/source-00000.gguf".to_string(),
                    byte_size: 128,
                    sha256: DIGEST.to_string(),
                }],
            },
            tensor_catalog: TensorCatalog {
                entries: vec![
                    Tensor {
                        id: "a".to_string(),
                        name: "native-only".to_string(),
                        ggml_type: 1,
                        dimensions: vec![4, 4],
                        layer_ordinal: None,
                        storage: TensorStorage::Owned {
                            artifact_id: "weights".to_string(),
                            data_offset: 0,
                            stored_length: 32,
                            alignment: 32,
                            integrity: TensorIntegrity::ArtifactSha256,
                        },
                    },
                    Tensor {
                        id: "b".to_string(),
                        name: "a".to_string(),
                        ggml_type: 1,
                        dimensions: vec![4, 4],
                        layer_ordinal: Some(1),
                        storage: TensorStorage::Owned {
                            artifact_id: "weights".to_string(),
                            data_offset: 64,
                            stored_length: 32,
                            alignment: 32,
                            integrity: TensorIntegrity::ArtifactSha256,
                        },
                    },
                ],
            },
            sidecars: Vec::new(),
            generation: None,
            native_abi_version: "0.1.49".to_string(),
            generator_version: "test".to_string(),
            created_at_unix_secs: 1,
        };
        manifest.package_id = manifest.computed_package_id().unwrap();
        manifest
    }
}
