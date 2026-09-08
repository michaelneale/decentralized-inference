use super::*;

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn valid_manifest_passes() {
    fixture().validate().unwrap();
}

#[test]
fn metadata_artifact_binding_is_required_and_source_identity_is_exact() {
    let manifest = fixture();
    let mut encoded = serde_json::to_value(&manifest).unwrap();
    encoded["source_model"]
        .as_object_mut()
        .unwrap()
        .remove("metadata_artifact_id");
    assert!(serde_json::from_value::<PackageManifest>(encoded).is_err());

    let mut empty = manifest.clone();
    empty.source_model.metadata_artifact_id.clear();
    empty.package_id = empty.computed_package_id().unwrap();
    assert!(empty.validate().unwrap_err().issues().iter().any(|issue| {
        issue.code == ValidationCode::MissingValue
            && issue.path == "source_model.metadata_artifact_id"
    }));

    let mut unknown = manifest.clone();
    unknown.source_model.metadata_artifact_id = "missing".to_string();
    unknown.package_id = unknown.computed_package_id().unwrap();
    assert!(
        unknown
            .validate()
            .unwrap_err()
            .issues()
            .iter()
            .any(|issue| {
                issue.code == ValidationCode::UnknownArtifact
                    && issue.path == "source_model.metadata_artifact_id"
            })
    );

    let mut missing_primary = manifest.clone();
    missing_primary.source_model.primary_file = None;
    missing_primary.package_id = missing_primary.computed_package_id().unwrap();
    assert!(
        missing_primary
            .validate()
            .unwrap_err()
            .issues()
            .iter()
            .any(|issue| {
                issue.code == ValidationCode::MissingValue
                    && issue.path == "source_model.primary_file"
            })
    );

    let mut generated_metadata = manifest.clone();
    generated_metadata.artifact_catalog.entries[0].path = "shared/metadata.gguf".to_string();
    generated_metadata.artifact_catalog.entries[0].sha256 = "1".repeat(64);
    generated_metadata.artifact_catalog.entries[0].byte_size += 1;
    generated_metadata.package_id = generated_metadata.computed_package_id().unwrap();
    generated_metadata.validate().unwrap();

    let mut changed_source = manifest;
    changed_source.source_model.sha256 = "1".repeat(64);
    changed_source.package_id = changed_source.computed_package_id().unwrap();
    assert!(
        changed_source
            .validate()
            .unwrap_err()
            .issues()
            .iter()
            .any(|issue| { issue.code == ValidationCode::SourceIdentityMismatch })
    );
}

#[test]
fn metadata_artifact_must_have_gguf_source_role() {
    let manifest = fixture();
    let mut wrong_format = manifest.clone();
    wrong_format.artifact_catalog.entries[0].path = "layers/layer-00000.bin".to_string();
    wrong_format.package_id = wrong_format.computed_package_id().unwrap();
    assert!(
        wrong_format
            .validate()
            .unwrap_err()
            .issues()
            .iter()
            .any(|issue| { issue.code == ValidationCode::InvalidMetadataArtifact })
    );

    let mut sidecar = manifest;
    sidecar.sidecars.push(Sidecar {
        kind: SidecarKind::Mmproj,
        artifact_id: sidecar.source_model.metadata_artifact_id.clone(),
        name: None,
    });
    sidecar.package_id = sidecar.computed_package_id().unwrap();
    assert!(
        sidecar
            .validate()
            .unwrap_err()
            .issues()
            .iter()
            .any(|issue| { issue.code == ValidationCode::InvalidMetadataArtifact })
    );
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
        path: "model-00002-of-00002.gguf".to_string(),
        byte_size: 32,
        sha256: DIGEST.to_string(),
    });
    manifest.artifact_catalog.entries.push(Artifact {
        id: "mmproj".to_string(),
        path: "sidecars/mmproj.gguf".to_string(),
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
        kind: SidecarKind::Mmproj,
        artifact_id: "mmproj".to_string(),
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
fn package_identity_excludes_creation_timestamp() {
    let manifest = fixture();
    let mut created_later = manifest.clone();
    created_later.created_at_unix_secs = manifest.created_at_unix_secs + 86_400;

    assert_eq!(
        created_later.computed_package_id().unwrap(),
        manifest.computed_package_id().unwrap()
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
    let cache_proposer = &value["generation"]["speculative_decoding"]["proposers"]["cache-head"];
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
                issue.code == ValidationCode::InvalidGenerationValue && issue.path.ends_with(field)
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
            metadata_artifact_id: "layer-0".to_string(),
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
