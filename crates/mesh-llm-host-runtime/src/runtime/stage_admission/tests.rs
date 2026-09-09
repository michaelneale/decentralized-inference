use super::*;
use skippy_package_format::{
    Artifact, ArtifactCatalog, PACKAGE_SCHEMA_VERSION, SidecarKind, SourceFile, SourceModel,
    Tensor, TensorCatalog, TensorIntegrity, TensorStorage,
};
use std::collections::BTreeMap;

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// A minimal, self-consistent v2 manifest whose tensor catalog contains
/// exactly `package.tensor.a` and `package.tensor.b`.
fn manifest() -> PackageManifest {
    let mut m = PackageManifest {
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
        layer_count: 16,
        model_metadata: BTreeMap::from([(
            "general.architecture".to_string(),
            serde_json::Value::String("llama".to_string()),
        )]),
        artifact_catalog: ArtifactCatalog {
            entries: vec![
                artifact("weights-a", "artifacts/source-00000.gguf", 256),
                artifact("weights-b", "artifacts/source-00001.gguf", 128),
            ],
        },
        tensor_catalog: TensorCatalog {
            entries: vec![
                tensor("package.tensor.a", "native.layer.0", "weights-a", Some(0)),
                tensor("package.tensor.b", "shared", "weights-b", None),
            ],
        },
        sidecars: Vec::new(),
        generation: None,
        native_abi_version: "0.1.52".to_string(),
        generator_version: "test".to_string(),
        created_at_unix_secs: 1,
    };
    m.package_id = m.computed_package_id().unwrap();
    m.validate().unwrap();
    m
}

fn artifact(id: &str, path: &str, byte_size: u64) -> Artifact {
    Artifact {
        id: id.to_string(),
        path: path.to_string(),
        byte_size,
        sha256: DIGEST.to_string(),
    }
}

fn tensor(id: &str, name: &str, artifact_id: &str, layer_ordinal: Option<u32>) -> Tensor {
    Tensor {
        id: id.to_string(),
        name: name.to_string(),
        ggml_type: 1,
        dimensions: vec![4, 4],
        layer_ordinal,
        storage: TensorStorage::Owned {
            artifact_id: artifact_id.to_string(),
            data_offset: 0,
            stored_length: 32,
            alignment: 32,
            integrity: TensorIntegrity::ArtifactSha256,
        },
    }
}

fn planned_profile(profile_id: &str) -> PlannedStageProfile {
    PlannedStageProfile {
        profile_id: profile_id.to_string(),
        graph_identity: "graph-1".to_string(),
        profile_identity: "profile-identity-1".to_string(),
        slice_identity: "slice-1".to_string(),
        source_snapshot_identity: "snapshot-1".to_string(),
        graph_configuration_id: "graph-config-1".to_string(),
        backend_id: "backend-1".to_string(),
    }
}

fn realized_profile(profile_id: &str) -> RealizedStageProfile {
    let planned = planned_profile(profile_id);
    RealizedStageProfile {
        profile_id: planned.profile_id,
        graph_identity: planned.graph_identity,
        profile_identity: planned.profile_identity,
        slice_identity: planned.slice_identity,
        source_snapshot_identity: planned.source_snapshot_identity,
        graph_configuration_id: planned.graph_configuration_id,
        backend_id: planned.backend_id,
        activation_imports: Vec::new(),
        activation_exports: Vec::new(),
        request_inputs: Vec::new(),
        state_effects: Vec::new(),
    }
}

fn planned(m: &PackageManifest, tensors: &[&str], profiles: &[&str]) -> PlannedStageAdmission {
    PlannedStageAdmission {
        package_id: m.package_id.clone(),
        plan_id: "skippy-plan:v1:t0".to_string(),
        layer_start: 0,
        layer_end: 16,
        resident_tensor_ids: tensors.iter().map(|t| t.to_string()).collect(),
        sidecars: Vec::new(),
        profiles: profiles.iter().map(|p| planned_profile(p)).collect(),
    }
}

fn realized(m: &PackageManifest, tensors: &[&str], profiles: &[&str]) -> RealizedStagePlan {
    RealizedStagePlan {
        package_id: m.package_id.clone(),
        plan_id: "skippy-plan:v1:t0".to_string(),
        layer_start: 0,
        layer_end: 16,
        resident_tensor_ids: tensors.iter().map(|t| t.to_string()).collect(),
        sidecars: Vec::new(),
        profiles: profiles.iter().map(|p| realized_profile(p)).collect(),
    }
}

#[test]
fn planned_admission_uses_manifest_topology_and_sidecar_inputs() {
    let mut m = manifest();
    let sidecar = Sidecar {
        kind: skippy_package_format::SidecarKind::Mmproj,
        artifact_id: "weights-b".to_string(),
        name: Some("vision".to_string()),
    };
    m.sidecars = vec![sidecar.clone()];
    let mut discovered = realized(&m, &["package.tensor.a"], &["decode"]);
    discovered.package_id = "wrong-package".to_string();
    discovered.layer_start = 99;
    discovered.layer_end = 100;
    discovered.sidecars.clear();

    let planned =
        planned_admission_from_discovery(&m, (2, 7), std::slice::from_ref(&sidecar), &discovered);

    assert_eq!(planned.package_id, m.package_id);
    assert_eq!((planned.layer_start, planned.layer_end), (2, 7));
    assert_eq!(planned.sidecars, vec![sidecar]);
    assert_eq!(planned.plan_id, discovered.plan_id);
    assert_eq!(planned.resident_tensor_ids, discovered.resident_tensor_ids);
}

#[test]
fn matching_plan_and_realization_is_admitted() {
    let m = manifest();
    let admitted = admit_stage_plan(
        &planned(
            &m,
            &["package.tensor.a", "package.tensor.b"],
            &["batched", "decode"],
        ),
        &realized(
            &m,
            &["package.tensor.a", "package.tensor.b"],
            &["batched", "decode"],
        ),
        &m,
    )
    .expect("matching plan must admit");
    assert_eq!(admitted.package_id, m.package_id);
    assert_eq!(admitted.resident_tensor_ids.len(), 2);
}

#[test]
fn package_id_mismatch_rejects() {
    let m = manifest();
    let mut r = realized(&m, &["package.tensor.a"], &["batched"]);
    r.package_id = format!("sha256:{}", "1".repeat(64));
    assert!(matches!(
        admit_stage_plan(&planned(&m, &["package.tensor.a"], &["batched"]), &r, &m),
        Err(StagePlanAdmissionError::PackageIdMismatch { .. })
    ));
}

#[test]
fn manifest_package_id_mismatch_fails_closed() {
    let m = manifest();
    // Both sides agree on a package id the manifest does not carry.
    let mut p = planned(&m, &["package.tensor.a"], &["batched"]);
    let mut r = realized(&m, &["package.tensor.a"], &["batched"]);
    p.package_id = format!("sha256:{}", "2".repeat(64));
    r.package_id = p.package_id.clone();
    assert!(matches!(
        admit_stage_plan(&p, &r, &m),
        Err(StagePlanAdmissionError::ManifestResolution(
            skippy_package_format::stage_admission::StageAdmissionError::PackageIdMismatch { .. }
        ))
    ));
}

#[test]
fn unknown_resident_tensor_fails_closed() {
    let m = manifest();
    let r = realized(&m, &["package.tensor.unknown"], &["batched"]);
    assert!(matches!(
        admit_stage_plan(
            &planned(&m, &["package.tensor.unknown"], &["batched"]),
            &r,
            &m
        ),
        Err(StagePlanAdmissionError::ManifestResolution(_))
    ));
}

#[test]
fn plan_id_mismatch_rejects() {
    let m = manifest();
    let mut r = realized(&m, &["package.tensor.a"], &["batched"]);
    r.plan_id = "skippy-plan:v1:other".to_string();
    assert!(matches!(
        admit_stage_plan(&planned(&m, &["package.tensor.a"], &["batched"]), &r, &m),
        Err(StagePlanAdmissionError::PlanIdMismatch { .. })
    ));
}

#[test]
fn layer_range_mismatch_rejects() {
    let m = manifest();
    let mut r = realized(&m, &["package.tensor.a"], &["batched"]);
    r.layer_end = 18;
    assert!(matches!(
        admit_stage_plan(&planned(&m, &["package.tensor.a"], &["batched"]), &r, &m),
        Err(StagePlanAdmissionError::LayerRangeMismatch { .. })
    ));
}

#[test]
fn tensor_closure_mismatch_rejects() {
    let m = manifest();
    let r = realized(&m, &["package.tensor.a", "package.tensor.b"], &["batched"]);
    assert!(matches!(
        admit_stage_plan(
            &planned(&m, &["package.tensor.a"], &["batched"]),
            &r,
            &m
        ),
        Err(StagePlanAdmissionError::TensorClosureMismatch { planned_only, realized_only })
            if planned_only.is_empty()
                && realized_only == vec!["package.tensor.b".to_string()]
    ));
}

#[test]
fn unsorted_realized_tensor_ids_reject_before_identity() {
    let m = manifest();
    let r = realized(&m, &["package.tensor.b", "package.tensor.a"], &["batched"]);
    assert!(matches!(
        admit_stage_plan(
            // planned side is sorted; only the realized side is not
            &planned(&m, &["package.tensor.a", "package.tensor.b"], &["batched"]),
            &r,
            &m
        ),
        Err(StagePlanAdmissionError::RealizedTensorIdsNotStrictlySorted { .. })
    ));
}

#[test]
fn unsorted_planned_tensor_ids_reject_first() {
    let m = manifest();
    let r = realized(&m, &["package.tensor.a", "package.tensor.b"], &["batched"]);
    assert!(matches!(
        admit_stage_plan(
            &planned(&m, &["package.tensor.b", "package.tensor.a"], &["batched"]),
            &r,
            &m
        ),
        Err(StagePlanAdmissionError::PlannedTensorIdsNotStrictlySorted { .. })
    ));
}

#[test]
fn duplicate_realized_tensor_ids_reject() {
    let m = manifest();
    let r = realized(&m, &["package.tensor.a", "package.tensor.a"], &["batched"]);
    assert!(matches!(
        admit_stage_plan(&planned(&m, &["package.tensor.a"], &["batched"]), &r, &m),
        Err(StagePlanAdmissionError::RealizedTensorIdsNotStrictlySorted { .. })
    ));
}

#[test]
fn sidecar_mismatch_rejects() {
    let m = manifest();
    let mut r = realized(&m, &["package.tensor.a"], &["batched"]);
    r.sidecars = vec![Sidecar {
        kind: SidecarKind::Mmproj,
        artifact_id: "weights-b".to_string(),
        name: None,
    }];
    assert!(matches!(
        admit_stage_plan(&planned(&m, &["package.tensor.a"], &["batched"]), &r, &m),
        Err(StagePlanAdmissionError::SidecarMismatch { .. })
    ));
}

#[test]
fn profile_identity_mismatch_rejects() {
    let m = manifest();
    let mut r = realized(&m, &["package.tensor.a"], &["batched"]);
    r.profiles[0].graph_identity = "graph-2".to_string();
    assert!(matches!(
        admit_stage_plan(&planned(&m, &["package.tensor.a"], &["batched"]), &r, &m),
        Err(StagePlanAdmissionError::ProfileIdentityMismatch {
            field: "graph_identity",
            ..
        })
    ));
}

#[test]
fn profile_set_mismatch_rejects() {
    let m = manifest();
    let r = realized(&m, &["package.tensor.a"], &["batched"]);
    assert!(matches!(
        admit_stage_plan(
            &planned(&m, &["package.tensor.a"], &["batched", "decode"]),
            &r,
            &m
        ),
        Err(StagePlanAdmissionError::ProfileSetMismatch { .. })
    ));
}

#[test]
fn backend_identity_mismatch_rejects() {
    let m = manifest();
    let mut r = realized(&m, &["package.tensor.a"], &["batched"]);
    r.profiles[0].backend_id = "backend-2".to_string();
    assert!(matches!(
        admit_stage_plan(&planned(&m, &["package.tensor.a"], &["batched"]), &r, &m),
        Err(StagePlanAdmissionError::ProfileIdentityMismatch {
            field: "backend_id",
            ..
        })
    ));
}

#[test]
#[ignore = "requires SKIPPY_PACKAGE_V2_TEST_DIR and the prepared native runtime"]
fn realizes_and_admits_a_real_package_v2_chain() {
    let package_dir = std::env::var_os("SKIPPY_PACKAGE_V2_TEST_DIR")
        .map(PathBuf::from)
        .expect("SKIPPY_PACKAGE_V2_TEST_DIR is required");
    let admissions = realize_stage_admissions(
        &package_dir,
        &[(0, 16), (16, 32)],
        &[
            StagePlannerProfile {
                profile_id: "batched".to_string(),
                n_tokens: 8,
                n_sequences: 2,
                n_outputs: 8,
                n_recurrent_rollback_sequences: 0,
            },
            StagePlannerProfile {
                profile_id: "decode".to_string(),
                n_tokens: 1,
                n_sequences: 1,
                n_outputs: 1,
                n_recurrent_rollback_sequences: 0,
            },
            StagePlannerProfile {
                profile_id: "prefill".to_string(),
                n_tokens: 8,
                n_sequences: 1,
                n_outputs: 8,
                n_recurrent_rollback_sequences: 0,
            },
        ],
        "skippy-graph-configuration:v1:real-package-test",
        "skippy-backend:cpu:v1",
    )
    .expect("real package-v2 chain must admit");
    assert_eq!(admissions.len(), 2);
    assert_eq!(admissions[0].package_id, admissions[1].package_id);
    assert_ne!(admissions[0].plan_id, admissions[1].plan_id);
    assert!(
        admissions
            .iter()
            .all(|admission| !admission.resident_tensor_ids.is_empty())
    );
    assert!(
        admissions
            .iter()
            .all(|admission| admission.profiles.len() == 3)
    );
    let package_ref = package_dir.to_string_lossy();
    for admission in &admissions {
        let descriptor = skippy_package_format::stage_admission::StageAdmissionDescriptor {
            package_id: admission.package_id.clone(),
            resident_tensor_ids: admission.resident_tensor_ids.clone(),
            sidecars: Vec::new(),
        };
        let (_, model_parts, projector) =
            crate::inference::skippy::resolve_package_v2_stage_to_local(&package_ref, &descriptor)
                .expect("real package-v2 stage artifacts must resolve");
        assert!(projector.is_none());
        assert!(
            model_parts
                .iter()
                .any(|path| path.ends_with("shared/metadata.gguf"))
        );
        assert!(
            model_parts
                .iter()
                .all(|path| path.starts_with(&package_dir))
        );
        assert!(
            model_parts
                .iter()
                .all(|path| !path.starts_with(package_dir.join("artifacts")))
        );
    }
}
