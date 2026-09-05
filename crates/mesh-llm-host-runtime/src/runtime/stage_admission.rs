//! Host-side admission of realized stage plans against planned stages.
//!
//! Binds the realized native stage descriptor (returned through the public
//! stage-plan ABI, patch 0074) to the planned stage before topology
//! publication. Every binding below is exact and fail-closed: package
//! identity is recomputed from manifest content, plan IDs must be stable,
//! layer ranges must match the plan, resident tensor closures resolve
//! through the v2 manifest with no name/layer/role fallbacks, and guarded
//! profile identities must agree on both sides. Any mismatch produces a
//! structured [`StagePlanAdmissionError`] before `activate_stage_topology`.

use std::collections::BTreeMap;
use std::fmt;

use skippy_package_format::{PackageManifest, Sidecar};

/// The planned admission expectations for one stage.
///
/// This is carried by the planner on `RuntimeSliceStagePlan` from planning
/// time and mirrored into the generation-8 control protocol descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedStageAdmission {
    /// Content-derived package identity (`sha256:...`).
    pub package_id: String,
    /// Coordinator-minted plan identity.
    pub plan_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    /// Exact sorted, unique resident tensor IDs.
    pub resident_tensor_ids: Vec<String>,
    /// Typed sidecar references, strictly sorted.
    pub sidecars: Vec<Sidecar>,
    /// Guarded per-profile identities, sorted by `profile_id`.
    pub profiles: Vec<PlannedStageProfile>,
}

/// One guarded execution profile's planned identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedStageProfile {
    pub profile_id: String,
    pub graph_identity: String,
    pub profile_identity: String,
    pub slice_identity: String,
    pub source_snapshot_identity: String,
    pub graph_configuration_id: String,
    pub backend_id: String,
}

/// The realized native stage descriptor as returned through the ABI.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RealizedStagePlan {
    pub package_id: String,
    pub plan_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    /// Exact sorted, unique resident tensor IDs realized by the native side.
    pub resident_tensor_ids: Vec<String>,
    /// Typed sidecar references realized by the native side, strictly sorted.
    pub sidecars: Vec<Sidecar>,
    pub profiles: Vec<RealizedStageProfile>,
}

/// One guarded execution profile's realized identities.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RealizedStageProfile {
    pub profile_id: String,
    pub graph_identity: String,
    pub profile_identity: String,
    pub slice_identity: String,
    pub source_snapshot_identity: String,
    pub graph_configuration_id: String,
    pub backend_id: String,
}

/// The admitted stage: exact identities that passed every check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedStage {
    pub package_id: String,
    pub plan_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub resident_tensor_ids: Vec<String>,
}

#[derive(Debug)]
pub enum StagePlanAdmissionError {
    PlannedTensorIdsNotStrictlySorted {
        index: usize,
        previous: String,
        current: String,
    },
    RealizedTensorIdsNotStrictlySorted {
        index: usize,
        previous: String,
        current: String,
    },
    PlannedSidecarsNotStrictlySorted {
        index: usize,
    },
    RealizedSidecarsNotStrictlySorted {
        index: usize,
    },
    PlannedProfilesNotSorted {
        index: usize,
    },
    RealizedProfilesNotSorted {
        index: usize,
    },
    PackageIdMismatch {
        planned: String,
        realized: String,
    },
    PlanIdMismatch {
        planned: String,
        realized: String,
    },
    LayerRangeMismatch {
        planned: (u32, u32),
        realized: (u32, u32),
    },
    TensorClosureMismatch {
        planned_only: Vec<String>,
        realized_only: Vec<String>,
    },
    SidecarMismatch {
        planned_only: Vec<Sidecar>,
        realized_only: Vec<Sidecar>,
    },
    ProfileSetMismatch {
        planned_only: Vec<String>,
        realized_only: Vec<String>,
    },
    ProfileIdentityMismatch {
        profile_id: String,
        field: &'static str,
        planned: String,
        realized: String,
    },
    ManifestResolution(skippy_package_format::stage_admission::StageAdmissionError),
}

impl fmt::Display for StagePlanAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlannedTensorIdsNotStrictlySorted {
                index,
                previous,
                current,
            } => write!(
                formatter,
                "planned resident tensor ids are not strictly sorted at index {index}: {previous:?} then {current:?}"
            ),
            Self::RealizedTensorIdsNotStrictlySorted {
                index,
                previous,
                current,
            } => write!(
                formatter,
                "realized resident tensor ids are not strictly sorted at index {index}: {previous:?} then {current:?}"
            ),
            Self::PlannedSidecarsNotStrictlySorted { index } => write!(
                formatter,
                "planned sidecars are not strictly sorted at index {index}"
            ),
            Self::RealizedSidecarsNotStrictlySorted { index } => write!(
                formatter,
                "realized sidecars are not strictly sorted at index {index}"
            ),
            Self::PlannedProfilesNotSorted { index } => write!(
                formatter,
                "planned profiles are not sorted by profile id at index {index}"
            ),
            Self::RealizedProfilesNotSorted { index } => write!(
                formatter,
                "realized profiles are not sorted by profile id at index {index}"
            ),
            Self::PackageIdMismatch { planned, realized } => write!(
                formatter,
                "realized stage package id {realized:?} does not match planned {planned:?}"
            ),
            Self::PlanIdMismatch { planned, realized } => write!(
                formatter,
                "realized stage plan id {realized:?} does not match planned {planned:?}"
            ),
            Self::LayerRangeMismatch { planned, realized } => write!(
                formatter,
                "realized stage layer range {:?} does not match planned {:?}",
                realized, planned
            ),
            Self::TensorClosureMismatch {
                planned_only,
                realized_only,
            } => write!(
                formatter,
                "realized resident tensor closure differs from plan: unplanned realized {planned_only:?}, unrealized planned {realized_only:?}"
            ),
            Self::SidecarMismatch {
                planned_only,
                realized_only,
            } => write!(
                formatter,
                "realized sidecars differ from plan: unplanned realized {planned_only:?}, unrealized planned {realized_only:?}"
            ),
            Self::ProfileSetMismatch {
                planned_only,
                realized_only,
            } => write!(
                formatter,
                "realized profile set differs from plan: unplanned realized {planned_only:?}, unrealized planned {realized_only:?}"
            ),
            Self::ProfileIdentityMismatch {
                profile_id,
                field,
                planned,
                realized,
            } => write!(
                formatter,
                "realized profile {profile_id:?} field {field} is {realized:?} but planned {planned:?}"
            ),
            Self::ManifestResolution(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for StagePlanAdmissionError {}

fn ensure_strictly_sorted(
    ids: &[String],
    mut error: impl FnMut(usize, String, String) -> StagePlanAdmissionError,
) -> Result<(), StagePlanAdmissionError> {
    for (index, window) in ids.windows(2).enumerate() {
        if window[0] >= window[1] {
            return Err(error(index, window[0].clone(), window[1].clone()));
        }
    }
    Ok(())
}

fn ensure_sidecars_sorted(
    sidecars: &[Sidecar],
    planned: bool,
) -> Result<(), StagePlanAdmissionError> {
    for window in sidecars.windows(2) {
        if window[0] >= window[1] {
            let index = 0;
            return Err(if planned {
                StagePlanAdmissionError::PlannedSidecarsNotStrictlySorted { index }
            } else {
                StagePlanAdmissionError::RealizedSidecarsNotStrictlySorted { index }
            });
        }
    }
    Ok(())
}

fn diff_sorted<T>(planned: &[T], realized: &[T]) -> (Vec<T>, Vec<T>)
where
    T: Ord + Clone,
{
    let mut planned_only = Vec::new();
    let mut realized_only = Vec::new();
    let mut planned_iter = planned.iter().peekable();
    let mut realized_iter = realized.iter().peekable();
    loop {
        match (planned_iter.peek(), realized_iter.peek()) {
            (None, None) => break,
            (Some(_), None) => {
                planned_only.push(planned_iter.next().unwrap().clone());
            }
            (None, Some(_)) => {
                realized_only.push(realized_iter.next().unwrap().clone());
            }
            (Some(p), Some(r)) => match p.cmp(r) {
                std::cmp::Ordering::Less => {
                    planned_only.push(planned_iter.next().unwrap().clone());
                }
                std::cmp::Ordering::Greater => {
                    realized_only.push(realized_iter.next().unwrap().clone());
                }
                std::cmp::Ordering::Equal => {
                    planned_iter.next();
                    realized_iter.next();
                }
            },
        }
    }
    (planned_only, realized_only)
}

/// Bind a realized stage plan to its planned admission expectations.
///
/// Resolution order:
/// 1. structural checks on both sides (sortedness of tensor ids, sidecars,
///    profiles);
/// 2. exact package and plan identity;
/// 3. exact layer range;
/// 4. exact resident tensor closure and sidecar set;
/// 5. guarded profile identities per profile;
/// 6. package identity recomputation and tensor closure resolution through
///    the v2 manifest (`PackageManifest::resolve_stage_admission`).
///
/// The returned [`AdmittedStage`] is the only signal that a stage may be
/// published into the topology.
pub fn admit_stage_plan(
    planned: &PlannedStageAdmission,
    realized: &RealizedStagePlan,
    manifest: &PackageManifest,
) -> Result<AdmittedStage, StagePlanAdmissionError> {
    ensure_strictly_sorted(&planned.resident_tensor_ids, |index, previous, current| {
        StagePlanAdmissionError::PlannedTensorIdsNotStrictlySorted {
            index,
            previous,
            current,
        }
    })?;
    ensure_strictly_sorted(&realized.resident_tensor_ids, |index, previous, current| {
        StagePlanAdmissionError::RealizedTensorIdsNotStrictlySorted {
            index,
            previous,
            current,
        }
    })?;
    ensure_sidecars_sorted(&planned.sidecars, true)?;
    ensure_sidecars_sorted(&realized.sidecars, false)?;

    if planned.package_id != realized.package_id {
        return Err(StagePlanAdmissionError::PackageIdMismatch {
            planned: planned.package_id.clone(),
            realized: realized.package_id.clone(),
        });
    }
    if planned.plan_id != realized.plan_id {
        return Err(StagePlanAdmissionError::PlanIdMismatch {
            planned: planned.plan_id.clone(),
            realized: realized.plan_id.clone(),
        });
    }
    let planned_range = (planned.layer_start, planned.layer_end);
    let realized_range = (realized.layer_start, realized.layer_end);
    if planned_range != realized_range {
        return Err(StagePlanAdmissionError::LayerRangeMismatch {
            planned: planned_range,
            realized: realized_range,
        });
    }

    let (planned_only, realized_only) =
        diff_sorted(&planned.resident_tensor_ids, &realized.resident_tensor_ids);
    if !planned_only.is_empty() || !realized_only.is_empty() {
        return Err(StagePlanAdmissionError::TensorClosureMismatch {
            planned_only,
            realized_only,
        });
    }

    let (planned_only, realized_only) = diff_sorted(&planned.sidecars, &realized.sidecars);
    if !planned_only.is_empty() || !realized_only.is_empty() {
        return Err(StagePlanAdmissionError::SidecarMismatch {
            planned_only,
            realized_only,
        });
    }

    admit_profiles(planned, realized)?;

    // Full package admission resolution: recomputes the package id from
    // manifest content, binds every resident tensor through the v2 catalog,
    // and resolves typed sidecar references. Fails closed on any mismatch.
    manifest
        .resolve_stage_admission(
            &skippy_package_format::stage_admission::StageAdmissionDescriptor {
                package_id: realized.package_id.clone(),
                resident_tensor_ids: realized.resident_tensor_ids.clone(),
                sidecars: realized.sidecars.clone(),
            },
        )
        .map_err(StagePlanAdmissionError::ManifestResolution)?;

    Ok(AdmittedStage {
        package_id: realized.package_id.clone(),
        plan_id: realized.plan_id.clone(),
        layer_start: realized.layer_start,
        layer_end: realized.layer_end,
        resident_tensor_ids: realized.resident_tensor_ids.clone(),
    })
}

fn admit_profiles(
    planned: &PlannedStageAdmission,
    realized: &RealizedStagePlan,
) -> Result<(), StagePlanAdmissionError> {
    let planned_ids: Vec<&str> = planned
        .profiles
        .iter()
        .map(|p| p.profile_id.as_str())
        .collect();
    let realized_ids: Vec<&str> = realized
        .profiles
        .iter()
        .map(|p| p.profile_id.as_str())
        .collect();
    for window in planned_ids.windows(2) {
        if window[0] >= window[1] {
            return Err(StagePlanAdmissionError::PlannedProfilesNotSorted { index: 0 });
        }
    }
    for window in realized_ids.windows(2) {
        if window[0] >= window[1] {
            return Err(StagePlanAdmissionError::RealizedProfilesNotSorted { index: 0 });
        }
    }
    let (planned_only, realized_only) = diff_sorted(&planned_ids, &realized_ids);
    if !planned_only.is_empty() || !realized_only.is_empty() {
        return Err(StagePlanAdmissionError::ProfileSetMismatch {
            planned_only: planned_only.into_iter().map(str::to_string).collect(),
            realized_only: realized_only.into_iter().map(str::to_string).collect(),
        });
    }
    let planned_by_id: BTreeMap<&str, &PlannedStageProfile> = planned
        .profiles
        .iter()
        .map(|p| (p.profile_id.as_str(), p))
        .collect();
    for realized_profile in &realized.profiles {
        let planned_profile = planned_by_id
            .get(realized_profile.profile_id.as_str())
            .expect("profile set equality checked above");
        for (field, planned_value, realized_value) in [
            (
                "graph_identity",
                &planned_profile.graph_identity,
                &realized_profile.graph_identity,
            ),
            (
                "profile_identity",
                &planned_profile.profile_identity,
                &realized_profile.profile_identity,
            ),
            (
                "slice_identity",
                &planned_profile.slice_identity,
                &realized_profile.slice_identity,
            ),
            (
                "source_snapshot_identity",
                &planned_profile.source_snapshot_identity,
                &realized_profile.source_snapshot_identity,
            ),
            (
                "graph_configuration_id",
                &planned_profile.graph_configuration_id,
                &realized_profile.graph_configuration_id,
            ),
            (
                "backend_id",
                &planned_profile.backend_id,
                &realized_profile.backend_id,
            ),
        ] {
            if planned_value != realized_value {
                return Err(StagePlanAdmissionError::ProfileIdentityMismatch {
                    profile_id: realized_profile.profile_id.clone(),
                    field,
                    planned: planned_value.clone(),
                    realized: realized_value.clone(),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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
            native_abi_version: "0.1.51".to_string(),
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
}
