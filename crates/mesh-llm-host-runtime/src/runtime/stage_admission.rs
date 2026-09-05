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

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::ptr;

use anyhow::Context as _;
use skippy_package_format::{PackageManifest, Sidecar};

/// The planned admission expectations for one stage.
///
/// This is carried by the planner on `RuntimeSliceStagePlan` from planning
/// time and mirrored into the generation-8 control protocol descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedStageAdmission {
    /// Content-derived package identity (`sha256:...`).
    pub package_id: String,
    /// Deterministic native semantic plan identity (`skippy-plan:v1:...`).
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
    pub activation_imports: Vec<String>,
    pub activation_exports: Vec<String>,
    pub request_inputs: Vec<String>,
    pub state_effects: Vec<RealizedStageStateEffect>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RealizedStageStateEffect {
    pub identity: String,
    pub kind: skippy_ffi::StagePlanStateKind,
    pub access: skippy_ffi::StagePlanStateAccess,
    pub layer: i32,
    pub write_ordinal: i64,
}

/// Exact native planning profiles. The IDs must be strictly sorted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagePlannerProfile {
    pub profile_id: String,
    pub n_tokens: u32,
    pub n_sequences: u32,
    pub n_outputs: u32,
    pub n_recurrent_rollback_sequences: u32,
}

impl From<&RealizedStagePlan> for PlannedStageAdmission {
    fn from(realized: &RealizedStagePlan) -> Self {
        Self {
            package_id: realized.package_id.clone(),
            plan_id: realized.plan_id.clone(),
            layer_start: realized.layer_start,
            layer_end: realized.layer_end,
            resident_tensor_ids: realized.resident_tensor_ids.clone(),
            sidecars: realized.sidecars.clone(),
            profiles: realized
                .profiles
                .iter()
                .map(|profile| PlannedStageProfile {
                    profile_id: profile.profile_id.clone(),
                    graph_identity: profile.graph_identity.clone(),
                    profile_identity: profile.profile_identity.clone(),
                    slice_identity: profile.slice_identity.clone(),
                    source_snapshot_identity: profile.source_snapshot_identity.clone(),
                    graph_configuration_id: profile.graph_configuration_id.clone(),
                    backend_id: profile.backend_id.clone(),
                })
                .collect(),
        }
    }
}

impl From<&PlannedStageAdmission> for skippy_protocol::StageAdmissionDescriptor {
    fn from(planned: &PlannedStageAdmission) -> Self {
        Self {
            version: skippy_protocol::STAGE_ADMISSION_DESCRIPTOR_VERSION,
            package_id: planned.package_id.clone(),
            plan_id: planned.plan_id.clone(),
            layer_start: planned.layer_start,
            layer_end: planned.layer_end,
            resident_tensor_ids: planned.resident_tensor_ids.clone(),
            sidecars: planned
                .sidecars
                .iter()
                .map(|sidecar| skippy_protocol::StageAdmissionSidecar {
                    kind: match sidecar.kind {
                        skippy_package_format::SidecarKind::Mmproj => {
                            skippy_protocol::StageAdmissionSidecarKind::Mmproj
                        }
                    },
                    artifact_id: sidecar.artifact_id.clone(),
                    name: sidecar.name.clone(),
                })
                .collect(),
            profiles: planned
                .profiles
                .iter()
                .map(|profile| skippy_protocol::StageAdmissionProfile {
                    profile_id: profile.profile_id.clone(),
                    graph_identity: profile.graph_identity.clone(),
                    profile_identity: profile.profile_identity.clone(),
                    slice_identity: profile.slice_identity.clone(),
                    source_snapshot_identity: profile.source_snapshot_identity.clone(),
                    graph_configuration_id: profile.graph_configuration_id.clone(),
                    backend_id: profile.backend_id.clone(),
                })
                .collect(),
        }
    }
}

/// A native realized plan kept alive until the complete chain has been
/// validated through the public ABI.
struct NativePlan {
    raw: *mut skippy_ffi::StagePlan,
    realized: RealizedStagePlan,
}

impl Drop for NativePlan {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { skippy_ffi::skippy_stage_plan_free(self.raw) };
        }
    }
}

struct NativePlanner(*mut skippy_ffi::StagePlanner);

impl Drop for NativePlanner {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { skippy_ffi::skippy_stage_planner_free(self.0) };
        }
    }
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
                "realized resident tensor closure differs from plan: unrealized planned {planned_only:?}, unplanned realized {realized_only:?}"
            ),
            Self::SidecarMismatch {
                planned_only,
                realized_only,
            } => write!(
                formatter,
                "realized sidecars differ from plan: unrealized planned {planned_only:?}, unplanned realized {realized_only:?}"
            ),
            Self::ProfileSetMismatch {
                planned_only,
                realized_only,
            } => write!(
                formatter,
                "realized profile set differs from plan: unrealized planned {planned_only:?}, unplanned realized {realized_only:?}"
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

/// Open a package-v2 manifest and realize an exact native plan for every
/// requested stage. The native plan objects remain alive until the complete
/// chain passes `skippy_stage_plan_validate_chain_v1`.
pub fn realize_native_stage_chain(
    package_dir: &Path,
    ranges: &[(u32, u32)],
    profiles: &[StagePlannerProfile],
    graph_configuration_id: &str,
    backend_id: &str,
    sidecars_by_stage: &[Vec<Sidecar>],
) -> anyhow::Result<(PackageManifest, Vec<RealizedStagePlan>)> {
    anyhow::ensure!(!ranges.is_empty(), "stage plan chain is empty");
    anyhow::ensure!(
        ranges.len() == sidecars_by_stage.len(),
        "stage ranges and sidecar selections differ in length"
    );
    let manifest_path = package_dir.join("model-package.json");
    let manifest: PackageManifest = serde_json::from_slice(
        &std::fs::read(&manifest_path)
            .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse package-v2 manifest {}", manifest_path.display()))?;
    manifest
        .validate()
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("validate package-v2 manifest")?;
    let computed_package_id = manifest
        .computed_package_id()
        .context("compute package-v2 identity")?;
    anyhow::ensure!(
        manifest.package_id == computed_package_id,
        "package-v2 manifest package_id does not match its content"
    );

    let inputs = PlannerInputs::new(
        package_dir,
        &manifest,
        profiles,
        graph_configuration_id,
        backend_id,
    )?;
    let planner = inputs.create_native_planner()?;
    let mut plans = Vec::with_capacity(ranges.len());
    for ((layer_start, layer_end), sidecars) in ranges.iter().zip(sidecars_by_stage) {
        plans.push(realize_native_plan(
            &planner,
            *layer_start,
            *layer_end,
            sidecars.clone(),
        )?);
    }
    let raw_plans = plans
        .iter()
        .map(|plan| plan.raw.cast_const())
        .collect::<Vec<_>>();
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_plan_validate_chain_v1(
            raw_plans.as_ptr(),
            raw_plans.len(),
            &mut error,
        )
    };
    ffi_result(status, error).context("validate native stage-plan chain")?;
    Ok((
        manifest,
        plans
            .into_iter()
            .map(|plan| plan.realized.clone())
            .collect(),
    ))
}

/// Realize, package-resolve, and admit every stage before a topology can be
/// published. Returned descriptors are canonical generation-8 wire values.
pub fn realize_stage_admissions(
    package_dir: &Path,
    ranges: &[(u32, u32)],
    profiles: &[StagePlannerProfile],
    graph_configuration_id: &str,
    backend_id: &str,
) -> anyhow::Result<Vec<skippy_protocol::StageAdmissionDescriptor>> {
    let manifest_bytes = std::fs::read(package_dir.join("model-package.json"))
        .context("read package-v2 manifest for sidecar assignment")?;
    let manifest: PackageManifest =
        serde_json::from_slice(&manifest_bytes).context("parse package-v2 manifest")?;
    let sidecars_by_stage = ranges
        .iter()
        .enumerate()
        .map(|(index, _)| {
            if index == 0 {
                manifest.sidecars.clone()
            } else {
                Vec::new()
            }
        })
        .collect::<Vec<_>>();
    let (manifest, realized) = realize_native_stage_chain(
        package_dir,
        ranges,
        profiles,
        graph_configuration_id,
        backend_id,
        &sidecars_by_stage,
    )?;
    realized
        .iter()
        .map(|realized| {
            let planned = PlannedStageAdmission::from(realized);
            admit_stage_plan(&planned, realized, &manifest).context("admit native stage plan")?;
            Ok(skippy_protocol::StageAdmissionDescriptor::from(&planned))
        })
        .collect()
}

struct PlannerInputs {
    package_id: CString,
    shard_paths: Vec<CString>,
    _tensor_ids: Vec<CString>,
    _tensor_names: Vec<CString>,
    tensors: Vec<skippy_ffi::StagePlannerTensorV1>,
    _profile_ids: Vec<CString>,
    profiles: Vec<skippy_ffi::StagePlannerProfileV1>,
    graph_configuration_id: CString,
    backend_id: CString,
}

impl PlannerInputs {
    fn new(
        package_dir: &Path,
        manifest: &PackageManifest,
        profiles: &[StagePlannerProfile],
        graph_configuration_id: &str,
        backend_id: &str,
    ) -> anyhow::Result<Self> {
        let artifact_by_id = manifest
            .artifact_catalog
            .entries
            .iter()
            .map(|artifact| (artifact.id.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();
        let mut source_artifacts = artifact_by_id
            .iter()
            .filter_map(|(id, artifact)| source_artifact_index(id).map(|index| (index, *artifact)))
            .collect::<Vec<_>>();
        source_artifacts.sort_by_key(|(index, _)| *index);
        anyhow::ensure!(
            !source_artifacts.is_empty()
                && source_artifacts
                    .iter()
                    .enumerate()
                    .all(|(expected, (actual, _))| expected == *actual),
            "package-v2 source artifacts are not a contiguous source-00000 shard set"
        );
        let mut shard_paths = Vec::with_capacity(source_artifacts.len());
        let mut shard_index_by_artifact = BTreeMap::new();
        for (index, artifact) in &source_artifacts {
            let path = contained_package_path(package_dir, &artifact.path)?;
            anyhow::ensure!(
                path.is_file(),
                "package-v2 artifact is missing: {}",
                path.display()
            );
            shard_paths.push(path_cstring(&path, "package-v2 shard path")?);
            shard_index_by_artifact.insert(artifact.id.as_str(), *index);
        }

        let tensor_by_id = manifest
            .tensor_catalog
            .entries
            .iter()
            .map(|tensor| (tensor.id.as_str(), tensor))
            .collect::<BTreeMap<_, _>>();
        anyhow::ensure!(
            tensor_by_id.len() == manifest.tensor_catalog.entries.len(),
            "package-v2 tensor IDs are duplicated"
        );
        let ordered_tensors = tensor_by_id.values().copied().collect::<Vec<_>>();
        let tensor_ids = ordered_tensors
            .iter()
            .map(|tensor| cstring(&tensor.id, "tensor ID"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tensor_names = ordered_tensors
            .iter()
            .map(|tensor| cstring(&tensor.name, "native tensor name"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut tensors = Vec::with_capacity(ordered_tensors.len());
        for (index, tensor) in ordered_tensors.iter().enumerate() {
            let (artifact_id, data_offset, stored_length) =
                tensor_storage(manifest, &tensor_by_id, &tensor.id)?;
            let split_no = *shard_index_by_artifact.get(artifact_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "tensor {:?} resolves to non-source artifact {:?}",
                    tensor.id,
                    artifact_id
                )
            })?;
            let mut dimensions = [0_i64; skippy_ffi::STAGE_PLAN_MAX_DIMS];
            anyhow::ensure!(
                !tensor.dimensions.is_empty()
                    && tensor.dimensions.len() <= skippy_ffi::STAGE_PLAN_MAX_DIMS,
                "tensor {:?} has unsupported rank {}",
                tensor.id,
                tensor.dimensions.len()
            );
            for (destination, source) in dimensions.iter_mut().zip(&tensor.dimensions) {
                *destination = i64::try_from(*source)
                    .with_context(|| format!("tensor {:?} dimension exceeds i64", tensor.id))?;
                anyhow::ensure!(
                    *destination > 0,
                    "tensor {:?} has an empty dimension",
                    tensor.id
                );
            }
            tensors.push(skippy_ffi::StagePlannerTensorV1 {
                abi_version: skippy_ffi::STAGE_PLANNER_TENSOR_V1_ABI_VERSION,
                struct_size: u32::try_from(std::mem::size_of::<skippy_ffi::StagePlannerTensorV1>())
                    .expect("stage planner tensor descriptor size fits u32"),
                tensor_id: tensor_ids[index].as_ptr(),
                native_name: tensor_names[index].as_ptr(),
                ggml_type: i32::try_from(tensor.ggml_type)
                    .context("GGML tensor type exceeds i32")?,
                dimension_count: u32::try_from(tensor.dimensions.len())
                    .expect("validated tensor rank fits u32"),
                dimensions,
                split_no: u32::try_from(split_no).context("shard index exceeds u32")?,
                reserved: 0,
                data_offset,
                stored_length,
            });
        }

        let mut previous_profile = None;
        let profile_ids = profiles
            .iter()
            .map(|profile| {
                if previous_profile
                    .is_some_and(|previous: &str| previous >= profile.profile_id.as_str())
                {
                    anyhow::bail!("stage planner profile IDs are not strictly sorted");
                }
                previous_profile = Some(profile.profile_id.as_str());
                cstring(&profile.profile_id, "profile ID")
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        anyhow::ensure!(
            !profile_ids.is_empty(),
            "stage planner profile set is empty"
        );
        let profiles = profiles
            .iter()
            .zip(&profile_ids)
            .map(|(profile, id)| skippy_ffi::StagePlannerProfileV1 {
                abi_version: skippy_ffi::STAGE_PLANNER_PROFILE_V1_ABI_VERSION,
                struct_size:
                    u32::try_from(std::mem::size_of::<skippy_ffi::StagePlannerProfileV1>())
                        .expect("stage planner profile descriptor size fits u32"),
                profile_id: id.as_ptr(),
                n_tokens: profile.n_tokens,
                n_sequences: profile.n_sequences,
                n_outputs: profile.n_outputs,
                n_recurrent_rollback_sequences: profile.n_recurrent_rollback_sequences,
            })
            .collect();
        Ok(Self {
            package_id: cstring(&manifest.package_id, "package ID")?,
            shard_paths,
            _tensor_ids: tensor_ids,
            _tensor_names: tensor_names,
            tensors,
            _profile_ids: profile_ids,
            profiles,
            graph_configuration_id: cstring(graph_configuration_id, "graph configuration ID")?,
            backend_id: cstring(backend_id, "backend ID")?,
        })
    }

    fn create_native_planner(&self) -> anyhow::Result<NativePlanner> {
        let shard_path_ptrs = self
            .shard_paths
            .iter()
            .map(|path| path.as_ptr())
            .collect::<Vec<_>>();
        let config = skippy_ffi::StagePlannerConfigV1 {
            abi_version: skippy_ffi::STAGE_PLANNER_CONFIG_V1_ABI_VERSION,
            struct_size: u32::try_from(std::mem::size_of::<skippy_ffi::StagePlannerConfigV1>())
                .expect("stage planner config size fits u32"),
            package_id: self.package_id.as_ptr(),
            shard_paths: shard_path_ptrs.as_ptr(),
            shard_count: shard_path_ptrs.len(),
            tensors: self.tensors.as_ptr(),
            tensor_count: self.tensors.len(),
            profiles: self.profiles.as_ptr(),
            profile_count: self.profiles.len(),
            graph_configuration_id: self.graph_configuration_id.as_ptr(),
            backend_id: self.backend_id.as_ptr(),
        };
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_stage_planner_create_v1(&config, &mut raw, &mut error) };
        ffi_result(status, error).context("create native stage planner")?;
        anyhow::ensure!(
            !raw.is_null(),
            "native stage planner returned a null handle"
        );
        Ok(NativePlanner(raw))
    }
}

fn realize_native_plan(
    planner: &NativePlanner,
    layer_start: u32,
    layer_end: u32,
    sidecars: Vec<Sidecar>,
) -> anyhow::Result<NativePlan> {
    let layer_start_i32 = i32::try_from(layer_start).context("stage layer start exceeds i32")?;
    let layer_end_i32 = i32::try_from(layer_end).context("stage layer end exceeds i32")?;
    let mut raw = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_planner_realize_v1(
            planner.0,
            layer_start_i32,
            layer_end_i32,
            &mut raw,
            &mut error,
        )
    };
    if let Err(error) = ffi_result(status, error).context("realize native stage plan") {
        if !raw.is_null() {
            unsafe { skippy_ffi::skippy_stage_plan_free(raw) };
        }
        return Err(error);
    }
    anyhow::ensure!(
        !raw.is_null(),
        "native stage realization returned a null plan"
    );

    let realized = describe_native_plan(raw, sidecars).inspect_err(|_| unsafe {
        skippy_ffi::skippy_stage_plan_free(raw);
    })?;
    anyhow::ensure!(
        realized.layer_start == layer_start && realized.layer_end == layer_end,
        "native stage descriptor range {}..{} differs from requested {}..{}",
        realized.layer_start,
        realized.layer_end,
        layer_start,
        layer_end
    );
    Ok(NativePlan { raw, realized })
}

fn describe_native_plan(
    raw: *const skippy_ffi::StagePlan,
    sidecars: Vec<Sidecar>,
) -> anyhow::Result<RealizedStagePlan> {
    let mut descriptor = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanDescV1>() };
    let mut error = ptr::null_mut();
    let status =
        unsafe { skippy_ffi::skippy_stage_plan_describe_v1(raw, &mut descriptor, &mut error) };
    ffi_result(status, error).context("describe native stage plan")?;
    ensure_descriptor_abi(
        "stage plan",
        descriptor.abi_version,
        skippy_ffi::STAGE_PLAN_DESC_V1_ABI_VERSION,
        descriptor.struct_size,
        std::mem::size_of::<skippy_ffi::StagePlanDescV1>(),
    )?;
    anyhow::ensure!(
        descriptor.layer_count > 0,
        "native stage plan has no layers"
    );
    anyhow::ensure!(
        descriptor.layer_start >= 0
            && descriptor.layer_end > descriptor.layer_start
            && descriptor.layer_end <= descriptor.layer_count,
        "native stage plan has invalid layer range {}..{} for {} layers",
        descriptor.layer_start,
        descriptor.layer_end,
        descriptor.layer_count
    );

    let resident_count = usize::try_from(descriptor.resident_tensor_count)
        .context("native resident tensor count exceeds usize")?;
    let mut resident_tensor_ids = Vec::with_capacity(resident_count);
    for index in 0..resident_count {
        let mut value = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanValueDescV1>() };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_stage_plan_resident_tensor_at_v1(raw, index, &mut value, &mut error)
        };
        ffi_result(status, error)
            .with_context(|| format!("read native resident tensor {index}"))?;
        ensure_descriptor_abi(
            "stage plan resident tensor",
            value.abi_version,
            skippy_ffi::STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION,
            value.struct_size,
            std::mem::size_of::<skippy_ffi::StagePlanValueDescV1>(),
        )?;
        resident_tensor_ids.push(read_plan_string(raw, value.identity)?);
    }
    ensure_canonical_strings("native resident tensor IDs", &resident_tensor_ids)?;

    let profile_count =
        usize::try_from(descriptor.profile_count).context("native profile count exceeds usize")?;
    anyhow::ensure!(
        profile_count > 0,
        "native stage plan has no guarded profiles"
    );
    let mut profiles = Vec::with_capacity(profile_count);
    for profile_index in 0..profile_count {
        profiles.push(read_native_profile(raw, profile_index)?);
    }
    ensure_canonical_strings(
        "native profile IDs",
        &profiles
            .iter()
            .map(|profile| profile.profile_id.clone())
            .collect::<Vec<_>>(),
    )?;

    for (index, window) in sidecars.windows(2).enumerate() {
        anyhow::ensure!(
            window[0] < window[1],
            "native stage host sidecars are not strictly sorted at index {index}"
        );
    }
    let package_id = read_plan_string(raw, descriptor.package_id)?;
    let plan_id = read_plan_string(raw, descriptor.plan_id)?;
    anyhow::ensure!(
        package_id
            .strip_prefix("sha256:")
            .is_some_and(is_lower_hex_digest),
        "native stage package identity is not canonical"
    );
    anyhow::ensure!(
        plan_id
            .strip_prefix("skippy-plan:v1:")
            .is_some_and(is_lower_hex_digest),
        "native stage plan identity is not canonical"
    );
    Ok(RealizedStagePlan {
        package_id,
        plan_id,
        layer_start: u32::try_from(descriptor.layer_start)
            .expect("validated nonnegative native layer start"),
        layer_end: u32::try_from(descriptor.layer_end)
            .expect("validated positive native layer end"),
        resident_tensor_ids,
        sidecars,
        profiles,
    })
}

fn read_native_profile(
    raw: *const skippy_ffi::StagePlan,
    profile_index: usize,
) -> anyhow::Result<RealizedStageProfile> {
    let mut descriptor = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanProfileDescV1>() };
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_plan_profile_at_v1(raw, profile_index, &mut descriptor, &mut error)
    };
    ffi_result(status, error)
        .with_context(|| format!("read native stage profile {profile_index}"))?;
    ensure_descriptor_abi(
        "stage plan profile",
        descriptor.abi_version,
        skippy_ffi::STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION,
        descriptor.struct_size,
        std::mem::size_of::<skippy_ffi::StagePlanProfileDescV1>(),
    )?;
    anyhow::ensure!(
        descriptor.n_tokens > 0
            && descriptor.n_sequences > 0
            && descriptor.n_outputs > 0
            && descriptor.n_tokens % descriptor.n_sequences == 0
            && descriptor.n_outputs <= descriptor.n_tokens,
        "native stage profile {profile_index} has an invalid execution guard"
    );

    let activation_imports = read_native_values(
        raw,
        profile_index,
        skippy_ffi::StagePlanValueKind::ActivationImport,
        descriptor.activation_import_count,
    )?;
    let activation_exports = read_native_values(
        raw,
        profile_index,
        skippy_ffi::StagePlanValueKind::ActivationExport,
        descriptor.activation_export_count,
    )?;
    let request_inputs = read_native_values(
        raw,
        profile_index,
        skippy_ffi::StagePlanValueKind::RequestInput,
        descriptor.request_input_count,
    )?;
    let state_count = usize::try_from(descriptor.state_effect_count)
        .context("native state effect count exceeds usize")?;
    let mut state_effects = Vec::with_capacity(state_count);
    let mut state_identities = BTreeSet::new();
    for index in 0..state_count {
        let mut state = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanStateDescV1>() };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_stage_plan_state_at_v1(
                raw,
                profile_index,
                index,
                &mut state,
                &mut error,
            )
        };
        ffi_result(status, error).with_context(|| {
            format!("read native state effect {index} for profile {profile_index}")
        })?;
        ensure_descriptor_abi(
            "stage plan state effect",
            state.abi_version,
            skippy_ffi::STAGE_PLAN_STATE_DESC_V1_ABI_VERSION,
            state.struct_size,
            std::mem::size_of::<skippy_ffi::StagePlanStateDescV1>(),
        )?;
        anyhow::ensure!(
            state.reserved == 0,
            "native state effect reserved field is nonzero"
        );
        let identity = read_plan_string(raw, state.identity)?;
        anyhow::ensure!(
            state_identities.insert(identity.clone()),
            "native state effect identity {identity:?} is duplicated"
        );
        let kind = match state.kind {
            value if value == skippy_ffi::StagePlanStateKind::KvKey as i32 => {
                skippy_ffi::StagePlanStateKind::KvKey
            }
            value if value == skippy_ffi::StagePlanStateKind::KvValue as i32 => {
                skippy_ffi::StagePlanStateKind::KvValue
            }
            value if value == skippy_ffi::StagePlanStateKind::RecurrentConv as i32 => {
                skippy_ffi::StagePlanStateKind::RecurrentConv
            }
            value if value == skippy_ffi::StagePlanStateKind::RecurrentSsm as i32 => {
                skippy_ffi::StagePlanStateKind::RecurrentSsm
            }
            unknown => anyhow::bail!("native state effect kind {unknown} is unsupported"),
        };
        let access = match state.access {
            value if value == skippy_ffi::StagePlanStateAccess::Read as i32 => {
                skippy_ffi::StagePlanStateAccess::Read
            }
            value if value == skippy_ffi::StagePlanStateAccess::Write as i32 => {
                skippy_ffi::StagePlanStateAccess::Write
            }
            unknown => anyhow::bail!("native state effect access {unknown} is unsupported"),
        };
        state_effects.push(RealizedStageStateEffect {
            identity,
            kind,
            access,
            layer: state.layer,
            write_ordinal: state.write_ordinal,
        });
    }

    Ok(RealizedStageProfile {
        profile_id: read_plan_string(raw, descriptor.profile_id)?,
        graph_identity: read_plan_string(raw, descriptor.graph_identity)?,
        profile_identity: read_plan_string(raw, descriptor.profile_identity)?,
        slice_identity: read_plan_string(raw, descriptor.slice_identity)?,
        source_snapshot_identity: read_plan_string(raw, descriptor.source_snapshot_identity)?,
        graph_configuration_id: read_plan_string(raw, descriptor.graph_configuration_id)?,
        backend_id: read_plan_string(raw, descriptor.backend_id)?,
        activation_imports,
        activation_exports,
        request_inputs,
        state_effects,
    })
}

fn read_native_values(
    raw: *const skippy_ffi::StagePlan,
    profile_index: usize,
    kind: skippy_ffi::StagePlanValueKind,
    count: u64,
) -> anyhow::Result<Vec<String>> {
    let count = usize::try_from(count).context("native stage value count exceeds usize")?;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let mut descriptor = unsafe { std::mem::zeroed::<skippy_ffi::StagePlanValueDescV1>() };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_stage_plan_value_at_v1(
                raw,
                profile_index,
                kind,
                index,
                &mut descriptor,
                &mut error,
            )
        };
        ffi_result(status, error).with_context(|| {
            format!("read native {kind:?} value {index} for profile {profile_index}")
        })?;
        ensure_descriptor_abi(
            "stage plan value",
            descriptor.abi_version,
            skippy_ffi::STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION,
            descriptor.struct_size,
            std::mem::size_of::<skippy_ffi::StagePlanValueDescV1>(),
        )?;
        values.push(read_plan_string(raw, descriptor.identity)?);
    }
    ensure_unique_strings(&format!("native {kind:?} identities"), &values)?;
    Ok(values)
}

fn read_plan_string(
    raw: *const skippy_ffi::StagePlan,
    reference: skippy_ffi::StagePlanStringRefV1,
) -> anyhow::Result<String> {
    let mut data = ptr::null();
    let mut length = 0;
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_stage_plan_string_v1(raw, reference, &mut data, &mut length, &mut error)
    };
    ffi_result(status, error).context("read native stage-plan string")?;
    anyhow::ensure!(length > 0, "native stage-plan string is empty");
    anyhow::ensure!(!data.is_null(), "native stage-plan string data is null");
    let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), length) };
    Ok(std::str::from_utf8(bytes)
        .context("native stage-plan string is not UTF-8")?
        .to_owned())
}

fn ensure_descriptor_abi(
    label: &str,
    actual_version: u32,
    expected_version: u32,
    actual_size: u32,
    expected_size: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        actual_version == expected_version,
        "native {label} ABI version {actual_version} differs from expected {expected_version}"
    );
    anyhow::ensure!(
        usize::try_from(actual_size).ok() == Some(expected_size),
        "native {label} size {actual_size} differs from expected {expected_size}"
    );
    Ok(())
}

fn ensure_canonical_strings(label: &str, values: &[String]) -> anyhow::Result<()> {
    for (index, window) in values.windows(2).enumerate() {
        anyhow::ensure!(
            window[0] < window[1],
            "{label} are not strictly sorted at index {index}"
        );
    }
    Ok(())
}

fn ensure_unique_strings(label: &str, values: &[String]) -> anyhow::Result<()> {
    let mut unique = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        anyhow::ensure!(
            unique.insert(value),
            "{label} contain a duplicate at index {index}: {value:?}"
        );
    }
    Ok(())
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn ffi_result(status: skippy_ffi::Status, error: *mut skippy_ffi::Error) -> anyhow::Result<()> {
    let message = if error.is_null() {
        String::new()
    } else {
        let message = unsafe { (*error).message };
        if message.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        }
    };
    if !error.is_null() {
        unsafe { skippy_ffi::skippy_error_free(error) };
    }
    anyhow::ensure!(
        status == skippy_ffi::Status::Ok,
        "native stage planner returned {status:?}: {message}"
    );
    Ok(())
}

fn source_artifact_index(id: &str) -> Option<usize> {
    id.strip_prefix("source-")?.parse().ok()
}

fn tensor_storage<'a>(
    manifest: &'a PackageManifest,
    tensors: &BTreeMap<&str, &'a skippy_package_format::Tensor>,
    tensor_id: &str,
) -> anyhow::Result<(&'a str, u64, u64)> {
    let mut current = tensor_id;
    let mut visited = BTreeSet::new();
    loop {
        anyhow::ensure!(visited.insert(current), "tensor alias cycle at {current:?}");
        let tensor = tensors
            .get(current)
            .with_context(|| format!("tensor alias target {current:?} is absent"))?;
        match &tensor.storage {
            skippy_package_format::TensorStorage::Owned {
                artifact_id,
                data_offset,
                stored_length,
                ..
            } => {
                anyhow::ensure!(
                    manifest
                        .artifact_catalog
                        .entries
                        .iter()
                        .any(|artifact| artifact.id == *artifact_id),
                    "tensor {tensor_id:?} references missing artifact {artifact_id:?}"
                );
                return Ok((artifact_id, *data_offset, *stored_length));
            }
            skippy_package_format::TensorStorage::Alias { target_tensor_id } => {
                current = target_tensor_id;
            }
        }
    }
}

fn contained_package_path(package_dir: &Path, relative: &str) -> anyhow::Result<PathBuf> {
    let relative = Path::new(relative);
    anyhow::ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "package-v2 artifact path is not a safe relative path: {relative:?}"
    );
    Ok(package_dir.join(relative))
}

fn cstring(value: &str, label: &str) -> anyhow::Result<CString> {
    anyhow::ensure!(!value.is_empty(), "{label} is empty");
    CString::new(value).with_context(|| format!("{label} contains an interior NUL byte"))
}

fn path_cstring(path: &Path, label: &str) -> anyhow::Result<CString> {
    let value = path
        .to_str()
        .with_context(|| format!("{label} is not valid UTF-8: {}", path.display()))?;
    cstring(value, label)
}

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
    }
}
