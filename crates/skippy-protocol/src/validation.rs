//! Stage protocol constants and wire-frame validation.

use crate::proto;
pub const SCHEMA_VERSION: u32 = 1;
pub const STAGE_ALPN_V2: &[u8] = b"skippy-stage/2";
pub const STAGE_SUBPROTOCOL_NAME: &str = "skippy-stage";
pub const STAGE_SUBPROTOCOL_MAJOR: u32 = 2;
pub const STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL: &str = "stage-control";
pub const STAGE_PROTOCOL_GENERATION: u32 = 8;
/// Generation-scoped stage capability. A peer can advertise `stage-control`
/// while still rejecting current-generation frames, so split planning gates on
/// this exact token before sending current-generation control requests.
pub const STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V8: &str = "stage-generation-8";
pub const STAGE_SUBPROTOCOL_FEATURE_STAGE_GENERATION: &str =
    STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V8;
pub const STAGE_SUBPROTOCOL_FEATURE_ARTIFACT_TRANSFER: &str = "artifact-transfer";
pub const STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST: &str = "status-list";
pub const STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1: &str = "local-gguf-content-id-v1";
pub const STAGE_STREAM_CONTROL: u8 = 0x01;
pub const STAGE_STREAM_TRANSPORT: u8 = 0x02;
pub const STAGE_STREAM_ARTIFACT_TRANSFER: u8 = 0x03;
pub const MAX_STAGE_FRAME_BYTES: usize = 8 * 1024 * 1024;
/// Maximum number of unresolved verify windows covered by native checkpoints.
pub const MAX_VERIFY_WINDOW_PIPELINE_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageFrameError {
    BadGeneration { got: u32 },
    InvalidEndpointId { got: usize },
    InvalidArtifactDigestLength { got: usize },
    InvalidSourceDigestLength { got: usize },
    MissingRequiredSourceDigest,
    LocalSourcePolicyRequired,
    LocalSourceCommandRequired,
    InvalidLocalSourceLoadMode { got: i32 },
    InvalidLocalSourceReference,
    InvalidSourceResolutionPolicy { got: i32 },
    InvalidArtifactPath,
    InvalidArtifactOffset,
    MissingStageControlCommand,
    MissingStageControlResponse,
    MissingStageAdmissionDescriptor,
    MissingLoadClaimHashes,
    InvalidActivationCodec { got: i32 },
    InvalidTopologyStages(&'static str),
    InvalidStageAdmissionDescriptor(&'static str),
    MissingStageTransportTarget,
    MissingStageArtifactTarget,
}

impl std::fmt::Display for StageFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageFrameError::BadGeneration { got } => write!(
                f,
                "bad skippy stage generation: expected {}, got {}",
                STAGE_PROTOCOL_GENERATION, got
            ),
            StageFrameError::InvalidEndpointId { got } => {
                write!(f, "invalid endpoint_id length: expected 32, got {got}")
            }
            StageFrameError::InvalidArtifactDigestLength { got } => write!(
                f,
                "invalid artifact sha256 length: expected 64 hex chars, got {got}"
            ),
            StageFrameError::InvalidSourceDigestLength { got } => write!(
                f,
                "invalid source model sha256 length: expected 64 lowercase hex chars, got {got}"
            ),
            StageFrameError::MissingRequiredSourceDigest => {
                write!(
                    f,
                    "local-required source resolution requires a source model sha256"
                )
            }
            StageFrameError::LocalSourcePolicyRequired => {
                write!(f, "strict local load requires local-required source policy")
            }
            StageFrameError::LocalSourceCommandRequired => {
                write!(
                    f,
                    "local-required source resolution requires the fail-closed local command"
                )
            }
            StageFrameError::InvalidLocalSourceLoadMode { got } => write!(
                f,
                "local-required source resolution requires RuntimeSlice load mode, got {got}"
            ),
            StageFrameError::InvalidLocalSourceReference => {
                write!(
                    f,
                    "strict local load requires a content-addressed GGUF reference"
                )
            }
            StageFrameError::InvalidSourceResolutionPolicy { got } => {
                write!(f, "unsupported source resolution policy {got}")
            }
            StageFrameError::InvalidArtifactPath => {
                write!(f, "artifact relative_path must be a safe relative path")
            }
            StageFrameError::InvalidArtifactOffset => {
                write!(f, "artifact offset exceeds expected artifact size")
            }
            StageFrameError::MissingStageControlCommand => {
                write!(f, "stage control command is required but missing")
            }
            StageFrameError::MissingStageControlResponse => {
                write!(f, "stage control response is required but missing")
            }
            StageFrameError::MissingStageAdmissionDescriptor => {
                write!(
                    f,
                    "generation 8 stage load/status requires an admission descriptor"
                )
            }
            StageFrameError::MissingLoadClaimHashes => {
                write!(
                    f,
                    "generation 8 stage load requires participant and topology hashes"
                )
            }
            StageFrameError::InvalidActivationCodec { got } => {
                write!(f, "unsupported generation-8 activation codec {got}")
            }
            StageFrameError::InvalidTopologyStages(reason) => {
                write!(f, "invalid generation-8 topology stage list: {reason}")
            }
            StageFrameError::InvalidStageAdmissionDescriptor(reason) => {
                write!(f, "invalid stage admission descriptor: {reason}")
            }
            StageFrameError::MissingStageTransportTarget => {
                write!(f, "stage transport target is required but missing")
            }
            StageFrameError::MissingStageArtifactTarget => {
                write!(f, "stage artifact transfer target is required but missing")
            }
        }
    }
}

impl std::error::Error for StageFrameError {}

pub fn validate_stage_control_request(
    frame: &proto::stage::StageControlRequest,
) -> Result<(), StageFrameError> {
    validate_generation(frame.r#gen)?;
    validate_endpoint_id(frame.requester_id.len())?;
    if frame.command.is_none() {
        return Err(StageFrameError::MissingStageControlCommand);
    }
    use proto::stage::stage_control_request::Command;
    match frame.command.as_ref() {
        Some(Command::LoadStage(load)) => {
            validate_load_stage_admission(load)?;
            reject_local_source_in_legacy_command(load)?;
            validate_source_resolution(
                load.source_model_sha256.as_deref(),
                load.source_resolution_policy,
            )?;
        }
        Some(Command::LoadLocalStage(load)) => {
            validate_load_stage_admission(load)?;
            validate_local_source_load(load)?;
        }
        Some(Command::GetLayerInventory(inventory)) => validate_source_resolution(
            inventory.expected_source_model_sha256.as_deref(),
            inventory.source_resolution_policy,
        )?,
        Some(Command::PrepareStage(prepare)) => {
            let load = prepare
                .load_stage
                .as_ref()
                .ok_or(StageFrameError::MissingStageAdmissionDescriptor)?;
            validate_load_stage_admission(load)?;
            reject_local_source_in_legacy_command(load)?;
            validate_source_resolution(
                load.source_model_sha256.as_deref(),
                load.source_resolution_policy,
            )?;
        }
        Some(Command::StageStatusUpdate(update)) => {
            let status = update
                .status
                .as_ref()
                .ok_or(StageFrameError::MissingStageAdmissionDescriptor)?;
            validate_preparation_stage_admission(status)?;
        }
        _ => {}
    }
    Ok(())
}

fn reject_local_source_in_legacy_command(
    load: &proto::stage::LoadStage,
) -> Result<(), StageFrameError> {
    if load.source_resolution_policy == proto::stage::SourceResolutionPolicy::LocalRequired as i32
        || load.package_ref.starts_with("local-gguf://sha256/")
    {
        return Err(StageFrameError::LocalSourceCommandRequired);
    }
    Ok(())
}

fn validate_local_source_load(load: &proto::stage::LoadStage) -> Result<(), StageFrameError> {
    if load.source_resolution_policy != proto::stage::SourceResolutionPolicy::LocalRequired as i32 {
        return Err(StageFrameError::LocalSourcePolicyRequired);
    }
    let Some(reference_digest) = load.package_ref.strip_prefix("local-gguf://sha256/") else {
        return Err(StageFrameError::InvalidLocalSourceReference);
    };
    if validate_source_digest(reference_digest).is_err() {
        return Err(StageFrameError::InvalidLocalSourceReference);
    }
    if load.load_mode != proto::stage::StageLoadMode::RuntimeSlice as i32 {
        return Err(StageFrameError::InvalidLocalSourceLoadMode {
            got: load.load_mode,
        });
    }
    validate_source_resolution(
        load.source_model_sha256.as_deref(),
        load.source_resolution_policy,
    )?;
    if load.source_model_sha256.as_deref() != Some(reference_digest) {
        return Err(StageFrameError::InvalidLocalSourceReference);
    }
    Ok(())
}

pub fn validate_stage_control_response(
    frame: &proto::stage::StageControlResponse,
) -> Result<(), StageFrameError> {
    validate_generation(frame.r#gen)?;
    if frame.response.is_none() {
        return Err(StageFrameError::MissingStageControlResponse);
    }
    if let Some(proto::stage::stage_control_response::Response::LayerInventory(inventory)) =
        frame.response.as_ref()
        && let Some(sha256) = inventory.source_model_sha256.as_deref()
    {
        validate_source_digest(sha256)?;
    }
    use proto::stage::stage_control_response::Response;
    match frame.response.as_ref() {
        Some(Response::StageReady(ready)) => {
            let status = ready
                .status
                .as_ref()
                .ok_or(StageFrameError::MissingStageAdmissionDescriptor)?;
            validate_status_stage_admission(status)?;
        }
        Some(Response::StageStatuses(statuses)) => {
            for status in &statuses.statuses {
                validate_status_stage_admission(status)?;
            }
        }
        Some(Response::PrepareStageAccepted(accepted)) => {
            let status = accepted
                .status
                .as_ref()
                .ok_or(StageFrameError::MissingStageAdmissionDescriptor)?;
            validate_preparation_stage_admission(status)?;
        }
        Some(Response::StagePreparationStatus(status)) => {
            validate_preparation_stage_admission(status)?;
        }
        _ => {}
    }
    Ok(())
}

fn validate_load_stage_admission(load: &proto::stage::LoadStage) -> Result<(), StageFrameError> {
    if load.participant_set_hash.is_empty() || load.topology_hash.is_empty() {
        return Err(StageFrameError::MissingLoadClaimHashes);
    }
    let admission = load
        .admission
        .as_ref()
        .ok_or(StageFrameError::MissingStageAdmissionDescriptor)?;
    validate_stage_admission_descriptor(admission)?;
    if admission.layer_start != load.layer_start || admission.layer_end != load.layer_end {
        return Err(StageFrameError::InvalidStageAdmissionDescriptor(
            "descriptor layer range does not match load range",
        ));
    }
    validate_activation_codec(load.activation_codec)?;
    validate_topology_stages(load)?;
    Ok(())
}

fn validate_topology_stages(load: &proto::stage::LoadStage) -> Result<(), StageFrameError> {
    if load.topology_stages.is_empty() {
        return Err(StageFrameError::InvalidTopologyStages(
            "canonical stage list is required",
        ));
    }
    let mut current_matches = 0usize;
    let mut previous_end = None;
    for (position, stage) in load.topology_stages.iter().enumerate() {
        if stage.stage_index as usize != position {
            return Err(StageFrameError::InvalidTopologyStages(
                "stage indexes must be ordered and contiguous from zero",
            ));
        }
        if stage.stage_id.is_empty() || stage.bind_addr.is_empty() {
            return Err(StageFrameError::InvalidTopologyStages(
                "stage id and bind address are required",
            ));
        }
        if load.topology_stages[..position]
            .iter()
            .any(|prior| prior.stage_id == stage.stage_id)
        {
            return Err(StageFrameError::InvalidTopologyStages(
                "stage ids must be unique",
            ));
        }
        if load.topology_stages[..position]
            .iter()
            .any(|prior| prior.node_id == stage.node_id)
        {
            return Err(StageFrameError::InvalidTopologyStages(
                "stage node ids must be unique",
            ));
        }
        validate_endpoint_id(stage.node_id.len())?;
        if stage.layer_start >= stage.layer_end {
            return Err(StageFrameError::InvalidTopologyStages(
                "stage layer ranges must be non-empty",
            ));
        }
        if previous_end.is_some_and(|end| end != stage.layer_start) {
            return Err(StageFrameError::InvalidTopologyStages(
                "stage layer ranges must be contiguous",
            ));
        }
        previous_end = Some(stage.layer_end);
        if stage.stage_id == load.stage_id {
            current_matches += 1;
            if stage.stage_index != load.stage_index
                || stage.layer_start != load.layer_start
                || stage.layer_end != load.layer_end
                || stage.bind_addr != load.bind_addr
            {
                return Err(StageFrameError::InvalidTopologyStages(
                    "current stage entry does not match load assignment",
                ));
            }
        }
    }
    if current_matches != 1 {
        return Err(StageFrameError::InvalidTopologyStages(
            "current stage must appear exactly once",
        ));
    }
    Ok(())
}

fn validate_status_stage_admission(
    status: &proto::stage::StageStatus,
) -> Result<(), StageFrameError> {
    let Some(admission) = status.admission.as_ref() else {
        if status.model_id.is_empty() {
            return Ok(());
        }
        return Err(StageFrameError::MissingStageAdmissionDescriptor);
    };
    validate_stage_admission_descriptor(admission)?;
    if admission.layer_start != status.layer_start || admission.layer_end != status.layer_end {
        return Err(StageFrameError::InvalidStageAdmissionDescriptor(
            "descriptor layer range does not match status range",
        ));
    }
    validate_activation_codec(status.activation_codec)?;
    Ok(())
}

fn validate_preparation_stage_admission(
    status: &proto::stage::StagePreparationStatus,
) -> Result<(), StageFrameError> {
    let Some(admission) = status.admission.as_ref() else {
        if status.model_id.is_empty() {
            return Ok(());
        }
        return Err(StageFrameError::MissingStageAdmissionDescriptor);
    };
    validate_stage_admission_descriptor(admission)?;
    if admission.layer_start != status.layer_start || admission.layer_end != status.layer_end {
        return Err(StageFrameError::InvalidStageAdmissionDescriptor(
            "descriptor layer range does not match preparation range",
        ));
    }
    validate_activation_codec(status.activation_codec)?;
    Ok(())
}

fn validate_activation_codec(value: i32) -> Result<(), StageFrameError> {
    match proto::stage::StageActivationCodec::try_from(value) {
        Ok(proto::stage::StageActivationCodec::RawF32V1)
        | Ok(proto::stage::StageActivationCodec::F16RneV1)
        | Ok(proto::stage::StageActivationCodec::Bf16RneV1)
        | Ok(proto::stage::StageActivationCodec::S8RowF32RneV1) => Ok(()),
        _ => Err(StageFrameError::InvalidActivationCodec { got: value }),
    }
}

pub fn validate_stage_admission_descriptor(
    descriptor: &proto::stage::StageAdmissionDescriptor,
) -> Result<(), StageFrameError> {
    if descriptor.version != crate::STAGE_ADMISSION_DESCRIPTOR_VERSION {
        return Err(StageFrameError::InvalidStageAdmissionDescriptor(
            "unsupported descriptor version",
        ));
    }
    if !valid_prefixed_sha256(&descriptor.package_id, "sha256:") {
        return Err(StageFrameError::InvalidStageAdmissionDescriptor(
            "package_id must be a canonical sha256 digest",
        ));
    }
    if !valid_prefixed_sha256(&descriptor.plan_id, "skippy-plan:v1:") {
        return Err(StageFrameError::InvalidStageAdmissionDescriptor(
            "plan_id must be a canonical native semantic digest",
        ));
    }
    if descriptor.layer_start >= descriptor.layer_end {
        return Err(StageFrameError::InvalidStageAdmissionDescriptor(
            "layer range must be non-empty",
        ));
    }
    if descriptor.resident_tensor_ids.is_empty()
        || !strictly_sorted_nonempty(&descriptor.resident_tensor_ids)
    {
        return Err(StageFrameError::InvalidStageAdmissionDescriptor(
            "resident tensor ids must be non-empty, strictly sorted, and unique",
        ));
    }
    if descriptor.profiles.is_empty()
        || !descriptor
            .profiles
            .windows(2)
            .all(|pair| !pair[0].profile_id.is_empty() && pair[0].profile_id < pair[1].profile_id)
        || descriptor
            .profiles
            .last()
            .is_some_and(|profile| profile.profile_id.is_empty())
    {
        return Err(StageFrameError::InvalidStageAdmissionDescriptor(
            "profiles must be non-empty, strictly sorted by profile_id, and unique",
        ));
    }
    for profile in &descriptor.profiles {
        if [
            profile.graph_identity.as_str(),
            profile.profile_identity.as_str(),
            profile.slice_identity.as_str(),
            profile.source_snapshot_identity.as_str(),
            profile.graph_configuration_id.as_str(),
            profile.backend_id.as_str(),
        ]
        .into_iter()
        .any(str::is_empty)
        {
            return Err(StageFrameError::InvalidStageAdmissionDescriptor(
                "profile identities are required",
            ));
        }
    }
    let mut previous_sidecar: Option<(i32, Option<&str>, &str)> = None;
    for sidecar in &descriptor.sidecars {
        if sidecar.artifact_id.is_empty()
            || proto::stage::StageAdmissionSidecarKind::try_from(sidecar.kind)
                .ok()
                .is_none_or(|kind| kind == proto::stage::StageAdmissionSidecarKind::Unspecified)
        {
            return Err(StageFrameError::InvalidStageAdmissionDescriptor(
                "sidecar kind and artifact_id are required",
            ));
        }
        let key = (
            sidecar.kind,
            sidecar.name.as_deref(),
            sidecar.artifact_id.as_str(),
        );
        if previous_sidecar.is_some_and(|previous| previous >= key) {
            return Err(StageFrameError::InvalidStageAdmissionDescriptor(
                "sidecars must be strictly sorted and unique",
            ));
        }
        previous_sidecar = Some(key);
    }
    Ok(())
}

fn strictly_sorted_nonempty(values: &[String]) -> bool {
    values.iter().all(|value| !value.is_empty()) && values.windows(2).all(|pair| pair[0] < pair[1])
}

fn valid_prefixed_sha256(value: &str, prefix: &str) -> bool {
    value.strip_prefix(prefix).is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn validate_source_resolution(
    source_sha256: Option<&str>,
    source_resolution_policy: i32,
) -> Result<(), StageFrameError> {
    let local_required =
        match proto::stage::SourceResolutionPolicy::try_from(source_resolution_policy) {
            Ok(proto::stage::SourceResolutionPolicy::Fallback) => false,
            Ok(proto::stage::SourceResolutionPolicy::LocalRequired) => true,
            Err(_) => {
                return Err(StageFrameError::InvalidSourceResolutionPolicy {
                    got: source_resolution_policy,
                });
            }
        };
    if local_required && source_sha256.is_none() {
        return Err(StageFrameError::MissingRequiredSourceDigest);
    }
    if let Some(sha256) = source_sha256 {
        validate_source_digest(sha256)?;
    }
    Ok(())
}

fn validate_source_digest(value: &str) -> Result<(), StageFrameError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(StageFrameError::InvalidSourceDigestLength { got: value.len() });
    }
    Ok(())
}

pub fn validate_stage_transport_open(
    frame: &proto::stage::StageTransportOpen,
) -> Result<(), StageFrameError> {
    validate_generation(frame.r#gen)?;
    validate_endpoint_id(frame.requester_id.len())?;
    if frame.topology_id.is_empty() || frame.run_id.is_empty() || frame.stage_id.is_empty() {
        return Err(StageFrameError::MissingStageTransportTarget);
    }
    Ok(())
}

pub fn validate_stage_artifact_transfer_request(
    frame: &proto::stage::StageArtifactTransferRequest,
) -> Result<(), StageFrameError> {
    validate_generation(frame.r#gen)?;
    validate_endpoint_id(frame.requester_id.len())?;
    if frame.topology_id.is_empty()
        || frame.run_id.is_empty()
        || frame.stage_id.is_empty()
        || !frame.package_ref.starts_with("hf://")
    {
        return Err(StageFrameError::MissingStageArtifactTarget);
    }
    validate_artifact_digest(&frame.manifest_sha256)?;
    if let Some(expected_sha) = frame.expected_sha256.as_deref() {
        validate_artifact_digest(expected_sha)?;
    }
    if frame.expected_size.is_some_and(|size| frame.offset > size) {
        return Err(StageFrameError::InvalidArtifactOffset);
    }
    validate_safe_relative_artifact_path(&frame.relative_path)?;
    Ok(())
}

pub fn validate_stage_artifact_transfer_response(
    frame: &proto::stage::StageArtifactTransferResponse,
) -> Result<(), StageFrameError> {
    validate_generation(frame.r#gen)?;
    if let Some(sha256) = frame.sha256.as_deref() {
        validate_artifact_digest(sha256)?;
    }
    Ok(())
}

fn validate_generation(r#gen: u32) -> Result<(), StageFrameError> {
    if r#gen != STAGE_PROTOCOL_GENERATION {
        return Err(StageFrameError::BadGeneration { got: r#gen });
    }
    Ok(())
}

fn validate_artifact_digest(value: &str) -> Result<(), StageFrameError> {
    if value.len() != 64 || !value.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(StageFrameError::InvalidArtifactDigestLength { got: value.len() });
    }
    Ok(())
}

fn validate_safe_relative_artifact_path(path: &str) -> Result<(), StageFrameError> {
    use std::path::{Component, Path};

    if path.trim().is_empty() {
        return Err(StageFrameError::InvalidArtifactPath);
    }
    let path = Path::new(path);
    let mut components = path.components();
    let Some(first) = components.next() else {
        return Err(StageFrameError::InvalidArtifactPath);
    };
    if !matches!(first, Component::Normal(_))
        || !components.all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(StageFrameError::InvalidArtifactPath);
    }
    Ok(())
}

fn validate_endpoint_id(len: usize) -> Result<(), StageFrameError> {
    if len != 32 {
        return Err(StageFrameError::InvalidEndpointId { got: len });
    }
    Ok(())
}
