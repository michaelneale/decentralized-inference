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
/// Sanity bound on the run-ahead speculative-token budget. Every in-flight
/// speculative token holds restorable recovery state downstream, so the budget
/// caps how much KV checkpoint memory one request can pin.
pub const MAX_VERIFY_WINDOW_RUNAHEAD_TOKENS: usize = 4096;

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
            reject_local_source_in_legacy_command(load)?;
            validate_source_resolution(
                load.source_model_sha256.as_deref(),
                load.source_resolution_policy,
            )?;
        }
        Some(Command::LoadLocalStage(load)) => validate_local_source_load(load)?,
        Some(Command::GetLayerInventory(inventory)) => validate_source_resolution(
            inventory.expected_source_model_sha256.as_deref(),
            inventory.source_resolution_policy,
        )?,
        Some(Command::PrepareStage(prepare)) => {
            if let Some(load) = prepare.load_stage.as_ref() {
                reject_local_source_in_legacy_command(load)?;
                validate_source_resolution(
                    load.source_model_sha256.as_deref(),
                    load.source_resolution_policy,
                )?;
            }
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
    Ok(())
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
