//! Proto <-> domain conversion helpers for Skippy staged-runtime control,
//! status, topology, and load-request messages exchanged over the mesh wire
//! protocol (`skippy_stage_proto`). Extracted from `mesh/mod.rs` because this
//! cluster of pure conversion functions had grown large enough to obscure the
//! rest of the mesh runtime; these functions depend only on
//! `crate::inference::skippy::*` domain types, `skippy_protocol`/
//! `skippy_stage_proto` wire types, and a handful of `mesh` module types
//! (`StageRuntimeStatus`, `StageTopologyInstance`, `StageAssignment`,
//! `StageEndpoint`), not on `Node` or other mesh runtime state.

use super::{StageAssignment, StageEndpoint, StageRuntimeStatus, StageTopologyInstance};
use anyhow::Context;
use iroh::EndpointId;
use skippy_protocol::proto::stage as skippy_stage_proto;
use std::collections::BTreeMap;

pub(super) fn stage_topology_key(topology_id: &str, run_id: &str) -> String {
    format!("{topology_id}\n{run_id}")
}

pub(super) fn stage_runtime_status_key(topology_id: &str, run_id: &str, stage_id: &str) -> String {
    format!("{topology_id}\n{run_id}\n{stage_id}")
}

fn path_free_stage_projector_path(
    package_ref: Option<&str>,
    projector_path: Option<String>,
) -> Option<String> {
    if package_ref.is_some_and(crate::inference::skippy::is_content_addressed_gguf_ref) {
        None
    } else {
        projector_path
    }
}

pub(super) fn endpoint_id_from_bytes(bytes: Vec<u8>) -> anyhow::Result<EndpointId> {
    let arr: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
        anyhow::anyhow!(
            "invalid endpoint id length: expected 32, got {}",
            bytes.len()
        )
    })?;
    let public_key = iroh::PublicKey::from_bytes(&arr)
        .map_err(|error| anyhow::anyhow!("invalid endpoint id bytes: {error}"))?;
    Ok(EndpointId::from(public_key))
}

pub(super) fn stage_runtime_status_from_snapshot(
    node_id: Option<EndpointId>,
    status: crate::inference::skippy::StageStatusSnapshot,
) -> StageRuntimeStatus {
    let projector_path =
        path_free_stage_projector_path(status.package_ref.as_deref(), status.projector_path);
    StageRuntimeStatus {
        topology_id: status.topology_id,
        run_id: status.run_id,
        model_id: status.model_id,
        backend: status.backend,
        package_ref: status.package_ref,
        manifest_sha256: status.manifest_sha256,
        source_model_path: status.source_model_path,
        source_model_sha256: status.source_model_sha256,
        source_model_bytes: status.source_model_bytes,
        materialized_path: status.materialized_path,
        materialized_pinned: status.materialized_pinned,
        projector_path,
        stage_id: status.stage_id,
        stage_index: status.stage_index,
        node_id,
        layer_start: status.layer_start,
        layer_end: status.layer_end,
        admission: status.admission,
        activation_codec: status.activation_codec,
        activation_codec_policy: status.activation_codec_policy,
        state: status.state,
        bind_addr: status.bind_addr,
        input_activation_boundary: status.input_activation_boundary,
        output_activation_boundary: status.output_activation_boundary,
        selected_device: status.selected_device,
        ctx_size: status.ctx_size,
        lane_count: status.lane_count,
        n_batch: status.n_batch,
        n_ubatch: status.n_ubatch,
        flash_attn_type: status.flash_attn_type,
        error: status.error,
        shutdown_generation: status.shutdown_generation,
    }
}

pub(crate) fn stage_snapshot_from_runtime_status(
    status: &StageRuntimeStatus,
    state: crate::inference::skippy::StageRuntimeState,
    error: Option<String>,
) -> crate::inference::skippy::StageStatusSnapshot {
    let projector_path = path_free_stage_projector_path(
        status.package_ref.as_deref(),
        status.projector_path.clone(),
    );
    crate::inference::skippy::StageStatusSnapshot {
        topology_id: status.topology_id.clone(),
        run_id: status.run_id.clone(),
        model_id: status.model_id.clone(),
        backend: status.backend.clone(),
        package_ref: status.package_ref.clone(),
        manifest_sha256: status.manifest_sha256.clone(),
        source_model_path: status.source_model_path.clone(),
        source_model_sha256: status.source_model_sha256.clone(),
        source_model_bytes: status.source_model_bytes,
        materialized_path: status.materialized_path.clone(),
        materialized_pinned: status.materialized_pinned,
        projector_path,
        stage_id: status.stage_id.clone(),
        stage_index: status.stage_index,
        layer_start: status.layer_start,
        layer_end: status.layer_end,
        admission: status.admission.clone(),
        activation_codec: status.activation_codec,
        activation_codec_policy: status.activation_codec_policy,
        state,
        bind_addr: status.bind_addr.clone(),
        input_activation_boundary: status.input_activation_boundary,
        output_activation_boundary: status.output_activation_boundary,
        selected_device: status.selected_device.clone(),
        ctx_size: status.ctx_size,
        lane_count: status.lane_count,
        n_batch: status.n_batch,
        n_ubatch: status.n_ubatch,
        flash_attn_type: status.flash_attn_type,
        error,
        shutdown_generation: status.shutdown_generation,
        coordinator_term: 0,
        coordinator_id: None,
        lease_until_unix_ms: 0,
    }
}

pub(super) fn stage_topology_from_load(
    _node_id: EndpointId,
    load: &crate::inference::skippy::StageLoadRequest,
) -> StageTopologyInstance {
    StageTopologyInstance {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        admissions: BTreeMap::from([(load.stage_id.clone(), load.admission.clone())]),
        stages: load
            .topology_stages
            .iter()
            .map(|stage| StageAssignment {
                stage_id: stage.stage_id.clone(),
                stage_index: stage.stage_index,
                node_id: stage.node_id,
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                endpoint: StageEndpoint {
                    bind_addr: stage.bind_addr.clone(),
                },
            })
            .collect(),
    }
}

pub(super) fn stage_control_request_to_proto(
    requester_id: EndpointId,
    request: crate::inference::skippy::StageControlRequest,
) -> anyhow::Result<skippy_stage_proto::StageControlRequest> {
    use skippy_stage_proto::stage_control_request::Command;

    let command = match request {
        crate::inference::skippy::StageControlRequest::Claim(claim) => {
            Command::ClaimCoordinator(stage_coordinator_claim_to_proto(claim))
        }
        crate::inference::skippy::StageControlRequest::Load(load) => {
            anyhow::ensure!(
                !load.local_source_required
                    && !crate::inference::skippy::is_content_addressed_gguf_ref(&load.package_ref),
                "local-required stage load must use the fail-closed LoadLocal command"
            );
            Command::LoadStage(stage_load_to_proto(load))
        }
        crate::inference::skippy::StageControlRequest::LoadLocal(mut load) => {
            // The command discriminant is the fail-closed contract. Do not let
            // a stale or manually constructed domain request weaken it by
            // carrying the fallback policy bit.
            load.local_source_required = true;
            Command::LoadLocalStage(stage_load_to_proto(load))
        }
        crate::inference::skippy::StageControlRequest::Stop(stop) => {
            Command::StopStage(skippy_stage_proto::StopStage {
                topology_id: stop.topology_id,
                run_id: stop.run_id,
                stage_id: stop.stage_id,
                shutdown_generation: stop.shutdown_generation,
                coordinator_term: stop.coordinator_term,
            })
        }
        crate::inference::skippy::StageControlRequest::Status(status) => {
            Command::GetStageStatus(skippy_stage_proto::GetStageStatus {
                topology_id: status.topology_id,
                run_id: status.run_id,
                stage_id: status.stage_id,
            })
        }
        crate::inference::skippy::StageControlRequest::Inventory(inventory) => {
            Command::GetLayerInventory(skippy_stage_proto::GetLayerInventory {
                model_id: inventory.model_id,
                runtime_profile: inventory.runtime_profile,
                package_ref: inventory.package_ref,
                manifest_sha256: inventory.manifest_sha256,
                expected_source_model_sha256: inventory.expected_source_model_sha256,
                source_resolution_policy: source_resolution_policy_to_proto(
                    inventory.local_source_required,
                ) as i32,
            })
        }
        crate::inference::skippy::StageControlRequest::Prepare(prepare) => {
            anyhow::ensure!(
                !prepare.load.local_source_required
                    && !crate::inference::skippy::is_content_addressed_gguf_ref(
                        &prepare.load.package_ref
                    ),
                "local-required stage load cannot use the legacy Prepare command"
            );
            Command::PrepareStage(skippy_stage_proto::PrepareStage {
                load_stage: Some(stage_load_to_proto(prepare.load)),
                coordinator_id: prepare.coordinator_id.map(|id| id.as_bytes().to_vec()),
            })
        }
        crate::inference::skippy::StageControlRequest::CancelPrepare(cancel) => {
            Command::CancelPrepareStage(skippy_stage_proto::CancelPrepareStage {
                topology_id: cancel.topology_id,
                run_id: cancel.run_id,
                stage_id: cancel.stage_id,
                shutdown_generation: cancel.shutdown_generation,
            })
        }
        crate::inference::skippy::StageControlRequest::StatusUpdate(status) => {
            Command::StageStatusUpdate(skippy_stage_proto::StageStatusUpdate {
                status: Some(stage_preparation_status_to_proto(status)),
            })
        }
    };

    let frame = skippy_stage_proto::StageControlRequest {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        requester_id: requester_id.as_bytes().to_vec(),
        command: Some(command),
    };
    skippy_protocol::validate_stage_control_request(&frame)
        .map_err(|error| anyhow::anyhow!("invalid outbound stage-control request: {error}"))?;
    Ok(frame)
}

pub(super) fn stage_load_to_proto(
    load: crate::inference::skippy::StageLoadRequest,
) -> skippy_stage_proto::LoadStage {
    let projector_path = (!load.local_source_required)
        .then(|| load.projector_path.clone())
        .flatten();
    skippy_stage_proto::LoadStage {
        topology_id: load.topology_id,
        run_id: load.run_id,
        model_id: load.model_id,
        runtime_profile: load.runtime_profile,
        backend: load.backend,
        package_ref: load.package_ref,
        manifest_sha256: load.manifest_sha256,
        stage_id: load.stage_id,
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        model_path: load.model_path,
        source_model_bytes: load.source_model_bytes,
        source_model_sha256: load.source_model_sha256,
        source_resolution_policy: source_resolution_policy_to_proto(load.local_source_required)
            as i32,
        projector_path,
        projector_use_gpu: load.projector_use_gpu,
        media_marker: load.media_marker,
        image_min_tokens: load.image_min_tokens,
        image_max_tokens: load.image_max_tokens,
        batch_max_tokens: load.batch_max_tokens,
        glm_dsa_policy: stage_glm_dsa_policy_to_proto(load.glm_dsa_policy) as i32,
        generation_signal_window: load.generation_signal_window,
        selected_device: load.selected_device.map(stage_device_to_proto),
        bind_addr: load.bind_addr,
        ctx_size: load.ctx_size,
        lane_count: load.lane_count,
        continuous_batching: Some(load.continuous_batching),
        n_batch: load.n_batch,
        n_ubatch: load.n_ubatch,
        n_gpu_layers: load.n_gpu_layers,
        mmap: load.mmap,
        mlock: Some(load.mlock),
        cache_type_k: load.cache_type_k,
        cache_type_v: load.cache_type_v,
        flash_attn_type: stage_flash_attn_type_to_proto(load.flash_attn_type) as i32,
        runtime_settings: Some(stage_runtime_settings_to_proto(load.runtime_settings)),
        native_mtp_enabled: Some(load.native_mtp_enabled),
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id.map(|id| id.to_string()),
        lease_until_unix_ms: load.lease_until_unix_ms,
        load_mode: match load.load_mode {
            skippy_protocol::LoadMode::RuntimeSlice => {
                skippy_stage_proto::StageLoadMode::RuntimeSlice as i32
            }
            skippy_protocol::LoadMode::LayerPackage => {
                skippy_stage_proto::StageLoadMode::LayerPackage as i32
            }
            skippy_protocol::LoadMode::ArtifactSlice => {
                skippy_stage_proto::StageLoadMode::ArtifactSlice as i32
            }
        },
        upstream: load.upstream.map(stage_peer_to_proto),
        downstream: load.downstream.map(stage_peer_to_proto),
        admission: Some(load.admission.into()),
        participant_set_hash: load.participant_set_hash,
        topology_hash: load.topology_hash,
        activation_codec: stage_activation_codec_to_proto(load.activation_codec) as i32,
        activation_codec_policy: stage_activation_codec_policy_to_proto(
            load.activation_codec_policy,
        ) as i32,
        topology_stages: load
            .topology_stages
            .into_iter()
            .map(stage_topology_stage_to_proto)
            .collect(),
    }
}

fn stage_topology_stage_to_proto(
    stage: crate::inference::skippy::StageTopologyStageDescriptor,
) -> skippy_stage_proto::StageTopologyStage {
    skippy_stage_proto::StageTopologyStage {
        stage_id: stage.stage_id,
        stage_index: stage.stage_index,
        node_id: stage.node_id.as_bytes().to_vec(),
        layer_start: stage.layer_start,
        layer_end: stage.layer_end,
        bind_addr: stage.bind_addr,
    }
}

fn stage_runtime_settings_to_proto(
    settings: crate::inference::skippy::StageLoadRuntimeSettings,
) -> skippy_stage_proto::StageLoadRuntimeSettings {
    skippy_stage_proto::StageLoadRuntimeSettings {
        repack: Some(settings.repack),
        op_offload: settings.op_offload,
        no_host_buffer: Some(settings.no_host_buffer),
        check_tensors: Some(settings.check_tensors),
        direct_io: Some(settings.direct_io),
        main_gpu: settings.main_gpu,
        split_mode: Some(match settings.split_mode {
            skippy_protocol::SplitMode::Auto => -1,
            skippy_protocol::SplitMode::None => 0,
            skippy_protocol::SplitMode::Layer => 1,
            skippy_protocol::SplitMode::Row => 2,
            skippy_protocol::SplitMode::Tensor => 3,
        }),
        kv_offload: settings.kv_offload,
        kv_unified: settings.kv_unified,
        swa_full: settings.swa_full,
        cache_idle_slots: settings.cache_idle_slots,
        activation_codec_policy: stage_activation_codec_policy_to_proto(
            settings.activation_codec_policy,
        ) as i32,
    }
}

pub(super) fn stage_coordinator_claim_to_proto(
    claim: crate::inference::skippy::StageCoordinatorClaim,
) -> skippy_stage_proto::ClaimCoordinator {
    skippy_stage_proto::ClaimCoordinator {
        model_id: claim.model_id,
        package_ref: claim.package_ref,
        manifest_sha256: claim.manifest_sha256,
        topology_id: claim.topology_id,
        run_id: claim.run_id,
        coordinator_id: claim.coordinator_id,
        coordinator_term: claim.coordinator_term,
        participant_set_hash: claim.participant_set_hash,
        topology_hash: claim.topology_hash,
        lease_until_unix_ms: claim.lease_until_unix_ms,
    }
}

pub(super) fn stage_peer_to_proto(
    peer: crate::inference::skippy::StagePeerDescriptor,
) -> skippy_stage_proto::StagePeer {
    skippy_stage_proto::StagePeer {
        stage_id: peer.stage_id,
        stage_index: peer.stage_index,
        endpoint: peer.endpoint,
        node_id: peer.node_id.map(|id| id.as_bytes().to_vec()),
    }
}

pub(super) fn stage_device_to_proto(
    device: skippy_protocol::StageDevice,
) -> skippy_stage_proto::StageDevice {
    skippy_stage_proto::StageDevice {
        backend_device: device.backend_device,
        stable_id: device.stable_id,
        index: device.index.map(|value| value as u64),
        vram_bytes: device.vram_bytes,
    }
}

pub(super) fn stage_control_request_from_proto(
    frame: skippy_stage_proto::StageControlRequest,
) -> anyhow::Result<crate::inference::skippy::StageControlRequest> {
    use skippy_stage_proto::stage_control_request::Command;

    match frame
        .command
        .ok_or_else(|| anyhow::anyhow!("missing stage control command"))?
    {
        Command::ClaimCoordinator(claim) => {
            Ok(crate::inference::skippy::StageControlRequest::Claim(
                stage_coordinator_claim_from_proto(claim)?,
            ))
        }
        Command::LoadStage(load) => Ok(crate::inference::skippy::StageControlRequest::Load(
            stage_load_from_proto(load)?,
        )),
        Command::LoadLocalStage(load) => {
            let mut load = stage_load_from_proto(load)?;
            // Preserve the command-level invariant even if this conversion is
            // called on a frame that did not pass the wire validator first.
            load.local_source_required = true;
            Ok(crate::inference::skippy::StageControlRequest::LoadLocal(
                load,
            ))
        }
        Command::StopStage(stop) => Ok(crate::inference::skippy::StageControlRequest::Stop(
            crate::inference::skippy::StageStopRequest {
                topology_id: stop.topology_id,
                run_id: stop.run_id,
                stage_id: stop.stage_id,
                shutdown_generation: stop.shutdown_generation,
                coordinator_term: stop.coordinator_term,
            },
        )),
        Command::GetStageStatus(status) => {
            Ok(crate::inference::skippy::StageControlRequest::Status(
                crate::inference::skippy::StageStatusFilter {
                    topology_id: status.topology_id,
                    run_id: status.run_id,
                    stage_id: status.stage_id,
                },
            ))
        }
        Command::GetLayerInventory(inventory) => {
            Ok(crate::inference::skippy::StageControlRequest::Inventory(
                crate::inference::skippy::StageInventoryRequest {
                    model_id: inventory.model_id,
                    runtime_profile: inventory.runtime_profile,
                    package_ref: inventory.package_ref,
                    manifest_sha256: inventory.manifest_sha256,
                    expected_source_model_sha256: inventory.expected_source_model_sha256,
                    local_source_required: source_resolution_policy_from_proto(
                        inventory.source_resolution_policy,
                    )?,
                },
            ))
        }
        Command::PrepareStage(prepare) => {
            let load = prepare
                .load_stage
                .ok_or_else(|| anyhow::anyhow!("prepare stage missing load_stage"))?;
            Ok(crate::inference::skippy::StageControlRequest::Prepare(
                crate::inference::skippy::StagePrepareRequest {
                    load: stage_load_from_proto(load)?,
                    coordinator_id: prepare
                        .coordinator_id
                        .map(endpoint_id_from_bytes)
                        .transpose()
                        .context("invalid prepare stage coordinator_id")?,
                },
            ))
        }
        Command::CancelPrepareStage(cancel) => Ok(
            crate::inference::skippy::StageControlRequest::CancelPrepare(
                crate::inference::skippy::StageCancelPrepareRequest {
                    topology_id: cancel.topology_id,
                    run_id: cancel.run_id,
                    stage_id: cancel.stage_id,
                    shutdown_generation: cancel.shutdown_generation,
                },
            ),
        ),
        Command::StageStatusUpdate(update) => {
            let status = update
                .status
                .ok_or_else(|| anyhow::anyhow!("stage status update missing status"))?;
            Ok(crate::inference::skippy::StageControlRequest::StatusUpdate(
                stage_preparation_status_from_proto(status)?,
            ))
        }
    }
}

pub(super) fn stage_load_from_proto(
    load: skippy_stage_proto::LoadStage,
) -> anyhow::Result<crate::inference::skippy::StageLoadRequest> {
    let local_source_required = source_resolution_policy_from_proto(load.source_resolution_policy)?;
    let projector_path = (!local_source_required)
        .then(|| load.projector_path.clone())
        .flatten();
    let admission = load
        .admission
        .clone()
        .ok_or_else(|| anyhow::anyhow!("stage load missing admission descriptor"))?
        .try_into()
        .map_err(|error| anyhow::anyhow!("invalid stage load admission descriptor: {error}"))?;
    Ok(crate::inference::skippy::StageLoadRequest {
        topology_id: load.topology_id,
        run_id: load.run_id,
        model_id: load.model_id,
        runtime_profile: load.runtime_profile,
        backend: load.backend,
        package_ref: load.package_ref,
        manifest_sha256: load.manifest_sha256,
        stage_id: load.stage_id,
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        admission,
        participant_set_hash: load.participant_set_hash,
        topology_hash: load.topology_hash,
        activation_codec: stage_activation_codec_from_proto(load.activation_codec)?,
        activation_codec_policy: stage_activation_codec_policy_from_proto(
            load.activation_codec_policy,
        )?,
        topology_stages: load
            .topology_stages
            .into_iter()
            .map(stage_topology_stage_from_proto)
            .collect::<anyhow::Result<Vec<_>>>()?,
        model_path: load.model_path,
        source_model_bytes: load.source_model_bytes,
        source_model_sha256: load.source_model_sha256,
        local_source_required,
        projector_path,
        projector_use_gpu: load.projector_use_gpu,
        media_marker: load.media_marker,
        image_min_tokens: load.image_min_tokens,
        image_max_tokens: load.image_max_tokens,
        batch_max_tokens: load.batch_max_tokens,
        glm_dsa_policy: stage_glm_dsa_policy_from_proto(load.glm_dsa_policy),
        generation_signal_window: load.generation_signal_window,
        selected_device: load
            .selected_device
            .map(stage_device_from_proto)
            .transpose()?,
        bind_addr: load.bind_addr,
        ctx_size: load.ctx_size,
        lane_count: if load.lane_count == 0 {
            4
        } else {
            load.lane_count
        },
        continuous_batching: load.continuous_batching.unwrap_or(true),
        n_batch: load.n_batch,
        n_ubatch: load.n_ubatch,
        n_gpu_layers: load.n_gpu_layers,
        mmap: load.mmap,
        mlock: load.mlock.unwrap_or(false),
        cache_type_k: load.cache_type_k,
        cache_type_v: load.cache_type_v,
        flash_attn_type: stage_flash_attn_type_from_proto(load.flash_attn_type),
        runtime_settings: stage_runtime_settings_from_proto(load.runtime_settings)?,
        native_mtp_enabled: load.native_mtp_enabled.unwrap_or(true),
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load
            .coordinator_id
            .map(|id| id.parse())
            .transpose()
            .context("invalid stage load coordinator_id")?,
        lease_until_unix_ms: load.lease_until_unix_ms,
        load_mode: stage_load_mode_from_proto(load.load_mode),
        upstream: load.upstream.map(stage_peer_from_proto).transpose()?,
        downstream: load.downstream.map(stage_peer_from_proto).transpose()?,
    })
}

fn stage_topology_stage_from_proto(
    stage: skippy_stage_proto::StageTopologyStage,
) -> anyhow::Result<crate::inference::skippy::StageTopologyStageDescriptor> {
    Ok(crate::inference::skippy::StageTopologyStageDescriptor {
        stage_id: stage.stage_id,
        stage_index: stage.stage_index,
        node_id: endpoint_id_from_bytes(stage.node_id).context("invalid topology stage node_id")?,
        layer_start: stage.layer_start,
        layer_end: stage.layer_end,
        bind_addr: stage.bind_addr,
    })
}

fn stage_runtime_settings_from_proto(
    settings: Option<skippy_stage_proto::StageLoadRuntimeSettings>,
) -> anyhow::Result<crate::inference::skippy::StageLoadRuntimeSettings> {
    let Some(settings) = settings else {
        return Ok(crate::inference::skippy::StageLoadRuntimeSettings::default());
    };
    Ok(crate::inference::skippy::StageLoadRuntimeSettings {
        repack: settings.repack.unwrap_or(false),
        op_offload: settings.op_offload,
        no_host_buffer: settings.no_host_buffer.unwrap_or(false),
        check_tensors: settings.check_tensors.unwrap_or(false),
        direct_io: settings.direct_io.unwrap_or(false),
        main_gpu: settings.main_gpu,
        split_mode: match settings.split_mode.unwrap_or(-1) {
            0 => skippy_protocol::SplitMode::None,
            1 => skippy_protocol::SplitMode::Layer,
            2 => skippy_protocol::SplitMode::Row,
            3 => skippy_protocol::SplitMode::Tensor,
            _ => skippy_protocol::SplitMode::Auto,
        },
        kv_offload: settings.kv_offload,
        kv_unified: settings.kv_unified,
        swa_full: settings.swa_full,
        cache_idle_slots: settings.cache_idle_slots,
        activation_codec_policy: stage_activation_codec_policy_from_proto(
            settings.activation_codec_policy,
        )?,
    })
}

pub(super) fn stage_coordinator_claim_from_proto(
    claim: skippy_stage_proto::ClaimCoordinator,
) -> anyhow::Result<crate::inference::skippy::StageCoordinatorClaim> {
    Ok(crate::inference::skippy::StageCoordinatorClaim {
        model_id: claim.model_id,
        package_ref: claim.package_ref,
        manifest_sha256: claim.manifest_sha256,
        topology_id: claim.topology_id,
        run_id: claim.run_id,
        coordinator_id: claim.coordinator_id,
        coordinator_term: claim.coordinator_term,
        participant_set_hash: claim.participant_set_hash,
        topology_hash: claim.topology_hash,
        lease_until_unix_ms: claim.lease_until_unix_ms,
    })
}

pub(super) fn stage_device_from_proto(
    device: skippy_stage_proto::StageDevice,
) -> anyhow::Result<skippy_protocol::StageDevice> {
    Ok(skippy_protocol::StageDevice {
        backend_device: device.backend_device,
        stable_id: device.stable_id,
        index: device
            .index
            .map(usize::try_from)
            .transpose()
            .context("stage selected_device.index exceeds usize")?,
        vram_bytes: device.vram_bytes,
    })
}

pub(super) fn stage_peer_from_proto(
    peer: skippy_stage_proto::StagePeer,
) -> anyhow::Result<crate::inference::skippy::StagePeerDescriptor> {
    Ok(crate::inference::skippy::StagePeerDescriptor {
        stage_id: peer.stage_id,
        stage_index: peer.stage_index,
        endpoint: peer.endpoint,
        node_id: peer
            .node_id
            .map(endpoint_id_from_bytes)
            .transpose()
            .context("invalid stage peer node_id")?,
    })
}

pub(super) fn stage_load_mode_from_proto(value: i32) -> skippy_protocol::LoadMode {
    match skippy_stage_proto::StageLoadMode::try_from(value)
        .unwrap_or(skippy_stage_proto::StageLoadMode::Unspecified)
    {
        skippy_stage_proto::StageLoadMode::Unspecified
        | skippy_stage_proto::StageLoadMode::RuntimeSlice => {
            skippy_protocol::LoadMode::RuntimeSlice
        }
        skippy_stage_proto::StageLoadMode::LayerPackage => skippy_protocol::LoadMode::LayerPackage,
        skippy_stage_proto::StageLoadMode::ArtifactSlice => {
            skippy_protocol::LoadMode::ArtifactSlice
        }
    }
}

fn stage_activation_codec_to_proto(
    codec: skippy_protocol::StageActivationCodec,
) -> skippy_stage_proto::StageActivationCodec {
    match codec {
        skippy_protocol::StageActivationCodec::RawF32V1 => {
            skippy_stage_proto::StageActivationCodec::RawF32V1
        }
        skippy_protocol::StageActivationCodec::F16RneV1 => {
            skippy_stage_proto::StageActivationCodec::F16RneV1
        }
        skippy_protocol::StageActivationCodec::Bf16RneV1 => {
            skippy_stage_proto::StageActivationCodec::Bf16RneV1
        }
        skippy_protocol::StageActivationCodec::S8RowF32RneV1 => {
            skippy_stage_proto::StageActivationCodec::S8RowF32RneV1
        }
    }
}

fn stage_activation_codec_from_proto(
    value: i32,
) -> anyhow::Result<skippy_protocol::StageActivationCodec> {
    match skippy_stage_proto::StageActivationCodec::try_from(value) {
        Ok(skippy_stage_proto::StageActivationCodec::RawF32V1) => {
            Ok(skippy_protocol::StageActivationCodec::RawF32V1)
        }
        Ok(skippy_stage_proto::StageActivationCodec::F16RneV1) => {
            Ok(skippy_protocol::StageActivationCodec::F16RneV1)
        }
        Ok(skippy_stage_proto::StageActivationCodec::Bf16RneV1) => {
            Ok(skippy_protocol::StageActivationCodec::Bf16RneV1)
        }
        Ok(skippy_stage_proto::StageActivationCodec::S8RowF32RneV1) => {
            Ok(skippy_protocol::StageActivationCodec::S8RowF32RneV1)
        }
        _ => anyhow::bail!("unsupported generation-8 activation codec {value}"),
    }
}

fn stage_activation_codec_policy_to_proto(
    policy: skippy_protocol::StageActivationCodecPolicy,
) -> skippy_stage_proto::StageActivationCodecPolicy {
    match policy {
        skippy_protocol::StageActivationCodecPolicy::Fixed => {
            skippy_stage_proto::StageActivationCodecPolicy::FixedV1
        }
        skippy_protocol::StageActivationCodecPolicy::AutoLosslessV1 => {
            skippy_stage_proto::StageActivationCodecPolicy::AutoLosslessV1
        }
    }
}

fn stage_activation_codec_policy_from_proto(
    value: i32,
) -> anyhow::Result<skippy_protocol::StageActivationCodecPolicy> {
    match skippy_stage_proto::StageActivationCodecPolicy::try_from(value) {
        // Legacy peer that never sent the policy field.
        Ok(skippy_stage_proto::StageActivationCodecPolicy::Unspecified) => {
            Ok(skippy_protocol::StageActivationCodecPolicy::Fixed)
        }
        Ok(skippy_stage_proto::StageActivationCodecPolicy::FixedV1) => {
            Ok(skippy_protocol::StageActivationCodecPolicy::Fixed)
        }
        Ok(skippy_stage_proto::StageActivationCodecPolicy::AutoLosslessV1) => {
            Ok(skippy_protocol::StageActivationCodecPolicy::AutoLosslessV1)
        }
        _ => anyhow::bail!("unsupported generation-8 activation codec policy {value}"),
    }
}

pub(super) fn stage_control_unavailable_response(
    request: crate::inference::skippy::StageControlRequest,
) -> crate::inference::skippy::StageControlResponse {
    let status = match request {
        crate::inference::skippy::StageControlRequest::Claim(claim) => {
            return crate::inference::skippy::StageControlResponse::ClaimAccepted(
                crate::inference::skippy::StageCoordinatorClaimAck {
                    accepted: false,
                    claim,
                    error: Some("stage control is not available".to_string()),
                },
            );
        }
        crate::inference::skippy::StageControlRequest::Load(load) => {
            stage_status_from_load(&load, crate::inference::skippy::StageRuntimeState::Failed)
        }
        crate::inference::skippy::StageControlRequest::LoadLocal(load) => {
            stage_status_from_load(&load, crate::inference::skippy::StageRuntimeState::Failed)
        }
        crate::inference::skippy::StageControlRequest::Stop(stop) => {
            crate::inference::skippy::StageStatusSnapshot {
                topology_id: stop.topology_id,
                run_id: stop.run_id,
                model_id: String::new(),
                backend: "skippy".to_string(),
                package_ref: None,
                manifest_sha256: None,
                source_model_path: None,
                source_model_sha256: None,
                source_model_bytes: None,
                materialized_path: None,
                materialized_pinned: false,
                projector_path: None,
                stage_id: stop.stage_id,
                stage_index: 0,
                layer_start: 0,
                layer_end: 0,
                admission: None,
                activation_codec: skippy_protocol::StageActivationCodec::default(),
                activation_codec_policy: Default::default(),
                state: crate::inference::skippy::StageRuntimeState::Failed,
                bind_addr: String::new(),
                input_activation_boundary: None,
                output_activation_boundary: None,
                selected_device: None,
                ctx_size: 0,
                lane_count: 0,
                n_batch: None,
                n_ubatch: None,
                flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
                error: Some("stage control is not available".to_string()),
                shutdown_generation: stop.shutdown_generation,
                coordinator_term: stop.coordinator_term,
                coordinator_id: None,
                lease_until_unix_ms: 0,
            }
        }
        crate::inference::skippy::StageControlRequest::Status(_) => {
            return crate::inference::skippy::StageControlResponse::Status(Vec::new());
        }
        crate::inference::skippy::StageControlRequest::Inventory(inventory) => {
            return crate::inference::skippy::StageControlResponse::Inventory(
                crate::inference::skippy::StageLayerInventory {
                    model_id: inventory.model_id,
                    package_ref: inventory.package_ref,
                    manifest_sha256: inventory.manifest_sha256,
                    layer_count: 0,
                    ready_ranges: Vec::new(),
                    available_ranges: Vec::new(),
                    missing_ranges: Vec::new(),
                    preparing_ranges: Vec::new(),
                    source_model_path: None,
                    source_model_bytes: None,
                    source_model_sha256: None,
                    content_addressed_local_source: None,
                    source_model_kind: crate::inference::skippy::SourceModelKind::Unknown,
                },
            );
        }
        crate::inference::skippy::StageControlRequest::Prepare(prepare) => {
            return crate::inference::skippy::StageControlResponse::PrepareAccepted(
                crate::inference::skippy::StagePrepareAcceptedResponse {
                    accepted: false,
                    status: stage_preparation_status_from_load(
                        &prepare.load,
                        crate::inference::skippy::StagePreparationState::Failed,
                        Some("stage control is not available".to_string()),
                    ),
                    error: Some("stage control is not available".to_string()),
                },
            );
        }
        crate::inference::skippy::StageControlRequest::CancelPrepare(cancel) => {
            return crate::inference::skippy::StageControlResponse::PreparationStatus(
                stage_preparation_status_from_cancel(
                    cancel,
                    crate::inference::skippy::StagePreparationState::Failed,
                    Some("stage control is not available".to_string()),
                ),
            );
        }
        crate::inference::skippy::StageControlRequest::StatusUpdate(_) => {
            return crate::inference::skippy::StageControlResponse::StatusAck(
                crate::inference::skippy::StageStatusAck {
                    accepted: false,
                    error: Some("stage control is not available".to_string()),
                },
            );
        }
    };
    crate::inference::skippy::StageControlResponse::Ready(
        crate::inference::skippy::StageReadyResponse {
            accepted: false,
            status,
            error: Some("stage control is not available".to_string()),
        },
    )
}

pub(crate) fn stage_status_from_load(
    load: &crate::inference::skippy::StageLoadRequest,
    state: crate::inference::skippy::StageRuntimeState,
) -> crate::inference::skippy::StageStatusSnapshot {
    let content_addressed_ref =
        crate::inference::skippy::is_content_addressed_gguf_ref(&load.package_ref);
    crate::inference::skippy::StageStatusSnapshot {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        backend: load.backend.clone(),
        package_ref: Some(load.package_ref.clone()),
        manifest_sha256: Some(load.manifest_sha256.clone()),
        source_model_path: (!load.local_source_required && !content_addressed_ref)
            .then(|| load.model_path.clone())
            .flatten(),
        source_model_sha256: load.source_model_sha256.clone(),
        source_model_bytes: load.source_model_bytes,
        materialized_path: None,
        materialized_pinned: false,
        projector_path: (!load.local_source_required && !content_addressed_ref)
            .then(|| load.projector_path.clone())
            .flatten(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        admission: Some(load.admission.clone()),
        activation_codec: load.activation_codec,
        activation_codec_policy: load.activation_codec_policy,
        state,
        bind_addr: load.bind_addr.clone(),
        input_activation_boundary: None,
        output_activation_boundary: None,
        selected_device: load.selected_device.clone(),
        ctx_size: load.ctx_size,
        lane_count: load.lane_count,
        n_batch: load.n_batch,
        n_ubatch: load.n_ubatch,
        flash_attn_type: load.flash_attn_type,
        error: Some("stage control is not available".to_string()),
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id,
        lease_until_unix_ms: load.lease_until_unix_ms,
    }
}

pub(super) fn stage_preparation_status_from_load(
    load: &crate::inference::skippy::StageLoadRequest,
    state: crate::inference::skippy::StagePreparationState,
    error: Option<String>,
) -> crate::inference::skippy::StagePreparationStatus {
    crate::inference::skippy::StagePreparationStatus {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        backend: load.backend.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        admission: Some(load.admission.clone()),
        activation_codec: load.activation_codec,
        activation_codec_policy: load.activation_codec_policy,
        state,
        bytes_done: None,
        bytes_total: None,
        bind_addr: None,
        error,
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id,
        lease_until_unix_ms: load.lease_until_unix_ms,
    }
}

pub(super) fn stage_preparation_status_from_cancel(
    cancel: crate::inference::skippy::StageCancelPrepareRequest,
    state: crate::inference::skippy::StagePreparationState,
    error: Option<String>,
) -> crate::inference::skippy::StagePreparationStatus {
    crate::inference::skippy::StagePreparationStatus {
        topology_id: cancel.topology_id,
        run_id: cancel.run_id,
        model_id: String::new(),
        backend: "skippy".to_string(),
        package_ref: String::new(),
        manifest_sha256: String::new(),
        stage_id: cancel.stage_id,
        stage_index: 0,
        layer_start: 0,
        layer_end: 0,
        admission: None,
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        activation_codec_policy: Default::default(),
        state,
        bytes_done: None,
        bytes_total: None,
        bind_addr: None,
        error,
        shutdown_generation: cancel.shutdown_generation,
        coordinator_term: 0,
        coordinator_id: None,
        lease_until_unix_ms: 0,
    }
}

pub(super) fn stage_control_response_to_proto(
    response: crate::inference::skippy::StageControlResponse,
) -> skippy_stage_proto::StageControlResponse {
    use skippy_stage_proto::stage_control_response::Response;

    let response = match response {
        crate::inference::skippy::StageControlResponse::ClaimAccepted(accepted) => {
            Response::CoordinatorClaimAccepted(skippy_stage_proto::CoordinatorClaimAccepted {
                accepted: accepted.accepted,
                claim: Some(stage_coordinator_claim_to_proto(accepted.claim)),
                error: accepted.error,
            })
        }
        crate::inference::skippy::StageControlResponse::Ready(ready) => {
            Response::StageReady(skippy_stage_proto::StageReady {
                accepted: ready.accepted,
                status: Some(stage_status_to_proto(ready.status)),
                error: ready.error,
            })
        }
        crate::inference::skippy::StageControlResponse::Status(statuses) => {
            Response::StageStatuses(skippy_stage_proto::StageStatusList {
                statuses: statuses.into_iter().map(stage_status_to_proto).collect(),
            })
        }
        crate::inference::skippy::StageControlResponse::Inventory(inventory) => {
            Response::LayerInventory(layer_inventory_to_proto(inventory))
        }
        crate::inference::skippy::StageControlResponse::PrepareAccepted(accepted) => {
            Response::PrepareStageAccepted(skippy_stage_proto::PrepareStageAccepted {
                accepted: accepted.accepted,
                status: Some(stage_preparation_status_to_proto(accepted.status)),
                error: accepted.error,
            })
        }
        crate::inference::skippy::StageControlResponse::PreparationStatus(status) => {
            Response::StagePreparationStatus(stage_preparation_status_to_proto(status))
        }
        crate::inference::skippy::StageControlResponse::StatusAck(ack) => {
            Response::StageStatusAck(skippy_stage_proto::StageStatusAck {
                accepted: ack.accepted,
                error: ack.error,
            })
        }
    };

    skippy_stage_proto::StageControlResponse {
        r#gen: skippy_protocol::STAGE_PROTOCOL_GENERATION,
        response: Some(response),
    }
}

pub(super) fn stage_control_response_from_proto(
    frame: skippy_stage_proto::StageControlResponse,
) -> anyhow::Result<crate::inference::skippy::StageControlResponse> {
    use skippy_stage_proto::stage_control_response::Response;

    match frame
        .response
        .ok_or_else(|| anyhow::anyhow!("missing stage control response"))?
    {
        Response::CoordinatorClaimAccepted(accepted) => {
            let claim = accepted
                .claim
                .ok_or_else(|| anyhow::anyhow!("coordinator claim accepted missing claim"))?;
            Ok(
                crate::inference::skippy::StageControlResponse::ClaimAccepted(
                    crate::inference::skippy::StageCoordinatorClaimAck {
                        accepted: accepted.accepted,
                        claim: stage_coordinator_claim_from_proto(claim)?,
                        error: accepted.error,
                    },
                ),
            )
        }
        Response::StageReady(ready) => {
            let status = ready
                .status
                .ok_or_else(|| anyhow::anyhow!("stage ready missing status"))?;
            Ok(crate::inference::skippy::StageControlResponse::Ready(
                crate::inference::skippy::StageReadyResponse {
                    accepted: ready.accepted,
                    status: stage_status_from_proto(status)?,
                    error: ready.error,
                },
            ))
        }
        Response::StageStatuses(statuses) => {
            Ok(crate::inference::skippy::StageControlResponse::Status(
                statuses
                    .statuses
                    .into_iter()
                    .map(stage_status_from_proto)
                    .collect::<anyhow::Result<Vec<_>>>()?,
            ))
        }
        Response::LayerInventory(inventory) => {
            Ok(crate::inference::skippy::StageControlResponse::Inventory(
                layer_inventory_from_proto(inventory)?,
            ))
        }
        Response::PrepareStageAccepted(accepted) => {
            let status = accepted
                .status
                .ok_or_else(|| anyhow::anyhow!("prepare stage accepted missing status"))?;
            Ok(
                crate::inference::skippy::StageControlResponse::PrepareAccepted(
                    crate::inference::skippy::StagePrepareAcceptedResponse {
                        accepted: accepted.accepted,
                        status: stage_preparation_status_from_proto(status)?,
                        error: accepted.error,
                    },
                ),
            )
        }
        Response::StagePreparationStatus(status) => Ok(
            crate::inference::skippy::StageControlResponse::PreparationStatus(
                stage_preparation_status_from_proto(status)?,
            ),
        ),
        Response::StageStatusAck(ack) => {
            Ok(crate::inference::skippy::StageControlResponse::StatusAck(
                crate::inference::skippy::StageStatusAck {
                    accepted: ack.accepted,
                    error: ack.error,
                },
            ))
        }
    }
}

pub(super) fn layer_inventory_to_proto(
    inventory: crate::inference::skippy::StageLayerInventory,
) -> skippy_stage_proto::LayerInventory {
    skippy_stage_proto::LayerInventory {
        model_id: inventory.model_id,
        package_ref: inventory.package_ref,
        manifest_sha256: inventory.manifest_sha256,
        layer_count: inventory.layer_count,
        ready_ranges: inventory
            .ready_ranges
            .into_iter()
            .map(layer_range_to_proto)
            .collect(),
        available_ranges: inventory
            .available_ranges
            .into_iter()
            .map(layer_range_to_proto)
            .collect(),
        missing_ranges: inventory
            .missing_ranges
            .into_iter()
            .map(layer_range_to_proto)
            .collect(),
        preparing_ranges: inventory
            .preparing_ranges
            .into_iter()
            .map(stage_preparation_status_to_proto)
            .collect(),
        source_model_path: inventory.source_model_path,
        source_model_bytes: inventory.source_model_bytes,
        source_model_kind: source_model_kind_to_proto(inventory.source_model_kind) as i32,
        source_model_sha256: inventory.source_model_sha256,
        content_addressed_local_source: inventory.content_addressed_local_source,
    }
}

pub(super) fn layer_inventory_from_proto(
    inventory: skippy_stage_proto::LayerInventory,
) -> anyhow::Result<crate::inference::skippy::StageLayerInventory> {
    Ok(crate::inference::skippy::StageLayerInventory {
        model_id: inventory.model_id,
        package_ref: inventory.package_ref,
        manifest_sha256: inventory.manifest_sha256,
        layer_count: inventory.layer_count,
        ready_ranges: inventory
            .ready_ranges
            .into_iter()
            .map(layer_range_from_proto)
            .collect(),
        available_ranges: inventory
            .available_ranges
            .into_iter()
            .map(layer_range_from_proto)
            .collect(),
        missing_ranges: inventory
            .missing_ranges
            .into_iter()
            .map(layer_range_from_proto)
            .collect(),
        preparing_ranges: inventory
            .preparing_ranges
            .into_iter()
            .map(stage_preparation_status_from_proto)
            .collect::<anyhow::Result<Vec<_>>>()?,
        source_model_path: inventory.source_model_path,
        source_model_bytes: inventory.source_model_bytes,
        source_model_sha256: inventory.source_model_sha256,
        content_addressed_local_source: inventory.content_addressed_local_source,
        source_model_kind: source_model_kind_from_proto(inventory.source_model_kind),
    })
}

fn source_resolution_policy_to_proto(
    local_source_required: bool,
) -> skippy_stage_proto::SourceResolutionPolicy {
    if local_source_required {
        skippy_stage_proto::SourceResolutionPolicy::LocalRequired
    } else {
        skippy_stage_proto::SourceResolutionPolicy::Fallback
    }
}

fn source_resolution_policy_from_proto(value: i32) -> anyhow::Result<bool> {
    match skippy_stage_proto::SourceResolutionPolicy::try_from(value) {
        Ok(skippy_stage_proto::SourceResolutionPolicy::Fallback) => Ok(false),
        Ok(skippy_stage_proto::SourceResolutionPolicy::LocalRequired) => Ok(true),
        Err(_) => anyhow::bail!("unsupported stage source resolution policy {value}"),
    }
}

pub(super) fn layer_range_to_proto(
    range: crate::inference::skippy::LayerRange,
) -> skippy_stage_proto::LayerRange {
    skippy_stage_proto::LayerRange {
        layer_start: range.layer_start,
        layer_end: range.layer_end,
    }
}

pub(super) fn layer_range_from_proto(
    range: skippy_stage_proto::LayerRange,
) -> crate::inference::skippy::LayerRange {
    crate::inference::skippy::LayerRange {
        layer_start: range.layer_start,
        layer_end: range.layer_end,
    }
}

pub(super) fn source_model_kind_to_proto(
    kind: crate::inference::skippy::SourceModelKind,
) -> skippy_stage_proto::SourceModelKind {
    match kind {
        crate::inference::skippy::SourceModelKind::Unknown => {
            skippy_stage_proto::SourceModelKind::Unspecified
        }
        crate::inference::skippy::SourceModelKind::LayerPackage => {
            skippy_stage_proto::SourceModelKind::LayerPackage
        }
        crate::inference::skippy::SourceModelKind::PlainGguf => {
            skippy_stage_proto::SourceModelKind::PlainGguf
        }
        crate::inference::skippy::SourceModelKind::SplitGguf => {
            skippy_stage_proto::SourceModelKind::SplitGguf
        }
    }
}

pub(super) fn source_model_kind_from_proto(
    value: i32,
) -> crate::inference::skippy::SourceModelKind {
    match skippy_stage_proto::SourceModelKind::try_from(value)
        .unwrap_or(skippy_stage_proto::SourceModelKind::Unspecified)
    {
        skippy_stage_proto::SourceModelKind::Unspecified => {
            crate::inference::skippy::SourceModelKind::Unknown
        }
        skippy_stage_proto::SourceModelKind::LayerPackage => {
            crate::inference::skippy::SourceModelKind::LayerPackage
        }
        skippy_stage_proto::SourceModelKind::PlainGguf => {
            crate::inference::skippy::SourceModelKind::PlainGguf
        }
        skippy_stage_proto::SourceModelKind::SplitGguf => {
            crate::inference::skippy::SourceModelKind::SplitGguf
        }
    }
}

pub(super) fn stage_preparation_status_to_proto(
    status: crate::inference::skippy::StagePreparationStatus,
) -> skippy_stage_proto::StagePreparationStatus {
    use skippy_stage_proto::stage_preparation_status::AdmissionState;
    let admission_state = Some(match status.admission {
        Some(descriptor) => AdmissionState::Admitted(skippy_stage_proto::StageAdmissionAdmitted {
            descriptor: Some(descriptor.into()),
        }),
        None => AdmissionState::Idle(skippy_stage_proto::StageAdmissionIdle {}),
    });
    skippy_stage_proto::StagePreparationStatus {
        topology_id: status.topology_id,
        run_id: status.run_id,
        model_id: status.model_id,
        backend: status.backend,
        package_ref: status.package_ref,
        manifest_sha256: status.manifest_sha256,
        stage_id: status.stage_id,
        stage_index: status.stage_index,
        layer_start: status.layer_start,
        layer_end: status.layer_end,
        state: stage_preparation_state_to_proto(status.state) as i32,
        bytes_done: status.bytes_done,
        bytes_total: status.bytes_total,
        bind_addr: status.bind_addr,
        error: status.error,
        shutdown_generation: status.shutdown_generation,
        coordinator_term: status.coordinator_term,
        coordinator_id: status.coordinator_id.map(|id| id.to_string()),
        lease_until_unix_ms: status.lease_until_unix_ms,
        admission_state,
        activation_codec: stage_activation_codec_to_proto(status.activation_codec) as i32,
        activation_codec_policy: stage_activation_codec_policy_to_proto(
            status.activation_codec_policy,
        ) as i32,
    }
}

pub(super) fn stage_preparation_status_from_proto(
    status: skippy_stage_proto::StagePreparationStatus,
) -> anyhow::Result<crate::inference::skippy::StagePreparationStatus> {
    use skippy_stage_proto::stage_preparation_status::AdmissionState;
    let coordinator_id = status.coordinator_id.and_then(|id| match id.parse() {
        Ok(id) => Some(id),
        Err(error) => {
            tracing::warn!(
                coordinator_id = %id,
                error = %error,
                "invalid stage preparation coordinator_id"
            );
            None
        }
    });
    let admission = match status.admission_state {
        Some(AdmissionState::Idle(_)) => None,
        Some(AdmissionState::Admitted(admitted)) => Some(
            admitted
                .descriptor
                .ok_or_else(|| anyhow::anyhow!("admitted stage preparation missing descriptor"))?
                .try_into()
                .map_err(|error| {
                    anyhow::anyhow!("invalid stage preparation admission descriptor: {error}")
                })?,
        ),
        None => return Err(anyhow::anyhow!("stage preparation missing admission state")),
    };
    Ok(crate::inference::skippy::StagePreparationStatus {
        topology_id: status.topology_id,
        run_id: status.run_id,
        model_id: status.model_id,
        backend: status.backend,
        package_ref: status.package_ref,
        manifest_sha256: status.manifest_sha256,
        stage_id: status.stage_id,
        stage_index: status.stage_index,
        layer_start: status.layer_start,
        layer_end: status.layer_end,
        admission,
        activation_codec: stage_activation_codec_from_proto(status.activation_codec)?,
        activation_codec_policy: stage_activation_codec_policy_from_proto(
            status.activation_codec_policy,
        )?,
        state: stage_preparation_state_from_proto(status.state),
        bytes_done: status.bytes_done,
        bytes_total: status.bytes_total,
        bind_addr: status.bind_addr,
        error: status.error,
        shutdown_generation: status.shutdown_generation,
        coordinator_term: status.coordinator_term,
        coordinator_id,
        lease_until_unix_ms: status.lease_until_unix_ms,
    })
}

pub(super) fn stage_status_to_proto(
    status: crate::inference::skippy::StageStatusSnapshot,
) -> skippy_stage_proto::StageStatus {
    use skippy_stage_proto::stage_status::AdmissionState;
    let projector_path =
        path_free_stage_projector_path(status.package_ref.as_deref(), status.projector_path);
    let admission_state = Some(match status.admission {
        Some(descriptor) => AdmissionState::Admitted(skippy_stage_proto::StageAdmissionAdmitted {
            descriptor: Some(descriptor.into()),
        }),
        None => AdmissionState::Idle(skippy_stage_proto::StageAdmissionIdle {}),
    });
    skippy_stage_proto::StageStatus {
        topology_id: status.topology_id,
        run_id: status.run_id,
        model_id: status.model_id,
        backend: status.backend,
        stage_id: status.stage_id,
        stage_index: status.stage_index,
        layer_start: status.layer_start,
        layer_end: status.layer_end,
        state: stage_runtime_state_to_proto(status.state) as i32,
        bind_addr: status.bind_addr,
        input_activation_boundary: status
            .input_activation_boundary
            .map(activation_boundary_to_proto),
        output_activation_boundary: status
            .output_activation_boundary
            .map(activation_boundary_to_proto),
        error: status.error,
        shutdown_generation: status.shutdown_generation,
        selected_device: status.selected_device.map(stage_device_to_proto),
        ctx_size: status.ctx_size,
        lane_count: status.lane_count,
        n_batch: status.n_batch,
        n_ubatch: status.n_ubatch,
        package_ref: status.package_ref,
        manifest_sha256: status.manifest_sha256,
        source_model_path: status.source_model_path,
        source_model_sha256: status.source_model_sha256,
        source_model_bytes: status.source_model_bytes,
        materialized_path: status.materialized_path,
        materialized_pinned: Some(status.materialized_pinned),
        projector_path,
        flash_attn_type: stage_flash_attn_type_to_proto(status.flash_attn_type) as i32,
        coordinator_term: status.coordinator_term,
        coordinator_id: status.coordinator_id.map(|id| id.to_string()),
        lease_until_unix_ms: status.lease_until_unix_ms,
        admission_state,
        activation_codec: stage_activation_codec_to_proto(status.activation_codec) as i32,
        activation_codec_policy: stage_activation_codec_policy_to_proto(
            status.activation_codec_policy,
        ) as i32,
    }
}

pub(super) fn stage_status_from_proto(
    status: skippy_stage_proto::StageStatus,
) -> anyhow::Result<crate::inference::skippy::StageStatusSnapshot> {
    use skippy_stage_proto::stage_status::AdmissionState;
    let projector_path =
        path_free_stage_projector_path(status.package_ref.as_deref(), status.projector_path);
    let admission = match status.admission_state {
        Some(AdmissionState::Idle(_)) => None,
        Some(AdmissionState::Admitted(admitted)) => Some(
            admitted
                .descriptor
                .ok_or_else(|| anyhow::anyhow!("admitted stage status missing descriptor"))?
                .try_into()
                .map_err(|error| {
                    anyhow::anyhow!("invalid stage status admission descriptor: {error}")
                })?,
        ),
        None => return Err(anyhow::anyhow!("stage status missing admission state")),
    };
    Ok(crate::inference::skippy::StageStatusSnapshot {
        topology_id: status.topology_id,
        run_id: status.run_id,
        model_id: status.model_id,
        backend: status.backend,
        stage_id: status.stage_id,
        stage_index: status.stage_index,
        layer_start: status.layer_start,
        layer_end: status.layer_end,
        admission,
        activation_codec: stage_activation_codec_from_proto(status.activation_codec)?,
        activation_codec_policy: stage_activation_codec_policy_from_proto(
            status.activation_codec_policy,
        )?,
        state: stage_runtime_state_from_proto(status.state),
        bind_addr: status.bind_addr,
        input_activation_boundary: status
            .input_activation_boundary
            .map(activation_boundary_from_proto),
        output_activation_boundary: status
            .output_activation_boundary
            .map(activation_boundary_from_proto),
        selected_device: status
            .selected_device
            .map(stage_device_from_proto)
            .transpose()?,
        ctx_size: status.ctx_size,
        lane_count: if status.lane_count == 0 {
            4
        } else {
            status.lane_count
        },
        n_batch: status.n_batch,
        n_ubatch: status.n_ubatch,
        package_ref: status.package_ref,
        manifest_sha256: status.manifest_sha256,
        source_model_path: status.source_model_path,
        source_model_sha256: status.source_model_sha256,
        source_model_bytes: status.source_model_bytes,
        materialized_path: status.materialized_path,
        materialized_pinned: status.materialized_pinned.unwrap_or(false),
        projector_path,
        flash_attn_type: stage_flash_attn_type_from_proto(status.flash_attn_type),
        error: status.error,
        shutdown_generation: status.shutdown_generation,
        coordinator_term: status.coordinator_term,
        coordinator_id: status
            .coordinator_id
            .map(|id| id.parse())
            .transpose()
            .context("invalid stage status coordinator_id")?,
        lease_until_unix_ms: status.lease_until_unix_ms,
    })
}

fn activation_boundary_to_proto(
    value: skippy_runtime::ActivationBoundaryDesc,
) -> skippy_stage_proto::ActivationBoundaryDescriptor {
    skippy_stage_proto::ActivationBoundaryDescriptor {
        version: value.version,
        ggml_type: value.ggml_type,
        layout: value.layout,
        elements_per_token: value.elements_per_token,
        bytes_per_token: value.bytes_per_token,
        required_frame_flags: value.required_frame_flags,
        required_sidebands: value.required_sidebands,
    }
}

fn activation_boundary_from_proto(
    value: skippy_stage_proto::ActivationBoundaryDescriptor,
) -> skippy_runtime::ActivationBoundaryDesc {
    skippy_runtime::ActivationBoundaryDesc {
        version: value.version,
        ggml_type: value.ggml_type,
        layout: value.layout,
        elements_per_token: value.elements_per_token,
        bytes_per_token: value.bytes_per_token,
        required_frame_flags: value.required_frame_flags,
        required_sidebands: value.required_sidebands,
    }
}

pub(super) fn stage_flash_attn_type_to_proto(
    value: skippy_protocol::FlashAttentionType,
) -> skippy_stage_proto::StageFlashAttnType {
    match value {
        skippy_protocol::FlashAttentionType::Auto => skippy_stage_proto::StageFlashAttnType::Auto,
        skippy_protocol::FlashAttentionType::Disabled => {
            skippy_stage_proto::StageFlashAttnType::Disabled
        }
        skippy_protocol::FlashAttentionType::Enabled => {
            skippy_stage_proto::StageFlashAttnType::Enabled
        }
    }
}

pub(super) fn stage_flash_attn_type_from_proto(value: i32) -> skippy_protocol::FlashAttentionType {
    match skippy_stage_proto::StageFlashAttnType::try_from(value)
        .unwrap_or(skippy_stage_proto::StageFlashAttnType::Unspecified)
    {
        skippy_stage_proto::StageFlashAttnType::Unspecified
        | skippy_stage_proto::StageFlashAttnType::Auto => skippy_protocol::FlashAttentionType::Auto,
        skippy_stage_proto::StageFlashAttnType::Disabled => {
            skippy_protocol::FlashAttentionType::Disabled
        }
        skippy_stage_proto::StageFlashAttnType::Enabled => {
            skippy_protocol::FlashAttentionType::Enabled
        }
    }
}

fn stage_glm_dsa_policy_to_proto(
    value: skippy_protocol::GlmDsaPolicy,
) -> skippy_stage_proto::StageGlmDsaPolicy {
    match value {
        skippy_protocol::GlmDsaPolicy::Auto => skippy_stage_proto::StageGlmDsaPolicy::Auto,
        skippy_protocol::GlmDsaPolicy::V1 => skippy_stage_proto::StageGlmDsaPolicy::V1,
    }
}

fn stage_glm_dsa_policy_from_proto(value: i32) -> skippy_protocol::GlmDsaPolicy {
    match skippy_stage_proto::StageGlmDsaPolicy::try_from(value)
        .unwrap_or(skippy_stage_proto::StageGlmDsaPolicy::Auto)
    {
        skippy_stage_proto::StageGlmDsaPolicy::Auto => skippy_protocol::GlmDsaPolicy::Auto,
        skippy_stage_proto::StageGlmDsaPolicy::V1 => skippy_protocol::GlmDsaPolicy::V1,
    }
}

pub(super) fn stage_runtime_state_from_proto(
    value: i32,
) -> crate::inference::skippy::StageRuntimeState {
    match skippy_stage_proto::StageRuntimeState::try_from(value)
        .unwrap_or(skippy_stage_proto::StageRuntimeState::Failed)
    {
        skippy_stage_proto::StageRuntimeState::Starting => {
            crate::inference::skippy::StageRuntimeState::Starting
        }
        skippy_stage_proto::StageRuntimeState::Ready => {
            crate::inference::skippy::StageRuntimeState::Ready
        }
        skippy_stage_proto::StageRuntimeState::Stopping => {
            crate::inference::skippy::StageRuntimeState::Stopping
        }
        skippy_stage_proto::StageRuntimeState::Stopped
        | skippy_stage_proto::StageRuntimeState::Unspecified => {
            crate::inference::skippy::StageRuntimeState::Stopped
        }
        skippy_stage_proto::StageRuntimeState::Failed => {
            crate::inference::skippy::StageRuntimeState::Failed
        }
    }
}

pub(super) fn stage_runtime_state_to_proto(
    state: crate::inference::skippy::StageRuntimeState,
) -> skippy_stage_proto::StageRuntimeState {
    match state {
        crate::inference::skippy::StageRuntimeState::Starting => {
            skippy_stage_proto::StageRuntimeState::Starting
        }
        crate::inference::skippy::StageRuntimeState::Ready => {
            skippy_stage_proto::StageRuntimeState::Ready
        }
        crate::inference::skippy::StageRuntimeState::Stopping => {
            skippy_stage_proto::StageRuntimeState::Stopping
        }
        crate::inference::skippy::StageRuntimeState::Stopped => {
            skippy_stage_proto::StageRuntimeState::Stopped
        }
        crate::inference::skippy::StageRuntimeState::Failed => {
            skippy_stage_proto::StageRuntimeState::Failed
        }
    }
}

pub(super) fn stage_preparation_state_from_proto(
    value: i32,
) -> crate::inference::skippy::StagePreparationState {
    match skippy_stage_proto::StagePreparationState::try_from(value)
        .unwrap_or(skippy_stage_proto::StagePreparationState::Unspecified)
    {
        skippy_stage_proto::StagePreparationState::Assigned
        | skippy_stage_proto::StagePreparationState::Unspecified => {
            crate::inference::skippy::StagePreparationState::Assigned
        }
        skippy_stage_proto::StagePreparationState::Downloading => {
            crate::inference::skippy::StagePreparationState::Downloading
        }
        skippy_stage_proto::StagePreparationState::Available => {
            crate::inference::skippy::StagePreparationState::Available
        }
        skippy_stage_proto::StagePreparationState::Resolving => {
            crate::inference::skippy::StagePreparationState::Resolving
        }
        skippy_stage_proto::StagePreparationState::Loading => {
            crate::inference::skippy::StagePreparationState::Loading
        }
        skippy_stage_proto::StagePreparationState::Ready => {
            crate::inference::skippy::StagePreparationState::Ready
        }
        skippy_stage_proto::StagePreparationState::Failed => {
            crate::inference::skippy::StagePreparationState::Failed
        }
        skippy_stage_proto::StagePreparationState::Cancelled => {
            crate::inference::skippy::StagePreparationState::Cancelled
        }
    }
}

pub(super) fn stage_preparation_state_to_proto(
    state: crate::inference::skippy::StagePreparationState,
) -> skippy_stage_proto::StagePreparationState {
    match state {
        crate::inference::skippy::StagePreparationState::Assigned => {
            skippy_stage_proto::StagePreparationState::Assigned
        }
        crate::inference::skippy::StagePreparationState::Downloading => {
            skippy_stage_proto::StagePreparationState::Downloading
        }
        crate::inference::skippy::StagePreparationState::Available => {
            skippy_stage_proto::StagePreparationState::Available
        }
        crate::inference::skippy::StagePreparationState::Resolving => {
            skippy_stage_proto::StagePreparationState::Resolving
        }
        crate::inference::skippy::StagePreparationState::Loading => {
            skippy_stage_proto::StagePreparationState::Loading
        }
        crate::inference::skippy::StagePreparationState::Ready => {
            skippy_stage_proto::StagePreparationState::Ready
        }
        crate::inference::skippy::StagePreparationState::Failed => {
            skippy_stage_proto::StagePreparationState::Failed
        }
        crate::inference::skippy::StagePreparationState::Cancelled => {
            skippy_stage_proto::StagePreparationState::Cancelled
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[derive(Clone, PartialEq, Message)]
    struct LegacyLoadStage {}

    #[derive(Clone, PartialEq, Message)]
    struct CompatLoadStage {
        #[prost(message, optional, tag = "43")]
        runtime_settings: Option<CompatRuntimeSettings>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct CompatRuntimeSettings {
        #[prost(bool, optional, tag = "1")]
        repack: Option<bool>,
        #[prost(bool, optional, tag = "2")]
        op_offload: Option<bool>,
        #[prost(bool, optional, tag = "3")]
        no_host_buffer: Option<bool>,
        #[prost(bool, optional, tag = "4")]
        check_tensors: Option<bool>,
        #[prost(bool, optional, tag = "5")]
        direct_io: Option<bool>,
        #[prost(uint32, optional, tag = "6")]
        main_gpu: Option<u32>,
        #[prost(int32, optional, tag = "7")]
        split_mode: Option<i32>,
        #[prost(bool, optional, tag = "8")]
        kv_offload: Option<bool>,
        #[prost(bool, optional, tag = "9")]
        kv_unified: Option<bool>,
        #[prost(bool, optional, tag = "10")]
        swa_full: Option<bool>,
        #[prost(uint32, optional, tag = "11")]
        cache_idle_slots: Option<u32>,
    }

    #[test]
    fn stage_load_wire_round_trip_preserves_runtime_controls() {
        let settings = CompatRuntimeSettings {
            repack: Some(true),
            op_offload: Some(false),
            no_host_buffer: Some(true),
            check_tensors: Some(true),
            direct_io: Some(true),
            main_gpu: Some(2),
            split_mode: Some(3),
            kv_offload: Some(false),
            kv_unified: Some(true),
            swa_full: Some(false),
            cache_idle_slots: Some(3),
        };
        let load = skippy_stage_proto::LoadStage {
            admission: Some(crate::inference::skippy::test_stage_admission(0, 1).into()),
            activation_codec: skippy_stage_proto::StageActivationCodec::F16RneV1 as i32,
            activation_codec_policy: Default::default(),
            ..Default::default()
        };
        let mut encoded = load.encode_to_vec();
        CompatLoadStage {
            runtime_settings: Some(settings.clone()),
        }
        .encode(&mut encoded)
        .expect("compat runtime settings encode");

        let legacy = LegacyLoadStage::decode(encoded.as_slice())
            .expect("legacy schema ignores additive runtime settings");
        assert!(legacy.encode_to_vec().is_empty());

        let decoded = skippy_stage_proto::LoadStage::decode(encoded.as_slice())
            .expect("production load stage decode");
        let request = stage_load_from_proto(decoded).expect("domain load request");
        let reencoded = stage_load_to_proto(request).encode_to_vec();
        let observed = CompatLoadStage::decode(reencoded.as_slice())
            .expect("compat runtime settings decode")
            .runtime_settings
            .expect("runtime settings must survive production round trip");

        assert_eq!(observed, settings);
        assert_eq!(
            stage_runtime_settings_from_proto(None).unwrap(),
            crate::inference::skippy::StageLoadRuntimeSettings::default()
        );
    }

    #[test]
    fn activation_boundary_descriptor_wire_round_trip_preserves_every_field() {
        let expected = skippy_runtime::ActivationBoundaryDesc {
            version: 7,
            ggml_type: 8,
            layout: 3,
            elements_per_token: 2_560,
            bytes_per_token: 10_240,
            required_frame_flags: 0x04,
            required_sidebands: 0x02,
        };
        let encoded = activation_boundary_to_proto(expected).encode_to_vec();
        let decoded = skippy_stage_proto::ActivationBoundaryDescriptor::decode(encoded.as_slice())
            .expect("decode activation boundary descriptor");

        assert_eq!(activation_boundary_from_proto(decoded), expected);
    }
}
