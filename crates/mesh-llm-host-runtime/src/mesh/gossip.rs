//! Gossip protocol: peer announcement exchange, transitive peer tracking,
//! and peer list management (add/remove/update).

use super::{
    DEAD_PEER_TTL, DisplayLatencySource, InviteTokenMaterial, MeshOperationalEvent,
    MeshPeerRemovalReason, MeshPolicyRejectionReason, ModelDemand, ModelRuntimeDescriptor, Node,
    NodeRole, PEER_CONNECT_AND_GOSSIP_TIMEOUT, PEER_STALE_SECS, PeerAnnouncement, PeerInfo,
    ServedModelDescriptor, SignedNodeOwnership, connect_mesh, elapsed_ms_u64, emit_mesh_info,
    infer_remote_served_descriptors, mesh_peer_operational_context, parse_invite_token,
    record_mesh_operational_event, record_mesh_operational_event_with_context,
};
use crate::crypto::{OwnershipSummary, verify_node_ownership};
use crate::mesh::cache_affinity_gossip;
use crate::mesh::peer_state::{PropagatedLatencyObservation, policy_accepts_peer};
use crate::mesh::requirements::current_time_unix_ms;
use crate::mesh::stage_transport::PeerLifecycleCaptureEvent;
use crate::models::append_external_inference_models;
use crate::protocol::{
    ControlProtocol, NODE_PROTOCOL_GENERATION, STREAM_GOSSIP, connection_protocol,
    decode_gossip_payload, read_len_prefixed, write_gossip_payload,
};
use anyhow::Result;
use iroh::{EndpointAddr, EndpointId, endpoint::Connection};
use std::collections::HashMap;

/// Minimum peer version we accept into the local mesh table and re-broadcast.
///
/// Peers below this floor are rejected at ingest in both `add_peer`
/// (direct gossip exchange) and `update_transitive_peer` (gossip relayed
/// by a bridge peer). They do not appear in `/api/status`, do not appear
/// in the UI, and are not included in outbound gossip. A peer that updates
/// and re-announces with a version at or above the floor is accepted on
/// the next exchange.
///
/// v0.60.0 is the cut where the on-wire `hardware` block landed; peers
/// older than that predate several gossip fields the current mesh relies
/// on. Peers that don't advertise a version at all (some legacy nodes
/// leave the field unset) are conservatively accepted, on the theory that
/// a missing version is more likely to be a legitimate old node than a
/// targeted bypass.
const MIN_REBROADCAST_VERSION_MAJOR: u64 = 0;
const MIN_REBROADCAST_VERSION_MINOR: u64 = 60;
const CLIENT_AUTO_JOIN_PROBE_LIMIT: usize = 4;
const CLIENT_AUTO_JOIN_PROBE_TIMEOUT: std::time::Duration = PEER_CONNECT_AND_GOSSIP_TIMEOUT;

#[derive(Clone, Copy)]
pub(crate) struct AnnouncedPeerContext {
    remote: EndpointId,
    rtt_ms: Option<u32>,
    negotiated_protocol_generation: Option<u32>,
    direct_peer_requirements_validated: bool,
}

impl AnnouncedPeerContext {
    const fn direct(remote: EndpointId, negotiated_protocol_generation: Option<u32>) -> Self {
        Self {
            remote,
            rtt_ms: None,
            negotiated_protocol_generation,
            direct_peer_requirements_validated: true,
        }
    }
}

pub(crate) struct JoinProbeCandidate {
    token: String,
    mesh_name: Option<String>,
    addr: EndpointAddr,
}

pub(crate) struct JoinProbeSuccess {
    candidate: JoinProbeCandidate,
    conn: Connection,
    announcements: Vec<(EndpointAddr, PeerAnnouncement)>,
    rtt_ms: u32,
    elapsed: std::time::Duration,
}

#[cfg(test)]
impl JoinProbeSuccess {
    /// Test-only constructor so sibling test modules can drive
    /// `commit_join_probe_success` against a real QUIC connection.
    pub(super) fn new_for_tests(
        token: String,
        mesh_name: Option<String>,
        addr: EndpointAddr,
        conn: Connection,
        announcements: Vec<(EndpointAddr, PeerAnnouncement)>,
        rtt_ms: u32,
    ) -> Self {
        Self {
            candidate: JoinProbeCandidate {
                token,
                mesh_name,
                addr,
            },
            conn,
            announcements,
            rtt_ms,
            elapsed: std::time::Duration::from_millis(0),
        }
    }
}

pub(crate) fn emit_join_probe_race_started(candidate_count: usize) {
    tracing::info!(
        candidates = candidate_count,
        timeout_ms = CLIENT_AUTO_JOIN_PROBE_TIMEOUT.as_millis(),
        "Racing auto-join bootstrap candidates"
    );
    emit_mesh_info(format!(
        "Racing {candidate_count} auto-join bootstrap candidates"
    ));
}

pub(crate) fn emit_join_probe_fallback(last_error: Option<&anyhow::Error>) {
    if let Some(error) = last_error {
        tracing::debug!(
            "No auto-join candidate completed the fast probe; falling back to serial join: {error:#}"
        );
    }
    emit_mesh_info(
        "No auto-join candidate completed the fast probe; falling back to serial join".to_string(),
    );
}

/// Returns `true` if `version` is recent enough to include in outbound
/// gossip. `None` (no advertised version) returns `true` for back-compat.
/// Build metadata after `+` is stripped before parsing.
pub(super) fn version_allowed_for_rebroadcast(version: Option<&str>) -> bool {
    let Some(raw) = version else {
        return true;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return true;
    }
    // Strip build metadata ("0.65.1+skippy.20260504.kv.2" → "0.65.1") and
    // pre-release tag ("0.63.0-rc5" → "0.63.0") so the comparison is
    // purely on the major.minor numeric pair.
    let core = trimmed
        .split('+')
        .next()
        .unwrap_or(trimmed)
        .split('-')
        .next()
        .unwrap_or(trimmed);
    let mut parts = core.split('.');
    let Some(major) = parts.next().and_then(|s| s.parse::<u64>().ok()) else {
        return true; // Unparseable — don't penalise; conservative default.
    };
    let Some(minor) = parts.next().and_then(|s| s.parse::<u64>().ok()) else {
        return true;
    };
    if major != MIN_REBROADCAST_VERSION_MAJOR {
        // Any major > floor (e.g. v1.x.y) is allowed; any major < floor is
        // refused. With MIN_REBROADCAST_VERSION_MAJOR == 0, the "less than"
        // case cannot occur, but we keep the comparison structure for the
        // day the floor bumps to a non-zero major.
        return major > MIN_REBROADCAST_VERSION_MAJOR;
    }
    minor >= MIN_REBROADCAST_VERSION_MINOR
}

/// Returns `true` if the announcement describes a peer the mesh has no
/// observable use for via transitive gossip: a `Client`-role peer that
/// advertises **no identity** (no hostname), has **never been directly
/// measured** by any peer in the mesh, and has **no model interests**
/// (no requested/serving/hosted models).
///
/// Three independent signals must all be absent before we treat a peer
/// as a gossip-only ghost:
///
/// 1. `hostname` — populated synchronously by `system::hardware::survey()`
///    at node construction. Every real client on every supported platform
///    has one from its first gossip frame.
///
/// 2. `latency_source == Direct` — set when *any* peer in the mesh has
///    measured this peer's RTT via direct contact, then propagated
///    through gossip. A peer with a direct measurement is real — someone
///    reached it on the network. The v0.57 swarm uniformly has
///    `latency_source = Unknown`; no peer has ever directly contacted
///    one.
///
/// 3. model interests (`requested`/`serving`/`hosted`) — any of these
///    being populated makes the peer useful to the mesh (demand signal
///    or routable capacity).
///
/// A peer that fails all three is invisible to routing, untraceable on
/// the network, and contributes no demand signal. Real idle clients
/// survive: they have a hostname. Real reachable clients survive: they
/// have a direct measurement. Real demand-signaling clients survive:
/// they have a requested model.
///
/// Direct ingest in `add_peer` ignores this check — a client we actually
/// connect to is admitted regardless of what they advertise.
pub(super) fn peer_is_idle_transitive_client(ann: &PeerAnnouncement) -> bool {
    let directly_measured = matches!(
        ann.latency_source,
        Some(crate::proto::node::LatencySource::Direct)
    );
    matches!(ann.role, NodeRole::Client)
        && ann.hostname.is_none()
        && !directly_measured
        && ann.requested_models.is_empty()
        && ann.serving_models.is_empty()
        && ann
            .hosted_models
            .as_ref()
            .map(|h| h.is_empty())
            .unwrap_or(true)
}

pub(crate) struct LocalAnnouncementData {
    role: NodeRole,
    first_joined_mesh_ts: Option<u64>,
    models: Vec<String>,
    model_source: Option<String>,
    serving_models: Vec<String>,
    hosted_models: Vec<String>,
    available_models: Vec<String>,
    requested_models: Vec<String>,
    explicit_model_interests: Vec<String>,
    model_demand: HashMap<String, ModelDemand>,
    mesh_id: Option<String>,
    mesh_policy_hash: Option<String>,
    signed_genesis_policy: Option<crate::SignedMeshGenesisPolicy>,
    release_attestation: Option<crate::ReleaseBuildAttestation>,
    direct_admission_proof: Option<crate::DirectNodeAdmissionProof>,
    available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    available_model_sizes: HashMap<String, u64>,
    served_model_descriptors: Vec<ServedModelDescriptor>,
    served_model_runtime: Vec<ModelRuntimeDescriptor>,
    owner_attestation: Option<SignedNodeOwnership>,
    artifact_transfer_supported: bool,
    advertised_model_throughput: Vec<crate::network::metrics::ModelThroughputHint>,
    cache_affinity: Option<mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement>,
    gpu_mem_bandwidth_gbps: Option<String>,
    gpu_compute_tflops_fp32: Option<String>,
    gpu_compute_tflops_fp16: Option<String>,
    inference_admission_state: Option<crate::proto::node::InferenceAdmissionState>,
}

pub(crate) struct RebroadcastAnnouncements {
    announcements: Vec<PeerAnnouncement>,
    filtered_old_version: usize,
}

pub fn backfill_legacy_descriptors(ann: &mut PeerAnnouncement) {
    if ann.served_model_descriptors.is_empty() {
        let primary_model_name = ann
            .serving_models
            .first()
            .map(String::as_str)
            .unwrap_or_default()
            .to_string();
        ann.served_model_descriptors = infer_remote_served_descriptors(
            &primary_model_name,
            &ann.serving_models,
            ann.model_source.as_deref(),
        );
    }
}

pub(super) fn peer_meaningfully_changed(old: &PeerInfo, new: &PeerInfo) -> bool {
    old.addr != new.addr
        || old.mesh_id != new.mesh_id
        || old.mesh_policy_hash != new.mesh_policy_hash
        || old.genesis_policy != new.genesis_policy
        || old.role != new.role
        || old.first_joined_mesh_ts != new.first_joined_mesh_ts
        || old.models != new.models
        || old.vram_bytes != new.vram_bytes
        || old.rtt_ms != new.rtt_ms
        || old.model_source != new.model_source
        || old.serving_models != new.serving_models
        || old.hosted_models_known != new.hosted_models_known
        || old.hosted_models != new.hosted_models
        || old.available_models != new.available_models
        || old.requested_models != new.requested_models
        || old.explicit_model_interests != new.explicit_model_interests
        || old.served_model_descriptors != new.served_model_descriptors
        || old.served_model_runtime != new.served_model_runtime
        || old.artifact_transfer_supported != new.artifact_transfer_supported
        || old.stage_protocol_generation_supported != new.stage_protocol_generation_supported
        || old.stage_status_list_supported != new.stage_status_list_supported
        || old.local_gguf_content_id_supported != new.local_gguf_content_id_supported
        || cache_affinity_gossip::advertised_state_changed(&old.cache_affinity, &new.cache_affinity)
        || old.version != new.version
        || old.owner_summary != new.owner_summary
        || old.gpu_reserved_bytes != new.gpu_reserved_bytes
        || old.memory != new.memory
        || old.propagated_latency != new.propagated_latency
        || old.inference_admission_state != new.inference_admission_state
}

pub(crate) fn merge_first_joined_mesh_ts(existing: &mut Option<u64>, incoming: Option<u64>) {
    match (*existing, incoming) {
        (None, Some(v)) => *existing = Some(v),
        (Some(_), None) => {}
        (Some(a), Some(b)) => *existing = Some(a.min(b)),
        (None, None) => {}
    }
}

pub(super) fn apply_transitive_ann(
    existing: &mut PeerInfo,
    addr: &EndpointAddr,
    ann: &PeerAnnouncement,
    bridge_id: EndpointId,
) -> bool {
    let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
    existing.mesh_id = ann.mesh_id.clone();
    existing.mesh_policy_hash = ann.mesh_policy_hash.clone();
    existing.genesis_policy = ann.genesis_policy.clone();
    let serving_changed = existing.serving_models != ann.serving_models
        || existing.hosted_models != ann_hosted_models
        || existing.hosted_models_known != ann.hosted_models.is_some();
    existing.serving_models = ann.serving_models.clone();
    existing.hosted_models = ann_hosted_models;
    existing.hosted_models_known = ann.hosted_models.is_some();
    existing.role = ann.role.clone();
    merge_first_joined_mesh_ts(&mut existing.first_joined_mesh_ts, ann.first_joined_mesh_ts);
    let capacity_changed = existing.vram_bytes != ann.vram_bytes;
    existing.vram_bytes = ann.vram_bytes;
    // Only advance addr if the transitive announcement is at least as path-rich,
    // so a direct peer's richer address is not overwritten by a weaker transitive one.
    if !addr.addrs.is_empty() && addr.addrs.len() >= existing.addr.addrs.len() {
        existing.addr = addr.clone();
    }
    if ann.version.is_some() {
        existing.version = ann.version.clone();
    }
    if ann.gpu_name.is_some() {
        existing.gpu_name = ann.gpu_name.clone();
    }
    if ann.hostname.is_some() {
        existing.hostname = ann.hostname.clone();
    }
    if ann.is_soc.is_some() {
        existing.is_soc = ann.is_soc;
    }
    if ann.gpu_vram.is_some() {
        existing.gpu_vram = ann.gpu_vram.clone();
    }
    if ann.gpu_reserved_bytes.is_some() {
        existing.gpu_reserved_bytes = ann.gpu_reserved_bytes.clone();
    }
    match ann.memory {
        Some(memory) => existing.memory = Some(memory),
        // A relay that predates the block strips it. The cached block only
        // explains the capacity it arrived with: keep it while that capacity
        // is unchanged, drop it once the capacity moved, so a stale breakdown
        // is never paired with the new budget and rebroadcast as such.
        None if capacity_changed => existing.memory = None,
        None => {}
    }
    if ann.gpu_mem_bandwidth_gbps.is_some() {
        existing.gpu_mem_bandwidth_gbps = ann.gpu_mem_bandwidth_gbps.clone();
    }
    if ann.gpu_compute_tflops_fp32.is_some() {
        existing.gpu_compute_tflops_fp32 = ann.gpu_compute_tflops_fp32.clone();
    }
    if ann.gpu_compute_tflops_fp16.is_some() {
        existing.gpu_compute_tflops_fp16 = ann.gpu_compute_tflops_fp16.clone();
    }
    existing.models = ann.models.clone();
    existing.available_models.clear();
    existing.requested_models = ann.requested_models.clone();
    existing.explicit_model_interests = ann.explicit_model_interests.clone();
    existing.owner_attestation = ann.owner_attestation.clone();
    if ann.model_source.is_some() {
        existing.model_source = ann.model_source.clone();
    }
    existing.served_model_descriptors = ann.served_model_descriptors.clone();
    existing.served_model_runtime = ann.served_model_runtime.clone();
    existing.artifact_transfer_supported = ann.artifact_transfer_supported;
    existing.stage_protocol_generation_supported = ann.stage_protocol_generation_supported;
    existing.stage_status_list_supported = ann.stage_status_list_supported;
    // Strict local-source admission requires capability provenance from the
    // peer itself. A transitive announcer is not authoritative in either
    // direction, so it may neither promote nor clear this support bit. Direct
    // announcements in `add_peer` update it authoritatively.
    existing.advertised_model_throughput = ann.advertised_model_throughput.clone();
    cache_affinity_gossip::merge_advertisement(
        &mut existing.cache_affinity,
        ann.cache_affinity.as_ref(),
        false,
    );
    if ann.inference_admission_state.is_some() {
        existing.inference_admission_state = ann.inference_admission_state;
    }
    if ann.experts_summary.is_some() {
        existing.experts_summary = ann.experts_summary.clone();
    }
    // Propagate latency from the announcement (transitive gossip).
    if let Some(latency_ms) = ann.latency_ms {
        let source = ann
            .latency_source
            .unwrap_or(crate::proto::node::LatencySource::Unspecified);
        let is_propagatable_source = matches!(
            source,
            crate::proto::node::LatencySource::Direct
                | crate::proto::node::LatencySource::Estimated
        );
        if latency_ms > 0 && is_propagatable_source {
            let observer_id = ann
                .latency_observer_id
                .as_ref()
                .and_then(|id_bytes| EndpointId::from_bytes(id_bytes).ok());
            existing.propagated_latency = Some(PropagatedLatencyObservation {
                latency_ms,
                age_ms_at_received: ann.latency_age_ms.unwrap_or(0),
                received_at: std::time::Instant::now(),
                observer_id: observer_id.or(Some(bridge_id)),
            });
        }
    }
    serving_changed
}

impl Node {
    async fn validate_direct_announcement_before_payload_apply(
        &self,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        context: AnnouncedPeerContext,
    ) -> Result<()> {
        let direct_announcement = their_announcements
            .iter()
            .find_map(|(addr, ann)| (addr.id == context.remote).then_some(ann))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "gossip payload from {} omitted its direct announcement",
                    context.remote.fmt_short()
                )
            })?;
        if let Err(reason) = self
            .validate_direct_peer_requirements(
                context.remote,
                direct_announcement,
                context.negotiated_protocol_generation,
            )
            .await
        {
            self.record_mesh_requirement_rejection(
                super::requirements::MeshRequirementRejectionSource::Gossip,
                Some(context.remote),
                reason.clone(),
            )
            .await;
            self.state
                .lock()
                .await
                .requirement_rejected_peers
                .insert(context.remote);
            anyhow::bail!(
                "peer {} rejected by mesh requirements: {}",
                context.remote.fmt_short(),
                reason.code()
            );
        }
        Ok(())
    }

    async fn validate_and_capture_inbound_gossip(
        &self,
        protocol: ControlProtocol,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        context: AnnouncedPeerContext,
    ) -> Result<()> {
        self.validate_direct_announcement_before_payload_apply(their_announcements, context)
            .await?;

        let (recovered_from_dead, prior_state) = {
            let mut state = self.state.lock().await;
            let recovered_from_dead = state.dead_peers.remove(&context.remote).is_some();
            let prior_state = state
                .peers
                .get(&context.remote)
                .map(|peer| {
                    if peer.last_seen >= peer.last_mentioned {
                        "direct"
                    } else {
                        "transitive"
                    }
                })
                .unwrap_or("unknown")
                .to_string();
            if recovered_from_dead {
                super::emit_mesh_info(format!(
                    "🔄 Dead peer {} is gossiping — clearing dead status",
                    context.remote.fmt_short()
                ));
            }
            (recovered_from_dead, prior_state)
        };
        self.capture_gossip_inbound(context.remote, protocol, their_announcements.len());
        self.capture_direct_proof_of_life(
            context.remote,
            protocol,
            their_announcements.len(),
            recovered_from_dead,
            &prior_state,
        );
        Ok(())
    }

    pub(crate) async fn apply_announced_peer(
        &self,
        peer_id: EndpointId,
        addr: &EndpointAddr,
        ann: &PeerAnnouncement,
        context: AnnouncedPeerContext,
    ) -> Result<()> {
        let remote = context.remote;
        if peer_id == self.endpoint.id() {
            return Ok(());
        }
        if peer_id == remote {
            if !context.direct_peer_requirements_validated
                && let Err(reason) = self
                    .validate_direct_peer_requirements(
                        remote,
                        ann,
                        context.negotiated_protocol_generation,
                    )
                    .await
            {
                self.record_mesh_requirement_rejection(
                    super::requirements::MeshRequirementRejectionSource::Gossip,
                    Some(remote),
                    reason.clone(),
                )
                .await;
                self.state
                    .lock()
                    .await
                    .requirement_rejected_peers
                    .insert(remote);
                anyhow::bail!(
                    "peer {} rejected by mesh requirements: {}",
                    remote.fmt_short(),
                    reason.code()
                );
            }
            if self
                .add_peer_after_direct_requirements_validated(
                    remote,
                    addr.clone(),
                    ann,
                    context.negotiated_protocol_generation,
                )
                .await
            {
                if let Some(ref their_id) = ann.mesh_id {
                    self.set_mesh_id(their_id.clone()).await;
                }
                self.merge_remote_demand(&ann.model_demand);
                if let Some(rtt_ms) = context.rtt_ms {
                    self.update_peer_rtt(remote, rtt_ms).await;
                }
            }
            return Ok(());
        }
        if let Err(err) = self
            .validate_peer_announcement_against_active_policy(peer_id, ann)
            .await
        {
            tracing::debug!(
                "ignoring transitive peer {} because its policy announcement did not match the active mesh: {}",
                peer_id.fmt_short(),
                err.code()
            );
            return Ok(());
        }
        self.update_transitive_peer(peer_id, addr, ann, remote)
            .await;
        Ok(())
    }

    pub(crate) async fn apply_announced_peers(
        &self,
        remote: EndpointId,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        rtt_ms: Option<u32>,
        negotiated_protocol_generation: Option<u32>,
        direct_peer_requirements_validated: bool,
    ) -> Result<()> {
        let context = AnnouncedPeerContext {
            remote,
            rtt_ms,
            negotiated_protocol_generation,
            direct_peer_requirements_validated,
        };
        if !direct_peer_requirements_validated {
            self.validate_direct_announcement_before_payload_apply(their_announcements, context)
                .await?;
        }
        let context = AnnouncedPeerContext {
            direct_peer_requirements_validated: true,
            ..context
        };
        for (addr, ann) in their_announcements {
            self.apply_announced_peer(addr.id, addr, ann, context)
                .await?;
        }
        Ok(())
    }

    pub(crate) async fn refresh_gossip_path_rtt(
        &self,
        remote: EndpointId,
        ceiling_rtt_ms: Option<u32>,
    ) {
        let conn = self.state.lock().await.connections.get(&remote).cloned();
        let Some(conn) = conn else {
            return;
        };
        self.refresh_gossip_path_rtt_for_connection(remote, &conn, ceiling_rtt_ms)
            .await;
    }

    pub(crate) async fn refresh_gossip_path_rtt_for_connection(
        &self,
        remote: EndpointId,
        conn: &Connection,
        ceiling_rtt_ms: Option<u32>,
    ) {
        let capture_source = if ceiling_rtt_ms.is_some() {
            "gossip_round_trip_path"
        } else {
            "inbound_gossip_path"
        };
        let Some(observation) = self.capture_selected_connection_path(remote, conn, capture_source)
        else {
            return;
        };
        if let Some(path_rtt_ms) = observation.rtt_ms {
            if ceiling_rtt_ms.is_some_and(|ceiling| path_rtt_ms >= ceiling) {
                self.update_peer_selected_path(remote, observation).await;
                return;
            }
            super::emit_mesh_info(format!(
                "📡 Peer {} RTT: {}ms ({}){}",
                remote.fmt_short(),
                path_rtt_ms,
                observation.path_type,
                if ceiling_rtt_ms.is_some() {
                    " [path info]"
                } else {
                    ""
                }
            ));
        }
        self.update_peer_selected_path(remote, observation).await;
    }

    pub(crate) async fn maybe_connect_discovered_peer(
        &self,
        my_role: &super::NodeRole,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        known_peer_check_uses_connections: bool,
        log_discovery_failure_as_warning: bool,
    ) {
        let peer_id = addr.id;
        if self.should_skip_discovered_peer(my_role, peer_id, ann)
            || self
                .discovered_peer_already_known(peer_id, known_peer_check_uses_connections)
                .await
            || Self::discovered_peer_is_filtered(peer_id, ann)
        {
            return;
        }
        if let Err(error) = Box::pin(self.connect_to_peer(addr)).await {
            if log_discovery_failure_as_warning {
                tracing::warn!("Failed to discover peer: {error}");
            } else {
                tracing::debug!(
                    "Could not connect to discovered peer {}: {error}",
                    peer_id.fmt_short()
                );
            }
        }
    }

    pub(crate) async fn connect_discovered_peers(
        &self,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        known_peer_check_uses_connections: bool,
        log_discovery_failure_as_warning: bool,
    ) {
        let my_role = self.role.lock().await.clone();
        for (addr, ann) in their_announcements {
            self.maybe_connect_discovered_peer(
                &my_role,
                addr.clone(),
                ann,
                known_peer_check_uses_connections,
                log_discovery_failure_as_warning,
            )
            .await;
        }
    }

    pub(crate) fn spawn_discovered_peer_connects(
        &self,
        their_announcements: Vec<(EndpointAddr, PeerAnnouncement)>,
        known_peer_check_uses_connections: bool,
        log_discovery_failure_as_warning: bool,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            node.connect_discovered_peers(
                &their_announcements,
                known_peer_check_uses_connections,
                log_discovery_failure_as_warning,
            )
            .await;
        });
    }

    /// Returns `true` if the announcement would be rejected by the same
    /// gates that filter ingest. Skipping the dial here avoids spending
    /// 30s per host walking through unreachable ghost addresses
    /// sequentially in the gossip exchange dial loop — the wedge that
    /// caused `--auto` startup to hang.
    pub(crate) fn discovered_peer_is_filtered(peer_id: EndpointId, ann: &PeerAnnouncement) -> bool {
        if !version_allowed_for_rebroadcast(ann.version.as_deref())
            || peer_is_idle_transitive_client(ann)
        {
            tracing::debug!(
                "Skipping discovered peer {} (filtered: version={:?} role={:?})",
                peer_id.fmt_short(),
                ann.version,
                ann.role
            );
            return true;
        }
        false
    }

    pub(crate) fn should_skip_discovered_peer(
        &self,
        my_role: &super::NodeRole,
        peer_id: EndpointId,
        ann: &PeerAnnouncement,
    ) -> bool {
        peer_id == self.endpoint.id()
            || (matches!(my_role, super::NodeRole::Client)
                && matches!(ann.role, super::NodeRole::Client))
    }

    pub(crate) async fn discovered_peer_already_known(
        &self,
        peer_id: EndpointId,
        use_connections: bool,
    ) -> bool {
        let mut state = self.state.lock().await;
        if state.pending_connection_is_active(peer_id) {
            return true;
        }
        if use_connections {
            state.connections.contains_key(&peer_id)
        } else {
            state.peers.contains_key(&peer_id)
        }
    }

    pub(crate) fn peer_hardware_changed(old_peer: &PeerInfo, updated_peer: &PeerInfo) -> bool {
        old_peer.gpu_name != updated_peer.gpu_name
            || old_peer.hostname != updated_peer.hostname
            || old_peer.is_soc != updated_peer.is_soc
            || old_peer.gpu_vram != updated_peer.gpu_vram
            || old_peer.gpu_reserved_bytes != updated_peer.gpu_reserved_bytes
            || old_peer.gpu_mem_bandwidth_gbps != updated_peer.gpu_mem_bandwidth_gbps
            || old_peer.gpu_compute_tflops_fp32 != updated_peer.gpu_compute_tflops_fp32
            || old_peer.gpu_compute_tflops_fp16 != updated_peer.gpu_compute_tflops_fp16
    }

    pub(crate) fn update_existing_direct_peer(
        existing: &mut PeerInfo,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
        now: std::time::Instant,
    ) -> (PeerInfo, bool, bool, bool) {
        let old_peer = existing.clone();
        let role_changed = existing.role != ann.role;
        let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
        let serving_changed = existing.serving_models != ann.serving_models
            || existing.hosted_models != ann_hosted_models
            || existing.hosted_models_known != ann.hosted_models.is_some();
        existing.admitted = true;
        existing.mesh_id = ann.mesh_id.clone();
        existing.mesh_policy_hash = ann.mesh_policy_hash.clone();
        existing.genesis_policy = ann.genesis_policy.clone();
        if role_changed {
            tracing::info!(
                "Peer {} role updated: {:?} → {:?}",
                existing.id.fmt_short(),
                existing.role,
                ann.role
            );
            existing.role = ann.role.clone();
        }
        if !addr.addrs.is_empty() {
            existing.addr = addr;
        }
        existing.models = ann.models.clone();
        merge_first_joined_mesh_ts(&mut existing.first_joined_mesh_ts, ann.first_joined_mesh_ts);
        existing.vram_bytes = ann.vram_bytes;
        if ann.model_source.is_some() {
            existing.model_source = ann.model_source.clone();
        }
        existing.serving_models = ann.serving_models.clone();
        existing.hosted_models = ann_hosted_models;
        existing.hosted_models_known = ann.hosted_models.is_some();
        existing.available_models.clear();
        existing
            .available_models
            .extend(ann.available_models.clone());
        existing.requested_models = ann.requested_models.clone();
        existing.explicit_model_interests = ann.explicit_model_interests.clone();
        existing.last_seen = now;
        existing.owner_attestation = ann.owner_attestation.clone();
        existing.owner_summary = owner_summary;
        existing.served_model_descriptors = ann.served_model_descriptors.clone();
        existing.served_model_runtime = ann.served_model_runtime.clone();
        existing.artifact_transfer_supported = ann.artifact_transfer_supported;
        existing.stage_protocol_generation_supported = ann.stage_protocol_generation_supported;
        existing.stage_status_list_supported = ann.stage_status_list_supported;
        existing.local_gguf_content_id_supported = ann.local_gguf_content_id_supported;
        existing.advertised_model_throughput = ann.advertised_model_throughput.clone();
        cache_affinity_gossip::merge_advertisement(
            &mut existing.cache_affinity,
            ann.cache_affinity.as_ref(),
            true,
        );
        existing.inference_admission_state = ann.inference_admission_state;
        if ann.version.is_some() {
            existing.version = ann.version.clone();
        }
        existing.gpu_name = ann.gpu_name.clone();
        existing.hostname = ann.hostname.clone();
        existing.is_soc = ann.is_soc;
        existing.gpu_vram = ann.gpu_vram.clone();
        existing.gpu_reserved_bytes = ann.gpu_reserved_bytes.clone();
        existing.memory = ann.memory;
        existing.gpu_mem_bandwidth_gbps = ann.gpu_mem_bandwidth_gbps.clone();
        existing.gpu_compute_tflops_fp32 = ann.gpu_compute_tflops_fp32.clone();
        existing.gpu_compute_tflops_fp16 = ann.gpu_compute_tflops_fp16.clone();
        if ann.experts_summary.is_some() {
            existing.experts_summary = ann.experts_summary.clone();
        }
        existing.release_attestation_summary = crate::verify_release_attestation(
            ann.release_attestation.as_ref(),
            &crate::ReleaseSignerTrustStore::default(),
        );
        let updated_peer = existing.clone();
        let changed = peer_meaningfully_changed(&old_peer, &updated_peer)
            || Self::peer_hardware_changed(&old_peer, &updated_peer);
        (updated_peer, changed, role_changed, serving_changed)
    }

    pub(crate) async fn remove_disallowed_peer(&self, id: EndpointId) {
        let mut state = self.state.lock().await;
        if state.peers.remove(&id).is_some() {
            let admitted_count = state
                .peers
                .values()
                .filter(|peer| peer.is_admitted())
                .count();
            let _ = self.peer_change_tx.send(admitted_count);
        }
    }

    pub(crate) async fn direct_peer_owner_summary(
        &self,
        id: EndpointId,
        ann: &PeerAnnouncement,
    ) -> OwnershipSummary {
        let trust_store = self.trust_store.lock().await.clone();
        verify_node_ownership(
            ann.owner_attestation.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        )
    }

    pub(crate) async fn reject_direct_peer_for_policy(
        &self,
        id: EndpointId,
        owner_summary: &OwnershipSummary,
    ) -> bool {
        if policy_accepts_peer(self.trust_policy, owner_summary) {
            return false;
        }

        let mut state = self.state.lock().await;
        let last_status = state.policy_rejected_peers.get(&id).cloned();
        let newly_rejected = last_status.as_ref() != Some(&owner_summary.status);
        if newly_rejected {
            tracing::warn!(
                "Rejecting peer {} due to owner policy: {:?}",
                id.fmt_short(),
                owner_summary.status
            );
            state
                .policy_rejected_peers
                .insert(id, owner_summary.status.clone());
        }
        if state.peers.remove(&id).is_some() {
            let admitted_count = state
                .peers
                .values()
                .filter(|peer| peer.is_admitted())
                .count();
            let _ = self.peer_change_tx.send(admitted_count);
        }
        drop(state);
        if newly_rejected
            && let Some(reason) =
                MeshPolicyRejectionReason::from_ownership_status(&owner_summary.status)
        {
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::GossipPolicyRejected(reason),
                mesh_peer_operational_context(id, self.authenticated_peer_path(id).await),
            );
        }
        true
    }

    pub(crate) async fn publish_direct_peer_update(
        &self,
        updated_peer: PeerInfo,
        changed: bool,
        should_publish_count: bool,
        count: usize,
    ) {
        let capture_event = if should_publish_count {
            "peer_direct_update"
        } else {
            "peer_direct_seen"
        };
        self.capture_peer_observation(capture_event, &updated_peer, "direct", None);
        if should_publish_count {
            let _ = self.peer_change_tx.send(count);
        }
        if changed {
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                Some(&updated_peer),
                String::new(),
            )
            .await;
        }
    }

    pub(crate) async fn upsert_existing_direct_peer(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
        now: std::time::Instant,
    ) -> bool {
        let mut state = self.state.lock().await;
        state.policy_rejected_peers.remove(&id);
        let Some(existing) = state.peers.get_mut(&id) else {
            return false;
        };
        let (updated_peer, changed, role_changed, serving_changed) =
            Self::update_existing_direct_peer(existing, addr, ann, owner_summary, now);
        let count = state
            .peers
            .values()
            .filter(|peer| peer.is_admitted())
            .count();
        let should_publish_count = role_changed || serving_changed;
        drop(state);
        self.publish_direct_peer_update(updated_peer, changed, should_publish_count, count)
            .await;
        true
    }

    pub(crate) async fn insert_new_direct_peer(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
    ) {
        let mut state = self.state.lock().await;
        state.policy_rejected_peers.remove(&id);
        tracing::info!(
            "Peer added: {} role={:?} vram={:.1}GB assigned={:?} catalog={:?} (total: {})",
            id.fmt_short(),
            ann.role,
            ann.vram_bytes as f64 / 1e9,
            ann.serving_models.first(),
            ann.available_models,
            state.peers.len() + 1
        );
        let mut peer = PeerInfo::from_announcement(id, addr, ann, owner_summary);
        peer.admitted = true;
        state.peers.insert(id, peer.clone());
        let count = state
            .peers
            .values()
            .filter(|peer| peer.is_admitted())
            .count();
        drop(state);
        self.capture_peer_observation("peer_direct_add", &peer, "direct", None);
        record_mesh_operational_event_with_context(
            MeshOperationalEvent::GossipDirectPeerPromoted,
            mesh_peer_operational_context(id, self.authenticated_peer_path(id).await)
                .numeric_summary("direct_peers", count as u64),
        );
        let _ = self.peer_change_tx.send(count);
        self.emit_plugin_mesh_event(
            crate::plugin::proto::mesh_event::Kind::PeerUp,
            Some(&peer),
            String::new(),
        )
        .await;
    }

    pub(crate) async fn collect_rebroadcast_announcements(
        &self,
        stale_cutoff: std::time::Instant,
    ) -> RebroadcastAnnouncements {
        let mut filtered_old_version = 0;
        let announcements = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .filter(|peer| {
                    peer.last_seen >= stale_cutoff || peer.last_mentioned >= stale_cutoff
                })
                .filter(|peer| {
                    let allowed = version_allowed_for_rebroadcast(peer.version.as_deref());
                    if !allowed {
                        filtered_old_version += 1;
                    }
                    allowed
                })
                .map(Self::announcement_from_peer)
                .collect()
        };
        RebroadcastAnnouncements {
            announcements,
            filtered_old_version,
        }
    }

    #[expect(
        clippy::cognitive_complexity,
        reason = "local gossip snapshots intentionally gather many independent advertised fields in one atomic view"
    )]
    pub(crate) async fn snapshot_local_announcement_data(&self) -> LocalAnnouncementData {
        let owner_summary = self.owner_summary.lock().await.clone();
        let plugin_models = self.plugin_inference_models().await;
        let mut models = self.models.lock().await.clone();
        append_external_inference_models(&mut models, &plugin_models);
        let mut serving_models = self.serving_models.lock().await.clone();
        append_external_inference_models(&mut serving_models, &plugin_models);
        let mut hosted_models = self.hosted_models.lock().await.clone();
        append_external_inference_models(&mut hosted_models, &plugin_models);
        let activity_advertisement = self
            .activity_policy_guard
            .advertisement_decision(self.public_mesh);
        if activity_advertisement.withdraw_model_availability {
            serving_models.clear();
            hosted_models.clear();
        }
        let advertised_model_throughput = self
            .routing_metrics
            .advertisable_model_throughput(&hosted_models);
        let now_unix_ms = current_time_unix_ms();
        let cache_affinity = cache_affinity_gossip::local_advertisement(
            &self.cache_affinity_inventory,
            self.endpoint.id().as_bytes(),
            now_unix_ms,
        );
        let release_attestation = self.release_attestation.lock().await.clone();
        let (mesh_id, mesh_policy_hash, signed_genesis_policy) =
            if let Some(state) = self.requirement_mesh_state.lock().await.clone() {
                (
                    Some(state.mesh_id),
                    Some(state.policy_hash),
                    state.signed_policy,
                )
            } else {
                (
                    self.mesh_id.lock().await.clone(),
                    self.mesh_policy_hash.lock().await.clone(),
                    self.signed_genesis_policy.lock().await.clone(),
                )
            };
        let direct_admission_proof = match (mesh_id.as_deref(), mesh_policy_hash.as_deref()) {
            (Some(mesh_id), Some(policy_hash)) => self.build_self_direct_admission_proof(
                mesh_id,
                policy_hash,
                release_attestation.as_ref(),
            ),
            _ => None,
        };
        LocalAnnouncementData {
            role: self.role.lock().await.clone(),
            first_joined_mesh_ts: *self.first_joined_mesh_ts.lock().await,
            models,
            model_source: self.model_source.lock().await.clone(),
            serving_models,
            hosted_models,
            available_models: self.available_models.lock().await.clone(),
            requested_models: self.requested_models.lock().await.clone(),
            explicit_model_interests: self.explicit_model_interests.lock().await.clone(),
            model_demand: self.get_demand(),
            mesh_id,
            mesh_policy_hash,
            signed_genesis_policy,
            release_attestation,
            direct_admission_proof,
            available_model_metadata: Vec::new(),
            available_model_sizes: HashMap::new(),
            served_model_descriptors: self.served_model_descriptors.lock().await.clone(),
            served_model_runtime: self.model_runtime_descriptors.lock().await.clone(),
            owner_attestation: self.owner_attestation.lock().await.clone(),
            artifact_transfer_supported:
                crate::models::artifact_transfer::artifact_transfer_advertised(&owner_summary),
            advertised_model_throughput,
            cache_affinity: Some(cache_affinity),
            gpu_mem_bandwidth_gbps: Self::format_optional_locked_f32_list(
                &self.gpu_mem_bandwidth_gbps,
            )
            .await,
            gpu_compute_tflops_fp32: Self::format_optional_locked_f32_list(
                &self.gpu_compute_tflops_fp32,
            )
            .await,
            gpu_compute_tflops_fp16: Self::format_optional_locked_f32_list(
                &self.gpu_compute_tflops_fp16,
            )
            .await,
            inference_admission_state: activity_advertisement.admission_state,
        }
    }

    pub(crate) async fn plugin_inference_models(&self) -> Vec<String> {
        let plugin_manager = self.plugin_manager.lock().await.clone();
        let Some(plugin_manager) = plugin_manager else {
            return Vec::new();
        };
        plugin_manager
            .inference_models()
            .await
            .unwrap_or_else(|error| {
                tracing::debug!(%error, "failed to collect plugin inference models for gossip");
                Vec::new()
            })
    }

    pub(crate) fn announcement_from_peer(peer: &PeerInfo) -> PeerAnnouncement {
        let latency = peer.display_latency();
        PeerAnnouncement {
            addr: peer.addr.clone(),
            role: peer.role.clone(),
            first_joined_mesh_ts: peer.first_joined_mesh_ts,
            models: peer.models.clone(),
            vram_bytes: peer.vram_bytes,
            model_source: peer.model_source.clone(),
            serving_models: peer.serving_models.clone(),
            hosted_models: peer.hosted_models_known.then(|| peer.hosted_models.clone()),
            available_models: peer.available_models.clone(),
            requested_models: peer.requested_models.clone(),
            explicit_model_interests: peer.explicit_model_interests.clone(),
            version: peer.version.clone(),
            model_demand: HashMap::new(),
            mesh_id: peer.mesh_id.clone(),
            mesh_policy_hash: peer.mesh_policy_hash.clone(),
            gpu_name: peer.gpu_name.clone(),
            hostname: peer.hostname.clone(),
            is_soc: peer.is_soc,
            gpu_vram: peer.gpu_vram.clone(),
            gpu_reserved_bytes: peer.gpu_reserved_bytes.clone(),
            memory: peer.memory,
            gpu_mem_bandwidth_gbps: peer.gpu_mem_bandwidth_gbps.clone(),
            gpu_compute_tflops_fp32: peer.gpu_compute_tflops_fp32.clone(),
            gpu_compute_tflops_fp16: peer.gpu_compute_tflops_fp16.clone(),
            available_model_metadata: peer.available_model_metadata.clone(),
            experts_summary: peer.experts_summary.clone(),
            available_model_sizes: peer.available_model_sizes.clone(),
            served_model_descriptors: peer.served_model_descriptors.clone(),
            served_model_runtime: peer.served_model_runtime.clone(),
            owner_attestation: peer.owner_attestation.clone(),
            genesis_policy: peer.genesis_policy.clone(),
            release_attestation: None,
            direct_admission_proof: None,
            artifact_transfer_supported: peer.artifact_transfer_supported,
            stage_protocol_generation_supported: peer.stage_protocol_generation_supported,
            stage_status_list_supported: peer.stage_status_list_supported,
            local_gguf_content_id_supported: peer.local_gguf_content_id_supported,
            advertised_model_throughput: peer.advertised_model_throughput.clone(),
            cache_affinity: peer.cache_affinity.clone(),
            latency_ms: latency.latency_ms,
            latency_source: Some(match latency.source {
                DisplayLatencySource::Direct => crate::proto::node::LatencySource::Direct,
                DisplayLatencySource::Estimated => crate::proto::node::LatencySource::Estimated,
                DisplayLatencySource::Unknown => crate::proto::node::LatencySource::Unknown,
            }),
            latency_age_ms: Some(latency.age_ms),
            latency_observer_id: latency.observer_id,
            inference_admission_state: peer.inference_admission_state,
        }
    }

    pub(crate) async fn format_optional_locked_f32_list(
        values: &tokio::sync::Mutex<Option<Vec<f64>>>,
    ) -> Option<String> {
        values.lock().await.as_ref().map(|values| {
            values
                .iter()
                .map(|f| format!("{:.2}", f))
                .collect::<Vec<_>>()
                .join(",")
        })
    }

    pub(crate) fn build_local_announcement(&self, data: LocalAnnouncementData) -> PeerAnnouncement {
        PeerAnnouncement {
            addr: self.endpoint_addr_for_advertisement(),
            role: data.role,
            first_joined_mesh_ts: data.first_joined_mesh_ts,
            models: data.models,
            vram_bytes: self.vram_bytes,
            model_source: data.model_source,
            serving_models: data.serving_models,
            hosted_models: Some(data.hosted_models),
            available_models: data.available_models,
            requested_models: data.requested_models,
            explicit_model_interests: data.explicit_model_interests,
            version: Some(crate::VERSION.to_string()),
            model_demand: data.model_demand,
            mesh_id: data.mesh_id,
            mesh_policy_hash: data.mesh_policy_hash,
            gpu_name: self.enumerate_host.then(|| self.gpu_name.clone()).flatten(),
            hostname: self.enumerate_host.then(|| self.hostname.clone()).flatten(),
            is_soc: self.is_soc,
            gpu_vram: self.enumerate_host.then(|| self.gpu_vram.clone()).flatten(),
            gpu_reserved_bytes: self
                .enumerate_host
                .then(|| self.gpu_reserved_bytes.clone())
                .flatten(),
            memory: self.enumerate_host.then_some(self.advertised_memory),
            gpu_mem_bandwidth_gbps: data.gpu_mem_bandwidth_gbps,
            gpu_compute_tflops_fp32: data.gpu_compute_tflops_fp32,
            gpu_compute_tflops_fp16: data.gpu_compute_tflops_fp16,
            available_model_metadata: data.available_model_metadata,
            experts_summary: None,
            available_model_sizes: data.available_model_sizes,
            served_model_descriptors: data.served_model_descriptors,
            served_model_runtime: data.served_model_runtime,
            owner_attestation: data.owner_attestation,
            genesis_policy: data.signed_genesis_policy,
            release_attestation: data.release_attestation,
            direct_admission_proof: data.direct_admission_proof,
            artifact_transfer_supported: data.artifact_transfer_supported,
            stage_protocol_generation_supported: true,
            stage_status_list_supported: true,
            local_gguf_content_id_supported: true,
            advertised_model_throughput: data.advertised_model_throughput,
            cache_affinity: data.cache_affinity,
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
            inference_admission_state: data.inference_admission_state,
        }
    }

    /// Open a gossip stream on an existing connection to exchange peer info.
    pub(super) async fn initiate_gossip(&self, conn: Connection, remote: EndpointId) -> Result<()> {
        // Timeout only the gossip round-trip. A misbehaving peer may accept the
        // QUIC connection and even the bi-stream but never send a gossip response,
        // blocking the join path indefinitely and preventing fallback to other
        // candidates.
        match tokio::time::timeout(
            PEER_CONNECT_AND_GOSSIP_TIMEOUT,
            self.gossip_round_trip(&conn, remote),
        )
        .await
        {
            Ok(Ok((their_announcements, rtt_ms))) => {
                self.apply_gossip_announcements(remote, rtt_ms, &their_announcements, true)
                    .await?;
                if !self.state.lock().await.connections.contains_key(&remote) {
                    self.refresh_gossip_path_rtt_for_connection(remote, &conn, Some(rtt_ms))
                        .await;
                }
                Ok(())
            }
            Ok(Err(e)) => Err(e),
            Err(_) => anyhow::bail!(
                "gossip exchange with {} timed out ({}s)",
                remote.fmt_short(),
                PEER_CONNECT_AND_GOSSIP_TIMEOUT.as_secs()
            ),
        }
    }

    pub(crate) async fn join_first_responsive_candidate(
        &self,
        join_attempts: &[(String, Option<String>)],
    ) -> Result<Option<(String, Option<String>)>> {
        let candidates = self.collect_join_probe_candidates(join_attempts).await;
        if candidates.len() <= 1 {
            tracing::debug!(
                valid_candidates = candidates.len(),
                "auto-join probe skipped"
            );
            return Ok(None);
        }

        emit_join_probe_race_started(candidates.len());
        match self.race_join_probe_candidates(candidates).await {
            Some(success) => self.commit_join_probe_success(success).await.map(Some),
            None => Ok(None),
        }
    }

    pub(crate) async fn collect_join_probe_candidates(
        &self,
        join_attempts: &[(String, Option<String>)],
    ) -> Vec<JoinProbeCandidate> {
        if join_attempts.len() <= 1 {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        let mut invalid = 0usize;
        for (token, mesh_name) in join_attempts.iter().take(CLIENT_AUTO_JOIN_PROBE_LIMIT) {
            match self
                .prepare_join_probe_candidate(token, mesh_name.clone())
                .await
            {
                Ok(Some(candidate)) => candidates.push(candidate),
                Ok(None) => {}
                Err(error) => {
                    invalid += 1;
                    tracing::debug!("Skipping invalid auto-join candidate: {error:#}");
                }
            }
        }
        tracing::debug!(
            valid_candidates = candidates.len(),
            invalid_candidates = invalid,
            "collected auto-join probe candidates"
        );
        candidates
    }

    pub(crate) async fn race_join_probe_candidates(
        &self,
        candidates: Vec<JoinProbeCandidate>,
    ) -> Option<JoinProbeSuccess> {
        let mut probes = tokio::task::JoinSet::new();
        for candidate in candidates {
            let node = self.clone();
            probes.spawn(async move { node.probe_join_candidate(candidate).await });
        }

        let mut last_error = None;
        while let Some(result) = probes.join_next().await {
            match result {
                Ok(Ok(success)) => {
                    probes.abort_all();
                    return Some(success);
                }
                Ok(Err(error)) => {
                    tracing::debug!("auto-join candidate probe failed: {error:#}");
                    last_error = Some(error);
                }
                Err(error) => {
                    tracing::debug!("auto-join candidate probe task failed: {error:#}");
                }
            }
        }

        emit_join_probe_fallback(last_error.as_ref());
        None
    }

    pub(crate) async fn prepare_join_probe_candidate(
        &self,
        token: &str,
        mesh_name: Option<String>,
    ) -> Result<Option<JoinProbeCandidate>> {
        let addr = match parse_invite_token(token)
            .map_err(|reason| anyhow::anyhow!("join rejected: {}", reason.code()))?
        {
            InviteTokenMaterial::Legacy(addr) => addr,
            // Requirement-aware bootstrap tokens may require installing the
            // signed policy before gossip. Keep those on the established
            // serial join path rather than probing them out-of-band.
            InviteTokenMaterial::Signed(_) => return Ok(None),
        };

        if addr.id == self.endpoint.id() {
            return Ok(None);
        }

        let mut state = self.state.lock().await;
        if state.connections.contains_key(&addr.id) || state.pending_connection_is_active(addr.id) {
            return Ok(None);
        }
        if state
            .dead_peers
            .get(&addr.id)
            .is_some_and(|t| t.elapsed() < DEAD_PEER_TTL)
        {
            return Ok(None);
        }
        drop(state);

        Ok(Some(JoinProbeCandidate {
            token: token.to_string(),
            mesh_name,
            addr,
        }))
    }

    pub(crate) async fn probe_join_candidate(
        &self,
        candidate: JoinProbeCandidate,
    ) -> Result<JoinProbeSuccess> {
        let peer_id = candidate.addr.id;
        let started = std::time::Instant::now();
        let result = tokio::time::timeout(CLIENT_AUTO_JOIN_PROBE_TIMEOUT, async {
            let conn = connect_mesh(&self.endpoint, candidate.addr.clone()).await?;
            let (announcements, rtt_ms) = self.gossip_round_trip(&conn, peer_id).await?;
            Ok::<_, anyhow::Error>((conn, announcements, rtt_ms))
        })
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "candidate {} timed out after {}s",
                peer_id.fmt_short(),
                CLIENT_AUTO_JOIN_PROBE_TIMEOUT.as_secs()
            )
        })??;

        Ok(JoinProbeSuccess {
            candidate,
            conn: result.0,
            announcements: result.1,
            rtt_ms: result.2,
            elapsed: started.elapsed(),
        })
    }

    pub(super) async fn commit_join_probe_success(
        &self,
        success: JoinProbeSuccess,
    ) -> Result<(String, Option<String>)> {
        let JoinProbeSuccess {
            candidate,
            conn,
            announcements,
            rtt_ms,
            elapsed,
        } = success;
        let peer_id = candidate.addr.id;

        {
            let mut state = self.state.lock().await;
            state.dead_peers.remove(&peer_id);
            state.connections.insert(peer_id, conn.clone());
        }
        let node_for_dispatch = self.clone();
        let conn_for_dispatch = conn.clone();
        tokio::spawn(async move {
            node_for_dispatch
                .dispatch_streams(conn_for_dispatch, peer_id)
                .await;
        });

        if let Err(error) = self
            .apply_gossip_announcements(peer_id, rtt_ms, &announcements, false)
            .await
        {
            // Drop the tracked entry AND close the QUIC connection. The
            // dispatcher task above holds its own `conn` clone, so removing the
            // map entry alone would leave a live, keep-alive'd connection and a
            // running dispatcher for a peer nobody tracks (and one whose
            // close-recovery path could even reconnect it). Closing here makes
            // the dispatcher's `accept_*` calls error so it unwinds cleanly.
            self.state.lock().await.connections.remove(&peer_id);
            conn.close(0u32.into(), b"join announcement-apply failed");
            record_mesh_operational_event(MeshOperationalEvent::AutoJoinFailed);
            return Err(error);
        }

        // Match `connect_to_peer`: the probe gossip RTT above likely reflects
        // relay latency, so refresh the selected-path/RTT after holepunch.
        self.schedule_selected_path_recheck(peer_id);
        self.spawn_discovered_peer_connects(announcements, true, false);

        tracing::info!(
            peer = %peer_id.fmt_short(),
            elapsed_ms = elapsed_ms_u64(elapsed),
            rtt_ms,
            "Fast auto-join probe selected bootstrap candidate"
        );
        emit_mesh_info(format!(
            "Fast auto-join selected peer {} in {}ms",
            peer_id.fmt_short(),
            elapsed_ms_u64(elapsed)
        ));
        record_mesh_operational_event(MeshOperationalEvent::AutoJoinSucceeded);

        Ok((candidate.token, candidate.mesh_name))
    }

    pub(super) async fn initiate_gossip_inner(
        &self,
        conn: Connection,
        remote: EndpointId,
        discover_peers: bool,
    ) -> Result<()> {
        let (their_announcements, rtt_ms) = self.gossip_round_trip(&conn, remote).await?;
        self.apply_gossip_announcements(remote, rtt_ms, &their_announcements, discover_peers)
            .await?;
        if !self.state.lock().await.connections.contains_key(&remote) {
            self.refresh_gossip_path_rtt_for_connection(remote, &conn, Some(rtt_ms))
                .await;
        }
        Ok(())
    }

    pub(crate) async fn gossip_round_trip(
        &self,
        conn: &Connection,
        remote: EndpointId,
    ) -> Result<(Vec<(EndpointAddr, PeerAnnouncement)>, u32)> {
        let protocol = connection_protocol(conn);
        let t0 = std::time::Instant::now();
        let (mut send, mut recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_GOSSIP]).await?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let buf = read_len_prefixed(&mut recv).await?;
        let rtt_ms = t0.elapsed().as_millis() as u32;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;

        let _ = recv.read_to_end(0).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Ok((their_announcements, rtt_ms))
    }

    pub(crate) async fn apply_gossip_announcements(
        &self,
        remote: EndpointId,
        rtt_ms: u32,
        their_announcements: &[(EndpointAddr, PeerAnnouncement)],
        discover_peers: bool,
    ) -> Result<()> {
        self.apply_announced_peers(
            remote,
            their_announcements,
            Some(rtt_ms),
            Some(NODE_PROTOCOL_GENERATION),
            false,
        )
        .await?;

        // Also check the connection's actual path info — the gossip round-trip
        // time above may reflect relay latency even if a direct path is now active.
        self.refresh_gossip_path_rtt(remote, Some(rtt_ms)).await;

        if discover_peers {
            self.connect_discovered_peers(their_announcements, true, false)
                .await;
        }

        Ok(())
    }

    pub(super) async fn handle_gossip_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        tracing::info!("Inbound gossip from {}", remote.fmt_short());

        let buf = read_len_prefixed(&mut recv).await?;
        let their_announcements = decode_gossip_payload(protocol, remote, &buf)?;
        let negotiated_protocol_generation = match protocol {
            ControlProtocol::ProtoV1 => Some(NODE_PROTOCOL_GENERATION),
        };
        let context = AnnouncedPeerContext::direct(remote, negotiated_protocol_generation);
        self.validate_and_capture_inbound_gossip(protocol, &their_announcements, context)
            .await?;

        let our_announcements = self.collect_announcements().await;
        write_gossip_payload(&mut send, protocol, &our_announcements, self.endpoint.id()).await?;
        send.finish()?;

        let _ = recv.read_to_end(0).await;

        self.apply_announced_peers(
            remote,
            &their_announcements,
            None,
            negotiated_protocol_generation,
            true,
        )
        .await?;
        self.refresh_gossip_path_rtt(remote, None).await;

        self.connect_discovered_peers(&their_announcements, false, true)
            .await;

        Ok(())
    }
    pub(super) async fn remove_peer(&self, id: EndpointId, reason: MeshPeerRemovalReason) {
        let mut state = self.state.lock().await;
        // Always clear any rejection-tracking entry so the map stays bounded.
        state.policy_rejected_peers.remove(&id);
        let had_connection = state.connections.contains_key(&id);
        state.requirement_rejected_peers.remove(&id);
        if let Some(peer) = state.peers.remove(&id) {
            let last_seen_age_ms = super::elapsed_ms_u64(peer.last_seen.elapsed());
            let last_mentioned_age_ms = super::elapsed_ms_u64(peer.last_mentioned.elapsed());
            let bridge_id = peer
                .propagated_latency
                .as_ref()
                .and_then(|latency| latency.observer_id);
            tracing::info!(
                "Peer removed: {} (total: {})",
                id.fmt_short(),
                state.peers.len()
            );
            let count = state
                .peers
                .values()
                .filter(|peer| peer.is_admitted())
                .count();
            drop(state);
            self.capture_peer_lifecycle_event(PeerLifecycleCaptureEvent {
                event: "peer_removed",
                peer: id,
                reason: reason.reason_code(),
                reporter: None,
                last_seen_age_ms: Some(last_seen_age_ms),
                last_mentioned_age_ms: Some(last_mentioned_age_ms),
                had_connection: Some(had_connection),
                bridge_id,
            });
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::GossipPeerRemoved(reason),
                mesh_peer_operational_context(id, peer.selected_path)
                    .numeric_summary("direct_peers", count as u64),
            );
            let _ = self.peer_change_tx.send(count);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerDown,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    #[cfg(test)]
    pub(super) async fn add_peer(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        negotiated_protocol_generation: Option<u32>,
    ) {
        if let Err(reason) = self
            .validate_direct_peer_requirements(id, ann, negotiated_protocol_generation)
            .await
        {
            self.record_mesh_requirement_rejection(
                super::requirements::MeshRequirementRejectionSource::Gossip,
                Some(id),
                reason.clone(),
            )
            .await;
            tracing::warn!(
                "Rejecting peer {} before promotion: {}",
                id.fmt_short(),
                reason.code()
            );
            let mut state = self.state.lock().await;
            state.requirement_rejected_peers.insert(id);
            if state.peers.remove(&id).is_some() {
                let admitted_count = state
                    .peers
                    .values()
                    .filter(|peer| peer.is_admitted())
                    .count();
                let _ = self.peer_change_tx.send(admitted_count);
            }
            return;
        }
        self.add_peer_after_direct_requirements_validated(
            id,
            addr,
            ann,
            negotiated_protocol_generation,
        )
        .await;
    }

    pub(crate) async fn add_peer_after_direct_requirements_validated(
        &self,
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        _negotiated_protocol_generation: Option<u32>,
    ) -> bool {
        // Reject ingest from peers below the supported version floor. They
        // are not added to local state, do not appear in /api/status, and
        // are not re-broadcast. A peer that updates and re-announces will
        // be accepted on the next exchange.
        if !version_allowed_for_rebroadcast(ann.version.as_deref()) {
            tracing::debug!(
                "Refusing direct peer {} below version floor (advertised {:?})",
                id.fmt_short(),
                ann.version
            );
            record_mesh_operational_event_with_context(
                MeshOperationalEvent::GossipIncompatibleVersionRejected,
                mesh_peer_operational_context(id, self.authenticated_peer_path(id).await),
            );
            self.remove_disallowed_peer(id).await;
            return false;
        }
        let owner_summary = self.direct_peer_owner_summary(id, ann).await;
        if self.reject_direct_peer_for_policy(id, &owner_summary).await {
            self.capture_peer_rejected(id, &addr, ann, &owner_summary, "direct", None);
            return false;
        }
        let mut state = self.state.lock().await;
        state.policy_rejected_peers.remove(&id);
        state.requirement_rejected_peers.remove(&id);
        if id == self.endpoint.id() {
            return false;
        }
        let now = std::time::Instant::now();
        // If this peer was previously dead, clear it — add_peer is only called
        // after a successful gossip exchange, which is proof of life.
        let recovered = state.dead_peers.remove(&id).is_some();
        if recovered {
            super::emit_mesh_info(format!(
                "🔄 Peer {} back from the dead (successful gossip)",
                id.fmt_short()
            ));
        }
        let peer_exists = state.peers.contains_key(&id);
        drop(state);
        if peer_exists
            && self
                .upsert_existing_direct_peer(id, addr.clone(), ann, owner_summary.clone(), now)
                .await
        {
            return true;
        }
        self.insert_new_direct_peer(id, addr, ann, owner_summary)
            .await;
        true
    }

    /// Update a peer learned transitively through gossip (not directly connected).
    /// Updates assigned/hosted state so models_being_served() includes their models.
    /// Refreshes `last_mentioned` (not `last_seen`) so the peer survives pruning
    /// and gossip propagation as long as a bridge peer keeps mentioning it, but
    /// PeerDown silencing uses only `last_seen` (direct proof-of-life).
    /// Does NOT trigger peer_change events for new transitive peers
    /// (avoids re-election storms at scale).
    pub(super) async fn update_transitive_peer(
        &self,
        id: EndpointId,
        addr: &EndpointAddr,
        ann: &PeerAnnouncement,
        bridge_id: EndpointId,
    ) {
        // Refuse transitive ingest from peers below the supported version
        // floor. Keeps the local table free of pre-floor gossip filler;
        // /api/status, the UI, and routing all stop seeing them.
        if !version_allowed_for_rebroadcast(ann.version.as_deref()) {
            let mut state = self.state.lock().await;
            if state.peers.remove(&id).is_some() {
                let admitted_count = state
                    .peers
                    .values()
                    .filter(|peer| peer.is_admitted())
                    .count();
                let _ = self.peer_change_tx.send(admitted_count);
            }
            return;
        }
        // Refuse transitive ingest of idle clients — clients that aren't
        // asking for any model, aren't serving anything, and aren't hosting
        // anything. They contribute nothing the mesh can use:
        //   - not routable to (no model to serve)
        //   - not findable (clients-don't-dial-clients by design)
        //   - no demand signal (empty requested_models)
        //   - not relaying for us (no connection — purely transitive)
        // The moment any of those become non-empty, this filter stops firing
        // and the peer is admitted normally. Direct connections (`add_peer`)
        // are never affected — a client that actually contacts us still
        // gets in.
        if peer_is_idle_transitive_client(ann) {
            let mut state = self.state.lock().await;
            if state.peers.remove(&id).is_some() {
                let admitted_count = state
                    .peers
                    .values()
                    .filter(|peer| peer.is_admitted())
                    .count();
                let _ = self.peer_change_tx.send(admitted_count);
            }
            return;
        }
        let trust_store = self.trust_store.lock().await.clone();
        let owner_summary = verify_node_ownership(
            ann.owner_attestation.as_ref(),
            id.as_bytes(),
            &trust_store,
            self.trust_policy,
            current_time_unix_ms(),
        );
        if !policy_accepts_peer(self.trust_policy, &owner_summary) {
            let mut state = self.state.lock().await;
            if state.peers.remove(&id).is_some() {
                let admitted_count = state
                    .peers
                    .values()
                    .filter(|peer| peer.is_admitted())
                    .count();
                let _ = self.peer_change_tx.send(admitted_count);
            }
            drop(state);
            self.capture_peer_rejected(
                id,
                addr,
                ann,
                &owner_summary,
                "transitive",
                Some(bridge_id),
            );
            return;
        }
        let mut state = self.state.lock().await;
        if id == self.endpoint.id() {
            return;
        }
        if state
            .dead_peers
            .get(&id)
            .is_some_and(|t| t.elapsed() < DEAD_PEER_TTL)
        {
            return;
        }
        if let Some(existing) = state.peers.get_mut(&id) {
            let old_peer = existing.clone();
            let serving_changed = apply_transitive_ann(existing, addr, ann, bridge_id);
            existing.owner_summary = owner_summary;
            // Refresh last_mentioned: the bridge peer vouches for this peer
            // being alive (collect_announcements already filters stale peers).
            // We update last_mentioned (not last_seen) so that PeerDown
            // silencing and collect_announcements use only direct proof-of-life,
            // while the prune decision considers both timestamps.
            existing.last_mentioned = std::time::Instant::now();
            let updated_peer = existing.clone();
            let changed = peer_meaningfully_changed(&old_peer, &updated_peer);
            if serving_changed {
                let count = state
                    .peers
                    .values()
                    .filter(|peer| peer.is_admitted())
                    .count();
                drop(state);
                self.capture_peer_observation(
                    "peer_transitive_update",
                    &updated_peer,
                    "transitive",
                    Some(bridge_id),
                );
                let _ = self.peer_change_tx.send(count);
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            } else {
                drop(state);
                self.capture_peer_observation(
                    "peer_transitive_seen",
                    &updated_peer,
                    "transitive",
                    Some(bridge_id),
                );
                if changed {
                    self.emit_plugin_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                        Some(&updated_peer),
                        String::new(),
                    )
                    .await;
                }
            }
        } else {
            // New transitive peer — not directly verified, so set last_seen to
            // epoch (not "now") to avoid incorrectly silencing PeerDown reports.
            // last_mentioned = now keeps the peer alive for the prune window.
            let mut peer = PeerInfo::from_announcement(id, addr.clone(), ann, owner_summary);
            // Capability provenance must be direct. A bridge can report that a
            // peer exists, but it cannot make that peer eligible for strict
            // local-GGUF election on the peer's behalf.
            peer.local_gguf_content_id_supported = false;
            // Mark as never directly seen — only transitively mentioned.
            peer.admitted = false;
            peer.last_seen =
                std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 2);
            state.peers.insert(id, peer.clone());
            drop(state);
            self.capture_peer_observation(
                "peer_transitive_add",
                &peer,
                "transitive",
                Some(bridge_id),
            );
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUp,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    pub(super) async fn collect_announcements(&self) -> Vec<PeerAnnouncement> {
        let stale_cutoff =
            std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS);
        let local = self.snapshot_local_announcement_data().await;
        let RebroadcastAnnouncements {
            mut announcements,
            filtered_old_version,
        } = self.collect_rebroadcast_announcements(stale_cutoff).await;
        if filtered_old_version > 0 {
            tracing::debug!(
                filtered = filtered_old_version,
                "gossip: omitting {} peer(s) below v{}.{}.0 from outbound rebroadcast",
                filtered_old_version,
                MIN_REBROADCAST_VERSION_MAJOR,
                MIN_REBROADCAST_VERSION_MINOR,
            );
        }
        announcements.push(self.build_local_announcement(local));
        announcements
    }
}

#[cfg(test)]
mod tests {
    include!("tests/gossip.rs");
}
