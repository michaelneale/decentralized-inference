use super::*;

pub(crate) fn peer_info_to_mesh_peer(peer: &PeerInfo) -> crate::plugin::proto::MeshPeer {
    crate::plugin::proto::MeshPeer {
        peer_id: endpoint_id_hex(peer.id),
        version: peer.version.clone().unwrap_or_default(),
        capabilities: Vec::new(),
        role: node_role_label(&peer.role),
        vram_bytes: peer.vram_bytes,
        models: peer.models.clone(),
        serving_models: peer.serving_models.clone(),
        available_models: Vec::new(),
        requested_models: peer.requested_models.clone(),
        rtt_ms: peer.current_direct_rtt_ms(),
        model_source: peer.model_source.clone().unwrap_or_default(),
        hosted_models: peer.hosted_models.clone(),
        hosted_models_known: Some(peer.hosted_models_known),
    }
}

pub(crate) fn policy_accepts_peer(policy: TrustPolicy, owner_summary: &OwnershipSummary) -> bool {
    match policy {
        TrustPolicy::Off | TrustPolicy::PreferOwned => true,
        TrustPolicy::RequireOwned | TrustPolicy::Allowlist => {
            owner_summary.status == OwnershipStatus::Verified
        }
    }
}

pub(crate) fn load_or_refresh_owner_attestation(
    owner_keypair: &crate::crypto::OwnerKeypair,
    endpoint_id: EndpointId,
    node_label: Option<String>,
    hostname_hint: Option<String>,
) -> Result<SignedNodeOwnership> {
    // Always sign a fresh attestation on startup when the owner key is available.
    // This ensures that key rotation is always reflected immediately and no stale
    // certificate can persist across restarts.
    let path = default_node_ownership_path()?;
    let ownership = sign_node_ownership(
        owner_keypair,
        endpoint_id.as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        node_label,
        hostname_hint,
    )?;
    save_node_ownership(&path, &ownership)?;
    Ok(ownership)
}

pub(crate) fn model_identity_score(identity: &ServedModelIdentity) -> u8 {
    let kind_score = match identity.source_kind {
        ModelSourceKind::HuggingFace => 4,
        ModelSourceKind::Catalog => 3,
        ModelSourceKind::DirectUrl => 2,
        ModelSourceKind::LocalGguf => 1,
        ModelSourceKind::Unknown => 0,
    };
    let canonical_bonus = if identity.canonical_ref.is_some() {
        2
    } else {
        0
    };
    let revision_bonus = if identity.revision.is_some() { 1 } else { 0 };
    kind_score + canonical_bonus + revision_bonus
}

pub(crate) fn model_descriptor_score(descriptor: &ServedModelDescriptor) -> u8 {
    let identity = &descriptor.identity;
    let capability_bonus = u8::from(descriptor.capabilities.multimodal)
        + u8::from(descriptor.capabilities.audio != crate::models::CapabilityLevel::None)
        + u8::from(descriptor.capabilities.vision != crate::models::CapabilityLevel::None)
        + u8::from(descriptor.capabilities.reasoning != crate::models::CapabilityLevel::None)
        + u8::from(descriptor.capabilities.tool_use != crate::models::CapabilityLevel::None);
    let metadata_bonus = u8::from(descriptor.metadata.is_some());
    model_identity_score(identity) + capability_bonus + metadata_bonus
}

pub(crate) fn upsert_mesh_catalog_descriptor(
    descriptors: &mut HashMap<String, ServedModelDescriptor>,
    descriptor: ServedModelDescriptor,
) {
    if descriptor.identity.model_name.is_empty() {
        return;
    }
    let mut keys = vec![descriptor.identity.model_name.clone()];
    if let Some(public_id) = public_model_id_from_identity(&descriptor.identity) {
        keys.push(public_id);
    }
    keys.sort();
    keys.dedup();
    for key in keys {
        match descriptors.get(&key) {
            Some(existing)
                if model_descriptor_score(existing) >= model_descriptor_score(&descriptor) => {}
            _ => {
                descriptors.insert(key, descriptor.clone());
            }
        }
    }
}

/// Merge two demand maps. For each model, take max of last_active and request_count.
/// Role a node plays in the mesh.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum NodeRole {
    /// Provides staged GPU compute for a specific model.
    #[default]
    Worker,
    /// Runs the local serving runtime for a specific model and provides the HTTP API.
    Host { http_port: u16 },
    /// Lite client — no compute, accesses the API via tunnel.
    Client,
}

/// Gossip payload — extends EndpointAddr with role metadata.
/// Internal mesh gossip model. Legacy JSON v0 is adapted at the boundary.
#[derive(Debug, Clone)]
pub struct PeerAnnouncement {
    pub(crate) addr: EndpointAddr,
    pub(crate) role: NodeRole,
    pub(crate) first_joined_mesh_ts: Option<u64>,
    pub(crate) models: Vec<String>,
    pub(crate) vram_bytes: u64,
    pub(crate) model_source: Option<String>,
    pub(crate) serving_models: Vec<String>,
    pub(crate) hosted_models: Option<Vec<String>>,
    /// All GGUF filenames on disk in managed or legacy local storage (for mesh catalog)
    pub(crate) available_models: Vec<String>,
    pub(crate) requested_models: Vec<String>,
    /// Advisory canonical refs this node wants the mesh to consider.
    pub(crate) explicit_model_interests: Vec<String>,
    pub(crate) version: Option<String>,
    pub(crate) model_demand: HashMap<String, ModelDemand>,
    pub(crate) mesh_id: Option<String>,
    pub(crate) mesh_policy_hash: Option<String>,
    pub(crate) gpu_name: Option<String>,
    pub(crate) hostname: Option<String>,
    pub(crate) is_soc: Option<bool>,
    pub(crate) gpu_vram: Option<String>,
    pub(crate) gpu_reserved_bytes: Option<String>,
    pub(crate) gpu_mem_bandwidth_gbps: Option<String>,
    pub(crate) gpu_compute_tflops_fp32: Option<String>,
    pub(crate) gpu_compute_tflops_fp16: Option<String>,
    pub(crate) available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    pub(crate) experts_summary: Option<crate::proto::node::ExpertsSummary>,
    pub(crate) available_model_sizes: HashMap<String, u64>,
    pub(crate) served_model_descriptors: Vec<ServedModelDescriptor>,
    pub(crate) served_model_runtime: Vec<ModelRuntimeDescriptor>,
    pub(crate) owner_attestation: Option<SignedNodeOwnership>,
    pub(crate) genesis_policy: Option<crate::SignedMeshGenesisPolicy>,
    pub(crate) release_attestation: Option<crate::ReleaseBuildAttestation>,
    pub(crate) direct_admission_proof: Option<crate::DirectNodeAdmissionProof>,
    pub(crate) artifact_transfer_supported: bool,
    pub(crate) stage_protocol_generation_supported: bool,
    pub(crate) stage_status_list_supported: bool,
    pub(crate) local_gguf_content_id_supported: bool,
    pub(crate) advertised_model_throughput: Vec<crate::network::metrics::ModelThroughputHint>,
    pub(crate) cache_affinity:
        Option<mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement>,
    pub(crate) latency_ms: Option<u32>,
    pub(crate) latency_source: Option<crate::proto::node::LatencySource>,
    pub(crate) latency_age_ms: Option<u64>,
    pub(crate) latency_observer_id: Option<EndpointId>,
    pub(crate) inference_admission_state: Option<crate::proto::node::InferenceAdmissionState>,
    /// An optional, signed attestation this peer MAY advertise about its own
    /// append-only history. Carried opaquely; never verified by mesh-llm.
    pub(crate) attested_log_head: Option<AttestedLogHead>,
}

/// A peer's latest signed attestation of the head of its append-only log —
/// see `AttestedLogHead` in `node.proto` for the wire shape and the
/// signing-scope note. Carried opaquely: mesh-llm never verifies `signature`
/// itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestedLogHead {
    pub log_id: String,
    pub size: u64,
    pub root: Vec<u8>,
    pub timestamp_unix_ms: u64,
    pub signature: Vec<u8>,
}

/// A single direct RTT measurement (e.g. from gossip exchange).
#[derive(Debug, Clone)]
pub struct DirectLatencyObservation {
    pub rtt_ms: u32,
    pub observed_at: std::time::Instant,
}

/// Latency propagated via transitive gossip (not measured directly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropagatedLatencyObservation {
    pub latency_ms: u32,
    pub age_ms_at_received: u64,
    pub received_at: std::time::Instant,
    pub observer_id: Option<EndpointId>,
}

/// Which source a display latency value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayLatencySource {
    Direct,
    Estimated,
    Unknown,
}

/// Computed display latency for UI/API consumption.
#[derive(Debug, Clone)]
pub struct DisplayLatency {
    pub latency_ms: Option<u32>,
    pub source: DisplayLatencySource,
    pub age_ms: u64,
    pub observer_id: Option<EndpointId>,
}

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: EndpointId,
    pub addr: EndpointAddr,
    pub mesh_id: Option<String>,
    pub mesh_policy_hash: Option<String>,
    pub genesis_policy: Option<crate::SignedMeshGenesisPolicy>,
    pub role: NodeRole,
    pub first_joined_mesh_ts: Option<u64>,
    pub models: Vec<String>,
    pub vram_bytes: u64,
    pub rtt_ms: Option<u32>,
    pub model_source: Option<String>,
    pub admitted: bool,
    /// All models assigned to this peer, even if not yet healthy.
    pub serving_models: Vec<String>,
    /// Models this node is actively routing inference for.
    pub hosted_models: Vec<String>,
    /// True when this peer explicitly advertised `hosted_models`.
    pub hosted_models_known: bool,
    /// All GGUFs on disk
    pub available_models: Vec<String>,
    /// Models this node has requested the mesh to serve
    pub requested_models: Vec<String>,
    /// Advisory canonical refs this peer wants the mesh to consider.
    pub explicit_model_interests: Vec<String>,
    /// Last time we directly communicated with this peer (gossip, heartbeat, tunnel).
    /// Only updated by direct bi-directional gossip exchanges, heartbeat probes,
    /// and inbound connections — never by transitive mentions.
    /// Used by PeerDown silencing to require independent proof-of-life.
    pub last_seen: std::time::Instant,
    /// Last time a bridge peer mentioned this peer in gossip.
    /// Updated on every transitive gossip update. Used together with `last_seen`
    /// for pruning and `collect_announcements`: a peer is included/kept as long
    /// as either timestamp is fresh.
    pub last_mentioned: std::time::Instant,
    /// mesh-llm version (e.g. "0.23.0")
    pub version: Option<String>,
    /// GPU name/model (e.g. "NVIDIA A100", "Apple M4 Max")
    pub gpu_name: Option<String>,
    /// Hostname of the node
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_reserved_bytes: Option<String>,
    pub gpu_mem_bandwidth_gbps: Option<String>,
    pub gpu_compute_tflops_fp32: Option<String>,
    pub gpu_compute_tflops_fp16: Option<String>,
    pub available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    pub experts_summary: Option<crate::proto::node::ExpertsSummary>,
    pub available_model_sizes: HashMap<String, u64>,
    pub served_model_descriptors: Vec<ServedModelDescriptor>,
    pub served_model_runtime: Vec<ModelRuntimeDescriptor>,
    pub owner_attestation: Option<SignedNodeOwnership>,
    pub release_attestation_summary: crate::ReleaseAttestationSummary,
    pub artifact_transfer_supported: bool,
    pub stage_protocol_generation_supported: bool,
    pub stage_status_list_supported: bool,
    pub local_gguf_content_id_supported: bool,
    pub(crate) advertised_model_throughput: Vec<crate::network::metrics::ModelThroughputHint>,
    pub(crate) cache_affinity:
        Option<mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement>,
    /// Most recent direct RTT sample for display purposes (refreshed periodically).
    pub display_rtt: Option<DirectLatencyObservation>,
    /// Last selected path observed on the mesh control connection to this peer.
    pub(crate) selected_path: Option<SelectedPathObservation>,
    /// Latency propagated via transitive gossip.
    pub propagated_latency: Option<PropagatedLatencyObservation>,
    pub owner_summary: OwnershipSummary,
    pub inference_admission_state: Option<crate::proto::node::InferenceAdmissionState>,
}

#[derive(Debug)]
pub struct OwnerRuntimeConfig {
    pub keypair: Option<crate::crypto::OwnerKeypair>,
    pub control_bind: Option<std::net::SocketAddr>,
    pub control_advertise_addr: Option<std::net::SocketAddr>,
    pub node_label: Option<String>,
    pub trust_store: TrustStore,
    pub trust_policy: TrustPolicy,
    pub activity: mesh_llm_config::RuntimeActivityConfig,
    pub drain_timeout_secs: u64,
    pub drain_timeout_max_secs: u64,
    pub public_mesh: bool,
}

pub(crate) struct ControlListenerLifecycle {
    pub(crate) endpoint: Endpoint,
    pub(crate) token: String,
    pub(crate) shutdown_requested: Arc<std::sync::atomic::AtomicBool>,
    pub(crate) shutdown: Arc<tokio::sync::Notify>,
    pub(crate) task: tokio::task::JoinHandle<()>,
}
#[derive(Debug, Clone)]
pub struct MeshCatalogEntry {
    pub model_name: String,
    pub descriptor: Option<ServedModelDescriptor>,
}

impl PeerInfo {
    pub(crate) fn from_announcement(
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
    ) -> Self {
        Self {
            id,
            addr,
            mesh_id: ann.mesh_id.clone(),
            mesh_policy_hash: ann.mesh_policy_hash.clone(),
            genesis_policy: ann.genesis_policy.clone(),
            role: ann.role.clone(),
            first_joined_mesh_ts: ann.first_joined_mesh_ts,
            models: ann.models.clone(),
            vram_bytes: ann.vram_bytes,
            rtt_ms: None,
            model_source: ann.model_source.clone(),
            admitted: false,
            serving_models: ann.serving_models.clone(),
            hosted_models: ann.hosted_models.clone().unwrap_or_default(),
            hosted_models_known: ann.hosted_models.is_some(),
            available_models: ann.available_models.clone(),
            requested_models: ann.requested_models.clone(),
            explicit_model_interests: ann.explicit_model_interests.clone(),
            last_seen: std::time::Instant::now(),
            last_mentioned: std::time::Instant::now(),
            version: ann.version.clone(),
            gpu_name: ann.gpu_name.clone(),
            hostname: ann.hostname.clone(),
            is_soc: ann.is_soc,
            gpu_vram: ann.gpu_vram.clone(),
            gpu_reserved_bytes: ann.gpu_reserved_bytes.clone(),
            gpu_mem_bandwidth_gbps: ann.gpu_mem_bandwidth_gbps.clone(),
            gpu_compute_tflops_fp32: ann.gpu_compute_tflops_fp32.clone(),
            gpu_compute_tflops_fp16: ann.gpu_compute_tflops_fp16.clone(),
            available_model_metadata: ann.available_model_metadata.clone(),
            experts_summary: ann.experts_summary.clone(),
            available_model_sizes: ann.available_model_sizes.clone(),
            served_model_descriptors: ann.served_model_descriptors.clone(),
            served_model_runtime: ann.served_model_runtime.clone(),
            owner_attestation: ann.owner_attestation.clone(),
            release_attestation_summary: crate::verify_release_attestation(
                ann.release_attestation.as_ref(),
                &crate::ReleaseSignerTrustStore::default(),
            ),
            artifact_transfer_supported: ann.artifact_transfer_supported,
            stage_protocol_generation_supported: ann.stage_protocol_generation_supported,
            stage_status_list_supported: ann.stage_status_list_supported,
            local_gguf_content_id_supported: ann.local_gguf_content_id_supported,
            advertised_model_throughput: ann.advertised_model_throughput.clone(),
            cache_affinity: ann.cache_affinity.clone(),
            display_rtt: None,
            selected_path: None,
            propagated_latency: None,
            owner_summary,
            inference_admission_state: ann.inference_admission_state,
        }
    }

    pub fn is_admitted(&self) -> bool {
        self.admitted
    }

    /// Return the most recent direct RTT sample for display, falling back to best-seen RTT.
    pub fn current_direct_rtt_ms(&self) -> Option<u32> {
        self.display_rtt.as_ref().map(|d| d.rtt_ms).or(self.rtt_ms)
    }

    pub(crate) fn split_stage_path_fallback(&self) -> Option<SelectedPathObservation> {
        let observation = self.selected_path?;
        if observation.path_type != "direct" {
            return Some(observation);
        }
        Some(SelectedPathObservation {
            rtt_ms: self.rtt_ms.or(observation.rtt_ms),
            ..observation
        })
    }

    /// Compute display latency from direct sample or propagated data.
    pub fn display_latency(&self) -> DisplayLatency {
        if let Some(ref direct) = self.display_rtt {
            return DisplayLatency {
                latency_ms: Some(direct.rtt_ms),
                source: DisplayLatencySource::Direct,
                age_ms: direct.observed_at.elapsed().as_millis() as u64,
                observer_id: None,
            };
        }
        if let Some(ref propagated) = self.propagated_latency {
            return DisplayLatency {
                latency_ms: Some(propagated.latency_ms),
                source: DisplayLatencySource::Estimated,
                age_ms: propagated.age_ms_at_received
                    + propagated.received_at.elapsed().as_millis() as u64,
                observer_id: propagated.observer_id,
            };
        }
        DisplayLatency {
            latency_ms: self.rtt_ms,
            source: DisplayLatencySource::Unknown,
            age_ms: 0,
            observer_id: None,
        }
    }

    #[cfg(test)]
    pub fn is_assigned_model(&self, model: &str) -> bool {
        self.serving_models.iter().any(|m| m == model)
    }

    pub fn routable_models(&self) -> Vec<String> {
        let raw = if self.hosted_models_known {
            &self.hosted_models
        } else {
            &self.serving_models
        };
        let mut models = raw
            .iter()
            .map(|model| self.public_model_id_for_routable_model(model))
            .collect::<Vec<_>>();
        models.sort();
        models.dedup();
        models
    }

    pub fn routes_model(&self, model: &str) -> bool {
        let raw = if self.hosted_models_known {
            &self.hosted_models
        } else {
            &self.serving_models
        };
        raw.iter().any(|candidate| {
            candidate == model || self.public_model_id_for_routable_model(candidate) == model
        })
    }

    pub fn accepts_http_inference(&self) -> bool {
        matches!(self.role, NodeRole::Host { .. })
    }

    pub fn http_routable_models(&self) -> Vec<String> {
        if self.accepts_http_inference() {
            self.routable_models()
        } else {
            Vec::new()
        }
    }

    pub fn routes_http_model(&self, model: &str) -> bool {
        self.accepts_http_inference() && self.routes_model(model)
    }

    pub(crate) fn public_model_id_for_routable_model(&self, model: &str) -> String {
        self.served_model_descriptors
            .iter()
            .find(|descriptor| descriptor.identity.model_name == model)
            .and_then(|descriptor| public_model_id_from_identity(&descriptor.identity))
            .unwrap_or_else(|| canonical_demand_model_ref(model))
    }

    pub fn advertised_context_length(&self, model: &str) -> Option<u32> {
        self.advertised_context_length_for_runtime_model(model)
            .or_else(|| {
                self.served_model_descriptors
                    .iter()
                    .filter(|descriptor| {
                        let runtime_name = descriptor.identity.model_name.as_str();
                        runtime_name != model
                            && self.public_model_id_for_routable_model(runtime_name) == model
                    })
                    .find_map(|descriptor| {
                        self.advertised_context_length_for_runtime_model(
                            &descriptor.identity.model_name,
                        )
                    })
            })
    }

    pub(crate) fn advertised_context_length_for_runtime_model(&self, model: &str) -> Option<u32> {
        self.served_model_runtime
            .iter()
            .find(|runtime| runtime.model_name == model)
            .and_then(ModelRuntimeDescriptor::advertised_context_length)
    }
}

pub(crate) fn public_model_id_from_identity(identity: &ServedModelIdentity) -> Option<String> {
    match identity.source_kind {
        ModelSourceKind::HuggingFace => identity
            .repository
            .as_deref()
            .map(|repo| {
                let selector = identity
                    .artifact
                    .as_deref()
                    .and_then(model_ref::quant_selector_from_gguf_file)
                    .or_else(|| identity.artifact.clone());
                model_ref::format_model_ref(repo, identity.revision.as_deref(), selector.as_deref())
            })
            .or_else(|| {
                identity
                    .canonical_ref
                    .as_deref()
                    .and_then(|model_ref| model_ref::ModelRef::parse(model_ref).ok())
                    .map(|model_ref| model_ref.display_id())
            }),
        ModelSourceKind::Catalog => identity
            .canonical_ref
            .as_deref()
            .and_then(|model_ref| model_ref::ModelRef::parse(model_ref).ok())
            .map(|model_ref| model_ref.display_id()),
        ModelSourceKind::LocalGguf | ModelSourceKind::DirectUrl | ModelSourceKind::Unknown => None,
    }
}

pub(crate) fn canonical_demand_model_ref(model: &str) -> String {
    if let Ok(model_ref) = model_ref::ModelRef::parse(model) {
        return model_ref.display_id();
    }
    crate::models::find_loaded_remote_catalog_model_exact(model)
        .map(|remote_model| crate::models::remote_catalog_model_ref(&remote_model))
        .unwrap_or_else(|| model.to_string())
}

/// Peers not directly verified within this window are considered stale
/// and excluded from gossip propagation. After 2x this duration they're removed entirely.
pub(crate) const PEER_STALE_SECS: u64 = 180; // 3 minutes

/// How long a dead-peer entry blocks transitive re-learning and outbound
/// reconnection. After this period the entry expires silently and the peer
/// can be re-discovered through normal gossip propagation. If the peer is
/// genuinely gone, no bridge peer will mention it and it stays forgotten.
pub(crate) const DEAD_PEER_TTL: std::time::Duration = std::time::Duration::from_secs(300); // 5 minutes
pub(crate) const PEER_DOWN_REPORTER_COOLDOWN_SECS: u64 = 600; // 10 minutes

pub(crate) struct MeshState {
    pub(crate) peers: HashMap<EndpointId, PeerInfo>,
    pub(crate) connections: HashMap<EndpointId, Connection>,
    pub(crate) pending_connections: HashMap<EndpointId, PendingConnectionHandshake>,
    pub(crate) next_pending_connection_attempt: u64,
    /// Remote peers' tunnel maps: peer_endpoint_id → { target_endpoint_id → tunnel_port_on_that_peer }
    pub(crate) remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>>,
    /// Peers confirmed dead — don't reconnect from gossip discovery.
    /// Cleared when the peer successfully reconnects via rejoin/join.
    /// Entries expire after [`DEAD_PEER_TTL`] so that peers recovered
    /// on other paths can be re-learned transitively through gossip.
    pub(crate) dead_peers: HashMap<EndpointId, std::time::Instant>,
    /// Tracks (reporter, target) pairs where a PeerDown claim was rejected
    /// (target was still reachable). Used to suppress repeated false reports
    /// from unreliable reporters (e.g. relay-partitioned nodes).
    pub(crate) peer_down_rejections: HashMap<(EndpointId, EndpointId), std::time::Instant>,
    /// Last accepted direct-path dial-back request per peer. This keeps path
    /// maintenance targeted even if a peer repeatedly asks us to reverse-dial.
    pub(crate) direct_path_request_last_at: HashMap<EndpointId, std::time::Instant>,
    pub(crate) seen_plugin_messages: HashMap<String, std::time::Instant>,
    pub(crate) seen_plugin_message_order: VecDeque<(std::time::Instant, String)>,
    /// Last policy-rejection status per peer — used to suppress duplicate log lines.
    /// Only logs when the status transitions (first rejection or status change).
    pub(crate) policy_rejected_peers: HashMap<EndpointId, OwnershipStatus>,
    /// Peers rejected by immutable mesh requirements. Used to keep pre-admission
    /// streams from disclosing topology after a deterministic requirement reject.
    pub(crate) requirement_rejected_peers: HashSet<EndpointId>,
    pub(crate) recent_mesh_rejections: VecDeque<MeshRequirementRejectionEvent>,
}

/// Returns `true` if the given peer has completed gossip validation and is
/// a full mesh member. Inbound unadmitted peers may be in `state.connections`
/// before `state.peers` — they are quarantined until gossip succeeds.
#[cfg(test)]
pub(crate) fn is_peer_admitted(peers: &HashMap<EndpointId, PeerInfo>, id: &EndpointId) -> bool {
    peers.get(id).is_some_and(PeerInfo::is_admitted)
}

/// Returns `true` if the given stream type is permitted before a peer has
/// been admitted through gossip, under the node's trust policy.
///
/// With a non-enforcing trust policy (`Off` or `PreferOwned`), three streams
/// bypass the quarantine gate:
/// - `STREAM_GOSSIP (0x01)`: the admission handshake itself.
/// - `STREAM_ROUTE_REQUEST (0x05)`: passive/client request-only path — caller
///   is NEVER promoted to `state.peers`.
/// - `STREAM_TUNNEL_HTTP (0x04)`: passive SDK inference path for callers that
///   have an invite token but should not need a local `/v1` HTTP listener.
///
/// When a trust policy enforces ownership (`RequireOwned` or `Allowlist`), only
/// `STREAM_GOSSIP` bypasses the gate. Otherwise a leaked invite token is a
/// bearer credential for inference: a caller rejected by the trust gate (e.g.
/// `UntrustedOwner` under `Allowlist`) could still route requests via the
/// passive paths without ever being admitted. If a node enforces who may join,
/// the same enforcement must cover who may consume. `PreferOwned` remains
/// advisory and therefore preserves the passive-client behavior of `Off`.
///
/// Every other stream — including raw tunnel (0x02) — always requires the
/// remote to have completed gossip first.
pub(crate) fn stream_allowed_before_admission(stream_type: u8, trust_policy: TrustPolicy) -> bool {
    if stream_type == STREAM_GOSSIP {
        return true;
    }
    if matches!(
        trust_policy,
        TrustPolicy::RequireOwned | TrustPolicy::Allowlist
    ) {
        return false;
    }
    stream_type == STREAM_ROUTE_REQUEST || stream_type == STREAM_TUNNEL_HTTP
}

pub(crate) fn ingest_tunnel_map(
    remote: EndpointId,
    frame: &crate::proto::node::TunnelMap,
    remote_tunnel_maps: &mut HashMap<EndpointId, HashMap<EndpointId, u16>>,
) -> Result<()> {
    if frame.owner_peer_id.as_slice() != remote.as_bytes() {
        anyhow::bail!(
            "TunnelMap owner_peer_id mismatch: frame claims owner {}, but connected peer is {}",
            hex::encode(&frame.owner_peer_id),
            remote.fmt_short()
        );
    }

    let mut tunnel_map: HashMap<EndpointId, u16> = HashMap::new();
    for entry in &frame.entries {
        if entry.target_peer_id.len() != 32 {
            anyhow::bail!(
                "TunnelMap entry has invalid target_peer_id length: {} (expected 32)",
                entry.target_peer_id.len()
            );
        }
        if entry.tunnel_port > u16::MAX as u32 {
            anyhow::bail!(
                "TunnelMap entry has out-of-range tunnel_port: {} (max {})",
                entry.tunnel_port,
                u16::MAX
            );
        }
        let arr: [u8; 32] = entry.target_peer_id.as_slice().try_into().unwrap();
        let eid = EndpointId::from(
            iroh::PublicKey::from_bytes(&arr)
                .map_err(|e| anyhow::anyhow!("Invalid target_peer_id bytes: {e}"))?,
        );
        tunnel_map.insert(eid, entry.tunnel_port as u16);
    }

    remote_tunnel_maps.insert(remote, tunnel_map);
    Ok(())
}

/// Validates the sender-identity rule for a validated `PeerLeaving` frame.
/// Returns `Ok(leaving_id)` if `frame.peer_id == remote` (sender is announcing its own departure).
/// Returns `Err(ForgedSender)` if `frame.peer_id != remote` — no peer should be removed.
pub(crate) fn resolve_peer_leaving(
    remote: EndpointId,
    frame: &crate::proto::node::PeerLeaving,
) -> Result<EndpointId, ControlFrameError> {
    if frame.peer_id.as_slice() != remote.as_bytes() {
        return Err(ControlFrameError::ForgedSender);
    }
    let arr: [u8; 32] =
        frame
            .peer_id
            .as_slice()
            .try_into()
            .map_err(|_| ControlFrameError::InvalidEndpointId {
                got: frame.peer_id.len(),
            })?;
    let pk =
        iroh::PublicKey::from_bytes(&arr).map_err(|_| ControlFrameError::InvalidEndpointId {
            got: frame.peer_id.len(),
        })?;
    Ok(EndpointId::from(pk))
}

impl Node {
    pub async fn mesh_catalog(&self) -> Vec<String> {
        // Snapshot each lock independently to avoid holding multiple locks.
        let my_available = self.available_models.lock().await.clone();
        let my_requested = self.requested_models.lock().await.clone();
        let my_serving_models = self.serving_models.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .map(|p| {
                    (
                        p.available_models.clone(),
                        p.requested_models.clone(),
                        p.serving_models.clone(),
                    )
                })
                .collect()
        };
        let mut all = std::collections::HashSet::new();
        for m in &my_available {
            all.insert(m.clone());
        }
        for m in &my_requested {
            all.insert(m.clone());
        }
        for m in &my_serving_models {
            all.insert(m.clone());
        }
        for (avail, req, serving_models) in &peer_data {
            for m in avail {
                all.insert(m.clone());
            }
            for m in req {
                all.insert(m.clone());
            }
            for m in serving_models {
                all.insert(m.clone());
            }
        }
        let mut result: Vec<String> = all.into_iter().collect();
        result.sort();
        result
    }

    pub async fn mesh_catalog_entries(&self) -> Vec<MeshCatalogEntry> {
        let names = self.mesh_catalog().await;
        let my_available = self.available_models.lock().await.clone();
        let my_served_descriptors = self.served_model_descriptors.lock().await.clone();
        let peer_descriptors: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .map(|p| p.served_model_descriptors.clone())
                .collect()
        };

        let mut by_name: HashMap<String, ServedModelDescriptor> = HashMap::new();
        for descriptor in infer_available_model_descriptors(&my_available)
            .into_iter()
            .chain(my_served_descriptors)
        {
            upsert_mesh_catalog_descriptor(&mut by_name, descriptor);
        }
        for served in peer_descriptors {
            for descriptor in served {
                upsert_mesh_catalog_descriptor(&mut by_name, descriptor);
            }
        }

        names
            .into_iter()
            .map(|model_name| MeshCatalogEntry {
                descriptor: by_name.get(&model_name).cloned(),
                model_name,
            })
            .collect()
    }

    /// Get all models currently reachable via the mesh HTTP/API ingress.
    ///
    /// This is intentionally stricter than "loaded in VRAM somewhere": split
    /// workers may contribute compute for a model but cannot accept chat
    /// requests directly.
    pub async fn models_being_served(&self) -> Vec<String> {
        let my_hosted_models = self.hosted_models.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state.peers.values().cloned().collect()
        };
        let mut served = std::collections::HashSet::new();
        for s in &my_hosted_models {
            served.insert(s.clone());
        }
        for peer in &peer_data {
            for m in peer.http_routable_models() {
                served.insert(m.clone());
            }
        }
        let mut result: Vec<String> = served.into_iter().collect();
        result.sort();
        result
    }

    /// All host IDs serving a model, with a healthy hash-preferred host first.
    /// Paused peers are excluded; deprioritized peers remain spillover capacity.
    pub async fn hosts_for_model(&self, model: &str) -> Vec<EndpointId> {
        let state = self.state.lock().await;
        let mut hosts: Vec<(EndpointId, bool)> = state
            .peers
            .values()
            .filter(|p| p.is_admitted())
            .filter(|p| p.routes_http_model(model))
            .filter_map(|p| {
                use crate::proto::node::InferenceAdmissionState;
                match p.inference_admission_state {
                    Some(
                        InferenceAdmissionState::RemotePaused | InferenceAdmissionState::AllPaused,
                    ) => None,
                    Some(InferenceAdmissionState::AcceptingDeprioritized) => Some((p.id, true)),
                    _ => Some((p.id, false)),
                }
            })
            .collect();
        hosts.sort_by_key(|(id, _)| *id);
        let mut healthy = Vec::new();
        let mut deprioritized = Vec::new();
        for (id, is_deprioritized) in hosts {
            if is_deprioritized {
                deprioritized.push(id);
            } else {
                healthy.push(id);
            }
        }
        // Rendezvous-order each health class independently. This keeps one
        // origin sticky to one replica for KV reuse, spreads different origins,
        // and when a replica becomes busy removes only that replica from the
        // healthy ordering instead of rotating every remaining peer.
        let origin = self.endpoint.id();
        let affinity_score = |peer: &EndpointId| {
            origin
                .as_bytes()
                .iter()
                .chain(peer.as_bytes())
                .fold(0xcbf29ce484222325u64, |hash, &byte| {
                    hash.wrapping_mul(0x100000001b3) ^ u64::from(byte)
                })
        };
        healthy.sort_by_key(|peer| std::cmp::Reverse(affinity_score(peer)));
        deprioritized.sort_by_key(|peer| std::cmp::Reverse(affinity_score(peer)));
        healthy.extend(deprioritized);
        healthy
    }

    /// Find ANY host in the mesh (fallback when no model match).
    pub async fn any_host(&self) -> Option<PeerInfo> {
        let state = self.state.lock().await;
        state
            .peers
            .values()
            .filter(|p| p.is_admitted())
            .find(|p| !p.http_routable_models().is_empty())
            .cloned()
    }

    /// Build the current routing table from this node's view of the mesh.
    pub async fn routing_table(&self) -> RoutingTable {
        let my_hosted_models = self.hosted_models.lock().await.clone();
        let my_role = self.role.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .filter(|peer| peer.is_admitted())
                .cloned()
                .collect()
        };
        let mut hosts = Vec::new();

        // Include self if we're serving through the local API proxy
        if !matches!(my_role, NodeRole::Client) {
            for model in my_hosted_models {
                hosts.push(RouteEntry {
                    model,
                    node_id: format!("{}", self.endpoint.id().fmt_short()),
                    endpoint_id: self.endpoint.id(),
                    vram_gb: self.vram_bytes as f64 / 1e9,
                });
            }
        }

        // Include peers that are serving through their local API proxies
        for peer in &peer_data {
            for model in peer.http_routable_models() {
                hosts.push(RouteEntry {
                    model,
                    node_id: format!("{}", peer.id.fmt_short()),
                    endpoint_id: peer.id,
                    vram_gb: peer.vram_bytes as f64 / 1e9,
                });
            }
        }

        let mesh_id = self.mesh_id().await;
        RoutingTable { hosts, mesh_id }
    }

    /// Accelerator-resident capacity used for mesh stage placement.
    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    /// Local model-fit budget, including supported CPU offload memory.
    pub fn local_runtime_capacity_bytes(&self) -> u64 {
        self.local_runtime_capacity_bytes
    }

    pub async fn peers(&self) -> Vec<PeerInfo> {
        self.state
            .lock()
            .await
            .peers
            .values()
            .filter(|peer| peer.is_admitted())
            .cloned()
            .collect()
    }

    pub(crate) async fn connection_to_peer(&self, peer_id: EndpointId) -> Result<Connection> {
        let state = self.state.lock().await;
        match state.connections.get(&peer_id).cloned() {
            Some(conn) if state.peers.get(&peer_id).is_some_and(PeerInfo::is_admitted) => Ok(conn),
            Some(conn) => {
                drop(state);
                if let Err(error) = self
                    .complete_cached_connection_gossip_single_flight(peer_id, conn, false)
                    .await
                {
                    anyhow::bail!(
                        "Failed to complete gossip with {} before opening mesh stream: {error}",
                        peer_id.fmt_short()
                    );
                }
                let state = self.state.lock().await;
                state
                    .connections
                    .get(&peer_id)
                    .cloned()
                    .filter(|_| state.peers.get(&peer_id).is_some_and(PeerInfo::is_admitted))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "No admitted connection for {} after gossip",
                            peer_id.fmt_short()
                        )
                    })
            }
            None => {
                let addr = state.peers.get(&peer_id).map(|p| p.addr.clone());
                drop(state);
                let Some(addr) = addr else {
                    anyhow::bail!("No connection or address for {}", peer_id.fmt_short());
                };
                let owner = match self.reserve_pending_connection(peer_id).await {
                    PendingConnectionReservation::Owner(owner) => owner,
                    PendingConnectionReservation::Waiter(waiter) => {
                        self.await_pending_connection(waiter).await?;
                        let state = self.state.lock().await;
                        return state
                            .connections
                            .get(&peer_id)
                            .cloned()
                            .filter(|_| {
                                state.peers.get(&peer_id).is_some_and(PeerInfo::is_admitted)
                            })
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "No admitted connection for {} after pending handshake",
                                    peer_id.fmt_short()
                                )
                            });
                    }
                };
                let conn = match tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    connect_mesh(&self.endpoint, addr),
                )
                .await
                {
                    Ok(Ok(conn)) => conn,
                    Ok(Err(error)) => {
                        let error = anyhow::anyhow!(
                            "Failed to connect to {}: {error}",
                            peer_id.fmt_short()
                        );
                        self.finish_pending_connection(
                            owner,
                            PendingConnectionOutcome::Failed(error.to_string()),
                        )
                        .await;
                        return Err(error);
                    }
                    Err(_) => {
                        let error =
                            anyhow::anyhow!("Timeout connecting to {}", peer_id.fmt_short());
                        self.finish_pending_connection(
                            owner,
                            PendingConnectionOutcome::Failed(error.to_string()),
                        )
                        .await;
                        return Err(error);
                    }
                };
                if let Err(error) = self
                    .initiate_gossip_inner(conn.clone(), peer_id, false)
                    .await
                {
                    conn.close(0u32.into(), b"on-demand-gossip-failed");
                    self.finish_pending_connection(
                        owner,
                        PendingConnectionOutcome::Failed(error.to_string()),
                    )
                    .await;
                    anyhow::bail!(
                        "Failed to complete gossip with {} before opening mesh stream: {error}",
                        peer_id.fmt_short()
                    );
                }
                self.state
                    .lock()
                    .await
                    .connections
                    .insert(peer_id, conn.clone());
                let node_for_dispatch = self.clone();
                let conn_for_dispatch = conn.clone();
                tokio::spawn(async move {
                    node_for_dispatch
                        .dispatch_streams(conn_for_dispatch, peer_id)
                        .await;
                });
                self.finish_pending_connection(owner, PendingConnectionOutcome::Admitted)
                    .await;
                Ok(conn)
            }
        }
    }

    pub(crate) async fn split_stage_path_snapshot(
        &self,
        peer_id: EndpointId,
    ) -> SplitStagePathSnapshot {
        let fallback = self.peer_stage_path_fallback(peer_id).await;
        match self.stage_connection_to_peer(peer_id).await {
            Ok(conn) => {
                split_stage_path_snapshot_from_connection(&conn).with_peer_path_fallback(fallback)
            }
            Err(error) => {
                tracing::debug!(
                    peer = %peer_id.fmt_short(),
                    error = %error,
                    "split stage path probe could not open stage connection"
                );
                SplitStagePathSnapshot::unknown().with_peer_path_fallback(fallback)
            }
        }
    }

    pub(crate) async fn peer_stage_path_fallback(
        &self,
        peer_id: EndpointId,
    ) -> Option<SelectedPathObservation> {
        let state = self.state.lock().await;
        state
            .peers
            .get(&peer_id)
            .and_then(PeerInfo::split_stage_path_fallback)
    }

    pub(crate) async fn open_mesh_subprotocol_stream(
        &self,
        peer_id: EndpointId,
        name: &str,
        major: u32,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        use prost::Message as _;

        let conn = self.connection_to_peer(peer_id).await?;
        let (mut send, recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_SUBPROTOCOL]).await?;
        let open = crate::proto::node::MeshSubprotocolOpen {
            r#gen: NODE_PROTOCOL_GENERATION,
            name: name.to_string(),
            major,
        };
        open.validate_frame()
            .map_err(|error| anyhow::anyhow!("invalid mesh subprotocol open: {error}"))?;
        write_len_prefixed(&mut send, &open.encode_to_vec()).await?;
        Ok((send, recv))
    }

    pub(crate) async fn open_skippy_stage_mesh_stream(
        &self,
        peer_id: EndpointId,
        stream_kind: u8,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        let (mut send, recv) = self
            .open_mesh_subprotocol_stream(
                peer_id,
                skippy_protocol::STAGE_SUBPROTOCOL_NAME,
                skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
            )
            .await?;
        send.write_all(&[stream_kind]).await?;
        Ok((send, recv))
    }

    pub(crate) async fn stage_connection_to_peer(&self, peer_id: EndpointId) -> Result<Connection> {
        let addr = {
            let state = self.state.lock().await;
            state.peers.get(&peer_id).map(|p| p.addr.clone())
        };
        let Some(addr) = addr else {
            anyhow::bail!("No address for stage peer {}", peer_id.fmt_short());
        };
        let conn = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            self.endpoint
                .connect(addr, skippy_protocol::STAGE_ALPN_V2)
                .await
        })
        .await
        .map_err(|_| anyhow::anyhow!("Timeout connecting to stage peer {}", peer_id.fmt_short()))?
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to connect to stage peer {}: {e}",
                peer_id.fmt_short()
            )
        })?;
        Ok(conn)
    }
}
impl Node {
    pub(crate) async fn handle_peer_down_stream(
        &self,
        remote: EndpointId,
        mut recv: iroh::endpoint::RecvStream,
    ) {
        let Some(dead_id) = self.decode_peer_down_frame(&mut recv).await else {
            return;
        };
        let report = self.peer_down_report(remote, dead_id).await;
        self.apply_peer_down_report(remote, dead_id, report).await;
    }

    pub(crate) async fn decode_peer_down_frame(
        &self,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> Option<EndpointId> {
        let frame = self.read_peer_down_frame(recv).await?;
        peer_down_endpoint_id(&frame)
    }

    pub(crate) async fn read_peer_down_frame(
        &self,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> Option<crate::proto::node::PeerDown> {
        let proto_buf = match read_len_prefixed(recv).await {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!("PeerDown: failed to read proto body — rejecting: {e}");
                return None;
            }
        };
        self.decode_peer_down_proto(&proto_buf)
    }

    pub(crate) fn decode_peer_down_proto(
        &self,
        proto_buf: &[u8],
    ) -> Option<crate::proto::node::PeerDown> {
        let frame = match crate::proto::node::PeerDown::decode(proto_buf) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("PeerDown: invalid protobuf — rejecting: {e}");
                return None;
            }
        };
        if let Err(e) = frame.validate_frame() {
            tracing::warn!("PeerDown: frame validation failed — rejecting: {e}");
            return None;
        }
        Some(frame)
    }

    pub(crate) async fn peer_down_report(
        &self,
        remote: EndpointId,
        dead_id: EndpointId,
    ) -> PeerDownReport {
        let state = self.state.lock().await;
        let conn_opt = state.connections.get(&dead_id).cloned();
        let peer = state.peers.get(&dead_id);
        let peer_addr = peer.map(|p| p.addr.clone());
        let recently_seen = peer
            .map(|p| p.last_seen.elapsed().as_secs() < PEER_STALE_SECS)
            .unwrap_or(false);
        let reporter_cooled = state
            .peer_down_rejections
            .get(&(remote, dead_id))
            .is_some_and(|t| t.elapsed().as_secs() < PEER_DOWN_REPORTER_COOLDOWN_SECS);
        PeerDownReport {
            conn_opt,
            peer_addr,
            recently_seen,
            reporter_cooled,
        }
    }

    pub(crate) async fn apply_peer_down_report(
        &self,
        remote: EndpointId,
        dead_id: EndpointId,
        report: PeerDownReport,
    ) {
        match peer_down_report_disposition(report.reporter_cooled, report.recently_seen) {
            PeerDownReportDisposition::SuppressReporterCooldown => tracing::debug!(
                "PeerDown: {} reported {} dead but reporter is in cooldown, ignoring",
                remote.fmt_short(),
                dead_id.fmt_short()
            ),
            PeerDownReportDisposition::RejectRecentlySeen => {
                self.reject_recent_peer_down_report(remote, dead_id).await;
            }
            PeerDownReportDisposition::ProbeReachability => {
                self.probe_and_apply_peer_down(remote, dead_id, report)
                    .await;
            }
        }
    }

    pub(crate) async fn reject_recent_peer_down_report(
        &self,
        remote: EndpointId,
        dead_id: EndpointId,
    ) {
        emit_mesh_info(format!(
            "ℹ️  Peer {} reported dead by {} but seen recently (direct alive), ignoring",
            dead_id.fmt_short(),
            remote.fmt_short()
        ));
        self.record_peer_down_rejection(remote, dead_id).await;
    }

    pub(crate) async fn probe_and_apply_peer_down(
        &self,
        remote: EndpointId,
        dead_id: EndpointId,
        report: PeerDownReport,
    ) {
        let should_remove = self
            .peer_down_probe_should_remove(dead_id, report.conn_opt, report.peer_addr)
            .await;
        if let Some(id) = resolve_peer_down(self.endpoint.id(), dead_id, should_remove) {
            self.remove_confirmed_peer_down(remote, id).await;
        } else if dead_id != self.endpoint.id() {
            emit_mesh_info(format!(
                "ℹ️  Peer {} reported dead by {} but still reachable, ignoring",
                dead_id.fmt_short(),
                remote.fmt_short()
            ));
            self.record_peer_down_rejection(remote, dead_id).await;
        }
    }

    pub(crate) async fn peer_down_probe_should_remove(
        &self,
        dead_id: EndpointId,
        conn_opt: Option<Connection>,
        peer_addr: Option<EndpointAddr>,
    ) -> bool {
        if let Some(conn) = conn_opt {
            return !matches!(
                tokio::time::timeout(std::time::Duration::from_secs(5), conn.open_bi()).await,
                Ok(Ok(_))
            );
        }
        let Some(addr) = peer_addr else {
            return true;
        };
        match tokio::time::timeout(
            std::time::Duration::from_secs(8),
            connect_mesh(&self.endpoint, addr),
        )
        .await
        {
            Ok(Ok(new_conn)) => {
                self.keep_reachable_peer_down_connection(dead_id, new_conn)
                    .await;
                false
            }
            _ => true,
        }
    }

    pub(crate) async fn keep_reachable_peer_down_connection(
        &self,
        dead_id: EndpointId,
        new_conn: Connection,
    ) {
        emit_mesh_info(format!(
            "ℹ️  Peer {} reported dead but we reached them, keeping",
            dead_id.fmt_short()
        ));
        let mut state = self.state.lock().await;
        if state.connections.contains_key(&dead_id) {
            return;
        }
        state.connections.insert(dead_id, new_conn.clone());
        drop(state);
        let node = self.clone();
        tokio::spawn(async move {
            node.dispatch_streams(new_conn, dead_id).await;
        });
    }

    pub(crate) async fn remove_confirmed_peer_down(&self, remote: EndpointId, id: EndpointId) {
        emit_mesh_warning(format!(
            "⚠️  Peer {} reported dead by {}, confirmed, removing",
            id.fmt_short(),
            remote.fmt_short()
        ));
        let mut state = self.state.lock().await;
        state.dead_peers.insert(id, std::time::Instant::now());
        state.connections.remove(&id);
        drop(state);
        self.remove_peer(id, MeshPeerRemovalReason::PeerDownProbeFailed)
            .await;
    }

    pub(crate) async fn record_peer_down_rejection(&self, remote: EndpointId, dead_id: EndpointId) {
        self.state
            .lock()
            .await
            .peer_down_rejections
            .insert((remote, dead_id), std::time::Instant::now());
    }
}
impl Node {
    pub(crate) async fn handle_tunnel_map_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        use prost::Message as _;

        let buf = read_len_prefixed(&mut recv).await?;
        let _ = protocol;
        let frame = crate::proto::node::TunnelMap::decode(buf.as_slice())
            .map_err(|e| anyhow::anyhow!("TunnelMap decode error: {e}"))?;

        frame
            .validate_frame()
            .map_err(|e| anyhow::anyhow!("TunnelMap validation failed: {e}"))?;

        let entry_count = frame.entries.len();
        {
            let mut state = self.state.lock().await;
            ingest_tunnel_map(remote, &frame, &mut state.remote_tunnel_maps)?;
        }

        tracing::info!(
            "Received tunnel map from {} ({} entries)",
            remote.fmt_short(),
            entry_count
        );

        Ok(())
    }
}
