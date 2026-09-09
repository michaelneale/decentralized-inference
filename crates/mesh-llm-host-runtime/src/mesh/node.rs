use super::*;
use mesh_llm_types::mesh::{DEMAND_TTL_SECS, merge_demand};
use serde_json::json;
use std::net::SocketAddr;

mod routing_telemetry;
mod startup;

pub use startup::detect_vram_bytes_capped;
use startup::{
    bind_mesh_endpoint, init_owner_runtime, startup_secret_key, wait_for_endpoint_online,
};
pub(crate) use startup::{
    default_plugin_event_source, hardware_snapshot_for_start, startup_transport_config,
};

/// Upper bound on how long shutdown waits for one iroh endpoint to close.
///
/// `Endpoint::close` queues connection-close frames and then waits for peers to
/// acknowledge them, which iroh documents as up to roughly 3s when connectivity
/// is bad or a peer has already gone away; it then waits for the endpoint and
/// connection drivers to stop. Shutdown must stay bounded, so an unresponsive
/// peer costs this budget and nothing more.
const ENDPOINT_CLOSE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Await `Endpoint::close` under [`ENDPOINT_CLOSE_TIMEOUT`].
///
/// Returning early on an already-closed endpoint keeps this idempotent, so a
/// shutdown path may call it without tracking whether an earlier one did.
///
/// On timeout iroh has still *started* closing, which downgrades the drop-time
/// `abort()` to a no-op, but the endpoint is not marked closed and the
/// ungraceful-drop ERROR can still be logged. The warning is what tells those
/// two cases apart in a log.
async fn close_endpoint_gracefully(endpoint: &Endpoint, label: &str) {
    if endpoint.is_closed() {
        return;
    }
    if tokio::time::timeout(ENDPOINT_CLOSE_TIMEOUT, endpoint.close())
        .await
        .is_err()
    {
        tracing::warn!(
            endpoint = label,
            timeout_secs = ENDPOINT_CLOSE_TIMEOUT.as_secs(),
            "iroh endpoint did not finish closing within the shutdown budget; peers may time its connections out instead of seeing them close"
        );
    }
}

/// Lightweight routing table for passive nodes (clients + standby GPU).
/// Contains just enough info to route requests to the right host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTable {
    pub hosts: Vec<RouteEntry>,
    /// Stable mesh identity — shared by all nodes in the same mesh.
    #[serde(default)]
    pub mesh_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteEntry {
    pub model: String,
    pub node_id: String,
    pub endpoint_id: EndpointId,
    pub vram_gb: f64,
}

#[derive(Clone)]
pub struct Node {
    pub(crate) endpoint: Endpoint,
    pub(crate) endpoint_secret_key: SecretKey,
    pub(crate) public_addr: Option<std::net::SocketAddr>,
    pub(crate) quic_bind: QuicBindSelection,
    pub(crate) relay_policy: RelayPolicy,
    pub(crate) owner_keypair: Option<crate::crypto::OwnerKeypair>,
    pub(crate) local_mesh_requirements: crate::MeshRequirements,
    pub(crate) state: Arc<Mutex<MeshState>>,
    /// Relay-disabled same-identity endpoints that own recovered direct connections.
    /// See [`super::direct_rescue`] for the ownership and shutdown rules.
    pub(crate) direct_rescue_endpoints: super::direct_rescue::DirectRescueEndpoints,
    pub(crate) role: Arc<Mutex<NodeRole>>,
    pub(crate) host_role_claims: Arc<Mutex<HostRoleClaims>>,
    pub(crate) models: Arc<Mutex<Vec<String>>>,
    pub(crate) model_source: Arc<Mutex<Option<String>>>,
    pub(crate) serving_models: Arc<Mutex<Vec<String>>>,
    pub(crate) served_model_descriptors: Arc<Mutex<Vec<ServedModelDescriptor>>>,
    pub(crate) model_runtime_descriptors: Arc<Mutex<Vec<ModelRuntimeDescriptor>>>,
    pub(crate) hosted_models: Arc<Mutex<Vec<String>>>,
    pub(crate) llama_ready: Arc<Mutex<bool>>,
    pub(crate) available_models: Arc<Mutex<Vec<String>>>,
    pub(crate) requested_models: Arc<Mutex<Vec<String>>>,
    pub(crate) explicit_model_interests: Arc<Mutex<Vec<String>>>,
    /// Mesh-wide demand map — merged from gossip + local API requests.
    /// This is the single source of truth for "what does the mesh want?"
    pub(crate) model_demand: Arc<std::sync::Mutex<HashMap<String, ModelDemand>>>,
    pub(crate) requirement_mesh_state: Arc<Mutex<Option<RequirementAwareMeshState>>>,
    pub(crate) mesh_id: Arc<Mutex<Option<String>>>,
    pub(crate) mesh_policy_hash: Arc<Mutex<Option<String>>>,
    pub(crate) signed_genesis_policy: Arc<Mutex<Option<crate::SignedMeshGenesisPolicy>>>,
    pub(crate) bootstrap_token: Arc<Mutex<Option<crate::SignedBootstrapToken>>>,
    /// Addresses we have been asked to join (from invite tokens), retained so
    /// the LAN beacon can unicast a dial-back hint to them even before a direct
    /// connection forms (relay-less multi-homed-initiator case).
    pub(crate) join_targets: Arc<Mutex<Vec<EndpointAddr>>>,
    pub(crate) first_joined_mesh_ts: Arc<Mutex<Option<u64>>>,
    pub(crate) accepting: Arc<(tokio::sync::Notify, std::sync::atomic::AtomicBool)>,
    /// Accelerator-resident capacity advertised to peers and split placement.
    pub(crate) vram_bytes: u64,
    /// Local fit budget, which may additionally include CPU offload memory.
    pub(crate) local_runtime_capacity_bytes: u64,
    pub(crate) peer_change_tx: watch::Sender<usize>,
    pub peer_change_rx: watch::Receiver<usize>,
    pub(crate) inflight_requests: Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) inflight_change_tx: watch::Sender<u64>,
    pub(crate) routing_metrics: crate::network::metrics::RoutingMetrics,
    pub(crate) routing_telemetry:
        Arc<std::sync::Mutex<Option<Arc<dyn crate::network::metrics::RoutingTelemetrySink>>>>,
    pub(crate) cache_affinity_inventory:
        Arc<std::sync::Mutex<mesh_llm_routing::cache_inventory::CacheInventory>>,
    pub(crate) swarm_capture: Arc<std::sync::Mutex<Option<crate::capture::SwarmCaptureRecorder>>>,
    pub(crate) local_request_metrics: Arc<LocalRequestMetricsSampler>,
    pub(crate) runtime_data_producer: crate::runtime_data::RuntimeDataProducer,
    pub(crate) tunnel_tx:
        tokio::sync::mpsc::Sender<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)>,
    pub(crate) tunnel_http_tx: tokio::sync::mpsc::Sender<(
        EndpointId,
        iroh::endpoint::SendStream,
        iroh::endpoint::RecvStream,
    )>,
    pub(crate) stage_transport_tx: tokio::sync::mpsc::Sender<(
        EndpointId,
        iroh::endpoint::SendStream,
        iroh::endpoint::RecvStream,
    )>,
    pub(crate) stage_control_tx: Arc<
        Mutex<
            Option<
                tokio::sync::mpsc::UnboundedSender<crate::inference::skippy::StageControlCommand>,
            >,
        >,
    >,
    pub(crate) stage_control_lifecycle:
        Arc<Mutex<Option<crate::inference::skippy::StageControlHandle>>>,
    pub(crate) stage_transport_bridges: Arc<Mutex<HashMap<String, StageTransportBridge>>>,
    pub(crate) stage_transport_aliases: Arc<Mutex<HashMap<String, String>>>,
    pub(crate) stage_topologies: Arc<Mutex<StageTopologyState>>,
    pub(crate) plugin_manager: Arc<Mutex<Option<crate::plugin::PluginManager>>>,
    pub(crate) display_name: Arc<Mutex<Option<String>>>,
    pub(crate) owner_attestation: Arc<Mutex<Option<SignedNodeOwnership>>>,
    pub(crate) release_attestation: Arc<Mutex<Option<crate::ReleaseBuildAttestation>>>,
    pub(crate) release_attestation_summary: Arc<Mutex<crate::ReleaseAttestationSummary>>,
    pub(crate) owner_summary: Arc<Mutex<OwnershipSummary>>,
    pub(crate) control_listener: Arc<Mutex<Option<ControlListenerLifecycle>>>,
    pub(crate) trust_store: Arc<Mutex<TrustStore>>,
    pub(crate) trust_policy: TrustPolicy,
    pub enumerate_host: bool,
    pub gpu_name: Option<String>,
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_reserved_bytes: Option<String>,
    pub gpu_mem_bandwidth_gbps: Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    pub gpu_compute_tflops_fp32: Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    pub gpu_compute_tflops_fp16: Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    pub(crate) config_state: Arc<tokio::sync::Mutex<crate::runtime::config_state::ConfigState>>,
    pub(crate) config_revision_tx: Arc<tokio::sync::watch::Sender<u64>>,
    /// Shared activity policy guard for ingress admission checks.
    pub(crate) activity_policy_guard: crate::runtime::activity_policy::ActivityPolicyGuard,
    /// Whether activity admission details are being advertised onto a public mesh.
    pub(crate) public_mesh: bool,
    /// Default private owner-requested graceful drain timeout.
    pub(crate) drain_timeout_secs: u64,
    /// Upper bound for private owner-requested graceful drain timeouts.
    pub(crate) drain_timeout_max_secs: u64,
    pub(crate) runtime_intents: Arc<std::sync::Mutex<Vec<crate::runtime::DesiredRuntimeIntent>>>,
    pub(crate) runtime_instance_lifecycles: Arc<
        std::sync::Mutex<
            HashMap<u16, Arc<tokio::sync::Mutex<crate::runtime::InstanceLifecycleRecord>>>,
        >,
    >,
    pub(crate) owner_lifecycle_response_cache: OwnerLifecycleResponseCache,
    /// Channel to send model lifecycle intents from owner-control commands to the reconciler.
    /// Set during runtime initialization; None when running outside the control loop.
    pub(crate) model_intent_tx:
        Arc<tokio::sync::Mutex<Option<tokio::sync::mpsc::Sender<crate::runtime::ModelIntent>>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RequirementAwareMeshState {
    pub(crate) mesh_id: String,
    pub(crate) policy_hash: String,
    pub(crate) policy: crate::MeshGenesisPolicy,
    pub(crate) signed_policy: Option<crate::SignedMeshGenesisPolicy>,
    pub(crate) bootstrap_token: Option<crate::SignedBootstrapToken>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LocalRequestMetricsSnapshot {
    pub accepted_request_counts: Vec<u64>,
    pub latency_samples_ms: Vec<u64>,
}

#[derive(Default)]
pub(crate) struct LocalRequestMetricsSampler {
    pub(crate) inner: std::sync::Mutex<LocalRequestMetricsWindow>,
}

#[derive(Default)]
pub(crate) struct LocalRequestMetricsWindow {
    pub(crate) accepted_by_second: VecDeque<(u64, u64)>,
    pub(crate) completed_latencies_ms: VecDeque<(u64, u64)>,
}

pub(crate) struct PeerDownReport {
    pub(crate) conn_opt: Option<Connection>,
    pub(crate) peer_addr: Option<EndpointAddr>,
    pub(crate) recently_seen: bool,
    pub(crate) reporter_cooled: bool,
}

pub(crate) fn peer_down_endpoint_id(frame: &crate::proto::node::PeerDown) -> Option<EndpointId> {
    let peer_id_arr: [u8; 32] = match frame.peer_id.as_slice().try_into() {
        Ok(bytes) => bytes,
        Err(_) => {
            tracing::warn!("PeerDown: peer_id is not 32 bytes — rejecting");
            return None;
        }
    };
    match iroh::PublicKey::from_bytes(&peer_id_arr) {
        Ok(key) => Some(EndpointId::from(key)),
        Err(_) => {
            tracing::warn!("PeerDown: peer_id is not a valid public key — rejecting");
            None
        }
    }
}

impl LocalRequestMetricsSampler {
    pub(crate) fn record_request_accepted(&self) {
        let now_sec = now_secs();
        let mut guard = self
            .inner
            .lock()
            .expect("pretty request metrics mutex poisoned");
        guard.prune(now_sec);
        if let Some((second, count)) = guard.accepted_by_second.back_mut()
            && *second == now_sec
        {
            *count += 1;
            return;
        }
        guard.accepted_by_second.push_back((now_sec, 1));
    }

    pub(crate) fn record_request_completed(&self, started_at: std::time::Instant) {
        let now_sec = now_secs();
        let latency_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        let mut guard = self
            .inner
            .lock()
            .expect("pretty request metrics mutex poisoned");
        guard.prune(now_sec);
        guard
            .completed_latencies_ms
            .push_back((now_sec, latency_ms));
    }

    pub(crate) fn snapshot(&self) -> LocalRequestMetricsSnapshot {
        let now_sec = now_secs();
        let window_start = now_sec.saturating_sub(PRETTY_LOCAL_REQUEST_WINDOW_SECS - 1);
        let mut guard = self
            .inner
            .lock()
            .expect("pretty request metrics mutex poisoned");
        guard.prune(now_sec);

        let accepted_by_second = guard
            .accepted_by_second
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        let accepted_request_counts = (window_start..=now_sec)
            .map(|second| accepted_by_second.get(&second).copied().unwrap_or(0))
            .collect();
        let latency_samples_ms = guard
            .completed_latencies_ms
            .iter()
            .filter_map(|(second, latency_ms)| (*second >= window_start).then_some(*latency_ms))
            .collect();

        LocalRequestMetricsSnapshot {
            accepted_request_counts,
            latency_samples_ms,
        }
    }
}

impl LocalRequestMetricsWindow {
    pub(crate) fn prune(&mut self, now_sec: u64) {
        let oldest_kept_second = now_sec.saturating_sub(PRETTY_LOCAL_REQUEST_WINDOW_SECS - 1);
        while let Some((second, _)) = self.accepted_by_second.front() {
            if *second < oldest_kept_second {
                self.accepted_by_second.pop_front();
            } else {
                break;
            }
        }
        while let Some((second, _)) = self.completed_latencies_ms.front() {
            if *second < oldest_kept_second {
                self.completed_latencies_ms.pop_front();
            } else {
                break;
            }
        }
    }
}

impl Node {
    pub(crate) fn set_swarm_capture_recorder(
        &self,
        recorder: Option<crate::capture::SwarmCaptureRecorder>,
    ) {
        *self
            .swarm_capture
            .lock()
            .expect("swarm capture recorder lock poisoned") = recorder;
    }

    pub(crate) fn swarm_capture_recorder(&self) -> Option<crate::capture::SwarmCaptureRecorder> {
        self.swarm_capture
            .lock()
            .expect("swarm capture recorder lock poisoned")
            .clone()
    }

    pub(crate) fn swarm_capture_enabled(&self) -> bool {
        self.swarm_capture
            .lock()
            .expect("swarm capture recorder lock poisoned")
            .is_some()
    }

    pub(crate) fn capture_event(&self, event: &str, fields: impl FnOnce() -> serde_json::Value) {
        if let Some(recorder) = self.swarm_capture_recorder() {
            recorder.record_event(event, fields());
        }
    }

    pub(crate) fn capture_peer_observation(
        &self,
        event: &str,
        peer: &PeerInfo,
        source: &str,
        bridge_id: Option<EndpointId>,
    ) {
        self.capture_event(event, || peer_capture_fields(peer, source, bridge_id));
    }

    pub(crate) fn capture_peer_rejected(
        &self,
        id: EndpointId,
        _addr: &EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: &OwnershipSummary,
        source: &str,
        bridge_id: Option<EndpointId>,
    ) {
        self.capture_event("peer_rejected", || {
            json!({
                "peer": endpoint_id_capture_fields(id),
                "source": source,
                "bridge": bridge_id.map(endpoint_id_capture_fields),
                "role": &ann.role,
                "version": &ann.version,
                "hostname": &ann.hostname,
                "mesh_id": &ann.mesh_id,
                "models": &ann.models,
                "serving_models": &ann.serving_models,
                "hosted_models": &ann.hosted_models,
                "available_models": &ann.available_models,
                "requested_models": &ann.requested_models,
                "gpu_name": &ann.gpu_name,
                "is_soc": ann.is_soc,
                "vram_bytes": ann.vram_bytes,
                "latency_ms": ann.latency_ms,
                "latency_source": ann.latency_source.map(|value| value.as_str_name()),
                "owner": owner_summary,
            })
        });
    }

    pub(crate) fn capture_gossip_inbound(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        announcement_count: usize,
    ) {
        self.capture_event("gossip_inbound", || {
            json!({
                "remote": endpoint_id_capture_fields(remote),
                "protocol": format!("{protocol:?}"),
                "announcement_count": announcement_count,
            })
        });
    }

    pub(crate) fn capture_path_observation(
        &self,
        remote: EndpointId,
        path_type: &str,
        rtt_ms: Option<u32>,
        observed_direct_remote_addr: Option<SocketAddr>,
        source: &str,
    ) {
        let observed_via_relay = path_type == "relay";
        self.capture_event("peer_path_observed", || json!({
            "remote": endpoint_id_capture_fields(remote),
            "path_type": path_type,
            "rtt_ms": rtt_ms,
            "observed_direct_remote_addr": observed_direct_remote_addr.map(|addr| addr.to_string()),
            "observed_via_relay": observed_via_relay,
            "direct_addr_available": observed_direct_remote_addr.is_some(),
            "source": source,
        }));
    }

    pub(crate) fn capture_selected_connection_path(
        &self,
        remote: EndpointId,
        conn: &Connection,
        source: &str,
    ) -> Option<SelectedPathObservation> {
        let observation = selected_path_observation(conn)?;
        self.capture_path_observation(
            remote,
            observation.path_type,
            observation.rtt_ms,
            observation.observed_direct_remote_addr,
            source,
        );
        Some(observation)
    }

    pub(crate) fn capture_connection_event(&self, event: ConnectionCaptureEvent<'_>) {
        self.capture_event(event.event, || {
            json!({
                "remote": endpoint_id_capture_fields(event.remote),
                "direction": event.direction,
                "phase": event.phase,
                "protocol": event.protocol.map(|value| format!("{value:?}")),
                "path_type": event.path_type,
                "rtt_ms": event.rtt_ms,
                "admitted_peer": event.admitted_peer,
                "reason": event.reason,
            })
        });
    }

    pub(crate) fn capture_direct_proof_of_life(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        announcement_count: usize,
        recovered_from_dead: bool,
        prior_state: &str,
    ) {
        self.capture_event("peer_direct_proof_of_life", || {
            json!({
                "remote": endpoint_id_capture_fields(remote),
                "protocol": format!("{protocol:?}"),
                "announcement_count": announcement_count,
                "recovered_from_dead": recovered_from_dead,
                "prior_state": prior_state,
            })
        });
    }

    pub(crate) fn capture_peer_lifecycle_event(&self, event: PeerLifecycleCaptureEvent<'_>) {
        self.capture_event(event.event, || {
            json!({
                "peer": endpoint_id_capture_fields(event.peer),
                "reason": event.reason,
                "reporter": event.reporter.map(endpoint_id_capture_fields),
                "last_seen_age_ms": event.last_seen_age_ms,
                "last_mentioned_age_ms": event.last_mentioned_age_ms,
                "had_connection": event.had_connection,
                "bridge": event.bridge_id.map(endpoint_id_capture_fields),
            })
        });
    }

    pub(crate) async fn capture_peer_lifecycle_snapshot(
        &self,
        event: &str,
        peer: EndpointId,
        reason: &str,
        reporter: Option<EndpointId>,
    ) {
        if !self.swarm_capture_enabled() {
            return;
        }

        let (last_seen_age_ms, last_mentioned_age_ms, had_connection, bridge_id) = {
            let state = self.state.lock().await;
            let peer_info = state.peers.get(&peer);
            (
                peer_info.map(|info| elapsed_ms_u64(info.last_seen.elapsed())),
                peer_info.map(|info| elapsed_ms_u64(info.last_mentioned.elapsed())),
                Some(state.connections.contains_key(&peer)),
                peer_info
                    .and_then(|info| info.propagated_latency.as_ref())
                    .and_then(|latency| latency.observer_id),
            )
        };
        self.capture_peer_lifecycle_event(PeerLifecycleCaptureEvent {
            event,
            peer,
            reason,
            reporter,
            last_seen_age_ms,
            last_mentioned_age_ms,
            had_connection,
            bridge_id,
        });
    }

    pub(crate) fn capture_stream_observation(
        &self,
        remote: EndpointId,
        stream_type: u8,
        protocol: ControlProtocol,
        admitted: bool,
    ) {
        self.capture_event("mesh_stream_observed", || {
            json!({
                "remote": endpoint_id_capture_fields(remote),
                "stream_type": stream_type,
                "protocol": format!("{protocol:?}"),
                "admitted": admitted,
            })
        });
    }

    pub(crate) fn capture_stream_rejected(
        &self,
        remote: EndpointId,
        stream_type: u8,
        protocol: ControlProtocol,
        reason: &str,
    ) {
        self.capture_event("mesh_stream_rejected", || {
            json!({
                "remote": endpoint_id_capture_fields(remote),
                "stream_type": stream_type,
                "protocol": format!("{protocol:?}"),
                "reason": reason,
            })
        });
    }

    pub(crate) fn capture_route_request(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        outcome: &str,
    ) {
        self.capture_event("route_request", || {
            json!({
                "remote": endpoint_id_capture_fields(remote),
                "protocol": format!("{protocol:?}"),
                "outcome": outcome,
            })
        });
    }

    pub(crate) fn capture_http_request(&self, event: HttpCaptureEvent<'_>) {
        self.capture_event(event.event, || {
            json!({
                "source_addr": event.source_addr.map(|addr| addr.to_string()),
                "method": event.method,
                "path": crate::capture::http_path_without_query(event.path),
                "query_present": event.path.contains('?'),
                "body_len_bytes": event.body_len_bytes,
                "model": event.model_name,
                "completion_tokens": event.completion_tokens,
                "stream": event.stream,
            })
        });
    }
}
impl Node {
    pub(crate) async fn stop_stage_transport_bridge(
        &self,
        topology_id: &str,
        run_id: &str,
        stage_id: &str,
    ) {
        let key = stage_runtime_status_key(topology_id, run_id, stage_id);
        if let Some(bridge) = self.stage_transport_bridges.lock().await.remove(&key) {
            bridge.abort();
        }
    }

    pub fn local_request_metrics_snapshot(&self) -> LocalRequestMetricsSnapshot {
        self.local_request_metrics.snapshot()
    }

    pub(crate) fn runtime_data_collector(&self) -> crate::runtime_data::RuntimeDataCollector {
        self.runtime_data_producer.collector()
    }

    pub async fn owner_summary(&self) -> OwnershipSummary {
        self.owner_summary.lock().await.clone()
    }

    pub async fn release_attestation_summary(&self) -> crate::ReleaseAttestationSummary {
        self.release_attestation_summary.lock().await.clone()
    }

    pub async fn control_endpoint(&self) -> Option<String> {
        let guard = self.control_listener.lock().await;
        guard.as_ref().map(|listener| listener.token.clone())
    }

    pub async fn shutdown_control_listener(&self) {
        let lifecycle = self.control_listener.lock().await.take();
        if let Some(lifecycle) = lifecycle {
            lifecycle
                .shutdown_requested
                .store(true, std::sync::atomic::Ordering::Release);
            lifecycle.shutdown.notify_waiters();
            let _ = lifecycle.task.await;
            close_endpoint_gracefully(&lifecycle.endpoint, "owner-control").await;
        }
        self.shutdown_stage_control().await;
    }

    async fn shutdown_stage_control(&self) {
        *self.stage_control_tx.lock().await = None;
        let lifecycle = self.stage_control_lifecycle.lock().await.take();
        if let Some(lifecycle) = lifecycle
            && let Err(error) = lifecycle.shutdown().await
        {
            tracing::warn!("stage control shutdown failed: {error:#}");
        }
    }

    /// Close the main mesh QUIC endpoint as the last step of runtime shutdown.
    ///
    /// iroh only releases an endpoint gracefully when `Endpoint::close` is
    /// awaited. Dropping it instead makes `EndpointInner::drop` log
    ///
    /// ```text
    /// ERROR iroh::socket: Endpoint dropped without calling `Endpoint::close`. Aborting ungracefully.
    /// ```
    ///
    /// and abort the socket, which also denies peers the connection-close
    /// frames — they are left to time the connections out and report this node
    /// as failed rather than as cleanly departed.
    ///
    /// Call this only after every subsystem that speaks over the mesh endpoint
    /// (control listener, plugins, loaded models and their stage connections,
    /// the final gossip updates) has finished. `accept_loop` observes the close
    /// as `Endpoint::accept` returning `None` and exits on its own.
    pub async fn close_endpoint(&self) {
        // Closing the registry to new installs is what makes this drain
        // complete: the relay-health monitor is an untracked task holding a
        // `Node` clone, so without the flag it could park a fresh rescue
        // endpoint after the drain had already passed, leaving it open and
        // unowned for the process lifetime.
        let rescue_endpoints = self.direct_rescue_endpoints.drain_for_shutdown().await;

        // One shared budget for all rescue endpoints, not one each. These are
        // per-peer, so a mesh with several rescued peers would otherwise add
        // ENDPOINT_CLOSE_TIMEOUT per endpoint to shutdown before the main
        // endpoint even starts closing.
        if !rescue_endpoints.is_empty() {
            let closes = rescue_endpoints
                .iter()
                .map(|endpoint| endpoint.close())
                .collect::<Vec<_>>();
            if tokio::time::timeout(
                ENDPOINT_CLOSE_TIMEOUT,
                futures_util::future::join_all(closes),
            )
            .await
            .is_err()
            {
                tracing::warn!(
                    endpoint = "direct-rescue",
                    count = rescue_endpoints.len(),
                    timeout_secs = ENDPOINT_CLOSE_TIMEOUT.as_secs(),
                    "direct-rescue endpoints did not finish closing within the shutdown budget; peers may time those connections out instead of seeing them close"
                );
            }
        }

        close_endpoint_gracefully(&self.endpoint, "mesh").await;
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "startup wires independent node/runtime subsystems; changing the public constructor shape is outside this rebase repair"
    )]
    pub async fn start(
        role: NodeRole,
        relay: RelayConfig<'_>,
        quic_bind: QuicBindSelection,
        max_vram_gb: Option<f64>,
        enumerate_host: bool,
        owner_config: Option<OwnerRuntimeConfig>,
        config_path: Option<&std::path::Path>,
        local_mesh_requirements: crate::MeshRequirements,
    ) -> Result<(Self, TunnelChannels)> {
        let secret_key = startup_secret_key(&role).await?;
        let endpoint = bind_mesh_endpoint(secret_key.clone(), relay, quic_bind).await?;
        if relay.policy.uses_relay() {
            // Wait briefly for relay connection so the invite token includes the relay URL.
            // On sinkholed networks this times out and we proceed without relay (direct UDP only).
            wait_for_endpoint_online(
                &endpoint,
                "Relay connected",
                "Relay connection timed out (5s) — proceeding without relay",
            )
            .await;
        }

        // Use iroh's net report because it probes through the endpoint's actual QUIC sockets.
        // Its direct address includes the observed NAT-mapped port; substituting the configured
        // bind port would advertise an address that was never verified externally.
        let public_addr = if relay.policy.uses_raw_stun() {
            stun_public_addr(&endpoint).await
        } else {
            tracing::info!("Raw STUN: disabled by LAN-only discovery mode");
            None
        };

        let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
        let (inflight_change_tx, _inflight_change_rx) = watch::channel(0u64);
        let (tunnel_tx, tunnel_rx) = tokio::sync::mpsc::channel(256);
        let (tunnel_http_tx, tunnel_http_rx) = tokio::sync::mpsc::channel(256);
        let (stage_transport_tx, stage_transport_rx) = tokio::sync::mpsc::channel(256);

        let hardware = hardware_snapshot_for_start(
            startup::hardware_survey_for_start(max_vram_gb, enumerate_host),
            &role,
            max_vram_gb,
        );
        let owner_runtime = init_owner_runtime(
            owner_config.as_ref(),
            endpoint.id(),
            hardware.hostname.clone(),
        )?;
        let owner_summary = verify_node_ownership(
            owner_runtime.owner_attestation.as_ref(),
            endpoint.id().as_bytes(),
            &owner_runtime.trust_store,
            TrustPolicy::Off,
            current_time_unix_ms(),
        );
        let config_state_init = {
            let path = crate::plugin::config_path(config_path)
                .unwrap_or_else(|_| std::path::PathBuf::from("config.toml"));
            crate::runtime::config_state::ConfigState::load(&path)?
        };
        let config_revision_init = config_state_init.revision();
        let runtime_data_collector = crate::runtime_data::RuntimeDataCollector::new();
        let runtime_data_producer =
            runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
                scope: "routing",
                plugin_data_key: None,
                plugin_endpoint_key: None,
            });

        let owner_keypair = owner_config
            .as_ref()
            .and_then(|config| config.keypair.clone());
        let activity_policy_config = owner_config
            .as_ref()
            .map(|config| config.activity.clone())
            .unwrap_or_default();
        let public_mesh = owner_config
            .as_ref()
            .is_some_and(|config| config.public_mesh);
        let drain_timeout_secs = owner_config
            .as_ref()
            .map_or(mesh_llm_config::DEFAULT_DRAIN_TIMEOUT_SECS, |config| {
                config.drain_timeout_secs
            });
        let drain_timeout_max_secs = owner_config
            .as_ref()
            .map_or(mesh_llm_config::DEFAULT_DRAIN_TIMEOUT_MAX_SECS, |config| {
                config.drain_timeout_max_secs
            });

        let node = Node {
            endpoint,
            endpoint_secret_key: secret_key.clone(),
            public_addr,
            quic_bind,
            relay_policy: relay.policy,
            owner_keypair,
            local_mesh_requirements,
            state: Arc::new(Mutex::new(MeshState {
                peers: HashMap::new(),
                connections: HashMap::new(),
                pending_connections: HashMap::new(),
                next_pending_connection_attempt: 1,
                remote_tunnel_maps: HashMap::new(),
                dead_peers: HashMap::new(),
                peer_down_rejections: HashMap::new(),
                direct_path_request_last_at: HashMap::new(),
                seen_plugin_messages: HashMap::new(),
                seen_plugin_message_order: VecDeque::new(),
                policy_rejected_peers: HashMap::new(),
                requirement_rejected_peers: HashSet::new(),
                recent_mesh_rejections: VecDeque::new(),
            })),
            direct_rescue_endpoints: super::direct_rescue::DirectRescueEndpoints::default(),
            role: Arc::new(Mutex::new(role)),
            host_role_claims: Arc::new(Mutex::new(HostRoleClaims::default())),
            models: Arc::new(Mutex::new(Vec::new())),
            model_source: Arc::new(Mutex::new(None)),
            serving_models: Arc::new(Mutex::new(Vec::new())),
            served_model_descriptors: Arc::new(Mutex::new(Vec::new())),
            model_runtime_descriptors: Arc::new(Mutex::new(Vec::new())),
            hosted_models: Arc::new(Mutex::new(Vec::new())),
            llama_ready: Arc::new(Mutex::new(false)),
            available_models: Arc::new(Mutex::new(Vec::new())),
            requested_models: Arc::new(Mutex::new(Vec::new())),
            explicit_model_interests: Arc::new(Mutex::new(Vec::new())),
            model_demand: Arc::new(std::sync::Mutex::new(HashMap::new())),
            requirement_mesh_state: Arc::new(Mutex::new(None)),
            mesh_id: Arc::new(Mutex::new(None)),
            mesh_policy_hash: Arc::new(Mutex::new(None)),
            signed_genesis_policy: Arc::new(Mutex::new(None)),
            bootstrap_token: Arc::new(Mutex::new(None)),
            join_targets: Arc::new(Mutex::new(Vec::new())),
            first_joined_mesh_ts: Arc::new(Mutex::new(None)),
            accepting: Arc::new((
                tokio::sync::Notify::new(),
                std::sync::atomic::AtomicBool::new(false),
            )),
            vram_bytes: hardware.vram_bytes,
            local_runtime_capacity_bytes: hardware.local_runtime_capacity_bytes,
            peer_change_tx,
            peer_change_rx,
            inflight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            inflight_change_tx,
            routing_metrics: crate::network::metrics::RoutingMetrics::default(),
            routing_telemetry: Arc::new(std::sync::Mutex::new(None)),
            cache_affinity_inventory: Arc::new(std::sync::Mutex::new(
                mesh_llm_routing::cache_inventory::CacheInventory::default(),
            )),
            swarm_capture: Arc::new(std::sync::Mutex::new(None)),
            local_request_metrics: Arc::new(LocalRequestMetricsSampler::default()),
            runtime_data_producer,
            tunnel_tx,
            tunnel_http_tx,
            stage_transport_tx,
            stage_control_tx: Arc::new(Mutex::new(None)),
            stage_control_lifecycle: Arc::new(Mutex::new(None)),
            stage_transport_bridges: Arc::new(Mutex::new(HashMap::new())),
            stage_transport_aliases: Arc::new(Mutex::new(HashMap::new())),
            stage_topologies: Arc::new(Mutex::new(StageTopologyState::default())),
            plugin_manager: Arc::new(Mutex::new(None)),
            display_name: Arc::new(Mutex::new(None)),
            owner_attestation: Arc::new(Mutex::new(owner_runtime.owner_attestation)),
            release_attestation: Arc::new(Mutex::new(None)),
            release_attestation_summary: Arc::new(Mutex::new(
                crate::ReleaseAttestationSummary::default(),
            )),
            owner_summary: Arc::new(Mutex::new(owner_summary)),
            control_listener: Arc::new(Mutex::new(None)),
            trust_store: Arc::new(Mutex::new(owner_runtime.trust_store)),
            trust_policy: owner_runtime.trust_policy,
            enumerate_host,
            gpu_name: hardware.gpu_name,
            hostname: hardware.hostname,
            is_soc: hardware.is_soc,
            gpu_vram: hardware.gpu_vram,
            gpu_reserved_bytes: hardware.gpu_reserved_bytes,
            gpu_mem_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
            gpu_compute_tflops_fp32: Arc::new(tokio::sync::Mutex::new(None)),
            gpu_compute_tflops_fp16: Arc::new(tokio::sync::Mutex::new(None)),
            config_state: Arc::new(tokio::sync::Mutex::new(config_state_init)),
            config_revision_tx: {
                let (tx, _rx) = tokio::sync::watch::channel(config_revision_init);
                Arc::new(tx)
            },
            activity_policy_guard: crate::runtime::activity_policy::ActivityPolicyGuard::new(
                &activity_policy_config,
            ),
            public_mesh,
            drain_timeout_secs,
            drain_timeout_max_secs,
            runtime_intents: Arc::new(std::sync::Mutex::new(Vec::new())),
            runtime_instance_lifecycles: Arc::new(std::sync::Mutex::new(HashMap::new())),
            owner_lifecycle_response_cache: OwnerLifecycleResponseCache::default(),
            model_intent_tx: Arc::new(tokio::sync::Mutex::new(None)),
        };

        node.maybe_start_control_listener(
            secret_key,
            owner_config.as_ref().and_then(|config| config.control_bind),
            owner_config
                .as_ref()
                .and_then(|config| config.control_advertise_addr),
        )
        .await?;

        // Accept loop starts but waits for start_accepting() before processing connections.
        // This lets a node exist before it is ready to accept mesh traffic.
        let node2 = node.clone();
        tokio::spawn(async move {
            node2.accept_loop().await;
        });

        Ok((
            node,
            TunnelChannels {
                rpc: tunnel_rx,
                http: tunnel_http_rx,
                stage: stage_transport_rx,
            },
        ))
    }

    #[cfg(test)]
    pub async fn new_for_tests(role: NodeRole) -> Result<Self> {
        let (node, _) = Self::new_for_tests_with_secret(role).await?;
        Ok(node)
    }

    #[cfg(test)]
    pub(crate) async fn new_for_tests_with_secret(role: NodeRole) -> Result<(Self, SecretKey)> {
        let (node, secret_key) = {
            let secret_key = SecretKey::generate();
            let transport_config = iroh::endpoint::QuicTransportConfig::builder()
                .max_concurrent_bidi_streams(1024u32.into())
                .build();
            let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
                .secret_key(secret_key.clone())
                .alpns(vec![ALPN.to_vec(), skippy_protocol::STAGE_ALPN_V2.to_vec()])
                .relay_mode(iroh::endpoint::RelayMode::Disabled)
                .transport_config(transport_config)
                .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
                .bind()
                .await?;
            (
                Self::new_test_node_from_endpoint(role, endpoint, secret_key.clone()),
                secret_key,
            )
        };
        Ok((node, secret_key))
    }

    #[cfg(test)]
    pub(crate) fn new_test_node_from_endpoint(
        role: NodeRole,
        endpoint: Endpoint,
        secret_key: SecretKey,
    ) -> Self {
        let (peer_change_tx, peer_change_rx) = watch::channel(0usize);
        let (inflight_change_tx, _inflight_change_rx) = watch::channel(0u64);
        let (tunnel_tx, _tunnel_rx) = tokio::sync::mpsc::channel(256);
        let (tunnel_http_tx, _tunnel_http_rx) = tokio::sync::mpsc::channel(256);
        let (stage_transport_tx, _stage_transport_rx) = tokio::sync::mpsc::channel(256);
        let runtime_data_collector = crate::runtime_data::RuntimeDataCollector::new();
        let runtime_data_producer =
            runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
                scope: "routing",
                plugin_data_key: None,
                plugin_endpoint_key: None,
            });

        Node {
            endpoint,
            endpoint_secret_key: secret_key,
            public_addr: None,
            quic_bind: QuicBindSelection::default(),
            relay_policy: RelayPolicy::Disabled,
            owner_keypair: None,
            local_mesh_requirements: crate::MeshRequirements::unrestricted(),
            state: Arc::new(Mutex::new(MeshState {
                peers: HashMap::new(),
                connections: HashMap::new(),
                pending_connections: HashMap::new(),
                next_pending_connection_attempt: 1,
                remote_tunnel_maps: HashMap::new(),
                dead_peers: HashMap::new(),
                peer_down_rejections: HashMap::new(),
                direct_path_request_last_at: HashMap::new(),
                seen_plugin_messages: HashMap::new(),
                seen_plugin_message_order: VecDeque::new(),
                policy_rejected_peers: HashMap::new(),
                requirement_rejected_peers: HashSet::new(),
                recent_mesh_rejections: VecDeque::new(),
            })),
            direct_rescue_endpoints: super::direct_rescue::DirectRescueEndpoints::default(),
            role: Arc::new(Mutex::new(role)),
            host_role_claims: Arc::new(Mutex::new(HostRoleClaims::default())),
            models: Arc::new(Mutex::new(Vec::new())),
            model_source: Arc::new(Mutex::new(None)),
            serving_models: Arc::new(Mutex::new(Vec::new())),
            served_model_descriptors: Arc::new(Mutex::new(Vec::new())),
            model_runtime_descriptors: Arc::new(Mutex::new(Vec::new())),
            hosted_models: Arc::new(Mutex::new(Vec::new())),
            llama_ready: Arc::new(Mutex::new(false)),
            available_models: Arc::new(Mutex::new(Vec::new())),
            requested_models: Arc::new(Mutex::new(Vec::new())),
            explicit_model_interests: Arc::new(Mutex::new(Vec::new())),
            model_demand: Arc::new(std::sync::Mutex::new(HashMap::new())),
            requirement_mesh_state: Arc::new(Mutex::new(None)),
            mesh_id: Arc::new(Mutex::new(None)),
            mesh_policy_hash: Arc::new(Mutex::new(None)),
            signed_genesis_policy: Arc::new(Mutex::new(None)),
            bootstrap_token: Arc::new(Mutex::new(None)),
            join_targets: Arc::new(Mutex::new(Vec::new())),
            first_joined_mesh_ts: Arc::new(Mutex::new(None)),
            accepting: Arc::new((
                tokio::sync::Notify::new(),
                std::sync::atomic::AtomicBool::new(false),
            )),
            vram_bytes: 0,
            local_runtime_capacity_bytes: 0,
            peer_change_tx,
            peer_change_rx,
            inflight_requests: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            inflight_change_tx,
            routing_metrics: crate::network::metrics::RoutingMetrics::default(),
            routing_telemetry: Arc::new(std::sync::Mutex::new(None)),
            cache_affinity_inventory: Arc::new(std::sync::Mutex::new(
                mesh_llm_routing::cache_inventory::CacheInventory::default(),
            )),
            swarm_capture: Arc::new(std::sync::Mutex::new(None)),
            local_request_metrics: Arc::new(LocalRequestMetricsSampler::default()),
            runtime_data_producer,
            tunnel_tx,
            tunnel_http_tx,
            stage_transport_tx,
            stage_control_tx: Arc::new(Mutex::new(None)),
            stage_control_lifecycle: Arc::new(Mutex::new(None)),
            stage_transport_bridges: Arc::new(Mutex::new(HashMap::new())),
            stage_transport_aliases: Arc::new(Mutex::new(HashMap::new())),
            stage_topologies: Arc::new(Mutex::new(StageTopologyState::default())),
            plugin_manager: Arc::new(Mutex::new(None)),
            display_name: Arc::new(Mutex::new(None)),
            owner_attestation: Arc::new(Mutex::new(None)),
            release_attestation: Arc::new(Mutex::new(None)),
            release_attestation_summary: Arc::new(Mutex::new(
                crate::ReleaseAttestationSummary::default(),
            )),
            owner_summary: Arc::new(Mutex::new(OwnershipSummary::default())),
            control_listener: Arc::new(Mutex::new(None)),
            trust_store: Arc::new(Mutex::new(TrustStore::default())),
            trust_policy: TrustPolicy::Off,
            enumerate_host: false,
            gpu_name: None,
            hostname: None,
            is_soc: None,
            gpu_vram: None,
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: Arc::new(tokio::sync::Mutex::new(None)),
            gpu_compute_tflops_fp32: Arc::new(tokio::sync::Mutex::new(None)),
            gpu_compute_tflops_fp16: Arc::new(tokio::sync::Mutex::new(None)),
            config_state: Arc::new(tokio::sync::Mutex::new(
                crate::runtime::config_state::ConfigState::default(),
            )),
            config_revision_tx: {
                let (tx, _rx) = tokio::sync::watch::channel(0);
                Arc::new(tx)
            },
            activity_policy_guard: crate::runtime::activity_policy::ActivityPolicyGuard::new(
                &mesh_llm_config::RuntimeActivityConfig::default(),
            ),
            public_mesh: false,
            drain_timeout_secs: mesh_llm_config::DEFAULT_DRAIN_TIMEOUT_SECS,
            drain_timeout_max_secs: mesh_llm_config::DEFAULT_DRAIN_TIMEOUT_MAX_SECS,
            runtime_intents: Arc::new(std::sync::Mutex::new(Vec::new())),
            runtime_instance_lifecycles: Arc::new(std::sync::Mutex::new(HashMap::new())),
            owner_lifecycle_response_cache: OwnerLifecycleResponseCache::default(),
            model_intent_tx: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    pub(crate) async fn maybe_start_control_listener(
        &self,
        secret_key: SecretKey,
        bind_addr: Option<std::net::SocketAddr>,
        advertise_addr: Option<std::net::SocketAddr>,
    ) -> Result<()> {
        if self.local_verified_owner_id().await.is_none() {
            return Ok(());
        }

        // The owner-control listener deliberately shares the node's secret key
        // (and therefore its iroh endpoint id) with the main mesh endpoint: the
        // control protocol validates the dialed `target_node_id` against the
        // main endpoint id (`verify_control_plane_target_node`), so the control
        // endpoint MUST present that same id.
        //
        // Because the id is shared, this endpoint must NOT register with the
        // relay. An iroh relay keeps only one active connection per endpoint id:
        // a second same-id registration evicts the first ("Another endpoint
        // connected with the same endpoint id. No more messages will be
        // received."). The control listener binds *after* the main mesh
        // endpoint, so if it also joined the relay it would steal the main
        // endpoint's relay slot and silently cut off all relay-delivered mesh
        // traffic (gossip, joins, inference routing) — breaking relay fallback
        // for any peer that cannot reach this node directly. Keeping the control
        // endpoint relay-disabled leaves the main mesh endpoint as the sole
        // relay registrant for this id. Owner-control is therefore reachable
        // over its direct / advertised address only; relay-assisted remote
        // owner-control is intentionally unsupported while the control and mesh
        // endpoints share one id.
        let endpoint = Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(secret_key)
            .alpns(vec![ALPN_CONTROL_V1.to_vec()])
            .clear_ip_transports()
            .bind_addr(bind_addr.unwrap_or_else(default_control_bind_addr))?
            .relay_mode(iroh::endpoint::RelayMode::Disabled)
            .bind()
            .await?;
        let token = encode_endpoint_addr_token(&control_endpoint_addr(&endpoint, advertise_addr));
        let shutdown_requested = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let task_endpoint = endpoint.clone();
        let task_shutdown_requested = shutdown_requested.clone();
        let task_shutdown = shutdown.clone();
        let node = self.clone();
        let task = tokio::spawn(Box::pin(async move {
            node.control_accept_loop(task_endpoint, task_shutdown_requested, task_shutdown)
                .await;
        }));
        *self.control_listener.lock().await = Some(ControlListenerLifecycle {
            endpoint,
            token,
            shutdown_requested,
            shutdown,
            task,
        });
        Ok(())
    }

    pub(crate) fn plugin_manager_local_kind(&self) -> crate::plugin::proto::mesh_event::Kind {
        if self.accepting.1.load(std::sync::atomic::Ordering::Acquire) {
            crate::plugin::proto::mesh_event::Kind::LocalAccepting
        } else {
            crate::plugin::proto::mesh_event::Kind::LocalStandby
        }
    }

    pub(crate) async fn broadcast_existing_mesh_snapshot(
        &self,
        plugin_manager: &crate::plugin::PluginManager,
        peers: Vec<PeerInfo>,
    ) {
        let _ = plugin_manager
            .broadcast_mesh_event(
                self.build_mesh_event(self.plugin_manager_local_kind(), None, String::new())
                    .await,
            )
            .await;
        if self.mesh_id().await.is_some() {
            let _ = plugin_manager
                .broadcast_mesh_event(
                    self.build_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::MeshIdUpdated,
                        None,
                        String::new(),
                    )
                    .await,
                )
                .await;
        }
        for peer in peers {
            if let Err(err) = plugin_manager
                .broadcast_mesh_event(
                    self.build_mesh_event(
                        crate::plugin::proto::mesh_event::Kind::PeerUp,
                        Some(peer_info_to_mesh_peer(&peer)),
                        String::new(),
                    )
                    .await,
                )
                .await
            {
                tracing::debug!(
                    "Failed to send existing peer snapshot to plugins for {}: {err}",
                    peer.id.fmt_short()
                );
            }
        }
    }

    #[cfg(test)]
    pub async fn insert_test_peer(&self, peer: PeerInfo) {
        self.state.lock().await.peers.insert(peer.id, peer);
    }

    /// Drop a peer from the local mesh view, simulating churn.
    #[cfg(test)]
    pub async fn remove_test_peer(&self, id: EndpointId) {
        self.state.lock().await.peers.remove(&id);
    }
}
impl Node {
    pub fn id(&self) -> EndpointId {
        self.endpoint.id()
    }

    pub async fn role(&self) -> NodeRole {
        self.role.lock().await.clone()
    }

    #[cfg(test)]
    pub async fn set_role(&self, role: NodeRole) {
        *self.role.lock().await = role;
    }

    pub async fn set_release_attestation_report(
        &self,
        summary: crate::ReleaseAttestationSummary,
        attestation: Option<crate::ReleaseBuildAttestation>,
    ) {
        *self.release_attestation.lock().await = attestation;
        *self.release_attestation_summary.lock().await = summary;
    }

    pub async fn set_models(&self, models: Vec<String>) {
        *self.models.lock().await = models;
    }

    pub async fn models(&self) -> Vec<String> {
        self.models.lock().await.clone()
    }

    pub async fn set_model_source(&self, source: String) {
        *self.model_source.lock().await = Some(source);
        self.refresh_served_model_descriptors().await;
    }

    pub async fn set_serving_models(&self, models: Vec<String>) {
        *self.serving_models.lock().await = models;
        self.refresh_served_model_descriptors().await;
    }

    pub async fn set_served_model_descriptors(&self, descriptors: Vec<ServedModelDescriptor>) {
        let model_names: std::collections::HashSet<_> = descriptors
            .iter()
            .map(|descriptor| descriptor.identity.model_name.clone())
            .collect();
        *self.served_model_descriptors.lock().await = descriptors;
        self.model_runtime_descriptors
            .lock()
            .await
            .retain(|runtime| model_names.contains(&runtime.model_name));
    }

    pub async fn upsert_served_model_descriptor(&self, descriptor: ServedModelDescriptor) {
        let mut descriptors = self.served_model_descriptors.lock().await;
        if let Some(existing) = descriptors
            .iter_mut()
            .find(|existing| existing.identity.model_name == descriptor.identity.model_name)
        {
            *existing = descriptor;
        } else {
            descriptors.push(descriptor);
        }
    }

    pub async fn remove_served_model_descriptor(&self, model_name: &str) {
        self.served_model_descriptors
            .lock()
            .await
            .retain(|descriptor| descriptor.identity.model_name != model_name);
        self.model_runtime_descriptors
            .lock()
            .await
            .retain(|runtime| runtime.model_name != model_name);
    }

    pub async fn set_model_runtime_context_length(
        &self,
        model_name: &str,
        context_length: Option<u32>,
    ) {
        let identity_hash = self
            .served_model_descriptors
            .lock()
            .await
            .iter()
            .find(|descriptor| descriptor.identity.model_name == model_name)
            .and_then(|descriptor| descriptor.identity.identity_hash.clone());
        let mut runtimes = self.model_runtime_descriptors.lock().await;
        if let Some(context_length) = context_length {
            if let Some(runtime) = runtimes
                .iter_mut()
                .find(|runtime| runtime.model_name == model_name)
            {
                runtime.identity_hash = identity_hash.or_else(|| runtime.identity_hash.clone());
                runtime.context_length = Some(context_length);
                runtime.ready = true;
            } else {
                runtimes.push(ModelRuntimeDescriptor {
                    model_name: model_name.to_string(),
                    identity_hash,
                    context_length: Some(context_length),
                    ready: true,
                });
            }
        } else {
            runtimes.retain(|runtime| runtime.model_name != model_name);
        }
    }

    pub async fn local_model_context_length(&self, model_name: &str) -> Option<u32> {
        self.model_runtime_descriptors
            .lock()
            .await
            .iter()
            .find(|runtime| runtime.model_name == model_name)
            .and_then(ModelRuntimeDescriptor::advertised_context_length)
    }

    pub async fn peer_model_context_length(
        &self,
        peer_id: EndpointId,
        model_name: &str,
    ) -> Option<u32> {
        self.state
            .lock()
            .await
            .peers
            .get(&peer_id)
            .and_then(|peer| peer.advertised_context_length(model_name))
    }

    pub(crate) async fn peer_model_throughput_hint(
        &self,
        peer_id: EndpointId,
        model_name: &str,
    ) -> Option<crate::network::metrics::ModelThroughputHint> {
        let state = self.state.lock().await;
        state.peers.get(&peer_id).and_then(|peer| {
            peer.advertised_model_throughput
                .iter()
                .find(|hint| hint.model_name == model_name)
                .cloned()
        })
    }

    pub async fn served_model_descriptors(&self) -> Vec<ServedModelDescriptor> {
        self.served_model_descriptors.lock().await.clone()
    }

    pub async fn all_served_model_descriptors(&self) -> Vec<ServedModelDescriptor> {
        let mut descriptors = self.served_model_descriptors.lock().await.clone();
        let peer_descriptors = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .flat_map(|peer| peer.served_model_descriptors.clone())
                .collect::<Vec<_>>()
        };
        descriptors.extend(peer_descriptors);
        descriptors
    }

    pub async fn all_model_runtime_descriptors(&self) -> Vec<ModelRuntimeDescriptor> {
        let mut runtimes = self.model_runtime_descriptors.lock().await.clone();
        let peer_runtimes = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .flat_map(|peer| peer.served_model_runtime.clone())
                .collect::<Vec<_>>()
        };
        runtimes.extend(peer_runtimes);
        runtimes
    }

    pub async fn serving_models(&self) -> Vec<String> {
        self.serving_models.lock().await.clone()
    }

    pub async fn set_hosted_models(&self, models: Vec<String>) {
        *self.hosted_models.lock().await = models;
    }

    pub async fn hosted_models(&self) -> Vec<String> {
        self.hosted_models.lock().await.clone()
    }

    pub(crate) fn register_runtime_instance_lifecycle(
        &self,
        port: u16,
        lifecycle: Arc<tokio::sync::Mutex<crate::runtime::InstanceLifecycleRecord>>,
    ) {
        self.runtime_instance_lifecycles
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(port, lifecycle);
    }

    pub(crate) fn unregister_runtime_instance_lifecycle(&self, port: u16) {
        self.runtime_instance_lifecycles
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&port);
    }

    pub(crate) async fn begin_runtime_instance_request(
        &self,
        port: u16,
    ) -> Result<Option<crate::runtime::InstanceRequestGuard>, crate::runtime::InstanceAdmissionError>
    {
        let lifecycle = self
            .runtime_instance_lifecycles
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&port)
            .cloned();
        let Some(lifecycle) = lifecycle else {
            return Ok(None);
        };
        let record = lifecycle.lock().await;
        if !record.is_accepting_work() {
            return Err(crate::runtime::InstanceAdmissionError::NotAccepting {
                state: record.state(),
            });
        }
        Ok(Some(crate::runtime::InstanceRequestGuard::new(
            record.in_flight_tracker(),
        )))
    }

    pub(crate) async fn refresh_served_model_descriptors(&self) {
        let serving_models = self.serving_models.lock().await.clone();
        let existing_by_name: HashMap<_, _> = self
            .served_model_descriptors
            .lock()
            .await
            .iter()
            .map(|descriptor| (descriptor.identity.model_name.clone(), descriptor.clone()))
            .collect();
        let mut descriptors = if let Some(primary_model_name) = serving_models.first() {
            let model_source = self.model_source.lock().await.clone();
            let primary_model_path = crate::models::find_model_path(primary_model_name);
            infer_served_model_descriptors(
                primary_model_name,
                &serving_models,
                model_source.as_deref(),
                Some(primary_model_path.as_path()),
            )
        } else {
            Vec::new()
        };
        for descriptor in &mut descriptors {
            if descriptor.metadata.is_none() {
                descriptor.metadata =
                    crate::models::served_model_metadata_for_model(&descriptor.identity.model_name);
            }
            if let Some(existing) = existing_by_name.get(&descriptor.identity.model_name) {
                descriptor.capabilities = existing.capabilities;
                descriptor.capabilities_known = existing.capabilities_known;
                if existing.topology.is_some() {
                    descriptor.topology = existing.topology.clone();
                }
                if existing.metadata.is_some() {
                    descriptor.metadata = existing.metadata.clone();
                }
            }
        }
        self.set_served_model_descriptors(descriptors).await;
    }

    /// Set the operator-facing display name for this node.
    pub async fn set_display_name(&self, name: String) {
        *self.display_name.lock().await = Some(name);
    }

    pub async fn set_plugin_manager(&self, plugin_manager: crate::plugin::PluginManager) {
        let peers = {
            let state = self.state.lock().await;
            state.peers.values().cloned().collect::<Vec<_>>()
        };
        *self.plugin_manager.lock().await = Some(plugin_manager.clone());
        self.broadcast_existing_mesh_snapshot(&plugin_manager, peers)
            .await;
    }

    pub async fn plugin_manager(&self) -> Option<crate::plugin::PluginManager> {
        self.plugin_manager.lock().await.clone()
    }

    pub fn start_plugin_channel_forwarder(
        &self,
        mut rx: tokio::sync::mpsc::Receiver<crate::plugin::PluginMeshEvent>,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if let Err(err) = node.forward_plugin_event(event).await {
                    tracing::debug!("Plugin mesh forward failed: {err}");
                }
            }
        });
    }

    pub(crate) async fn emit_plugin_mesh_event(
        &self,
        kind: crate::plugin::proto::mesh_event::Kind,
        peer: Option<&PeerInfo>,
        detail_json: String,
    ) {
        let plugin_manager = self.plugin_manager.lock().await.clone();
        if let Some(plugin_manager) = plugin_manager
            && let Err(err) = plugin_manager
                .broadcast_mesh_event(
                    self.build_mesh_event(kind, peer.map(peer_info_to_mesh_peer), detail_json)
                        .await,
                )
                .await
        {
            tracing::debug!(
                "Failed to deliver plugin mesh event {:?} for {}: {err}",
                kind,
                peer.map(|p| p.id.fmt_short().to_string())
                    .unwrap_or_else(|| self.endpoint.id().fmt_short().to_string())
            );
        }
    }

    pub(crate) async fn update_peer_rtt(&self, id: EndpointId, rtt_ms: u32) {
        // 0ms is not a valid network RTT — it indicates a measurement artifact
        // (e.g. local buffer time before the actual network round-trip).
        if rtt_ms == 0 {
            return;
        }
        let updated_peer = {
            let mut state = self.state.lock().await;
            if let Some(peer) = state.peers.get_mut(&id) {
                let prev = peer.rtt_ms;
                // Only accept equal-or-lower RTT for planner preference and display.
                // Gossip round-trip timing can inflate the value when routed via
                // relay, overwriting a better path measurement. Liveness remains
                // responsible for detecting an unreachable peer.
                if prev.is_some_and(|p| rtt_ms > p) {
                    // Store display_rtt regardless (for UI refresh), but don't update best RTT.
                    peer.display_rtt = Some(DirectLatencyObservation {
                        rtt_ms,
                        observed_at: std::time::Instant::now(),
                    });
                    return;
                }
                peer.rtt_ms = Some(rtt_ms);
                peer.display_rtt = Some(DirectLatencyObservation {
                    rtt_ms,
                    observed_at: std::time::Instant::now(),
                });
                Some(peer.clone())
            } else {
                None
            }
        };
        if let Some(peer) = updated_peer {
            tracing::info!("Peer {} RTT: {}ms", id.fmt_short(), rtt_ms);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::PeerUpdated,
                Some(&peer),
                String::new(),
            )
            .await;
        }
    }

    pub(crate) async fn update_peer_selected_path(
        &self,
        id: EndpointId,
        observation: SelectedPathObservation,
    ) {
        let direct_rtt_ms = if observation.path_type == "direct" {
            observation.rtt_ms
        } else {
            None
        };
        {
            let mut state = self.state.lock().await;
            if let Some(peer) = state.peers.get_mut(&id) {
                peer.selected_path = Some(observation);
            }
        }
        if let Some(rtt_ms) = direct_rtt_ms {
            self.update_peer_rtt(id, rtt_ms).await;
        }
    }

    /// Re-gossip our state to all connected peers.
    /// Call after changing assigned/hosted state, role, or configured models.
    pub async fn regossip(&self) {
        let conns: Vec<(EndpointId, Connection)> = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .map(|(id, c)| (*id, c.clone()))
                .collect()
        };
        for (peer_id, conn) in conns {
            let node = self.clone();
            tokio::spawn(async move {
                if let Err(e) = node.initiate_gossip(conn, peer_id).await {
                    tracing::debug!("Regossip to {} failed: {e}", peer_id.fmt_short());
                }
            });
        }
    }

    /// Gossip with one connected peer to update routing table.
    /// Used by: (1) passive nodes' periodic 60s heartbeat, (2) background
    /// refresh on tunnel failure so future requests have fresh routing.
    pub async fn gossip_one_peer(&self) {
        let conn = {
            let state = self.state.lock().await;
            state
                .connections
                .iter()
                .next()
                .map(|(id, c)| (*id, c.clone()))
        };
        if let Some((peer_id, conn)) = conn {
            let _ = self.initiate_gossip_inner(conn, peer_id, false).await;
        }
    }

    pub async fn is_llama_ready(&self) -> bool {
        *self.llama_ready.lock().await
    }

    pub async fn mesh_id(&self) -> Option<String> {
        if let Some(state) = self.requirement_mesh_state.lock().await.clone() {
            return Some(state.mesh_id);
        }
        self.mesh_id.lock().await.clone()
    }

    pub async fn first_joined_mesh_ts(&self) -> Option<u64> {
        *self.first_joined_mesh_ts.lock().await
    }

    pub async fn set_first_joined_mesh_ts_if_absent(&self, ts: u64) -> bool {
        let mut current = self.first_joined_mesh_ts.lock().await;
        if current.is_none() {
            *current = Some(ts);
            true
        } else {
            false
        }
    }

    /// Set the mesh identity. If None was set, adopts the given ID (from gossip).
    /// If already set, ignores (originator's ID wins).
    pub async fn set_mesh_id(&self, id: String) {
        if let Some(state) = self.requirement_mesh_state.lock().await.clone()
            && state.mesh_id != id
        {
            tracing::warn!(
                "ignoring conflicting mesh ID '{}' for requirement-aware mesh {}",
                id,
                state.mesh_id
            );
            return;
        }
        let mut current = self.mesh_id.lock().await;
        if current.is_none() {
            *current = Some(id);
            drop(current);
            self.emit_plugin_mesh_event(
                crate::plugin::proto::mesh_event::Kind::MeshIdUpdated,
                None,
                String::new(),
            )
            .await;
        }
    }

    /// Set mesh ID unconditionally (for originator).
    pub async fn set_mesh_id_force(&self, id: String) {
        if let Some(state) = self.requirement_mesh_state.lock().await.clone() {
            assert_eq!(
                state.mesh_id, id,
                "requirement-aware mesh state must keep mesh ID aligned with policy hash"
            );
        }
        *self.mesh_id.lock().await = Some(id);
        self.emit_plugin_mesh_event(
            crate::plugin::proto::mesh_event::Kind::MeshIdUpdated,
            None,
            String::new(),
        )
        .await;
    }

    pub async fn set_available_models(&self, models: Vec<String>) {
        *self.available_models.lock().await = models;
    }

    pub async fn available_models(&self) -> Vec<String> {
        self.available_models.lock().await.clone()
    }

    /// Record a request for a model — updates the demand map.
    /// Called from API proxy on every request (including misses for unserved models).
    /// Uses std::sync::Mutex (not tokio) so it can be called from sync context too.
    pub fn record_request(&self, model: &str) {
        // "auto" is a routing directive, not a real model — don't pollute demand
        if model == "auto" || model.is_empty() {
            return;
        }
        let model_ref = canonical_demand_model_ref(model);
        let mut demand = self.model_demand.lock().unwrap();
        let entry = demand.entry(model_ref).or_default();
        entry.last_active = now_secs();
        entry.request_count += 1;
    }

    /// Get the current demand map (for gossip and assignment decisions).
    pub fn get_demand(&self) -> HashMap<String, ModelDemand> {
        self.model_demand.lock().unwrap().clone()
    }

    /// Merge incoming demand from gossip into our local map.
    pub fn merge_remote_demand(&self, remote: &HashMap<String, ModelDemand>) {
        let mut demand = self.model_demand.lock().unwrap();
        merge_demand(&mut demand, remote);
    }

    /// Remove demand entries that have expired (past TTL and not pinned).
    /// Call periodically to prevent unbounded map growth.
    pub async fn gc_demand(&self) {
        let now = now_secs();
        let pinned = self.pinned_requested_models().await;

        let mut demand = self.model_demand.lock().unwrap();
        demand.retain(|model, d| {
            pinned.contains(model) || now.saturating_sub(d.last_active) < DEMAND_TTL_SECS
        });
    }

    /// Get active demand entries (within TTL or pinned by a live node).
    /// This replaces mesh_wanted_models().
    pub async fn active_demand(&self) -> HashMap<String, ModelDemand> {
        let now = now_secs();
        let demand = self.model_demand.lock().unwrap().clone();
        let pinned = self.pinned_requested_models().await;

        demand
            .into_iter()
            .filter(|(model, d)| {
                pinned.contains(model) || now.saturating_sub(d.last_active) < DEMAND_TTL_SECS
            })
            .collect()
    }

    async fn pinned_requested_models(&self) -> std::collections::HashSet<String> {
        let my_requested = self.requested_models.lock().await;
        let peers = self.state.lock().await;
        let mut pinned: std::collections::HashSet<String> = my_requested.iter().cloned().collect();
        for peer in peers.peers.values() {
            pinned.extend(peer.requested_models.iter().cloned());
        }
        pinned
    }

    pub async fn set_requested_models(&self, models: Vec<String>) {
        let models = models
            .into_iter()
            .map(|model| canonical_demand_model_ref(&model))
            .collect::<Vec<_>>();
        // Seed demand entries for --model declarations
        {
            let mut demand = self.model_demand.lock().unwrap();
            let now = now_secs();
            for m in &models {
                let entry = demand.entry(m.clone()).or_default();
                entry.last_active = entry.last_active.max(now);
            }
        }
        *self.requested_models.lock().await = models;
    }

    pub async fn requested_models(&self) -> Vec<String> {
        self.requested_models.lock().await.clone()
    }

    pub async fn set_explicit_model_interests(&self, mut model_refs: Vec<String>) {
        model_refs.retain(|model_ref| !model_ref.trim().is_empty());
        model_refs.sort();
        model_refs.dedup();
        *self.explicit_model_interests.lock().await = model_refs;
    }

    pub async fn explicit_model_interests(&self) -> Vec<String> {
        self.explicit_model_interests.lock().await.clone()
    }
}
