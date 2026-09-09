//! Mesh management API — read-only dashboard on port 3131 (default).
//!
//! Endpoints:
//!   GET  /health         — lightweight liveness plus mesh/serving summary (JSON, always 200 when answering)
//!   GET  /api/status    — live mesh state plus local-only routing metrics (JSON)
//!   GET  /api/models    — mesh model inventory plus local-only routing metrics (JSON)
//!   GET  /api/search    — catalog or Hugging Face model search with the same JSON payload as `mesh-llm models search --json`
//!   GET  /api/model-interests — local explicit-interest readback (JSON)
//!   POST /api/model-interests — register local explicit interest for a canonical model ref
//!   DELETE /api/model-interests/{model_ref} — clear local explicit interest
//!   GET  /api/model-targets — ranked model targets from explicit interest and demand
//!   GET  /api/diagnostics/split-readiness — split peer eligibility and operator guidance
//!   GET  /api/runtime   — local model state (JSON)
//!   GET  /api/runtime/llama — local llama.cpp runtime metrics + slots snapshots (JSON)
//!   GET  /api/runtime/events — SSE stream of llama.cpp runtime metrics + slots snapshots
//!   GET  /api/runtime/endpoints — registered plugin endpoint state (JSON, trusted-local only)
//!   GET  /api/runtime/processes — local inference process state (JSON)
//!   GET  /api/runtime/stages — backend-neutral staged-serving state (JSON)
//!   GET  /api/runtime/config-schema — merged built-in and installed-plugin config schema (JSON)
//!   GET  /api/runtime/config-control-state — local-only runtime config availability/options overlay (JSON)
//!   POST /api/runtime/pick-directory — open the trusted-local host directory picker
//!   GET  /api/runtime/control-bootstrap — local-only owner-control bootstrap policy (JSON)
//!   POST /api/runtime/control/get-config — run local owner-control get-config against an explicit endpoint
//!   POST /api/runtime/control/refresh-inventory — run local owner-control refresh-inventory against an explicit endpoint
//!   POST /api/runtime/control/apply-config — run local owner-control apply-config against an explicit endpoint
//!   POST /api/runtime/control/kv-cache — run authenticated L3 KV cache operations against owned nodes
//!   POST /api/runtime/models — load a local model
//!   DELETE /api/runtime/models/{model} — unload a local model
//!   DELETE /api/runtime/instances/{instance_id} — unload one local runtime instance
//!   GET  /api/events    — SSE stream of status updates
//!   GET  /api/discover  — browse Nostr meshes or LAN mDNS advertisements
//!   POST /api/discovery/lan-details — invite-token proof-gated LAN detail
//!   POST /api/chat      — proxy to chat completions API
//!   POST /api/responses — proxy to responses API
//!   POST /api/objects   — upload a request-scoped media object
//!   POST /mcp           — streamable HTTP MCP endpoint for all mesh plugin tools
//!   GET  /              — embedded web dashboard
//!
//! The dashboard is mostly read-only — shows status, topology, and models.
//! Local model load/unload is exposed for operator control.
//!
//! Broad runtime reads should stay behind `runtime_data` helpers so the API
//! layer keeps using stable collector-backed views instead of fresh fan-in.
//!
//! `routing_metrics`, `routing_metrics.local_node`, `routing_metrics.pressure`,
//! and `/api/models` per-model `routing_metrics.targets` are measured on the
//! current node only; not mesh-wide aggregates.

mod access;
mod assets;
mod http;
mod management_lifecycle;
mod model_target_capacity;
mod model_targets;
pub(crate) mod routes;
mod server;
mod split_readiness;
mod state;
pub(crate) mod status;

pub(crate) use self::server::start_with_listener;
#[cfg(test)]
pub(crate) use self::server::{handle_request, is_ui_only_route};
pub use self::state::{
    ControlBootstrapPayload, LocalModelInterest, MeshApi, OpenAiGuardrailModeUpdateResponse,
    PublicationState, RuntimeControlRequest, RuntimeLoadResponse, RuntimeModelPayload,
    RuntimeProcessPayload, RuntimeUnloadResponse,
};
pub(crate) use self::status::classify_runtime_error;

use self::state::ApiInner;
use self::status::{
    IntentSummary, LifecycleInstancePayload, LoggingStatusPayload, MeshModelPayload,
    OpenAiGuardrailsPayload, RuntimeCapabilityFlags, RuntimeLlamaPayload, RuntimeProcessesPayload,
    RuntimeStatusPayload, StatusPayload, build_runtime_processes_payload,
    build_runtime_stage_payloads, build_runtime_status_payload, derive_daemon_state,
    runtime_stage_state_label,
};
use crate::mesh;
use crate::models::append_external_inference_models;
use crate::network::{affinity, nostr};
use crate::plugin;
use crate::runtime_data;
use mesh_llm_node::serving::{
    DevicePolicy as NodeDevicePolicy, LoadModelRequest, ServedModel, ServingController,
    ServingError, ServingFuture, ServingModelState, ServingStatus, UnloadModelRequest,
};
use mesh_llm_types::models::capabilities::merge_name_signals;
use std::sync::Arc;
use tokio::sync::Mutex;

#[cfg(test)]
use self::http::http_body_text;
#[cfg(test)]
use self::status::{NodeState, WakeableNode, WakeableNodeState, build_gpus};
#[cfg(test)]
use crate::inference::election;
#[cfg(test)]
use crate::network::proxy;
#[cfg(test)]
use crate::runtime::wakeable::{WakeableInventoryEntry, WakeableState};

const MESH_LLM_BUILD_VERSION: &str = crate::BUILD_VERSION;

async fn external_inference_models(plugin_manager: &plugin::PluginManager) -> Vec<String> {
    plugin_manager
        .inference_models()
        .await
        .unwrap_or_else(|error| {
            tracing::debug!(%error, "failed to collect plugin inference models for status");
            Vec::new()
        })
}

#[cfg(test)]
#[derive(Debug, Default, PartialEq)]
pub(crate) struct HttpRouteStats {
    node_count: usize,
    active_nodes: Vec<String>,
    mesh_vram_gb: f64,
}

#[cfg(test)]
pub(crate) fn http_route_stats(
    model_name: &str,
    peers: &[mesh::PeerInfo],
    my_hosted_models: &[String],
    my_hostname: Option<&str>,
    my_vram_gb: f64,
) -> HttpRouteStats {
    let mut active_nodes = Vec::new();
    let mut node_count = 0usize;
    let mut mesh_vram_gb = 0.0;

    if my_hosted_models.iter().any(|hosted| hosted == model_name) {
        node_count += 1;
        mesh_vram_gb += my_vram_gb;
        active_nodes.push(
            my_hostname
                .filter(|hostname| !hostname.trim().is_empty())
                .unwrap_or("This node")
                .to_string(),
        );
    }

    for peer in peers {
        if !peer.routes_http_model(model_name) {
            continue;
        }
        node_count += 1;
        mesh_vram_gb += peer.vram_bytes as f64 / 1e9;
        active_nodes.push(
            peer.hostname
                .clone()
                .filter(|hostname| !hostname.trim().is_empty())
                .unwrap_or_else(|| peer.id.fmt_short().to_string()),
        );
    }

    active_nodes.sort();
    active_nodes.dedup();

    HttpRouteStats {
        node_count,
        active_nodes,
        mesh_vram_gb,
    }
}

pub struct MeshApiConfig {
    pub(crate) node: mesh::Node,
    pub(crate) model_name: String,
    pub(crate) api_port: u16,
    pub(crate) model_size_bytes: u64,
    pub(crate) owner_key_path: Option<std::path::PathBuf>,
    pub(crate) plugin_manager: plugin::PluginManager,
    pub(crate) affinity_router: affinity::AffinityRouter,
    pub(crate) runtime_data_collector: runtime_data::RuntimeDataCollector,
    pub(crate) runtime_data_producer: runtime_data::RuntimeDataProducer,
}

impl MeshApi {
    pub fn new(config: MeshApiConfig) -> Self {
        let MeshApiConfig {
            node,
            model_name,
            api_port,
            model_size_bytes,
            owner_key_path,
            plugin_manager,
            affinity_router,
            runtime_data_collector,
            runtime_data_producer,
        } = config;

        runtime_data_producer.publish_runtime_status(|runtime_status| {
            if runtime_status.primary_model.as_deref() == Some(model_name.as_str()) {
                return false;
            }
            runtime_status.primary_model = Some(model_name.clone());
            true
        });
        let mcp_http = plugin::mcp::PluginMcpHttpEndpoint::new(plugin_manager.clone());
        let initial_runtime_data_views = runtime_data::collect_views(&runtime_data_collector);
        let _ = (
            initial_runtime_data_views
                .runtime_status
                .primary_model
                .as_ref(),
            initial_runtime_data_views
                .runtime_status
                .primary_backend
                .as_ref(),
            initial_runtime_data_views.runtime_status.is_host,
            initial_runtime_data_views.runtime_status.is_client,
            initial_runtime_data_views.runtime_status.llama_ready,
            initial_runtime_data_views.runtime_status.llama_port,
            initial_runtime_data_views
                .runtime_status
                .local_processes
                .len(),
            initial_runtime_data_views.local_instances.instances.len(),
            initial_runtime_data_views.plugin_data.entries.len(),
            initial_runtime_data_views.plugin_endpoints.entries.len(),
            runtime_data_producer.scope(),
            runtime_data_producer.has_plugin_data_key(),
            runtime_data_producer.has_plugin_endpoint_key(),
            runtime_data_producer.initial_process_count(),
        );
        MeshApi {
            capture_node: node.clone(),
            inner: Arc::new(Mutex::new(ApiInner {
                node,
                plugin_manager,
                mcp_http,
                affinity_router,
                runtime_data_collector,
                runtime_data_producer,
                headless: false,
                is_host: false,
                is_client: false,
                llama_ready: false,
                listeners_ready: false,
                llama_port: None,
                model_name,
                primary_backend: None,
                openai_guardrails: None,
                draft_name: None,
                api_port,
                model_size_bytes,
                mesh_name: None,
                mesh_region: None,
                mesh_max_clients: None,
                latest_version: None,
                nostr_relays: nostr::DEFAULT_RELAYS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                mesh_discovery_mode: crate::network::discovery::MeshDiscoveryMode::Nostr,
                nostr_discovery: false,
                publication_state: state::PublicationState::Private,
                runtime_control: None,
                control_bootstrap: state::ControlBootstrapPayload::default(),
                owner_key_path,
                local_processes: Vec::new(),
                sse_clients: Vec::new(),
                model_interests: std::collections::HashMap::new(),
                wakeable_inventory: crate::runtime::wakeable::WakeableInventory::default(),
            })),
        }
    }

    pub async fn node(&self) -> mesh::Node {
        self.inner.lock().await.node.clone()
    }

    pub(super) async fn model_interests(&self) -> Vec<LocalModelInterest> {
        let mut interests = {
            let inner = self.inner.lock().await;
            inner
                .model_interests
                .values()
                .cloned()
                .collect::<Vec<LocalModelInterest>>()
        };
        interests.sort_by(|left, right| {
            right
                .updated_at_unix
                .cmp(&left.updated_at_unix)
                .then_with(|| left.model_ref.cmp(&right.model_ref))
        });
        interests
    }

    pub(super) async fn upsert_model_interest(
        &self,
        model_ref: String,
        submission_source: Option<String>,
    ) -> (LocalModelInterest, bool) {
        let now = current_unix_secs();
        let (interest, created, model_refs) = {
            let mut inner = self.inner.lock().await;
            let (interest, created) = match inner.model_interests.entry(model_ref.clone()) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get().clone();
                    let updated = LocalModelInterest {
                        model_ref,
                        submission_source: submission_source.or(existing.submission_source),
                        created_at_unix: existing.created_at_unix,
                        updated_at_unix: now,
                    };
                    entry.insert(updated.clone());
                    (updated, false)
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let created = LocalModelInterest {
                        model_ref,
                        submission_source,
                        created_at_unix: now,
                        updated_at_unix: now,
                    };
                    entry.insert(created.clone());
                    (created, true)
                }
            };
            let mut model_refs = inner.model_interests.keys().cloned().collect::<Vec<_>>();
            model_refs.sort();
            (interest, created, model_refs)
        };
        self.sync_node_model_interests(model_refs).await;
        (interest, created)
    }

    pub(super) async fn remove_model_interest(&self, model_ref: &str) -> bool {
        let (removed, model_refs) = {
            let mut inner = self.inner.lock().await;
            let removed = inner.model_interests.remove(model_ref).is_some();
            let mut model_refs = inner.model_interests.keys().cloned().collect::<Vec<_>>();
            model_refs.sort();
            (removed, model_refs)
        };
        if removed {
            self.sync_node_model_interests(model_refs).await;
        }
        removed
    }

    async fn sync_node_model_interests(&self, model_refs: Vec<String>) {
        let node = { self.inner.lock().await.node.clone() };
        node.set_explicit_model_interests(model_refs).await;
        self.push_status().await;
    }

    pub async fn set_primary_backend(&self, backend: String) {
        let mut inner = self.inner.lock().await;
        inner.primary_backend = Some(backend.clone());
        inner
            .runtime_data_producer
            .publish_runtime_status(|runtime_status| {
                if runtime_status.primary_backend.as_deref() == Some(backend.as_str()) {
                    return false;
                }
                runtime_status.primary_backend = Some(backend.clone());
                true
            });
    }

    pub async fn set_openai_guardrails(&self, openai_guardrails: Option<OpenAiGuardrailsPayload>) {
        self.inner.lock().await.openai_guardrails = openai_guardrails;
    }

    pub async fn set_draft_name(&self, name: String) {
        self.inner.lock().await.draft_name = Some(name);
    }

    #[expect(
        dead_code,
        reason = "retained for embedded callers that toggle client presentation state"
    )]
    pub async fn set_client(&self, is_client: bool) {
        let mut inner = self.inner.lock().await;
        inner.is_client = is_client;
        inner
            .runtime_data_producer
            .publish_runtime_status(|runtime_status| {
                if runtime_status.is_client == is_client {
                    return false;
                }
                runtime_status.is_client = is_client;
                true
            });
    }

    pub async fn set_mesh_publication_metadata(
        &self,
        name: Option<String>,
        region: Option<String>,
        max_clients: Option<usize>,
    ) {
        let mut inner = self.inner.lock().await;
        inner.mesh_name = name;
        inner.mesh_region = region;
        inner.mesh_max_clients = max_clients;
    }

    pub async fn set_nostr_relays(&self, relays: Vec<String>) {
        self.inner.lock().await.nostr_relays = relays;
    }

    pub async fn set_mesh_discovery_mode(
        &self,
        mode: crate::network::discovery::MeshDiscoveryMode,
    ) {
        self.inner.lock().await.mesh_discovery_mode = mode;
    }

    pub async fn set_nostr_discovery(&self, v: bool) {
        self.inner.lock().await.nostr_discovery = v;
    }

    pub async fn set_publication_state(&self, state: state::PublicationState) {
        {
            let mut inner = self.inner.lock().await;
            inner.publication_state = state;
        }
        self.push_status().await;
    }

    #[cfg(test)]
    pub(crate) async fn publication_state(&self) -> state::PublicationState {
        self.inner.lock().await.publication_state
    }

    pub(crate) async fn runtime_data_producer(&self) -> runtime_data::RuntimeDataProducer {
        self.inner.lock().await.runtime_data_producer.clone()
    }

    pub async fn set_runtime_control(
        &self,
        tx: tokio::sync::mpsc::UnboundedSender<RuntimeControlRequest>,
    ) {
        self.inner.lock().await.runtime_control = Some(tx);
    }

    pub async fn control_bootstrap(&self) -> ControlBootstrapPayload {
        self.inner.lock().await.control_bootstrap.clone()
    }

    pub async fn set_control_bootstrap(&self, control_bootstrap: ControlBootstrapPayload) {
        self.inner.lock().await.control_bootstrap = control_bootstrap;
    }

    pub(crate) async fn owner_key_path(&self) -> Option<std::path::PathBuf> {
        self.inner.lock().await.owner_key_path.clone()
    }

    #[cfg(test)]
    pub(crate) async fn set_owner_key_path(&self, owner_key_path: Option<std::path::PathBuf>) {
        self.inner.lock().await.owner_key_path = owner_key_path;
    }

    pub(crate) async fn status_snapshot_string(&self) -> String {
        let status = self.status().await;
        match serde_json::to_string_pretty(&status) {
            Ok(json) => json,
            Err(err) => {
                tracing::warn!("failed to serialize local status snapshot: {err}");
                format!(
                    "{{\n  \"error\": \"status snapshot unavailable\",\n  \"detail\": {:?}\n}}",
                    err.to_string()
                )
            }
        }
    }

    pub async fn upsert_local_process(&self, process: RuntimeProcessPayload) {
        {
            let mut inner = self.inner.lock().await;
            inner.local_processes.retain(|p| {
                runtime_process_payload_identity(p) != runtime_process_payload_identity(&process)
            });
            inner.local_processes.push(process.clone());
            inner
                .runtime_data_producer
                .publish_local_processes(|local_processes| {
                    runtime_data::upsert_runtime_process_snapshot(
                        local_processes,
                        runtime_data::RuntimeProcessSnapshot::from_payload(&process),
                    )
                });
        }
    }

    pub async fn remove_local_process(&self, target: &str) {
        {
            let mut inner = self.inner.lock().await;
            let has_instance_match = inner
                .local_processes
                .iter()
                .any(|process| process.instance_id.as_deref() == Some(target));
            inner.local_processes.retain(|process| {
                if has_instance_match {
                    process.instance_id.as_deref() != Some(target)
                } else {
                    process.name != target
                }
            });
            inner
                .runtime_data_producer
                .publish_local_processes(|local_processes| {
                    runtime_data::remove_runtime_process_snapshot(local_processes, target)
                });
        }
    }

    pub async fn update(&self, is_host: bool, llama_ready: bool) {
        {
            let mut inner = self.inner.lock().await;
            inner.is_host = is_host;
            inner.llama_ready = llama_ready;
            inner
                .runtime_data_producer
                .publish_runtime_status(|runtime_status| {
                    let mut changed = false;
                    if runtime_status.is_host != is_host {
                        runtime_status.is_host = is_host;
                        changed = true;
                    }
                    if runtime_status.llama_ready != llama_ready {
                        runtime_status.llama_ready = llama_ready;
                        changed = true;
                    }
                    changed
                });
        }
    }

    pub async fn set_llama_port(&self, port: Option<u16>) {
        let mut inner = self.inner.lock().await;
        inner.llama_port = port;
        inner
            .runtime_data_producer
            .publish_runtime_status(|runtime_status| {
                if runtime_status.llama_port == port {
                    return false;
                }
                runtime_status.llama_port = port;
                true
            });
    }

    pub async fn set_headless(&self, headless: bool) {
        self.inner.lock().await.headless = headless;
    }

    pub(super) async fn is_headless(&self) -> bool {
        self.inner.lock().await.headless
    }

    async fn runtime_status(&self) -> RuntimeStatusPayload {
        let (runtime_status, openai_guardrails) = {
            let inner = self.inner.lock().await;
            (
                inner.runtime_data_collector.runtime_status_snapshot(),
                inner.openai_guardrails.clone(),
            )
        };
        build_runtime_status_payload(
            runtime_status.primary_model.as_deref().unwrap_or_default(),
            runtime_status.primary_backend,
            openai_guardrails,
            runtime_status.is_host,
            runtime_status.llama_ready,
            runtime_status.llama_port,
            runtime_data::runtime_process_payloads(&runtime_status.local_processes),
        )
    }

    async fn runtime_processes(&self) -> RuntimeProcessesPayload {
        let runtime_processes = self
            .inner
            .lock()
            .await
            .runtime_data_collector
            .runtime_processes_snapshot();
        build_runtime_processes_payload(runtime_data::runtime_process_payloads(&runtime_processes))
    }

    async fn runtime_stages(&self) -> serde_json::Value {
        let node = self.inner.lock().await.node.clone();
        node.refresh_stage_runtime_statuses(std::time::Duration::from_secs(2))
            .await;
        let topologies = node.stage_topologies().await;
        let statuses = node.stage_runtime_statuses().await;
        let stage_statuses = statuses
            .iter()
            .map(|status| {
                serde_json::json!({
                    "topology_id": status.topology_id.clone(),
                    "run_id": status.run_id.clone(),
                    "model_id": status.model_id.clone(),
                    "backend": status.backend.clone(),
                    "package_ref": status.package_ref.clone(),
                    "manifest_sha256": status.manifest_sha256.clone(),
                    "source_model_path": status.source_model_path.clone(),
                    "source_model_sha256": status.source_model_sha256.clone(),
                    "source_model_bytes": status.source_model_bytes,
                    "materialized_path": status.materialized_path.clone(),
                    "materialized_bytes": status
                        .materialized_path
                        .as_deref()
                        .and_then(|path| std::fs::metadata(path).ok())
                        .filter(|metadata| metadata.is_file())
                        .map(|metadata| metadata.len()),
                    "materialized_pinned": status.materialized_pinned,
                    "projector_path": status.projector_path.clone(),
                    "multimodal": status.projector_path.is_some(),
                    "stage_id": status.stage_id.clone(),
                    "stage_index": status.stage_index,
                    "node_id": status.node_id.map(|id| id.to_string()),
                    "layer_start": status.layer_start,
                    "layer_end": status.layer_end,
                    "state": runtime_stage_state_label(status.state),
                    "bind_addr": status.bind_addr.clone(),
                    "input_activation_boundary": status.input_activation_boundary,
                    "output_activation_boundary": status.output_activation_boundary,
                    "selected_device": status.selected_device.as_ref().map(|device| {
                        serde_json::json!({
                            "backend_device": device.backend_device,
                            "stable_id": device.stable_id,
                            "index": device.index,
                            "vram_bytes": device.vram_bytes,
                        })
                    }),
                    "ctx_size": status.ctx_size,
                    "lane_count": status.lane_count,
                    "error": status.error.clone(),
                    "shutdown_generation": status.shutdown_generation,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "stages": stage_statuses.clone(),
            "topologies": topologies.into_iter().map(|topology| {
                serde_json::json!({
                    "topology_id": topology.topology_id,
                    "run_id": topology.run_id,
                    "model_id": topology.model_id,
                    "package_ref": topology.package_ref,
                    "manifest_sha256": topology.manifest_sha256,
                    "stages": topology.stages.into_iter().map(|stage| {
                        serde_json::json!({
                            "stage_id": stage.stage_id,
                            "stage_index": stage.stage_index,
                            "node_id": stage.node_id.to_string(),
                            "layer_start": stage.layer_start,
                            "layer_end": stage.layer_end,
                            "endpoint": {
                                "bind_addr": stage.endpoint.bind_addr,
                            },
                        })
                    }).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>(),
            "statuses": stage_statuses,
        })
    }

    async fn runtime_llama(&self) -> RuntimeLlamaPayload {
        let (runtime_llama, runtime_llama_by_instance) = {
            let inner = self.inner.lock().await;
            (
                inner.runtime_data_collector.runtime_llama_snapshot(),
                inner
                    .runtime_data_collector
                    .runtime_llama_snapshots_by_instance(),
            )
        };
        status::build_runtime_llama_payload(runtime_llama, runtime_llama_by_instance)
    }

    async fn runtime_endpoints(&self) -> anyhow::Result<Vec<plugin::PluginEndpointSummary>> {
        let plugin_manager = self.inner.lock().await.plugin_manager.clone();
        plugin_manager.endpoints().await
    }

    async fn plugins(&self) -> Vec<plugin::PluginSummary> {
        let plugin_manager = self.inner.lock().await.plugin_manager.clone();
        plugin_manager.list().await
    }

    async fn plugin_capability_providers(
        &self,
    ) -> anyhow::Result<Vec<plugin::PluginCapabilityProvider>> {
        let plugin_manager = self.inner.lock().await.plugin_manager.clone();
        plugin_manager.capability_providers().await
    }

    async fn plugin_provider_for_capability(
        &self,
        capability: &str,
    ) -> anyhow::Result<Option<plugin::PluginCapabilityProvider>> {
        let plugin_manager = self.inner.lock().await.plugin_manager.clone();
        plugin_manager.provider_for_capability(capability).await
    }

    async fn local_inventory_snapshot(&self) -> crate::models::LocalModelInventorySnapshot {
        let runtime_data_collector = self.inner.lock().await.runtime_data_collector.clone();
        runtime_data_collector
            .coalesce_local_inventory_scan(|| {
                crate::models::scan_local_inventory_snapshot_with_progress(|_| {})
            })
            .await
            .unwrap_or_else(|_| runtime_data_collector.local_inventory_snapshot())
    }

    async fn mesh_models(&self) -> Vec<MeshModelPayload> {
        let (runtime_data_collector, node, my_vram_gb, fallback_model_name, model_size_bytes) = {
            let inner = self.inner.lock().await;
            (
                inner.runtime_data_collector.clone(),
                inner.node.clone(),
                inner.node.vram_bytes() as f64 / 1e9,
                inner.model_name.clone(),
                inner.model_size_bytes,
            )
        };

        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let runtime_status = runtime_data_collector.runtime_status_snapshot();
        let model_name = runtime_status.primary_model.unwrap_or(fallback_model_name);

        let target_lookup = self.model_target_lookup().await;
        let mut models = runtime_data::mesh_models(runtime_data_collector.build_model_view(
            runtime_data::ModelViewInput {
                peers: node.peers().await,
                catalog: node.mesh_catalog_entries().await,
                served_models: node.models_being_served().await,
                active_demand: node.active_demand().await,
                my_serving_models: node.serving_models().await,
                my_hosted_models: node.hosted_models().await,
                local_inventory: self.local_inventory_snapshot().await,
                node_hostname: node.hostname.clone(),
                my_vram_gb,
                model_name,
                model_size_bytes,
                now_unix_secs: now_ts,
            },
        ));
        for model in &mut models {
            let target = target_lookup
                .target_by_model_name(&model.name)
                .or_else(|| target_lookup.target_by_model_ref(&model.name));
            if let Some(target) = target {
                model.target_rank = Some(target.rank);
                model.explicit_interest_count = Some(target.explicit_interest_count);
                model.wanted = Some(target.wanted);
            }
        }
        models
    }

    #[cfg(test)]
    fn derive_local_node_state(
        is_client: bool,
        effective_is_host: bool,
        effective_llama_ready: bool,
        has_local_worker_activity: bool,
        display_model_name: &str,
    ) -> NodeState {
        let has_declared_local_serving_work = (effective_is_host || has_local_worker_activity)
            && !display_model_name.trim().is_empty();

        if is_client {
            NodeState::Client
        } else if effective_llama_ready && has_declared_local_serving_work {
            NodeState::Serving
        } else if has_declared_local_serving_work {
            NodeState::Loading
        } else {
            NodeState::Standby
        }
    }

    #[cfg(test)]
    fn derive_node_status(node_state: NodeState) -> String {
        node_state.node_status_alias().to_string()
    }

    #[cfg(test)]
    fn derive_peer_state(peer: &mesh::PeerInfo) -> NodeState {
        fn has_nonempty_models(models: &[String]) -> bool {
            models.iter().any(|model| !model.trim().is_empty())
        }

        match peer.role {
            mesh::NodeRole::Client => NodeState::Client,
            mesh::NodeRole::Host { .. } | mesh::NodeRole::Worker => {
                let has_runtime_descriptors = peer
                    .served_model_runtime
                    .iter()
                    .any(|runtime| !runtime.model_name.trim().is_empty());
                let has_ready_runtime = peer
                    .served_model_runtime
                    .iter()
                    .any(|runtime| runtime.ready && !runtime.model_name.trim().is_empty());
                let has_assigned_model_work = has_runtime_descriptors
                    || has_nonempty_models(&peer.serving_models)
                    || has_nonempty_models(&peer.hosted_models);
                let has_legacy_serving_signal = has_nonempty_models(&peer.hosted_models)
                    || has_nonempty_models(&peer.serving_models)
                    || peer
                        .routable_models()
                        .iter()
                        .any(|model| !model.trim().is_empty());

                if has_ready_runtime {
                    NodeState::Serving
                } else if has_runtime_descriptors && has_assigned_model_work {
                    NodeState::Loading
                } else if has_legacy_serving_signal {
                    NodeState::Serving
                } else {
                    NodeState::Standby
                }
            }
        }
    }

    #[cfg(test)]
    fn build_wakeable_node(entry: WakeableInventoryEntry) -> WakeableNode {
        WakeableNode {
            logical_id: entry.logical_id,
            models: entry.models,
            vram_gb: entry.vram_gb,
            provider: entry.provider,
            state: match entry.state {
                WakeableState::Sleeping => WakeableNodeState::Sleeping,
                WakeableState::Waking => WakeableNodeState::Waking,
            },
            wake_eta_secs: entry.wake_eta_secs,
        }
    }

    async fn status(&self) -> StatusPayload {
        let (
            runtime_data_collector,
            node,
            node_id,
            my_vram_gb,
            inflight_requests,
            routing_affinity,
            model_size_bytes,
            is_client,
            api_port,
            draft_name,
            mesh_name,
            latest_version,
            mesh_discovery_mode,
            nostr_discovery,
            publication_state,
            wakeable_inventory,
            openai_guardrails,
            plugin_manager,
            listeners_ready,
        ) = {
            let inner = self.inner.lock().await;
            (
                inner.runtime_data_collector.clone(),
                inner.node.clone(),
                inner.node.id().fmt_short().to_string(),
                inner.node.vram_bytes() as f64 / 1e9,
                inner.node.inflight_requests(),
                inner.affinity_router.stats_snapshot(),
                inner.model_size_bytes,
                inner.is_client,
                inner.api_port,
                inner.draft_name.clone(),
                inner.mesh_name.clone(),
                inner.latest_version.clone(),
                inner.mesh_discovery_mode,
                inner.nostr_discovery,
                inner.publication_state,
                inner.wakeable_inventory.clone(),
                inner.openai_guardrails.clone(),
                inner.plugin_manager.clone(),
                inner.listeners_ready,
            )
        };
        let token = node.invite_token().await;
        let runtime_status = runtime_data_collector.runtime_status_snapshot();
        let model_name = runtime_status.primary_model.clone().unwrap_or_default();
        let local_processes =
            runtime_data::runtime_process_payloads(&runtime_status.local_processes);
        let mut runtime = build_runtime_status_payload(
            &model_name,
            runtime_status.primary_backend.clone(),
            openai_guardrails,
            runtime_status.is_host,
            runtime_status.llama_ready,
            runtime_status.llama_port,
            local_processes.clone(),
        );
        node.refresh_stage_runtime_statuses(std::time::Duration::from_secs(2))
            .await;
        runtime.stages = build_runtime_stage_payloads(node.stage_runtime_statuses().await);

        let wakeable_nodes = wakeable_inventory.status_snapshot().await;
        let hardware = runtime_data_collector
            .build_hardware_view(node_hardware_input(&node, my_vram_gb, model_size_bytes).await);

        let plugin_models = external_inference_models(&plugin_manager).await;
        let mut advertised_models = node.models().await;
        append_external_inference_models(&mut advertised_models, &plugin_models);
        let mut serving_models = node.serving_models().await;
        append_external_inference_models(&mut serving_models, &plugin_models);
        let mut hosted_models = node.hosted_models().await;
        append_external_inference_models(&mut hosted_models, &plugin_models);
        let peers = node.peers().await;

        let local_serving = local_processes
            .iter()
            .any(|process| matches!(process.status.as_str(), "serving" | "ready"));
        let plugin_ingress = !plugin_models.is_empty();
        let proxying = plugin_ingress
            || peers
                .iter()
                .any(|peer| !peer.http_routable_models().is_empty());
        let lifecycle_instances = build_lifecycle_instances(&local_processes);
        let has_terminal_failure = lifecycle_instances
            .iter()
            .any(|instance| instance.lifecycle_state == "failed");
        let accepting_local = node
            .activity_policy_guard
            .check_admission(crate::runtime::IngressType::LocalOpenAi)
            == crate::runtime::AdmissionResult::Allowed;
        let accepting_remote = node
            .activity_policy_guard
            .check_admission(crate::runtime::IngressType::RemoteQuicHttp)
            == crate::runtime::AdmissionResult::Allowed;
        let intents = node
            .runtime_intents
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let intent_summary = summarize_intents(&intents);
        runtime.daemon_state = Some(derive_daemon_state(
            crate::system::backend::runtime_shutting_down(),
            has_terminal_failure,
            node.activity_policy_guard.priority_degraded(),
            local_serving,
            proxying,
            listeners_ready,
        ));
        runtime.capabilities = Some(derive_capability_flags(
            is_client,
            local_serving,
            proxying,
            plugin_ingress,
            accepting_local,
            accepting_remote,
        ));
        runtime.lifecycle_instances = lifecycle_instances;
        runtime.intent_summary = Some(intent_summary);

        let mut payload = runtime_data::status_payload(runtime_data_collector.build_status_view(
            runtime_data::StatusViewInput {
                version: MESH_LLM_BUILD_VERSION.to_string(),
                latest_version,
                node_id,
                owner: node.owner_summary().await,
                release_attestation: node.release_attestation_summary().await,
                token,
                is_host: runtime_status.is_host,
                is_client,
                llama_ready: runtime_status.llama_ready,
                model_name,
                models: advertised_models,
                available_models: node.available_models().await,
                requested_models: node.requested_models().await,
                serving_models,
                hosted_models,
                draft_name,
                api_port,
                inflight_requests,
                mesh_id: node.mesh_id().await,
                mesh_name,
                mesh_discovery_mode: mesh_discovery_mode.as_str().into(),
                discovery_scope: mesh_discovery_mode.scope().as_str().into(),
                discovery_source: mesh_discovery_mode.source().into(),
                nostr_discovery,
                publication_state: publication_state.as_str().into(),
                local_processes,
                peers,
                wakeable_nodes,
                routing_affinity,
                hardware,
            },
        ));
        payload.runtime = runtime;
        payload.wanted_model_refs = self.wanted_model_refs().await;
        payload.mesh_requirements = node.mesh_requirement_policy_summary().await;
        payload.recent_mesh_rejections = node.recent_mesh_requirement_rejections().await;
        payload.logging =
            crate::logging_runtime_state().map(|state| LoggingStatusPayload::from(state.status()));
        payload
    }

    async fn push_status(&self) {
        let mut inner = self.inner.lock().await;
        inner.runtime_data_producer.mark_status_dirty();
        inner.sse_clients.retain(|tx| !tx.is_closed());
    }
}

fn build_lifecycle_instances(
    local_processes: &[RuntimeProcessPayload],
) -> Vec<LifecycleInstancePayload> {
    local_processes
        .iter()
        .filter_map(|process| {
            process
                .instance_id
                .as_ref()
                .map(|instance_id| LifecycleInstancePayload {
                    instance_id: instance_id.clone(),
                    model_ref: process.name.clone(),
                    lifecycle_state: match process.status.as_str() {
                        "ready" | "serving" => "serving",
                        "shutting down" => "draining",
                        "exited" => "failed",
                        "starting" => "loading",
                        other => other,
                    }
                    .to_string(),
                })
        })
        .take(256)
        .collect()
}

fn summarize_intents(intents: &[crate::runtime::DesiredRuntimeIntent]) -> IntentSummary {
    IntentSummary {
        durable_count: intents
            .iter()
            .filter(|intent| intent.persistence == crate::runtime::IntentPersistence::Process)
            .count(),
        session_count: intents
            .iter()
            .filter(|intent| intent.persistence == crate::runtime::IntentPersistence::Session)
            .count(),
        recent_errors: intents
            .iter()
            .rev()
            .filter_map(|intent| intent.last_error.clone())
            .take(16)
            .collect(),
    }
}

fn derive_capability_flags(
    is_client: bool,
    local_serving: bool,
    proxying: bool,
    plugin_ingress: bool,
    accepting_local: bool,
    accepting_remote: bool,
) -> RuntimeCapabilityFlags {
    RuntimeCapabilityFlags {
        worker_capable: !is_client,
        local_serving,
        proxying,
        plugin_ingress,
        accepting_local,
        accepting_remote,
    }
}

impl ServingController for MeshApi {
    fn load<'a>(&'a self, request: LoadModelRequest) -> ServingFuture<'a, ServedModel> {
        Box::pin(async move {
            let model_ref = request.model_ref;
            if !matches!(request.device_policy, NodeDevicePolicy::Auto) {
                return Err(anyhow::anyhow!(ServingError::UnsupportedDevicePolicy {
                    policy: request.device_policy,
                }));
            }
            let control_tx = self
                .inner
                .lock()
                .await
                .runtime_control
                .clone()
                .ok_or_else(|| runtime_unavailable("runtime control unavailable"))?;
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            control_tx
                .send(RuntimeControlRequest::Load {
                    spec: model_ref.clone(),
                    config_model_id: None,
                    profile: request.profile.clone(),
                    resp: resp_tx,
                })
                .map_err(|_| runtime_unavailable("runtime control unavailable"))?;
            let loaded = resp_rx
                .await
                .map_err(|_| runtime_unavailable("runtime control response dropped"))?
                .map_err(|error| {
                    anyhow::anyhow!(ServingError::LoadFailed {
                        model_ref: model_ref.clone(),
                        message: error.to_string(),
                    })
                })?;
            let capabilities = infer_served_model_capabilities(&model_ref, &loaded.model);
            Ok(ServedModel {
                model_ref: loaded.model_ref,
                profile: loaded.profile,
                model_id: loaded.model,
                instance_id: Some(loaded.instance_id),
                state: ServingModelState::Ready,
                backend: loaded.backend,
                capabilities,
                context_length: loaded.context_length,
                error: None,
            })
        })
    }

    fn unload<'a>(&'a self, request: UnloadModelRequest) -> ServingFuture<'a, ()> {
        Box::pin(async move {
            let control_tx = self
                .inner
                .lock()
                .await
                .runtime_control
                .clone()
                .ok_or_else(|| runtime_unavailable("runtime control unavailable"))?;
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            let target = request.target;
            control_tx
                .send(RuntimeControlRequest::Unload {
                    target: target.clone(),
                    options: request.options,
                    resp: resp_tx,
                })
                .map_err(|_| runtime_unavailable("runtime control unavailable"))?;
            let _ = resp_rx
                .await
                .map_err(|_| runtime_unavailable("runtime control response dropped"))?
                .map_err(|error| {
                    anyhow::anyhow!(ServingError::UnloadFailed {
                        target,
                        message: error.to_string(),
                    })
                })?;
            Ok(())
        })
    }

    fn served_models<'a>(&'a self) -> ServingFuture<'a, Vec<ServedModel>> {
        Box::pin(async move {
            Ok(self
                .runtime_status()
                .await
                .models
                .into_iter()
                .map(served_model_from_runtime_payload)
                .collect())
        })
    }

    fn status<'a>(&'a self) -> ServingFuture<'a, ServingStatus> {
        Box::pin(async move {
            let enabled = self.inner.lock().await.runtime_control.is_some();
            let models = self
                .runtime_status()
                .await
                .models
                .into_iter()
                .map(served_model_from_runtime_payload)
                .collect();
            Ok(ServingStatus { enabled, models })
        })
    }

    fn set_device_policy<'a>(&'a self, policy: NodeDevicePolicy) -> ServingFuture<'a, ()> {
        Box::pin(async move {
            match policy {
                NodeDevicePolicy::Auto => Ok(()),
                policy => Err(anyhow::anyhow!(ServingError::UnsupportedDevicePolicy {
                    policy,
                })),
            }
        })
    }
}

fn served_model_from_runtime_payload(model: RuntimeModelPayload) -> ServedModel {
    let capabilities = infer_served_model_capabilities(&model.name, &model.name);
    // Build model_ref with profile suffix for non-default profiles
    let model_ref = if model.profile.is_empty() {
        model.name.clone()
    } else {
        format!("{}#{}", model.name, model.profile)
    };
    ServedModel {
        model_ref,
        profile: model.profile,
        model_id: model.name,
        instance_id: model.instance_id,
        state: serving_model_state_from_runtime_status(&model.status),
        backend: Some(model.backend),
        capabilities,
        context_length: model.context_length,
        error: None,
    }
}

fn runtime_unavailable(message: impl Into<String>) -> anyhow::Error {
    anyhow::anyhow!(ServingError::RuntimeUnavailable {
        message: message.into(),
    })
}

fn infer_served_model_capabilities(
    model_ref: &str,
    model_id: &str,
) -> mesh_llm_node::models::ModelCapabilities {
    merge_name_signals(Default::default(), &[model_ref, model_id]).normalize()
}

fn serving_model_state_from_runtime_status(status: &str) -> ServingModelState {
    match status.to_ascii_lowercase().as_str() {
        "loading" | "starting" => ServingModelState::Loading,
        "ready" | "running" => ServingModelState::Ready,
        "failed" | "error" => ServingModelState::Failed,
        "unloading" | "stopping" => ServingModelState::Unloading,
        "stopped" => ServingModelState::Stopped,
        other => ServingModelState::Unknown(other.to_string()),
    }
}

fn runtime_process_payload_identity(process: &RuntimeProcessPayload) -> &str {
    process.instance_id.as_deref().unwrap_or(&process.name)
}

async fn node_hardware_input(
    node: &mesh::Node,
    my_vram_gb: f64,
    model_size_bytes: u64,
) -> runtime_data::HardwareViewInput {
    runtime_data::HardwareViewInput {
        gpu_name: node.gpu_name.clone(),
        gpu_vram: node.gpu_vram.clone(),
        gpu_reserved_bytes: node.gpu_reserved_bytes.clone(),
        gpu_mem_bandwidth_gbps: node_metric_csv(&node.gpu_mem_bandwidth_gbps).await,
        gpu_compute_tflops_fp32: node_metric_csv(&node.gpu_compute_tflops_fp32).await,
        gpu_compute_tflops_fp16: node_metric_csv(&node.gpu_compute_tflops_fp16).await,
        my_hostname: node.hostname.clone(),
        my_is_soc: node.is_soc,
        my_vram_gb,
        model_size_gb: model_size_bytes as f64 / 1e9,
        first_joined_mesh_ts: node.first_joined_mesh_ts().await,
    }
}

async fn node_metric_csv(metric: &Arc<Mutex<Option<Vec<f64>>>>) -> Option<String> {
    metric.lock().await.as_ref().map(|values| {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",")
    })
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
pub(crate) mod tests;
