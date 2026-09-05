//! Collector-backed snapshot storage and synchronous publish helpers.
//!
//! Keep mutation local, drop locks before publish, and let readers observe
//! shared snapshots through this boundary.

use super::event_cutover::{
    FieldId, ShadowHealth, field_of_dirty, merge_legacy_publish, should_apply_legacy_write,
};
use super::inventory::{
    InventoryScanCoordinator, InventoryScanDisposition, InventoryScanError, InventoryScanOutcome,
    InventoryScanResult, replace_local_instances_snapshot, replace_local_inventory_snapshot,
};
use super::plugins::{
    PluginDataValue, PluginsSnapshotView, clear_plugin_data, clear_plugin_endpoints,
    plugins_snapshot, upsert_plugin_data, upsert_plugin_endpoint,
};
#[cfg(test)]
use super::plugins::{PluginScopedSnapshot, plugin_endpoint_snapshot, plugin_snapshot};
use super::processes::RuntimeProcessSnapshot;
use super::producers::{RuntimeDataProducer, RuntimeDataSource};
use super::snapshots::{
    HardwareViewInput, HardwareViewSnapshot, LocalInstancesSnapshot, ModelRouteStats,
    ModelViewInput, ModelViewSnapshot, PluginDataKey, PluginDataSnapshot, PluginEndpointKey,
    PluginEndpointsSnapshot, RuntimeDataSnapshots, RuntimeStatusDerivation, RuntimeStatusSnapshot,
    StatusViewInput, StatusViewSnapshot,
};
use super::subscriptions::{
    RuntimeDataDirty, RuntimeDataSubscriptionState, RuntimeDataSubscriptions,
};
use super::{
    RuntimeLlamaMetricItem, RuntimeLlamaMetricsSnapshot, RuntimeLlamaRuntimeItems,
    RuntimeLlamaRuntimeSnapshot, RuntimeLlamaSlotItem, RuntimeLlamaSlotsSnapshot,
};
use crate::api::status::{
    LatencySource, LocalInstance, MeshModelPayload, NodeState, PeerPayload, WakeableNode,
    WakeableNodeState, build_gpus, build_ownership_payload,
};
use crate::mesh;
use crate::models::LocalModelInventorySnapshot;
use crate::network::metrics::RoutingCollectorSnapshot;
use crate::plugin::PluginEndpointSummary;
use crate::runtime::instance::LocalInstanceSnapshot;
use crate::runtime::wakeable::{WakeableInventoryEntry, WakeableState};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use tokio::sync::watch;

#[derive(Default)]
struct RuntimeDataSharedState {
    snapshots: RwLock<RuntimeDataSnapshots>,
    subscriptions: RuntimeDataSubscriptions,
    inventory_scan: Mutex<InventoryScanCoordinator>,
    event_cutover_health: ShadowHealth,
    #[cfg(test)]
    shadow_test_hook: Mutex<Option<Box<dyn FnOnce() + Send + 'static>>>,
}

#[derive(Clone, Default)]
pub(crate) struct RuntimeDataCollector {
    shared: Arc<RuntimeDataSharedState>,
}

enum LegacyShadowSnapshot {
    Status(bool),
    Inventory(HashSet<String>),
}

#[cfg(test)]
impl RuntimeDataSharedState {
    fn set_shadow_test_hook(&self, hook: impl FnOnce() + Send + 'static) {
        *self
            .shadow_test_hook
            .lock()
            .expect("runtime data shadow test hook lock poisoned") = Some(Box::new(hook));
    }

    fn run_shadow_test_hook(&self) {
        let hook = self
            .shadow_test_hook
            .lock()
            .expect("runtime data shadow test hook lock poisoned")
            .take();
        if let Some(hook) = hook {
            hook();
        }
    }
}

impl RuntimeDataCollector {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn producer(&self, source: RuntimeDataSource) -> RuntimeDataProducer {
        RuntimeDataProducer::new(self.clone(), source)
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<RuntimeDataSubscriptionState> {
        self.shared.subscriptions.subscribe()
    }

    #[cfg(test)]
    pub(crate) fn subscription_state(&self) -> RuntimeDataSubscriptionState {
        self.shared.subscriptions.state()
    }

    #[cfg(test)]
    pub(crate) fn inventory_scan_waiter_count(&self) -> usize {
        self.shared
            .inventory_scan
            .lock()
            .expect("runtime data inventory scan lock poisoned")
            .waiters_len()
    }

    pub(crate) fn mark_dirty(&self, dirty: RuntimeDataDirty) -> RuntimeDataSubscriptionState {
        self.shared.subscriptions.publish(dirty)
    }

    pub(crate) fn update_runtime_status<F>(&self, dirty: RuntimeDataDirty, update: F) -> bool
    where
        F: FnOnce(&mut RuntimeStatusSnapshot) -> bool,
    {
        self.update_snapshots(dirty, |snapshots| update(&mut snapshots.runtime_status))
    }

    pub(crate) fn snapshots(&self) -> RuntimeDataSnapshots {
        self.shared
            .snapshots
            .read()
            .expect("runtime data snapshots lock poisoned")
            .clone()
    }

    pub(crate) fn runtime_status_snapshot(&self) -> RuntimeStatusSnapshot {
        self.snapshots().runtime_status
    }

    pub(crate) fn runtime_processes_snapshot(&self) -> Vec<RuntimeProcessSnapshot> {
        self.runtime_status_snapshot().local_processes
    }

    pub(crate) fn runtime_llama_snapshot(&self) -> RuntimeLlamaRuntimeSnapshot {
        self.runtime_status_snapshot().llama_runtime
    }

    pub(crate) fn runtime_llama_snapshots_by_model(
        &self,
    ) -> BTreeMap<String, RuntimeLlamaRuntimeSnapshot> {
        self.runtime_status_snapshot().llama_runtime_by_model
    }

    pub(crate) fn runtime_llama_snapshots_by_instance(
        &self,
    ) -> BTreeMap<String, RuntimeLlamaRuntimeSnapshot> {
        self.runtime_status_snapshot().llama_runtime_by_instance
    }

    pub(crate) fn routing_snapshot(&self) -> RoutingCollectorSnapshot {
        self.snapshots().routing
    }

    pub(crate) fn local_instances_snapshot(&self) -> LocalInstancesSnapshot {
        self.snapshots().local_instances
    }

    pub(crate) fn local_inventory_snapshot(&self) -> LocalModelInventorySnapshot {
        self.snapshots().local_inventory
    }

    pub(crate) fn replace_local_instances_snapshot(
        &self,
        instances: Vec<LocalInstanceSnapshot>,
    ) -> bool {
        self.update_snapshots(RuntimeDataDirty::INVENTORY, |snapshots| {
            replace_local_instances_snapshot(&mut snapshots.local_instances, instances)
        })
    }

    #[cfg(test)]
    pub(crate) fn replace_llama_metrics_snapshot(
        &self,
        snapshot: RuntimeLlamaMetricsSnapshot,
    ) -> bool {
        self.update_runtime_status(RuntimeDataDirty::RUNTIME, |runtime_status| {
            let next_items =
                build_llama_runtime_items(&snapshot, &runtime_status.llama_runtime.slots);
            let mut changed = false;
            if runtime_status.llama_runtime.metrics != snapshot
                || runtime_status.llama_runtime.items != next_items
            {
                runtime_status.llama_runtime.metrics = snapshot.clone();
                runtime_status.llama_runtime.items = next_items;
                changed = true;
            }

            for runtime in runtime_status.llama_runtime_by_model.values_mut() {
                let next_items = build_llama_runtime_items(&snapshot, &runtime.slots);
                if runtime.metrics != snapshot || runtime.items != next_items {
                    runtime.metrics = snapshot.clone();
                    runtime.items = next_items;
                    changed = true;
                }
            }
            for runtime in runtime_status.llama_runtime_by_instance.values_mut() {
                let next_items = build_llama_runtime_items(&snapshot, &runtime.slots);
                if runtime.metrics != snapshot || runtime.items != next_items {
                    runtime.metrics = snapshot.clone();
                    runtime.items = next_items;
                    changed = true;
                }
            }

            let selected = select_runtime_llama_projection(runtime_status);
            if runtime_status.llama_runtime != selected {
                runtime_status.llama_runtime = selected;
                changed = true;
            }
            changed
        })
    }

    pub(crate) fn replace_llama_slots_snapshot(&self, snapshot: RuntimeLlamaSlotsSnapshot) -> bool {
        self.update_runtime_status(RuntimeDataDirty::RUNTIME, |runtime_status| {
            let next_runtime =
                build_llama_runtime_snapshot(&runtime_status.llama_runtime.metrics, snapshot);
            let mut changed = false;

            if let Some(instance_id) = next_runtime.slots.instance_id.clone()
                && runtime_status.llama_runtime_by_instance.get(&instance_id) != Some(&next_runtime)
            {
                runtime_status
                    .llama_runtime_by_instance
                    .insert(instance_id, next_runtime.clone());
                changed = true;
            }

            if let Some(model) = next_runtime.slots.model.clone() {
                if should_replace_model_runtime_projection(
                    runtime_status.llama_runtime_by_model.get(&model),
                    &next_runtime,
                ) {
                    runtime_status
                        .llama_runtime_by_model
                        .insert(model, next_runtime.clone());
                    changed = true;
                }

                let selected = select_runtime_llama_projection(runtime_status);
                if runtime_status.llama_runtime != selected {
                    runtime_status.llama_runtime = selected;
                    changed = true;
                }
            } else if runtime_status.llama_runtime != next_runtime {
                runtime_status.llama_runtime = next_runtime;
                changed = true;
            }

            changed
        })
    }

    pub(crate) async fn coalesce_local_inventory_scan<F>(&self, load: F) -> InventoryScanResult
    where
        F: FnOnce() -> LocalModelInventorySnapshot + Send + 'static,
    {
        self.coalesce_local_inventory_scan_outcome(move || Ok(load()))
            .await
            .map(|outcome| outcome.snapshot)
    }

    pub(crate) async fn coalesce_local_inventory_scan_outcome<F>(
        &self,
        load: F,
    ) -> Result<InventoryScanOutcome, InventoryScanError>
    where
        F: FnOnce() -> InventoryScanResult + Send + 'static,
    {
        let (rx, start_scan) = {
            let mut inventory_scan = self
                .shared
                .inventory_scan
                .lock()
                .expect("runtime data inventory scan lock poisoned");
            inventory_scan.begin_or_join()
        };

        if start_scan {
            let collector = self.clone();
            tokio::spawn(async move {
                let scan_result = match tokio::task::spawn_blocking(load).await {
                    Ok(result) => result,
                    Err(err) => {
                        tracing::warn!("Local inventory scan failed: {err}");
                        Err(InventoryScanError::TaskPanicked(err.to_string()))
                    }
                };

                if let Ok(snapshot) = &scan_result {
                    collector.replace_local_inventory_snapshot(snapshot.clone());
                }
                let waiters = {
                    let mut inventory_scan = collector
                        .shared
                        .inventory_scan
                        .lock()
                        .expect("runtime data inventory scan lock poisoned");
                    inventory_scan.finish()
                };
                for waiter in waiters {
                    let _ = waiter.send(scan_result.clone());
                }
            });
        }

        let snapshot = rx.await.map_err(|_| InventoryScanError::TaskCancelled)??;
        Ok(InventoryScanOutcome {
            snapshot,
            disposition: if start_scan {
                InventoryScanDisposition::Executed
            } else {
                InventoryScanDisposition::Coalesced
            },
        })
    }

    pub(crate) fn plugin_data_snapshot(&self) -> PluginDataSnapshot {
        self.snapshots().plugin_data
    }

    pub(crate) fn plugin_endpoints_snapshot(&self) -> PluginEndpointsSnapshot {
        self.snapshots().plugin_endpoints
    }

    pub(crate) fn plugins_snapshot(&self) -> PluginsSnapshotView {
        let snapshots = self.snapshots();
        plugins_snapshot(&snapshots.plugin_data, &snapshots.plugin_endpoints)
    }

    #[cfg(test)]
    pub(crate) fn plugin_snapshot(&self, plugin_name: &str) -> PluginScopedSnapshot {
        let snapshots = self.snapshots();
        plugin_snapshot(
            &snapshots.plugin_data,
            &snapshots.plugin_endpoints,
            plugin_name,
        )
    }

    #[cfg(test)]
    pub(crate) fn plugin_endpoint_snapshot(
        &self,
        plugin_name: &str,
        endpoint_id: &str,
    ) -> Option<PluginEndpointSummary> {
        plugin_endpoint_snapshot(&self.snapshots().plugin_endpoints, plugin_name, endpoint_id)
    }

    pub(crate) fn publish_plugin_data(&self, key: PluginDataKey, value: PluginDataValue) -> bool {
        self.update_snapshots(RuntimeDataDirty::PLUGINS, |snapshots| {
            upsert_plugin_data(&mut snapshots.plugin_data, key, value)
        })
    }

    pub(crate) fn publish_plugin_endpoint(
        &self,
        key: PluginEndpointKey,
        value: PluginEndpointSummary,
    ) -> bool {
        self.update_snapshots(RuntimeDataDirty::PLUGINS, |snapshots| {
            upsert_plugin_endpoint(&mut snapshots.plugin_endpoints, key, value)
        })
    }

    pub(crate) fn clear_plugin_reports(&self, plugin_name: &str) -> bool {
        self.update_snapshots(RuntimeDataDirty::PLUGINS, |snapshots| {
            let data_changed = clear_plugin_data(&mut snapshots.plugin_data, plugin_name);
            let endpoints_changed =
                clear_plugin_endpoints(&mut snapshots.plugin_endpoints, plugin_name);
            data_changed || endpoints_changed
        })
    }

    pub(crate) fn build_hardware_view(&self, input: HardwareViewInput) -> HardwareViewSnapshot {
        HardwareViewSnapshot {
            my_hostname: input.my_hostname,
            my_is_soc: input.my_is_soc,
            my_vram_gb: input.my_vram_gb,
            model_size_gb: input.model_size_gb,
            gpus: build_gpus(
                input.gpu_name.as_deref(),
                input.gpu_vram.as_deref(),
                input.gpu_reserved_bytes.as_deref(),
                input.gpu_mem_bandwidth_gbps.as_deref(),
                input.gpu_compute_tflops_fp32.as_deref(),
                input.gpu_compute_tflops_fp16.as_deref(),
            ),
            first_joined_mesh_ts: input.first_joined_mesh_ts,
        }
    }

    pub(crate) fn build_status_view(&self, input: StatusViewInput) -> StatusViewSnapshot {
        let derivation = derive_runtime_status(RuntimeStatusDerivationInput {
            is_client: input.is_client,
            is_host: input.is_host,
            llama_ready: input.llama_ready,
            local_processes: &input.local_processes,
            hosted_models: &input.hosted_models,
            serving_models: &input.serving_models,
            model_name: &input.model_name,
            api_port: input.api_port,
        });
        let routing_snapshot = self.routing_snapshot();

        StatusViewSnapshot {
            version: input.version.clone(),
            latest_version: input.latest_version,
            node_id: input.node_id,
            owner: build_ownership_payload(&input.owner),
            release_attestation: input.release_attestation,
            token: input.token,
            node_state: derivation.node_state,
            node_status: derivation.node_status,
            is_host: derivation.effective_is_host,
            is_client: input.is_client,
            llama_ready: derivation.effective_llama_ready,
            model_name: derivation.display_model_name,
            models: input.models,
            available_models: input.available_models,
            requested_models: input.requested_models,
            serving_models: input.serving_models,
            hosted_models: input.hosted_models,
            draft_name: input.draft_name,
            api_port: input.api_port,
            peers: input.peers.iter().map(build_peer_payload).collect(),
            wakeable_nodes: input
                .wakeable_nodes
                .into_iter()
                .map(build_wakeable_node)
                .collect(),
            local_instances: build_local_instances(
                self.local_instances_snapshot().instances,
                input.api_port,
                &input.version,
            ),
            launch_pi: derivation.launch_pi,
            launch_goose: derivation.launch_goose,
            inflight_requests: input.inflight_requests,
            mesh_id: input.mesh_id,
            mesh_name: input.mesh_name,
            mesh_discovery_mode: input.mesh_discovery_mode,
            discovery_scope: input.discovery_scope,
            discovery_source: input.discovery_source,
            nostr_discovery: input.nostr_discovery,
            publication_state: input.publication_state,
            routing_affinity: input.routing_affinity,
            routing_metrics: routing_snapshot.status,
            hardware: input.hardware,
        }
    }

    pub(crate) fn build_model_view(&self, mut input: ModelViewInput) -> ModelViewSnapshot {
        let routing_metrics_by_model = self.routing_snapshot().models;
        let local_model_names = std::mem::take(&mut input.local_inventory.model_names);
        let mut metadata_by_name = std::mem::take(&mut input.local_inventory.metadata_by_name);
        let mut size_by_name = std::mem::take(&mut input.local_inventory.size_by_name);
        let display_name_by_name = std::mem::take(&mut input.local_inventory.display_name_by_name);
        for peer in &input.peers {
            for meta in &peer.available_model_metadata {
                metadata_by_name
                    .entry(meta.model_key.clone())
                    .or_insert_with(|| meta.clone());
            }
            for (model_name, size) in &peer.available_model_sizes {
                size_by_name.entry(model_name.clone()).or_insert(*size);
            }
        }

        let mut catalog = std::mem::take(&mut input.catalog);
        let mut catalog_names = catalog
            .iter()
            .map(|entry| entry.model_name.clone())
            .collect::<HashSet<_>>();
        for model_name in input
            .served_models
            .iter()
            .chain(input.my_hosted_models.iter())
        {
            if model_name.trim().is_empty() || !catalog_names.insert(model_name.clone()) {
                continue;
            }
            catalog.push(mesh::MeshCatalogEntry {
                model_name: model_name.clone(),
                descriptor: None,
            });
        }

        let build_ctx = ModelViewBuildContext {
            input: &input,
            routing_metrics_by_model: &routing_metrics_by_model,
            local_model_names: &local_model_names,
            metadata_by_name: &metadata_by_name,
            size_by_name: &size_by_name,
            display_name_by_name: &display_name_by_name,
        };
        let models = catalog
            .iter()
            .map(|entry| build_model_payload_from_catalog_entry(entry, &build_ctx))
            .collect();

        ModelViewSnapshot { models }
    }

    pub(crate) fn replace_routing_snapshot(&self, snapshot: RoutingCollectorSnapshot) -> bool {
        self.update_snapshots(RuntimeDataDirty::ROUTING, |snapshots| {
            if snapshots.routing == snapshot {
                false
            } else {
                snapshots.routing = snapshot;
                true
            }
        })
    }

    fn replace_local_inventory_snapshot(&self, snapshot: LocalModelInventorySnapshot) -> bool {
        self.update_snapshots(RuntimeDataDirty::INVENTORY, |snapshots| {
            replace_local_inventory_snapshot(&mut snapshots.local_inventory, snapshot)
        })
    }

    /// Generation-tagged field-level merge gate for the collector's
    /// publish path (task 5). A legacy whole-snapshot write for a field
    /// still at `Legacy` generation applies exactly as before. A field
    /// that has cut over to `Reducer` generation never runs `update` at
    /// all: the legacy write is ignored (never merged and reverted) and
    /// counted as stale, and the reducer projection remains authoritative.
    /// Every field is `Legacy` today (see `event_cutover::CUTOVER_MATRIX`),
    /// so this preserves existing behavior exactly while being the real
    /// integration point once a producer task lands a reducer projection.
    ///
    /// After a `Status`/`Inventory` write, also runs the task 6 shadow
    /// comparison (`record_shadow_divergence`) against the reducer's own
    /// domain projection -- observability only, never a cutover trigger.
    fn update_snapshots<F>(&self, dirty: RuntimeDataDirty, update: F) -> bool
    where
        F: FnOnce(&mut RuntimeDataSnapshots) -> bool,
    {
        let field = field_of_dirty(dirty);
        if let Some(field) = field
            && !should_apply_legacy_write(field, &self.shared.event_cutover_health)
        {
            return false;
        }

        let (changed, legacy_shadow) = {
            let mut snapshots = self
                .shared
                .snapshots
                .write()
                .expect("runtime data snapshots lock poisoned");
            let changed = update(&mut snapshots);
            let legacy_shadow = field.and_then(|field| match field {
                FieldId::Status => Some(LegacyShadowSnapshot::Status(
                    snapshots.runtime_status.llama_ready,
                )),
                FieldId::Inventory => Some(LegacyShadowSnapshot::Inventory(
                    snapshots.local_inventory.model_names.clone(),
                )),
                FieldId::Routing | FieldId::Processes | FieldId::Plugins | FieldId::Runtime => None,
            });
            (changed, legacy_shadow)
        };

        #[cfg(test)]
        self.shared.run_shadow_test_hook();

        if let Some(legacy_shadow) = legacy_shadow {
            self.record_shadow_divergence(legacy_shadow);
        }

        if changed {
            self.shared.subscriptions.publish(dirty);
        }

        changed
    }

    /// Task 6 (`.omo/plans/event-system-fixes.md`, defect D14): compare the
    /// just-written legacy value against the reducer's own domain
    /// projection for the two fields task 6 gives a real reducer
    /// projection to compare against -- `Status` (does the reducer also
    /// think a model is available, matching `llama_ready`?) and
    /// `Inventory` (does the reducer's known model-id set match the legacy
    /// local-inventory model-name set?). `merge_legacy_publish` records a
    /// divergence but its returned winner is discarded: every
    /// `CUTOVER_MATRIX` row stays `Legacy`, so the legacy value already
    /// written above remains authoritative regardless of this comparison.
    /// A `None` reducer projection (no engine installed, e.g.
    /// `--local-model-only` before the engine starts) is a silent no-op,
    /// matching `merge_legacy_publish`'s own `Option` contract.
    fn record_shadow_divergence(&self, legacy_shadow: LegacyShadowSnapshot) {
        match legacy_shadow {
            LegacyShadowSnapshot::Status(legacy) => {
                let reducer_available = crate::runtime_events::runtime_event_engine()
                    .map(|engine| engine.reducer_snapshot().domain().has_available_model());
                let _ = merge_legacy_publish(
                    FieldId::Status,
                    legacy,
                    reducer_available.as_ref(),
                    &self.shared.event_cutover_health,
                );
            }
            LegacyShadowSnapshot::Inventory(legacy) => {
                let reducer_models = crate::runtime_events::runtime_event_engine()
                    .map(|engine| engine.reducer_snapshot().domain().model_id_set());
                let _ = merge_legacy_publish(
                    FieldId::Inventory,
                    legacy,
                    reducer_models.as_ref(),
                    &self.shared.event_cutover_health,
                );
            }
        }
    }
}

struct ModelViewBuildContext<'a> {
    input: &'a ModelViewInput,
    routing_metrics_by_model:
        &'a HashMap<String, crate::network::metrics::ModelRoutingMetricsSnapshot>,
    local_model_names: &'a HashSet<String>,
    metadata_by_name: &'a HashMap<String, crate::proto::node::CompactModelMetadata>,
    size_by_name: &'a HashMap<String, u64>,
    display_name_by_name: &'a HashMap<String, String>,
}

fn build_model_payload_from_catalog_entry(
    entry: &mesh::MeshCatalogEntry,
    ctx: &ModelViewBuildContext<'_>,
) -> MeshModelPayload {
    let input = ctx.input;
    let name = &entry.model_name;
    let descriptor = entry.descriptor.as_ref();
    let identity = descriptor.map(|descriptor| &descriptor.identity);
    let catalog_entry = find_catalog_model(name);
    let is_warm = input.served_models.iter().any(|served| served == name);
    let local_known = ctx.local_model_names.contains(name)
        || input.my_hosted_models.iter().any(|served| served == name)
        || input.my_serving_models.iter().any(|served| served == name)
        || name == &input.model_name;
    let display_name = crate::models::loaded_remote_catalog_display_name(name)
        .or_else(|| ctx.display_name_by_name.get(name).cloned())
        .unwrap_or_else(|| name.to_string());
    let route_stats = is_warm.then(|| {
        http_route_stats(
            name,
            &input.peers,
            &input.my_hosted_models,
            input.node_hostname.as_deref(),
            input.my_vram_gb,
        )
    });
    let node_count = route_stats
        .as_ref()
        .map(|stats| stats.node_count)
        .unwrap_or(0);
    let active_nodes = route_stats
        .as_ref()
        .map(|stats| stats.active_nodes.clone())
        .unwrap_or_default();
    let mesh_vram_gb = route_stats
        .as_ref()
        .map(|stats| stats.mesh_vram_gb)
        .unwrap_or(0.0);
    let size_gb = model_size_gb_for_view(name, &catalog_entry, ctx, input);
    let (request_count, last_active_secs_ago) = match input.active_demand.get(name) {
        Some(demand) => (
            Some(demand.request_count),
            Some(input.now_unix_secs.saturating_sub(demand.last_active)),
        ),
        None => (None, None),
    };
    let routing_metrics = ctx.routing_metrics_by_model.get(name).cloned();
    let capabilities =
        model_capabilities_for_view(name, descriptor, catalog_entry.as_ref(), local_known);
    let description = catalog_entry
        .as_ref()
        .and_then(|model| model.description.clone());
    let metadata = ctx.metadata_by_name.get(name);
    let architecture = metadata
        .map(|m| m.architecture.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let context_length = metadata
        .map(|m| m.context_length)
        .filter(|value| *value > 0);
    let quantization = model_quantization_for_view(metadata, catalog_entry.as_ref());
    let tokenizer = metadata
        .map(|m| m.tokenizer_model_name.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let layer_count = compact_metadata_nonzero(metadata, |m| m.layer_count);
    let head_count = compact_metadata_nonzero(metadata, |m| m.head_count);
    let embedding_size = compact_metadata_nonzero(metadata, |m| m.embedding_size);
    let draft_model = catalog_entry
        .as_ref()
        .and_then(crate::models::remote_catalog_model_draft_ref);
    let source_page_url = model_source_page_url(identity, catalog_entry.as_ref(), local_known);
    let source_ref = identity
        .and_then(huggingface_repository_from_identity)
        .or_else(|| {
            source_page_url
                .as_deref()
                .map(|url| url.replace("https://huggingface.co/", ""))
        });
    let source_revision = identity.and_then(|identity| identity.revision.clone());
    let source_file = identity.and_then(source_file_from_identity).or_else(|| {
        local_known
            .then(|| catalog_entry.as_ref().map(|model| model.file.clone()))
            .flatten()
    });
    let command_ref = identity
        .and_then(|identity| identity.canonical_ref.clone())
        .or_else(|| {
            local_known
                .then(|| {
                    catalog_entry
                        .as_ref()
                        .map(crate::models::remote_catalog_model_ref)
                })
                .flatten()
        })
        .unwrap_or_else(|| name.clone());
    let (fit_label, fit_detail) = fit_hint_for_machine(size_gb, input.my_vram_gb);
    let capability_view = model_capability_view(&capabilities);

    MeshModelPayload {
        name: name.clone(),
        display_name,
        status: if is_warm {
            "warm".into()
        } else {
            "cold".into()
        },
        node_count,
        mesh_vram_gb,
        size_gb,
        architecture,
        context_length,
        quantization,
        tokenizer,
        layer_count,
        head_count,
        embedding_size,
        description,
        multimodal: capability_view.multimodal,
        multimodal_status: capability_view.multimodal_status,
        vision: capability_view.vision,
        vision_status: capability_view.vision_status,
        audio: capability_view.audio,
        audio_status: capability_view.audio_status,
        reasoning: capability_view.reasoning,
        reasoning_status: capability_view.reasoning_status,
        tool_use: capability_view.tool_use,
        tool_use_status: capability_view.tool_use_status,
        draft_model,
        request_count,
        last_active_secs_ago,
        target_rank: None,
        explicit_interest_count: None,
        wanted: None,
        routing_metrics,
        source_page_url,
        source_ref,
        source_revision,
        source_file,
        active_nodes,
        fit_label,
        fit_detail,
        download_command: format!("mesh-llm models download {}", command_ref),
        run_command: format!("mesh-llm serve --model {}", command_ref),
        auto_command: format!("mesh-llm serve --auto --model {}", command_ref),
    }
}

fn compact_metadata_nonzero(
    metadata: Option<&crate::proto::node::CompactModelMetadata>,
    field: impl FnOnce(&crate::proto::node::CompactModelMetadata) -> u32,
) -> Option<u32> {
    metadata.map(field).filter(|value| *value > 0)
}

fn model_size_gb_for_view(
    name: &str,
    catalog_entry: &Option<crate::models::remote_catalog::RemoteCatalogModel>,
    ctx: &ModelViewBuildContext<'_>,
    input: &ModelViewInput,
) -> f64 {
    if name == input.model_name && input.model_size_bytes > 0 {
        input.model_size_bytes as f64 / 1e9
    } else {
        ctx.size_by_name
            .get(name)
            .map(|size| *size as f64 / 1e9)
            .unwrap_or_else(|| {
                crate::models::catalog::parse_size_gb(
                    catalog_entry
                        .as_ref()
                        .and_then(|model| model.size.as_deref())
                        .unwrap_or("0"),
                )
            })
    }
}

fn model_capabilities_for_view(
    name: &str,
    descriptor: Option<&mesh::ServedModelDescriptor>,
    catalog_entry: Option<&crate::models::remote_catalog::RemoteCatalogModel>,
    local_known: bool,
) -> crate::models::ModelCapabilities {
    let mut capabilities = descriptor
        .filter(|descriptor| descriptor.capabilities_known)
        .map(|descriptor| descriptor.capabilities)
        .unwrap_or_else(|| {
            if local_known {
                crate::models::installed_model_capabilities(name)
            } else {
                crate::models::ModelCapabilities::default()
            }
        });
    let description = catalog_entry.and_then(|model| model.description.as_deref());
    let capabilities_known = descriptor
        .map(|descriptor| descriptor.capabilities_known)
        .unwrap_or(false);
    if local_known && likely_reasoning_model(name, description) {
        capabilities.reasoning = capabilities
            .reasoning
            .max(crate::models::capabilities::CapabilityLevel::Likely);
    }
    if local_known && !capabilities_known && likely_vision_model(name, description) {
        capabilities.vision = capabilities
            .vision
            .max(crate::models::capabilities::CapabilityLevel::Likely);
        capabilities.multimodal = true;
    }
    if local_known && !capabilities_known && likely_audio_model(name, description) {
        capabilities.audio = capabilities
            .audio
            .max(crate::models::capabilities::CapabilityLevel::Likely);
        capabilities.multimodal = true;
    }
    capabilities
}

struct ModelCapabilityView {
    multimodal: bool,
    multimodal_status: Option<&'static str>,
    vision: bool,
    vision_status: Option<&'static str>,
    audio: bool,
    audio_status: Option<&'static str>,
    reasoning: bool,
    reasoning_status: Option<&'static str>,
    tool_use: bool,
    tool_use_status: Option<&'static str>,
}

fn model_capability_view(capabilities: &crate::models::ModelCapabilities) -> ModelCapabilityView {
    let multimodal = capabilities.supports_multimodal_runtime();
    let vision = capabilities.supports_vision_runtime();
    let audio = matches!(
        capabilities.audio,
        crate::models::capabilities::CapabilityLevel::Supported
            | crate::models::capabilities::CapabilityLevel::Likely
    );
    let reasoning = matches!(
        capabilities.reasoning,
        crate::models::capabilities::CapabilityLevel::Supported
            | crate::models::capabilities::CapabilityLevel::Likely
    );
    let tool_use = capabilities.tool_use_label().is_some();
    ModelCapabilityView {
        multimodal,
        multimodal_status: (multimodal || capabilities.multimodal_label().is_some())
            .then_some(capabilities.multimodal_status()),
        vision,
        vision_status: (vision || capabilities.vision_label().is_some())
            .then_some(capabilities.vision_status()),
        audio,
        audio_status: (audio || capabilities.audio_label().is_some())
            .then_some(capabilities.audio_status()),
        reasoning,
        reasoning_status: (reasoning || capabilities.reasoning_label().is_some())
            .then_some(capabilities.reasoning_status()),
        tool_use,
        tool_use_status: capabilities
            .tool_use_label()
            .map(|_| capabilities.tool_use_status()),
    }
}

fn model_quantization_for_view(
    metadata: Option<&crate::proto::node::CompactModelMetadata>,
    catalog_entry: Option<&crate::models::remote_catalog::RemoteCatalogModel>,
) -> Option<String> {
    metadata
        .map(|m| m.quantization_type.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            catalog_entry
                .map(|model| model.file.clone())
                .and_then(|file| {
                    let quant = file
                        .strip_suffix(".gguf")
                        .map(crate::models::inventory::derive_quantization_type)
                        .filter(|value| !value.is_empty())?;
                    Some(quant)
                })
        })
}

fn model_source_page_url(
    identity: Option<&mesh::ServedModelIdentity>,
    catalog_entry: Option<&crate::models::remote_catalog::RemoteCatalogModel>,
    local_known: bool,
) -> Option<String> {
    identity
        .and_then(source_page_url_from_identity)
        .or_else(|| {
            if local_known {
                catalog_entry.map(|model| format!("https://huggingface.co/{}", model.source_repo()))
            } else {
                None
            }
        })
}

fn build_llama_runtime_items(
    metrics: &RuntimeLlamaMetricsSnapshot,
    slots: &RuntimeLlamaSlotsSnapshot,
) -> RuntimeLlamaRuntimeItems {
    let slot_items = slots
        .slots
        .iter()
        .enumerate()
        .map(|(index, slot)| RuntimeLlamaSlotItem {
            index,
            id: slot.id,
            id_task: slot.id_task,
            n_ctx: slot.n_ctx,
            is_processing: slot.is_processing.unwrap_or(false),
        })
        .collect::<Vec<_>>();
    RuntimeLlamaRuntimeItems {
        metrics: metrics
            .samples
            .iter()
            .map(|sample| RuntimeLlamaMetricItem {
                name: sample.name.clone(),
                labels: sample.labels.clone(),
                value: sample.value,
            })
            .collect(),
        slots_total: slot_items.len(),
        slots_busy: slot_items.iter().filter(|slot| slot.is_processing).count(),
        slots: slot_items,
    }
}

fn build_llama_runtime_snapshot(
    metrics: &RuntimeLlamaMetricsSnapshot,
    slots: RuntimeLlamaSlotsSnapshot,
) -> RuntimeLlamaRuntimeSnapshot {
    RuntimeLlamaRuntimeSnapshot {
        items: build_llama_runtime_items(metrics, &slots),
        metrics: metrics.clone(),
        slots,
    }
}

fn should_replace_model_runtime_projection(
    current: Option<&RuntimeLlamaRuntimeSnapshot>,
    next: &RuntimeLlamaRuntimeSnapshot,
) -> bool {
    let Some(current) = current else {
        return true;
    };
    if current == next {
        return false;
    }
    if current.slots.instance_id == next.slots.instance_id {
        return true;
    }
    matches!(
        (current.slots.status, next.slots.status),
        (
            super::RuntimeLlamaEndpointStatus::Unavailable,
            super::RuntimeLlamaEndpointStatus::Ready
        )
    )
}

fn select_runtime_llama_projection(
    runtime_status: &RuntimeStatusSnapshot,
) -> RuntimeLlamaRuntimeSnapshot {
    if let Some(primary_ready) = runtime_status.primary_model.as_ref().and_then(|model| {
        runtime_status
            .llama_runtime_by_instance
            .values()
            .find(|snapshot| {
                snapshot.slots.model.as_deref() == Some(model.as_str())
                    && snapshot.slots.status == super::RuntimeLlamaEndpointStatus::Ready
            })
    }) {
        return primary_ready.clone();
    }

    if let Some(primary_ready) = runtime_status
        .primary_model
        .as_ref()
        .and_then(|model| runtime_status.llama_runtime_by_model.get(model))
        .filter(|snapshot| snapshot.slots.status == super::RuntimeLlamaEndpointStatus::Ready)
    {
        return primary_ready.clone();
    }

    if let Some((_, ready)) = runtime_status
        .llama_runtime_by_instance
        .iter()
        .find(|(_, snapshot)| snapshot.slots.status == super::RuntimeLlamaEndpointStatus::Ready)
    {
        return ready.clone();
    }

    if let Some((_, ready)) = runtime_status
        .llama_runtime_by_model
        .iter()
        .find(|(_, snapshot)| snapshot.slots.status == super::RuntimeLlamaEndpointStatus::Ready)
    {
        return ready.clone();
    }

    if let Some(primary) = runtime_status
        .primary_model
        .as_ref()
        .and_then(|model| runtime_status.llama_runtime_by_model.get(model))
    {
        return primary.clone();
    }

    if let Some(primary) = runtime_status.primary_model.as_ref().and_then(|model| {
        runtime_status
            .llama_runtime_by_instance
            .values()
            .find(|snapshot| snapshot.slots.model.as_deref() == Some(model.as_str()))
    }) {
        return primary.clone();
    }

    if let Some((_, snapshot)) = runtime_status.llama_runtime_by_instance.iter().next() {
        return snapshot.clone();
    }

    runtime_status
        .llama_runtime_by_model
        .iter()
        .next()
        .map(|(_, snapshot)| snapshot.clone())
        .unwrap_or_else(|| runtime_status.llama_runtime.clone())
}

struct RuntimeStatusDerivationInput<'a> {
    is_client: bool,
    is_host: bool,
    llama_ready: bool,
    local_processes: &'a [crate::api::RuntimeProcessPayload],
    hosted_models: &'a [String],
    serving_models: &'a [String],
    model_name: &'a str,
    api_port: u16,
}

fn derive_runtime_status(input: RuntimeStatusDerivationInput<'_>) -> RuntimeStatusDerivation {
    let has_local_processes = !input.local_processes.is_empty();
    let effective_llama_ready = input.llama_ready || has_local_processes;
    let effective_is_host = input.is_host || has_local_processes;
    let display_model_name = input
        .local_processes
        .first()
        .map(|process| process.name.clone())
        .or_else(|| input.hosted_models.first().cloned())
        .or_else(|| input.serving_models.first().cloned())
        .unwrap_or_else(|| input.model_name.to_string());
    let has_local_worker_activity = has_local_processes || !input.hosted_models.is_empty();
    let node_state = derive_local_node_state(
        input.is_client,
        effective_is_host,
        effective_llama_ready,
        has_local_worker_activity,
        &display_model_name,
    );
    let launch_pi = if effective_llama_ready {
        Some(format!(
            "mesh-llm pi --host 127.0.0.1:{} --model {}",
            input.api_port,
            single_quote_shell_arg(&display_model_name)
        ))
    } else {
        None
    };
    let launch_goose = if effective_llama_ready {
        let api_port = input.api_port;
        Some(format!(
            "GOOSE_PROVIDER=openai OPENAI_HOST=http://localhost:{api_port} OPENAI_API_KEY=mesh GOOSE_MODEL={display_model_name} goose session"
        ))
    } else {
        None
    };

    RuntimeStatusDerivation {
        effective_is_host,
        effective_llama_ready,
        display_model_name,
        node_state,
        node_status: node_state.node_status_alias().to_string(),
        launch_pi,
        launch_goose,
    }
}

fn single_quote_shell_arg(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn derive_local_node_state(
    is_client: bool,
    effective_is_host: bool,
    effective_llama_ready: bool,
    has_local_worker_activity: bool,
    display_model_name: &str,
) -> NodeState {
    let has_declared_local_serving_work =
        (effective_is_host || has_local_worker_activity) && !display_model_name.trim().is_empty();

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

fn build_peer_payload(peer: &mesh::PeerInfo) -> PeerPayload {
    let display_latency = peer.display_latency();
    PeerPayload {
        id: peer.id.fmt_short().to_string(),
        owner: build_ownership_payload(&peer.owner_summary),
        release_attestation: peer.release_attestation_summary.clone(),
        role: match peer.role {
            mesh::NodeRole::Worker => "Worker".into(),
            mesh::NodeRole::Host { .. } => "Host".into(),
            mesh::NodeRole::Client => "Client".into(),
        },
        state: derive_peer_state(peer),
        models: peer.models.clone(),
        available_models: peer.available_models.clone(),
        requested_models: peer.requested_models.clone(),
        vram_gb: peer.vram_bytes as f64 / 1e9,
        serving_models: peer.serving_models.clone(),
        hosted_models: peer.hosted_models.clone(),
        hosted_models_known: peer.hosted_models_known,
        advertised_model_throughput: peer.advertised_model_throughput.clone(),
        version: peer.version.clone(),
        rtt_ms: peer.rtt_ms,
        latency_ms: display_latency.latency_ms,
        latency_source: Some(match display_latency.source {
            mesh::DisplayLatencySource::Direct => LatencySource::Direct,
            mesh::DisplayLatencySource::Estimated => LatencySource::Estimated,
            mesh::DisplayLatencySource::Unknown => LatencySource::Unknown,
        }),
        latency_age_ms: Some(display_latency.age_ms),
        latency_observer_id: display_latency
            .observer_id
            .as_ref()
            .map(|id| id.fmt_short().to_string()),
        hostname: peer.hostname.clone(),
        is_soc: peer.is_soc,
        gpus: build_gpus(
            peer.gpu_name.as_deref(),
            peer.gpu_vram.as_deref(),
            peer.gpu_reserved_bytes.as_deref(),
            peer.gpu_mem_bandwidth_gbps.as_deref(),
            peer.gpu_compute_tflops_fp32.as_deref(),
            peer.gpu_compute_tflops_fp16.as_deref(),
        ),
        first_joined_mesh_ts: peer.first_joined_mesh_ts,
    }
}

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

fn build_local_instances(
    snapshots: Vec<crate::runtime::instance::LocalInstanceSnapshot>,
    api_port: u16,
    version: &str,
) -> Vec<LocalInstance> {
    let mut instances: Vec<LocalInstance> = snapshots
        .iter()
        .map(|snapshot| LocalInstance {
            pid: snapshot.pid,
            api_port: snapshot.api_port,
            version: snapshot.version.clone(),
            started_at_unix: snapshot.started_at_unix,
            runtime_dir: snapshot.runtime_dir.to_string_lossy().to_string(),
            is_self: snapshot.is_self,
        })
        .collect();

    if instances.is_empty() {
        instances.push(LocalInstance {
            pid: std::process::id(),
            api_port: Some(api_port),
            version: Some(version.to_string()),
            started_at_unix: 0,
            runtime_dir: String::new(),
            is_self: true,
        });
    }

    instances
}

fn find_catalog_model(name: &str) -> Option<crate::models::remote_catalog::RemoteCatalogModel> {
    crate::models::remote_catalog::find_loaded_model_exact(name)
}

fn is_huggingface_repository_like(repository: &str) -> bool {
    let trimmed = repository.trim();
    !trimmed.is_empty()
        && !trimmed.starts_with('/')
        && !trimmed.ends_with('/')
        && !trimmed.contains('\\')
        && trimmed.split('/').count() == 2
}

fn huggingface_repository_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    matches!(identity.source_kind, mesh::ModelSourceKind::HuggingFace)
        .then(|| {
            identity
                .repository
                .clone()
                .filter(|repo| is_huggingface_repository_like(repo))
        })
        .flatten()
}

fn source_page_url_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    huggingface_repository_from_identity(identity)
        .map(|repository| format!("https://huggingface.co/{repository}"))
}

fn source_file_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    identity
        .artifact
        .clone()
        .or_else(|| identity.local_file_name.clone())
}

fn likely_reasoning_model(name: &str, description: Option<&str>) -> bool {
    let haystack = format!("{} {}", name, description.unwrap_or_default()).to_ascii_lowercase();
    ["reasoning", "thinking", "deepseek-r1"]
        .iter()
        .any(|needle| haystack.contains(needle))
}

fn likely_vision_model(name: &str, description: Option<&str>) -> bool {
    let haystack = format!("{} {}", name, description.unwrap_or_default()).to_ascii_lowercase();
    ["vision", "-vl", "llava", "omni", "qwen2.5-vl", "mllama"]
        .iter()
        .any(|needle| haystack.contains(needle))
}

fn likely_audio_model(name: &str, description: Option<&str>) -> bool {
    let haystack = format!("{} {}", name, description.unwrap_or_default()).to_ascii_lowercase();
    [
        "audio",
        "speech",
        "voice",
        "omni",
        "ultravox",
        "qwen2-audio",
    ]
    .iter()
    .any(|needle| haystack.contains(needle))
}

fn fit_hint_for_machine(size_gb: f64, my_vram_gb: f64) -> (String, String) {
    if size_gb <= 0.0 || my_vram_gb <= 0.0 {
        return (
            "Unknown".into(),
            "No local capacity signal is available for this machine yet.".into(),
        );
    }
    if size_gb * 1.2 <= my_vram_gb {
        return (
            "Likely comfortable".into(),
            format!(
                "This machine has {:.1} GB capacity, which should handle a {:.1} GB model comfortably.",
                my_vram_gb, size_gb
            ),
        );
    }
    if size_gb * 1.05 <= my_vram_gb {
        return (
            "Likely fits".into(),
            format!(
                "This machine has {:.1} GB capacity. A {:.1} GB model should fit, but headroom will be tight.",
                my_vram_gb, size_gb
            ),
        );
    }
    if size_gb * 0.8 <= my_vram_gb {
        return (
            "Possible with tradeoffs".into(),
            format!(
                "This machine has {:.1} GB capacity. A {:.1} GB model may load, but expect tighter memory pressure.",
                my_vram_gb, size_gb
            ),
        );
    }
    (
        "Likely too large".into(),
        format!(
            "This machine has {:.1} GB capacity, which is likely not enough for a {:.1} GB model locally.",
            my_vram_gb, size_gb
        ),
    )
}

fn http_route_stats(
    model_name: &str,
    peers: &[mesh::PeerInfo],
    my_hosted_models: &[String],
    my_hostname: Option<&str>,
    my_vram_gb: f64,
) -> ModelRouteStats {
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

    ModelRouteStats {
        node_count,
        active_nodes,
        mesh_vram_gb,
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeDataCollector;
    use crate::models::LocalModelInventorySnapshot;
    use crate::runtime_data::RuntimeDataDirty;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use std::collections::HashSet;

    /// The shadow value must be copied while the legacy snapshot write still
    /// owns the write lock. The hook represents a concurrent legacy publisher
    /// that runs after that lock is released but before the first publisher's
    /// reducer comparison; rereading the snapshot at that point would compare
    /// the wrong write.
    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn shadow_compare_uses_the_snapshot_from_its_own_write() {
        crate::runtime_events::clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        crate::runtime_events::install_runtime_event_engine(engine.clone());

        let collector = RuntimeDataCollector::new();
        let concurrent_collector = collector.clone();
        collector.shared.set_shadow_test_hook(move || {
            concurrent_collector.update_runtime_status(RuntimeDataDirty::STATUS, |status| {
                status.llama_ready = false;
                true
            });
        });

        let before = engine.health().snapshot().event_cutover_divergence;
        assert!(
            collector.update_runtime_status(RuntimeDataDirty::STATUS, |status| {
                status.llama_ready = true;
                true
            })
        );

        assert_eq!(
            engine.health().snapshot().event_cutover_divergence,
            before + 1,
            "the first writer's true value must be compared even after a concurrent writer replaces it"
        );
        assert!(
            !collector.runtime_status_snapshot().llama_ready,
            "the concurrent writer should still be visible in the final legacy snapshot"
        );

        let concurrent_collector = collector.clone();
        collector.shared.set_shadow_test_hook(move || {
            concurrent_collector.replace_local_inventory_snapshot(LocalModelInventorySnapshot {
                model_names: HashSet::new(),
                ..LocalModelInventorySnapshot::default()
            });
        });
        let before = engine.health().snapshot().event_cutover_divergence;
        assert!(
            collector.replace_local_inventory_snapshot(LocalModelInventorySnapshot {
                model_names: HashSet::from(["first".to_string()]),
                ..LocalModelInventorySnapshot::default()
            })
        );

        assert_eq!(
            engine.health().snapshot().event_cutover_divergence,
            before + 1,
            "the first inventory writer's model set must be compared after a concurrent replacement"
        );
        assert_eq!(
            collector.local_inventory_snapshot().model_names,
            HashSet::new()
        );

        crate::runtime_events::clear_runtime_event_engine();
    }
}
