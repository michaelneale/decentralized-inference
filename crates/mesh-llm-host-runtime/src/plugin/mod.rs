mod channel_broadcast;
mod config;
mod health;
mod installed;
pub(crate) mod mcp;
pub mod openai_exchange;
mod runtime;
mod schema_validation;
pub(crate) mod shared_endpoint;
pub(crate) mod stapler;
mod startup;
#[cfg(test)]
mod startup_tests;
mod support;
mod transport;
mod types;
mod web_ui;

pub(crate) use self::types::BridgeFuture;
pub use self::types::{
    InferenceEndpointRoute, PluginCapabilityProvider, PluginEndpointSummary,
    PluginManifestOverview, PluginMeshEvent, PluginRpcBridge, PluginSummary, RpcResult,
    ToolCallResult, ToolSummary,
};
pub use self::web_ui::{
    PluginWebUiConfigSectionOverview, PluginWebUiManifestOverview, PluginWebUiPageOverview,
    PluginWebUiState, PluginWebUiStateKind,
};
pub(crate) use self::web_ui::{PluginWebUiStateInput, derive_plugin_web_ui_state};
use self::web_ui::{
    inactive_web_ui_state, installed_plugin_metadata, plugin_web_ui_manifest_overview_from_proto,
};
#[cfg(test)]
use self::web_ui::{installed_metadata_with_web_ui, projected_existing_web_ui_state};

use crate::runtime_data::{
    PluginDataKey, PluginEndpointKey, RuntimeDataCollector, RuntimeDataSource,
};
use anyhow::{Context, Result, anyhow, bail};
pub use mesh_llm_plugin::proto;
use rmcp::model::ServerInfo;
use rmcp::model::{
    CompleteRequestParams, CompleteResult, GetPromptRequestParams, GetPromptResult,
    ReadResourceRequestParams, ReadResourceResult,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::BTreeSet;
#[cfg(test)]
use std::future::Future;
#[cfg(test)]
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::{Mutex, mpsc};

pub use self::config::ExternalPluginSpec;
pub(crate) use self::config::{
    BoolOrAuto, HardwareConfig, IntegerOrString, ModelConfigDefaults, ModelFitConfig,
    ReasoningBudget, ReasoningEnabled, RequestDefaultsConfig, SkippyConfig, StringOrStringList,
    ThroughputConfig,
};
pub use self::config::{
    ConfigEditor, ConfigStore, GpuAssignment, GpuConfig, LocalServingNodeConfig, MeshConfig,
    MeshRequirementsConfig, ModelConfigEditor, ModelConfigEntry, ModelDefaultsEditor,
    ModelRuntimeKind, OwnerControlConfig, PluginConfigEditor, PluginConfigEntry, PluginHostMode,
    PluginStartupConfig, PluginWebUiPreference, ResolvedPlugins, SpeculativeConfig,
    TelemetryConfig, TelemetryMetricsConfig, bundled_cli_plugin_spec, config_path, config_to_toml,
    load_config, parse_config_toml, resolve_plugins, validate_config_file,
};
#[cfg(test)]
pub(crate) use self::config::{
    assert_mesh_requirements_config_accepts_unset_min_only_max_only_and_full_ranges,
    assert_mesh_requirements_config_rejects_non_ed25519_signer_key,
    assert_mesh_requirements_config_rejects_required_attestation_without_signer_keys,
};
pub(crate) use self::config::{
    mesh_requirements_config_from_runtime, mesh_requirements_config_to_runtime,
    mesh_requirements_validation_error, validate_config_diagnostics_with_installed_plugin_schemas,
};
use self::health::EndpointHealthState;
#[cfg(test)]
use self::health::{endpoint_declared_capabilities, endpoint_record_from_plugin_status};
use self::runtime::ExternalPlugin;
pub use self::startup::{PluginStartupOptions, PluginStartupSummary};
pub(crate) use self::support::parse_optional_json;
use self::support::{format_args_for_log, format_slice_for_log, format_tool_names_for_log};
#[cfg(all(test, unix))]
use self::transport::unix_socket_path;
#[cfg(all(test, windows))]
use self::transport::windows_pipe_name;
pub(crate) use self::transport::{
    LocalListener, LocalStream, bind_local_listener, connect_side_stream, make_instance_id,
};
#[cfg(test)]
use mesh_llm_plugin::MeshVisibility;
#[cfg(test)]
use mesh_llm_plugin_manager::store::InstalledPluginWebUiValidationStatus;

pub const BLOBSTORE_PLUGIN_ID: &str = "blobstore";
pub(crate) const PROTOCOL_VERSION: u32 = mesh_llm_plugin::PROTOCOL_VERSION;
const REQUEST_TIMEOUT_SECS: u64 = 30;
#[cfg(test)]
type TestStreamFuture = Pin<Box<dyn Future<Output = Result<LocalStream>> + Send>>;
#[cfg(test)]
type TestStreamHandler = Arc<dyn Fn(proto::OpenStreamRequest) -> TestStreamFuture + Send + Sync>;

#[derive(Clone)]
pub struct PluginManager {
    pub(in crate::plugin) inner: Arc<PluginManagerInner>,
}

pub(in crate::plugin) struct PluginManagerInner {
    pub(in crate::plugin) plugins: BTreeMap<String, ExternalPlugin>,
    pub(in crate::plugin) inactive: BTreeMap<String, PluginSummary>,
    /// One already-running upstream this node serves without a plugin process.
    /// Fixed for the manager's lifetime; its health and models live in
    /// `endpoint_health` alongside plugin endpoints.
    pub(in crate::plugin) shared_endpoint: Option<String>,
    pub(in crate::plugin) endpoint_health: Arc<Mutex<BTreeMap<String, EndpointHealthState>>>,
    pub(in crate::plugin) runtime_data: RuntimeDataCollector,
    pub(in crate::plugin) rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    pub(in crate::plugin) shutting_down: AtomicBool,
    #[cfg(test)]
    pub(in crate::plugin) bridged_plugins: BTreeSet<String>,
    #[cfg(test)]
    pub(in crate::plugin) test_endpoints: Arc<Mutex<Vec<PluginEndpointSummary>>>,
    #[cfg(test)]
    pub(in crate::plugin) test_inference_endpoints: Arc<Mutex<Vec<InferenceEndpointRoute>>>,
    #[cfg(test)]
    pub(in crate::plugin) test_manifests: Arc<Mutex<BTreeMap<String, proto::PluginManifest>>>,
    #[cfg(test)]
    pub(in crate::plugin) test_stream_handlers: Arc<Mutex<BTreeMap<String, TestStreamHandler>>>,
}

impl PluginManager {
    pub async fn start(
        specs: &ResolvedPlugins,
        host_mode: PluginHostMode,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
    ) -> Result<Self> {
        Self::log_startup_plan(specs);

        let rpc_bridge = Arc::new(Mutex::new(None));
        let runtime_data = RuntimeDataCollector::new();
        let instance_id = make_instance_id();
        let (plugins, failed_plugins) = Self::load_external_plugins(
            specs,
            host_mode,
            mesh_tx,
            instance_id,
            rpc_bridge.clone(),
            &runtime_data,
        )
        .await?;
        let manager = Self {
            inner: Arc::new(PluginManagerInner {
                plugins,
                inactive: Self::inactive_plugins(specs, failed_plugins),
                shared_endpoint: specs.shared_endpoint.clone(),
                endpoint_health: Arc::new(Mutex::new(BTreeMap::new())),
                runtime_data,
                rpc_bridge,
                shutting_down: AtomicBool::new(false),
                #[cfg(test)]
                bridged_plugins: BTreeSet::new(),
                #[cfg(test)]
                test_endpoints: Arc::new(Mutex::new(Vec::new())),
                #[cfg(test)]
                test_inference_endpoints: Arc::new(Mutex::new(Vec::new())),
                #[cfg(test)]
                test_manifests: Arc::new(Mutex::new(BTreeMap::new())),
                #[cfg(test)]
                test_stream_handlers: Arc::new(Mutex::new(BTreeMap::new())),
            }),
        };
        for summary in manager.inner.inactive.values().cloned() {
            manager.publish_plugin_summary(&summary);
            manager.publish_plugin_manifest(&summary.name, None);
            manager.publish_plugin_providers(&summary.name, Vec::new());
        }
        let plugin_names = manager.inner.plugins.keys().cloned().collect::<Vec<_>>();
        for plugin_name in plugin_names {
            manager.refresh_plugin_endpoints(&plugin_name).await?;
        }
        manager.start_supervisor();
        Ok(manager)
    }

    fn log_startup_plan(specs: &ResolvedPlugins) {
        Self::log_inactive_plugins(&specs.inactive);
        if specs.externals.is_empty() {
            tracing::info!("Plugin manager: no plugins enabled");
            return;
        }

        Self::log_enabled_plugins(&specs.externals);
    }

    fn log_inactive_plugins(inactive: &[PluginSummary]) {
        for summary in inactive {
            tracing::warn!(
                plugin = %summary.name,
                status = %summary.status,
                error = %summary.error.as_deref().unwrap_or(""),
                "Plugin inactive at startup"
            );
        }
    }

    fn log_enabled_plugins(externals: &[ExternalPluginSpec]) {
        let names = externals
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        tracing::info!(
            "Plugin manager: loading {} plugin(s): {}",
            externals.len(),
            names
        );
    }

    fn summary_runtime_source(plugin_name: String) -> RuntimeDataSource {
        RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: Some(PluginDataKey {
                plugin_name,
                data_key: "summary".into(),
            }),
            plugin_endpoint_key: None,
        }
    }

    async fn load_external_plugins(
        specs: &ResolvedPlugins,
        host_mode: PluginHostMode,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
        instance_id: String,
        rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
        runtime_data: &RuntimeDataCollector,
    ) -> Result<(BTreeMap<String, ExternalPlugin>, Vec<PluginSummary>)> {
        let mut plugins = BTreeMap::new();
        let mut failed = Vec::new();
        for spec in &specs.externals {
            match Self::load_external_plugin(
                spec,
                host_mode,
                mesh_tx.clone(),
                instance_id.clone(),
                rpc_bridge.clone(),
                runtime_data,
            )
            .await
            {
                Ok(plugin) => {
                    plugins.insert(spec.name.clone(), plugin);
                }
                Err(error) => {
                    if spec.startup.optional {
                        failed.push(Self::plugin_load_failure_summary(spec, &error));
                    } else {
                        for plugin in plugins.values() {
                            plugin.shutdown().await;
                        }
                        return Err(error).with_context(|| {
                            format!("Required plugin '{}' failed to start", spec.name)
                        });
                    }
                }
            }
        }
        Ok((plugins, failed))
    }

    async fn load_external_plugin(
        spec: &ExternalPluginSpec,
        host_mode: PluginHostMode,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
        instance_id: String,
        rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
        runtime_data: &RuntimeDataCollector,
    ) -> Result<ExternalPlugin> {
        tracing::info!(
            plugin = %spec.name,
            command = %spec.command,
            args = %format_args_for_log(&spec.args),
            "Loading plugin"
        );
        let plugin = ExternalPlugin::spawn(
            spec,
            instance_id,
            host_mode,
            mesh_tx,
            rpc_bridge,
            runtime_data.producer(Self::summary_runtime_source(spec.name.clone())),
        )
        .await
        .map_err(|err| {
            tracing::error!(
                plugin = %spec.name,
                error = %err,
                "Plugin failed to load"
            );
            err
        })?;

        let summary = plugin.summary().await;
        tracing::info!(
            plugin = %summary.name,
            version = %summary.version.as_deref().unwrap_or("unknown"),
            capabilities = %format_slice_for_log(&summary.capabilities),
            tools = %format_tool_names_for_log(&summary.tools),
            "Plugin loaded successfully"
        );
        Ok(plugin)
    }

    fn plugin_load_failure_summary(
        spec: &ExternalPluginSpec,
        error: &anyhow::Error,
    ) -> PluginSummary {
        let error_message = crate::logging::policy::redact_urls_in_text(&error.to_string());
        tracing::warn!(
            plugin = %spec.name,
            command = %spec.command,
            args = %format_args_for_log(&spec.args),
            error = %error_message,
            "Plugin disabled after load failure"
        );
        PluginSummary {
            name: spec.name.clone(),
            kind: "external".to_string(),
            enabled: false,
            status: "error".to_string(),
            pid: None,
            version: None,
            capabilities: Vec::new(),
            command: (!spec.command.is_empty()).then(|| spec.command.clone()),
            args: spec.args.clone(),
            tools: Vec::new(),
            manifest: None,
            web_ui: derive_plugin_web_ui_state(PluginWebUiStateInput {
                plugin_name: &spec.name,
                live_manifest: None,
                installed_metadata: spec.installed_metadata.as_ref(),
                web_ui_enabled: spec.web_ui_enabled,
                runtime_available: false,
                runtime_unavailable_reason: Some(&error_message),
            }),
            startup: Some(spec.startup.summary()),
            error: Some(error_message),
        }
    }

    fn inactive_plugins(
        specs: &ResolvedPlugins,
        failed_plugins: Vec<PluginSummary>,
    ) -> BTreeMap<String, PluginSummary> {
        specs
            .inactive
            .iter()
            .cloned()
            .chain(failed_plugins)
            .map(|summary| (summary.name.clone(), summary))
            .collect()
    }

    #[cfg(test)]
    pub fn for_test_bridge(plugin_names: &[&str], bridge: Arc<dyn PluginRpcBridge>) -> Self {
        Self {
            inner: Arc::new(PluginManagerInner {
                plugins: BTreeMap::new(),
                inactive: BTreeMap::new(),
                shared_endpoint: None,
                endpoint_health: Arc::new(Mutex::new(BTreeMap::new())),
                runtime_data: RuntimeDataCollector::new(),
                rpc_bridge: Arc::new(Mutex::new(Some(bridge))),
                shutting_down: AtomicBool::new(false),
                bridged_plugins: plugin_names
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
                test_endpoints: Arc::new(Mutex::new(Vec::new())),
                test_inference_endpoints: Arc::new(Mutex::new(Vec::new())),
                test_manifests: Arc::new(Mutex::new(BTreeMap::new())),
                test_stream_handlers: Arc::new(Mutex::new(BTreeMap::new())),
            }),
        }
    }

    /// A manager whose only inference source is one shared upstream.
    #[cfg(test)]
    pub(crate) fn for_test_shared_endpoint(address: &str) -> Self {
        Self {
            inner: Arc::new(PluginManagerInner {
                plugins: BTreeMap::new(),
                inactive: BTreeMap::new(),
                shared_endpoint: Some(address.to_string()),
                endpoint_health: Arc::new(Mutex::new(BTreeMap::new())),
                runtime_data: RuntimeDataCollector::new(),
                rpc_bridge: Arc::new(Mutex::new(None)),
                shutting_down: AtomicBool::new(false),
                bridged_plugins: BTreeSet::new(),
                test_endpoints: Arc::new(Mutex::new(Vec::new())),
                test_inference_endpoints: Arc::new(Mutex::new(Vec::new())),
                test_manifests: Arc::new(Mutex::new(BTreeMap::new())),
                test_stream_handlers: Arc::new(Mutex::new(BTreeMap::new())),
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test_summaries(summaries: Vec<PluginSummary>) -> Self {
        Self {
            inner: Arc::new(PluginManagerInner {
                plugins: BTreeMap::new(),
                inactive: summaries
                    .into_iter()
                    .map(|summary| (summary.name.clone(), summary))
                    .collect(),
                shared_endpoint: None,
                endpoint_health: Arc::new(Mutex::new(BTreeMap::new())),
                runtime_data: RuntimeDataCollector::new(),
                rpc_bridge: Arc::new(Mutex::new(None)),
                shutting_down: AtomicBool::new(false),
                bridged_plugins: BTreeSet::new(),
                test_endpoints: Arc::new(Mutex::new(Vec::new())),
                test_inference_endpoints: Arc::new(Mutex::new(Vec::new())),
                test_manifests: Arc::new(Mutex::new(BTreeMap::new())),
                test_stream_handlers: Arc::new(Mutex::new(BTreeMap::new())),
            }),
        }
    }

    pub(in crate::plugin) fn plugin_summary_producer(
        &self,
        plugin_name: &str,
    ) -> crate::runtime_data::RuntimeDataProducer {
        self.inner.runtime_data.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: Some(PluginDataKey {
                plugin_name: plugin_name.to_string(),
                data_key: "summary".into(),
            }),
            plugin_endpoint_key: None,
        })
    }

    pub(in crate::plugin) fn plugin_endpoint_producer(
        &self,
        plugin_name: &str,
        endpoint_id: &str,
    ) -> crate::runtime_data::RuntimeDataProducer {
        self.inner.runtime_data.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: None,
            plugin_endpoint_key: Some(PluginEndpointKey {
                plugin_name: plugin_name.to_string(),
                endpoint_id: endpoint_id.to_string(),
            }),
        })
    }

    pub(in crate::plugin) fn publish_plugin_summary(&self, summary: &PluginSummary) {
        self.plugin_summary_producer(&summary.name)
            .publish_plugin_summary(summary.clone());
    }

    pub(in crate::plugin) fn publish_plugin_manifest(
        &self,
        plugin_name: &str,
        manifest: Option<PluginManifestOverview>,
    ) {
        if let Some(manifest) = manifest {
            self.plugin_summary_producer(plugin_name)
                .publish_plugin_manifest(manifest);
        }
    }

    pub(in crate::plugin) fn publish_plugin_providers(
        &self,
        plugin_name: &str,
        providers: Vec<PluginCapabilityProvider>,
    ) {
        self.plugin_summary_producer(plugin_name)
            .publish_plugin_providers(providers);
    }

    pub async fn list(&self) -> Vec<PluginSummary> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.inactive.is_empty() {
            let manifests = self.inner.test_manifests.lock().await.clone();
            if !manifests.is_empty() {
                let mut summaries = manifests
                    .into_iter()
                    .map(|(name, manifest)| PluginSummary {
                        name: name.clone(),
                        kind: "bridge".into(),
                        enabled: true,
                        status: "running".into(),
                        pid: None,
                        version: None,
                        capabilities: manifest.capabilities.clone(),
                        command: None,
                        args: Vec::new(),
                        tools: Vec::new(),
                        manifest: Some(plugin_manifest_overview(&manifest)),
                        web_ui: derive_plugin_web_ui_state(PluginWebUiStateInput {
                            plugin_name: &name,
                            live_manifest: Some(&manifest),
                            installed_metadata: None,
                            web_ui_enabled: None,
                            runtime_available: true,
                            runtime_unavailable_reason: None,
                        }),
                        startup: None,
                        error: None,
                    })
                    .collect::<Vec<_>>();
                summaries.sort_by(|a, b| a.name.cmp(&b.name));
                return summaries;
            }
        }
        let mut summaries = self.inner.runtime_data.plugins_snapshot().plugins;
        if !summaries.is_empty() {
            summaries.sort_by(|a, b| a.name.cmp(&b.name));
            return summaries;
        }
        let mut summaries =
            Vec::with_capacity(self.inner.plugins.len() + self.inner.inactive.len());
        for plugin in self.inner.plugins.values() {
            summaries.push(plugin.summary().await);
        }
        summaries.extend(self.inner.inactive.values().cloned());
        summaries.sort_by(|a, b| a.name.cmp(&b.name));
        summaries
    }

    pub async fn web_ui_state(&self, name: &str) -> Result<PluginWebUiState> {
        Ok(self.plugin_summary(name).await?.web_ui)
    }

    pub async fn set_web_ui_enabled(&self, name: &str, enabled: bool) -> Result<PluginWebUiState> {
        if let Some(plugin) = self.inner.plugins.get(name) {
            return Ok(plugin.set_web_ui_enabled(enabled).await);
        }

        if let Some(summary) = self.inner.inactive.get(name) {
            let web_ui = inactive_web_ui_state(summary, Some(enabled));
            let mut updated = summary.clone();
            updated.web_ui = web_ui.clone();
            self.publish_plugin_summary(&updated);
            return Ok(web_ui);
        }

        #[cfg(test)]
        if self.is_test_bridge_enabled(name) {
            let summary = self.plugin_summary(name).await?;
            let web_ui = projected_existing_web_ui_state(&summary, Some(enabled));
            let mut updated = summary;
            updated.web_ui = web_ui.clone();
            self.publish_plugin_summary(&updated);
            return Ok(web_ui);
        }

        anyhow::bail!("Unknown plugin '{name}'")
    }

    pub async fn web_ui_asset_root(&self, name: &str) -> Result<Option<std::path::PathBuf>> {
        if let Some(plugin) = self.inner.plugins.get(name) {
            return Ok(plugin.web_ui_asset_root());
        }
        if self.inner.inactive.contains_key(name) || self.is_test_bridge_enabled(name) {
            return Ok(installed_plugin_metadata(name).and_then(|metadata| {
                metadata
                    .web_ui_asset_root_path()
                    .filter(|asset_root| asset_root.is_dir())
            }));
        }
        anyhow::bail!("Unknown plugin '{name}'")
    }

    async fn plugin_summary(&self, name: &str) -> Result<PluginSummary> {
        #[cfg(test)]
        if self.is_test_bridge_enabled(name) {
            let _ = self.publish_test_bridge_snapshot(name).await;
        }
        self.list()
            .await
            .into_iter()
            .find(|summary| summary.name == name)
            .with_context(|| format!("Unknown plugin '{name}'"))
    }

    pub async fn shutdown(&self) {
        self.inner.shutting_down.store(true, Ordering::SeqCst);
        for plugin in self.inner.plugins.values() {
            plugin.shutdown().await;
        }
        self.inner.endpoint_health.lock().await.clear();
    }

    pub async fn endpoints(&self) -> Result<Vec<PluginEndpointSummary>> {
        #[cfg(test)]
        if self.inner.plugins.is_empty() && self.inner.inactive.is_empty() {
            let mut endpoints = self.inner.test_endpoints.lock().await.clone();
            endpoints.sort_by(|a, b| {
                a.plugin_name
                    .cmp(&b.plugin_name)
                    .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
            });
            if !endpoints.is_empty() {
                return Ok(endpoints);
            }
        }
        Ok(self.inner.runtime_data.plugins_snapshot().endpoints)
    }

    #[cfg(test)]
    pub async fn set_test_endpoints(&self, endpoints: Vec<PluginEndpointSummary>) {
        *self.inner.test_endpoints.lock().await = endpoints;
    }

    #[cfg(test)]
    pub fn set_test_capability_providers(&self, providers: Vec<PluginCapabilityProvider>) {
        self.publish_plugin_providers("__test__", providers);
    }

    #[cfg(test)]
    pub async fn set_test_inference_endpoints(&self, endpoints: Vec<InferenceEndpointRoute>) {
        *self.inner.test_inference_endpoints.lock().await = endpoints;
    }

    #[cfg(test)]
    pub async fn set_test_manifests(&self, manifests: BTreeMap<String, proto::PluginManifest>) {
        let plugin_names = manifests.keys().cloned().collect::<Vec<_>>();
        *self.inner.test_manifests.lock().await = manifests;
        for plugin_name in plugin_names {
            let _ = self.publish_test_bridge_snapshot(&plugin_name).await;
        }
    }

    #[cfg(test)]
    pub async fn publish_test_bridge_snapshot(&self, plugin_name: &str) -> Result<()> {
        let manifest = self
            .inner
            .test_manifests
            .lock()
            .await
            .get(plugin_name)
            .cloned()
            .with_context(|| format!("Unknown test bridge plugin '{plugin_name}'"))?;

        let summary = PluginSummary {
            name: plugin_name.to_string(),
            kind: "bridge".into(),
            enabled: true,
            status: "running".into(),
            pid: None,
            version: None,
            capabilities: manifest.capabilities.clone(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: Some(plugin_manifest_overview(&manifest)),
            web_ui: derive_plugin_web_ui_state(PluginWebUiStateInput {
                plugin_name,
                live_manifest: Some(&manifest),
                installed_metadata: None,
                web_ui_enabled: None,
                runtime_available: true,
                runtime_unavailable_reason: None,
            }),
            startup: None,
            error: None,
        };
        self.publish_plugin_summary(&summary);
        self.publish_plugin_manifest(plugin_name, summary.manifest.clone());
        let plugin_default = endpoint_record_from_plugin_status(&summary);

        let endpoint_summaries = manifest
            .endpoints
            .iter()
            .map(|endpoint| PluginEndpointSummary {
                plugin_name: plugin_name.to_string(),
                plugin_status: summary.status.clone(),
                endpoint_id: endpoint.endpoint_id.clone(),
                state: "configured".into(),
                available: false,
                kind: endpoint_kind_name(endpoint.kind).to_string(),
                transport_kind: endpoint_transport_kind_name(endpoint.transport_kind).to_string(),
                protocol: endpoint.protocol.clone(),
                address: endpoint.address.clone(),
                args: endpoint.args.clone(),
                namespace: endpoint.namespace.clone(),
                supports_streaming: endpoint.supports_streaming,
                managed_by_plugin: endpoint.managed_by_plugin,
                detail: None,
                models: Vec::new(),
            })
            .collect::<Vec<_>>();
        let mut providers = manifest
            .capabilities
            .iter()
            .map(|capability| PluginCapabilityProvider {
                capability: capability.clone(),
                plugin_name: plugin_name.to_string(),
                plugin_status: summary.status.clone(),
                endpoint_id: None,
                available: plugin_default.available,
                detail: plugin_default.detail.clone(),
            })
            .collect::<Vec<_>>();
        for endpoint in &manifest.endpoints {
            for capability in endpoint_declared_capabilities(endpoint) {
                providers.push(PluginCapabilityProvider {
                    capability,
                    plugin_name: plugin_name.to_string(),
                    plugin_status: summary.status.clone(),
                    endpoint_id: Some(endpoint.endpoint_id.clone()),
                    available: false,
                    detail: None,
                });
            }
        }
        self.publish_plugin_providers(plugin_name, providers);
        for endpoint_summary in endpoint_summaries {
            self.plugin_endpoint_producer(plugin_name, &endpoint_summary.endpoint_id)
                .publish_plugin_endpoint(endpoint_summary);
        }
        Ok(())
    }

    pub async fn tools(&self, name: &str) -> Result<Vec<ToolSummary>> {
        if let Some(summary) = self.inner.inactive.get(name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(name)
            .with_context(|| format!("Unknown plugin '{name}'"))?;
        plugin.list_tools().await
    }

    pub async fn call_tool(
        &self,
        plugin_name: &str,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        if self.is_test_bridge_enabled(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .with_context(|| format!("No bridge configured for test plugin '{plugin_name}'"))?;
            let arguments = parse_optional_json(arguments_json)?;
            let params_json = serde_json::to_string(&serde_json::json!({
                "name": tool_name,
                "arguments": arguments,
            }))
            .with_context(|| format!("Serialize tool call for test plugin '{plugin_name}'"))?;
            let result = bridge
                .handle_request(plugin_name.to_string(), "tools/call".into(), params_json)
                .await
                .map_err(|err| anyhow!("{}", err.message))?;
            let decoded: rmcp::model::CallToolResult = serde_json::from_str(&result.result_json)
                .with_context(|| format!("Decode tool result from test plugin '{plugin_name}'"))?;
            return Ok(ToolCallResult {
                content_json: normalize_test_tool_result_content(&decoded)?,
                is_error: decoded.is_error.unwrap_or(false),
            });
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.call_tool(tool_name, arguments_json).await
    }

    pub async fn call_tool_without_timeout(
        &self,
        plugin_name: &str,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        if self.is_test_bridge_enabled(plugin_name) {
            return self.call_tool(plugin_name, tool_name, arguments_json).await;
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin
            .call_tool_without_timeout(tool_name, arguments_json)
            .await
    }

    pub async fn invoke_operation(
        &self,
        plugin_name: &str,
        operation_name: &str,
        input_json: &str,
    ) -> Result<ToolCallResult> {
        self.call_tool(plugin_name, operation_name, input_json)
            .await
    }

    pub async fn invoke_operation_without_timeout(
        &self,
        plugin_name: &str,
        operation_name: &str,
        input_json: &str,
    ) -> Result<ToolCallResult> {
        self.call_tool_without_timeout(plugin_name, operation_name, input_json)
            .await
    }

    pub async fn inference_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();
        for endpoint in self.inference_endpoints().await? {
            models.extend(endpoint.models);
        }
        models.sort();
        models.dedup();
        Ok(models)
    }

    pub async fn inference_endpoint_for_model(
        &self,
        model: &str,
    ) -> Result<Option<InferenceEndpointRoute>> {
        let mut endpoints = self.inference_endpoints().await?;
        endpoints.sort_by(|a, b| {
            a.plugin_name
                .cmp(&b.plugin_name)
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(endpoints
            .into_iter()
            .find(|endpoint| endpoint.models.iter().any(|candidate| candidate == model)))
    }

    pub async fn capability_providers(&self) -> Result<Vec<PluginCapabilityProvider>> {
        Ok(self.inner.runtime_data.plugins_snapshot().providers)
    }

    pub async fn provider_for_capability(
        &self,
        capability: &str,
    ) -> Result<Option<PluginCapabilityProvider>> {
        let mut providers = self.capability_providers().await?;
        providers.sort_by(|a, b| {
            b.available
                .cmp(&a.available)
                .then_with(|| a.plugin_name.cmp(&b.plugin_name))
                .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
        });
        Ok(providers
            .into_iter()
            .find(|provider| provider.capability == capability))
    }

    pub async fn available_provider_for_capability(
        &self,
        capability: &str,
    ) -> Result<Option<PluginCapabilityProvider>> {
        Ok(self
            .provider_for_capability(capability)
            .await?
            .filter(|provider| provider.available))
    }

    pub async fn is_capability_available(&self, capability: &str) -> bool {
        self.available_provider_for_capability(capability)
            .await
            .ok()
            .flatten()
            .is_some()
    }

    pub async fn invoke_operation_by_capability(
        &self,
        capability: &str,
        operation_name: &str,
        input_json: &str,
    ) -> Result<ToolCallResult> {
        let provider = self
            .available_provider_for_capability(capability)
            .await?
            .ok_or_else(|| anyhow!("No provider for capability '{capability}'"))?;
        self.invoke_operation(&provider.plugin_name, operation_name, input_json)
            .await
    }

    pub async fn get_prompt(
        &self,
        plugin_name: &str,
        prompt_name: &str,
        params: GetPromptRequestParams,
    ) -> Result<GetPromptResult> {
        self.invoke_service_json(
            plugin_name,
            proto::ServiceKind::Prompt,
            prompt_name,
            &params,
        )
        .await
    }

    pub async fn read_resource(
        &self,
        plugin_name: &str,
        resource_uri: &str,
        params: ReadResourceRequestParams,
    ) -> Result<ReadResourceResult> {
        self.invoke_service_json(
            plugin_name,
            proto::ServiceKind::Resource,
            resource_uri,
            &params,
        )
        .await
    }

    pub async fn complete(
        &self,
        plugin_name: &str,
        argument_ref: &str,
        params: CompleteRequestParams,
    ) -> Result<CompleteResult> {
        self.invoke_service_json(
            plugin_name,
            proto::ServiceKind::Completion,
            argument_ref,
            &params,
        )
        .await
    }

    pub async fn mcp_request<T, P>(&self, plugin_name: &str, method: &str, params: P) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: Serialize,
    {
        if self.is_test_bridge_enabled(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .with_context(|| format!("No bridge configured for test plugin '{plugin_name}'"))?;
            let params_json = serde_json::to_string(&params)
                .with_context(|| format!("Serialize params for test plugin '{plugin_name}'"))?;
            let result = bridge
                .handle_request(plugin_name.to_string(), method.to_string(), params_json)
                .await
                .map_err(|err| anyhow!("{}", err.message))?;
            return serde_json::from_str(&result.result_json)
                .with_context(|| format!("Decode response from test plugin '{plugin_name}'"));
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.mcp_request(method, params).await
    }

    pub async fn mcp_notify<P>(&self, plugin_name: &str, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        if self.is_test_bridge_enabled(plugin_name) {
            let bridge = self
                .inner
                .rpc_bridge
                .lock()
                .await
                .clone()
                .with_context(|| format!("No bridge configured for test plugin '{plugin_name}'"))?;
            let params_json = serde_json::to_string(&params)
                .with_context(|| format!("Serialize params for test plugin '{plugin_name}'"))?;
            bridge
                .handle_notification(plugin_name.to_string(), method.to_string(), params_json)
                .await;
            return Ok(());
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.mcp_notify(method, params).await
    }

    async fn invoke_service_json<T, P>(
        &self,
        plugin_name: &str,
        kind: proto::ServiceKind,
        service_name: &str,
        params: &P,
    ) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: Serialize,
    {
        if self.is_test_bridge_enabled(plugin_name) {
            let method = match kind {
                proto::ServiceKind::Operation => "tools/call",
                proto::ServiceKind::Prompt => "prompts/get",
                proto::ServiceKind::Resource => "resources/read",
                proto::ServiceKind::Completion => "completion/complete",
                proto::ServiceKind::Unspecified => {
                    bail!("Service kind is required for test plugin '{plugin_name}'")
                }
            };
            if method == "tools/call" {
                let arguments = serde_json::to_value(params).with_context(|| {
                    format!("Serialize operation params for test plugin '{plugin_name}'")
                })?;
                return self
                    .mcp_request(
                        plugin_name,
                        method,
                        serde_json::json!({
                            "name": service_name,
                            "arguments": arguments,
                        }),
                    )
                    .await;
            }
            return self.mcp_request(plugin_name, method, params).await;
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        let input_json = serde_json::to_string(params)
            .with_context(|| format!("Serialize service params for plugin '{plugin_name}'"))?;
        let response = plugin
            .invoke_service(
                kind,
                service_name,
                &input_json,
                Some(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS)),
            )
            .await?;
        serde_json::from_str(&response.output_json).with_context(|| {
            format!(
                "Decode service response '{}' from plugin '{}'",
                service_name, plugin_name
            )
        })
    }

    fn is_test_bridge_enabled(&self, _plugin_name: &str) -> bool {
        #[cfg(test)]
        {
            return self.inner.bridged_plugins.contains(_plugin_name);
        }
        #[allow(unreachable_code)]
        false
    }

    pub async fn list_server_infos(&self) -> Vec<(String, ServerInfo)> {
        let mut infos = Vec::new();
        for (name, plugin) in &self.inner.plugins {
            if let Ok(info) = plugin.server_info().await {
                infos.push((name.clone(), info));
            }
        }
        infos
    }

    pub async fn manifest(&self, plugin_name: &str) -> Result<Option<proto::PluginManifest>> {
        if self.is_test_bridge_enabled(plugin_name) {
            #[cfg(test)]
            if let Some(manifest) = self
                .inner
                .test_manifests
                .lock()
                .await
                .get(plugin_name)
                .cloned()
            {
                return Ok(Some(manifest));
            }
            bail!(
                "Plugin '{}' does not expose a manifest in bridge mode",
                plugin_name
            );
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.manifest().await
    }

    /// Non-starting counterpart to [`Self::manifest`]: returns the plugin's
    /// already-cached manifest without calling `ensure_running`, so a
    /// delivery-filtering check (declares this channel? subscribes to this
    /// event?) never lazily starts a stopped plugin or blocks its caller on
    /// connection/init timeouts. `Plugin::shutdown` clears the cached
    /// manifest, so this doubles as an "already running" check: a stopped or
    /// never-started plugin has no cached manifest and returns `None` here.
    async fn manifest_snapshot(&self, plugin_name: &str) -> Option<proto::PluginManifest> {
        self.inner
            .plugins
            .get(plugin_name)?
            .manifest_snapshot()
            .await
    }

    pub async fn manifest_json(&self, plugin_name: &str) -> Result<Option<Value>> {
        Ok(self
            .manifest(plugin_name)
            .await?
            .as_ref()
            .map(plugin_manifest_to_json))
    }

    pub async fn set_rpc_bridge(&self, bridge: Option<Arc<dyn PluginRpcBridge>>) {
        *self.inner.rpc_bridge.lock().await = bridge;
    }

    #[cfg(test)]
    pub(crate) async fn set_test_stream_handler<F>(&self, plugin_name: &str, handler: F)
    where
        F: Fn(proto::OpenStreamRequest) -> TestStreamFuture + Send + Sync + 'static,
    {
        self.inner
            .test_stream_handlers
            .lock()
            .await
            .insert(plugin_name.to_string(), Arc::new(handler));
    }

    pub async fn dispatch_channel_message(&self, event: PluginMeshEvent) -> Result<()> {
        let PluginMeshEvent::Channel { plugin_id, message } = event else {
            bail!("expected plugin channel event");
        };
        if !self
            .plugin_declares_mesh_channel(&plugin_id, &message.channel)
            .await
        {
            tracing::debug!(
                plugin = %plugin_id,
                channel = %message.channel,
                "Dropping channel message for undeclared mesh channel"
            );
            return Ok(());
        }
        let Some(plugin) = self.inner.plugins.get(&plugin_id) else {
            tracing::debug!(
                "Dropping channel message for unloaded plugin '{}'",
                plugin_id
            );
            return Ok(());
        };
        plugin.send_channel_message(message).await
    }

    pub async fn dispatch_bulk_transfer_message(&self, event: PluginMeshEvent) -> Result<()> {
        let PluginMeshEvent::BulkTransfer { plugin_id, message } = event else {
            bail!("expected plugin bulk transfer event");
        };
        if !self
            .plugin_declares_mesh_channel(&plugin_id, &message.channel)
            .await
        {
            tracing::debug!(
                plugin = %plugin_id,
                channel = %message.channel,
                "Dropping bulk transfer for undeclared mesh channel"
            );
            return Ok(());
        }
        let Some(plugin) = self.inner.plugins.get(&plugin_id) else {
            tracing::debug!(
                "Dropping bulk transfer message for unloaded plugin '{}'",
                plugin_id
            );
            return Ok(());
        };
        plugin.send_bulk_transfer_message(message).await
    }

    pub async fn broadcast_mesh_event(&self, event: proto::MeshEvent) -> Result<()> {
        for (name, plugin) in &self.inner.plugins {
            if !self.plugin_subscribes_mesh_event(name, event.kind).await {
                continue;
            }
            plugin.send_mesh_event(event.clone()).await?;
        }
        Ok(())
    }

    /// Whether an already-running `plugin_name` subscribes to mesh event
    /// `kind`. Same non-starting contract as
    /// [`Self::plugin_declares_mesh_channel`], for [`Self::broadcast_mesh_event`].
    pub async fn plugin_subscribes_mesh_event(&self, plugin_name: &str, kind: i32) -> bool {
        self.manifest_snapshot(plugin_name)
            .await
            .is_some_and(|manifest| manifest_subscribes_mesh_event(&manifest, kind))
    }

    pub async fn open_stream(
        &self,
        plugin_name: &str,
        request: proto::OpenStreamRequest,
    ) -> Result<proto::OpenStreamResponse> {
        if self.is_test_bridge_enabled(plugin_name) {
            bail!(
                "Plugin '{}' does not support stream control in bridge mode",
                plugin_name
            );
        }
        if let Some(summary) = self.inner.inactive.get(plugin_name) {
            bail!(
                "Plugin '{}' is disabled: {}",
                plugin_name,
                summary.error.as_deref().unwrap_or("unavailable")
            );
        }
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.open_stream(request).await
    }

    pub(crate) async fn connect_stream(
        &self,
        plugin_name: &str,
        request: proto::OpenStreamRequest,
    ) -> Result<LocalStream> {
        #[cfg(test)]
        if let Some(handler) = self
            .inner
            .test_stream_handlers
            .lock()
            .await
            .get(plugin_name)
            .cloned()
        {
            return handler(request).await;
        }
        let response = self.open_stream(plugin_name, request).await?;
        if !response.accepted {
            bail!(
                "Plugin '{}' rejected stream request: {}",
                plugin_name,
                response.message.as_deref().unwrap_or("no reason provided")
            );
        }
        let endpoint = response.endpoint.as_deref().with_context(|| {
            format!(
                "Plugin '{}' accepted stream request without an endpoint",
                plugin_name
            )
        })?;
        connect_side_stream(endpoint, response.transport_kind).await
    }

    fn start_supervisor(&self) {
        let manager = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(
                health::HEALTH_CHECK_INTERVAL_SECS,
            ));
            loop {
                ticker.tick().await;
                if manager.inner.shutting_down.load(Ordering::SeqCst) {
                    break;
                }
                manager.refresh_shared_endpoint().await;
                let plugin_names = manager.inner.plugins.keys().cloned().collect::<Vec<_>>();
                for plugin_name in plugin_names {
                    let Some(plugin) = manager.inner.plugins.get(&plugin_name) else {
                        continue;
                    };
                    if let Err(err) = plugin.supervise().await {
                        tracing::warn!(
                            plugin = %plugin.name(),
                            error = %err,
                            "Plugin supervision round failed"
                        );
                    }
                    if let Err(err) = manager.refresh_plugin_endpoints(&plugin_name).await {
                        tracing::warn!(
                            plugin = %plugin_name,
                            error = %err,
                            "Endpoint supervision round failed"
                        );
                    }
                }
            }
        });
    }
}

#[cfg(test)]
pub(crate) async fn connect_test_side_stream(
    endpoint: &str,
    transport_kind: i32,
) -> Result<LocalStream> {
    connect_side_stream(endpoint, transport_kind).await
}

pub(crate) fn plugin_manifest_overview(manifest: &proto::PluginManifest) -> PluginManifestOverview {
    let web_ui = plugin_web_ui_manifest_overview_from_proto(manifest.web_ui.as_ref());
    PluginManifestOverview {
        operations: manifest.operations.len(),
        resources: manifest.resources.len(),
        resource_templates: manifest.resource_templates.len(),
        prompts: manifest.prompts.len(),
        completions: manifest.completions.len(),
        http_bindings: manifest.http_bindings.len(),
        endpoints: manifest.endpoints.len(),
        mesh_channels: manifest.mesh_channels.len(),
        mesh_event_subscriptions: manifest.mesh_event_subscriptions.len(),
        capabilities: manifest.capabilities.clone(),
        web_ui,
    }
}

pub(crate) fn plugin_manifest_to_json(manifest: &proto::PluginManifest) -> Value {
    json!({
        "operations": manifest.operations.iter().map(|operation| {
            json!({
                "name": operation.name,
                "description": operation.description,
                "input_schema_json": operation.input_schema_json,
                "output_schema_json": operation.output_schema_json,
                "title": operation.title,
            })
        }).collect::<Vec<_>>(),
        "resources": manifest.resources.iter().map(|resource| {
            json!({
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mime_type": resource.mime_type,
            })
        }).collect::<Vec<_>>(),
        "resource_templates": manifest.resource_templates.iter().map(|resource| {
            json!({
                "uri_template": resource.uri_template,
                "name": resource.name,
                "description": resource.description,
                "mime_type": resource.mime_type,
            })
        }).collect::<Vec<_>>(),
        "prompts": manifest.prompts.iter().map(|prompt| {
            json!({
                "name": prompt.name,
                "description": prompt.description,
            })
        }).collect::<Vec<_>>(),
        "completions": manifest.completions.iter().map(|completion| {
            json!({
                "argument_ref": completion.argument_ref,
                "description": completion.description,
            })
        }).collect::<Vec<_>>(),
        "http_bindings": manifest.http_bindings.iter().map(|binding| {
            json!({
                "binding_id": binding.binding_id,
                "method": http_method_name(binding.method),
                "path": binding.path,
                "operation_name": binding.operation_name,
                "request_body_mode": http_body_mode_name(binding.request_body_mode),
                "response_body_mode": http_body_mode_name(binding.response_body_mode),
                "request_schema_json": binding.request_schema_json,
                "response_schema_json": binding.response_schema_json,
            })
        }).collect::<Vec<_>>(),
        "endpoints": manifest.endpoints.iter().map(|endpoint| {
            json!({
                "endpoint_id": endpoint.endpoint_id,
                "kind": endpoint_kind_name(endpoint.kind),
                "transport_kind": endpoint_transport_kind_name(endpoint.transport_kind),
                "protocol": endpoint.protocol,
                "address": endpoint.address,
                "args": endpoint.args,
                "namespace": endpoint.namespace,
                "supports_streaming": endpoint.supports_streaming,
                "managed_by_plugin": endpoint.managed_by_plugin,
            })
        }).collect::<Vec<_>>(),
        "mesh_channels": manifest.mesh_channels.iter().map(|channel| {
            json!({
                "name": channel.name,
            })
        }).collect::<Vec<_>>(),
        "mesh_event_subscriptions": manifest.mesh_event_subscriptions.iter().map(|subscription| {
            json!({
                "kind": mesh_event_kind_name(subscription.kind),
            })
        }).collect::<Vec<_>>(),
        "capabilities": manifest.capabilities,
    })
}

fn manifest_subscribes_mesh_event(manifest: &proto::PluginManifest, kind: i32) -> bool {
    manifest
        .mesh_event_subscriptions
        .iter()
        .any(|entry| entry.kind == kind)
}

fn http_method_name(value: i32) -> &'static str {
    match proto::HttpMethod::try_from(value).unwrap_or(proto::HttpMethod::Unspecified) {
        proto::HttpMethod::Get => "GET",
        proto::HttpMethod::Post => "POST",
        proto::HttpMethod::Put => "PUT",
        proto::HttpMethod::Patch => "PATCH",
        proto::HttpMethod::Delete => "DELETE",
        proto::HttpMethod::Unspecified => "UNSPECIFIED",
    }
}

fn http_body_mode_name(value: i32) -> &'static str {
    match proto::HttpBodyMode::try_from(value).unwrap_or(proto::HttpBodyMode::Unspecified) {
        proto::HttpBodyMode::Buffered => "buffered",
        proto::HttpBodyMode::Streamed => "streamed",
        proto::HttpBodyMode::Unspecified => "unspecified",
    }
}

fn mesh_event_kind_name(value: i32) -> &'static str {
    match proto::mesh_event::Kind::try_from(value).unwrap_or(proto::mesh_event::Kind::Unspecified) {
        proto::mesh_event::Kind::PeerUp => "peer_up",
        proto::mesh_event::Kind::PeerDown => "peer_down",
        proto::mesh_event::Kind::PeerUpdated => "peer_updated",
        proto::mesh_event::Kind::LocalAccepting => "local_accepting",
        proto::mesh_event::Kind::LocalStandby => "local_standby",
        proto::mesh_event::Kind::MeshIdUpdated => "mesh_id_updated",
        proto::mesh_event::Kind::Unspecified => "unspecified",
    }
}

pub(in crate::plugin) fn endpoint_kind_name(value: i32) -> &'static str {
    match proto::EndpointKind::try_from(value).unwrap_or(proto::EndpointKind::Unspecified) {
        proto::EndpointKind::Inference => "inference",
        proto::EndpointKind::Mcp => "mcp",
        proto::EndpointKind::Unspecified => "unspecified",
    }
}

pub(in crate::plugin) fn endpoint_transport_kind_name(value: i32) -> &'static str {
    match proto::EndpointTransportKind::try_from(value)
        .unwrap_or(proto::EndpointTransportKind::Unspecified)
    {
        proto::EndpointTransportKind::EndpointTransportHttp => "http",
        proto::EndpointTransportKind::EndpointTransportUnixSocket => "unix_socket",
        proto::EndpointTransportKind::EndpointTransportStdio => "stdio",
        proto::EndpointTransportKind::EndpointTransportNamedPipe => "named_pipe",
        proto::EndpointTransportKind::EndpointTransportTcp => "tcp",
        proto::EndpointTransportKind::Unspecified => "unspecified",
    }
}

fn normalize_test_tool_result_content(result: &rmcp::model::CallToolResult) -> Result<String> {
    if let Some(value) = &result.structured_content {
        return serde_json::to_string(value).map_err(Into::into);
    }
    if let Some(text) = result.content.first().and_then(|content| content.as_text()) {
        return Ok(text.text.clone());
    }
    serde_json::to_string(&result.content).map_err(Into::into)
}

pub async fn run_plugin_process(name: String) -> Result<()> {
    match name.as_str() {
        BLOBSTORE_PLUGIN_ID => crate::plugins::blobstore::run_plugin(name).await,
        _ => bail!("Unknown built-in plugin '{}'", name),
    }
}

#[cfg(test)]
mod tests {
    use super::config::{MeshConfig, PluginConfigEntry};
    use super::*;

    fn private_host_mode() -> PluginHostMode {
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        }
    }

    fn web_ui_manifest() -> proto::PluginWebUiManifest {
        proto::PluginWebUiManifest {
            pages: vec![proto::PluginWebUiPageManifest {
                id: "home".into(),
                label: "Home".into(),
                icon: Some("icons/home.svg".into()),
                route: "index.html".into(),
                bundle_id: "main".into(),
                entry_script: "assets/app.js".into(),
            }],
            config_sections: vec![proto::PluginWebUiConfigSectionManifest {
                id: "settings".into(),
                title: "Settings".into(),
                entry_script: "assets/settings.js".into(),
                parent_tab: Some("integrations".into()),
                bundle_id: "main".into(),
            }],
            bundles: vec![proto::PluginWebUiBundleManifest {
                id: "main".into(),
                root_path: "web".into(),
            }],
        }
    }

    #[test]
    fn plugin_manifest_overview_includes_web_ui_declaration() {
        let manifest = proto::PluginManifest {
            web_ui: Some(web_ui_manifest()),
            ..proto::PluginManifest::default()
        };

        let overview = plugin_manifest_overview(&manifest);

        let web_ui = overview.web_ui.expect("web UI overview should be present");
        assert_eq!(web_ui.pages[0].id, "home");
        assert_eq!(web_ui.config_sections[0].id, "settings");
    }

    #[test]
    fn resolves_default_builtin_plugins() {
        let resolved = resolve_plugins(&MeshConfig::default(), private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLOBSTORE_PLUGIN_ID);
        assert!(resolved.externals[0].startup.optional);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn external_plugin_can_be_configured() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "demo".into(),
                enabled: Some(true),
                web_ui_enabled: None,
                command: Some("mesh-llm-plugin-demo".into()),
                args: vec!["--stdio".into()],
                url: None,
                settings: Default::default(),
                startup: Default::default(),
            }],
            defaults: None,
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[0].name, "demo");
        assert_eq!(resolved.externals[0].command, "mesh-llm-plugin-demo");
        assert_eq!(resolved.externals[0].args, ["--stdio"]);
        assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn failed_plugin_summary_redacts_urls_before_serialization() {
        let spec = ExternalPluginSpec {
            name: "remote".into(),
            command: "mesh-llm-plugin-remote".into(),
            args: Vec::new(),
            url: Some("https://plugin.example.test/v1".into()),
            env: BTreeMap::new(),
            startup: PluginStartupOptions {
                optional: true,
                ..PluginStartupOptions::default()
            },
            web_ui_enabled: None,
            installed_metadata: None,
        };
        let error = anyhow::anyhow!(
            "connection to tcp://user:secret@127.0.0.1:19091/control?token=private failed"
        );

        let summary = PluginManager::plugin_load_failure_summary(&spec, &error);
        let serialized = serde_json::to_string(&summary).expect("summary serializes");

        assert!(!serialized.contains("user"));
        assert!(!serialized.contains("secret"));
        assert!(!serialized.contains("private"));
    }

    #[test]
    fn blobstore_can_be_disabled() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: BLOBSTORE_PLUGIN_ID.into(),
                enabled: Some(false),
                web_ui_enabled: None,
                command: None,
                args: Vec::new(),
                url: None,
                settings: Default::default(),
                startup: Default::default(),
            }],
            defaults: None,
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert!(resolved.externals.is_empty());
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn external_plugin_can_be_enabled_with_url() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "endpoint-plugin".into(),
                enabled: Some(true),
                web_ui_enabled: None,
                command: Some("endpoint-plugin".into()),
                args: Vec::new(),
                url: Some("http://gpu-box:8000/v1".into()),
                settings: Default::default(),
                startup: Default::default(),
            }],
            defaults: None,
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[0].name, "endpoint-plugin");
        assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
        let spec = &resolved.externals[0];
        assert_eq!(spec.command, "endpoint-plugin");
        assert!(spec.args.is_empty());
        assert_eq!(spec.url.as_deref(), Some("http://gpu-box:8000/v1"));
    }

    #[test]
    fn external_plugin_rejects_url_that_is_empty_after_normalization() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "endpoint-plugin".into(),
                enabled: Some(true),
                web_ui_enabled: None,
                command: Some("endpoint-plugin".into()),
                args: Vec::new(),
                url: Some("\u{2003}\t\n".into()),
                settings: Default::default(),
                startup: Default::default(),
            }],
            ..MeshConfig::default()
        };

        let error = resolve_plugins(&config, private_host_mode())
            .expect_err("normalized plugin URL must not be empty");
        assert!(error.to_string().contains("plugin URL must not be empty"));
    }

    #[test]
    fn remote_plugin_control_url_is_rejected_without_authentication() {
        let raw_url = "\u{2003}tcp://user:secret@127.0.0.1:19091/control?token=private\u{2003}";
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "remote-plugin".into(),
                enabled: Some(true),
                web_ui_enabled: None,
                command: None,
                args: Vec::new(),
                url: Some(raw_url.into()),
                settings: Default::default(),
                startup: Default::default(),
            }],
            defaults: None,
            ..MeshConfig::default()
        };

        let error = resolve_plugins(&config, private_host_mode())
            .expect_err("unauthenticated remote plugin control must be rejected");
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("authenticated capability handshake"));
        assert!(!diagnostic.contains("user"));
        assert!(!diagnostic.contains("secret"));
        assert!(!diagnostic.contains("private"));
        assert!(!diagnostic.contains(raw_url));
    }

    #[test]
    fn external_plugin_can_be_enabled_with_command_args() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "endpoint-plugin".into(),
                enabled: Some(true),
                web_ui_enabled: None,
                command: Some("/opt/plugins/endpoint-plugin".into()),
                args: vec!["--verbose".into()],
                url: None,
                settings: Default::default(),
                startup: Default::default(),
            }],
            defaults: None,
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[0].name, "endpoint-plugin");
        assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
        let spec = &resolved.externals[0];
        assert_eq!(spec.command, "/opt/plugins/endpoint-plugin");
        assert_eq!(spec.args, vec!["--verbose"]);
    }

    #[test]
    fn external_plugin_ignores_disabled_entry_without_install() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "endpoint-plugin".into(),
                enabled: Some(false),
                web_ui_enabled: None,
                command: None,
                args: Vec::new(),
                url: Some("http://gpu-box:8000/v1".into()),
                settings: Default::default(),
                startup: Default::default(),
            }],
            defaults: None,
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLOBSTORE_PLUGIN_ID);
    }

    #[test]
    fn default_builtins_are_resolved_on_public_meshes() {
        let resolved = resolve_plugins(
            &MeshConfig::default(),
            PluginHostMode {
                mesh_visibility: MeshVisibility::Public,
            },
        )
        .unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLOBSTORE_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn resolves_external_plugin() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "demo".into(),
                enabled: Some(true),
                web_ui_enabled: None,
                command: Some("/tmp/demo".into()),
                args: vec!["--flag".into()],
                url: None,
                settings: Default::default(),
                startup: Default::default(),
            }],
            defaults: None,
            ..MeshConfig::default()
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[0].name, "demo");
        assert_eq!(resolved.externals[1].name, BLOBSTORE_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[tokio::test]
    async fn plugin_load_failure_keeps_declared_web_ui_metadata() {
        let specs = ResolvedPlugins {
            shared_endpoint: None,
            externals: vec![ExternalPluginSpec {
                name: "demo".into(),
                command: "mesh-llm-definitely-missing-plugin-binary".into(),
                args: Vec::new(),
                url: None,
                env: BTreeMap::new(),
                startup: PluginStartupOptions {
                    optional: true,
                    ..PluginStartupOptions::default()
                },
                web_ui_enabled: None,
                installed_metadata: Some(installed_metadata_with_web_ui(
                    InstalledPluginWebUiValidationStatus::Valid,
                    Some("web"),
                )),
            }],
            inactive: Vec::new(),
        };
        let (mesh_tx, _mesh_rx) = mpsc::channel(1);

        let manager = PluginManager::start(&specs, private_host_mode(), mesh_tx)
            .await
            .expect("broken plugin should not stop manager startup");
        let summaries = manager.list().await;
        manager.shutdown().await;

        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].name, "demo");
        assert_eq!(summaries[0].status, "error");
        assert_eq!(
            summaries[0].web_ui.state,
            PluginWebUiStateKind::PluginNotRunning
        );
        assert_eq!(summaries[0].web_ui.pages.len(), 1);
        assert_eq!(summaries[0].web_ui.config_sections.len(), 1);
    }

    #[test]
    fn instance_ids_include_pid_and_random_suffix() {
        let instance_id = make_instance_id();
        let prefix = format!("p{}-", std::process::id());
        assert!(instance_id.starts_with(&prefix));
        assert_eq!(instance_id.len(), prefix.len() + 8);
        assert!(
            instance_id[prefix.len()..]
                .chars()
                .all(|ch| ch.is_ascii_hexdigit())
        );
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_path_is_namespaced_by_instance_id() {
        let path = unix_socket_path("p1234-deadbeef", "Pipes").unwrap();
        assert_eq!(
            path.file_name().and_then(|value| value.to_str()),
            Some("p1234-deadbeef-Pipes.sock")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_pipe_name_is_namespaced_by_instance_id() {
        assert_eq!(
            windows_pipe_name("p1234-deadbeef", "Pipes"),
            r"\\.\pipe\mesh-llm-p1234-deadbeef-Pipes"
        );
    }
}
