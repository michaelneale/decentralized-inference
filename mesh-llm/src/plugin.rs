use anyhow::{anyhow, bail, Context, Result};
pub use mesh_llm_plugin::proto;
use mesh_llm_plugin::{MeshVisibility, STARTUP_DISABLED_ERROR_CODE};
use prost::Message;
use rmcp::model::{
    CallToolResult as McpCallToolResult, ErrorCode, InitializeRequestParams, ListToolsResult,
    PaginatedRequestParams, ServerInfo,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex};

pub const BLACKBOARD_PLUGIN_ID: &str = "blackboard";
pub(crate) const PROTOCOL_VERSION: u32 = mesh_llm_plugin::PROTOCOL_VERSION;
const CONNECT_TIMEOUT_SECS: u64 = 10;
const REQUEST_TIMEOUT_SECS: u64 = 30;
const HEALTH_CHECK_INTERVAL_SECS: u64 = 15;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct MeshConfig {
    #[serde(default)]
    pub self_update: Option<bool>,
    #[serde(rename = "plugin", default)]
    pub plugins: Vec<PluginConfigEntry>,
}

impl MeshConfig {
    pub fn self_update_enabled(&self) -> bool {
        self.self_update.unwrap_or(true)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct PluginConfigEntry {
    pub name: String,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedPlugins {
    pub externals: Vec<ExternalPluginSpec>,
    pub inactive: Vec<PluginSummary>,
}

#[derive(Clone, Debug)]
pub struct ExternalPluginSpec {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct PluginHostMode {
    pub mesh_visibility: MeshVisibility,
}

#[derive(Clone, Debug)]
pub enum PluginMeshEvent {
    Channel {
        plugin_id: String,
        message: proto::ChannelMessage,
    },
    BulkTransfer {
        plugin_id: String,
        message: proto::BulkTransferMessage,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolSummary {
    pub name: String,
    pub description: String,
    pub input_schema_json: String,
}

#[derive(Clone, Debug)]
pub struct ToolCallResult {
    pub content_json: String,
    pub is_error: bool,
}

#[derive(Clone, Debug)]
pub struct RpcResult {
    pub result_json: String,
}

pub(crate) type BridgeFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

pub trait PluginRpcBridge: Send + Sync {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>>;

    fn handle_notification(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<()>;
}

#[derive(Clone, Debug, Serialize)]
pub struct PluginSummary {
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct PluginManager {
    inner: Arc<PluginManagerInner>,
}

struct PluginManagerInner {
    plugins: BTreeMap<String, ExternalPlugin>,
    inactive: BTreeMap<String, PluginSummary>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
}

struct ExternalPlugin {
    spec: ExternalPluginSpec,
    host_mode: PluginHostMode,
    summary: Arc<Mutex<PluginSummary>>,
    server_info: Arc<Mutex<Option<ServerInfo>>>,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    restart_lock: Arc<Mutex<()>>,
    next_request_id: AtomicU64,
    next_generation: AtomicU64,
}

struct PluginRuntime {
    generation: u64,
    _child: Child,
    outbound_tx: mpsc::Sender<proto::Envelope>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
}

pub(crate) enum LocalStream {
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
    #[cfg(windows)]
    PipeServer(tokio::net::windows::named_pipe::NamedPipeServer),
}

enum LocalListener {
    #[cfg(unix)]
    Unix(tokio::net::UnixListener, PathBuf),
    #[cfg(windows)]
    Pipe(String, tokio::net::windows::named_pipe::NamedPipeServer),
}

pub fn config_path(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.to_path_buf());
    }
    if let Ok(path) = std::env::var("MESH_LLM_CONFIG") {
        return Ok(PathBuf::from(path));
    }
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".mesh-llm").join("config.toml"))
}

pub fn load_config(override_path: Option<&Path>) -> Result<MeshConfig> {
    let path = config_path(override_path)?;
    if !path.exists() {
        return Ok(MeshConfig::default());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("Failed to parse config {}", path.display()))
}

pub fn resolve_plugins(config: &MeshConfig, _host_mode: PluginHostMode) -> Result<ResolvedPlugins> {
    let mut externals = Vec::new();
    let inactive = Vec::new();
    let mut names = BTreeMap::<String, ()>::new();
    let mut blackboard_enabled = true;
    for entry in &config.plugins {
        if names.insert(entry.name.clone(), ()).is_some() {
            bail!("Duplicate plugin entry '{}'", entry.name);
        }
        let enabled = entry.enabled.unwrap_or(true);
        if entry.name == BLACKBOARD_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    BLACKBOARD_PLUGIN_ID
                );
            }
            blackboard_enabled = enabled;
            continue;
        }
        if !enabled {
            continue;
        }
        let command = entry
            .command
            .clone()
            .with_context(|| format!("Plugin '{}' is enabled but missing command", entry.name))?;
        externals.push(ExternalPluginSpec {
            name: entry.name.clone(),
            command,
            args: entry.args.clone(),
        });
    }

    if blackboard_enabled {
        externals.insert(0, blackboard_plugin_spec()?);
    }

    Ok(ResolvedPlugins {
        externals,
        inactive,
    })
}

pub fn blackboard_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: BLACKBOARD_PLUGIN_ID.to_string(),
        command,
        args: vec!["--plugin".into(), BLACKBOARD_PLUGIN_ID.into()],
    })
}

impl PluginManager {
    pub async fn start(
        specs: &ResolvedPlugins,
        host_mode: PluginHostMode,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
    ) -> Result<Self> {
        if specs.externals.is_empty() {
            tracing::info!("Plugin manager: no plugins enabled");
        } else {
            let names = specs
                .externals
                .iter()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            tracing::info!(
                "Plugin manager: loading {} plugin(s): {}",
                specs.externals.len(),
                names
            );
        }

        let rpc_bridge = Arc::new(Mutex::new(None));
        let mut plugins = BTreeMap::new();
        for spec in &specs.externals {
            tracing::info!(
                plugin = %spec.name,
                command = %spec.command,
                args = %format_args_for_log(&spec.args),
                "Loading plugin"
            );
            let plugin =
                match ExternalPlugin::spawn(spec, host_mode, mesh_tx.clone(), rpc_bridge.clone())
                    .await
                {
                    Ok(plugin) => plugin,
                    Err(err) => {
                        tracing::error!(
                            plugin = %spec.name,
                            error = %err,
                            "Plugin failed to load"
                        );
                        return Err(err);
                    }
                };
            let summary = plugin.summary.lock().await.clone();
            tracing::info!(
                plugin = %summary.name,
                version = %summary.version.as_deref().unwrap_or("unknown"),
                capabilities = %format_slice_for_log(&summary.capabilities),
                tools = %format_tool_names_for_log(&summary.tools),
                "Plugin loaded successfully"
            );
            plugins.insert(spec.name.clone(), plugin);
        }
        let manager = Self {
            inner: Arc::new(PluginManagerInner {
                plugins,
                inactive: specs
                    .inactive
                    .iter()
                    .cloned()
                    .map(|summary| (summary.name.clone(), summary))
                    .collect(),
                rpc_bridge,
            }),
        };
        manager.start_supervisor();
        Ok(manager)
    }

    pub async fn list(&self) -> Vec<PluginSummary> {
        let mut summaries =
            Vec::with_capacity(self.inner.plugins.len() + self.inner.inactive.len());
        for plugin in self.inner.plugins.values() {
            summaries.push(plugin.summary.lock().await.clone());
        }
        summaries.extend(self.inner.inactive.values().cloned());
        summaries.sort_by(|a, b| a.name.cmp(&b.name));
        summaries
    }

    pub async fn is_enabled(&self, name: &str) -> bool {
        if let Some(plugin) = self.inner.plugins.get(name) {
            let summary = plugin.summary.lock().await;
            summary.enabled && summary.status == "running"
        } else {
            false
        }
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

    pub async fn mcp_request<T, P>(&self, plugin_name: &str, method: &str, params: P) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: Serialize,
    {
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

    pub async fn list_server_infos(&self) -> Vec<(String, ServerInfo)> {
        let mut infos = Vec::new();
        for (name, plugin) in &self.inner.plugins {
            if let Ok(info) = plugin.server_info().await {
                infos.push((name.clone(), info));
            }
        }
        infos
    }

    pub async fn set_rpc_bridge(&self, bridge: Option<Arc<dyn PluginRpcBridge>>) {
        *self.inner.rpc_bridge.lock().await = bridge;
    }

    pub async fn dispatch_channel_message(&self, event: PluginMeshEvent) -> Result<()> {
        let PluginMeshEvent::Channel { plugin_id, message } = event else {
            bail!("expected plugin channel event");
        };
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
        for plugin in self.inner.plugins.values() {
            plugin.send_mesh_event(event.clone()).await?;
        }
        Ok(())
    }

    fn start_supervisor(&self) {
        let manager = self.clone();
        tokio::spawn(async move {
            let mut ticker =
                tokio::time::interval(std::time::Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS));
            loop {
                ticker.tick().await;
                for plugin in manager.inner.plugins.values() {
                    if let Err(err) = plugin.supervise().await {
                        tracing::warn!(
                            plugin = %plugin.spec.name,
                            error = %err,
                            "Plugin supervision round failed"
                        );
                    }
                }
            }
        });
    }
}

impl ExternalPlugin {
    async fn spawn(
        spec: &ExternalPluginSpec,
        host_mode: PluginHostMode,
        mesh_tx: mpsc::Sender<PluginMeshEvent>,
        rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    ) -> Result<Self> {
        let plugin = Self {
            spec: spec.clone(),
            host_mode,
            summary: Arc::new(Mutex::new(PluginSummary {
                name: spec.name.clone(),
                kind: "external".into(),
                enabled: true,
                status: "starting".into(),
                version: None,
                capabilities: Vec::new(),
                command: Some(spec.command.clone()),
                args: spec.args.clone(),
                tools: Vec::new(),
                error: None,
            })),
            server_info: Arc::new(Mutex::new(None)),
            runtime: Arc::new(Mutex::new(None)),
            mesh_tx,
            rpc_bridge,
            restart_lock: Arc::new(Mutex::new(())),
            next_request_id: AtomicU64::new(1),
            next_generation: AtomicU64::new(1),
        };
        if let Err(err) = plugin.ensure_running().await {
            if plugin.is_disabled().await {
                return Ok(plugin);
            }
            return Err(err);
        }
        Ok(plugin)
    }

    async fn supervise(&self) -> Result<()> {
        if self.is_disabled().await {
            return Ok(());
        }
        self.ensure_running().await?;
        let response = self
            .request(proto::envelope::Payload::HealthRequest(
                proto::HealthRequest {},
            ))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::HealthResponse(resp))
                if resp.status == proto::health_response::Status::Ok as i32 =>
            {
                let mut summary = self.summary.lock().await;
                summary.status = "running".into();
                summary.error = None;
                Ok(())
            }
            Some(proto::envelope::Payload::HealthResponse(resp)) => {
                self.handle_runtime_failure(
                    None,
                    format!("health check reported status {}", resp.status),
                )
                .await;
                self.ensure_running().await
            }
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                self.handle_runtime_failure(None, err.message).await;
                self.ensure_running().await
            }
            _ => {
                self.handle_runtime_failure(None, "unexpected health payload".into())
                    .await;
                self.ensure_running().await
            }
        }
    }

    async fn ensure_running(&self) -> Result<()> {
        if let Some(reason) = self.disabled_reason().await {
            bail!("Plugin '{}' is disabled: {}", self.spec.name, reason);
        }
        if self.runtime.lock().await.is_some() {
            return Ok(());
        }
        let _guard = self.restart_lock.lock().await;
        if self.runtime.lock().await.is_some() {
            return Ok(());
        }

        {
            let mut summary = self.summary.lock().await;
            summary.status = "starting".into();
            summary.error = None;
        }

        let listener = bind_local_listener(&self.spec.name).await?;
        let endpoint = listener.endpoint();
        let transport = listener.transport_name();
        tracing::debug!(
            plugin = %self.spec.name,
            endpoint = %endpoint,
            transport,
            "Waiting for plugin connection"
        );

        let mut child = Command::new(&self.spec.command);
        child.args(&self.spec.args);
        child.env("MESH_LLM_PLUGIN_ENDPOINT", &endpoint);
        child.env("MESH_LLM_PLUGIN_TRANSPORT", transport);
        child.env("MESH_LLM_PLUGIN_NAME", &self.spec.name);
        child.stdin(std::process::Stdio::null());
        child.stdout(std::process::Stdio::null());
        child.stderr(std::process::Stdio::inherit());
        child.kill_on_drop(true);

        let child = child.spawn().with_context(|| {
            format!(
                "Failed to launch plugin '{}' via {}",
                self.spec.name, self.spec.command
            )
        })?;

        let stream = tokio::time::timeout(
            std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS),
            listener.accept(),
        )
        .await
        .with_context(|| format!("Timed out waiting for plugin '{}'", self.spec.name))??;

        let (outbound_tx, outbound_rx) = mpsc::channel(256);
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        let outbound_tx_for_runtime = outbound_tx.clone();
        *self.runtime.lock().await = Some(PluginRuntime {
            generation,
            _child: child,
            outbound_tx,
            pending: pending.clone(),
        });
        tokio::spawn(connection_loop(
            stream,
            outbound_rx,
            pending,
            self.mesh_tx.clone(),
            self.spec.name.clone(),
            self.summary.clone(),
            self.rpc_bridge.clone(),
            self.runtime.clone(),
            outbound_tx_for_runtime,
            generation,
        ));

        let (_, outbound_tx, pending) = self.runtime_handles().await?;
        let init_result: Result<proto::InitializeResponse> = async {
            let host_info_json = serde_json::to_string(&InitializeRequestParams::default())?;
            let response = self
                .request_once(
                    generation,
                    outbound_tx.clone(),
                    pending.clone(),
                    proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
                        host_protocol_version: PROTOCOL_VERSION,
                        host_version: crate::VERSION.to_string(),
                        host_info_json,
                        mesh_visibility: proto_mesh_visibility(self.host_mode.mesh_visibility),
                    }),
                )
                .await?;

            let init = match response.payload {
                Some(proto::envelope::Payload::InitializeResponse(resp)) => resp,
                Some(proto::envelope::Payload::ErrorResponse(err))
                    if err.code == STARTUP_DISABLED_ERROR_CODE =>
                {
                    self.mark_disabled(generation, err.message).await;
                    bail!("Plugin '{}' is disabled", self.spec.name);
                }
                Some(proto::envelope::Payload::ErrorResponse(err)) => {
                    bail!(
                        "Plugin '{}' rejected initialize: {}",
                        self.spec.name,
                        err.message
                    )
                }
                _ => bail!(
                    "Plugin '{}' returned an unexpected initialize payload",
                    self.spec.name
                ),
            };

            if init.plugin_id != self.spec.name {
                bail!(
                    "Plugin '{}' identified itself as '{}'",
                    self.spec.name,
                    init.plugin_id
                );
            }
            if init.plugin_protocol_version != PROTOCOL_VERSION {
                bail!(
                    "Plugin '{}' uses protocol {}, host uses {}",
                    self.spec.name,
                    init.plugin_protocol_version,
                    PROTOCOL_VERSION
                );
            }

            Ok(init)
        }
        .await;
        let init = match init_result {
            Ok(init) => init,
            Err(err) => {
                if self.is_disabled().await {
                    return Err(err);
                }
                self.handle_runtime_failure(
                    Some(generation),
                    format!("Plugin '{}' failed initialize: {err}", self.spec.name),
                )
                .await;
                return Err(err);
            }
        };

        let server_info: ServerInfo =
            serde_json::from_str(&init.server_info_json).with_context(|| {
                format!(
                    "Plugin '{}' returned invalid server_info_json",
                    self.spec.name
                )
            })?;
        *self.server_info.lock().await = Some(server_info.clone());

        let tools = if server_info.capabilities.tools.is_some() {
            let response = self
                .request_once(
                    generation,
                    outbound_tx.clone(),
                    pending.clone(),
                    proto::envelope::Payload::RpcRequest(proto::RpcRequest {
                        method: "tools/list".into(),
                        params_json: serialize_params(Option::<PaginatedRequestParams>::None)?,
                    }),
                )
                .await;
            match response {
                Ok(envelope) => match envelope.payload {
                    Some(proto::envelope::Payload::RpcResponse(resp)) => {
                        let result: ListToolsResult = serde_json::from_str(&resp.result_json)
                            .with_context(|| {
                                format!(
                                    "Plugin '{}' returned invalid result for 'tools/list'",
                                    self.spec.name
                                )
                            })?;
                        result
                            .tools
                            .into_iter()
                            .map(|tool| ToolSummary {
                                name: tool.name.to_string(),
                                description: tool.description.unwrap_or_default().to_string(),
                                input_schema_json: serde_json::to_string(
                                    tool.input_schema.as_ref(),
                                )
                                .unwrap_or_else(|_| "{}".into()),
                            })
                            .collect()
                    }
                    Some(proto::envelope::Payload::ErrorResponse(err)) => {
                        tracing::warn!(
                            plugin = %self.spec.name,
                            error = %plugin_error(&self.spec.name, "tools/list", &err),
                            "Plugin tools/list failed during startup"
                        );
                        Vec::new()
                    }
                    _ => {
                        tracing::warn!(
                            plugin = %self.spec.name,
                            "Plugin returned unexpected payload for tools/list during startup"
                        );
                        Vec::new()
                    }
                },
                Err(err) => {
                    tracing::warn!(
                        plugin = %self.spec.name,
                        error = %err,
                        "Plugin tools/list request failed during startup"
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        let mut summary = self.summary.lock().await;
        summary.status = "running".into();
        summary.version = Some(init.plugin_version);
        summary.capabilities = summarize_capabilities(&server_info, &init.capabilities);
        summary.tools = tools;
        summary.error = None;
        Ok(())
    }

    async fn server_info(&self) -> Result<ServerInfo> {
        self.ensure_running().await?;
        self.server_info
            .lock()
            .await
            .clone()
            .with_context(|| format!("Plugin '{}' did not publish server info", self.spec.name))
    }

    async fn list_tools(&self) -> Result<Vec<ToolSummary>> {
        let info = self.server_info().await?;
        if info.capabilities.tools.is_none() {
            return Ok(Vec::new());
        }
        let result: ListToolsResult = self
            .mcp_request("tools/list", None::<PaginatedRequestParams>)
            .await?;
        Ok(result
            .tools
            .into_iter()
            .map(|tool| ToolSummary {
                name: tool.name.to_string(),
                description: tool.description.unwrap_or_default().to_string(),
                input_schema_json: serde_json::to_string(tool.input_schema.as_ref())
                    .unwrap_or_else(|_| "{}".into()),
            })
            .collect())
    }

    async fn call_tool(&self, tool_name: &str, arguments_json: &str) -> Result<ToolCallResult> {
        let arguments = parse_optional_json(arguments_json)?;
        let result: McpCallToolResult = self
            .mcp_request(
                "tools/call",
                serde_json::json!({
                    "name": tool_name,
                    "arguments": arguments,
                }),
            )
            .await?;
        Ok(ToolCallResult {
            content_json: serde_json::to_string(&result)?,
            is_error: result.is_error.unwrap_or(false),
        })
    }

    async fn mcp_request<T, P>(&self, method: &str, params: P) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        P: Serialize,
    {
        let params_json = serialize_params(params)?;
        let response = self
            .request(proto::envelope::Payload::RpcRequest(proto::RpcRequest {
                method: method.to_string(),
                params_json,
            }))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::RpcResponse(resp)) => {
                serde_json::from_str(&resp.result_json).with_context(|| {
                    format!(
                        "Plugin '{}' returned invalid result for '{}'",
                        self.spec.name, method
                    )
                })
            }
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                Err(plugin_error(&self.spec.name, method, &err))
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected RPC payload for '{}'",
                self.spec.name,
                method
            ),
        }
    }

    async fn mcp_notify<P>(&self, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        self.send_unsolicited(
            proto::envelope::Payload::RpcNotification(proto::RpcNotification {
                method: method.to_string(),
                params_json: serialize_params(params)?,
            }),
            method,
        )
        .await
    }

    async fn send_channel_message(&self, message: proto::ChannelMessage) -> Result<()> {
        self.send_unsolicited(
            proto::envelope::Payload::ChannelMessage(message),
            "messages",
        )
        .await
    }

    async fn send_bulk_transfer_message(&self, message: proto::BulkTransferMessage) -> Result<()> {
        self.send_unsolicited(
            proto::envelope::Payload::BulkTransferMessage(message),
            "bulk transfers",
        )
        .await
    }

    async fn send_mesh_event(&self, event: proto::MeshEvent) -> Result<()> {
        self.send_unsolicited(proto::envelope::Payload::MeshEvent(event), "mesh events")
            .await
    }

    async fn request(&self, payload: proto::envelope::Payload) -> Result<proto::Envelope> {
        for attempt in 0..2 {
            self.ensure_running().await?;
            let (generation, outbound_tx, pending) = self.runtime_handles().await?;
            match self
                .request_once(generation, outbound_tx, pending, payload.clone())
                .await
            {
                Ok(response) => return Ok(response),
                Err(err) if attempt == 0 => {
                    tracing::debug!(
                        plugin = %self.spec.name,
                        error = %err,
                        "Retrying plugin request after restart"
                    );
                }
                Err(err) => return Err(err),
            }
        }
        bail!("Plugin '{}' request failed after restart", self.spec.name)
    }

    async fn send_unsolicited(&self, payload: proto::envelope::Payload, kind: &str) -> Result<()> {
        for attempt in 0..2 {
            self.ensure_running().await?;
            let (generation, outbound_tx, _) = self.runtime_handles().await?;
            let envelope = proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.spec.name.clone(),
                request_id: 0,
                payload: Some(payload.clone()),
            };
            if outbound_tx.send(envelope).await.is_ok() {
                return Ok(());
            }
            self.handle_runtime_failure(
                Some(generation),
                format!("Plugin '{}' is not accepting {kind}", self.spec.name),
            )
            .await;
            if attempt == 1 {
                break;
            }
        }
        bail!("Plugin '{}' is not accepting {}", self.spec.name, kind)
    }

    async fn runtime_handles(
        &self,
    ) -> Result<(
        u64,
        mpsc::Sender<proto::Envelope>,
        Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
    )> {
        let runtime = self.runtime.lock().await;
        let runtime = runtime
            .as_ref()
            .with_context(|| format!("Plugin '{}' is not running", self.spec.name))?;
        Ok((
            runtime.generation,
            runtime.outbound_tx.clone(),
            runtime.pending.clone(),
        ))
    }

    async fn request_once(
        &self,
        generation: u64,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
        payload: proto::envelope::Payload,
    ) -> Result<proto::Envelope> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        pending.lock().await.insert(request_id, tx);

        let envelope = proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: self.spec.name.clone(),
            request_id,
            payload: Some(payload),
        };

        if let Err(_send_err) = outbound_tx.send(envelope).await {
            pending.lock().await.remove(&request_id);
            self.handle_runtime_failure(
                Some(generation),
                format!("Plugin '{}' is not accepting requests", self.spec.name),
            )
            .await;
            bail!("Plugin '{}' is not accepting requests", self.spec.name);
        }

        match tokio::time::timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS), rx).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(_)) => {
                self.handle_runtime_failure(
                    Some(generation),
                    format!("Plugin '{}' dropped the response channel", self.spec.name),
                )
                .await;
                bail!("Plugin '{}' dropped the response channel", self.spec.name);
            }
            Err(_) => {
                pending.lock().await.remove(&request_id);
                self.handle_runtime_failure(
                    Some(generation),
                    format!("Plugin '{}' timed out", self.spec.name),
                )
                .await;
                bail!("Plugin '{}' timed out", self.spec.name);
            }
        }
    }

    async fn handle_runtime_failure(&self, generation: Option<u64>, reason: String) {
        let mut runtime = self.runtime.lock().await;
        let should_clear = generation
            .map(|generation| runtime.as_ref().map(|r| r.generation) == Some(generation))
            .unwrap_or(true);
        if should_clear {
            *runtime = None;
        }
        drop(runtime);
        let mut summary = self.summary.lock().await;
        summary.status = "restarting".into();
        summary.error = Some(reason);
    }

    async fn disabled_reason(&self) -> Option<String> {
        let summary = self.summary.lock().await;
        if summary.status == "disabled" {
            Some(
                summary
                    .error
                    .clone()
                    .unwrap_or_else(|| "disabled".to_string()),
            )
        } else {
            None
        }
    }

    async fn is_disabled(&self) -> bool {
        self.disabled_reason().await.is_some()
    }

    async fn mark_disabled(&self, generation: u64, reason: String) {
        let mut runtime = self.runtime.lock().await;
        if runtime.as_ref().map(|runtime| runtime.generation) == Some(generation) {
            *runtime = None;
        }
        drop(runtime);

        let mut server_info = self.server_info.lock().await;
        *server_info = None;
        drop(server_info);

        let mut summary = self.summary.lock().await;
        summary.enabled = false;
        summary.status = "disabled".into();
        summary.version = None;
        summary.capabilities.clear();
        summary.tools.clear();
        summary.error = Some(reason);
    }
}

async fn connection_loop(
    mut stream: LocalStream,
    mut outbound_rx: mpsc::Receiver<proto::Envelope>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
    plugin_name: String,
    summary: Arc<Mutex<PluginSummary>>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    outbound_tx: mpsc::Sender<proto::Envelope>,
    generation: u64,
) {
    let result: Result<()> = async {
        loop {
            tokio::select! {
                maybe_outbound = outbound_rx.recv() => {
                    let Some(envelope) = maybe_outbound else {
                        break;
                    };
                    write_envelope(&mut stream, &envelope).await?;
                }
                inbound = read_envelope(&mut stream) => {
                    let envelope = inbound?;
                    let request_id = envelope.request_id;
                    let plugin_id_from_env = envelope.plugin_id.clone();
                    let payload = envelope.payload.clone();
                    match payload {
                        Some(proto::envelope::Payload::ChannelMessage(message)) => {
                            let plugin_id = if plugin_id_from_env.is_empty() {
                                plugin_name.clone()
                            } else {
                                plugin_id_from_env
                            };
                            let _ = mesh_tx
                                .send(PluginMeshEvent::Channel { plugin_id, message })
                                .await;
                        }
                        Some(proto::envelope::Payload::BulkTransferMessage(message)) => {
                            let plugin_id = if plugin_id_from_env.is_empty() {
                                plugin_name.clone()
                            } else {
                                plugin_id_from_env
                            };
                            let _ = mesh_tx
                                .send(PluginMeshEvent::BulkTransfer {
                                    plugin_id,
                                    message,
                                })
                                .await;
                        }
                        Some(proto::envelope::Payload::RpcRequest(request)) => {
                            forward_plugin_request(
                                plugin_name.clone(),
                                request_id,
                                request,
                                rpc_bridge.clone(),
                                outbound_tx.clone(),
                            );
                        }
                        Some(proto::envelope::Payload::RpcNotification(notification)) => {
                            forward_plugin_notification(
                                plugin_name.clone(),
                                notification,
                                rpc_bridge.clone(),
                            );
                        }
                        _ => {
                            let responder = pending.lock().await.remove(&request_id);
                            if let Some(responder) = responder {
                                let _ = responder.send(Ok(envelope));
                            } else {
                                tracing::debug!(
                                    "Plugin '{}' sent an unsolicited response id={}",
                                    plugin_name,
                                    request_id
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    .await;

    if let Err(err) = result {
        tracing::warn!(
            plugin = %plugin_name,
            error = %err,
            "Plugin connection closed"
        );
    }

    {
        let mut runtime = runtime.lock().await;
        if runtime.as_ref().map(|runtime| runtime.generation) == Some(generation) {
            *runtime = None;
            let mut summary = summary.lock().await;
            summary.status = "stopped".into();
            summary.error = Some(format!("Plugin '{}' disconnected", plugin_name));
        }
    }

    let mut pending = pending.lock().await;
    for (_, responder) in pending.drain() {
        let _ = responder.send(Err(anyhow!("Plugin '{}' disconnected", plugin_name)));
    }
}

fn forward_plugin_request(
    plugin_name: String,
    request_id: u64,
    request: proto::RpcRequest,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    outbound_tx: mpsc::Sender<proto::Envelope>,
) {
    tokio::spawn(async move {
        let bridge = rpc_bridge.lock().await.clone();
        let payload = match bridge {
            Some(bridge) => match bridge
                .handle_request(
                    plugin_name.clone(),
                    request.method.clone(),
                    request.params_json.clone(),
                )
                .await
            {
                Ok(result) => proto::envelope::Payload::RpcResponse(proto::RpcResponse {
                    result_json: result.result_json,
                }),
                Err(err) => proto::envelope::Payload::ErrorResponse(err),
            },
            None => proto::envelope::Payload::ErrorResponse(proto::ErrorResponse {
                code: ErrorCode::INTERNAL_ERROR.0,
                message: "No active MCP bridge".into(),
                data_json: String::new(),
            }),
        };

        let _ = outbound_tx
            .send(proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: plugin_name,
                request_id,
                payload: Some(payload),
            })
            .await;
    });
}

fn forward_plugin_notification(
    plugin_name: String,
    notification: proto::RpcNotification,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
) {
    tokio::spawn(async move {
        if let Some(bridge) = rpc_bridge.lock().await.clone() {
            bridge
                .handle_notification(plugin_name, notification.method, notification.params_json)
                .await;
        }
    });
}

impl LocalListener {
    async fn accept(self) -> Result<LocalStream> {
        match self {
            #[cfg(unix)]
            LocalListener::Unix(listener, path) => {
                let (stream, _) = listener.accept().await?;
                let _ = std::fs::remove_file(path);
                Ok(LocalStream::Unix(stream))
            }
            #[cfg(windows)]
            LocalListener::Pipe(_name, server) => {
                server.connect().await?;
                Ok(LocalStream::PipeServer(server))
            }
        }
    }

    fn endpoint(&self) -> String {
        match self {
            #[cfg(unix)]
            LocalListener::Unix(_, path) => path.display().to_string(),
            #[cfg(windows)]
            LocalListener::Pipe(name, _) => name.clone(),
        }
    }

    fn transport_name(&self) -> &'static str {
        #[cfg(unix)]
        {
            "unix"
        }
        #[cfg(windows)]
        {
            "pipe"
        }
    }
}

impl LocalStream {
    async fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => stream.write_all(bytes).await?,
            #[cfg(windows)]
            LocalStream::PipeServer(stream) => stream.write_all(bytes).await?,
        }
        Ok(())
    }

    async fn read_exact(&mut self, bytes: &mut [u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => {
                let _ = stream.read_exact(bytes).await?;
            }
            #[cfg(windows)]
            LocalStream::PipeServer(stream) => {
                let _ = stream.read_exact(bytes).await?;
            }
        }
        Ok(())
    }
}

async fn bind_local_listener(name: &str) -> Result<LocalListener> {
    #[cfg(unix)]
    {
        let dir = runtime_dir()?;
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create plugin runtime dir {}", dir.display()))?;
        let path = dir.join(format!("{name}.sock"));
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
        let listener = tokio::net::UnixListener::bind(&path)
            .with_context(|| format!("Failed to bind plugin socket {}", path.display()))?;
        return Ok(LocalListener::Unix(listener, path));
    }
    #[cfg(windows)]
    {
        let endpoint = format!(r"\\.\pipe\mesh-llm-{name}");
        let server = tokio::net::windows::named_pipe::ServerOptions::new()
            .create(&endpoint)
            .with_context(|| format!("Failed to create plugin pipe {endpoint}"))?;
        return Ok(LocalListener::Pipe(endpoint, server));
    }
}

fn runtime_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".mesh-llm").join("run").join("plugins"))
}

pub async fn run_plugin_process(name: String) -> Result<()> {
    match name.as_str() {
        BLACKBOARD_PLUGIN_ID => crate::plugins::blackboard::run_plugin(name).await,
        _ => bail!("Unknown built-in plugin '{}'", name),
    }
}

pub(crate) fn serialize_params<T: Serialize>(params: T) -> Result<String> {
    serde_json::to_string(&params).context("serialize plugin RPC params")
}

pub(crate) fn parse_optional_json(raw: &str) -> Result<Option<serde_json::Value>> {
    if raw.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(
            serde_json::from_str(raw).context("parse plugin JSON payload")?,
        ))
    }
}

fn plugin_error(plugin_name: &str, method: &str, err: &proto::ErrorResponse) -> anyhow::Error {
    if err.data_json.trim().is_empty() {
        anyhow!(
            "Plugin '{}' failed '{}' (code {}): {}",
            plugin_name,
            method,
            err.code,
            err.message
        )
    } else {
        anyhow!(
            "Plugin '{}' failed '{}' (code {}): {} ({})",
            plugin_name,
            method,
            err.code,
            err.message,
            err.data_json
        )
    }
}

fn summarize_capabilities(server_info: &ServerInfo, extra: &[String]) -> Vec<String> {
    let mut capabilities = extra.to_vec();
    let caps = &server_info.capabilities;
    if caps.tools.is_some() {
        capabilities.push("mcp:tools".into());
    }
    if caps.prompts.is_some() {
        capabilities.push("mcp:prompts".into());
    }
    if caps.resources.is_some() {
        capabilities.push("mcp:resources".into());
    }
    if caps.completions.is_some() {
        capabilities.push("mcp:completions".into());
    }
    if caps.logging.is_some() {
        capabilities.push("mcp:logging".into());
    }
    if caps.tasks.is_some() {
        capabilities.push("mcp:tasks".into());
    }
    if let Some(extensions) = &caps.extensions {
        for key in extensions.keys() {
            capabilities.push(format!("mcp:extension:{key}"));
        }
    }
    capabilities.sort();
    capabilities.dedup();
    capabilities
}

pub(crate) async fn write_envelope(
    stream: &mut LocalStream,
    envelope: &proto::Envelope,
) -> Result<()> {
    let mut body = Vec::new();
    envelope.encode(&mut body)?;
    stream.write_all(&(body.len() as u32).to_le_bytes()).await?;
    stream.write_all(&body).await?;
    Ok(())
}

pub(crate) async fn read_envelope(stream: &mut LocalStream) -> Result<proto::Envelope> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 16 * 1024 * 1024 {
        bail!("Plugin frame too large");
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).await?;
    Ok(proto::Envelope::decode(body.as_slice())?)
}

fn format_args_for_log(args: &[String]) -> String {
    if args.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", args.join(", "))
    }
}

fn format_slice_for_log(values: &[String]) -> String {
    if values.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", values.join(", "))
    }
}

fn format_tool_names_for_log(tools: &[ToolSummary]) -> String {
    let names = tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<Vec<_>>();
    format_slice_for_log(&names)
}

fn proto_mesh_visibility(mesh_visibility: MeshVisibility) -> i32 {
    match mesh_visibility {
        MeshVisibility::Private => proto::MeshVisibility::Private as i32,
        MeshVisibility::Public => proto::MeshVisibility::Public as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn private_host_mode() -> PluginHostMode {
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        }
    }

    #[test]
    fn resolves_default_blackboard_plugin() {
        let resolved = resolve_plugins(&MeshConfig::default(), private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn blackboard_can_be_disabled() {
        let config = MeshConfig {
            self_update: None,
            plugins: vec![PluginConfigEntry {
                name: BLACKBOARD_PLUGIN_ID.into(),
                enabled: Some(false),
                command: None,
                args: Vec::new(),
            }],
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert!(resolved.externals.is_empty());
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn blackboard_is_resolved_on_public_meshes() {
        let resolved = resolve_plugins(
            &MeshConfig::default(),
            PluginHostMode {
                mesh_visibility: MeshVisibility::Public,
            },
        )
        .unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn resolves_external_plugin() {
        let config = MeshConfig {
            self_update: None,
            plugins: vec![PluginConfigEntry {
                name: "demo".into(),
                enabled: Some(true),
                command: Some("/tmp/demo".into()),
                args: vec!["--flag".into()],
            }],
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[1].name, "demo");
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn self_update_defaults_to_enabled() {
        assert!(MeshConfig::default().self_update_enabled());
        let config: MeshConfig = toml::from_str("self_update = false").unwrap();
        assert!(!config.self_update_enabled());
    }
}
