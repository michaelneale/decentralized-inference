use anyhow::{anyhow, bail, Context, Result};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex};

pub const BLACKBOARD_PLUGIN_ID: &str = "blackboard";
const PROTOCOL_VERSION: u32 = 2;
const CONNECT_TIMEOUT_SECS: u64 = 10;
const REQUEST_TIMEOUT_SECS: u64 = 30;
const HEALTH_CHECK_INTERVAL_SECS: u64 = 15;

#[allow(dead_code)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/meshllm.plugin.v1.rs"));
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct MeshConfig {
    #[serde(rename = "plugin", default)]
    pub plugins: Vec<PluginConfigEntry>,
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
}

#[derive(Clone, Debug)]
pub struct ExternalPluginSpec {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
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
}

struct ExternalPlugin {
    spec: ExternalPluginSpec,
    summary: Arc<Mutex<PluginSummary>>,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
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

enum LocalStream {
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
    #[cfg(windows)]
    PipeServer(tokio::net::windows::named_pipe::NamedPipeServer),
    #[cfg(windows)]
    PipeClient(tokio::net::windows::named_pipe::NamedPipeClient),
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

pub fn resolve_plugins(config: &MeshConfig) -> Result<ResolvedPlugins> {
    let mut externals = Vec::new();
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

    Ok(ResolvedPlugins { externals })
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

        let mut plugins = BTreeMap::new();
        for spec in &specs.externals {
            tracing::info!(
                plugin = %spec.name,
                command = %spec.command,
                args = %format_args_for_log(&spec.args),
                "Loading plugin"
            );
            let plugin = match ExternalPlugin::spawn(spec, mesh_tx.clone()).await {
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
            inner: Arc::new(PluginManagerInner { plugins }),
        };
        manager.start_supervisor();
        Ok(manager)
    }

    pub async fn list(&self) -> Vec<PluginSummary> {
        let mut summaries = Vec::with_capacity(self.inner.plugins.len());
        for plugin in self.inner.plugins.values() {
            summaries.push(plugin.summary.lock().await.clone());
        }
        summaries
    }

    pub async fn is_enabled(&self, name: &str) -> bool {
        if let Some(plugin) = self
            .inner
            .plugins
            .get(name)
        {
            let summary = plugin.summary.lock().await;
            summary.enabled && summary.status == "running"
        } else {
            false
        }
    }

    pub async fn tools(&self, name: &str) -> Result<Vec<ToolSummary>> {
        let plugin = self
            .inner
            .plugins
            .get(name)
            .with_context(|| format!("Unknown plugin '{name}'"))?;
        Ok(plugin.summary.lock().await.tools.clone())
    }

    pub async fn call_tool(
        &self,
        plugin_name: &str,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.call_tool(tool_name, arguments_json).await
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
    async fn spawn(spec: &ExternalPluginSpec, mesh_tx: mpsc::Sender<PluginMeshEvent>) -> Result<Self> {
        let plugin = Self {
            spec: spec.clone(),
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
            runtime: Arc::new(Mutex::new(None)),
            mesh_tx,
            restart_lock: Arc::new(Mutex::new(())),
            next_request_id: AtomicU64::new(1),
            next_generation: AtomicU64::new(1),
        };
        plugin.ensure_running().await?;
        Ok(plugin)
    }

    async fn supervise(&self) -> Result<()> {
        self.ensure_running().await?;
        let response = self
            .request(proto::envelope::Payload::HealthRequest(proto::HealthRequest {}))
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
            self.runtime.clone(),
            generation,
        ));

        let (_, outbound_tx, pending) = self.runtime_handles().await?;
        let init_result: Result<proto::InitializeResponse> = async {
            let response = self
                .request_once(
                    generation,
                    outbound_tx,
                    pending,
                    proto::envelope::Payload::InitializeRequest(proto::InitializeRequest {
                        host_protocol_version: PROTOCOL_VERSION,
                        host_version: crate::VERSION.to_string(),
                        requested_capabilities: Vec::new(),
                    }),
                )
                .await?;

            let init = match response.payload {
                Some(proto::envelope::Payload::InitializeResponse(resp)) => resp,
                Some(proto::envelope::Payload::ErrorResponse(err)) => {
                    bail!("Plugin '{}' rejected initialize: {}", self.spec.name, err.message)
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
                self.handle_runtime_failure(
                    Some(generation),
                    format!("Plugin '{}' failed initialize: {err}", self.spec.name),
                )
                .await;
                return Err(err);
            }
        };

        let mut summary = self.summary.lock().await;
        summary.status = "running".into();
        summary.version = Some(init.plugin_version);
        summary.capabilities = init.capabilities;
        summary.tools = init
            .tool_schemas
            .into_iter()
            .map(|tool| ToolSummary {
                name: tool.name,
                description: tool.description,
                input_schema_json: tool.input_schema_json,
            })
            .collect();
        summary.error = None;
        Ok(())
    }

    async fn call_tool(&self, tool_name: &str, arguments_json: &str) -> Result<ToolCallResult> {
        let response = self
            .request(proto::envelope::Payload::ToolCallRequest(
                proto::ToolCallRequest {
                    name: tool_name.to_string(),
                    arguments_json: arguments_json.to_string(),
                },
            ))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::ToolCallResponse(resp)) => Ok(ToolCallResult {
                content_json: resp.content_json,
                is_error: resp.is_error,
            }),
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                bail!("Plugin tool call failed: {}", err.message)
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected tool payload",
                self.spec.name
            ),
        }
    }

    async fn send_channel_message(&self, message: proto::ChannelMessage) -> Result<()> {
        self.send_unsolicited(proto::envelope::Payload::ChannelMessage(message), "messages")
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

    async fn send_unsolicited(
        &self,
        payload: proto::envelope::Payload,
        kind: &str,
    ) -> Result<()> {
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
}

async fn connection_loop(
    mut stream: LocalStream,
    mut outbound_rx: mpsc::Receiver<proto::Envelope>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
    plugin_name: String,
    summary: Arc<Mutex<PluginSummary>>,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
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
            #[cfg(windows)]
            LocalStream::PipeClient(stream) => stream.write_all(bytes).await?,
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
            #[cfg(windows)]
            LocalStream::PipeClient(stream) => {
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
    let endpoint = std::env::var("MESH_LLM_PLUGIN_ENDPOINT")
        .context("MESH_LLM_PLUGIN_ENDPOINT is not set for plugin process")?;
    let transport =
        std::env::var("MESH_LLM_PLUGIN_TRANSPORT").unwrap_or_else(|_| default_transport().into());
    let stream = connect_to_host(&endpoint, &transport).await?;

    match name.as_str() {
        BLACKBOARD_PLUGIN_ID => run_blackboard_plugin(name, stream).await,
        _ => bail!("Unknown built-in plugin '{}'", name),
    }
}

async fn run_blackboard_plugin(name: String, mut stream: LocalStream) -> Result<()> {
    use crate::blackboard::{
        BlackboardItem, BlackboardMessage, BlackboardStore, FeedRequest, PostRequest, SearchRequest,
        BLACKBOARD_CHANNEL,
    };

    fn blackboard_channel_message(
        target_peer_id: String,
        body: Vec<u8>,
    ) -> proto::ChannelMessage {
        proto::ChannelMessage {
            channel: BLACKBOARD_CHANNEL.to_string(),
            source_peer_id: String::new(),
            target_peer_id,
            content_type: "application/json".into(),
            body,
            message_kind: "blackboard".into(),
            correlation_id: String::new(),
            metadata_json: String::new(),
        }
    }

    let store = BlackboardStore::new(true);
    loop {
        let envelope = read_envelope(&mut stream).await?;
        match envelope.payload {
            Some(proto::envelope::Payload::InitializeRequest(_)) => {
                let response = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: name.clone(),
                    request_id: envelope.request_id,
                    payload: Some(proto::envelope::Payload::InitializeResponse(
                        proto::InitializeResponse {
                            plugin_id: name.clone(),
                            plugin_protocol_version: PROTOCOL_VERSION,
                            plugin_version: crate::VERSION.to_string(),
                            capabilities: vec![
                                "channel:blackboard".into(),
                                "mcp:blackboard".into(),
                            ],
                            tool_schemas: vec![
                                proto::ToolSchema {
                                    name: "feed".into(),
                                    description: "Read the recent blackboard feed.".into(),
                                    input_schema_json: serde_json::json!({
                                        "type": "object",
                                        "properties": {
                                            "since": {"type": "integer"},
                                            "from": {"type": "string"},
                                            "limit": {"type": "integer"}
                                        }
                                    })
                                    .to_string(),
                                },
                                proto::ToolSchema {
                                    name: "search".into(),
                                    description: "Search blackboard messages.".into(),
                                    input_schema_json: serde_json::json!({
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string"},
                                            "since": {"type": "integer"},
                                            "limit": {"type": "integer"}
                                        },
                                        "required": ["query"]
                                    })
                                    .to_string(),
                                },
                                proto::ToolSchema {
                                    name: "post".into(),
                                    description: "Post a blackboard message.".into(),
                                    input_schema_json: serde_json::json!({
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "from": {"type": "string"},
                                            "peer_id": {"type": "string"}
                                        },
                                        "required": ["text"]
                                    })
                                    .to_string(),
                                },
                            ],
                        },
                    )),
                };
                write_envelope(&mut stream, &response).await?;
                send_plugin_channel_message(
                    &mut stream,
                    &name,
                    blackboard_channel_message(
                        String::new(),
                        serde_json::to_vec(&BlackboardMessage::SyncRequest)?,
                    ),
                )
                .await?;
            }
            Some(proto::envelope::Payload::HealthRequest(_)) => {
                let response = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: name.clone(),
                    request_id: envelope.request_id,
                    payload: Some(proto::envelope::Payload::HealthResponse(
                        proto::HealthResponse {
                            status: proto::health_response::Status::Ok as i32,
                            detail: "ok".into(),
                        },
                    )),
                };
                write_envelope(&mut stream, &response).await?;
            }
            Some(proto::envelope::Payload::ShutdownRequest(_)) => {
                let response = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: name.clone(),
                    request_id: envelope.request_id,
                    payload: Some(proto::envelope::Payload::ShutdownResponse(
                        proto::ShutdownResponse {},
                    )),
                };
                write_envelope(&mut stream, &response).await?;
                break;
            }
            Some(proto::envelope::Payload::ToolCallRequest(call)) => {
                let payload = match call.name.as_str() {
                    "feed" => {
                        let request = serde_json::from_str::<FeedRequest>(&call.arguments_json)
                            .unwrap_or_default();
                        let items = store.feed(request.since, request.from.as_deref(), request.limit).await;
                        proto::envelope::Payload::ToolCallResponse(proto::ToolCallResponse {
                            content_json: serde_json::to_string(&items)?,
                            is_error: false,
                        })
                    }
                    "search" => {
                        match serde_json::from_str::<SearchRequest>(&call.arguments_json) {
                            Ok(request) => {
                                let mut items = store.search(&request.query, request.since).await;
                                items.truncate(request.limit.max(1));
                                proto::envelope::Payload::ToolCallResponse(
                                    proto::ToolCallResponse {
                                        content_json: serde_json::to_string(&items)?,
                                        is_error: false,
                                    },
                                )
                            }
                            Err(err) => {
                                proto::envelope::Payload::ErrorResponse(proto::ErrorResponse {
                                    code: proto::error_response::Code::InvalidRequest as i32,
                                    message: format!("Invalid search arguments: {err}"),
                                })
                            }
                        }
                    }
                    "post" => {
                        match serde_json::from_str::<PostRequest>(&call.arguments_json) {
                            Ok(request) => {
                                let from = if request.from.trim().is_empty() {
                                    "mcp".to_string()
                                } else {
                                    request.from
                                };
                                let peer_id = if request.peer_id.trim().is_empty() {
                                    "mcp".to_string()
                                } else {
                                    request.peer_id
                                };
                                let item = BlackboardItem::new(from, peer_id, request.text);
                                match store.post(item).await {
                                    Ok(posted) => {
                                        send_plugin_channel_message(
                                            &mut stream,
                                            &name,
                                            blackboard_channel_message(
                                                String::new(),
                                                serde_json::to_vec(&BlackboardMessage::Post(
                                                    posted.clone(),
                                                ))?,
                                            ),
                                        )
                                        .await?;
                                        proto::envelope::Payload::ToolCallResponse(
                                            proto::ToolCallResponse {
                                                content_json: serde_json::to_string(&posted)?,
                                                is_error: false,
                                            },
                                        )
                                    }
                                    Err(reason) => {
                                        proto::envelope::Payload::ErrorResponse(
                                            proto::ErrorResponse {
                                                code: proto::error_response::Code::InvalidRequest
                                                    as i32,
                                                message: reason,
                                            },
                                        )
                                    }
                                }
                            }
                            Err(err) => {
                                proto::envelope::Payload::ErrorResponse(proto::ErrorResponse {
                                    code: proto::error_response::Code::InvalidRequest as i32,
                                    message: format!("Invalid post arguments: {err}"),
                                })
                            }
                        }
                    }
                    _ => proto::envelope::Payload::ErrorResponse(proto::ErrorResponse {
                        code: proto::error_response::Code::UnsupportedCapability as i32,
                        message: format!("Unknown tool '{}'", call.name),
                    }),
                };
                let response = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: name.clone(),
                    request_id: envelope.request_id,
                    payload: Some(payload),
                };
                write_envelope(&mut stream, &response).await?;
            }
            Some(proto::envelope::Payload::ChannelMessage(message)) => {
                if message.channel != BLACKBOARD_CHANNEL {
                    continue;
                }
                let payload: BlackboardMessage = serde_json::from_slice(&message.body)?;
                match payload {
                    BlackboardMessage::Post(item) => {
                        let _ = store.insert(item).await;
                    }
                    BlackboardMessage::SyncRequest => {
                        let ids = store.ids().await;
                        let response = BlackboardMessage::SyncDigest(ids);
                        send_plugin_channel_message(
                            &mut stream,
                            &name,
                            blackboard_channel_message(
                                message.source_peer_id,
                                serde_json::to_vec(&response)?,
                            ),
                        )
                        .await?;
                    }
                    BlackboardMessage::SyncDigest(ids) => {
                        let our_ids = store.ids().await;
                        let missing: Vec<u64> = ids
                            .into_iter()
                            .filter(|id| !our_ids.contains(id))
                            .collect();
                        if !missing.is_empty() {
                            send_plugin_channel_message(
                                &mut stream,
                                &name,
                                blackboard_channel_message(
                                    message.source_peer_id,
                                    serde_json::to_vec(&BlackboardMessage::FetchRequest(missing))?,
                                ),
                            )
                            .await?;
                        }
                    }
                    BlackboardMessage::FetchRequest(ids) => {
                        let items = store.get_by_ids(&ids).await;
                        send_plugin_channel_message(
                            &mut stream,
                            &name,
                            blackboard_channel_message(
                                message.source_peer_id,
                                serde_json::to_vec(&BlackboardMessage::FetchResponse(items))?,
                            ),
                        )
                        .await?;
                    }
                    BlackboardMessage::FetchResponse(items) => {
                        for item in items {
                            let _ = store.insert(item).await;
                        }
                    }
                }
            }
            Some(proto::envelope::Payload::MeshEvent(_)) => {
                continue;
            }
            _ => {
                let response = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: name.clone(),
                    request_id: envelope.request_id,
                    payload: Some(proto::envelope::Payload::ErrorResponse(
                        proto::ErrorResponse {
                            code: proto::error_response::Code::InvalidRequest as i32,
                            message: "Unsupported request".into(),
                        },
                    )),
                };
                write_envelope(&mut stream, &response).await?;
            }
        }
    }

    Ok(())
}

async fn send_plugin_channel_message(
    stream: &mut LocalStream,
    plugin_id: &str,
    message: proto::ChannelMessage,
) -> Result<()> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: plugin_id.to_string(),
            request_id: 0,
            payload: Some(proto::envelope::Payload::ChannelMessage(message)),
        },
    )
    .await
}

async fn connect_to_host(endpoint: &str, transport: &str) -> Result<LocalStream> {
    match transport {
        #[cfg(unix)]
        "unix" => Ok(LocalStream::Unix(tokio::net::UnixStream::connect(endpoint).await?)),
        #[cfg(windows)]
        "pipe" => Ok(LocalStream::PipeClient(
            tokio::net::windows::named_pipe::ClientOptions::new().open(endpoint)?,
        )),
        _ => bail!("Unsupported plugin transport '{transport}'"),
    }
}

async fn write_envelope(stream: &mut LocalStream, envelope: &proto::Envelope) -> Result<()> {
    let mut body = Vec::new();
    envelope.encode(&mut body)?;
    stream.write_all(&(body.len() as u32).to_le_bytes()).await?;
    stream.write_all(&body).await?;
    Ok(())
}

async fn read_envelope(stream: &mut LocalStream) -> Result<proto::Envelope> {
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

fn default_transport() -> &'static str {
    #[cfg(unix)]
    {
        "unix"
    }
    #[cfg(windows)]
    {
        "pipe"
    }
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
    let names = tools.iter().map(|tool| tool.name.clone()).collect::<Vec<_>>();
    format_slice_for_log(&names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_default_blackboard_plugin() {
        let resolved = resolve_plugins(&MeshConfig::default()).unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
    }

    #[test]
    fn blackboard_can_be_disabled() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: BLACKBOARD_PLUGIN_ID.into(),
                enabled: Some(false),
                command: None,
                args: Vec::new(),
            }],
        };
        let resolved = resolve_plugins(&config).unwrap();
        assert!(resolved.externals.is_empty());
    }

    #[test]
    fn resolves_external_plugin() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "demo".into(),
                enabled: Some(true),
                command: Some("/tmp/demo".into()),
                args: vec!["--flag".into()],
            }],
        };
        let resolved = resolve_plugins(&config).unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[1].name, "demo");
    }
}
