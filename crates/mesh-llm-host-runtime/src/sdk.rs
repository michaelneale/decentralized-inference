use crate::inference::skippy::{SkippyDeviceDescriptor, SkippyModelHandle, SkippyModelLoadOptions};
use crate::models;
use anyhow::{Context, Result};
#[cfg(test)]
use mesh_llm_node::serving::UnloadOptions;
use mesh_llm_node::serving::{
    DevicePolicy, LoadModelRequest, ServedModel, ServingController, ServingFuture,
    ServingModelState, ServingStatus, UnloadModelRequest, UnloadTarget,
};
use mesh_llm_system::hardware::{self, Metric};
#[cfg(test)]
use mesh_llm_types::models::capabilities::ModelCapabilities;
use openai_frontend::{ChatCompletionRequest, ChatMessage, MessageContent, OpenAiBackend};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;
use tokio::sync::Mutex;

mod embedded_config;
pub(crate) mod embedded_logging;
mod embedded_startup;

pub use embedded_config::*;

pub mod config {
    pub use mesh_llm_config::{
        AdvancedConfig, AdvancedServerConfig, BoolOrAuto, BoolOrString, ConfigEditor, ConfigStore,
        FlashAttentionType, GpuAssignment, GpuConfig, HardwareConfig, IntegerOrString,
        LocalServingNodeConfig, MeshConfig, ModelConfigDefaults, ModelConfigEditor,
        ModelConfigEntry, ModelDefaultsEditor, ModelFitConfig, ModelRuntimeKind, MultimodalConfig,
        OwnerControlConfig, PluginConfigEditor, PluginConfigEntry, PrefixCacheConfig,
        ReasoningBudget, ReasoningEnabled, RequestDefaultsConfig, ReservedObjectConfig,
        SkippyConfig, SpeculativeConfig, StringOrStringList, TelemetryConfig,
        TelemetryMetricsConfig, TensorSplitConfig, ThroughputConfig, config_path, config_to_toml,
        load_config, parse_config_toml, validate_config,
    };
}

#[path = "sdk/native_runtime.rs"]
pub mod native_runtime;

const DEFAULT_EMBEDDED_WORKER_STACK_SIZE: usize = 8 * 1024 * 1024;
const EMBEDDED_STARTUP_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Debug)]
pub struct EmbeddedServeStatus {
    pub api_base_url: String,
    pub console_url: String,
    pub invite_token: Option<String>,
    pub payload: serde_json::Value,
}

pub type EmbeddedMeshNodeStatus = EmbeddedServeStatus;

pub struct EmbeddedServeHandle {
    api_base_url: String,
    console_url: String,
    invite_token: Option<String>,
    control_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::api::RuntimeControlRequest>>,
    task: Option<std::thread::JoinHandle<Result<()>>>,
    _config_snapshot: NamedTempFile,
}

pub type EmbeddedMeshNodeHandle = EmbeddedServeHandle;

impl EmbeddedServeHandle {
    pub fn api_base_url(&self) -> &str {
        &self.api_base_url
    }

    pub fn console_url(&self) -> &str {
        &self.console_url
    }

    pub fn invite_token(&self) -> Option<&str> {
        self.invite_token.as_deref()
    }

    pub async fn status(&self) -> Result<EmbeddedServeStatus> {
        let payload = fetch_json(&format!("{}/api/status", self.console_url)).await?;
        Ok(EmbeddedServeStatus {
            api_base_url: self.api_base_url.clone(),
            console_url: self.console_url.clone(),
            invite_token: token_from_status(&payload),
            payload,
        })
    }

    pub async fn join_token(&self, invite_token: impl Into<String>) -> Result<()> {
        let control_tx = self
            .control_tx
            .as_ref()
            .context("embedded mesh runtime control channel is unavailable")?;
        let (resp, rx) = tokio::sync::oneshot::channel();
        control_tx
            .send(crate::api::RuntimeControlRequest::Join {
                invite_token: invite_token.into(),
                resp,
            })
            .map_err(|_| anyhow::anyhow!("embedded mesh runtime control channel is closed"))?;
        rx.await
            .context("embedded mesh runtime join response dropped")?
    }

    pub async fn stop(mut self) -> Result<()> {
        if !self.request_shutdown("sdk") && !self.task_finished() {
            anyhow::bail!("embedded mesh runtime control channel is unavailable");
        }
        let task = self
            .task
            .take()
            .context("embedded mesh runtime thread handle is unavailable")?;
        join_embedded_runtime_thread(task).await?;
        Ok(())
    }

    fn request_shutdown(&mut self, source: &'static str) -> bool {
        self.control_tx.take().is_some_and(|tx| {
            tx.send(crate::api::RuntimeControlRequest::Shutdown { source })
                .is_ok()
        })
    }

    fn task_finished(&self) -> bool {
        self.task
            .as_ref()
            .is_none_or(std::thread::JoinHandle::is_finished)
    }
}

impl Drop for EmbeddedServeHandle {
    fn drop(&mut self) {
        let _ = self.request_shutdown("sdk-drop");
    }
}

pub async fn start_embedded_node(
    mut config: EmbeddedMeshNodeConfig,
) -> Result<EmbeddedServeHandle> {
    embedded_startup::prepare_embedded_native_runtime(&config.mode)?;
    let isolated_config = prepare_isolated_config(&mut config)?;
    let config_snapshot =
        embedded_logging::snapshot_validated_config(config.storage.config_path.as_deref())?;
    config.storage.config_path = Some(embedded_logging::snapshot_path(&config_snapshot));
    drop(isolated_config);
    let (control_tx, control_rx) = tokio::sync::mpsc::unbounded_channel();
    let runtime_options = embedded_runtime_options(&config, Some(control_rx));
    let api_base_url = format!("http://127.0.0.1:{}/v1", config.http.api_port);
    let console_url = format!("http://127.0.0.1:{}", config.http.console_port);
    let startup_timeout = config.startup_timeout;
    let stack_size = embedded_worker_stack_size();
    let task = std::thread::Builder::new()
        .name("mesh-llm-embedded-serve".to_string())
        .stack_size(stack_size)
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("mesh-llm-embedded-worker")
                .thread_stack_size(stack_size)
                .build()
                .context("build embedded mesh runtime")?;
            runtime.block_on(crate::runtime::run_embedded_runtime(runtime_options))
        })
        .context("spawn embedded mesh runtime thread")?;
    let status = match wait_for_embedded_status(&console_url, startup_timeout, &task).await {
        Ok(status) => status,
        Err(error) => {
            if let Err(shutdown_error) = shutdown_failed_embedded_startup(control_tx, task).await {
                return Err(error).with_context(|| {
                    format!(
                        "failed to shut down embedded mesh runtime after startup error: {shutdown_error}"
                    )
                });
            }
            return Err(error);
        }
    };
    Ok(EmbeddedServeHandle {
        api_base_url,
        console_url,
        invite_token: token_from_status(&status),
        control_tx: Some(control_tx),
        task: Some(task),
        _config_snapshot: config_snapshot,
    })
}

pub async fn start_embedded_serve(config: EmbeddedServeConfig) -> Result<EmbeddedServeHandle> {
    start_embedded_node(config.into()).await
}

fn prepare_isolated_config(config: &mut EmbeddedMeshNodeConfig) -> Result<Option<NamedTempFile>> {
    if config.storage.config_path.is_some() || !config.storage.isolated_config {
        return Ok(None);
    }
    let mut file = NamedTempFile::new().context("create isolated embedded mesh config")?;
    file.write_all(
        b"[[plugin]]\nname = \"telemetry\"\nenabled = false\n\n[[plugin]]\nname = \"blobstore\"\nenabled = false\n",
    )
        .context("write isolated embedded mesh config")?;
    config.storage.config_path = Some(file.path().to_path_buf());
    Ok(Some(file))
}

fn embedded_runtime_options(
    config: &EmbeddedMeshNodeConfig,
    control_rx: Option<tokio::sync::mpsc::UnboundedReceiver<crate::api::RuntimeControlRequest>>,
) -> crate::runtime::EmbeddedRuntimeOptions {
    crate::runtime::EmbeddedRuntimeOptions {
        mode: match &config.mode {
            EmbeddedMeshNodeMode::SharedEndpoint { address } => {
                crate::runtime::EmbeddedRuntimeMode::SharedEndpoint {
                    address: address.clone(),
                }
            }
            EmbeddedMeshNodeMode::Serve => crate::runtime::EmbeddedRuntimeMode::Serve,
            EmbeddedMeshNodeMode::Client => crate::runtime::EmbeddedRuntimeMode::Client,
        },
        models: config.serving.models.clone(),
        join: config.network.join_tokens.clone(),
        auto: config.network.auto_join,
        api_port: config.http.api_port,
        console_port: config.http.console_port,
        mesh_name: config.network.mesh_name.clone(),
        max_vram_gb: config.serving.max_vram_gb,
        publish: config.network.publish,
        discovery_mode: match config.network.discovery_mode {
            EmbeddedMeshDiscoveryMode::Nostr => crate::runtime::EmbeddedRuntimeDiscoveryMode::Nostr,
            EmbeddedMeshDiscoveryMode::Mdns => crate::runtime::EmbeddedRuntimeDiscoveryMode::Mdns,
        },
        relay: config.network.iroh_relays.clone(),
        disable_iroh_relays: config.network.disable_iroh_relays,
        relay_auth: config
            .network
            .iroh_relay_auth
            .iter()
            .map(|(relay, token)| (relay.clone(), token.clone()))
            .collect(),
        nostr_relay: config.network.nostr_relays.clone(),
        region: config.network.region.clone(),
        node_name: config.network.node_name.clone(),
        bind_ip: config.network.bind_ip,
        bind_port: config.network.bind_port,
        listen_all: config.network.listen_all,
        enumerate_host: config.network.enumerate_host,
        owner_key: config.admission.owner_key.clone(),
        owner_required: config.admission.owner_required,
        node_label: config.admission.node_label.clone(),
        trust_policy: config.admission.trust_policy.map(Into::into),
        trust_owner: config.admission.trusted_owners.clone(),
        mesh_requirements: crate::plugin::MeshRequirementsConfig {
            min_node_version: config.admission.mesh_requirements.min_node_version.clone(),
            max_node_version: config.admission.mesh_requirements.max_node_version.clone(),
            min_protocol_version: config.admission.mesh_requirements.min_protocol_version,
            max_protocol_version: config.admission.mesh_requirements.max_protocol_version,
            require_release_attestation: config
                .admission
                .mesh_requirements
                .require_release_attestation,
            release_signer_keys: config
                .admission
                .mesh_requirements
                .release_signer_keys
                .clone(),
        },
        config_path: config.storage.config_path.clone(),
        log_format: config.log_format.into(),
        headless: !config.http.console_ui,
        control_rx,
    }
}

async fn shutdown_failed_embedded_startup(
    control_tx: tokio::sync::mpsc::UnboundedSender<crate::api::RuntimeControlRequest>,
    task: std::thread::JoinHandle<Result<()>>,
) -> Result<()> {
    let _ = control_tx.send(crate::api::RuntimeControlRequest::Shutdown {
        source: "sdk-startup-error",
    });
    join_embedded_runtime_thread_with_timeout(task, EMBEDDED_STARTUP_SHUTDOWN_TIMEOUT).await
}

async fn join_embedded_runtime_thread(task: std::thread::JoinHandle<Result<()>>) -> Result<()> {
    tokio::task::spawn_blocking(move || join_embedded_runtime_thread_blocking(task))
        .await
        .context("join embedded mesh runtime thread")?
}

async fn join_embedded_runtime_thread_with_timeout(
    task: std::thread::JoinHandle<Result<()>>,
    timeout: Duration,
) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        let deadline = Instant::now() + timeout;
        loop {
            if task.is_finished() {
                return join_embedded_runtime_thread_blocking(task);
            }
            if Instant::now() >= deadline {
                anyhow::bail!(
                    "timed out after {:?} waiting for embedded mesh runtime thread to exit",
                    timeout
                );
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    })
    .await
    .context("join embedded mesh runtime thread after startup failure")?
}

fn join_embedded_runtime_thread_blocking(task: std::thread::JoinHandle<Result<()>>) -> Result<()> {
    task.join()
        .map_err(|_| anyhow::anyhow!("embedded mesh runtime thread panicked"))?
}

async fn wait_for_embedded_status(
    console_url: &str,
    timeout: Duration,
    task: &std::thread::JoinHandle<Result<()>>,
) -> Result<serde_json::Value> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if task.is_finished() {
            anyhow::bail!("embedded mesh runtime exited before the console became ready");
        }
        if let Ok(status) = fetch_json(&format!("{console_url}/api/status")).await {
            return Ok(status);
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for embedded mesh console at {console_url}");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn fetch_json(url: &str) -> Result<serde_json::Value> {
    let response = reqwest::Client::new()
        .get(url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("GET {url} returned an error status"))?;
    response
        .json::<serde_json::Value>()
        .await
        .with_context(|| format!("decode JSON from {url}"))
}

fn token_from_status(payload: &serde_json::Value) -> Option<String> {
    payload
        .get("token")
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
}

fn embedded_worker_stack_size() -> usize {
    std::env::var("MESH_TOKIO_STACK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_EMBEDDED_WORKER_STACK_SIZE)
}

#[derive(Clone, Debug)]
pub struct EmbeddedChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
pub struct EmbeddedServingController {
    inner: Arc<Mutex<EmbeddedServingState>>,
}

struct EmbeddedServingState {
    next_instance_id: u64,
    default_device_policy: DevicePolicy,
    /// Maps (model_ref, profile) -> served model.
    /// The compound key ensures two profiles of the same model coexist
    /// without silently replacing each other.
    models: HashMap<(String, String), Arc<EmbeddedServedModel>>,
}

struct EmbeddedServedModel {
    served: ServedModel,
    handle: Option<SkippyModelHandle>,
}

impl Default for EmbeddedServingController {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddedServingController {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(EmbeddedServingState {
                next_instance_id: 1,
                default_device_policy: DevicePolicy::Auto,
                models: HashMap::new(),
            })),
        }
    }

    pub async fn chat_completion_text(
        &self,
        model: &str,
        messages: Vec<EmbeddedChatMessage>,
    ) -> Result<String> {
        let loaded = self.loaded_model(model).await?;
        let request = ChatCompletionRequest {
            model: loaded.served.model_id.clone(),
            messages: messages
                .into_iter()
                .map(|message| ChatMessage {
                    role: message.role,
                    content: Some(MessageContent::Text(message.content)),
                    extra: BTreeMap::new(),
                })
                .collect(),
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            logprobs: None,
            top_logprobs: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            user: None,
            stop: None,
            seed: None,
            reasoning: None,
            reasoning_effort: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            stream_options: None,
            extra: BTreeMap::new(),
        };
        let handle = loaded
            .handle
            .as_ref()
            .context("model handle not available")?;
        let response = handle
            .chat_completion(request)
            .await
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default())
    }

    pub async fn model_list(&self) -> Vec<(String, String)> {
        self.inner
            .lock()
            .await
            .models
            .values()
            .map(|model| {
                (
                    model.served.model_id.clone(),
                    model.served.model_ref.clone(),
                )
            })
            .collect()
    }

    async fn loaded_model(&self, model: &str) -> Result<Arc<EmbeddedServedModel>> {
        let state = self.inner.lock().await;
        state
            .models
            .values()
            .find(|loaded| {
                loaded.served.model_id == model
                    || loaded.served.model_ref == model
                    || loaded.served.instance_id.as_deref() == Some(model)
            })
            .cloned()
            .with_context(|| format!("model is not loaded for local serving: {model}"))
    }
}

impl ServingController for EmbeddedServingController {
    fn load<'a>(&'a self, request: LoadModelRequest) -> ServingFuture<'a, ServedModel> {
        Box::pin(async move {
            let model_path =
                models::resolve_model_spec_with_progress(Path::new(&request.model_ref), true)
                    .await
                    .with_context(|| format!("resolve model {}", request.model_ref))?;
            let model_id = models::model_ref_for_path(&model_path);
            let device_policy = self.effective_device_policy(&request.device_policy).await;
            reject_obvious_vram_overcommit(&model_path, &device_policy)?;
            let options = apply_device_policy(
                SkippyModelLoadOptions::for_direct_gguf(&model_id, &model_path),
                &device_policy,
            )?;
            let handle = tokio::task::spawn_blocking(move || SkippyModelHandle::load(options))
                .await
                .context("join embedded model load task")??;
            let capabilities = models::runtime_verified_model_capabilities(
                &model_id,
                &model_path,
                models::RuntimeMediaCapabilityEvidence::default(),
            );

            let mut state = self.inner.lock().await;
            let instance_id = format!("embedded-{}", state.next_instance_id);
            state.next_instance_id += 1;
            let model_ref = request.model_ref.clone();
            let profile = request.profile.clone();
            let served = ServedModel {
                model_ref: request.model_ref,
                profile: profile.clone(),
                model_id: model_id.clone(),
                instance_id: Some(instance_id),
                state: ServingModelState::Ready,
                backend: Some("skippy".to_string()),
                capabilities,
                context_length: Some(handle.status().ctx_size),
                error: None,
            };
            state.models.insert(
                (model_ref, profile),
                Arc::new(EmbeddedServedModel {
                    served: served.clone(),
                    handle: Some(handle),
                }),
            );
            Ok(served)
        })
    }

    fn unload<'a>(&'a self, request: UnloadModelRequest) -> ServingFuture<'a, ()> {
        Box::pin(async move {
            let mut state = self.inner.lock().await;
            match request.target {
                UnloadTarget::Model(model_ref) => {
                    let key = resolve_model_unload_key(&state.models, &model_ref)?;
                    state.models.remove(&key);
                    Ok(())
                }
                UnloadTarget::Instance(instance_id) => {
                    let keys = matching_instance_unload_keys(&state.models, &instance_id);
                    match keys.as_slice() {
                        [key] => {
                            state.models.remove(key);
                            Ok(())
                        }
                        [] => {
                            anyhow::bail!("instance is not loaded for local serving: {instance_id}")
                        }
                        _ => {
                            anyhow::bail!(
                                "ambiguous instance unload target {instance_id}: matched {} loaded instances",
                                keys.len()
                            )
                        }
                    }
                }
            }
        })
    }

    fn served_models<'a>(&'a self) -> ServingFuture<'a, Vec<ServedModel>> {
        Box::pin(async move {
            Ok(self
                .inner
                .lock()
                .await
                .models
                .values()
                .map(|model| model.served.clone())
                .collect())
        })
    }

    fn status<'a>(&'a self) -> ServingFuture<'a, ServingStatus> {
        Box::pin(async move {
            let models = self.served_models().await?;
            Ok(ServingStatus {
                enabled: true,
                models,
            })
        })
    }

    fn set_device_policy<'a>(&'a self, policy: DevicePolicy) -> ServingFuture<'a, ()> {
        Box::pin(async move {
            self.inner.lock().await.default_device_policy = policy;
            Ok(())
        })
    }
}

impl EmbeddedServingController {
    async fn effective_device_policy(&self, request_policy: &DevicePolicy) -> DevicePolicy {
        match request_policy {
            DevicePolicy::Auto => self.inner.lock().await.default_device_policy.clone(),
            explicit => explicit.clone(),
        }
    }
}

fn resolve_model_unload_key(
    models: &HashMap<(String, String), Arc<EmbeddedServedModel>>,
    target: &str,
) -> Result<(String, String)> {
    let keys = matching_model_unload_keys(models, target);
    match keys.as_slice() {
        [key] => Ok(key.clone()),
        [] => anyhow::bail!("model is not loaded for local serving: {target}"),
        _ => anyhow::bail!(
            "ambiguous model unload target {target}: matched {} loaded profiles; use model#profile or an instance id",
            keys.len()
        ),
    }
}

fn matching_model_unload_keys(
    models: &HashMap<(String, String), Arc<EmbeddedServedModel>>,
    target: &str,
) -> Vec<(String, String)> {
    let (model_target, profile_target) = split_model_ref_and_profile(target);
    models
        .iter()
        .filter_map(|(key, loaded)| {
            let model_matches =
                loaded.served.model_id == model_target || loaded.served.model_ref == model_target;
            let profile_matches = profile_target
                .map(|profile| loaded.served.profile == profile)
                .unwrap_or(true);
            (model_matches && profile_matches).then(|| key.clone())
        })
        .collect()
}

fn matching_instance_unload_keys(
    models: &HashMap<(String, String), Arc<EmbeddedServedModel>>,
    instance_id: &str,
) -> Vec<(String, String)> {
    models
        .iter()
        .filter(|(_, loaded)| loaded.served.instance_id.as_deref() == Some(instance_id))
        .map(|(key, _)| key.clone())
        .collect()
}

fn split_model_ref_and_profile(model_ref: &str) -> (&str, Option<&str>) {
    if let Some(hash_pos) = model_ref.rfind('#') {
        (&model_ref[..hash_pos], Some(&model_ref[hash_pos + 1..]))
    } else {
        (model_ref, None)
    }
}

fn reject_obvious_vram_overcommit(model_path: &Path, policy: &DevicePolicy) -> Result<()> {
    if matches!(policy, DevicePolicy::Cpu) {
        return Ok(());
    }
    let survey = hardware::query(&[Metric::GpuFacts]);
    let total_vram_bytes = survey.gpus.iter().map(|gpu| gpu.vram_bytes).sum::<u64>();
    if total_vram_bytes == 0 {
        return Ok(());
    }
    let model_size_bytes = std::fs::metadata(model_path)
        .with_context(|| format!("read model metadata {}", model_path.display()))?
        .len();
    anyhow::ensure!(
        model_size_bytes <= total_vram_bytes,
        "model file is larger than detected total GPU VRAM: model={} bytes, vram={} bytes",
        model_size_bytes,
        total_vram_bytes
    );
    Ok(())
}

fn apply_device_policy(
    mut options: SkippyModelLoadOptions,
    policy: &DevicePolicy,
) -> Result<SkippyModelLoadOptions> {
    match policy {
        DevicePolicy::Auto => Ok(options),
        DevicePolicy::Cpu => {
            options.n_gpu_layers = 0;
            Ok(options)
        }
        DevicePolicy::Gpu { device_ids } => {
            if device_ids.is_empty() {
                return Ok(options);
            }
            anyhow::ensure!(
                device_ids.len() == 1,
                "embedded serving can pin one GPU per loaded model; got {} device ids",
                device_ids.len()
            );
            let survey = hardware::query(&[Metric::GpuFacts]);
            let gpu =
                hardware::resolve_pinned_gpu_strict(Some(device_ids[0].as_str()), &survey.gpus)
                    .with_context(|| {
                        format!(
                            "resolve requested serving GPU '{}' from local hardware",
                            device_ids[0]
                        )
                    })?;
            let backend_device = gpu.backend_device.clone().with_context(|| {
                format!(
                    "requested serving GPU '{}' has no backend device name",
                    device_ids[0]
                )
            })?;
            Ok(options.with_selected_device(SkippyDeviceDescriptor {
                backend_device,
                stable_id: gpu.stable_id.clone(),
                index: Some(gpu.index),
                vram_bytes: Some(gpu.vram_bytes),
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn explicit_load_policy_overrides_stored_default() {
        let controller = EmbeddedServingController::new();
        controller
            .set_device_policy(DevicePolicy::Cpu)
            .await
            .unwrap();

        assert_eq!(
            controller
                .effective_device_policy(&DevicePolicy::Gpu {
                    device_ids: vec!["metal:0".to_string()],
                })
                .await,
            DevicePolicy::Gpu {
                device_ids: vec!["metal:0".to_string()],
            }
        );
    }

    #[tokio::test]
    async fn auto_load_policy_uses_stored_default() {
        let controller = EmbeddedServingController::new();
        controller
            .set_device_policy(DevicePolicy::Cpu)
            .await
            .unwrap();

        assert_eq!(
            controller
                .effective_device_policy(&DevicePolicy::Auto)
                .await,
            DevicePolicy::Cpu
        );
    }

    #[test]
    fn cpu_policy_forces_cpu_only_runtime_load() {
        let options =
            apply_device_policy(test_load_options(), &DevicePolicy::Cpu).expect("cpu policy");

        assert_eq!(options.n_gpu_layers, 0);
        assert!(options.selected_device.is_none());
    }

    #[test]
    fn multi_gpu_policy_is_rejected_instead_of_ignored() {
        let err = apply_device_policy(
            test_load_options(),
            &DevicePolicy::Gpu {
                device_ids: vec!["metal:0".to_string(), "metal:1".to_string()],
            },
        )
        .expect_err("multi-gpu policy should be rejected");

        assert!(
            err.to_string().contains("can pin one GPU per loaded model"),
            "{err}"
        );
    }

    #[test]
    fn embedded_serve_config_maps_to_runtime_surface() {
        let config = EmbeddedMeshNodeConfig::builder()
            .model("Qwen3-8B-Q4_K_M")
            .mesh_name("sprout")
            .api_port(19337)
            .console_port(13131)
            .max_vram_gb(3.0)
            .iroh_relay("https://relay.example")
            .iroh_relay_auth("https://relay.example", "token")
            .disable_iroh_relays(true)
            .nostr_relay("wss://nostr.example")
            .bind_port(17777)
            .owner_key("/tmp/sprout-owner.json")
            .owner_required(true)
            .node_label("sprout-desktop")
            .trust_policy(EmbeddedTrustPolicy::RequireOwned)
            .trust_owner("owner-a")
            .trust_owner("owner-b")
            .min_node_version("0.65.0")
            .signed_join_tokens(true)
            .build();
        let options = embedded_runtime_options(&config, None);

        assert_eq!(options.mode, crate::runtime::EmbeddedRuntimeMode::Serve);
        assert_eq!(options.models, vec!["Qwen3-8B-Q4_K_M".to_string()]);
        assert_eq!(options.api_port, 19337);
        assert_eq!(options.console_port, 13131);
        assert_eq!(options.mesh_name.as_deref(), Some("sprout"));
        assert_eq!(options.max_vram_gb, Some(3.0));
        assert_embedded_runtime_network_options(&options);
        assert_embedded_runtime_admission_options(&options);
        assert_eq!(options.log_format, mesh_llm_events::LogFormat::Json);
        assert!(options.headless);
    }

    fn assert_embedded_runtime_network_options(options: &crate::runtime::EmbeddedRuntimeOptions) {
        assert_eq!(options.relay, vec!["https://relay.example".to_string()]);
        assert_eq!(
            options.relay_auth,
            vec![("https://relay.example".to_string(), "token".to_string())]
        );
        assert!(options.disable_iroh_relays);
        assert_eq!(options.nostr_relay, vec!["wss://nostr.example".to_string()]);
        assert_eq!(options.bind_port, Some(17777));
    }

    fn assert_embedded_runtime_admission_options(options: &crate::runtime::EmbeddedRuntimeOptions) {
        assert_eq!(
            options.owner_key.as_deref(),
            Some(std::path::Path::new("/tmp/sprout-owner.json"))
        );
        assert!(options.owner_required);
        assert_eq!(options.node_label.as_deref(), Some("sprout-desktop"));
        assert_eq!(
            options.trust_policy,
            Some(crate::crypto::TrustPolicy::RequireOwned)
        );
        assert_eq!(options.trust_owner, vec!["owner-a", "owner-b"]);
        assert_eq!(
            options.mesh_requirements.min_node_version.as_deref(),
            Some("0.65.0")
        );
        assert_eq!(options.mesh_requirements.min_protocol_version, Some(1));
        assert!(!options.mesh_requirements.require_release_attestation);
    }

    #[test]
    fn signed_join_tokens_sets_genesis_requirement_without_lowering_existing_bound() {
        let config = EmbeddedMeshNodeConfig::builder()
            .signed_join_tokens(true)
            .build();
        assert_eq!(
            config.admission.mesh_requirements.min_protocol_version,
            Some(SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION)
        );

        let config = EmbeddedMeshNodeConfig::builder()
            .min_protocol_version(2)
            .signed_join_tokens(true)
            .build();
        assert_eq!(
            config.admission.mesh_requirements.min_protocol_version,
            Some(2)
        );
    }

    #[test]
    fn embedded_client_config_maps_to_auto_join_runtime_surface() {
        let config = EmbeddedMeshNodeConfig::builder()
            .client()
            .join_token("mesh-test-token")
            .auto_join(true)
            .api_port(29337)
            .console_port(23131)
            .discovery_mode(EmbeddedMeshDiscoveryMode::Mdns)
            .listen_all(true)
            .enumerate_host(false)
            .console_ui(true)
            .build();
        let options = embedded_runtime_options(&config, None);

        assert_eq!(options.mode, crate::runtime::EmbeddedRuntimeMode::Client);
        assert_eq!(options.join, vec!["mesh-test-token".to_string()]);
        assert!(options.auto);
        assert!(options.models.is_empty());
        assert_eq!(options.api_port, 29337);
        assert_eq!(options.console_port, 23131);
        assert_eq!(
            options.discovery_mode,
            crate::runtime::EmbeddedRuntimeDiscoveryMode::Mdns
        );
        assert!(options.listen_all);
        assert!(!options.enumerate_host);
        assert!(!options.headless);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "opens localhost mesh runtime sockets"]
    async fn embedded_client_start_stop_exposes_local_status() {
        let api_port = free_local_port();
        let console_port = free_local_port();
        let handle = start_embedded_serve(EmbeddedServeConfig {
            mode: EmbeddedMeshNodeMode::Client,
            api_port,
            console_port,
            startup_timeout: Duration::from_secs(15),
            ..EmbeddedServeConfig::default()
        })
        .await
        .expect("start embedded mesh client");

        let status = handle.status().await.expect("embedded status");
        assert_eq!(
            status.api_base_url,
            format!("http://127.0.0.1:{api_port}/v1")
        );
        assert_eq!(
            status.console_url,
            format!("http://127.0.0.1:{console_port}")
        );
        assert!(status.payload.is_object());

        handle.stop().await.expect("stop embedded mesh client");
    }

    fn make_served_model(
        model_ref: &str,
        profile: &str,
        instance_id: u64,
    ) -> Arc<EmbeddedServedModel> {
        Arc::new(EmbeddedServedModel {
            served: ServedModel {
                model_ref: model_ref.to_string(),
                profile: profile.to_string(),
                model_id: format!("{model_ref}-model-id"),
                instance_id: Some(format!("embedded-{instance_id}")),
                state: ServingModelState::Ready,
                backend: Some("skippy".to_string()),
                capabilities: ModelCapabilities::default(),
                context_length: Some(4096),
                error: None,
            },
            handle: None,
        })
    }

    #[tokio::test]
    async fn model_list_returns_both_profiles_for_same_model() {
        let controller = EmbeddedServingController::new();
        {
            let mut state = controller.inner.lock().await;
            state.models.insert(
                ("model-a".to_string(), "gaming".to_string()),
                make_served_model("model-a", "gaming", 1),
            );
            state.models.insert(
                ("model-a".to_string(), "coding".to_string()),
                make_served_model("model-a", "coding", 2),
            );
        }

        let models = controller.model_list().await;
        assert_eq!(models.len(), 2, "should return both profile entries");
        assert_eq!(models[0].1, "model-a", "model_ref matches");
        assert_eq!(models[1].1, "model-a", "both entries have same model_ref");
    }

    #[tokio::test]
    async fn served_models_returns_both_profiles() {
        let controller = EmbeddedServingController::new();
        {
            let mut state = controller.inner.lock().await;
            state.models.insert(
                ("model-a".to_string(), "gaming".to_string()),
                make_served_model("model-a", "gaming", 1),
            );
            state.models.insert(
                ("model-a".to_string(), "coding".to_string()),
                make_served_model("model-a", "coding", 2),
            );
        }

        let list = controller.served_models().await.unwrap();
        assert_eq!(list.len(), 2, "should return both profile entries");
        let profiles: Vec<&str> = list.iter().map(|m| m.profile.as_str()).collect();
        assert!(profiles.contains(&"gaming"));
        assert!(profiles.contains(&"coding"));
    }

    #[tokio::test]
    async fn unload_by_bare_model_rejects_ambiguous_profiles() {
        let controller = EmbeddedServingController::new();
        {
            let mut state = controller.inner.lock().await;
            state.models.insert(
                ("model-a".to_string(), "gaming".to_string()),
                make_served_model("model-a", "gaming", 1),
            );
            state.models.insert(
                ("model-a".to_string(), "coding".to_string()),
                make_served_model("model-a", "coding", 2),
            );
        }

        let err = controller
            .unload(UnloadModelRequest {
                target: UnloadTarget::Model("model-a".to_string()),
                options: UnloadOptions::default(),
            })
            .await
            .expect_err("bare model unload should reject ambiguous profiles");

        assert!(
            err.to_string().contains("ambiguous"),
            "error should explain ambiguity: {err}"
        );
        let remaining = controller.served_models().await.unwrap();
        assert_eq!(
            remaining.len(),
            2,
            "ambiguous bare unload must not remove an arbitrary profile"
        );
    }

    #[tokio::test]
    async fn unload_by_profile_qualified_model_removes_only_target_profile() {
        let controller = EmbeddedServingController::new();
        {
            let mut state = controller.inner.lock().await;
            state.models.insert(
                ("model-a".to_string(), "gaming".to_string()),
                make_served_model("model-a", "gaming", 1),
            );
            state.models.insert(
                ("model-a".to_string(), "coding".to_string()),
                make_served_model("model-a", "coding", 2),
            );
        }

        controller
            .unload(UnloadModelRequest {
                target: UnloadTarget::Model("model-a#gaming".to_string()),
                options: UnloadOptions::default(),
            })
            .await
            .expect("unload gaming profile");

        let remaining = controller.served_models().await.unwrap();
        assert_eq!(remaining.len(), 1, "one entry should remain");
        assert_eq!(
            remaining[0].profile.as_str(),
            "coding",
            "coding profile should survive"
        );
    }

    #[tokio::test]
    async fn unload_by_instance_id_removes_only_target_entry() {
        let controller = EmbeddedServingController::new();
        {
            let mut state = controller.inner.lock().await;
            state.models.insert(
                ("model-a".to_string(), "gaming".to_string()),
                make_served_model("model-a", "gaming", 1),
            );
            state.models.insert(
                ("model-a".to_string(), "coding".to_string()),
                make_served_model("model-a", "coding", 2),
            );
        }

        controller
            .unload(UnloadModelRequest {
                target: UnloadTarget::Instance("embedded-1".to_string()),
                options: UnloadOptions::default(),
            })
            .await
            .expect("unload gaming profile");

        let remaining = controller.served_models().await.unwrap();
        assert_eq!(remaining.len(), 1, "one entry should remain");
        assert_eq!(
            remaining[0].profile.as_str(),
            "coding",
            "coding profile should survive"
        );
    }

    fn test_load_options() -> SkippyModelLoadOptions {
        SkippyModelLoadOptions::for_direct_gguf("test-model", PathBuf::from("/tmp/test.gguf"))
    }

    fn free_local_port() -> u16 {
        std::net::TcpListener::bind(("127.0.0.1", 0))
            .expect("bind local port")
            .local_addr()
            .expect("local addr")
            .port()
    }
}
