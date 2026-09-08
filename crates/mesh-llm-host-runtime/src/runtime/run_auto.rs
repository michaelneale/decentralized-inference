use super::daemon_startup::{check_mode_conflicts, resolve_effective_mode};
use super::plugin_host_role;
use super::startup_identity::{emit_private_mesh_name_warning, handle_public_identity_transition};
use super::status::mesh_guardrail_mode_to_openai;
use super::{
    AutoRuntimeNodeSetup, BootstrapProxyStopTx, DashboardContextUsage, ManagedModelController,
    ModelTargetReconciliationPolicy, ModelTargetReconciliationState, OpenAiGuardrailPolicyHandle,
    PreparedRuntimeStartup, RunAutoAdditionalModelsContext, RunAutoConsoleStateContext,
    RunAutoRuntimeLifecycleContext, RunAutoServingSurface, RunAutoServingSurfaceContext,
    RuntimeCapacityLedger, RuntimeDashboardSnapshotProvider, RuntimeEvent, RuntimeInstanceRegistry,
    RuntimeModelHandleEntry, RuntimeOperationalEvent, RuntimeOptions,
    RuntimeResourcePlanningProfile, RuntimeSurface, SkippyNativeLogForwardingGuard,
    StartupLocalModelTask, StartupMeshCreationState, StartupModelPlan, StartupModelSpec,
    StartupReadyReporter, bridge_skippy_native_logs, build_serving_list, cli_has_explicit_models,
    configure_skippy_native_logging, emit_configuration_ui_read_only_hint,
    initialize_embedded_runtime_entrypoint, initialize_runtime_entrypoint,
    maybe_discover_join_candidates, next_runtime_instance_id, nostr_rediscovery, nostr_relays,
    openai_guardrail_policy_handle, owner_runtime_config, prepare_runtime_startup,
    publish_initial_openai_guardrails_status, record_first_joined_mesh_ts,
    record_runtime_operational_event, resolve_runtime_owner_key_path,
    resolve_startup_mesh_creation_state, run_auto_join_mesh_phase, run_auto_model_identity,
    run_auto_model_path_or_shutdown, run_auto_runtime_loop_and_shutdown, run_local_model_only,
    runtime_data_producer_for_console, runtime_startup_requirements, setup_run_auto_console_state,
    setup_run_auto_serving_surface, spawn_embedded_runtime_control_forwarder,
    spawn_run_auto_additional_model_tasks, spawn_run_auto_discovery_publisher,
    start_run_auto_bootstrap_proxy, startup_device_override, startup_local_model_loop,
    swarm_capture_observer_requested,
};
use crate::api;
use crate::inference::{election, skippy};
use crate::mesh::{self, NodeRole};
use crate::models;
use crate::network::{
    affinity, discovery as mesh_discovery,
    lan_bootstrap::{effective_quic_bind_ip, spawn_mdns_reverse_dial},
    nostr, tunnel,
};
use crate::plugin;
use crate::runtime::release_attestation;
use crate::runtime::survey;
use crate::runtime::{
    InstanceLifecycleRecord, InstanceLifecycleState, tracing_writer::init_audit_logging,
};
use crate::system::{autoupdate, benchmark, hardware};
use anyhow::{Context, Result};
use mesh_llm_events::{LogFormat, OutputEvent, RuntimeStatus, emit_event, output_sink};
use skippy_protocol::FlashAttentionType;
use std::collections::{BTreeSet, HashMap};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU16},
};

#[expect(
    dead_code,
    reason = "the legacy advertised-model selection lane remains available to compatibility helpers and focused tests"
)]
pub(super) enum RunAutoModelSelection {
    Model(PathBuf),
    Shutdown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RuntimeUnloadOwner {
    Runtime,
    Managed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RuntimeUnloadCandidate {
    pub(super) owner: RuntimeUnloadOwner,
    pub(super) instance_id: String,
    pub(super) model_name: String,
    pub(super) profile: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EmbeddedRuntimeMode {
    Serve,
    Client,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EmbeddedRuntimeDiscoveryMode {
    Nostr,
    Mdns,
}

pub(crate) struct EmbeddedRuntimeOptions {
    pub(crate) mode: EmbeddedRuntimeMode,
    pub(crate) models: Vec<String>,
    pub(crate) join: Vec<String>,
    pub(crate) auto: bool,
    pub(crate) api_port: u16,
    pub(crate) console_port: u16,
    pub(crate) mesh_name: Option<String>,
    pub(crate) max_vram_gb: Option<f64>,
    pub(crate) publish: bool,
    pub(crate) discovery_mode: EmbeddedRuntimeDiscoveryMode,
    pub(crate) relay: Vec<String>,
    pub(crate) relay_auth: Vec<(String, String)>,
    pub(crate) disable_iroh_relays: bool,
    pub(crate) nostr_relay: Vec<String>,
    pub(crate) region: Option<String>,
    pub(crate) node_name: Option<String>,
    pub(crate) bind_ip: Option<IpAddr>,
    pub(crate) bind_port: Option<u16>,
    pub(crate) listen_all: bool,
    pub(crate) enumerate_host: bool,
    pub(crate) owner_key: Option<PathBuf>,
    pub(crate) owner_required: bool,
    pub(crate) node_label: Option<String>,
    pub(crate) trust_policy: Option<crate::crypto::TrustPolicy>,
    pub(crate) trust_owner: Vec<String>,
    pub(crate) mesh_requirements: crate::plugin::MeshRequirementsConfig,
    pub(crate) config_path: Option<PathBuf>,
    pub(crate) log_format: LogFormat,
    pub(crate) headless: bool,
    pub(crate) control_rx: Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
}

impl EmbeddedRuntimeOptions {
    pub(super) fn runtime_surface(&self) -> RuntimeSurface {
        match self.mode {
            EmbeddedRuntimeMode::Serve => RuntimeSurface::Serve,
            EmbeddedRuntimeMode::Client => RuntimeSurface::Client,
        }
    }
}

pub(super) fn acquire_instance_runtime(
    options: &RuntimeOptions,
) -> Option<Arc<crate::runtime::instance::InstanceRuntime>> {
    if options.client && !swarm_capture_observer_requested(options) {
        return None;
    }

    match crate::runtime::instance::InstanceRuntime::acquire(std::process::id()) {
        Ok(rt) => Some(Arc::new(rt)),
        Err(err) => {
            tracing::warn!("failed to acquire instance runtime: {err}");
            None
        }
    }
}

pub(super) fn write_runtime_owner_metadata(
    runtime: Option<&Arc<crate::runtime::instance::InstanceRuntime>>,
    console_port: u16,
) {
    let Some(rt) = runtime else {
        return;
    };

    let started_at =
        crate::runtime::instance::validate::current_process_start_time_unix().unwrap_or(0);
    let owner_meta = serde_json::json!({
        "pid": std::process::id(),
        "api_port": console_port,
        "version": crate::BUILD_VERSION,
        "started_at_unix": started_at,
        "mesh_llm_binary": std::env::current_exe()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default(),
    });
    let owner_path = rt.dir().join("owner.json");
    if let Ok(json) = serde_json::to_string_pretty(&owner_meta) {
        let _ = crate::runtime::instance::write_text_file_atomic(&owner_path, &json);
    }
}

pub(crate) async fn run() -> Result<()> {
    initialize_runtime_entrypoint()?;
    run_runtime_cli(RuntimeOptions::default(), None, None, None).await
}

pub(crate) async fn run_cli(
    options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    legacy_warning: Option<String>,
) -> Result<()> {
    initialize_runtime_entrypoint()?;
    run_runtime_cli(options, explicit_surface, legacy_warning, None).await
}

pub(crate) async fn run_embedded_runtime(mut options: EmbeddedRuntimeOptions) -> Result<()> {
    initialize_embedded_runtime_entrypoint()?;
    crate::sdk::embedded_logging::initialize_embedded_logging(options.config_path.as_deref())
        .await?;

    let surface = options.runtime_surface();
    let control_rx = options.control_rx.take();
    let options = options_from_embedded_options(options);
    run_runtime_cli(options, Some(surface), None, control_rx).await
}

pub(super) fn options_from_embedded_options(embedded: EmbeddedRuntimeOptions) -> RuntimeOptions {
    RuntimeOptions {
        log_format: embedded.log_format,
        client: matches!(embedded.mode, EmbeddedRuntimeMode::Client),
        model: embedded.models.into_iter().map(PathBuf::from).collect(),
        join: embedded.join,
        auto: embedded.auto,
        port: embedded.api_port,
        console: embedded.console_port,
        headless: embedded.headless,
        publish: embedded.publish,
        mesh_name: embedded.mesh_name,
        max_vram: embedded.max_vram_gb,
        mesh_discovery_mode: match embedded.discovery_mode {
            EmbeddedRuntimeDiscoveryMode::Nostr => mesh_discovery::MeshDiscoveryMode::Nostr,
            EmbeddedRuntimeDiscoveryMode::Mdns => mesh_discovery::MeshDiscoveryMode::Mdns,
        },
        relay: embedded.relay,
        relay_auth: embedded.relay_auth,
        disable_iroh_relays: embedded.disable_iroh_relays,
        nostr_relay: embedded.nostr_relay,
        region: embedded.region,
        name: embedded.node_name,
        bind_ip: embedded.bind_ip,
        bind_port: embedded.bind_port,
        listen_all: embedded.listen_all,
        no_enumerate_host: !embedded.enumerate_host,
        owner_key: embedded.owner_key,
        owner_required: embedded.owner_required,
        node_label: embedded.node_label,
        trust_policy: embedded.trust_policy,
        trust_owner: embedded.trust_owner,
        min_node_version: embedded.mesh_requirements.min_node_version,
        max_node_version: embedded.mesh_requirements.max_node_version,
        min_protocol_version: embedded.mesh_requirements.min_protocol_version,
        max_protocol_version: embedded.mesh_requirements.max_protocol_version,
        require_release_attestation: embedded.mesh_requirements.require_release_attestation,
        release_signer_key: embedded.mesh_requirements.release_signer_keys,
        config: embedded.config_path,
        ..RuntimeOptions::default()
    }
}

pub(super) async fn run_runtime_cli(
    mut options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    legacy_warning: Option<String>,
    embedded_control_rx: Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
) -> Result<()> {
    options.validate_discovery_mode_args()?;

    if let Some(warning) = legacy_warning {
        let _ = emit_event(OutputEvent::Warning {
            message: warning,
            context: None,
        });
    }

    // These topologies are intentionally selected before plugin startup,
    // release lookup, config-driven mesh discovery, and `mesh::Node::start`.
    // Load config only to configure their optional audit sink; failures stay
    // nonfatal so they retain their early-return behavior.
    if options.local_model_only {
        initialize_early_topology_audit_logging(&mut options)?;
        return run_local_model_only(options).await;
    }

    if let Some(name) = options.plugin.clone() {
        initialize_early_topology_audit_logging(&mut options)?;
        return plugin::run_plugin_process(name).await;
    }

    let checked_updates = autoupdate::maybe_auto_update(autoupdate::AutoUpdateOptions {
        auto_update: options.auto_update,
        plugin_requested: options.plugin.is_some(),
        command_is_update: options.command_is_update,
        llama_flavor: options.llama_flavor,
        current_version: crate::BUILD_VERSION,
    })
    .await?;

    // Finish the release check before startup continues.
    if !checked_updates
        && !options.command_is_update
        && !options.command_uses_machine_output
        && let Some(notice) = autoupdate::check_for_update(crate::BUILD_VERSION).await
    {
        let _ = emit_event(OutputEvent::Info {
            message: notice.message,
            context: None,
        });
    }

    let mut config = plugin::load_config(options.config.as_deref())?;
    let effective_mode = resolve_effective_mode(&options, config.runtime.mode);
    if let Err(error) = check_mode_conflicts(&options, explicit_surface, config.runtime.mode) {
        anyhow::bail!("mode conflict: {error}");
    }
    options.client = effective_mode == mesh_llm_config::RuntimeMode::Client;
    anyhow::ensure!(
        !options.client
            || (options.checkpoint_quantization.is_none() && options.checkpoint_imatrix.is_none()),
        "--checkpoint-quantization and --checkpoint-imatrix are only valid with mesh-llm serve"
    );
    apply_runtime_cli_speculative_overrides(&mut config, options.speculative_overrides.as_ref());
    apply_runtime_cli_checkpoint_overrides(
        &mut config,
        options.checkpoint_quantization.as_deref(),
        options.checkpoint_imatrix.as_deref(),
    )?;
    apply_runtime_config_options(&mut options, &config);

    initialize_audit_logging_for_options(&options)?;

    let startup_mesh_creation_state = resolve_startup_mesh_creation_state(&options, &config)?;
    let cli_has_explicit_models = cli_has_explicit_models(&options);
    let has_config_models = !config.models.is_empty();
    let has_startup_models = cli_has_explicit_models || has_config_models;

    // Acquire the per-instance runtime directory and flock. Plain --client still
    // skips this, but capture observers register so detached runs can be found
    // and stopped by `mesh-llm stop`.
    // Wrap in Arc so it can be cheaply shared with local model tasks.
    let runtime = acquire_instance_runtime(&options);

    // Write owner.json into the runtime dir so sibling-instance discovery can find us.
    write_runtime_owner_metadata(runtime.as_ref(), options.console);

    // Publication intent is now explicit only: --publish gates Nostr discovery.
    // --mesh-name alone never implies publication (Issue #240).

    // Warn users who set --mesh-name without --publish — but only when they
    // are creating a new mesh, not when they are joining one via --discover
    // or --auto (where --mesh-name is just a filter for which mesh to join).
    emit_private_mesh_name_warning(&options);

    // --- Public-to-private identity transition ---
    // If the previous run was public (--auto or --publish) but this run is
    // private, clear the stored identity so the private mesh gets a fresh key
    // that isn't associated with the old public listing.
    handle_public_identity_transition(&options)?;

    let mut auto_join_candidates: Vec<(String, Option<String>)> = Vec::new();
    maybe_discover_join_candidates(&mut options, has_startup_models, &mut auto_join_candidates)
        .await?;
    let Some(PreparedRuntimeStartup {
        startup_specs,
        requested_model_names,
        bin_dir,
    }) = prepare_runtime_startup(&options, &config, explicit_surface, effective_mode).await?
    else {
        return Ok(());
    };

    run_auto(RunAutoContext {
        options,
        config,
        startup_mesh_creation_state,
        startup_specs,
        requested_model_names,
        bin_dir,
        runtime,
        auto_join_candidates,
        embedded_control_rx,
    })
    .await
}

pub(super) fn apply_runtime_config_options(
    options: &mut RuntimeOptions,
    config: &plugin::MeshConfig,
) {
    options.debug |= config.runtime.debug;
    options.listen_all |= config.runtime.listen_all;

    let cli_path_supplied = options.audit_log_path.is_some();
    let audit = &config.logging.audit;
    let config_enabled =
        config.logging.enabled && audit.enabled.unwrap_or(audit.log_path.is_some());
    options.audit_max_file_size = audit
        .max_file_size_mb
        .unwrap_or(100)
        .saturating_mul(1024 * 1024);
    options.audit_max_files = audit.max_files.unwrap_or(10);
    if !cli_path_supplied && config_enabled {
        options.audit_log_path = audit.log_path.clone();
        options.audit_log_format = audit
            .log_format
            .as_deref()
            .and_then(|s| {
                (s == "json_lines").then_some(mesh_llm_events::audit::AuditLogFormat::JsonLines)
            })
            .unwrap_or(mesh_llm_events::audit::AuditLogFormat::JsonLines);
        options.audit_log_level = audit
            .log_level
            .as_deref()
            .and_then(|s| match s {
                "info" => Some(mesh_llm_events::audit::AuditLevel::Info),
                "warn" => Some(mesh_llm_events::audit::AuditLevel::Warn),
                "error" => Some(mesh_llm_events::audit::AuditLevel::Error),
                "critical" => Some(mesh_llm_events::audit::AuditLevel::Critical),
                _ => None,
            })
            .unwrap_or(mesh_llm_events::audit::AuditLevel::Info);
    }
}

fn initialize_early_topology_audit_logging(options: &mut RuntimeOptions) -> Result<()> {
    let config_path = options.config.clone();
    initialize_early_topology_audit_logging_with(
        options,
        || plugin::load_config(config_path.as_deref()),
        initialize_audit_logging_for_options,
    )
}

pub(super) fn initialize_early_topology_audit_logging_with(
    options: &mut RuntimeOptions,
    load_config: impl FnOnce() -> Result<plugin::MeshConfig>,
    initialize_audit_logging: impl FnOnce(&RuntimeOptions) -> Result<()>,
) -> Result<()> {
    match load_config() {
        Ok(config) => apply_runtime_config_options(options, &config),
        Err(error) => {
            tracing::warn!(
                error = %error,
                "failed to load config for early topology audit logging; continuing without config-derived audit settings"
            );
        }
    }
    initialize_audit_logging(options)
}

fn initialize_audit_logging_for_options(options: &RuntimeOptions) -> Result<()> {
    // An effective path is the enabling condition. A CLI path is an explicit
    // opt-in even when the persisted config disables its own sink.
    if options.audit_log_path.is_some() {
        init_audit_logging(
            options.audit_log_path.clone(),
            options.audit_log_format,
            options.audit_log_level,
            options.audit_max_file_size,
            options.audit_max_files,
        )?;
    }
    Ok(())
}

pub(in crate::runtime) fn apply_runtime_cli_speculative_overrides(
    config: &mut plugin::MeshConfig,
    overrides: Option<&plugin::SpeculativeConfig>,
) {
    let Some(overrides) = overrides else {
        return;
    };
    let defaults = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.speculative.as_ref())
        .cloned();
    let resolved_defaults =
        plugin::SpeculativeConfig::with_precedence(Some(overrides), None, defaults.as_ref());
    config
        .defaults
        .get_or_insert_with(plugin::ModelConfigDefaults::default)
        .speculative = Some(resolved_defaults);
    for model in &mut config.models {
        model.speculative = Some(plugin::SpeculativeConfig::with_precedence(
            Some(overrides),
            model.speculative.as_ref(),
            defaults.as_ref(),
        ));
    }
}

pub(in crate::runtime) fn apply_runtime_cli_checkpoint_overrides(
    config: &mut plugin::MeshConfig,
    quantization: Option<&str>,
    imatrix: Option<&Path>,
) -> Result<()> {
    if quantization.is_none() && imatrix.is_none() {
        return Ok(());
    }

    let quantization = quantization
        .map(|value| {
            value
                .parse::<skippy_runtime::CheckpointQuantization>()
                .map(|parsed| parsed.canonical_name().to_string())
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("invalid --quant {value:?}"))
        })
        .transpose()?;
    let imatrix = imatrix
        .map(|path| {
            let canonical = std::fs::canonicalize(path).with_context(|| {
                format!(
                    "resolve --checkpoint-imatrix path {} from the current directory",
                    path.display()
                )
            })?;
            anyhow::ensure!(
                canonical.is_file(),
                "--checkpoint-imatrix must name a file: {}",
                canonical.display()
            );
            Ok::<_, anyhow::Error>(canonical.to_string_lossy().into_owned())
        })
        .transpose()?;

    let defaults = config
        .defaults
        .get_or_insert_with(plugin::ModelConfigDefaults::default);
    let default_hardware = defaults
        .hardware
        .get_or_insert_with(plugin::HardwareConfig::default);
    if let Some(value) = quantization.as_ref() {
        default_hardware.checkpoint_quantization = Some(value.clone());
    }
    if let Some(value) = imatrix.as_ref() {
        default_hardware.checkpoint_imatrix = Some(value.clone());
    }

    // Runtime CLI flags are process-wide serving overrides. Apply them to
    // config-declared models as well as defaults so model-local config cannot
    // unexpectedly take precedence over the explicit command line.
    for model in &mut config.models {
        let hardware = model
            .hardware
            .get_or_insert_with(plugin::HardwareConfig::default);
        if let Some(value) = quantization.as_ref() {
            hardware.checkpoint_quantization = Some(value.clone());
        }
        if let Some(value) = imatrix.as_ref() {
            hardware.checkpoint_imatrix = Some(value.clone());
        }
    }

    Ok(())
}

pub fn load_resolved_plugins(options: &RuntimeOptions) -> Result<plugin::ResolvedPlugins> {
    let config = plugin::load_config(options.config.as_deref())?;
    resolve_plugins_from_config(&config, options)
}

pub(super) fn resolve_plugins_from_config(
    config: &plugin::MeshConfig,
    options: &RuntimeOptions,
) -> Result<plugin::ResolvedPlugins> {
    plugin::resolve_plugins(config, plugin_host_mode(options))
}

pub(super) fn plugin_host_mode(options: &RuntimeOptions) -> plugin::PluginHostMode {
    plugin::PluginHostMode {
        mesh_visibility: if options.publish || options.nostr_discovery {
            mesh_llm_plugin::MeshVisibility::Public
        } else {
            mesh_llm_plugin::MeshVisibility::Private
        },
    }
}

pub(super) fn node_display_name(options: &RuntimeOptions, node: &mesh::Node) -> String {
    options
        .name
        .clone()
        .or_else(|| std::env::var("USER").ok())
        .or_else(|| std::env::var("USERNAME").ok())
        .unwrap_or_else(|| node.id().fmt_short().to_string())
}

pub(super) async fn store_benchmark_metrics(
    mem_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    fp32_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    fp16_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    result: Option<&benchmark::BenchmarkResult>,
) {
    *mem_arc.lock().await = result.map(|r| r.mem_bandwidth_gbps.clone());
    *fp32_arc.lock().await = result.and_then(|r| r.compute_tflops_fp32.clone());
    *fp16_arc.lock().await = result.and_then(|r| r.compute_tflops_fp16.clone());
}

#[expect(
    clippy::cognitive_complexity,
    reason = "release attestation loading logs missing, valid, and invalid embedded states before advertising the result"
)]
pub(super) async fn attach_local_release_attestation(node: &mesh::Node) -> Result<()> {
    let loaded = match release_attestation::load_for_current_binary() {
        Ok(loaded) => loaded,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "failed to load local embedded release attestation; continuing without advertising one"
            );
            return Ok(());
        }
    };
    node.set_release_attestation_report(loaded.summary.clone(), loaded.attestation.clone())
        .await;
    match loaded.summary.status {
        crate::ReleaseAttestationStatus::Missing => {
            tracing::info!(
                path = %loaded.binary_path.display(),
                "no embedded release attestation found for local binary"
            );
            return Ok(());
        }
        crate::ReleaseAttestationStatus::Valid => {}
        crate::ReleaseAttestationStatus::Invalid => {
            tracing::warn!(
                path = %loaded.binary_path.display(),
                error = %loaded.summary.error.as_deref().unwrap_or("unknown release attestation error"),
                "local binary has an invalid embedded release attestation; continuing without advertising one"
            );
            return Ok(());
        }
    }
    let Some(attestation) = loaded.attestation else {
        tracing::warn!(
            path = %loaded.binary_path.display(),
            "embedded release attestation verified but no release attestation payload was produced"
        );
        return Ok(());
    };
    let attestation_hash = attestation.canonical_hash_hex().ok();
    if loaded.summary.verified {
        tracing::info!(
            path = %loaded.binary_path.display(),
            signer_key_id = %attestation.signer_key_id,
            attestation_hash = attestation_hash.as_deref().unwrap_or("unknown"),
            "loaded local embedded release attestation"
        );
    }
    node.set_release_attestation_report(loaded.summary, Some(attestation))
        .await;
    Ok(())
}

/// Install the model-intent channel on `node` and return the receiver the
/// runtime control loop drains.
async fn install_run_auto_model_intent_channel(
    node: mesh::Node,
) -> tokio::sync::mpsc::Receiver<super::ModelIntent> {
    let (model_intent_tx, model_intent_rx) = tokio::sync::mpsc::channel(64);
    let mut tx_guard = node.model_intent_tx.lock().await;
    *tx_guard = Some(model_intent_tx);
    model_intent_rx
}

pub(super) fn skippy_telemetry_options(options: &RuntimeOptions) -> skippy::SkippyTelemetryOptions {
    let endpoint = options
        .skippy_metrics_otlp_grpc
        .as_deref()
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
        .map(str::to_owned);
    match (endpoint, options.debug) {
        (Some(endpoint), true) => skippy::SkippyTelemetryOptions::debug(Some(endpoint)),
        (Some(endpoint), false) => skippy::SkippyTelemetryOptions::summary(endpoint),
        (None, _) => skippy::SkippyTelemetryOptions::off(),
    }
}

pub(super) fn configure_run_auto_process_state(
    options: &RuntimeOptions,
    runtime: Option<&std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
) {
    // SAFETY: UNSAFE CONTRACT — callers must invoke this before concurrent
    // runtime work can access the process environment. The current runtime
    // startup path does not enforce that boundary; retain the audit TODO.
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe {
        if options.local_model_only {
            // SAFETY: UNSAFE CONTRACT — callers must establish the startup ordering above.
            // TODO: Audit that the environment access only happens in single-threaded code.
            std::env::remove_var("MESH_API_PORT");
        } else {
            // SAFETY: UNSAFE CONTRACT — callers must establish the startup ordering above.
            // TODO: Audit that the environment access only happens in single-threaded code.
            std::env::set_var("MESH_API_PORT", options.console.to_string());
        }
    }

    let verbose_native_debug = options.debug
        && std::env::var("MESH_LLM_DEBUG_NATIVE_VERBOSE")
            .ok()
            .as_deref()
            == Some("1");
    if verbose_native_debug {
        skippy_runtime::enable_verbose_native_logs();
    } else {
        skippy_runtime::disable_verbose_native_logs();
    }

    let native_log_rx = skippy_runtime::register_filtered_native_logs();
    skippy_runtime::set_filtered_native_logs_enabled(true);
    bridge_skippy_native_logs(native_log_rx);
    skippy::configure_materialized_stage_cache();
    configure_skippy_native_logging(runtime.as_ref().map(|runtime| runtime.dir()));
}

pub(super) fn spawn_node_benchmark_task(node: &mesh::Node, bin_dir: &Path) {
    let mem_arc = node.gpu_mem_bandwidth_gbps.clone();
    let compute_fp32_arc = node.gpu_compute_tflops_fp32.clone();
    let compute_fp16_arc = node.gpu_compute_tflops_fp16.clone();
    let bin_dir_clone = bin_dir.to_path_buf();
    let node_bench = node.clone();
    tokio::spawn(async move {
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            tokio::task::spawn_blocking(move || {
                let hw = hardware::survey();
                if hw.gpu_count == 0 {
                    tracing::debug!("no GPUs detected — skipping memory bandwidth benchmark");
                    return None;
                }
                benchmark::run_or_load(&hw, &bin_dir_clone, benchmark::BENCHMARK_TIMEOUT)
            }),
        )
        .await
        .map_err(|_| {
            tracing::warn!("benchmark timed out after 30s — bandwidth will not be gossiped")
        })
        .ok()
        .and_then(|r| r.ok())
        .flatten();

        if let Some(ref run) = result {
            let total: f64 = run.mem_bandwidth_gbps.iter().sum();
            tracing::info!(
                "Memory bandwidth fingerprint: {} GPUs, {:.1} GB/s total",
                run.mem_bandwidth_gbps.len(),
                total
            );
            for (i, gbps) in run.mem_bandwidth_gbps.iter().enumerate() {
                tracing::debug!("  GPU {}: {:.1} GB/s", i, gbps);
            }
            if let Some(fp32s) = &run.compute_tflops_fp32 {
                let total_fp32: f64 = fp32s.iter().sum();
                tracing::info!(
                    "Compute FP32 TFLOPS: {} GPUs, {:.1} TFLOPS total",
                    fp32s.len(),
                    total_fp32
                );
                for (i, tf) in fp32s.iter().enumerate() {
                    tracing::debug!("  GPU {}: {:.1} TF32", i, tf);
                }
            }
            if let Some(fp16s) = &run.compute_tflops_fp16 {
                let total_fp16: f64 = fp16s.iter().sum();
                tracing::info!(
                    "Compute FP16 TFLOPS: {} GPUs, {:.1} TFLOPS total",
                    fp16s.len(),
                    total_fp16
                );
                for (i, tf) in fp16s.iter().enumerate() {
                    tracing::debug!("  GPU {}: {:.1} TF16", i, tf);
                }
            }
        }
        store_benchmark_metrics(
            mem_arc.clone(),
            compute_fp32_arc.clone(),
            compute_fp16_arc.clone(),
            result.as_ref(),
        )
        .await;
        node_bench.regossip().await;
    });
}

pub(super) async fn start_run_auto_node_and_plugins(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
    resolved_plugins: &plugin::ResolvedPlugins,
    swarm_capture: Option<crate::capture::SwarmCaptureRecorder>,
    startup_mesh_creation_state: &StartupMeshCreationState,
) -> Result<(mesh::Node, mesh::TunnelChannels, plugin::PluginManager)> {
    let role = if options.client {
        NodeRole::Client
    } else {
        NodeRole::Worker
    };
    let owner_config = owner_runtime_config(options, config)?;
    if !options.headless && owner_config.keypair.is_none() {
        emit_configuration_ui_read_only_hint();
    }
    let max_vram = if options.client {
        Some(0.0)
    } else {
        options.max_vram
    };
    let relay_auths: std::collections::HashMap<String, String> =
        options.relay_auth.iter().cloned().collect();
    let (node, channels) = mesh::Node::start(
        role,
        mesh::RelayConfig {
            urls: &options.relay,
            auths: &relay_auths,
            policy: relay_policy_for_runtime_options(options),
        },
        mesh::QuicBindSelection {
            ip: effective_quic_bind_ip(options),
            port: options.bind_port,
        },
        max_vram,
        !options.no_enumerate_host,
        Some(owner_config),
        options.config.as_deref(),
        startup_mesh_creation_state.requirements.clone(),
    )
    .await?;
    node.set_swarm_capture_recorder(swarm_capture);
    attach_local_release_attestation(&node).await?;
    node.set_stage_control_handle(skippy::spawn_stage_control_loop(
        Some(Arc::new(node.clone())),
        skippy_telemetry_options(options),
    ))
    .await;
    node.start_accepting();
    node.set_display_name(node_display_name(options, &node))
        .await;

    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(resolved_plugins, plugin_host_mode(options), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);
    Ok((node, channels, plugin_manager))
}

pub(super) fn register_pre_accept_local_source_policies(
    config: &plugin::MeshConfig,
    startup_specs: &[StartupModelSpec],
) {
    let default_skippy = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.skippy.as_ref());
    let default_model_path = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.hardware.as_ref())
        .and_then(|hardware| hardware.model_path.as_deref());
    for model in &config.models {
        let mut model_ids = BTreeSet::from([model.model.clone()]);
        let configured_path = model
            .hardware
            .as_ref()
            .and_then(|hardware| hardware.model_path.as_deref())
            .or(default_model_path);
        if let Some(path) = configured_path {
            let path = PathBuf::from(path);
            model_ids.insert(path.to_string_lossy().into_owned());
            if path.is_absolute() {
                let canonical = path.canonicalize().unwrap_or(path);
                model_ids.insert(models::model_ref_for_path(&canonical));
            }
        }
        let required = super::startup_models::skippy_local_source_required(
            model.skippy.as_ref(),
            default_skippy,
        );
        let runtime_profile = model
            .with_profile_defaults(config.defaults.as_ref())
            .derived_profile();
        for model_id in model_ids {
            skippy::register_local_source_policy(&model_id, &runtime_profile, required);
        }
    }
    for spec in startup_specs {
        let mut model_ids = BTreeSet::new();
        if let Some(declared_ref) = spec.declared_ref.as_deref() {
            model_ids.insert(declared_ref.to_string());
        }
        model_ids.insert(spec.model_ref.to_string_lossy().into_owned());
        if spec.model_ref.is_absolute() {
            let canonical = spec
                .model_ref
                .canonicalize()
                .unwrap_or_else(|_| spec.model_ref.clone());
            model_ids.insert(models::model_ref_for_path(&canonical));
        }
        for model_id in model_ids {
            skippy::register_local_source_policy(
                &model_id,
                &spec.profile,
                spec.local_source_required,
            );
        }
    }
}

pub(super) fn relay_policy_for_runtime_options(options: &RuntimeOptions) -> mesh::RelayPolicy {
    if options.disable_iroh_relays {
        mesh::RelayPolicy::ExplicitlyDisabled
    } else {
        relay_policy_for_mesh_discovery_mode(options.mesh_discovery_mode)
    }
}

pub(super) fn relay_policy_for_mesh_discovery_mode(
    mode: mesh_discovery::MeshDiscoveryMode,
) -> mesh::RelayPolicy {
    match mode {
        mesh_discovery::MeshDiscoveryMode::Nostr => mesh::RelayPolicy::DefaultPublic,
        mesh_discovery::MeshDiscoveryMode::Mdns => mesh::RelayPolicy::Disabled,
    }
}

pub(super) fn runtime_resource_planning_profile(
    options: &RuntimeOptions,
) -> RuntimeResourcePlanningProfile {
    if options.auto || options.publish || options.discover.is_some() || !options.join.is_empty() {
        RuntimeResourcePlanningProfile::SharedMesh
    } else {
        RuntimeResourcePlanningProfile::DedicatedLocal
    }
}

pub(super) fn runtime_model_ctx_size_override(
    options: &RuntimeOptions,
    model_overrides: Option<&plugin::ModelConfigEntry>,
) -> Option<u32> {
    options
        .ctx_size
        .or_else(|| model_overrides.and_then(|model| model.ctx_size))
}

pub(super) fn should_start_relay_health_monitor(mode: mesh_discovery::MeshDiscoveryMode) -> bool {
    matches!(
        relay_policy_for_mesh_discovery_mode(mode),
        mesh::RelayPolicy::DefaultPublic
    )
}

pub(super) fn should_start_lan_rediscovery(
    mode: mesh_discovery::MeshDiscoveryMode,
    join_tokens: &[String],
) -> bool {
    mode == mesh_discovery::MeshDiscoveryMode::Mdns
        && join_tokens.iter().any(|token| !token.trim().is_empty())
}

pub(super) fn start_relay_health_monitor_for_discovery_mode(
    node: &mesh::Node,
    mode: mesh_discovery::MeshDiscoveryMode,
) {
    if should_start_relay_health_monitor(mode) {
        node.start_relay_health_monitor();
    } else {
        tracing::debug!("Relay health monitor disabled for LAN-only mesh discovery");
    }
}

pub(super) fn run_auto_survey_hardware(is_client: bool) -> hardware::HardwareSurvey {
    if is_client {
        hardware::HardwareSurvey::default()
    } else {
        hardware::query(&[
            hardware::Metric::GpuName,
            hardware::Metric::GpuCount,
            hardware::Metric::IsSoc,
            hardware::Metric::GpuFacts,
        ])
    }
}

fn attach_logging_metrics_to_survey(survey_telemetry: &survey::SurveyTelemetry) {
    if let Some(logging) = crate::logging_runtime_state() {
        logging.set_metrics_sink(survey_telemetry.logging_sink());
    }
}

pub(super) async fn build_run_auto_node_setup(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
    resolved_plugins: &plugin::ResolvedPlugins,
    bin_dir: &Path,
    swarm_capture: Option<crate::capture::SwarmCaptureRecorder>,
    startup_mesh_creation_state: &StartupMeshCreationState,
) -> Result<AutoRuntimeNodeSetup> {
    let console_port = Some(options.console);
    let is_client = options.client;
    let skippy_telemetry = skippy_telemetry_options(options);
    let local_models = if is_client {
        vec![]
    } else {
        models::scan_local_models()
    };
    tracing::info!("Local models on disk: {:?}", local_models);
    let (node, channels, plugin_manager) = start_run_auto_node_and_plugins(
        options,
        config,
        resolved_plugins,
        swarm_capture,
        startup_mesh_creation_state,
    )
    .await?;
    let survey_hardware = run_auto_survey_hardware(is_client);
    let survey_telemetry = survey::SurveyTelemetry::start(
        config,
        survey_hardware,
        survey::SurveyTelemetrySource {
            node_id: node.id().fmt_short().to_string(),
            node_role: if is_client { "client" } else { "worker" }.into(),
        },
    );
    attach_logging_metrics_to_survey(&survey_telemetry);
    node.set_routing_telemetry_sink(survey_telemetry.routing_sink());
    node.set_available_models(local_models.clone()).await;
    let _activity_policy_task = super::activity_policy::spawn_native_activity_policy(
        node.activity_policy_guard.clone(),
        config.runtime.activity.clone(),
    );
    node.start_heartbeat();
    node.start_rtt_refresh();
    // iroh owns NAT traversal and relay-to-direct migration; modern nodes do
    // not start the legacy application-level reverse-dial sender.
    start_relay_health_monitor_for_discovery_mode(&node, options.mesh_discovery_mode);
    let lan_bootstrap_tasks = spawn_mdns_reverse_dial(options, &node);

    if !is_client {
        spawn_node_benchmark_task(&node, bin_dir);
    } else {
        tracing::debug!("client node — skipping memory bandwidth benchmark");
    }

    Ok(AutoRuntimeNodeSetup {
        is_client,
        console_port,
        skippy_telemetry,
        local_models,
        node,
        channels,
        plugin_manager,
        survey_telemetry,
        lan_bootstrap_tasks,
    })
}

pub(super) async fn run_auto_start_new_mesh(
    options: &RuntimeOptions,
    node: &mesh::Node,
) -> Result<()> {
    let nostr_pubkey = if options.publish
        && options.mesh_discovery_mode == mesh_discovery::MeshDiscoveryMode::Nostr
    {
        nostr::load_or_create_keys()
            .ok()
            .map(|k| k.public_key().to_hex())
    } else {
        None
    };
    let mesh_id = node
        .initialize_mesh_identity_as_originator(
            options.mesh_name.as_deref(),
            nostr_pubkey.as_deref(),
        )
        .await?;
    record_first_joined_mesh_ts(node).await;
    mesh::save_last_mesh_id(&mesh_id)?;
    tracing::info!("Mesh ID: {mesh_id}");
    let _ = emit_event(OutputEvent::InviteToken {
        token: node.invite_token().await,
        mesh_id: mesh_id.clone(),
        mesh_name: options.mesh_name.clone(),
    });
    let _ = emit_event(OutputEvent::WaitingForPeers { detail: None });

    if options.mesh_discovery_mode == mesh_discovery::MeshDiscoveryMode::Nostr
        && (options.auto || options.discover.is_some())
    {
        let rediscover_node = node.clone();
        let rediscover_relays = nostr_relays(&options.nostr_relay);
        let rediscover_relay_urls = options.relay.clone();
        let rediscover_mesh_name = options.mesh_name.clone();
        tokio::spawn(Box::pin(nostr_rediscovery(
            rediscover_node,
            rediscover_relays,
            rediscover_relay_urls,
            rediscover_mesh_name,
        )));
    }

    Ok(())
}

/// Returns true if `run_auto` should spawn the bootstrap proxy.
///
/// The bootstrap proxy binds the API port and tunnels OpenAI requests to
/// whichever mesh peer can serve them, so the local API stays usable while
/// this node's GPU loads its model.
///
/// Historically this gated solely on `options.join` being non-empty, which worked
/// because both `--client --auto` and `serve --auto` pushed their discovered
/// token into `options.join`. Commit 1bd62389 changed the serve path to stage
/// candidates in `auto_join_candidates` instead, leaving `options.join` empty and
/// silently disabling the bootstrap proxy for `serve --auto`. Accepting either
/// signal restores the original contract without changing any other path:
///
/// - `--join <token>` (any mode): `options.join` non-empty → fires (unchanged).
/// - `--client --auto` with discovery hit: `options.join` populated by
///   `handle_auto_decision` → fires (unchanged).
/// - `serve --auto` with discovery hit: `auto_join_candidates` non-empty,
///   `options.join` empty → **now fires** (the fix).
/// - Anything with no candidates and no join token (bare `mesh-llm`, bare
///   `--client`, `--auto` with zero discovery results): both empty → does
///   not fire (unchanged — there is nowhere to tunnel to).
pub(super) async fn advertise_run_auto_models(
    node: &mesh::Node,
    startup_models: &[StartupModelPlan],
    model_name: &str,
    model_source: String,
) {
    node.set_model_source(model_source).await;
    let all_declared = build_serving_list(startup_models, model_name);
    node.set_requested_models(all_declared.clone()).await;
    node.set_serving_models(all_declared.clone()).await;
    node.set_hosted_models(Vec::new()).await;
    node.set_models(all_declared).await;
    node.regossip().await;
}

fn initial_run_auto_requested_models(
    startup_specs: &[StartupModelSpec],
    requested_model_names: &[String],
) -> Vec<String> {
    startup_specs
        .iter()
        .zip(requested_model_names)
        .filter(|(spec, _)| {
            spec.declared_ref.is_some()
                || spec.config_model_id.is_some()
                || !super::startup_models::is_direct_local_gguf_source(&spec.model_ref)
        })
        .map(|(_, model_name)| model_name.clone())
        .collect()
}

pub(super) struct RunAutoRuntimeLoopContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) node: &'a mesh::Node,
    pub(super) primary_model_name: &'a str,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) runtime_models: &'a mut HashMap<String, RuntimeModelHandleEntry>,
    pub(super) runtime_survey_models: &'a mut HashMap<String, survey::SurveyLoadedModel>,
    pub(super) managed_models: &'a mut HashMap<String, ManagedModelController>,
    pub(super) runtime_capacity_ledger: &'a RuntimeCapacityLedger,
    pub(super) next_runtime_instance_sequence: &'a mut u64,
    pub(super) runtime_instance_registry: &'a RuntimeInstanceRegistry,
    pub(super) dashboard_processes: &'a Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) dashboard_context_usage: &'a DashboardContextUsage,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
    pub(super) runtime_event_tx: &'a tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) startup_ready_reporter: &'a StartupReadyReporter,
    pub(super) openai_guardrail_policy: &'a OpenAiGuardrailPolicyHandle,
    pub(super) model_target_reconciliation_policy: ModelTargetReconciliationPolicy,
    pub(super) model_target_reconciliation_state: ModelTargetReconciliationState,
}

pub(super) struct RunAutoRuntimeState {
    pub(super) runtime_models: HashMap<String, RuntimeModelHandleEntry>,
    pub(super) runtime_survey_models: HashMap<String, survey::SurveyLoadedModel>,
    pub(super) managed_models: HashMap<String, ManagedModelController>,
    pub(super) runtime_instance_registry: RuntimeInstanceRegistry,
    pub(super) runtime_capacity_ledger: RuntimeCapacityLedger,
    pub(super) next_runtime_instance_sequence: u64,
    pub(super) dashboard_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) dashboard_context_usage: DashboardContextUsage,
    pub(super) input_handler_enabled: bool,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    /// The installed process-local logging service owned by this runtime
    /// invocation. It is started only after the Phase 1 state has completed
    /// its confined store/artifact recovery, then drained before runtime exit.
    pub(super) logging_service: Option<Arc<crate::logging::LoggingService>>,
}

pub(super) struct RunAutoStartupTasksContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) node: &'a mesh::Node,
    pub(super) tunnel_mgr: &'a tunnel::Manager,
    pub(super) startup_models: &'a [StartupModelPlan],
    pub(super) primary_startup_model: Option<&'a StartupModelPlan>,
    pub(super) model_name: &'a str,
    pub(super) model_path: &'a Path,
    pub(super) startup_ready_reporter: &'a StartupReadyReporter,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) managed_models: &'a mut HashMap<String, ManagedModelController>,
    pub(super) next_runtime_instance_sequence: &'a mut u64,
    pub(super) runtime_capacity_ledger: &'a RuntimeCapacityLedger,
    pub(super) runtime_instance_registry: &'a RuntimeInstanceRegistry,
    pub(super) dashboard_processes: &'a Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) dashboard_context_usage: &'a DashboardContextUsage,
    pub(super) input_handler_enabled: bool,
    pub(super) openai_guardrail_policy: &'a OpenAiGuardrailPolicyHandle,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) runtime_event_tx: &'a tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) skippy_telemetry: &'a skippy::SkippyTelemetryOptions,
    pub(super) api_port: u16,
    pub(super) interactive_started: Arc<AtomicBool>,
}

pub(super) fn initialize_run_auto_runtime_state(options: &RuntimeOptions) -> RunAutoRuntimeState {
    RunAutoRuntimeState {
        runtime_models: HashMap::new(),
        runtime_survey_models: HashMap::new(),
        managed_models: HashMap::new(),
        runtime_instance_registry: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        runtime_capacity_ledger: RuntimeCapacityLedger::default(),
        next_runtime_instance_sequence: 1_u64,
        dashboard_processes: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        dashboard_context_usage: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        input_handler_enabled: output_sink()
            .and_then(|sink| sink.console_session_mode())
            .is_some(),
        openai_guardrail_policy: openai_guardrail_policy_handle(mesh_guardrail_mode_to_openai(
            options.mesh_guardrails,
        )),
        logging_service: None,
    }
}

/// Start the one persistence worker owned by the current runtime invocation.
///
/// The logging state is optional because disabled or fail-open initialization
/// must never prevent serving. The state itself owns the corrected shared
/// `LogStore`/artifact-capture resources, including startup recovery, so this
/// function intentionally does not reopen either resource.
pub(super) async fn start_run_auto_logging_service() -> Option<Arc<crate::logging::LoggingService>>
{
    crate::logging_runtime_state()?
        .start_persistence_worker()
        .await
}

pub(super) async fn spawn_run_auto_startup_model_tasks(ctx: RunAutoStartupTasksContext<'_>) {
    let RunAutoStartupTasksContext {
        options,
        config,
        node,
        tunnel_mgr,
        startup_models,
        primary_startup_model,
        model_name,
        model_path,
        startup_ready_reporter,
        target_tx,
        managed_models,
        next_runtime_instance_sequence,
        runtime_capacity_ledger,
        runtime_instance_registry,
        dashboard_processes,
        dashboard_context_usage,
        input_handler_enabled,
        openai_guardrail_policy,
        console_state,
        control_tx,
        runtime_event_tx,
        survey_telemetry,
        skippy_telemetry,
        api_port,
        interactive_started,
    } = ctx;

    let startup_load_gate = Arc::new(tokio::sync::Mutex::new(()));
    let primary_parallel_override = super::startup_models::resolve_model_parallel_override(
        primary_startup_model.and_then(|m| m.parallel),
        &config.gpu,
    );
    let resource_planning_profile = runtime_resource_planning_profile(options);
    let console_state_for_election = console_state.cloned();
    let interactive_console_state = console_state.cloned();
    let primary_mmproj = primary_startup_model.and_then(|model| model.mmproj_path.clone());
    let primary_ctx_size = primary_startup_model.and_then(|model| model.ctx_size);
    let primary_pinned_gpu = primary_startup_model.and_then(|model| model.pinned_gpu.clone());
    let primary_device_override =
        primary_startup_model.and_then(|model| startup_device_override(model.gpu_id.as_deref()));
    let primary_cache_type_k = primary_startup_model.and_then(|model| model.cache_type_k.clone());
    let primary_cache_type_v = primary_startup_model.and_then(|model| model.cache_type_v.clone());
    let primary_n_batch = primary_startup_model.and_then(|model| model.n_batch);
    let primary_n_ubatch = primary_startup_model.and_then(|model| model.n_ubatch);
    let primary_flash_attention = primary_startup_model
        .map(|model| model.flash_attention)
        .unwrap_or(FlashAttentionType::Auto);
    let primary_model_ref = primary_startup_model
        .map(|model| model.declared_ref.clone())
        .unwrap_or_else(|| model_name.to_string());
    let primary_config_model_id =
        primary_startup_model.and_then(|model| model.config_model_id.clone());
    let (primary_stop_tx, primary_stop_rx) = tokio::sync::watch::channel(false);
    let primary_instance_id = next_runtime_instance_id(next_runtime_instance_sequence);
    let primary_lifecycle = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
        InstanceLifecycleState::Planned,
        32,
    )));
    let primary_lifecycle_port = Arc::new(AtomicU16::new(0));
    let primary_task = tokio::spawn(Box::pin(startup_local_model_loop(StartupLocalModelTask {
        node: node.clone(),
        config: config.clone(),
        tunnel_mgr: tunnel_mgr.clone(),
        target_tx: target_tx.clone(),
        model_path: model_path.to_path_buf(),
        model_ref: primary_model_ref,
        preindexed_split_package: primary_startup_model
            .and_then(|model| model.preindexed_split_package.clone()),
        config_model_id: primary_config_model_id,
        readiness_index: 0,
        profile: primary_startup_model
            .map(|model| model.profile.clone())
            .unwrap_or_default(),
        model_name: model_name.to_string(),
        instance_id: primary_instance_id.clone(),
        primary_model_name: model_name.to_string(),
        mmproj_path: primary_mmproj,
        ctx_size: primary_ctx_size,
        pinned_gpu: primary_pinned_gpu,
        device_override: primary_device_override,
        runtime_capacity_ledger: runtime_capacity_ledger.clone(),
        cache_type_k: primary_cache_type_k,
        cache_type_v: primary_cache_type_v,
        n_batch: primary_n_batch,
        n_ubatch: primary_n_ubatch,
        flash_attention: primary_flash_attention,
        parallel_override: primary_parallel_override,
        local_source_required: primary_startup_model
            .is_some_and(|model| model.local_source_required),
        split_topology_lock: options.split_topology_lock.clone(),
        resource_planning_profile,
        openai_guardrail_policy: openai_guardrail_policy.clone(),
        split: options.split,
        skippy_telemetry: skippy_telemetry.clone(),
        survey_telemetry: survey_telemetry.clone(),
        survey_launch_kind: survey::SurveyLaunchKind::Startup,
        stop_rx: primary_stop_rx,
        dashboard_processes: dashboard_processes.clone(),
        dashboard_context_usage: dashboard_context_usage.clone(),
        runtime_instance_registry: runtime_instance_registry.clone(),
        console_state: console_state_for_election,
        api_port,
        startup_ready_reporter: startup_ready_reporter.clone(),
        runtime_event_tx: runtime_event_tx.clone(),
        startup_load_gate: startup_load_gate.clone(),
        input_handler_enabled,
        interactive_started,
        interactive_control_tx: control_tx.clone(),
        interactive_console_state,
        lifecycle: primary_lifecycle.clone(),
        lifecycle_port: primary_lifecycle_port.clone(),
    })));
    managed_models.insert(
        primary_instance_id,
        ManagedModelController {
            model_name: model_name.to_string(),
            profile: primary_startup_model
                .map(|model| model.profile.clone())
                .unwrap_or_default(),
            stop_tx: primary_stop_tx,
            task: primary_task,
            lifecycle: primary_lifecycle,
            port: primary_lifecycle_port,
        },
    );

    spawn_run_auto_additional_model_tasks(RunAutoAdditionalModelsContext {
        options,
        config,
        node,
        tunnel_mgr,
        startup_models,
        primary_model_name: model_name,
        target_tx,
        managed_models,
        next_runtime_instance_sequence,
        dashboard_processes,
        dashboard_context_usage,
        runtime_instance_registry,
        runtime_capacity_ledger,
        console_state,
        startup_ready_reporter,
        startup_load_gate: &startup_load_gate,
        control_tx,
        runtime_event_tx,
        survey_telemetry,
        skippy_telemetry,
        openai_guardrail_policy,
    })
    .await;
}

pub(super) fn configure_swarm_capture(
    options: &RuntimeOptions,
) -> Result<Option<crate::capture::SwarmCaptureRecorder>> {
    let recorder =
        crate::capture::SwarmCaptureRecorder::from_cli_or_env(options.swarm_capture.as_deref())?;
    if let Some(recorder) = recorder.as_ref() {
        tracing::info!(
            path = %recorder.path().display(),
            "passive swarm capture enabled; writing local debug capture JSONL"
        );
    }
    Ok(recorder)
}

#[expect(
    dead_code,
    reason = "the legacy advertised-model selection context is retained for compatibility helpers and focused tests"
)]
pub(super) struct RunAutoModelSelectionContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) node: &'a mesh::Node,
    pub(super) startup_models: &'a [StartupModelPlan],
    pub(super) local_models: &'a [String],
    pub(super) is_client: bool,
    pub(super) plugin_manager: &'a plugin::PluginManager,
    pub(super) bootstrap_listener_tx: &'a mut Option<BootstrapProxyStopTx>,
    pub(super) primary_startup_model: Option<&'a StartupModelPlan>,
    pub(super) embedded_control_rx:
        &'a mut Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
}

#[expect(
    dead_code,
    reason = "the daemon startup path supersedes advertised-model selection while compatibility tests still exercise it"
)]
pub(super) async fn select_advertised_run_auto_model(
    mut ctx: RunAutoModelSelectionContext<'_>,
) -> Result<Option<(PathBuf, String)>> {
    let Some(model) = run_auto_model_path_or_shutdown(&mut ctx).await? else {
        return Ok(None);
    };

    let (model_name, model_source) = run_auto_model_identity(ctx.primary_startup_model, &model);
    advertise_run_auto_models(ctx.node, ctx.startup_models, &model_name, model_source).await;
    Ok(Some((model, model_name)))
}

/// Serve mode: join the mesh and serve local models through the embedded runtime.
pub(super) struct RunAutoContext {
    pub(super) options: RuntimeOptions,
    pub(super) config: plugin::MeshConfig,
    pub(super) startup_mesh_creation_state: StartupMeshCreationState,
    pub(super) startup_specs: Vec<StartupModelSpec>,
    pub(super) requested_model_names: Vec<String>,
    pub(super) bin_dir: PathBuf,
    pub(super) runtime: Option<std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
    pub(super) auto_join_candidates: Vec<(String, Option<String>)>,
    pub(super) embedded_control_rx:
        Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
}

#[expect(
    clippy::cognitive_complexity,
    reason = "run_auto is the top-level runtime orchestration path and preserves startup/shutdown ordering"
)]
pub(super) async fn run_auto(ctx: RunAutoContext) -> Result<()> {
    let RunAutoContext {
        mut options,
        config,
        startup_mesh_creation_state,
        startup_specs,
        requested_model_names,
        bin_dir,
        runtime,
        auto_join_candidates,
        mut embedded_control_rx,
    } = ctx;
    // Stage-control starts accepting before eager model resolution. Register
    // every spelling that can become the model's runtime identity now so a
    // legacy/profile-unaware request cannot bypass local-required policy in
    // that window. False entries deliberately clear stale in-process policy.
    register_pre_accept_local_source_policies(&config, &startup_specs);
    let resolved_plugins = resolve_plugins_from_config(&config, &options)?;
    let swarm_capture = configure_swarm_capture(&options)?;
    tracing::debug!(
        mesh_requirements = ?runtime_startup_requirements(&startup_mesh_creation_state),
        "loaded creation-time mesh requirements into runtime startup state"
    );
    let api_port = options.port;
    configure_run_auto_process_state(&options, runtime.as_ref());
    let _native_log_forwarding = SkippyNativeLogForwardingGuard;
    // Embedded native logs are process-global and are redirected to the runtime log
    // file before model load. We also forward the filtered, aggregated model-loading
    // summaries through OutputEvent/JSONL so structured startup progress remains visible
    // without streaming every raw native line through the dashboard.
    let AutoRuntimeNodeSetup {
        is_client,
        console_port,
        skippy_telemetry,
        local_models,
        node,
        channels,
        plugin_manager,
        survey_telemetry,
        lan_bootstrap_tasks,
    } = build_run_auto_node_setup(
        &options,
        &config,
        &resolved_plugins,
        &bin_dir,
        swarm_capture,
        &startup_mesh_creation_state,
    )
    .await?;

    // A bare local GGUF has no stable mesh identity until its content scan
    // completes. Do not advertise its node-local path as temporary demand.
    let initial_requested_models =
        initial_run_auto_requested_models(&startup_specs, &requested_model_names);
    node.set_requested_models(initial_requested_models).await;

    run_auto_join_mesh_phase(&mut options, &node, &auto_join_candidates).await?;

    let affinity_router = affinity::AffinityRouter::new();

    // Start bootstrap proxy if we have somewhere to tunnel to. This gives
    // instant API access via tunnel while our GPU loads.
    let bootstrap_listener_tx = start_run_auto_bootstrap_proxy(
        &options,
        &node,
        api_port,
        &affinity_router,
        &auto_join_candidates,
    );

    // All daemon surfaces start from a valid zero-model state. Eager startup
    // model resolution is intentionally deferred until after those surfaces bind.
    node.set_models(Vec::new()).await;
    node.set_serving_models(Vec::new()).await;
    node.set_hosted_models(Vec::new()).await;
    node.regossip().await;

    // Election publishes per-model targets. The same receiver and affinity
    // router serve local TCP and remote QUIC OpenAI ingress.
    let (target_tx, target_rx) = tokio::sync::watch::channel(election::ModelTargets::default());
    let target_tx = std::sync::Arc::new(target_tx);

    let tunnel_mgr =
        tunnel::Manager::start(node.clone(), channels.rpc, channels.http, channels.stage).await?;
    // Serving hosts terminate inbound HTTP tunnel streams directly in the
    // shared OpenAI ingress. Client-only nodes remain unreachable as hosts.
    if !is_client {
        tunnel_mgr.set_http_port(api_port);
        tunnel_mgr.set_http_ingress(target_rx.clone(), affinity_router.clone());
        plugin_host_role::spawn(node.clone(), plugin_manager.clone(), api_port);
    }

    // Runtime control for local load/unload of extra models.
    let (control_tx, mut control_rx) =
        tokio::sync::mpsc::unbounded_channel::<api::RuntimeControlRequest>();
    spawn_embedded_runtime_control_forwarder(embedded_control_rx.take(), control_tx.clone());
    let (runtime_event_tx, mut runtime_event_rx) =
        tokio::sync::mpsc::unbounded_channel::<RuntimeEvent>();
    let mut runtime_state = initialize_run_auto_runtime_state(&options);
    runtime_state.logging_service = start_run_auto_logging_service().await;
    record_runtime_operational_event(RuntimeOperationalEvent::StartupStarted);

    // Model intent channel: owner-control commands send intents here, control loop polls.
    let mut model_intent_rx = install_run_auto_model_intent_channel(node.clone()).await;

    let model_name_for_console = String::new();
    let model_path_for_console = PathBuf::new();
    let runtime_owner_key_path = resolve_runtime_owner_key_path(&options)?;
    let console_state = setup_run_auto_console_state(RunAutoConsoleStateContext {
        options: &options,
        node: &node,
        console_enabled: console_port.is_some(),
        model_name: &model_name_for_console,
        model_path: &model_path_for_console,
        api_port,
        plugin_manager: &plugin_manager,
        affinity_router: &affinity_router,
        control_tx: &control_tx,
        owner_key_path: &runtime_owner_key_path,
    })
    .await?;
    publish_initial_openai_guardrails_status(
        console_state.as_ref(),
        &runtime_state.openai_guardrail_policy,
    )
    .await;

    if let Some(sink) = output_sink() {
        sink.register_dashboard_snapshot_provider(Arc::new(RuntimeDashboardSnapshotProvider::new(
            node.clone(),
            runtime_state.dashboard_processes.clone(),
            runtime_state.dashboard_context_usage.clone(),
            Some(plugin_manager.clone()),
            api_port,
            console_port,
            options.headless,
        )));
    }

    let interactive_started = Arc::new(AtomicBool::new(false));
    let RunAutoServingSurface {
        api_proxy_handle,
        console_server_handle,
        api_ready_url,
        ready_console_url,
        ready_api_port,
        ready_console_port,
    } = setup_run_auto_serving_surface(RunAutoServingSurfaceContext {
        options: &options,
        node: &node,
        api_port,
        console_port,
        is_client,
        target_rx: &target_rx,
        control_tx: &control_tx,
        affinity_router: &affinity_router,
        bootstrap_listener_tx,
        input_handler_enabled: runtime_state.input_handler_enabled,
        interactive_started: &interactive_started,
        console_state: console_state.as_ref(),
        model_name_for_console: &model_name_for_console,
    })
    .await?;

    let primary_model_name = requested_model_names.first().cloned().unwrap_or_default();
    let startup_ready_reporter = StartupReadyReporter::new_with_failure_policy(
        &requested_model_names,
        primary_model_name.clone(),
        api_ready_url,
        ready_console_url,
        ready_api_port,
        ready_console_port,
        config.runtime.startup_failure_policy,
    );
    if startup_specs.is_empty() {
        let _ = emit_event(OutputEvent::PassiveMode {
            role: if is_client { "client" } else { "standby" }.to_string(),
            status: RuntimeStatus::Ready,
            capacity_gb: (!is_client).then(|| node.vram_bytes() as f64 / 1e9),
            models_on_disk: (!is_client).then_some(local_models),
            detail: Some(if is_client {
                "Client daemon ready; local model loading is disabled".to_string()
            } else {
                "Runtime daemon ready; no local models are loaded".to_string()
            }),
        });
        record_runtime_operational_event(RuntimeOperationalEvent::Ready);
    }

    // Discovery publish loop (if --publish) or Nostr watchdog (if --auto, to take over if publisher dies).
    let discovery_publisher =
        spawn_run_auto_discovery_publisher(&options, &node, console_state.as_ref()).await;

    let runtime_data_producer = runtime_data_producer_for_console(console_state.as_ref()).await;
    run_auto_runtime_loop_and_shutdown(RunAutoRuntimeLifecycleContext {
        options: &options,
        config: &config,
        node: &node,
        primary_model_name: &primary_model_name,
        target_tx: &target_tx,
        control_rx: &mut control_rx,
        control_tx: &control_tx,
        runtime_event_rx: &mut runtime_event_rx,
        model_intent_rx: &mut model_intent_rx,
        runtime_state: &mut runtime_state,
        console_state: console_state.as_ref(),
        runtime_data_producer: runtime_data_producer.as_ref(),
        runtime_event_tx: &runtime_event_tx,
        survey_telemetry: &survey_telemetry,
        startup_ready_reporter: &startup_ready_reporter,
        plugin_manager: &plugin_manager,
        api_proxy_handle,
        console_server_handle,
        discovery_publisher,
        startup_specs: &startup_specs,
        tunnel_mgr: &tunnel_mgr,
        skippy_telemetry: &skippy_telemetry,
        api_port,
        console_port,
        interactive_started,
        lan_bootstrap_tasks,
        runtime,
    })
    .await;
    if let Some(summary) = startup_ready_reporter.fail_fast_summary() {
        anyhow::bail!("{summary}");
    }
    Ok(())
}
