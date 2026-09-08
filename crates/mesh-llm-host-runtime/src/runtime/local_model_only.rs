use super::{
    LocalOpenAiModelStartSpec, RuntimeOptions, RuntimeResourcePlanningProfile,
    SkippyNativeLogForwardingGuard, acquire_instance_runtime,
    apply_runtime_cli_checkpoint_overrides, apply_runtime_cli_speculative_overrides,
    apply_runtime_config_options, build_startup_model_specs, cleanup_run_auto_runtime_dir,
    configure_run_auto_process_state, emit_shutdown, openai_guardrail_policy_handle,
    preflight_pinned_startup_models, resolve_local_model_only_startup_models,
    runtime_model_required_bytes, skippy_telemetry_options, start_local_openai_model,
    startup_device_override, wait_shutdown_signal,
};
use crate::inference::election;
use crate::plugin;
use crate::runtime::survey;
use crate::system::hardware;
use anyhow::{Context, Result};
use mesh_llm_events::{OutputEvent, emit_event};
use skippy_server::EmbeddedState;
use skippy_server::serving_hooks::SharedModelServingHooksFactory;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

const OPENAI_STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
const OPENAI_STATUS_POLL_INTERVAL: Duration = Duration::from_millis(25);

pub(super) fn validate_local_model_only_options(options: &RuntimeOptions) -> Result<()> {
    anyhow::ensure!(!options.client, "--local-model-only cannot run as a client");
    anyhow::ensure!(
        !options.auto && options.discover.is_none() && options.join.is_empty(),
        "--local-model-only cannot discover or join a mesh"
    );
    anyhow::ensure!(
        !options.publish
            && options.mesh_name.is_none()
            && options.region.is_none()
            && options.name.is_none(),
        "--local-model-only cannot publish or describe a mesh"
    );
    anyhow::ensure!(
        !options.split && options.split_topology_lock.is_none() && options.tensor_split.is_none(),
        "--local-model-only does not support split serving"
    );
    anyhow::ensure!(
        options.relay.is_empty()
            && options.relay_auth.is_empty()
            && options.nostr_relay.is_empty()
            && !options.disable_iroh_relays
            && options.bind_ip.is_none()
            && options.bind_port.is_none()
            && options.max_clients.is_none(),
        "--local-model-only does not accept mesh transport options"
    );
    anyhow::ensure!(
        options.min_node_version.is_none()
            && options.max_node_version.is_none()
            && options.min_protocol_version.is_none()
            && options.max_protocol_version.is_none()
            && !options.require_release_attestation
            && options.release_signer_key.is_empty(),
        "--local-model-only does not accept mesh admission options"
    );
    anyhow::ensure!(
        options.owner_key.is_none()
            && options.control_bind.is_none()
            && options.control_advertise_addr.is_none()
            && !options.owner_required
            && options.node_label.is_none()
            && options.trust_policy.is_none()
            && options.trust_owner.is_empty(),
        "--local-model-only does not start owner control or management APIs"
    );
    anyhow::ensure!(
        options.plugin.is_none() && options.swarm_capture.is_none(),
        "--local-model-only does not start plugins or swarm capture"
    );
    anyhow::ensure!(
        !options.auto_update,
        "--local-model-only does not perform release updates"
    );
    anyhow::ensure!(
        !options.headless,
        "--local-model-only never starts a console; remove --headless"
    );
    if let Some(max_vram) = options.max_vram {
        anyhow::ensure!(
            max_vram.is_finite() && max_vram > 0.0,
            "--max-vram must be a finite positive number"
        );
    }
    match options.native_serving_plugin.as_ref() {
        Some(_) => {
            anyhow::ensure!(
                options.native_serving_plugin_config.is_some()
                    && options.native_serving_plugin_state.is_some()
                    && options.native_serving_plugin_deadline_ms.is_some(),
                "--native-serving-plugin requires config, state, and deadline options"
            );
        }
        None => {
            anyhow::ensure!(
                options.native_serving_plugin_config.is_none()
                    && options.native_serving_plugin_state.is_none()
                    && options.native_serving_plugin_deadline_ms.is_none(),
                "native serving plugin config, state, and deadline require --native-serving-plugin"
            );
        }
    }
    Ok(())
}

pub(super) async fn run_local_model_only(options: RuntimeOptions) -> Result<()> {
    validate_local_model_only_options(&options)?;
    // Local-model-only serving starts the runtime-event engine with zero
    // management subscribers attached: nothing here calls `.subscribers()`,
    // matching the plan's "zero management subscribers" requirement for
    // this mode without needing a separate no-op engine variant.
    let runtime_event_engine = crate::runtime_events::engine::RuntimeEventEngine::new();
    crate::runtime_events::install_runtime_event_engine(runtime_event_engine.clone());
    // Task 3: same engine-owned driver as the mesh-serve path in
    // `run_auto.rs` (defect D3) -- this mode's own "zero management
    // subscribers" invariant only ever meant no PRESENTATION subscriber;
    // the driver needs no subscriber at all to apply and publish a fact
    // (see `runtime_events::driver`'s module doc and this module's own
    // `tests::engine_driver`).
    let mut runtime_event_driver = Some(crate::runtime_events::driver::spawn_engine_driver(
        runtime_event_engine.clone(),
    ));

    let result =
        run_local_model_only_inner(options, &runtime_event_engine, &mut runtime_event_driver).await;
    cleanup_local_model_only_runtime_event_state(&runtime_event_engine, &mut runtime_event_driver)
        .await;
    result
}

async fn cleanup_local_model_only_runtime_event_state(
    runtime_event_engine: &Arc<crate::runtime_events::engine::RuntimeEventEngine>,
    runtime_event_driver: &mut Option<crate::runtime_events::driver::EngineDriverHandle>,
) {
    if let Some(driver) = runtime_event_driver.take() {
        driver.stop_and_wait().await;
    }
    crate::runtime_events::clear_runtime_event_engine_if_owned(runtime_event_engine);
}

async fn run_local_model_only_inner(
    mut options: RuntimeOptions,
    runtime_event_engine: &Arc<crate::runtime_events::engine::RuntimeEventEngine>,
    runtime_event_driver: &mut Option<crate::runtime_events::driver::EngineDriverHandle>,
) -> Result<()> {
    // Task 19: same hidden, TEST-ONLY `event-disabled` A/B certification
    // selector as the mesh-serve path in `run_auto.rs` -- see its comment
    // for the gate/selector relationship; a no-op on every normal startup.
    runtime_event_engine.set_progress_diagnostic_class_bypass(
        mesh_llm_config::event_system_progress_diagnostic_bypass_enabled()?,
    );
    super::node_lifecycle_events::emit_node_starting();
    let serving_hooks_factory = native_serving_plugin_factory(&options)?;
    let mut config = plugin::load_config(options.config.as_deref())?;
    apply_runtime_cli_speculative_overrides(&mut config, options.speculative_overrides.as_ref());
    apply_runtime_cli_checkpoint_overrides(
        &mut config,
        options.checkpoint_quantization.as_deref(),
        options.checkpoint_imatrix.as_deref(),
    )?;
    apply_runtime_config_options(&mut options, &config);
    // Task 16: same OTLP-specific runtime-event telemetry consumer as the
    // mesh-serve path in `run_auto.rs`; a disabled or failed exporter
    // degrades to a no-op instance and never affects startup. Installs its
    // sample queue onto `runtime_event_engine` so the single local model's
    // real submissions feed the ingress-latency and class-outcome
    // instruments too.
    let _runtime_event_telemetry =
        survey::runtime_events::RuntimeEventTelemetry::start(&config, runtime_event_engine);

    let startup_specs = build_startup_model_specs(&options, &config)?;
    anyhow::ensure!(
        startup_specs.len() == 1,
        "--local-model-only requires exactly one startup model"
    );
    let mut startup_models = resolve_local_model_only_startup_models(&startup_specs).await?;
    preflight_pinned_startup_models(
        &config,
        &startup_specs,
        &mut startup_models,
        options.llama_flavor,
        None,
    )?;
    let model = startup_models
        .pop()
        .context("local model resolution produced no startup model")?;
    anyhow::ensure!(
        model.resolved_path.is_file(),
        "--local-model-only requires one complete local model file: {}",
        model.resolved_path.display()
    );

    let model_bytes = election::total_model_bytes(&model.resolved_path);
    anyhow::ensure!(
        model_bytes > 0,
        "could not determine local model size: {}",
        model.resolved_path.display()
    );
    let local_capacity_bytes = local_capacity_bytes(&options, model.pinned_gpu.as_ref());
    let required_bytes = runtime_model_required_bytes(model_bytes);
    anyhow::ensure!(
        local_capacity_bytes >= required_bytes,
        "local model requires {:.2} GB but this process has {:.2} GB; local model-only serving never falls back to a split",
        required_bytes as f64 / 1e9,
        local_capacity_bytes as f64 / 1e9
    );

    let bind_addr = local_openai_bind_addr(&options);
    let runtime = acquire_instance_runtime(&options);
    configure_run_auto_process_state(&options, runtime.as_ref(), &config);
    let _native_log_forwarding = SkippyNativeLogForwardingGuard;

    let model_name = model.declared_ref.clone();
    let survey_telemetry = survey::SurveyTelemetry::start(
        &config,
        hardware::survey(),
        survey::SurveyTelemetrySource {
            node_id: "local-model-only".into(),
            node_role: "worker".into(),
        },
    );
    let launch = LocalOpenAiModelStartSpec {
        mesh_config: &config,
        config_model_id: model.config_model_id.as_deref(),
        model_path: &model.resolved_path,
        model_bytes,
        mmproj_override: model.mmproj_path.as_deref(),
        ctx_size_override: model.ctx_size,
        pinned_gpu: model.pinned_gpu.as_ref(),
        device_override: startup_device_override(model.gpu_id.as_deref()),
        capacity_budget_bytes: local_capacity_bytes,
        cache_type_k_override: model.cache_type_k.as_deref(),
        cache_type_v_override: model.cache_type_v.as_deref(),
        n_batch_override: model.n_batch,
        n_ubatch_override: model.n_ubatch,
        flash_attention_override: model.flash_attention,
        parallel_override: model.parallel,
        planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
        openai_guardrail_policy: openai_guardrail_policy_handle(
            super::status::mesh_guardrail_mode_to_openai(options.mesh_guardrails),
        ),
        skippy_telemetry: skippy_telemetry_options(&options),
        survey_telemetry,
        hook_policy: None,
        serving_hooks_factory,
        http_bind_addr: bind_addr,
    };

    let result = run_loaded_local_model(launch, &model_name, bind_addr, runtime_event_driver).await;
    cleanup_run_auto_runtime_dir(runtime);
    result
}

fn native_serving_plugin_factory(
    options: &RuntimeOptions,
) -> Result<Option<SharedModelServingHooksFactory>> {
    let Some(library_path) = options.native_serving_plugin.as_deref() else {
        return Ok(None);
    };
    let config_path = options
        .native_serving_plugin_config
        .clone()
        .context("--native-serving-plugin-config is required")?;
    let state_directory = options
        .native_serving_plugin_state
        .clone()
        .context("--native-serving-plugin-state is required")?;
    let deadline_ms = options
        .native_serving_plugin_deadline_ms
        .context("--native-serving-plugin-deadline-ms is required")?;
    anyhow::ensure!(
        deadline_ms > 0,
        "--native-serving-plugin-deadline-ms must be greater than zero"
    );
    let factory = mesh_native_serving_plugin_host::NativeServingPluginFactory::load(
        library_path,
        config_path,
        state_directory,
        Duration::from_millis(deadline_ms),
    )?;
    Ok(Some(std::sync::Arc::new(factory)))
}

fn local_capacity_bytes(
    options: &RuntimeOptions,
    pinned_gpu: Option<&super::StartupPinnedGpuTarget>,
) -> u64 {
    let detected = pinned_gpu
        .map(super::StartupPinnedGpuTarget::allocatable_vram_bytes)
        .unwrap_or_else(|| hardware::survey().vram_bytes);
    options
        .max_vram
        .map(|gb| (gb * 1e9) as u64)
        .map_or(detected, |cap| detected.min(cap))
}

fn local_openai_bind_addr(options: &RuntimeOptions) -> SocketAddr {
    let ip = if options.listen_all {
        IpAddr::V4(Ipv4Addr::UNSPECIFIED)
    } else {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    };
    SocketAddr::new(ip, options.port)
}

async fn run_loaded_local_model(
    launch: LocalOpenAiModelStartSpec<'_>,
    model_name: &str,
    bind_addr: SocketAddr,
    runtime_event_driver: &mut Option<crate::runtime_events::driver::EngineDriverHandle>,
) -> Result<()> {
    // `--local-model-only` has no `LoadOperation` reservation (event-system-
    // fixes deferral D2) -- degrade rather than fabricate an uncorrelated
    // root.
    let (_, model, _death_rx) = start_local_openai_model(launch, model_name, None).await?;
    if let Err(error) = wait_for_openai_ready(&model, bind_addr).await {
        model.shutdown().await;
        return Err(error);
    }

    let ready_url = format!("http://{}:{}/v1", connect_ip(bind_addr), bind_addr.port());
    let _ = emit_event(OutputEvent::ApiReady {
        url: ready_url.clone(),
    });
    let _ = emit_event(OutputEvent::RuntimeReady {
        api_url: ready_url,
        console_url: None,
        api_port: bind_addr.port(),
        console_port: None,
        models_count: Some(1),
        pi_command: None,
        goose_command: None,
    });
    super::node_lifecycle_events::emit_node_accepting_requests();

    let outcome = wait_for_openai_exit_or_shutdown(&model).await;
    let reason = outcome
        .as_ref()
        .map_or_else(|error| error.to_string(), |signal| signal.to_string());
    super::node_lifecycle_events::emit_node_draining();
    emit_shutdown(Some(reason)).await;
    model.shutdown().await;
    super::node_lifecycle_events::emit_node_stopped();
    // Task 3: finalize AFTER node_stopped, same ordering rationale as
    // `control_loop.rs::shutdown_run_auto_runtime` -- the driver's own
    // final drain is what applies and publishes both node lifecycle facts.
    if let Some(driver) = runtime_event_driver.take() {
        crate::runtime_events::driver::finalize_engine_driver(driver).await;
    }
    outcome.map(|_| ())
}

async fn wait_for_openai_ready(
    model: &super::LocalRuntimeModelHandle,
    bind_addr: SocketAddr,
) -> Result<()> {
    let deadline = tokio::time::Instant::now() + OPENAI_STARTUP_TIMEOUT;
    loop {
        let status = model.openai_server_status();
        if status.state == EmbeddedState::Failed {
            anyhow::bail!(
                "local OpenAI API failed during startup: {}",
                status.last_error.as_deref().unwrap_or("unknown error")
            );
        }
        if status.state == EmbeddedState::Ready && status.bind_addr == bind_addr {
            return Ok(());
        }
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "local OpenAI API did not bind {bind_addr}"
        );
        tokio::time::sleep(OPENAI_STATUS_POLL_INTERVAL).await;
    }
}

async fn wait_for_openai_exit_or_shutdown(
    model: &super::LocalRuntimeModelHandle,
) -> Result<&'static str> {
    let mut interval = tokio::time::interval(OPENAI_STATUS_POLL_INTERVAL);
    loop {
        tokio::select! {
            signal = wait_shutdown_signal() => return Ok(signal),
            _ = interval.tick() => {
                let status = model.openai_server_status();
                match status.state {
                    EmbeddedState::Failed => anyhow::bail!(
                        "local OpenAI API stopped: {}",
                        status.last_error.as_deref().unwrap_or("unknown error")
                    ),
                    EmbeddedState::Stopped => anyhow::bail!("local OpenAI API stopped unexpectedly"),
                    EmbeddedState::Starting | EmbeddedState::Ready | EmbeddedState::Stopping => {}
                }
            }
        }
    }
}

fn connect_ip(bind_addr: SocketAddr) -> IpAddr {
    if bind_addr.ip().is_unspecified() {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    } else {
        bind_addr.ip()
    }
}

#[cfg(test)]
mod tests {
    use super::{cleanup_local_model_only_runtime_event_state, run_local_model_only};
    use crate::inference::skippy::{SkippyModelHandle, SkippyModelLoadOptions};
    use crate::models::local::{huggingface_hub_cache_dir, scan_hf_cache_fast};
    use crate::runtime::RuntimeOptions;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{
        clear_runtime_event_engine, install_runtime_event_engine, runtime_event_engine,
    };
    use openai_frontend::ChatCompletionRequest;
    use skippy_protocol::{StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload};
    use std::path::PathBuf;
    use std::sync::Arc;

    /// The smallest real GGUF already cached under the user's local
    /// Hugging Face hub cache (the same cache `mesh-llm serve` populates),
    /// so this test exercises real inference without a network fetch or a
    /// checked-in fixture. Mirrors the existing `SKIPPY_MM_MODEL`-gated
    /// smoke tests' skip-when-unavailable convention when no cache exists.
    fn smallest_cached_gguf() -> Option<PathBuf> {
        scan_hf_cache_fast(&huggingface_hub_cache_dir())
            .into_iter()
            .filter_map(|path| {
                std::fs::metadata(&path)
                    .ok()
                    .filter(std::fs::Metadata::is_file)
                    .map(|metadata| (path, metadata.len()))
            })
            .min_by_key(|(_, size)| *size)
            .map(|(path, _)| path)
    }

    /// Review defect D1: local-model-only serving of a KV-disabled model
    /// failed every request with "session ... has no tracked position".
    /// `model_id` is unrecognizable to every family heuristic and
    /// `kv_cache.payload` is forced to `Auto`, so KV integration disables
    /// itself through the real unknown-family path in
    /// `kv_integration::model_capability` (the file's own dense tensor
    /// names, not certified-family knowledge, decide the outcome) exactly
    /// as it does for a genuinely uncertified model family.
    #[tokio::test]
    async fn kv_disabled_model_serves() {
        let Some(model_path) = smallest_cached_gguf() else {
            eprintln!(
                "skipping kv_disabled_model_serves: no cached GGUF found under {}",
                huggingface_hub_cache_dir().display()
            );
            return;
        };
        #[cfg(feature = "dynamic-native-runtime")]
        {
            let _ = crate::system::native_runtime::load_local_native_runtime_for_embedded_serving();
            if !skippy_runtime::native_runtime_loaded() {
                eprintln!(
                    "skipping kv_disabled_model_serves: no local native runtime bundle discovered (run `just build` first)"
                );
                return;
            }
        }

        let options = SkippyModelLoadOptions::for_direct_gguf(
            "local-test/unrecognized-family-model",
            model_path,
        )
        .with_ctx_size(512)
        .with_generation_concurrency(1)
        .with_kv_cache(Some(StageKvCacheConfig {
            mode: StageKvCacheMode::LookupRecord,
            payload: StageKvCachePayload::Auto,
            max_entries: 8,
            max_bytes: 0,
            min_tokens: 8,
            shared_prefix_stride_tokens: 8,
            shared_prefix_record_limit: 2,
        }));

        let handle = SkippyModelHandle::load(options).expect("load kv-disabled local model");
        let backend = handle.backend();

        for attempt in 0..3 {
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "local-test/unrecognized-family-model",
                "messages": [{"role": "user", "content": format!("ping {attempt}")}],
                "max_tokens": 4,
            }))
            .expect("build chat completion request");
            backend
                .chat_completion(request)
                .await
                .unwrap_or_else(|error| panic!("completion {attempt} failed: {error}"));
        }
    }

    /// Review defect D3: `--local-model-only` never drained a single
    /// runtime-event fact, because the only production drain call site was
    /// the presentation subscriber's own tick, and this mode deliberately
    /// never attaches a presentation (or any other management) subscriber
    /// -- see this module's own top-of-function comment in
    /// `run_local_model_only`. Proves the SAME wiring pattern
    /// `run_local_model_only` now uses (`spawn_engine_driver` right after
    /// `install_runtime_event_engine`) applies and releases a submitted
    /// terminal with zero subscribers ever attached, matching the plan's
    /// "zero management subscribers" invariant for this mode.
    #[tokio::test]
    async fn engine_driver() {
        use mesh_llm_runtime_event_contracts::{
            FamilyFact, NativeRuntimeEventKind, OperationId, RuntimeEventIngress, RuntimeFact,
            SubmitOutcome,
        };

        let engine = crate::runtime_events::engine::RuntimeEventEngine::new();
        assert_eq!(
            engine.subscribers().active_count(),
            0,
            "local-model-only attaches zero management subscribers before the driver even \
             starts"
        );
        let driver = crate::runtime_events::driver::spawn_engine_driver(engine.clone());

        let reservation = engine
            .reserve_root(OperationId::new(), || {
                RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
            })
            .expect("reservation");
        let fact =
            RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped));
        assert_eq!(
            reservation.ingress().try_submit(fact),
            SubmitOutcome::Accepted
        );

        for _ in 0..200 {
            if engine.occupied_count() == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }

        assert_eq!(
            engine.occupied_count(),
            0,
            "the engine-owned driver must apply and release the terminal with no subscriber \
             ever attached"
        );
        assert_eq!(
            engine.subscribers().active_count(),
            0,
            "draining must never require or create a subscriber"
        );

        driver.abort();
    }

    /// Startup errors after the engine and driver are installed must stop the
    /// driver before removing the process-local engine. An invalid config
    /// reaches the first fallible startup step after that installation.
    #[tokio::test]
    #[serial_test::serial(runtime_event_engine_state)]
    async fn startup_config_error_cleans_the_installed_engine() {
        clear_runtime_event_engine();
        let config_file = tempfile::NamedTempFile::new().expect("temporary config path");
        std::fs::write(config_file.path(), b"[").expect("write invalid config");

        let result = run_local_model_only(RuntimeOptions {
            local_model_only: true,
            config: Some(config_file.path().to_path_buf()),
            ..RuntimeOptions::default()
        })
        .await;

        let error = result.expect_err("invalid config must fail startup");
        assert!(
            format!("{error:#}").contains("Invalid config"),
            "startup must reach config loading after engine installation: {error:#}"
        );
        assert!(
            runtime_event_engine().is_none(),
            "startup failure must uninstall its engine after stopping its driver"
        );
    }

    /// The weak reference observes the driver's Arc disappearing only after
    /// `stop_and_wait` has completed, proving cleanup does not uninstall an
    /// engine while its driver task is still retaining it.
    #[tokio::test]
    #[serial_test::serial(runtime_event_engine_state)]
    async fn startup_cleanup_waits_for_driver_before_uninstalling_engine() {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        let weak_engine = Arc::downgrade(&engine);
        install_runtime_event_engine(engine.clone());
        let mut driver = Some(crate::runtime_events::driver::spawn_engine_driver(
            engine.clone(),
        ));

        cleanup_local_model_only_runtime_event_state(&engine, &mut driver).await;

        assert!(driver.is_none());
        assert!(runtime_event_engine().is_none());
        drop(engine);
        assert!(
            weak_engine.upgrade().is_none(),
            "stopped driver must not retain the uninstalled engine"
        );
    }
}
