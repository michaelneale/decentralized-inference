use super::model_lifecycle::runtime_model_audit_context;
use super::startup_retry::is_retryable_split_start_failure;
use super::status::current_time_unix_ms;
use super::status::single_quote_shell_arg;
use super::{
    DASHBOARD_CONTEXT_USAGE_REFRESH_INTERVAL, DashboardContextUsage, InitialPromptMode,
    InstanceLifecycleRecord, InstanceLifecycleState, LocalRuntimeModelHandle,
    LocalRuntimeModelStartSpec, OpenAiGuardrailPolicyHandle, RuntimeCapacityLedger,
    RuntimeCapacityReservation, RuntimeInstanceRegistry, RuntimeOperationalEvent,
    RuntimeResourcePlanningProfile, SPLIT_STANDBY_RETRY_INTERVAL, SplitCoordinatorAck,
    SplitCoordinatorEvent, SplitRuntimeReason, SplitRuntimeStart, StartupPinnedGpuTarget,
    StartupRuntimePlan, add_runtime_local_target, local_process_payload,
    publish_runtime_llama_slots, publish_runtime_llama_unavailable,
    record_runtime_operational_event, record_runtime_operational_event_with_context,
    refresh_dashboard_context_usage, register_runtime_instance, remove_dashboard_context_usage,
    remove_dashboard_process, remove_runtime_local_target, reserve_runtime_capacity_for_model,
    runtime_model_planning_bytes, runtime_model_required_bytes,
    runtime_process_payload_with_status, start_runtime_local_model, start_runtime_split_model,
    startup_runtime_plan, stop_split_generation_cleanup, unregister_runtime_instance,
    update_pi_models_json, upsert_dashboard_process,
};
use crate::api;
use crate::inference::{election, skippy};
use crate::mesh;
use crate::network::tunnel;
use crate::plugin;
use crate::runtime::interactive;
use crate::runtime::local;
use crate::runtime::survey;
use anyhow::Context;
use mesh_llm_events::{OutputEvent, emit_event, output_sink, schedule_ready_prompt};
use skippy_protocol::FlashAttentionType;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU16, Ordering},
};
use std::time::{Duration, Instant};

mod startup_loop;
pub(super) use startup_loop::*;

pub(super) type BootstrapProxyStopTx =
    tokio::sync::mpsc::Sender<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>;

pub(super) struct StartupLaunchHandles {
    pub(super) loaded_name: String,
    pub(super) handle: LocalRuntimeModelHandle,
    pub(super) death_rx: tokio::sync::oneshot::Receiver<()>,
    pub(super) split_cleanup: Option<local::SplitGenerationCleanup>,
    pub(super) split_event_rx: Option<tokio::sync::mpsc::Receiver<SplitCoordinatorEvent>>,
    pub(super) coordinator_task: Option<tokio::task::JoinHandle<()>>,
    pub(super) capacity_reservation: Option<RuntimeCapacityReservation>,
}

pub(super) struct StartupLocalModelTask {
    pub(super) node: mesh::Node,
    pub(super) config: plugin::MeshConfig,
    pub(super) tunnel_mgr: tunnel::Manager,
    pub(super) target_tx: Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) model_path: PathBuf,
    pub(super) model_ref: String,
    pub(super) preindexed_split_package: Option<skippy::SkippyPackageIdentity>,
    pub(super) config_model_id: Option<String>,
    pub(super) readiness_index: usize,
    pub(super) profile: String,
    pub(super) model_name: String,
    pub(super) instance_id: String,
    pub(super) primary_model_name: String,
    pub(super) mmproj_path: Option<PathBuf>,
    pub(super) ctx_size: Option<u32>,
    pub(super) pinned_gpu: Option<StartupPinnedGpuTarget>,
    pub(super) device_override: Option<String>,
    pub(super) runtime_capacity_ledger: RuntimeCapacityLedger,
    pub(super) cache_type_k: Option<String>,
    pub(super) cache_type_v: Option<String>,
    pub(super) n_batch: Option<u32>,
    pub(super) n_ubatch: Option<u32>,
    pub(super) flash_attention: FlashAttentionType,
    pub(super) parallel_override: Option<usize>,
    pub(super) local_source_required: bool,
    pub(super) split_topology_lock: Option<PathBuf>,
    pub(super) resource_planning_profile: RuntimeResourcePlanningProfile,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) split: bool,
    pub(super) skippy_telemetry: skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: survey::SurveyTelemetry,
    pub(super) survey_launch_kind: survey::SurveyLaunchKind,
    pub(super) stop_rx: tokio::sync::watch::Receiver<bool>,
    pub(super) dashboard_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) dashboard_context_usage: DashboardContextUsage,
    pub(super) runtime_instance_registry: RuntimeInstanceRegistry,
    pub(super) console_state: Option<api::MeshApi>,
    pub(super) api_port: u16,
    pub(super) startup_ready_reporter: StartupReadyReporter,
    pub(super) runtime_event_tx: tokio::sync::mpsc::UnboundedSender<super::RuntimeEvent>,
    pub(super) startup_load_gate: Arc<tokio::sync::Mutex<()>>,
    pub(super) input_handler_enabled: bool,
    pub(super) interactive_started: Arc<AtomicBool>,
    pub(super) interactive_control_tx:
        tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) interactive_console_state: Option<api::MeshApi>,
    pub(super) lifecycle: Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
    pub(super) lifecycle_port: Arc<AtomicU16>,
}

pub(super) struct StartupLaunchFailureContext<'a> {
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
}

pub(super) struct StartupSplitRuntimeLoopParams<'a, F, G>
where
    F: Fn() -> LocalRuntimeModelStartSpec<'a>,
    G: Fn() -> survey::SurveyModelSpec<'a> + Copy,
{
    make_start_spec: F,
    model_ref: &'a str,
    model_name: &'a str,
    local_capacity: u64,
    model_bytes: u64,
    node: &'a mesh::Node,
    startup_load_gate: &'a Arc<tokio::sync::Mutex<()>>,
    stop_rx: &'a mut tokio::sync::watch::Receiver<bool>,
    launch_failure: StartupLaunchFailureContext<'a>,
    make_survey_spec: G,
    announce_capacity_fallback: bool,
}

pub(super) struct StartupLocalRuntimeOnceParams<'a, F>
where
    F: Fn() -> survey::SurveyModelSpec<'a>,
{
    make_start_spec: LocalRuntimeModelStartSpec<'a>,
    runtime_capacity_ledger: &'a RuntimeCapacityLedger,
    instance_id: &'a str,
    model_name: &'a str,
    pinned_gpu: Option<&'a StartupPinnedGpuTarget>,
    local_capacity: u64,
    model_bytes: u64,
    startup_load_gate: &'a Arc<tokio::sync::Mutex<()>>,
    launch_failure: StartupLaunchFailureContext<'a>,
    make_survey_spec: F,
    model_ref: &'a str,
}

pub(super) struct PreparedRuntimeStartup {
    pub(super) startup_specs: Vec<super::StartupModelSpec>,
    pub(super) requested_model_names: Vec<String>,
    pub(super) bin_dir: PathBuf,
}

pub(super) struct RunAutoJoinOutcome {
    pub(super) joined: bool,
    pub(super) last_join_error: Option<String>,
    pub(super) successful_join: Option<(String, Option<String>)>,
}

pub(super) struct ShutdownRuntimeLoadedModelsContext<'a> {
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) dashboard_processes: &'a Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) runtime_instance_registry: &'a RuntimeInstanceRegistry,
    pub(super) node: &'a mesh::Node,
    pub(super) runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
    pub(super) dashboard_context_usage: &'a DashboardContextUsage,
}

pub(super) async fn startup_reset_model_target(
    target_tx: &Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    console_state: Option<&api::MeshApi>,
) {
    update_startup_target(target_tx, model_name, election::InferenceTarget::None);
    if let Some(cs) = console_state {
        cs.update(false, false).await;
    }
}

pub(super) async fn startup_emit_model_inspection_failure(
    target_tx: &Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    err: &anyhow::Error,
    console_state: Option<&api::MeshApi>,
) {
    let _ = emit_event(OutputEvent::Error {
        message: format!("Failed to inspect model {model_name}: {err:#}"),
        context: Some(format!("model={model_name}")),
    });
    startup_reset_model_target(target_tx, model_name, console_state).await;
}

pub(super) async fn startup_emit_launch_failure(
    survey_telemetry: &survey::SurveyTelemetry,
    survey_spec: survey::SurveyModelSpec<'_>,
    launch_started: Instant,
    err: anyhow::Error,
    target_tx: &Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    console_state: Option<&api::MeshApi>,
) {
    survey_telemetry.record_launch_failure(
        survey_spec,
        launch_started.elapsed(),
        survey::classify_launch_failure(&err),
    );
    let _ = emit_event(OutputEvent::Error {
        message: format!("Failed to start model {model_name}: {err:#}"),
        context: Some(format!("model={model_name}")),
    });
    startup_reset_model_target(target_tx, model_name, console_state).await;
}

pub(super) async fn startup_start_split_runtime_loop<'a, F, G>(
    params: StartupSplitRuntimeLoopParams<'a, F, G>,
) -> Option<(StartupLaunchHandles, Instant)>
where
    F: Fn() -> LocalRuntimeModelStartSpec<'a>,
    G: Fn() -> survey::SurveyModelSpec<'a> + Copy,
{
    let StartupSplitRuntimeLoopParams {
        make_start_spec,
        model_ref,
        model_name,
        local_capacity,
        model_bytes,
        node,
        startup_load_gate,
        stop_rx,
        launch_failure,
        make_survey_spec,
        announce_capacity_fallback,
    } = params;
    let StartupLaunchFailureContext {
        target_tx,
        console_state,
        survey_telemetry,
    } = launch_failure;

    if announce_capacity_fallback {
        let required_bytes = runtime_model_required_bytes(model_bytes);
        let _ = emit_event(OutputEvent::Info {
            message: format!(
                "Model {model_name} exceeds local runtime capacity; attempting split runtime"
            ),
            context: Some(format!(
                "model={model_name} local_capacity_gb={:.1} required_capacity_gb={:.1} model_size_gb={:.1}",
                local_capacity as f64 / 1e9,
                required_bytes as f64 / 1e9,
                model_bytes as f64 / 1e9
            )),
        });
    }

    let mut peer_rx = node.peer_change_rx.clone();
    loop {
        let startup_load_guard = startup_load_gate.lock().await;
        let launch_started = Instant::now();
        match start_runtime_split_model(make_start_spec(), model_ref).await {
            Ok(SplitRuntimeStart::Started(loaded)) => {
                drop(startup_load_guard);
                let mut loaded = *loaded;
                return Some((
                    StartupLaunchHandles {
                        loaded_name: loaded.loaded_name,
                        handle: loaded.handle,
                        death_rx: loaded.death_rx,
                        split_cleanup: loaded.cleanup.take(),
                        split_event_rx: loaded.coordinator_rx.take(),
                        coordinator_task: loaded.coordinator_task.take(),
                        capacity_reservation: None,
                    },
                    launch_started,
                ));
            }
            Ok(SplitRuntimeStart::Standby { coordinator }) => {
                drop(startup_load_guard);
                let _ = emit_event(OutputEvent::Info {
                    message: format!(
                        "Split runtime coordinator is {}; standing by for stage assignment",
                        coordinator.fmt_short()
                    ),
                    context: Some(format!("model={model_ref}")),
                });
                startup_reset_model_target(target_tx, model_name, console_state).await;
            }
            Err(err) => {
                drop(startup_load_guard);
                let err_msg = format!("{err:#}");
                if is_retryable_split_start_failure(&err_msg) {
                    let _ = emit_event(OutputEvent::Info {
                        message: format!("Split waiting to retry: {err_msg}"),
                        context: Some(format!("model={model_name}")),
                    });
                } else {
                    startup_emit_launch_failure(
                        survey_telemetry,
                        make_survey_spec(),
                        launch_started,
                        err,
                        target_tx,
                        model_name,
                        console_state,
                    )
                    .await;
                    return None;
                }
            }
        }

        tokio::select! {
            result = peer_rx.changed() => {
                if result.is_err() {
                    return None;
                }
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(2)) => {}
                    result = stop_rx.changed() => {
                        if result.is_err() || *stop_rx.borrow() {
                            return None;
                        }
                    }
                }
            }
            _ = tokio::time::sleep(SPLIT_STANDBY_RETRY_INTERVAL) => {}
            result = stop_rx.changed() => {
                if result.is_err() || *stop_rx.borrow() {
                    return None;
                }
            }
        }
    }
}

pub(super) async fn startup_start_local_runtime_once<'a, F>(
    params: StartupLocalRuntimeOnceParams<'a, F>,
) -> Option<(StartupLaunchHandles, Instant)>
where
    F: Fn() -> survey::SurveyModelSpec<'a>,
{
    let StartupLocalRuntimeOnceParams {
        mut make_start_spec,
        runtime_capacity_ledger,
        instance_id,
        model_name,
        pinned_gpu,
        local_capacity,
        model_bytes,
        startup_load_gate,
        launch_failure,
        make_survey_spec,
        model_ref,
    } = params;
    let StartupLaunchFailureContext {
        target_tx,
        console_state,
        survey_telemetry,
    } = launch_failure;

    let startup_load_guard = startup_load_gate.lock().await;
    let launch_started = Instant::now();
    let reservation = match reserve_runtime_capacity_for_model(
        runtime_capacity_ledger,
        instance_id,
        model_name,
        pinned_gpu,
        local_capacity,
        model_bytes,
    ) {
        Ok(reservation) => reservation,
        Err(err) => {
            drop(startup_load_guard);
            startup_emit_launch_failure(
                survey_telemetry,
                make_survey_spec(),
                launch_started,
                err,
                target_tx,
                model_name,
                console_state,
            )
            .await;
            return None;
        }
    };

    make_start_spec.capacity_budget_bytes = Some(reservation.capacity_budget_bytes());
    // Startup-time loads have no `LoadOperation` reservation of their own
    // (event-system-fixes deferral D2 scopes `ModelLoadProgress` to the
    // runtime-load path via `LoadOperation::progress_ingress`) -- degrade
    // to no progress reporting here rather than fabricate an uncorrelated
    // root.
    let start_result = start_runtime_local_model(make_start_spec, model_ref, None).await;
    drop(startup_load_guard);

    match start_result {
        Ok((loaded_name, handle, death_rx)) => Some((
            StartupLaunchHandles {
                loaded_name,
                handle,
                death_rx,
                split_cleanup: None,
                split_event_rx: None,
                coordinator_task: None,
                capacity_reservation: Some(reservation),
            },
            launch_started,
        )),
        Err(err) => {
            drop(reservation);
            startup_emit_launch_failure(
                survey_telemetry,
                make_survey_spec(),
                launch_started,
                err,
                target_tx,
                model_name,
                console_state,
            )
            .await;
            None
        }
    }
}

pub(super) fn startup_split_unavailable_stage_nodes(nodes: &[iroh::EndpointId]) -> String {
    nodes
        .iter()
        .map(|node| node.fmt_short().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

pub(super) async fn startup_unregister_runtime_instance(
    ctx: &StartupLoopContext<'_>,
    model_name: &str,
) -> bool {
    let port = ctx.lifecycle_port.swap(0, Ordering::AcqRel);
    if port != 0 {
        ctx.node.unregister_runtime_instance_lifecycle(port);
    }
    unregister_runtime_instance(
        ctx.runtime_instance_registry,
        ctx.node,
        model_name,
        ctx.instance_id,
    )
    .await
}

pub(super) async fn startup_remove_runtime_instance_artifacts(
    ctx: &StartupLoopContext<'_>,
    model_name: &str,
) {
    if startup_unregister_runtime_instance(ctx, model_name).await {
        publish_runtime_llama_unavailable(
            ctx.runtime_data_producer,
            model_name,
            Some(ctx.instance_id),
        );
    }
    remove_dashboard_process(ctx.dashboard_processes, ctx.instance_id).await;
    if let Some(cs) = ctx.console_state {
        cs.remove_local_process(ctx.instance_id).await;
        cs.update(false, false).await;
    }
}

pub(super) async fn startup_register_loaded_runtime(
    ctx: &StartupLoopContext<'_>,
    loaded_name: &str,
    handle: &LocalRuntimeModelHandle,
) -> api::RuntimeProcessPayload {
    let previous_port = ctx.lifecycle_port.swap(handle.port, Ordering::AcqRel);
    if previous_port != 0 && previous_port != handle.port {
        ctx.node
            .unregister_runtime_instance_lifecycle(previous_port);
    }
    {
        let mut lifecycle = ctx.lifecycle.lock().await;
        lifecycle
            .transition_to(InstanceLifecycleState::Serving)
            .expect("managed runtime lifecycle must reach serving from warming");
    }
    ctx.node
        .register_runtime_instance_lifecycle(handle.port, ctx.lifecycle.clone());
    add_runtime_local_target(ctx.target_tx, loaded_name, handle.port);
    ctx.tunnel_mgr.set_http_port(ctx.api_port);
    register_runtime_instance(
        ctx.runtime_instance_registry,
        ctx.node,
        ctx.primary_model_name,
        loaded_name,
        ctx.instance_id,
        Some(handle.context_length),
        handle.capabilities,
    )
    .await;
    let payload = local_process_payload(
        loaded_name,
        Some(ctx.instance_id),
        "",
        &handle.backend,
        handle.port,
        handle.pid(),
        handle.slots,
        handle.context_length,
    );
    upsert_dashboard_process(ctx.dashboard_processes, payload.clone()).await;
    payload
}

pub(super) async fn startup_prepare_launch(
    ctx: StartupPrepareLaunchContext<'_>,
) -> Option<StartupPreparedLaunch> {
    let local_capacity = ctx
        .pinned_gpu
        .map(|gpu| gpu.allocatable_vram_bytes())
        .unwrap_or_else(|| ctx.node.local_runtime_capacity_bytes());
    let model_bytes = startup_planning_model_bytes(&ctx).await?;
    let runtime_plan = startup_runtime_plan(ctx.split, local_capacity, model_bytes);
    let launch_kind = startup_launch_kind(runtime_plan, ctx.survey_launch_kind);
    Some(StartupPreparedLaunch {
        local_capacity,
        model_bytes,
        runtime_plan,
        launch_kind,
    })
}

pub(super) async fn startup_planning_model_bytes(
    ctx: &StartupPrepareLaunchContext<'_>,
) -> Option<u64> {
    let model_path_for_sizing = ctx.model_path.to_path_buf();
    match tokio::task::spawn_blocking(move || runtime_model_planning_bytes(&model_path_for_sizing))
        .await
        .context("join runtime model sizing task")
        .and_then(|result| result)
    {
        Ok(model_bytes) => Some(model_bytes),
        Err(err) => {
            startup_emit_model_inspection_failure(
                ctx.target_tx,
                ctx.model_name,
                &err,
                ctx.console_state,
            )
            .await;
            None
        }
    }
}

pub(super) fn startup_launch_kind(
    runtime_plan: StartupRuntimePlan,
    survey_launch_kind: survey::SurveyLaunchKind,
) -> survey::SurveyLaunchKind {
    match runtime_plan {
        StartupRuntimePlan::Local => survey_launch_kind,
        StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::Forced,
        } => survey::SurveyLaunchKind::MoeShard,
        StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::LocalCapacity,
        } => survey::SurveyLaunchKind::MoeFallback,
    }
}

pub(super) async fn startup_launch_runtime(
    ctx: StartupLaunchRuntimeContext<'_>,
) -> Option<(StartupLaunchHandles, Instant)> {
    let StartupLaunchRuntimeContext {
        node,
        config,
        target_tx,
        model_path,
        preindexed_split_package,
        model_ref,
        config_model_id,
        runtime_profile,
        model_name,
        instance_id,
        mmproj_path,
        ctx_size,
        pinned_gpu,
        device_override,
        runtime_capacity_ledger,
        cache_type_k,
        cache_type_v,
        n_batch,
        n_ubatch,
        flash_attention,
        parallel_override,
        local_source_required,
        split_topology_lock,
        resource_planning_profile,
        openai_guardrail_policy,
        skippy_telemetry,
        survey_telemetry,
        console_state,
        startup_load_gate,
        stop_rx,
        local_capacity,
        model_bytes,
        runtime_plan,
        launch_kind,
    } = ctx;
    let make_start_spec = || LocalRuntimeModelStartSpec {
        node,
        mesh_config: config,
        config_model_id,
        runtime_profile,
        model_path,
        preindexed_split_package,
        model_bytes,
        mmproj_override: mmproj_path.map(PathBuf::as_path),
        ctx_size_override: ctx_size,
        pinned_gpu,
        device_override: device_override.clone(),
        capacity_budget_bytes: None,
        cache_type_k_override: cache_type_k,
        cache_type_v_override: cache_type_v,
        n_batch_override: n_batch,
        n_ubatch_override: n_ubatch,
        flash_attention_override: flash_attention,
        parallel_override,
        local_source_required,
        split_topology_lock,
        planning_profile: resource_planning_profile,
        openai_guardrail_policy: openai_guardrail_policy.clone(),
        skippy_telemetry: skippy_telemetry.clone(),
        survey_telemetry: survey_telemetry.clone(),
    };
    let make_launch_failure_spec = || survey::SurveyModelSpec {
        model: model_name,
        configured_model_selector: config_model_id,
        model_path: Some(model_path),
        launch_kind,
        pinned_gpu,
        backend: None,
        context_length: ctx_size.map(u64::from),
    };
    match runtime_plan {
        StartupRuntimePlan::Split { reason } => {
            startup_start_split_runtime_loop(StartupSplitRuntimeLoopParams {
                make_start_spec,
                model_ref,
                model_name,
                local_capacity,
                model_bytes,
                node,
                startup_load_gate,
                stop_rx,
                launch_failure: StartupLaunchFailureContext {
                    target_tx,
                    console_state,
                    survey_telemetry,
                },
                make_survey_spec: make_launch_failure_spec,
                announce_capacity_fallback: reason == SplitRuntimeReason::LocalCapacity,
            })
            .await
        }
        StartupRuntimePlan::Local => {
            startup_start_local_runtime_once(StartupLocalRuntimeOnceParams {
                make_start_spec: make_start_spec(),
                runtime_capacity_ledger,
                instance_id,
                model_name,
                pinned_gpu,
                local_capacity,
                model_bytes,
                startup_load_gate,
                launch_failure: StartupLaunchFailureContext {
                    target_tx,
                    console_state,
                    survey_telemetry,
                },
                make_survey_spec: make_launch_failure_spec,
                model_ref,
            })
            .await
        }
    }
}

pub(super) fn maybe_spawn_startup_interactive_handler(
    input_handler_enabled: bool,
    loaded_name: &str,
    primary_model_name: &str,
    interactive_started: &AtomicBool,
    interactive_control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    interactive_console_state: Option<api::MeshApi>,
) {
    if !input_handler_enabled || loaded_name != primary_model_name {
        return;
    }
    if interactive_started.swap(true, Ordering::AcqRel) || !std::io::stdin().is_terminal() {
        return;
    }
    if let Some(cs) = interactive_console_state {
        let Some(sink) = output_sink() else {
            return;
        };
        interactive::spawn_handler(
            interactive_control_tx,
            cs,
            sink,
            InitialPromptMode::Deferred,
        );
    }
}

pub(super) async fn runtime_data_producer_for_console(
    console_state: Option<&api::MeshApi>,
) -> Option<crate::runtime_data::RuntimeDataProducer> {
    match console_state {
        Some(cs) => Some(cs.runtime_data_producer().await),
        None => None,
    }
}

fn record_startup_load_started(params: &StartupLocalModelTask) -> Instant {
    let load_started = Instant::now();
    record_runtime_operational_event_with_context(
        RuntimeOperationalEvent::ModelLoadStarted,
        runtime_model_audit_context(Some(&params.model_name), &params.instance_id)
            .outcome("started"),
    );
    load_started
}

async fn run_startup_load_attempt<T, Start, Prepare, PrepareFuture, Failure, FailureFuture>(
    start: Start,
    prepare: Prepare,
    failure: Failure,
) -> Option<(T, Instant)>
where
    Start: FnOnce() -> Instant,
    Prepare: FnOnce() -> PrepareFuture,
    PrepareFuture: std::future::Future<Output = Option<T>>,
    Failure: FnOnce(Instant) -> FailureFuture,
    FailureFuture: std::future::Future<Output = ()>,
{
    let load_started = start();
    match prepare().await {
        Some(prepared) => Some((prepared, load_started)),
        None => {
            failure(load_started).await;
            None
        }
    }
}

pub(super) fn update_startup_target(
    target_tx: &Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    target: election::InferenceTarget,
) {
    let mut targets = target_tx.borrow().clone();
    targets.targets.insert(model_name.to_string(), vec![target]);
    target_tx.send_replace(targets);
}

#[derive(Clone)]
pub(super) struct StartupReadyReporter {
    pub(super) ready_by_model: Arc<Mutex<Vec<bool>>>,
    pub(super) emitted: Arc<AtomicBool>,
    pub(super) shutdown_requested: Arc<AtomicBool>,
    startup_failure_policy: mesh_llm_config::StartupFailurePolicy,
    terminal_failures: Arc<Mutex<Vec<String>>>,
    pub(super) primary_model: String,
    pub(super) api_url: String,
    pub(super) console_url: Option<String>,
    pub(super) api_port: u16,
    pub(super) console_port: Option<u16>,
}

impl StartupReadyReporter {
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "default-policy constructor remains covered by readiness tests"
        )
    )]
    pub(super) fn new(
        models: &[String],
        primary_model: String,
        api_url: String,
        console_url: Option<String>,
        api_port: u16,
        console_port: Option<u16>,
    ) -> Self {
        Self::new_with_failure_policy(
            models,
            primary_model,
            api_url,
            console_url,
            api_port,
            console_port,
            mesh_llm_config::StartupFailurePolicy::BestEffort,
        )
    }

    pub(super) fn new_with_failure_policy(
        models: &[String],
        primary_model: String,
        api_url: String,
        console_url: Option<String>,
        api_port: u16,
        console_port: Option<u16>,
        startup_failure_policy: mesh_llm_config::StartupFailurePolicy,
    ) -> Self {
        let ready_by_model = vec![false; models.len()];
        Self {
            ready_by_model: Arc::new(Mutex::new(ready_by_model)),
            emitted: Arc::new(AtomicBool::new(false)),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            startup_failure_policy,
            terminal_failures: Arc::new(Mutex::new(Vec::new())),
            primary_model,
            api_url,
            console_url,
            api_port,
            console_port,
        }
    }

    /// Record an eager startup failure. Returns true when fail-fast shutdown is required.
    pub(super) fn record_terminal_failure(&self, model_name: &str, detail: &str) -> bool {
        let mut failures = self
            .terminal_failures
            .lock()
            .expect("startup failure mutex poisoned");
        if failures.len() < 32 {
            let mut failure = format!("{model_name}: {detail}");
            if failure.len() > 512 {
                let mut end = 512;
                while !failure.is_char_boundary(end) {
                    end -= 1;
                }
                failure.truncate(end);
            }
            failures.push(failure);
        }
        self.startup_failure_policy == mesh_llm_config::StartupFailurePolicy::FailFast
    }

    pub(super) fn fail_fast_summary(&self) -> Option<String> {
        if self.startup_failure_policy != mesh_llm_config::StartupFailurePolicy::FailFast {
            return None;
        }
        let mut failures = self
            .terminal_failures
            .lock()
            .expect("startup failure mutex poisoned")
            .clone();
        if failures.is_empty() {
            return None;
        }
        failures.sort();
        let mut summary = format!(
            "eager startup failed ({}): {}",
            failures.len(),
            failures.join("; ")
        );
        if summary.len() > 1_024 {
            let mut end = 1_024;
            while !summary.is_char_boundary(end) {
                end -= 1;
            }
            summary.truncate(end);
        }
        Some(summary)
    }

    pub(super) fn mark_shutdown_requested(&self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
    }

    pub(super) fn mark_ready_and_build_event(&self, readiness_index: usize) -> Option<OutputEvent> {
        let models_count = {
            let mut ready_by_model = self
                .ready_by_model
                .lock()
                .expect("startup readiness mutex poisoned");
            if let Some(entry) = ready_by_model.get_mut(readiness_index) {
                *entry = true;
            }
            if ready_by_model.iter().all(|ready| *ready) {
                Some(ready_by_model.len())
            } else {
                None
            }
        };

        let models_count = models_count?;

        if self.shutdown_requested.load(Ordering::SeqCst) {
            return None;
        };

        if self.emitted.swap(true, Ordering::SeqCst) {
            return None;
        }

        let pi_command = Some(format!(
            "mesh-llm pi --host 127.0.0.1:{} --model {}",
            self.api_port,
            single_quote_shell_arg(&self.primary_model)
        ));
        let goose_command = Some(format!(
            "GOOSE_PROVIDER=openai OPENAI_HOST={} OPENAI_API_KEY=mesh GOOSE_MODEL={} goose session",
            self.api_url, self.primary_model
        ));
        Some(OutputEvent::RuntimeReady {
            api_url: self.api_url.clone(),
            console_url: self.console_url.clone(),
            api_port: self.api_port,
            console_port: self.console_port,
            models_count: Some(models_count),
            pi_command,
            goose_command,
        })
    }

    fn mark_ready_and_maybe_emit(&self, readiness_index: usize) {
        let Some(event) = self.mark_ready_and_build_event(readiness_index) else {
            return;
        };
        let _ = emit_event(event);
        record_runtime_operational_event(RuntimeOperationalEvent::Ready);
        super::node_lifecycle_events::emit_node_accepting_requests();
        let _ = schedule_ready_prompt();
    }
}

pub(super) async fn record_first_joined_mesh_ts(node: &mesh::Node) {
    let now_ms = current_time_unix_ms();
    node.set_first_joined_mesh_ts_if_absent(now_ms).await;
}

#[cfg(test)]
mod startup_failure_policy_tests {
    use super::{StartupReadyReporter, run_startup_load_attempt};
    use mesh_llm_config::StartupFailurePolicy;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    fn reporter(policy: StartupFailurePolicy) -> StartupReadyReporter {
        StartupReadyReporter::new_with_failure_policy(
            &["model-a".to_string()],
            "model-a".to_string(),
            "http://127.0.0.1:1".to_string(),
            None,
            1,
            None,
            policy,
        )
    }

    #[test]
    fn best_effort_startup_failure_does_not_request_shutdown() {
        let reporter = reporter(StartupFailurePolicy::BestEffort);
        assert!(!reporter.record_terminal_failure("model-a", "launch failed"));
        assert_eq!(reporter.fail_fast_summary(), None);
    }

    #[test]
    fn fail_fast_startup_failure_has_bounded_deterministic_summary() {
        let reporter = reporter(StartupFailurePolicy::FailFast);
        assert!(reporter.record_terminal_failure("model-z", &"é".repeat(400)));
        assert!(reporter.record_terminal_failure("model-a", "inspection failed"));

        let summary = reporter.fail_fast_summary().expect("fail-fast summary");
        assert!(summary.starts_with("eager startup failed (2): model-a:"));
        assert!(summary.len() <= 1_024);
    }

    #[tokio::test]
    async fn startup_preparation_failure_orders_start_before_failure() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let start_events = Arc::clone(&events);
        let prepare_events = Arc::clone(&events);
        let failure_events = Arc::clone(&events);

        let result = run_startup_load_attempt(
            move || {
                start_events.lock().unwrap().push("started");
                Instant::now()
            },
            move || async move {
                prepare_events.lock().unwrap().push("prepared");
                None::<()>
            },
            move |_load_started| async move {
                failure_events.lock().unwrap().push("failed");
            },
        )
        .await;

        assert!(result.is_none());
        assert_eq!(*events.lock().unwrap(), ["started", "prepared", "failed"]);
    }
}
