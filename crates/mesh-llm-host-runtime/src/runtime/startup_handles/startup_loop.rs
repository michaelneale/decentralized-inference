use super::*;
use crate::runtime::RuntimeEvent;

pub(in crate::runtime) struct StartupLoopContext<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) tunnel_mgr: &'a tunnel::Manager,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) model_path: &'a PathBuf,
    pub(super) model_ref: &'a str,
    pub(super) preindexed_split_package: Option<&'a skippy::SkippyPackageIdentity>,
    pub(super) config_model_id: Option<&'a str>,
    pub(super) runtime_profile: &'a str,
    pub(super) readiness_index: usize,
    pub(super) instance_id: &'a str,
    pub(super) primary_model_name: &'a str,
    pub(super) mmproj_path: Option<&'a PathBuf>,
    pub(super) ctx_size: Option<u32>,
    pub(super) pinned_gpu: Option<&'a StartupPinnedGpuTarget>,
    pub(super) device_override: Option<&'a str>,
    pub(super) runtime_capacity_ledger: &'a RuntimeCapacityLedger,
    pub(super) cache_type_k: Option<&'a str>,
    pub(super) cache_type_v: Option<&'a str>,
    pub(super) n_batch: Option<u32>,
    pub(super) n_ubatch: Option<u32>,
    pub(super) flash_attention: FlashAttentionType,
    pub(super) parallel_override: Option<usize>,
    pub(super) local_source_required: bool,
    pub(super) resource_planning_profile: RuntimeResourcePlanningProfile,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) skippy_telemetry: &'a skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) launch_kind: survey::SurveyLaunchKind,
    pub(super) dashboard_processes: &'a Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) dashboard_context_usage: &'a DashboardContextUsage,
    pub(super) runtime_instance_registry: &'a RuntimeInstanceRegistry,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) api_port: u16,
    pub(super) runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
    pub(super) lifecycle: &'a Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
    pub(super) lifecycle_port: &'a Arc<AtomicU16>,
}

pub(in crate::runtime) struct StartupLoopState {
    pub(super) loaded_name: String,
    pub(super) handle: Option<LocalRuntimeModelHandle>,
    pub(super) death_rx: tokio::sync::oneshot::Receiver<()>,
    pub(super) split_cleanup: Option<local::SplitGenerationCleanup>,
    pub(super) split_event_rx: Option<tokio::sync::mpsc::Receiver<SplitCoordinatorEvent>>,
    pub(super) survey_loaded_model: survey::SurveyLoadedModel,
    pub(super) capacity_reservation: Option<RuntimeCapacityReservation>,
    pub(super) survey_exited_unexpectedly: bool,
}

pub(in crate::runtime) struct StartupLoopEventContext<'a> {
    pub(super) context_usage_tick: &'a mut tokio::time::Interval,
    pub(super) stop_rx: &'a mut tokio::sync::watch::Receiver<bool>,
    pub(super) local_capacity: u64,
    pub(super) model_bytes: u64,
}

pub(in crate::runtime) enum StartupLoopControl {
    Continue,
    Break,
    Return,
    /// Tear down the current split runtime, then re-enter the launch phase
    /// (wait for eligible split participants again) instead of ending the
    /// model task. Used when a split topology is withdrawn but the model
    /// cannot fit locally: the peer may come back, and a healthy standing-by
    /// worker must not require a manual restart of both sides.
    RelaunchSplit,
}

/// Outcome of the local-model event loop, consumed by
/// `startup_local_model_loop` to decide between clean teardown, immediate
/// return, and split relaunch.
pub(in crate::runtime) enum StartupLoopOutcome {
    /// Tear down and end the model task.
    Shutdown,
    /// End the model task without the shared teardown (already handled).
    Exit,
    /// Tear down, then loop back to the launch phase and wait for peers.
    RelaunchSplit,
}

pub(in crate::runtime) struct StartupPreparedLaunch {
    pub(super) local_capacity: u64,
    pub(super) model_bytes: u64,
    pub(super) runtime_plan: StartupRuntimePlan,
    pub(super) launch_kind: survey::SurveyLaunchKind,
}

pub(in crate::runtime) struct StartupPrepareLaunchContext<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) pinned_gpu: Option<&'a StartupPinnedGpuTarget>,
    pub(super) model_path: &'a Path,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) model_name: &'a str,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) split: bool,
    pub(super) survey_launch_kind: survey::SurveyLaunchKind,
}

pub(in crate::runtime) struct StartupLaunchRuntimeContext<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) model_path: &'a PathBuf,
    pub(super) model_ref: &'a str,
    pub(super) preindexed_split_package: Option<&'a skippy::SkippyPackageIdentity>,
    pub(super) config_model_id: Option<&'a str>,
    pub(super) runtime_profile: &'a str,
    pub(super) model_name: &'a str,
    pub(super) instance_id: &'a str,
    pub(super) mmproj_path: Option<&'a PathBuf>,
    pub(super) ctx_size: Option<u32>,
    pub(super) pinned_gpu: Option<&'a StartupPinnedGpuTarget>,
    pub(super) device_override: Option<String>,
    pub(super) runtime_capacity_ledger: &'a RuntimeCapacityLedger,
    pub(super) cache_type_k: Option<&'a str>,
    pub(super) cache_type_v: Option<&'a str>,
    pub(super) n_batch: Option<u32>,
    pub(super) n_ubatch: Option<u32>,
    pub(super) flash_attention: FlashAttentionType,
    pub(super) parallel_override: Option<usize>,
    pub(super) local_source_required: bool,
    pub(super) split_topology_lock: Option<&'a Path>,
    pub(super) resource_planning_profile: RuntimeResourcePlanningProfile,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) skippy_telemetry: &'a skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) startup_load_gate: &'a Arc<tokio::sync::Mutex<()>>,
    pub(super) stop_rx: &'a mut tokio::sync::watch::Receiver<bool>,
    pub(super) local_capacity: u64,
    pub(super) model_bytes: u64,
    pub(super) runtime_plan: StartupRuntimePlan,
    pub(super) launch_kind: survey::SurveyLaunchKind,
}

pub(in crate::runtime) fn startup_fallback_survey_spec<'a>(
    ctx: &'a StartupLoopContext<'a>,
    model_name: &'a str,
    backend: Option<&'a str>,
    context_length: Option<u32>,
) -> survey::SurveyModelSpec<'a> {
    survey::SurveyModelSpec {
        model: model_name,
        configured_model_selector: ctx.config_model_id,
        model_path: Some(ctx.model_path),
        launch_kind: survey::SurveyLaunchKind::MoeFallback,
        pinned_gpu: ctx.pinned_gpu,
        backend,
        context_length: context_length.map(u64::from),
    }
}

pub(in crate::runtime) async fn startup_handle_fallback_failure(
    ctx: &StartupLoopContext<'_>,
    event: &local::SplitCoordinatorLocalFallbackEvent,
    model_name: &str,
    launch_started: Instant,
    err: &anyhow::Error,
    unavailable_stage_nodes: &str,
) -> StartupLoopControl {
    ctx.survey_telemetry.record_launch_failure(
        startup_fallback_survey_spec(ctx, model_name, None, ctx.ctx_size),
        launch_started.elapsed(),
        survey::classify_launch_failure(err),
    );
    let _ = emit_event(OutputEvent::Warning {
        message: format!(
            "Split runtime topology '{}' lost required stage peer(s); local fallback failed, withdrawing model '{}'",
            event.topology_id, model_name
        ),
        context: Some(format!(
            "reason={} generation={} unavailable_stage_nodes=[{}] error={err:#}",
            event.reason, event.generation, unavailable_stage_nodes
        )),
    });
    // The failed fallback exits without the shared shutdown path, so release
    // the local-model host claim explicitly before returning to the runtime.
    ctx.node
        .release_host_role(mesh::HostRoleClaim::LocalModel)
        .await;
    startup_remove_runtime_instance_artifacts(ctx, model_name).await;
    StartupLoopControl::Return
}

pub(in crate::runtime) async fn startup_handle_local_fallback_event(
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    event: local::SplitCoordinatorLocalFallbackEvent,
    local_capacity: u64,
    model_bytes: u64,
) -> StartupLoopControl {
    let unavailable_stage_nodes =
        startup_split_unavailable_stage_nodes(&event.unavailable_stage_nodes);
    let old_loaded_name = state.loaded_name.clone();
    let withdrew_topology = ctx
        .node
        .withdraw_stage_topology(&event.topology_id, &event.run_id)
        .await;
    let Some(old_handle) = state.handle.take() else {
        let _ = event.ack.send(SplitCoordinatorAck::Accepted);
        return StartupLoopControl::Break;
    };

    let old_port = old_handle.port;
    remove_runtime_local_target(ctx.target_tx, &old_loaded_name, old_port);
    remove_dashboard_context_usage(ctx.dashboard_context_usage, &old_loaded_name, &old_handle)
        .await;
    old_handle.shutdown().await;
    ctx.survey_telemetry
        .record_unload(&state.survey_loaded_model);
    if let Some(cleanup) = state.split_cleanup.take() {
        stop_split_generation_cleanup(ctx.node, cleanup, event.generation.saturating_add(1)).await;
    }

    let launch_started = Instant::now();
    let reservation = match reserve_runtime_capacity_for_model(
        ctx.runtime_capacity_ledger,
        ctx.instance_id,
        &old_loaded_name,
        ctx.pinned_gpu,
        local_capacity,
        model_bytes,
    ) {
        Ok(reservation) => reservation,
        Err(err) => {
            let result = startup_handle_fallback_failure(
                ctx,
                &event,
                &old_loaded_name,
                launch_started,
                &err,
                &unavailable_stage_nodes,
            )
            .await;
            let _ = event.ack.send(SplitCoordinatorAck::Accepted);
            return result;
        }
    };

    let start_result = start_runtime_local_model(
        LocalRuntimeModelStartSpec {
            node: ctx.node,
            mesh_config: ctx.config,
            config_model_id: ctx.config_model_id,
            runtime_profile: ctx.runtime_profile,
            model_path: ctx.model_path,
            preindexed_split_package: ctx.preindexed_split_package,
            model_bytes,
            mmproj_override: ctx.mmproj_path.map(PathBuf::as_path),
            ctx_size_override: ctx.ctx_size,
            pinned_gpu: ctx.pinned_gpu,
            device_override: ctx.device_override.map(str::to_string),
            capacity_budget_bytes: Some(reservation.capacity_budget_bytes()),
            cache_type_k_override: ctx.cache_type_k,
            cache_type_v_override: ctx.cache_type_v,
            n_batch_override: ctx.n_batch,
            n_ubatch_override: ctx.n_ubatch,
            flash_attention_override: ctx.flash_attention,
            parallel_override: ctx.parallel_override,
            local_source_required: ctx.local_source_required,
            split_topology_lock: None,
            planning_profile: ctx.resource_planning_profile,
            openai_guardrail_policy: ctx.openai_guardrail_policy.clone(),
            skippy_telemetry: ctx.skippy_telemetry.clone(),
            survey_telemetry: ctx.survey_telemetry.clone(),
        },
        ctx.model_ref,
    )
    .await;

    let (next_loaded_name, next_handle, next_death_rx) = match start_result {
        Ok(result) => result,
        Err(err) => {
            drop(reservation);
            let result = startup_handle_fallback_failure(
                ctx,
                &event,
                &old_loaded_name,
                launch_started,
                &err,
                &unavailable_stage_nodes,
            )
            .await;
            let _ = event.ack.send(SplitCoordinatorAck::Accepted);
            return result;
        }
    };

    state.capacity_reservation = Some(reservation);
    state.loaded_name = next_loaded_name;
    let payload = startup_register_loaded_runtime(ctx, &state.loaded_name, &next_handle).await;
    if let Some(cs) = ctx.console_state {
        cs.upsert_local_process(payload).await;
        cs.update(true, true).await;
    }
    state.survey_loaded_model = ctx.survey_telemetry.model(startup_fallback_survey_spec(
        ctx,
        &state.loaded_name,
        Some(&next_handle.backend),
        Some(next_handle.context_length),
    ));
    ctx.survey_telemetry
        .record_launch_success(&state.survey_loaded_model, launch_started.elapsed());
    refresh_dashboard_context_usage(
        ctx.dashboard_context_usage,
        &state.loaded_name,
        &next_handle,
    )
    .await;
    publish_runtime_llama_slots(
        ctx.runtime_data_producer,
        &state.loaded_name,
        Some(ctx.instance_id),
        &next_handle,
    );
    let new_port = next_handle.port;
    let new_context_length = next_handle.context_length;
    state.handle = Some(next_handle);
    state.death_rx = next_death_rx;
    state.split_event_rx = None;
    let _ = event.ack.send(SplitCoordinatorAck::Accepted);
    let _ = emit_event(OutputEvent::Warning {
        message: format!(
            "Split runtime topology '{}' lost required stage peer(s); recovered model '{}' locally",
            event.topology_id, state.loaded_name
        ),
        context: Some(format!(
            "reason={} generation={} run_id={} topology_withdrawn={} unavailable_stage_nodes=[{}] previous_port={} new_port={} new_ctx={}",
            event.reason,
            event.generation,
            event.run_id,
            withdrew_topology,
            unavailable_stage_nodes,
            old_port,
            new_port,
            new_context_length
        )),
    });
    StartupLoopControl::Continue
}

pub(in crate::runtime) async fn startup_handle_replace_event(
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    event: local::SplitCoordinatorReplaceEvent,
) -> StartupLoopControl {
    let mut next = event.loaded;
    let old_loaded_name = state.loaded_name.clone();
    let Some(old_handle) = state.handle.take() else {
        let _ = event.ack.send(SplitCoordinatorAck::Accepted);
        return StartupLoopControl::Break;
    };

    let old_port = old_handle.port;
    let old_context_length = old_handle.context_length;
    remove_runtime_local_target(ctx.target_tx, &old_loaded_name, old_port);
    add_runtime_local_target(ctx.target_tx, &next.loaded_name, next.handle.port);
    ctx.tunnel_mgr.set_http_port(ctx.api_port);
    if old_loaded_name != next.loaded_name
        && startup_unregister_runtime_instance(ctx, &old_loaded_name).await
    {
        publish_runtime_llama_unavailable(
            ctx.runtime_data_producer,
            &old_loaded_name,
            Some(ctx.instance_id),
        );
    }
    let payload = startup_register_loaded_runtime(ctx, &next.loaded_name, &next.handle).await;
    if let Some(cs) = ctx.console_state {
        cs.upsert_local_process(payload).await;
        cs.update(true, true).await;
    }
    remove_dashboard_context_usage(ctx.dashboard_context_usage, &old_loaded_name, &old_handle)
        .await;
    ctx.survey_telemetry
        .record_unload(&state.survey_loaded_model);
    state.loaded_name = next.loaded_name;
    state.survey_loaded_model = ctx.survey_telemetry.model(survey::SurveyModelSpec {
        model: &state.loaded_name,
        configured_model_selector: ctx.config_model_id,
        model_path: Some(ctx.model_path),
        launch_kind: ctx.launch_kind,
        pinned_gpu: ctx.pinned_gpu,
        backend: Some(&next.handle.backend),
        context_length: Some(u64::from(next.handle.context_length)),
    });
    ctx.survey_telemetry
        .record_launch_success(&state.survey_loaded_model, Duration::from_secs(0));
    refresh_dashboard_context_usage(
        ctx.dashboard_context_usage,
        &state.loaded_name,
        &next.handle,
    )
    .await;
    publish_runtime_llama_slots(
        ctx.runtime_data_producer,
        &state.loaded_name,
        Some(ctx.instance_id),
        &next.handle,
    );
    let new_port = next.handle.port;
    let new_context_length = next.handle.context_length;
    state.death_rx = next.death_rx;
    state.split_cleanup = next.cleanup.take();
    state.handle = Some(next.handle);
    let _ = event.ack.send(SplitCoordinatorAck::Accepted);
    old_handle.shutdown().await;
    drop(state.capacity_reservation.take());
    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Split runtime cut over model '{}' from :{} to :{}",
            state.loaded_name, old_port, new_port
        ),
        context: Some(format!(
            "reason={} generation={} previous_ctx={} new_ctx={}",
            event.reason, event.generation, old_context_length, new_context_length
        )),
    });
    StartupLoopControl::Continue
}

pub(in crate::runtime) async fn startup_handle_split_event(
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    event: SplitCoordinatorEvent,
    local_capacity: u64,
    model_bytes: u64,
) -> StartupLoopControl {
    match event {
        SplitCoordinatorEvent::Replace(event) => {
            startup_handle_replace_event(ctx, state, *event).await
        }
        SplitCoordinatorEvent::LocalFallback(event) => {
            startup_handle_local_fallback_event(ctx, state, event, local_capacity, model_bytes)
                .await
        }
        SplitCoordinatorEvent::Withdraw(event) => {
            let unavailable_stage_nodes =
                startup_split_unavailable_stage_nodes(&event.unavailable_stage_nodes);
            let withdrew_topology = ctx
                .node
                .withdraw_stage_topology(&event.topology_id, &event.run_id)
                .await;
            let _ = emit_event(OutputEvent::Warning {
                message: format!(
                    "Split runtime topology '{}' lost required stage peer(s); withdrawing model '{}' and waiting for peers to relaunch",
                    event.topology_id, state.loaded_name
                ),
                context: Some(format!(
                    "reason={} generation={} run_id={} topology_withdrawn={} unavailable_stage_nodes=[{}]",
                    event.reason,
                    event.generation,
                    event.run_id,
                    withdrew_topology,
                    unavailable_stage_nodes
                )),
            });
            let _ = event.ack.send(SplitCoordinatorAck::Accepted);
            StartupLoopControl::RelaunchSplit
        }
    }
}

pub(in crate::runtime) async fn startup_shutdown_local_model_loop(
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    coordinator_task: &mut Option<tokio::task::JoinHandle<()>>,
) {
    if let Some(task) = coordinator_task.take() {
        task.abort();
        let _ = task.await;
    }
    ctx.node
        .release_host_role(mesh::HostRoleClaim::LocalModel)
        .await;
    if !state.survey_exited_unexpectedly {
        ctx.survey_telemetry
            .record_unload(&state.survey_loaded_model);
    }
    let Some(handle) = state.handle.take() else {
        drop(state.capacity_reservation.take());
        return;
    };
    let port = handle.port;
    remove_runtime_local_target(ctx.target_tx, &state.loaded_name, port);
    ctx.tunnel_mgr.set_http_port(ctx.api_port);
    if startup_unregister_runtime_instance(ctx, &state.loaded_name).await {
        publish_runtime_llama_unavailable(
            ctx.runtime_data_producer,
            &state.loaded_name,
            Some(ctx.instance_id),
        );
    }
    let shutting_down_payload = runtime_process_payload_with_status(
        &state.loaded_name,
        Some(ctx.instance_id),
        &handle,
        "shutting down",
    );
    upsert_dashboard_process(ctx.dashboard_processes, shutting_down_payload.clone()).await;
    if let Some(cs) = ctx.console_state {
        cs.upsert_local_process(shutting_down_payload).await;
    }
    remove_dashboard_context_usage(ctx.dashboard_context_usage, &state.loaded_name, &handle).await;
    handle.shutdown().await;
    drop(state.capacity_reservation.take());
    if let Some(cleanup) = state.split_cleanup.take() {
        stop_split_generation_cleanup(ctx.node, cleanup, u64::MAX).await;
    }
    remove_dashboard_process(ctx.dashboard_processes, ctx.instance_id).await;
    if let Some(cs) = ctx.console_state {
        cs.remove_local_process(ctx.instance_id).await;
        cs.update(false, false).await;
    }
    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Stopped startup model '{}' from :{}",
            state.loaded_name, port
        ),
        context: None,
    });
}

pub(in crate::runtime) async fn startup_local_model_loop(params: StartupLocalModelTask) {
    let runtime_data_producer =
        runtime_data_producer_for_console(params.console_state.as_ref()).await;
    let Some((prepared, mut load_started)) = run_startup_load_attempt(
        || record_startup_load_started(&params),
        || prepare_startup_local_model_task(&params),
        |load_started| {
            record_startup_task_failure(&params, "model inspection failed", load_started)
        },
    )
    .await
    else {
        return;
    };
    let mut stop_rx = params.stop_rx.clone();
    let mut first_attempt = true;
    loop {
        reset_startup_lifecycle(&params.lifecycle).await;
        if !first_attempt {
            load_started = record_startup_load_started(&params);
        }
        first_attempt = false;
        let Some((launch_handles, launch_started)) =
            launch_startup_local_model_task(&params, &mut stop_rx, &prepared).await
        else {
            record_startup_task_failure(&params, "model launch failed", load_started).await;
            return;
        };
        let StartupLaunchHandles {
            loaded_name,
            handle,
            death_rx,
            split_cleanup,
            split_event_rx,
            mut coordinator_task,
            capacity_reservation,
        } = launch_handles;
        params
            .lifecycle
            .lock()
            .await
            .transition_to(InstanceLifecycleState::Warming)
            .expect("managed runtime lifecycle loading to warming");

        let survey_loaded_model = params.survey_telemetry.model(survey::SurveyModelSpec {
            model: &loaded_name,
            configured_model_selector: params.config_model_id.as_deref(),
            model_path: Some(&params.model_path),
            launch_kind: prepared.launch_kind,
            pinned_gpu: params.pinned_gpu.as_ref(),
            backend: Some(&handle.backend),
            context_length: Some(u64::from(handle.context_length)),
        });
        params
            .survey_telemetry
            .record_launch_success(&survey_loaded_model, launch_started.elapsed());

        let ctx = startup_loop_context(
            &params,
            runtime_data_producer.as_ref(),
            prepared.launch_kind,
        );
        publish_startup_local_model(
            &params,
            &ctx,
            &loaded_name,
            &handle,
            launch_started.elapsed(),
        )
        .await;

        let mut state = StartupLoopState {
            loaded_name,
            handle: Some(handle),
            death_rx,
            split_cleanup,
            split_event_rx,
            survey_loaded_model,
            capacity_reservation,
            survey_exited_unexpectedly: false,
        };
        let mut context_usage_tick =
            tokio::time::interval(DASHBOARD_CONTEXT_USAGE_REFRESH_INTERVAL);
        context_usage_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let outcome = startup_run_local_model_event_loop(
            &ctx,
            &mut state,
            StartupLoopEventContext {
                context_usage_tick: &mut context_usage_tick,
                stop_rx: &mut stop_rx,
                local_capacity: prepared.local_capacity,
                model_bytes: prepared.model_bytes,
            },
        )
        .await;

        if !startup_resolve_loop_outcome(
            outcome,
            &ctx,
            &mut state,
            &mut coordinator_task,
            &stop_rx,
            &params.model_name,
        )
        .await
        {
            return;
        }
    }
}

async fn prepare_startup_local_model_task(
    params: &StartupLocalModelTask,
) -> Option<StartupPreparedLaunch> {
    startup_prepare_launch(StartupPrepareLaunchContext {
        node: &params.node,
        pinned_gpu: params.pinned_gpu.as_ref(),
        model_path: &params.model_path,
        target_tx: &params.target_tx,
        model_name: &params.model_name,
        console_state: params.console_state.as_ref(),
        split: params.split,
        survey_launch_kind: params.survey_launch_kind,
    })
    .await
}

async fn reset_startup_lifecycle(lifecycle: &Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>) {
    let mut record = lifecycle.lock().await;
    *record = InstanceLifecycleRecord::new(InstanceLifecycleState::Planned, 32);
    record
        .transition_to(InstanceLifecycleState::Resolving)
        .expect("managed runtime lifecycle planned to resolving");
    record
        .transition_to(InstanceLifecycleState::Loading)
        .expect("managed runtime lifecycle resolving to loading");
}

async fn launch_startup_local_model_task(
    params: &StartupLocalModelTask,
    stop_rx: &mut tokio::sync::watch::Receiver<bool>,
    prepared: &StartupPreparedLaunch,
) -> Option<(StartupLaunchHandles, Instant)> {
    startup_launch_runtime(StartupLaunchRuntimeContext {
        node: &params.node,
        config: &params.config,
        target_tx: &params.target_tx,
        model_path: &params.model_path,
        model_ref: &params.model_ref,
        preindexed_split_package: params.preindexed_split_package.as_ref(),
        config_model_id: params.config_model_id.as_deref(),
        runtime_profile: &params.profile,
        model_name: &params.model_name,
        instance_id: &params.instance_id,
        mmproj_path: params.mmproj_path.as_ref(),
        ctx_size: params.ctx_size,
        pinned_gpu: params.pinned_gpu.as_ref(),
        device_override: params.device_override.clone(),
        runtime_capacity_ledger: &params.runtime_capacity_ledger,
        cache_type_k: params.cache_type_k.as_deref(),
        cache_type_v: params.cache_type_v.as_deref(),
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        flash_attention: params.flash_attention,
        parallel_override: params.parallel_override,
        local_source_required: params.local_source_required,
        split_topology_lock: params.split_topology_lock.as_deref(),
        resource_planning_profile: params.resource_planning_profile,
        openai_guardrail_policy: params.openai_guardrail_policy.clone(),
        skippy_telemetry: &params.skippy_telemetry,
        survey_telemetry: &params.survey_telemetry,
        console_state: params.console_state.as_ref(),
        startup_load_gate: &params.startup_load_gate,
        stop_rx,
        local_capacity: prepared.local_capacity,
        model_bytes: prepared.model_bytes,
        runtime_plan: prepared.runtime_plan,
        launch_kind: prepared.launch_kind,
    })
    .await
}

fn startup_loop_context<'a>(
    params: &'a StartupLocalModelTask,
    runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
    launch_kind: survey::SurveyLaunchKind,
) -> StartupLoopContext<'a> {
    StartupLoopContext {
        node: &params.node,
        config: &params.config,
        tunnel_mgr: &params.tunnel_mgr,
        target_tx: &params.target_tx,
        model_path: &params.model_path,
        model_ref: &params.model_ref,
        preindexed_split_package: params.preindexed_split_package.as_ref(),
        config_model_id: params.config_model_id.as_deref(),
        runtime_profile: &params.profile,
        readiness_index: params.readiness_index,
        instance_id: &params.instance_id,
        primary_model_name: &params.primary_model_name,
        mmproj_path: params.mmproj_path.as_ref(),
        ctx_size: params.ctx_size,
        pinned_gpu: params.pinned_gpu.as_ref(),
        device_override: params.device_override.as_deref(),
        runtime_capacity_ledger: &params.runtime_capacity_ledger,
        cache_type_k: params.cache_type_k.as_deref(),
        cache_type_v: params.cache_type_v.as_deref(),
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        flash_attention: params.flash_attention,
        parallel_override: params.parallel_override,
        local_source_required: params.local_source_required,
        resource_planning_profile: params.resource_planning_profile,
        openai_guardrail_policy: params.openai_guardrail_policy.clone(),
        skippy_telemetry: &params.skippy_telemetry,
        survey_telemetry: &params.survey_telemetry,
        launch_kind,
        dashboard_processes: &params.dashboard_processes,
        dashboard_context_usage: &params.dashboard_context_usage,
        runtime_instance_registry: &params.runtime_instance_registry,
        console_state: params.console_state.as_ref(),
        api_port: params.api_port,
        runtime_data_producer,
        lifecycle: &params.lifecycle,
        lifecycle_port: &params.lifecycle_port,
    }
}

async fn publish_startup_local_model(
    params: &StartupLocalModelTask,
    ctx: &StartupLoopContext<'_>,
    loaded_name: &str,
    handle: &LocalRuntimeModelHandle,
    load_duration: Duration,
) {
    startup_publish_loaded_runtime(
        ctx,
        loaded_name,
        handle,
        &params.startup_ready_reporter,
        load_duration,
    )
    .await;
    let response = api::RuntimeLoadResponse {
        model_ref: params.model_ref.clone(),
        model: loaded_name.to_string(),
        instance_id: params.instance_id.clone(),
        profile: params.profile.clone(),
        backend: Some(handle.backend.clone()),
        context_length: Some(handle.context_length),
    };
    let _ = params
        .runtime_event_tx
        .send(RuntimeEvent::StartupModelLoadFinished {
            model_ref: params.model_ref.clone(),
            profile: params.profile.clone(),
            result: Ok(response),
        });
    maybe_spawn_startup_interactive_handler(
        params.input_handler_enabled,
        loaded_name,
        &params.primary_model_name,
        &params.interactive_started,
        params.interactive_control_tx.clone(),
        params.interactive_console_state.clone(),
    );
}

async fn record_startup_task_failure(
    params: &StartupLocalModelTask,
    detail: &str,
    load_started: Instant,
) {
    record_startup_terminal_failure(
        &params.lifecycle,
        &params.startup_ready_reporter,
        &params.interactive_control_tx,
        &params.runtime_event_tx,
        &params.model_ref,
        &params.profile,
        &params.model_name,
        &params.instance_id,
        detail,
        load_started,
    )
    .await;
}

#[expect(
    clippy::too_many_arguments,
    reason = "terminal failure reporting explicitly updates lifecycle, reconciler, readiness, and fail-fast control sinks"
)]
async fn record_startup_terminal_failure(
    lifecycle: &Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
    reporter: &StartupReadyReporter,
    control_tx: &tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    runtime_event_tx: &tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    model_ref: &str,
    profile: &str,
    model_name: &str,
    instance_id: &str,
    detail: &str,
    load_started: Instant,
) {
    let mut record = lifecycle.lock().await;
    let transitioned_to_failure = if !record.state().is_terminal() {
        let _ = record.transition_to(InstanceLifecycleState::Failed);
        true
    } else {
        false
    };
    drop(record);

    if transitioned_to_failure {
        record_runtime_operational_event(RuntimeOperationalEvent::StartupFailed);
        record_runtime_operational_event_with_context(
            RuntimeOperationalEvent::ModelLoadFailed,
            runtime_model_audit_context(Some(model_name), instance_id)
                .outcome("failed")
                .duration_ms(u64::try_from(load_started.elapsed().as_millis()).unwrap_or(u64::MAX)),
        );
    }

    let _ = runtime_event_tx.send(RuntimeEvent::StartupModelLoadFinished {
        model_ref: model_ref.to_string(),
        profile: profile.to_string(),
        result: Err(detail.to_string()),
    });
    if reporter.record_terminal_failure(model_name, detail) {
        let _ = control_tx.send(api::RuntimeControlRequest::Shutdown {
            source: "startup-fail-fast",
        });
    }
}

/// Handles the outcome of one event-loop pass. Returns `true` when the model
/// task should loop back to the launch phase (split relaunch), `false` when
/// the task should end.
async fn startup_resolve_loop_outcome(
    outcome: StartupLoopOutcome,
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    coordinator_task: &mut Option<tokio::task::JoinHandle<()>>,
    stop_rx: &tokio::sync::watch::Receiver<bool>,
    model_name: &str,
) -> bool {
    match outcome {
        StartupLoopOutcome::Exit => false,
        StartupLoopOutcome::Shutdown => {
            startup_shutdown_local_model_loop(ctx, state, coordinator_task).await;
            false
        }
        StartupLoopOutcome::RelaunchSplit => {
            startup_shutdown_local_model_loop(ctx, state, coordinator_task).await;
            if *stop_rx.borrow() {
                return false;
            }
            let _ = emit_event(OutputEvent::Info {
                message: format!(
                    "Split model '{model_name}' withdrawn; waiting for eligible peers to relaunch"
                ),
                context: None,
            });
            true
        }
    }
}

pub(in crate::runtime) async fn startup_publish_loaded_runtime(
    ctx: &StartupLoopContext<'_>,
    loaded_name: &str,
    handle: &LocalRuntimeModelHandle,
    startup_ready_reporter: &StartupReadyReporter,
    load_duration: Duration,
) {
    let payload = startup_register_loaded_runtime(ctx, loaded_name, handle).await;
    ctx.node
        .claim_host_role(mesh::HostRoleClaim::LocalModel, ctx.api_port)
        .await;
    refresh_dashboard_context_usage(ctx.dashboard_context_usage, loaded_name, handle).await;
    publish_runtime_llama_slots(
        ctx.runtime_data_producer,
        loaded_name,
        Some(ctx.instance_id),
        handle,
    );
    if let Some(cs) = ctx.console_state {
        cs.upsert_local_process(payload).await;
        cs.update(true, true).await;
    }
    update_pi_models_json(loaded_name, ctx.api_port);
    startup_ready_reporter.mark_ready_and_maybe_emit(ctx.readiness_index);
    let _ = emit_event(OutputEvent::ModelReady {
        model: loaded_name.to_string(),
        internal_port: Some(handle.port),
        role: Some(handle.backend.clone()),
    });
    let _ = emit_event(OutputEvent::Info {
        message: format!("Startup-loaded model '{}' on :{}", loaded_name, handle.port),
        context: None,
    });
    record_runtime_operational_event_with_context(
        RuntimeOperationalEvent::ModelReady,
        runtime_model_audit_context(Some(loaded_name), ctx.instance_id)
            .outcome("ready")
            .duration_ms(u64::try_from(load_duration.as_millis()).unwrap_or(u64::MAX)),
    );
}

pub(in crate::runtime) async fn startup_run_local_model_event_loop(
    ctx: &StartupLoopContext<'_>,
    state: &mut StartupLoopState,
    event_ctx: StartupLoopEventContext<'_>,
) -> StartupLoopOutcome {
    let StartupLoopEventContext {
        context_usage_tick,
        stop_rx,
        local_capacity,
        model_bytes,
    } = event_ctx;
    loop {
        tokio::select! {
            _ = context_usage_tick.tick() => {
                if let Some(handle) = state.handle.as_ref() {
                    refresh_dashboard_context_usage(ctx.dashboard_context_usage, &state.loaded_name, handle).await;
                    publish_runtime_llama_slots(ctx.runtime_data_producer, &state.loaded_name, Some(ctx.instance_id), handle);
                }
            }
            _ = &mut state.death_rx => {
                state.survey_exited_unexpectedly = true;
                ctx.survey_telemetry.record_unexpected_exit(&state.survey_loaded_model);
                record_runtime_operational_event_with_context(
                    RuntimeOperationalEvent::ModelExited,
                    runtime_model_audit_context(Some(&state.loaded_name), ctx.instance_id)
                        .reason_code("runtime_process_exited")
                        .outcome("failed"),
                );
                let port = state.handle.as_ref().map(|handle| handle.port).unwrap_or_default();
                let _ = emit_event(OutputEvent::Warning {
                    message: format!("Startup model '{}' exited unexpectedly", state.loaded_name),
                    context: Some(format!("model={} port={port}", state.loaded_name)),
                });
                return StartupLoopOutcome::Shutdown;
            }
            event = async {
                if let Some(rx) = state.split_event_rx.as_mut() {
                    rx.recv().await
                } else {
                    std::future::pending().await
                }
            } => {
                let Some(event) = event else {
                    state.split_event_rx = None;
                    continue;
                };
                match startup_handle_split_event(ctx, state, event, local_capacity, model_bytes).await {
                    StartupLoopControl::Continue => continue,
                    StartupLoopControl::Break => return StartupLoopOutcome::Shutdown,
                    StartupLoopControl::Return => return StartupLoopOutcome::Exit,
                    StartupLoopControl::RelaunchSplit => return StartupLoopOutcome::RelaunchSplit,
                }
            }
            res = stop_rx.changed() => {
                let _ = res;
                return StartupLoopOutcome::Shutdown;
            }
        }
    }
}
