/// Wait for either SIGINT (ctrl-c) or SIGTERM. Without this, an unhandled
/// SIGTERM aborts the process before runtime cleanup can run.
use super::{
    DASHBOARD_CONTEXT_USAGE_REFRESH_INTERVAL, IntentSource, MODEL_TARGET_RECONCILIATION_INTERVAL,
    ModelIntent, ModelTargetReconciliationState, OpenAiGuardrailPolicyHandle,
    RunAutoRuntimeLifecycleContext, RunAutoRuntimeLoopContext, RunAutoShutdownContext,
    RunAutoStartupTasksContext, RuntimeEvent, RuntimeOperationalEvent,
    ShutdownRuntimeLoadedModelsContext, UnloadTarget, advertise_run_auto_models,
    apply_startup_model_load_finished, cleanup_run_auto_runtime_dir, current_time_secs,
    dashboard_context_usage_source, emit_shutdown, model_target_reconciliation_policy,
    normalize_runtime_model_request_for_config, publish_runtime_llama_slots,
    record_runtime_operational_event_with_context, refresh_dashboard_context_usage_batch,
    resolve_eager_startup_models, resolve_runtime_unload_target,
    run_auto_handle_model_target_reconciliation_result, run_auto_handle_runtime_exit,
    run_auto_load_runtime_model, run_auto_model_identity, run_auto_reconcile_model_targets,
    run_auto_record_model_target_manual_unload, run_auto_unload_runtime_model,
    runtime_unload_candidates, set_openai_guardrail_policy_mode, shutdown_run_auto_services,
    shutdown_runtime_loaded_models, shutdown_runtime_managed_models,
    spawn_run_auto_startup_model_tasks, startup_default_backend_device, startup_launch_plan,
    suppress_desired_for_resolved_unload_candidate, unpublish_run_auto_nostr_listing,
};
use crate::api;
use crate::inference::skippy;
use anyhow::Result;
use mesh_llm_events::{OutputEvent, emit_event, flush_output};
use std::time::Instant;

pub(super) async fn wait_shutdown_signal() -> &'static str {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return "SIGINT";
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => "SIGINT",
            _ = term.recv() => "SIGTERM",
        }
    }
    #[cfg(windows)]
    {
        use tokio::signal::windows::ctrl_break;

        // CTRL_BREAK_EVENT is distinct from CTRL_C_EVENT on Windows. The CI
        // readiness smoke starts MeshLLM in a dedicated process group and
        // uses CTRL_BREAK_EVENT to request graceful shutdown.
        let mut ctrl_break = match ctrl_break() {
            Ok(signal) => signal,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return "CTRL-C";
            }
        };

        tokio::select! {
            _ = tokio::signal::ctrl_c() => "CTRL-C",
            _ = ctrl_break.recv() => "CTRL-BREAK",
        }
    }
    #[cfg(all(not(unix), not(windows)))]
    {
        let _ = tokio::signal::ctrl_c().await;
        "CTRL-C"
    }
}

async fn drain_pending_startup_commands(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    control_rx: &mut tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>,
    model_intent_rx: &mut tokio::sync::mpsc::Receiver<ModelIntent>,
) -> bool {
    while let Ok(intent) = model_intent_rx.try_recv() {
        run_auto_handle_model_intent(ctx, intent).await;
    }
    while let Ok(command) = control_rx.try_recv() {
        if run_auto_handle_control_request(ctx, command).await {
            return true;
        }
    }
    false
}

#[expect(
    clippy::cognitive_complexity,
    reason = "startup reconciliation intentionally keeps surface readiness, intent precedence, resolution, and launch ordering visible"
)]
#[expect(
    clippy::too_many_lines,
    reason = "startup reconciliation keeps the daemon-before-model ordering and shutdown ownership in one auditable orchestration boundary"
)]
pub(super) async fn run_auto_runtime_loop_and_shutdown(ctx: RunAutoRuntimeLifecycleContext<'_>) {
    let RunAutoRuntimeLifecycleContext {
        options,
        config,
        node,
        primary_model_name,
        target_tx,
        control_rx,
        control_tx,
        runtime_event_rx,
        model_intent_rx,
        runtime_state,
        console_state,
        runtime_data_producer,
        runtime_event_tx,
        survey_telemetry,
        startup_ready_reporter,
        plugin_manager,
        api_proxy_handle,
        console_server_handle,
        discovery_publisher,
        presentation_subscriber,
        runtime_event_driver,
        startup_specs,
        tunnel_mgr,
        skippy_telemetry,
        api_port,
        console_port,
        interactive_started,
        lan_bootstrap_tasks,
        runtime,
    } = ctx;
    let input_handler_enabled = runtime_state.input_handler_enabled;
    let mut loop_ctx = RunAutoRuntimeLoopContext {
        options,
        config,
        node,
        primary_model_name,
        target_tx,
        control_tx,
        runtime_models: &mut runtime_state.runtime_models,
        runtime_survey_models: &mut runtime_state.runtime_survey_models,
        managed_models: &mut runtime_state.managed_models,
        runtime_capacity_ledger: &runtime_state.runtime_capacity_ledger,
        next_runtime_instance_sequence: &mut runtime_state.next_runtime_instance_sequence,
        runtime_instance_registry: &runtime_state.runtime_instance_registry,
        dashboard_processes: &runtime_state.dashboard_processes,
        dashboard_context_usage: &runtime_state.dashboard_context_usage,
        console_state,
        runtime_data_producer,
        runtime_event_tx,
        survey_telemetry,
        startup_ready_reporter,
        openai_guardrail_policy: &runtime_state.openai_guardrail_policy,
        model_target_reconciliation_policy: model_target_reconciliation_policy(config),
        model_target_reconciliation_state: ModelTargetReconciliationState::with_shared_history(
            node.runtime_intents.clone(),
        ),
    };

    // Seed startup config as desired state before any resolution I/O.
    let mut startup_intent_ids = Vec::with_capacity(startup_specs.len());
    for spec in startup_specs {
        let model_ref = spec.model_ref.to_string_lossy().into_owned();
        let intent_id = loop_ctx.model_target_reconciliation_state.add_desired(
            &model_ref,
            &spec.profile,
            IntentSource::StartupConfig,
        );
        startup_intent_ids.push((intent_id, model_ref.clone(), spec.profile.clone()));
        tracing::debug!(
            model = %model_ref,
            profile = %spec.profile,
            "seeded startup config model as desired intent"
        );
    }

    // Commands may arrive while eager models are resolving. Apply those
    // higher-priority session intents before launching startup work so an
    // unload/drain accepted during resolution can suppress the startup load.
    let mut shutdown_requested =
        drain_pending_startup_commands(&mut loop_ctx, control_rx, model_intent_rx).await;

    let eligible_startup_specs = startup_specs
        .iter()
        .filter(|spec| {
            let model_ref = spec.model_ref.to_string_lossy();
            loop_ctx
                .model_target_reconciliation_state
                .is_desired(&model_ref, &spec.profile)
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut eligible_startup_models = Vec::new();
    if !shutdown_requested {
        for spec in &eligible_startup_specs {
            match resolve_eager_startup_models(options, config, std::slice::from_ref(spec)).await {
                Ok(mut models) => {
                    let model_ref = spec.model_ref.to_string_lossy();
                    if let Some((intent_id, _, _)) =
                        startup_intent_ids.iter().find(|(_, candidate, profile)| {
                            candidate == &model_ref && profile == &spec.profile
                        })
                    {
                        for model in &models {
                            loop_ctx
                                .model_target_reconciliation_state
                                .retarget_intent_model_ref(intent_id, &model.declared_ref);
                        }
                    }
                    eligible_startup_models.append(&mut models);
                }
                Err(error) => {
                    let detail = format!("{error:#}");
                    let model_ref = spec.model_ref.to_string_lossy().into_owned();
                    if let Some((intent_id, _, _)) =
                        startup_intent_ids.iter().find(|(_, candidate, profile)| {
                            candidate == &model_ref && profile == &spec.profile
                        })
                    {
                        loop_ctx
                            .model_target_reconciliation_state
                            .set_intent_error(intent_id, detail.clone());
                    }
                    if startup_ready_reporter.record_terminal_failure(&model_ref, &detail) {
                        shutdown_requested = true;
                    }
                    let _ = emit_event(OutputEvent::Warning {
                        message: format!(
                            "Eager startup model '{model_ref}' failed to resolve after daemon surfaces became ready"
                        ),
                        context: Some(detail),
                    });
                }
            }
        }
    }

    // Re-check accepted commands after resolution and immediately before task
    // creation. This closes the launch-boundary race for unload/drain intents
    // that arrived while catalog or filesystem resolution was in progress.
    shutdown_requested |=
        drain_pending_startup_commands(&mut loop_ctx, control_rx, model_intent_rx).await;
    eligible_startup_models.retain(|model| {
        loop_ctx
            .model_target_reconciliation_state
            .is_desired(&model.declared_ref, &model.profile)
    });

    if !shutdown_requested && let Some(primary) = eligible_startup_models.first() {
        for model in &eligible_startup_models {
            loop_ctx
                .model_target_reconciliation_state
                .mark_load_started(&model.declared_ref, &model.profile);
        }
        let (eligible_primary_name, model_source) =
            run_auto_model_identity(Some(primary), &primary.resolved_path);
        advertise_run_auto_models(
            node,
            &eligible_startup_models,
            &eligible_primary_name,
            model_source,
        )
        .await;
        let _ = emit_event(OutputEvent::LaunchPlan {
            plan: startup_launch_plan(
                &eligible_startup_models,
                &eligible_primary_name,
                api_port,
                console_port,
                options.headless,
                config.gpu.parallel,
                startup_default_backend_device(options.llama_flavor),
            ),
        });
        spawn_run_auto_startup_model_tasks(RunAutoStartupTasksContext {
            options,
            config,
            node,
            tunnel_mgr,
            startup_models: &eligible_startup_models,
            primary_startup_model: Some(primary),
            model_name: &eligible_primary_name,
            model_path: &primary.resolved_path,
            startup_ready_reporter,
            target_tx,
            managed_models: loop_ctx.managed_models,
            next_runtime_instance_sequence: loop_ctx.next_runtime_instance_sequence,
            runtime_capacity_ledger: loop_ctx.runtime_capacity_ledger,
            runtime_instance_registry: loop_ctx.runtime_instance_registry,
            dashboard_processes: loop_ctx.dashboard_processes,
            dashboard_context_usage: loop_ctx.dashboard_context_usage,
            input_handler_enabled,
            openai_guardrail_policy: loop_ctx.openai_guardrail_policy,
            console_state,
            control_tx,
            runtime_event_tx,
            survey_telemetry,
            skippy_telemetry,
            api_port,
            interactive_started,
        })
        .await;
    }

    if !shutdown_requested {
        run_auto_runtime_event_loop(&mut loop_ctx, control_rx, runtime_event_rx, model_intent_rx)
            .await;
    }

    // This audit must precede the cleanup-worker stop and service drain below,
    // so normal shutdown retains the same durable boundary as other lifecycle
    // records without delaying teardown.
    let shutdown_started = Instant::now();
    record_runtime_operational_event_with_context(
        RuntimeOperationalEvent::ShutdownStarted,
        crate::logging::OperationalAuditContext::new()
            .subject(crate::logging::OperationalAuditSubjectKind::Runtime, "host")
            .outcome("started"),
    );

    // Stop scheduled cleanup before draining persistence so the scheduler
    // cannot enqueue a late audit after the durable delivery boundary closes.
    if let Some(logging_runtime) = crate::logging_runtime_state() {
        logging_runtime.shutdown_cleanup_worker().await;
    }

    // Stop terminal webhook dispatch before closing the persistence hand-off.
    // Its bounded scheduler leaves unfinished durable rows restart-reclaimable.
    if let Some(logging_runtime) = crate::logging_runtime_state() {
        logging_runtime.shutdown_webhook_delivery_worker().await;
    }

    shutdown_run_auto_runtime(RunAutoShutdownContext {
        options,
        node,
        plugin_manager,
        api_proxy_handle,
        console_server_handle,
        discovery_publisher,
        presentation_subscriber,
        runtime_event_driver,
        lan_bootstrap_tasks,
        runtime_models: &mut runtime_state.runtime_models,
        runtime_survey_models: &mut runtime_state.runtime_survey_models,
        managed_models: &mut runtime_state.managed_models,
        survey_telemetry,
        dashboard_processes: &runtime_state.dashboard_processes,
        console_state,
        target_tx,
        runtime_instance_registry: &runtime_state.runtime_instance_registry,
        runtime_data_producer,
        dashboard_context_usage: &runtime_state.dashboard_context_usage,
        runtime,
    })
    .await;

    // The Rust control loop now owns an authoritative completed boundary: all
    // runtime services and model processes above have finished their shutdown
    // work. Emit it before draining the logging worker so both start and
    // completion survive normal process exit.
    record_runtime_operational_event_with_context(
        RuntimeOperationalEvent::ShutdownCompleted,
        crate::logging::OperationalAuditContext::new()
            .subject(crate::logging::OperationalAuditSubjectKind::Runtime, "host")
            .outcome("completed")
            .duration_ms(u64::try_from(shutdown_started.elapsed().as_millis()).unwrap_or(u64::MAX)),
    );

    // Drain and join only after the final authoritative runtime boundary.
    if let Some(logging_service) = runtime_state.logging_service.take() {
        let _ = logging_service.shutdown().await;
    }
}

pub(super) async fn shutdown_run_auto_runtime(ctx: RunAutoShutdownContext<'_>) {
    let RunAutoShutdownContext {
        options,
        node,
        plugin_manager,
        api_proxy_handle,
        console_server_handle,
        discovery_publisher,
        presentation_subscriber,
        runtime_event_driver,
        lan_bootstrap_tasks,
        runtime_models,
        runtime_survey_models,
        managed_models,
        survey_telemetry,
        dashboard_processes,
        console_state,
        target_tx,
        runtime_instance_registry,
        runtime_data_producer,
        dashboard_context_usage,
        runtime,
    } = ctx;
    super::node_lifecycle_events::emit_node_draining();
    node.broadcast_leaving().await;

    unpublish_run_auto_nostr_listing(options).await;
    if let Some(handle) = discovery_publisher {
        handle.abort();
    }
    // Stop the relay-less LAN bootstrap loops (mDNS publisher, reverse-dial,
    // and beacon) so they release their sockets and stop dialing on shutdown.
    lan_bootstrap_tasks.abort();
    if let Some(handle) = presentation_subscriber {
        handle.abort();
        let _ = handle.await;
    }

    shutdown_run_auto_services(
        node,
        plugin_manager,
        api_proxy_handle,
        console_server_handle,
    )
    .await;

    shutdown_runtime_loaded_models(
        runtime_models,
        runtime_survey_models,
        ShutdownRuntimeLoadedModelsContext {
            survey_telemetry,
            dashboard_processes,
            console_state,
            target_tx,
            runtime_instance_registry,
            node,
            runtime_data_producer,
            dashboard_context_usage,
        },
    )
    .await;
    shutdown_runtime_managed_models(managed_models).await;

    node.set_serving_models(Vec::new()).await;
    node.set_hosted_models(Vec::new()).await;

    // Last mesh-facing step: every subsystem that speaks over the mesh endpoint
    // has now stopped, so close it explicitly. Dropping it instead makes iroh
    // log `Endpoint dropped without calling `Endpoint::close`. Aborting
    // ungracefully.` and abandon the connection-close frames, leaving peers to
    // time this node out rather than seeing it depart.
    node.close_endpoint().await;

    cleanup_run_auto_runtime_dir(runtime);
    super::node_lifecycle_events::emit_node_stopped();
    // Task 3: finalize AFTER node_stopped so the driver's own final drain
    // is what applies and publishes both node_draining and node_stopped to
    // any attached v1 stream -- aborting the driver any earlier would leave
    // node_stopped sitting in the wake list with nothing left to drain it.
    crate::runtime_events::driver::finalize_engine_driver(runtime_event_driver).await;
}

pub(super) async fn run_auto_handle_control_request(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    cmd: api::RuntimeControlRequest,
) -> bool {
    use super::{IntentSource, ModelIntent};

    match cmd {
        api::RuntimeControlRequest::Join { invite_token, resp } => {
            let result = ctx.node.join_with_retry(&invite_token).await;
            let _ = resp.send(result);
            false
        }
        api::RuntimeControlRequest::Load {
            spec,
            config_model_id,
            profile,
            resp,
        } => {
            let intent = ModelIntent::Load {
                intent_id: None,
                spec,
                config_model_id,
                profile,
                source: IntentSource::ApiLoad,
                completion: Some(resp),
            };
            run_auto_handle_model_intent(ctx, intent).await;
            false
        }
        api::RuntimeControlRequest::Unload {
            target,
            options,
            resp,
        } => {
            let intent = ModelIntent::Unload {
                intent_id: None,
                canonical_model_ref: None,
                target,
                options,
                source: IntentSource::ApiUnload,
                completion: Some(resp),
            };
            run_auto_handle_model_intent(ctx, intent).await;
            false
        }
        api::RuntimeControlRequest::SetOpenAiGuardrailMode { mode, resp } => {
            let result = run_auto_set_openai_guardrail_mode(ctx, mode).await;
            let _ = resp.send(result);
            false
        }
        api::RuntimeControlRequest::Shutdown { source } => {
            let _ = emit_event(OutputEvent::ShutdownRequested { signal: source });
            ctx.startup_ready_reporter.mark_shutdown_requested();
            let _ = flush_output().await;
            emit_shutdown(None).await;
            true
        }
    }
}

pub(super) async fn run_auto_handle_model_intent(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    intent: ModelIntent,
) {
    use super::ModelIntent;

    match intent {
        ModelIntent::Load {
            intent_id,
            spec,
            config_model_id,
            profile,
            source,
            completion,
        } => {
            if ctx.options.client {
                if let Some(tx) = completion {
                    let _ = tx.send(Err(anyhow::anyhow!(
                        "runtime mode client does not allow local model loading"
                    )));
                }
                return;
            }
            let (spec, profile) = match normalize_runtime_model_request_for_config(
                ctx.config,
                config_model_id.as_deref(),
                spec,
                profile,
            ) {
                Ok(normalized) => normalized,
                Err(error) => {
                    if let Some(tx) = completion {
                        let _ = tx.send(Err(error));
                    }
                    return;
                }
            };
            if ctx
                .model_target_reconciliation_state
                .is_load_pending(&spec, &profile)
            {
                if let Some(tx) = completion {
                    ctx.model_target_reconciliation_state
                        .stack_load_completion(&spec, &profile, tx);
                }
                return;
            }

            let intent_id = ctx
                .model_target_reconciliation_state
                .add_desired_with_id(&spec, &profile, source, intent_id);
            if !ctx
                .model_target_reconciliation_state
                .is_effective_intent(&intent_id, &spec, &profile)
            {
                ctx.model_target_reconciliation_state
                    .retire_one_shot_present(&intent_id);
                if let Some(tx) = completion {
                    let _ = tx.send(Err(anyhow::anyhow!(
                        "model load intent is suppressed by a higher-priority desired state"
                    )));
                }
                return;
            }
            if let Some(tx) = completion {
                ctx.model_target_reconciliation_state
                    .stack_load_completion(&spec, &profile, tx);
            }

            let result =
                run_auto_load_runtime_model(ctx, spec.clone(), config_model_id, profile.clone())
                    .await;
            match &result {
                Ok(response) => {
                    ctx.model_target_reconciliation_state
                        .record_load_success(&spec, &profile);
                    ctx.model_target_reconciliation_state.notify_load_success(
                        &spec,
                        &profile,
                        response.clone(),
                    );
                    ctx.model_target_reconciliation_state
                        .retire_one_shot_present(&intent_id);
                }
                Err(e) => {
                    ctx.model_target_reconciliation_state
                        .set_intent_error(&intent_id, e.to_string());
                    ctx.model_target_reconciliation_state
                        .retire_one_shot_present(&intent_id);
                    ctx.model_target_reconciliation_state
                        .notify_load_failure(&spec, &profile, e);
                }
            }
        }
        intent @ ModelIntent::Unload { .. } => {
            run_auto_handle_unload_intent(ctx, intent).await;
        }
    }
}

async fn run_auto_handle_unload_intent(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    intent: ModelIntent,
) {
    let ModelIntent::Unload {
        intent_id,
        canonical_model_ref,
        target,
        options,
        source,
        completion,
    } = intent
    else {
        return;
    };
    let unload_key = match &target {
        UnloadTarget::Model(m) => m.clone(),
        UnloadTarget::Instance(i) => i.clone(),
    };

    if ctx
        .model_target_reconciliation_state
        .is_unload_pending(&unload_key)
    {
        if let Some(tx) = completion {
            ctx.model_target_reconciliation_state
                .stack_unload_completion(&unload_key, tx);
        }
        return;
    }

    let resolved_unload = resolve_runtime_unload_target(
        target.as_runtime_target(),
        runtime_unload_candidates(ctx.runtime_models, ctx.managed_models),
    )
    .ok();
    let intent_id = if let Some(candidate) = &resolved_unload {
        suppress_desired_for_resolved_unload_candidate(
            &mut ctx.model_target_reconciliation_state,
            candidate,
            source,
            matches!(source, IntentSource::OwnerDrain),
            intent_id,
        )
    } else {
        let (intent_model, instance_target) = match &target {
            UnloadTarget::Model(model) => {
                (canonical_model_ref.unwrap_or_else(|| model.clone()), None)
            }
            UnloadTarget::Instance(instance) => (
                canonical_model_ref.unwrap_or_else(|| instance.clone()),
                Some(instance.clone()),
            ),
        };
        let profile = ctx
            .model_target_reconciliation_state
            .desired_profile(&intent_model)
            .unwrap_or_default()
            .to_string();
        ctx.model_target_reconciliation_state
            .suppress_desired_with_id(
                &intent_model,
                &profile,
                instance_target,
                source,
                matches!(source, IntentSource::OwnerDrain),
                intent_id,
            )
    };

    if let Some(tx) = completion {
        ctx.model_target_reconciliation_state
            .stack_unload_completion(&unload_key, tx);
    }

    let result = run_auto_unload_runtime_model(ctx, target.clone(), options).await;
    run_auto_record_model_target_manual_unload(
        ctx,
        resolved_unload.as_ref(),
        target.as_runtime_target(),
        &result,
    );
    match &result {
        Ok(response) => {
            if matches!(source, IntentSource::OwnerDrain) {
                ctx.model_target_reconciliation_state
                    .transition_drain_to_absent(
                        &intent_id,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    );
            }
            ctx.model_target_reconciliation_state
                .notify_unload_success(&unload_key, response.clone());
        }
        Err(e) => {
            ctx.model_target_reconciliation_state
                .set_intent_error(&intent_id, e.to_string());
            ctx.model_target_reconciliation_state
                .notify_unload_failure(&unload_key, e);
        }
    }
}

pub(super) async fn run_auto_set_openai_guardrail_mode(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    mode: openai_frontend::GuardrailMode,
) -> Result<api::OpenAiGuardrailModeUpdateResponse> {
    set_openai_guardrail_policy_mode(ctx.openai_guardrail_policy, mode);
    let mut updated_models = 0_usize;
    let mut latest_status = None;
    for entry in ctx.runtime_models.values() {
        if let Some(status) = entry.handle.set_openai_guardrail_mode(mode) {
            updated_models += 1;
            latest_status = Some(status);
        }
    }

    let status_payload = Some(
        latest_status
            .map(api::status::OpenAiGuardrailsPayload::from)
            .unwrap_or_else(|| openai_guardrails_payload_from_policy(ctx.openai_guardrail_policy)),
    );
    if let Some(console_state) = ctx.console_state {
        console_state
            .set_openai_guardrails(status_payload.clone())
            .await;
    }

    Ok(api::OpenAiGuardrailModeUpdateResponse {
        mode: guardrail_mode_status_label(mode),
        updated_models,
        status: status_payload,
    })
}

pub(super) fn guardrail_mode_status_label(mode: openai_frontend::GuardrailMode) -> &'static str {
    match mode {
        openai_frontend::GuardrailMode::Disabled => "disabled",
        openai_frontend::GuardrailMode::MetricsOnly => "metrics",
        openai_frontend::GuardrailMode::Enforce => "enforce",
    }
}

pub(super) fn openai_guardrails_payload_from_policy(
    policy: &OpenAiGuardrailPolicyHandle,
) -> api::status::OpenAiGuardrailsPayload {
    api::status::OpenAiGuardrailsPayload::from(
        skippy::skippy_openai_guardrails_for_policy_handle(policy.clone()).status(),
    )
}

pub(super) async fn publish_initial_openai_guardrails_status(
    console_state: Option<&api::MeshApi>,
    policy: &OpenAiGuardrailPolicyHandle,
) {
    let Some(console_state) = console_state else {
        return;
    };
    console_state
        .set_openai_guardrails(Some(openai_guardrails_payload_from_policy(policy)))
        .await;
}

pub(super) async fn run_auto_runtime_event_loop(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    control_rx: &mut tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>,
    runtime_event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
    model_intent_rx: &mut tokio::sync::mpsc::Receiver<ModelIntent>,
) {
    let mut dashboard_context_usage_tick =
        tokio::time::interval(DASHBOARD_CONTEXT_USAGE_REFRESH_INTERVAL);
    dashboard_context_usage_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut model_target_reconciliation_tick =
        tokio::time::interval(MODEL_TARGET_RECONCILIATION_INTERVAL);
    model_target_reconciliation_tick
        .set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = dashboard_context_usage_tick.tick() => {
                let updates = ctx.runtime_models
                    .iter()
                    .map(|(instance_id, entry)| {
                        publish_runtime_llama_slots(
                            ctx.runtime_data_producer,
                            &entry.model_name,
                            Some(instance_id.as_str()),
                            &entry.handle,
                        );
                        (
                            entry.model_name.clone(),
                            dashboard_context_usage_source(&entry.handle),
                            entry.handle.ctx_used_tokens(),
                        )
                    })
                    .collect();
                refresh_dashboard_context_usage_batch(ctx.dashboard_context_usage, updates).await;
            }
            _ = model_target_reconciliation_tick.tick() => {
                run_auto_reconcile_model_targets(ctx).await;
            }
            signal = wait_shutdown_signal() => {
                let _ = emit_event(OutputEvent::ShutdownRequested { signal });
                ctx.startup_ready_reporter.mark_shutdown_requested();
                let _ = flush_output().await;
                emit_shutdown(None).await;
                break;
            }
            Some(cmd) = control_rx.recv() => {
                if run_auto_handle_control_request(ctx, cmd).await {
                    break;
                }
            }
            Some(event) = runtime_event_rx.recv() => {
                match event {
                    RuntimeEvent::ModelTargetReconciliationLoadFinished {
                        model_ref,
                        profile,
                        result,
                    } => {
                        run_auto_handle_model_target_reconciliation_result(
                            ctx,
                            model_ref,
                            profile,
                            result,
                        );
                    }
                    RuntimeEvent::StartupModelLoadFinished {
                        model_ref,
                        profile,
                        result,
                    } => {
                        apply_startup_model_load_finished(
                            &mut ctx.model_target_reconciliation_state,
                            &ctx.model_target_reconciliation_policy,
                            &model_ref,
                            &profile,
                            result,
                            current_time_secs(),
                        );
                    }
                    RuntimeEvent::Exited { instance_id, model, port } => {
                        run_auto_handle_runtime_exit(ctx, instance_id, model, port).await;
                    }
                }
            }
            Some(intent) = model_intent_rx.recv() => {
                run_auto_handle_model_intent(ctx, intent).await;
            }
        }
    }
}

pub(super) fn spawn_embedded_runtime_control_forwarder(
    embedded_control_rx: Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
    control_tx: tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
) {
    let Some(mut embedded_control_rx) = embedded_control_rx else {
        return;
    };
    tokio::spawn(async move {
        while let Some(command) = embedded_control_rx.recv().await {
            if control_tx.send(command).is_err() {
                break;
            }
        }
    });
}
