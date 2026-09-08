use super::*;
use crate::runtime::{DrainCoordinator, DrainResult};

pub(super) fn unregister_local_source_policy_if_unused(
    ctx: &RunAutoRuntimeLoopContext<'_>,
    model: &str,
    profile: &str,
) {
    let runtime_profiles = ctx
        .runtime_models
        .values()
        .map(|entry| (entry.model_name.as_str(), entry.profile.as_str()));
    let managed_profiles = ctx
        .managed_models
        .values()
        .map(|controller| (controller.model_name.as_str(), controller.profile.as_str()));
    unregister_local_source_policy_after_lifecycle_removal(
        runtime_profiles.chain(managed_profiles),
        model,
        profile,
    );
}

fn unregister_local_source_policy_after_lifecycle_removal<'a>(
    mut active_profiles: impl Iterator<Item = (&'a str, &'a str)>,
    model: &str,
    profile: &str,
) {
    if !active_profiles.any(|active| active == (model, profile)) {
        crate::inference::skippy::unregister_local_source_policy(model, profile);
    }
}

/// Unload a runtime model (entry point for both runtime and managed models).
#[expect(
    clippy::cognitive_complexity,
    reason = "unload dispatch keeps runtime and managed ownership cleanup symmetric and auditable"
)]
pub(crate) async fn run_auto_unload_runtime_model(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    target: UnloadTarget,
    options: UnloadOptions,
) -> Result<api::RuntimeUnloadResponse> {
    let unload_started = Instant::now();
    let unload = match resolve_runtime_unload_target(
        target.as_runtime_target(),
        runtime_unload_candidates(ctx.runtime_models, ctx.managed_models),
    ) {
        Ok(unload) => unload,
        Err(error) => {
            record_runtime_operational_event_with_context(
                RuntimeOperationalEvent::ModelUnloadFailed,
                OperationalAuditContext::new()
                    .reason_code("target_not_resolved")
                    .outcome("failed")
                    .duration_ms(
                        u64::try_from(unload_started.elapsed().as_millis()).unwrap_or(u64::MAX),
                    ),
            );
            return Err(error);
        }
    };
    record_runtime_operational_event_with_context(
        RuntimeOperationalEvent::ModelUnloadStarted,
        runtime_model_audit_context(Some(&unload.model_name), &unload.instance_id)
            .outcome("started"),
    );
    // Reservation acquired before any unload work begins, per plan task 9.
    let unload_op = UnloadOperation::begin(&unload.model_name);
    let drain_delay = if options.force {
        unload_op.forced(&unload.model_name);
        Duration::ZERO
    } else {
        options.drain_timeout
    };
    match unload.owner {
        RuntimeUnloadOwner::Runtime => {
            run_auto_unload_runtime_entry(ctx, unload, drain_delay, unload_started, unload_op).await
        }
        RuntimeUnloadOwner::Managed => {
            let Some(controller) = ctx.managed_models.remove(&unload.instance_id) else {
                record_runtime_operational_event_with_context(
                    RuntimeOperationalEvent::ModelUnloadFailed,
                    runtime_model_audit_context(Some(&unload.model_name), &unload.instance_id)
                        .reason_code("instance_missing")
                        .outcome("failed")
                        .duration_ms(
                            u64::try_from(unload_started.elapsed().as_millis()).unwrap_or(u64::MAX),
                        ),
                );
                unload_op.failed(&unload.model_name);
                anyhow::bail!(
                    "model or runtime instance '{}' is not loaded",
                    unload.instance_id
                );
            };
            let ManagedModelController {
                model_name: model,
                profile,
                stop_tx,
                task,
                lifecycle,
                port,
                ..
            } = controller;
            let active_port = port.load(std::sync::atomic::Ordering::Acquire);
            if active_port != 0 {
                let draining = {
                    let mut record = lifecycle.lock().await;
                    match record.mark_draining(Instant::now() + drain_delay) {
                        Ok(()) => true,
                        Err(error) => {
                            tracing::warn!(
                                model,
                                instance_id = unload.instance_id,
                                %error,
                                "managed instance could not enter draining; forcing unload"
                            );
                            false
                        }
                    }
                };
                if draining {
                    unload_op.session_draining_started(&model);
                    let result = DrainCoordinator::default()
                        .wait_for_unload_ready(&lifecycle)
                        .await;
                    unload_op.session_draining_completed(&model);
                    if result == DrainResult::ForceCancelled {
                        unload_op.forced(&model);
                        tracing::warn!(
                            model,
                            instance_id = unload.instance_id,
                            "managed instance drain deadline expired; forcing unload"
                        );
                    }
                    if let Err(error) = lifecycle.lock().await.transition_to_unloading() {
                        tracing::warn!(
                            model,
                            instance_id = unload.instance_id,
                            %error,
                            "managed instance could not enter unloading; forcing unload"
                        );
                    }
                }
            }
            let _ = stop_tx.send(true);
            await_managed_model_stop(task, drain_delay, options.force, &model).await;
            ctx.node.unregister_runtime_instance_lifecycle(active_port);
            let process_already_crashed = {
                let mut record = lifecycle.lock().await;
                let crashed = record.state() == InstanceLifecycleState::Failed;
                if record.state() == InstanceLifecycleState::Unloading
                    && let Err(error) = record.transition_to(InstanceLifecycleState::Stopped)
                {
                    tracing::warn!(
                        model,
                        instance_id = unload.instance_id,
                        %error,
                        "managed instance stopped with a stale lifecycle state"
                    );
                }
                crashed
            };
            if !runtime_registry_has_model(ctx.runtime_instance_registry, &model).await {
                publish_runtime_llama_unavailable(
                    ctx.runtime_data_producer,
                    &model,
                    Some(&unload.instance_id),
                );
                withdraw_advertised_model(ctx.node, &model, "").await;
                set_advertised_model_context(ctx.node, &model, None).await;
                remove_serving_assignment(ctx.node, &model).await;
            }
            remove_dashboard_process(ctx.dashboard_processes, &unload.instance_id).await;
            if let Some(cs) = ctx.console_state {
                cs.remove_local_process(&unload.instance_id).await;
            }
            unregister_local_source_policy_if_unused(ctx, &model, &profile);
            let _ = emit_event(OutputEvent::Info {
                message: format!("Unloaded managed model '{}'", model),
                context: None,
            });
            record_runtime_operational_event_with_context(
                RuntimeOperationalEvent::ModelUnloaded,
                runtime_model_audit_context(Some(&model), &unload.instance_id)
                    .outcome("completed")
                    .duration_ms(
                        u64::try_from(unload_started.elapsed().as_millis()).unwrap_or(u64::MAX),
                    ),
            );
            unload_op.reconcile(&model, process_already_crashed);
            Ok(api::RuntimeUnloadResponse {
                model,
                instance_id: unload.instance_id,
                unloaded: true,
            })
        }
    }
}

/// Await a managed model's task to stop, with optional timeout and force.
pub(crate) async fn await_managed_model_stop(
    mut task: tokio::task::JoinHandle<()>,
    drain_timeout: Duration,
    force: bool,
    model: &str,
) {
    if force {
        task.abort();
        let _ = task.await;
        return;
    }

    match tokio::time::timeout(drain_timeout, &mut task).await {
        Ok(join_result) => {
            let _ = join_result;
        }
        Err(_) => {
            tracing::warn!(
                model,
                drain_timeout_ms = drain_timeout.as_millis(),
                "managed model task did not stop within unload drain timeout; aborting"
            );
            task.abort();
            let _ = task.await;
        }
    }
}

/// Unload a runtime-managed model entry (not externally managed).
#[expect(
    clippy::cognitive_complexity,
    reason = "runtime entry unload keeps drain, registry, telemetry, and capacity cleanup in one ordered transaction"
)]
pub(crate) async fn run_auto_unload_runtime_entry(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    unload: RuntimeUnloadCandidate,
    drain_delay: Duration,
    unload_started: Instant,
    unload_op: UnloadOperation,
) -> Result<api::RuntimeUnloadResponse> {
    let Some(entry) = ctx.runtime_models.remove(&unload.instance_id) else {
        record_runtime_operational_event_with_context(
            RuntimeOperationalEvent::ModelUnloadFailed,
            runtime_model_audit_context(Some(&unload.model_name), &unload.instance_id)
                .reason_code("instance_missing")
                .outcome("failed")
                .duration_ms(
                    u64::try_from(unload_started.elapsed().as_millis()).unwrap_or(u64::MAX),
                ),
        );
        unload_op.failed(&unload.model_name);
        anyhow::bail!(
            "model or runtime instance '{}' is not loaded",
            unload.instance_id
        );
    };
    let RuntimeModelHandleEntry {
        model_name: model,
        profile,
        handle,
        capacity_reservation,
        lifecycle,
        ..
    } = entry;
    let port = handle.port;
    let draining = {
        let mut record = lifecycle.lock().await;
        match record.mark_draining(std::time::Instant::now() + drain_delay) {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(
                    model,
                    instance_id = unload.instance_id,
                    %error,
                    "runtime instance could not enter draining; forcing unload"
                );
                false
            }
        }
    };
    if let Some(survey_model) = ctx.runtime_survey_models.remove(&unload.instance_id) {
        ctx.survey_telemetry.record_unload(&survey_model);
    }
    remove_runtime_local_target(ctx.target_tx, &model, port);
    if unregister_runtime_instance(
        ctx.runtime_instance_registry,
        ctx.node,
        &model,
        &unload.instance_id,
    )
    .await
    {
        publish_runtime_llama_unavailable(
            ctx.runtime_data_producer,
            &model,
            Some(&unload.instance_id),
        );
    }
    upsert_dashboard_process(
        ctx.dashboard_processes,
        runtime_process_payload_with_status(
            &model,
            Some(&unload.instance_id),
            &handle,
            "shutting down",
        ),
    )
    .await;
    if let Some(cs) = ctx.console_state {
        cs.upsert_local_process(runtime_process_payload_with_status(
            &model,
            Some(&unload.instance_id),
            &handle,
            "shutting down",
        ))
        .await;
    }
    if draining {
        unload_op.session_draining_started(&model);
        let result = DrainCoordinator::default()
            .wait_for_unload_ready(&lifecycle)
            .await;
        unload_op.session_draining_completed(&model);
        if result == DrainResult::ForceCancelled {
            unload_op.forced(&model);
            tracing::warn!(
                model,
                instance_id = unload.instance_id,
                "instance drain deadline expired; forcing unload"
            );
        }
        if let Err(error) = lifecycle.lock().await.transition_to_unloading() {
            tracing::warn!(
                model,
                instance_id = unload.instance_id,
                %error,
                "runtime instance could not enter unloading; forcing unload"
            );
        }
    }
    ctx.node.unregister_runtime_instance_lifecycle(port);
    remove_dashboard_context_usage(ctx.dashboard_context_usage, &model, &handle).await;
    handle.shutdown().await;
    // Native/process-supervision reconciliation: a racing exit-watcher event
    // may have already marked this instance Failed before this orderly
    // unload reached here. Read the state BEFORE the Stopped transition
    // below so the reservation resolves through the observed crash, not a
    // false "unload completed".
    let process_already_crashed = {
        let mut record = lifecycle.lock().await;
        let crashed = record.state() == InstanceLifecycleState::Failed;
        if record.state() == InstanceLifecycleState::Unloading
            && let Err(error) = record.transition_to(InstanceLifecycleState::Stopped)
        {
            tracing::warn!(
                model,
                instance_id = unload.instance_id,
                %error,
                "runtime instance stopped with a stale lifecycle state"
            );
        }
        crashed
    };
    drop(capacity_reservation);
    remove_dashboard_process(ctx.dashboard_processes, &unload.instance_id).await;
    if let Some(cs) = ctx.console_state {
        cs.remove_local_process(&unload.instance_id).await;
    }
    unregister_local_source_policy_if_unused(ctx, &model, &profile);
    let _ = emit_event(OutputEvent::Info {
        message: format!("Unloaded local model '{}' from :{}", model, port),
        context: None,
    });
    record_runtime_operational_event_with_context(
        RuntimeOperationalEvent::ModelUnloaded,
        runtime_model_audit_context(Some(&model), &unload.instance_id)
            .outcome("completed")
            .duration_ms(u64::try_from(unload_started.elapsed().as_millis()).unwrap_or(u64::MAX)),
    );
    unload_op.reconcile(&model, process_already_crashed);
    Ok(api::RuntimeUnloadResponse {
        model,
        instance_id: unload.instance_id,
        unloaded: true,
    })
}

/// Handle unexpected exit of a runtime-loaded model.
pub(crate) async fn run_auto_handle_runtime_exit(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    instance_id: String,
    model: String,
    port: u16,
) {
    let matches = ctx
        .runtime_models
        .get(&instance_id)
        .map(|entry| entry.model_name == model && entry.handle.port == port)
        .unwrap_or(false);
    if !matches {
        return;
    }
    // Process supervision observed a crash with no orderly unload sequence
    // in flight; reconcile with a synthesized terminal through the frozen
    // `ProcessCrash` reason rather than leaving the operation unresolved.
    reconcile_process_crash(&model);
    if let Some(entry) = ctx.runtime_models.remove(&instance_id) {
        let RuntimeModelHandleEntry {
            handle,
            profile,
            capacity_reservation,
            lifecycle,
            ..
        } = entry;
        ctx.node.unregister_runtime_instance_lifecycle(port);
        let _ = lifecycle
            .lock()
            .await
            .transition_to(InstanceLifecycleState::Failed);
        if let Some(survey_model) = ctx.runtime_survey_models.remove(&instance_id) {
            ctx.survey_telemetry.record_unexpected_exit(&survey_model);
        }
        if unregister_runtime_instance(
            ctx.runtime_instance_registry,
            ctx.node,
            &model,
            &instance_id,
        )
        .await
        {
            publish_runtime_llama_unavailable(
                ctx.runtime_data_producer,
                &model,
                Some(&instance_id),
            );
        }
        upsert_dashboard_process(
            ctx.dashboard_processes,
            runtime_process_payload_with_status(&model, Some(&instance_id), &handle, "exited"),
        )
        .await;
        if let Some(cs) = ctx.console_state {
            cs.upsert_local_process(runtime_process_payload_with_status(
                &model,
                Some(&instance_id),
                &handle,
                "exited",
            ))
            .await;
        }
        remove_dashboard_context_usage(ctx.dashboard_context_usage, &model, &handle).await;
        handle.shutdown().await;
        drop(capacity_reservation);
        unregister_local_source_policy_if_unused(ctx, &model, &profile);
    }
    remove_runtime_local_target(ctx.target_tx, &model, port);
    let _ = emit_event(OutputEvent::Warning {
        message: format!("Runtime model '{model}' exited unexpectedly"),
        context: Some(format!("model={model} port={port}")),
    });
    record_runtime_operational_event_with_context(
        RuntimeOperationalEvent::ModelExited,
        runtime_model_audit_context(Some(&model), &instance_id)
            .reason_code("unexpected_exit")
            .outcome("failed"),
    );
}

#[cfg(test)]
mod tests {
    use super::unregister_local_source_policy_after_lifecycle_removal;

    #[test]
    fn last_strict_lifecycle_removal_allows_profile_unaware_fallback_reload() {
        let model = format!("lifecycle-policy-model-{}", std::process::id());
        let strict_profile = "strict";
        crate::inference::skippy::register_local_source_policy(&model, strict_profile, true);

        unregister_local_source_policy_after_lifecycle_removal(
            std::iter::once((model.as_str(), strict_profile)),
            &model,
            strict_profile,
        );
        assert!(crate::inference::skippy::effective_local_source_required(
            &model, None, false
        ));

        unregister_local_source_policy_after_lifecycle_removal(
            std::iter::empty::<(&str, &str)>(),
            &model,
            strict_profile,
        );
        crate::inference::skippy::register_local_source_policy(&model, "fallback", false);

        assert!(!crate::inference::skippy::effective_local_source_required(
            &model, None, false
        ));
        crate::inference::skippy::unregister_local_source_policy(&model, "fallback");
    }
}
