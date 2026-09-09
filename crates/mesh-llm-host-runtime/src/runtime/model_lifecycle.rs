use super::{
    DashboardContextUsage, InstanceLifecycleRecord, InstanceLifecycleState, IntentSource,
    LocalRuntimeModelHandle, LocalRuntimeModelStartSpec, ManagedModelController, ModelIntent,
    ModelTargetReconciliationAction, ModelTargetReconciliationCandidate,
    ModelTargetReconciliationCapacityState, ModelTargetReconciliationInput,
    ModelTargetReconciliationPolicy, ModelTargetReconciliationState, RunAutoRuntimeLoopContext,
    RunAutoRuntimeState, RuntimeCapacityReservation, RuntimeEvent, RuntimeInstanceRegistry,
    RuntimeOperationalEvent, RuntimeOptions, RuntimeUnloadCandidate, RuntimeUnloadOwner,
    ShutdownRuntimeLoadedModelsContext, StartupModelSpec, StartupReadyReporter,
    add_runtime_local_target, add_serving_assignment, find_remote_catalog_model_exact_blocking,
    local_process_payload, next_runtime_instance_id, plan_model_target_reconciliation,
    publish_runtime_llama_slots, publish_runtime_llama_unavailable,
    record_runtime_operational_event_with_context, refresh_dashboard_context_usage,
    register_runtime_instance, remove_dashboard_context_usage, remove_dashboard_process,
    remove_runtime_local_target, remove_serving_assignment, reserve_runtime_capacity_for_model,
    resolve_model, runtime_model_ctx_size_override, runtime_model_planning_bytes,
    runtime_process_payload_with_status, runtime_registry_has_model,
    runtime_resource_planning_profile, set_advertised_model_context, skippy_telemetry_options,
    start_runtime_local_model, unregister_runtime_instance, upsert_dashboard_process,
    withdraw_advertised_model,
};
use crate::api;
use crate::inference::election;
use crate::logging::{OperationalAuditContext, OperationalAuditSubjectKind};
use crate::mesh;
use crate::models;
use crate::network::lan_bootstrap::LanBootstrapTasks;
use crate::plugin;
use crate::runtime::survey;
use anyhow::Result;
use mesh_llm_events::{OutputEvent, emit_event};
use mesh_llm_node::serving::{UnloadOptions, UnloadTarget};
use skippy_protocol::FlashAttentionType;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(super) fn runtime_model_audit_context(
    model: Option<&str>,
    instance_id: &str,
) -> OperationalAuditContext {
    match model {
        Some(model) => OperationalAuditContext::new()
            .subject(OperationalAuditSubjectKind::Model, model)
            .operation_id(instance_id),
        None => OperationalAuditContext::new()
            .subject(OperationalAuditSubjectKind::RuntimeInstance, instance_id),
    }
}

mod load;
pub(crate) mod reconciliation;
mod unload;

pub(crate) use load::{normalize_runtime_model_request_for_config, run_auto_load_runtime_model};
pub(crate) use unload::{run_auto_handle_runtime_exit, run_auto_unload_runtime_model};

// Re-export reconciliation module's public API for callers in parent scope
pub(crate) use reconciliation::{
    model_target_reconciliation_policy, reconcile_model_targets_once, runtime_unix_secs,
};

pub(crate) use reconciliation::ReconcileModelTargetsContext;
#[cfg(test)]
pub(crate) use reconciliation::{
    model_target_reconciliation_local_fit, run_model_target_reconciliation_action,
};

pub(super) struct RunAutoShutdownContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) node: &'a mesh::Node,
    pub(super) plugin_manager: &'a plugin::PluginManager,
    pub(super) api_proxy_handle: tokio::task::JoinHandle<()>,
    pub(super) console_server_handle: Option<tokio::task::JoinHandle<()>>,
    pub(super) discovery_publisher: Option<tokio::task::JoinHandle<()>>,
    pub(super) lan_bootstrap_tasks: LanBootstrapTasks,
    pub(super) runtime_models: &'a mut HashMap<String, RuntimeModelHandleEntry>,
    pub(super) runtime_survey_models: &'a mut HashMap<String, survey::SurveyLoadedModel>,
    pub(super) managed_models: &'a mut HashMap<String, ManagedModelController>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) dashboard_processes: &'a Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) runtime_instance_registry: &'a RuntimeInstanceRegistry,
    pub(super) runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
    pub(super) dashboard_context_usage: &'a DashboardContextUsage,
    pub(super) runtime: Option<std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
}

pub(super) struct RunAutoRuntimeLifecycleContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) node: &'a mesh::Node,
    pub(super) primary_model_name: &'a str,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) control_rx: &'a mut tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) runtime_event_rx: &'a mut tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
    pub(super) model_intent_rx: &'a mut tokio::sync::mpsc::Receiver<ModelIntent>,
    pub(super) runtime_state: &'a mut RunAutoRuntimeState,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) runtime_data_producer: Option<&'a crate::runtime_data::RuntimeDataProducer>,
    pub(super) runtime_event_tx: &'a tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) startup_ready_reporter: &'a StartupReadyReporter,
    pub(super) plugin_manager: &'a plugin::PluginManager,
    pub(super) api_proxy_handle: tokio::task::JoinHandle<()>,
    pub(super) console_server_handle: Option<tokio::task::JoinHandle<()>>,
    pub(super) discovery_publisher: Option<tokio::task::JoinHandle<()>>,
    pub(super) startup_specs: &'a [StartupModelSpec],
    pub(super) tunnel_mgr: &'a crate::network::tunnel::Manager,
    pub(super) skippy_telemetry: &'a crate::inference::skippy::SkippyTelemetryOptions,
    pub(super) api_port: u16,
    pub(super) console_port: Option<u16>,
    pub(super) interactive_started: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub(super) lan_bootstrap_tasks: LanBootstrapTasks,
    pub(super) runtime: Option<std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
}

pub(super) async fn run_auto_reconcile_model_targets(ctx: &mut RunAutoRuntimeLoopContext<'_>) {
    // Reconciliation exists to drive local models toward a desired set. A
    // sharing node has no such set and no way to act on one, so running it
    // would only produce load attempts that must fail. Scoped to sharing
    // rather than `allows_local_inference()` so client startup is unchanged.
    if ctx.options.shared_endpoint.is_some() {
        return;
    }
    reconcile_model_targets_once(ReconcileModelTargetsContext {
        policy: &ctx.model_target_reconciliation_policy,
        state: &mut ctx.model_target_reconciliation_state,
        node: ctx.node,
        console_state: ctx.console_state,
        runtime_models: ctx.runtime_models,
        managed_models: ctx.managed_models,
        control_tx: ctx.control_tx,
        runtime_event_tx: ctx.runtime_event_tx,
    })
    .await;
}

pub(super) fn run_auto_record_model_target_manual_unload(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    resolved_unload: Option<&RuntimeUnloadCandidate>,
    requested_target: &str,
    result: &Result<api::RuntimeUnloadResponse>,
) {
    let Ok(response) = result else {
        return;
    };
    let now_secs = runtime_unix_secs();
    let requested_profile = resolved_unload
        .filter(|candidate| {
            candidate.instance_id == requested_target || candidate.model_name == requested_target
        })
        .map(|candidate| candidate.profile.as_str())
        .unwrap_or("");
    ctx.model_target_reconciliation_state.record_manual_unload(
        requested_target,
        requested_profile,
        now_secs,
        &ctx.model_target_reconciliation_policy,
    );
    if response.model != requested_target {
        let response_profile = resolved_unload
            .filter(|candidate| candidate.model_name == response.model)
            .map(|candidate| candidate.profile.as_str())
            .unwrap_or("");
        ctx.model_target_reconciliation_state.record_manual_unload(
            &response.model,
            response_profile,
            now_secs,
            &ctx.model_target_reconciliation_policy,
        );
    }
}

pub(super) fn run_auto_handle_model_target_reconciliation_result(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    model_ref: String,
    profile: String,
    result: std::result::Result<api::RuntimeLoadResponse, String>,
) {
    match apply_model_target_reconciliation_load_finished(
        &mut ctx.model_target_reconciliation_state,
        &ctx.model_target_reconciliation_policy,
        &model_ref,
        &profile,
        result,
        runtime_unix_secs(),
    ) {
        Ok(response) => {
            let load_profile = if response.profile.is_empty() {
                profile.clone()
            } else {
                response.profile.clone()
            };
            if !load_profile.is_empty() && load_profile != profile {
                tracing::warn!(
                    model_ref = %model_ref,
                    requested_profile = %profile,
                    loaded_profile = %load_profile,
                    "model target reconciliation load response profile differs from requested profile"
                );
            }
            let _ = emit_event(OutputEvent::Info {
                message: format!("Model target reconciliation loaded '{}'", response.model),
                context: Some(format!(
                    "model_ref={} instance={}",
                    model_ref, response.instance_id
                )),
            });
        }
        Err(error) => {
            let _ = emit_event(OutputEvent::Warning {
                message: format!("Model target reconciliation failed for '{model_ref}'"),
                context: Some(error),
            });
        }
    }
}

pub(super) fn apply_model_target_reconciliation_load_finished(
    state: &mut ModelTargetReconciliationState,
    policy: &ModelTargetReconciliationPolicy,
    model_ref: &str,
    profile: &str,
    result: std::result::Result<api::RuntimeLoadResponse, String>,
    now_secs: u64,
) -> std::result::Result<api::RuntimeLoadResponse, String> {
    match &result {
        Ok(response) => {
            state.record_load_success(model_ref, profile);
            state.notify_load_success(model_ref, profile, response.clone());
        }
        Err(error) => {
            state.record_load_failure(model_ref, profile, now_secs, policy);
            state.notify_load_failure(model_ref, profile, &anyhow::Error::msg(error.clone()));
        }
    }
    result
}

pub(super) fn apply_startup_model_load_finished(
    state: &mut ModelTargetReconciliationState,
    policy: &ModelTargetReconciliationPolicy,
    model_ref: &str,
    profile: &str,
    result: std::result::Result<api::RuntimeLoadResponse, String>,
    now_secs: u64,
) {
    match result {
        Ok(response) => {
            state.record_load_success(model_ref, profile);
            state.notify_load_success(model_ref, profile, response);
        }
        Err(error) => {
            let error_message = error;
            let error = anyhow::Error::msg(error_message.clone());
            state.record_load_failure(model_ref, profile, now_secs, policy);
            state.set_effective_intent_error(model_ref, profile, error_message);
            state.notify_load_failure(model_ref, profile, &error);
        }
    }
}

pub(super) async fn shutdown_runtime_loaded_models(
    runtime_models: &mut HashMap<String, RuntimeModelHandleEntry>,
    runtime_survey_models: &mut HashMap<String, survey::SurveyLoadedModel>,
    ctx: ShutdownRuntimeLoadedModelsContext<'_>,
) {
    let ShutdownRuntimeLoadedModelsContext {
        survey_telemetry,
        dashboard_processes,
        console_state,
        target_tx,
        runtime_instance_registry,
        node,
        runtime_data_producer,
        dashboard_context_usage,
    } = ctx;

    for (instance_id, entry) in runtime_models.drain() {
        let unload_started = Instant::now();
        let RuntimeModelHandleEntry {
            model_name: name,
            handle,
            capacity_reservation,
            lifecycle,
            ..
        } = entry;
        if let Some(survey_model) = runtime_survey_models.remove(&instance_id) {
            survey_telemetry.record_unload(&survey_model);
        }
        let shutting_down_payload = runtime_process_payload_with_status(
            &name,
            Some(&instance_id),
            &handle,
            "shutting down",
        );
        upsert_dashboard_process(dashboard_processes, shutting_down_payload.clone()).await;
        if let Some(cs) = console_state {
            cs.upsert_local_process(shutting_down_payload).await;
        }
        remove_runtime_local_target(target_tx, &name, handle.port);
        if unregister_runtime_instance(runtime_instance_registry, node, &name, &instance_id).await {
            publish_runtime_llama_unavailable(runtime_data_producer, &name, Some(&instance_id));
        }
        remove_dashboard_context_usage(dashboard_context_usage, &name, &handle).await;
        let _ = emit_event(OutputEvent::ModelUnloading {
            model: name.clone(),
        });
        record_runtime_operational_event_with_context(
            RuntimeOperationalEvent::ModelUnloadStarted,
            runtime_model_audit_context(Some(&name), &instance_id).outcome("started"),
        );
        let stopped_payload =
            runtime_process_payload_with_status(&name, Some(&instance_id), &handle, "stopped");
        {
            let mut record = lifecycle.lock().await;
            if record.mark_draining_force().is_ok() {
                let _ = record.transition_to_unloading();
            }
        }
        node.unregister_runtime_instance_lifecycle(handle.port);
        handle.shutdown().await;
        let _ = lifecycle
            .lock()
            .await
            .transition_to(InstanceLifecycleState::Stopped);
        drop(capacity_reservation);
        let _ = emit_event(OutputEvent::ModelUnloaded {
            model: name.clone(),
        });
        record_runtime_operational_event_with_context(
            RuntimeOperationalEvent::ModelUnloaded,
            runtime_model_audit_context(Some(&name), &instance_id)
                .outcome("completed")
                .duration_ms(
                    u64::try_from(unload_started.elapsed().as_millis()).unwrap_or(u64::MAX),
                ),
        );
        upsert_dashboard_process(dashboard_processes, stopped_payload.clone()).await;
        if let Some(cs) = console_state {
            cs.upsert_local_process(stopped_payload).await;
        }
    }
}

pub(super) async fn shutdown_runtime_managed_models(
    managed_models: &mut HashMap<String, ManagedModelController>,
) {
    for (instance_id, controller) in managed_models.drain() {
        let unload_started = Instant::now();
        let _ = emit_event(OutputEvent::ModelUnloading {
            model: controller.model_name.clone(),
        });
        record_runtime_operational_event_with_context(
            RuntimeOperationalEvent::ModelUnloadStarted,
            runtime_model_audit_context(Some(&controller.model_name), &instance_id)
                .outcome("started"),
        );
        let _ = controller.stop_tx.send(true);
        let mut task = controller.task;
        match tokio::time::timeout(std::time::Duration::from_secs(3), &mut task).await {
            Ok(join_result) => {
                let _ = join_result;
            }
            Err(_) => {
                tracing::warn!("local model task did not stop within 3s during shutdown");
                task.abort();
                let _ = task.await;
            }
        }
        let _ = emit_event(OutputEvent::ModelUnloaded {
            model: controller.model_name.clone(),
        });
        record_runtime_operational_event_with_context(
            RuntimeOperationalEvent::ModelUnloaded,
            runtime_model_audit_context(Some(&controller.model_name), &instance_id)
                .outcome("completed")
                .duration_ms(
                    u64::try_from(unload_started.elapsed().as_millis()).unwrap_or(u64::MAX),
                ),
        );
    }
}

pub(super) struct RuntimeModelHandleEntry {
    pub(super) model_name: String,
    pub(super) profile: String,
    pub(super) handle: LocalRuntimeModelHandle,
    pub(super) capacity_reservation: RuntimeCapacityReservation,
    /// Per-instance lifecycle state machine for admission and drain control.
    #[allow(
        dead_code,
        reason = "retained with each loaded runtime so lifecycle ownership follows the instance"
    )]
    pub(super) lifecycle: std::sync::Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
}

pub(super) fn runtime_unload_candidates(
    runtime_models: &HashMap<String, RuntimeModelHandleEntry>,
    managed_models: &HashMap<String, ManagedModelController>,
) -> Vec<RuntimeUnloadCandidate> {
    runtime_models
        .iter()
        .map(|(instance_id, entry)| RuntimeUnloadCandidate {
            owner: RuntimeUnloadOwner::Runtime,
            instance_id: instance_id.clone(),
            model_name: entry.model_name.clone(),
            profile: entry.profile.clone(),
        })
        .chain(
            managed_models
                .iter()
                .map(|(instance_id, controller)| RuntimeUnloadCandidate {
                    owner: RuntimeUnloadOwner::Managed,
                    instance_id: instance_id.clone(),
                    model_name: controller.model_name.clone(),
                    profile: controller.profile.clone(),
                }),
        )
        .collect()
}

pub(super) fn suppress_desired_for_resolved_unload_candidate(
    state: &mut ModelTargetReconciliationState,
    candidate: &RuntimeUnloadCandidate,
    source: IntentSource,
    draining: bool,
    intent_id: Option<String>,
) -> String {
    state.suppress_desired_with_id(
        &candidate.model_name,
        &candidate.profile,
        Some(candidate.instance_id.clone()),
        source,
        draining,
        intent_id,
    )
}

pub(super) fn resolve_runtime_unload_target(
    target: &str,
    candidates: Vec<RuntimeUnloadCandidate>,
) -> Result<RuntimeUnloadCandidate> {
    let mut instance_matches = candidates
        .iter()
        .filter(|candidate| candidate.instance_id == target);
    if let Some(candidate) = instance_matches.next() {
        return Ok(candidate.clone());
    }

    let model_matches: Vec<_> = candidates
        .into_iter()
        .filter(|candidate| candidate.model_name == target)
        .collect();
    match model_matches.len() {
        0 => Err(anyhow::anyhow!(
            "model or runtime instance '{target}' is not loaded"
        )),
        1 => Ok(model_matches.into_iter().next().expect("one model match")),
        _ => {
            let ids = model_matches
                .iter()
                .map(|candidate| candidate.instance_id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            Err(anyhow::anyhow!(
                "model '{target}' has multiple loaded instances ({ids}); unload by runtime instance id"
            ))
        }
    }
}
