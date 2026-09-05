use super::*;
use crate::inference::skippy;
use anyhow::Context;
use std::path::{Path, PathBuf};

fn audit_runtime_model_load_result<T>(
    result: Result<T>,
    context: OperationalAuditContext,
    load_started: Instant,
) -> Result<T> {
    result.inspect_err(|_| {
        record_runtime_operational_event_with_context(
            RuntimeOperationalEvent::ModelLoadFailed,
            context
                .outcome("failed")
                .duration_ms(u64::try_from(load_started.elapsed().as_millis()).unwrap_or(u64::MAX)),
        );
    })
}

fn record_runtime_model_load_terminal(
    event: RuntimeOperationalEvent,
    model: &str,
    instance_id: &str,
    outcome: &'static str,
    load_started: Instant,
) {
    record_runtime_operational_event_with_context(
        event,
        runtime_model_audit_context(Some(model), instance_id)
            .outcome(outcome)
            .duration_ms(u64::try_from(load_started.elapsed().as_millis()).unwrap_or(u64::MAX)),
    );
}

fn spawn_runtime_model_exit_watcher(
    event_tx: tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    instance_id: String,
    model: String,
    port: u16,
    death_rx: tokio::sync::oneshot::Receiver<()>,
) {
    tokio::spawn(async move {
        let _ = death_rx.await;
        let _ = event_tx.send(RuntimeEvent::Exited {
            instance_id,
            model,
            port,
        });
    });
}

async fn plan_runtime_model_bytes(model_path: &Path, requested_model: &str) -> u64 {
    let planning_path = model_path.to_path_buf();
    tokio::task::spawn_blocking(move || runtime_model_planning_bytes(&planning_path))
        .await
        .unwrap_or_else(|err| {
            Err(anyhow::anyhow!(
                "join runtime model byte planning task: {err}"
            ))
        })
        .unwrap_or_else(|err| {
            let fallback = election::total_model_bytes(model_path);
            tracing::warn!(
                model = %requested_model,
                error = %err,
                fallback_bytes = fallback,
                "failed to resolve runtime model planning bytes; using filesystem size fallback"
            );
            fallback
        })
}

fn resolve_runtime_model_config<'a>(
    config: &'a plugin::MeshConfig,
    config_model_id: Option<&str>,
    spec: &str,
    requested_profile: &str,
) -> Result<(Option<&'a plugin::ModelConfigEntry>, String)> {
    let selector = config_model_id.unwrap_or(spec);
    let matching = config
        .models
        .iter()
        .filter(|model| model.model == selector)
        .collect::<Vec<_>>();
    if matching.is_empty() {
        anyhow::ensure!(
            config_model_id.is_none(),
            "configured model {selector} is no longer present"
        );
        return Ok((None, requested_profile.to_string()));
    }
    anyhow::ensure!(
        !requested_profile.is_empty() || matching.len() == 1,
        "configured model {selector} has multiple runtime profiles; an explicit profile is required"
    );
    if let Some(model) = matching.iter().find(|model| {
        model
            .with_profile_defaults(config.defaults.as_ref())
            .derived_profile()
            == requested_profile
    }) {
        return Ok((Some(*model), requested_profile.to_string()));
    }
    if requested_profile.is_empty() && matching.len() == 1 {
        let model = matching[0];
        let profile = model
            .with_profile_defaults(config.defaults.as_ref())
            .derived_profile();
        return Ok((Some(model), profile));
    }
    anyhow::bail!(
        "configured model {selector} does not define runtime profile {requested_profile:?}"
    )
}

pub(crate) fn normalize_runtime_model_request(
    spec: String,
    profile: String,
) -> Result<(String, String)> {
    anyhow::ensure!(
        spec == spec.trim(),
        "runtime model reference must not contain leading or trailing whitespace"
    );
    anyhow::ensure!(
        profile == profile.trim(),
        "runtime profile must not contain leading or trailing whitespace"
    );
    anyhow::ensure!(!spec.is_empty(), "runtime model reference is empty");
    let Some((model_ref, embedded_profile)) = spec.split_once('#') else {
        return Ok((spec, profile));
    };
    anyhow::ensure!(!model_ref.is_empty(), "runtime model reference is empty");
    anyhow::ensure!(
        model_ref == model_ref.trim(),
        "runtime model reference must not contain whitespace before its profile delimiter"
    );
    anyhow::ensure!(
        embedded_profile == embedded_profile.trim(),
        "runtime profile must not contain whitespace after its delimiter"
    );
    anyhow::ensure!(
        !embedded_profile.contains('#'),
        "runtime model request may contain at most one profile delimiter"
    );
    anyhow::ensure!(
        profile.is_empty() || embedded_profile.is_empty() || profile == embedded_profile,
        "runtime model request contains conflicting profiles"
    );
    let resolved_profile = if profile.is_empty() {
        embedded_profile.to_string()
    } else {
        profile
    };
    Ok((model_ref.to_string(), resolved_profile))
}

/// Canonicalize the reconciliation identity before desired/pending state is
/// keyed. Config-driven reconciliation carries the logical config selector
/// separately from the concrete load path, so profile policy must be resolved
/// through that selector rather than through the path.
pub(crate) fn normalize_runtime_model_request_for_config(
    config: &plugin::MeshConfig,
    config_model_id: Option<&str>,
    spec: String,
    profile: String,
) -> Result<(String, String)> {
    let (spec, profile) = normalize_runtime_model_request(spec, profile)?;
    let (_, profile) = resolve_runtime_model_config(config, config_model_id, &spec, &profile)?;
    Ok((spec, profile))
}

fn local_required_runtime_model_name(
    configured_model_id: Option<&str>,
    model_path: &Path,
) -> String {
    configured_model_id
        .map(str::to_string)
        .unwrap_or_else(|| models::model_ref_for_path(model_path))
}

struct ResolvedRuntimeModelSource {
    model_path: PathBuf,
    runtime_model_name: String,
    local_source_required: bool,
}

/// State owned by the post-launch registration phase.
struct RuntimeModelLaunchSuccess {
    requested_model: String,
    config_model_id: Option<String>,
    profile: String,
    instance_id: String,
    model_path: PathBuf,
    loaded_name: String,
    handle: LocalRuntimeModelHandle,
    death_rx: tokio::sync::oneshot::Receiver<()>,
    capacity_reservation: RuntimeCapacityReservation,
    launch_started: Instant,
    load_started: Instant,
    load_op: LoadOperation,
}

fn configured_runtime_model_path(
    config: &plugin::MeshConfig,
    model_overrides: Option<&plugin::ModelConfigEntry>,
) -> Option<PathBuf> {
    let model = model_overrides?;
    model
        .hardware
        .as_ref()
        .and_then(|hardware| hardware.model_path.as_ref())
        .or_else(|| {
            config
                .defaults
                .as_ref()
                .and_then(|defaults| defaults.hardware.as_ref())
                .and_then(|hardware| hardware.model_path.as_ref())
        })
        .map(PathBuf::from)
}

fn local_required_runtime_model_path(
    config: &plugin::MeshConfig,
    model_overrides: Option<&plugin::ModelConfigEntry>,
    spec: &str,
) -> Option<PathBuf> {
    configured_runtime_model_path(config, model_overrides).or_else(|| {
        let direct = PathBuf::from(spec);
        direct.is_absolute().then_some(direct)
    })
}

/// Resolve source policy before entering any remote resolver. An unknown
/// caller-supplied profile must never discard a configured strict-local policy.
async fn resolve_runtime_model_source(
    ctx: &RunAutoRuntimeLoopContext<'_>,
    spec: &str,
    config_model_id: Option<&str>,
    profile: &str,
    model_overrides: Option<&plugin::ModelConfigEntry>,
    instance_id: &str,
    load_started: Instant,
) -> Result<ResolvedRuntimeModelSource> {
    let default_skippy = ctx
        .config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.skippy.as_ref());
    let local_source_required = crate::runtime::startup_models::skippy_local_source_required(
        model_overrides.and_then(|model| model.skippy.as_ref()),
        default_skippy,
    );
    let model_path = if local_source_required {
        let selector = config_model_id.unwrap_or(spec);
        audit_runtime_model_load_result(
            (|| {
                let path = local_required_runtime_model_path(ctx.config, model_overrides, spec)
                    .with_context(|| {
                        format!(
                            "skippy.source_policy = \"local-required\" for {selector} requires hardware.model_path or an absolute model path"
                        )
                    })?;
                crate::runtime::startup_models::validate_local_required_source(&path, selector)?;
                path.canonicalize().with_context(|| {
                    format!("canonicalize local-required GGUF: {}", path.display())
                })
            })(),
            runtime_model_audit_context(None, instance_id),
            load_started,
        )?
    } else {
        audit_runtime_model_load_result(
            resolve_model(&PathBuf::from(spec)).await,
            runtime_model_audit_context(None, instance_id),
            load_started,
        )?
    };
    let runtime_model_name = if local_source_required {
        let configured_model_id = config_model_id.or_else(|| model_overrides.map(|_| spec));
        local_required_runtime_model_name(configured_model_id, &model_path)
    } else {
        find_remote_catalog_model_exact_blocking(spec.to_string())
            .await
            .map(|model| models::remote_catalog_model_ref(&model))
            .unwrap_or_else(|| models::model_ref_for_path(&model_path))
    };
    // Once strict identity is known, register it before the first await. The
    // serving assignment is gossiped and stage-control requests can arrive
    // while a large GGUF is still being hashed and indexed.
    skippy::register_local_source_policy(&runtime_model_name, profile, local_source_required);
    Ok(ResolvedRuntimeModelSource {
        model_path,
        runtime_model_name,
        local_source_required,
    })
}

/// Register a successfully launched model across routing, status, lifecycle,
/// telemetry, and reconciliation state.
async fn finish_runtime_model_load(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    success: RuntimeModelLaunchSuccess,
) -> api::RuntimeLoadResponse {
    let RuntimeModelLaunchSuccess {
        requested_model,
        config_model_id,
        profile,
        instance_id,
        model_path,
        loaded_name,
        handle,
        death_rx,
        capacity_reservation,
        launch_started,
        load_started,
        load_op,
    } = success;
    // §8.3 "backend or device selected": known now that native load
    // succeeded, before consuming `load_op`.
    load_op.backend_device_selected(&loaded_name, &handle.backend);
    // Native load already succeeded (this function only runs on the launch
    // success path). Resolves the loading-family terminal only; public
    // availability is decided separately, below, once Rust serving surfaces
    // (routing, instance registry, dashboard) are actually usable.
    let availability = load_op.native_load_completed(&loaded_name);
    // §8.4 "Rust backend initialization started": Rust is about to begin
    // registering serving surfaces below.
    availability.rust_backend_initialization_started(&loaded_name);
    let survey_loaded_model = ctx.survey_telemetry.model(survey::SurveyModelSpec {
        model: &loaded_name,
        configured_model_selector: config_model_id.as_deref(),
        model_path: Some(&model_path),
        launch_kind: survey::SurveyLaunchKind::RuntimeLoad,
        pinned_gpu: None,
        backend: Some(&handle.backend),
        context_length: Some(u64::from(handle.context_length)),
    });
    ctx.survey_telemetry
        .record_launch_success(&survey_loaded_model, launch_started.elapsed());
    add_runtime_local_target(ctx.target_tx, &loaded_name, handle.port);
    register_runtime_instance(
        ctx.runtime_instance_registry,
        ctx.node,
        ctx.primary_model_name,
        &loaded_name,
        &instance_id,
        Some(handle.context_length),
        handle.capabilities,
    )
    .await;
    ctx.node
        .set_available_models(models::scan_local_models())
        .await;
    let payload = local_process_payload(
        &loaded_name,
        Some(&instance_id),
        &profile,
        &handle.backend,
        handle.port,
        handle.pid(),
        handle.slots,
        handle.context_length,
    );
    upsert_dashboard_process(ctx.dashboard_processes, payload.clone()).await;
    if let Some(cs) = ctx.console_state {
        cs.set_openai_guardrails(
            handle
                .openai_guardrails()
                .map(crate::api::status::OpenAiGuardrailsPayload::from),
        )
        .await;
        cs.upsert_local_process(payload).await;
    }

    spawn_runtime_model_exit_watcher(
        ctx.runtime_event_tx.clone(),
        instance_id.clone(),
        loaded_name.clone(),
        handle.port,
        death_rx,
    );

    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Runtime-loaded {} model '{}' on :{}",
            handle.backend, loaded_name, handle.port
        ),
        context: None,
    });
    refresh_dashboard_context_usage(ctx.dashboard_context_usage, &loaded_name, &handle).await;
    publish_runtime_llama_slots(
        ctx.runtime_data_producer,
        &loaded_name,
        Some(&instance_id),
        &handle,
    );
    ctx.runtime_survey_models
        .insert(instance_id.clone(), survey_loaded_model);
    let loaded_backend = handle.backend.clone();
    let loaded_context_length = handle.context_length;
    let lifecycle = std::sync::Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
        InstanceLifecycleState::Serving,
        50,
    )));
    ctx.node
        .register_runtime_instance_lifecycle(handle.port, lifecycle.clone());
    ctx.runtime_models.insert(
        instance_id.clone(),
        RuntimeModelHandleEntry {
            model_name: loaded_name.clone(),
            profile: profile.clone(),
            handle,
            capacity_reservation,
            lifecycle,
        },
    );
    record_runtime_model_load_terminal(
        RuntimeOperationalEvent::ModelReady,
        &loaded_name,
        &instance_id,
        "ready",
        load_started,
    );
    availability.model_available(&loaded_name);
    api::RuntimeLoadResponse {
        model_ref: requested_model,
        model: loaded_name,
        instance_id,
        profile,
        backend: Some(loaded_backend),
        context_length: Some(loaded_context_length),
    }
}

/// Run auto-load for a runtime model.
pub(crate) async fn run_auto_load_runtime_model(
    ctx: &mut RunAutoRuntimeLoopContext<'_>,
    spec: String,
    config_model_id: Option<String>,
    profile: String,
) -> Result<api::RuntimeLoadResponse> {
    let (spec, profile) = normalize_runtime_model_request_for_config(
        ctx.config,
        config_model_id.as_deref(),
        spec,
        profile,
    )?;
    let load_started = Instant::now();
    let instance_id = next_runtime_instance_id(ctx.next_runtime_instance_sequence);
    // Reservation acquired before any load work begins, per plan task 9. A
    // dropped `load_op` on an early `?` failure below synthesizes its own
    // terminal through the engine's guard-drop mechanism.
    let mut load_op = LoadOperation::begin(&spec);
    record_runtime_operational_event_with_context(
        RuntimeOperationalEvent::ModelLoadStarted,
        runtime_model_audit_context(None, &instance_id).outcome("started"),
    );
    let (model_overrides, profile) = audit_runtime_model_load_result(
        resolve_runtime_model_config(ctx.config, config_model_id.as_deref(), &spec, &profile),
        runtime_model_audit_context(None, &instance_id),
        load_started,
    )?;
    let ResolvedRuntimeModelSource {
        model_path,
        runtime_model_name,
        local_source_required,
    } = resolve_runtime_model_source(
        ctx,
        &spec,
        config_model_id.as_deref(),
        &profile,
        model_overrides,
        &instance_id,
        load_started,
    )
    .await?;
    // §8.2 "model resolution completed": the model's source is now known
    // (local path or remote catalog ref), before any native load work.
    load_op.resolution_completed(&runtime_model_name);
    let requested_model = spec.clone();
    let model_bytes = plan_runtime_model_bytes(&model_path, &requested_model).await;
    let ctx_size_override = runtime_model_ctx_size_override(ctx.options, model_overrides);
    let parallel_override = crate::runtime::startup_models::resolve_model_parallel_override(
        model_overrides.and_then(|m| m.parallel),
        &ctx.config.gpu,
    );
    let capacity_reservation = match audit_runtime_model_load_result(
        reserve_runtime_capacity_for_model(
            ctx.runtime_capacity_ledger,
            &instance_id,
            &runtime_model_name,
            None,
            ctx.node.local_runtime_capacity_bytes(),
            model_bytes,
        ),
        runtime_model_audit_context(Some(&runtime_model_name), &instance_id),
        load_started,
    ) {
        Ok(reservation) => {
            // §8.4 "model capacity changed": local capacity is now
            // reserved for this load.
            load_op.capacity_changed(&runtime_model_name);
            reservation
        }
        Err(error) => {
            super::unload::unregister_local_source_policy_if_unused(
                ctx,
                &runtime_model_name,
                &profile,
            );
            return Err(error);
        }
    };
    add_serving_assignment(ctx.node, ctx.primary_model_name, &runtime_model_name).await;
    let launch_started = Instant::now();
    let capacity_budget_bytes = capacity_reservation.capacity_budget_bytes();
    let (loaded_name, handle, death_rx) = match start_runtime_local_model(
        LocalRuntimeModelStartSpec {
            node: ctx.node,
            mesh_config: ctx.config,
            config_model_id: config_model_id
                .as_deref()
                .or_else(|| model_overrides.map(|model| model.model.as_str())),
            runtime_profile: &profile,
            model_path: &model_path,
            preindexed_split_package: None,
            model_bytes,
            mmproj_override: None,
            ctx_size_override,
            pinned_gpu: None,
            device_override: None,
            capacity_budget_bytes: Some(capacity_budget_bytes),
            cache_type_k_override: model_overrides.and_then(|m| m.cache_type_k.as_deref()),
            cache_type_v_override: model_overrides.and_then(|m| m.cache_type_v.as_deref()),
            n_batch_override: model_overrides.and_then(|m| m.batch),
            n_ubatch_override: model_overrides.and_then(|m| m.ubatch),
            flash_attention_override: model_overrides
                .and_then(|m| m.flash_attention)
                .unwrap_or(FlashAttentionType::Auto),
            parallel_override,
            local_source_required,
            split_topology_lock: None,
            planning_profile: runtime_resource_planning_profile(ctx.options),
            openai_guardrail_policy: ctx.openai_guardrail_policy.clone(),
            skippy_telemetry: skippy_telemetry_options(ctx.options),
            survey_telemetry: ctx.survey_telemetry.clone(),
        },
        &runtime_model_name,
        // D2 fix (event-system-fixes deferral): correlates every native
        // `ModelOpenProgress` tick under this SAME load operation's root
        // (`load_op.progress_ingress`), rather than a fresh, uncorrelated
        // one. `&self` borrow only -- `load_op` itself is still needed
        // below on both the success and failure paths.
        load_op.progress_ingress(),
    )
    .await
    {
        Ok(result) => result,
        Err(err) => {
            drop(capacity_reservation);
            remove_serving_assignment(ctx.node, &runtime_model_name).await;
            super::unload::unregister_local_source_policy_if_unused(
                ctx,
                &runtime_model_name,
                &profile,
            );
            ctx.survey_telemetry.record_launch_failure(
                survey::SurveyModelSpec {
                    model: &requested_model,
                    configured_model_selector: config_model_id.as_deref(),
                    model_path: Some(&model_path),
                    launch_kind: survey::SurveyLaunchKind::RuntimeLoad,
                    pinned_gpu: None,
                    backend: None,
                    context_length: ctx_size_override.map(u64::from),
                },
                launch_started.elapsed(),
                survey::classify_launch_failure(&err),
            );
            record_runtime_model_load_terminal(
                RuntimeOperationalEvent::ModelLoadFailed,
                &runtime_model_name,
                &instance_id,
                "failed",
                load_started,
            );
            load_op.load_failed(&runtime_model_name);
            return Err(err);
        }
    };
    Ok(finish_runtime_model_load(
        ctx,
        RuntimeModelLaunchSuccess {
            requested_model,
            config_model_id,
            profile,
            instance_id,
            model_path,
            loaded_name,
            handle,
            death_rx,
            capacity_reservation,
            launch_started,
            load_started,
            load_op,
        },
    )
    .await)
}

#[cfg(test)]
mod tests {
    use super::{
        local_required_runtime_model_name, local_required_runtime_model_path,
        normalize_runtime_model_request, normalize_runtime_model_request_for_config,
        resolve_runtime_model_config,
    };
    use mesh_llm_config::{
        HardwareConfig, MeshConfig, ModelConfigDefaults, ModelConfigEntry, ModelFitConfig,
        SkippyConfig,
    };

    fn strict_model(model: &str) -> ModelConfigEntry {
        ModelConfigEntry {
            model: model.to_string(),
            skippy: Some(SkippyConfig {
                source_policy: Some("local-required".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn config_with(models: Vec<ModelConfigEntry>) -> MeshConfig {
        MeshConfig {
            models,
            ..Default::default()
        }
    }

    #[test]
    fn unconfigured_strict_absolute_path_ignores_default_model_path() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                hardware: Some(HardwareConfig {
                    model_path: Some("/models/default.gguf".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };

        assert_eq!(
            local_required_runtime_model_path(&config, None, "/models/requested.gguf"),
            Some("/models/requested.gguf".into())
        );
    }

    #[test]
    fn configured_strict_model_inherits_default_model_path() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                hardware: Some(HardwareConfig {
                    model_path: Some("/models/default.gguf".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let configured = ModelConfigEntry {
            model: "org/model".to_string(),
            ..Default::default()
        };

        assert_eq!(
            local_required_runtime_model_path(&config, Some(&configured), "/models/requested.gguf",),
            Some("/models/default.gguf".into())
        );
    }

    #[test]
    fn configured_model_rejects_unknown_profile_before_source_resolution() {
        let config = config_with(vec![strict_model("org/model")]);

        let error = resolve_runtime_model_config(&config, None, "org/model", "attacker")
            .expect_err("an unknown profile must not discard configured source policy");

        assert!(
            error
                .to_string()
                .contains("does not define runtime profile")
        );
    }

    #[test]
    fn config_selector_applies_policy_to_a_concrete_runtime_path() {
        let configured = strict_model("org/model");
        let expected_profile = configured.derived_profile();
        let config = config_with(vec![configured]);

        let (resolved, profile) =
            resolve_runtime_model_config(&config, Some("org/model"), "/models/node-local.gguf", "")
                .expect("the logical selector should resolve independently of the local path");

        assert_eq!(
            resolved.map(|model| model.model.as_str()),
            Some("org/model")
        );
        assert_eq!(profile, expected_profile);
        assert_eq!(
            resolved
                .and_then(|model| model.skippy.as_ref())
                .and_then(|skippy| skippy.source_policy.as_deref()),
            Some("local-required")
        );
    }

    #[test]
    fn missing_config_selector_fails_closed() {
        let config = config_with(vec![strict_model("org/model")]);

        let error = resolve_runtime_model_config(
            &config,
            Some("removed/model"),
            "/models/node-local.gguf",
            "",
        )
        .expect_err("a stale config selector must not enter unconfigured resolution");

        assert!(error.to_string().contains("is no longer present"));
    }

    #[test]
    fn unique_configured_model_canonicalizes_empty_profile() {
        let configured = strict_model("org/model");
        let expected_profile = configured.derived_profile();
        let config = config_with(vec![configured]);

        let (resolved, profile) = resolve_runtime_model_config(&config, None, "org/model", "")
            .expect("a unique configured model should accept an omitted profile");

        assert!(resolved.is_some());
        assert_eq!(profile, expected_profile);
    }

    #[test]
    fn configured_model_rejects_ambiguous_omitted_profile() {
        let first = ModelConfigEntry {
            model: "org/model".to_string(),
            model_fit: Some(ModelFitConfig {
                ctx_size: Some(4096),
                ..Default::default()
            }),
            ..Default::default()
        };
        let second = ModelConfigEntry {
            model: "org/model".to_string(),
            model_fit: Some(ModelFitConfig {
                ctx_size: Some(8192),
                ..Default::default()
            }),
            ..Default::default()
        };
        let config = config_with(vec![first, second]);

        let error = resolve_runtime_model_config(&config, None, "org/model", "")
            .expect_err("an omitted profile is ambiguous across configured variants");

        assert!(
            error
                .to_string()
                .contains("an explicit profile is required")
        );
    }

    #[test]
    fn configured_model_rejects_empty_profile_fallback_when_selector_is_ambiguous() {
        let fallback = ModelConfigEntry {
            model: "org/model".to_string(),
            ..Default::default()
        };
        assert!(fallback.derived_profile().is_empty());
        let config = config_with(vec![fallback, strict_model("org/model")]);

        let error = resolve_runtime_model_config(&config, None, "org/model", "")
            .expect_err("an empty-profile fallback must not bypass a strict sibling profile");

        assert!(
            error
                .to_string()
                .contains("an explicit profile is required")
        );
    }

    #[test]
    fn embedded_profile_is_normalized_before_config_lookup() {
        let config = config_with(vec![strict_model("org/model")]);
        let (spec, profile) =
            normalize_runtime_model_request("org/model#attacker".to_string(), String::new())
                .expect("combined model profile should normalize");

        let error = resolve_runtime_model_config(&config, None, &spec, &profile)
            .expect_err("an embedded forged profile must not bypass configured policy");

        assert_eq!(spec, "org/model");
        assert_eq!(profile, "attacker");
        assert!(
            error
                .to_string()
                .contains("does not define runtime profile")
        );
    }

    #[test]
    fn whitespace_alias_is_rejected_before_config_lookup() {
        let error =
            normalize_runtime_model_request(" org/model ".to_string(), "strict".to_string())
                .expect_err("resolver-trimmed aliases must not bypass exact config matching");

        assert!(error.to_string().contains("whitespace"));
    }

    #[test]
    fn profile_delimiter_aliases_are_rejected_before_config_lookup() {
        for spec in [
            "org/model #attacker",
            "org/model# attacker",
            "https://huggingface.co/org/model#x#attacker",
        ] {
            let error = normalize_runtime_model_request(spec.to_string(), String::new())
                .expect_err("non-canonical profile spelling must not bypass configured policy");

            assert!(
                error.to_string().contains("whitespace")
                    || error.to_string().contains("profile delimiter"),
                "unexpected error for {spec:?}: {error}"
            );
        }
    }

    #[test]
    fn matching_embedded_and_explicit_profiles_normalize_once() {
        let (spec, profile) =
            normalize_runtime_model_request("org/model#strict".to_string(), "strict".to_string())
                .expect("matching profile spellings should normalize");

        assert_eq!(spec, "org/model");
        assert_eq!(profile, "strict");
    }

    #[test]
    fn equivalent_profile_spellings_share_one_reconciliation_key() {
        let config = MeshConfig::default();
        let embedded = normalize_runtime_model_request_for_config(
            &config,
            None,
            "org/model#strict".to_string(),
            String::new(),
        )
        .unwrap();
        let separate = normalize_runtime_model_request_for_config(
            &config,
            None,
            "org/model".to_string(),
            "strict".to_string(),
        )
        .unwrap();

        assert_eq!(embedded, separate);
    }

    #[test]
    fn omitted_and_explicit_derived_profiles_share_one_reconciliation_key() {
        let configured = strict_model("org/model");
        let derived_profile = configured.derived_profile();
        let config = config_with(vec![configured]);

        let omitted = normalize_runtime_model_request_for_config(
            &config,
            None,
            "org/model".to_string(),
            String::new(),
        )
        .unwrap();
        let explicit = normalize_runtime_model_request_for_config(
            &config,
            None,
            "org/model".to_string(),
            derived_profile,
        )
        .unwrap();

        assert_eq!(omitted, explicit);
    }

    #[test]
    fn unconfigured_model_preserves_requested_profile() {
        let config = MeshConfig::default();
        let (resolved, profile) =
            resolve_runtime_model_config(&config, None, "org/model", "custom")
                .expect("unconfigured models keep legacy resolution behavior");

        assert!(resolved.is_none());
        assert_eq!(profile, "custom");
    }

    #[test]
    fn unconfigured_strict_path_registers_policy_under_derived_runtime_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dynamic.gguf");
        std::fs::write(&path, b"gguf").unwrap();
        let canonical = path.canonicalize().unwrap();
        let runtime_model_name = local_required_runtime_model_name(None, &canonical);

        crate::inference::skippy::register_local_source_policy(
            &runtime_model_name,
            "strict-dynamic",
            true,
        );

        assert_ne!(runtime_model_name, path.to_string_lossy());
        assert!(crate::inference::skippy::local_source_required_for_model(
            &runtime_model_name,
            Some("strict-dynamic")
        ));
    }
}
