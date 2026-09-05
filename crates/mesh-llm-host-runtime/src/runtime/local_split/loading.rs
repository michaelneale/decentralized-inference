use super::attestation::{split_stage_source_is_ready, strict_ready_status_matches};
use super::{
    LocalRuntimeBackendHandle, LocalRuntimeModelHandle, OpenAiGuardrailPolicyHandle,
    RuntimeSliceStagePlan, SplitGenerationCleanup, SplitRuntimeGenerationHandle,
    SplitTopologyGeneration, alloc_local_port, pinned_stage_device, split_participant_labels,
    split_stage_plan_labels, stop_split_generation,
};
use crate::inference::skippy;
use crate::mesh;
use crate::models;
use crate::plugin;
use crate::runtime::local::skippy_native_model_open_event_reporter;
use crate::runtime::local_package::{
    split_node_labels, split_participant_set_hash, split_topology_hash,
};
use crate::runtime::survey;
use anyhow::{Context, Result};
use mesh_llm_events::{OutputEvent, emit_event};
use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig};
use skippy_runtime::ActivationBoundaryDesc;
use skippy_server::serving_hooks::SharedModelServingHooksFactory;
use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(super) const MIN_STAGE_SOURCE_PREPARE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const STAGE_SOURCE_PREPARE_ALLOWANCE: Duration = Duration::from_secs(10 * 60);
const STAGE_SOURCE_MIN_BYTES_PER_SEC: u64 = 16 * 1024 * 1024;
const DEFAULT_STAGE_STARTUP_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const DEFAULT_STAGE_READINESS_INTERVAL: Duration = Duration::from_secs(2);
pub(super) const DEFAULT_STAGE_HEALTH_INTERVAL: Duration = Duration::from_secs(30);

pub(super) async fn await_stage_startup<F, T>(
    timeout: Duration,
    future: F,
) -> std::result::Result<T, tokio::time::error::Elapsed>
where
    F: Future<Output = T>,
{
    tokio::time::timeout(timeout, future).await
}

pub(super) async fn wait_for_stage_readiness_poll(interval: Duration) {
    tokio::time::sleep(interval).await;
}

pub(super) fn stage_health_ticks(interval: Duration) -> tokio::time::Interval {
    tokio::time::interval(interval)
}

pub(super) fn peer_supports_strict_local_load(
    local_node_id: iroh::EndpointId,
    target_node_id: iroh::EndpointId,
    peers: &[mesh::PeerInfo],
) -> bool {
    target_node_id == local_node_id
        || peers
            .iter()
            .any(|peer| peer.id == target_node_id && peer.local_gguf_content_id_supported)
}

async fn ensure_peer_supports_strict_local_load(
    node: &mesh::Node,
    target_node_id: iroh::EndpointId,
    stage_id: &str,
) -> Result<()> {
    let peers = node.peers().await;
    anyhow::ensure!(
        peer_supports_strict_local_load(node.id(), target_node_id, &peers),
        "stage {stage_id} peer {} no longer advertises content-addressed local GGUF support",
        target_node_id.fmt_short()
    );
    Ok(())
}

pub(super) struct SplitGenerationLoadSpec<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) mesh_config: &'a plugin::MeshConfig,
    pub(super) model_ref: &'a str,
    pub(super) config_model_id: Option<&'a str>,
    pub(super) runtime_profile: &'a str,
    pub(super) model_path: &'a Path,
    pub(super) package: &'a skippy::SkippyPackageIdentity,
    pub(super) generation: &'a SplitTopologyGeneration,
    pub(super) projector_path: Option<String>,
    pub(super) ctx_size: u32,
    pub(super) compact_meta: &'a models::gguf::GgufCompactMeta,
    pub(super) capacity_budget_bytes: Option<u64>,
    pub(super) pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    pub(super) device_override: Option<&'a str>,
    pub(super) slots: usize,
    pub(super) cache_type_k_override: Option<&'a str>,
    pub(super) cache_type_v_override: Option<&'a str>,
    pub(super) n_batch_override: Option<u32>,
    pub(super) n_ubatch_override: Option<u32>,
    pub(super) flash_attention_override: FlashAttentionType,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) skippy_telemetry: skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: survey::SurveyTelemetry,
    pub(super) serving_hooks_factory: Option<SharedModelServingHooksFactory>,
    pub(super) local_source_required: bool,
}

pub(super) struct SplitGenerationLoadSettings<'a> {
    pub(super) stage0: &'a RuntimeSliceStagePlan,
    pub(super) runtime_options: skippy_server::EmbeddedRuntimeOptions,
    pub(super) embedded_openai: skippy::ResolvedEmbeddedOpenAiArgs,
    pub(super) load_mode: LoadMode,
    pub(super) startup_timeout: Duration,
    pub(super) readiness_interval: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StageLifecycleIntervals {
    pub(super) startup_timeout: Duration,
    pub(super) readiness_interval: Duration,
    pub(super) health_interval: Duration,
}

pub(super) fn configured_stage_lifecycle_intervals(
    mesh_config: &plugin::MeshConfig,
    config_model_id: Option<&str>,
) -> StageLifecycleIntervals {
    let model = config_model_id.and_then(|selector| model_skippy_config(mesh_config, selector));
    let defaults = mesh_config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.skippy.as_ref());
    StageLifecycleIntervals {
        startup_timeout: Duration::from_millis(
            model
                .and_then(|config| config.lifecycle_startup_timeout_ms)
                .or_else(|| defaults.and_then(|config| config.lifecycle_startup_timeout_ms))
                .unwrap_or(DEFAULT_STAGE_STARTUP_TIMEOUT.as_millis() as u64),
        ),
        readiness_interval: Duration::from_millis(
            model
                .and_then(|config| config.lifecycle_readiness_interval_ms)
                .or_else(|| defaults.and_then(|config| config.lifecycle_readiness_interval_ms))
                .unwrap_or(DEFAULT_STAGE_READINESS_INTERVAL.as_millis() as u64),
        ),
        health_interval: Duration::from_millis(
            model
                .and_then(|config| config.lifecycle_health_interval_ms)
                .or_else(|| defaults.and_then(|config| config.lifecycle_health_interval_ms))
                .unwrap_or(DEFAULT_STAGE_HEALTH_INTERVAL.as_millis() as u64),
        ),
    }
}

fn model_skippy_config<'a>(
    mesh_config: &'a plugin::MeshConfig,
    model_ref: &str,
) -> Option<&'a plugin::SkippyConfig> {
    mesh_config
        .models
        .iter()
        .find(|model| model.model == model_ref)
        .and_then(|model| model.skippy.as_ref())
}

pub(super) async fn load_split_runtime_generation(
    spec: SplitGenerationLoadSpec<'_>,
) -> Result<SplitRuntimeGenerationHandle> {
    let mut cleanup_on_error = false;
    let result = Box::pin(load_split_runtime_generation_inner(
        &spec,
        &mut cleanup_on_error,
    ))
    .await;
    if let Err(error) = &result
        && cleanup_on_error
    {
        tracing::warn!(
            model_ref = spec.model_ref,
            topology_id = %spec.generation.topology_id,
            run_id = %spec.generation.run_id,
            generation = spec.generation.generation,
            error = %error,
            "cleaning up split runtime generation after failed load"
        );
        stop_split_generation(spec.node, spec.generation, spec.generation.generation).await;
    }
    result
}

pub(super) async fn load_split_runtime_generation_inner(
    spec: &SplitGenerationLoadSpec<'_>,
    cleanup_on_error: &mut bool,
) -> Result<SplitRuntimeGenerationHandle> {
    let settings = split_generation_load_settings(spec).await?;
    anyhow::ensure!(
        settings.stage0.node_id == spec.node.id(),
        "split topology stage 0 moved to {}; local coordinator is {}",
        settings.stage0.node_id.fmt_short(),
        spec.node.id().fmt_short()
    );

    claim_split_coordinator_lease(spec.node, spec.model_ref, spec.package, spec.generation).await?;

    let mut ready_by_stage: HashMap<String, skippy::StageStatusSnapshot> = HashMap::new();
    let mut downstream: Option<skippy::StagePeerDescriptor> = None;

    if settings.load_mode == LoadMode::LayerPackage {
        spec.node
            .record_stage_topology(split_stage_topology_instance(
                &spec.generation.topology_id,
                &spec.generation.run_id,
                spec.model_ref,
                spec.package,
                &spec.generation.stages,
                None,
                &ready_by_stage,
            ))
            .await;
    }

    let stage0_return_port = alloc_local_port().await?;
    let stage0_return_endpoint = format!("127.0.0.1:{stage0_return_port}");
    spec.node
        .register_stage_transport_alias(
            &spec.generation.topology_id,
            &spec.generation.run_id,
            &settings.stage0.stage_id,
            stage0_return_endpoint.clone(),
        )
        .await;
    let downstream = Box::pin(load_downstream_split_runtime_stages(
        spec,
        &settings,
        cleanup_on_error,
        &mut ready_by_stage,
        &mut downstream,
        &stage0_return_endpoint,
    ))
    .await?;
    let downstream_endpoint = if downstream.node_id == Some(spec.node.id()) {
        downstream.endpoint.clone()
    } else {
        spec.node
            .ensure_stage_transport_bridge(
                downstream
                    .node_id
                    .context("downstream split stage is missing node id")?,
                spec.generation.topology_id.clone(),
                spec.generation.run_id.clone(),
                downstream.stage_id.clone(),
            )
            .await?
    };
    let first_downstream_stage_id = downstream.stage_id.clone();
    let stage0_status_load = split_runtime_stage_load_request(
        spec,
        &settings,
        settings.stage0,
        Some(downstream.clone()),
        &stage0_return_endpoint,
        &ready_by_stage,
    )?;
    let runtime_options = stage0_runtime_options(
        spec,
        &settings,
        &downstream,
        &downstream_endpoint,
        &stage0_return_endpoint,
    )
    .await?;
    let media_capability_evidence = models::runtime_media_capability_evidence(
        runtime_options
            .config
            .projector_path
            .as_deref()
            .map(std::path::PathBuf::from),
    )
    .await;
    let node_for_hook = spec.node.clone();
    let model_ref = spec.model_ref.to_string();
    let reporter_model_ref = model_ref.clone();
    let skippy_telemetry = spec.skippy_telemetry.clone();
    let guardrail_telemetry = spec.survey_telemetry.clone();
    let serving_hooks_factory = spec.serving_hooks_factory.clone();
    let openai_guardrails =
        skippy::skippy_openai_guardrails_for_policy_handle(spec.openai_guardrail_policy.clone());
    let _ = emit_event(OutputEvent::ModelLoading {
        model: model_ref.clone(),
        source: None,
    });
    let handle = tokio::task::spawn_blocking(move || {
        skippy::SkippyModelHandle::load_stage0_runtime_options_with_openai_args_and_open_events(
            runtime_options,
            settings.embedded_openai.clone(),
            Some(skippy::MeshAutoHookPolicy::new(node_for_hook)),
            skippy_telemetry,
            Some(skippy_native_model_open_event_reporter(reporter_model_ref)),
            skippy::SkippyOpenAiGuardrailOptions::new(Some(openai_guardrails), guardrail_telemetry),
            serving_hooks_factory,
        )
    })
    .await
    .context("join load skippy stage0 config task")??;
    let stage0_output = handle
        .output_activation_boundary()
        .context("stage 0 graph did not expose its output activation boundary")?;
    let first_downstream_status = ready_by_stage
        .get(&first_downstream_stage_id)
        .with_context(|| format!("missing ready status for {first_downstream_stage_id}"))?;
    validate_activation_edge(
        &settings.stage0.stage_id,
        stage0_output,
        &first_downstream_status.stage_id,
        required_boundary(
            first_downstream_status.input_activation_boundary,
            &first_downstream_status.stage_id,
            "input",
        )?,
    )?;
    let mut stage0_status =
        mesh::stage_status_from_load(&stage0_status_load, skippy::StageRuntimeState::Ready);
    stage0_status.bind_addr = stage0_return_endpoint.clone();
    stage0_status.output_activation_boundary = Some(stage0_output);
    stage0_status.error = None;
    spec.node
        .record_stage_status(Some(spec.node.id()), stage0_status)
        .await;
    let _ = emit_event(OutputEvent::ModelLoaded {
        model: model_ref,
        bytes: None,
    });
    let http = handle.start_http(alloc_local_port().await?)?;
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();
    let capabilities = models::runtime_verified_model_capabilities(
        spec.model_ref,
        spec.model_path,
        media_capability_evidence,
    );

    spec.node
        .activate_stage_topology(split_stage_topology_instance(
            &spec.generation.topology_id,
            &spec.generation.run_id,
            spec.model_ref,
            spec.package,
            &spec.generation.stages,
            Some(&stage0_return_endpoint),
            &ready_by_stage,
        ))
        .await;

    Ok(SplitRuntimeGenerationHandle {
        loaded_name: spec.model_ref.to_string(),
        handle: LocalRuntimeModelHandle {
            port: http.port(),
            backend: "skippy".into(),
            context_length: spec.ctx_size,
            slots: spec.slots,
            capabilities,
            inner: LocalRuntimeBackendHandle::Skippy {
                model: handle,
                http,
                _death_tx: death_tx,
            },
        },
        death_rx,
        cleanup: Some(SplitGenerationCleanup {
            generation: spec.generation.clone(),
        }),
        coordinator_rx: None,
        coordinator_task: None,
    })
}

async fn stage0_runtime_options(
    spec: &SplitGenerationLoadSpec<'_>,
    settings: &SplitGenerationLoadSettings<'_>,
    downstream: &skippy::StagePeerDescriptor,
    downstream_endpoint: &str,
    stage0_return_endpoint: &str,
) -> Result<skippy_server::EmbeddedRuntimeOptions> {
    let mut runtime_options = settings.runtime_options.clone();
    runtime_options.config.run_id = spec.generation.run_id.clone();
    runtime_options.config.topology_id = spec.generation.topology_id.clone();
    runtime_options.config.model_id = spec.model_ref.to_string();
    runtime_options.config.package_ref = Some(spec.package.package_ref.clone());
    runtime_options.config.manifest_sha256 = Some(spec.package.manifest_sha256.clone());
    let verified_stage0_model_path = if spec.local_source_required {
        let mut stage0_load = split_runtime_stage_load_request(
            spec,
            settings,
            settings.stage0,
            Some(downstream.clone()),
            stage0_return_endpoint,
            &HashMap::new(),
        )?;
        let stage0_load = tokio::task::spawn_blocking(move || {
            let verified = skippy::apply_verified_local_source(&mut stage0_load)?;
            anyhow::ensure!(verified, "local-required stage 0 was not content-verified");
            anyhow::Ok(stage0_load)
        })
        .await
        .context("join verify local-required stage 0 source task")??;
        Some(
            stage0_load
                .model_path
                .context("verified local-required stage 0 is missing its worker-local path")?,
        )
    } else {
        None
    };
    let effective_model_path = verified_stage0_model_path.unwrap_or_else(|| {
        stage_load_model_path(
            settings.load_mode.clone(),
            &spec.package.package_ref,
            &spec.package.source_model_path,
        )
    });
    runtime_options.config.source_model_path = Some(effective_model_path.clone());
    runtime_options.config.source_model_sha256 = Some(spec.package.source_model_sha256.clone());
    runtime_options.config.source_model_bytes = Some(spec.package.source_model_bytes);
    runtime_options.config.materialized_path = None;
    runtime_options.config.materialized_pinned = false;
    runtime_options.config.model_path = Some(effective_model_path);
    if runtime_options.config.projector_path.is_none() {
        runtime_options.config.projector_path = spec.projector_path.clone();
    }
    runtime_options.config.stage_id = settings.stage0.stage_id.clone();
    runtime_options.config.stage_index = settings.stage0.stage_index;
    runtime_options.config.layer_start = settings.stage0.layer_start;
    runtime_options.config.layer_end = settings.stage0.layer_end;
    runtime_options.config.ctx_size = spec.ctx_size;
    runtime_options.config.lane_count = spec.slots as u32;
    runtime_options.config.filter_tensors_on_load = true;
    apply_split_generation_pinned_device(
        &mut runtime_options.config,
        spec.pinned_gpu,
        spec.device_override,
    );
    runtime_options.config.load_mode = settings.load_mode.clone();
    runtime_options.config.bind_addr = stage0_return_endpoint.to_string();
    runtime_options.config.upstream = None;
    runtime_options.config.downstream = Some(PeerConfig {
        stage_id: downstream.stage_id.clone(),
        stage_index: downstream.stage_index,
        endpoint: downstream_endpoint.to_string(),
    });
    Ok(runtime_options)
}

pub(super) async fn load_downstream_split_runtime_stages(
    spec: &SplitGenerationLoadSpec<'_>,
    settings: &SplitGenerationLoadSettings<'_>,
    cleanup_on_error: &mut bool,
    ready_by_stage: &mut HashMap<String, skippy::StageStatusSnapshot>,
    downstream: &mut Option<skippy::StagePeerDescriptor>,
    stage0_return_endpoint: &str,
) -> Result<skippy::StagePeerDescriptor> {
    for stage in spec.generation.stages.iter().skip(1).rev() {
        *cleanup_on_error = true;
        let load = split_runtime_stage_load_request(
            spec,
            settings,
            stage,
            downstream.clone(),
            stage0_return_endpoint,
            ready_by_stage,
        )?;
        if load.local_source_required {
            let inventory = query_stage_inventory(spec.node, stage.node_id, &load)
                .await
                .with_context(|| {
                    stage_control_unreachable_message(&load.stage_id, stage.node_id)
                })?;
            anyhow::ensure!(
                split_stage_source_is_ready(&inventory, &load),
                "stage {} no longer has the exact local GGUF content required by this topology",
                stage.stage_id
            );
        } else {
            prepare_split_stage(spec.node, stage.node_id, load.clone()).await?;
            wait_for_split_stage_source(
                spec.node,
                stage.node_id,
                &load,
                stage_source_prepare_timeout(
                    spec.model_path,
                    spec.package,
                    stage,
                    downstream.is_none(),
                )?,
                settings.readiness_interval,
            )
            .await
            .with_context(|| {
                format!(
                    "prepare split stage {} on {}",
                    stage.stage_id,
                    stage.node_id.fmt_short()
                )
            })?;
        }
        if load.local_source_required {
            ensure_peer_supports_strict_local_load(spec.node, stage.node_id, &load.stage_id)
                .await?;
        }
        let load_request = if load.local_source_required {
            skippy::StageControlRequest::LoadLocal(load.clone())
        } else {
            skippy::StageControlRequest::Load(load.clone())
        };
        let response = await_stage_startup(settings.startup_timeout, async {
            if stage.node_id == spec.node.id() {
                spec.node.send_local_stage_control(load_request).await
            } else {
                spec.node
                    .send_stage_control(stage.node_id, load_request)
                    .await
            }
        })
        .await
        .context("split stage startup timed out")?
        .with_context(|| {
            format!(
                "load split stage {} on {}",
                stage.stage_id,
                stage.node_id.fmt_short()
            )
        })?;
        let skippy::StageControlResponse::Ready(ready) = response else {
            anyhow::bail!(
                "unexpected status response while loading {}",
                stage.stage_id
            );
        };
        anyhow::ensure!(
            ready.accepted,
            "stage {} rejected load: {}",
            stage.stage_id,
            ready.error.unwrap_or_else(|| "unknown error".to_string())
        );
        if load.local_source_required {
            anyhow::ensure!(
                strict_ready_status_matches(&ready.status, &load),
                "stage {} did not attest the expected local GGUF content after load",
                stage.stage_id
            );
        }
        if let Some(consumer) = downstream.as_ref() {
            let consumer_status = ready_by_stage
                .get(&consumer.stage_id)
                .with_context(|| format!("missing ready status for {}", consumer.stage_id))?;
            validate_activation_edge(
                &ready.status.stage_id,
                required_boundary(
                    ready.status.output_activation_boundary,
                    &ready.status.stage_id,
                    "output",
                )?,
                &consumer_status.stage_id,
                required_boundary(
                    consumer_status.input_activation_boundary,
                    &consumer_status.stage_id,
                    "input",
                )?,
            )?;
        }
        *downstream = Some(skippy::StagePeerDescriptor {
            stage_id: stage.stage_id.clone(),
            stage_index: stage.stage_index,
            endpoint: ready.status.bind_addr.clone(),
            node_id: Some(stage.node_id),
        });
        ready_by_stage.insert(stage.stage_id.clone(), ready.status);
    }

    downstream
        .clone()
        .context("split topology missing downstream stage")
}

fn required_boundary(
    descriptor: Option<ActivationBoundaryDesc>,
    stage_id: &str,
    edge: &str,
) -> Result<ActivationBoundaryDesc> {
    descriptor.with_context(|| {
        format!("stage {stage_id} did not advertise its graph-observed {edge} activation boundary")
    })
}

fn validate_activation_edge(
    producer_id: &str,
    producer: ActivationBoundaryDesc,
    consumer_id: &str,
    consumer: ActivationBoundaryDesc,
) -> Result<()> {
    producer
        .raw_f32_width("output")
        .with_context(|| format!("invalid graph boundary advertised by producer {producer_id}"))?;
    consumer
        .raw_f32_width("input")
        .with_context(|| format!("invalid graph boundary advertised by consumer {consumer_id}"))?;
    anyhow::ensure!(
        producer == consumer,
        "activation boundary mismatch between producer {producer_id} and consumer {consumer_id}: producer={producer:?} consumer={consumer:?}"
    );
    Ok(())
}

pub(super) fn stage_source_prepare_timeout(
    model_path: &Path,
    package: &skippy::SkippyPackageIdentity,
    stage: &RuntimeSliceStagePlan,
    include_output: bool,
) -> Result<Duration> {
    let assigned_bytes = if skippy::is_layer_package_ref(&package.package_ref)
        && !skippy::is_package_v2_ref(&package.package_ref)
    {
        crate::models::artifact_transfer::required_stage_package_bytes(
            model_path,
            &package.package_ref,
            &package.manifest_sha256,
            crate::models::artifact_transfer::StageArtifactSelection {
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                include_embeddings: stage.layer_start == 0,
                include_output,
                include_projectors: stage.layer_start == 0,
            },
        )?
    } else if package.layer_weight_bytes.len() == package.layer_count as usize {
        package
            .layer_weight_bytes
            .get(stage.layer_start as usize..stage.layer_end as usize)
            .unwrap_or_default()
            .iter()
            .copied()
            .fold(0_u64, u64::saturating_add)
    } else {
        let package_layers = u64::from(package.layer_count.max(1));
        let stage_layers = u64::from(stage.layer_end.saturating_sub(stage.layer_start).max(1));
        package
            .source_model_bytes
            .saturating_mul(stage_layers)
            .div_ceil(package_layers)
    };
    let transfer_secs = assigned_bytes.div_ceil(STAGE_SOURCE_MIN_BYTES_PER_SEC);
    Ok(Duration::from_secs(transfer_secs)
        .saturating_add(STAGE_SOURCE_PREPARE_ALLOWANCE)
        .max(MIN_STAGE_SOURCE_PREPARE_TIMEOUT))
}

pub(super) fn split_runtime_stage_load_request(
    spec: &SplitGenerationLoadSpec<'_>,
    settings: &SplitGenerationLoadSettings<'_>,
    stage: &RuntimeSliceStagePlan,
    downstream: Option<skippy::StagePeerDescriptor>,
    stage0_return_endpoint: &str,
    ready_by_stage: &HashMap<String, skippy::StageStatusSnapshot>,
) -> Result<skippy::StageLoadRequest> {
    let resolved_config = &settings.runtime_options.config;
    let upstream = if downstream.is_none() {
        split_runtime_stage_upstream(spec, stage0_return_endpoint)
    } else {
        None
    };
    let admission = spec
        .generation
        .admissions
        .get(&stage.stage_id)
        .with_context(|| {
            format!(
                "stage {} is missing native admission evidence",
                stage.stage_id
            )
        })?
        .clone();
    Ok(skippy::StageLoadRequest {
        topology_id: spec.generation.topology_id.clone(),
        run_id: spec.generation.run_id.clone(),
        model_id: spec.model_ref.to_string(),
        runtime_profile: Some(spec.runtime_profile.to_string()),
        backend: "skippy".to_string(),
        package_ref: spec.package.package_ref.clone(),
        manifest_sha256: spec.package.manifest_sha256.clone(),
        stage_id: stage.stage_id.clone(),
        stage_index: stage.stage_index,
        layer_start: stage.layer_start,
        layer_end: stage.layer_end,
        admission,
        participant_set_hash: split_participant_set_hash(&spec.generation.participants),
        topology_hash: split_topology_hash(
            &spec.generation.stages,
            &spec.generation.admissions,
            spec.generation.activation_codec,
        ),
        activation_codec: spec.generation.activation_codec,
        topology_stages: spec
            .generation
            .stages
            .iter()
            .map(|planned| skippy::StageTopologyStageDescriptor {
                stage_id: planned.stage_id.clone(),
                stage_index: planned.stage_index,
                node_id: planned.node_id,
                layer_start: planned.layer_start,
                layer_end: planned.layer_end,
                bind_addr: if planned.stage_id == stage.stage_id {
                    "127.0.0.1:0".to_string()
                } else if planned.stage_index == 0 {
                    stage0_return_endpoint.to_string()
                } else {
                    ready_by_stage
                        .get(&planned.stage_id)
                        .map(|status| status.bind_addr.clone())
                        .unwrap_or_else(|| "127.0.0.1:0".to_string())
                },
            })
            .collect(),
        model_path: (!spec.local_source_required).then(|| {
            stage_load_model_path(
                settings.load_mode.clone(),
                &spec.package.package_ref,
                &spec.package.source_model_path,
            )
        }),
        source_model_bytes: Some(spec.package.source_model_bytes),
        source_model_sha256: Some(spec.package.source_model_sha256.clone()),
        local_source_required: spec.local_source_required,
        projector_path: (!spec.local_source_required || stage.stage_index == 0)
            .then(|| resolved_config.projector_path.clone())
            .flatten(),
        projector_use_gpu: resolved_config.projector_use_gpu,
        media_marker: resolved_config.media_marker.clone(),
        image_min_tokens: resolved_config.image_min_tokens,
        image_max_tokens: resolved_config.image_max_tokens,
        batch_max_tokens: resolved_config.batch_max_tokens,
        glm_dsa_policy: resolved_config.glm_dsa_policy,
        generation_signal_window: resolved_config.generation_signal_window,
        selected_device: None,
        bind_addr: "127.0.0.1:0".to_string(),
        ctx_size: spec.ctx_size,
        lane_count: spec.slots as u32,
        continuous_batching: settings.embedded_openai.continuous_batching,
        n_batch: resolved_config.n_batch,
        n_ubatch: resolved_config.n_ubatch,
        n_gpu_layers: resolved_config.n_gpu_layers,
        mmap: resolved_config.mmap,
        mlock: resolved_config.mlock,
        cache_type_k: resolved_config.cache_type_k.clone(),
        cache_type_v: resolved_config.cache_type_v.clone(),
        flash_attn_type: resolved_config.flash_attn_type,
        runtime_settings: skippy::StageLoadRuntimeSettings {
            repack: resolved_config.repack,
            op_offload: resolved_config.op_offload,
            no_host_buffer: resolved_config.no_host_buffer,
            check_tensors: resolved_config.check_tensors,
            direct_io: resolved_config.direct_io,
            main_gpu: resolved_config.main_gpu,
            split_mode: resolved_config.split_mode,
            kv_offload: resolved_config.kv_offload,
            kv_unified: resolved_config.kv_unified,
            swa_full: resolved_config.swa_full,
            cache_idle_slots: resolved_config.cache_idle_slots,
        },
        native_mtp_enabled: resolved_config.native_mtp_enabled,
        shutdown_generation: spec.generation.generation,
        coordinator_term: spec.generation.coordinator_term,
        coordinator_id: Some(spec.node.id()),
        lease_until_unix_ms: spec.generation.lease_until_unix_ms,
        load_mode: settings.load_mode.clone(),
        upstream,
        downstream,
    })
}

pub(super) fn split_runtime_stage_upstream(
    spec: &SplitGenerationLoadSpec<'_>,
    stage0_return_endpoint: &str,
) -> Option<skippy::StagePeerDescriptor> {
    let stage0 = spec.generation.stages.first()?;
    Some(skippy::StagePeerDescriptor {
        stage_id: stage0.stage_id.clone(),
        stage_index: stage0.stage_index,
        endpoint: stage0_return_endpoint.to_string(),
        node_id: Some(stage0.node_id),
    })
}

pub(super) async fn split_generation_load_settings<'a>(
    spec: &'a SplitGenerationLoadSpec<'_>,
) -> Result<SplitGenerationLoadSettings<'a>> {
    let stage0 = spec
        .generation
        .stages
        .first()
        .context("split topology did not produce stage 0")?;
    let load_mode = split_generation_load_mode(spec.package);
    let mut resolved = skippy::resolve_skippy_config_for_selector(
        skippy::SkippyConfigResolveRequest {
            mesh_config: spec.mesh_config,
            model_id: spec.model_ref,
            model_path: spec.model_path,
            model_bytes: spec.package.source_model_bytes,
            allocatable_memory_bytes: split_allocatable_memory_bytes(spec),
            request_defaults: None,
            package_generation: spec.package.generation.as_ref(),
            // Split stage load uses the compact metadata scanned during planning
            // so the resolver guards both the size-tiered default and the family
            // K/V default exactly like the split planner does.
            compact_meta: Some(spec.compact_meta),
        },
        spec.config_model_id,
    )?;
    resolved.materialize_projector_url().await?;
    resolved.model_fit.ctx_size = spec.ctx_size;
    resolved.throughput.parallel = spec.slots;
    if let Some(cache_type_k) = spec.cache_type_k_override {
        resolved.model_fit.cache_type_k = cache_type_k.to_string();
    }
    if let Some(cache_type_v) = spec.cache_type_v_override {
        resolved.model_fit.cache_type_v = cache_type_v.to_string();
    }
    if let Some(n_batch) = spec.n_batch_override {
        resolved.model_fit.batch = n_batch;
    }
    if let Some(n_ubatch) = spec.n_ubatch_override {
        resolved.model_fit.ubatch = n_ubatch;
    }
    if spec.flash_attention_override != FlashAttentionType::Auto {
        resolved.model_fit.flash_attention = spec.flash_attention_override;
    }
    if resolved.hardware.projector_path.is_none() {
        resolved.hardware.projector_path = spec.projector_path.as_ref().map(PathBuf::from);
    }
    if let Some(gpu) = spec.pinned_gpu {
        resolved.hardware.device = Some(gpu.backend_device.clone());
    }
    if let Some(device) = spec.device_override {
        resolved.hardware.device = Some(device.to_string());
    }
    // Stage zero replaces this placeholder with its graph-observed output
    // boundary after the native runtime has been constructed.
    let embedded_openai = resolved.to_embedded_openai_args(0, true)?;
    let lifecycle = configured_stage_lifecycle_intervals(spec.mesh_config, spec.config_model_id);
    let runtime_options = resolved.to_embedded_runtime_options(
        &spec.skippy_telemetry,
        Some(spec.package.clone()),
        load_mode.clone(),
    )?;
    tracing::info!(
        model = spec.model_ref,
        "KV cache: {} K + {} V",
        runtime_options.config.cache_type_k.to_ascii_uppercase(),
        runtime_options.config.cache_type_v.to_ascii_uppercase(),
    );
    Ok(SplitGenerationLoadSettings {
        stage0,
        runtime_options,
        embedded_openai,
        load_mode,
        startup_timeout: lifecycle.startup_timeout,
        readiness_interval: lifecycle.readiness_interval,
    })
}

pub(super) fn split_allocatable_memory_bytes(spec: &SplitGenerationLoadSpec<'_>) -> Option<u64> {
    spec.capacity_budget_bytes
        .filter(|bytes| *bytes > 0)
        .or_else(|| spec.pinned_gpu.map(|gpu| gpu.allocatable_vram_bytes()))
}

pub(super) fn apply_split_generation_pinned_device(
    config: &mut skippy_protocol::StageConfig,
    pinned_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
    device_override: Option<&str>,
) {
    if device_override.is_none()
        && let Some(gpu) = pinned_gpu
    {
        config.selected_device = Some(pinned_stage_device(gpu));
    }
}

pub(super) fn split_generation_load_mode(package: &skippy::SkippyPackageIdentity) -> LoadMode {
    if skippy::is_layer_package_ref(&package.package_ref)
        && !skippy::is_package_v2_ref(&package.package_ref)
    {
        LoadMode::LayerPackage
    } else {
        LoadMode::RuntimeSlice
    }
}

pub(super) async fn claim_split_coordinator_lease(
    node: &mesh::Node,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    generation: &SplitTopologyGeneration,
) -> Result<()> {
    let claim = split_coordinator_claim(node.id(), model_ref, package, generation);
    let required_accepts = skippy_coordinator::quorum_requirement(generation.stages.len());
    let mut accepted = 0usize;
    let mut accepted_nodes = Vec::new();
    let mut errors = Vec::new();
    tracing::info!(
        model_ref,
        topology_id = generation.topology_id,
        run_id = generation.run_id,
        generation = generation.generation,
        coordinator_term = generation.coordinator_term,
        coordinator = %node.id().fmt_short(),
        planned_stages = generation.stages.len(),
        required_accepts,
        stages = ?split_stage_plan_labels(&generation.stages),
        participants = ?split_participant_labels(&generation.participants),
        "claiming split topology coordinator lease"
    );

    for stage in &generation.stages {
        record_split_coordinator_claim_result(
            model_ref,
            generation,
            stage,
            claim_split_coordinator_stage(node, stage, claim.clone()).await,
            &mut accepted,
            &mut accepted_nodes,
            &mut errors,
        );
    }

    anyhow::ensure!(
        accepted >= required_accepts,
        "coordinator claim for {model_ref} accepted by {accepted}/{} planned stage(s), need {required_accepts}: {}",
        generation.stages.len(),
        errors.join("; ")
    );
    tracing::info!(
        model_ref,
        topology_id = generation.topology_id,
        run_id = generation.run_id,
        generation = generation.generation,
        coordinator_term = generation.coordinator_term,
        accepted,
        required_accepts,
        accepted_nodes = ?split_node_labels(&accepted_nodes),
        "split topology coordinator lease quorum reached"
    );
    Ok(())
}

pub(super) enum SplitCoordinatorClaimResult {
    Accepted,
    Rejected(String),
    Unexpected(Box<skippy::StageControlResponse>),
    Failed(anyhow::Error),
}

pub(super) fn record_split_coordinator_claim_result(
    model_ref: &str,
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    result: SplitCoordinatorClaimResult,
    accepted: &mut usize,
    accepted_nodes: &mut Vec<iroh::EndpointId>,
    errors: &mut Vec<String>,
) {
    match result {
        SplitCoordinatorClaimResult::Accepted => {
            record_claim_accepted(model_ref, generation, stage, accepted, accepted_nodes)
        }
        SplitCoordinatorClaimResult::Rejected(error) => {
            record_claim_rejected(model_ref, generation, stage, error, errors)
        }
        SplitCoordinatorClaimResult::Unexpected(response) => {
            record_claim_unexpected(model_ref, generation, stage, response, errors)
        }
        SplitCoordinatorClaimResult::Failed(err) => {
            record_claim_failed(model_ref, generation, stage, err, errors)
        }
    }
}

pub(super) fn record_claim_accepted(
    model_ref: &str,
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    accepted: &mut usize,
    accepted_nodes: &mut Vec<iroh::EndpointId>,
) {
    *accepted += 1;
    accepted_nodes.push(stage.node_id);
    tracing::debug!(
        model_ref,
        topology_id = generation.topology_id,
        generation = generation.generation,
        stage_id = stage.stage_id,
        stage_node = %stage.node_id.fmt_short(),
        "split topology coordinator claim accepted by stage"
    );
}

pub(super) fn record_claim_rejected(
    model_ref: &str,
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    error: String,
    errors: &mut Vec<String>,
) {
    tracing::warn!(
        model_ref,
        topology_id = generation.topology_id,
        generation = generation.generation,
        stage_id = stage.stage_id,
        stage_node = %stage.node_id.fmt_short(),
        error = %error,
        "split topology coordinator claim rejected by stage"
    );
    errors.push(format!(
        "{} rejected claim: {}",
        stage.node_id.fmt_short(),
        error
    ));
}

pub(super) fn record_claim_unexpected(
    model_ref: &str,
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    response: Box<skippy::StageControlResponse>,
    errors: &mut Vec<String>,
) {
    tracing::warn!(
        model_ref,
        topology_id = generation.topology_id,
        generation = generation.generation,
        stage_id = stage.stage_id,
        stage_node = %stage.node_id.fmt_short(),
        response = ?response,
        "split topology coordinator claim returned unexpected response"
    );
    errors.push(format!(
        "{} returned unexpected claim response: {response:?}",
        stage.node_id.fmt_short()
    ));
}

pub(super) fn record_claim_failed(
    model_ref: &str,
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    err: anyhow::Error,
    errors: &mut Vec<String>,
) {
    tracing::warn!(
        model_ref,
        topology_id = generation.topology_id,
        generation = generation.generation,
        stage_id = stage.stage_id,
        stage_node = %stage.node_id.fmt_short(),
        error = %err,
        "split topology coordinator claim failed for stage"
    );
    errors.push(format!(
        "{} claim failed: {err:#}",
        stage.node_id.fmt_short()
    ));
}

pub(super) async fn claim_split_coordinator_stage(
    node: &mesh::Node,
    stage: &RuntimeSliceStagePlan,
    claim: skippy::StageCoordinatorClaim,
) -> SplitCoordinatorClaimResult {
    let request = skippy::StageControlRequest::Claim(claim);
    let response = if stage.node_id == node.id() {
        node.send_local_stage_control(request).await
    } else {
        node.send_stage_control(stage.node_id, request).await
    };
    match response {
        Ok(skippy::StageControlResponse::ClaimAccepted(ack)) if ack.accepted => {
            SplitCoordinatorClaimResult::Accepted
        }
        Ok(skippy::StageControlResponse::ClaimAccepted(ack)) => {
            SplitCoordinatorClaimResult::Rejected(
                ack.error.unwrap_or_else(|| "unknown rejection".to_string()),
            )
        }
        Ok(other) => SplitCoordinatorClaimResult::Unexpected(Box::new(other)),
        Err(err) => SplitCoordinatorClaimResult::Failed(err),
    }
}

pub(super) fn split_coordinator_claim(
    coordinator_id: iroh::EndpointId,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    generation: &SplitTopologyGeneration,
) -> skippy::StageCoordinatorClaim {
    skippy::StageCoordinatorClaim {
        model_id: model_ref.to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        topology_id: generation.topology_id.clone(),
        run_id: generation.run_id.clone(),
        coordinator_id: coordinator_id.to_string(),
        coordinator_term: generation.coordinator_term,
        participant_set_hash: split_participant_set_hash(&generation.participants),
        topology_hash: split_topology_hash(
            &generation.stages,
            &generation.admissions,
            generation.activation_codec,
        ),
        lease_until_unix_ms: generation.lease_until_unix_ms,
    }
}

pub(super) fn stage_load_model_path(
    load_mode: LoadMode,
    package_ref: &str,
    source_model_path: &Path,
) -> String {
    match load_mode {
        LoadMode::LayerPackage => package_ref.to_string(),
        LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => {
            source_model_path.to_string_lossy().to_string()
        }
    }
}

pub(super) async fn prepare_split_stage(
    node: &mesh::Node,
    stage_node_id: iroh::EndpointId,
    load: skippy::StageLoadRequest,
) -> Result<()> {
    let prepare = skippy::StagePrepareRequest {
        load,
        coordinator_id: Some(node.id()),
    };
    let prepare_stage_id = prepare.load.stage_id.clone();
    let response = if stage_node_id == node.id() {
        node.send_local_stage_control(skippy::StageControlRequest::Prepare(prepare))
            .await
    } else {
        node.send_stage_control(stage_node_id, skippy::StageControlRequest::Prepare(prepare))
            .await
    }
    .with_context(|| stage_control_unreachable_message(&prepare_stage_id, stage_node_id))?;
    let skippy::StageControlResponse::PrepareAccepted(accepted) = response else {
        anyhow::bail!(
            "{}",
            stage_control_unreachable_message(&prepare_stage_id, stage_node_id)
        );
    };
    anyhow::ensure!(
        accepted.accepted,
        "{}",
        stage_source_prepare_failed_message(
            &accepted.status.stage_id,
            &accepted
                .error
                .unwrap_or_else(|| "unknown error".to_string())
        )
    );
    Ok(())
}

pub(super) async fn wait_for_split_stage_source(
    node: &mesh::Node,
    stage_node_id: iroh::EndpointId,
    load: &skippy::StageLoadRequest,
    timeout: Duration,
    readiness_interval: Duration,
) -> Result<()> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let inventory = query_stage_inventory(node, stage_node_id, load)
            .await
            .with_context(|| stage_control_unreachable_message(&load.stage_id, stage_node_id))?;
        if split_stage_source_is_ready(&inventory, load) {
            tracing::info!(
                topology_id = %load.topology_id,
                run_id = %load.run_id,
                stage_id = %load.stage_id,
                node = %stage_node_id.fmt_short(),
                "split stage source is available; loading runtime"
            );
            return Ok(());
        }
        if let Some(failed) = inventory.preparing_ranges.iter().find(|status| {
            status.stage_id == load.stage_id
                && matches!(status.state, skippy::StagePreparationState::Failed)
        }) {
            anyhow::bail!(
                "{}",
                stage_source_prepare_failed_message(
                    &load.stage_id,
                    failed.error.as_deref().unwrap_or("unknown error")
                )
            );
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "{}",
                stage_source_prepare_timeout_message(&load.stage_id, timeout)
            );
        }
        wait_for_stage_readiness_poll(readiness_interval).await;
    }
}

pub(super) fn stage_control_unreachable_message(
    stage_id: &str,
    stage_node_id: iroh::EndpointId,
) -> String {
    format!(
        "stage_control_unreachable: inventory/control request failed for stage {} on {}",
        stage_id,
        stage_node_id.fmt_short()
    )
}

pub(super) fn stage_source_prepare_failed_message(stage_id: &str, error: &str) -> String {
    format!("stage_source_prepare_failed: stage {stage_id} source prepare failed: {error}")
}

pub(super) fn stage_source_prepare_timeout_message(stage_id: &str, timeout: Duration) -> String {
    format!(
        "stage_source_prepare_timeout: timed out waiting for stage {stage_id} source availability after {timeout:?}"
    )
}

pub(super) async fn query_stage_inventory(
    node: &mesh::Node,
    stage_node_id: iroh::EndpointId,
    load: &skippy::StageLoadRequest,
) -> Result<skippy::StageLayerInventory> {
    let request = skippy::StageInventoryRequest {
        model_id: load.model_id.clone(),
        runtime_profile: load.runtime_profile.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        expected_source_model_sha256: load.source_model_sha256.clone(),
        local_source_required: load.local_source_required,
    };
    let response = if stage_node_id == node.id() {
        node.send_local_stage_control(skippy::StageControlRequest::Inventory(request))
            .await
    } else {
        node.send_stage_control(
            stage_node_id,
            skippy::StageControlRequest::Inventory(request),
        )
        .await
    }?;
    let skippy::StageControlResponse::Inventory(inventory) = response else {
        anyhow::bail!("unexpected response while querying stage inventory");
    };
    Ok(inventory)
}

pub(super) fn split_stage_topology_instance(
    topology_id: &str,
    run_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    stages: &[RuntimeSliceStagePlan],
    stage0_bind_addr: Option<&str>,
    ready_by_stage: &HashMap<String, skippy::StageStatusSnapshot>,
) -> mesh::StageTopologyInstance {
    mesh::StageTopologyInstance {
        topology_id: topology_id.to_string(),
        run_id: run_id.to_string(),
        model_id: model_ref.to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        stages: stages
            .iter()
            .map(|stage| mesh::StageAssignment {
                stage_id: stage.stage_id.clone(),
                stage_index: stage.stage_index,
                node_id: stage.node_id,
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                endpoint: mesh::StageEndpoint {
                    bind_addr: ready_by_stage
                        .get(&stage.stage_id)
                        .map(|status| status.bind_addr.clone())
                        .or_else(|| {
                            if stage.stage_index == 0 {
                                stage0_bind_addr.map(str::to_string)
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default(),
                },
            })
            .collect(),
    }
}

#[cfg(test)]
mod activation_boundary_tests {
    use super::{required_boundary, validate_activation_edge};
    use skippy_runtime::ActivationBoundaryDesc;

    fn boundary(elements_per_token: u64) -> ActivationBoundaryDesc {
        ActivationBoundaryDesc {
            version: 1,
            ggml_type: 0,
            layout: 1,
            elements_per_token,
            bytes_per_token: elements_per_token * std::mem::size_of::<f32>() as u64,
            required_frame_flags: 0,
            required_sidebands: 0,
        }
    }

    #[test]
    fn matching_graph_boundaries_form_a_valid_stage_edge() {
        validate_activation_edge("stage-0", boundary(1024), "stage-1", boundary(1024))
            .expect("identical graph boundary contracts must match");
    }

    #[test]
    fn three_stage_topology_validates_each_edge_independently() {
        let first_output = boundary(1024);
        let middle_input = boundary(1024);
        let middle_output = boundary(2048);
        let final_input = boundary(2048);

        validate_activation_edge("stage-0", first_output, "stage-1", middle_input)
            .expect("first edge must match");
        validate_activation_edge("stage-1", middle_output, "stage-2", final_input)
            .expect("second edge must match even when its contract differs from the first");

        let error = validate_activation_edge("stage-1", middle_output, "stage-2", boundary(1024))
            .expect_err("a mismatch on the second edge must block the topology");
        assert!(error.to_string().contains("stage-1"));
        assert!(error.to_string().contains("stage-2"));
    }

    #[test]
    fn graph_boundary_mismatch_blocks_topology_activation() {
        let error = validate_activation_edge("stage-0", boundary(1024), "stage-1", boundary(896))
            .expect_err("different graph boundary contracts must not connect");
        assert!(error.to_string().contains("activation boundary mismatch"));
    }

    #[test]
    fn every_graph_boundary_field_participates_in_edge_matching() {
        let producer = boundary(1024);
        let mutations = [
            ActivationBoundaryDesc {
                version: 2,
                ..producer
            },
            ActivationBoundaryDesc {
                ggml_type: 1,
                ..producer
            },
            ActivationBoundaryDesc {
                layout: 2,
                ..producer
            },
            ActivationBoundaryDesc {
                elements_per_token: 896,
                bytes_per_token: 896 * std::mem::size_of::<f32>() as u64,
                ..producer
            },
            ActivationBoundaryDesc {
                bytes_per_token: producer.bytes_per_token + 4,
                ..producer
            },
            ActivationBoundaryDesc {
                required_frame_flags: skippy_ffi::ACTIVATION_FLAG_GEMMA3N_ALTUP,
                ..producer
            },
            ActivationBoundaryDesc {
                required_sidebands: skippy_ffi::ACTIVATION_SIDEBAND_TOKEN_IDS,
                ..producer
            },
        ];

        for consumer in mutations {
            assert!(
                validate_activation_edge("stage-0", producer, "stage-1", consumer).is_err(),
                "mutated consumer contract must not activate: {consumer:?}"
            );
        }
    }

    #[test]
    fn identical_but_invalid_graph_boundaries_do_not_activate() {
        let invalid = ActivationBoundaryDesc {
            version: 99,
            ..boundary(1024)
        };
        let error = validate_activation_edge("stage-0", invalid, "stage-1", invalid)
            .expect_err("matching invalid descriptors must fail semantic validation");
        assert!(error.to_string().contains("invalid graph boundary"));
    }

    #[test]
    fn missing_graph_boundary_is_not_reconstructed_from_manifest_width() {
        let error = required_boundary(None, "stage-1", "input")
            .expect_err("generation 7 requires graph-observed boundary descriptors");
        assert!(
            error
                .to_string()
                .contains("did not advertise its graph-observed input")
        );
    }
}
