use crate::logging::{LoggingMetric, LoggingMetricsSink};
use crate::network::metrics::{
    AttemptOutcome, AttemptTarget, RequestOutcome, RequestService, RoutingTelemetrySink,
};
use crate::plugin;
use crate::system::hardware;
use anyhow::{Context, Result};
use openai_frontend::{GuardrailMode, GuardrailTelemetrySink};
use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge, Histogram, MeterProvider as _};
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

mod logging_metrics;

// Event-system telemetry ownership: new instruments for task 16 (aggregate
// privacy-safe runtime-event health metrics) live in this sibling module
// rather than growing this already-large file further. See
// `runtime_events.rs`'s module doc for the metrics-only/no-ID v1 decision
// and the existing-Skippy-telemetry scope exclusion.
pub(super) mod runtime_events;

#[cfg(test)]
#[path = "survey_exact_head_tests.rs"]
mod survey_exact_head_tests;

#[cfg(test)]
#[path = "survey_prompt_shape_tests.rs"]
mod survey_prompt_shape_tests;

const DEFAULT_SERVICE_NAME: &str = "mesh-llm";
const DEFAULT_EXPORT_INTERVAL_SECS: u64 = 15;
const DEFAULT_QUEUE_SIZE: usize = 2048;
const OTLP_ENDPOINT_ENV: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
const OTLP_METRICS_ENDPOINT_ENV: &str = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT";
const MODEL_SELECTOR_ID_DOMAIN: &[u8] = b"mesh-llm.telemetry.model-selector.v1\0";
const MODEL_SELECTOR_ID_DIGEST_BYTES: usize = 16;
#[cfg(any(debug_assertions, test))]
const TELEMETRY_ATTRIBUTE_ALLOWLIST: &[&str] = &[
    "llama_stage.verify_window.direct_return_reverse_fallback",
    "llama_stage.verify_window.direct_return_upstream_opened",
    "llama_stage.linear_proposal.source_callback_us",
    "llama_stage.linear_proposal.source_outcome",
    "llama_stage.linear_proposal.source_queue_wait_us",
    "mesh_llm.architecture",
    "mesh_llm.attempt_outcome",
    "mesh_llm.backend",
    "mesh_llm.backend_device",
    "mesh_llm.context_bucket",
    "mesh_llm.failure_reason",
    "mesh_llm.guardrail.attempt_bucket",
    "mesh_llm.guardrail.bypass_reason",
    "mesh_llm.guardrail.contract",
    "mesh_llm.guardrail.decision",
    "mesh_llm.guardrail.mode",
    "mesh_llm.guardrail.outcome",
    "mesh_llm.gpu_count",
    "mesh_llm.gpu_name",
    "mesh_llm.gpu_stable_id",
    "mesh_llm.is_soc",
    "mesh_llm.launch_kind",
    "mesh_llm.logging_artifact_capture_status",
    "mesh_llm.logging_cleanup_outcome",
    "mesh_llm.logging_terminal_outcome",
    "mesh_llm.logging_webhook_attempt_state",
    "mesh_llm.logging_webhook_delivery_outcome",
    "mesh_llm.model_selector_id",
    "mesh_llm.quantization",
    "mesh_llm.request_outcome",
    "mesh_llm.route_attempt_bucket",
    "mesh_llm.route_service",
    "mesh_llm.service_version",
    "mesh_llm.source_node_id",
    "mesh_llm.source_node_role",
    "mesh_llm.target_kind",
    "mesh_llm.target_node_id",
];

#[derive(Clone)]
pub(crate) struct SurveyTelemetry {
    inner: Option<Arc<SurveyTelemetryInner>>,
}

struct SurveyTelemetryInner {
    queue: Arc<SurveyEventQueue>,
    hardware: hardware::HardwareSurvey,
    source: SurveyTelemetrySource,
    prompt_shape_metrics: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct SurveyTelemetrySource {
    pub(crate) node_id: String,
    pub(crate) node_role: String,
}

impl SurveyTelemetrySource {
    fn key_values(&self) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("mesh_llm.source_node_role", self.node_role.clone()),
            KeyValue::new("mesh_llm.service_version", crate::VERSION),
        ];
        if let Some(node_id) = redact_stable_id(&self.node_id) {
            attrs.push(KeyValue::new("mesh_llm.source_node_id", node_id));
        }
        debug_assert_telemetry_attrs_allowlisted(&attrs);
        attrs
    }
}

#[derive(Clone, Debug)]
pub(super) struct SurveyLoadedModel {
    attrs: SurveyAttributes,
    loaded_at: Instant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SurveyLaunchKind {
    Startup,
    RuntimeLoad,
    MultiModel,
    MoeFallback,
    MoeShard,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SurveyFailureReason {
    SpawnFailed,
    HealthTimeout,
    ExitedBeforeHealthy,
    BackendProxyFailed,
    CapacityRejected,
    KnownKvCacheCrash,
    MmprojMissing,
    Other,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SurveyModelSpec<'a> {
    pub(super) model: &'a str,
    pub(super) configured_model_selector: Option<&'a str>,
    pub(super) model_path: Option<&'a Path>,
    pub(super) launch_kind: SurveyLaunchKind,
    pub(super) pinned_gpu: Option<&'a super::StartupPinnedGpuTarget>,
    pub(super) backend: Option<&'a str>,
    pub(super) context_length: Option<u64>,
}

#[derive(Clone, Debug)]
struct SurveySettings {
    service_name: String,
    endpoint: String,
    headers: std::collections::HashMap<String, String>,
    export_interval: Duration,
    queue_size: usize,
    prompt_shape_metrics: bool,
}

impl SurveySettings {
    fn from_config(config: &plugin::MeshConfig) -> Option<Self> {
        Self::from_config_with_env(config, |key| std::env::var(key).ok())
    }

    fn from_config_with_env<F>(config: &plugin::MeshConfig, env: F) -> Option<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        if config.telemetry.enabled == Some(false) {
            return None;
        }
        let endpoint = resolve_metrics_endpoint(
            &config.telemetry,
            env,
            config.telemetry.enabled == Some(true),
        )?;
        let service_name = config
            .telemetry
            .service_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(DEFAULT_SERVICE_NAME)
            .to_string();
        let headers = config
            .telemetry
            .headers
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let export_interval = Duration::from_secs(
            config
                .telemetry
                .export_interval_secs
                .unwrap_or(DEFAULT_EXPORT_INTERVAL_SECS),
        );
        let queue_size = config.telemetry.queue_size.unwrap_or(DEFAULT_QUEUE_SIZE);
        Some(Self {
            service_name,
            endpoint,
            headers,
            export_interval,
            queue_size,
            prompt_shape_metrics: config.telemetry.prompt_shape_metrics,
        })
    }
}

impl SurveyTelemetry {
    pub(crate) fn disabled() -> Self {
        Self { inner: None }
    }

    pub(crate) fn start(
        config: &plugin::MeshConfig,
        hardware: hardware::HardwareSurvey,
        source: SurveyTelemetrySource,
    ) -> Self {
        let Some(settings) = SurveySettings::from_config(config) else {
            return Self::disabled();
        };
        let queue = Arc::new(SurveyEventQueue::new(settings.queue_size));
        let recorder = match SurveyRecorder::otlp(&settings) {
            Ok(recorder) => recorder,
            Err(err) => {
                tracing::warn!("disabling telemetry OTLP metrics exporter: {err:#}");
                return Self::disabled();
            }
        };
        spawn_survey_worker(queue.clone(), recorder);
        Self {
            inner: Some(Arc::new(SurveyTelemetryInner {
                queue,
                hardware,
                source,
                prompt_shape_metrics: settings.prompt_shape_metrics,
            })),
        }
    }

    pub(crate) fn routing_sink(&self) -> Option<Arc<dyn RoutingTelemetrySink>> {
        self.inner.as_ref()?;
        Some(Arc::new(self.clone()))
    }

    pub(crate) fn guardrail_sink(&self) -> Option<Arc<dyn GuardrailTelemetrySink>> {
        self.inner.as_ref()?;
        Some(Arc::new(self.clone()))
    }

    pub(crate) fn logging_sink(&self) -> Option<Arc<dyn LoggingMetricsSink>> {
        self.inner.as_ref()?;
        Some(Arc::new(self.clone()))
    }

    pub(super) fn model(&self, spec: SurveyModelSpec<'_>) -> SurveyLoadedModel {
        let attrs = if let Some(inner) = self.inner.as_ref() {
            SurveyAttributes::from_spec(spec, &inner.hardware)
        } else {
            SurveyAttributes::from_disabled_spec(spec)
        };
        SurveyLoadedModel {
            attrs,
            loaded_at: Instant::now(),
        }
    }

    pub(super) fn record_launch_success(&self, model: &SurveyLoadedModel, duration: Duration) {
        self.emit(SurveyEvent::LaunchSuccess {
            attrs: model.attrs.clone(),
            duration_ms: duration.as_secs_f64() * 1000.0,
        });
    }

    pub(super) fn record_launch_failure(
        &self,
        spec: SurveyModelSpec<'_>,
        duration: Duration,
        reason: SurveyFailureReason,
    ) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        self.emit(SurveyEvent::LaunchFailure {
            attrs: SurveyAttributes::from_spec(spec, &inner.hardware),
            duration_ms: duration.as_secs_f64() * 1000.0,
            reason,
        });
    }

    pub(super) fn record_unload(&self, model: &SurveyLoadedModel) {
        self.emit(SurveyEvent::Unload {
            attrs: model.attrs.clone(),
            uptime_s: model.loaded_at.elapsed().as_secs_f64(),
        });
    }

    pub(super) fn record_unexpected_exit(&self, model: &SurveyLoadedModel) {
        self.emit(SurveyEvent::UnexpectedExit {
            attrs: model.attrs.clone(),
            uptime_s: model.loaded_at.elapsed().as_secs_f64(),
        });
    }

    fn emit(&self, event: SurveyEvent) {
        if let Some(inner) = self.inner.as_ref() {
            inner.queue.push(event);
        }
    }
}

impl GuardrailTelemetrySink for SurveyTelemetry {
    fn record_decision(
        &self,
        mode: GuardrailMode,
        contract: Option<&'static str>,
        decision: &'static str,
        bypass_reason: Option<&'static str>,
    ) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        self.emit(SurveyEvent::GuardrailDecision {
            attrs: GuardrailDecisionAttributes {
                source: inner.source.clone(),
                mode: guardrail_mode_label(mode),
                contract: match contract {
                    Some(value) => match guardrail_contract_attr(value) {
                        Some(label) => Some(label),
                        None => return,
                    },
                    None => None,
                },
                decision: match guardrail_decision_attr(decision) {
                    Some(value) => value,
                    None => return,
                },
                bypass_reason: match bypass_reason {
                    Some(value) => match guardrail_bypass_reason_attr(value) {
                        Some(label) => Some(label),
                        None => return,
                    },
                    None => None,
                },
            },
        });
    }

    fn record_outcome(
        &self,
        mode: GuardrailMode,
        contract: Option<&'static str>,
        outcome: &'static str,
        attempt_bucket: Option<&'static str>,
    ) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        self.emit(SurveyEvent::GuardrailOutcome {
            attrs: GuardrailOutcomeAttributes {
                source: inner.source.clone(),
                mode: guardrail_mode_label(mode),
                contract: match contract {
                    Some(value) => match guardrail_contract_attr(value) {
                        Some(label) => Some(label),
                        None => return,
                    },
                    None => None,
                },
                outcome: match guardrail_outcome_attr(outcome) {
                    Some(value) => value,
                    None => return,
                },
                attempt_bucket: match attempt_bucket {
                    Some(value) => match guardrail_attempt_bucket_attr(value) {
                        Some(label) => Some(label),
                        None => return,
                    },
                    None => None,
                },
            },
        });
    }
}

impl RoutingTelemetrySink for SurveyTelemetry {
    fn observe_inflight_requests(&self, current: u64) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        self.emit(SurveyEvent::InflightRequests {
            source: inner.source.clone(),
            current,
        });
    }

    fn record_model_request(&self, model: Option<&str>, attempts: usize, outcome: RequestOutcome) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        self.emit(SurveyEvent::ModelRequest {
            attrs: RequestAttributes::from_request(model, attempts, outcome, inner.source.clone()),
        });
    }

    fn record_route_attempt(
        &self,
        model: Option<&str>,
        target: &AttemptTarget,
        outcome: AttemptOutcome,
    ) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        self.emit(SurveyEvent::RouteAttempt {
            attrs: RouteAttemptAttributes::from_attempt(
                model,
                target,
                outcome,
                inner.source.clone(),
            ),
        });
    }

    fn record_prompt_shape(
        &self,
        _model: Option<&str>,
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
        outcome: RequestOutcome,
    ) {
        if !self
            .inner
            .as_ref()
            .is_some_and(|inner| inner.prompt_shape_metrics)
        {
            return;
        }
        if prompt_tokens.is_none() && completion_tokens.is_none() {
            return;
        }
        self.emit(SurveyEvent::PromptShape {
            attrs: PromptShapeAttributes::from_outcome(outcome),
            prompt_tokens,
            completion_tokens,
        });
    }
}

pub(super) fn classify_launch_failure(err: &anyhow::Error) -> SurveyFailureReason {
    let message = format!("{err:#}").to_ascii_lowercase();
    if message.contains("capacity")
        || message.contains("fit locally")
        || message.contains("requires")
    {
        SurveyFailureReason::CapacityRejected
    } else if message.contains("mmproj") {
        SurveyFailureReason::MmprojMissing
    } else if message.contains("health") || message.contains("timeout") {
        SurveyFailureReason::HealthTimeout
    } else if message.contains("kv cache") {
        SurveyFailureReason::KnownKvCacheCrash
    } else if message.contains("proxy") {
        SurveyFailureReason::BackendProxyFailed
    } else if message.contains("exit") || message.contains("exited") {
        SurveyFailureReason::ExitedBeforeHealthy
    } else if message.contains("spawn") || message.contains("start") || message.contains("launch") {
        SurveyFailureReason::SpawnFailed
    } else {
        SurveyFailureReason::Other
    }
}

fn spawn_survey_worker(queue: Arc<SurveyEventQueue>, mut recorder: SurveyRecorder) {
    tokio::spawn(async move {
        loop {
            let events = queue.drain();
            if events.is_empty() {
                queue.notified().await;
                continue;
            }
            for event in events {
                recorder.record(event);
            }
        }
    });
}

fn resolve_metrics_endpoint<F>(
    config: &plugin::TelemetryConfig,
    env: F,
    allow_env_endpoint: bool,
) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    let configured = trimmed_nonempty(config.metrics.endpoint.as_deref())
        .map(ToOwned::to_owned)
        .or_else(|| trimmed_nonempty(config.endpoint.as_deref()).map(metrics_endpoint_from_base));
    if configured.is_some() || !allow_env_endpoint {
        return configured;
    }
    trimmed_nonempty(env(OTLP_METRICS_ENDPOINT_ENV).as_deref())
        .map(ToOwned::to_owned)
        .or_else(|| {
            trimmed_nonempty(env(OTLP_ENDPOINT_ENV).as_deref()).map(metrics_endpoint_from_base)
        })
}

fn metrics_endpoint_from_base(endpoint: &str) -> String {
    let endpoint = endpoint.trim().trim_end_matches('/');
    if endpoint.ends_with("/v1/metrics") {
        endpoint.to_string()
    } else {
        format!("{endpoint}/v1/metrics")
    }
}

fn trimmed_nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[derive(Clone, Debug)]
struct SurveyAttributes {
    model_selector_id: Option<String>,
    architecture: Option<String>,
    quantization: Option<String>,
    launch_kind: SurveyLaunchKind,
    gpu_name: Option<String>,
    gpu_stable_id: Option<String>,
    backend_device: Option<String>,
    gpu_count: u64,
    is_soc: bool,
    backend: Option<String>,
    context_length: Option<u64>,
}

impl SurveyAttributes {
    fn from_disabled_spec(spec: SurveyModelSpec<'_>) -> Self {
        Self {
            model_selector_id: pseudonymous_model_selector_id(spec.configured_model_selector),
            architecture: None,
            quantization: None,
            launch_kind: spec.launch_kind,
            gpu_name: None,
            gpu_stable_id: None,
            backend_device: None,
            gpu_count: 0,
            is_soc: false,
            backend: spec
                .backend
                .and_then(|value| trimmed_nonempty(Some(value)))
                .map(ToOwned::to_owned),
            context_length: spec.context_length,
        }
    }

    fn from_spec(spec: SurveyModelSpec<'_>, hardware: &hardware::HardwareSurvey) -> Self {
        let gpu = spec
            .pinned_gpu
            .and_then(|pinned| hardware.gpus.iter().find(|gpu| gpu.index == pinned.index))
            .or_else(|| hardware.gpus.first());
        let gpu_name = gpu
            .map(|gpu| gpu.display_name.as_str())
            .or(hardware.gpu_name.as_deref())
            .and_then(|value| trimmed_nonempty(Some(value)))
            .map(ToOwned::to_owned);
        let stable_id = spec
            .pinned_gpu
            .map(|gpu| gpu.stable_id.as_str())
            .or_else(|| gpu.and_then(|gpu| gpu.stable_id.as_deref()));
        let backend_device = spec
            .pinned_gpu
            .map(|gpu| gpu.backend_device.as_str())
            .or_else(|| gpu.and_then(|gpu| gpu.backend_device.as_deref()))
            .and_then(|value| trimmed_nonempty(Some(value)))
            .map(ToOwned::to_owned);
        let architecture = spec
            .model_path
            .and_then(crate::models::gguf::scan_gguf_compact_meta)
            .and_then(|meta| {
                trimmed_nonempty(Some(meta.architecture.as_str())).map(ToOwned::to_owned)
            });
        let quantization = spec
            .model_path
            .and_then(|path| path.file_stem())
            .and_then(|stem| stem.to_str())
            .map(crate::models::inventory::derive_quantization_type)
            .and_then(|value| trimmed_nonempty(Some(value.as_str())).map(ToOwned::to_owned))
            .or_else(|| super::dashboard_quantization_from_model_name(spec.model));
        Self {
            model_selector_id: pseudonymous_model_selector_id(spec.configured_model_selector),
            architecture,
            quantization,
            launch_kind: spec.launch_kind,
            gpu_name,
            gpu_stable_id: stable_id.and_then(redact_stable_id),
            backend_device,
            gpu_count: u64::from(hardware.gpu_count).max(hardware.gpus.len() as u64),
            is_soc: hardware.is_soc,
            backend: spec
                .backend
                .and_then(|value| trimmed_nonempty(Some(value)))
                .map(ToOwned::to_owned),
            context_length: spec.context_length,
        }
    }

    fn key_values(&self, failure_reason: Option<SurveyFailureReason>) -> Vec<KeyValue> {
        let mut attrs = vec![
            KeyValue::new("mesh_llm.launch_kind", self.launch_kind.as_str()),
            KeyValue::new("mesh_llm.gpu_count", self.gpu_count as i64),
            KeyValue::new("mesh_llm.is_soc", self.is_soc),
            KeyValue::new("mesh_llm.service_version", crate::VERSION),
        ];
        if let Some(value) = &self.model_selector_id {
            attrs.push(KeyValue::new("mesh_llm.model_selector_id", value.clone()));
        }
        if let Some(value) = &self.architecture {
            attrs.push(KeyValue::new("mesh_llm.architecture", value.clone()));
        }
        if let Some(value) = &self.quantization {
            attrs.push(KeyValue::new("mesh_llm.quantization", value.clone()));
        }
        if let Some(value) = &self.gpu_name {
            attrs.push(KeyValue::new("mesh_llm.gpu_name", value.clone()));
        }
        if let Some(value) = &self.gpu_stable_id {
            attrs.push(KeyValue::new("mesh_llm.gpu_stable_id", value.clone()));
        }
        if let Some(value) = &self.backend_device {
            attrs.push(KeyValue::new("mesh_llm.backend_device", value.clone()));
        }
        if let Some(value) = &self.backend {
            attrs.push(KeyValue::new("mesh_llm.backend", value.clone()));
        }
        if let Some(context_length) = self.context_length {
            attrs.push(KeyValue::new(
                "mesh_llm.context_bucket",
                context_bucket(context_length),
            ));
        }
        if let Some(reason) = failure_reason {
            attrs.push(KeyValue::new("mesh_llm.failure_reason", reason.as_str()));
        }
        debug_assert_telemetry_attrs_allowlisted(&attrs);
        attrs
    }
}

fn pseudonymous_model_selector_id(configured_model_selector: Option<&str>) -> Option<String> {
    let configured_model_selector = trimmed_nonempty(configured_model_selector)?;
    let mut hasher = Sha256::new();
    hasher.update(MODEL_SELECTOR_ID_DOMAIN);
    hasher.update(configured_model_selector.as_bytes());
    let digest = hasher.finalize();
    Some(format!(
        "sha256:{}",
        hex::encode(&digest[..MODEL_SELECTOR_ID_DIGEST_BYTES])
    ))
}

fn redact_stable_id(stable_id: &str) -> Option<String> {
    let stable_id = stable_id.trim();
    if stable_id.is_empty() {
        return None;
    }
    let digest = Sha256::digest(stable_id.as_bytes());
    Some(format!("sha256:{}", hex::encode(&digest[..8])))
}

fn context_bucket(context_length: u64) -> &'static str {
    match context_length {
        0..=8192 => "<=8k",
        8193..=16_384 => "8k_16k",
        16_385..=32_768 => "16k_32k",
        32_769..=65_536 => "32k_64k",
        65_537..=131_072 => "64k_128k",
        _ => ">128k",
    }
}

#[cfg(any(debug_assertions, test))]
fn telemetry_attribute_allowed(key: &str) -> bool {
    TELEMETRY_ATTRIBUTE_ALLOWLIST.contains(&key)
}

#[cfg(debug_assertions)]
fn debug_assert_telemetry_attrs_allowlisted(attrs: &[KeyValue]) {
    for attr in attrs {
        let key = attr.key.to_string();
        debug_assert!(
            telemetry_attribute_allowed(&key),
            "OTLP telemetry attribute '{key}' must be added to the privacy-reviewed allowlist"
        );
    }
}

#[cfg(not(debug_assertions))]
fn debug_assert_telemetry_attrs_allowlisted(_attrs: &[KeyValue]) {}

impl SurveyLaunchKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::RuntimeLoad => "runtime_load",
            Self::MultiModel => "multi_model",
            Self::MoeFallback => "moe_fallback",
            Self::MoeShard => "moe_shard",
        }
    }
}

impl SurveyFailureReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::SpawnFailed => "spawn_failed",
            Self::HealthTimeout => "health_timeout",
            Self::ExitedBeforeHealthy => "exited_before_healthy",
            Self::BackendProxyFailed => "backend_proxy_failed",
            Self::CapacityRejected => "capacity_rejected",
            Self::KnownKvCacheCrash => "known_kv_cache_crash",
            Self::MmprojMissing => "mmproj_missing",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug)]
struct RequestAttributes {
    source: SurveyTelemetrySource,
    route_service: &'static str,
    request_outcome: &'static str,
    attempts: u64,
}

impl RequestAttributes {
    fn from_request(
        _model: Option<&str>,
        attempts: usize,
        outcome: RequestOutcome,
        source: SurveyTelemetrySource,
    ) -> Self {
        let (request_outcome, route_service) = match outcome {
            RequestOutcome::Success(service) => ("success", request_service_label(service)),
            RequestOutcome::Rejected(service) => ("rejected", request_service_label(service)),
            RequestOutcome::Unavailable => ("unavailable", "unavailable"),
        };
        Self {
            source,
            route_service,
            request_outcome,
            attempts: attempts as u64,
        }
    }

    fn key_values(&self) -> Vec<KeyValue> {
        let mut attrs = self.source.key_values();
        attrs.push(KeyValue::new("mesh_llm.route_service", self.route_service));
        attrs.push(KeyValue::new(
            "mesh_llm.request_outcome",
            self.request_outcome,
        ));
        attrs.push(KeyValue::new(
            "mesh_llm.route_attempt_bucket",
            request_attempt_bucket(self.attempts),
        ));
        debug_assert_telemetry_attrs_allowlisted(&attrs);
        attrs
    }
}

#[derive(Clone, Debug)]
struct PromptShapeAttributes {
    route_service: &'static str,
    request_outcome: &'static str,
}

impl PromptShapeAttributes {
    fn from_outcome(outcome: RequestOutcome) -> Self {
        let (request_outcome, route_service) = match outcome {
            RequestOutcome::Success(service) => ("success", request_service_label(service)),
            RequestOutcome::Rejected(service) => ("rejected", request_service_label(service)),
            RequestOutcome::Unavailable => ("unavailable", "unavailable"),
        };
        Self {
            route_service,
            request_outcome,
        }
    }

    fn key_values(&self) -> Vec<KeyValue> {
        let attrs = vec![
            KeyValue::new("mesh_llm.route_service", self.route_service),
            KeyValue::new("mesh_llm.request_outcome", self.request_outcome),
        ];
        debug_assert_telemetry_attrs_allowlisted(&attrs);
        attrs
    }
}

#[derive(Clone, Debug)]
struct RouteAttemptAttributes {
    source: SurveyTelemetrySource,
    target_kind: &'static str,
    target_node_id: Option<String>,
    attempt_outcome: &'static str,
}

impl RouteAttemptAttributes {
    fn from_attempt(
        _model: Option<&str>,
        target: &AttemptTarget,
        outcome: AttemptOutcome,
        source: SurveyTelemetrySource,
    ) -> Self {
        let (target_kind, target_node_id) = match target {
            AttemptTarget::Local(_) => ("local", redact_stable_id(&source.node_id)),
            AttemptTarget::Remote(node_id) => ("remote", redact_stable_id(node_id)),
            AttemptTarget::Endpoint(_) => ("endpoint", None),
        };
        Self {
            source,
            target_kind,
            target_node_id,
            attempt_outcome: attempt_outcome_label(outcome),
        }
    }

    fn key_values(&self) -> Vec<KeyValue> {
        let mut attrs = self.source.key_values();
        attrs.push(KeyValue::new("mesh_llm.target_kind", self.target_kind));
        if let Some(node_id) = &self.target_node_id {
            attrs.push(KeyValue::new("mesh_llm.target_node_id", node_id.clone()));
        }
        attrs.push(KeyValue::new(
            "mesh_llm.attempt_outcome",
            self.attempt_outcome,
        ));
        debug_assert_telemetry_attrs_allowlisted(&attrs);
        attrs
    }
}

fn request_service_label(service: RequestService) -> &'static str {
    match service {
        RequestService::Local => "local",
        RequestService::Remote => "remote",
        RequestService::Endpoint => "endpoint",
    }
}

fn request_attempt_bucket(attempts: u64) -> &'static str {
    match attempts {
        0 | 1 => "1",
        2 => "2",
        3 | 4 => "3_4",
        _ => "5_plus",
    }
}

fn attempt_outcome_label(outcome: AttemptOutcome) -> &'static str {
    match outcome {
        AttemptOutcome::Success => "success",
        AttemptOutcome::Timeout => "timeout",
        AttemptOutcome::Unavailable => "unavailable",
        AttemptOutcome::ContextOverflow => "context_overflow",
        AttemptOutcome::Rejected => "rejected",
    }
}

fn guardrail_mode_label(mode: GuardrailMode) -> &'static str {
    match mode {
        GuardrailMode::Disabled => "disabled",
        GuardrailMode::MetricsOnly => "metrics",
        GuardrailMode::Enforce => "enforce",
    }
}

fn guardrail_contract_attr(value: &'static str) -> Option<&'static str> {
    match value {
        "tools" | "structured" => Some(value),
        _ => None,
    }
}

fn guardrail_decision_attr(value: &'static str) -> Option<&'static str> {
    match value {
        "eligible" | "bypassed" | "unsupported" | "rejected" => Some(value),
        _ => None,
    }
}

fn guardrail_bypass_reason_attr(value: &'static str) -> Option<&'static str> {
    match value {
        "disabled"
        | "streaming"
        | "no_contract"
        | "unsupported_surface"
        | "reserved_collision"
        | "mixed_tools_structured" => Some(value),
        _ => None,
    }
}

fn guardrail_outcome_attr(value: &'static str) -> Option<&'static str> {
    match value {
        "pass_through" | "valid" | "retried" | "failed" | "metrics_only_failure" => Some(value),
        _ => None,
    }
}

fn guardrail_attempt_bucket_attr(value: &'static str) -> Option<&'static str> {
    match value {
        "1" | "2" | "3_plus" => Some(value),
        _ => None,
    }
}

#[derive(Clone, Debug)]
struct GuardrailDecisionAttributes {
    source: SurveyTelemetrySource,
    mode: &'static str,
    contract: Option<&'static str>,
    decision: &'static str,
    bypass_reason: Option<&'static str>,
}

impl GuardrailDecisionAttributes {
    fn key_values(&self) -> Vec<KeyValue> {
        let mut attrs = self.source.key_values();
        attrs.push(KeyValue::new("mesh_llm.guardrail.mode", self.mode));
        attrs.push(KeyValue::new("mesh_llm.guardrail.decision", self.decision));
        if let Some(contract) = self.contract {
            attrs.push(KeyValue::new("mesh_llm.guardrail.contract", contract));
        }
        if let Some(reason) = self.bypass_reason {
            attrs.push(KeyValue::new("mesh_llm.guardrail.bypass_reason", reason));
        }
        debug_assert_telemetry_attrs_allowlisted(&attrs);
        attrs
    }
}

#[derive(Clone, Debug)]
struct GuardrailOutcomeAttributes {
    source: SurveyTelemetrySource,
    mode: &'static str,
    contract: Option<&'static str>,
    outcome: &'static str,
    attempt_bucket: Option<&'static str>,
}

impl GuardrailOutcomeAttributes {
    fn key_values(&self) -> Vec<KeyValue> {
        let mut attrs = self.source.key_values();
        attrs.push(KeyValue::new("mesh_llm.guardrail.mode", self.mode));
        attrs.push(KeyValue::new("mesh_llm.guardrail.outcome", self.outcome));
        if let Some(contract) = self.contract {
            attrs.push(KeyValue::new("mesh_llm.guardrail.contract", contract));
        }
        if let Some(attempt_bucket) = self.attempt_bucket {
            attrs.push(KeyValue::new(
                "mesh_llm.guardrail.attempt_bucket",
                attempt_bucket,
            ));
        }
        debug_assert_telemetry_attrs_allowlisted(&attrs);
        attrs
    }
}

#[derive(Clone, Debug)]
enum SurveyEvent {
    LaunchSuccess {
        attrs: SurveyAttributes,
        duration_ms: f64,
    },
    LaunchFailure {
        attrs: SurveyAttributes,
        duration_ms: f64,
        reason: SurveyFailureReason,
    },
    Unload {
        attrs: SurveyAttributes,
        uptime_s: f64,
    },
    UnexpectedExit {
        attrs: SurveyAttributes,
        uptime_s: f64,
    },
    ModelRequest {
        attrs: RequestAttributes,
    },
    RouteAttempt {
        attrs: RouteAttemptAttributes,
    },
    PromptShape {
        attrs: PromptShapeAttributes,
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
    },
    GuardrailDecision {
        attrs: GuardrailDecisionAttributes,
    },
    GuardrailOutcome {
        attrs: GuardrailOutcomeAttributes,
    },
    InflightRequests {
        source: SurveyTelemetrySource,
        current: u64,
    },
    LoggingMetric {
        metric: LoggingMetric,
    },
}

#[derive(Debug)]
struct SurveyEventQueue {
    capacity: usize,
    events: Mutex<VecDeque<SurveyEvent>>,
    notify: Notify,
}

impl SurveyEventQueue {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            events: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
            notify: Notify::new(),
        }
    }

    fn push(&self, event: SurveyEvent) {
        let mut events = self
            .events
            .lock()
            .expect("telemetry event queue lock poisoned");
        if events.len() == self.capacity {
            events.pop_front();
        }
        events.push_back(event);
        drop(events);
        self.notify.notify_one();
    }

    /// Logging metrics use this path so a contended telemetry queue cannot
    /// delay logging or request-serving work. The loss is intentionally
    /// fail-open because telemetry is observational only.
    fn try_push(&self, event: SurveyEvent) {
        let Ok(mut events) = self.events.try_lock() else {
            return;
        };
        if events.len() == self.capacity {
            events.pop_front();
        }
        events.push_back(event);
        drop(events);
        self.notify.notify_one();
    }

    fn drain(&self) -> Vec<SurveyEvent> {
        let mut events = self
            .events
            .lock()
            .expect("telemetry event queue lock poisoned");
        events.drain(..).collect()
    }

    async fn notified(&self) {
        self.notify.notified().await;
    }
}

struct SurveyRecorder {
    _provider: SdkMeterProvider,
    launch_total: Counter<u64>,
    launch_success_total: Counter<u64>,
    launch_failure_total: Counter<u64>,
    unload_total: Counter<u64>,
    unexpected_exit_total: Counter<u64>,
    loaded_models: Gauge<u64>,
    model_loaded: Gauge<u64>,
    model_context_length: Gauge<u64>,
    model_request_total: Counter<u64>,
    route_attempt_total: Counter<u64>,
    prompt_tokens: Histogram<u64>,
    completion_tokens: Histogram<u64>,
    guardrail_decision_total: Counter<u64>,
    guardrail_outcome_total: Counter<u64>,
    requests_inflight: Gauge<u64>,
    logging_lifecycle_terminal_total: Counter<u64>,
    logging_persistence_queue_dropped_total: Counter<u64>,
    logging_persistence_failure_total: Counter<u64>,
    logging_persistence_shutdown_loss_total: Counter<u64>,
    logging_persistence_outstanding: Gauge<u64>,
    logging_replay_evicted_total: Counter<u64>,
    logging_replay_gap_total: Counter<u64>,
    logging_replay_dropped_total: Counter<u64>,
    logging_cleanup_total: Counter<u64>,
    logging_webhook_delivery_total: Counter<u64>,
    logging_webhook_attempt_total: Counter<u64>,
    logging_artifact_capture_total: Counter<u64>,
    launch_duration_ms: Histogram<f64>,
    uptime_s: Histogram<f64>,
    loaded_count: u64,
}

impl SurveyRecorder {
    fn otlp(settings: &SurveySettings) -> Result<Self> {
        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(settings.endpoint.clone())
            .with_timeout(Duration::from_secs(10))
            .with_headers(settings.headers.clone())
            .build()
            .context("build OTLP metrics exporter")?;
        let reader = PeriodicReader::builder(exporter)
            .with_interval(settings.export_interval)
            .build();
        let provider = SdkMeterProvider::builder()
            .with_resource(
                Resource::builder()
                    .with_service_name(settings.service_name.clone())
                    .with_attribute(KeyValue::new("service.version", crate::VERSION))
                    .build(),
            )
            .with_reader(reader)
            .build();
        Ok(Self::new(provider))
    }

    fn new(provider: SdkMeterProvider) -> Self {
        let meter = provider.meter("mesh-llm.telemetry");
        Self {
            _provider: provider,
            launch_total: meter
                .u64_counter("mesh_llm_model_launch_total")
                .with_description("Total local model launch attempts.")
                .build(),
            launch_success_total: meter
                .u64_counter("mesh_llm_model_launch_success_total")
                .with_description("Successful local model launches.")
                .build(),
            launch_failure_total: meter
                .u64_counter("mesh_llm_model_launch_failure_total")
                .with_description("Failed local model launches.")
                .build(),
            unload_total: meter
                .u64_counter("mesh_llm_model_unload_total")
                .with_description("Intentional local model unloads.")
                .build(),
            unexpected_exit_total: meter
                .u64_counter("mesh_llm_model_exit_unexpected_total")
                .with_description("Unexpected local model exits.")
                .build(),
            loaded_models: meter
                .u64_gauge("mesh_llm_loaded_models")
                .with_description("Current number of locally loaded models.")
                .build(),
            model_loaded: meter
                .u64_gauge("mesh_llm_model_loaded")
                .with_description("Whether a local model is currently loaded.")
                .build(),
            model_context_length: meter
                .u64_gauge("mesh_llm_model_context_length")
                .with_description("Effective context length for a loaded local model.")
                .with_unit("{token}")
                .build(),
            model_request_total: meter
                .u64_counter("mesh_llm_model_request_total")
                .with_description("Requests fronted by this node for a model.")
                .build(),
            route_attempt_total: meter
                .u64_counter("mesh_llm_route_attempt_total")
                .with_description(
                    "Routing attempts from this node to local, remote, or endpoint targets.",
                )
                .build(),
            prompt_tokens: meter
                .u64_histogram("mesh_llm_prompt_tokens")
                .with_description("Observed prompt token count for completed requests.")
                .with_unit("{token}")
                .build(),
            completion_tokens: meter
                .u64_histogram("mesh_llm_completion_tokens")
                .with_description("Observed completion token count for completed requests.")
                .with_unit("{token}")
                .build(),
            guardrail_decision_total: meter
                .u64_counter("mesh_llm_guardrail_decision_total")
                .with_description(
                    "Guardrail request decisions for hosted OpenAI backends on this node.",
                )
                .build(),
            guardrail_outcome_total: meter
                .u64_counter("mesh_llm_guardrail_outcome_total")
                .with_description(
                    "Guardrail attempt and final outcomes for hosted OpenAI backends on this node.",
                )
                .build(),
            requests_inflight: meter
                .u64_gauge("mesh_llm_requests_inflight")
                .with_description("Current in-flight requests fronted by this node.")
                .build(),
            logging_lifecycle_terminal_total: meter
                .u64_counter("mesh_llm_logging_lifecycle_terminal_total")
                .with_description("Logging lifecycle terminal outcomes.")
                .build(),
            logging_persistence_queue_dropped_total: meter
                .u64_counter("mesh_llm_logging_persistence_queue_dropped_total")
                .with_description("Logging persistence queue entries dropped.")
                .build(),
            logging_persistence_failure_total: meter
                .u64_counter("mesh_llm_logging_persistence_failure_total")
                .with_description("Logging persistence sink failures.")
                .build(),
            logging_persistence_shutdown_loss_total: meter
                .u64_counter("mesh_llm_logging_persistence_shutdown_loss_total")
                .with_description("Logging persistence entries lost after bounded shutdown.")
                .build(),
            logging_persistence_outstanding: meter
                .u64_gauge("mesh_llm_logging_persistence_outstanding")
                .with_description(
                    "Logging persistence entries currently owned by a queue or worker.",
                )
                .build(),
            logging_replay_evicted_total: meter
                .u64_counter("mesh_llm_logging_replay_evicted_total")
                .with_description("Logging replay entries evicted by the bounded window.")
                .build(),
            logging_replay_gap_total: meter
                .u64_counter("mesh_llm_logging_replay_gap_total")
                .with_description("Replay recovery gaps emitted by the logging SSE session.")
                .build(),
            logging_replay_dropped_total: meter
                .u64_counter("mesh_llm_logging_replay_dropped_total")
                .with_description(
                    "Logging replay entries rejected because the replay window is disabled.",
                )
                .build(),
            logging_cleanup_total: meter
                .u64_counter("mesh_llm_logging_cleanup_total")
                .with_description("Logging retention cleanup outcomes.")
                .build(),
            logging_webhook_delivery_total: meter
                .u64_counter("mesh_llm_logging_webhook_delivery_total")
                .with_description("Durable logging webhook delivery outcomes.")
                .build(),
            logging_webhook_attempt_total: meter
                .u64_counter("mesh_llm_logging_webhook_attempt_total")
                .with_description("Durable logging webhook attempt states.")
                .build(),
            logging_artifact_capture_total: meter
                .u64_counter("mesh_llm_logging_artifact_capture_total")
                .with_description("Logging artifact capture outcomes.")
                .build(),
            launch_duration_ms: meter
                .f64_histogram("mesh_llm_model_launch_duration_ms")
                .with_description("Local model launch duration.")
                .with_unit("ms")
                .build(),
            uptime_s: meter
                .f64_histogram("mesh_llm_model_uptime_s")
                .with_description("Local model uptime before unload or unexpected exit.")
                .with_unit("s")
                .build(),
            loaded_count: 0,
        }
    }

    fn record(&mut self, event: SurveyEvent) {
        match event {
            SurveyEvent::LaunchSuccess { attrs, duration_ms } => {
                let kv = attrs.key_values(None);
                self.launch_total.add(1, &kv);
                self.launch_success_total.add(1, &kv);
                self.launch_duration_ms.record(duration_ms, &kv);
                self.loaded_count = self.loaded_count.saturating_add(1);
                self.loaded_models
                    .record(self.loaded_count, &service_version_attrs());
                self.model_loaded.record(1, &kv);
                if let Some(context_length) = attrs.context_length {
                    self.model_context_length.record(context_length, &kv);
                }
            }
            SurveyEvent::LaunchFailure {
                attrs,
                duration_ms,
                reason,
            } => {
                let kv = attrs.key_values(Some(reason));
                self.launch_total.add(1, &kv);
                self.launch_failure_total.add(1, &kv);
                self.launch_duration_ms.record(duration_ms, &kv);
            }
            SurveyEvent::Unload { attrs, uptime_s } => {
                let kv = attrs.key_values(None);
                self.unload_total.add(1, &kv);
                self.uptime_s.record(uptime_s, &kv);
                self.loaded_count = self.loaded_count.saturating_sub(1);
                self.loaded_models
                    .record(self.loaded_count, &service_version_attrs());
                self.model_loaded.record(0, &kv);
            }
            SurveyEvent::UnexpectedExit { attrs, uptime_s } => {
                let kv = attrs.key_values(None);
                self.unexpected_exit_total.add(1, &kv);
                self.uptime_s.record(uptime_s, &kv);
                self.loaded_count = self.loaded_count.saturating_sub(1);
                self.loaded_models
                    .record(self.loaded_count, &service_version_attrs());
                self.model_loaded.record(0, &kv);
            }
            SurveyEvent::ModelRequest { attrs } => {
                let kv = attrs.key_values();
                self.model_request_total.add(1, &kv);
            }
            SurveyEvent::RouteAttempt { attrs } => {
                let kv = attrs.key_values();
                self.route_attempt_total.add(1, &kv);
            }
            SurveyEvent::PromptShape {
                attrs,
                prompt_tokens,
                completion_tokens,
            } => {
                let kv = attrs.key_values();
                if let Some(tokens) = prompt_tokens {
                    self.prompt_tokens.record(tokens, &kv);
                }
                if let Some(tokens) = completion_tokens {
                    self.completion_tokens.record(tokens, &kv);
                }
            }
            SurveyEvent::GuardrailDecision { attrs } => {
                let kv = attrs.key_values();
                self.guardrail_decision_total.add(1, &kv);
            }
            SurveyEvent::GuardrailOutcome { attrs } => {
                let kv = attrs.key_values();
                self.guardrail_outcome_total.add(1, &kv);
            }
            SurveyEvent::InflightRequests { source, current } => {
                self.requests_inflight.record(current, &source.key_values());
            }
            SurveyEvent::LoggingMetric { metric } => self.record_logging_metric(metric),
        }
    }
}

fn service_version_attrs() -> Vec<KeyValue> {
    let attrs = vec![KeyValue::new("mesh_llm.service_version", crate::VERSION)];
    debug_assert_telemetry_attrs_allowlisted(&attrs);
    attrs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{MeshConfig, TelemetryConfig, TelemetryMetricsConfig};
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    pub(super) fn test_source() -> SurveyTelemetrySource {
        SurveyTelemetrySource {
            node_id: "source-node-raw".into(),
            node_role: "client".into(),
        }
    }

    fn assert_attrs_allowlisted(attrs: Vec<KeyValue>) {
        for attr in attrs {
            let key = attr.key.to_string();
            assert!(
                telemetry_attribute_allowed(&key),
                "unexpected telemetry attribute key: {key}"
            );
        }
    }

    #[test]
    fn model_values_are_absent_from_request_and_route_attributes() {
        let request = RequestAttributes::from_request(
            Some("https://user:secret@example.test/private/model.gguf?token=leaked"),
            1,
            RequestOutcome::Success(RequestService::Local),
            test_source(),
        );
        let route = RouteAttemptAttributes::from_attempt(
            Some("org/private-model:variant"),
            &AttemptTarget::Local("local".into()),
            AttemptOutcome::Success,
            test_source(),
        );

        for attrs in [request.key_values(), route.key_values()] {
            assert!(
                attrs
                    .iter()
                    .all(|attr| attr.key.as_str() != "mesh_llm.model")
            );
            let serialized = format!("{attrs:?}");
            assert!(!serialized.contains("secret"));
            assert!(!serialized.contains("private-model"));
        }
    }

    pub(super) fn survey_config() -> MeshConfig {
        MeshConfig {
            telemetry: TelemetryConfig {
                enabled: Some(true),
                service_name: Some("mesh-llm-test".into()),
                endpoint: Some("https://config.example.com".into()),
                export_interval_secs: Some(5),
                queue_size: Some(2),
                ..Default::default()
            },
            defaults: None,
            ..Default::default()
        }
    }

    #[test]
    fn settings_default_to_enabled_with_endpoint() {
        let mut config = survey_config();
        config.plugins.clear();
        assert!(SurveySettings::from_config_with_env(&config, |_| None).is_some());
    }

    #[test]
    fn settings_disable_when_telemetry_config_opted_out() {
        let mut config = survey_config();
        config.telemetry.enabled = Some(false);
        assert!(SurveySettings::from_config_with_env(&config, |_| None).is_none());
    }

    #[test]
    fn metrics_endpoint_prefers_config_metrics_endpoint_over_base_and_env() {
        let mut config = survey_config();
        config.telemetry.endpoint = Some("https://base.example.com".into());
        config.telemetry.metrics = TelemetryMetricsConfig {
            endpoint: Some("https://metrics.example.com/custom".into()),
        };

        let settings = SurveySettings::from_config_with_env(&config, |key| match key {
            OTLP_METRICS_ENDPOINT_ENV => Some("https://env-metrics.example.com/v1/metrics".into()),
            OTLP_ENDPOINT_ENV => Some("https://env-base.example.com".into()),
            _ => None,
        })
        .expect("settings");

        assert_eq!(settings.endpoint, "https://metrics.example.com/custom");
        assert_eq!(settings.queue_size, 2);
        assert_eq!(settings.export_interval, Duration::from_secs(5));
    }

    #[test]
    fn metrics_endpoint_normalizes_base_endpoint_from_env() {
        let mut config = survey_config();
        config.telemetry.endpoint = None;
        config.telemetry.metrics.endpoint = None;

        let settings = SurveySettings::from_config_with_env(&config, |key| match key {
            OTLP_ENDPOINT_ENV => Some("https://collector.example.com/".into()),
            _ => None,
        })
        .expect("settings");

        assert_eq!(
            settings.endpoint,
            "https://collector.example.com/v1/metrics"
        );
    }

    #[test]
    fn ambient_otel_env_does_not_enable_export_without_explicit_telemetry_enable() {
        let mut config = survey_config();
        config.telemetry.enabled = None;
        config.telemetry.endpoint = None;
        config.telemetry.metrics.endpoint = None;

        let settings = SurveySettings::from_config_with_env(&config, |key| match key {
            OTLP_ENDPOINT_ENV => Some("https://ambient.example.com".into()),
            _ => None,
        });
        assert!(settings.is_none());

        config.telemetry.enabled = Some(true);
        let settings = SurveySettings::from_config_with_env(&config, |key| match key {
            OTLP_ENDPOINT_ENV => Some("https://ambient.example.com".into()),
            _ => None,
        })
        .expect("explicit telemetry enable should allow OTel env endpoint");
        assert_eq!(settings.endpoint, "https://ambient.example.com/v1/metrics");
    }

    #[test]
    fn config_endpoint_enables_export_without_boolean_flag() {
        let mut config = survey_config();
        config.telemetry.enabled = None;
        config.telemetry.endpoint = Some("https://config-owned.example.com".into());
        config.telemetry.metrics.endpoint = None;

        let settings =
            SurveySettings::from_config_with_env(&config, |_| None).expect("config endpoint");
        assert_eq!(
            settings.endpoint,
            "https://config-owned.example.com/v1/metrics"
        );
    }

    #[test]
    fn event_queue_drops_oldest_when_full() {
        let queue = SurveyEventQueue::new(2);
        for context_length in [1, 2, 3] {
            let attrs = SurveyAttributes {
                model_selector_id: None,
                architecture: None,
                quantization: None,
                launch_kind: SurveyLaunchKind::Startup,
                gpu_name: None,
                gpu_stable_id: None,
                backend_device: None,
                gpu_count: 0,
                is_soc: false,
                backend: None,
                context_length: Some(context_length),
            };
            queue.push(SurveyEvent::LaunchSuccess {
                attrs,
                duration_ms: 1.0,
            });
        }

        let drained = queue.drain();
        let context_lengths: Vec<_> = drained
            .iter()
            .filter_map(|event| match event {
                SurveyEvent::LaunchSuccess { attrs, .. } => attrs.context_length,
                _ => None,
            })
            .collect();
        assert_eq!(context_lengths, vec![2, 3]);
    }

    #[test]
    fn attributes_hash_gpu_stable_id_and_bucket_context() {
        let hardware = hardware::HardwareSurvey {
            gpu_count: 1,
            is_soc: true,
            gpus: vec![hardware::GpuFacts {
                index: 0,
                display_name: "NVIDIA Test".into(),
                backend_device: Some("CUDA0".into()),
                stable_id: Some("uuid:SECRET-GPU".into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let attrs = SurveyAttributes::from_spec(
            SurveyModelSpec {
                model: "/private/models/Qwen3-8B-Q4_K_M.gguf",
                configured_model_selector: None,
                model_path: None,
                launch_kind: SurveyLaunchKind::RuntimeLoad,
                pinned_gpu: None,
                backend: Some("skippy"),
                context_length: Some(32_768),
            },
            &hardware,
        );
        let kv: HashMap<_, _> = attrs
            .key_values(None)
            .into_iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();

        assert!(!kv.contains_key("mesh_llm.model"));
        assert_eq!(
            kv.get("mesh_llm.context_bucket").map(String::as_str),
            Some("16k_32k")
        );
        let stable_id = kv.get("mesh_llm.gpu_stable_id").expect("stable id");
        assert!(stable_id.starts_with("sha256:"));
        assert!(!stable_id.contains("SECRET-GPU"));
        assert_eq!(
            kv.get("mesh_llm.backend").map(String::as_str),
            Some("skippy")
        );
        assert_eq!(kv.get("mesh_llm.is_soc").map(String::as_str), Some("true"));
        assert!(!kv.values().any(|value| value.contains("/private/models")));
    }

    #[test]
    fn telemetry_attribute_allowlist_has_unique_reviewed_keys() {
        let keys: BTreeSet<_> = TELEMETRY_ATTRIBUTE_ALLOWLIST.iter().copied().collect();
        assert_eq!(keys.len(), TELEMETRY_ATTRIBUTE_ALLOWLIST.len());
        assert_eq!(
            keys,
            BTreeSet::from([
                "llama_stage.linear_proposal.source_callback_us",
                "llama_stage.linear_proposal.source_outcome",
                "llama_stage.linear_proposal.source_queue_wait_us",
                "llama_stage.verify_window.direct_return_reverse_fallback",
                "llama_stage.verify_window.direct_return_upstream_opened",
                "mesh_llm.architecture",
                "mesh_llm.attempt_outcome",
                "mesh_llm.backend",
                "mesh_llm.backend_device",
                "mesh_llm.context_bucket",
                "mesh_llm.failure_reason",
                "mesh_llm.guardrail.attempt_bucket",
                "mesh_llm.guardrail.bypass_reason",
                "mesh_llm.guardrail.contract",
                "mesh_llm.guardrail.decision",
                "mesh_llm.guardrail.mode",
                "mesh_llm.guardrail.outcome",
                "mesh_llm.gpu_count",
                "mesh_llm.gpu_name",
                "mesh_llm.gpu_stable_id",
                "mesh_llm.is_soc",
                "mesh_llm.launch_kind",
                "mesh_llm.logging_artifact_capture_status",
                "mesh_llm.logging_cleanup_outcome",
                "mesh_llm.logging_terminal_outcome",
                "mesh_llm.logging_webhook_attempt_state",
                "mesh_llm.logging_webhook_delivery_outcome",
                "mesh_llm.model_selector_id",
                "mesh_llm.quantization",
                "mesh_llm.request_outcome",
                "mesh_llm.route_attempt_bucket",
                "mesh_llm.route_service",
                "mesh_llm.service_version",
                "mesh_llm.source_node_id",
                "mesh_llm.source_node_role",
                "mesh_llm.target_kind",
                "mesh_llm.target_node_id",
            ])
        );
    }

    #[test]
    fn generated_telemetry_attributes_are_allowlisted() {
        let lifecycle_attrs = SurveyAttributes {
            model_selector_id: Some("sha256:0123456789abcdef0123456789abcdef".into()),
            architecture: Some("qwen3".into()),
            quantization: Some("Q4_K_M".into()),
            launch_kind: SurveyLaunchKind::Startup,
            gpu_name: Some("NVIDIA Test".into()),
            gpu_stable_id: Some("sha256:abcdef1234567890".into()),
            backend_device: Some("CUDA0".into()),
            gpu_count: 1,
            is_soc: false,
            backend: Some("skippy".into()),
            context_length: Some(131_072),
        };
        assert_attrs_allowlisted(lifecycle_attrs.key_values(Some(SurveyFailureReason::Other)));

        assert_attrs_allowlisted(
            RequestAttributes::from_request(
                Some("Qwen/Qwen3-8B-GGUF:Q4_K_M"),
                2,
                RequestOutcome::Rejected(RequestService::Endpoint),
                test_source(),
            )
            .key_values(),
        );
        assert_attrs_allowlisted(
            RouteAttemptAttributes::from_attempt(
                Some("Qwen/Qwen3-8B-GGUF:Q4_K_M"),
                &AttemptTarget::Remote("remote-node-raw".into()),
                AttemptOutcome::Success,
                test_source(),
            )
            .key_values(),
        );
        assert_attrs_allowlisted(
            GuardrailDecisionAttributes {
                source: test_source(),
                mode: "enforce",
                contract: Some("tools"),
                decision: "eligible",
                bypass_reason: None,
            }
            .key_values(),
        );
        assert_attrs_allowlisted(
            GuardrailOutcomeAttributes {
                source: test_source(),
                mode: "metrics",
                contract: Some("structured"),
                outcome: "metrics_only_failure",
                attempt_bucket: Some("2"),
            }
            .key_values(),
        );
        assert_attrs_allowlisted(test_source().key_values());
        assert_attrs_allowlisted(service_version_attrs());
    }

    #[test]
    fn guardrail_attributes_stay_bounded_and_allowlisted() {
        let decision = GuardrailDecisionAttributes {
            source: test_source(),
            mode: "disabled",
            contract: Some("tools"),
            decision: "bypassed",
            bypass_reason: Some("streaming"),
        };
        let decision_kv: HashMap<_, _> = decision
            .key_values()
            .into_iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();
        assert_eq!(
            decision_kv
                .get("mesh_llm.guardrail.mode")
                .map(String::as_str),
            Some("disabled")
        );
        assert_eq!(
            decision_kv
                .get("mesh_llm.guardrail.bypass_reason")
                .map(String::as_str),
            Some("streaming")
        );

        let outcome = GuardrailOutcomeAttributes {
            source: test_source(),
            mode: "enforce",
            contract: Some("structured"),
            outcome: "valid",
            attempt_bucket: Some("3_plus"),
        };
        let outcome_kv: HashMap<_, _> = outcome
            .key_values()
            .into_iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();
        assert_eq!(
            outcome_kv
                .get("mesh_llm.guardrail.contract")
                .map(String::as_str),
            Some("structured")
        );
        assert!(outcome_kv.values().all(|value| {
            !value.contains("prompt")
                && !value.contains("completion")
                && !value.contains("http://")
                && !value.contains("https://")
                && !value.contains('/')
        }));
    }

    #[test]
    fn request_attributes_omit_model_and_capture_service_and_attempt_count() {
        let attrs = RequestAttributes::from_request(
            Some("/private/models/Qwen3-8B-Q4_K_M.gguf"),
            2,
            RequestOutcome::Success(RequestService::Remote),
            test_source(),
        );
        let kv: HashMap<_, _> = attrs
            .key_values()
            .into_iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();

        assert!(!kv.contains_key("mesh_llm.model"));
        assert_eq!(
            kv.get("mesh_llm.route_service").map(String::as_str),
            Some("remote")
        );
        assert_eq!(
            kv.get("mesh_llm.request_outcome").map(String::as_str),
            Some("success")
        );
        assert_eq!(
            kv.get("mesh_llm.route_attempt_bucket").map(String::as_str),
            Some("2")
        );
        assert_eq!(
            kv.get("mesh_llm.source_node_role").map(String::as_str),
            Some("client")
        );
    }

    #[test]
    fn request_attempt_count_is_exported_as_bounded_bucket() {
        assert_eq!(request_attempt_bucket(0), "1");
        assert_eq!(request_attempt_bucket(1), "1");
        assert_eq!(request_attempt_bucket(2), "2");
        assert_eq!(request_attempt_bucket(3), "3_4");
        assert_eq!(request_attempt_bucket(4), "3_4");
        assert_eq!(request_attempt_bucket(5), "5_plus");
        assert_eq!(request_attempt_bucket(100), "5_plus");
    }

    #[test]
    fn route_attempt_attributes_hash_source_and_remote_node_ids() {
        let attrs = RouteAttemptAttributes::from_attempt(
            Some("Qwen/Qwen3-8B-GGUF:Q4_K_M"),
            &AttemptTarget::Remote("remote-node-raw".into()),
            AttemptOutcome::Timeout,
            test_source(),
        );
        let kv: HashMap<_, _> = attrs
            .key_values()
            .into_iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();

        assert!(!kv.contains_key("mesh_llm.model"));
        assert_eq!(
            kv.get("mesh_llm.target_kind").map(String::as_str),
            Some("remote")
        );
        assert_eq!(
            kv.get("mesh_llm.attempt_outcome").map(String::as_str),
            Some("timeout")
        );
        let source_node_id = kv.get("mesh_llm.source_node_id").expect("source node id");
        assert!(source_node_id.starts_with("sha256:"));
        assert!(!source_node_id.contains("source-node-raw"));
        let target_node_id = kv.get("mesh_llm.target_node_id").expect("target node id");
        assert!(target_node_id.starts_with("sha256:"));
        assert!(!target_node_id.contains("remote-node-raw"));
    }

    #[test]
    fn route_attempt_attributes_do_not_export_endpoint_urls() {
        let attrs = RouteAttemptAttributes::from_attempt(
            None,
            &AttemptTarget::Endpoint("https://private-endpoint.example.com/v1".into()),
            AttemptOutcome::Rejected,
            test_source(),
        );
        let kv: HashMap<_, _> = attrs
            .key_values()
            .into_iter()
            .map(|kv| (kv.key.to_string(), kv.value.to_string()))
            .collect();

        assert_eq!(
            kv.get("mesh_llm.target_kind").map(String::as_str),
            Some("endpoint")
        );
        assert!(!kv.contains_key("mesh_llm.target_node_id"));
        assert!(!kv.values().any(|value| value.contains("private-endpoint")));
    }

    #[test]
    fn lifecycle_attributes_omit_model_values() {
        let attrs = SurveyAttributes::from_disabled_spec(SurveyModelSpec {
            model: "org/private-model:variant",
            configured_model_selector: None,
            model_path: None,
            launch_kind: SurveyLaunchKind::Startup,
            pinned_gpu: None,
            backend: None,
            context_length: None,
        })
        .key_values(None);

        assert!(
            attrs
                .iter()
                .all(|attr| attr.key.as_str() != "mesh_llm.model")
        );
        assert!(!format!("{attrs:?}").contains("private-model"));
    }

    #[test]
    fn telemetry_headers_are_copied_from_config() {
        let mut config = survey_config();
        config.telemetry.headers = BTreeMap::from([("authorization".into(), "Bearer abc".into())]);

        let settings = SurveySettings::from_config_with_env(&config, |_| None).expect("settings");

        assert_eq!(
            settings.headers.get("authorization").map(String::as_str),
            Some("Bearer abc")
        );
    }
}
