use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TuneTargetFailure {
    pub requested_input: String,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TuneTargetStatus {
    Ready,
    Written,
    Skipped,
    Failed,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TuneRenderedSettingStatus {
    Applied,
    Preserved,
    ReportOnly,
    Unsupported,
    Error,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TuneRenderedSetting {
    pub field: TuneField,
    pub support: TuneFieldSupport,
    pub status: TuneRenderedSettingStatus,
    pub config_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<TuneRecommendedValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic: Option<TuneDiagnostic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edit: Option<TuneConfigEdit>,
    pub applied_write: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TuneLaunchSetting {
    pub config_path: String,
    pub field: TuneField,
    pub value: TuneRecommendedValue,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TuneLaunchPreview {
    pub argv: Vec<String>,
    pub shell: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config_settings: Vec<TuneLaunchSetting>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub report_only: Vec<TuneRenderedSetting>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unsupported: Vec<TuneRenderedSetting>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(crate) struct TuneTargetReport {
    pub target: TuneTarget,
    pub status: TuneTargetStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_model_ref: Option<String>,
    pub selection: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field_summary: Option<TunePlanSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<TuneDiagnostic>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub settings: Vec<TuneRenderedSetting>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config_edits: Vec<TuneRenderedSetting>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch: Option<TuneLaunchPreview>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct TuneBenchmarkCandidate {
    pub ctx_size: u32,
    pub batch: u32,
    pub ubatch: u32,
    pub cache_type_k: TuneKvCacheType,
    pub cache_type_v: TuneKvCacheType,
    pub mmap: TuneBoolOrAutoValue,
    pub mlock: bool,
    pub speculative: TuneBenchmarkSpeculativeCandidate,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flash_attention: Option<TuneFlashAttentionValue>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub(crate) enum TuneBenchmarkSpeculativeCandidate {
    Disabled,
    Mtp {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        draft_model: Option<String>,
        draft_max_tokens: u32,
        draft_min_tokens: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        draft_acceptance_threshold: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        draft_split_probability: Option<f64>,
    },
    Draft {
        draft_model: String,
        draft_max_tokens: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        draft_min_tokens: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        draft_acceptance_threshold: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        draft_split_probability: Option<f64>,
    },
    MtpNgram {
        ngram_min: u32,
        ngram_max: u32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct TuneBenchmarkTrial {
    pub candidate: TuneBenchmarkCandidate,
    pub status: TuneBenchmarkTrialStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tok_s: Option<f64>,
    // Time to first non-empty streamed content delta, in milliseconds.
    // Null (never zero) on trial failure, on a timeout/disconnect/malformed
    // stream before any content arrived, or on any pre-streaming
    // (`TuneBenchmarkMetricsSchema::NonStreamingHistorical`) trial.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    // `completion_tokens / max(total_request_elapsed - ttft, epsilon)`.
    // Null (never zero) whenever `ttft_ms` is null or the decode interval is
    // zero/negative -- see `streaming::decode_only_tok_s`, the single
    // function that computes this value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_only_tok_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timings: Option<TuneBenchmarkTimingStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct TuneBenchmarkTimingStats {
    pub total_ms: f64,
    pub setup_ms: f64,
    pub readiness_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shutdown_ms: Option<f64>,
    pub readiness_attempts: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TuneBenchmarkTrialStatus {
    Succeeded,
    Failed,
}

// Schema/comparability marker for `ttft_ms` / `decode_only_tok_s`. Recorded
// on every benchmark target report; `#[serde(default)]` makes a pre-task-18
// JSON report (missing this field entirely) deserialize as
// `NonStreamingHistorical` instead of silently reading as comparable
// streaming data. `decode_tok_s` itself is unaffected: its definition never
// changed and stays comparable across both schema versions.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TuneBenchmarkMetricsSchema {
    #[default]
    NonStreamingHistorical,
    StreamingV1,
}

// Frozen definition of what one benchmark "trial" and one "pair" mean,
// recorded for auditability by downstream comparison/certification tooling.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TuneBenchmarkTrialUnit {
    pub trial: String,
    pub pair: String,
}

pub(crate) fn benchmark_trial_unit_definition() -> TuneBenchmarkTrialUnit {
    TuneBenchmarkTrialUnit {
        trial: "One trial is one fresh process launch, one readiness wait, \
                one warmup request excluded from metrics, one measured \
                streaming request, and one shutdown."
            .to_string(),
        pair: "A pair is two trials, one per side, with the same prompt \
               and seed, side order randomized per pair."
            .to_string(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct TuneBenchmarkTargetReport {
    pub requested: String,
    pub throughput_tolerance_pct: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best: Option<TuneBenchmarkTrial>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_best: Option<TuneBenchmarkTrial>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pareto_frontier: Vec<TuneBenchmarkTrial>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trials: Vec<TuneBenchmarkTrial>,
    #[serde(default)]
    pub metrics_schema: TuneBenchmarkMetricsSchema,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trial_unit: Option<TuneBenchmarkTrialUnit>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub(crate) struct TuneRunReport {
    pub command: &'static str,
    pub apply_mode: TuneApplyMode,
    pub summary: TuneResultSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub global_blockers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<TuneTargetReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub benchmarks: Vec<TuneBenchmarkTargetReport>,
}
