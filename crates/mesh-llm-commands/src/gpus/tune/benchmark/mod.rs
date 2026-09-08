// Benchmark module for GPU tune benchmarking.
// Split from benchmark.rs to improve maintainability.

mod candidates;
mod streaming;
#[cfg(test)]
mod tests;
mod trial;
mod trial_config;

const MAX_BENCHMARK_TRIALS_PER_TARGET: usize = 512;

pub(crate) use candidates::*;
// Re-export flattened benchmark helpers for downstream tune call-sites.
pub(crate) use trial::run_trial;
#[cfg(test)]
use trial::{
    build_trial_child_command, trial_startup_failure_from_log, trial_startup_failure_from_log_line,
};
pub(crate) use trial_config::trial_config;

// Re-export types from sibling modules that benchmark consumers need.
// output_types.rs and benchmark_selection.rs are flat-included in the tune
// parent, so we reference them via crate::gpus::tune::*.
pub(crate) use crate::gpus::tune::{
    TuneBenchmarkCandidate, TuneBenchmarkMetricsSchema, TuneBenchmarkSpeculativeCandidate,
    TuneBenchmarkTargetReport, TuneBenchmarkTimingStats, TuneBenchmarkTrial,
    TuneBenchmarkTrialStatus, benchmark_trial_unit_definition, select_benchmark_trials,
};

/// Request structure for running benchmark plans.
pub(crate) struct TuneBenchmarkRunRequest<'a> {
    pub(crate) config: &'a mesh_llm_config::MeshConfig,
    pub(crate) prepared: &'a [crate::gpus::tune_apply::PreparedTunePlan],
    pub(crate) ctx_sizes: &'a [u32],
    pub(crate) batch_sizes: &'a [u32],
    pub(crate) ubatch_sizes: &'a [u32],
    pub(crate) mmap_values: &'a [mesh_llm_cli::benchmark::BenchmarkBoolOrAuto],
    pub(crate) mlock_values: &'a [mesh_llm_cli::benchmark::BenchmarkBool],
    pub(crate) flash_attention_values: &'a [mesh_llm_cli::benchmark::BenchmarkFlashAttention],
    pub(crate) speculative_types: &'a [mesh_llm_cli::benchmark::BenchmarkSpeculativeType],
    pub(crate) no_speculative_tune: bool,
    pub(crate) spec_draft_models: &'a [std::path::PathBuf],
    pub(crate) spec_draft_max_tokens: &'a [u32],
    pub(crate) spec_draft_min_tokens: &'a [u32],
    pub(crate) spec_draft_acceptance_threshold: &'a [f64],
    pub(crate) spec_draft_split_probability: &'a [f64],
    pub(crate) spec_ngram_min: &'a [u32],
    pub(crate) spec_ngram_max: &'a [u32],
    pub(crate) throughput_tolerance_pct: f64,
    pub(crate) max_tokens: u32,
    pub(crate) startup_timeout_secs: u64,
    pub(crate) request_timeout_secs: u64,
    pub(crate) debug_telemetry: bool,
    pub(crate) prompt: &'a str,
}

/// Run benchmark plans for the given request.
pub(crate) fn run_benchmark_plans(
    request: TuneBenchmarkRunRequest<'_>,
) -> anyhow::Result<Vec<TuneBenchmarkTargetReport>> {
    // Validate throughput tolerance before proceeding; debug_assert is not
    // enough for release builds.
    if !request.throughput_tolerance_pct.is_finite() || request.throughput_tolerance_pct < 0.0 {
        anyhow::bail!(
            "benchmark tune: invalid throughput_tolerance_pct {}; expected a finite non-negative value",
            request.throughput_tolerance_pct
        );
    }
    request
        .prepared
        .iter()
        .filter(|prepared| !plan_has_errors(&prepared.plan))
        .map(|prepared| run_target_benchmarks(&request, prepared))
        .collect::<anyhow::Result<Vec<_>>>()
}

fn run_target_benchmarks(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
) -> anyhow::Result<TuneBenchmarkTargetReport> {
    let candidates = benchmark_candidates(request, prepared);
    if candidates.len() > MAX_BENCHMARK_TRIALS_PER_TARGET {
        anyhow::bail!(
            "benchmark tune: target `{}` produced {} trial candidates, exceeding the hard cap of {}",
            prepared.target.requested_input,
            candidates.len(),
            MAX_BENCHMARK_TRIALS_PER_TARGET
        );
    }
    eprintln!(
        "benchmark tune: target `{}` running {} trials (throughput tolerance {:.2}%)",
        prepared.target.requested_input,
        candidates.len(),
        request.throughput_tolerance_pct,
    );
    let total = candidates.len();
    let trials = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| {
            super::run_trial_with_progress(request, prepared, index, total, candidate)
        })
        .collect::<Vec<_>>();
    let selection = select_benchmark_trials(&trials, request.throughput_tolerance_pct);
    super::log_target_selection(&prepared.target.requested_input, &selection);

    Ok(TuneBenchmarkTargetReport {
        requested: prepared.target.requested_input.clone(),
        throughput_tolerance_pct: request.throughput_tolerance_pct,
        best: selection.recommended,
        raw_best: selection.raw_best,
        pareto_frontier: selection.pareto_frontier,
        selection_reason: selection.reason,
        trials,
        metrics_schema: TuneBenchmarkMetricsSchema::StreamingV1,
        trial_unit: Some(benchmark_trial_unit_definition()),
    })
}

fn plan_has_errors(plan: &crate::gpus::tune::TunePlan) -> bool {
    plan.field_statuses
        .iter()
        .any(|status| matches!(status, crate::gpus::tune::TuneFieldStatus::Error { .. }))
        || plan.diagnostics.iter().any(|diagnostic| {
            matches!(
                diagnostic.severity,
                crate::gpus::tune::TuneDiagnosticSeverity::Error
            )
        })
}
