use super::*;

pub(crate) fn log_target_selection(requested: &str, selection: &BenchmarkSelection) {
    if let Some(best) = &selection.recommended {
        eprintln!(
            "benchmark tune: target `{requested}` recommended {} decode_tok_s={}",
            render_benchmark_candidate(&best.candidate),
            best.decode_tok_s
                .map(|rate| format!("{rate:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
        );
    } else {
        eprintln!("benchmark tune: target `{requested}` produced no successful trials");
    }
}

pub(crate) fn run_trial_with_progress(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
    index: usize,
    total: usize,
    candidate: TuneBenchmarkCandidate,
) -> TuneBenchmarkTrial {
    eprintln!(
        "benchmark tune: trial {}/{} start {}",
        index + 1,
        total,
        render_benchmark_candidate(&candidate),
    );
    let trial = run_trial(request, prepared, index, candidate);
    log_trial_result(index, total, &trial);
    trial
}

fn log_trial_result(index: usize, total: usize, trial: &TuneBenchmarkTrial) {
    match trial.status {
        TuneBenchmarkTrialStatus::Succeeded => eprintln!(
            "benchmark tune: trial {}/{} ok {} decode_tok_s={} ttft_ms={} decode_only_tok_s={}{}",
            index + 1,
            total,
            render_benchmark_candidate(&trial.candidate),
            trial
                .decode_tok_s
                .map(|rate| format!("{rate:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
            trial
                .ttft_ms
                .map(|value| format!("{value:.0}"))
                .unwrap_or_else(|| "n/a".to_string()),
            trial
                .decode_only_tok_s
                .map(|rate| format!("{rate:.2}"))
                .unwrap_or_else(|| "n/a".to_string()),
            render_progress_timing(trial.timings.as_ref()),
        ),
        TuneBenchmarkTrialStatus::Failed => eprintln!(
            "benchmark tune: trial {}/{} failed {} error={}",
            index + 1,
            total,
            render_benchmark_candidate(&trial.candidate),
            trial.error.as_deref().unwrap_or("unknown"),
        ),
    }
}

fn render_progress_timing(timings: Option<&TuneBenchmarkTimingStats>) -> String {
    timings
        .map(|timings| {
            let request_ms = timings
                .request_ms
                .map(|value| format!("{value:.0}"))
                .unwrap_or_else(|| "n/a".to_string());
            format!(
                " readiness_ms={:.0} request_ms={request_ms}",
                timings.readiness_ms
            )
        })
        .unwrap_or_default()
}
