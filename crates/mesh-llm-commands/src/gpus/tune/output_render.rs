use std::fmt::Write as _;

use super::*;

pub(crate) fn render_tune_human_output(report: &TuneRunReport) -> String {
    let mut rendered = String::new();
    let _ = writeln!(
        &mut rendered,
        "{} {} summary",
        render_command_label(report.command),
        render_apply_mode(report.apply_mode)
    );
    let _ = writeln!(
        &mut rendered,
        "  Targets: total={} ready={} written={} skipped={} failed={}",
        report.summary.total_targets,
        report.summary.ready_targets,
        report.summary.written_targets,
        report.summary.skipped_targets,
        report.summary.failed_targets,
    );
    let _ = writeln!(
        &mut rendered,
        "  Field counts: applied={} preserved={} report_only={} unsupported={} error={}",
        report.summary.fields.applied,
        report.summary.fields.preserved,
        report.summary.fields.report_only,
        report.summary.fields.unsupported,
        report.summary.fields.error,
    );
    if !report.global_blockers.is_empty() {
        let _ = writeln!(&mut rendered, "Global blockers:");
        for blocker in &report.global_blockers {
            let _ = writeln!(&mut rendered, "  - {blocker}");
        }
    }

    for target in &report.targets {
        let _ = writeln!(&mut rendered);
        let _ = writeln!(&mut rendered, "Target: {}", target.target.requested);
        let _ = writeln!(
            &mut rendered,
            "  Status: {}",
            render_target_status(target.status)
        );
        let _ = writeln!(&mut rendered, "  Selection: {}", target.selection);
        if let Some(resolved) = &target.target.resolved {
            let _ = writeln!(&mut rendered, "  Resolved: {resolved}");
        }
        if let Some(model_ref) = &target.target.config_model_ref {
            let _ = writeln!(&mut rendered, "  Config model: {model_ref}");
        }
        if let Some(reason) = &target.reason {
            let _ = writeln!(&mut rendered, "  Reason: {reason}");
        }
        if let Some(summary) = &target.field_summary {
            let _ = writeln!(
                &mut rendered,
                "  Review summary: applied={} preserved={} report_only={} unsupported={} error={}",
                summary.applied,
                summary.preserved,
                summary.report_only,
                summary.unsupported,
                summary.error,
            );
        }
        write_section(
            &mut rendered,
            "Config edits",
            &target.config_edits,
            render_config_edit_line,
        );
        write_section(
            &mut rendered,
            "Preserved",
            &collect_settings(target, TuneRenderedSettingStatus::Preserved),
            render_setting_line,
        );
        write_section(
            &mut rendered,
            "Report-only",
            &collect_settings(target, TuneRenderedSettingStatus::ReportOnly),
            render_setting_line,
        );
        write_section(
            &mut rendered,
            "Unsupported",
            &collect_settings(target, TuneRenderedSettingStatus::Unsupported),
            render_setting_line,
        );
        write_section(
            &mut rendered,
            "Errors",
            &collect_settings(target, TuneRenderedSettingStatus::Error),
            render_setting_line,
        );
        let warnings = target
            .diagnostics
            .iter()
            .filter(|diagnostic| matches!(diagnostic.severity, TuneDiagnosticSeverity::Warning))
            .collect::<Vec<_>>();
        if !warnings.is_empty() {
            let _ = writeln!(&mut rendered, "  Warnings:");
            for warning in warnings {
                let _ = writeln!(&mut rendered, "    - {}", warning.message);
            }
        }
    }
    write_benchmark_section(&mut rendered, &report.benchmarks);

    rendered
}

fn render_command_label(command: &str) -> &'static str {
    match command {
        "benchmark_tune" => "Benchmark tune",
        _ => "GPU tune",
    }
}

pub(crate) fn render_tune_launch_args_output(report: &TuneRunReport) -> String {
    let mut rendered = String::new();
    let _ = writeln!(&mut rendered, "# tune --launch-args");
    let _ = writeln!(
        &mut rendered,
        "# total={} ready={} written={} skipped={} failed={}",
        report.summary.total_targets,
        report.summary.ready_targets,
        report.summary.written_targets,
        report.summary.skipped_targets,
        report.summary.failed_targets,
    );
    for blocker in &report.global_blockers {
        let _ = writeln!(&mut rendered, "# blocker: {blocker}");
    }
    for target in &report.targets {
        let _ = writeln!(&mut rendered);
        let _ = writeln!(&mut rendered, "# target: {}", target.target.requested);
        let _ = writeln!(
            &mut rendered,
            "# status: {}",
            render_target_status(target.status)
        );
        if let Some(reason) = &target.reason {
            let _ = writeln!(&mut rendered, "# reason: {reason}");
        }
        match &target.launch {
            Some(launch) => {
                let _ = writeln!(&mut rendered, "{}", launch.shell);
                if !launch.config_settings.is_empty() {
                    let _ = writeln!(&mut rendered, "# effective config settings:");
                    for setting in &launch.config_settings {
                        let _ = writeln!(
                            &mut rendered,
                            "#   {} = {}",
                            setting.config_path,
                            render_recommended_value(&setting.value),
                        );
                    }
                }
                if !launch.report_only.is_empty() {
                    let _ = writeln!(&mut rendered, "# report-only:");
                    for setting in &launch.report_only {
                        let _ = writeln!(&mut rendered, "#   {}", render_setting_line(setting));
                    }
                }
                if !launch.unsupported.is_empty() {
                    let _ = writeln!(&mut rendered, "# unsupported:");
                    for setting in &launch.unsupported {
                        let _ = writeln!(&mut rendered, "#   {}", render_setting_line(setting));
                    }
                }
            }
            None => {
                let _ = writeln!(&mut rendered, "# no launch args emitted for this target");
            }
        }
    }
    rendered
}

fn write_benchmark_section(rendered: &mut String, benchmarks: &[TuneBenchmarkTargetReport]) {
    if benchmarks.is_empty() {
        return;
    }
    let _ = writeln!(rendered);
    let _ = writeln!(rendered, "Benchmark results:");
    for benchmark in benchmarks {
        let _ = writeln!(rendered, "  Target: {}", benchmark.requested);
        let _ = writeln!(
            rendered,
            "    Metrics schema: {}",
            render_metrics_schema(benchmark.metrics_schema),
        );
        match &benchmark.best {
            Some(best) => {
                let _ = writeln!(
                    rendered,
                    "    Recommended: {} decode_tok_s={}{}{}",
                    render_benchmark_candidate(&best.candidate),
                    best.decode_tok_s
                        .map(|value| format!("{value:.2}"))
                        .unwrap_or_else(|| "n/a".to_string()),
                    render_new_metrics(best),
                    render_timing_summary(best.timings.as_ref()),
                );
                if let Some(reason) = &benchmark.selection_reason {
                    let _ = writeln!(rendered, "      reason: {reason}");
                }
            }
            None => {
                let _ = writeln!(rendered, "    Recommended: none");
            }
        }
        if let Some(raw_best) = &benchmark.raw_best {
            let _ = writeln!(
                rendered,
                "    Raw best: {} decode_tok_s={}{}{}",
                render_benchmark_candidate(&raw_best.candidate),
                raw_best
                    .decode_tok_s
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_else(|| "n/a".to_string()),
                render_new_metrics(raw_best),
                render_timing_summary(raw_best.timings.as_ref()),
            );
        }
        if !benchmark.pareto_frontier.is_empty() {
            let _ = writeln!(rendered, "    Pareto frontier (decode tok/s vs ctx_size):");
            for trial in &benchmark.pareto_frontier {
                let _ = writeln!(
                    rendered,
                    "      - {} decode_tok_s={}{}{}",
                    render_benchmark_candidate(&trial.candidate),
                    trial
                        .decode_tok_s
                        .map(|value| format!("{value:.2}"))
                        .unwrap_or_else(|| "n/a".to_string()),
                    render_new_metrics(trial),
                    render_timing_summary(trial.timings.as_ref()),
                );
            }
        }
        let _ = writeln!(
            rendered,
            "    Throughput tolerance: {:.2}%",
            benchmark.throughput_tolerance_pct,
        );
        for trial in &benchmark.trials {
            let status = match trial.status {
                TuneBenchmarkTrialStatus::Succeeded => "ok",
                TuneBenchmarkTrialStatus::Failed => "failed",
            };
            let _ = write!(
                rendered,
                "    - {status}: {}",
                render_benchmark_candidate(&trial.candidate),
            );
            if let Some(rate) = trial.decode_tok_s {
                let _ = write!(rendered, " decode_tok_s={rate:.2}");
            }
            let _ = write!(rendered, "{}", render_new_metrics(trial));
            if let Some(timings) = &trial.timings {
                write_timing_fields(rendered, timings);
            }
            if let Some(error) = &trial.error {
                let _ = write!(rendered, " error={error}");
            }
            if let Some(log_path) = &trial.log_path {
                let _ = write!(rendered, " log={log_path}");
            }
            let _ = writeln!(rendered);
        }
    }
}

fn render_metrics_schema(schema: TuneBenchmarkMetricsSchema) -> &'static str {
    match schema {
        TuneBenchmarkMetricsSchema::NonStreamingHistorical => {
            "non_streaming_historical (ttft/decode-only not comparable)"
        }
        TuneBenchmarkMetricsSchema::StreamingV1 => "streaming_v1",
    }
}

fn render_new_metrics(trial: &TuneBenchmarkTrial) -> String {
    let mut rendered = String::new();
    if let Some(ttft_ms) = trial.ttft_ms {
        let _ = write!(&mut rendered, " ttft_ms={ttft_ms:.0}");
    }
    if let Some(rate) = trial.decode_only_tok_s {
        let _ = write!(&mut rendered, " decode_only_tok_s={rate:.2}");
    }
    rendered
}

fn render_timing_summary(timings: Option<&TuneBenchmarkTimingStats>) -> String {
    timings
        .map(|timings| {
            let request_ms = timings
                .request_ms
                .map(|value| format!("{value:.0}"))
                .unwrap_or_else(|| "n/a".to_string());
            format!(" request_ms={request_ms} total_ms={:.0}", timings.total_ms)
        })
        .unwrap_or_default()
}

fn write_timing_fields(rendered: &mut String, timings: &TuneBenchmarkTimingStats) {
    let _ = write!(
        rendered,
        " setup_ms={:.0} readiness_ms={:.0}",
        timings.setup_ms, timings.readiness_ms,
    );
    if let Some(request_ms) = timings.request_ms {
        let _ = write!(rendered, " request_ms={request_ms:.0}");
    }
    if let Some(shutdown_ms) = timings.shutdown_ms {
        let _ = write!(rendered, " shutdown_ms={shutdown_ms:.0}");
    }
    let _ = write!(
        rendered,
        " total_ms={:.0} readiness_attempts={}",
        timings.total_ms, timings.readiness_attempts,
    );
}

fn render_setting_line(setting: &TuneRenderedSetting) -> String {
    let mut rendered = format!(
        "{} ({})",
        setting.config_path,
        render_field_name(setting.field)
    );
    if let Some(value) = &setting.value {
        let _ = write!(&mut rendered, " = {}", render_recommended_value(value));
    }
    if let Some(reason) = &setting.reason {
        let _ = write!(&mut rendered, ": {reason}");
    }
    if let Some(rationale) = &setting.rationale {
        let _ = write!(&mut rendered, ": {rationale}");
    }
    if let Some(diagnostic) = &setting.diagnostic {
        let _ = write!(&mut rendered, ": {}", diagnostic.message);
    }
    rendered
}

fn render_config_edit_line(setting: &TuneRenderedSetting) -> String {
    let mut rendered = format!(
        "{} = {}",
        setting.config_path,
        setting
            .value
            .as_ref()
            .map(render_recommended_value)
            .unwrap_or_else(|| "<unknown>".to_string()),
    );
    if let Some(rationale) = &setting.rationale {
        let _ = write!(&mut rendered, " ({rationale})");
    }
    rendered
}

fn write_section(
    rendered: &mut String,
    title: &str,
    settings: &[TuneRenderedSetting],
    line_renderer: fn(&TuneRenderedSetting) -> String,
) {
    if settings.is_empty() {
        return;
    }
    let _ = writeln!(rendered, "  {title}:");
    for setting in settings {
        let _ = writeln!(rendered, "    - {}", line_renderer(setting));
    }
}
