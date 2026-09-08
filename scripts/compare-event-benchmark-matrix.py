#!/usr/bin/env python3
"""Paired, multiplicity-aware comparator for the event-system A/B benchmark
matrix (spec §17.5, `.omo/plans/event-system.md` task 19).

Consumes three manifests written by `run-event-benchmark-matrix.py`
(production/current, event-disabled/current, production/baseline) and
produces `comparison_a` (production vs event-disabled on the current
binary) and `comparison_b` (current-binary production vs baseline-binary
production).

Frozen statistical rules (verbatim from the plan's Verification Strategy --
none of this is negotiable by an executor):

- 20 valid pairs required for the primary comparison, 10 for each scenario.
  Invalid/null metrics are excluded PAIRWISE ONLY, never resampled or
  silently dropped to reach a minimum; fewer valid pairs than required is
  INVALID INPUT, not a downgraded pass.
- Fixed-seed 10,000-resample bootstrap 95% CIs on paired relative
  degradation. A metric FAILS when its entire two-sided 95% CI lies above
  the degradation threshold (default 3%).
- NO multiplicity correction may relax a fail. Holm-adjusted p-values
  across the predefined metric/scenario family are computed and reported
  for WORDING ONLY.
- A metric whose minimal detectable degradation (80% power, from the
  bootstrap standard error) exceeds `--max-mdd-percent` marks the screen
  UNDERPOWERED and BLOCKS certification.
- Correct wording is "not proven worse by this screen" -- never "proven
  within N%" or any phrasing that claims statistical proof.
- Callback ingress p99 must be measurable and within budget on the frozen
  certification hosts (macOS arm64 Metal, Linux x86_64 CUDA); Windows CPU
  is informational only. An unmeasurable p99 on a certification host BLOCKS
  rather than passes.
- Terminal/state drops must be zero; progress/diagnostic coalescing/drop
  counts must exactly match each manifest's recorded tiny-bound expected
  counts.
- Exactly one predefined FULL-SET retry is permitted after an adverse
  result, only after recording/correcting a thermal/load/runtime mismatch;
  a second adverse result blocks release. This script never retries
  individual favorable pairs -- `evaluate_retry_state` classifies a
  reported `attempt` number, it does not re-run anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

CORRECT_WORDING = "not proven worse by this screen"
FORBIDDEN_WORDING_SUBSTRINGS = ("proven within", "proven to be within", "statistically proven")

MANIFEST_SCHEMA_VERSION = 1
EXPECTED_METRICS_SCHEMA = "streaming_v1"

PRIMARY_SCENARIO = "__primary__"

# Higher-is-better throughput metrics vs lower-is-better latency metrics --
# the predefined metric family every comparison screens, in a fixed order
# so Holm adjustment operates over a stable, reproducible ordering.
METRIC_DIRECTIONS: dict[str, str] = {
    "decode_tok_s": "higher_better",
    "decode_only_tok_s": "higher_better",
    "ttft_ms": "lower_better",
}
METRICS: tuple[str, ...] = tuple(METRIC_DIRECTIONS)

CERTIFICATION_P99_BUDGET_US_DEFAULT = 100.0

# Two-sided 97.5th percentile and 80th-percentile standard-normal quantiles,
# used only to derive the minimal detectable degradation from the bootstrap
# standard error (80% power, two-sided alpha = 0.05).
Z_ALPHA_TWO_SIDED = 1.959963985
Z_BETA_80_POWER = 0.841621234


class InvalidManifestError(ValueError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise InvalidManifestError(f"{path}: unsupported schema_version {manifest.get('schema_version')!r}")
    if manifest.get("metrics_schema") != EXPECTED_METRICS_SCHEMA:
        raise InvalidManifestError(
            f"{path}: metrics_schema {manifest.get('metrics_schema')!r} is not comparable "
            f"(expected {EXPECTED_METRICS_SCHEMA!r}); a manifest without this exact marker "
            "is treated as non-comparable, never silently mixed into a paired comparison"
        )
    return manifest


def compare_environments(
    env_a: dict[str, dict[str, Any]],
    env_b: dict[str, dict[str, Any]],
    *,
    planned_selector: str | None,
) -> list[str]:
    """Both sides must carry IDENTICAL normalized allowlisted values and
    IDENTICAL redacted-presence sets, except `planned_selector` (the one
    intentional difference for this comparison), which is expected and
    excluded from the equality check entirely."""
    violations: list[str] = []
    names = set(env_a) | set(env_b)
    for name in sorted(names):
        if name == planned_selector:
            continue
        a = env_a.get(name)
        b = env_b.get(name)
        if a is None or b is None:
            violations.append(f"{name}: present on one side only")
            continue
        if a["redacted"] != b["redacted"]:
            violations.append(f"{name}: redacted-presence mismatch")
            continue
        if not a["redacted"] and a["value"] != b["value"]:
            violations.append(f"{name}: normalized value mismatch ({a['value']!r} vs {b['value']!r})")
    return violations


@dataclass(frozen=True)
class PairedMetricSample:
    key: tuple[str, int]
    baseline_value: float
    candidate_value: float


def index_trials_by_key(trials: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(trial["scenario"], trial["pair_index"]): trial for trial in trials}


def evaluate_executed_order_consistency(
    production_manifest: dict[str, Any],
    event_disabled_manifest: dict[str, Any],
) -> list[str]:
    """Replaces the label-only `side_order_first` consistency check: since
    Task 14, ONE runner invocation executes both sides of every pair back
    to back (see `run_paired_trial_plan` in the runner) and records the
    REAL observed order as a manifest-level `executed_order` list (`[{
    "scenario", "pair_index", "order": [first_side_id, second_side_id]},
    ...]`), rather than a per-trial `side_order_first` label with no
    execution effect. Both manifests come from the SAME invocation's SAME
    plan, so this verifies: (a) `executed_order` is present (non-null,
    non-empty) on BOTH manifests -- an absent value here means the runner
    never actually ran an interleaved plan for this pair of manifests; (b)
    the two manifests' `executed_order` lists are byte-equal -- any
    divergence means one side's manifest was corrupted or built from a
    different plan than the other; and (c) the recorded order is
    genuinely varied across pairs rather than a degenerate constant one --
    a constant ordering means the "side order randomized per pair"
    trial-unit requirement was not actually honored, even though the
    plan-derivation code ran."""
    production_order = production_manifest.get("executed_order")
    event_disabled_order = event_disabled_manifest.get("executed_order")
    violations: list[str] = []
    if not production_order:
        violations.append("production manifest is missing executed_order")
    if not event_disabled_order:
        violations.append("event_disabled manifest is missing executed_order")
    if violations:
        return violations
    if production_order != event_disabled_order:
        violations.append(
            "executed_order disagrees between the production and event_disabled manifests"
        )
        return violations
    observed_orders = {tuple(pair_entry["order"]) for pair_entry in production_order}
    if len(observed_orders) < 2:
        violations.append(
            "executed_order is constant across all matched pairs "
            f"(observed: {sorted(observed_orders)}); the 'randomized per pair' "
            "trial-unit requirement is not honored"
        )
    return violations


def pair_metric_samples(
    baseline_trials: Sequence[dict[str, Any]],
    candidate_trials: Sequence[dict[str, Any]],
    metric: str,
    group: str,
) -> tuple[list[PairedMetricSample], list[str]]:
    """Pairs trials in `group` (either `PRIMARY_SCENARIO` or a named
    scenario) by (scenario, pair_index). A key missing on either side, a
    non-`succeeded` trial, or a null metric value on either side is EXCLUDED
    from this metric's pairing. Non-numeric, non-finite, or non-positive
    baselines are excluded for the same reason: relative degradation requires
    a finite, positive denominator. Pairwise only, never resampled, never
    silently replaced. Returns (valid_samples, exclusion_reasons)."""
    baseline_index = index_trials_by_key(baseline_trials)
    candidate_index = index_trials_by_key(candidate_trials)
    keys = sorted(
        {key for key in baseline_index if key[0] == group} | {key for key in candidate_index if key[0] == group}
    )
    samples: list[PairedMetricSample] = []
    exclusions: list[str] = []
    for key in keys:
        baseline_trial = baseline_index.get(key)
        candidate_trial = candidate_index.get(key)
        if baseline_trial is None or candidate_trial is None:
            exclusions.append(f"{key}: missing on one side")
            continue
        if baseline_trial.get("status") != "succeeded" or candidate_trial.get("status") != "succeeded":
            exclusions.append(f"{key}: non-succeeded trial")
            continue
        baseline_value = baseline_trial.get(metric)
        candidate_value = candidate_trial.get(metric)
        if baseline_value is None or candidate_value is None:
            exclusions.append(f"{key}: null {metric}")
            continue
        if not _is_finite_number(baseline_value) or not _is_finite_number(candidate_value):
            exclusions.append(f"{key}: non-finite or non-numeric {metric}")
            continue
        if baseline_value <= 0:
            exclusions.append(f"{key}: non-positive baseline {metric}")
            continue
        samples.append(PairedMetricSample(key=key, baseline_value=baseline_value, candidate_value=candidate_value))
    return samples, exclusions


def _is_finite_number(value: object) -> bool:
    """Return whether a metric value is a finite JSON-number-like scalar.

    ``math.isfinite`` converts integers to a C double first. Python raises
    ``OverflowError`` for an integer too large for that conversion, and a
    malformed benchmark manifest must be excluded rather than crash the
    comparison.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def relative_degradation(baseline_value: float, candidate_value: float, direction: str) -> float:
    """Positive == candidate is worse than baseline. `baseline_value` of
    zero or less is treated as an unusable pair by the caller before reaching
    here (comparator excludes it during pairing, not here)."""
    if direction == "higher_better":
        return (baseline_value - candidate_value) / baseline_value
    if direction == "lower_better":
        return (candidate_value - baseline_value) / baseline_value
    raise ValueError(f"unknown metric direction {direction!r}")


def bootstrap_ci(
    deltas: Sequence[float], samples: int, seed: int
) -> tuple[float, float, float, list[float]]:
    """Fixed-seed, samples-resample-with-replacement percentile bootstrap.
    Returns (mean, ci_low, ci_high, resample_means) -- pure Python, no
    numpy, matching this repo's stdlib-only script convention."""
    n = len(deltas)
    if n == 0:
        raise ValueError("cannot bootstrap zero pairs")
    mean = sum(deltas) / n
    if n == 1:
        return mean, deltas[0], deltas[0], [deltas[0]] * samples
    rng = random.Random(seed)
    resample_means = []
    for _ in range(samples):
        resample_sum = 0.0
        for _ in range(n):
            resample_sum += deltas[rng.randrange(n)]
        resample_means.append(resample_sum / n)
    sorted_means = sorted(resample_means)
    lo_idx = max(0, min(samples - 1, round(0.025 * (samples - 1))))
    hi_idx = max(0, min(samples - 1, round(0.975 * (samples - 1))))
    return mean, sorted_means[lo_idx], sorted_means[hi_idx], resample_means


def bootstrap_standard_error(resample_means: Sequence[float]) -> float:
    n = len(resample_means)
    if n < 2:
        return 0.0
    mean = sum(resample_means) / n
    variance = sum((value - mean) ** 2 for value in resample_means) / (n - 1)
    return math.sqrt(variance)


def minimal_detectable_degradation(standard_error: float) -> float:
    """Minimal detectable degradation at 80% power, two-sided alpha=0.05,
    from the bootstrap standard error: `(z_alpha/2 + z_beta) * SE`."""
    return (Z_ALPHA_TWO_SIDED + Z_BETA_80_POWER) * standard_error


def bootstrap_two_sided_p_value(resample_means: Sequence[float]) -> float:
    """A standard bootstrap two-sided p-value against the null of zero
    degradation: `2 * min(P(resample <= 0), P(resample >= 0))`, capped at
    1.0. Used ONLY to feed Holm adjustment for WORDING -- see module
    docstring; it never changes a fail decided by the raw CI."""
    n = len(resample_means)
    if n == 0:
        return 1.0
    below = sum(1 for value in resample_means if value <= 0.0) / n
    above = sum(1 for value in resample_means if value >= 0.0) / n
    return min(1.0, 2.0 * min(below, above))


def holm_adjusted_p_values(p_values: Sequence[float]) -> list[float]:
    """Standard Holm step-down adjustment, monotone non-decreasing by
    construction. Reported for WORDING ONLY -- see module docstring: this
    NEVER relaxes a screening fail decided by the raw bootstrap CI."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min((m - rank) * p_values[idx], 1.0)
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


@dataclass(frozen=True)
class MetricScreenResult:
    group: str
    metric: str
    valid_pairs: int
    required_pairs: int
    status: str  # "pass" | "fail" | "underpowered" | "invalid_input"
    mean_relative_degradation: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    minimal_detectable_degradation: float | None = None
    raw_p_value: float | None = None
    exclusions: tuple[str, ...] = ()


def metric_seed(root_seed: int, group: str, metric: str) -> int:
    """Derive a stable per-metric bootstrap seed from the root seed.

    Python's built-in ``hash`` is randomized per process, so it cannot
    provide reproducible reports across subprocesses. Hashing an explicit,
    length-delimited byte representation keeps the root seed in the mixing
    input while making the derived stream independent of ``PYTHONHASHSEED``.
    """
    root = root_seed & ((1 << 64) - 1)
    payload = b"event-benchmark-metric-seed\0" + root.to_bytes(8, "big")
    for value in (group, metric):
        encoded = value.encode("utf-8")
        payload += len(encoded).to_bytes(4, "big") + encoded
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def screen_metric(
    baseline_trials: Sequence[dict[str, Any]],
    candidate_trials: Sequence[dict[str, Any]],
    group: str,
    metric: str,
    *,
    required_pairs: int,
    bootstrap_samples: int,
    seed: int,
    max_degradation_fraction: float,
    max_mdd_fraction: float,
) -> MetricScreenResult:
    samples, exclusions = pair_metric_samples(baseline_trials, candidate_trials, metric, group)
    if len(samples) < required_pairs:
        return MetricScreenResult(
            group=group,
            metric=metric,
            valid_pairs=len(samples),
            required_pairs=required_pairs,
            status="invalid_input",
            exclusions=tuple(exclusions),
        )
    direction = METRIC_DIRECTIONS[metric]
    deltas = [relative_degradation(s.baseline_value, s.candidate_value, direction) for s in samples]
    # Seed the bootstrap deterministically per (group, metric) so re-running
    # the comparator with the same --seed reproduces byte-identical CIs,
    # while different metrics/groups never share a resampling stream.
    mean, ci_low, ci_high, resample_means = bootstrap_ci(
        deltas, bootstrap_samples, metric_seed(seed, group, metric)
    )
    standard_error = bootstrap_standard_error(resample_means)
    mdd = minimal_detectable_degradation(standard_error)
    raw_p_value = bootstrap_two_sided_p_value(resample_means)

    if ci_low > max_degradation_fraction:
        status = "fail"
    elif mdd > max_mdd_fraction:
        status = "underpowered"
    else:
        status = "pass"

    return MetricScreenResult(
        group=group,
        metric=metric,
        valid_pairs=len(samples),
        required_pairs=required_pairs,
        status=status,
        mean_relative_degradation=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        minimal_detectable_degradation=mdd,
        raw_p_value=raw_p_value,
        exclusions=tuple(exclusions),
    )


def screened_groups(scenarios: Sequence[str]) -> list[str]:
    return [PRIMARY_SCENARIO, *scenarios]


def run_comparison_screen(
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    *,
    min_primary_pairs: int,
    min_scenario_pairs: int,
    bootstrap_samples: int,
    seed: int,
    max_degradation_percent: float,
    max_mdd_percent: float,
) -> list[MetricScreenResult]:
    scenarios = candidate_manifest.get("scenarios", [])
    groups = screened_groups(scenarios)
    baseline_trials = baseline_manifest.get("trials", [])
    candidate_trials = candidate_manifest.get("trials", [])
    max_degradation_fraction = max_degradation_percent / 100.0
    max_mdd_fraction = max_mdd_percent / 100.0
    results: list[MetricScreenResult] = []
    for group in groups:
        required = min_primary_pairs if group == PRIMARY_SCENARIO else min_scenario_pairs
        for metric in METRICS:
            results.append(
                screen_metric(
                    baseline_trials,
                    candidate_trials,
                    group,
                    metric,
                    required_pairs=required,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                    max_degradation_fraction=max_degradation_fraction,
                    max_mdd_fraction=max_mdd_fraction,
                )
            )
    return results


def apply_holm_wording(results: Sequence[MetricScreenResult]) -> list[dict[str, Any]]:
    scored = [r for r in results if r.raw_p_value is not None]
    adjusted = holm_adjusted_p_values([r.raw_p_value for r in scored])
    adjusted_by_key = {(r.group, r.metric): value for r, value in zip(scored, adjusted)}
    report: list[dict[str, Any]] = []
    for result in results:
        entry = {
            "group": result.group,
            "metric": result.metric,
            "status": result.status,
            "valid_pairs": result.valid_pairs,
            "required_pairs": result.required_pairs,
            "mean_relative_degradation_pct": _pct(result.mean_relative_degradation),
            "ci_low_pct": _pct(result.ci_low),
            "ci_high_pct": _pct(result.ci_high),
            "minimal_detectable_degradation_pct": _pct(result.minimal_detectable_degradation),
            "raw_p_value": result.raw_p_value,
            "holm_adjusted_p_value": adjusted_by_key.get((result.group, result.metric)),
            "exclusions": list(result.exclusions),
            "wording": _wording_for_status(result.status),
        }
        report.append(entry)
    return report


def _pct(fraction: float | None) -> float | None:
    return None if fraction is None else fraction * 100.0


def _wording_for_status(status: str) -> str:
    if status == "pass":
        return CORRECT_WORDING
    if status == "fail":
        return "this screen found a degradation whose 95% CI lies entirely above the threshold"
    if status == "underpowered":
        return "UNDERPOWERED: minimal detectable degradation exceeds the configured bound; collect more pairs"
    return "invalid input: fewer valid pairs than required"


def overall_status(results: Sequence[MetricScreenResult]) -> str:
    """Fail beats underpowered beats invalid_input beats pass -- multiplicity
    correction is never consulted here, only the raw per-metric status."""
    statuses = {result.status for result in results}
    if "fail" in statuses:
        return "fail"
    if "invalid_input" in statuses:
        return "invalid_input"
    if "underpowered" in statuses:
        return "underpowered"
    return "pass"


@dataclass(frozen=True)
class RetryDecision:
    action: str  # "accept" | "retry_permitted" | "blocked_retry_exhausted"
    reason: str


def evaluate_retry_state(current_verdict: str, attempt: int) -> RetryDecision:
    """Exactly one predefined full-set retry is permitted after an adverse
    (non-`pass`) result, and only after recording/correcting a
    thermal/load/runtime mismatch on the first attempt. A second adverse
    result -- at `attempt >= 2` -- blocks release; there is no third
    attempt. This function classifies a reported `attempt` number; it never
    re-runs anything itself (individual favorable-pair retries are
    forbidden outright and have no code path here at all)."""
    if current_verdict == "pass":
        return RetryDecision("accept", "screen passed")
    if attempt <= 1:
        return RetryDecision(
            "retry_permitted",
            "first adverse result: one full-set retry is permitted after recording "
            "and correcting a thermal/load/runtime mismatch",
        )
    return RetryDecision(
        "blocked_retry_exhausted",
        "second adverse result after the one predefined full-set retry: release is blocked",
    )


def evaluate_p99_gate(
    production_manifest: dict[str, Any],
    event_disabled_manifest: dict[str, Any],
    max_p99_us: float,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Callback ingress p99 must be measurable and within `max_p99_us` on a
    frozen certification host; Windows CPU (and any other non-certification
    host) is informational only. An unmeasurable p99 on a certification
    host BLOCKS rather than passes -- this is the one gate where "we don't
    know" is treated as a failure, not a skip."""
    results: dict[str, dict[str, Any]] = {}
    blocking = False
    for label, manifest in (("production", production_manifest), ("event_disabled", event_disabled_manifest)):
        host = manifest.get("host", {})
        p99 = manifest.get("callback_ingress_p99_us")
        if host.get("p99_gate") != "enforced":
            results[label] = {"status": "informational", "value": p99}
            continue
        if p99 is None:
            results[label] = {"status": "blocked", "reason": "p99 unmeasurable on a certification host"}
            blocking = True
            continue
        if p99 > max_p99_us:
            results[label] = {
                "status": "blocked",
                "value": p99,
                "reason": f"p99 {p99}us exceeds the {max_p99_us}us budget",
            }
            blocking = True
            continue
        results[label] = {"status": "ok", "value": p99}
    return results, blocking


def health_is_available(manifest: dict[str, Any]) -> bool:
    """A manifest's `health` block is `None` (JSON null) when no call site
    collected it -- e.g. every real `--local-model-only` trial run today,
    which starts no console/management API (see the runner's `build_manifest`
    comment). Distinguishing "unavailable" from "present" is the caller's
    job precisely so an unavailable block is never misread as an all-zero
    one -- see `evaluate_health_expectations`."""
    return manifest.get("health") is not None


def evaluate_health_expectations(manifest: dict[str, Any]) -> list[str]:
    """Terminal/state drops must be zero; progress/diagnostic
    coalescing/drop counts must exactly match this manifest's own recorded
    tiny-bound expected counts (see `summarize_health_expectations` in the
    runner). Returns `[]` when `health` is unavailable (`None`) -- an
    unavailable block must NEVER be silently read as all-zero counts (that
    would vacuously pass every `!= 0` check below and falsely flag a real
    expected-count mismatch that was never actually measured). Callers
    MUST separately gate on `health_is_available` for the `health_unavailable`
    blocking reason: an empty violations list here means "no violation
    found", not "healthy" -- those are different claims when the data was
    never collected at all. Intentional cancelled-reservation rejections are
    retained as a non-negative health count; a positive count does not imply
    state degradation or capacity loss."""
    health = manifest.get("health")
    if health is None:
        return []
    violations: list[str] = []
    if health.get("terminal_delivery_failed", 0) != 0:
        violations.append("terminal_delivery_failed must be zero")
    if health.get("state_lane_evictions", 0) != 0:
        violations.append("state-transition drops/evictions must be zero")
    if health.get("state_transition_rejected", 0) != 0:
        violations.append("state_transition_rejected must be zero")
    cancelled_reservation_rejected = health.get("cancelled_reservation_rejected", 0)
    if (
        isinstance(cancelled_reservation_rejected, bool)
        or not isinstance(cancelled_reservation_rejected, int)
        or cancelled_reservation_rejected < 0
    ):
        violations.append(
            "cancelled_reservation_rejected must be a non-negative integer"
        )
    for field in ("state_degraded", "rebuild_required"):
        if health.get(field, False):
            violations.append(f"{field} must be false")
    expected_progress = manifest.get("expected_dropped_progress")
    if expected_progress is not None and health.get("dropped_progress", 0) != expected_progress:
        violations.append(
            f"dropped_progress {health.get('dropped_progress', 0)} does not exactly match "
            f"the expected count {expected_progress}"
        )
    expected_diagnostic = manifest.get("expected_dropped_diagnostic")
    if expected_diagnostic is not None and health.get("dropped_diagnostic", 0) != expected_diagnostic:
        violations.append(
            f"dropped_diagnostic {health.get('dropped_diagnostic', 0)} does not exactly match "
            f"the expected count {expected_diagnostic}"
        )
    return violations


def check_seed_consistency(manifests: dict[str, dict[str, Any]], expected_seed: int) -> list[str]:
    violations = []
    for label, manifest in manifests.items():
        if manifest.get("seed") != expected_seed:
            violations.append(
                f"{label}: manifest seed {manifest.get('seed')!r} does not match --seed {expected_seed}; "
                "pairing requires the SAME seed on every side"
            )
    return violations


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    production = load_manifest(args.production)
    event_disabled = load_manifest(args.event_disabled)
    baseline = load_manifest(args.baseline)

    manifests = {"production": production, "event_disabled": event_disabled, "baseline": baseline}
    input_violations = check_seed_consistency(manifests, args.seed)

    env_violations_a = compare_environments(
        production["environment"], event_disabled["environment"], planned_selector="MESH_LLM_EVENT_SYSTEM_TRIAL_MODE"
    )
    env_violations_b = compare_environments(production["environment"], baseline["environment"], planned_selector=None)

    screen_kwargs = dict(
        min_primary_pairs=args.min_primary_pairs,
        min_scenario_pairs=args.min_scenario_pairs,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        max_degradation_percent=args.max_degradation_percent,
        max_mdd_percent=args.max_mdd_percent,
    )
    # Comparison A: is production (candidate) not proven worse than
    # event-disabled (baseline/reference) on the SAME current binary.
    comparison_a_results = run_comparison_screen(event_disabled, production, **screen_kwargs)
    # Comparison B: is the current binary's production behavior (candidate)
    # not proven worse than the verified baseline release binary's
    # (baseline/reference).
    comparison_b_results = run_comparison_screen(baseline, production, **screen_kwargs)

    comparison_a_status = overall_status(comparison_a_results)
    comparison_b_status = overall_status(comparison_b_results)

    p99_results, p99_blocking = evaluate_p99_gate(production, event_disabled, CERTIFICATION_P99_BUDGET_US_DEFAULT)

    health_availability = {
        "production": health_is_available(production),
        "event_disabled": health_is_available(event_disabled),
        "baseline": health_is_available(baseline),
    }
    health_violations = {
        "production": evaluate_health_expectations(production),
        "event_disabled": evaluate_health_expectations(event_disabled),
        "baseline": evaluate_health_expectations(baseline),
    }
    any_health_unavailable = not all(health_availability.values())
    any_health_violation = any(health_violations.values())

    executed_order_violations = evaluate_executed_order_consistency(production, event_disabled)

    attempt = int(production.get("attempt", 1))
    overall_adverse = comparison_a_status != "pass" or comparison_b_status != "pass"
    retry = evaluate_retry_state("fail" if overall_adverse else "pass", attempt)

    blocking_reasons: list[str] = []
    if input_violations:
        blocking_reasons.append("seed_mismatch")
    if env_violations_a:
        blocking_reasons.append("environment_mismatch_comparison_a")
    if env_violations_b:
        blocking_reasons.append("environment_mismatch_comparison_b")
    if comparison_a_status == "fail" or comparison_b_status == "fail":
        blocking_reasons.append("degradation_fail")
    if comparison_a_status == "underpowered" or comparison_b_status == "underpowered":
        blocking_reasons.append("underpowered")
    if comparison_a_status == "invalid_input" or comparison_b_status == "invalid_input":
        blocking_reasons.append("insufficient_pairs")
    if p99_blocking:
        blocking_reasons.append("callback_ingress_p99")
    if any_health_unavailable:
        blocking_reasons.append("health_unavailable")
    if any_health_violation:
        blocking_reasons.append("health_expectation_violation")
    if executed_order_violations:
        blocking_reasons.append("executed_order_inconsistent")
    if retry.action == "blocked_retry_exhausted":
        blocking_reasons.append("retry_exhausted")

    certification_status = "pass" if not blocking_reasons else "blocked"

    report = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_with_seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "max_degradation_percent": args.max_degradation_percent,
        "max_mdd_percent": args.max_mdd_percent,
        "min_primary_pairs": args.min_primary_pairs,
        "min_scenario_pairs": args.min_scenario_pairs,
        "report_holm": bool(args.report_holm),
        "input_violations": input_violations,
        "comparison_a": {
            "description": "production vs event-disabled on the current binary",
            "status": comparison_a_status,
            "environment_violations": env_violations_a,
            "metrics": apply_holm_wording(comparison_a_results) if args.report_holm else [
                {
                    "group": r.group,
                    "metric": r.metric,
                    "status": r.status,
                    "valid_pairs": r.valid_pairs,
                    "required_pairs": r.required_pairs,
                    "mean_relative_degradation_pct": _pct(r.mean_relative_degradation),
                    "ci_low_pct": _pct(r.ci_low),
                    "ci_high_pct": _pct(r.ci_high),
                    "minimal_detectable_degradation_pct": _pct(r.minimal_detectable_degradation),
                    "exclusions": list(r.exclusions),
                    "wording": _wording_for_status(r.status),
                }
                for r in comparison_a_results
            ],
        },
        "comparison_b": {
            "description": "current-binary production vs baseline-binary production",
            "status": comparison_b_status,
            "environment_violations": env_violations_b,
            "metrics": apply_holm_wording(comparison_b_results) if args.report_holm else [
                {
                    "group": r.group,
                    "metric": r.metric,
                    "status": r.status,
                    "valid_pairs": r.valid_pairs,
                    "required_pairs": r.required_pairs,
                    "mean_relative_degradation_pct": _pct(r.mean_relative_degradation),
                    "ci_low_pct": _pct(r.ci_low),
                    "ci_high_pct": _pct(r.ci_high),
                    "minimal_detectable_degradation_pct": _pct(r.minimal_detectable_degradation),
                    "exclusions": list(r.exclusions),
                    "wording": _wording_for_status(r.status),
                }
                for r in comparison_b_results
            ],
        },
        "callback_ingress_p99": p99_results,
        "health": health_violations,
        "health_availability": health_availability,
        "executed_order_violations": executed_order_violations,
        "retry": {"attempt": attempt, "action": retry.action, "reason": retry.reason},
        "binary_identity": {
            "production": production.get("binary"),
            "event_disabled": event_disabled.get("binary"),
            "baseline": baseline.get("binary"),
        },
        "thermal_state": {
            "production": production.get("thermal_state"),
            "event_disabled": event_disabled.get("thermal_state"),
            "baseline": baseline.get("thermal_state"),
        },
        "certification_status": certification_status,
        "blocking_reasons": blocking_reasons,
        "wording": CORRECT_WORDING,
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare-event-benchmark-matrix.py",
        description="Paired, multiplicity-aware comparator for the event-system A/B benchmark matrix.",
    )
    parser.add_argument("--production", required=True, type=Path, help="Production-mode manifest (current binary).")
    parser.add_argument(
        "--event-disabled", required=True, type=Path, dest="event_disabled", help="event-disabled manifest (current binary)."
    )
    parser.add_argument("--baseline", required=True, type=Path, help="Production-mode manifest (baseline release binary).")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the comparison report JSON.")
    parser.add_argument("--bootstrap-samples", required=True, type=int, dest="bootstrap_samples")
    parser.add_argument("--seed", required=True, type=int, help="Bootstrap seed; reuse the SAME seed the runner used.")
    parser.add_argument("--max-degradation-percent", required=True, type=float, dest="max_degradation_percent")
    parser.add_argument("--min-primary-pairs", required=True, type=int, dest="min_primary_pairs")
    parser.add_argument("--min-scenario-pairs", required=True, type=int, dest="min_scenario_pairs")
    parser.add_argument("--max-mdd-percent", required=True, type=float, dest="max_mdd_percent")
    parser.add_argument("--report-holm", action="store_true", dest="report_holm")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = build_report(args)
    except InvalidManifestError as exc:
        parser.error(str(exc))
        return 2

    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    for forbidden in FORBIDDEN_WORDING_SUBSTRINGS:
        assert forbidden not in serialized, f"forbidden wording {forbidden!r} leaked into the report"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    print(json.dumps({"output_path": str(args.output), "certification_status": report["certification_status"]}))
    return 0 if report["certification_status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
