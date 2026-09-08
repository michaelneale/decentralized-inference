from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compare-event-benchmark-matrix.py"


def load_module():
    module_name = "compare_event_benchmark_matrix"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # See the paired runner test's load_module() for why this must happen
    # BEFORE exec_module (a Python 3.9 dataclasses/sys.modules quirk).
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_trial(
    scenario,
    pair_index,
    *,
    decode_tok_s=50.0,
    ttft_ms=100.0,
    decode_only_tok_s=55.0,
    status="succeeded",
    side_order_first="production",
):
    return {
        "scenario": scenario,
        "pair_index": pair_index,
        "side_order_first": side_order_first,
        "status": status,
        "completion_tokens": 100,
        "elapsed_ms": 2000.0,
        "decode_tok_s": decode_tok_s,
        "ttft_ms": ttft_ms,
        "decode_only_tok_s": decode_only_tok_s,
        "setup_ms": 10.0,
        "readiness_ms": 20.0,
        "shutdown_ms": 5.0,
        "error": None,
    }


def alternating_trials(scenario, count):
    """Trial fixtures with a genuinely-varied (non-constant)
    `side_order_first`, matching what a real deterministic plan produces --
    use this instead of the `make_trial` default in any fixture that feeds
    `build_report`, so the side-order-variety gate doesn't spuriously fire
    for tests that aren't exercising that gate."""
    return [
        make_trial(scenario, i, side_order_first="production" if i % 2 == 0 else "event-disabled")
        for i in range(count)
    ]


def alternating_executed_order(scenario, count):
    """Manifest-level `executed_order` fixture matching what
    `run_paired_trial_plan` (the runner) actually records: a genuinely
    varied (non-constant) order across pairs -- use this as `make_manifest`'s
    default so the `evaluate_executed_order_consistency` gate doesn't
    spuriously fire for tests that feed `build_report` without exercising
    that gate directly."""
    return [
        {
            "scenario": scenario,
            "pair_index": i,
            "order": ["production", "event-disabled"] if i % 2 == 0 else ["event-disabled", "production"],
        }
        for i in range(count)
    ]


_UNSET = object()


def make_manifest(
    *,
    mode="production",
    seed=42,
    pairs_primary=20,
    scenarios=None,
    trials=None,
    environment=None,
    host=None,
    thermal_state=None,
    callback_ingress_p99_us=50.0,
    health=_UNSET,
    expected_dropped_progress=0,
    expected_dropped_diagnostic=0,
    attempt=1,
    binary_path="/fixtures/mesh-llm",
    executed_order=_UNSET,
):
    scenarios = scenarios if scenarios is not None else []
    trials = trials if trials is not None else alternating_trials("__primary__", pairs_primary)
    if executed_order is _UNSET:
        executed_order = alternating_executed_order("__primary__", pairs_primary)
    # `executed_order=None` (distinct from omitting the arg) explicitly
    # represents a manifest that never ran an interleaved plan -- passed
    # through unchanged, never defaulted to a fabricated list.
    environment = (
        environment
        if environment is not None
        else {
            "MESH_LLM_LIFECYCLE_LOG_PARSER": {"value": "auto", "redacted": False},
            "MESH_LLM_BENCHMARK_TUNE_TRIAL": {"value": True, "redacted": False},
            "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE": {"value": mode, "redacted": False},
        }
    )
    host = host if host is not None else {
        "system": "Darwin",
        "machine": "arm64",
        "certification_host": "macos-arm64-metal",
        "p99_gate": "enforced",
    }
    thermal_state = thermal_state if thermal_state is not None else {"available": True, "source": "pmset -g therm", "raw": "No thermal warning"}
    if health is _UNSET:
        health = {
            "terminal_delivery_failed": 0,
            "cancelled_reservation_rejected": 0,
            "dropped_progress": expected_dropped_progress,
            "dropped_diagnostic": expected_dropped_diagnostic,
        }
    # `health=None` (distinct from omitting the arg) explicitly represents
    # an unavailable health block (JSON null), e.g. a real
    # `--local-model-only` trial run where no live console API ever
    # collected health counters -- passed through unchanged, never
    # defaulted to a fabricated dict.
    return {
        "schema_version": 1,
        "metrics_schema": "streaming_v1",
        "mode": mode,
        "binary": {"path": binary_path, "sha256": "a" * 64, "version": "mesh-llm 0.76.0"},
        "model": "fixture-model",
        "seed": seed,
        "pairs_primary": pairs_primary,
        "pairs_scenario": 10,
        "scenarios": scenarios,
        "attempt": attempt,
        "generated_at": "2026-01-01T00:00:00Z",
        "host": host,
        "thermal_state": thermal_state,
        "environment": environment,
        "trial_unit": {"trial": "one trial", "pair": "one pair"},
        "callback_ingress_p99_us": callback_ingress_p99_us,
        "health": health,
        "expected_dropped_progress": expected_dropped_progress,
        "expected_dropped_diagnostic": expected_dropped_diagnostic,
        "trials": trials,
        "executed_order": executed_order,
    }


def write_manifest(directory: Path, name: str, manifest: dict) -> Path:
    path = directory / name
    path.write_text(json.dumps(manifest))
    return path


def build_argv(
    production: Path,
    event_disabled: Path,
    baseline: Path,
    output: Path,
    *,
    bootstrap_samples: int = 200,
    seed: int = 42,
    max_degradation_percent: int = 3,
    min_primary_pairs: int = 20,
    min_scenario_pairs: int = 10,
    max_mdd_percent: int = 10,
    report_holm: bool = False,
) -> list[str]:
    argv = [
        "--production",
        str(production),
        "--event-disabled",
        str(event_disabled),
        "--baseline",
        str(baseline),
        "--output",
        str(output),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--seed",
        str(seed),
        "--max-degradation-percent",
        str(max_degradation_percent),
        "--min-primary-pairs",
        str(min_primary_pairs),
        "--min-scenario-pairs",
        str(min_scenario_pairs),
        "--max-mdd-percent",
        str(max_mdd_percent),
    ]
    if report_holm:
        argv.append("--report-holm")
    return argv


class ManifestLoadingTests(unittest.TestCase):
    def test_rejects_non_streaming_schema(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_manifest()
            manifest["metrics_schema"] = "non_streaming_historical"
            path = write_manifest(Path(tmp), "m.json", manifest)
            with self.assertRaises(harness.InvalidManifestError):
                harness.load_manifest(path)

    def test_rejects_missing_metrics_schema_field(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_manifest()
            del manifest["metrics_schema"]
            path = write_manifest(Path(tmp), "m.json", manifest)
            with self.assertRaises(harness.InvalidManifestError):
                harness.load_manifest(path)

    def test_rejects_unsupported_schema_version(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_manifest()
            manifest["schema_version"] = 2
            path = write_manifest(Path(tmp), "m.json", manifest)
            with self.assertRaises(harness.InvalidManifestError):
                harness.load_manifest(path)

    def test_accepts_a_valid_manifest(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(Path(tmp), "m.json", make_manifest())
            loaded = harness.load_manifest(path)
            self.assertEqual(loaded["mode"], "production")


class EnvironmentComparisonTests(unittest.TestCase):
    def test_identical_environments_have_no_violations(self):
        harness = load_module()
        env = {"MESH_LLM_LIFECYCLE_LOG_PARSER": {"value": "auto", "redacted": False}}
        self.assertEqual(harness.compare_environments(env, dict(env), planned_selector=None), [])

    def test_planned_selector_difference_is_excluded(self):
        harness = load_module()
        env_a = {"MESH_LLM_EVENT_SYSTEM_TRIAL_MODE": {"value": "production", "redacted": False}}
        env_b = {"MESH_LLM_EVENT_SYSTEM_TRIAL_MODE": {"value": "event-disabled", "redacted": False}}
        violations = harness.compare_environments(
            env_a, env_b, planned_selector="MESH_LLM_EVENT_SYSTEM_TRIAL_MODE"
        )
        self.assertEqual(violations, [])

    def test_unplanned_value_mismatch_is_a_violation(self):
        harness = load_module()
        env_a = {"MESH_LLM_LIFECYCLE_LOG_PARSER": {"value": "auto", "redacted": False}}
        env_b = {"MESH_LLM_LIFECYCLE_LOG_PARSER": {"value": "disabled", "redacted": False}}
        violations = harness.compare_environments(env_a, env_b, planned_selector=None)
        self.assertEqual(len(violations), 1)
        self.assertIn("MESH_LLM_LIFECYCLE_LOG_PARSER", violations[0])

    def test_redacted_presence_mismatch_is_a_violation(self):
        harness = load_module()
        env_a = {"MESH_LLM_CONFIG": {"value": "<redacted:present>", "redacted": True}}
        env_b = {"MESH_LLM_CONFIG": {"value": "/tmp/other.toml", "redacted": False}}
        violations = harness.compare_environments(env_a, env_b, planned_selector=None)
        self.assertEqual(len(violations), 1)

    def test_present_on_one_side_only_is_a_violation(self):
        harness = load_module()
        env_a = {"MESH_LLM_LIFECYCLE_LOG_PARSER": {"value": "auto", "redacted": False}}
        violations = harness.compare_environments(env_a, {}, planned_selector=None)
        self.assertEqual(len(violations), 1)


class PairingAndNullHandlingTests(unittest.TestCase):
    def test_null_metric_is_excluded_pairwise_not_the_whole_trial(self):
        harness = load_module()
        baseline = [make_trial("__primary__", 0, ttft_ms=None)]
        candidate = [make_trial("__primary__", 0, ttft_ms=None)]
        samples, exclusions = harness.pair_metric_samples(baseline, candidate, "ttft_ms", "__primary__")
        self.assertEqual(samples, [])
        self.assertEqual(len(exclusions), 1)
        self.assertIn("null ttft_ms", exclusions[0])

        # A DIFFERENT metric on the same pair, both non-null, still pairs.
        decode_samples, decode_exclusions = harness.pair_metric_samples(
            baseline, candidate, "decode_tok_s", "__primary__"
        )
        self.assertEqual(len(decode_samples), 1)
        self.assertEqual(decode_exclusions, [])

    def test_missing_on_one_side_is_excluded_not_silently_dropped_to_reach_minimum(self):
        harness = load_module()
        baseline = [make_trial("__primary__", 0), make_trial("__primary__", 1)]
        candidate = [make_trial("__primary__", 0)]
        samples, exclusions = harness.pair_metric_samples(baseline, candidate, "decode_tok_s", "__primary__")
        self.assertEqual(len(samples), 1)
        self.assertEqual(len(exclusions), 1)
        self.assertIn("missing on one side", exclusions[0])

    def test_non_succeeded_trial_is_excluded(self):
        harness = load_module()
        baseline = [make_trial("__primary__", 0, status="failed")]
        candidate = [make_trial("__primary__", 0)]
        samples, exclusions = harness.pair_metric_samples(baseline, candidate, "decode_tok_s", "__primary__")
        self.assertEqual(samples, [])
        self.assertIn("non-succeeded", exclusions[0])

    def test_non_positive_baseline_is_excluded_before_relative_degradation(self):
        harness = load_module()
        baseline = [make_trial("__primary__", 0, decode_tok_s=0.0)]
        candidate = [make_trial("__primary__", 0, decode_tok_s=10.0)]
        samples, exclusions = harness.pair_metric_samples(
            baseline, candidate, "decode_tok_s", "__primary__"
        )
        self.assertEqual(samples, [])
        self.assertEqual(len(exclusions), 1)
        self.assertIn("non-positive baseline decode_tok_s", exclusions[0])

    def test_non_finite_metric_is_excluded_and_reported(self):
        harness = load_module()
        baseline = [make_trial("__primary__", 0, decode_tok_s=float("nan"))]
        candidate = [make_trial("__primary__", 0, decode_tok_s=10.0)]
        samples, exclusions = harness.pair_metric_samples(
            baseline, candidate, "decode_tok_s", "__primary__"
        )
        self.assertEqual(samples, [])
        self.assertEqual(len(exclusions), 1)
        self.assertIn("non-finite or non-numeric decode_tok_s", exclusions[0])

    def test_oversized_integer_metric_is_excluded_without_overflow(self):
        harness = load_module()
        oversized = 10**1000
        baseline = [make_trial("__primary__", 0, decode_tok_s=oversized)]
        candidate = [make_trial("__primary__", 0, decode_tok_s=oversized)]
        samples, exclusions = harness.pair_metric_samples(
            baseline, candidate, "decode_tok_s", "__primary__"
        )
        self.assertEqual(samples, [])
        self.assertEqual(len(exclusions), 1)
        self.assertIn("non-finite or non-numeric decode_tok_s", exclusions[0])


class InsufficientPairsTests(unittest.TestCase):
    def test_fewer_than_required_pairs_is_invalid_input_never_a_downgraded_pass(self):
        harness = load_module()
        baseline = [make_trial("__primary__", i) for i in range(5)]
        candidate = [make_trial("__primary__", i) for i in range(5)]
        result = harness.screen_metric(
            baseline,
            candidate,
            "__primary__",
            "decode_tok_s",
            required_pairs=20,
            bootstrap_samples=100,
            seed=1,
            max_degradation_fraction=0.03,
            max_mdd_fraction=0.10,
        )
        self.assertEqual(result.status, "invalid_input")
        self.assertEqual(result.valid_pairs, 5)
        self.assertEqual(result.required_pairs, 20)


class AdverseCiFailTests(unittest.TestCase):
    def test_clear_degradation_fails_the_screen(self):
        harness = load_module()
        # candidate consistently 20% slower than baseline on decode_tok_s
        baseline = [make_trial("__primary__", i, decode_tok_s=100.0) for i in range(20)]
        candidate = [make_trial("__primary__", i, decode_tok_s=80.0) for i in range(20)]
        result = harness.screen_metric(
            baseline,
            candidate,
            "__primary__",
            "decode_tok_s",
            required_pairs=20,
            bootstrap_samples=500,
            seed=1,
            max_degradation_fraction=0.03,
            max_mdd_fraction=0.10,
        )
        self.assertEqual(result.status, "fail")
        self.assertGreater(result.ci_low, 0.03)

    def test_no_degradation_passes(self):
        harness = load_module()
        baseline = [make_trial("__primary__", i, decode_tok_s=100.0) for i in range(20)]
        candidate = [make_trial("__primary__", i, decode_tok_s=100.0) for i in range(20)]
        result = harness.screen_metric(
            baseline,
            candidate,
            "__primary__",
            "decode_tok_s",
            required_pairs=20,
            bootstrap_samples=500,
            seed=1,
            max_degradation_fraction=0.03,
            max_mdd_fraction=0.10,
        )
        self.assertEqual(result.status, "pass")


class UnderpoweredBlockTests(unittest.TestCase):
    def test_high_variance_small_effect_marks_underpowered_not_pass(self):
        harness = load_module()
        import random

        # Independent noise draws on EACH side (not `candidate = baseline *
        # constant`) so the paired delta actually varies pair-to-pair --
        # a linear-in-baseline candidate would make every delta identical
        # and the bootstrap SE exactly zero, which can never be
        # underpowered regardless of threshold.
        baseline_rng = random.Random(1)
        candidate_rng = random.Random(2)
        baseline = [
            make_trial("__primary__", i, decode_tok_s=100.0 + baseline_rng.uniform(-40, 40)) for i in range(20)
        ]
        candidate = [
            make_trial("__primary__", i, decode_tok_s=99.5 + candidate_rng.uniform(-40, 40)) for i in range(20)
        ]
        result = harness.screen_metric(
            baseline,
            candidate,
            "__primary__",
            "decode_tok_s",
            required_pairs=20,
            bootstrap_samples=500,
            seed=2,
            max_degradation_fraction=0.03,
            max_mdd_fraction=0.001,
        )
        self.assertEqual(result.status, "underpowered")
        self.assertGreater(result.minimal_detectable_degradation, 0.001)


class BootstrapSeedDeterminismTests(unittest.TestCase):
    def test_bootstrap_status_and_values_are_stable_across_hash_seeds(self):
        code = r"""
import importlib.util
import json
import sys
from pathlib import Path

script = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("compare_event_benchmark_matrix", script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

def trial(index, value):
    return {
        "scenario": "__primary__",
        "pair_index": index,
        "status": "succeeded",
        "decode_tok_s": value,
    }

baseline = [trial(i, 100.0 + i * 0.7) for i in range(20)]
candidate = [trial(i, 99.5 + i * 0.7) for i in range(20)]
result = module.screen_metric(
    baseline,
    candidate,
    "__primary__",
    "decode_tok_s",
    required_pairs=20,
    bootstrap_samples=500,
    seed=42,
    max_degradation_fraction=0.03,
    max_mdd_fraction=0.10,
)
print(json.dumps({
    "status": result.status,
    "mean": result.mean_relative_degradation,
    "ci_low": result.ci_low,
    "ci_high": result.ci_high,
}, sort_keys=True))
"""
        outputs = []
        for hash_seed in ("1", "2"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            result = subprocess.run(
                [sys.executable, "-c", code, str(SCRIPT)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])


class HolmMultiplicityTests(unittest.TestCase):
    def test_holm_never_relaxes_a_fail_status(self):
        harness = load_module()
        results = [
            harness.MetricScreenResult(
                group="__primary__",
                metric="decode_tok_s",
                valid_pairs=20,
                required_pairs=20,
                status="fail",
                mean_relative_degradation=0.10,
                ci_low=0.08,
                ci_high=0.12,
                minimal_detectable_degradation=0.02,
                raw_p_value=0.001,
            ),
            harness.MetricScreenResult(
                group="__primary__",
                metric="ttft_ms",
                valid_pairs=20,
                required_pairs=20,
                status="pass",
                mean_relative_degradation=0.0,
                ci_low=-0.01,
                ci_high=0.01,
                minimal_detectable_degradation=0.02,
                raw_p_value=0.9,
            ),
        ]
        report = harness.apply_holm_wording(results)
        fail_entry = next(entry for entry in report if entry["metric"] == "decode_tok_s")
        self.assertEqual(fail_entry["status"], "fail")
        self.assertIsNotNone(fail_entry["holm_adjusted_p_value"])

    def test_holm_adjustment_is_monotone_non_decreasing_by_rank(self):
        harness = load_module()
        adjusted = harness.holm_adjusted_p_values([0.001, 0.02, 0.5])
        self.assertEqual(len(adjusted), 3)
        ordered = sorted(range(3), key=lambda i: [0.001, 0.02, 0.5][i])
        values_in_rank_order = [adjusted[i] for i in ordered]
        self.assertEqual(values_in_rank_order, sorted(values_in_rank_order))


class WordingTests(unittest.TestCase):
    def test_pass_wording_uses_the_correct_phrase(self):
        harness = load_module()
        self.assertEqual(harness._wording_for_status("pass"), harness.CORRECT_WORDING)

    def test_forbidden_phrases_never_appear_for_any_status(self):
        harness = load_module()
        for status in ("pass", "fail", "underpowered", "invalid_input"):
            wording = harness._wording_for_status(status)
            for forbidden in harness.FORBIDDEN_WORDING_SUBSTRINGS:
                self.assertNotIn(forbidden, wording)


class RetryExhaustionTests(unittest.TestCase):
    def test_first_adverse_result_permits_one_retry(self):
        harness = load_module()
        decision = harness.evaluate_retry_state("fail", attempt=1)
        self.assertEqual(decision.action, "retry_permitted")

    def test_second_adverse_result_blocks_release(self):
        harness = load_module()
        decision = harness.evaluate_retry_state("fail", attempt=2)
        self.assertEqual(decision.action, "blocked_retry_exhausted")

    def test_pass_is_accepted_regardless_of_attempt(self):
        harness = load_module()
        self.assertEqual(harness.evaluate_retry_state("pass", attempt=1).action, "accept")
        self.assertEqual(harness.evaluate_retry_state("pass", attempt=2).action, "accept")


class P99GateTests(unittest.TestCase):
    def test_unmeasurable_p99_on_certification_host_blocks(self):
        harness = load_module()
        production = make_manifest(callback_ingress_p99_us=None)
        event_disabled = make_manifest(mode="event-disabled", callback_ingress_p99_us=50.0)
        results, blocking = harness.evaluate_p99_gate(production, event_disabled, 100.0)
        self.assertTrue(blocking)
        self.assertEqual(results["production"]["status"], "blocked")

    def test_p99_within_budget_is_ok(self):
        harness = load_module()
        production = make_manifest(callback_ingress_p99_us=42.0)
        event_disabled = make_manifest(mode="event-disabled", callback_ingress_p99_us=42.0)
        results, blocking = harness.evaluate_p99_gate(production, event_disabled, 100.0)
        self.assertFalse(blocking)
        self.assertEqual(results["production"]["status"], "ok")

    def test_p99_over_budget_blocks(self):
        harness = load_module()
        production = make_manifest(callback_ingress_p99_us=150.0)
        event_disabled = make_manifest(mode="event-disabled", callback_ingress_p99_us=50.0)
        results, blocking = harness.evaluate_p99_gate(production, event_disabled, 100.0)
        self.assertTrue(blocking)
        self.assertEqual(results["production"]["status"], "blocked")

    def test_non_certification_host_is_informational_even_when_unmeasurable(self):
        harness = load_module()
        windows_host = {"system": "Windows", "machine": "AMD64", "certification_host": None, "p99_gate": "informational"}
        production = make_manifest(callback_ingress_p99_us=None, host=windows_host)
        event_disabled = make_manifest(mode="event-disabled", callback_ingress_p99_us=None, host=windows_host)
        results, blocking = harness.evaluate_p99_gate(production, event_disabled, 100.0)
        self.assertFalse(blocking)
        self.assertEqual(results["production"]["status"], "informational")


class ExactHealthCountTests(unittest.TestCase):
    def test_state_rejection_and_degradation_block_certification(self):
        harness = load_module()
        for field, value in (
            ("state_transition_rejected", 1),
            ("state_degraded", True),
            ("rebuild_required", True),
        ):
            with self.subTest(field=field):
                violations = harness.evaluate_health_expectations({"health": {field: value}})
                self.assertTrue(any(field in violation for violation in violations))

    def test_terminal_delivery_failed_nonzero_is_a_violation(self):
        harness = load_module()
        manifest = make_manifest(health={"terminal_delivery_failed": 1, "dropped_progress": 0, "dropped_diagnostic": 0})
        violations = harness.evaluate_health_expectations(manifest)
        self.assertTrue(any("terminal_delivery_failed" in v for v in violations))

    def test_cancelled_reservation_rejections_are_accepted_as_nonnegative_counts(self):
        harness = load_module()
        manifest = make_manifest(
            health={
                "terminal_delivery_failed": 0,
                "cancelled_reservation_rejected": 3,
                "dropped_progress": 0,
                "dropped_diagnostic": 0,
            }
        )
        self.assertEqual(harness.evaluate_health_expectations(manifest), [])

    def test_negative_cancelled_reservation_count_is_invalid(self):
        harness = load_module()
        manifest = make_manifest(
            health={
                "terminal_delivery_failed": 0,
                "cancelled_reservation_rejected": -1,
                "dropped_progress": 0,
                "dropped_diagnostic": 0,
            }
        )
        violations = harness.evaluate_health_expectations(manifest)
        self.assertTrue(any("cancelled_reservation_rejected" in v for v in violations))

    def test_dropped_progress_must_exactly_match_expected_count(self):
        harness = load_module()
        manifest = make_manifest(
            expected_dropped_progress=20,
            health={"terminal_delivery_failed": 0, "dropped_progress": 19, "dropped_diagnostic": 20},
        )
        violations = harness.evaluate_health_expectations(manifest)
        self.assertTrue(any("dropped_progress" in v for v in violations))

    def test_exact_match_has_no_violations(self):
        harness = load_module()
        manifest = make_manifest(
            expected_dropped_progress=20,
            expected_dropped_diagnostic=20,
            health={"terminal_delivery_failed": 0, "dropped_progress": 20, "dropped_diagnostic": 20},
        )
        self.assertEqual(harness.evaluate_health_expectations(manifest), [])

    def test_state_lane_evictions_nonzero_is_a_violation(self):
        harness = load_module()
        manifest = make_manifest(
            health={"terminal_delivery_failed": 0, "state_lane_evictions": 3, "dropped_progress": 0, "dropped_diagnostic": 0}
        )
        violations = harness.evaluate_health_expectations(manifest)
        self.assertTrue(any("state-transition" in v for v in violations))


class HealthAvailabilityTests(unittest.TestCase):
    def test_health_is_available_true_when_health_dict_present(self):
        harness = load_module()
        self.assertTrue(harness.health_is_available(make_manifest()))

    def test_health_is_available_false_when_health_is_null(self):
        harness = load_module()
        self.assertFalse(harness.health_is_available(make_manifest(health=None)))

    def test_evaluate_health_expectations_returns_no_fabricated_violations_when_unavailable(self):
        """Defect A: an absent health block must never be silently read as
        all-zero counts, which would vacuously pass the !=0 checks and
        falsely fire dropped_progress/dropped_diagnostic mismatches
        against a manifest's real expected counts."""
        harness = load_module()
        manifest = make_manifest(health=None, expected_dropped_progress=20, expected_dropped_diagnostic=20)
        self.assertEqual(harness.evaluate_health_expectations(manifest), [])


class EvaluateExecutedOrderConsistencyTests(unittest.TestCase):
    """Replaces `SideOrderConsistencyTests`: since Task 14,
    `evaluate_executed_order_consistency` reads the manifest-level
    `executed_order` field (populated from REAL interleaved execution),
    not a per-trial `side_order_first` label."""

    def test_agreeing_and_varied_order_has_no_violations(self):
        harness = load_module()
        production = make_manifest(executed_order=alternating_executed_order("__primary__", 4))
        event_disabled = make_manifest(executed_order=alternating_executed_order("__primary__", 4))
        violations = harness.evaluate_executed_order_consistency(production, event_disabled)
        self.assertEqual(violations, [])

    def test_missing_on_production_manifest_is_a_violation(self):
        harness = load_module()
        production = make_manifest(executed_order=None)
        event_disabled = make_manifest(executed_order=alternating_executed_order("__primary__", 4))
        violations = harness.evaluate_executed_order_consistency(production, event_disabled)
        self.assertTrue(any("production manifest is missing" in v for v in violations))

    def test_missing_on_event_disabled_manifest_is_a_violation(self):
        harness = load_module()
        production = make_manifest(executed_order=alternating_executed_order("__primary__", 4))
        event_disabled = make_manifest(executed_order=None)
        violations = harness.evaluate_executed_order_consistency(production, event_disabled)
        self.assertTrue(any("event_disabled manifest is missing" in v for v in violations))

    def test_missing_on_both_reports_both_reasons(self):
        harness = load_module()
        production = make_manifest(executed_order=None)
        event_disabled = make_manifest(executed_order=None)
        violations = harness.evaluate_executed_order_consistency(production, event_disabled)
        self.assertEqual(len(violations), 2)

    def test_disagreement_between_manifests_is_a_violation(self):
        harness = load_module()
        production = make_manifest(
            executed_order=[
                {"scenario": "__primary__", "pair_index": 0, "order": ["production", "event-disabled"]},
                {"scenario": "__primary__", "pair_index": 1, "order": ["event-disabled", "production"]},
            ]
        )
        event_disabled = make_manifest(
            executed_order=[
                {"scenario": "__primary__", "pair_index": 0, "order": ["event-disabled", "production"]},
                {"scenario": "__primary__", "pair_index": 1, "order": ["event-disabled", "production"]},
            ]
        )
        violations = harness.evaluate_executed_order_consistency(production, event_disabled)
        self.assertTrue(any("disagrees" in v for v in violations))

    def test_constant_order_across_all_pairs_is_a_violation(self):
        harness = load_module()
        constant_order = [
            {"scenario": "__primary__", "pair_index": i, "order": ["production", "event-disabled"]} for i in range(5)
        ]
        production = make_manifest(executed_order=constant_order)
        event_disabled = make_manifest(executed_order=constant_order)
        violations = harness.evaluate_executed_order_consistency(production, event_disabled)
        self.assertTrue(any("constant" in v for v in violations))


class ThermalRecordTests(unittest.TestCase):
    def test_thermal_state_is_recorded_per_side_in_the_final_report(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            production = write_manifest(directory, "production.json", make_manifest(mode="production"))
            event_disabled = write_manifest(
                directory, "event-disabled.json", make_manifest(mode="event-disabled")
            )
            baseline = write_manifest(directory, "baseline.json", make_manifest(mode="production", binary_path="/fixtures/baseline"))
            args = harness.build_arg_parser().parse_args(
                build_argv(
                    production,
                    event_disabled,
                    baseline,
                    directory / "report.json",
                )
            )
            report = harness.build_report(args)
            self.assertIn("production", report["thermal_state"])
            self.assertTrue(report["thermal_state"]["production"]["available"])


class BaselineComparisonTests(unittest.TestCase):
    def test_comparison_b_screens_current_binary_against_baseline_binary(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            trials_matching = [make_trial("__primary__", i, decode_tok_s=100.0) for i in range(20)]
            production = write_manifest(
                directory, "production.json", make_manifest(mode="production", trials=trials_matching)
            )
            event_disabled = write_manifest(
                directory,
                "event-disabled.json",
                make_manifest(mode="event-disabled", trials=[make_trial("__primary__", i, decode_tok_s=100.0) for i in range(20)]),
            )
            # decode_tok_s is higher-is-better; a LOWER baseline value means
            # the baseline binary was slower, so the current binary looks
            # BETTER than baseline here.
            slower_baseline_trials = [make_trial("__primary__", i, decode_tok_s=70.0) for i in range(20)]
            baseline = write_manifest(
                directory,
                "baseline.json",
                make_manifest(mode="production", trials=slower_baseline_trials, binary_path="/fixtures/baseline"),
            )
            args = harness.build_arg_parser().parse_args(
                build_argv(
                    production,
                    event_disabled,
                    baseline,
                    directory / "report.json",
                )
            )
            report = harness.build_report(args)
            # A worse baseline means the CURRENT binary looks BETTER than
            # baseline, so comparison B (current vs baseline) must pass.
            self.assertEqual(report["comparison_b"]["status"], "pass")
            self.assertIn(
                "current-binary production vs baseline-binary production",
                report["comparison_b"]["description"],
            )


class SeedConsistencyTests(unittest.TestCase):
    def test_mismatched_manifest_seed_is_an_input_violation(self):
        harness = load_module()
        manifests = {"a": make_manifest(seed=1), "b": make_manifest(seed=2)}
        violations = harness.check_seed_consistency(manifests, expected_seed=1)
        self.assertEqual(len(violations), 1)
        self.assertIn("b", violations[0])


class HealthUnavailableBlockingTests(unittest.TestCase):
    def test_missing_health_blocks_with_the_accurately_named_reason(self):
        """Defect A: a null health block must block certification (never a
        silent pass) but with an honestly-named reason -- never the
        misleading `health_expectation_violation`, which claims a real
        count mismatch that was never actually observed."""
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            trials = alternating_trials("__primary__", 20)
            production = write_manifest(
                directory, "production.json", make_manifest(mode="production", trials=trials, health=None)
            )
            event_disabled = write_manifest(
                directory,
                "event-disabled.json",
                make_manifest(mode="event-disabled", trials=alternating_trials("__primary__", 20), health=None),
            )
            baseline = write_manifest(
                directory,
                "baseline.json",
                make_manifest(mode="production", trials=trials, binary_path="/fixtures/baseline", health=None),
            )
            args = harness.build_arg_parser().parse_args(
                build_argv(
                    production,
                    event_disabled,
                    baseline,
                    directory / "report.json",
                )
            )
            report = harness.build_report(args)
            self.assertIn("health_unavailable", report["blocking_reasons"])
            self.assertNotIn("health_expectation_violation", report["blocking_reasons"])
            self.assertEqual(report["certification_status"], "blocked")


class ExecutedOrderBlockingTests(unittest.TestCase):
    def test_constant_executed_order_blocks_with_a_clearly_named_reason(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            constant_order = [
                {"scenario": "__primary__", "pair_index": i, "order": ["production", "event-disabled"]}
                for i in range(20)
            ]
            production = write_manifest(
                directory, "production.json", make_manifest(mode="production", executed_order=constant_order)
            )
            event_disabled = write_manifest(
                directory,
                "event-disabled.json",
                make_manifest(mode="event-disabled", executed_order=constant_order),
            )
            baseline = write_manifest(
                directory,
                "baseline.json",
                make_manifest(mode="production", binary_path="/fixtures/baseline"),
            )
            args = harness.build_arg_parser().parse_args(
                build_argv(
                    production,
                    event_disabled,
                    baseline,
                    directory / "report.json",
                )
            )
            report = harness.build_report(args)
            self.assertIn("executed_order_inconsistent", report["blocking_reasons"])

    def test_missing_executed_order_blocks_with_a_clearly_named_reason(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            production = write_manifest(
                directory, "production.json", make_manifest(mode="production", executed_order=None)
            )
            event_disabled = write_manifest(
                directory, "event-disabled.json", make_manifest(mode="event-disabled", executed_order=None)
            )
            baseline = write_manifest(
                directory, "baseline.json", make_manifest(mode="production", binary_path="/fixtures/baseline")
            )
            args = harness.build_arg_parser().parse_args(
                build_argv(
                    production,
                    event_disabled,
                    baseline,
                    directory / "report.json",
                )
            )
            report = harness.build_report(args)
            self.assertIn("executed_order_inconsistent", report["blocking_reasons"])


class EndToEndReportTests(unittest.TestCase):
    def test_full_report_pass_case_has_correct_wording_and_no_forbidden_phrases(self):
        harness = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            identical_trials = [
                make_trial(
                    "__primary__",
                    i,
                    decode_tok_s=100.0,
                    ttft_ms=50.0,
                    side_order_first="production" if i % 2 == 0 else "event-disabled",
                )
                for i in range(20)
            ]
            production = write_manifest(
                directory,
                "production.json",
                make_manifest(mode="production", trials=identical_trials, expected_dropped_progress=0, expected_dropped_diagnostic=0),
            )
            event_disabled = write_manifest(
                directory,
                "event-disabled.json",
                make_manifest(
                    mode="event-disabled",
                    trials=[
                        make_trial(
                            "__primary__",
                            i,
                            decode_tok_s=100.0,
                            ttft_ms=50.0,
                            side_order_first="production" if i % 2 == 0 else "event-disabled",
                        )
                        for i in range(20)
                    ],
                    expected_dropped_progress=20,
                    expected_dropped_diagnostic=20,
                    health={"terminal_delivery_failed": 0, "dropped_progress": 20, "dropped_diagnostic": 20},
                ),
            )
            baseline = write_manifest(
                directory,
                "baseline.json",
                make_manifest(
                    mode="production",
                    trials=[
                        make_trial(
                            "__primary__",
                            i,
                            decode_tok_s=100.0,
                            ttft_ms=50.0,
                            side_order_first="production" if i % 2 == 0 else "event-disabled",
                        )
                        for i in range(20)
                    ],
                    binary_path="/fixtures/baseline",
                ),
            )
            output_path = directory / "report.json"
            argv = build_argv(
                production,
                event_disabled,
                baseline,
                output_path,
                bootstrap_samples=300,
                report_holm=True,
            )
            exit_code = harness.main(argv)
            self.assertEqual(exit_code, 0)
            written = output_path.read_text()
            self.assertIn(harness.CORRECT_WORDING, written)
            for forbidden in harness.FORBIDDEN_WORDING_SUBSTRINGS:
                self.assertNotIn(forbidden, written)
            report = json.loads(written)
            self.assertEqual(report["certification_status"], "pass")


class CliHelpTests(unittest.TestCase):
    def test_help_lists_every_frozen_flag(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        help_text = parser.format_help()
        for flag in (
            "--production",
            "--event-disabled",
            "--baseline",
            "--output",
            "--bootstrap-samples",
            "--seed",
            "--max-degradation-percent",
            "--min-primary-pairs",
            "--min-scenario-pairs",
            "--max-mdd-percent",
            "--report-holm",
        ):
            self.assertIn(flag, help_text)

    def test_help_exits_zero_via_subprocess(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=30, check=False
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--report-holm", result.stdout)


if __name__ == "__main__":
    unittest.main()
