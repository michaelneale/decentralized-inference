from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run-event-benchmark-matrix.py"
COMPARATOR_SCRIPT = ROOT / "scripts" / "compare-event-benchmark-matrix.py"
RUST_OUTPUT_TYPES = (
    ROOT / "crates" / "mesh-llm-commands" / "src" / "gpus" / "tune" / "output_types.rs"
)


def load_module():
    module_name = "run_event_benchmark_matrix"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registering in sys.modules BEFORE exec_module is required for
    # `dataclasses` to resolve `cls.__module__` on Python 3.9 (its
    # `_is_type` helper looks the module up via `sys.modules`); omitting
    # this raises `AttributeError: 'NoneType' object has no attribute
    # '__dict__'` the moment the loaded module defines a dataclass.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_comparator_module():
    """D-6: loads `compare-event-benchmark-matrix.py` (same mechanism as
    `load_module()` above) so a manifest built by THIS file's real
    `build_manifest`/`summarize_health_expectations` can be fed straight
    into the comparator's real `evaluate_health_expectations` -- an
    end-to-end proof that a fixture built independently via each module's
    own hand-authored dict (never exercising the actual producer function)
    cannot give."""
    module_name = "compare_event_benchmark_matrix"
    spec = importlib.util.spec_from_file_location(module_name, COMPARATOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_rust_string_literal(raw: str) -> str:
    joined = re.sub(r"\\\s*\n\s*", " ", raw)
    return normalize_whitespace(joined)


class ModeAliasMappingTests(unittest.TestCase):
    def test_production_and_event_disabled_map_to_identical_wire_values(self):
        harness = load_module()
        self.assertEqual(harness.resolve_trial_env_value("production"), "production")
        self.assertEqual(harness.resolve_trial_env_value("event-disabled"), "event-disabled")

    def test_unknown_mode_is_a_hard_error(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.resolve_trial_env_value("bogus")

    def test_valid_modes_matches_the_alias_map_keys(self):
        harness = load_module()
        self.assertEqual(set(harness.VALID_MODES), {"production", "event-disabled"})


class SeedValidationTests(unittest.TestCase):
    def test_accepts_zero_and_u64_max(self):
        harness = load_module()
        harness.validate_seed(0)
        harness.validate_seed(harness.U64_MAX)

    def test_rejects_negative(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.validate_seed(-1)

    def test_rejects_above_u64_max(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.validate_seed(harness.U64_MAX + 1)


class EnvironmentRedactionTests(unittest.TestCase):
    def test_allowlisted_names_persist_normalized_raw_values(self):
        harness = load_module()
        snapshot = harness.capture_environment_snapshot(
            {
                "MESH_LLM_LIFECYCLE_LOG_PARSER": "auto",
                "MESH_LLM_BENCHMARK_TUNE_TRIAL": "1",
                "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE": "event-disabled",
            }
        )
        self.assertEqual(
            snapshot["MESH_LLM_LIFECYCLE_LOG_PARSER"], {"value": "auto", "redacted": False}
        )
        self.assertEqual(
            snapshot["MESH_LLM_BENCHMARK_TUNE_TRIAL"], {"value": True, "redacted": False}
        )
        self.assertEqual(
            snapshot["MESH_LLM_EVENT_SYSTEM_TRIAL_MODE"],
            {"value": "event-disabled", "redacted": False},
        )

    def test_non_allowlisted_mesh_llm_names_are_redacted_to_presence_only(self):
        harness = load_module()
        snapshot = harness.capture_environment_snapshot({"MESH_LLM_CONFIG": "/tmp/secret-path.toml"})
        self.assertEqual(
            snapshot["MESH_LLM_CONFIG"], {"value": harness.REDACTED_PRESENT, "redacted": True}
        )
        self.assertNotIn("/tmp/secret-path.toml", json.dumps(snapshot))

    def test_names_matching_sensitive_pattern_are_always_redacted(self):
        harness = load_module()
        # Simulates a hypothetical future allowlist collision: even a name
        # containing a sensitive substring must redact, defense in depth
        # beyond the hand-curated allowlist.
        snapshot = harness.capture_environment_snapshot({"MESH_LLM_API_TOKEN": "abc123"})
        self.assertTrue(snapshot["MESH_LLM_API_TOKEN"]["redacted"])
        self.assertNotIn("abc123", json.dumps(snapshot))

    def test_non_mesh_llm_names_are_ignored_entirely(self):
        harness = load_module()
        snapshot = harness.capture_environment_snapshot({"HOME": "/Users/example", "PATH": "/usr/bin"})
        self.assertEqual(snapshot, {})


class BinaryIdentityTests(unittest.TestCase):
    def test_missing_binary_reports_none_sha256_and_none_version(self):
        harness = load_module()
        identity = harness.capture_binary_identity(
            Path("/nonexistent/mesh-llm"), run_version=lambda _binary: None
        )
        self.assertIsNone(identity["sha256"])
        self.assertIsNone(identity["version"])
        self.assertTrue(identity["path"].endswith("mesh-llm"))

    def test_existing_file_hashes_deterministically(self):
        harness = load_module()
        with self._temp_file(b"fake-binary-bytes") as path:
            identity = harness.capture_binary_identity(path, run_version=lambda _b: "mesh-llm 0.76.0")
            self.assertIsNotNone(identity["sha256"])
            self.assertEqual(len(identity["sha256"]), 64)
            self.assertEqual(identity["version"], "mesh-llm 0.76.0")

    @staticmethod
    def _temp_file(data: bytes):
        import contextlib
        import tempfile

        @contextlib.contextmanager
        def _ctx():
            with tempfile.NamedTemporaryFile(delete=False) as handle:
                handle.write(data)
                handle.flush()
                path = Path(handle.name)
            try:
                yield path
            finally:
                path.unlink(missing_ok=True)

        return _ctx()


class TrialPlanDeterminismTests(unittest.TestCase):
    def test_same_seed_produces_identical_plans_across_invocations(self):
        harness = load_module()
        plan_a = harness.build_trial_plan(42, 3, 2, ["chat_short", "chat_long"])
        plan_b = harness.build_trial_plan(42, 3, 2, ["chat_short", "chat_long"])
        self.assertEqual(plan_a, plan_b)

    def test_different_seeds_produce_different_prompt_seeds(self):
        harness = load_module()
        plan_a = harness.build_trial_plan(1, 2, 1, ["s"])
        plan_b = harness.build_trial_plan(2, 2, 1, ["s"])
        self.assertNotEqual(
            [entry.prompt_seed for entry in plan_a], [entry.prompt_seed for entry in plan_b]
        )

    def test_plan_shape_covers_primary_then_each_scenario_in_order(self):
        harness = load_module()
        plan = harness.build_trial_plan(1, 2, 3, ["alpha", "beta"])
        scenarios = [entry.scenario for entry in plan]
        self.assertEqual(
            scenarios,
            [harness.PRIMARY_SCENARIO] * 2 + ["alpha"] * 3 + ["beta"] * 3,
        )
        primary_indices = [entry.pair_index for entry in plan if entry.scenario == harness.PRIMARY_SCENARIO]
        self.assertEqual(primary_indices, [0, 1])

    def test_zero_pairs_primary_is_rejected(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.build_trial_plan(1, 0, 1, ["s"])

    def test_zero_pairs_scenario_is_rejected(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.build_trial_plan(1, 1, 0, ["s"])

    def test_no_scenarios_is_rejected(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.build_trial_plan(1, 1, 1, [])

    def test_side_order_first_is_a_valid_mode_for_every_entry(self):
        harness = load_module()
        plan = harness.build_trial_plan(7, 3, 2, ["alpha"])
        for entry in plan:
            self.assertIn(entry.side_order_first, harness.VALID_MODES)

    def test_sides_kwarg_overrides_the_default_mode_domain(self):
        """`build_trial_plan(sides=...)` -- added for comparison B, where
        the two sides differ by BINARY (e.g. "current"/"baseline"), not by
        trial mode -- must mint `side_order_first` from the GIVEN domain,
        not the default `VALID_MODES`."""
        harness = load_module()
        plan = harness.build_trial_plan(7, 3, 2, ["alpha"], sides=("current", "baseline"))
        for entry in plan:
            self.assertIn(entry.side_order_first, ("current", "baseline"))
        observed = {
            harness.build_trial_plan(seed, 1, 1, ["s"], sides=("current", "baseline"))[0].side_order_first
            for seed in range(50)
        }
        self.assertEqual(observed, {"current", "baseline"})

    def test_side_order_first_uses_both_modes_across_many_seeds(self):
        """Side order is minted from the SAME deterministic per-plan rng
        that mints prompt_seed (see build_trial_plan), so re-running with
        the same seed/counts/scenarios always reproduces the same order --
        proven separately by
        test_same_seed_produces_identical_plans_across_invocations, since
        TrialPlanEntry equality now covers side_order_first too. This test
        instead proves the ordering genuinely VARIES rather than being a
        constant: across many different seeds, both modes must appear."""
        harness = load_module()
        observed = {
            harness.build_trial_plan(seed, 1, 1, ["s"])[0].side_order_first for seed in range(50)
        }
        self.assertEqual(observed, set(harness.VALID_MODES))


class ResolveComparisonSidesTests(unittest.TestCase):
    """`resolve_comparison_sides` is what turns the CLI's repeatable
    `--mode` plus optional `--baseline-binary` into the invocation's two
    sides -- see the runner module docstring for the two accepted shapes."""

    def test_comparison_a_requires_exactly_two_modes_no_baseline_binary(self):
        harness = load_module()
        side_a, side_b = harness.resolve_comparison_sides(
            Path("/bin/mesh-llm"), None, ["production", "event-disabled"]
        )
        self.assertEqual((side_a.mode, side_a.side_id), ("production", "production"))
        self.assertEqual((side_b.mode, side_b.side_id), ("event-disabled", "event-disabled"))
        self.assertEqual(side_a.binary, side_b.binary)

    def test_comparison_a_rejects_a_single_mode(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.resolve_comparison_sides(Path("/bin/mesh-llm"), None, ["production"])

    def test_comparison_a_rejects_a_repeated_mode(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.resolve_comparison_sides(Path("/bin/mesh-llm"), None, ["production", "production"])

    def test_comparison_b_requires_exactly_one_mode_with_baseline_binary(self):
        harness = load_module()
        side_a, side_b = harness.resolve_comparison_sides(
            Path("/bin/current"), Path("/bin/baseline"), ["production"]
        )
        self.assertEqual((side_a.binary, side_a.side_id), (Path("/bin/current"), "current"))
        self.assertEqual((side_b.binary, side_b.side_id), (Path("/bin/baseline"), "baseline"))
        self.assertEqual(side_a.mode, side_b.mode)

    def test_comparison_b_rejects_two_modes(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.resolve_comparison_sides(
                Path("/bin/current"), Path("/bin/baseline"), ["production", "event-disabled"]
            )


class EventSystemHealthLogParsingTests(unittest.TestCase):
    """Proves parse_final_event_system_health against fixture log text
    (never a real process) -- the manifest-populated-from-a-fixture-log
    acceptance criterion."""

    def test_state_pressure_health_survives_log_to_manifest_conversion(self):
        harness = load_module()
        fields = harness.parse_event_system_health_message(
            "state_transition_rejected=2 state_degraded=true rebuild_required=true "
            "terminal_delivery_failed=0 ingress_p99_us=5"
        )
        health, p99 = harness.derive_health_and_p99(fields)
        self.assertEqual(health["state_transition_rejected"], 2)
        self.assertIs(health["state_degraded"], True)
        self.assertIs(health["rebuild_required"], True)
        self.assertEqual(p99, 5)
        self.assertIs(harness.parse_event_system_health_message("state_degraded=false")["state_degraded"], False)

    HEALTH_LINE_NO_P99 = (
        '{"context":"event_system_health","event":"info","level":"info",'
        '"message":"version=0 reservation_exhausted=0 cancelled_reservation_rejected=0 terminal_delivery_failed=0 '
        "dropped_progress=0 dropped_diagnostic=0 replay_evicted=0 "
        "subscriber_disconnected=0 shutdown_degraded=0 reducer_rejected=0 "
        "rebuild_generation=0 bounds.reservation_table_capacity=3136 "
        "bounds.state_transition_lane_depth=4096 bounds.diagnostic_lane_depth=2048 "
        "bounds.wake_list_depth=3136 bounds.replay_max_frames=4096 "
        "bounds.subscriber_lag_max_frames=1024 bounds.max_concurrent_subscribers=32 "
        'ingress_p99_us=null","timestamp":"2026-09-04T17:14:57.966Z"}'
    )
    HEALTH_LINE_WITH_P99 = (
        '{"context":"event_system_health","event":"info","level":"info",'
        '"message":"version=1 reservation_exhausted=0 cancelled_reservation_rejected=0 terminal_delivery_failed=2 '
        "dropped_progress=7 dropped_diagnostic=3 replay_evicted=0 "
        "subscriber_disconnected=0 shutdown_degraded=0 reducer_rejected=0 "
        "rebuild_generation=0 bounds.reservation_table_capacity=3136 "
        "bounds.state_transition_lane_depth=4096 bounds.diagnostic_lane_depth=2048 "
        "bounds.wake_list_depth=3136 bounds.replay_max_frames=4096 "
        "bounds.subscriber_lag_max_frames=1024 bounds.max_concurrent_subscribers=32 "
        'ingress_p99_us=6","timestamp":"2026-09-04T17:15:25.088Z"}'
    )

    def test_no_health_line_at_all_returns_none(self):
        harness = load_module()
        log_text = '{"context":"other","event":"info","level":"info","message":"unrelated","timestamp":"x"}\n'
        self.assertIsNone(harness.parse_final_event_system_health(log_text))

    def test_single_health_line_is_parsed(self):
        harness = load_module()
        fields = harness.parse_final_event_system_health(self.HEALTH_LINE_NO_P99 + "\n")
        self.assertEqual(fields["version"], 0)
        self.assertEqual(fields["cancelled_reservation_rejected"], 0)
        self.assertIsNone(fields["ingress_p99_us"])
        self.assertEqual(fields["dropped_progress"], 0)

    def test_repeated_health_lines_take_the_final_one(self):
        """The docstring's core promise: with MULTIPLE health lines in one
        log (the version bumps as counters change), the FINAL line wins --
        never the first."""
        harness = load_module()
        log_text = "\n".join(
            [
                '{"context":"other","event":"info","level":"info","message":"noise","timestamp":"x"}',
                self.HEALTH_LINE_NO_P99,
                '{"context":"other","event":"info","level":"info","message":"noise2","timestamp":"y"}',
                self.HEALTH_LINE_WITH_P99,
            ]
        )
        fields = harness.parse_final_event_system_health(log_text)
        self.assertEqual(fields["version"], 1)
        self.assertEqual(fields["ingress_p99_us"], 6)

    def test_malformed_json_line_is_skipped_not_fatal(self):
        harness = load_module()
        log_text = "\n".join(["not json at all {{{", self.HEALTH_LINE_WITH_P99])
        fields = harness.parse_final_event_system_health(log_text)
        self.assertEqual(fields["version"], 1)

    def test_derive_health_and_p99_absent_line_is_honest_none_none(self):
        """The absence-stays-null contract this task's Must-NOT forbids
        relaxing: no health line anywhere in the log means BOTH `health`
        and `callback_ingress_p99_us` must come back `None` -- never a
        fabricated zero-filled health dict and never a fabricated p99."""
        harness = load_module()
        health, p99 = harness.derive_health_and_p99(None)
        self.assertIsNone(health)
        self.assertIsNone(p99)

    def test_derive_health_and_p99_null_p99_token_stays_none_but_health_populates(self):
        """A health line CAN exist (real counters collected) while
        `ingress_p99_us` is still the literal `null` token (fewer than 100
        submissions in that trial's process lifetime) -- `health` must
        populate while `callback_ingress_p99_us` independently stays
        `None`; these two null checks are NOT the same claim."""
        harness = load_module()
        fields = harness.parse_final_event_system_health(self.HEALTH_LINE_NO_P99)
        health, p99 = harness.derive_health_and_p99(fields)
        self.assertIsNotNone(health)
        self.assertEqual(health["terminal_delivery_failed"], 0)
        self.assertEqual(health["cancelled_reservation_rejected"], 0)
        self.assertIsNone(p99)

    def test_derive_health_and_p99_populates_both_when_present(self):
        harness = load_module()
        fields = harness.parse_final_event_system_health(self.HEALTH_LINE_WITH_P99)
        health, p99 = harness.derive_health_and_p99(fields)
        self.assertEqual(health["terminal_delivery_failed"], 2)
        self.assertEqual(health["cancelled_reservation_rejected"], 0)
        self.assertEqual(health["dropped_progress"], 7)
        self.assertEqual(health["dropped_diagnostic"], 3)
        self.assertEqual(p99, 6.0)


class DecodeOnlyTokSTests(unittest.TestCase):
    def test_null_when_ttft_is_null(self):
        harness = load_module()
        self.assertIsNone(harness.compute_decode_only_tok_s(100, 5000.0, None))

    def test_null_when_decode_interval_is_zero(self):
        harness = load_module()
        self.assertIsNone(harness.compute_decode_only_tok_s(100, 500.0, 500.0))

    def test_null_when_decode_interval_is_negative(self):
        harness = load_module()
        self.assertIsNone(harness.compute_decode_only_tok_s(100, 500.0, 600.0))

    def test_computed_value_uses_the_epsilon_guarded_interval(self):
        harness = load_module()
        # completion_tokens=100 over a 4.5s decode interval (5.0s total - 0.5s ttft)
        value = harness.compute_decode_only_tok_s(100, 5000.0, 500.0)
        self.assertAlmostEqual(value, 100 / 4.5, places=6)

    def test_never_returns_zero_on_failure_paths(self):
        harness = load_module()
        for args in [(None, 1.0, 1.0), (1, None, 1.0), (1, 1.0, None), (1, 0.0, 5.0)]:
            self.assertIsNone(harness.compute_decode_only_tok_s(*args))


class DecodeTokSTests(unittest.TestCase):
    def test_historical_definition_preserved(self):
        harness = load_module()
        self.assertAlmostEqual(harness.compute_decode_tok_s(100, 2000.0), 50.0)

    def test_null_on_non_positive_elapsed(self):
        harness = load_module()
        self.assertIsNone(harness.compute_decode_tok_s(100, 0.0))
        self.assertIsNone(harness.compute_decode_tok_s(None, 2000.0))


class ModelIdResolutionTests(unittest.TestCase):
    """`--local-model-only` rejects `model: "auto"` with `404 model_not_found`
    (no mesh/routing layer to resolve the alias) -- confirmed against a real
    running binary. `build_chat_request_body` must therefore send the model
    id resolved from `/v1/models`, never a hardcoded `"auto"`."""

    def test_build_chat_request_body_uses_the_passed_model_not_auto(self):
        harness = load_module()
        body = harness.build_chat_request_body("prompt", 16, "local-gguf/sha256-abc")
        self.assertEqual(body["model"], "local-gguf/sha256-abc")
        self.assertNotEqual(body["model"], "auto")

    def test_first_models_list_id_extracts_the_first_entry(self):
        harness = load_module()
        payload = {"object": "list", "data": [{"id": "local-gguf/sha256-abc", "object": "model"}]}
        self.assertEqual(harness.first_models_list_id(payload), "local-gguf/sha256-abc")

    def test_first_models_list_id_is_none_on_empty_data(self):
        harness = load_module()
        self.assertIsNone(harness.first_models_list_id({"object": "list", "data": []}))

    def test_first_models_list_id_is_none_on_malformed_payload(self):
        harness = load_module()
        self.assertIsNone(harness.first_models_list_id({}))
        self.assertIsNone(harness.first_models_list_id({"data": "not-a-list"}))
        self.assertIsNone(harness.first_models_list_id({"data": [{"id": ""}]}))


class SseStreamParsingTests(unittest.TestCase):
    def test_happy_path_extracts_ttft_and_completion_tokens(self):
        # `parse_sse_stream` calls `clock()` exactly once, at the moment
        # the first non-empty content delta is seen -- the fake returns
        # `started_at + 0.25s` on that single call.
        harness = load_module()
        lines = [
            'data: {"choices":[{"delta":{"content":""}}]}\n',
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
            'data: {"choices":[{"delta":{}}],"usage":{"completion_tokens":7}}\n',
            "data: [DONE]\n",
        ]
        result = harness.parse_sse_stream(lines, clock=lambda: 0.25, started_at=0.0)
        self.assertEqual(result.completion_tokens, 7)
        self.assertAlmostEqual(result.ttft_ms, 250.0, places=3)
        self.assertFalse(result.malformed)

    def test_split_chunks_and_empty_deltas_do_not_set_ttft(self):
        harness = load_module()
        lines = [
            'data: {"choices":[{"delta":{"content":""}}]}\n',
            'data: {"choices":[{"delta":{}}]}\n',
            'data: {"choices":[{"delta":{"content":"ok"}}],"usage":{"completion_tokens":3}}\n',
            "data: [DONE]\n",
        ]
        result = harness.parse_sse_stream(lines, clock=lambda: 0.1, started_at=0.0)
        self.assertAlmostEqual(result.ttft_ms, 100.0, places=3)
        self.assertEqual(result.completion_tokens, 3)

    def test_malformed_json_line_is_skipped_not_fatal(self):
        harness = load_module()
        lines = [
            "data: {not valid json\n",
            'data: {"choices":[{"delta":{"content":"x"}}],"usage":{"completion_tokens":1}}\n',
            "data: [DONE]\n",
        ]
        result = harness.parse_sse_stream(lines, clock=lambda: 0.05, started_at=0.0)
        self.assertEqual(result.completion_tokens, 1)
        self.assertFalse(result.malformed)

    def test_stream_without_terminal_usage_is_malformed_with_null_tokens(self):
        harness = load_module()
        lines = ['data: {"choices":[{"delta":{"content":"x"}}]}\n', "data: [DONE]\n"]
        result = harness.parse_sse_stream(lines, clock=lambda: 0.0, started_at=0.0)
        self.assertIsNone(result.completion_tokens)
        self.assertTrue(result.malformed)

    def test_non_data_lines_are_ignored(self):
        harness = load_module()
        lines = [
            ": keepalive\n",
            "\n",
            'data: {"choices":[{"delta":{"content":"x"}}],"usage":{"completion_tokens":2}}\n',
            "data: [DONE]\n",
        ]
        result = harness.parse_sse_stream(lines, clock=lambda: 0.0, started_at=0.0)
        self.assertEqual(result.completion_tokens, 2)


class HealthExpectationTests(unittest.TestCase):
    def test_event_disabled_expects_the_fixed_per_trial_counts_not_the_run_total(self):
        """D-6 fix (`.omo/evidence/event-system-fixes/deferrals/d6/`):
        `event-disabled` mode's expectation is scoped to the SAME single
        trial `health` actually reflects (Task 14's
        `health_and_p99_from_trial_log` reads only the side's LAST trial's
        log) -- it must NOT scale with `len(results)`, the OLD per-RUN
        total that fired `health_expectation_violation` on every real
        event-disabled manifest regardless of pair count (F4 certification
        wave, `.omo/evidence/event-system-fixes/final/f4/f4-verdict.md`,
        "New finding" section). 5 and 30 results (30 matches a real
        20+10-pair matrix's per-side trial count) must produce the
        IDENTICAL fixed pair -- proof the expectation no longer grows with
        run length."""
        harness = load_module()
        five_results = [_fake_trial_result() for _ in range(5)]
        thirty_results = [_fake_trial_result() for _ in range(30)]
        self.assertEqual(
            harness.summarize_health_expectations("event-disabled", five_results),
            {"expected_dropped_progress": 1, "expected_dropped_diagnostic": 0},
        )
        self.assertEqual(
            harness.summarize_health_expectations("event-disabled", thirty_results),
            {"expected_dropped_progress": 1, "expected_dropped_diagnostic": 0},
        )

    def test_event_disabled_with_zero_trials_expects_zero_drops(self):
        """No trial ran on this side at all, so there is no `health`
        snapshot to expect anything from -- expected stays honestly zero
        rather than the fixed per-trial pair."""
        harness = load_module()
        expectations = harness.summarize_health_expectations("event-disabled", [])
        self.assertEqual(expectations, {"expected_dropped_progress": 0, "expected_dropped_diagnostic": 0})

    def test_production_expects_zero_drops(self):
        harness = load_module()
        results = [_fake_trial_result() for _ in range(5)]
        expectations = harness.summarize_health_expectations("production", results)
        self.assertEqual(expectations, {"expected_dropped_progress": 0, "expected_dropped_diagnostic": 0})


def _fake_trial_result():
    """`summarize_health_expectations` only counts `results` -- content is
    irrelevant, so a plain placeholder is enough."""
    return object()


class ManifestBuildingTests(unittest.TestCase):
    def test_manifest_carries_schema_and_trial_unit_and_expectations(self):
        harness = load_module()
        result = harness.TrialResult(
            scenario=harness.PRIMARY_SCENARIO,
            pair_index=0,
            side_order_first="production",
            status="succeeded",
            completion_tokens=10,
            elapsed_ms=1000.0,
            decode_tok_s=10.0,
            ttft_ms=100.0,
            decode_only_tok_s=11.1,
            setup_ms=50.0,
            readiness_ms=200.0,
            shutdown_ms=25.0,
        )
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="event-disabled",
            seed=7,
            pairs_primary=1,
            pairs_scenario=1,
            scenarios=["chat_short"],
            results=[result],
            environ={"MESH_LLM_BENCHMARK_TUNE_TRIAL": "1", "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE": "event-disabled"},
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
        )
        self.assertEqual(manifest["schema_version"], harness.MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest["metrics_schema"], "streaming_v1")
        self.assertEqual(manifest["mode"], "event-disabled")
        self.assertEqual(manifest["trial_unit"], harness.TRIAL_UNIT_DEFINITION)
        # D-6 fix: fixed per-trial pair (1, 0), not the OLD len(results)
        # total -- see the full-scale test below, where the two formulas
        # actually diverge (this one result count doesn't distinguish them).
        self.assertEqual(manifest["expected_dropped_progress"], 1)
        self.assertEqual(manifest["expected_dropped_diagnostic"], 0)
        self.assertEqual(len(manifest["trials"]), 1)
        self.assertEqual(manifest["trials"][0]["status"], "succeeded")

    def test_manifest_expectations_stay_fixed_at_full_run_scale_not_scaled_by_trial_count(self):
        """D-6: at a real 20+10-pair matrix's per-side scale (30 trials),
        expectations must stay (1, 0), never scale to 30 like the OLD
        per-run-total formula did (F4's health_expectation_violation)."""
        harness = load_module()
        results = [
            harness.TrialResult(
                scenario=harness.PRIMARY_SCENARIO,
                pair_index=i,
                side_order_first="production" if i % 2 == 0 else "event-disabled",
                status="succeeded",
                completion_tokens=10,
                elapsed_ms=1000.0,
                decode_tok_s=10.0,
                ttft_ms=100.0,
                decode_only_tok_s=11.1,
                setup_ms=50.0,
                readiness_ms=200.0,
                shutdown_ms=25.0,
            )
            for i in range(30)
        ]
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="event-disabled",
            seed=7,
            pairs_primary=20,
            pairs_scenario=10,
            scenarios=["smoke"],
            results=results,
            environ={"MESH_LLM_BENCHMARK_TUNE_TRIAL": "1", "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE": "event-disabled"},
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
        )
        self.assertEqual(manifest["expected_dropped_progress"], 1)
        self.assertEqual(manifest["expected_dropped_diagnostic"], 0)
        self.assertEqual(len(manifest["trials"]), 30)

    def test_manifest_records_a_non_default_attempt_number(self):
        harness = load_module()
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="production",
            seed=1,
            pairs_primary=1,
            pairs_scenario=1,
            scenarios=["s"],
            results=[],
            environ={},
            attempt=2,
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
        )
        self.assertEqual(manifest["attempt"], 2)

    def test_manifest_health_is_null_when_not_collected(self):
        """Defect A: no real call site collects health data (see
        summarize_health_expectations's own docstring on what CAN be
        proven without a live console API); build_manifest must report
        that honestly as JSON null, never a silently-zero {}."""
        harness = load_module()
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="production",
            seed=1,
            pairs_primary=1,
            pairs_scenario=1,
            scenarios=["s"],
            results=[],
            environ={},
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
        )
        self.assertIsNone(manifest["health"])

    def test_manifest_preserves_an_explicitly_supplied_health_dict(self):
        """A future caller that CAN collect real health data must have its
        value pass through unchanged -- build_manifest never overwrites a
        caller-supplied health block."""
        harness = load_module()
        supplied = {"terminal_delivery_failed": 0, "dropped_progress": 3, "dropped_diagnostic": 3}
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="event-disabled",
            seed=1,
            pairs_primary=1,
            pairs_scenario=1,
            scenarios=["s"],
            results=[],
            environ={},
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
            health=supplied,
        )
        self.assertEqual(manifest["health"], supplied)

    def test_manifest_executed_order_is_null_when_not_supplied(self):
        """A caller that never ran an interleaved plan (e.g. this test,
        building a manifest in isolation) must see `executed_order` as
        JSON null -- never a fabricated empty list standing in for "this
        pair's real order was never recorded"."""
        harness = load_module()
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="production",
            seed=1,
            pairs_primary=1,
            pairs_scenario=1,
            scenarios=["s"],
            results=[],
            environ={},
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
        )
        self.assertIsNone(manifest["executed_order"])

    def test_manifest_executed_order_is_populated_and_json_serializable(self):
        harness = load_module()
        entries = [
            harness.ExecutedOrderEntry(scenario=harness.PRIMARY_SCENARIO, pair_index=0, order=("production", "event-disabled")),
            harness.ExecutedOrderEntry(scenario=harness.PRIMARY_SCENARIO, pair_index=1, order=("event-disabled", "production")),
        ]
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="production",
            seed=1,
            pairs_primary=2,
            pairs_scenario=1,
            scenarios=["s"],
            results=[],
            environ={},
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
            executed_order=entries,
        )
        self.assertEqual(len(manifest["executed_order"]), 2)
        self.assertEqual(manifest["executed_order"][0]["order"], ["production", "event-disabled"])
        # Must round-trip through json.dumps without a custom encoder --
        # this is exactly what main() does when writing the manifest file.
        json.dumps(manifest)


def _event_disabled_manifest_at_full_scale(harness, *, dropped_progress, dropped_diagnostic):
    """Builds a manifest through the REAL `build_manifest` (30 trials,
    matching a 20+10-pair matrix's per-side count) with the given `health`
    counts -- the same shape D-6's fix targets. Used by
    `EndToEndHealthExpectationReconciliationTests` so the comparator sees
    exactly what a real invocation would write, not a hand-authored dict."""
    results = [
        harness.TrialResult(
            scenario=harness.PRIMARY_SCENARIO,
            pair_index=i,
            side_order_first="production" if i % 2 == 0 else "event-disabled",
            status="succeeded",
            completion_tokens=10,
            elapsed_ms=1000.0,
            decode_tok_s=10.0,
            ttft_ms=100.0,
            decode_only_tok_s=11.1,
            setup_ms=50.0,
            readiness_ms=200.0,
            shutdown_ms=25.0,
        )
        for i in range(30)
    ]
    return harness.build_manifest(
        binary=Path("/nonexistent/mesh-llm"),
        model="fixture-model",
        mode="event-disabled",
        seed=7,
        pairs_primary=20,
        pairs_scenario=10,
        scenarios=["smoke"],
        results=results,
        environ={"MESH_LLM_BENCHMARK_TUNE_TRIAL": "1", "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE": "event-disabled"},
        generated_at="2026-01-01T00:00:00Z",
        run_version=lambda _b: None,
        health={
            "terminal_delivery_failed": 0,
            "dropped_progress": dropped_progress,
            "dropped_diagnostic": dropped_diagnostic,
        },
    )


class EndToEndHealthExpectationReconciliationTests(unittest.TestCase):
    """D-6 (`.omo/evidence/event-system-fixes/deferrals/d6/`): feeds a
    manifest built by THIS module's real `build_manifest` straight into
    the comparator's real `evaluate_health_expectations`, so these tests
    exercise the actual producer/consumer pair the F4 certification wave
    ran, not two independently hand-authored fixtures that could agree by
    construction regardless of whether the producer is fixed."""

    def test_real_f4_captured_event_disabled_manifest_has_no_expectation_violations(self):
        """Health counts taken verbatim from every one of 3 independent
        full 30-pair event-disabled manifests in
        `.omo/evidence/event-system-fixes/final/f4/f4-manifests.txt`
        (dropped_progress=1, dropped_diagnostic=0). Before the D-6 fix,
        `build_manifest` computed expected=30/30 here (len(results)), so
        this genuinely-captured, well-formed manifest showed violations
        against real data -- the exact defect F4 found."""
        harness = load_module()
        comparator = load_comparator_module()
        manifest = _event_disabled_manifest_at_full_scale(harness, dropped_progress=1, dropped_diagnostic=0)
        self.assertEqual(comparator.evaluate_health_expectations(manifest), [])

    def test_reconciled_expectation_still_catches_genuine_progress_divergence(self):
        """The reconciliation must not become a vacuous always-pass: a
        progress count that genuinely diverges from the reconciled
        per-trial expectation (1) must still be flagged, and the violation
        message must cite THAT reconciled count -- not the old,
        run-scaled 30."""
        harness = load_module()
        comparator = load_comparator_module()
        manifest = _event_disabled_manifest_at_full_scale(harness, dropped_progress=3, dropped_diagnostic=0)
        violations = comparator.evaluate_health_expectations(manifest)
        self.assertTrue(any("dropped_progress" in v and "expected count 1" in v for v in violations), violations)

    def test_reconciled_expectation_still_catches_genuine_diagnostic_divergence(self):
        """Same proof for dropped_diagnostic's independent reconciled
        value (0): a genuine divergence must still be flagged, citing 0 --
        not the old, run-scaled 30."""
        harness = load_module()
        comparator = load_comparator_module()
        manifest = _event_disabled_manifest_at_full_scale(harness, dropped_progress=1, dropped_diagnostic=2)
        violations = comparator.evaluate_health_expectations(manifest)
        self.assertTrue(any("dropped_diagnostic" in v and "expected count 0" in v for v in violations), violations)


class MainThreadsAttemptIntoManifestTests(unittest.TestCase):
    """`main()` must forward the parsed `--attempt` value into
    `build_manifest` (via `_build_side_manifest`, its one-side-at-a-time
    helper) -- a source-level check (rather than invoking `main()`, which
    spawns REAL trial subprocesses by default) mirroring this file's
    existing source-inspection convention (see `HiddenSelectorWiringTests`)."""

    def test_build_side_manifest_passes_attempt_from_args_to_build_manifest(self):
        source = SCRIPT.read_text()
        helper_start = source.index("def _build_side_manifest(")
        helper_end = source.index("\ndef ", helper_start + 1)
        helper_body = source[helper_start:helper_end]
        self.assertIn("attempt=args.attempt", helper_body)

    def test_main_calls_build_side_manifest_for_both_sides(self):
        source = SCRIPT.read_text()
        main_start = source.index("def main(")
        main_body = source[main_start:]
        self.assertEqual(main_body.count("_build_side_manifest("), 1)
        self.assertIn("for side, results, health, ingress_p99_us in (", main_body)


class TrialUnitDefinitionMatchesRustSourceTests(unittest.TestCase):
    def test_trial_and_pair_wording_matches_rust_verbatim(self):
        harness = load_module()
        rust_source = RUST_OUTPUT_TYPES.read_text()
        match = re.search(
            r"benchmark_trial_unit_definition\(\).*?trial:\s*\"(?P<trial>.*?)\"\s*\.to_string\(\)"
            r".*?pair:\s*\"(?P<pair>.*?)\"\s*\.to_string\(\)",
            rust_source,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "could not locate benchmark_trial_unit_definition() in the Rust source")
        rust_trial = normalize_rust_string_literal(match.group("trial"))
        rust_pair = normalize_rust_string_literal(match.group("pair"))
        self.assertEqual(normalize_whitespace(harness.TRIAL_UNIT_DEFINITION["trial"]), rust_trial)
        self.assertEqual(normalize_whitespace(harness.TRIAL_UNIT_DEFINITION["pair"]), rust_pair)


class HostAndThermalCaptureTests(unittest.TestCase):
    def test_certification_host_classification_for_macos_arm64(self):
        harness = load_module()
        # capture_host_classification reads the REAL platform; verify the
        # lookup table it consults instead, which is what actually
        # determines certification-host status.
        self.assertEqual(
            harness.CERTIFICATION_HOSTS[("Darwin", "arm64")], "macos-arm64-metal"
        )
        self.assertEqual(
            harness.CERTIFICATION_HOSTS[("Linux", "x86_64")], "linux-x86_64-cuda"
        )

    def test_unknown_host_is_informational_only(self):
        harness = load_module()
        self.assertNotIn(("Windows", "AMD64"), harness.CERTIFICATION_HOSTS)

    def test_thermal_state_capture_never_raises_and_has_available_flag(self):
        harness = load_module()
        state = harness.capture_thermal_state(run_pmset=lambda: None, thermal_root=Path("/nonexistent"))
        self.assertIn("available", state)


class HiddenSelectorWiringTests(unittest.TestCase):
    def test_execute_trial_always_sets_gate_and_selector_together(self):
        harness = load_module()
        source = SCRIPT.read_text()
        needle = 'env[TRIAL_GATE_ENV_NAME] = "1"\n    env[TRIAL_ENV_NAME] = resolve_trial_env_value(mode)'
        self.assertIn(needle, source)


class LocalModelOnlyCliCompatibilityTests(unittest.TestCase):
    """`--local-model-only` rejects `--headless` at CLI validation
    (`validate_local_model_only_options` in
    `crates/mesh-llm-host-runtime/src/runtime/local_model_only.rs`: "never
    starts a console; remove --headless") and never starts a console/
    management API at all ("does not start owner control or management
    APIs", same function) -- so `execute_trial`'s argv must never pass
    `--console` or `--headless` alongside `--local-model-only`, or every
    real trial launch fails at startup before readiness is even polled."""

    def test_argv_never_passes_console_or_headless(self):
        harness = load_module()
        source = SCRIPT.read_text()
        argv_start = source.index("argv = [", source.index("def execute_trial"))
        argv_end = source.index("]", argv_start)
        argv_block = source[argv_start:argv_end]
        self.assertNotIn("--console", argv_block)
        self.assertNotIn("--headless", argv_block)
        self.assertIn("--local-model-only", argv_block)

    def test_argv_always_disables_native_mtp_speculative_decoding(self):
        """Confirmed against a real running binary: native in-model MTP
        crashes every inference request (`llama_decode failed` /
        `backend sampling requires at most one output token per sequence`)
        unless `--speculative-strategy disabled` is passed; `--no-draft`
        does not help (it only covers separate sibling draft-model files)."""
        harness = load_module()
        source = SCRIPT.read_text()
        argv_start = source.index("argv = [", source.index("def execute_trial"))
        argv_end = source.index("]", argv_start)
        argv_block = source[argv_start:argv_end]
        self.assertIn("--speculative-strategy", argv_block)
        self.assertIn('"disabled"', argv_block)


class ExecuteTrialTimeoutTests(unittest.TestCase):
    def test_model_resolution_and_requests_use_request_timeout(self):
        harness = load_module()
        entry = harness.TrialPlanEntry(
            scenario="__primary__",
            pair_index=0,
            prompt_seed=1,
            side_order_first="production",
        )
        process = mock.Mock()
        process.wait.return_value = 0
        resolved_timeouts = []

        def resolve_model_id(base_url, timeout_secs):
            resolved_timeouts.append((base_url, timeout_secs))
            return "served-model"

        parsed = harness.StreamParseResult(completion_tokens=1, ttft_ms=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(harness, "reserve_local_port", return_value=12345),
                mock.patch.object(harness, "wait_for_readiness", return_value=True),
                mock.patch.object(
                    harness,
                    "resolve_ready_model_id",
                    side_effect=resolve_model_id,
                ),
                mock.patch.object(
                    harness,
                    "send_streaming_chat_request",
                    return_value=parsed,
                ) as send_request,
                mock.patch.object(harness.subprocess, "Popen", return_value=process),
            ):
                result = harness.execute_trial(
                    Path("/bin/true"),
                    "fixture-model",
                    "production",
                    entry,
                    Path(tmp) / "trial.log",
                    readiness_timeout_secs=31.0,
                    readiness_poll_interval_secs=0.25,
                    request_timeout_secs=23.5,
                )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(resolved_timeouts, [("http://127.0.0.1:12345", 23.5)])
        self.assertEqual(send_request.call_count, 2)
        self.assertTrue(all(call.args[3] == 23.5 for call in send_request.call_args_list))


class CliParsingTests(unittest.TestCase):
    def test_help_lists_every_frozen_flag(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        help_text = parser.format_help()
        for flag in (
            "--binary",
            "--baseline-binary",
            "--model",
            "--output-dir",
            "--pairs-primary",
            "--pairs-scenario",
            "--seed",
            "--mode",
            "--scenario",
        ):
            self.assertIn(flag, help_text)

    def test_mode_is_repeatable_for_comparison_a(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/bin/true",
                "--model",
                "fixture-model",
                "--output-dir",
                "/tmp/out",
                "--pairs-primary",
                "20",
                "--pairs-scenario",
                "10",
                "--seed",
                "42",
                "--mode",
                "production",
                "--mode",
                "event-disabled",
                "--scenario",
                "chat_short",
            ]
        )
        self.assertEqual(args.modes, ["production", "event-disabled"])
        self.assertEqual(args.scenarios, ["chat_short"])
        self.assertIsNone(args.baseline_binary)

    def test_baseline_binary_accepted_alongside_a_single_mode(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/bin/current",
                "--baseline-binary",
                "/bin/baseline",
                "--model",
                "fixture-model",
                "--output-dir",
                "/tmp/out",
                "--pairs-primary",
                "20",
                "--pairs-scenario",
                "10",
                "--seed",
                "42",
                "--mode",
                "production",
                "--scenario",
                "chat_short",
            ]
        )
        self.assertEqual(args.modes, ["production"])
        self.assertEqual(args.baseline_binary, Path("/bin/baseline"))

    def test_attempt_defaults_to_one(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/bin/true",
                "--model",
                "m",
                "--output-dir",
                "/tmp/out",
                "--pairs-primary",
                "1",
                "--pairs-scenario",
                "1",
                "--seed",
                "1",
                "--mode",
                "production",
                "--scenario",
                "a",
            ]
        )
        self.assertEqual(args.attempt, 1)

    def test_attempt_accepts_explicit_retry_value(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/bin/true",
                "--model",
                "m",
                "--output-dir",
                "/tmp/out",
                "--pairs-primary",
                "1",
                "--pairs-scenario",
                "1",
                "--seed",
                "1",
                "--mode",
                "production",
                "--scenario",
                "a",
                "--attempt",
                "2",
            ]
        )
        self.assertEqual(args.attempt, 2)

    def test_scenario_is_repeatable(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/bin/true",
                "--model",
                "m",
                "--output-dir",
                "/tmp/out",
                "--pairs-primary",
                "1",
                "--pairs-scenario",
                "1",
                "--seed",
                "1",
                "--mode",
                "production",
                "--scenario",
                "a",
                "--scenario",
                "b",
            ]
        )
        self.assertEqual(args.scenarios, ["a", "b"])

    def test_help_exits_zero_via_subprocess(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=30, check=False
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--mode", result.stdout)


class RunPairedTrialPlanInterleavingTests(unittest.TestCase):
    """`run_paired_trial_plan` is the D13 fix: ONE invocation must run BOTH
    sides of every pair BACK TO BACK, in the seeded per-pair order --
    never all of side A's trials followed by all of side B's (the old
    "separate invocations" shape). Every call here injects a fake
    executor; this module's tests must never spawn a real mesh-llm
    process (see the module docstring)."""

    @staticmethod
    def _make_fake_executor(harness, call_log):
        def fake_executor(binary, model, mode, entry, log_path):
            call_log.append((mode, entry.scenario, entry.pair_index))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("")
            return harness.TrialResult(
                scenario=entry.scenario,
                pair_index=entry.pair_index,
                side_order_first=entry.side_order_first,
                status="succeeded",
                completion_tokens=1,
                elapsed_ms=1.0,
                decode_tok_s=1.0,
                ttft_ms=1.0,
                decode_only_tok_s=1.0,
                setup_ms=1.0,
                readiness_ms=1.0,
                shutdown_ms=1.0,
            )

        return fake_executor

    def test_never_calls_the_real_executor_by_default_in_tests(self):
        harness = load_module()
        plan = harness.build_trial_plan(1, 1, 1, ["s"])
        side_a = harness.SideSpec(binary=Path("/nonexistent"), mode="production", side_id="production")
        side_b = harness.SideSpec(binary=Path("/nonexistent"), mode="event-disabled", side_id="event-disabled")
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            result = harness.run_paired_trial_plan(
                "model",
                plan,
                side_a,
                side_b,
                log_dir=Path(tmp),
                trial_executor=self._make_fake_executor(harness, calls),
            )
        self.assertEqual(len(result.side_a_results) + len(result.side_b_results), 2 * len(plan))
        self.assertEqual(len(calls), 2 * len(plan))

    def test_each_pairs_two_trials_run_back_to_back_in_plan_order(self):
        """For a 3-pair plan, the fake executor's call sequence must be
        [pair0-first, pair0-second, pair1-first, pair1-second, pair2-first,
        pair2-second] -- NEVER [pair0-first, pair1-first, pair2-first,
        pair0-second, ...] (all of one side, then all of the other),
        which is what running the two sides as separate invocations
        produced before this task. The expected sequence is reconstructed
        directly from the plan's own `side_order_first` labels."""
        harness = load_module()
        plan = harness.build_trial_plan(3, 3, 1, ["s"])
        side_a = harness.SideSpec(binary=Path("/nonexistent"), mode="production", side_id="production")
        side_b = harness.SideSpec(binary=Path("/nonexistent"), mode="event-disabled", side_id="event-disabled")
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            harness.run_paired_trial_plan(
                "model",
                plan,
                side_a,
                side_b,
                log_dir=Path(tmp),
                trial_executor=self._make_fake_executor(harness, calls),
            )
        expected: list[tuple[str, str, int]] = []
        for entry in plan:
            other = side_b.mode if entry.side_order_first == side_a.mode else side_a.mode
            expected.append((entry.side_order_first, entry.scenario, entry.pair_index))
            expected.append((other, entry.scenario, entry.pair_index))
        self.assertEqual(calls, expected)

    def test_executed_order_records_the_real_first_and_second_side_ids(self):
        harness = load_module()
        plan = harness.build_trial_plan(9, 2, 1, ["s"])
        side_a = harness.SideSpec(binary=Path("/nonexistent"), mode="production", side_id="production")
        side_b = harness.SideSpec(binary=Path("/nonexistent"), mode="event-disabled", side_id="event-disabled")
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            result = harness.run_paired_trial_plan(
                "model",
                plan,
                side_a,
                side_b,
                log_dir=Path(tmp),
                trial_executor=self._make_fake_executor(harness, calls),
            )
        self.assertEqual(len(result.executed_order), len(plan))
        for order_entry, plan_entry in zip(result.executed_order, plan, strict=True):
            self.assertEqual(order_entry.scenario, plan_entry.scenario)
            self.assertEqual(order_entry.pair_index, plan_entry.pair_index)
            self.assertEqual(order_entry.order[0], plan_entry.side_order_first)
            self.assertEqual({order_entry.order[0], order_entry.order[1]}, {"production", "event-disabled"})

    def test_side_a_and_side_b_results_are_bucketed_correctly_regardless_of_launch_order(self):
        harness = load_module()
        # 4 primary pairs, no named scenarios beyond the required minimum
        # one pair -- keeps `pair_index` unique-by-position across the
        # whole plan so this test can assert positional correspondence
        # directly, rather than conflating the primary group's indices
        # with a named scenario's own independently-numbered indices.
        plan = harness.build_trial_plan(2, 4, 1, ["s"])
        side_a = harness.SideSpec(binary=Path("/nonexistent"), mode="production", side_id="production")
        side_b = harness.SideSpec(binary=Path("/nonexistent"), mode="event-disabled", side_id="event-disabled")
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            result = harness.run_paired_trial_plan(
                "model",
                plan,
                side_a,
                side_b,
                log_dir=Path(tmp),
                trial_executor=self._make_fake_executor(harness, calls),
            )
        self.assertEqual(len(result.side_a_results), len(plan))
        self.assertEqual(len(result.side_b_results), len(plan))
        for position, plan_entry in enumerate(plan):
            self.assertEqual(result.side_a_results[position].scenario, plan_entry.scenario)
            self.assertEqual(result.side_a_results[position].pair_index, plan_entry.pair_index)
            self.assertEqual(result.side_b_results[position].scenario, plan_entry.scenario)
            self.assertEqual(result.side_b_results[position].pair_index, plan_entry.pair_index)

    def test_health_and_p99_are_derived_from_each_sides_own_last_trial_log(self):
        """`run_paired_trial_plan` must read health/p99 from a REAL fixture
        log written by the (fake) executor for the LAST trial of each
        side -- proving `execute_trial`'s `log_path` plumbing round-trips
        through `health_and_p99_from_trial_log` end to end, without ever
        spawning a real process."""
        harness = load_module()
        plan = harness.build_trial_plan(1, 2, 1, ["s"])
        side_a = harness.SideSpec(binary=Path("/nonexistent"), mode="production", side_id="production")
        side_b = harness.SideSpec(binary=Path("/nonexistent"), mode="event-disabled", side_id="event-disabled")

        health_line_by_mode = {
            "production": (
                '{"context":"event_system_health","message":'
                '"version=1 reservation_exhausted=0 cancelled_reservation_rejected=0 terminal_delivery_failed=0 '
                "dropped_progress=0 dropped_diagnostic=0 replay_evicted=0 "
                "subscriber_disconnected=0 shutdown_degraded=0 reducer_rejected=0 "
                'rebuild_generation=0 ingress_p99_us=12"}'
            ),
            "event-disabled": (
                '{"context":"event_system_health","message":'
                '"version=1 reservation_exhausted=0 cancelled_reservation_rejected=0 terminal_delivery_failed=0 '
                "dropped_progress=1 dropped_diagnostic=1 replay_evicted=0 "
                "subscriber_disconnected=0 shutdown_degraded=0 reducer_rejected=0 "
                'rebuild_generation=0 ingress_p99_us=null"}'
            ),
        }

        def fake_executor(binary, model, mode, entry, log_path):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(health_line_by_mode[mode] + "\n")
            return harness.TrialResult(
                scenario=entry.scenario,
                pair_index=entry.pair_index,
                side_order_first=entry.side_order_first,
                status="succeeded",
                completion_tokens=1,
                elapsed_ms=1.0,
                decode_tok_s=1.0,
                ttft_ms=1.0,
                decode_only_tok_s=1.0,
                setup_ms=1.0,
                readiness_ms=1.0,
                shutdown_ms=1.0,
            )

        with tempfile.TemporaryDirectory() as tmp:
            result = harness.run_paired_trial_plan(
                "model", plan, side_a, side_b, log_dir=Path(tmp), trial_executor=fake_executor
            )
        self.assertEqual(result.side_a_ingress_p99_us, 12.0)
        self.assertIsNotNone(result.side_a_health)
        self.assertIsNone(result.side_b_ingress_p99_us)
        self.assertIsNotNone(result.side_b_health)
        self.assertEqual(result.side_b_health["dropped_progress"], 1)

    def test_health_unavailable_when_no_trial_ran_for_a_side(self):
        """Degenerate but honest: if a side never ran a single trial (e.g.
        an empty plan), health/p99 must stay `None`, not raise or
        fabricate -- `health_and_p99_from_trial_log` is never called with
        a nonexistent path guess."""
        harness = load_module()
        side_a = harness.SideSpec(binary=Path("/nonexistent"), mode="production", side_id="production")
        side_b = harness.SideSpec(binary=Path("/nonexistent"), mode="event-disabled", side_id="event-disabled")
        with tempfile.TemporaryDirectory() as tmp:
            result = harness.run_paired_trial_plan(
                "model", [], side_a, side_b, log_dir=Path(tmp), trial_executor=self._make_fake_executor(harness, [])
            )
        self.assertIsNone(result.side_a_health)
        self.assertIsNone(result.side_a_ingress_p99_us)
        self.assertIsNone(result.side_b_health)
        self.assertIsNone(result.side_b_ingress_p99_us)


class SideOrderThreadingTests(unittest.TestCase):
    """`execute_trial` must thread `entry.side_order_first` into every
    `TrialResult` it constructs (all 5 return paths) so the manifest can
    record the deterministic per-pair ordering for the comparator to
    verify -- a source-level check since `execute_trial` spawns a real
    process and is never invoked by this module's own test suite (see the
    module docstring)."""

    def test_execute_trial_threads_side_order_first_into_every_trial_result(self):
        harness = load_module()
        source = SCRIPT.read_text()
        start = source.index("def execute_trial(")
        end = source.index("\ndef ", start + 1)
        body = source[start:end]
        occurrences = body.count("side_order_first=entry.side_order_first")
        self.assertEqual(
            occurrences,
            5,
            f"expected 5 TrialResult(...) constructions to thread side_order_first, found {occurrences}",
        )


if __name__ == "__main__":
    unittest.main()
