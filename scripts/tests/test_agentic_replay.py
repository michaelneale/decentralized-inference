from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evals/agentic-replay.py"


def load_module():
    spec = importlib.util.spec_from_file_location("agentic_replay", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module()


class AgenticReplayTest(unittest.TestCase):
    def specs(self):
        return [
            BENCH.RefSpec("rc8", "v0.76.0-rc8", "a" * 40),
            BENCH.RefSpec("main", "origin/main", "b" * 40),
        ]

    def test_ab_order_reverses_every_other_pass(self) -> None:
        order = BENCH.ab_order(self.specs(), 3)

        self.assertEqual(
            [(pass_index, spec.label) for pass_index, spec in order],
            [
                (0, "rc8"),
                (0, "main"),
                (1, "main"),
                (1, "rc8"),
                (2, "rc8"),
                (2, "main"),
            ],
        )

    def test_single_ref_supports_absolute_measurement(self) -> None:
        with mock.patch.object(BENCH, "git", return_value="a" * 40):
            specs = BENCH.parse_ref_specs(REPO, ["main=HEAD"])

        self.assertEqual(specs, [BENCH.RefSpec("main", "HEAD", "a" * 40)])

    def test_trajectory_replay_uses_recorded_history_in_strict_turn_order(self) -> None:
        trajectory = {
            "session_id": "session-1",
            "source_dataset": "source",
            "agent_framework": "framework",
            "recorded_model": "recorded-model",
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "recorded answer 1"},
                {"role": "user", "content": "tool observation"},
                {"role": "assistant", "content": "recorded answer 2"},
            ],
        }
        calls = []
        original = BENCH.stream_request

        def fake_stream(
            request_id, messages, tools, metadata, model_id, output_tokens, timeout
        ):
            calls.append((request_id, list(messages), list(tools), dict(metadata)))
            return {**metadata, "request_id": request_id, "error": "fixture"}

        BENCH.stream_request = fake_stream
        try:
            BENCH.replay_trajectory(trajectory, "model", 256, 10)
        finally:
            BENCH.stream_request = original

        self.assertEqual([call[3]["assistant_turn"] for call in calls], [0, 1])
        self.assertEqual(
            [message["content"] for message in calls[0][1]],
            ["rules", "task"],
        )
        self.assertEqual(
            [message["content"] for message in calls[1][1]],
            ["rules", "task", "recorded answer 1", "tool observation"],
        )

    def test_trajectory_replay_honors_warmup_turn_limit(self) -> None:
        trajectory = {
            "session_id": "session-1",
            "source_dataset": "source",
            "agent_framework": "framework",
            "recorded_model": "recorded-model",
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "observation"},
                {"role": "assistant", "content": "second"},
            ],
        }
        original = BENCH.stream_request
        BENCH.stream_request = lambda *args, **kwargs: {"request_id": args[0]}
        try:
            results = BENCH.replay_trajectory(
                trajectory, "model", 2048, 10, turn_limit=1
            )
        finally:
            BENCH.stream_request = original

        self.assertEqual([result["request_id"] for result in results], ["session-1:0"])

    def test_stream_request_preserves_in_stream_server_errors(self) -> None:
        class ErrorResponse:
            status = 200

            def __iter__(self):
                return iter(
                    [
                        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
                        b'data: {"error":{"message":"cache operation deadline exceeded","type":"timeout"}}\n',
                        b"data: [DONE]\n",
                    ]
                )

        class ErrorConnection:
            def __init__(self, *_args, **_kwargs):
                pass

            def request(self, *_args, **_kwargs):
                pass

            def getresponse(self):
                return ErrorResponse()

            def close(self):
                pass

        with mock.patch.object(BENCH.http.client, "HTTPConnection", ErrorConnection):
            result = BENCH.stream_request(
                "request-1",
                [{"role": "user", "content": "task"}],
                [],
                {"session_id": "session-1"},
                "model",
                128,
                10,
            )

        self.assertEqual(result["request_id"], "request-1")
        self.assertEqual(result["session_id"], "session-1")
        self.assertEqual(
            result["error"],
            "stream failed with server error: cache operation deadline exceeded",
        )

    def test_stream_request_rejects_truncated_stream_without_done(self) -> None:
        class TruncatedResponse:
            status = 200

            def __iter__(self):
                return iter(
                    [
                        b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
                        b'data: {"choices":[],"usage":{"completion_tokens":1,"prompt_tokens":10}}\n',
                    ]
                )

        class TruncatedConnection:
            def __init__(self, *_args, **_kwargs):
                pass

            def request(self, *_args, **_kwargs):
                pass

            def getresponse(self):
                return TruncatedResponse()

            def close(self):
                pass

        with mock.patch.object(
            BENCH.http.client, "HTTPConnection", TruncatedConnection
        ):
            result = BENCH.stream_request(
                "request-1",
                [{"role": "user", "content": "task"}],
                [],
                {"session_id": "session-1"},
                "model",
                128,
                10,
            )

        self.assertEqual(
            result["error"], "stream ended without terminal [DONE] marker"
        )

    def test_output_hash_ignores_tool_call_chunking_and_generated_ids(self) -> None:
        chunked = {}
        BENCH.merge_tool_call_delta(
            chunked,
            {
                "index": 0,
                "id": "call-random-a",
                "type": "function",
                "function": {"name": "shell", "arguments": '{"command":'},
            },
            0,
        )
        BENCH.merge_tool_call_delta(
            chunked,
            {"index": 0, "function": {"arguments": '"ls"}'}},
            0,
        )
        unchunked = {}
        BENCH.merge_tool_call_delta(
            unchunked,
            {
                "index": 0,
                "id": "call-random-b",
                "type": "function",
                "function": {"name": "shell", "arguments": '{"command":"ls"}'},
            },
            0,
        )

        self.assertEqual(
            BENCH.response_content_sha256([], [], chunked),
            BENCH.response_content_sha256([], [], unchunked),
        )

    def test_checkpoint_replay_reconstructs_full_recorded_prefix(self) -> None:
        trajectory = {
            "session_id": "session-1",
            "source_dataset": "source",
            "agent_framework": "framework",
            "recorded_model": "recorded-model",
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "recorded first"},
                {"role": "tool", "content": "observation", "tool_call_id": "1"},
                {"role": "assistant", "content": "recorded second"},
            ],
        }
        calls = []
        original = BENCH.stream_request

        def fake_stream(*args, **kwargs):
            calls.append((args[0], list(args[1]), args[2], dict(args[3])))
            return {"request_id": args[0]}

        BENCH.stream_request = fake_stream
        try:
            BENCH.replay_trajectory(
                trajectory,
                "model",
                2048,
                10,
                measured_assistant_turns={1},
                checkpoint_stage="final",
            )
        finally:
            BENCH.stream_request = original

        self.assertEqual([call[0] for call in calls], ["session-1:1"])
        self.assertEqual(
            [message["content"] for message in calls[0][1]],
            ["task", "recorded first", "observation"],
        )
        self.assertEqual(calls[0][3]["checkpoint_stage"], "final")

    def test_checkpoint_schedule_balances_four_stages_per_framework(self) -> None:
        trajectories = []
        for index in range(4):
            messages = []
            for turn in range(8):
                messages.extend(
                    [
                        {"role": "user", "content": f"u{turn}"},
                        {"role": "assistant", "content": f"a{turn}"},
                    ]
                )
            trajectories.append(
                {
                    "session_id": f"session-{index}",
                    "agent_framework": "framework",
                    "messages": messages,
                }
            )

        schedule = BENCH.checkpoint_schedule(trajectories)

        self.assertEqual(
            list(schedule.values()),
            [(1, "early"), (3, "middle"), (5, "late"), (7, "final")],
        )

    def test_trajectory_tools_are_stable_and_schema_shaped(self) -> None:
        trajectory = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls_json": json.dumps(
                        [
                            {"type": "function", "function": {"name": "shell"}},
                            {"type": "function", "function": {"name": "editor"}},
                        ]
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls_json": json.dumps(
                        [{"type": "function", "function": {"name": "shell"}}]
                    ),
                },
            ]
        }

        tools = BENCH.trajectory_tools(trajectory)

        self.assertEqual(
            [tool["function"]["name"] for tool in tools], ["editor", "shell"]
        )
        self.assertTrue(
            all(tool["function"]["parameters"]["additionalProperties"] for tool in tools)
        )

    def test_trajectory_tools_preserve_captured_harness_schemas(self) -> None:
        captured = [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

        self.assertEqual(BENCH.trajectory_tools({"tools": captured}), captured)

    def test_captured_message_preserves_openai_fields_and_tool_calls(self) -> None:
        recorded = {
            "role": "assistant",
            "content": "",
            "name": "agent",
            "reasoning_content": "private reasoning",
            "provider_extension": {"cache_control": "ephemeral"},
            "tool_calls": [{"id": "call-1", "type": "function"}],
        }

        self.assertEqual(BENCH.openai_message(recorded), recorded)

    def test_captured_message_decodes_legacy_tool_calls_without_forwarding_alias(
        self,
    ) -> None:
        tool_calls = [{"id": "call-1", "type": "function"}]
        recorded = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "private reasoning",
            "tool_calls_json": json.dumps(tool_calls),
        }

        self.assertEqual(
            BENCH.openai_message(recorded),
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "private reasoning",
                "tool_calls": tool_calls,
            },
        )

    def test_captured_message_omits_null_optional_fields(self) -> None:
        recorded = {
            "role": "user",
            "content": "task",
            "tool_call_id": None,
            "tool_calls_json": None,
        }

        self.assertEqual(
            BENCH.openai_message(recorded),
            {"role": "user", "content": "task"},
        )

    def test_captured_tool_call_sizes_the_generated_output_budget(self) -> None:
        recorded = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": '{"command":"find . -type f"}',
                    },
                }
            ],
        }

        self.assertGreater(BENCH.recorded_output_budget(recorded, 2048), 8)

    def test_final_replay_measures_only_the_last_assistant_prefix(self) -> None:
        trajectory = {
            "session_id": "session-1",
            "source_dataset": "buzz-capture",
            "agent_framework": "buzz",
            "recorded_model": "model",
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "first"},
                {"role": "tool", "content": "observation", "tool_call_id": "1"},
                {"role": "assistant", "content": "second"},
            ],
        }
        calls = []
        original = BENCH.stream_request
        BENCH.stream_request = lambda *args, **kwargs: calls.append(args) or {
            "request_id": args[0],
            "session_id": args[3]["session_id"],
            "started": 0.0,
            "first_token_at": 1.0,
            "completed": 2.0,
            "ttft_seconds": 1.0,
            "elapsed_seconds": 2.0,
            "generation_seconds": 1.0,
            "completion_tokens": 1,
            "prompt_tokens": 100,
            "cached_tokens": 0,
            "content_sha256": "a" * 64,
        }
        try:
            BENCH.run_trajectory_cell(
                trajectories=[trajectory],
                model_id="model",
                concurrency=1,
                max_output_tokens=8,
                timeout=10,
                raw_path=Path(tempfile.gettempdir()) / "agentic-final-test.jsonl",
                replay_mode="final",
            )
        finally:
            BENCH.stream_request = original

        self.assertEqual([call[0] for call in calls], ["session-1:1"])
        self.assertEqual(calls[0][3]["checkpoint_stage"], "final")

    def test_server_command_keeps_mesh_planning_at_defaults(self) -> None:
        command = BENCH.server_command(Path("/product/mesh-llm"), "model-uri")

        self.assertEqual(
            command,
            [
                "/product/mesh-llm",
                "serve",
                "--model",
                "model-uri",
                "--log-format",
                "json",
            ],
        )
        for option in BENCH.FORBIDDEN_STARTUP_OPTIONS:
            self.assertNotIn(option, command)

    def test_isolated_server_state_preserves_the_model_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hf_home = root / "shared-hf"

            env = BENCH.isolated_server_env(
                root / "runtime-bundle", root / "state", hf_home
            )

            self.assertEqual(env["HF_HOME"], str(hf_home))
            self.assertEqual(env["HOME"], str(root / "state/home"))
            self.assertEqual(
                env["MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR"],
                str(root / "runtime-bundle"),
            )

    def test_runtime_evidence_collects_logs_without_copying_identity_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            native_log = state / "runtime/123/logs/skippy-native.log"
            identity = state / "home/.mesh-llm/identity.key"
            native_log.parent.mkdir(parents=True)
            identity.parent.mkdir(parents=True)
            native_log.write_text("runtime evidence", encoding="utf-8")
            identity.write_text("secret", encoding="utf-8")

            BENCH.collect_runtime_logs(state, root / "artifact")

            self.assertEqual(
                (root / "artifact/123/logs/skippy-native.log").read_text(
                    encoding="utf-8"
                ),
                "runtime evidence",
            )
            self.assertFalse((root / "artifact/identity.key").exists())

    def test_default_cli_workload_has_external_concurrency_only(self) -> None:
        args = BENCH.parse_args(
            [
                "plan",
                "--ref",
                "rc8=v0.76.0-rc8",
                "--ref",
                "main=origin/main",
                "--model",
                "model-uri",
                "--trajectories-per-framework",
                "4",
            ]
        )

        self.assertEqual(args.concurrency, [1, 2, 4])
        self.assertEqual(args.minimum_worker_waves, 2)
        self.assertEqual(args.passes, 1)
        self.assertEqual(args.replay_mode, "checkpoints")
        self.assertEqual(args.trajectories_per_framework, 4)
        self.assertEqual(args.framework, ["swe-agent", "mini-swe-agent", "openhands"])
        self.assertEqual(args.max_output_tokens, 2048)
        self.assertEqual(args.warmup_turns, 4)
        self.assertEqual(args.max_isl, 65536)

    def test_cli_rejects_a_single_wave_concurrency_cohort(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                BENCH.parse_args(
                    [
                        "plan",
                        "--ref",
                        "rc8=v0.76.0-rc8",
                        "--ref",
                        "main=origin/main",
                        "--model",
                        "model-uri",
                        "--trajectories-per-framework",
                        "2",
                    ]
                )

    def test_cli_rejects_zero_cache_threshold(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                BENCH.parse_args(
                    [
                        "plan",
                        "--ref",
                        "rc8=v0.76.0-rc8",
                        "--ref",
                        "main=origin/main",
                        "--model",
                        "model-uri",
                        "--trajectories-per-framework",
                        "4",
                        "--min-cache-pct",
                        "0",
                    ]
                )

    def test_manifest_contract_is_validated_before_builds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "cohorts": {
                            "warmup": [
                                {
                                    "session_id": "session-1",
                                    "source_dataset": "source",
                                    "agent_framework": "framework",
                                    "recorded_model": "model",
                                    "messages": [{"role": "user", "content": "task"}],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing cohorts"):
                BENCH.load_trajectory_cohorts(manifest, ["warmup", "1"])

    def test_manifest_rejects_non_object_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps({"cohorts": {"warmup": ["not-an-object"]}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "non-object trajectory"):
                BENCH.load_trajectory_cohorts(manifest, ["warmup"])

    def test_manifest_allows_dataset_null_recorded_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            trajectory = {
                "session_id": "session-1",
                "source_dataset": "source",
                "agent_framework": "framework",
                "recorded_model": None,
                "messages": [{"role": "user", "content": "task"}],
            }
            manifest.write_text(
                json.dumps({"cohorts": {"warmup": [trajectory]}}),
                encoding="utf-8",
            )

            cohorts = BENCH.load_trajectory_cohorts(manifest, ["warmup"])

            self.assertIsNone(cohorts["warmup"][0]["recorded_model"])

    def test_imported_manifest_is_copied_hashed_and_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "capture.json"
            output = root / "artifact"
            trajectory = {
                "session_id": "buzz-1",
                "source_dataset": "buzz-capture",
                "agent_framework": "buzz",
                "recorded_model": None,
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "answer"},
                ],
                "tools": [],
            }
            source.write_text(
                json.dumps(
                    {
                        "metadata": {"name": "real harness capture", "revision": "r1"},
                        "cohorts": {"warmup": [trajectory], "1": [trajectory]},
                    }
                ),
                encoding="utf-8",
            )

            imported = BENCH.import_trajectory_manifest(
                source, output, ["warmup", "1"]
            )

            self.assertEqual(imported["kind"], "captured")
            self.assertEqual(imported["dataset"]["revision"], "r1")
            self.assertEqual(imported["cohorts"]["1"]["assistant_turns"], 1)
            self.assertEqual(
                imported["source_manifest_sha256"], imported["manifest_sha256"]
            )

    def test_warmup_capacity_is_validated_before_builds(self) -> None:
        trajectories = [
            {
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "action"},
                ]
            }
        ]

        with self.assertRaisesRegex(ValueError, "1 assistant turns; 14 required"):
            BENCH.validate_warmup_capacity(trajectories, 14)

    def test_measured_cohort_requires_two_worker_waves(self) -> None:
        with self.assertRaisesRegex(ValueError, "8 required"):
            BENCH.validate_measured_cohort_capacity(
                {"4": [{"session_id": str(index)} for index in range(7)]}, [4]
            )

    def test_measured_cohort_can_explicitly_allow_one_worker_wave(self) -> None:
        BENCH.validate_measured_cohort_capacity(
            {"8": [{"session_id": str(index)} for index in range(8)]},
            [8],
            minimum_worker_waves=1,
        )

    def test_required_frameworks_are_checked_in_every_measured_cohort(self) -> None:
        cohorts = {
            "1": [
                {"agent_framework": "buzz"},
                {"agent_framework": "opencode"},
            ],
            "2": [
                {"agent_framework": "buzz"},
                {"agent_framework": "goose"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "concurrency 1.*goose"):
            BENCH.validate_required_frameworks(
                cohorts, [1, 2], ["buzz", "opencode", "goose"]
            )

    def test_summarize_requests_computes_stream_windows_and_cache_rate(self) -> None:
        requests = [
            {
                "started": 0.0,
                "first_token_at": 1.0,
                "completed": 3.0,
                "ttft_seconds": 1.0,
                "elapsed_seconds": 3.0,
                "generation_seconds": 2.0,
                "completion_tokens": 10,
                "prompt_tokens": 100,
                "cached_tokens": 50,
                "requested_output_tokens": 10,
                "finish_reason": "length",
                "request_id": "request-1",
                "content_sha256": "a" * 64,
            },
            {
                "started": 0.0,
                "first_token_at": 2.0,
                "completed": 4.0,
                "ttft_seconds": 2.0,
                "elapsed_seconds": 4.0,
                "generation_seconds": 2.0,
                "completion_tokens": 10,
                "prompt_tokens": 100,
                "cached_tokens": 100,
                "requested_output_tokens": 10,
                "finish_reason": "stop",
                "request_id": "request-2",
                "content_sha256": "b" * 64,
            },
        ]

        summary = BENCH.summarize_requests(requests, offered_concurrency=2)

        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["failed_request_ids"], [])
        self.assertEqual(summary["budget_exhausted_requests"], 2)
        self.assertEqual(summary["finish_reason_length_requests"], 1)
        self.assertEqual(summary["agent_steps_per_second"], 0.5)
        self.assertEqual(summary["workload_output_tokens_per_second"], 5)
        self.assertEqual(summary["decode_tokens_per_second"], 5)
        self.assertEqual(summary["mean_in_flight"], 1.75)
        self.assertEqual(summary["concurrency_utilization_pct"], 87.5)
        self.assertEqual(summary["ttft_samples"], [1.0, 2.0])
        self.assertEqual(summary["ttft_p50_seconds"], 1.5)
        self.assertEqual(summary["ttft_p95_seconds"], 2.0)
        self.assertEqual(summary["cache_pct"], 75)
        self.assertEqual(summary["prompt_tokens_min"], 100)
        self.assertEqual(summary["prompt_tokens_max"], 100)
        self.assertEqual(
            summary["content_sha256_by_request"]["request-1"], "a" * 64
        )

    def test_acceptance_gates_cover_19k_cache_output_and_ttft(self) -> None:
        baseline = {
            "label": "base",
            "concurrency": 1,
            "failed_requests": 0,
            "prompt_tokens_min": 19_100,
            "prompt_tokens_max": 19_900,
            "cache_pct": 80.0,
            "delta_comparable": True,
            "content_identity_known": True,
            "ttft_p50_seconds_delta_pct": 0.0,
        }
        candidate = {
            **baseline,
            "label": "candidate",
            "cache_pct": 85.0,
            "ttft_p50_seconds_delta_pct": 3.0,
        }

        gates = BENCH.evaluate_gates(
            [baseline, candidate],
            prompt_token_range=(18_000, 21_000),
            min_cache_pct=75.0,
            require_output_match=True,
            max_ttft_regression_pct=5.0,
        )

        self.assertTrue(gates["passed"])
        self.assertTrue(gates["evaluated"])
        self.assertEqual(len(gates["checks"]), 10)

    def test_acceptance_gate_failure_is_explicit(self) -> None:
        gates = BENCH.evaluate_gates(
            [
                {
                    "label": "candidate",
                    "concurrency": 4,
                    "failed_requests": 1,
                    "prompt_tokens_min": 17_999,
                    "prompt_tokens_max": 19_000,
                    "cache_pct": 20.0,
                    "delta_comparable": False,
                    "ttft_p50_seconds_delta_pct": 8.0,
                }
            ],
            prompt_token_range=(18_000, 21_000),
            min_cache_pct=75.0,
            require_output_match=True,
            max_ttft_regression_pct=5.0,
        )

        self.assertFalse(gates["passed"])
        self.assertTrue(all(not check["passed"] for check in gates["checks"]))

    def test_ttft_gate_fails_closed_when_delta_is_unavailable(self) -> None:
        gates = BENCH.evaluate_gates(
            [
                {
                    "label": "candidate",
                    "concurrency": 1,
                    "failed_requests": 0,
                    "ttft_p50_seconds_delta_pct": None,
                }
            ],
            max_ttft_regression_pct=5.0,
        )

        self.assertFalse(gates["passed"])
        self.assertEqual(len(gates["checks"]), 2)
        self.assertIn("observed=unavailable", gates["checks"][1]["detail"])

    def test_acceptance_gate_rejects_failed_measured_requests(self) -> None:
        gates = BENCH.evaluate_gates(
            [
                {
                    "label": "candidate",
                    "concurrency": 1,
                    "failed_requests": 49,
                    "cache_pct": 100.0,
                }
            ],
            min_cache_pct=75.0,
        )

        self.assertTrue(gates["evaluated"])
        self.assertFalse(gates["passed"])
        self.assertEqual(gates["checks"][0]["name"], "failed-requests:candidate/c1")
        self.assertFalse(gates["checks"][0]["passed"])

    def test_acceptance_gates_are_not_evaluated_without_configuration(self) -> None:
        gates = BENCH.evaluate_gates(
            [{"label": "candidate", "concurrency": 1, "failed_requests": 0}]
        )

        self.assertFalse(gates["evaluated"])
        self.assertIsNone(gates["passed"])
        self.assertEqual(gates["checks"], [])

    def test_resumed_empty_cells_cannot_pass_configured_gates(self) -> None:
        resumed_results = [
            {
                "label": "candidate",
                "ref": "candidate",
                "commit": "a" * 40,
                "pass": 1,
                "cells": [],
            }
        ]

        gates = BENCH.evaluate_gates(
            BENCH.pooled_rows(resumed_results), min_cache_pct=50.0
        )

        self.assertTrue(gates["evaluated"])
        self.assertFalse(gates["passed"])
        self.assertEqual(
            gates["checks"],
            [
                {
                    "name": "measured-rows",
                    "passed": False,
                    "detail": "observed=0 required>=1",
                }
            ],
        )

    def test_pooled_rows_preserve_legacy_comparability_without_hashes(self) -> None:
        def successful_cell():
            return {
                "concurrency": 1,
                "trajectories": 2,
                "requests": 2,
                "successful_requests": 2,
                "failed_request_ids": [],
                "budget_exhausted_requests": 0,
                "agent_steps_per_second": 1.0,
                "workload_output_tokens_per_second": 2.0,
                "decode_tokens_per_second": 3.0,
                "ttft_p50_seconds": 4.0,
                "ttft_p95_seconds": 5.0,
                "ttft_samples": [4.0],
                "mean_in_flight": 1.0,
                "concurrency_utilization_pct": 100.0,
                "cache_pct": 50.0,
            }

        rows = BENCH.pooled_rows(
            [
                {
                    "label": "base",
                    "ref": "base",
                    "commit": "a" * 40,
                    "cells": [successful_cell()],
                },
                {
                    "label": "candidate",
                    "ref": "candidate",
                    "commit": "b" * 40,
                    "cells": [successful_cell()],
                },
            ]
        )

        self.assertTrue(all(row["delta_comparable"] for row in rows))

    def test_pooled_rows_require_a_hash_for_every_successful_pass(self) -> None:
        def successful_cell(include_hash: bool):
            cell = {
                "concurrency": 1,
                "trajectories": 1,
                "requests": 1,
                "successful_requests": 1,
                "successful_request_ids": ["session:0"],
                "content_sha256_by_request": {},
                "failed_request_ids": [],
                "budget_exhausted_requests": 0,
                "agent_steps_per_second": 1.0,
                "workload_output_tokens_per_second": 2.0,
                "decode_tokens_per_second": 3.0,
                "ttft_p50_seconds": 4.0,
                "ttft_p95_seconds": 5.0,
                "ttft_samples": [4.0],
                "mean_in_flight": 1.0,
                "concurrency_utilization_pct": 100.0,
                "cache_pct": 50.0,
            }
            if include_hash:
                cell["content_sha256_by_request"] = {"session:0": "a" * 64}
            return cell

        rows = BENCH.pooled_rows(
            [
                {
                    "label": "base",
                    "ref": "base",
                    "commit": "a" * 40,
                    "cells": [successful_cell(True), successful_cell(False)],
                }
            ]
        )

        self.assertFalse(rows[0]["content_identity_known"])

    def test_pooled_rows_suppress_deltas_for_different_failed_requests(self) -> None:
        def failed_cell(request_id):
            return {
                "concurrency": 1,
                "trajectories": 3,
                "requests": 5,
                "successful_requests": 4,
                "failed_request_ids": [request_id],
                "budget_exhausted_requests": 0,
                "agent_steps_per_second": 1.0,
                "workload_output_tokens_per_second": 2.0,
                "decode_tokens_per_second": 3.0,
                "ttft_p50_seconds": 4.0,
                "ttft_p95_seconds": 5.0,
                "ttft_samples": [4.0],
                "mean_in_flight": 1.0,
                "concurrency_utilization_pct": 100.0,
                "cache_pct": 50.0,
            }

        rows = BENCH.pooled_rows(
            [
                {
                    "label": "rc8",
                    "ref": "rc8",
                    "commit": "a" * 40,
                    "cells": [failed_cell("session-a:1")],
                },
                {
                    "label": "main",
                    "ref": "main",
                    "commit": "b" * 40,
                    "cells": [failed_cell("session-b:1")],
                },
            ]
        )

        main = next(row for row in rows if row["label"] == "main")
        self.assertFalse(main["delta_comparable"])
        self.assertIsNone(main["decode_tokens_per_second_delta_pct"])

    def test_pooled_rows_compare_each_ref_to_first_ref(self) -> None:
        def cell(concurrency, generation, ttft):
            return {
                "concurrency": concurrency,
                "trajectories": 3,
                "requests": 5,
                "successful_requests": 5,
                "budget_exhausted_requests": 5,
                "agent_steps_per_second": generation,
                "workload_output_tokens_per_second": generation - 1,
                "decode_tokens_per_second": generation + 1,
                "ttft_p50_seconds": ttft,
                "ttft_p95_seconds": ttft + 1,
                "ttft_samples": [ttft, ttft + 1],
                "mean_in_flight": concurrency,
                "concurrency_utilization_pct": 100,
                "cache_pct": 50,
            }

        rows = BENCH.pooled_rows(
            [
                {
                    "label": "rc8",
                    "ref": "v0.76.0-rc8",
                    "commit": "a" * 40,
                    "cells": [cell(2, 20, 4)],
                },
                {
                    "label": "main",
                    "ref": "origin/main",
                    "commit": "b" * 40,
                    "cells": [cell(2, 15, 3)],
                },
            ]
        )

        main = next(row for row in rows if row["label"] == "main")
        self.assertEqual(main["agent_steps_per_second_delta_pct"], -25)
        self.assertEqual(main["ttft_p50_seconds_delta_pct"], -25)
        self.assertAlmostEqual(
            main["decode_tokens_per_second_delta_pct"], -23.8095238095
        )
        self.assertEqual(main["decode_tokens_per_second_min"], 16)

    def test_pooled_rows_weight_decode_and_pool_raw_ttft_samples(self) -> None:
        def cell(tokens, generation_seconds, ttft_samples):
            return {
                "concurrency": 1,
                "trajectories": 3,
                "requests": 2,
                "successful_requests": 2,
                "failed_request_ids": [],
                "completion_tokens": tokens,
                "prompt_tokens": 100,
                "cached_tokens": 50,
                "generation_seconds": generation_seconds,
                "workload_window_seconds": 10.0,
                "budget_exhausted_requests": 0,
                "agent_steps_per_second": 0.2,
                "workload_output_tokens_per_second": tokens / 10.0,
                "decode_tokens_per_second": tokens / generation_seconds,
                "ttft_p50_seconds": BENCH.percentile(ttft_samples, 0.5),
                "ttft_p95_seconds": BENCH.percentile(ttft_samples, 0.95),
                "ttft_samples": ttft_samples,
                "mean_in_flight": 1.0,
                "concurrency_utilization_pct": 100.0,
                "cache_pct": 50.0,
            }

        rows = BENCH.pooled_rows(
            [
                {
                    "label": "rc8",
                    "ref": "rc8",
                    "commit": "a" * 40,
                    "cells": [
                        cell(100, 10.0, [1.0, 2.0]),
                        cell(100, 20.0, [10.0, 20.0]),
                    ],
                }
            ]
        )

        self.assertAlmostEqual(rows[0]["decode_tokens_per_second"], 200 / 30)
        self.assertEqual(rows[0]["ttft_p50_seconds"], 2.0)
        self.assertEqual(rows[0]["ttft_p95_seconds"], 20.0)
        self.assertEqual(rows[0]["ttft_p50_seconds_min"], 1.0)
        self.assertEqual(rows[0]["ttft_p50_seconds_max"], 10.0)

    def test_report_writes_tables_charts_and_inventory(self) -> None:
        cell = {
            "concurrency": 1,
            "trajectories": 3,
            "requests": 5,
            "successful_requests": 5,
            "budget_exhausted_requests": 5,
            "agent_steps_per_second": 20.0,
            "workload_output_tokens_per_second": 18.0,
            "decode_tokens_per_second": 22.0,
            "ttft_p50_seconds": 1.0,
            "ttft_p95_seconds": 1.2,
            "ttft_samples": [1.0, 1.2],
            "mean_in_flight": 1.0,
            "concurrency_utilization_pct": 100.0,
            "cache_pct": 60.0,
        }
        document = {
            "config": {
                "model": "model",
                "concurrency": [1],
                "warmup_turns": 14,
                "replay_mode": "checkpoints",
            },
            "inputs": {
                "dataset": {"revision": "c" * 40},
                "manifest_sha256": "d" * 64,
                "cohorts": {
                    "warmup": {
                        "trajectory_count": 3,
                        "assistant_turns": 30,
                        "framework_trajectories": {
                            "swe-agent": 1,
                            "mini-swe-agent": 1,
                            "openhands": 1,
                        },
                        "framework_assistant_turns": {
                            "swe-agent": 10,
                            "mini-swe-agent": 10,
                            "openhands": 10,
                        },
                    },
                    "1": {
                        "trajectory_count": 3,
                        "assistant_turns": 100,
                        "framework_trajectories": {
                            "swe-agent": 1,
                            "mini-swe-agent": 1,
                            "openhands": 1,
                        },
                        "framework_assistant_turns": {
                            "swe-agent": 20,
                            "mini-swe-agent": 30,
                            "openhands": 50,
                        },
                    }
                },
            },
            "builds": [
                {"label": "rc8"},
                {"label": "main"},
            ],
            "order": [
                {"label": "rc8"},
                {"label": "main"},
            ],
            "results": [
                {
                    "label": "rc8",
                    "ref": "v0.76.0-rc8",
                    "commit": "a" * 40,
                    "cells": [cell],
                },
                {
                    "label": "main",
                    "ref": "origin/main",
                    "commit": "b" * 40,
                    "cells": [{**cell, "agent_steps_per_second": 22.0}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            report = BENCH.write_report(artifact, document)

            self.assertTrue(report.is_file())
            self.assertTrue((artifact / "summary/comparison.csv").is_file())
            self.assertTrue(
                (artifact / "summary/charts/decode-throughput.svg").is_file()
            )
            self.assertTrue(
                (artifact / "summary/charts/workload-output-throughput.svg").is_file()
            )
            self.assertTrue((artifact / "summary/charts/ttft-p50.svg").is_file())
            self.assertTrue((artifact / "artifact-sha256.txt").is_file())
            self.assertIn(
                "Mesh chooses context size",
                report.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "swe-agent 1 / 20",
                report.read_text(encoding="utf-8"),
            )
            self.assertIn("Decode tok/s", report.read_text(encoding="utf-8"))
            self.assertIn("Slot use", report.read_text(encoding="utf-8"))
            self.assertIn(
                "other cohort sizes use rank/count labels",
                report.read_text(encoding="utf-8"),
            )

    def test_plan_records_exact_default_server_command(self) -> None:
        args = SimpleNamespace(
            repo=REPO,
            backend="metal",
            model="model-uri",
            passes=2,
            source_dataset=["swe-smith-claude-3-7-sonnet"],
            framework=["swe-agent", "mini-swe-agent", "openhands"],
            trajectories_per_framework=4,
            min_isl=8192,
            max_isl=65536,
            min_turns=5,
            concurrency=[1, 2, 4],
            max_output_tokens=2048,
            warmup_turns=14,
            replay_mode="checkpoints",
        )

        plan = BENCH.benchmark_plan(args, self.specs())

        self.assertEqual(
            plan["server_command"],
            [
                "<release-binary>",
                "serve",
                "--model",
                "model-uri",
                "--log-format",
                "json",
            ],
        )
        self.assertEqual(
            [item["label"] for item in plan["order"]],
            ["rc8", "main", "main", "rc8"],
        )
        self.assertEqual(plan["selection"]["measured_unique_trajectory_count"], 36)
        self.assertEqual(plan["selection"]["warmup_unique_trajectory_count"], 12)
        self.assertEqual(plan["workload"]["measured_requests_per_arm_pass"], 36)
        self.assertEqual(plan["workload"]["measured_requests_total"], 144)

    def test_plan_appends_external_arms_to_the_same_abba_order(self) -> None:
        args = SimpleNamespace(
            repo=REPO,
            backend="metal",
            model="model-uri",
            passes=2,
            source_dataset=["swe-smith-claude-3-7-sonnet"],
            framework=["swe-agent", "mini-swe-agent", "openhands"],
            trajectories_per_framework=4,
            min_isl=8192,
            max_isl=65536,
            min_turns=5,
            concurrency=[1, 2, 4],
            max_output_tokens=2048,
            warmup_turns=4,
            replay_mode="checkpoints",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "comparison": {"model": "model-uri"},
                        "arms": [
                            {
                                "label": "vllm",
                                "engine": "vllm",
                                "executable": "vllm",
                                "model": "org/model",
                                "context_size": 65536,
                                "max_concurrency": 4,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            engine_config = BENCH.load_engine_config(path)

            plan = BENCH.benchmark_plan(args, self.specs(), engine_config)

        self.assertEqual(
            [item["label"] for item in plan["order"]],
            ["rc8", "main", "vllm", "vllm", "main", "rc8"],
        )
        self.assertEqual(plan["workload"]["measured_requests_total"], 216)
        self.assertIn("--max-num-seqs", plan["external_server_commands"][0])

    def test_plan_order_uses_verified_engine_identity_when_supplied(self) -> None:
        args = SimpleNamespace(
            repo=REPO,
            backend="metal",
            model="model-uri",
            passes=1,
            source_dataset=["swe-smith-claude-3-7-sonnet"],
            framework=["swe-agent", "mini-swe-agent", "openhands"],
            trajectories_per_framework=4,
            min_isl=8192,
            max_isl=65536,
            min_turns=5,
            concurrency=[1, 2, 4],
            max_output_tokens=2048,
            warmup_turns=4,
            replay_mode="checkpoints",
        )
        external_builds = [
            {
                "label": "rc8",
                "engine": "mesh",
                "version_sha256": "mesh-hash",
            },
            {
                "label": "vllm",
                "engine": "vllm",
                "version_sha256": "c" * 64,
            },
        ]
        verified = BENCH.verified_version_sha256_by_label(external_builds)
        self.assertEqual(verified, {"vllm": "c" * 64})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "comparison": {"model": "model-uri"},
                        "arms": [
                            {
                                "label": "vllm",
                                "engine": "vllm",
                                "executable": "vllm",
                                "model": "org/model",
                                "context_size": 65536,
                                "max_concurrency": 4,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            engine_config = BENCH.load_engine_config(path)

            plan = BENCH.benchmark_plan(
                args, self.specs(), engine_config, version_sha256_by_label=verified
            )

        self.assertEqual(
            [item["commit"] for item in plan["order"]],
            ["a" * 40, "b" * 40, "c" * 64],
        )
        self.assertEqual(
            [spec.commit for spec in BENCH.combined_specs(self.specs(), None)],
            ["a" * 40, "b" * 40],
        )

    def test_build_resume_identity_compares_external_version_hashes(self) -> None:
        mesh = {
            "label": "rc8",
            "engine": "mesh",
            "commit": "a" * 40,
            "binary_sha256": "x" * 64,
            "runtime_sha256": "y" * 64,
        }
        external = {
            "label": "vllm",
            "engine": "vllm",
            "version_sha256": "c" * 64,
        }

        self.assertEqual(
            BENCH.build_resume_identity(mesh),
            ("a" * 40, "x" * 64, "y" * 64),
        )
        self.assertEqual(
            BENCH.build_resume_identity(external),
            ("c" * 64,),
        )

    def test_resume_mismatch_message_reports_arm_identity_not_binary_claims(
        self,
    ) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("engine version_sha256 changed", source)
        self.assertNotIn("built binary or runtime hashes differ", source)

    def test_report_escapes_backticks_and_pipes_in_version_cells(self) -> None:
        self.assertEqual(
            BENCH.markdown_code_cell("v`0`|x"),
            "<code>v&#96;0&#96;&#124;x</code>",
        )
        self.assertEqual(
            BENCH.markdown_code_cell("a\\|b"),
            "<code>a\\&#124;b</code>",
        )
        self.assertEqual(
            BENCH.markdown_code_cell("multi\r\nline"),
            "<code>multi  line</code>",
        )

    def test_command_for_build_launches_the_preflight_resolved_executable(
        self,
    ) -> None:
        build = {
            "engine": "vllm",
            "worktree": "/engine",
            "external_engine": {
                "label": "vllm",
                "engine": "vllm",
                "executable": "vllm",
                "model": "org/model",
                "served_model": "org/model",
                "context_size": 65536,
                "max_concurrency": 4,
                "tokenizer": None,
                "hf_config": None,
                "prefix_cache": True,
                "batch_size": 2048,
                "ubatch_size": 512,
                "extra_args": [],
                "cwd": "/engine",
            },
            "provenance": {
                "resolved_executable": "/opt/vllm/bin/vllm-resolved",
            },
        }

        command = BENCH.command_for_build(build, "model-uri")

        self.assertEqual(command[0], "/opt/vllm/bin/vllm-resolved")

    def test_external_config_reuses_the_validated_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "comparison": {"model": "model-uri"},
                        "arms": [
                            {
                                "label": "vllm",
                                "engine": "vllm",
                                "executable": "vllm",
                                "model": "org/model",
                                "context_size": 4096,
                                "max_concurrency": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            snapshot = BENCH.load_engine_config(path)
            args = SimpleNamespace(
                engine_config=path, engine_config_snapshot=snapshot
            )

            with mock.patch.object(
                BENCH,
                "load_engine_config",
                side_effect=AssertionError(
                    "validated snapshot must be reused, not reloaded"
                ),
            ):
                config = BENCH.external_config(args)

            self.assertIs(config, snapshot)

    def test_external_config_without_snapshot_loads_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "comparison": {"model": "model-uri"},
                        "arms": [
                            {
                                "label": "vllm",
                                "engine": "vllm",
                                "executable": "vllm",
                                "model": "org/model",
                                "context_size": 4096,
                                "max_concurrency": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(engine_config=path)

            config = BENCH.external_config(args)

            self.assertEqual(config.path, path.resolve())
            self.assertEqual(config.arms[0].label, "vllm")


if __name__ == "__main__":
    unittest.main()
