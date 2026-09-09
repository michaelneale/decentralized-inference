from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "evals/skippy-competitive-benchmark.py"
CONFIG = REPO / "evals/skippy-competitive-benchmark.json"


def load_module():
    spec = importlib.util.spec_from_file_location("skippy_competitive_benchmark", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module()


class CompetitiveBenchmarkTest(unittest.TestCase):
    def test_checked_in_plan_covers_both_platforms_all_models_and_full_ladder(self) -> None:
        config = BENCH.load_config(CONFIG)
        plan = BENCH.build_plan(
            config,
            ["cuda", "metal"],
            config["models"],
            ["synthetic", "thoughtworks"],
        )

        self.assertEqual(plan["cell_count"], 576)
        self.assertEqual(
            sorted({cell["concurrency"] for cell in plan["cells"]}),
            [1, 2, 4, 8, 16, 32, 64, 128, 256],
        )
        self.assertEqual(
            sorted({cell["platform"] for cell in plan["cells"]}),
            ["cuda", "metal"],
        )
        self.assertEqual(len({cell["model"] for cell in plan["cells"]}), 4)
        trace = [cell for cell in plan["cells"] if cell["workload"] == "thoughtworks"]
        self.assertTrue(all(cell["prompt_count"] % cell["concurrency"] == 0 for cell in trace))
        self.assertTrue(all(cell["prompt_count"] >= cell["concurrency"] for cell in trace))
        dense_trace = [cell for cell in trace if cell["model"] == "llama32-dense"]
        self.assertTrue(all(cell["context_size"] == 131072 for cell in dense_trace))
        self.assertTrue(all(cell["active_lanes"] == 16 for cell in dense_trace))
        moe_trace = [cell for cell in trace if cell["model"] == "deepseek-v2-moe"]
        self.assertTrue(all(cell["context_size"] == 16384 for cell in moe_trace))
        self.assertTrue(all(cell["active_lanes"] == 8 for cell in moe_trace))
        recurrent_trace = [
            cell for cell in trace if cell["model"] == "falcon-h1-recurrent"
        ]
        self.assertTrue(all(cell["context_size"] == 16384 for cell in recurrent_trace))
        self.assertTrue(all(cell["active_lanes"] == 2 for cell in recurrent_trace))
        hybrid_trace = [
            cell for cell in trace if cell["model"] == "granite-h1-hybrid"
        ]
        self.assertTrue(all(cell["context_size"] == 131072 for cell in hybrid_trace))
        self.assertTrue(all(cell["active_lanes"] == 16 for cell in hybrid_trace))

    def test_thoughtworks_runtime_shape_allows_per_model_override(self) -> None:
        thoughtworks = {"context_size": 16384, "active_lanes": 2}

        self.assertEqual(
            BENCH.thoughtworks_runtime_shape(thoughtworks, {}), (16384, 2)
        )
        self.assertEqual(
            BENCH.thoughtworks_runtime_shape(
                thoughtworks,
                {
                    "thoughtworks_context_size": 131072,
                    "thoughtworks_active_lanes": 16,
                },
            ),
            (131072, 16),
        )

    def test_mesh_stage_uses_native_sequence_budget_as_resident_ceiling(self) -> None:
        config = BENCH.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage.json"
            BENCH.write_stage_config(
                path,
                config["models"][0],
                Path(directory) / "model.gguf",
                8080,
                131072,
                16,
                True,
            )

            stage = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(stage["kv_cache"]["max_entries"], 512)

    def test_config_rejects_nonpositive_model_trace_shape(self) -> None:
        document = json.loads(CONFIG.read_text(encoding="utf-8"))
        document["models"][0]["thoughtworks_active_lanes"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive thoughtworks_active_lanes"):
                BENCH.load_config(path)

    def test_config_rejects_boolean_model_trace_shape(self) -> None:
        document = json.loads(CONFIG.read_text(encoding="utf-8"))
        document["models"][0]["thoughtworks_active_lanes"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive thoughtworks_active_lanes"):
                BENCH.load_config(path)

    def test_config_requires_a_reason_for_comparison_exclusions(self) -> None:
        document = json.loads(CONFIG.read_text(encoding="utf-8"))
        document["models"][0]["comparison_support"] = {
            "vllm": {"available": False}
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "needs an exclusion reason"):
                BENCH.load_config(path)

    def test_trace_arm_order_alternates_to_reduce_time_order_bias(self) -> None:
        config = BENCH.load_config(CONFIG)
        plan = BENCH.build_plan(
            config,
            ["metal"],
            [config["models"][0]],
            ["thoughtworks"],
        )
        cells = plan["cells"]

        self.assertEqual([cell["arm"] for cell in cells[:4]], ["llama", "mesh", "mesh", "llama"])

    def test_manifest_verification_rejects_provenance_drift(self) -> None:
        config = BENCH.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "prompts.json"
            document = {
                "metadata": {"rows": [{"session_id": "wrong"}]},
                "prompts": [
                    {"family": "fixture", "prompt": "trace"}
                    for _ in range(256)
                ],
            }
            manifest.write_text(json.dumps(document), encoding="utf-8")
            config["thoughtworks"]["selection"]["manifest_sha256"] = BENCH.sha256(manifest)

            with self.assertRaisesRegex(RuntimeError, "row provenance"):
                BENCH.verify_manifest(manifest, config)

    def test_server_commands_keep_raw_and_mesh_lane_counts_equal(self) -> None:
        config = BENCH.load_config(CONFIG)
        model = config["models"][0]
        args = SimpleNamespace(mesh_binary=Path("mesh"), llama_binary=Path("llama"))
        common = (model, Path("model.gguf"), Path("stage.json"), 19000, 16384, 2, 8)

        mesh = BENCH.server_command("mesh", args, *common, True)
        raw = BENCH.server_command("llama", args, *common, True)

        self.assertEqual(mesh[mesh.index("--generation-concurrency") + 1], "2")
        self.assertEqual(mesh[mesh.index("--generation-queue-capacity") + 1], "256")
        self.assertEqual(
            mesh[mesh.index("--generation-admission-timeout-secs") + 1], "600"
        )
        self.assertEqual(raw[raw.index("--parallel") + 1], "2")
        self.assertIn("--kv-unified", raw)
        self.assertNotIn("--no-cache-prompt", raw)

    def test_adaptive_mesh_command_keeps_fixed_lane_ceiling(self) -> None:
        config = BENCH.load_config(CONFIG)
        model = config["models"][0]
        args = SimpleNamespace(
            mesh_binary=Path("mesh"),
            llama_binary=Path("llama"),
            tokenizer_root=Path("tokenizers"),
        )
        command = BENCH.server_command(
            "mesh-adaptive",
            args,
            model,
            Path("model.gguf"),
            Path("stage.json"),
            19000,
            16384,
            4,
            8,
            True,
        )

        self.assertEqual(command[command.index("--generation-concurrency") + 1], "4")
        self.assertIn("--adaptive-generation-concurrency", command)
        self.assertEqual(
            command[
                command.index("--adaptive-generation-min-concurrency") + 1
            ],
            "1",
        )

    def test_optional_comparisons_skip_cleanly_off_linux_cuda(self) -> None:
        config = BENCH.load_config(CONFIG)
        args = SimpleNamespace(
            comparison_backend=["vllm", "sglang"],
            platform="metal",
            vllm_binary=None,
            sglang_python=None,
            sglang_model_root=None,
        )

        status = BENCH.resolve_optional_comparisons(args, config["models"])

        self.assertFalse(status["vllm"]["available"])
        self.assertFalse(status["sglang"]["available"])
        self.assertIn("Linux CUDA", status["vllm"]["reason"])

    def test_model_capability_exclusions_remove_only_unsupported_arms(self) -> None:
        config = BENCH.load_config(CONFIG)
        dense, moe = config["models"][:2]
        args = SimpleNamespace(
            comparison_backend=["vllm", "sglang"],
            mesh_adaptive=False,
        )
        comparisons = {
            arm: {
                "available": True,
                "models": {
                    dense["key"]: {"available": True},
                    moe["key"]: {"available": False},
                },
            }
            for arm in args.comparison_backend
        }

        self.assertEqual(
            BENCH.arms_for_model(args, dense, comparisons),
            ("llama", "mesh", "vllm", "sglang"),
        )
        self.assertEqual(
            BENCH.arms_for_model(args, moe, comparisons),
            ("llama", "mesh"),
        )

    def test_required_backends_allow_only_pinned_capability_exclusions(self) -> None:
        comparisons = {
            "vllm": {
                "available": True,
                "models": {
                    "dense": {"available": True},
                    "moe": {
                        "available": False,
                        "source": "pinned-capability-exclusion",
                        "reason": "unsupported exact input",
                    },
                },
            },
            "sglang": {
                "available": True,
                "models": {
                    "dense": {"available": True},
                    "recurrent": {
                        "available": False,
                        "reason": "model path not found",
                    },
                },
            },
        }

        self.assertEqual(
            BENCH.required_comparison_errors(["vllm"], comparisons), []
        )
        self.assertEqual(
            BENCH.required_comparison_errors(["sglang"], comparisons),
            ["sglang: missing model inputs for recurrent"],
        )

    def test_optional_backends_can_match_unified_total_kv_capacity(self) -> None:
        config = BENCH.load_config(CONFIG)
        model = config["models"][0]
        common = {
            "tokenizer_root": Path("tokenizers"),
            "capacity_match_comparison_kv": True,
        }
        vllm = BENCH.server_command(
            "vllm",
            SimpleNamespace(
                **common,
                vllm_binary=Path("vllm"),
                vllm_hf_config_root=Path("hf-config"),
            ),
            model,
            Path("model.gguf"),
            Path("stage.json"),
            19000,
            131072,
            16,
            8,
            True,
        )
        sglang = BENCH.server_command(
            "sglang",
            SimpleNamespace(
                **common,
                sglang_python=Path("sglang-python"),
                sglang_model_root=None,
                model_root=Path("models"),
            ),
            model,
            Path("model.gguf"),
            Path("stage.json"),
            19000,
            131072,
            16,
            8,
            True,
        )

        self.assertEqual(vllm[vllm.index("--block-size") + 1], "16")
        self.assertEqual(
            vllm[vllm.index("--num-gpu-blocks-override") + 1], "8192"
        )
        self.assertEqual(
            sglang[sglang.index("--max-total-tokens") + 1], "131072"
        )

        granite = config["models"][-1]
        granite_vllm = BENCH.server_command(
            "vllm",
            SimpleNamespace(
                **common,
                vllm_binary=Path("vllm"),
                vllm_hf_config_root=Path("hf-config"),
            ),
            granite,
            Path("granite.gguf"),
            Path("stage.json"),
            19000,
            131072,
            16,
            8,
            True,
        )
        self.assertEqual(
            granite_vllm[
                granite_vllm.index("--num-gpu-blocks-override") + 1
            ],
            "2560",
        )

    def test_sglang_defaults_to_the_pinned_gguf_and_tokenizer(self) -> None:
        config = BENCH.load_config(CONFIG)
        model = config["models"][0]
        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory) / "models"
            gguf = model_root / model["key"] / model["filename"]
            gguf.parent.mkdir(parents=True)
            gguf.write_bytes(b"fixture")
            args = SimpleNamespace(
                sglang_python=Path("sglang-python"),
                sglang_model_root=None,
                model_root=model_root,
                tokenizer_root=Path("tokenizers"),
            )

            command = BENCH.server_command(
                "sglang",
                args,
                model,
                gguf,
                Path("stage.json"),
                19000,
                16384,
                4,
                8,
                True,
            )
            no_cache_command = BENCH.server_command(
                "sglang",
                args,
                model,
                gguf,
                Path("stage.json"),
                19000,
                16384,
                4,
                8,
                False,
            )

        self.assertEqual(command[command.index("--model-path") + 1], str(gguf))
        self.assertEqual(
            command[command.index("--tokenizer-path") + 1],
            str(Path("tokenizers") / model["key"]),
        )
        self.assertEqual(command[command.index("--max-running-requests") + 1], "4")
        self.assertEqual(command[command.index("--load-format") + 1], "gguf")
        self.assertEqual(command[command.index("--quantization") + 1], "gguf")
        served_name = command[command.index("--served-model-name") + 1]
        self.assertNotIn(":", served_name)
        self.assertEqual(BENCH.served_model_id("sglang", model), served_name)
        self.assertNotIn("--disable-radix-cache", command)
        self.assertIn("--disable-radix-cache", no_cache_command)

    def test_vllm_uses_the_verified_config_for_the_pinned_gguf(self) -> None:
        config = BENCH.load_config(CONFIG)
        model = config["models"][0]
        args = SimpleNamespace(
            vllm_binary=Path("vllm"),
            tokenizer_root=Path("tokenizers"),
            vllm_hf_config_root=Path("hf-configs"),
        )

        command = BENCH.server_command(
            "vllm",
            args,
            model,
            Path("model.gguf"),
            Path("stage.json"),
            19000,
            16384,
            4,
            8,
            True,
        )

        self.assertEqual(
            command[command.index("--hf-config-path") + 1],
            str(Path("hf-configs") / model["key"]),
        )
        self.assertEqual(command[command.index("--load-format") + 1], "gguf")
        self.assertEqual(command[command.index("--quantization") + 1], "gguf")
        self.assertIn("--enable-prefix-caching", command)

        no_cache_command = BENCH.server_command(
            "vllm",
            args,
            model,
            Path("model.gguf"),
            Path("stage.json"),
            19000,
            16384,
            4,
            8,
            False,
        )
        self.assertIn("--no-enable-prefix-caching", no_cache_command)
        self.assertNotIn("--enable-prefix-caching", no_cache_command)

    def test_vllm_native_override_omits_gguf_loader_flags(self) -> None:
        config = BENCH.load_config(CONFIG)
        model = config["models"][-1]
        args = SimpleNamespace(
            vllm_binary=Path("vllm"),
            vllm_model_root=Path("native-models"),
            tokenizer_root=Path("tokenizers"),
            vllm_hf_config_root=Path("hf-configs"),
        )

        command = BENCH.server_command(
            "vllm",
            args,
            model,
            Path("baseline.gguf"),
            Path("stage.json"),
            19000,
            131072,
            16,
            8,
            True,
        )

        self.assertEqual(command[2], str(Path("native-models") / model["key"]))
        self.assertNotIn("--load-format", command)
        self.assertNotIn("--quantization", command)

    def test_alternate_container_is_fail_closed_against_pinned_hash(self) -> None:
        config = BENCH.load_config(CONFIG)
        model = dict(config["models"][-1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "native-models"
            model_dir = root / model["key"]
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            actual = BENCH.directory_sha256(model_dir)
            model["comparison_inputs"] = {
                "sglang": {
                    **model["comparison_inputs"]["sglang"],
                    "sha256": actual,
                }
            }
            args = SimpleNamespace(
                sglang_model_root=root,
                model_root=Path("baseline-models"),
            )

            verified = BENCH.comparison_model_input(args, model, "sglang")
            (model_dir / "config.json").write_text("drift", encoding="utf-8")
            drifted = BENCH.comparison_model_input(args, model, "sglang")

        self.assertTrue(verified["available"])
        self.assertEqual(verified["source"], "pinned-alternate-container")
        self.assertFalse(drifted["available"])
        self.assertIn("SHA-256 mismatch", drifted["reason"])

    def test_synthetic_benchy_command_is_fail_closed_and_sglang_compatible(self) -> None:
        config = BENCH.load_config(CONFIG)
        model = config["models"][0]
        args = SimpleNamespace(
            benchy=Path("llama-benchy"), tokenizer_root=Path("tokenizers")
        )

        sglang = BENCH.synthetic_benchy_common_command(
            args,
            model,
            "sglang",
            BENCH.served_model_id("sglang", model),
            19000,
            config["synthetic"],
        )
        raw = BENCH.synthetic_benchy_common_command(
            args,
            model,
            "llama",
            model["model_id"],
            19000,
            config["synthetic"],
        )
        vllm = BENCH.synthetic_benchy_common_command(
            args,
            model,
            "vllm",
            model["model_id"],
            19000,
            config["synthetic"],
        )

        self.assertIn("--exit-on-first-fail", sglang)
        self.assertIn("--no-results-on-fail", sglang)
        sglang_extra = sglang[sglang.index("--extra-body") + 1]
        vllm_extra = vllm[vllm.index("--extra-body") + 1]
        raw_extra = raw[raw.index("--extra-body") + 1]
        self.assertIn("return_token_ids=false", sglang_extra)
        self.assertIn("return_token_ids=false", vllm_extra)
        self.assertNotIn("return_token_ids", raw_extra)

    def test_synthetic_cell_validation_rejects_hidden_request_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "tg-64-c-2"
            stem.with_name(stem.name + "-progress.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "request_end",
                                "error": "",
                                "total_tokens": 64,
                            }
                        ),
                        json.dumps(
                            {
                                "type": "request_end",
                                "error": "HTTP 400",
                                "total_tokens": 0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            stem.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "response_size": 64,
                                "tg_throughput": {"mean": None},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "1 failures"):
                BENCH.validate_synthetic_cell(stem, 2, 64)

    def test_synthetic_cell_validation_rejects_null_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "tg-64-c-2"
            stem.with_name(stem.name + "-progress.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "type": "request_end",
                            "error": "",
                            "total_tokens": 64,
                        }
                    )
                    for _ in range(2)
                )
                + "\n",
                encoding="utf-8",
            )
            stem.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "response_size": 64,
                                "tg_throughput": {"mean": None},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "invalid generation throughput"):
                BENCH.validate_synthetic_cell(stem, 2, 64)

    def test_synthetic_cell_validation_accepts_complete_finite_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "tg-64-c-2"
            stem.with_name(stem.name + "-progress.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "type": "request_end",
                            "error": "",
                            "total_tokens": 64,
                        }
                    )
                    for _ in range(2)
                )
                + "\n",
                encoding="utf-8",
            )
            stem.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "response_size": 64,
                                "tg_throughput": {"mean": 123.5},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            BENCH.validate_synthetic_cell(stem, 2, 64)

    def test_controlled_run_can_require_all_comparison_backends(self) -> None:
        args = BENCH.parse_args(
            [
                "run",
                "--platform",
                "cuda",
                "--output",
                "artifact",
                "--model-root",
                "models",
                "--tokenizer-root",
                "tokenizers",
                "--manifest",
                "manifest.json",
                "--mesh-root",
                "mesh",
                "--mesh-binary",
                "skippy-server",
                "--native-dir",
                "native",
                "--llama-root",
                "llama",
                "--llama-binary",
                "llama-server",
                "--benchy",
                "llama-benchy",
                "--comparison-backend",
                "vllm",
                "--comparison-backend",
                "sglang",
                "--require-comparison-backends",
                "--capacity-match-comparison-kv",
            ]
        )

        self.assertEqual(args.comparison_backend, ["vllm", "sglang"])
        self.assertTrue(args.require_comparison_backends)
        self.assertTrue(args.capacity_match_comparison_kv)

    def test_parity_rejects_empty_or_short_success_responses(self) -> None:
        empty = {
            "status": 200,
            "content": "",
            "content_sha256": BENCH.hashlib.sha256(b"").hexdigest(),
            "completion_tokens": 0,
            "requested_completion_tokens": 32,
            "error": None,
        }
        short = {
            **empty,
            "content": "partial",
            "completion_tokens": 31,
        }
        valid = {
            **empty,
            "content": "complete",
            "completion_tokens": 32,
        }

        self.assertFalse(BENCH.parity_result_valid(empty))
        self.assertFalse(BENCH.parity_result_valid(short))
        self.assertTrue(BENCH.parity_result_valid(valid))

    def test_loaders_ignore_results_without_matching_completion_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm_dir = root / "data" / "cuda" / "model" / "mesh"
            arm_dir.mkdir(parents=True)
            (arm_dir / "tg-8-c-1.json").write_text("{}", encoding="utf-8")
            trace_dir = root / "trace" / "cuda" / "model" / "c-1" / "mesh"
            trace_dir.mkdir(parents=True)
            (trace_dir / "result.json").write_text("{}", encoding="utf-8")

            self.assertEqual(BENCH.load_synthetic_rows(root), [])
            self.assertEqual(BENCH.load_trace_rows(root), [])

    def test_forced_rerun_quarantines_the_previous_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "data" / "cuda" / "model" / "mesh"
            output_dir.mkdir(parents=True)
            (output_dir / "result.json").write_text("stale", encoding="utf-8")

            BENCH.quarantine_cell(output_dir, root)

            self.assertFalse(output_dir.exists())
            quarantined = list((root / "quarantine").glob("*/data/cuda/model/mesh/result.json"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].read_text(encoding="utf-8"), "stale")

    def test_report_writes_csv_svg_markdown_and_hash_manifest(self) -> None:
        config = BENCH.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, throughput in (
                ("llama", 100.0),
                ("mesh", 120.0),
                ("vllm", 130.0),
                ("sglang", 110.0),
            ):
                arm_dir = root / "data" / "metal" / "llama32-dense" / arm
                arm_dir.mkdir(parents=True)
                (arm_dir / "tg-8-c-1.json").write_text(
                    json.dumps(
                        {
                            "benchmarks": [
                                {
                                    "tg_throughput": {"mean": throughput},
                                    "e2e_ttft": {"mean": 10.0},
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                (arm_dir / "tg-8-c-1-progress.jsonl").write_text(
                    json.dumps({"type": "request_end", "error": None}) + "\n",
                    encoding="utf-8",
                )
                (arm_dir / "tg-8-c-1.out").write_text("", encoding="utf-8")
                (arm_dir / "status.tsv").write_text(
                    "tg\tconcurrency\texit_code\n8\t1\t0\n", encoding="utf-8"
                )
                (arm_dir / "parity.json").write_text(
                    json.dumps(
                        {
                            "cells": [
                                {
                                    "concurrency": 1,
                                    "results": [
                                        {
                                            "request_index": 0,
                                            "status": 200,
                                            "content": "same",
                                            "content_sha256": "same",
                                            "completion_tokens": 32,
                                            "requested_completion_tokens": 32,
                                            "error": None,
                                        }
                                    ],
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                synthetic_cell = {
                    "platform": "metal",
                    "model": "llama32-dense",
                    "arm": arm,
                    "workload": "synthetic",
                }
                (arm_dir / "complete.json").write_text(
                    json.dumps(
                        {
                            "cell": synthetic_cell,
                            "cell_sha256": BENCH.stable_hash(synthetic_cell),
                        }
                    ),
                    encoding="utf-8",
                )
                trace_dir = root / "trace" / "metal" / "llama32-dense" / "c-1" / arm
                trace_dir.mkdir(parents=True)
                (trace_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "platform": "metal",
                            "model": "llama32-dense",
                            "arm": arm,
                            "concurrency": 1,
                            "successful_requests": 60,
                            "failed_requests": 0,
                            "output_tokens_per_second": throughput,
                        }
                    ),
                    encoding="utf-8",
                )
                trace_cell = {
                    "platform": "metal",
                    "model": "llama32-dense",
                    "arm": arm,
                    "workload": "thoughtworks",
                    "concurrency": 1,
                }
                (trace_dir / "complete.json").write_text(
                    json.dumps(
                        {
                            "cell": trace_cell,
                            "cell_sha256": BENCH.stable_hash(trace_cell),
                        }
                    ),
                    encoding="utf-8",
                )

            provenance = root / "provenance"
            provenance.mkdir()
            (provenance / "metal.json").write_text(
                json.dumps({"capacity_match_comparison_kv": True}),
                encoding="utf-8",
            )

            BENCH.report(argparse.Namespace(artifact=root), config)

            report = (root / "summary" / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Mesh correctness gate: **PASS**", report)
            self.assertIn("Capacity-matched comparison", report)
            self.assertIn("+20.00%", report)
            self.assertIn(
                "| raw llama.cpp tok/s | Mesh tok/s | vLLM tok/s | SGLang tok/s |",
                report,
            )
            self.assertTrue((root / "summary" / "synthetic.csv").is_file())
            self.assertTrue((root / "summary" / "thoughtworks.csv").is_file())
            charts = list((root / "summary" / "charts").glob("*.svg"))
            self.assertTrue(charts)
            throughput_chart = next(
                chart for chart in charts if chart.name.endswith("throughput.svg")
            )
            chart_text = throughput_chart.read_text(encoding="utf-8")
            self.assertIn("vLLM", chart_text)
            self.assertIn("SGLang", chart_text)
            self.assertTrue((root / "artifact-sha256.txt").is_file())


if __name__ == "__main__":
    unittest.main()
