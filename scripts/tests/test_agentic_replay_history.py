"""Unit tests for the agentic replay nightly history normalizer and gate."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/agentic-replay-history.py"


def load_module():
    spec = importlib.util.spec_from_file_location("agentic_replay_history", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


history = load_module()


def make_row(
    created: str,
    model: str = "qwen25-coder-14b",
    decode: float = 70.0,
    ttft: float = 1000.0,
    complete: bool = True,
) -> dict:
    return {
        "schema_version": 2,
        "created_utc": created,
        "source_sha": "a" * 40,
        "cohort_key": "k" * 64,
        "cohort": {"model": model, "concurrency": 4},
        "backend_binary_sha256": None,
        "hardware_fingerprint": {},
        "model": {"family": model, "quant": "Q4_K_M", "repo": "r", "revision": "b" * 40, "file": "f", "sha256": "c" * 64, "class": "dense"},
        "replay": {"mode": "checkpoint", "dataset": "d", "dataset_revision": "d" * 40, "dataset_file": "sessions.parquet", "dataset_sha256": "e" * 64, "trajectories_per_framework": 4, "passes": 2, "warmup_turns": 4, "concurrency": 4, "max_output_tokens": 2048},
        "prompt_count": 36,
        "successful_requests": 36 if complete else 30,
        "failed_requests": 0 if complete else 6,
        "output_tokens": 100000,
        "measured_wall_ms": 1000.0,
        "decode_tokens_per_second": decode,
        "end_to_end_tokens_per_second": decode * 0.8,
        "ttft_ms_mean": ttft,
        "ttft_ms_p90": ttft * 2,
        "cache_hit_pct": 70.0,
        "finish_reason_length_pct": 6.0,
        "complete": complete,
        "artifact_result": "ok" if complete else "incomplete",
    }


class GateTests(unittest.TestCase):
    @staticmethod
    def build_rows(cells: list[dict]) -> list[dict]:
        return history.build_rows(
            cells,
            model=make_row("2026-09-09")["model"],
            replay=make_row("2026-09-09")["replay"],
            hardware={"machine_model": "Mac15,14"},
            source_sha="a" * 40,
            backend_binary_sha256="b" * 64,
        )

    def test_history_rejects_unpinned_model(self):
        model = make_row("2026-09-09")["model"]
        model["sha256"] = None
        with self.assertRaisesRegex(ValueError, "sha256 must be a 64-hex digest"):
            history.build_rows(
                [],
                model=model,
                replay={},
                hardware={},
                source_sha="a" * 40,
                backend_binary_sha256=None,
            )

    def test_bootstrap_does_not_gate(self):
        baseline = [make_row(f"2026-09-0{d}") for d in (1, 2)]
        candidate = make_row("2026-09-09", decode=10.0)
        self.assertEqual(history.compare(candidate, baseline), [])

    def test_regression_after_bootstrap_fails(self):
        baseline = [make_row(f"2026-09-0{d}") for d in (1, 2, 3)]
        candidate = make_row("2026-09-09", decode=50.0)  # -28%
        problems = history.compare(candidate, baseline)
        self.assertTrue(any("decode_tokens_per_second" in p for p in problems), problems)

    def test_small_drift_within_tolerance_passes(self):
        baseline = [make_row(f"2026-09-0{d}") for d in (1, 2, 3)]
        candidate = make_row("2026-09-09", decode=68.0)  # -2.9%
        self.assertEqual(history.compare(candidate, baseline), [])

    def test_incomplete_run_always_flags(self):
        problems = history.compare(make_row("2026-09-09", complete=False), [])
        self.assertTrue(problems)

    def test_finish_reason_length_regression_is_gateable(self):
        baseline = [make_row(f"2026-09-0{d}") for d in (1, 2, 3)]
        for row in baseline:
            row["finish_reason_length_pct"] = 0.0
        candidate = make_row("2026-09-09")
        candidate["finish_reason_length_pct"] = 5.0

        problems = history.compare(candidate, baseline)

        self.assertTrue(any("finish_reason_length_pct" in p for p in problems), problems)

    def test_build_rows_counts_terminal_length_finishes(self):
        rows = self.build_rows(
            [
                {
                    "_mtime": history.dt.datetime(2026, 9, 9),
                    "concurrency": 4,
                    "requests": 4,
                    "successful_requests": 4,
                    "failed_requests": 0,
                    "completion_tokens": 40,
                    "generation_seconds": 2.0,
                    "workload_window_seconds": 3.0,
                    "finish_reason_length_requests": 1,
                }
            ]
        )

        self.assertEqual(rows[0]["finish_reason_length_pct"], 25.0)
        self.assertTrue(rows[0]["complete"])

    def test_dropped_requests_make_row_incomplete(self):
        rows = self.build_rows(
            [
                {
                    "_mtime": history.dt.datetime(2026, 9, 9),
                    "concurrency": 4,
                    "requests": 4,
                    "successful_requests": 3,
                    "failed_requests": 0,
                    "completion_tokens": 30,
                    "generation_seconds": 2.0,
                    "workload_window_seconds": 3.0,
                }
            ]
        )

        self.assertFalse(rows[0]["complete"])
        self.assertEqual(rows[0]["artifact_result"], "incomplete")

    def test_replay_requires_pinned_dataset_identity(self):
        replay = make_row("2026-09-09")["replay"]
        replay["dataset_sha256"] = None

        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            history.validate_replay_pin(replay)

    def test_baseline_replay_identity_includes_dataset(self):
        baseline = [make_row(f"2026-09-0{day}") for day in (1, 2, 3)]
        candidate = make_row("2026-09-09", decode=1.0)
        candidate["replay"] = dict(candidate["replay"], dataset_sha256="f" * 64)

        eligible = [row for row in baseline if row["replay"] == candidate["replay"]]

        self.assertEqual(history.compare(candidate, eligible), [])

    def test_bootstrap_incomplete_row_fails_without_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_dir = root / "replay"
            cell_dir = replay_dir / "qwen25-coder-14b/data/pass-1/main"
            cell_dir.mkdir(parents=True)
            (cell_dir / "c-4.json").write_text(
                json.dumps(
                    {
                        "concurrency": 4,
                        "requests": 4,
                        "successful_requests": 3,
                        "failed_requests": 0,
                    }
                )
            )
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"models": [make_row("2026-09-09")["model"]]}))
            replay = root / "replay.json"
            replay.write_text(json.dumps(make_row("2026-09-09")["replay"]))
            hardware = root / "hardware.json"
            hardware.write_text(json.dumps({"machine_model": "Mac15,14"}))

            result = history.main(
                [
                    "--matrix",
                    str(matrix),
                    "--replay-dir",
                    str(replay_dir),
                    "--label",
                    "main",
                    "--hardware",
                    str(hardware),
                    "--source-sha",
                    "a" * 40,
                    "--replay",
                    str(replay),
                    "--output",
                    str(root / "history.jsonl"),
                ]
            )

        self.assertEqual(result, 1)

    def test_missing_family_preserves_complete_family_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_dir = root / "replay"
            cell_dir = replay_dir / "qwen25-coder-14b/data/pass-1/main"
            cell_dir.mkdir(parents=True)
            (cell_dir / "c-4.json").write_text(
                json.dumps(
                    {
                        "concurrency": 4,
                        "requests": 4,
                        "successful_requests": 4,
                        "failed_requests": 0,
                    }
                )
            )
            present = make_row("2026-09-09")["model"]
            missing = dict(present, family="gpt-oss-20b")
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"models": [present, missing]}))
            replay = root / "replay.json"
            replay.write_text(json.dumps(make_row("2026-09-09")["replay"]))
            hardware = root / "hardware.json"
            hardware.write_text(json.dumps({"machine_model": "Mac15,14"}))
            output = root / "history.jsonl"

            result = history.main(
                [
                    "--matrix",
                    str(matrix),
                    "--replay-dir",
                    str(replay_dir),
                    "--label",
                    "main",
                    "--hardware",
                    str(hardware),
                    "--source-sha",
                    "a" * 40,
                    "--replay",
                    str(replay),
                    "--output",
                    str(output),
                ]
            )
            rows = output.read_text().splitlines()

        self.assertEqual(result, 0)
        self.assertEqual(len(rows), 1)

    def test_baseline_with_other_model_sha_does_not_count_toward_bootstrap(self):
        other = make_row("2026-09-01")
        other["model"] = dict(other["model"], sha256="b" * 64)
        baseline = [other, make_row("2026-09-02"), make_row("2026-09-03")]
        # Only rows for the same model identity count; two matching rows is
        # still bootstrap, so a big drop must not gate.
        candidate = make_row("2026-09-09", decode=1.0)
        self.assertEqual(history.compare(candidate, baseline), [])

    def test_zero_baseline_median_uses_absolute_tolerance_only(self):
        baseline = [
            make_row(f"2026-09-0{d}", ttft=0.0) for d in (1, 2, 3)
        ]
        candidate = make_row("2026-09-09", ttft=0.0)
        # Zero candidate against zero baseline: no percentage blow-up.
        self.assertEqual(history.compare(candidate, baseline), [])

    def test_incomplete_model_does_not_mask_regression_of_complete_model(self):
        baseline = [make_row(f"2026-09-0{d}") for d in (1, 2, 3)]
        incomplete = make_row("2026-09-09", model="gpt-oss-20b", complete=False)
        regressed = make_row("2026-09-09", decode=1.0)
        problems = history.compare(incomplete, []) + history.compare(
            regressed,
            [r for r in baseline if r["cohort"]["model"] == regressed["cohort"]["model"]],
        )
        self.assertTrue(any("incomplete" in p for p in problems), problems)
        self.assertTrue(any("decode_tokens_per_second" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
