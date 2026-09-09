#!/usr/bin/env python3
"""Normalize agentic-replay nightly artifacts and gate on cohort-matched drift.

Reads the per-concurrency cell JSON files that evals/agentic-replay.py writes
under ``<output>/<family>/data/pass-N/<label>/c-<concurrency>.json`` for the
candidate label, emits schema-version-2 JSONL history rows (one per model x
concurrency), and compares them against the append-only baseline in the HF
dataset checkout. Mirrors scripts/performance-history.py semantics: exact
cohort keys only, bootstrap-then-gate (three prior complete runs before a
regression fails).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
BASELINE_MIN_RUNS = 3
# Relative + absolute tolerance, both must be exceeded to fail (mirrors the
# reviewed-budget rule; tuned per metric after bootstrap evidence exists).
DEFAULT_TOLERANCE = {"pct": 5.0, "abs": 0.0}


def validate_model_pin(model: dict[str, Any]) -> None:
    """Reject mutable or unverifiable model entries before emitting history."""
    family = model.get("family", "unknown")
    revision = model.get("revision")
    digest = model.get("sha256")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{family}: revision must be an immutable 40-hex commit")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{family}: sha256 must be a 64-hex digest")


def validate_replay_pin(replay: dict[str, Any]) -> None:
    """Reject mutable or unverifiable trajectory inputs."""
    revision = replay.get("dataset_revision")
    digest = replay.get("dataset_sha256")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("dataset_revision must be an immutable 40-hex commit")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("dataset_sha256 must be a 64-hex digest")


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_cells(replay_root: Path, family: str, label: str) -> list[dict[str, Any]]:
    """Load every candidate-label cell for a family across passes."""
    cells: list[dict[str, Any]] = []
    pattern = f"{family}/data/pass-*/{label}/c-*.json"
    for path in sorted(replay_root.glob(pattern)):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object: {path}")
        value["_mtime"] = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC)
        cells.append(value)
    if not cells:
        raise FileNotFoundError(f"no replay cells matched {replay_root / pattern}")
    return cells


def build_rows(
    cells: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    replay: dict[str, Any],
    hardware: dict[str, Any],
    source_sha: str,
    backend_binary_sha256: str | None,
) -> list[dict[str, Any]]:
    """Group cells by concurrency and emit one history row per level."""
    validate_model_pin(model)
    validate_replay_pin(replay)
    by_concurrency: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_concurrency[int(cell["concurrency"])].append(cell)

    rows: list[dict[str, Any]] = []
    for concurrency in sorted(by_concurrency):
        group = sorted(by_concurrency[concurrency], key=lambda c: c["_mtime"])
        requests = sum(int(c["requests"]) for c in group)
        successful = sum(int(c["successful_requests"]) for c in group)
        failed = sum(int(c["failed_requests"]) for c in group)
        completion_tokens = sum(int(c.get("completion_tokens", 0)) for c in group)
        generation_seconds = sum(float(c.get("generation_seconds", 0.0)) for c in group)
        workload_seconds = sum(float(c.get("workload_window_seconds", 0.0)) for c in group)
        ttft_ms = sorted(
            1000.0 * s for c in group for s in c.get("ttft_samples", [])
        )
        cache_pcts = [c["cache_pct"] for c in group if c.get("cache_pct") is not None]
        length_finishes = sum(
            int(
                c.get(
                    "finish_reason_length_requests",
                    c.get("budget_exhausted_requests", 0),
                )
            )
            for c in group
        )
        newest = max(c["_mtime"] for c in group)

        def mean_or_none(values: list[float]) -> float | None:
            return statistics.fmean(values) if values else None

        cohort = {
            "model": model["family"],
            "quant": model["quant"],
            "concurrency": concurrency,
            "backend": "mesh",
            "runner": hardware["machine_model"],
        }
        replay_params = {k: v for k, v in replay.items() if k != "concurrency"} | {
            "concurrency": concurrency
        }
        complete = (
            requests > 0
            and failed == 0
            and successful + failed == requests
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "created_utc": newest.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_sha": source_sha,
                "cohort_key": stable_hash(
                    {
                        "cohort": cohort,
                        "source": source_sha,
                        "replay": replay_params,
                        "model_revision": model["revision"],
                        "model_sha": model.get("sha256"),
                    }
                ),
                "cohort": cohort,
                "backend_binary_sha256": backend_binary_sha256,
                "hardware_fingerprint": hardware,
                "model": model,
                "replay": replay_params,
                "prompt_count": requests,
                "successful_requests": successful,
                "failed_requests": failed,
                "output_tokens": completion_tokens,
                "measured_wall_ms": 1000.0 * workload_seconds,
                "decode_tokens_per_second": (
                    completion_tokens / generation_seconds
                    if generation_seconds > 0
                    else float(mean_or_none(
                        [c["decode_tokens_per_second"] for c in group
                         if c.get("decode_tokens_per_second") is not None]
                    ) or 0.0)
                ),
                "end_to_end_tokens_per_second": (
                    completion_tokens / workload_seconds if workload_seconds > 0 else 0.0
                ),
                "ttft_ms_mean": statistics.fmean(ttft_ms) if ttft_ms else 0.0,
                "ttft_ms_p90": (
                    ttft_ms[int(0.9 * (len(ttft_ms) - 1))] if ttft_ms else 0.0
                ),
                "cache_hit_pct": mean_or_none(cache_pcts),
                "finish_reason_length_pct": (
                    100.0 * length_finishes / successful if successful else None
                ),
                "complete": complete,
                "artifact_result": "ok" if complete else "incomplete",
            }
        )
    return rows


def baseline_key(row: dict[str, Any]) -> str:
    return f'{row["cohort"]["model"]}|c{row["replay"]["concurrency"]}'


def load_baseline(root: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                runs[baseline_key(row)].append(row)
    return runs


def incomplete_problem(row: dict[str, Any]) -> str | None:
    if row["complete"]:
        return None
    successful = int(row["successful_requests"])
    failed = int(row["failed_requests"])
    requests = int(row["prompt_count"])
    reasons: list[str] = []
    if failed:
        reasons.append(f"{failed} failed")
    if successful + failed != requests:
        reasons.append(f"{successful + failed} of {requests} requests accounted for")
    if requests <= 0:
        reasons.append("no requests recorded")
    detail = "; ".join(reasons) or "completion contract not satisfied"
    return (
        f"{row['cohort']['model']} c{row['replay']['concurrency']}: "
        f"incomplete run ({detail})"
    )


def compare(row: dict[str, Any], prior: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    incomplete = incomplete_problem(row)
    if incomplete is not None:
        problems.append(incomplete)
        return problems
    prior = [
        r
        for r in prior
        if r.get("complete")
        # Cohort contract: a model artifact digest change starts a new cohort.
        and r.get("model", {}).get("sha256") == row["model"].get("sha256")
    ]
    if len(prior) < BASELINE_MIN_RUNS:
        return []  # bootstrap: informational only
    metrics = (
        ("decode_tokens_per_second", 1.0),
        ("end_to_end_tokens_per_second", 1.0),
        ("ttft_ms_mean", -1.0),
        ("ttft_ms_p90", -1.0),
        ("finish_reason_length_pct", -1.0),
    )
    for metric, direction in metrics:
        values = [
            r.get(metric)
            for r in prior[-BASELINE_MIN_RUNS:]
            if isinstance(r.get(metric), (int, float))
        ]
        if len(values) < BASELINE_MIN_RUNS:
            continue
        base = statistics.median(values)
        candidate = row[metric]
        if not isinstance(candidate, (int, float)):
            continue
        if base == 0:
            # Zero baseline median (e.g. no TTFT samples): only the absolute
            # tolerance can fire; the percentage is undefined.
            if abs(candidate) > DEFAULT_TOLERANCE["abs"]:
                problems.append(
                    f"{row['cohort']['model']} c{row['replay']['concurrency']}: "
                    f"{metric} moved from 0.0 baseline to {candidate:.2f}"
                )
            continue
        delta_pct = 100.0 * (candidate - base) / base * direction
        if delta_pct < -(DEFAULT_TOLERANCE["pct"]) and abs(candidate - base) > DEFAULT_TOLERANCE["abs"]:
            problems.append(
                f"{row['cohort']['model']} c{row['replay']['concurrency']}: "
                f"{metric} regressed {delta_pct:.1f}% vs baseline median {base:.2f} "
                f"(candidate {candidate:.2f})"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="pinned model matrix JSON")
    parser.add_argument("--replay-dir", "--summary-dir", dest="replay_dir", type=Path, required=True, help="per-family replay output directory")
    parser.add_argument("--label", default="pr", help="candidate ref label whose cells become history rows")
    parser.add_argument("--hardware", type=Path, required=True, help="hardware fingerprint JSON")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--backend-binary-sha256", default=None)
    parser.add_argument("--replay", type=Path, required=True, help="replay parameters JSON")
    parser.add_argument("--baseline", type=Path, default=None, help="HF dataset runs/ checkout")
    parser.add_argument("--output", type=Path, required=True, help="history JSONL to write")
    parser.add_argument("--gate", action="store_true", help="fail on regression after bootstrap")
    args = parser.parse_args(argv)

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    hardware = json.loads(args.hardware.read_text(encoding="utf-8"))
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    validate_replay_pin(replay)

    rows: list[dict[str, Any]] = []
    for model in matrix["models"]:
        validate_model_pin(model)
        try:
            cells = load_cells(args.replay_dir, model["family"], args.label)
        except FileNotFoundError as error:
            print(f"warning: {error}; preserving other family results", file=sys.stderr)
            continue
        rows.extend(
            build_rows(
                cells,
                model=model,
                replay=replay,
                hardware=hardware,
                source_sha=args.source_sha,
                backend_binary_sha256=args.backend_binary_sha256,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    integrity_problems = [
        problem for row in rows if (problem := incomplete_problem(row)) is not None
    ]
    if not rows:
        integrity_problems.append("no replay rows were produced")

    regression_problems: list[str] = []
    if args.baseline and args.baseline.is_dir():
        baseline = load_baseline(args.baseline)
        for row in rows:
            if not row["complete"]:
                continue
            prior = [
                r for r in baseline.get(baseline_key(row), [])
                # Source SHA intentionally changes across history. All replay
                # inputs, including dataset identity, must otherwise match.
                if r.get("replay") == row["replay"]
            ]
            regression_problems.extend(compare(row, prior))
    problems = [*integrity_problems, *regression_problems]
    if problems:
        print("regressions detected:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    if integrity_problems or (args.gate and regression_problems):
        return 1
    print(f"wrote {len(rows)} history rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
