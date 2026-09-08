#!/usr/bin/env python3
"""Normalize agentic-replay nightly artifacts and gate on cohort-matched drift.

Reads the summary JSON produced by evals/agentic-replay.py for each model in
the pinned micstudio matrix, emits schema-version-2 JSONL history rows, and
compares them against the append-only baseline in the HF dataset checkout.
Mirrors scripts/performance-history.py semantics: exact cohort keys only,
bootstrap-then-gate (three prior complete runs before a regression fails).
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def build_row(
    summary: dict[str, Any],
    *,
    model: dict[str, Any],
    replay: dict[str, Any],
    hardware: dict[str, Any],
    source_sha: str,
    backend_binary_sha256: str | None,
) -> dict[str, Any]:
    requests = summary["requests"]
    cohort = {
        "model": model["family"],
        "quant": model["quant"],
        "concurrency": replay["concurrency"],
        "backend": "mesh",
        "runner": hardware["machine_model"],
    }
    successful = [r for r in requests if r.get("ok")]
    failed = len(requests) - len(successful)
    output_tokens = sum(int(r.get("completion_tokens", 0)) for r in successful)
    wall_ms = float(summary["measured_wall_ms"])
    ttfts = [float(r["ttft_ms"]) for r in successful if r.get("ttft_ms") is not None]
    ttfts.sort()
    p90 = ttfts[int(0.9 * (len(ttfts) - 1))] if ttfts else 0.0
    clipped = sum(
        1 for r in successful if (r.get("finish_reason") or "") == "length"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": summary["created_utc"],
        "source_sha": source_sha,
        "cohort_key": stable_hash(
            {"cohort": cohort, "source": source_sha, "replay": replay, "model_sha": model.get("sha256")}
        ),
        "cohort": cohort,
        "backend_binary_sha256": backend_binary_sha256,
        "hardware_fingerprint": hardware,
        "model": model,
        "replay": replay,
        "prompt_count": len(requests),
        "successful_requests": len(successful),
        "failed_requests": failed,
        "output_tokens": output_tokens,
        "measured_wall_ms": wall_ms,
        "decode_tokens_per_second": float(summary["decode_tokens_per_second"]),
        "end_to_end_tokens_per_second": float(summary["end_to_end_tokens_per_second"]),
        "ttft_ms_mean": statistics.fmean(ttfts) if ttfts else 0.0,
        "ttft_ms_p90": p90,
        "cache_hit_pct": summary.get("cache_hit_pct"),
        "finish_reason_length_pct": (100.0 * clipped / len(successful)) if successful else None,
        "complete": failed == 0,
        "artifact_result": summary["artifact_result"],
    }


def load_baseline(root: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                runs[row["cohort"]["model"] + f'|c{row["replay"]["concurrency"]}'].append(row)
    return runs


def compare(rows: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    prior = [r for r in baseline if r["complete"] and r["model"] == rows[0]["model"]]
    for row in rows:
        if not row["complete"]:
            problems.append(f"{row['cohort']['model']}: incomplete run ({row['failed_requests']} failed)")
            continue
        if len(prior) < BASELINE_MIN_RUNS:
            continue  # bootstrap: informational only
        metrics = (
            ("decode_tokens_per_second", 1.0),
            ("end_to_end_tokens_per_second", 1.0),
            ("ttft_ms_mean", -1.0),
            ("ttft_ms_p90", -1.0),
        )
        for metric, direction in metrics:
            base = statistics.median(r[metric] for r in prior[-BASELINE_MIN_RUNS:])
            candidate = row[metric]
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
    parser.add_argument("--summary-dir", type=Path, required=True, help="per-model summary JSON directory")
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

    rows: list[dict[str, Any]] = []
    for model in matrix["models"]:
        summary = load_summary(args.summary_dir / f"{model['family']}.json")
        rows.append(
            build_row(
                summary,
                model=model,
                replay={**replay, "concurrency": summary["concurrency"]},
                hardware=hardware,
                source_sha=args.source_sha,
                backend_binary_sha256=args.backend_binary_sha256,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    problems: list[str] = []
    if args.baseline and args.baseline.is_dir():
        baseline = load_baseline(args.baseline)
        problems = compare(rows, baseline.get("all", [])) or [
            p
            for row in rows
            for p in compare(
                [row],
                [r for r in baseline.get(row["cohort"]["model"] + f'|c{row["replay"]["concurrency"]}', [])],
            )
        ]
    if problems:
        print("regressions detected:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        if args.gate:
            return 1
    print(f"wrote {len(rows)} history rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
