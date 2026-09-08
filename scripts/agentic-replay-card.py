#!/usr/bin/env python3
"""Render the published benchmark card from the latest history rows + trend."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, help="baseline runs/ directory")
    parser.add_argument("--latest", type=Path, required=True, help="latest run history.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    latest = [json.loads(l) for l in args.latest.read_text(encoding="utf-8").splitlines() if l.strip()]
    all_rows = load_rows(args.history) if args.history and args.history.is_dir() else []
    all_rows += latest
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row["complete"]:
            by_model[row["model"]["family"]].append(row)
    for rows in by_model.values():
        rows.sort(key=lambda r: r["created_utc"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write("# MeshLLM Coding-Agent Serving Benchmark\n\n")
        out.write(
            "Nightly agentic-replay results on the pinned micstudio runner "
            "(Apple M3 Ultra, 256 GB, Metal 4). Full history in this dataset; "
            "schema in `schema.json`.\n\n"
        )
        out.write("## Latest run\n\n")
        newest = max(r["created_utc"] for r in latest)
        out.write(f"Latest complete run: **{newest}** @ `{latest[0]['source_sha'][:9]}`\n\n")
        out.write("| Model | Class | Conc. | Decode tok/s | E2E tok/s | TTFT mean | TTFT p90 | Cache hit % | Finish=length % |\n")
        out.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in sorted(latest, key=lambda r: -r["decode_tokens_per_second"]):
            m = row["model"]
            cache = "—" if row["cache_hit_pct"] is None else f"{row['cache_hit_pct']:.1f}"
            clipped = "—" if row["finish_reason_length_pct"] is None else f"{row['finish_reason_length_pct']:.1f}"
            out.write(
                f"| {m['repo']} {m['quant']} | {m['class']} | {row['replay']['concurrency']} "
                f"| {row['decode_tokens_per_second']:.1f} | {row['end_to_end_tokens_per_second']:.1f} "
                f"| {row['ttft_ms_mean']:.0f} ms | {row['ttft_ms_p90']:.0f} ms "
                f"| {cache} | {clipped} |\n"
            )
        out.write("\n## Trend — decode tok/s\n\n")
        models = sorted(by_model)
        recent = sorted({r["created_utc"][:10] for r in all_rows})[-7:]
        out.write("| Model | " + " | ".join(recent) + " |\n")
        out.write("|---" * (len(models) + 1) + "|\n")
        for model in models:
            cells = []
            for date in recent:
                match = [r for r in by_model[model] if r["created_utc"][:10] == date]
                cells.append(f"{match[-1]['decode_tokens_per_second']:.1f}" if match else "—")
            out.write(f"| {model} | " + " | ".join(cells) + " |\n")
    print(f"wrote card to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
