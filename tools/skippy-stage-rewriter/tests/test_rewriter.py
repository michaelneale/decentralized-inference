#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(
    tool: Path,
    source_root: Path,
    report: Path,
    *,
    source_name: str = "conventional.cpp",
    apply: bool,
) -> dict:
    source = source_root / "src/models" / source_name
    command = [
        str(tool),
        "--source-root",
        str(source_root),
        "--llama-commit",
        "fixture",
        "--report",
        str(report),
    ]
    if apply:
        command.append("--apply")
    command.extend([str(source), "--", "-std=c++17"])
    subprocess.run(command, check=True)
    return json.loads(report.read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_rewriter.py TOOL FIXTURE_ROOT")
    tool = Path(sys.argv[1]).resolve()
    fixture_root = Path(sys.argv[2]).resolve()

    with tempfile.TemporaryDirectory(prefix="skippy-rewriter-") as temporary:
        source_root = Path(temporary) / "source"
        shutil.copytree(fixture_root, source_root)

        first = run(tool, source_root, Path(temporary) / "first.json", apply=False)
        builder = first["builders"][0]
        assert first["summary"] == {
            "already_transformed": 0,
            "error": 0,
            "transformable": 1,
            "unsupported_shape": 0,
        }
        assert builder["verdict"] == "transformable"
        assert builder["proof"]["activation_in"] == "inpL"
        assert builder["proof"]["activation_out"] == "cur"
        assert {edit["kind"] for edit in builder["edits"]} == {
            "insert_filter_declarations",
            "rewrite_embedding_owner",
            "rewrite_output_owner",
            "rewrite_loop_start",
            "rewrite_loop_end",
            "rewrite_terminal_endpoint",
            "insert_begin_block",
            "insert_end_block",
            "insert_stage_boundary",
        }

        run(tool, source_root, Path(temporary) / "applied.json", apply=True)
        transformed = (source_root / "src/models/conventional.cpp").read_text(
            encoding="utf-8"
        )
        assert "begin_block(inpL, il);" in transformed
        assert "end_block(cur, il);" in transformed
        assert "for (int il = il_start; il < il_end; ++il)" in transformed

        second = run(tool, source_root, Path(temporary) / "second.json", apply=False)
        assert second["summary"]["transformable"] == 0
        assert second["summary"]["already_transformed"] == 1
        assert second["builders"][0]["edits"] == []

        two_loops = run(
            tool,
            source_root,
            Path(temporary) / "two-loops.json",
            source_name="two-loops.cpp",
            apply=False,
        )["builders"][0]
        assert two_loops["verdict"] == "unsupported_shape"
        assert two_loops["unsupported_reason"] == "multiple n_layer block loops"

        nonlocal_exit = run(
            tool,
            source_root,
            Path(temporary) / "nonlocal-exit.json",
            source_name="nonlocal-exit.cpp",
            apply=False,
        )["builders"][0]
        assert nonlocal_exit["verdict"] == "unsupported_shape"
        assert nonlocal_exit["unsupported_reason"] == (
            "block loop contains a non-local exit"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
