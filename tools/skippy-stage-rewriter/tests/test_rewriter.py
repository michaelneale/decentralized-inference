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
            "supported_auxiliary": 0,
            "supported_whole_model": 0,
            "transformable": 1,
            "unsupported_shape": 0,
        }
        assert builder["verdict"] == "transformable"
        assert builder["proof"]["activation_in"] == "inpL"
        assert builder["proof"]["activation_out"] == "inpL"
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
        assert "end_block(inpL, il);" in transformed
        assert "for (int il = il_start; il < il_end; ++il)" in transformed

        second = run(tool, source_root, Path(temporary) / "second.json", apply=False)
        assert second["summary"]["transformable"] == 0
        assert second["summary"]["already_transformed"] == 1
        assert second["builders"][0]["edits"] == []

        filter_only = run(
            tool,
            source_root,
            Path(temporary) / "filter-only.json",
            source_name="filter-only.cpp",
            apply=False,
        )["builders"][0]
        assert filter_only["verdict"] == "transformable"
        assert {edit["kind"] for edit in filter_only["edits"]} == {
            "insert_begin_block",
            "insert_end_block",
        }

        run(
            tool,
            source_root,
            Path(temporary) / "filter-only-applied.json",
            source_name="filter-only.cpp",
            apply=True,
        )
        filter_only_second = run(
            tool,
            source_root,
            Path(temporary) / "filter-only-second.json",
            source_name="filter-only.cpp",
            apply=False,
        )["builders"][0]
        assert filter_only_second["verdict"] == "already_transformed"
        assert filter_only_second["edits"] == []

        multiple_assignments = run(
            tool,
            source_root,
            Path(temporary) / "multiple-assignments.json",
            source_name="multiple-assignments.cpp",
            apply=False,
        )["builders"][0]
        assert multiple_assignments["verdict"] == "transformable"
        assert multiple_assignments["proof"]["loop"]["var"] == "layer_index"
        assert multiple_assignments["proof"]["activation_out"] == "inpL"

        self_carried = run(
            tool,
            source_root,
            Path(temporary) / "self-carried.json",
            source_name="self-carried.cpp",
            apply=False,
        )["builders"][0]
        assert self_carried["verdict"] == "transformable"
        assert self_carried["proof"]["activation_in"] == "cur"
        assert self_carried["proof"]["activation_out"] == "cur"

        two_loops = run(
            tool,
            source_root,
            Path(temporary) / "two-loops.json",
            source_name="two-loops.cpp",
            apply=False,
        )["builders"][0]
        assert two_loops["verdict"] == "unsupported_shape"
        assert two_loops["unsupported_reason"] == (
            "multiple equally ranked layer block loops"
        )

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

        trailing_work = run(
            tool,
            source_root,
            Path(temporary) / "trailing-work.json",
            source_name="trailing-work.cpp",
            apply=False,
        )["builders"][0]
        assert trailing_work["verdict"] == "transformable"
        assert trailing_work["proof"]["activation_out"] == "inpL"

        auxiliary = run(
            tool,
            source_root,
            Path(temporary) / "auxiliary.json",
            source_name="auxiliary.cpp",
            apply=False,
        )["builders"][0]
        assert auxiliary["verdict"] == "supported_auxiliary"
        assert auxiliary["proof"]["execution_scope"] == "final_stage_sidecar"
        assert auxiliary["proof"]["scope_evidence"] == ["typed_mtp_builder"]
        assert auxiliary["edits"] == []

        multiple_domains = run(
            tool,
            source_root,
            Path(temporary) / "multiple-domains.json",
            source_name="multiple-domains.cpp",
            apply=False,
        )["builders"][0]
        assert multiple_domains["verdict"] == "supported_whole_model"
        assert multiple_domains["proof"]["execution_scope"] == (
            "multiple_sequential_layer_domains"
        )
        assert multiple_domains["proof"]["scope_evidence"] == [
            "first.n_layer",
            "second.n_layer",
        ]
        assert multiple_domains["edits"] == []

        local_loop_bound = run(
            tool,
            source_root,
            Path(temporary) / "local-loop-bound.json",
            source_name="local-loop-bound.cpp",
            apply=False,
        )["builders"][0]
        assert local_loop_bound["verdict"] == "transformable"
        assert local_loop_bound["proof"]["loop"]["end"] == (
            "hparams.dec_n_layer"
        )
        run(
            tool,
            source_root,
            Path(temporary) / "local-loop-bound-applied.json",
            source_name="local-loop-bound.cpp",
            apply=True,
        )
        local_loop_source = (
            source_root / "src/models/local-loop-bound.cpp"
        ).read_text(encoding="utf-8")
        assert "stage_filter.layer_end   : hparams.dec_n_layer" in local_loop_source
        assert "\n    end_block(inpL, il);\n  }" in local_loop_source

        continue_path = run(
            tool,
            source_root,
            Path(temporary) / "continue-path.json",
            source_name="continue-path.cpp",
            apply=False,
        )["builders"][0]
        assert continue_path["verdict"] == "transformable"
        assert "continue" not in continue_path["proof"]["nonlocal_exits"]
        assert sum(
            edit["kind"] == "insert_end_block_before_continue"
            for edit in continue_path["edits"]
        ) == 1
        run(
            tool,
            source_root,
            Path(temporary) / "continue-path-applied.json",
            source_name="continue-path.cpp",
            apply=True,
        )
        continue_second = run(
            tool,
            source_root,
            Path(temporary) / "continue-path-second.json",
            source_name="continue-path.cpp",
            apply=False,
        )["builders"][0]
        assert continue_second["verdict"] == "already_transformed"

        unbraced_source = (
            fixture_root / "src/models/continue-path.cpp"
        ).read_text(encoding="utf-8").replace(
            "if (il == 2) {\n      continue;\n    }",
            "if (il == 2) continue;",
        )
        (source_root / "src/models/unbraced-continue.cpp").write_text(
            unbraced_source, encoding="utf-8"
        )
        unbraced_continue = run(
            tool,
            source_root,
            Path(temporary) / "unbraced-continue.json",
            source_name="unbraced-continue.cpp",
            apply=False,
        )["builders"][0]
        assert unbraced_continue["verdict"] == "transformable"
        assert sum(
            edit["kind"] == "wrap_end_block_before_continue"
            for edit in unbraced_continue["edits"]
        ) == 1
        run(
            tool,
            source_root,
            Path(temporary) / "unbraced-continue-applied.json",
            source_name="unbraced-continue.cpp",
            apply=True,
        )
        unbraced_transformed = (
            source_root / "src/models/unbraced-continue.cpp"
        ).read_text(encoding="utf-8")
        assert "if (il == 2) {\n" in unbraced_transformed
        assert "end_block(inpL, il);\n        continue;\n    }" in unbraced_transformed
        unbraced_second = run(
            tool,
            source_root,
            Path(temporary) / "unbraced-continue-second.json",
            source_name="unbraced-continue.cpp",
            apply=False,
        )["builders"][0]
        assert unbraced_second["verdict"] == "already_transformed"

        unbraced_else_source = unbraced_source.replace(
            "if (il == 2) continue;\n    inpL = block(inpL, il);",
            "if (il == 2) continue;\n    else inpL = block(inpL, il);",
        )
        (source_root / "src/models/unbraced-continue-else.cpp").write_text(
            unbraced_else_source, encoding="utf-8"
        )
        run(
            tool,
            source_root,
            Path(temporary) / "unbraced-continue-else-applied.json",
            source_name="unbraced-continue-else.cpp",
            apply=True,
        )
        unbraced_else_second = run(
            tool,
            source_root,
            Path(temporary) / "unbraced-continue-else-second.json",
            source_name="unbraced-continue-else.cpp",
            apply=False,
        )["builders"][0]
        assert unbraced_else_second["verdict"] == "already_transformed"

        switch_break = run(
            tool,
            source_root,
            Path(temporary) / "switch-break.json",
            source_name="switch-break.cpp",
            apply=False,
        )["builders"][0]
        assert switch_break["verdict"] == "transformable"
        assert switch_break["proof"]["nonlocal_exits"] == []

        hyperconnection = run(
            tool,
            source_root,
            Path(temporary) / "hyperconnection.json",
            source_name="hyperconnection.cpp",
            apply=False,
        )["builders"][0]
        assert hyperconnection["verdict"] == "transformable"
        assert hyperconnection["proof"]["activation_in"] == "inpL"
        assert hyperconnection["proof"]["scope_evidence"] == [
            "hyperconnection_activation_frontier"
        ]
        assert {
            "rewrite_hyperconnection_embedding_owner",
            "rewrite_hyperconnection_initializer",
            "insert_hyperconnection_import",
            "guard_hyperconnection_repeat",
        }.issubset({edit["kind"] for edit in hyperconnection["edits"]})
        run(
            tool,
            source_root,
            Path(temporary) / "hyperconnection-applied.json",
            source_name="hyperconnection.cpp",
            apply=True,
        )
        hyperconnection_source = (
            source_root / "src/models/hyperconnection.cpp"
        ).read_text(encoding="utf-8")
        assert "std::make_unique<llm_graph_input_hyperconnection>" in hyperconnection_source
        assert "res->t_skippy_activation_input = inpL;" in hyperconnection_source
        hyperconnection_second = run(
            tool,
            source_root,
            Path(temporary) / "hyperconnection-second.json",
            source_name="hyperconnection.cpp",
            apply=False,
        )["builders"][0]
        assert hyperconnection_second["verdict"] == "already_transformed"

        hyperconnection_initializer = run(
            tool,
            source_root,
            Path(temporary) / "hyperconnection-initializer.json",
            source_name="hyperconnection-initializer.cpp",
            apply=False,
        )["builders"][0]
        assert hyperconnection_initializer["verdict"] == "transformable"
        assert hyperconnection_initializer["proof"]["activation_in"] == "res_hc"
        initializer_edit_kinds = {
            edit["kind"] for edit in hyperconnection_initializer["edits"]
        }
        assert "insert_hyperconnection_import" in initializer_edit_kinds
        assert "rewrite_hyperconnection_initializer" in initializer_edit_kinds
        assert "guard_hyperconnection_embedding_prelude" in initializer_edit_kinds
        assert "guard_hyperconnection_repeat" not in initializer_edit_kinds
        run(
            tool,
            source_root,
            Path(temporary) / "hyperconnection-initializer-applied.json",
            source_name="hyperconnection-initializer.cpp",
            apply=True,
        )
        hyperconnection_initializer_source = (
            source_root / "src/models/hyperconnection-initializer.cpp"
        ).read_text(encoding="utf-8")
        assert "res->t_skippy_activation_input = res_hc;" in hyperconnection_initializer_source
        assert hyperconnection_initializer_source.count(
            "if (!stage_filtered || il_start == 0)"
        ) == 2
        hyperconnection_initializer_second = run(
            tool,
            source_root,
            Path(temporary) / "hyperconnection-initializer-second.json",
            source_name="hyperconnection-initializer.cpp",
            apply=False,
        )["builders"][0]
        assert hyperconnection_initializer_second["verdict"] == "already_transformed"

        embedding_prelude = run(
            tool,
            source_root,
            Path(temporary) / "embedding-prelude.json",
            source_name="embedding-prelude.cpp",
            apply=False,
        )["builders"][0]
        assert embedding_prelude["verdict"] == "transformable"
        assert embedding_prelude["proof"]["scope_evidence"] == [
            "guarded_embedding_prelude"
        ]
        assert any(
            edit["kind"] == "guard_embedding_prelude"
            for edit in embedding_prelude["edits"]
        )
        run(
            tool,
            source_root,
            Path(temporary) / "embedding-prelude-applied.json",
            source_name="embedding-prelude.cpp",
            apply=True,
        )
        embedding_prelude_source = (
            source_root / "src/models/embedding-prelude.cpp"
        ).read_text(encoding="utf-8")
        assert (
            "inpL = stage_filtered && il_start > 0 ? inpL : scale(inpL);"
            in embedding_prelude_source
        )
        assert (
            "if ((!stage_filtered || il_start == 0) && (scale_embeddings))"
            in embedding_prelude_source
        )
        embedding_prelude_second = run(
            tool,
            source_root,
            Path(temporary) / "embedding-prelude-second.json",
            source_name="embedding-prelude.cpp",
            apply=False,
        )["builders"][0]
        assert embedding_prelude_second["verdict"] == "already_transformed"

        embedding_prelude_else = run(
            tool,
            source_root,
            Path(temporary) / "embedding-prelude-else.json",
            source_name="embedding-prelude-else.cpp",
            apply=False,
        )["builders"][0]
        assert embedding_prelude_else["verdict"] == "unsupported_shape"
        assert embedding_prelude_else["unsupported_reason"] == (
            "pre-loop activation conditional has an else branch"
        )
        assert embedding_prelude_else["edits"] == []

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
