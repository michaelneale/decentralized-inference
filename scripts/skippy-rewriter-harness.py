#!/usr/bin/env python3
"""CI harness for the Skippy stage-rewriter (Clang Transformer generator).

Consumes the rewriter's outputs and enforces the contract described in
PLANS/SKIPPY_REWRITER_HARNESS_CONTRACT_V0.md (workspace copy; repo copy lands
with the generator commit):

- report validation: per-builder verdicts, refusal reasons, proof blocks
- patch-drift checking: `--check` result forwarded as pass/fail
- idempotence assertion: second-run report must contain an empty edit set

The generator (scama's tool) owns producing these artifacts; this script only
validates and gates. It never invokes the compiler or the graph verifier
directly -- those results arrive as pre-computed pass/fail inputs so the
harness stays decoupled from the generator's toolchain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
VERDICTS = {
    "transformable",
    "already_transformed",
    "supported_auxiliary",
    "supported_whole_model",
    "unsupported_shape",
    "error",
}
# Proof fields required for every transformable builder. The enforcement
# matrix hangs off these; a missing field is a contract violation, not a
# soft warning.
REQUIRED_PROOF_FIELDS = (
    "loop",
    "activation_in",
    "activation_out",
    "embedding_owner",
    "output_owner",
    "terminal_predicates",
    "nonlocal_exits",
    "execution_scope",
    "scope_evidence",
)


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    return report


def validate_report(report: dict[str, Any]) -> list[str]:
    """Return a list of contract violations (empty means valid)."""
    errors: list[str] = []
    version = report.get("schema_version")
    if version != SCHEMA_VERSION:
        errors.append(
            f"schema_version {version!r} != supported {SCHEMA_VERSION}"
        )
    for key in ("llama_cpp_commit", "generator_version"):
        value = report.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"missing required string field {key!r}")

    builders = report.get("builders")
    if not isinstance(builders, list) or not builders:
        errors.append("missing non-empty 'builders' array")
        return errors

    seen_keys: set[tuple[str, str]] = set()
    for idx, builder in enumerate(builders):
        label = builder.get("file", f"<builders[{idx}]>")
        # (file, constructor) is the unique builder key: 19 model files
        # contain multiple graph constructors, so file path alone is not
        # unique. `file` alone remains the display label.
        key = (label, str(builder.get("constructor", "")))
        if key in seen_keys:
            errors.append(f"duplicate builder record for {key[0]}::{key[1]}")
        seen_keys.add(key)

        verdict = builder.get("verdict")
        if verdict not in VERDICTS:
            errors.append(f"{label}: unknown verdict {verdict!r}")
            continue

        if verdict == "transformable":
            constructor = builder.get("constructor")
            if not isinstance(constructor, str) or not constructor:
                errors.append(
                    f"{label}: transformable without qualified 'constructor' name"
                )
            proof = builder.get("proof")
            if not isinstance(proof, dict):
                errors.append(f"{label}: transformable without proof block")
            else:
                for field in REQUIRED_PROOF_FIELDS:
                    if field not in proof:
                        errors.append(f"{label}: proof missing field {field!r}")
            edits = builder.get("edits")
            if not isinstance(edits, list) or not edits:
                errors.append(f"{label}: transformable with empty edit set")
        elif verdict == "unsupported_shape":
            reason = builder.get("unsupported_reason")
            if not isinstance(reason, str) or not reason:
                errors.append(
                    f"{label}: unsupported_shape without unsupported_reason"
                )
        elif verdict in {
            "already_transformed",
            "supported_auxiliary",
            "supported_whole_model",
        }:
            edits = builder.get("edits")
            if edits:
                errors.append(
                    f"{label}: {verdict} must carry no edits"
                )
            if verdict.startswith("supported_"):
                proof = builder.get("proof")
                expected_scope = {
                    "supported_auxiliary": "final_stage_sidecar",
                    "supported_whole_model": "multiple_sequential_layer_domains",
                }[verdict]
                if not isinstance(proof, dict):
                    errors.append(f"{label}: {verdict} without proof block")
                elif proof.get("execution_scope") != expected_scope:
                    errors.append(
                        f"{label}: {verdict} requires execution_scope "
                        f"{expected_scope!r}"
                    )
                elif not proof.get("scope_evidence"):
                    errors.append(f"{label}: {verdict} without scope evidence")
        else:  # error
            errors.append(f"{label}: error verdict present in report")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("missing 'summary' object")
    else:
        counted = sum(
            summary.get(verdict, 0) for verdict in VERDICTS
        )
        if counted != len(builders):
            errors.append(
                f"summary counts ({counted}) != builder records ({len(builders)})"
            )
    return errors


def check_idempotence(report: dict[str, Any]) -> list[str]:
    """A second run over the transformed tree must edit nothing."""
    errors: list[str] = []
    for builder in report.get("builders", []):
        if builder.get("verdict") == "transformable":
            label = builder.get("file", "<unknown>")
            errors.append(
                f"{label}: idempotence violation -- transformable on second run"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="rewriter JSON report to validate",
    )
    parser.add_argument(
        "--mode",
        choices=("validate", "idempotence"),
        default="validate",
        help="validate a first-run report, or assert a second run edited nothing",
    )
    parser.add_argument(
        "--patch-check",
        choices=("pass", "fail"),
        default=None,
        help="forwarded result of the generator's --check byte-compare",
    )
    parser.add_argument(
        "--patch-drift-gate",
        choices=("warn", "fail"),
        default="warn",
        help="policy for patch drift while the queue regeneration is in flight",
    )
    parser.add_argument(
        "--compile-result",
        choices=("pass", "fail", "skipped"),
        default="skipped",
        help="compile result of the transformed tree",
    )
    parser.add_argument(
        "--graph-verify-result",
        choices=("pass", "fail", "skipped"),
        default="skipped",
        help="no-allocation graph verifier result for the transformed tree",
    )
    args = parser.parse_args(argv)

    try:
        report = load_report(args.report)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load report: {exc}", file=sys.stderr)
        return 2

    if args.mode == "idempotence":
        violations = check_idempotence(report)
    else:
        violations = validate_report(report)

    failures: list[str] = list(violations)

    if args.patch_check == "fail":
        message = "generated family patch drifts from checked-in patch"
        if args.patch_drift_gate == "fail":
            failures.append(message)
        else:
            print(f"warn: {message}")
    elif args.patch_check not in ("pass", None):
        failures.append(f"invalid --patch-check value: {args.patch_check}")

    if args.compile_result == "fail":
        failures.append("transformed tree failed to compile")
    if args.graph_verify_result == "fail":
        failures.append("no-allocation graph verifier failed on transformed tree")

    for failure in failures:
        print(f"fail: {failure}", file=sys.stderr)
    if not failures:
        print(
            f"ok: report valid ({args.mode}); "
            f"{len(report.get('builders', []))} builders checked"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
