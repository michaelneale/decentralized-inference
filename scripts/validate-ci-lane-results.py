#!/usr/bin/env python3
"""Validate one dispatched CI lane against its bounded planner projection."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


class LaneResultError(RuntimeError):
    """Raised when a planned lane did not complete successfully."""


def _load_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise LaneResultError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise LaneResultError(f"{label} must be a JSON object")
    return parsed


def _ids(lane_plan: dict[str, Any], matrix: str) -> set[str]:
    matrices = lane_plan.get("matrices")
    if not isinstance(matrices, dict):
        raise LaneResultError("lane plan matrices must be an object")
    rows = matrices.get(matrix, [])
    if not isinstance(rows, list):
        raise LaneResultError(f"lane plan matrix {matrix!r} must be an array")
    ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise LaneResultError(f"lane plan matrix {matrix}[{index}] needs an ID")
        ids.add(row["id"])
    return ids


def _required_jobs(lane_plan: dict[str, Any]) -> set[str]:
    lane = lane_plan.get("lane")
    slices = lane_plan.get("required_slices")
    if not isinstance(lane, str) or not isinstance(slices, list):
        raise LaneResultError("lane plan needs lane and required_slices")
    selected = set(slices)
    jobs: set[str] = set()

    if lane == "quality":
        if "quality" in selected:
            jobs.add("quality")
        if "runner-contract" in selected:
            jobs.add("runner_contract")
    elif lane == "website":
        if "web" in selected:
            jobs.add("web")
    elif lane == "linux":
        hosts = _ids(lane_plan, "hosts")
        runtimes = _ids(lane_plan, "runtime_products")
        sdk = _ids(lane_plan, "sdk")
        if hosts:
            jobs.update({"ui_artifact", "hosts"})
        if "static-abi" in selected:
            jobs.add("static_abi")
        if _ids(lane_plan, "rust_tests"):
            jobs.add("rust_tests")
        if runtimes:
            jobs.update({"native_runtimes", "runtime_product"})
        if "kotlin" in sdk:
            jobs.add("kotlin_sdk_input")
        if sdk:
            jobs.add("sdk")
        if _ids(lane_plan, "smoke"):
            jobs.add("product_smoke")
    elif lane == "macos":
        hosts = _ids(lane_plan, "hosts")
        runtimes = _ids(lane_plan, "runtime_products")
        sdk = _ids(lane_plan, "sdk")
        if hosts:
            jobs.update({"ui_artifact", "hosts"})
        if runtimes:
            jobs.update({"native_runtimes", "runtime_product"})
        if _ids(lane_plan, "platform_checks"):
            jobs.add("platform_checks")
        if "swift" in sdk:
            jobs.add("swift_sdk_input")
        if sdk:
            jobs.add("sdk")
        if _ids(lane_plan, "smoke"):
            jobs.add("product_smoke")
        if jobs:
            jobs.add("validate_plan")
    elif lane == "windows":
        hosts = _ids(lane_plan, "hosts")
        runtimes = _ids(lane_plan, "runtime_products")
        if hosts:
            jobs.update({"ui_artifact", "hosts"})
        if runtimes:
            jobs.update({"native_runtimes", "runtime_product"})
        if _ids(lane_plan, "platform_checks"):
            jobs.add("platform_checks")
        if _ids(lane_plan, "smoke"):
            jobs.add("product_smoke")
    else:
        raise LaneResultError(f"unknown CI lane {lane!r}")
    return jobs


def validate(lane_plan: dict[str, Any], needs: dict[str, Any]) -> None:
    required = lane_plan.get("required")
    if not isinstance(required, bool):
        raise LaneResultError("lane plan required must be a boolean")

    expected = _required_jobs(lane_plan)
    if required and not expected:
        raise LaneResultError("required lane has no planned jobs")
    for job in expected:
        state = needs.get(job)
        result = state.get("result") if isinstance(state, dict) else None
        if result != "success":
            raise LaneResultError(f"planned job {job!r} finished with {result!r}")

    for job, state in needs.items():
        result = state.get("result") if isinstance(state, dict) else None
        if job in expected:
            continue
        if result != "skipped":
            raise LaneResultError(f"lane job {job!r} finished with {result!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-plan", required=True)
    parser.add_argument("--needs", required=True)
    args = parser.parse_args()
    try:
        lane_plan = _load_object(args.lane_plan, "lane plan")
        needs = _load_object(args.needs, "needs")
        validate(lane_plan, needs)
    except LaneResultError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
