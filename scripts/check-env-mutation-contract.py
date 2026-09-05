#!/usr/bin/env python3
"""Check the Rust process-environment mutation contract and census.

`std::env::set_var` and `remove_var` are unsafe on Rust 2024 platforms where a
concurrent environment reader can race the mutation.  The audited call sites
fall into two deliberately separate contracts:

* test-only overrides are scoped by a guard and every test that owns one is
  marked ``#[serial]``;
* build scripts run in their own process, while the four runtime calls that do
  not yet have a proven single-threaded boundary remain explicit TODOs.

The source tree contains several independent crates (and two build scripts),
so a shared Rust test helper would introduce an unnecessary dev-dependency
coupling. This repository-level check discovers every Rust mutation site,
strictly verifies the original TODO audit surface, and freezes the count in
pre-existing mutation files that have not yet joined that stricter contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


TODO = "// TODO: Audit that the environment access only happens in single-threaded code."

# This is the complete census of the 128 TODO comments that this check was
# introduced to audit. These files receive the strict serial-test/deferred-site
# checks below.
AUDITED_FILES = (
    "crates/skippy-protocol/build.rs",
    "crates/mesh-llm-plugin/build.rs",
    "crates/mesh-llm-config/src/env_overrides.rs",
    "crates/mesh-llm-host-runtime/src/plugin/config/tests.rs",
    "crates/mesh-llm-host-runtime/src/capture.rs",
    "crates/mesh-llm-host-runtime/src/runtime/instance.rs",
    "crates/mesh-llm-host-runtime/src/models/maintenance.rs",
    "crates/mesh-llm-host-runtime/src/models/remote_catalog.rs",
    "crates/mesh-llm-host-runtime/src/models/artifact_transfer.rs",
    "crates/mesh-llm-host-runtime/src/models/delete_tests.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/metal_pipeline_cache.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization/package_download.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization/cache_management.rs",
    "crates/model-hf/src/store/local.rs",
    "crates/mesh-llm-system/src/autoupdate.rs",
    "crates/mesh-llm-system/src/autoupdate/release_fetch.rs",
    "crates/mesh-llm-system/src/benchmark/tests.rs",
    "crates/skippy-runtime/src/logging.rs",
    "crates/mesh-llm-host-runtime/src/runtime/run_auto.rs",
)

# Other process-environment mutations predate the 128-TODO audit. They are
# frozen by exact file/count so this checker cannot overstate their safety, and
# so a new call (or a new mutation-bearing file) requires explicit review.
KNOWN_UNAUDITED_MUTATION_COUNTS = {
    "crates/mesh-llm-host-runtime/src/api/routes/plugins.rs": 3,
    "crates/mesh-llm-host-runtime/src/api/tests/apply_config_diagnostics.rs": 3,
    "crates/mesh-llm-host-runtime/src/api/tests/mod.rs": 3,
    "crates/mesh-llm-host-runtime/src/api/tests/runtime_config_validation_authority.rs": 3,
    "crates/mesh-llm-host-runtime/src/mesh/tests/admission/requirements.rs": 3,
    "crates/mesh-llm-host-runtime/src/mesh/tests/owner_control.rs": 5,
    "crates/mesh-llm-host-runtime/src/models/inventory.rs": 13,
    "crates/mesh-llm-host-runtime/src/models/resolve/tests.rs": 4,
    "crates/mesh-llm-host-runtime/src/network/nostr/auto.rs": 3,
    "crates/mesh-llm-host-runtime/src/network/nostr/keys.rs": 3,
    "crates/mesh-llm-host-runtime/src/runtime/config_state_tests/support.rs": 3,
    "crates/mesh-llm-host-runtime/src/runtime/tests/auto_join.rs": 1,
    "crates/mesh-llm-host-runtime/src/runtime/tests/mod.rs": 2,
    "crates/mesh-llm-host-runtime/src/runtime/tests/startup_models.rs": 2,
    "crates/mesh-llm-runtime-install/src/lib.rs": 16,
    "crates/mesh-llm/src/commands/plugin_cli.rs": 3,
    "crates/model-hf/src/cache_paths.rs": 2,
}

# These are the only intentionally unresolved sites.  They execute on runtime
# startup / native-runtime setup paths that may already have Tokio worker
# threads, so replacing the TODO with a guessed SAFETY claim would be unsafe.
DEFERRED_FILES = {
    "crates/skippy-runtime/src/logging.rs",
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization.rs",
    "crates/mesh-llm-host-runtime/src/runtime/run_auto.rs",
}

# Production mutations whose callers establish a real synchronous bootstrap
# boundary. These are neither tests nor deferred assertions: the exact owning
# function and the shipped binary's call order are part of the contract.
SYNCHRONOUS_BOOTSTRAP_FILES = {
    "crates/mesh-llm-host-runtime/src/inference/skippy/metal_pipeline_cache.rs":
        "configure_metal_pipeline_cache",
}

MUTATION_RE = re.compile(r"(?:std::)?env::(?:set_var|remove_var)\s*\(")
FUNCTION_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*<[^>{}]*>)?\s*\(")
SERIAL_ATTR_RE = re.compile(r"^\s*#\[(?:serial|serial_test::serial)\]\s*$")

# These helpers own scoped mutations for call sites that were manually
# verified as serial tests. Listing the helpers prevents a production function
# from passing merely because a nearby comment contains the text `#[serial]`.
SERIAL_TEST_HELPERS = {
    "crates/mesh-llm-config/src/env_overrides.rs": {
        "drop",
        "set",
        "with_env_override_for_test",
    },
    "crates/mesh-llm-host-runtime/src/capture.rs": {"drop"},
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization/cache_management.rs": {
        "restore_env"
    },
    "crates/mesh-llm-host-runtime/src/inference/skippy/materialization/package_download.rs": {
        "restore_env"
    },
    "crates/mesh-llm-host-runtime/src/models/artifact_transfer.rs": {"restore_env"},
    "crates/mesh-llm-host-runtime/src/models/delete_tests.rs": {"restore_env"},
    "crates/mesh-llm-host-runtime/src/models/maintenance.rs": {"restore_env"},
    "crates/mesh-llm-host-runtime/src/runtime/instance.rs": {
        "drop",
        "save_and_remove",
        "save_and_set",
    },
    "crates/mesh-llm-system/src/benchmark/tests.rs": {
        "drop",
        "with_benchmark_child_override",
    },
    "crates/model-hf/src/store/local.rs": {"restore_env"},
}


def mutation_lines(lines: list[str]) -> list[int]:
    """Return zero-based lines containing a process-env mutation call."""

    return [index for index, line in enumerate(lines) if MUTATION_RE.search(line)]


def nearest_function(lines: list[str], line_index: int) -> tuple[int, str] | None:
    for index in range(line_index, -1, -1):
        match = FUNCTION_RE.search(lines[index])
        if match:
            return index, match.group(1)
    return None


def preceding_comment_block(lines: list[str], line_index: int) -> str:
    """Return only the comment block directly adjacent to a mutation."""

    block: list[str] = []
    index = line_index - 1
    in_block_comment = False
    while index >= 0:
        stripped = lines[index].strip()
        if stripped.endswith("*/"):
            in_block_comment = True
        if in_block_comment or stripped.startswith(("//", "/*", "*")):
            block.append(lines[index])
            if stripped.startswith("/*"):
                in_block_comment = False
            index -= 1
            continue
        break
    return "\n".join(reversed(block))


def test_contract(
    relative_path: str,
    lines: list[str],
    line_index: int,
    function: tuple[int, str] | None,
) -> bool:
    """Whether a mutation has an explicit serial-test contract nearby."""

    if function is None:
        return False
    function_index, function_name = function
    attrs = lines[max(0, function_index - 8) : function_index]
    if any(SERIAL_ATTR_RE.match(line) for line in attrs):
        return True
    helpers = SERIAL_TEST_HELPERS.get(relative_path, set())
    nearby = preceding_comment_block(lines, line_index).lower()
    return function_name in helpers and "serial" in nearby


def check_file(root: Path, relative_path: str) -> list[str]:
    path = root / relative_path
    if not path.is_file():
        return [f"{relative_path}: audited source file is missing"]

    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    is_build_script = path.name == "build.rs"
    is_deferred = relative_path in DEFERRED_FILES
    bootstrap_function = SYNCHRONOUS_BOOTSTRAP_FILES.get(relative_path)
    has_test_module = (
        any("#[cfg(test)]" in line for line in lines)
        or "/tests/" in relative_path
        or path.name == "tests.rs"
        or path.name.endswith("_tests.rs")
    )

    for line_index in mutation_lines(lines):
        line_number = line_index + 1
        nearby = preceding_comment_block(lines, line_index)
        normalized_nearby = " ".join(nearby.split())
        function = nearest_function(lines, line_index)
        function_name = function[1] if function else "<module>"

        if is_build_script:
            if "SAFETY:" not in nearby or "build script" not in nearby.lower():
                errors.append(
                    f"{relative_path}:{line_number} ({function_name}): "
                    "build-script environment mutation needs a build-script SAFETY comment"
                )
            continue

        if is_deferred:
            # Keep an explicit marker at every unresolved runtime mutation. A
            # future audit can remove it only after establishing an ordering
            # guarantee or eliminating the process-global mutation.
            if "SAFETY:" not in nearby or TODO not in nearby:
                errors.append(
                    f"{relative_path}:{line_number} ({function_name}): "
                    "deferred runtime mutation needs adjacent SAFETY and audit TODO comments"
                )
            continue

        if bootstrap_function is not None:
            if function_name != bootstrap_function:
                errors.append(
                    f"{relative_path}:{line_number} ({function_name}): "
                    f"bootstrap mutation must remain in {bootstrap_function}"
                )
            if (
                "SAFETY:" not in normalized_nearby
                or "single-threaded" not in normalized_nearby
                or "Tokio runtime" not in normalized_nearby
            ):
                errors.append(
                    f"{relative_path}:{line_number} ({function_name}): "
                    "bootstrap mutation needs an adjacent SAFETY comment with the "
                    "single-threaded pre-Tokio ordering guarantee"
                )
            continue

        if not has_test_module:
            errors.append(
                f"{relative_path}:{line_number} ({function_name}): "
                "audited mutation is outside a recognized test module"
            )
            continue

        if "SAFETY:" not in nearby:
            errors.append(
                f"{relative_path}:{line_number} ({function_name}): "
                "test environment mutation needs a SAFETY comment"
            )
        if not test_contract(relative_path, lines, line_index, function):
            errors.append(
                f"{relative_path}:{line_number} ({function_name}): "
                "test environment mutation is not covered by #[serial]"
            )

    # Test/build call sites have all been audited; no old placeholder should
    # remain there. Deferred runtime sites are checked above instead.
    if not is_deferred and not is_build_script:
        for index, line in enumerate(lines):
            if TODO in line:
                errors.append(
                    f"{relative_path}:{index + 1}: stale environment audit TODO remains"
                )
    return errors


def check_synchronous_bootstrap_order(root: Path) -> list[str]:
    main_path = root / "crates/mesh-llm/src/main.rs"
    if not main_path.is_file():
        return ["crates/mesh-llm/src/main.rs: bootstrap caller is missing"]
    text = main_path.read_text(encoding="utf-8")
    start = text.find("fn main()")
    mutation = text.find("configure_metal_pipeline_cache();", start)
    thread_start = text.find("run_on_application_thread(", start)
    runtime_build = text.find("tokio::runtime::Builder::new_multi_thread()", start)
    if min(start, mutation, thread_start, runtime_build) < 0:
        return ["crates/mesh-llm/src/main.rs: synchronous Metal bootstrap markers are missing"]
    if not (start < mutation < thread_start < runtime_build):
        return [
            "crates/mesh-llm/src/main.rs: Metal cache environment mutation must run "
            "before application-thread and Tokio runtime construction"
        ]
    return []


def discover_mutation_files(root: Path) -> dict[str, int]:
    discovered: dict[str, int] = {}
    for path in root.rglob("*.rs"):
        if any(part in {".git", "target"} for part in path.relative_to(root).parts):
            continue
        count = len(mutation_lines(path.read_text(encoding="utf-8").splitlines()))
        if count:
            discovered[path.relative_to(root).as_posix()] = count
    return discovered


def run(root: Path, files: tuple[str, ...] | None = None) -> int:
    errors: list[str] = []
    if files is not None:
        mutation_count = 0
        for relative_path in files:
            path = root / relative_path
            if path.is_file():
                mutation_count += len(mutation_lines(path.read_text(encoding="utf-8").splitlines()))
            errors.extend(check_file(root, relative_path))
        checked_file_count = len(files)
        audited_file_count = len(files)
    else:
        discovered = discover_mutation_files(root)
        registered = set(AUDITED_FILES) | set(KNOWN_UNAUDITED_MUTATION_COUNTS)
        for relative_path in sorted(set(discovered) - registered):
            errors.append(
                f"{relative_path}: unregistered process-environment mutation file "
                f"({discovered[relative_path]} sites)"
            )
        for relative_path, expected_count in KNOWN_UNAUDITED_MUTATION_COUNTS.items():
            actual_count = discovered.get(relative_path)
            if actual_count is not None and actual_count != expected_count:
                errors.append(
                    f"{relative_path}: unaudited mutation census changed "
                    f"from {expected_count} to {actual_count} sites"
                )
        for relative_path in AUDITED_FILES:
            if relative_path in discovered or (root / relative_path).is_file():
                errors.extend(check_file(root, relative_path))
        errors.extend(check_synchronous_bootstrap_order(root))
        mutation_count = sum(discovered.values())
        checked_file_count = len(discovered)
        audited_file_count = sum(relative_path in discovered for relative_path in AUDITED_FILES)

    if errors:
        print("environment mutation contract violations:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"environment mutation contract: discovered {checked_file_count} Rust files and "
        f"{mutation_count} mutation sites; {audited_file_count} contract-audited files; "
        "unresolved runtime sites remain explicit"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the checkout containing this script)",
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        help="strictly audit one relative source file (repeatable; defaults to repository discovery)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = tuple(args.files) if args.files else None
    raise SystemExit(run(args.root.resolve(), selected))
