#!/usr/bin/env python3
"""Build the versioned, provider-neutral PR/main CI plan.

The checked-in ``ci/ownership.yml`` and ``ci/slices.yml`` files intentionally
use JSON-compatible YAML.  Keeping the syntax in the JSON subset means the
planner has no undeclared Python package dependency on a hosted runner while
remaining consumable by normal YAML tooling.

This planner is the eligibility boundary for both PR and main CI. It is strict:
an invalid manifest, unknown path, missing producer, or malformed matrix is an
error rather than an implicit broad build.

``--manifest-root`` is the audit path, not a convenience.  The planner itself
always runs from the protected default-branch checkout, which is the authority:
by default it reads the catalogs beside its own code.  A PR caller instead
extracts the source revision's ``ci/ownership.yml`` and ``ci/slices.yml`` into a
runner-temp directory and points ``--manifest-root`` at it, so the routing
inputs that reviewers see on the PR are the ones the plan is audited against
while the executable planner, Cargo workspace discovery, and every fan-out
ceiling stay protected.  The caller first requires those source catalogs to
match the protected copies byte for byte.  In that protected PR path, the
comparison prevents the switch from widening routing.  Do not collapse it into
a single read of the source tree, and do not delete it as redundant: the two
roots are deliberately different trust domains.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OWNERSHIP_PATH = ROOT / "ci" / "ownership.yml"
SLICES_PATH = ROOT / "ci" / "slices.yml"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PROFILES = ("pr-draft", "pr-ready", "main", "manual-full")
EVENTS = ("pull_request", "push", "workflow_dispatch")
CACHE_MODES = ("none", "pr-isolated", "trusted-readonly", "trusted-readwrite")


class PlanError(ValueError):
    """Raised when the plan input or checked-in CI manifests are invalid."""


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"unable to load {path}: {error}") from error
    if not isinstance(value, dict):
        raise PlanError(f"{path} must contain an object")
    if value.get("schema_version") != 1:
        raise PlanError(f"{path} has unsupported schema_version")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlanError(f"{field} must be a non-empty string")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise PlanError(f"{field} must be an array of non-empty strings")
    if len(set(value)) != len(value):
        raise PlanError(f"{field} must not contain duplicates")
    return list(value)


def _validate_manifests(ownership: dict[str, Any], slices: dict[str, Any]) -> None:
    domains = _string_list(ownership.get("domains"), "ownership.domains")
    domain_set = set(domains)
    if ownership.get("unknown_path_policy") != "fail":
        raise PlanError("ownership.unknown_path_policy must be 'fail'")

    path_rules = ownership.get("path_rules")
    if not isinstance(path_rules, list):
        raise PlanError("ownership.path_rules must be an array")
    for index, rule in enumerate(path_rules):
        if not isinstance(rule, dict):
            raise PlanError(f"ownership.path_rules[{index}] must be an object")
        domain = _nonempty_string(rule.get("domain"), f"path_rules[{index}].domain")
        if domain not in domain_set:
            raise PlanError(f"path rule references unknown domain {domain!r}")
        patterns = _string_list(rule.get("patterns"), f"path_rules[{index}].patterns")
        if not patterns:
            raise PlanError(f"path rule {domain!r} has no patterns")

    crate_rules = ownership.get("crate_rules")
    if not isinstance(crate_rules, list):
        raise PlanError("ownership.crate_rules must be an array")
    for index, rule in enumerate(crate_rules):
        if not isinstance(rule, dict):
            raise PlanError(f"ownership.crate_rules[{index}] must be an object")
        domain = _nonempty_string(rule.get("domain"), f"crate_rules[{index}].domain")
        if domain not in domain_set:
            raise PlanError(f"crate rule references unknown domain {domain!r}")
        _string_list(rule.get("crates"), f"crate_rules[{index}].crates")

    profiles = slices.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(PROFILES):
        raise PlanError("slices.profiles must define exactly the four supported profiles")

    slice_defs = slices.get("slices")
    if not isinstance(slice_defs, list):
        raise PlanError("slices.slices must be an array")
    slice_ids: list[str] = []
    for index, definition in enumerate(slice_defs):
        if not isinstance(definition, dict):
            raise PlanError(f"slices.slices[{index}] must be an object")
        slice_id = _nonempty_string(definition.get("id"), f"slices.slices[{index}].id")
        slice_ids.append(slice_id)
        _nonempty_string(definition.get("kind"), f"slice {slice_id}.kind")
        _nonempty_string(definition.get("runner_role"), f"slice {slice_id}.runner_role")
        cache_mode = _nonempty_string(definition.get("cache_mode"), f"slice {slice_id}.cache_mode")
        if cache_mode not in CACHE_MODES:
            raise PlanError(f"slice {slice_id} has invalid cache_mode {cache_mode!r}")
        dependencies = _string_list(
            definition.get("depends_on", []),
            f"slice {slice_id}.depends_on",
        )
        if slice_id in dependencies:
            raise PlanError(f"slice {slice_id} cannot depend on itself")
    if len(set(slice_ids)) != len(slice_ids):
        raise PlanError("slices.slices contains duplicate IDs")
    slice_set = set(slice_ids)

    for profile, definition in profiles.items():
        if not isinstance(definition, dict):
            raise PlanError(f"profile {profile} must be an object")
        for field in ("base_slices", "control_slices"):
            selected = _string_list(definition.get(field), f"profile {profile}.{field}")
            unknown = sorted(set(selected) - slice_set)
            if unknown:
                raise PlanError(f"profile {profile}.{field} references {unknown}")
        if type(definition.get("all_rows")) is not bool:
            raise PlanError(f"profile {profile}.all_rows must be boolean")
        budgets = definition.get("budgets")
        if not isinstance(budgets, dict):
            raise PlanError(f"profile {profile}.budgets must be an object")
        budget_keys = {
            "linux_max_parallel",
            "macos_max_parallel",
            "windows_max_parallel",
            "total_max_workers",
        }
        if set(budgets) != budget_keys:
            raise PlanError(f"profile {profile}.budgets has an unexpected shape")
        for key in budget_keys:
            if type(budgets.get(key)) is not int or budgets[key] < 1:
                raise PlanError(f"profile {profile}.budgets.{key} must be positive")
        platform_workers = sum(
            budgets[key]
            for key in (
                "linux_max_parallel",
                "macos_max_parallel",
                "windows_max_parallel",
            )
        )
        if platform_workers > budgets["total_max_workers"]:
            raise PlanError(
                f"profile {profile}.budgets platform ceilings exceed total_max_workers"
            )

    domain_rules = slices.get("domain_rules")
    if not isinstance(domain_rules, list):
        raise PlanError("slices.domain_rules must be an array")
    seen_domains: set[str] = set()
    for index, rule in enumerate(domain_rules):
        if not isinstance(rule, dict):
            raise PlanError(f"slices.domain_rules[{index}] must be an object")
        domain = _nonempty_string(rule.get("domain"), f"domain_rules[{index}].domain")
        if domain not in domain_set:
            raise PlanError(f"slice rule references unknown domain {domain!r}")
        if domain in seen_domains:
            raise PlanError(f"duplicate slice rule for domain {domain!r}")
        seen_domains.add(domain)
        selected = _string_list(rule.get("slices"), f"domain rule {domain}.slices")
        unknown = sorted(set(selected) - slice_set)
        if unknown:
            raise PlanError(f"domain {domain} references unknown slices {unknown}")
    if seen_domains != domain_set:
        raise PlanError(
            "slice domain rules do not cover: "
            + ", ".join(sorted(domain_set - seen_domains))
        )

    row_catalogs = {
        "domain_rows": _validate_rows(
            slices,
            "runtime_rows",
            required=("platform", "architecture", "runner_role"),
        ),
        "platform_domain_rows": _validate_rows(
            slices, "platform_rows", required=("platform", "architecture")
        ),
        "sdk_domain_rows": _validate_rows(
            slices,
            "sdk_rows",
            required=("language", "platform", "architecture"),
        ),
        "smoke_domain_rows": _validate_rows(slices, "smoke_rows", required=("kind",)),
    }
    batch_limits = slices.get("batch_limits")
    if not isinstance(batch_limits, dict):
        raise PlanError("slices.batch_limits must be an object")
    for key in ("clippy", "rust_tests"):
        if type(batch_limits.get(key)) is not int or batch_limits[key] < 1:
            raise PlanError(f"slices.batch_limits.{key} must be positive")

    for field in ("domain_rows", "smoke_domain_rows", "platform_domain_rows", "sdk_domain_rows"):
        mapping = slices.get(field)
        if not isinstance(mapping, dict):
            raise PlanError(f"slices.{field} must be an object")
        for domain, row_ids in mapping.items():
            if domain not in domain_set:
                raise PlanError(f"{field} references unknown domain {domain!r}")
            mapped_ids = _string_list(row_ids, f"{field}.{domain}")
            unknown = sorted(set(mapped_ids) - row_catalogs[field])
            if unknown:
                raise PlanError(f"{field}.{domain} references unknown rows {unknown}")

    dependencies_by_id = {
        definition["id"]: definition.get("depends_on", [])
        for definition in slice_defs
    }
    for slice_id, dependencies in dependencies_by_id.items():
        unknown = sorted(set(dependencies) - slice_set)
        if unknown:
            raise PlanError(f"slice {slice_id} depends on unknown slices {unknown}")
    _assert_acyclic(dependencies_by_id)


def _validate_rows(
    slices: dict[str, Any],
    field: str,
    *,
    required: tuple[str, ...],
) -> set[str]:
    rows = slices.get(field)
    if not isinstance(rows, list):
        raise PlanError(f"slices.{field} must be an array")
    ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PlanError(f"slices.{field}[{index}] must be an object")
        row_id = _nonempty_string(row.get("id"), f"{field}[{index}].id")
        ids.append(row_id)
        for key in required:
            _nonempty_string(row.get(key), f"{field}[{index}].{key}")
    if len(set(ids)) != len(ids):
        raise PlanError(f"slices.{field} contains duplicate IDs")
    return set(ids)


def _assert_acyclic(dependencies: dict[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise PlanError(f"slice dependency cycle includes {node!r}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in dependencies[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in dependencies:
        visit(node)


def _normalise_changed_files(raw: object) -> list[str]:
    files = _string_list(raw, "changed_files")
    normalised: list[str] = []
    for path in files:
        candidate = path[2:] if path.startswith("./") else path
        if candidate == "__force_all__":
            normalised.append(candidate)
            continue
        if candidate.startswith("/") or "\\" in candidate:
            raise PlanError(f"changed file is not a repository-relative POSIX path: {path!r}")
        parts = Path(candidate).parts
        if ".." in parts or candidate in {"", "."}:
            raise PlanError(f"changed file is not a normal repository path: {path!r}")
        normalised.append(candidate)
    return list(dict.fromkeys(normalised))


def _workspace_packages(root: Path, raw: object) -> list[dict[str, str]]:
    if raw is not None:
        if not isinstance(raw, list):
            raise PlanError("workspace_packages must be an array")
        packages: list[dict[str, str]] = []
        names: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise PlanError(f"workspace_packages[{index}] must be an object")
            name = _nonempty_string(item.get("name"), f"workspace_packages[{index}].name")
            path = _nonempty_string(item.get("path"), f"workspace_packages[{index}].path")
            if name in names:
                raise PlanError(f"workspace_packages contains duplicate {name!r}")
            names.add(name)
            packages.append({"name": name, "path": path.rstrip("/")})
        return packages

    result = subprocess.run(
        ["cargo", "metadata", "--format-version=1", "--no-deps"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PlanError(f"cargo metadata failed: {result.stderr.strip()}")
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PlanError(f"cargo metadata emitted invalid JSON: {error}") from error

    workspace_members = set(metadata.get("workspace_members", []))
    workspace_root = Path(metadata.get("workspace_root", root)).resolve()
    packages = []
    for package in metadata.get("packages", []):
        if package.get("id") not in workspace_members:
            continue
        manifest = Path(package["manifest_path"]).resolve()
        package_path = manifest.parent.relative_to(workspace_root).as_posix()
        packages.append({"name": package["name"], "path": package_path or "."})
    if not packages:
        raise PlanError("cargo metadata returned no workspace packages")
    return packages


def _direct_crates(changed_files: Iterable[str], packages: list[dict[str, str]]) -> list[str]:
    direct: list[str] = []
    for package in packages:
        package_path = package["path"]
        if package_path == ".":
            continue
        prefix = package_path.rstrip("/") + "/"
        if any(path == package_path or path.startswith(prefix) for path in changed_files):
            direct.append(package["name"])
    return direct


def _affected_crates(
    *,
    root: Path,
    changed_files: list[str],
    packages: list[dict[str, str]],
    profile: str,
    raw: object,
) -> list[str]:
    workspace_names = [package["name"] for package in packages]
    if profile in {"main", "manual-full"}:
        affected = workspace_names
    elif raw is not None and raw != []:
        affected = _string_list(raw, "affected_crates")
    else:
        script = root / "scripts" / "affected-crates.sh"
        result = subprocess.run(
            ["bash", str(script), "--stdin"],
            cwd=root,
            input="\n".join(changed_files) + "\n",
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise PlanError(f"affected-crates.sh failed: {result.stderr.strip()}")
        try:
            affected = _string_list(json.loads(result.stdout).get("affected"), "affected_crates")
        except (json.JSONDecodeError, AttributeError) as error:
            raise PlanError(f"affected-crates.sh emitted invalid JSON: {error}") from error

    unknown = sorted(set(affected) - set(workspace_names))
    if unknown:
        raise PlanError(f"affected_crates contains non-workspace crates: {unknown}")
    return [name for name in workspace_names if name in set(affected)]


def _matched_domains(
    ownership: dict[str, Any],
    changed_files: list[str],
    direct_crates: list[str],
) -> list[str]:
    domains = _string_list(ownership["domains"], "ownership.domains")
    if "__force_all__" in changed_files:
        return domains
    matched: set[str] = set()
    unknown_paths: list[str] = []
    for path in changed_files:
        path_matches = {
            rule["domain"]
            for rule in ownership["path_rules"]
            if any(fnmatchcase(path, pattern) for pattern in rule["patterns"])
        }
        if not path_matches:
            unknown_paths.append(path)
        matched.update(path_matches)
    for crate in direct_crates:
        matched.update(
            rule["domain"]
            for rule in ownership["crate_rules"]
            if any(fnmatchcase(crate, pattern) for pattern in rule["crates"])
        )
    if unknown_paths:
        raise PlanError(
            "ownership has no rule for changed paths: " + ", ".join(unknown_paths)
        )
    return [domain for domain in domains if domain in matched]


def _make_batches(crates: list[str], bins: int) -> list[dict[str, Any]]:
    if not crates:
        return []
    weights = {
        "mesh-llm": 10,
        "mesh-llm-host-runtime": 10,
        "mesh-llm-embedded-runtime": 8,
        "mesh-llm-client": 6,
        "skippy-runtime": 5,
        "skippy-server": 5,
        "model-artifact": 4,
        "model-hf": 4,
        "openai-frontend": 4,
        "skippy-correctness": 4,
        "mesh-llm-api-server": 3,
        "mesh-llm-system": 3,
        "skippy-prompt": 3,
    }
    buckets = [{"idx": index, "weight": 0, "crates": []} for index in range(bins)]
    indexed = [
        (crate, weights.get(crate, 1), index)
        for index, crate in enumerate(dict.fromkeys(crates))
    ]
    indexed.sort(key=lambda item: (-item[1], item[2], item[0]))
    for crate, weight, _index in indexed:
        target = min(buckets, key=lambda bucket: (bucket["weight"], bucket["idx"]))
        target["crates"].append(crate)
        target["weight"] += weight
    return [
        {
            **bucket,
            "id": f"batch-{bucket['idx']}",
        }
        for bucket in buckets
        if bucket["crates"]
    ]


def _row_map(slices: dict[str, Any], field: str) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in slices[field]}


def _select_rows(
    slices: dict[str, Any],
    *,
    profile: str,
    domains: list[str],
    selected: set[str],
    force_all_rows: bool,
) -> dict[str, list[dict[str, Any]]]:
    runtime_rows = _row_map(slices, "runtime_rows")
    platform_rows = _row_map(slices, "platform_rows")
    sdk_rows = _row_map(slices, "sdk_rows")
    smoke_rows = _row_map(slices, "smoke_rows")

    if profile == "pr-draft" and not force_all_rows:
        return {
            "hosts": [],
            "runtime_products": [],
            "platform_checks": [],
            "sdk": [],
            "smoke": [],
        }

    if force_all_rows:
        runtime_ids = list(runtime_rows)
        platform_ids = list(platform_rows)
        sdk_ids = list(sdk_rows)
        smoke_ids = list(smoke_rows)
    else:
        runtime_ids: list[str] = []
        for domain in domains:
            runtime_ids.extend(slices.get("domain_rows", {}).get(domain, []))
        if "runtime-product" in selected and not runtime_ids:
            runtime_ids.extend(["linux-cpu"])
        platform_ids = []
        for domain in domains:
            platform_ids.extend(slices.get("platform_domain_rows", {}).get(domain, []))
        sdk_ids = []
        for domain in domains:
            sdk_ids.extend(slices.get("sdk_domain_rows", {}).get(domain, []))
        smoke_ids = []
        for domain in domains:
            smoke_ids.extend(slices.get("smoke_domain_rows", {}).get(domain, []))
        if "product-smoke" in selected and not smoke_ids:
            smoke_ids.append("product-integration-cpu")
        if profile == "pr-draft" and smoke_ids:
            smoke_ids = [
                row_id for row_id in smoke_ids if row_id == "product-integration-cpu"
            ] or [smoke_ids[0]]

    def unique_rows(mapping: dict[str, dict[str, Any]], ids: Iterable[str], field: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row_id in ids:
            if row_id in seen:
                continue
            if row_id not in mapping:
                raise PlanError(f"{field} references unknown row {row_id!r}")
            seen.add(row_id)
            result.append(dict(mapping[row_id]))
        return result

    runtime = unique_rows(runtime_rows, runtime_ids, "runtime_rows")
    macos_architectures = {
        row["architecture"] for row in runtime if row["platform"] == "macos"
    }
    if len(macos_architectures) > 1:
        raise PlanError(
            "macOS runtime_products must use one architecture until SDK and smoke consumers are row-scoped"
        )
    hosts: list[dict[str, Any]] = []
    seen_hosts: set[tuple[str, str]] = set()
    for row in runtime:
        key = (row["platform"], row["architecture"])
        if key in seen_hosts:
            continue
        seen_hosts.add(key)
        hosts.append(
            {
                "id": f"{row['platform']}-{row['architecture']}-host",
                "platform": row["platform"],
                "architecture": row["architecture"],
                "runner_role": row["runner_role"],
                "profile": row.get("profile", profile),
            }
        )
    return {
        "hosts": hosts,
        "runtime_products": runtime,
        "platform_checks": unique_rows(platform_rows, platform_ids, "platform_rows"),
        "sdk": unique_rows(sdk_rows, sdk_ids, "sdk_rows"),
        "smoke": unique_rows(smoke_rows, smoke_ids, "smoke_rows"),
    }


def _slice_definitions(slices: dict[str, Any]) -> OrderedDict[str, dict[str, Any]]:
    return OrderedDict((definition["id"], definition) for definition in slices["slices"])


def _planned_cache_mode(definition: dict[str, Any], profile: str) -> str:
    mode = definition["cache_mode"]
    if mode != "pr-isolated":
        return mode
    if profile == "main":
        return "trusted-readwrite"
    if profile == "manual-full":
        return "trusted-readonly"
    return mode


def _documentation_only(domains: list[str]) -> bool:
    return bool(domains) and set(domains) - {"ci-control"} == {"docs"}


def _signal_value(
    changed_files: list[str], domains: list[str], profile: str, name: str
) -> bool:
    """Return stable, planner-owned signals consumed by reusable workflows.

    Signals deliberately live next to the ownership decision.  This keeps the
    orchestrator declarative: workflow YAML selects a slice from the plan and
    never re-implements path matching in expressions.
    """
    force_all = "__force_all__" in changed_files
    if force_all or profile in {"main", "manual-full"}:
        if name == "docs_only":
            return False
        return True
    if name == "rust_changed":
        return any(domain in domains for domain in ("rust", "native-abi", "protocol", "split-serving", "model-download", "cli"))
    if name == "ui_changed":
        return "ui" in domains
    if name == "website_changed":
        return "website" in domains
    if name == "website_docs_changed":
        return any(
            path.startswith("website/src/docs/pages/")
            or path.startswith("website/src/_includes/")
            for path in changed_files
        )
    if name == "cli_surface_changed":
        return "cli" in domains
    if name == "docs_only":
        return _documentation_only(domains)
    if name == "backend_changed":
        return any(
            domain in domains
            for domain in (
                "native-abi",
                "runtime-product",
                "backend-cuda",
                "backend-rocm",
                "backend-vulkan",
            )
        )
    if name == "runner_contract_required":
        return "ci-control" in domains or "runner-infra" in domains
    raise PlanError(f"unknown planner signal {name!r}")


def _select_slices(
    slices: dict[str, Any],
    *,
    profile: str,
    domains: list[str],
    documentation_only: bool,
) -> tuple[list[str], dict[str, list[str]], bool]:
    profile_definition = slices["profiles"][profile]
    definitions = _slice_definitions(slices)
    selected = set(profile_definition["base_slices"])
    reasons: dict[str, list[str]] = {
        slice_id: ["profile:base"] for slice_id in selected
    }
    domain_rules = {rule["domain"]: rule["slices"] for rule in slices["domain_rules"]}
    if profile != "pr-draft" or (
        "ci-control" in domains and documentation_only
    ):
        for domain in domains:
            for slice_id in domain_rules[domain]:
                selected.add(slice_id)
                reasons.setdefault(slice_id, []).append(f"domain:{domain}")

    force_all_rows = bool(profile_definition["all_rows"])
    if ("ci-control" in domains and not documentation_only) or "runner-infra" in domains:
        force_all_rows = True
        for slice_id in profile_definition["control_slices"]:
            selected.add(slice_id)
            reasons.setdefault(slice_id, []).append("control-plane:fail-open")

    changed = True
    while changed:
        changed = False
        for slice_id in list(selected):
            for dependency in definitions[slice_id].get("depends_on", []):
                if dependency not in selected:
                    selected.add(dependency)
                    reasons.setdefault(dependency, []).append(f"dependency:{slice_id}")
                    changed = True

    ordered = [slice_id for slice_id in definitions if slice_id in selected]
    for slice_id in ordered:
        reasons[slice_id] = sorted(set(reasons.get(slice_id, ["planner"])))
    return ordered, reasons, force_all_rows


def _validate_input(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PlanError("planner input must be an object")
    allowed = {
        "profile",
        "event_name",
        "source_sha",
        "base_sha",
        "changed_files",
        "affected_crates",
        "workspace_packages",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise PlanError("planner input has unknown fields: " + ", ".join(unknown))
    for field in ("profile", "event_name", "source_sha", "base_sha", "changed_files"):
        if field not in payload:
            raise PlanError(f"planner input is missing {field}")
    profile = _nonempty_string(payload["profile"], "profile")
    if profile not in PROFILES:
        raise PlanError(f"unsupported profile {profile!r}")
    event_name = _nonempty_string(payload["event_name"], "event_name")
    if event_name not in EVENTS:
        raise PlanError(f"unsupported event_name {event_name!r}")
    source_sha = _nonempty_string(payload["source_sha"], "source_sha")
    if not SHA_RE.fullmatch(source_sha):
        raise PlanError("source_sha must be a lowercase 40-character SHA")
    base_sha = payload["base_sha"]
    if not isinstance(base_sha, str) or (base_sha and not SHA_RE.fullmatch(base_sha)):
        raise PlanError("base_sha must be empty or a lowercase 40-character SHA")
    if profile in {"pr-draft", "pr-ready"} and event_name != "pull_request":
        raise PlanError(f"profile {profile} requires pull_request")
    if profile == "main" and event_name not in {"push", "workflow_dispatch"}:
        raise PlanError("profile main requires push or workflow_dispatch")
    if profile == "manual-full" and event_name != "workflow_dispatch":
        raise PlanError("profile manual-full requires workflow_dispatch")
    return payload


def _validate_plan(plan: dict[str, Any], slices: dict[str, Any], packages: list[dict[str, str]]) -> None:
    required = {
        "schema_version",
        "profile",
        "event_name",
        "source_sha",
        "base_sha",
        "direct_crates",
        "affected_crates",
        "domains",
        "required_slices",
        "matrices",
        "reasons",
        "dependencies",
        "runner_roles",
        "cache_modes",
        "budgets",
        "signals",
    }
    if set(plan) != required:
        raise PlanError("planner emitted an unexpected plan shape")
    slice_defs = _slice_definitions(slices)
    selected = plan["required_slices"]
    if len(selected) != len(set(selected)) or any(item not in slice_defs for item in selected):
        raise PlanError("plan has duplicate or unknown required slices")
    for field in ("reasons", "dependencies", "runner_roles", "cache_modes"):
        value = plan[field]
        if not isinstance(value, dict) or set(value) != set(selected):
            raise PlanError(f"plan {field} keys must equal required_slices")
    if any(
        not isinstance(items, list)
        or not items
        or any(not isinstance(item, str) or not item for item in items)
        for items in plan["reasons"].values()
    ):
        raise PlanError("plan reasons must contain non-empty string arrays")
    for slice_id in selected:
        dependencies = plan["dependencies"][slice_id]
        expected = slice_defs[slice_id].get("depends_on", [])
        if set(dependencies) != set(expected) or not set(expected).issubset(selected):
            raise PlanError(f"plan has an invalid dependency closure for {slice_id}")
        if plan["runner_roles"][slice_id] != slice_defs[slice_id]["runner_role"]:
            raise PlanError(f"plan runner role drift for {slice_id}")
        if plan["cache_modes"][slice_id] != _planned_cache_mode(
            slice_defs[slice_id], plan["profile"]
        ):
            raise PlanError(f"plan cache mode drift for {slice_id}")
    expected_budgets = slices["profiles"][plan["profile"]]["budgets"]
    if plan["budgets"] != expected_budgets:
        raise PlanError("plan budget drift for selected profile")
    planned_platform_workers = sum(
        plan["budgets"][key]
        for key in (
            "linux_max_parallel",
            "macos_max_parallel",
            "windows_max_parallel",
        )
    )
    if planned_platform_workers > plan["budgets"]["total_max_workers"]:
        raise PlanError("plan platform ceilings exceed total_max_workers")
    workspace_names = {package["name"] for package in packages}
    for field in ("direct_crates", "affected_crates"):
        if not set(plan[field]).issubset(workspace_names):
            raise PlanError(f"plan {field} contains a non-workspace crate")
    signals = plan["signals"]
    if not isinstance(signals, dict):
        raise PlanError("plan signals must be an object")
    expected_signals = {
        "rust_changed",
        "ui_changed",
        "website_changed",
        "website_docs_changed",
        "cli_surface_changed",
        "docs_only",
        "backend_changed",
        "runner_contract_required",
    }
    if set(signals) != expected_signals or any(type(value) is not bool for value in signals.values()):
        raise PlanError("plan signals have an unexpected shape")
    for matrix_name, rows in plan["matrices"].items():
        ids = [row.get("id") for row in rows]
        if any(not isinstance(row_id, str) or not row_id for row_id in ids):
            raise PlanError(f"plan matrix {matrix_name} has an invalid row")
        if len(ids) != len(set(ids)):
            raise PlanError(f"plan matrix {matrix_name} contains duplicate rows")
    if "rust-tests" in selected and plan["profile"] in {"main", "manual-full"}:
        tested = {
            crate
            for batch in plan["matrices"]["rust_tests"]
            for crate in batch.get("crates", [])
        }
        if tested != workspace_names:
            raise PlanError("main rust test matrix does not cover every workspace crate")


def build_plan(
    payload: object,
    *,
    root: Path = ROOT,
    manifest_root: Path | None = None,
) -> dict[str, Any]:
    input_data = _validate_input(payload)
    manifest_source = root if manifest_root is None else manifest_root
    ownership = _load_manifest(manifest_source / "ci" / "ownership.yml")
    slices = _load_manifest(manifest_source / "ci" / "slices.yml")
    _validate_manifests(ownership, slices)

    profile = input_data["profile"]
    changed_files = _normalise_changed_files(input_data["changed_files"])
    packages = _workspace_packages(root, input_data.get("workspace_packages"))
    direct_crates = _direct_crates(changed_files, packages)
    affected_crates = _affected_crates(
        root=root,
        changed_files=changed_files,
        packages=packages,
        profile=profile,
        raw=input_data.get("affected_crates"),
    )
    domains = _matched_domains(ownership, changed_files, direct_crates)
    documentation_only = _documentation_only(domains)
    required_slices, reasons, force_all_rows = _select_slices(
        slices,
        profile=profile,
        domains=domains,
        documentation_only=documentation_only,
    )

    matrices = _select_rows(
        slices,
        profile=profile,
        domains=domains,
        selected=set(required_slices),
        force_all_rows=force_all_rows,
    )
    all_rust = force_all_rows or profile in {"main", "manual-full"}
    rust_scope = affected_crates or ([package["name"] for package in packages] if all_rust else [])
    batch_limits = slices["batch_limits"]
    matrices["clippy"] = (
        _make_batches(rust_scope, batch_limits["clippy"])
        if "quality" in required_slices and ("rust" in domains or all_rust)
        else []
    )
    matrices["rust_tests"] = (
        _make_batches(rust_scope, batch_limits["rust_tests"])
        if "rust-tests" in required_slices
        else []
    )

    signal_names = (
        "rust_changed",
        "ui_changed",
        "website_changed",
        "website_docs_changed",
        "cli_surface_changed",
        "docs_only",
        "backend_changed",
        "runner_contract_required",
    )

    definitions = _slice_definitions(slices)
    plan = {
        "schema_version": 1,
        "profile": profile,
        "event_name": input_data["event_name"],
        "source_sha": input_data["source_sha"],
        "base_sha": input_data["base_sha"],
        "direct_crates": direct_crates,
        "affected_crates": affected_crates,
        "domains": domains,
        "required_slices": required_slices,
        "matrices": matrices,
        "reasons": reasons,
        "dependencies": {
            slice_id: list(definitions[slice_id].get("depends_on", []))
            for slice_id in required_slices
        },
        "runner_roles": {
            slice_id: definitions[slice_id]["runner_role"] for slice_id in required_slices
        },
        "cache_modes": {
            slice_id: _planned_cache_mode(definitions[slice_id], profile)
            for slice_id in required_slices
        },
        "budgets": dict(slices["profiles"][profile]["budgets"]),
        "signals": {
            name: _signal_value(changed_files, domains, profile, name)
            for name in signal_names
        },
    }
    _validate_plan(plan, slices, packages)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path)
    arguments = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        plan = build_plan(payload, manifest_root=arguments.manifest_root)
    except (json.JSONDecodeError, PlanError) as error:
        print(f"ERROR: unable to build CI plan: {error}", file=sys.stderr)
        return 2
    print(json.dumps(plan, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
