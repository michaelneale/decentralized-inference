#!/usr/bin/env python3
"""Inspect checked-in runner identities without changing CI execution.

Examples:
  python3 scripts/runner-image-identity.py check
  python3 scripts/runner-image-identity.py lookup ui-quality --field reference
  python3 scripts/runner-image-identity.py seed-key --recipe-hash <hash>
  python3 scripts/runner-image-identity.py diagnose

Historical images have no verified tool observations or provenance. A null
receipt means unknown, not verified or compatible. Compiler workload coverage
is independent of image compatibility; this catalog never authorizes a restore.

Workflow inspection covers .yml and .yaml files and deliberately supports only
the existing block-style job, container and CUDA matrix declarations. Runner
references in unsupported declarations fail closed. This is not a general YAML
parser or a workflow generator. Planner
rows are checked through the repository's real, stdlib-only selection function;
no Cargo metadata, builds, network access, or cache operations are performed.
Consequently --root must be a trusted executable checkout, not an untrusted PR
manifest directory. This tool is not wired into the protected planner workflow.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "ghcr.io/mesh-llm/mesh-llm-cuda-runner"
REFERENCE = re.compile(re.escape(REPOSITORY) + r"@sha256:([0-9a-f]{64})\Z")
IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]*\Z")
EPOCH_PREFIX = "mesh-llm-cuda-runner-sha256-"
_evidence_spec = importlib.util.spec_from_file_location("runner_image_evidence", Path(__file__).with_name("runner-image-evidence.py"))
EVIDENCE = importlib.util.module_from_spec(_evidence_spec)
_evidence_spec.loader.exec_module(EVIDENCE)


class IdentityError(ValueError):
    """A catalog or checked-in consumer disagrees with the identity contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IdentityError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = EVIDENCE.decode(EVIDENCE.read_bytes(path))
    require(isinstance(value, dict), f"{path}: expected an object")
    return value


def fields(value: Any, expected: str, where: str) -> None:
    require(isinstance(value, dict), f"{where}: expected an object")
    require(set(value) == set(expected.split()), f"{where}: unexpected or missing fields")


def named_map(value: Any, where: str) -> None:
    require(isinstance(value, dict) and bool(value), f"{where}: expected a nonempty object")
    require(all(isinstance(key, str) and IDENTIFIER.fullmatch(key) for key in value),
            f"{where}: invalid identifier")


def nonempty(value: Any, where: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{where}: expected a string")


def validate(catalog: dict[str, Any], root: Path = ROOT) -> None:
    fields(catalog, "schema_version images consumer_roles runtime_rows compiler_seed sdk_rust", "catalog")
    require(type(catalog["schema_version"]) is int and catalog["schema_version"] == 1,
            "unsupported schema_version")
    images = catalog["images"]
    named_map(images, "images")
    references = []
    for image_id, image in images.items():
        fields(image, "reference environment backend native_toolchain_epoch receipt provenance", image_id)
        require(isinstance(image["reference"], str) and bool(REFERENCE.fullmatch(image["reference"])),
                f"{image_id}: image must be an immutable runner digest")
        references.append(image["reference"])
        require(image["environment"] == "public", f"{image_id}: unsupported environment")
        require(image["backend"] in ("cpu", "web", "cuda12", "cuda13", "rocm", "vulkan", "ui", "browser"),
                f"{image_id}: unsupported backend")
        epoch = image["native_toolchain_epoch"]
        require(epoch is None or epoch == EPOCH_PREFIX + digest(image), f"{image_id}: epoch/digest mismatch")
    try:
        EVIDENCE.validate_catalog(catalog, root)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise IdentityError(str(error)) from error
    require(len(references) == len(set(references)), "duplicate image reference")
    roles = catalog["consumer_roles"]
    named_map(roles, "consumer_roles")
    bindings = []
    for role_id, role in roles.items():
        fields(role, "image_id scope bindings", role_id)
        require(role["image_id"] in images, f"{role_id}: unknown image_id")
        require(role["scope"] in ("ordinary", "release"), f"{role_id}: invalid scope")
        require(isinstance(role["bindings"], list) and bool(role["bindings"]), f"{role_id}: no bindings")
        for binding in role["bindings"]:
            fields(binding, "workflow job matrix_selector image_expression epoch_field", role_id)
            require(isinstance(binding["workflow"], str) and
                    bool(re.fullmatch(r"[a-z0-9_-]+\.ya?ml", binding["workflow"])), f"{role_id}: invalid workflow")
            require(isinstance(binding["job"], str) and bool(IDENTIFIER.fullmatch(binding["job"])),
                    f"{role_id}: invalid job")
            require((binding["workflow"] == "release.yml") == (role["scope"] == "release"),
                    f"{role_id}: release scope mismatch")
            selector = binding["matrix_selector"]
            require(selector is None or selector in ({"cuda_major": "12"}, {"cuda_major": "13"}),
                    f"{role_id}: unsupported matrix selector")
            expression = binding["image_expression"]
            require(isinstance(expression, str) and expression.count("{image}") == 1 and "\n" not in expression,
                    f"{role_id}: image_expression must contain one image placeholder")
            require(binding["epoch_field"] in (None, "pinned_epoch", "toolchain_epoch"),
                    f"{role_id}: invalid epoch_field")
            require(not binding["epoch_field"] or images[role["image_id"]]["native_toolchain_epoch"] is not None,
                    f"{role_id}: native epoch is unknown")
            bindings.append((binding["workflow"], binding["job"], json.dumps(selector, sort_keys=True)))
    require(len(bindings) == len(set(bindings)), "duplicate consumer binding")
    named_map(catalog["runtime_rows"], "runtime_rows")
    for row_id, row in catalog["runtime_rows"].items():
        fields(row, "image_id platform architecture backend", row_id)
        require(row["image_id"] in images, f"{row_id}: unknown image_id")
        require(row["platform"] == "linux" and row["architecture"] in ("amd64", "arm64"),
                f"{row_id}: unsupported runtime platform")
        require(row["backend"] in ("cpu", "cuda", "rocm", "vulkan"), f"{row_id}: unsupported runtime backend")
        require(images[row["image_id"]]["backend"].startswith(row["backend"]), f"{row_id}: image/backend mismatch")
        require(images[row["image_id"]]["native_toolchain_epoch"] is not None, f"{row_id}: unknown native epoch")
    seed = catalog["compiler_seed"]
    fields(seed, "image_id architecture key_prefix recipe_inputs recipe workload_coverage publisher_role consumer_roles runtime_consumer", "compiler_seed")
    require(seed["image_id"] in images, "compiler_seed: unknown image_id")
    require(seed["architecture"] == "amd64", "compiler_seed: unsupported architecture")
    short_digest = digest(images[seed["image_id"]])[:8]
    require(isinstance(seed["key_prefix"], str) and bool(re.fullmatch(
        f"mesh-llm-sccache-seed-linux-x86_64-img-{short_digest}-epoch-{short_digest}-v[0-9]+-",
        seed["key_prefix"])), "compiler_seed: key prefix/image mismatch")
    require(isinstance(seed["recipe_inputs"], list) and bool(seed["recipe_inputs"]) and
            all(isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_./*?-]+", item) for item in seed["recipe_inputs"]),
            "compiler_seed: invalid recipe_inputs")
    require(len(seed["recipe_inputs"]) == len(set(seed["recipe_inputs"])), "compiler_seed: duplicate recipe input")
    require(isinstance(seed["recipe"], str) and bool(IDENTIFIER.fullmatch(seed["recipe"])), "compiler_seed: invalid recipe")
    require(seed["workload_coverage"] is None, "compiler_seed: workload coverage has not been qualified")
    require(isinstance(seed["consumer_roles"], list) and bool(seed["consumer_roles"]) and
            len(seed["consumer_roles"]) == len(set(seed["consumer_roles"])), "compiler_seed: invalid consumer roles")
    for role_id in [seed["publisher_role"], *seed["consumer_roles"]]:
        require(role_id in roles and roles[role_id]["image_id"] == seed["image_id"] and
                roles[role_id]["scope"] == "ordinary", f"compiler_seed: incompatible role {role_id}")
    fields(seed["runtime_consumer"], "workflow job row_id", "compiler_seed.runtime_consumer")
    runtime = seed["runtime_consumer"]
    require(isinstance(runtime["workflow"], str) and bool(re.fullmatch(r"[a-z0-9_-]+\.ya?ml", runtime["workflow"])) and
            isinstance(runtime["job"], str) and bool(IDENTIFIER.fullmatch(runtime["job"])), "compiler_seed: invalid runtime consumer")
    require(runtime["row_id"] in catalog["runtime_rows"] and
            catalog["runtime_rows"][runtime["row_id"]]["image_id"] == seed["image_id"], "compiler_seed: runtime image mismatch")
    sdk = catalog["sdk_rust"]
    fields(sdk, "role target rust_action_ref toolchain_epoch profile_linker cache_namespace cache_recipe_inputs", "sdk_rust")
    require(sdk["role"] in roles, "sdk_rust: unknown role")
    for key in ("target", "toolchain_epoch", "profile_linker", "cache_namespace"):
        nonempty(sdk[key], f"sdk_rust.{key}")
    require(isinstance(sdk["rust_action_ref"], str) and
            bool(re.fullmatch(r"dtolnay/rust-toolchain@[0-9a-f]{40}", sdk["rust_action_ref"])), "sdk_rust: action must be pinned")
    require(sdk["toolchain_epoch"].endswith(sdk["rust_action_ref"].split("@")[1]), "sdk_rust: action/epoch mismatch")
    require(isinstance(sdk["cache_recipe_inputs"], list) and bool(sdk["cache_recipe_inputs"]) and
            all(isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_./*?-]+", item) for item in sdk["cache_recipe_inputs"]),
            "sdk_rust: invalid cache_recipe_inputs")


def digest(image: dict[str, Any]) -> str:
    return image["reference"].rsplit(":", 1)[1]


def scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def one_field(text: str, name: str, where: str, indent: int | None = None) -> str:
    spacing = " " * indent if indent is not None else r" +"
    values = re.findall(r"^" + spacing + re.escape(name) + r":\s*(\S[^\n]*)$", text, re.MULTILINE)
    require(len(values) == 1, f"{where}: expected exactly one {name} declaration")
    return scalar(values[0])


def workflow_jobs(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    # Scan the complete file before selecting jobs, so shorthand/flow containers
    # and indirect literal declarations cannot disappear from the census. The
    # sole non-image occurrence today is the independently checked seed guard.
    for number, line in enumerate(text.splitlines(), 1):
        if REPOSITORY in line and not line.lstrip().startswith("#"):
            require(bool(re.match(r"^(?: {6}image| {12}runner_image| {10}allow_trusted_seed):[ \t]+\S", line)),
                    f"{path.name}:{number}: unsupported runner image reference declaration")
    starts = list(re.finditer(r"^jobs:\s*$", text, re.MULTILINE))
    require(len(starts) == 1, f"{path.name}: expected one block-style jobs mapping")
    body = text[starts[0].end():]
    # Stop before another top-level YAML key, if present.
    body = re.split(r"^\S", body, maxsplit=1, flags=re.MULTILINE)[0]
    matches = list(re.finditer(r"^  ([A-Za-z_][A-Za-z0-9_-]*):\s*$", body, re.MULTILINE))
    jobs = {}
    for index, match in enumerate(matches):
        name = match[1]
        require(name not in jobs, f"{path.name}: duplicate job {name}")
        jobs[name] = body[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(body)]
    require(bool(jobs), f"{path.name}: unsupported jobs mapping")
    return jobs


def matrix_row(job: str, selector: dict[str, str], where: str) -> str:
    rows = re.findall(r"^          - [^\n]+\n(?:^            [^\n]*\n)*", job, re.MULTILINE)
    matches = [row for row in rows if re.search(
        r"^            cuda_major: ['\"]?" + selector["cuda_major"] + r"['\"]?\s*$", row, re.MULTILINE)]
    require(len(matches) == 1, f"{where}: missing or duplicate CUDA matrix row {selector}")
    return matches[0]


def job_steps(job: str) -> list[str]:
    matches = list(re.finditer(r"^      - ", job, re.MULTILINE))
    return [job[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(job)]
            for index, match in enumerate(matches)]


def seed_expression(catalog: dict[str, Any]) -> str:
    seed = catalog["compiler_seed"]
    return seed["key_prefix"] + "${{ hashFiles(" + ", ".join(repr(item) for item in seed["recipe_inputs"]) + ") }}"


def sdk_cache_expression(catalog: dict[str, Any]) -> str:
    inputs = ", ".join(repr(item) for item in catalog["sdk_rust"]["cache_recipe_inputs"])
    return ("${{ format('{0}-sdk-rust-cargo-v1-{1}-{2}-{3}-image-{4}-{5}-{6}-{7}', "
            "env.CACHE_NAMESPACE, runner.os, runner.arch, env.SDK_RUST_TARGET, "
            "env.SDK_RUST_IMAGE_DIGEST, env.SDK_RUST_TOOLCHAIN_EPOCH, "
            "env.SDK_RUST_PROFILE_LINKER, hashFiles(" + inputs + ")) }}")


def planner_rows(root: Path) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("runner_identity_planner", root / "scripts/plan-ci.py")
    require(spec is not None and spec.loader is not None, "unable to load actual CI planner")
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    slices = read_json(root / "ci/slices.yml")
    planner._validate_manifests(read_json(root / "ci/ownership.yml"), slices)
    return planner._select_rows(slices, profile="main", domains=[], selected=set(), force_all_rows=True)["runtime_products"]


def check(catalog: dict[str, Any], root: Path) -> dict[str, int]:
    validate(catalog, root)
    workflow_paths = sorted(path for path in (root / ".github/workflows").iterdir() if path.suffix in (".yml", ".yaml"))
    workflows = {path.name: workflow_jobs(path) for path in workflow_paths}
    images, roles = catalog["images"], catalog["consumer_roles"]
    expected_locations: Counter[tuple[str, str]] = Counter()
    for role_id, role in roles.items():
        image = images[role["image_id"]]
        for binding in role["bindings"]:
            workflow, job_id = binding["workflow"], binding["job"]
            where = f"{workflow}:{job_id} ({role_id})"
            require(workflow in workflows and job_id in workflows[workflow], f"{where}: missing job")
            job = workflows[workflow][job_id]
            selected = job
            if binding["matrix_selector"] is not None:
                require(one_field(job, "image", where, 6) == "${{ matrix.runner_image }}", f"{where}: matrix image consumer drift")
                require(one_field(job, "pinned_epoch", where) == "${{ matrix.toolchain_epoch }}", f"{where}: matrix epoch consumer drift")
                selected = matrix_row(job, binding["matrix_selector"], where)
            field = "runner_image" if binding["matrix_selector"] else "image"
            actual = one_field(selected, field, where, None if binding["matrix_selector"] else 6)
            require(actual == binding["image_expression"].replace("{image}", image["reference"]), f"{where}: image reference drift")
            if binding["epoch_field"]:
                require(one_field(selected, binding["epoch_field"], where) == image["native_toolchain_epoch"], f"{where}: native toolchain epoch drift")
            expected_locations[(workflow, job_id)] += 1
    actual_locations: Counter[tuple[str, str]] = Counter()
    for workflow, jobs in workflows.items():
        for job_id, job in jobs.items():
            values = re.findall(r"^ +(?:image|runner_image):\s*([^\n]+)$", job, re.MULTILINE)
            actual_locations[(workflow, job_id)] += sum(REPOSITORY in value for value in values)
    require(+actual_locations == expected_locations, "runner image consumer census drift; register every literal image binding")

    runtime = catalog["compiler_seed"]["runtime_consumer"]
    runtime_job = workflows[runtime["workflow"]][runtime["job"]]
    require(one_field(runtime_job, "image", "runtime consumer", 6) == "${{ matrix.runtime.container_image }}", "runtime matrix image consumer drift")
    require(one_field(runtime_job, "pinned_epoch", "runtime consumer") == "${{ matrix.runtime.toolchain_epoch }}", "runtime matrix epoch consumer drift")
    rows = planner_rows(root)
    actual_rows = {row["id"]: row for row in rows if "container_image" in row}
    require(len(actual_rows) == sum("container_image" in row for row in rows), "duplicate planner image row")
    require(set(actual_rows) == set(catalog["runtime_rows"]), "planner image row census drift")
    for row_id, expected in catalog["runtime_rows"].items():
        actual, image = actual_rows[row_id], images[expected["image_id"]]
        for field in ("platform", "architecture", "backend"):
            require(actual[field] == expected[field], f"{row_id}: planner {field} drift")
        require(actual["container_image"] == image["reference"], f"{row_id}: planner image drift")
        require(actual["toolchain_epoch"] == image["native_toolchain_epoch"], f"{row_id}: planner epoch drift")

    seed = catalog["compiler_seed"]
    expression = seed_expression(catalog)
    expected_seed_jobs = []
    for role_id in [seed["publisher_role"], *seed["consumer_roles"]]:
        for binding in roles[role_id]["bindings"]:
            expected_seed_jobs.append((binding["workflow"], binding["job"]))
    expected_seed_jobs.append((runtime["workflow"], runtime["job"]))
    observed_seed_jobs = []
    observed_restore_jobs = []
    for workflow, jobs in workflows.items():
        for job_id, job in jobs.items():
            keys = re.findall(r"mesh-llm-sccache-seed-[^\n]*?\$\{\{[^\n]*?\}\}", job)
            if keys:
                require(keys == [expression], f"{workflow}:{job_id}: compiler seed key drift")
                observed_seed_jobs.append((workflow, job_id))
            for step in job_steps(job):
                if re.search(r"^ *(?:- )?uses: \./\.github/actions/restore-sccache-seed\s*$", step, re.MULTILINE):
                    require(one_field(step, "cache_key", f"{workflow}:{job_id}") == expression,
                            f"{workflow}:{job_id}: compiler seed restore key drift")
                    observed_restore_jobs.append((workflow, job_id))
    require(Counter(expected_seed_jobs) == Counter(observed_seed_jobs), "compiler seed producer/consumer census drift")
    publisher = roles[seed["publisher_role"]]["bindings"][0]
    publisher_id = (publisher["workflow"], publisher["job"])
    require(Counter(observed_restore_jobs) == Counter(item for item in expected_seed_jobs if item != publisher_id), "compiler seed restore action census drift")
    publisher_job = workflows[publisher_id[0]][publisher_id[1]]
    publisher_cache_actions = []
    for step in job_steps(publisher_job):
        cache_action = re.search(r"uses: actions/cache/(restore|save)@", step)
        if cache_action:
            require(one_field(step, "key", "compiler seed publisher") == "${{ steps.seed.outputs.key }}",
                    "compiler seed publisher cache key drift")
            publisher_cache_actions.append(cache_action[1])
    require(publisher_cache_actions == ["restore", "save"], "compiler seed publisher cache action census drift")
    require(f"run: just {seed['recipe']}\n" in publisher_job, "compiler seed recipe drift")
    recipe = (root / "just/ci.just").read_text(encoding="utf-8")
    require(bool(re.search(r"^" + re.escape(seed["recipe"]) + r":\s*$", recipe, re.MULTILINE)), "compiler seed recipe is missing")
    seed_image = images[seed["image_id"]]
    for field, expected in (("container_image", seed_image["reference"]), ("toolchain_epoch", seed_image["native_toolchain_epoch"])):
        matches = re.findall(r"matrix\.runtime\." + field + r" == '([^']+)'", runtime_job)
        require(matches == [expected], f"runtime seed guard {field} drift")

    sdk = catalog["sdk_rust"]
    sdk_binding = roles[sdk["role"]]["bindings"][0]
    sdk_workflow = (root / ".github/workflows" / sdk_binding["workflow"]).read_text(encoding="utf-8")
    sdk_image = images[roles[sdk["role"]]["image_id"]]
    for name, expected in (("TARGET", sdk["target"]), ("IMAGE_DIGEST", digest(sdk_image)),
                           ("TOOLCHAIN_EPOCH", sdk["toolchain_epoch"]), ("PROFILE_LINKER", sdk["profile_linker"])):
        require(one_field(sdk_workflow, "SDK_RUST_" + name, "sdk_rust", 2) == expected, f"SDK Rust {name} drift")
    action_refs = re.findall(r"uses: (dtolnay/rust-toolchain@\S+)", workflows[sdk_binding["workflow"]][sdk_binding["job"]])
    require(action_refs == [sdk["rust_action_ref"]], "SDK Rust action drift")
    require(one_field(sdk_workflow, "CACHE_NAMESPACE", "sdk_rust", 2) == sdk["cache_namespace"], "SDK Rust namespace drift")
    require(one_field(workflows[sdk_binding["workflow"]][sdk_binding["job"]], "prefix-key", "sdk_rust") ==
            sdk_cache_expression(catalog), "SDK Rust cache key expression drift")
    return {"images": len(images), "roles": len(roles), "workflow_bindings": sum(expected_locations.values()),
            "runtime_rows": len(actual_rows), "seed_consumers": len(observed_restore_jobs)}


def diagnose(catalog: dict[str, Any], root: Path) -> list[str]:
    """Report eligibility concerns separately; a matching image is not coverage."""
    runtime = catalog["compiler_seed"]["runtime_consumer"]
    job = workflow_jobs(root / ".github/workflows" / runtime["workflow"])[runtime["job"]]
    guard_architectures = re.findall(r"matrix\.runtime\.architecture == '([^']+)'", job)
    actual = next(row for row in planner_rows(root) if row["id"] == runtime["row_id"])["architecture"]
    messages = []
    if guard_architectures != [actual]:
        messages.append(f"runtime seed architecture guard {guard_architectures!r} differs from planner {actual!r}; eligibility requires a separate workload coverage decision")
    if catalog["compiler_seed"]["workload_coverage"] is None:
        messages.append("compiler seed workload coverage is unqualified; image identity alone does not establish warm-cache coverage")
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--catalog", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "check", "diagnose"):
        commands.add_parser(command)
    lookup = commands.add_parser("lookup")
    lookup.add_argument("role")
    lookup.add_argument("--field", choices=("reference", "native_toolchain_epoch", "receipt", "provenance"))
    key = commands.add_parser("seed-key")
    key.add_argument("--recipe-hash", required=True)
    binder = commands.add_parser("bind")
    binder.add_argument("--image-id", required=True)
    binder.add_argument("--cohort", type=Path, required=True)
    binder.add_argument("--anchor", type=Path, required=True)
    binder.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        catalog = read_json(args.catalog or args.root / "ci/runner-images.json")
        validate(catalog, args.root)
        if args.command == "bind":
            result = EVIDENCE.bind(catalog, args.image_id, args.cohort, args.anchor, args.output, args.root)
        elif args.command == "check":
            result: Any = check(catalog, args.root)
        elif args.command == "diagnose":
            result = diagnose(catalog, args.root)
        elif args.command == "lookup":
            require(args.role in catalog["consumer_roles"], f"unknown consumer role: {args.role}")
            role = catalog["consumer_roles"][args.role]
            result = catalog["images"][role["image_id"]]
            if args.field:
                result = result[args.field]
        elif args.command == "seed-key":
            require(bool(re.fullmatch(r"[0-9a-f]{64}", args.recipe_hash)), "recipe hash must be lowercase SHA-256")
            result = catalog["compiler_seed"]["key_prefix"] + args.recipe_hash
        else:
            result = {"schema_version": catalog["schema_version"], "valid": True}
        print(result if isinstance(result, str) else json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (IdentityError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"runner image identity: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
