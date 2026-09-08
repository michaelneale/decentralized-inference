#!/usr/bin/env python3
"""Agentic Replay: compare inference engines on ordered real-agent trajectories.

The runner creates detached worktrees, builds the release host and native
runtime for every requested ref, replays a deterministic subset of the pinned
Thoughtworks agentic-coding-trajectories corpus, and writes raw evidence,
tables, CSV, and dependency-free SVG charts.

Mesh ref arms are deliberately launched without context-size, lane-count,
KV-budget, or backend-tuning arguments. External llama.cpp, vLLM, and SGLang
arms use an explicit engine configuration so their capacity and model choices
are visible in the artifact. Their identity is the SHA-256 of the engine's
reported version. Client concurrency is offered by this runner and is not a
server startup setting.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import http.client
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlsplit


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agentic_replay_engines import (  # noqa: E402
    EngineArm,
    EngineConfig,
    engine_order_specs,
    engine_server_command,
    external_server_environment,
    labels_overlap,
    load_engine_config,
    verify_engine_arm,
)


def verified_version_sha256_by_label(
    builds: Sequence[dict[str, Any]],
) -> dict[str, str]:
    return {
        build["label"]: build["version_sha256"]
        for build in builds
        if build.get("engine", "mesh") != "mesh"
    }


REPO = Path(__file__).resolve().parents[1]
COMPETITIVE_CONFIG = REPO / "evals/skippy-competitive-benchmark.json"
TRAJECTORY_GENERATOR = REPO / "evals/agentic-trajectory-manifest.py"
DEFAULT_BASE_URL = "http://127.0.0.1:9337/v1"
DEFAULT_ENDPOINT = urlsplit(DEFAULT_BASE_URL)
DEFAULT_HOST = DEFAULT_ENDPOINT.hostname or "127.0.0.1"
DEFAULT_PORT = DEFAULT_ENDPOINT.port or 80
FORBIDDEN_STARTUP_OPTIONS = (
    "--ctx-size",
    "--generation-concurrency",
    "--generation-queue-capacity",
    "--max-vram",
    "--parallel",
)
COLORS = (
    "#0284c7",
    "#dc2626",
    "#16a34a",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
)


@dataclass(frozen=True)
class RefSpec:
    label: str
    ref: str
    commit: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not normalized:
        raise ValueError(f"cannot derive a safe label from {value!r}")
    return normalized


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def parse_ref_specs(repo: Path, values: Sequence[str]) -> list[RefSpec]:
    if len(values) < 2:
        raise ValueError("at least two --ref LABEL=GIT_REF values are required")
    specs: list[RefSpec] = []
    labels: set[str] = set()
    commits: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"ref must use LABEL=GIT_REF syntax: {value}")
        label, ref = value.split("=", 1)
        label, ref = slug(label), ref.strip()
        if not ref:
            raise ValueError(f"empty git ref for label {label}")
        if label in labels:
            raise ValueError(f"duplicate ref label: {label}")
        commit = git(repo, "rev-parse", f"{ref}^{{commit}}")
        if commit in commits:
            raise ValueError(f"multiple labels resolve to commit {commit}")
        labels.add(label)
        commits.add(commit)
        specs.append(RefSpec(label=label, ref=ref, commit=commit))
    return specs


def external_config(args: argparse.Namespace) -> EngineConfig | None:
    path = getattr(args, "engine_config", None)
    return load_engine_config(path) if path is not None else None


def combined_specs(
    mesh_specs: Sequence[RefSpec],
    config: EngineConfig | None,
    version_sha256_by_label: Mapping[str, str] | None = None,
) -> list[RefSpec]:
    if config is None:
        return list(mesh_specs)
    overlap = labels_overlap([spec.label for spec in mesh_specs], config)
    if overlap:
        raise ValueError(
            f"Mesh and external engine labels overlap: {', '.join(sorted(overlap))}"
        )
    return [
        *mesh_specs,
        *(
            RefSpec(**item)
            for item in engine_order_specs(config, version_sha256_by_label)
        ),
    ]


def build_resume_identity(build: dict[str, Any]) -> tuple[str, ...]:
    if build.get("engine", "mesh") == "mesh":
        return (
            build["commit"],
            build["binary_sha256"],
            build["runtime_sha256"],
        )
    return (build["version_sha256"],)


def ab_order(specs: Sequence[RefSpec], passes: int) -> list[tuple[int, RefSpec]]:
    if passes <= 0:
        raise ValueError("passes must be positive")
    ordered: list[tuple[int, RefSpec]] = []
    for pass_index in range(passes):
        pass_specs = specs if pass_index % 2 == 0 else tuple(reversed(specs))
        ordered.extend((pass_index, spec) for spec in pass_specs)
    return ordered


class CommandLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        log_path: Path,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "started_at": utc_now(),
            "cwd": str(cwd),
            "command": list(command),
            "log": str(log_path),
        }
        with self.path.open("a", encoding="utf-8") as command_log:
            command_log.write(json.dumps(event, sort_keys=True) + "\n")
        with log_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                list(command),
                cwd=cwd,
                env=env,
                text=True,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        event["completed_at"] = utc_now()
        event["exit_code"] = result.returncode
        with self.path.open("a", encoding="utf-8") as command_log:
            command_log.write(json.dumps(event, sort_keys=True) + "\n")
        if result.returncode:
            raise RuntimeError(
                f"command failed ({result.returncode}); see {log_path}: "
                + " ".join(command)
            )


def prepare_worktree(repo: Path, root: Path, spec: RefSpec) -> Path:
    path = root / f"{spec.label}-{spec.commit[:10]}"
    if path.exists():
        try:
            actual = git(path, "rev-parse", "HEAD")
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            raise RuntimeError(f"existing path is not a git worktree: {path}") from error
        if actual != spec.commit:
            raise RuntimeError(
                f"worktree {path} is at {actual}, expected {spec.commit}; "
                "choose a different --worktree-root"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(path), spec.commit],
            check=True,
        )
    if git(path, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"benchmark worktree is dirty: {path}")
    return path


def runtime_backend_kind(backend: str) -> str:
    return {"cuda-blackwell": "cuda", "hip": "rocm"}.get(backend, backend)


def find_runtime(root: Path, backend: str) -> Path:
    candidates: list[Path] = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            kind = manifest["runtime"]["backend"]["kind"]
        except (KeyError, json.JSONDecodeError):
            continue
        if kind == runtime_backend_kind(backend):
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one {backend} runtime under {root}, found {len(candidates)}"
        )
    return candidates[0]


def build_ref(
    spec: RefSpec,
    worktree: Path,
    backend: str,
    output: Path,
    commands: CommandLog,
    skip_build: bool,
) -> dict[str, Any]:
    binary = worktree / "target/release/mesh-llm"
    runtime_root = worktree / "dist/native-runtimes"
    if not skip_build:
        commands.run(
            ["just", "release-host-build"],
            cwd=worktree,
            log_path=output / "logs" / f"build-{spec.label}-host.log",
        )
        commands.run(
            ["just", "release-runtime-build", backend],
            cwd=worktree,
            log_path=output / "logs" / f"build-{spec.label}-runtime.log",
        )
    if not binary.is_file():
        raise FileNotFoundError(f"release host not found: {binary}")
    runtime = find_runtime(runtime_root, backend)
    actual_head = git(worktree, "rev-parse", "HEAD")
    if actual_head != spec.commit:
        raise RuntimeError(
            f"worktree moved during build: expected {spec.commit}, found {actual_head}"
        )
    return {
        "label": spec.label,
        "engine": "mesh",
        "ref": spec.ref,
        "commit": spec.commit,
        "worktree": str(worktree),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "runtime_root": str(runtime_root),
        "runtime": str(runtime),
        "runtime_sha256": tree_sha256(runtime),
        "backend": backend,
    }


def load_competitive_config() -> dict[str, Any]:
    return json.loads(COMPETITIVE_CONFIG.read_text(encoding="utf-8"))


def verify_dataset(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Thoughtworks parquet not found: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Thoughtworks parquet SHA-256 mismatch: expected={expected_sha256} actual={actual}"
        )


def cohort_metadata(cohorts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name, trajectories in cohorts.items():
        framework_trajectories: dict[str, int] = {}
        framework_assistant_turns: dict[str, int] = {}
        for trajectory in trajectories:
            framework = trajectory["agent_framework"]
            framework_trajectories[framework] = (
                framework_trajectories.get(framework, 0) + 1
            )
            framework_assistant_turns[framework] = (
                framework_assistant_turns.get(framework, 0)
                + assistant_turn_count(trajectory)
            )
        metadata[name] = {
            "trajectory_count": len(trajectories),
            "assistant_turns": sum(assistant_turn_count(item) for item in trajectories),
            "framework_trajectories": framework_trajectories,
            "framework_assistant_turns": framework_assistant_turns,
            "session_ids_sha256": hashlib.sha256(
                "\n".join(item["session_id"] for item in trajectories).encode()
            ).hexdigest(),
        }
    return metadata


def build_trajectory_manifest(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    config = load_competitive_config()
    dataset = config["thoughtworks"]["dataset"]
    verify_dataset(args.dataset_file, dataset["sha256"])
    manifest = output / "inputs" / "thoughtworks-trajectories.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TRAJECTORY_GENERATOR),
        "--dataset-file",
        str(args.dataset_file),
        "--dataset-revision",
        dataset["revision"],
        "--output",
        str(manifest),
        "--trajectories-per-framework",
        str(args.trajectories_per_framework),
        "--min-isl",
        str(args.min_isl),
        "--max-isl",
        str(args.max_isl),
        "--min-turns",
        str(args.min_turns),
    ]
    command.extend(("--cohort", "warmup"))
    for concurrency in args.concurrency:
        command.extend(("--cohort", str(concurrency)))
    for framework in args.framework:
        command.extend(("--framework", framework))
    for source in args.source_dataset:
        command.extend(("--source-dataset", source))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "prompt generation failed; install DuckDB in this Python environment "
            f"and verify the requested selection window ({args.min_isl}-{args.max_isl})"
        ) from error
    document = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "kind": "thoughtworks",
        "dataset": dataset,
        "dataset_file": str(args.dataset_file),
        "dataset_file_sha256": sha256(args.dataset_file),
        "trajectory_generator": str(TRAJECTORY_GENERATOR),
        "trajectory_generator_sha256": sha256(TRAJECTORY_GENERATOR),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "metadata": document["metadata"],
        "cohorts": document["metadata"]["cohorts"],
    }


def import_trajectory_manifest(
    source: Path, output: Path, expected_cohorts: Sequence[str]
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"trajectory manifest not found: {source}")
    manifest = output / "inputs" / "captured-trajectories.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, manifest)
    cohorts = load_trajectory_cohorts(manifest, expected_cohorts)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "kind": "captured",
        "dataset": {
            "name": metadata.get("name", source.stem),
            "revision": metadata.get("revision", sha256(source)),
        },
        "source_manifest": str(source),
        "source_manifest_sha256": sha256(source),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "metadata": metadata,
        "cohorts": cohort_metadata(cohorts),
    }


def build_trajectory_inputs(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    expected_cohorts = ["warmup", *(str(value) for value in args.concurrency)]
    if args.trajectory_manifest is not None:
        return import_trajectory_manifest(
            args.trajectory_manifest, output, expected_cohorts
        )
    return build_trajectory_manifest(args, output)


def load_trajectory_cohorts(
    path: Path, expected_cohorts: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cohorts = document.get("cohorts")
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError("trajectory manifest must contain nonempty cohorts")
    missing = [name for name in expected_cohorts if name not in cohorts]
    if missing:
        raise ValueError(f"trajectory manifest is missing cohorts: {missing}")
    for name, trajectories in cohorts.items():
        if not isinstance(name, str) or not isinstance(trajectories, list) or not trajectories:
            raise ValueError("each trajectory cohort must be a nonempty list")
        for trajectory in trajectories:
            if not isinstance(trajectory, dict):
                raise ValueError(f"cohort {name} contains a non-object trajectory")
            session_id = trajectory.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(f"cohort {name} contains a trajectory without session_id")
            for field in ("source_dataset", "agent_framework"):
                if not isinstance(trajectory.get(field), str) or not trajectory[field]:
                    raise ValueError(f"trajectory {session_id} has no {field}")
            if "recorded_model" not in trajectory:
                raise ValueError(f"trajectory {session_id} has no recorded_model field")
            recorded_model = trajectory["recorded_model"]
            if recorded_model is not None and (
                not isinstance(recorded_model, str) or not recorded_model
            ):
                raise ValueError(f"trajectory {session_id} has invalid recorded_model")
            messages = trajectory.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"trajectory {session_id} has no messages list"
                )
            for index, message in enumerate(messages):
                if not isinstance(message, dict) or not isinstance(
                    message.get("role"), str
                ):
                    raise ValueError(
                        f"trajectory {session_id} message {index} has no role"
                    )
            tools = trajectory.get("tools")
            if tools is not None:
                if not isinstance(tools, list) or any(
                    not isinstance(tool, dict) for tool in tools
                ):
                    raise ValueError(
                        f"trajectory {session_id} tools must be a list of objects"
                    )
    return cohorts


def validate_warmup_capacity(
    trajectories: Sequence[dict[str, Any]], required_turns: int
) -> None:
    available_turns = sum(
        message["role"] == "assistant"
        for trajectory in trajectories
        for message in trajectory["messages"]
    )
    if available_turns < required_turns:
        raise ValueError(
            f"warm-up cohort has {available_turns} assistant turns; "
            f"{required_turns} required"
        )


def validate_measured_cohort_capacity(
    cohorts: dict[str, list[dict[str, Any]]],
    concurrency_values: Sequence[int],
    minimum_worker_waves: int = 2,
) -> None:
    for concurrency in concurrency_values:
        trajectories = len(cohorts[str(concurrency)])
        required = minimum_worker_waves * concurrency
        if trajectories < required:
            raise ValueError(
                f"concurrency {concurrency} cohort has {trajectories} trajectories; "
                f"{required} required for at least {minimum_worker_waves} worker "
                f"wave{'s' if minimum_worker_waves != 1 else ''}"
            )


def validate_required_frameworks(
    cohorts: dict[str, list[dict[str, Any]]],
    concurrency_values: Sequence[int],
    required_frameworks: Sequence[str],
) -> None:
    required = set(required_frameworks)
    for concurrency in concurrency_values:
        present = {
            trajectory["agent_framework"] for trajectory in cohorts[str(concurrency)]
        }
        missing = sorted(required - present)
        if missing:
            raise ValueError(
                f"concurrency {concurrency} cohort is missing required frameworks: "
                + ", ".join(missing)
            )


def server_command(binary: Path, model: str) -> list[str]:
    command = [str(binary), "serve", "--model", model, "--log-format", "json"]
    for option in FORBIDDEN_STARTUP_OPTIONS:
        if option in command:
            raise AssertionError(f"default-startup benchmark cannot use {option}")
    return command


def command_for_build(build: dict[str, Any], model: str) -> list[str]:
    if build.get("engine", "mesh") == "mesh":
        return server_command(Path(build["binary"]), model)
    arm = EngineArm(
        label=build["external_engine"]["label"],
        engine=build["external_engine"]["engine"],
        executable=build["external_engine"]["executable"],
        model=build["external_engine"]["model"],
        served_model=build["external_engine"]["served_model"],
        context_size=build["external_engine"]["context_size"],
        max_concurrency=build["external_engine"]["max_concurrency"],
        tokenizer=(
            build["external_engine"]["tokenizer"]
            if build["external_engine"]["tokenizer"] is not None
            else None
        ),
        hf_config=(
            build["external_engine"]["hf_config"]
            if build["external_engine"]["hf_config"] is not None
            else None
        ),
        prefix_cache=build["external_engine"]["prefix_cache"],
        batch_size=build["external_engine"]["batch_size"],
        ubatch_size=build["external_engine"]["ubatch_size"],
        extra_args=tuple(build["external_engine"]["extra_args"]),
        cwd=Path(build["external_engine"]["cwd"]),
    )
    return engine_server_command(arm, DEFAULT_HOST, DEFAULT_PORT)


def port_is_open(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.2)
        return connection.connect_ex((host, port)) == 0


def wait_for_model(
    base_url: str,
    timeout: float,
    process: Optional[subprocess.Popen[bytes]] = None,
) -> str:
    if base_url != DEFAULT_BASE_URL:
        raise ValueError(f"agentic replay requires {DEFAULT_BASE_URL}")
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"inference server exited before readiness with status {process.returncode}"
            )
        connection = http.client.HTTPConnection(DEFAULT_HOST, DEFAULT_PORT, timeout=5)
        try:
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            body = response.read()
            if response.status == 200:
                document = json.loads(body)
                models = document.get("data") or []
                if models:
                    return models[0]["id"]
            last_error = f"HTTP {response.status}: {body[:300]!r}"
        except (OSError, json.JSONDecodeError) as error:
            last_error = str(error)
        finally:
            connection.close()
        time.sleep(1)
    raise TimeoutError(
        f"inference server did not become ready after {timeout}s: {last_error}"
    )


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(math.ceil(len(ordered) * fraction) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def merge_tool_call_delta(
    tool_calls: dict[int, dict[str, Any]], delta: Any, fallback_index: int
) -> None:
    if not isinstance(delta, dict):
        return
    index = delta.get("index")
    if not isinstance(index, int):
        index = fallback_index
    target = tool_calls.setdefault(index, {})
    call_type = delta.get("type")
    if isinstance(call_type, str):
        target["type"] = call_type
    function = delta.get("function")
    if not isinstance(function, dict):
        return
    target_function = target.setdefault("function", {})
    name = function.get("name")
    if isinstance(name, str):
        target_function["name"] = name
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        target_function["arguments"] = (
            target_function.get("arguments", "") + arguments
        )


def response_content_sha256(
    content_parts: Sequence[str],
    reasoning_parts: Sequence[str],
    tool_calls: dict[int, dict[str, Any]],
) -> str:
    identity = {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def stream_request(
    request_id: str,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    model_id: str,
    output_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    first_token_at: Optional[float] = None
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    content_events = 0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_call_parts: dict[int, dict[str, Any]] = {}
    saw_done = False
    connection = http.client.HTTPConnection(DEFAULT_HOST, DEFAULT_PORT, timeout=timeout)
    payload = {
        "model": model_id,
        "messages": list(messages),
        "max_tokens": output_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = list(tools)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(payload),
            {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        )
        response = connection.getresponse()
        if response.status != 200:
            body = response.read(4096).decode("utf-8", errors="replace")
            return {
                "request_id": request_id,
                **metadata,
                "error": f"HTTP {response.status}: {body}",
            }
        for raw_line in response:
            line = raw_line.strip()
            if not line.startswith(b"data: "):
                continue
            event_bytes = line[6:]
            if event_bytes == b"[DONE]":
                saw_done = True
                break
            try:
                event = json.loads(event_bytes)
            except json.JSONDecodeError:
                continue
            server_error = event.get("error")
            if server_error is not None:
                if isinstance(server_error, dict):
                    message = server_error.get("message")
                    if not isinstance(message, str) or not message:
                        message = json.dumps(server_error, sort_keys=True)
                else:
                    message = str(server_error)
                return {
                    "request_id": request_id,
                    **metadata,
                    "error": f"stream failed with server error: {message}",
                }
            usage = event.get("usage")
            if isinstance(usage, dict):
                completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                details = usage.get("prompt_tokens_details")
                if isinstance(details, dict):
                    cached_tokens = int(details.get("cached_tokens") or cached_tokens)
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            reasoning_content = delta.get("reasoning_content")
            tool_calls = delta.get("tool_calls")
            if content or reasoning_content or tool_calls:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                content_events += 1
                if isinstance(content, str):
                    content_parts.append(content)
                if isinstance(reasoning_content, str):
                    reasoning_parts.append(reasoning_content)
                if isinstance(tool_calls, list):
                    for fallback_index, tool_call in enumerate(tool_calls):
                        merge_tool_call_delta(
                            tool_call_parts, tool_call, fallback_index
                        )
        completed = time.monotonic()
    except Exception as error:  # preserve request-level failures in the artifact
        return {
            "request_id": request_id,
            **metadata,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        connection.close()
    if not saw_done:
        return {
            "request_id": request_id,
            **metadata,
            "error": "stream ended without terminal [DONE] marker",
        }
    if first_token_at is None:
        return {
            "request_id": request_id,
            **metadata,
            "error": "stream completed without generated content",
        }
    if completion_tokens <= 0:
        return {
            "request_id": request_id,
            **metadata,
            "error": "stream completed without completion-token usage",
        }
    return {
        "request_id": request_id,
        **metadata,
        "started": started,
        "first_token_at": first_token_at,
        "completed": completed,
        "ttft_seconds": first_token_at - started,
        "elapsed_seconds": completed - started,
        "generation_seconds": completed - first_token_at,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "content_events": content_events,
        "content_sha256": response_content_sha256(
            content_parts, reasoning_parts, tool_call_parts
        ),
    }


def openai_message(recorded: dict[str, Any]) -> dict[str, Any]:
    # ChatMessage preserves arbitrary OpenAI-compatible fields, so retain the
    # captured object apart from null optional fields and the dataset-only JSON
    # encoding below. Older OpenAI-compatible servers reject null fields that
    # their schemas model as optional strings.
    message = {
        key: value
        for key, value in recorded.items()
        if key != "tool_calls_json" and value is not None
    }
    tool_calls_json = recorded.get("tool_calls_json")
    if tool_calls_json:
        message["tool_calls"] = json.loads(tool_calls_json)
    return message


def recorded_output_budget(recorded: dict[str, Any], maximum: int) -> int:
    content_length = len(recorded.get("content") or "")
    tool_calls = recorded.get("tool_calls")
    tool_length = len(recorded.get("tool_calls_json") or "")
    if tool_calls is not None:
        tool_length += len(json.dumps(tool_calls, separators=(",", ":")))
    approximate_tokens = math.ceil((content_length + tool_length) / 4)
    return min(max(approximate_tokens, 8), maximum)


def trajectory_tools(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    captured = trajectory.get("tools")
    if captured is not None:
        # Captured manifests retain the exact harness schemas because those
        # tokens are part of the reusable prefix identity.
        return list(captured)
    names: set[str] = set()
    for recorded in trajectory["messages"]:
        tool_calls_json = recorded.get("tool_calls_json")
        if not tool_calls_json:
            continue
        for tool_call in json.loads(tool_calls_json):
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.add(function["name"])
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Tool available in the recorded agent trajectory.",
                "parameters": {"type": "object", "additionalProperties": True},
            },
        }
        for name in sorted(names)
    ]


def replay_trajectory(
    trajectory: dict[str, Any],
    model_id: str,
    max_output_tokens: int,
    timeout: float,
    turn_limit: Optional[int] = None,
    measured_assistant_turns: Optional[set[int]] = None,
    checkpoint_stage: Optional[str] = None,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    assistant_turn = 0
    tools = trajectory_tools(trajectory)
    for message_index, recorded in enumerate(trajectory["messages"]):
        if recorded["role"] == "assistant":
            if turn_limit is not None and assistant_turn >= turn_limit:
                break
            if (
                measured_assistant_turns is None
                or assistant_turn in measured_assistant_turns
            ):
                requested_output_tokens = recorded_output_budget(
                    recorded, max_output_tokens
                )
                metadata = {
                    "session_id": trajectory["session_id"],
                    "source_dataset": trajectory["source_dataset"],
                    "agent_framework": trajectory["agent_framework"],
                    "recorded_model": trajectory["recorded_model"],
                    "assistant_turn": assistant_turn,
                    "recorded_message_index": message_index,
                    "history_message_count": len(history),
                    "requested_output_tokens": requested_output_tokens,
                    "recorded_output_characters": len(recorded.get("content") or ""),
                    "available_tools": len(tools),
                    "checkpoint_stage": checkpoint_stage,
                }
                result = stream_request(
                    f"{trajectory['session_id']}:{assistant_turn}",
                    history,
                    tools,
                    metadata,
                    model_id,
                    requested_output_tokens,
                    timeout,
                )
                results.append(result)
            assistant_turn += 1
        # Continue with the recorded trajectory, not the generated benchmark
        # output, so every experiment arm receives the same ordered history.
        history.append(openai_message(recorded))
    return results


def assistant_turn_count(trajectory: dict[str, Any]) -> int:
    return sum(
        message["role"] == "assistant" for message in trajectory["messages"]
    )


def checkpoint_schedule(
    trajectories: Sequence[dict[str, Any]],
) -> dict[str, tuple[int, str]]:
    """Assign deterministic early/middle/late/final checkpoints per framework."""
    by_framework: dict[str, list[dict[str, Any]]] = {}
    for trajectory in trajectories:
        by_framework.setdefault(trajectory["agent_framework"], []).append(trajectory)
    schedule: dict[str, tuple[int, str]] = {}
    stage_names = {1: "early", 2: "middle", 3: "late", 4: "final"}
    for framework_trajectories in by_framework.values():
        denominator = len(framework_trajectories)
        for rank, trajectory in enumerate(framework_trajectories, start=1):
            turns = assistant_turn_count(trajectory)
            if turns <= 0:
                raise ValueError(
                    f"trajectory {trajectory['session_id']} has no assistant turns"
                )
            assistant_turn = max(0, math.ceil(turns * rank / denominator) - 1)
            stage = stage_names.get(rank) if denominator == 4 else None
            schedule[trajectory["session_id"]] = (
                assistant_turn,
                stage or f"{rank}/{denominator}",
            )
    return schedule


def summarize_requests(
    requests: Sequence[dict[str, Any]], offered_concurrency: int = 1
) -> dict[str, Any]:
    successful = [request for request in requests if "error" not in request]
    ttft = [request["ttft_seconds"] for request in successful]
    completion_tokens = sum(request["completion_tokens"] for request in successful)
    prompt_tokens = sum(request["prompt_tokens"] for request in successful)
    cached_tokens = sum(request["cached_tokens"] for request in successful)
    elapsed_seconds = sum(request["elapsed_seconds"] for request in successful)
    generation_seconds = sum(
        request["generation_seconds"] for request in successful
    )
    if successful:
        workload_window = max(request["completed"] for request in successful) - min(
            request["started"] for request in successful
        )
    else:
        workload_window = 0.0
    mean_in_flight = (
        elapsed_seconds / workload_window if workload_window > 0 else None
    )
    return {
        "requests": len(requests),
        "successful_requests": len(successful),
        "failed_requests": len(requests) - len(successful),
        "failed_request_ids": sorted(
            str(request.get("request_id", "unknown"))
            for request in requests
            if "error" in request
        ),
        "successful_request_ids": sorted(
            str(request["request_id"])
            for request in successful
            if request.get("request_id")
        ),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "prompt_tokens_min": min(
            (request["prompt_tokens"] for request in successful), default=None
        ),
        "prompt_tokens_max": max(
            (request["prompt_tokens"] for request in successful), default=None
        ),
        "content_sha256_by_request": {
            request["request_id"]: request["content_sha256"]
            for request in successful
            if request.get("request_id") and request.get("content_sha256")
        },
        "generation_seconds": generation_seconds,
        "workload_window_seconds": workload_window,
        "budget_exhausted_requests": sum(
            request.get("completion_tokens") == request.get("requested_output_tokens")
            for request in successful
        ),
        "ttft_samples": ttft,
        "ttft_p50_seconds": statistics.median(ttft) if ttft else None,
        "ttft_p95_seconds": percentile(ttft, 0.95),
        "agent_steps_per_second": (
            len(successful) / workload_window if workload_window > 0 else None
        ),
        "workload_output_tokens_per_second": (
            completion_tokens / workload_window if workload_window > 0 else None
        ),
        "decode_tokens_per_second": (
            completion_tokens / generation_seconds
            if generation_seconds > 0
            else None
        ),
        "mean_in_flight": mean_in_flight,
        "concurrency_utilization_pct": (
            100 * mean_in_flight / offered_concurrency
            if mean_in_flight is not None and offered_concurrency > 0
            else None
        ),
        "cache_pct": 100 * cached_tokens / prompt_tokens if prompt_tokens else None,
    }


def write_request_records(
    path: Path,
    requests: Sequence[dict[str, Any]],
    concurrency: int,
    *,
    warmup: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as raw:
        for request in requests:
            request["concurrency"] = concurrency
            request["warmup"] = warmup
            raw.write(json.dumps(request, sort_keys=True) + "\n")


def run_warmup(
    trajectories: Sequence[dict[str, Any]],
    model_id: str,
    turns: int,
    max_output_tokens: int,
    timeout: float,
    raw_path: Path,
) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    for trajectory in trajectories:
        remaining = turns - len(requests)
        if remaining <= 0:
            break
        requests.extend(
            replay_trajectory(
                trajectory,
                model_id,
                max_output_tokens,
                timeout,
                turn_limit=remaining,
            )
        )
    if len(requests) < turns:
        raise RuntimeError(
            f"warm-up cohort produced {len(requests)} turns; {turns} required"
        )
    write_request_records(raw_path, requests, 1, warmup=True)
    summary = summarize_requests(requests, 1)
    if summary["failed_requests"]:
        raise RuntimeError(
            f"warm-up failed {summary['failed_requests']} of {summary['requests']} turns"
        )
    return summary


def run_trajectory_cell(
    *,
    trajectories: Sequence[dict[str, Any]],
    model_id: str,
    concurrency: int,
    max_output_tokens: int,
    timeout: float,
    raw_path: Path,
    replay_mode: str,
) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("trajectory cell cannot be empty")
    requests: list[dict[str, Any]] = []
    if replay_mode == "checkpoints":
        checkpoints = checkpoint_schedule(trajectories)
    elif replay_mode == "final":
        checkpoints = {
            trajectory["session_id"]: (
                assistant_turn_count(trajectory) - 1,
                "final",
            )
            for trajectory in trajectories
        }
    else:
        checkpoints = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(concurrency, len(trajectories))
    ) as pool:
        futures = [
            pool.submit(
                replay_trajectory,
                trajectory,
                model_id,
                max_output_tokens,
                timeout,
                measured_assistant_turns=(
                    {checkpoints[trajectory["session_id"]][0]}
                    if replay_mode in {"checkpoints", "final"}
                    else None
                ),
                checkpoint_stage=(
                    checkpoints[trajectory["session_id"]][1]
                    if replay_mode in {"checkpoints", "final"}
                    else None
                ),
            )
            for trajectory in trajectories
        ]
        for future in futures:
            requests.extend(future.result())
    write_request_records(raw_path, requests, concurrency)
    summary = summarize_requests(requests, concurrency)
    framework_counts: dict[str, int] = {}
    for trajectory in trajectories:
        framework = trajectory["agent_framework"]
        framework_counts[framework] = framework_counts.get(framework, 0) + 1
    successful_sessions = {
        request["session_id"]
        for request in requests
        if "error" not in request
    }
    failed_sessions = {
        request["session_id"] for request in requests if "error" in request
    }
    summary.update(
        {
            "concurrency": concurrency,
            "trajectories": len(trajectories),
            "successful_trajectories": len(successful_sessions - failed_sessions),
            "failed_trajectories": len(failed_sessions),
            "framework_trajectories": framework_counts,
            "max_output_tokens": max_output_tokens,
            "ordered_replay": True,
            "replay_mode": replay_mode,
            "recorded_assistant_turns": sum(
                assistant_turn_count(trajectory) for trajectory in trajectories
            ),
        }
    )
    return summary


def isolated_server_env(
    runtime_root: Path, state_dir: Path, hf_home: Optional[Path]
) -> dict[str, str]:
    env = os.environ.copy()
    inherited_home = Path.home()
    home = state_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "XDG_CACHE_HOME": str(state_dir / "xdg-cache"),
            "XDG_CONFIG_HOME": str(state_dir / "xdg-config"),
            "MESH_LLM_RUNTIME_ROOT": str(state_dir / "runtime"),
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR": str(runtime_root),
        }
    )
    if hf_home is not None:
        env["HF_HOME"] = str(hf_home)
    elif "HF_HOME" not in env:
        env["HF_HOME"] = str(inherited_home / ".cache/huggingface")
    return env


def start_server(
    build: dict[str, Any],
    model: str,
    state_dir: Path,
    log_path: Path,
    hf_home: Optional[Path],
) -> tuple[subprocess.Popen[bytes], list[str]]:
    if port_is_open():
        raise RuntimeError(
            f"TCP {DEFAULT_PORT} is already in use; stop the existing inference server"
        )
    command = command_for_build(build, model)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("wb")
    external = build.get("engine", "mesh") != "mesh"
    process = subprocess.Popen(
        command,
        cwd=Path(build["worktree"]),
        env=(
            external_server_environment(hf_home)
            if external
            else isolated_server_env(
                Path(build["runtime_root"]), state_dir, hf_home
            )
        ),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    return process, command


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    deadline = time.monotonic() + 10
    while port_is_open() and time.monotonic() < deadline:
        time.sleep(0.2)
    if port_is_open():
        raise RuntimeError(
            f"inference server stopped but TCP {DEFAULT_PORT} is still occupied"
        )


def collect_runtime_logs(state_dir: Path, output_dir: Path) -> None:
    runtime_root = state_dir / "runtime"
    if not runtime_root.is_dir():
        return
    for source in runtime_root.rglob("*"):
        if not source.is_file() or "logs" not in source.parts:
            continue
        destination = output_dir / source.relative_to(runtime_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_arm_pass(
    *,
    args: argparse.Namespace,
    output: Path,
    build: dict[str, Any],
    pass_index: int,
    cohorts: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    label = build["label"]
    pass_dir = output / "data" / f"pass-{pass_index + 1}" / label
    state_dir = Path(
        tempfile.mkdtemp(prefix=f"agentic-replay-{label}-pass-{pass_index + 1}-")
    )
    log_path = pass_dir / (
        "mesh.log" if build.get("engine", "mesh") == "mesh" else "server.log"
    )
    started_at = utc_now()
    process: Optional[subprocess.Popen[bytes]] = None
    command = command_for_build(build, args.model)
    cells: list[dict[str, Any]] = []
    warmup: Optional[dict[str, Any]] = None
    try:
        process, command = start_server(
            build, args.model, state_dir, log_path, args.hf_home
        )
        model_id = wait_for_model(DEFAULT_BASE_URL, args.startup_timeout, process)
        warmup = run_warmup(
            trajectories=cohorts["warmup"],
            model_id=model_id,
            turns=args.warmup_turns,
            max_output_tokens=args.max_output_tokens,
            timeout=args.request_timeout,
            raw_path=pass_dir / "warmup-requests.jsonl",
        )
        write_json(pass_dir / "warmup.json", warmup)
        concurrency_values = (
            args.concurrency
            if pass_index % 2 == 0
            else list(reversed(args.concurrency))
        )
        for concurrency in concurrency_values:
            cell = run_trajectory_cell(
                trajectories=cohorts[str(concurrency)],
                model_id=model_id,
                concurrency=concurrency,
                max_output_tokens=args.max_output_tokens,
                timeout=args.request_timeout,
                raw_path=pass_dir / f"c-{concurrency}-requests.jsonl",
                replay_mode=args.replay_mode,
            )
            cells.append(cell)
            write_json(pass_dir / f"c-{concurrency}.json", cell)
    finally:
        try:
            if process is not None:
                stop_server(process)
        finally:
            collect_runtime_logs(state_dir, pass_dir / "native-runtime")
            shutil.rmtree(state_dir, ignore_errors=True)
    return {
        "label": label,
        "ref": build["ref"],
        "commit": build["commit"],
        "pass": pass_index + 1,
        "started_at": started_at,
        "completed_at": utc_now(),
        "server_command": command,
        "server_log": str(log_path),
        "model_id": model_id,
        "warmup": warmup,
        "cells": cells,
    }


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def pooled_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for arm_pass in results:
        metadata[arm_pass["label"]] = (arm_pass["ref"], arm_pass["commit"])
        for cell in arm_pass["cells"]:
            groups.setdefault((arm_pass["label"], cell["concurrency"]), []).append(cell)
    rows: list[dict[str, Any]] = []
    for (label, concurrency), cells in sorted(groups.items()):
        ref, commit = metadata[label]
        requests = sum(cell["requests"] for cell in cells)
        successes = sum(cell["successful_requests"] for cell in cells)
        failures = requests - successes
        failure_identity_known = failures == 0 or all(
            "failed_request_ids" in cell for cell in cells
        )
        failed_request_ids = sorted(
            request_id
            for cell in cells
            for request_id in cell.get("failed_request_ids", [])
        )
        content_hashes: dict[str, set[str]] = {}
        content_hash_records = 0
        for cell in cells:
            for request_id, digest in cell.get(
                "content_sha256_by_request", {}
            ).items():
                content_hashes.setdefault(request_id, set()).add(digest)
                content_hash_records += 1
        successful_request_records = sum(
            len(cell.get("successful_request_ids", [])) for cell in cells
        )
        successful_request_ids = {
            request_id
            for cell in cells
            for request_id in cell.get("successful_request_ids", [])
        }
        content_identity = {
            request_id: sorted(digests)
            for request_id, digests in sorted(content_hashes.items())
        }
        prompt_token_mins = [
            cell["prompt_tokens_min"]
            for cell in cells
            if cell.get("prompt_tokens_min") is not None
        ]
        prompt_token_maxes = [
            cell["prompt_tokens_max"]
            for cell in cells
            if cell.get("prompt_tokens_max") is not None
        ]
        completion_tokens = sum(cell.get("completion_tokens", 0) for cell in cells)
        prompt_tokens = sum(cell.get("prompt_tokens", 0) for cell in cells)
        cached_tokens = sum(cell.get("cached_tokens", 0) for cell in cells)
        generation_seconds = sum(
            cell.get("generation_seconds", 0.0) for cell in cells
        )
        workload_seconds = sum(
            cell.get("workload_window_seconds", 0.0) for cell in cells
        )
        pooled_ttft = [
            value for cell in cells for value in cell.get("ttft_samples", [])
        ]
        decode_values = [
            cell.get(
                "decode_tokens_per_second",
                cell.get("mean_request_decode_tokens_per_second"),
            )
            for cell in cells
        ]
        decode_values = [value for value in decode_values if value is not None]
        step_values = [
            cell["agent_steps_per_second"]
            for cell in cells
            if cell["agent_steps_per_second"] is not None
        ]
        workload_values = [
            cell["workload_output_tokens_per_second"]
            for cell in cells
            if cell["workload_output_tokens_per_second"] is not None
        ]
        ttft_p50_values = [
            cell["ttft_p50_seconds"]
            for cell in cells
            if cell["ttft_p50_seconds"] is not None
        ]
        realized_concurrency = (
            sum(
                cell.get("mean_in_flight", 0.0)
                * cell.get("workload_window_seconds", 0.0)
                for cell in cells
                if cell.get("mean_in_flight") is not None
            )
            / workload_seconds
            if workload_seconds > 0
            else mean_or_none(cell.get("mean_in_flight") for cell in cells)
        )
        rows.append(
            {
                "label": label,
                "ref": ref,
                "commit": commit,
                "concurrency": concurrency,
                "passes": len(cells),
                "trajectories_per_pass": cells[0]["trajectories"],
                "trajectory_replays": sum(cell["trajectories"] for cell in cells),
                "requests": requests,
                "successful_requests": successes,
                "failed_requests": failures,
                "failed_request_ids": failed_request_ids,
                "failure_identity_known": failure_identity_known,
                "content_identity": content_identity,
                "content_identity_known": successful_request_records > 0
                and content_hash_records == successful_request_records
                and set(content_hashes) == successful_request_ids,
                "content_stable_across_passes": all(
                    len(digests) == 1 for digests in content_hashes.values()
                ),
                "prompt_tokens_min": min(prompt_token_mins)
                if prompt_token_mins
                else None,
                "prompt_tokens_max": max(prompt_token_maxes)
                if prompt_token_maxes
                else None,
                "success_pct": 100 * successes / requests if requests else None,
                "budget_exhausted_pct": 100
                * sum(
                    cell.get(
                        "budget_exhausted_requests",
                        cell.get("exact_output_requests", 0),
                    )
                    for cell in cells
                )
                / successes
                if successes
                else None,
                "agent_steps_per_second": (
                    successes / workload_seconds
                    if workload_seconds > 0
                    else mean_or_none(step_values)
                ),
                "agent_steps_per_second_min": min(step_values) if step_values else None,
                "agent_steps_per_second_max": max(step_values) if step_values else None,
                "workload_output_tokens_per_second": (
                    completion_tokens / workload_seconds
                    if workload_seconds > 0
                    else mean_or_none(workload_values)
                ),
                "workload_output_tokens_per_second_min": (
                    min(workload_values) if workload_values else None
                ),
                "workload_output_tokens_per_second_max": (
                    max(workload_values) if workload_values else None
                ),
                "decode_tokens_per_second": (
                    completion_tokens / generation_seconds
                    if generation_seconds > 0
                    else mean_or_none(decode_values)
                ),
                "decode_tokens_per_second_min": (
                    min(decode_values) if decode_values else None
                ),
                "decode_tokens_per_second_max": (
                    max(decode_values) if decode_values else None
                ),
                "ttft_p50_seconds": (
                    percentile(pooled_ttft, 0.50)
                    if pooled_ttft
                    else mean_or_none(cell["ttft_p50_seconds"] for cell in cells)
                ),
                "ttft_p50_seconds_min": (
                    min(ttft_p50_values) if ttft_p50_values else None
                ),
                "ttft_p50_seconds_max": (
                    max(ttft_p50_values) if ttft_p50_values else None
                ),
                "ttft_p95_seconds": (
                    percentile(pooled_ttft, 0.95)
                    if pooled_ttft
                    else mean_or_none(cell["ttft_p95_seconds"] for cell in cells)
                ),
                "mean_in_flight": realized_concurrency,
                "concurrency_utilization_pct": (
                    100 * realized_concurrency / concurrency
                    if realized_concurrency is not None
                    else None
                ),
                "cache_pct": (
                    100 * cached_tokens / prompt_tokens if prompt_tokens else None
                ),
            }
        )
    baseline_label = results[0]["label"] if results else None
    baseline = {
        row["concurrency"]: row
        for row in rows
        if row["label"] == baseline_label
    }
    for row in rows:
        reference = baseline.get(row["concurrency"])
        output_identity_comparable = bool(
            reference
            and (
                (
                    row["content_identity_known"]
                    and reference["content_identity_known"]
                    and row["content_stable_across_passes"]
                    and reference["content_stable_across_passes"]
                    and row["content_identity"] == reference["content_identity"]
                )
                or (
                    not row["content_identity_known"]
                    and not reference["content_identity_known"]
                )
            )
        )
        row["delta_comparable"] = bool(
            reference
            and row["requests"] == reference["requests"]
            and row["failure_identity_known"]
            and reference["failure_identity_known"]
            and row["failed_request_ids"] == reference["failed_request_ids"]
            and output_identity_comparable
        )
        for metric in (
            "agent_steps_per_second",
            "workload_output_tokens_per_second",
            "decode_tokens_per_second",
            "ttft_p50_seconds",
        ):
            base_value = reference.get(metric) if reference else None
            value = row.get(metric)
            row[f"{metric}_delta_pct"] = (
                100 * (value / base_value - 1)
                if row["delta_comparable"]
                and value is not None
                and base_value not in (None, 0)
                else None
            )
    return rows


def parse_token_range(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", value)
    if match is None:
        raise ValueError("prompt token range must use positive MIN:MAX syntax")
    minimum, maximum = (int(part) for part in match.groups())
    if maximum < minimum:
        raise ValueError("prompt token range maximum must be at least its minimum")
    return minimum, maximum


def evaluate_gates(
    rows: Sequence[dict[str, Any]],
    *,
    prompt_token_range: Optional[tuple[int, int]] = None,
    min_cache_pct: Optional[float] = None,
    require_output_match: bool = False,
    max_ttft_regression_pct: Optional[float] = None,
) -> dict[str, Any]:
    configured = any(
        (
            prompt_token_range is not None,
            min_cache_pct is not None,
            require_output_match,
            max_ttft_regression_pct is not None,
        )
    )
    if not configured:
        return {"evaluated": False, "passed": None, "checks": []}

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    if not rows:
        record("measured-rows", False, "observed=0 required>=1")

    for row in rows:
        cell = f"{row['label']}/c{row['concurrency']}"
        failed_requests = row.get("failed_requests")
        record(
            f"failed-requests:{cell}",
            failed_requests == 0,
            f"observed={failed_requests} required=0",
        )
        if prompt_token_range is not None:
            minimum, maximum = prompt_token_range
            observed_min = row.get("prompt_tokens_min")
            observed_max = row.get("prompt_tokens_max")
            passed = (
                observed_min is not None
                and observed_max is not None
                and observed_min >= minimum
                and observed_max <= maximum
            )
            record(
                f"prompt-token-range:{cell}",
                passed,
                f"observed={observed_min}:{observed_max} required={minimum}:{maximum}",
            )
        if min_cache_pct is not None:
            observed = row.get("cache_pct")
            record(
                f"cached-prompt:{cell}",
                observed is not None and observed >= min_cache_pct,
                f"observed={observed} required>={min_cache_pct}",
            )
        if require_output_match:
            output_matches = bool(row.get("content_identity_known")) and bool(
                row.get("delta_comparable")
            )
            record(
                f"deterministic-output:{cell}",
                output_matches,
                "content hashes and failed request identities match baseline"
                if output_matches
                else "content hashes are missing or differ, or failed request identities differ from baseline",
            )
        if max_ttft_regression_pct is not None:
            observed = row.get("ttft_p50_seconds_delta_pct")
            record(
                f"ttft-regression:{cell}",
                observed is not None and observed <= max_ttft_regression_pct,
                (
                    f"observed={observed:.3f}% allowed<={max_ttft_regression_pct:.3f}%"
                    if observed is not None
                    else f"observed=unavailable allowed<={max_ttft_regression_pct:.3f}%"
                ),
            )
    return {
        "evaluated": True,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def fmt(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def fmt_range(
    low: Optional[float], high: Optional[float], digits: int = 2
) -> str:
    if low is None or high is None:
        return "—"
    return f"{low:.{digits}f}–{high:.{digits}f}"


def escape(value: Any) -> str:
    return html.escape(str(value))


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def markdown_code_cell(value: Any) -> str:
    """Render an arbitrary string as a delimiter-safe Markdown code cell.

    The report wraps these values in fixed backtick fences, so a literal
    backtick would terminate the span, and backslash-pipe escaping is unsafe
    when the value itself ends in a backslash (the pair would un-escape the
    pipe and re-split the table cell). Render as an HTML code element with
    every Markdown/HTML delimiter entity-escaped: table renderers honor HTML
    escapes inside GitHub-flavored Markdown tables, and with no raw pipe,
    backtick, or angle brackets left in the cell it cannot split the table or
    re-enter Markdown syntax.
    """
    text = html.escape(str(value), quote=False)
    text = text.replace("|", "&#124;").replace("`", "&#96;")
    text = text.replace("\r", " ").replace("\n", " ")
    return f"<code>{text}</code>"


def svg_chart(
    title: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    metric: str,
    y_label: str,
    output: Path,
) -> None:
    width, height = 960, 540
    left, top, plot_width, plot_height = 100, 80, 790, 350
    concurrency_values = sorted({row["concurrency"] for row in rows})
    values = [row[metric] for row in rows if row.get(metric) is not None]
    y_max = max(max(values) * 1.12, 1e-9) if values else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="480" y="36" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = top + plot_height - plot_height * tick / 5
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e2e8f0"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>'
        )
    x_denominator = max(len(concurrency_values) - 1, 1)
    for index, concurrency in enumerate(concurrency_values):
        x = left + plot_width * index / x_denominator
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{concurrency}</text>'
        )
    for label_index, label in enumerate(labels):
        color = COLORS[label_index % len(COLORS)]
        points: list[tuple[float, float]] = []
        indexed = {
            row["concurrency"]: row[metric]
            for row in rows
            if row["label"] == label and row.get(metric) is not None
        }
        for index, concurrency in enumerate(concurrency_values):
            if concurrency not in indexed:
                continue
            x = left + plot_width * index / x_denominator
            y = top + plot_height - plot_height * indexed[concurrency] / y_max
            points.append((x, y))
        coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        parts.extend(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
            for x, y in points
        )
        legend_x = 110 + label_index * 150
        parts.append(
            f'<text x="{legend_x}" y="490" font-family="sans-serif" font-size="13" fill="{color}">{escape(label)}</text>'
        )
    parts.extend(
        [
            '<text x="480" y="525" text-anchor="middle" font-family="sans-serif" font-size="13">Client concurrency</text>',
            f'<text x="22" y="255" text-anchor="middle" transform="rotate(-90 22 255)" font-family="sans-serif" font-size="13">{escape(y_label)}</text>',
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")


def write_report(output: Path, run_document: dict[str, Any]) -> Path:
    rows = pooled_rows(run_document["results"])
    summary = output / "summary"
    charts = summary / "charts"
    summary.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with (summary / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    labels = [build["label"] for build in run_document["builds"]]
    cohort_metadata = run_document["inputs"]["cohorts"]
    measured_cohorts = [
        cohort_metadata[str(value)] for value in run_document["config"]["concurrency"]
    ]
    selected_trajectories = sum(
        cohort["trajectory_count"] for cohort in measured_cohorts
    )
    selected_turns = sum(cohort["assistant_turns"] for cohort in measured_cohorts)
    warmup_cohort = cohort_metadata["warmup"]
    svg_chart(
        "Decode throughput by arm",
        rows,
        labels,
        "decode_tokens_per_second",
        "Generated tokens / decode second",
        charts / "decode-throughput.svg",
    )
    svg_chart(
        "End-to-end workload output throughput by arm",
        rows,
        labels,
        "workload_output_tokens_per_second",
        "Generated tokens / wall-clock second",
        charts / "workload-output-throughput.svg",
    )
    svg_chart(
        "Median time to first token by arm",
        rows,
        labels,
        "ttft_p50_seconds",
        "Seconds (lower is better)",
        charts / "ttft-p50.svg",
    )
    lines = [
        "# Agentic Replay",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Arm identities",
        "",
        "| Arm | Engine | Reported version or ref | Identity |",
        "|---|---|---|---|",
    ]
    for build in run_document["builds"]:
        engine = build.get("engine", "mesh")
        version_or_ref = build.get("version") or build.get("ref", build["label"])
        identity = build.get("commit", "unknown")
        lines.append(
            f"| {markdown_cell(build['label'])} | {markdown_cell(engine)} | "
            f"{markdown_code_cell(version_or_ref)} | `{identity[:12]}` |"
        )
    lines.extend(
        [
            "",
            "## Trajectory selection",
            "",
            "| Client concurrency | Whole trajectories | Measured requests/pass | Recorded source steps | Framework trajectory / step breakdown |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for concurrency in run_document["config"]["concurrency"]:
        cohort = cohort_metadata[str(concurrency)]
        breakdown = " · ".join(
            f"{framework} {count} / {cohort['framework_assistant_turns'][framework]}"
            for framework, count in cohort["framework_trajectories"].items()
        )
        lines.append(
            f"| {concurrency} | {cohort['trajectory_count']} | "
            f"{cohort['trajectory_count'] if run_document['config'].get('replay_mode', 'all') in {'checkpoints', 'final'} else cohort['assistant_turns']} | "
            f"{cohort['assistant_turns']} | {breakdown} |"
        )
    lines.extend(
        [
            "",
            "## Result table",
            "",
            "| Arm | Identity | Offered C | Realized C | Slot use | Trajectories/pass | Measured requests | Prompt tokens | Failures | Output match | Decode tok/s | Pass range | vs baseline | E2E output tok/s | Pass range | TTFT p50 | Pass range | TTFT p95 | TTFT p50 vs baseline | Cached prompt | Budget exhausted |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | `{commit}` | {concurrency} | {realized} | {slot_use} | {trajectories} | {requests} | {prompt_range} | {failures} | {comparable} | {decode_tps} | {decode_range} | {decode_delta} | {output_tps} | {output_range} | {p50} | {p50_range} | {p95} | {p50_delta} | {cache} | {budget} |".format(
                label=row["label"],
                commit=row["commit"][:10],
                concurrency=row["concurrency"],
                realized=fmt(row["mean_in_flight"]),
                slot_use=fmt(row["concurrency_utilization_pct"], 1, "%"),
                trajectories=row["trajectories_per_pass"],
                requests=row["requests"],
                prompt_range=(
                    f"{row.get('prompt_tokens_min')}–{row.get('prompt_tokens_max')}"
                    if row.get("prompt_tokens_min") is not None
                    else "—"
                ),
                failures=row["failed_requests"],
                comparable="yes" if row["delta_comparable"] else "no",
                decode_tps=fmt(row["decode_tokens_per_second"]),
                decode_range=fmt_range(
                    row["decode_tokens_per_second_min"],
                    row["decode_tokens_per_second_max"],
                ),
                decode_delta=fmt(
                    row["decode_tokens_per_second_delta_pct"], 1, "%"
                ),
                output_tps=fmt(row["workload_output_tokens_per_second"]),
                output_range=fmt_range(
                    row["workload_output_tokens_per_second_min"],
                    row["workload_output_tokens_per_second_max"],
                ),
                p50=fmt(row["ttft_p50_seconds"], 3, "s"),
                p50_range=fmt_range(
                    row["ttft_p50_seconds_min"],
                    row["ttft_p50_seconds_max"],
                    3,
                ),
                p95=fmt(row["ttft_p95_seconds"], 3, "s"),
                p50_delta=fmt(row["ttft_p50_seconds_delta_pct"], 1, "%"),
                cache=fmt(row["cache_pct"], 1, "%"),
                budget=fmt(row["budget_exhausted_pct"], 1, "%"),
            )
        )
    server_method = (
        [
            "- Mesh startup: `mesh-llm serve --model <model> --log-format json`.",
            "- Mesh chooses context size, execution lanes, KV budget, and backend tuning.",
        ]
        if run_document["config"].get("engine_config") is None
        else [
            "- Mesh refs use product-default startup; external engines use the exact commands recorded in `plan.json` and each pass result.",
            "- Each external engine reports its version before launch; the report records that string and its SHA-256 identity. Model/tokenizer paths are provenance, not file-hash gates.",
        ]
    )
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "![Decode throughput](charts/decode-throughput.svg)",
            "",
            "![End-to-end workload output throughput](charts/workload-output-throughput.svg)",
            "",
            "![Median TTFT](charts/ttft-p50.svg)",
            "",
            "## Method",
            "",
            f"- Model: `{run_document['config']['model']}`",
            *server_method,
            f"- Client concurrency: `{','.join(map(str, run_document['config']['concurrency']))}`.",
            f"- Pass order: `{' → '.join(item['label'] for item in run_document['order'])}`.",
            f"- Warm-up: `{run_document['config']['warmup_turns']}` discarded turns from a disjoint cohort after every model-ready event.",
            f"- Warm-up cohort: `{warmup_cohort['trajectory_count']}` whole trajectories, disjoint from measured cohorts.",
            f"- Dataset revision: `{run_document['inputs']['dataset']['revision']}`.",
            f"- Selected trajectories: `{selected_trajectories}` unique whole sessions across disjoint concurrency cohorts.",
            f"- Recorded source steps: `{selected_turns}` assistant turns are represented across the selected trajectories.",
            f"- Replay mode: `{run_document['config'].get('replay_mode', 'all')}`. Checkpoint and final modes measure one request per trajectory; skipped recorded turns are still appended in order to reconstruct the exact prefix.",
            "- Within each framework's trajectories, checkpoint mode deterministically spreads one measured turn per trajectory across the trajectory timeline. Four trajectories are labeled early, middle, late, and final; other cohort sizes use rank/count labels. Different trajectories may overlap up to the offered client concurrency.",
            "- Realized concurrency is the time-weighted mean number of in-flight requests. Slot use makes cohort tail drain explicit; do not interpret offered-concurrency scaling as steady-state when utilization is low.",
            "- Each next request uses the recorded conversation history, so experiment arms receive identical growing prefixes and tool observations.",
            "- Per-turn output budgets approximate each recorded assistant action from its character length, capped by the configured maximum; generated output is measured but never fed into the next turn.",
            "- Decode tok/s is token-weighted generation throughput after first content. E2E output tok/s includes prompt ingestion and scheduling.",
            "- Percent deltas are suppressed unless compared arms fail the same request IDs. Pass ranges expose run-to-run spread.",
            (
                "- Tool definitions and message fields come from the supplied captured manifest unchanged, so captured runs retain their exact reusable-prefix identity."
                if run_document["inputs"].get("kind") == "captured"
                else "- Tool definitions preserve recorded names but use permissive synthetic schemas. This benchmark measures serving performance on reconstructed prompts, not answer quality or byte-identical production prompts."
            ),
            "- Selection is deterministic hash order, not stratified by context length or difficulty. Treat small deltas as directional unless repeated with a larger cohort.",
            f"- Trajectory manifest SHA-256: `{run_document['inputs']['manifest_sha256']}`.",
            "- Raw request records, server logs, build logs, commands, and exact binary/runtime hashes are retained beside this report.",
            "",
            "Gates are opt-in. When configured, the command exits non-zero after preserving the complete artifact if any gate fails.",
            "",
        ]
    )
    gates = run_document.get("gates")
    if gates is not None:
        lines.extend(["## Acceptance gates", ""])
        if not gates.get("evaluated", True):
            lines.append("- **NOT EVALUATED** — no acceptance gates were configured.")
        else:
            for check in gates["checks"]:
                marker = "PASS" if check["passed"] else "FAIL"
                lines.append(f"- **{marker}** `{check['name']}` — {check['detail']}")
        lines.append("")
    report_path = summary / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(summary / "comparison.json", rows)
    inventory: list[str] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "artifact-sha256.txt":
            continue
        inventory.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "artifact-sha256.txt").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    return report_path


def benchmark_plan(
    args: argparse.Namespace,
    specs: Sequence[RefSpec],
    engine_config: EngineConfig | None = None,
    version_sha256_by_label: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    config = load_competitive_config()
    dataset = config["thoughtworks"]["dataset"]
    order_specs = combined_specs(specs, engine_config, version_sha256_by_label)
    return {
        "schema_version": 3,
        "repo": str(args.repo),
        "refs": [spec.__dict__ for spec in specs],
        "engine_config": (
            {
                "path": str(engine_config.path),
                "sha256": engine_config.sha256,
                "comparison": engine_config.comparison.as_dict(),
                "arms": [arm.plan_identity() for arm in engine_config.arms],
            }
            if engine_config is not None
            else None
        ),
        "order": [
            {"pass": pass_index + 1, "label": spec.label, "commit": spec.commit}
            for pass_index, spec in ab_order(order_specs, args.passes)
        ],
        "build_commands": [
            ["just", "release-host-build"],
            ["just", "release-runtime-build", args.backend],
        ],
        "server_command": [
            "<release-binary>",
            "serve",
            "--model",
            args.model,
            "--log-format",
            "json",
        ],
        "external_server_commands": (
            [engine_server_command(arm) for arm in engine_config.arms]
            if engine_config is not None
            else []
        ),
        "dataset": dataset,
        "trajectory_manifest": (
            str(args.trajectory_manifest)
            if getattr(args, "trajectory_manifest", None) is not None
            else None
        ),
        "selection": {
            "source_datasets": args.source_dataset,
            "frameworks": args.framework,
            "trajectories_per_framework_per_concurrency": args.trajectories_per_framework,
            "measured_unique_trajectory_count": (
                len(args.concurrency)
                * len(args.framework)
                * args.trajectories_per_framework
                if args.trajectories_per_framework is not None
                else None
            ),
            "warmup_unique_trajectory_count": (
                len(args.framework) * args.trajectories_per_framework
                if args.trajectories_per_framework is not None
                else None
            ),
            "min_isl": args.min_isl,
            "max_isl_exclusive": args.max_isl,
            "min_assistant_turns": args.min_turns,
        },
        "workload": {
            "concurrency": args.concurrency,
            "passes": args.passes,
            "minimum_worker_waves": getattr(args, "minimum_worker_waves", 2),
            "replay_mode": args.replay_mode,
            "ordered_recorded_prefix_replay": True,
            "measured_requests_per_arm_pass": (
                len(args.concurrency)
                * len(args.framework)
                * args.trajectories_per_framework
                if args.replay_mode in {"checkpoints", "final"}
                and args.trajectories_per_framework is not None
                else None
            ),
            "measured_requests_total": (
                len(args.concurrency)
                * len(args.framework)
                * args.trajectories_per_framework
                * len(order_specs)
                * args.passes
                if args.replay_mode in {"checkpoints", "final"}
                and args.trajectories_per_framework is not None
                else None
            ),
            "max_output_tokens": args.max_output_tokens,
            "warmup_turns_per_arm_pass": args.warmup_turns,
        },
        "gates": {
            "required_frameworks": getattr(args, "require_framework", []),
            "prompt_token_range": getattr(args, "prompt_token_range", None),
            "min_cache_pct": getattr(args, "min_cache_pct", None),
            "require_output_match": getattr(args, "require_output_match", False),
            "max_ttft_regression_pct": getattr(
                args, "max_ttft_regression_pct", None
            ),
        },
        "outputs": [
            "raw request JSONL",
            "per-cell JSON",
            "server/build logs",
            "comparison CSV/JSON/Markdown",
            "throughput and TTFT SVG charts",
            "SHA-256 inventory",
        ],
    }


def run_benchmark(args: argparse.Namespace) -> Path:
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    if args.dataset_file is not None:
        args.dataset_file = args.dataset_file.resolve()
    if args.trajectory_manifest is not None:
        args.trajectory_manifest = args.trajectory_manifest.resolve()
    if args.hf_home is not None:
        args.hf_home = args.hf_home.resolve()
    specs = parse_ref_specs(args.repo, args.ref)
    engine_config = external_config(args)
    external_builds = (
        [verify_engine_arm(arm) for arm in engine_config.arms]
        if engine_config is not None
        else []
    )
    plan = benchmark_plan(
        args,
        specs,
        engine_config,
        version_sha256_by_label=verified_version_sha256_by_label(external_builds),
    )
    order_specs = combined_specs(
        specs,
        engine_config,
        version_sha256_by_label=verified_version_sha256_by_label(external_builds),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    existing = args.output / "run.json"
    if existing.exists() and not args.resume:
        raise RuntimeError(f"output already contains run.json; pass --resume: {args.output}")
    write_json(args.output / "plan.json", plan)
    commands = CommandLog(args.output / "commands.jsonl")
    inputs = build_trajectory_inputs(args, args.output)
    expected_cohorts = ["warmup", *(str(value) for value in args.concurrency)]
    cohorts = load_trajectory_cohorts(Path(inputs["manifest"]), expected_cohorts)
    validate_warmup_capacity(cohorts["warmup"], args.warmup_turns)
    validate_measured_cohort_capacity(
        cohorts, args.concurrency, args.minimum_worker_waves
    )
    validate_required_frameworks(
        cohorts, args.concurrency, args.require_framework
    )
    worktree_root = (
        args.worktree_root or (args.repo.parent / ".agentic-replay-worktrees")
    ).resolve()
    builds = []
    for spec in specs:
        worktree = prepare_worktree(args.repo, worktree_root, spec)
        builds.append(
            build_ref(spec, worktree, args.backend, args.output, commands, args.skip_build)
        )
    builds.extend(external_builds)
    run_document: dict[str, Any] = {
        "schema_version": 3,
        "started_at": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": sys.platform,
            "python": sys.version,
        },
        "config": {
            "model": args.model,
            "backend": args.backend,
            "concurrency": args.concurrency,
            "passes": args.passes,
            "minimum_worker_waves": args.minimum_worker_waves,
            "replay_mode": args.replay_mode,
            "max_output_tokens": args.max_output_tokens,
            "warmup_turns": args.warmup_turns,
            "required_frameworks": args.require_framework,
            "prompt_token_range": args.prompt_token_range,
            "min_cache_pct": args.min_cache_pct,
            "require_output_match": args.require_output_match,
            "max_ttft_regression_pct": args.max_ttft_regression_pct,
            "engine_config": (
                {
                    "path": str(engine_config.path),
                    "sha256": engine_config.sha256,
                    "comparison": engine_config.comparison.as_dict(),
                }
                if engine_config is not None
                else None
            ),
        },
        "plan_sha256": stable_hash(plan),
        "inputs": inputs,
        "builds": builds,
        "order": plan["order"],
        "results": [],
    }
    build_by_label = {build["label"]: build for build in builds}
    completed = {
        (item["pass"], item["label"])
        for item in json.loads(existing.read_text(encoding="utf-8")).get("results", [])
    } if existing.exists() and args.resume else set()
    if existing.exists() and args.resume:
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous.get("plan_sha256") != run_document["plan_sha256"]:
            raise RuntimeError("cannot resume: plan differs from existing run.json")
        previous_builds = {
            build["label"]: build_resume_identity(build)
            for build in previous.get("builds", [])
        }
        current_builds = {
            build["label"]: build_resume_identity(build)
            for build in builds
        }
        if previous_builds != current_builds:
            raise RuntimeError(
                "cannot resume: arm build identity differs from the existing "
                "run.json (commit, binary or runtime hashes, or reported "
                "engine version_sha256 changed)"
            )
        if previous.get("inputs", {}).get("manifest_sha256") != inputs["manifest_sha256"]:
            raise RuntimeError("cannot resume: Thoughtworks trajectory manifest differs")
        run_document["results"] = previous["results"]
    for pass_index, spec in ab_order(order_specs, args.passes):
        key = (pass_index + 1, spec.label)
        if key in completed:
            continue
        result = run_arm_pass(
            args=args,
            output=args.output,
            build=build_by_label[spec.label],
            pass_index=pass_index,
            cohorts=cohorts,
        )
        run_document["results"].append(result)
        write_json(existing, run_document)
    run_document["completed_at"] = utc_now()
    run_document["gates"] = evaluate_gates(
        pooled_rows(run_document["results"]),
        prompt_token_range=(
            parse_token_range(args.prompt_token_range)
            if args.prompt_token_range is not None
            else None
        ),
        min_cache_pct=args.min_cache_pct,
        require_output_match=args.require_output_match,
        max_ttft_regression_pct=args.max_ttft_regression_pct,
    )
    write_json(existing, run_document)
    report = write_report(args.output, run_document)
    if run_document["gates"]["passed"] is False:
        raise RuntimeError(f"benchmark acceptance gates failed; see {report}")
    return report


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument(
        "--ref",
        action="append",
        required=True,
        help="repeatable LABEL=GIT_REF; at least two distinct commits",
    )
    parser.add_argument("--model", required=True, help="model URI or local package path")
    parser.add_argument(
        "--engine-config",
        type=Path,
        help="JSON config for version-identified llama.cpp, vLLM, and SGLang arms",
    )
    parser.add_argument("--backend", default="metal")
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="A/B passes; use 2 for reverse-order ABBA confirmation",
    )
    parser.add_argument(
        "--replay-mode",
        choices=("checkpoints", "final", "all"),
        default="checkpoints",
        help="measure one stage-balanced checkpoint, the final prefix, or every assistant turn",
    )
    parser.add_argument("--concurrency", type=int, action="append", default=[])
    parser.add_argument(
        "--minimum-worker-waves",
        type=int,
        default=2,
        help=(
            "minimum complete request waves required in each measured cohort; "
            "set to 1 only for an exact captured single-wave workload"
        ),
    )
    parser.add_argument(
        "--trajectories-per-framework",
        type=int,
        help="whole trajectories from each framework in each concurrency cohort",
    )
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument(
        "--warmup-turns",
        type=int,
        default=4,
        help="discarded ordered turns from a disjoint cohort after model readiness",
    )
    parser.add_argument("--min-isl", type=int, default=8192)
    parser.add_argument("--max-isl", type=int, default=65536)
    parser.add_argument("--min-turns", type=int, default=5)
    parser.add_argument("--framework", action="append", default=[])
    parser.add_argument(
        "--source-dataset",
        action="append",
        default=[],
        help="Thoughtworks source_dataset; defaults to all three pinned sources",
    )
    parser.add_argument(
        "--trajectory-manifest",
        type=Path,
        help="captured exact-harness manifest; replaces --dataset-file for run",
    )
    parser.add_argument(
        "--require-framework",
        action="append",
        default=[],
        help="repeatable agent_framework required in every measured cohort",
    )
    parser.add_argument(
        "--prompt-token-range",
        help="optional acceptance range MIN:MAX applied to every measured request",
    )
    parser.add_argument(
        "--min-cache-pct",
        type=float,
        help="optional minimum cached prompt-token percentage for every ref/concurrency cell",
    )
    parser.add_argument(
        "--require-output-match",
        action="store_true",
        help="fail unless generated-content hashes and failure identities match the baseline",
    )
    parser.add_argument(
        "--max-ttft-regression-pct",
        type=float,
        help="optional maximum candidate median-TTFT regression versus the first ref",
    )


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command == "run" and (
        (args.dataset_file is None) == (args.trajectory_manifest is None)
    ):
        parser.error("run requires exactly one of --dataset-file or --trajectory-manifest")
    if not args.concurrency:
        args.concurrency = [1, 2, 4]
    if len(set(args.concurrency)) != len(args.concurrency) or any(
        value <= 0 for value in args.concurrency
    ):
        parser.error("--concurrency values must be unique and positive")
    positive = (
        args.passes,
        args.max_output_tokens,
        args.warmup_turns,
        args.min_isl,
        args.min_turns,
        args.minimum_worker_waves,
    )
    if any(value <= 0 for value in positive):
        parser.error("passes and workload sizes must be positive")
    if args.trajectories_per_framework is not None and args.trajectories_per_framework <= 0:
        parser.error("--trajectories-per-framework must be positive")
    if args.trajectory_manifest is None and args.trajectories_per_framework is None:
        parser.error("--trajectories-per-framework is required with Thoughtworks input")
    if args.max_isl <= args.min_isl:
        parser.error("--max-isl must exceed --min-isl")
    if args.prompt_token_range is not None:
        try:
            parse_token_range(args.prompt_token_range)
        except ValueError as error:
            parser.error(str(error))
    if args.min_cache_pct is not None and not 0 < args.min_cache_pct <= 100:
        parser.error("--min-cache-pct must be greater than 0 and at most 100")
    if args.max_ttft_regression_pct is not None and args.max_ttft_regression_pct < 0:
        parser.error("--max-ttft-regression-pct cannot be negative")
    if not args.source_dataset:
        args.source_dataset = [
            "swe-smith-claude-3-7-sonnet",
            "kwai-klear-swe-smith-mini",
            "nebius-swe-rebench-openhands",
        ]
    if not args.framework:
        args.framework = ["swe-agent", "mini-swe-agent", "openhands"]
    if len(set(args.framework)) != len(args.framework):
        parser.error("--framework values must be unique")
    if len(set(args.require_framework)) != len(args.require_framework):
        parser.error("--require-framework values must be unique")
    if args.engine_config is not None:
        try:
            config = load_engine_config(args.engine_config)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"invalid --engine-config: {error}")
        if config.comparison.model != args.model:
            parser.error(
                "--model must exactly match engine config comparison.model"
            )
    trajectories_per_cohort = (
        args.trajectories_per_framework * len(args.framework)
        if args.trajectories_per_framework is not None
        else None
    )
    minimum_trajectories = args.minimum_worker_waves * max(args.concurrency)
    if trajectories_per_cohort is not None and trajectories_per_cohort < minimum_trajectories:
        parser.error(
            "each concurrency cohort must contain the configured minimum worker "
            f"waves ({minimum_trajectories} trajectories required)"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print the exact side-effect-free benchmark plan")
    add_common_arguments(plan)
    run = subparsers.add_parser("run", help="build refs, run the matrix, and render the report")
    add_common_arguments(run)
    run.add_argument("--dataset-file", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--worktree-root", type=Path)
    run.add_argument("--hf-home", type=Path)
    run.add_argument("--startup-timeout", type=float, default=1800)
    run.add_argument("--request-timeout", type=float, default=900)
    run.add_argument("--skip-build", action="store_true")
    run.add_argument("--resume", action="store_true")
    report = subparsers.add_parser("report", help="rerender tables and charts from run.json")
    report.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in {"plan", "run"}:
        validate_args(args, parser)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        args.repo = args.repo.resolve()
        specs = parse_ref_specs(args.repo, args.ref)
        print(
            json.dumps(
                benchmark_plan(args, specs, external_config(args)),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run":
        print(run_benchmark(args))
        return 0
    artifact = args.artifact.resolve()
    document = json.loads((artifact / "run.json").read_text(encoding="utf-8"))
    print(write_report(artifact, document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
