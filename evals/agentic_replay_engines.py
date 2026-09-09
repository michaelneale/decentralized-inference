#!/usr/bin/env python3
"""External OpenAI-compatible engine arms for the agentic replay benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUPPORTED_ENGINES = ("llama.cpp", "vllm", "sglang")
LABEL_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
VERSION_COMMAND_TIMEOUT_SECONDS = 30
# Placeholder for the not-yet-verified external arm identity. The runner
# replaces it with the verified version SHA-256 before writing artifacts.
PENDING_VERSION_SHA256 = "reported-version-sha256-at-run"


def resolve_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def optional_path(root: Path, value: Any, field: str) -> Path | None:
    return None if value is None else resolve_path(root, value, field)


def required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def optional_string(value: Any, field: str) -> str | None:
    return None if value is None else required_string(value, field)


def positive_int(value: Any, field: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{field} is required")
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def string_list(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{field} must be an array of non-empty strings")
    return tuple(value)


@dataclass(frozen=True)
class ComparisonIdentity:
    model: str

    def as_dict(self) -> dict[str, str]:
        return {"model": self.model}


@dataclass(frozen=True)
class EngineArm:
    label: str
    engine: str
    executable: str
    model: str
    served_model: str
    context_size: int
    max_concurrency: int
    tokenizer: str | None
    hf_config: str | None
    prefix_cache: bool
    batch_size: int
    ubatch_size: int
    extra_args: tuple[str, ...]
    cwd: Path

    def plan_identity(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "engine": self.engine,
            "executable": str(self.executable),
            "model": str(self.model),
            "served_model": self.served_model,
            "context_size": self.context_size,
            "max_concurrency": self.max_concurrency,
            "tokenizer": str(self.tokenizer) if self.tokenizer else None,
            "hf_config": str(self.hf_config) if self.hf_config else None,
            "prefix_cache": self.prefix_cache,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "extra_args": list(self.extra_args),
            "cwd": str(self.cwd),
        }


@dataclass(frozen=True)
class EngineConfig:
    path: Path
    sha256: str
    comparison: ComparisonIdentity
    arms: tuple[EngineArm, ...]


def normalize_engine(value: Any) -> str:
    aliases = {
        "llama": "llama.cpp",
        "llama.cpp": "llama.cpp",
        "vllm": "vllm",
        "sglang": "sglang",
    }
    if not isinstance(value, str) or value.lower() not in aliases:
        raise ValueError(
            f"engine must be one of {', '.join(SUPPORTED_ENGINES)}"
        )
    return aliases[value.lower()]


def parse_comparison(document: dict[str, Any]) -> ComparisonIdentity:
    value = document.get("comparison")
    if not isinstance(value, dict):
        raise ValueError("engine config requires a comparison object")
    model = value.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("comparison.model must be a non-empty string")
    return ComparisonIdentity(model=model)


def parse_arm(
    value: Any, root: Path, comparison: ComparisonIdentity, index: int
) -> EngineArm:
    field = f"arms[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    label = value.get("label")
    if not isinstance(label, str) or LABEL_PATTERN.fullmatch(label) is None:
        raise ValueError(
            f"{field}.label must contain only letters, digits, '.', '_' or '-'"
        )
    engine = normalize_engine(value.get("engine"))
    executable = required_string(value.get("executable"), f"{field}.executable")
    model = required_string(value.get("model"), f"{field}.model")
    prefix_cache = value.get("prefix_cache", True)
    if not isinstance(prefix_cache, bool):
        raise ValueError(f"{field}.prefix_cache must be a boolean")
    served_model = value.get("served_model", comparison.model)
    if not isinstance(served_model, str) or not served_model:
        raise ValueError(f"{field}.served_model must be a non-empty string")
    cwd = optional_path(root, value.get("cwd"), f"{field}.cwd") or root
    return EngineArm(
        label=label,
        engine=engine,
        executable=executable,
        model=model,
        served_model=served_model,
        context_size=positive_int(value.get("context_size"), f"{field}.context_size"),
        max_concurrency=positive_int(
            value.get("max_concurrency"), f"{field}.max_concurrency"
        ),
        tokenizer=optional_string(value.get("tokenizer"), f"{field}.tokenizer"),
        hf_config=optional_string(value.get("hf_config"), f"{field}.hf_config"),
        prefix_cache=prefix_cache,
        batch_size=positive_int(
            value.get("batch_size"), f"{field}.batch_size", 2048
        ),
        ubatch_size=positive_int(
            value.get("ubatch_size"), f"{field}.ubatch_size", 512
        ),
        extra_args=string_list(value.get("extra_args"), f"{field}.extra_args"),
        cwd=cwd,
    )


def load_engine_config(path: Path) -> EngineConfig:
    resolved = path.expanduser().resolve()
    raw = resolved.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("engine config must be a JSON object")
    if document.get("schema_version") != 1:
        raise ValueError("engine config schema_version must be 1")
    comparison = parse_comparison(document)
    raw_arms = document.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("engine config requires at least one arm")
    arms = tuple(
        parse_arm(value, resolved.parent, comparison, index)
        for index, value in enumerate(raw_arms)
    )
    labels = [arm.label for arm in arms]
    if len(labels) != len(set(labels)):
        raise ValueError("engine arm labels must be unique")
    return EngineConfig(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        comparison=comparison,
        arms=arms,
    )


def engine_server_command(
    arm: EngineArm, host: str = "127.0.0.1", port: int = 9337
) -> list[str]:
    if arm.engine == "llama.cpp":
        command = [
            str(arm.executable),
            "--model",
            str(arm.model),
            "--alias",
            arm.served_model,
            "--host",
            host,
            "--port",
            str(port),
            "--ctx-size",
            str(arm.context_size),
            "--parallel",
            str(arm.max_concurrency),
            "--batch-size",
            str(arm.batch_size),
            "--ubatch-size",
            str(arm.ubatch_size),
            "--n-gpu-layers",
            "all",
            "--cont-batching",
            "--kv-unified",
            "--no-context-shift",
            "--metrics",
            "--no-webui",
        ]
        if not arm.prefix_cache:
            command.append("--no-cache-prompt")
    elif arm.engine == "vllm":
        command = [
            str(arm.executable),
            "serve",
            str(arm.model),
            "--served-model-name",
            arm.served_model,
            "--host",
            host,
            "--port",
            str(port),
            "--max-model-len",
            str(arm.context_size),
            "--max-num-seqs",
            str(arm.max_concurrency),
        ]
        if arm.tokenizer is not None:
            command.extend(("--tokenizer", str(arm.tokenizer)))
        if arm.hf_config is not None:
            command.extend(("--hf-config-path", str(arm.hf_config)))
        if arm.model.lower().endswith(".gguf"):
            command.extend(("--load-format", "gguf", "--quantization", "gguf"))
        command.append(
            "--enable-prefix-caching"
            if arm.prefix_cache
            else "--no-enable-prefix-caching"
        )
    elif arm.engine == "sglang":
        command = [
            str(arm.executable),
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(arm.model),
            "--served-model-name",
            arm.served_model,
            "--host",
            host,
            "--port",
            str(port),
            "--context-length",
            str(arm.context_size),
            "--max-running-requests",
            str(arm.max_concurrency),
            "--disable-prefill-cuda-graph",
        ]
        if arm.tokenizer is not None:
            command.extend(("--tokenizer-path", str(arm.tokenizer)))
        if arm.model.lower().endswith(".gguf"):
            command.extend(("--load-format", "gguf", "--quantization", "gguf"))
        if not arm.prefix_cache:
            command.append("--disable-radix-cache")
    else:  # pragma: no cover - load_engine_config validates this.
        raise ValueError(f"unsupported engine: {arm.engine}")
    command.extend(arm.extra_args)
    return command


def version_command(arm: EngineArm, executable: str) -> list[str]:
    if arm.engine == "sglang":
        return [
            executable,
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('sglang'))",
        ]
    return [executable, "--version"]


def engine_version(arm: EngineArm, executable: str) -> str:
    try:
        result = subprocess.run(
            version_command(arm, executable),
            text=True,
            capture_output=True,
            check=False,
            timeout=VERSION_COMMAND_TIMEOUT_SECONDS,
            # The SGLang query imports through sys.path, which includes the
            # current directory; run it where the server will run so a
            # configured checkout reports its own version.
            cwd=arm.cwd,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{arm.label} version command timed out after "
            f"{VERSION_COMMAND_TIMEOUT_SECONDS}s"
        ) from error
    value = (result.stdout or result.stderr).strip()
    if result.returncode != 0 or not value:
        raise RuntimeError(
            f"{arm.label} version command failed with status "
            f"{result.returncode}: {value or 'no output'}"
        )
    return value


def resolve_engine_executable(arm: EngineArm) -> str:
    """Resolve the arm executable the way the server launch will run it.

    Preflight must inspect the exact binary that ``start_server`` launches.
    That launch runs the command with ``cwd=arm.cwd``, which gives standard
    ``execvp`` semantics: a relative path with a separator (``./server``) is
    resolved against the launch working directory, an absolute path is used
    as-is, and a bare name is searched on the runner's ``PATH``. Matching that
    here guarantees the recorded version belongs to the launched executable.
    """
    has_separator = os.sep in arm.executable or (
        os.altsep is not None and os.altsep in arm.executable
    )
    if has_separator:
        candidate = Path(arm.executable).expanduser()
        if not candidate.is_absolute():
            candidate = arm.cwd / candidate
        if candidate.is_file():
            # absolutize WITHOUT dereferencing: resolve() would follow a
            # virtualenv's python symlink to the system interpreter, which
            # loses the venv's installed packages (sglang preflight/launch).
            return str(Path(os.path.abspath(candidate)))
        return ""
    return shutil.which(arm.executable) or ""


def verify_engine_arm(arm: EngineArm) -> dict[str, Any]:
    executable = resolve_engine_executable(arm)
    if not executable:
        raise FileNotFoundError(
            f"{arm.label} executable not found: {arm.executable} "
            f"(resolved against cwd {arm.cwd})"
        )
    if not arm.cwd.is_dir():
        raise FileNotFoundError(f"{arm.label} cwd not found: {arm.cwd}")
    version = engine_version(arm, executable)
    version_sha256 = hashlib.sha256(version.encode()).hexdigest()
    provenance = {
        **arm.plan_identity(),
        "resolved_executable": executable,
        "version": version,
        "version_sha256": version_sha256,
    }
    return {
        "label": arm.label,
        "engine": arm.engine,
        "ref": f"{arm.engine}@{version}",
        "commit": version_sha256,
        "version": version,
        "version_sha256": version_sha256,
        "worktree": str(arm.cwd),
        "binary": str(arm.executable),
        "runtime_root": "",
        "backend": arm.engine,
        "served_model": arm.served_model,
        "external_engine": arm.plan_identity(),
        "provenance": provenance,
    }


def external_server_environment(hf_home: Path | None) -> dict[str, str]:
    environment = os.environ.copy()
    if hf_home is not None:
        environment["HF_HOME"] = str(hf_home)
    return environment


def engine_order_specs(
    config: EngineConfig,
    version_sha256_by_label: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Return plan-order specs for the external arms.

    ``version_sha256_by_label`` supplies the verified arm identity once the
    runner has run preflight; without it the placeholder
    ``PENDING_VERSION_SHA256`` marks the identity as not yet verified.
    """
    verified = version_sha256_by_label or {}
    return [
        {
            "label": arm.label,
            "ref": f"external:{arm.engine}",
            "commit": verified.get(arm.label, PENDING_VERSION_SHA256),
        }
        for arm in config.arms
    ]


def labels_overlap(mesh_labels: Sequence[str], config: EngineConfig) -> set[str]:
    return set(mesh_labels).intersection(arm.label for arm in config.arms)
