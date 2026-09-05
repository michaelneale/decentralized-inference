#!/usr/bin/env python3
"""Run BOTH sides of one event-system A/B benchmark comparison, in ONE
invocation, interleaved per pair (spec §17.5, `.omo/plans/event-system.md`
task 19; `.omo/plans/event-system-fixes.md` task 14).

One invocation drives a fixed, deterministic set of paired trials for
EITHER comparison A (`--mode production --mode event-disabled` on one
`--binary`: the two sides differ by trial mode) OR comparison B (one
`--mode`, `--binary`/`--baseline-binary`: the two sides differ by binary)
-- see `resolve_comparison_sides`. For every pair, both sides' trials run
back to back in the seeded per-pair order (`build_trial_plan`'s
`side_order_first`, executed by `run_paired_trial_plan`), and the manifest
records the REAL observed `executed_order` -- never a label with no
execution effect. Full certification therefore runs this script twice:
once for comparison A (writing `manifest-production.json` +
`manifest-event-disabled.json`) and once for comparison B (writing
`manifest-current.json` + `manifest-baseline.json`), and the paired
comparator (`compare-event-benchmark-matrix.py`) consumes the resulting
manifests. A manifest records binary identity (path/sha256/`--version`)
rather than assuming which binary produced it.

`--mode event-disabled` forwards the hidden, TEST-ONLY selector
`MESH_LLM_EVENT_SYSTEM_TRIAL_MODE=event-disabled` to the spawned process,
which is accepted ONLY alongside `MESH_LLM_BENCHMARK_TUNE_TRIAL=1` (see
`crates/mesh-llm-config/src/env_overrides.rs`). This script always sets both
gate and selector consistently -- it never asks the binary to run this trial
mode without the trial gate.

Trial unit (mirrors `benchmark_trial_unit_definition()` in
`crates/mesh-llm-commands/src/gpus/tune/output_types.rs` VERBATIM -- see
`TRIAL_UNIT_DEFINITION` below and its cross-check test): one trial is one
fresh process launch, one readiness wait, one warmup request excluded from
metrics, one measured streaming request, and one shutdown. A pair is two
trials, one per side, with the same prompt and seed, side order randomized
per pair. Every trial's subject process is launched with `--log-format
json`; after its shutdown, `parse_final_event_system_health` reads back
its captured log for the FINAL `event_system_health` line, populating each
side's manifest `health`/`callback_ingress_p99_us` from that side's LAST
trial -- absence stays an honest JSON `null` that the comparator's
`health_is_available`/`evaluate_p99_gate` block on, never a fabricated
value.

This module deliberately never runs the real benchmark end-to-end as part of
its own test suite (see `scripts/tests/test_run_event_benchmark_matrix.py`):
every unit test exercises a pure function or injects a fake trial executor.
Real end-to-end certification runs are Task 21's job.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

MANIFEST_SCHEMA_VERSION = 1
METRICS_SCHEMA = "streaming_v1"

# The CLI's `--mode` values ARE the values forwarded verbatim as
# `MESH_LLM_EVENT_SYSTEM_TRIAL_MODE`; kept as an explicit alias map (rather
# than passing `args.mode` straight through) so a future CLI spelling change
# cannot silently become a wire-value change without a deliberate edit here.
MODE_TO_TRIAL_ENV_VALUE: dict[str, str] = {
    "production": "production",
    "event-disabled": "event-disabled",
}
VALID_MODES = tuple(MODE_TO_TRIAL_ENV_VALUE)

TRIAL_ENV_NAME = "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE"
TRIAL_GATE_ENV_NAME = "MESH_LLM_BENCHMARK_TUNE_TRIAL"

# Persist RAW values only for this explicit, non-sensitive allowlist,
# normalized as booleans/enums/numbers -- matches the plan's benchmark and
# certification protocol verbatim. Every other `MESH_LLM_*` name is
# redacted to name + `<redacted:present>`, never a raw value.
ENV_ALLOWLIST = (
    "MESH_LLM_LIFECYCLE_LOG_PARSER",
    TRIAL_GATE_ENV_NAME,
    TRIAL_ENV_NAME,
)

# Names matching this pattern are ALWAYS redacted, even if a future
# allowlist entry collided with one of them -- defense in depth, not just
# reliance on the allowlist staying hand-curated correctly.
SENSITIVE_NAME_PATTERN = re.compile(
    r"(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH|URL|PATH)", re.IGNORECASE
)
REDACTED_PRESENT = "<redacted:present>"

# Frozen `decode_only_tok_s` epsilon -- MUST match
# `streaming::DECODE_ONLY_TOK_S_EPSILON_SECS` in
# `crates/mesh-llm-commands/src/gpus/tune/benchmark/streaming.rs`. Duplicated
# (not imported -- this is a standalone Python tool) rather than redefined
# differently; the cross-check test in the paired test file pins this.
DECODE_ONLY_TOK_S_EPSILON_SECS = 1e-6

# Verbatim mirror of `benchmark_trial_unit_definition()` in
# `crates/mesh-llm-commands/src/gpus/tune/output_types.rs`. This is a REUSE
# of the frozen wording, not a redefinition -- the paired test file parses
# the Rust source and asserts these strings match after whitespace
# normalization, so the two can never silently drift apart.
TRIAL_UNIT_DEFINITION: dict[str, str] = {
    "trial": (
        "One trial is one fresh process launch, one readiness wait, "
        "one warmup request excluded from metrics, one measured "
        "streaming request, and one shutdown."
    ),
    "pair": (
        "A pair is two trials, one per side, with the same prompt "
        "and seed, side order randomized per pair."
    ),
}

PRIMARY_SCENARIO = "__primary__"

CERTIFICATION_HOSTS = {
    ("Darwin", "arm64"): "macos-arm64-metal",
    ("Linux", "x86_64"): "linux-x86_64-cuda",
}

U64_MAX = (1 << 64) - 1

DEFAULT_MAX_TOKENS = 64
DEFAULT_READINESS_TIMEOUT_SECS = 120.0
DEFAULT_REQUEST_TIMEOUT_SECS = 120.0
DEFAULT_READINESS_POLL_INTERVAL_SECS = 0.5
DEFAULT_SHUTDOWN_TIMEOUT_SECS = 15.0


def resolve_trial_env_value(mode: str) -> str:
    """Maps a `--mode` CLI value to the `MESH_LLM_EVENT_SYSTEM_TRIAL_MODE`
    wire value. Raises `ValueError` for any value outside `VALID_MODES` --
    argparse's `choices=` already prevents this in practice, but the mapping
    stays a hard error rather than a silent passthrough for direct callers."""
    try:
        return MODE_TO_TRIAL_ENV_VALUE[mode]
    except KeyError as exc:
        raise ValueError(f"unknown --mode {mode!r}; expected one of {VALID_MODES}") from exc


def validate_seed(seed: int) -> None:
    if seed < 0 or seed > U64_MAX:
        raise ValueError(f"--seed must fit in a u64 (0..={U64_MAX}), got {seed}")


def normalize_env_value(name: str, raw: str) -> Any:
    """Normalizes an allowlisted raw env-var string into a bool/enum/number
    for the manifest. Unknown (forward-compatible) allowlisted names
    normalize to the raw string unchanged."""
    if name == TRIAL_GATE_ENV_NAME:
        return raw == "1"
    return raw


def capture_environment_snapshot(environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Builds the manifest's persisted environment snapshot: only
    `MESH_LLM_*` names are recorded at all. An allowlisted, non-sensitive
    name is stored as its normalized value; every other name is stored as
    the redacted-presence marker only -- no raw value ever leaves this
    function for a non-allowlisted name, so a comparator reading two
    manifests in memory can compare presence/equality without ever holding
    a secret in the persisted artifact."""
    snapshot: dict[str, dict[str, Any]] = {}
    for name, raw in environ.items():
        if not name.startswith("MESH_LLM_"):
            continue
        if name in ENV_ALLOWLIST and not SENSITIVE_NAME_PATTERN.search(name):
            snapshot[name] = {"value": normalize_env_value(name, raw), "redacted": False}
        else:
            snapshot[name] = {"value": REDACTED_PRESENT, "redacted": True}
    return snapshot


def compute_file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_binary_identity(
    binary: Path,
    *,
    run_version: Callable[[Path], str | None] | None = None,
) -> dict[str, Any]:
    """Records enough identity for THIS SAME script to serve the baseline
    binary too: absolute path, sha256 of the file bytes (`None` if the path
    does not exist, e.g. a placeholder in a unit test), and `<binary>
    --version` stdout (best-effort; `None` when the binary cannot be
    executed). `run_version` is injectable so tests never spawn a real
    process."""
    resolved = binary.expanduser()
    sha256 = compute_file_sha256(resolved)
    if run_version is not None:
        version_output = run_version(resolved)
    else:
        version_output = _run_real_version_probe(resolved)
    return {"path": str(resolved), "sha256": sha256, "version": version_output}


def _run_real_version_probe(binary: Path) -> str | None:
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or "").strip() or (result.stderr or "").strip()
    return output or None


def capture_host_classification() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    certification_host = CERTIFICATION_HOSTS.get((system, machine))
    return {
        "system": system,
        "machine": machine,
        "certification_host": certification_host,
        "p99_gate": "enforced" if certification_host else "informational",
    }


def capture_thermal_state(
    *, run_pmset: Callable[[], subprocess.CompletedProcess[str] | None] | None = None,
    thermal_root: Path = Path("/sys/class/thermal"),
) -> dict[str, Any]:
    """Best-effort thermal/power/clock-state capture "where available" (the
    plan's exact phrase). Always returns a well-formed record -- never
    raises -- so a host without any readable thermal source still produces
    an explicit `{"available": False, ...}` record rather than a missing
    field."""
    system = platform.system()
    if system == "Darwin":
        result = run_pmset() if run_pmset is not None else _run_pmset_therm()
        if result is not None and result.returncode == 0 and result.stdout.strip():
            return {"available": True, "source": "pmset -g therm", "raw": result.stdout.strip()}
        return {"available": False, "source": "pmset -g therm"}
    if system == "Linux":
        zones: dict[str, int] = {}
        if thermal_root.is_dir():
            for zone_dir in sorted(thermal_root.glob("thermal_zone*")):
                temp_file = zone_dir / "temp"
                if not temp_file.is_file():
                    continue
                with contextlib.suppress(OSError, ValueError):
                    zones[zone_dir.name] = int(temp_file.read_text().strip())
        if zones:
            return {"available": True, "source": "sysfs", "zones_millidegrees_c": zones}
        return {"available": False, "source": "sysfs"}
    return {"available": False, "source": None}


def _run_pmset_therm() -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass(frozen=True)
class TrialPlanEntry:
    scenario: str
    pair_index: int
    prompt_seed: int
    side_order_first: str


def build_trial_plan(
    seed: int,
    pairs_primary: int,
    pairs_scenario: int,
    scenarios: Sequence[str],
    *,
    sides: tuple[str, str] = VALID_MODES,
) -> list[TrialPlanEntry]:
    """Deterministic trial plan from `seed`: `pairs_primary` entries in the
    synthetic `__primary__` group, then `pairs_scenario` entries per named
    `--scenario`, in the order scenarios were given. Two invocations of this
    function with the SAME `seed`/counts/scenarios -- as required when
    running production, event-disabled, and baseline through this script --
    produce an IDENTICAL plan, so the comparator can pair trial i of one
    manifest with trial i of another by (scenario, pair_index): "same
    prompt and seed" pairing without the two sides needing to run in the
    same process. Each entry also carries `side_order_first` -- which
    `sides` value is nominally "first" for that pair -- minted from the
    SAME seeded `rng` as `prompt_seed`. `sides` defaults to `VALID_MODES`
    (comparison A: one binary, two trial modes); pass
    `(side_a.side_id, side_b.side_id)` from `resolve_comparison_sides` for
    comparison B (one mode, two binaries -- e.g. `("current",
    "baseline")`), so the same deterministic-plan machinery covers both
    invocation shapes. Since Task 14, ONE invocation runs both sides of
    every pair back to back (see `run_paired_trial_plan`), so
    `side_order_first` is no longer just a label with no execution effect:
    it is the literal order `run_paired_trial_plan` executes trials in,
    and `executed_order` on the resulting manifests records what actually
    ran."""
    if pairs_primary < 1:
        raise ValueError("--pairs-primary must be at least 1")
    if pairs_scenario < 1:
        raise ValueError("--pairs-scenario must be at least 1")
    if not scenarios:
        raise ValueError("at least one --scenario is required")
    rng = random.Random(seed)
    plan: list[TrialPlanEntry] = []
    for index in range(pairs_primary):
        plan.append(TrialPlanEntry(PRIMARY_SCENARIO, index, rng.getrandbits(64), rng.choice(sides)))
    for scenario in scenarios:
        for index in range(pairs_scenario):
            plan.append(TrialPlanEntry(scenario, index, rng.getrandbits(64), rng.choice(sides)))
    return plan


@dataclass(frozen=True)
class SideSpec:
    """One side of a paired comparison run by ONE invocation of this
    script: a (binary, trial mode) pair plus a short `side_id` used both
    to mint the seeded per-pair order (`build_trial_plan`'s `sides`) and
    to name this side's output manifest file
    (`manifest-<side_id>.json`). Comparison A varies `mode` (both sides
    share `binary`, side_id == mode, e.g. "production"/"event-disabled");
    comparison B varies `binary` (both sides share `mode`, side_id ==
    "current"/"baseline")."""

    binary: Path
    mode: str
    side_id: str


def resolve_comparison_sides(
    binary: Path, baseline_binary: Path | None, modes: Sequence[str]
) -> tuple[SideSpec, SideSpec]:
    """Resolves the CLI's repeatable `--mode` and optional
    `--baseline-binary` into the invocation's two sides. Exactly one of
    two shapes is accepted -- see the module docstring -- any other
    combination is a hard `ValueError` (never a silent default, never a
    silent pick-one-and-ignore-the-rest):

    - Comparison B (`--baseline-binary` given): exactly one `--mode`
      value is required (both sides run the SAME mode; only the binary
      differs). Sides: `(binary, mode, "current")`,
      `(baseline_binary, mode, "baseline")`.
    - Comparison A (`--baseline-binary` omitted): `--mode` must be given
      exactly twice, covering both `VALID_MODES` with no repeat (the two
      sides differ by mode on the SAME binary). Sides:
      `(binary, modes[0], modes[0])`, `(binary, modes[1], modes[1])`.
    """
    if baseline_binary is not None:
        if len(modes) != 1:
            raise ValueError(
                "--baseline-binary (comparison B) requires exactly one --mode value "
                f"(both sides run the same mode); got {len(modes)}: {list(modes)!r}"
            )
        (mode,) = modes
        if mode not in VALID_MODES:
            raise ValueError(f"unknown --mode {mode!r}; expected one of {VALID_MODES}")
        return (
            SideSpec(binary=binary, mode=mode, side_id="current"),
            SideSpec(binary=baseline_binary, mode=mode, side_id="baseline"),
        )
    if len(modes) != 2 or set(modes) != set(VALID_MODES):
        raise ValueError(
            "without --baseline-binary (comparison A), --mode must be given exactly "
            f"twice, once for each of {VALID_MODES}; got {len(modes)}: {list(modes)!r}"
        )
    return (
        SideSpec(binary=binary, mode=modes[0], side_id=modes[0]),
        SideSpec(binary=binary, mode=modes[1], side_id=modes[1]),
    )


def compute_decode_only_tok_s(
    completion_tokens: int | None,
    total_elapsed_ms: float | None,
    ttft_ms: float | None,
) -> float | None:
    """`completion_tokens / max(total_elapsed - ttft, epsilon)`, mirroring
    `streaming::decode_only_tok_s` in
    `crates/mesh-llm-commands/src/gpus/tune/benchmark/streaming.rs`
    field-for-field: null (never zero) whenever `ttft_ms` is null or the
    decode interval is zero/negative."""
    if completion_tokens is None or total_elapsed_ms is None or ttft_ms is None:
        return None
    interval_secs = (total_elapsed_ms - ttft_ms) / 1000.0
    if interval_secs <= 0.0:
        return None
    return completion_tokens / max(interval_secs, DECODE_ONLY_TOK_S_EPSILON_SECS)


def compute_decode_tok_s(completion_tokens: int | None, total_elapsed_ms: float | None) -> float | None:
    """Preserves the historical `decode_tok_s = completion_tokens /
    total_request_elapsed` definition unchanged."""
    if completion_tokens is None or total_elapsed_ms is None or total_elapsed_ms <= 0.0:
        return None
    return completion_tokens / (total_elapsed_ms / 1000.0)


def sse_data_payload(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def first_choice_delta_content(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    return content if isinstance(content, str) else None


def terminal_usage_completion_tokens(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("completion_tokens")
    return tokens if isinstance(tokens, int) else None


@dataclass
class StreamParseResult:
    completion_tokens: int | None
    ttft_ms: float | None
    malformed: bool = False


def parse_sse_stream(
    lines: Iterable[str],
    *,
    clock: Callable[[], float],
    started_at: float,
) -> StreamParseResult:
    """Parses an SSE chat-completion stream line-by-line, mirroring
    `streaming::parse_streaming_chat_response` in
    `crates/mesh-llm-commands/src/gpus/tune/benchmark/streaming.rs`:
    malformed individual chunks are skipped (not fatal), `[DONE]` or EOF
    ends the stream, TTFT is measured at the first non-empty content delta,
    and `completion_tokens` comes from the terminal `usage` object. A
    stream that never produces terminal usage returns `completion_tokens =
    None` (never zero) with `malformed = True`."""
    ttft_ms: float | None = None
    completion_tokens: int | None = None
    for raw_line in lines:
        payload = sse_data_payload(raw_line)
        if payload is None:
            continue
        if payload == "[DONE]":
            break
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if ttft_ms is None:
            content = first_choice_delta_content(value)
            if content:
                ttft_ms = (clock() - started_at) * 1000.0
        tokens = terminal_usage_completion_tokens(value)
        if tokens is not None:
            completion_tokens = tokens
    return StreamParseResult(
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        malformed=completion_tokens is None,
    )


def reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def build_chat_request_body(prompt: str, max_tokens: int, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def first_models_list_id(payload: dict[str, Any]) -> str | None:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    model_id = first.get("id")
    return model_id if isinstance(model_id, str) and model_id else None


def resolve_ready_model_id(base_url: str, timeout_secs: float) -> str | None:
    """`--local-model-only` (the mode every trial runs under) has no mesh/
    routing layer, so the `model: "auto"` alias every OTHER OpenAI-compatible
    surface in this repo accepts is REJECTED here with `404 model_not_found`
    -- confirmed against a real running binary (task 21; discovered on the
    FIRST real end-to-end run of this harness, exactly the kind of edge task
    19's own problems.md note predicted). The trial must resolve the actual
    served model id from `/v1/models` and address it explicitly."""
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout_secs) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None
    return first_models_list_id(payload)


def send_streaming_chat_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    timeout_secs: float,
    model: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> StreamParseResult:
    body = json.dumps(build_chat_request_body(prompt, max_tokens, model)).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = clock()
    with urllib.request.urlopen(request, timeout=timeout_secs) as response:
        lines = (raw.decode("utf-8", errors="replace") for raw in response)
        return parse_sse_stream(lines, clock=clock, started_at=started_at)


def wait_for_readiness(
    base_url: str,
    timeout_secs: float,
    poll_interval_secs: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    deadline = clock() + timeout_secs
    while clock() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=poll_interval_secs) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        sleep(poll_interval_secs)
    return False


@dataclass
class TrialResult:
    scenario: str
    pair_index: int
    side_order_first: str
    status: str
    completion_tokens: int | None
    elapsed_ms: float | None
    decode_tok_s: float | None
    ttft_ms: float | None
    decode_only_tok_s: float | None
    setup_ms: float | None
    readiness_ms: float | None
    shutdown_ms: float | None
    error: str | None = None


def prompt_for_entry(entry: TrialPlanEntry) -> str:
    """A short, deterministic prompt derived from the entry's seeded value
    so every pair has a reproducible, comparable prompt without a fixture
    corpus dependency."""
    return f"Respond with a short factual sentence. token={entry.prompt_seed:016x}"


HEALTH_LOG_CONTEXT = "event_system_health"

# Verbatim field names from `health_projection_event()` in
# `crates/mesh-llm-host-runtime/src/runtime_events/presentation/projection.rs`
# -- the space-separated `key=value` tokens inside the JSON log line's
# `message` string (NOT nested JSON; the envelope is flat: `timestamp`,
# `level`, `event`, `message`, `context`, with `context ==
# "event_system_health"` identifying this line). Persisted into the
# manifest's `health` dict verbatim (as ints) when present; `bounds.*` and
# `ingress_p99_us` are intentionally excluded here -- p99 has its own
# manifest field (`callback_ingress_p99_us`) and `bounds` is a
# certification-tooling concern the comparator does not consume from this
# manifest.
HEALTH_COUNTER_FIELDS: tuple[str, ...] = (
    "version",
    "rebuild_generation",
    "reservation_exhausted",
    "cancelled_reservation_rejected",
    "terminal_delivery_failed",
    "dropped_progress",
    "dropped_diagnostic",
    "replay_evicted",
    "subscriber_disconnected",
    "shutdown_degraded",
    "reducer_rejected",
    "state_transition_rejected",
    "state_degraded",
    "rebuild_required",
)

INGRESS_P99_FIELD = "ingress_p99_us"


def _coerce_health_field_value(raw_value: str) -> int | float | None:
    """`null` (the literal token `health_projection_event` writes when
    `ingress_p99_us` is unmeasured) becomes `None`; every other value is an
    int when possible, else a float -- never a string, so callers never
    have to re-parse a numeric-looking value themselves."""
    if raw_value == "null":
        return None
    if raw_value in ("true", "false"):
        return raw_value == "true"
    try:
        return int(raw_value)
    except ValueError:
        return float(raw_value)


def parse_event_system_health_message(message: str) -> dict[str, int | float | None]:
    """Parses ONE `event_system_health` log line's `message` string (the
    space-separated `key=value` tokens `health_projection_event` writes,
    e.g. `"version=1 ... dropped_progress=0 ... ingress_p99_us=6"`) into a
    flat dict keyed by field name. Unknown/malformed tokens (no `=`) are
    skipped rather than raising -- a forward-compatible field this parser
    does not yet know about must never abort the whole parse."""
    fields: dict[str, int | float | None] = {}
    for token in message.split():
        if "=" not in token:
            continue
        key, _, raw_value = token.partition("=")
        try:
            fields[key] = _coerce_health_field_value(raw_value)
        except ValueError:
            continue
    return fields


def parse_final_event_system_health(log_text: str) -> dict[str, int | float | None] | None:
    """Scans a subject process's `--log-format json` combined stdout/stderr
    log, line by line, for `event_system_health` lines -- JSON envelopes
    shaped `{"timestamp": ..., "level": ..., "event": ..., "message": "key=value
    ...", "context": "event_system_health"}` -- and returns the FIELDS
    parsed from the message of the FINAL such line, or `None` if no such
    line appears anywhere in the log (e.g. the process crashed before its
    startup health line, or never wrote one at all). This is the last line
    the per-subscriber health-delivery gate allowed to be emitted before
    the trial's shutdown, NOT a synthesized end-of-trial snapshot -- see
    `health-parsing-note.txt` in this task's evidence for what that does
    and does not prove. A line that fails to `json.loads` (partial write,
    non-JSON TUI banner, etc.) is skipped, never fatal."""
    last_fields: dict[str, int | float | None] | None = None
    for raw_line in log_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            envelope = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(envelope, dict) or envelope.get("context") != HEALTH_LOG_CONTEXT:
            continue
        message = envelope.get("message")
        if not isinstance(message, str):
            continue
        last_fields = parse_event_system_health_message(message)
    return last_fields


def derive_health_and_p99(
    fields: dict[str, int | float | None] | None,
) -> tuple[dict[str, int | float | None] | None, float | None]:
    """Maps parsed `event_system_health` fields onto this manifest's
    `health` dict and `callback_ingress_p99_us` shapes. Absence of a health
    line at all (`fields is None`) stays an honest `(None, None)` -- never
    a fabricated all-zero health or a zero p99 -- so
    `health_is_available`/`evaluate_p99_gate` in the comparator keep
    blocking exactly as they did before this task, now fed real data
    instead of an always-`None` manifest. When a health line WAS found but
    its `ingress_p99_us` token is the literal `null` (fewer than
    `INGRESS_LATENCY_MIN_SAMPLES` submissions recorded in that trial's
    process lifetime), `health` is still populated (real counters exist)
    while `callback_ingress_p99_us` stays `None` -- these two null checks
    are independent by design, matching the pre-existing comparator gates
    this task does not change."""
    if fields is None:
        return None, None
    health = {name: fields[name] for name in HEALTH_COUNTER_FIELDS if name in fields}
    p99_raw = fields.get(INGRESS_P99_FIELD)
    p99 = float(p99_raw) if isinstance(p99_raw, (int, float)) else None
    return health, p99


def execute_trial(
    binary: Path,
    model: str,
    mode: str,
    entry: TrialPlanEntry,
    log_path: Path,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    readiness_timeout_secs: float = DEFAULT_READINESS_TIMEOUT_SECS,
    readiness_poll_interval_secs: float = DEFAULT_READINESS_POLL_INTERVAL_SECS,
    request_timeout_secs: float = DEFAULT_REQUEST_TIMEOUT_SECS,
    shutdown_timeout_secs: float = DEFAULT_SHUTDOWN_TIMEOUT_SECS,
) -> TrialResult:
    """Real trial execution: one fresh `<binary> --local-model-only`
    process launch, one readiness wait, one warmup request (excluded from
    metrics), one measured streaming request, one shutdown -- the exact
    trial unit `TRIAL_UNIT_DEFINITION` describes. `log_path` is where this
    ONE trial's subject process combined stdout/stderr is captured
    (`--log-format json` is always passed so the paired orchestrator,
    `run_paired_trial_plan`, can parse the subject's own final
    `event_system_health` line via `parse_final_event_system_health` after
    shutdown -- Task 14's health/p99 plumbing). NEVER called by this
    module's own test suite; `run_paired_trial_plan` takes an injectable
    `trial_executor` so tests exercise manifest assembly without spawning a
    real process (see the paired test file)."""
    port = reserve_local_port()
    base_url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env[TRIAL_GATE_ENV_NAME] = "1"
    env[TRIAL_ENV_NAME] = resolve_trial_env_value(mode)
    # `--local-model-only` REJECTS `--headless` at CLI validation
    # ("--local-model-only never starts a console; remove --headless" --
    # `validate_local_model_only_options` in
    # `crates/mesh-llm-host-runtime/src/runtime/local_model_only.rs`) and
    # never starts a console/management API at all ("--local-model-only
    # does not start owner control or management APIs", same function) --
    # discovered running this path against a REAL binary for the first
    # time (task 21; task 19 explicitly never exercised this code path).
    # `--console`/`--headless` are therefore NEVER passed here: readiness
    # only ever polls the OpenAI API port (`wait_for_readiness` below), and
    # a `--console` value would be silently pointless even if accepted.
    # `--speculative-strategy disabled` is REQUIRED, not optional: without
    # it, native in-model MTP speculative decoding (baked into some GGUFs,
    # e.g. the approved Task 8 fixture) crashes every real inference request
    # with `RuntimeError: llama_decode failed` -- confirmed against a real
    # running binary (task 21). The native skippy log shows the exact
    # native cause: `decode: backend sampling requires at most one output
    # token per sequence (seq_id 0 had 2)`, `llama_decode: failed to
    # decode, ret = -1`. `--no-draft` does NOT fix this (it only disables
    # detection of a separate sibling draft-model file, not native
    # in-model MTP tensors). This is a genuine native llama.cpp/skippy
    # correctness bug -- fixing it is out of scope for this certification
    # task; disabling native MTP uniformly on every trial side (current,
    # event-disabled, baseline) keeps the comparison fair (identical
    # serving profile on every side) while letting real measurements land
    # at all.
    argv = [
        str(binary),
        "serve",
        "--local-model-only",
        "--model",
        model,
        "--port",
        str(port),
        "--speculative-strategy",
        "disabled",
        "--log-format",
        "json",
    ]
    prompt = prompt_for_entry(entry)
    setup_started = time.monotonic()
    # `--log-format json` (added above) plus redirecting to a real FILE
    # (never `subprocess.PIPE`, which this script never drains -- an
    # undrained pipe deadlocks once the OS pipe buffer fills) is what lets
    # `run_paired_trial_plan` read this trial's `event_system_health` line
    # back after shutdown via `parse_final_event_system_health`.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    process = subprocess.Popen(  # noqa: S603 - trusted local binary under test
        argv, env=env, stdout=log_file, stderr=subprocess.STDOUT
    )
    setup_ms = (time.monotonic() - setup_started) * 1000.0
    try:
        readiness_started = time.monotonic()
        ready = wait_for_readiness(base_url, readiness_timeout_secs, readiness_poll_interval_secs)
        readiness_ms = (time.monotonic() - readiness_started) * 1000.0
        if not ready:
            return TrialResult(
                scenario=entry.scenario,
                pair_index=entry.pair_index,
                side_order_first=entry.side_order_first,
                status="failed",
                completion_tokens=None,
                elapsed_ms=None,
                decode_tok_s=None,
                ttft_ms=None,
                decode_only_tok_s=None,
                setup_ms=setup_ms,
                readiness_ms=readiness_ms,
                shutdown_ms=None,
                error="readiness timeout",
            )
        served_model_id = resolve_ready_model_id(base_url, request_timeout_secs)
        if served_model_id is None:
            return TrialResult(
                scenario=entry.scenario,
                pair_index=entry.pair_index,
                side_order_first=entry.side_order_first,
                status="failed",
                completion_tokens=None,
                elapsed_ms=None,
                decode_tok_s=None,
                ttft_ms=None,
                decode_only_tok_s=None,
                setup_ms=setup_ms,
                readiness_ms=readiness_ms,
                shutdown_ms=None,
                error="/v1/models returned no model id after readiness",
            )
        with contextlib.suppress(urllib.error.URLError, OSError, TimeoutError):
            send_streaming_chat_request(base_url, prompt, max_tokens, request_timeout_secs, served_model_id)
        measured_started = time.monotonic()
        try:
            parsed = send_streaming_chat_request(
                base_url, prompt, max_tokens, request_timeout_secs, served_model_id
            )
            elapsed_ms = (time.monotonic() - measured_started) * 1000.0
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return TrialResult(
                scenario=entry.scenario,
                pair_index=entry.pair_index,
                side_order_first=entry.side_order_first,
                status="failed",
                completion_tokens=None,
                elapsed_ms=None,
                decode_tok_s=None,
                ttft_ms=None,
                decode_only_tok_s=None,
                setup_ms=setup_ms,
                readiness_ms=readiness_ms,
                shutdown_ms=None,
                error=str(exc),
            )
    finally:
        shutdown_started = time.monotonic()
        process.terminate()
        try:
            process.wait(timeout=shutdown_timeout_secs)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=shutdown_timeout_secs)
        shutdown_ms = (time.monotonic() - shutdown_started) * 1000.0
        log_file.close()

    if parsed.malformed or parsed.completion_tokens is None:
        return TrialResult(
            scenario=entry.scenario,
            pair_index=entry.pair_index,
            side_order_first=entry.side_order_first,
            status="failed",
            completion_tokens=None,
            elapsed_ms=elapsed_ms,
            decode_tok_s=None,
            ttft_ms=None,
            decode_only_tok_s=None,
            setup_ms=setup_ms,
            readiness_ms=readiness_ms,
            shutdown_ms=shutdown_ms,
            error="stream ended without terminal usage",
        )
    return TrialResult(
        scenario=entry.scenario,
        pair_index=entry.pair_index,
        side_order_first=entry.side_order_first,
        status="succeeded",
        completion_tokens=parsed.completion_tokens,
        elapsed_ms=elapsed_ms,
        decode_tok_s=compute_decode_tok_s(parsed.completion_tokens, elapsed_ms),
        ttft_ms=parsed.ttft_ms,
        decode_only_tok_s=compute_decode_only_tok_s(parsed.completion_tokens, elapsed_ms, parsed.ttft_ms),
        setup_ms=setup_ms,
        readiness_ms=readiness_ms,
        shutdown_ms=shutdown_ms,
        error=None,
    )


TrialExecutor = Callable[[Path, str, str, TrialPlanEntry, Path], TrialResult]


@dataclass(frozen=True)
class ExecutedOrderEntry:
    """One pair's REAL observed execution order -- as opposed to the
    plan's `side_order_first` label, which (before Task 14) was recorded
    with no execution effect since production/event-disabled ran as
    separate invocations. `order` is `(first.side_id, second.side_id)`,
    the side_ids of the two trials `run_paired_trial_plan` actually
    launched back to back for this pair, in launch order."""

    scenario: str
    pair_index: int
    order: tuple[str, str]


@dataclass(frozen=True)
class PairedTrialPlanResult:
    """Everything one interleaved invocation produces for its two sides:
    each side's trial results (for its own manifest), the shared
    `executed_order` record (written identically onto both manifests --
    see `evaluate_executed_order_consistency` in the comparator), and each
    side's `health`/`ingress_p99_us` as parsed from that side's OWN LAST
    trial's `event_system_health` log line (see
    `health_and_p99_from_trial_log`)."""

    side_a_results: list[TrialResult]
    side_b_results: list[TrialResult]
    executed_order: list[ExecutedOrderEntry]
    side_a_health: dict[str, int | float | None] | None
    side_a_ingress_p99_us: float | None
    side_b_health: dict[str, int | float | None] | None
    side_b_ingress_p99_us: float | None


def health_and_p99_from_trial_log(log_path: Path) -> tuple[dict[str, int | float | None] | None, float | None]:
    """Reads back ONE trial's captured `--log-format json` log file and
    extracts health/p99 via `parse_final_event_system_health` +
    `derive_health_and_p99`. A missing file (e.g. the trial never even
    launched) is treated identically to an empty log: an honest `(None,
    None)`, never an exception that would abort the whole run over one
    unreadable trial log."""
    if not log_path.is_file():
        return None, None
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return derive_health_and_p99(parse_final_event_system_health(log_text))


def run_paired_trial_plan(
    model: str,
    plan: Sequence[TrialPlanEntry],
    side_a: SideSpec,
    side_b: SideSpec,
    *,
    log_dir: Path,
    trial_executor: TrialExecutor = execute_trial,
) -> PairedTrialPlanResult:
    """Runs BOTH sides of every pair in ONE invocation -- the fix for
    D13's "each invocation runs one side" defect. For each pair, in the
    SEEDED per-pair order (`entry.side_order_first`, which names one of
    `side_a.side_id`/`side_b.side_id`), executes that side's trial THEN
    the other side's trial immediately after (back to back: pair 0's two
    trials, then pair 1's two trials, ... -- never all of side A's trials
    followed by all of side B's, which is what running the two sides as
    separate invocations produced before this task). `executed_order`
    records, for every pair, the two side_ids in the order they were
    ACTUALLY launched -- by construction equal to the plan, since this is
    the same loop that both decides and executes that order. Each side's
    `health`/`ingress_p99_us` come from that side's LAST trial's captured
    log (each trial is a fresh process, per `TRIAL_UNIT_DEFINITION`, so a
    fresh trial's own health line is the only honest per-side reading
    available without a live console API)."""
    side_a_results: list[TrialResult] = []
    side_b_results: list[TrialResult] = []
    executed_order: list[ExecutedOrderEntry] = []
    side_a_last_log: Path | None = None
    side_b_last_log: Path | None = None
    for entry in plan:
        first, second = (side_a, side_b) if entry.side_order_first == side_a.side_id else (side_b, side_a)

        first_log = log_dir / f"{entry.scenario}-{entry.pair_index}-{first.side_id}.log"
        first_result = trial_executor(first.binary, model, first.mode, entry, first_log)
        if first is side_a:
            side_a_results.append(first_result)
            side_a_last_log = first_log
        else:
            side_b_results.append(first_result)
            side_b_last_log = first_log

        second_log = log_dir / f"{entry.scenario}-{entry.pair_index}-{second.side_id}.log"
        second_result = trial_executor(second.binary, model, second.mode, entry, second_log)
        if second is side_a:
            side_a_results.append(second_result)
            side_a_last_log = second_log
        else:
            side_b_results.append(second_result)
            side_b_last_log = second_log

        executed_order.append(
            ExecutedOrderEntry(scenario=entry.scenario, pair_index=entry.pair_index, order=(first.side_id, second.side_id))
        )

    side_a_health, side_a_p99 = (
        health_and_p99_from_trial_log(side_a_last_log) if side_a_last_log is not None else (None, None)
    )
    side_b_health, side_b_p99 = (
        health_and_p99_from_trial_log(side_b_last_log) if side_b_last_log is not None else (None, None)
    )
    return PairedTrialPlanResult(
        side_a_results=side_a_results,
        side_b_results=side_b_results,
        executed_order=executed_order,
        side_a_health=side_a_health,
        side_a_ingress_p99_us=side_a_p99,
        side_b_health=side_b_health,
        side_b_ingress_p99_us=side_b_p99,
    )


# D-6 (`.omo/evidence/event-system-fixes/deferrals/d6/`): per-TRIAL drop
# counts confirmed IDENTICAL across 3 independent full 30-pair
# certification runs (`final/f4/f4-manifests.txt`). Not scaled by trial
# count -- see the docstring below.
EVENT_DISABLED_EXPECTED_DROPPED_PROGRESS_PER_TRIAL = 1
EVENT_DISABLED_EXPECTED_DROPPED_DIAGNOSTIC_PER_TRIAL = 0


def summarize_health_expectations(mode: str, results: Sequence[TrialResult]) -> dict[str, int]:
    """The exact-count health expectation this manifest can prove without a
    live console API attached, SCOPED TO THE SAME SINGLE TRIAL that
    `health` actually reflects (`health_and_p99_from_trial_log` reads only
    the side's LAST trial's log -- never a per-run total across every
    trial the matrix attempted).

    D-6 fix: before this change, `event-disabled` mode's expectation
    scaled with `len(results)` (the whole side's attempted-trial count
    across the matrix, e.g. 30 for a 20+10-pair run) under the assumption
    that `health` would be a cumulative count across the whole run. It
    never was -- `health` has been single-last-trial-scoped since Task 14
    -- so this fired `health_expectation_violation` on every real
    event-disabled manifest regardless of pair count, seed, or parser mode
    (F4 certification wave, `.omo/evidence/event-system-fixes/final/f4/
    f4-verdict.md`, "New finding" section). Reconciled here to the fixed
    per-trial counts above, which do NOT scale with `len(results)` --
    matching what a single trial's Progress/Diagnostic bypass reproducibly
    produces. Under `production`, expected drops are zero regardless of
    scope: a correctly sized reservation table coalesces progress and has
    ample diagnostic headroom for one benchmark trial's traffic. `results`
    is used only to detect the no-trials-ran edge case (nothing attempted,
    so nothing expected) -- once at least one trial ran, the expectation
    does not grow with how many more trials the matrix goes on to run."""
    if not results:
        return {"expected_dropped_progress": 0, "expected_dropped_diagnostic": 0}
    if mode == "event-disabled":
        return {
            "expected_dropped_progress": EVENT_DISABLED_EXPECTED_DROPPED_PROGRESS_PER_TRIAL,
            "expected_dropped_diagnostic": EVENT_DISABLED_EXPECTED_DROPPED_DIAGNOSTIC_PER_TRIAL,
        }
    return {"expected_dropped_progress": 0, "expected_dropped_diagnostic": 0}


def build_manifest(
    *,
    binary: Path,
    model: str,
    mode: str,
    seed: int,
    pairs_primary: int,
    pairs_scenario: int,
    scenarios: Sequence[str],
    results: Sequence[TrialResult],
    environ: Mapping[str, str],
    attempt: int = 1,
    generated_at: str,
    run_version: Callable[[Path], str | None] | None = None,
    thermal_state: dict[str, Any] | None = None,
    host: dict[str, Any] | None = None,
    callback_ingress_p99_us: float | None = None,
    health: dict[str, int] | None = None,
    executed_order: Sequence[ExecutedOrderEntry] | None = None,
) -> dict[str, Any]:
    expectations = summarize_health_expectations(mode, results)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metrics_schema": METRICS_SCHEMA,
        "mode": mode,
        "binary": capture_binary_identity(binary, run_version=run_version),
        "model": model,
        "seed": seed,
        "pairs_primary": pairs_primary,
        "pairs_scenario": pairs_scenario,
        "scenarios": list(scenarios),
        "attempt": attempt,
        "generated_at": generated_at,
        "host": host if host is not None else capture_host_classification(),
        "thermal_state": thermal_state if thermal_state is not None else capture_thermal_state(),
        "environment": capture_environment_snapshot(environ),
        "trial_unit": dict(TRIAL_UNIT_DEFINITION),
        "callback_ingress_p99_us": callback_ingress_p99_us,
        # Honest reporting, not fabrication: a caller that never collected
        # health data must leave `health` as `None` (JSON null), which the
        # comparator's `health_is_available` reads as "unmeasured" and
        # BLOCKS on -- never as a silently-zero `{}`, which would let a
        # missing `!= 0` invariant check vacuously pass. `health` stays
        # unchanged when a caller supplies one; since Task 14,
        # `run_paired_trial_plan` supplies a real value parsed from the
        # subject process's own `event_system_health` log line whenever
        # that line was found (see `health_and_p99_from_trial_log`).
        "health": health,
        "expected_dropped_progress": expectations["expected_dropped_progress"],
        "expected_dropped_diagnostic": expectations["expected_dropped_diagnostic"],
        "trials": [asdict(result) for result in results],
        # Present on BOTH sides' manifests from the SAME invocation --
        # see `evaluate_executed_order_consistency` in the comparator.
        # `None` (JSON null) when the caller never ran an interleaved plan
        # (e.g. a unit test building a manifest in isolation) -- never a
        # fabricated empty list standing in for "never executed".
        "executed_order": (
            [
                {"scenario": entry.scenario, "pair_index": entry.pair_index, "order": list(entry.order)}
                for entry in executed_order
            ]
            if executed_order is not None
            else None
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-event-benchmark-matrix.py",
        description=(
            "Run BOTH sides of one event-system paired-benchmark comparison in ONE "
            "invocation, interleaved back to back per pair in the seeded order, and "
            "write one manifest per side. Accepts EITHER `--mode production --mode "
            "event-disabled` on one --binary (comparison A) OR one --mode with "
            "--binary/--baseline-binary (comparison B)."
        ),
    )
    parser.add_argument("--binary", required=True, type=Path, help="Path to the mesh-llm binary under test.")
    parser.add_argument(
        "--baseline-binary",
        type=Path,
        default=None,
        help=(
            "Path to the verified baseline release binary. When given, selects "
            "comparison B (--mode must then be given exactly once; both sides run "
            "that same mode, one per binary)."
        ),
    )
    parser.add_argument("--model", required=True, help="Approved deterministic local model reference.")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to write the manifests and evidence into."
    )
    parser.add_argument(
        "--pairs-primary", required=True, type=int, help="Number of primary-comparison trial pairs (>=1)."
    )
    parser.add_argument(
        "--pairs-scenario", required=True, type=int, help="Number of trial pairs per named scenario (>=1)."
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Deterministic u64 seed; reuse the SAME seed across the production/event-disabled/baseline runs being compared.",
    )
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        required=True,
        choices=list(VALID_MODES),
        help=(
            "Hidden trial selector forwarded as MESH_LLM_EVENT_SYSTEM_TRIAL_MODE (also "
            "sets MESH_LLM_BENCHMARK_TUNE_TRIAL=1). Repeatable: give it twice (once per "
            "value) for comparison A, or once for comparison B."
        ),
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        required=True,
        help="Named scenario; repeatable, at least one required.",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help=(
            "The predefined-retry attempt number (1 or 2) this run represents, recorded "
            "verbatim into the manifest's `attempt` field for the comparator's "
            "evaluate_retry_state gate. Defaults to 1 (first attempt). Pass 2 for the ONE "
            "predefined full-set retry permitted after an adverse first-attempt result and "
            "after recording/correcting a thermal/load/runtime mismatch -- never invent a "
            "3rd attempt."
        ),
    )
    return parser


def _build_side_manifest(
    *,
    side: SideSpec,
    args: argparse.Namespace,
    plan_result: PairedTrialPlanResult,
    results: Sequence[TrialResult],
    health: dict[str, int | float | None] | None,
    ingress_p99_us: float | None,
    generated_at: str,
) -> dict[str, Any]:
    return build_manifest(
        binary=side.binary,
        model=args.model,
        mode=side.mode,
        seed=args.seed,
        pairs_primary=args.pairs_primary,
        pairs_scenario=args.pairs_scenario,
        scenarios=args.scenarios,
        results=results,
        environ=os.environ,
        attempt=args.attempt,
        generated_at=generated_at,
        callback_ingress_p99_us=ingress_p99_us,
        health=health,
        executed_order=plan_result.executed_order,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        validate_seed(args.seed)
        side_a, side_b = resolve_comparison_sides(args.binary, args.baseline_binary, args.modes)
        plan = build_trial_plan(
            args.seed, args.pairs_primary, args.pairs_scenario, args.scenarios, sides=(side_a.side_id, side_b.side_id)
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "trial-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plan_result = run_paired_trial_plan(args.model, plan, side_a, side_b, log_dir=log_dir)
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    manifest_paths: dict[str, str] = {}
    for side, results, health, ingress_p99_us in (
        (side_a, plan_result.side_a_results, plan_result.side_a_health, plan_result.side_a_ingress_p99_us),
        (side_b, plan_result.side_b_results, plan_result.side_b_health, plan_result.side_b_ingress_p99_us),
    ):
        manifest = _build_side_manifest(
            side=side,
            args=args,
            plan_result=plan_result,
            results=results,
            health=health,
            ingress_p99_us=ingress_p99_us,
            generated_at=generated_at,
        )
        manifest_path = args.output_dir / f"manifest-{side.side_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_paths[side.side_id] = str(manifest_path)

    print(json.dumps({"manifest_paths": manifest_paths}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
