#!/usr/bin/env python3
"""KV restart replay: measure serving latency across a process restart.

Issue #1647-A. Runs one frozen multi-turn conversation against a default-startup
``mesh-llm serve`` in three cohorts:

- ``fill``    — cold server, conversation grows turn by turn (prefix reuse).
- ``restore`` — the server is stopped and restarted on the same state directory,
                then the frozen full conversation is replayed. On a build with a
                durable KV tier this measures first-request-after-restart
                restoration; without one it is the cold-prefill reference.
- ``warm``    — repeat replays without restart (resident reuse reference).

The runner never sets context size, lanes, KV budget, or backend tuning; the
only serving arguments are ``--model``, ``--log-format json`` and the explicit
``--serve-extra-args`` pass-through an operator asks for (for example a
``--kv-cache-disk`` mode under test). Everything measured is observational:
streaming TTFT from the first chunk, usage from ``stream_options.include_usage``,
cached tokens from ``prompt_tokens_details``.

Artifacts: ``run.json`` (schema_version 1, provenance + config + cohort
summaries), ``requests.jsonl`` (one row per request), ``report.md`` (human
summary). The manifest itself is deterministic from ``--turns`` and
``--turn-target-tokens`` and is embedded (with its SHA-256) into ``run.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://127.0.0.1:9337/v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9337
SCHEMA_VERSION = 1

# Same discipline as evals/agentic-replay.py: a default-startup benchmark may
# not tune the server. Extra serving arguments must arrive explicitly via
# --serve-extra-args and are recorded verbatim in run.json.
FORBIDDEN_STARTUP_OPTIONS = (
    "--ctx-size",
    "--generation-concurrency",
    "--generation-queue-capacity",
    "--max-vram",
    "--parallel",
)

# Deterministic manifest vocabulary. The conversation simulates a long-running
# coding-agent session: a stable scaffold, a growing project brief, and a
# per-turn request. Content is drawn from a fixed word list with a fixed PRNG
# seed so the same settings always produce the same conversation.
SEED = 20260909
_VOCAB = (
    "cache prefix token restore restart segment manifest budget eviction "
    "prefill decode latency throughput checkpoint durable radix tier admission "
    "pipeline stream verify digest commit quarantine pin lease reserve node "
    "mesh relay model runtime kernel attention matrix layer head batch queue "
    "trace replay harness baseline cohort percentile regression gate promote"
).split()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class DeterministicRandom:
    """Small LCG so manifests do not depend on the host Python random module."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFF

    def next(self) -> int:
        self.state = (self.state * 25214903917 + 11) & 0xFFFFFFFFFFFF
        return self.state >> 16

    def below(self, bound: int) -> int:
        return self.next() % bound

    def words(self, count: int) -> list[str]:
        return [_VOCAB[self.below(len(_VOCAB))] for _ in range(count)]


def build_manifest(turns: int, turn_target_tokens: int, system_tokens: int) -> dict[str, Any]:
    """Build the frozen conversation. Turn sizes are approximate (words * 4/3);
    the authoritative prompt token counts come from server usage at run time."""

    rng = DeterministicRandom(SEED)

    def block(target_tokens: int, topic: str) -> str:
        words = max(1, int(target_tokens * 3 / 4))
        chunks = []
        while len(chunks) * 8 < words:
            chunks.append(" ".join(rng.words(8)))
        return f"[{topic}] " + " ".join(chunks)

    scaffold = block(system_tokens, "scaffold")
    turn_specs = []
    for index in range(turns):
        body = block(turn_target_tokens, f"turn-{index + 1}-context")
        request = (
            f"Turn {index + 1}: given the project brief above, summarize the "
            f"{' '.join(rng.words(6))} constraint in one sentence and list the "
            f"{' '.join(rng.words(4))} next step."
        )
        turn_specs.append({"context": body, "request": request})

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "kv-restart-replay/manifest",
        "seed": SEED,
        "settings": {
            "turns": turns,
            "turn_target_tokens": turn_target_tokens,
            "system_tokens": system_tokens,
            "approx_total_prompt_tokens": system_tokens + turns * turn_target_tokens,
        },
        "system": scaffold,
        "turns": turn_specs,
    }


# ---------------------------------------------------------------------------
# Server lifecycle (mirrors evals/agentic-replay.py)
# ---------------------------------------------------------------------------


def server_command(binary: Path, model: str, extra_args: Sequence[str]) -> list[str]:
    command = [str(binary), "serve", "--model", model, "--log-format", "json"]
    command.extend(extra_args)
    for argument in command:
        for option in FORBIDDEN_STARTUP_OPTIONS:
            if argument == option or argument.startswith(f"{option}="):
                raise AssertionError(f"default-startup benchmark cannot use {option}")
    return command


def port_is_open(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.2)
        return connection.connect_ex((host, port)) == 0


def wait_for_model(timeout: float, process: subprocess.Popen[bytes]) -> str:
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Mesh exited before readiness with status {process.returncode}")
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
    raise TimeoutError(f"Mesh did not become ready after {timeout}s: {last_error}")


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
        raise RuntimeError("Mesh stopped but the serving port is still occupied")


def isolated_server_env(state_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    home = state_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "XDG_CACHE_HOME": str(state_dir / "xdg-cache"),
            "XDG_CONFIG_HOME": str(state_dir / "xdg-config"),
            "MESH_LLM_RUNTIME_ROOT": str(state_dir / "runtime"),
        }
    )
    if "HF_HOME" not in env:
        env["HF_HOME"] = str(Path.home() / ".cache/huggingface")
    return env


def start_server(
    binary: Path,
    model: str,
    extra_args: Sequence[str],
    state_dir: Path,
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], list[str]]:
    if port_is_open():
        raise RuntimeError(f"TCP {DEFAULT_PORT} is already in use; stop the existing Mesh instance")
    command = server_command(binary, model, extra_args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("wb")
    process = subprocess.Popen(
        command,
        cwd=str(REPO),
        env=isolated_server_env(state_dir),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    return process, command


# ---------------------------------------------------------------------------
# Requests (mirrors evals/agentic-replay.py stream_request)
# ---------------------------------------------------------------------------


def stream_request(
    request_id: str,
    messages: Sequence[dict[str, Any]],
    model_id: str,
    max_output_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    first_token_at: Optional[float] = None
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    saw_done = False
    connection = http.client.HTTPConnection(DEFAULT_HOST, DEFAULT_PORT, timeout=timeout)
    payload = {
        "model": model_id,
        "messages": list(messages),
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
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
            return {"request_id": request_id, "error": f"HTTP {response.status}: {body}"}
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
                return {
                    "request_id": request_id,
                    "error": f"stream failed with server error: {server_error}",
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
            delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
            if isinstance(delta, dict) and delta.get("content") and first_token_at is None:
                first_token_at = time.monotonic()
        if first_token_at is None:
            return {"request_id": request_id, "error": "stream completed without content tokens"}
        if not saw_done and completion_tokens == 0:
            return {"request_id": request_id, "error": "stream completed without completion-token usage"}
        ended = time.monotonic()
        return {
            "request_id": request_id,
            "ttft_seconds": first_token_at - started,
            "total_seconds": ended - started,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "decode_tokens_per_second": (
                completion_tokens / (ended - first_token_at) if ended > first_token_at else None
            ),
        }
    except (OSError, TimeoutError, http.client.HTTPException) as error:
        return {"request_id": request_id, "error": str(error)}
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def hardware_fingerprint() -> dict[str, Any]:
    fingerprint: dict[str, Any] = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
    }
    if sys.platform == "darwin":
        try:
            fingerprint["chip"] = (
                subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
            fingerprint["machine_model"] = (
                subprocess.run(
                    ["sysctl", "-n", "hw.model"], capture_output=True, text=True, check=True
                ).stdout.strip()
            )
        except subprocess.CalledProcessError:
            pass
        try:
            fingerprint["physical_memory_bytes"] = os.sysconf("HW_PHYSMEM")
        except (ValueError, OSError):
            fingerprint["physical_memory_bytes"] = None
        fingerprint["cpu_core_count"] = os.cpu_count()
    else:
        fingerprint["cpu_core_count"] = os.cpu_count()
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
            match = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
            if match:
                fingerprint["physical_memory_bytes"] = int(match.group(1)) * 1024
        except OSError:
            pass
    return fingerprint


def binary_provenance(binary: Path) -> dict[str, Any]:
    provenance: dict[str, Any] = {"binary": str(binary), "binary_sha256": sha256_file(binary)}
    try:
        described = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=True,
        )
        provenance["git_describe"] = described.stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, check=True
        )
        provenance["source_sha"] = commit.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        provenance["git_describe"] = "unknown"
    return provenance


# ---------------------------------------------------------------------------
# Cohorts and run
# ---------------------------------------------------------------------------


def summarize_cohort(name: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if "error" not in row]
    failed = [row for row in rows if "error" in row]
    ttft = [row["ttft_seconds"] for row in successful]

    def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        index = min(math.ceil(len(ordered) * fraction) - 1, len(ordered) - 1)
        return ordered[max(index, 0)]

    prompt_tokens = sum(row.get("prompt_tokens", 0) for row in successful)
    cached_tokens = sum(row.get("cached_tokens", 0) for row in successful)
    decode = [row["decode_tokens_per_second"] for row in successful if row.get("decode_tokens_per_second")]
    return {
        "cohort": name,
        "requests": len(rows),
        "failed": len(failed),
        "ttft_p50_seconds": percentile(ttft, 0.50),
        "ttft_p95_seconds": percentile(ttft, 0.95),
        "total_seconds_mean": statistics.fmean(row["total_seconds"] for row in successful) if successful else None,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_pct": (100 * cached_tokens / prompt_tokens) if prompt_tokens else None,
        "decode_tokens_per_second_mean": statistics.fmean(decode) if decode else None,
    }


def messages_through(messages: list[dict[str, Any]], turn_index: int) -> list[dict[str, Any]]:
    """Conversation prefix ending on the user message of ``turn_index`` (0-based).

    Fill requests must look like the requests a real session produces: the last
    message is always the user turn being answered, never the canned assistant
    reply that follows it in the canonical conversation.
    """
    return messages[: 2 * (turn_index + 1)]


def run_arm(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    binary = Path(args.binary).resolve()
    model_path = Path(args.model).resolve()
    if not binary.exists():
        raise FileNotFoundError(f"binary not found: {binary}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    manifest = build_manifest(args.turns, args.turn_target_tokens, args.system_tokens)
    manifest_sha = stable_hash(manifest)
    conversation: list[dict[str, Any]] = [{"role": "system", "content": manifest["system"]}]
    for spec in manifest["turns"]:
        conversation.append({"role": "user", "content": f"{spec['context']}\n\n{spec['request']}"})
        conversation.append({"role": "assistant", "content": spec["request"]})

    state_dir = (output / "server-state").resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output / "requests.jsonl"
    rows: list[dict[str, Any]] = []

    def record(cohort: str, index: int, result: dict[str, Any]) -> None:
        row = {"cohort": cohort, "request_index": index, **result}
        rows.append(row)
        with requests_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def replay_frozen(cohort: str, repeats: Optional[int] = None) -> None:
        connection = http.client.HTTPConnection(DEFAULT_HOST, DEFAULT_PORT, timeout=5)
        try:
            connection.request("GET", "/v1/models")
            document = json.loads(connection.getresponse().read())
            model_id = (document.get("data") or [{}])[0].get("id", "default")
        finally:
            connection.close()
        for repeat in range(args.restore_repeats if repeats is None else repeats):
            result = stream_request(
                f"{cohort}-{repeat + 1}",
                conversation,
                model_id,
                args.max_output_tokens,
                args.request_timeout,
            )
            record(cohort, repeat, result)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "kind": "kv-restart-replay/run",
        "started_at": utc_now(),
        "binary": binary_provenance(binary),
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "size_bytes": model_path.stat().st_size,
        },
        "hardware": hardware_fingerprint(),
        "config": {
            "base_url": DEFAULT_BASE_URL,
            "turns": args.turns,
            "turn_target_tokens": args.turn_target_tokens,
            "system_tokens": args.system_tokens,
            "restore_repeats": args.restore_repeats,
            "max_output_tokens": args.max_output_tokens,
            "request_timeout": args.request_timeout,
            "serve_extra_args": list(args.serve_extra_args),
        },
        "manifest_sha256": manifest_sha,
        "manifest": manifest,
    }

    process = None
    try:
        # Cohort: fill — cold server, conversation grows turn by turn.
        process, command = start_server(
            binary, str(model_path), args.serve_extra_args, state_dir, output / "logs" / "fill.log"
        )
        provenance["serve_command"] = command
        model_id = wait_for_model(args.ready_timeout, process)
        for index in range(args.turns):
            result = stream_request(
                f"fill-{index + 1}",
                messages_through(conversation, index),
                model_id,
                args.max_output_tokens,
                args.request_timeout,
            )
            record("fill", index, result)

        # Cohort: restore — full process restart on the same state directory.
        stop_server(process)
        process = None
        stopped_at = time.monotonic()
        process, _ = start_server(
            binary, str(model_path), args.serve_extra_args, state_dir, output / "logs" / "restore.log"
        )
        wait_for_model(args.ready_timeout, process)
        restart_gap_seconds = time.monotonic() - stopped_at
        provenance["restart"] = {
            "method": "SIGINT to the serving process group, then fresh start on the same state directory",
            "restart_to_ready_seconds": restart_gap_seconds,
        }
        # Cohort: restore — the FIRST post-restart replay alone is the
        # first-request-after-restart measurement; later replays warm from the
        # resident cache and are recorded under the warm cohort instead.
        replay_frozen("restore", repeats=1)
        replay_frozen("warm", repeats=max(args.restore_repeats - 1, 0))

        # Cohort: warm — repeat replays without restart.
        replay_frozen("warm")
    finally:
        if process is not None:
            try:
                stop_server(process)
            except RuntimeError:
                pass

    provenance["cohorts"] = [
        summarize_cohort("fill", [row for row in rows if row["cohort"] == "fill"]),
        summarize_cohort("restore", [row for row in rows if row["cohort"] == "restore"]),
        summarize_cohort("warm", [row for row in rows if row["cohort"] == "warm"]),
    ]
    provenance["completed_at"] = utc_now()
    return provenance


def write_report(run: dict[str, Any], path: Path) -> None:
    lines = [
        "# KV restart replay",
        "",
        f"- source: `{run['binary'].get('source_sha', 'unknown')}` (`{run['binary'].get('git_describe', '?')}`)",
        f"- model sha256: `{run['model']['sha256'][:16]}…`",
        f"- manifest sha256: `{run['manifest_sha256'][:16]}…` ({run['config']['turns']} turns, "
        f"≈{run['manifest']['settings']['approx_total_prompt_tokens']} prompt tokens)",
        f"- serve extra args: `{run['config']['serve_extra_args'] or 'none'}`",
        f"- restart-to-ready: {run.get('restart', {}).get('restart_to_ready_seconds', float('nan')):.1f}s",
        "",
        "| cohort | requests | failed | TTFT p50 (s) | TTFT p95 (s) | cached % | decode tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort in run["cohorts"]:
        fmt = lambda value: "—" if value is None else f"{value:.3f}" if isinstance(value, float) else value
        lines.append(
            f"| {cohort['cohort']} | {cohort['requests']} | {cohort['failed']} "
            f"| {fmt(cohort['ttft_p50_seconds'])} | {fmt(cohort['ttft_p95_seconds'])} "
            f"| {fmt(cohort['cache_pct'])} | {fmt(cohort['decode_tokens_per_second_mean'])} |"
        )
    lines += [
        "",
        "`restore` is the first-request-after-restart cohort; on a build without a",
        "durable KV tier it is the cold-prefill reference. `warm` repeats the same",
        "replay without restart (resident reuse reference).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(REPO / "target/release/mesh-llm"))
    parser.add_argument("--model", required=True, help="path to a GGUF model file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--turn-target-tokens", type=int, default=4750)
    parser.add_argument("--system-tokens", type=int, default=500)
    parser.add_argument("--restore-repeats", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument(
        "--serve-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="explicit extra serving arguments (recorded verbatim in run.json)",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if (output / "run.json").exists():
        raise SystemExit(f"output already contains run.json: {output}")
    output.mkdir(parents=True, exist_ok=True)

    run = run_arm(args, output)
    write_json = output / "run.json"
    write_json.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(run, output / "report.md")
    print(f"wrote {write_json}")
    for cohort in run["cohorts"]:
        p50 = cohort["ttft_p50_seconds"]
        print(
            f"  {cohort['cohort']:8s} p50={p50 if p50 is None else round(p50, 3)}s "
            f"cache={cohort['cache_pct'] if cohort['cache_pct'] is None else round(cohort['cache_pct'], 1)}% "
            f"failed={cohort['failed']}"
        )
    return 0 if all(cohort["failed"] == 0 for cohort in run["cohorts"]) else 1


if __name__ == "__main__":
    sys.exit(main())
