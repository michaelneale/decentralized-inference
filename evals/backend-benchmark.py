#!/usr/bin/env python3
"""
Small local backend benchmark for mesh-llm.

Runs the same prompt through:
  - llama backend via a GGUF model path
  - vllm backend via a HuggingFace-style model directory

The script starts mesh-llm separately for each backend, waits for /v1/models,
warms the backend, then runs repeated streaming chat completions to measure:
  - startup time
  - time to first token (TTFT)
  - end-to-end request time
  - completion throughput when usage is reported
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import concurrent.futures
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Write a short explanation of Rust ownership, then give a minimal example "
    "that borrows a string slice safely."
)


@dataclass
class BackendSpec:
    name: str
    model: Path
    api_port: int
    console_port: int
    extra_args: list[str]
    env: dict[str, str]


@dataclass
class RunMetrics:
    ttft_s: float
    total_s: float
    completion_tokens: int | None
    prompt_tokens: int | None
    chars: int


@dataclass
class BatchMetrics:
    concurrency: int
    batch_wall_s: float
    requests: list[RunMetrics]


COMMON_VARIANT_SUFFIXES = (
    "q2_k",
    "q3_k_s",
    "q3_k_m",
    "q3_k_l",
    "q4_0",
    "q4_1",
    "q4_k_s",
    "q4_k_m",
    "q5_0",
    "q5_1",
    "q5_k_s",
    "q5_k_m",
    "q6_k",
    "q8_0",
    "4bit",
    "8bit",
    "f16",
    "bf16",
    "fp16",
    "fp32",
    "int4",
    "int8",
    "gptq",
    "awq",
)

MODEL_PRESETS = {
    "qwen2.5-0.5b-instruct": {
        "comparison_key": "qwen2.5-0.5b-instruct",
        "description": (
            "Recommended apples-to-apples pair: official HF "
            "Qwen2.5-0.5B-Instruct for vllm and an F16 GGUF converted from "
            "that same snapshot for llama."
        ),
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare mesh-llm llama and vllm backends locally."
    )
    parser.add_argument(
        "--mesh-bin",
        default="target/release/mesh-llm",
        help="Path to the mesh-llm binary",
    )
    parser.add_argument(
        "--bin-dir",
        default="llama.cpp/build/bin",
        help="Path to llama.cpp runtime binaries",
    )
    parser.add_argument(
        "--llama-model",
        required=True,
        help="GGUF model path for the llama backend",
    )
    parser.add_argument(
        "--vllm-model",
        required=True,
        help="HF-style model directory for the vllm backend",
    )
    parser.add_argument(
        "--vllm-bin",
        help="Optional vllm executable path",
    )
    parser.add_argument(
        "--vllm-metal-memory-fraction",
        type=float,
        help="Set VLLM_METAL_MEMORY_FRACTION for vllm-metal runs",
    )
    parser.add_argument(
        "--vllm-mlx-device",
        help="Set VLLM_MLX_DEVICE for vllm-metal runs, for example gpu",
    )
    parser.add_argument(
        "--vllm-metal-use-mlx",
        choices=["0", "1"],
        help="Set VLLM_METAL_USE_MLX for vllm-metal runs",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to benchmark",
    )
    parser.add_argument(
        "--prompt-file",
        help="Read the prompt from a file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Timed iterations per backend and per concurrency level",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup requests per backend",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="max_tokens for each request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=240.0,
        help="Seconds to wait for /v1/models",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=240.0,
        help="Seconds to wait for one request",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=19337,
        help="Base API port; the script uses base/base+1 and base+10/base+11",
    )
    parser.add_argument(
        "--concurrency",
        default="1",
        help="Comma-separated concurrency levels to benchmark, for example 1,4,8",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a text table",
    )
    parser.add_argument(
        "--allow-mismatched-models",
        action="store_true",
        help="Allow benchmarking model pairs that do not normalize to the same base identity",
    )
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Use a named apples-to-apples comparison key for a recommended model pair",
    )
    return parser.parse_args()


def http_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def parse_concurrency_levels(raw: str) -> list[int]:
    levels: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("concurrency levels must be >= 1")
        levels.append(value)
    if not levels:
        raise ValueError("at least one concurrency level is required")
    return list(dict.fromkeys(levels))


def sidecar_comparison_key(model_path: Path) -> str | None:
    sidecar = model_path.with_name(f"{model_path.name}.mesh.json")
    if not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except Exception:
        return None

    source = payload.get("source") or {}
    repo = source.get("repo")
    revision = source.get("revision")
    if repo and revision:
        return f"{repo}@{revision}"
    if repo:
        return str(repo)

    identity = payload.get("identity") or {}
    canonical_id = identity.get("canonical_id")
    if canonical_id:
        return str(canonical_id)
    return None


def normalize_model_identity(name: str) -> str:
    value = Path(name).name.lower()
    if value.endswith(".gguf"):
        value = value[: -len(".gguf")]
    value = re.sub(r"[-_](split|part)-\d+of\d+$", "", value)
    return value


def post_stream(url: str, payload: dict[str, Any], timeout: float) -> RunMetrics:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    first_content_at: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    chars = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)

            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    chars += len(content)
                    if first_content_at is None:
                        first_content_at = time.monotonic()

    end = time.monotonic()
    if first_content_at is None:
        raise RuntimeError("No streamed content was returned")

    return RunMetrics(
        ttft_s=first_content_at - start,
        total_s=end - start,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        chars=chars,
    )


def wait_for_models(api_port: int, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    url = f"http://127.0.0.1:{api_port}/v1/models"
    while time.monotonic() < deadline:
        try:
            payload = http_json(url, timeout=5)
            data = payload.get("data") or []
            if data:
                return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1.0)
    if last_error:
        raise RuntimeError(f"Timed out waiting for {url}: {last_error}") from last_error
    raise RuntimeError(f"Timed out waiting for {url}")


def tail_text(path: Path, limit: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def stop_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=5)


def start_backend(
    args: argparse.Namespace, spec: BackendSpec
) -> tuple[subprocess.Popen[Any], Path, float, str]:
    log_path = Path("/tmp") / f"mesh-llm-bench-{spec.name}.log"
    log_file = log_path.open("w")
    cmd = [
        str(Path(args.mesh_bin)),
        "--no-self-update",
        "--model",
        str(spec.model),
        "--port",
        str(spec.api_port),
        "--console",
        str(spec.console_port),
        "--bin-dir",
        str(Path(args.bin_dir)),
    ]
    cmd.extend(spec.extra_args)

    env = os.environ.copy()
    env.update(spec.env)
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    try:
        payload = wait_for_models(spec.api_port, args.startup_timeout)
    except Exception:  # noqa: BLE001
        stop_process(proc)
        raise RuntimeError(
            f"{spec.name} failed to start.\n\nLog tail:\n{tail_text(log_path)}"
        )

    data = payload.get("data") or []
    model_id = data[0]["id"]
    startup_s = time.monotonic() - start
    return proc, log_path, startup_s, model_id


def warm_backend(
    spec: BackendSpec,
    model_id: str,
    prompt: str,
    warmup: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> None:
    if warmup <= 0:
        return
    url = f"http://127.0.0.1:{spec.api_port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": min(max_tokens, 32),
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for _ in range(warmup):
        post_stream(url, payload, timeout)


def run_batch(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    concurrency: int,
) -> BatchMetrics:
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(post_stream, url, payload, timeout)
            for _ in range(concurrency)
        ]
        requests = [future.result() for future in futures]
    ended = time.monotonic()
    return BatchMetrics(
        concurrency=concurrency,
        batch_wall_s=ended - started,
        requests=requests,
    )


def benchmark_backend(
    args: argparse.Namespace, spec: BackendSpec, prompt: str
) -> dict[str, Any]:
    proc, log_path, startup_s, model_id = start_backend(args, spec)
    try:
        warm_backend(
            spec,
            model_id,
            prompt,
            args.warmup,
            args.max_tokens,
            args.temperature,
            args.request_timeout,
        )

        url = f"http://127.0.0.1:{spec.api_port}/v1/chat/completions"
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        def maybe_mean(values: list[float]) -> float | None:
            return statistics.mean(values) if values else None
        concurrency_levels = parse_concurrency_levels(args.concurrency)
        results_by_concurrency: list[dict[str, Any]] = []
        for concurrency in concurrency_levels:
            batches: list[BatchMetrics] = []
            for _ in range(args.iterations):
                batches.append(run_batch(url, payload, args.request_timeout, concurrency))

            request_runs = [run for batch in batches for run in batch.requests]
            completion_token_values = [
                run.completion_tokens
                for run in request_runs
                if run.completion_tokens is not None
            ]
            overall_tok_s_values = [
                run.completion_tokens / run.total_s
                for run in request_runs
                if run.completion_tokens is not None and run.total_s > 0
            ]
            decode_tok_s_values = [
                run.completion_tokens / max(run.total_s - run.ttft_s, 1e-9)
                for run in request_runs
                if run.completion_tokens is not None and run.total_s > run.ttft_s
            ]
            batch_completion_tokens = [
                sum(
                    run.completion_tokens or 0
                    for run in batch.requests
                )
                for batch in batches
            ]
            batch_overall_tok_s_values = [
                total_tokens / batch.batch_wall_s
                for batch, total_tokens in zip(batches, batch_completion_tokens, strict=True)
                if batch.batch_wall_s > 0
            ]

            results_by_concurrency.append(
                {
                    "concurrency": concurrency,
                    "avg_ttft_s": maybe_mean([run.ttft_s for run in request_runs]),
                    "avg_total_s": maybe_mean([run.total_s for run in request_runs]),
                    "avg_batch_wall_s": maybe_mean([batch.batch_wall_s for batch in batches]),
                    "avg_completion_tokens": maybe_mean(
                        [float(value) for value in completion_token_values]
                    ),
                    "avg_overall_tok_s": maybe_mean(overall_tok_s_values),
                    "avg_decode_tok_s": maybe_mean(decode_tok_s_values),
                    "avg_batch_overall_tok_s": maybe_mean(batch_overall_tok_s_values),
                    "avg_chars": maybe_mean([float(run.chars) for run in request_runs]),
                    "batches": [
                        {
                            "concurrency": batch.concurrency,
                            "batch_wall_s": batch.batch_wall_s,
                            "requests": [
                                {
                                    "ttft_s": run.ttft_s,
                                    "total_s": run.total_s,
                                    "completion_tokens": run.completion_tokens,
                                    "prompt_tokens": run.prompt_tokens,
                                    "chars": run.chars,
                                }
                                for run in batch.requests
                            ],
                        }
                        for batch in batches
                    ],
                }
            )

        return {
            "backend": spec.name,
            "model_id": model_id,
            "normalized_model_id": sidecar_comparison_key(spec.model)
            or normalize_model_identity(model_id),
            "model_path": str(spec.model),
            "startup_s": startup_s,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "concurrency_levels": concurrency_levels,
            "results_by_concurrency": results_by_concurrency,
            "log_path": str(log_path),
        }
    finally:
        stop_process(proc)


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def print_text_summary(results: list[dict[str, Any]]) -> None:
    preset = results[0].get("model_preset")
    if preset:
        print(f"model_preset  {preset}")
        print("")
    pair_key = results[0].get("comparison_key")
    if pair_key:
        print(f"paired_model  {pair_key}")
        print("")
    for result in results:
        print(f"backend  {result['backend']}  startup_s  {format_metric(result['startup_s'])}  model_id  {result['model_id']}")
        print(
            "concurrency  avg_ttft_s  avg_total_s  avg_batch_wall_s  avg_decode_tok_s  avg_overall_tok_s  avg_batch_overall_tok_s"
        )
        for level in result["results_by_concurrency"]:
            print(
                f"{level['concurrency']:>11}  "
                f"{format_metric(level['avg_ttft_s']):>10}  "
                f"{format_metric(level['avg_total_s']):>11}  "
                f"{format_metric(level['avg_batch_wall_s']):>16}  "
                f"{format_metric(level['avg_decode_tok_s']):>16}  "
                f"{format_metric(level['avg_overall_tok_s']):>17}  "
                f"{format_metric(level['avg_batch_overall_tok_s']):>23}"
            )
        print("")
    for result in results:
        print(f"{result['backend']} log: {result['log_path']}")


def main() -> int:
    args = parse_args()
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    expected_comparison_key = None
    if args.model_preset:
        expected_comparison_key = MODEL_PRESETS[args.model_preset]["comparison_key"]

    llama_spec = BackendSpec(
        name="llama",
        model=Path(args.llama_model),
        api_port=args.base_port,
        console_port=args.base_port + 10,
        extra_args=[],
        env={},
    )
    vllm_extra_args: list[str] = []
    vllm_env: dict[str, str] = {}
    if args.vllm_bin:
        vllm_extra_args.extend(["--vllm-bin", args.vllm_bin])
    if args.vllm_metal_memory_fraction is not None:
        vllm_env["VLLM_METAL_MEMORY_FRACTION"] = str(args.vllm_metal_memory_fraction)
    if args.vllm_mlx_device:
        vllm_env["VLLM_MLX_DEVICE"] = args.vllm_mlx_device
    if args.vllm_metal_use_mlx:
        vllm_env["VLLM_METAL_USE_MLX"] = args.vllm_metal_use_mlx
    vllm_spec = BackendSpec(
        name="vllm",
        model=Path(args.vllm_model),
        api_port=args.base_port + 1,
        console_port=args.base_port + 11,
        extra_args=vllm_extra_args,
        env=vllm_env,
    )

    try:
        results = [
            benchmark_backend(args, llama_spec, prompt),
            benchmark_backend(args, vllm_spec, prompt),
        ]
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1

    comparison_keys = {result["normalized_model_id"] for result in results}
    if expected_comparison_key is not None:
        bad_results = [
            result
            for result in results
            if result["normalized_model_id"] != expected_comparison_key
        ]
        if bad_results and not args.allow_mismatched_models:
            print(
                "benchmark failed: model pair does not match the selected preset. "
                "Use equivalent exports or pass --allow-mismatched-models.\n"
                + "\n".join(
                    f"{result['backend']}={result['model_id']} -> {result['normalized_model_id']}"
                    for result in results
                ),
                file=sys.stderr,
            )
            return 1
        comparison_key = expected_comparison_key
    elif len(comparison_keys) != 1 and not args.allow_mismatched_models:
        print(
            "benchmark failed: model pair does not normalize to the same identity. "
            "Use equivalent exports or pass --allow-mismatched-models.\n"
            f"llama={results[0]['model_id']} -> {results[0]['normalized_model_id']}\n"
            f"vllm={results[1]['model_id']} -> {results[1]['normalized_model_id']}",
            file=sys.stderr,
        )
        return 1
    else:
        comparison_key = next(iter(comparison_keys)) if len(comparison_keys) == 1 else None
    for result in results:
        result["comparison_key"] = comparison_key
        result["model_preset"] = args.model_preset

    if args.json:
        payload: dict[str, Any] = {"prompt": prompt, "results": results}
        if args.model_preset:
            payload["preset"] = {
                "name": args.model_preset,
                **MODEL_PRESETS[args.model_preset],
            }
        print(json.dumps(payload, indent=2))
    else:
        print_text_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
