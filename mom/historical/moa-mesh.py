#!/usr/bin/env python3
"""
Mixture-of-Agents over mesh-llm.

Fan out to all mesh models in parallel, then aggregate with the strongest.
Uses the local mesh client's OpenAI-compatible API on localhost:9337.

Usage:
    python3 moa-mesh.py "What are 3 fun things to do in SF?"
    python3 moa-mesh.py --solo "Explain quantum computing"
    python3 moa-mesh.py --layers 2 "Review this code for bugs"
"""

from __future__ import annotations
import asyncio
import httpx
import json
import sys
import time
import argparse
from typing import Optional, List, Dict, Tuple

MESH_URL = "http://localhost:9337/v1"

# Disable thinking mode for all models (avoids wasting tokens on reasoning_content)
NO_THINK_PREFIX = "/no_think\n\n"


async def get_models() -> List[str]:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{MESH_URL}/models")
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"]]


async def generate(client: httpx.AsyncClient, model: str, messages: List[dict],
                   temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """Streaming completion from a mesh model. Returns full text."""
    chunks = []
    async with client.stream(
        "POST",
        f"{MESH_URL}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        timeout=300,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content") or ""
                    if content:
                        chunks.append(content)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
    return "".join(chunks)


async def generate_print(client: httpx.AsyncClient, model: str, messages: List[dict],
                         temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """Streaming completion that prints chunks as they arrive."""
    chunks = []
    async with client.stream(
        "POST",
        f"{MESH_URL}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        timeout=300,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content") or ""
                    if content:
                        print(content, end="", flush=True)
                        chunks.append(content)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
    return "".join(chunks)


def prepend_no_think(messages: List[dict]) -> List[dict]:
    """Prepend /no_think to the first user message to disable thinking mode."""
    result = []
    patched = False
    for m in messages:
        if m["role"] == "user" and not patched:
            result.append({"role": "user", "content": NO_THINK_PREFIX + m["content"]})
            patched = True
        else:
            result.append(m)
    return result


AGGREGATOR_PROMPT = """You have been provided with a set of responses from various models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. Critically evaluate the information — some may be biased or incorrect. Offer a refined, accurate, and comprehensive reply. Do not simply replicate; improve.

Responses from models:"""


async def moa(user_prompt: str, models: List[str], aggregator: str,
              layers: int = 1, temperature: float = 0.7,
              system_prompt: Optional[str] = None) -> dict:
    """
    Run MoA: fan out to all models in parallel, then aggregate.
    Returns dict with timing, individual responses, and final answer.
    """
    result = {"models": models, "aggregator": aggregator, "layers": layers, "rounds": []}

    async with httpx.AsyncClient() as client:
        prev_responses = None  # type: Optional[List[Tuple[str, str]]]

        for layer in range(layers):
            t0 = time.time()

            if prev_responses:
                # Layer 2+: each model sees previous layer's responses
                sys_content = AGGREGATOR_PROMPT + "\n" + "\n".join(
                    f"{i+1}. [{m}]: {r}" for i, (m, r) in enumerate(prev_responses)
                )
                messages_per_model = {
                    m: prepend_no_think([
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": user_prompt}
                    ]) for m in models
                }
            else:
                # Layer 1: raw prompt
                base = []
                if system_prompt:
                    base.append({"role": "system", "content": system_prompt})
                base.append({"role": "user", "content": user_prompt})
                base = prepend_no_think(base)
                messages_per_model = {m: base for m in models}

            # Fan out to all models in parallel
            tasks = {m: asyncio.create_task(generate(client, m, msgs, temperature))
                     for m, msgs in messages_per_model.items()}

            responses = {}
            for model_name, task in tasks.items():
                try:
                    responses[model_name] = await task
                except Exception as e:
                    responses[model_name] = f"[ERROR: {e}]"

            elapsed = time.time() - t0
            round_info = {
                "layer": layer + 1,
                "elapsed_s": round(elapsed, 1),
                "responses": {}
            }
            for m, r in responses.items():
                preview = r[:300] + "..." if len(r) > 300 else r
                round_info["responses"][m] = preview
                print(f"  [{m}]: {len(r)} chars", file=sys.stderr)

            result["rounds"].append(round_info)
            prev_responses = list(responses.items())
            print(f"  Layer {layer+1} done: {elapsed:.1f}s\n", file=sys.stderr)

        # Final aggregation with the strongest model (streaming, printed)
        t0 = time.time()
        sys_content = AGGREGATOR_PROMPT + "\n" + "\n".join(
            f"{i+1}. [{m}]: {r}" for i, (m, r) in enumerate(prev_responses)
        )

        print(f"  Aggregating with {aggregator}...\n", file=sys.stderr)
        final_text = await generate_print(client, aggregator, prepend_no_think([
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_prompt},
        ]), temperature=0.3)

        agg_elapsed = time.time() - t0
        result["aggregation_s"] = round(agg_elapsed, 1)
        result["final"] = final_text
        print(file=sys.stderr)

    return result


async def solo_run(user_prompt: str, model: str, system_prompt: Optional[str] = None) -> dict:
    """Run single model for comparison."""
    async with httpx.AsyncClient() as client:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        messages = prepend_no_think(messages)

        t0 = time.time()
        print(f"\n  Solo: {model}...\n", file=sys.stderr)
        final_text = await generate_print(client, model, messages, temperature=0.7)
        elapsed = time.time() - t0
        print(file=sys.stderr)
        return {"model": model, "elapsed_s": round(elapsed, 1), "response": final_text}


async def main():
    parser = argparse.ArgumentParser(description="MoA over mesh-llm")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--layers", type=int, default=1, help="MoA layers (default: 1)")
    parser.add_argument("--solo", action="store_true", help="Also run solo with aggregator for comparison")
    parser.add_argument("--aggregator", type=str, default=None, help="Aggregator model (default: largest)")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Reference models (default: all)")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    parser.add_argument("--json", action="store_true", help="Output JSON results to stdout")
    args = parser.parse_args()

    models = await get_models()
    print(f"Mesh models: {models}", file=sys.stderr)

    if args.models:
        models = [m for m in models if any(a.lower() in m.lower() for a in args.models)]

    # Pick aggregator — prefer the strongest
    aggregator = args.aggregator
    if not aggregator:
        for candidate in ["MiniMax", "Qwen2.5-72B"]:
            for m in models:
                if candidate in m:
                    aggregator = m
                    break
            if aggregator:
                break
        if not aggregator:
            aggregator = models[0]

    print(f"Aggregator: {aggregator}", file=sys.stderr)
    print(f"Reference models: {models}", file=sys.stderr)
    print(f"Layers: {args.layers}\n", file=sys.stderr)

    # ── MoA ──
    print("═══ MoA ═══", file=sys.stderr)
    t0 = time.time()
    moa_result = await moa(args.prompt, models, aggregator,
                           layers=args.layers, system_prompt=args.system)
    moa_total = time.time() - t0
    moa_result["total_s"] = round(moa_total, 1)

    # ── Solo comparison ──
    solo_result = None
    if args.solo:
        print("\n═══ Solo ═══", file=sys.stderr)
        solo_result = await solo_run(args.prompt, aggregator, system_prompt=args.system)

    # ── Summary ──
    print(f"\n{'═'*60}", file=sys.stderr)
    print(f"MoA ({args.layers} layer, {len(models)} models): {moa_total:.1f}s total", file=sys.stderr)
    if solo_result:
        print(f"Solo ({aggregator}): {solo_result['elapsed_s']}s", file=sys.stderr)

    if args.json:
        out = {"moa": moa_result}
        if solo_result:
            out["solo"] = solo_result
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
