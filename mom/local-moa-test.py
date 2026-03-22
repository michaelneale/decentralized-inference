#!/usr/bin/env python3
"""
Local MoA algorithm comparison — 3 models on localhost.
Fast iteration on strategies before deploying to mesh.
"""
from __future__ import annotations
import asyncio, httpx, json, sys, time
from typing import List, Dict, Tuple, Optional

MODELS = [
    ("Qwen3-30B-A3B",    "http://localhost:8081/v1"),
    ("Qwen2.5-Coder-7B", "http://localhost:8082/v1"),
    ("Mistral-Small-24B", "http://localhost:8083/v1"),
]

# ── Helpers ──

async def generate(client: httpx.AsyncClient, name: str, url: str,
                   messages: list, max_tokens=2000, temperature=0.7) -> Tuple[str, str, float]:
    """Returns (model_name, response_text, elapsed_s)."""
    t0 = time.time()
    chunks = []
    async with client.stream("POST", f"{url}/chat/completions", json={
        "model": name, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature, "stream": True,
    }, timeout=120) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]": break
                try:
                    d = json.loads(data)["choices"][0]["delta"]
                    c = d.get("content") or d.get("reasoning_content") or ""
                    if c: chunks.append(c)
                except: pass
    return name, "".join(chunks), time.time() - t0


async def fan_out(client: httpx.AsyncClient, messages: list,
                  models=None, max_tokens=2000, temperature=0.7) -> List[Tuple[str, str, float]]:
    """Fan out to all models in parallel."""
    models = models or MODELS
    tasks = [generate(client, n, u, messages, max_tokens, temperature) for n, u in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    good = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}", file=sys.stderr)
        else:
            name, text, elapsed = r
            print(f"  {name}: {len(text)} chars in {elapsed:.1f}s", file=sys.stderr)
            good.append(r)
    return good


def inject_refs(messages: list, refs: List[Tuple[str, str, float]], system: str) -> list:
    """Inject model responses as system context."""
    ref_text = system
    for i, (name, text, _) in enumerate(refs):
        ref_text += f"\n\n{i+1}. [{name}]:\n{text}"
    result = list(messages)
    if result and result[0].get("role") == "system":
        result[0] = {**result[0], "content": result[0]["content"] + "\n\n" + ref_text}
    else:
        result.insert(0, {"role": "system", "content": ref_text})
    return result


SYNTH_SYS = "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n\nResponses from models:"

PICK_SYS = "You have been provided with a set of responses from various models. Your task is to select the BEST response — the one that most accurately, completely, and helpfully answers the query. Output ONLY the selected response verbatim with no modifications or preamble. Pick the single best one.\n\nResponses:"

RANK_SYS = "You have been provided with responses from various models. Rank them from best to worst. For each, give a 1-sentence reason. Then output ONLY the text of the #1 ranked response verbatim after a line reading '---BEST---'.\n\nResponses:"

# ── Strategies ──

async def solo(client, prompt, model_idx=0, **kw):
    """Single model — baseline."""
    name, url = MODELS[model_idx]
    msgs = [{"role": "user", "content": prompt}]
    _, text, elapsed = await generate(client, name, url, msgs, **kw)
    return {"strategy": f"solo-{name}", "answer": text, "elapsed": elapsed}


async def moa_synthesize(client, prompt, **kw):
    """MoA: fan-out → aggregator synthesizes."""
    msgs = [{"role": "user", "content": prompt}]
    t0 = time.time()
    refs = await fan_out(client, msgs, **kw)
    if not refs: return {"strategy": "moa", "answer": "(no responses)", "elapsed": 0}
    
    # Aggregator = first model (strongest)
    agg_name, agg_url = MODELS[0]
    agg_msgs = inject_refs(msgs, refs, SYNTH_SYS)
    _, answer, _ = await generate(client, agg_name, agg_url, agg_msgs, temperature=0.3)
    return {"strategy": "moa", "answer": answer, "elapsed": time.time() - t0}


async def best_of_n(client, prompt, **kw):
    """Best-of-N: fan-out → aggregator picks best verbatim."""
    msgs = [{"role": "user", "content": prompt}]
    t0 = time.time()
    refs = await fan_out(client, msgs, **kw)
    if not refs: return {"strategy": "best-of-n", "answer": "(no responses)", "elapsed": 0}
    
    agg_name, agg_url = MODELS[0]
    agg_msgs = inject_refs(msgs, refs, PICK_SYS)
    _, answer, _ = await generate(client, agg_name, agg_url, agg_msgs, temperature=0.1)
    return {"strategy": "best-of-n", "answer": answer, "elapsed": time.time() - t0}


async def rank_and_pick(client, prompt, **kw):
    """Rank-and-pick: fan-out → aggregator ranks all, returns #1."""
    msgs = [{"role": "user", "content": prompt}]
    t0 = time.time()
    refs = await fan_out(client, msgs, **kw)
    if not refs: return {"strategy": "rank-pick", "answer": "(no responses)", "elapsed": 0}
    
    agg_name, agg_url = MODELS[0]
    agg_msgs = inject_refs(msgs, refs, RANK_SYS)
    _, answer, _ = await generate(client, agg_name, agg_url, agg_msgs, temperature=0.1)
    # Try to extract after ---BEST---
    if "---BEST---" in answer:
        answer = answer.split("---BEST---", 1)[1].strip()
    return {"strategy": "rank-pick", "answer": answer, "elapsed": time.time() - t0}


async def moa_2layer(client, prompt, **kw):
    """Two-layer MoA: fan-out → all refine → aggregator synthesizes."""
    msgs = [{"role": "user", "content": prompt}]
    t0 = time.time()
    
    # Layer 1
    print("  Layer 1:", file=sys.stderr)
    refs1 = await fan_out(client, msgs, **kw)
    if not refs1: return {"strategy": "moa-2", "answer": "(no responses)", "elapsed": 0}
    
    # Layer 2: each model sees Layer 1 refs
    print("  Layer 2:", file=sys.stderr)
    refined_msgs = inject_refs(msgs, refs1, SYNTH_SYS)
    refs2 = await fan_out(client, refined_msgs, **kw)
    final_refs = refs2 if refs2 else refs1
    
    # Aggregate
    agg_name, agg_url = MODELS[0]
    agg_msgs = inject_refs(msgs, final_refs, SYNTH_SYS)
    _, answer, _ = await generate(client, agg_name, agg_url, agg_msgs, temperature=0.3)
    return {"strategy": "moa-2", "answer": answer, "elapsed": time.time() - t0}


async def debate(client, prompt, rounds=2, **kw):
    """Debate: models critique each other's responses iteratively."""
    msgs = [{"role": "user", "content": prompt}]
    t0 = time.time()
    
    refs = await fan_out(client, msgs, **kw)
    if not refs: return {"strategy": "debate", "answer": "(no responses)", "elapsed": 0}
    
    for r in range(rounds - 1):
        print(f"  Debate round {r+2}:", file=sys.stderr)
        critique_sys = "You have seen other models' responses to this query. Provide your own improved response, considering the strengths and weaknesses of the other responses. Be concise and accurate.\n\nOther responses:"
        debate_msgs = inject_refs(msgs, refs, critique_sys)
        refs = await fan_out(client, debate_msgs, **kw)
        if not refs: break
    
    # Final aggregation
    agg_name, agg_url = MODELS[0]
    agg_msgs = inject_refs(msgs, refs, SYNTH_SYS)
    _, answer, _ = await generate(client, agg_name, agg_url, agg_msgs, temperature=0.3)
    return {"strategy": f"debate-{rounds}r", "answer": answer, "elapsed": time.time() - t0}


# ── Main ──

STRATEGIES = {
    "solo": lambda c, p, **k: solo(c, p, model_idx=0, **k),
    "solo-coder": lambda c, p, **k: solo(c, p, model_idx=1, **k),
    "solo-mistral": lambda c, p, **k: solo(c, p, model_idx=2, **k),
    "moa": moa_synthesize,
    "best-of-n": best_of_n,
    "rank-pick": rank_and_pick,
    "moa-2": moa_2layer,
    "debate": debate,
}


async def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write a Python function that finds the longest palindromic substring in a string. Include docstring and handle edge cases."
    
    strategies = sys.argv[2:] if len(sys.argv) > 2 else ["solo", "moa", "best-of-n"]
    
    print(f"Prompt: {prompt[:80]}...\n", file=sys.stderr)
    
    async with httpx.AsyncClient() as client:
        results = []
        for name in strategies:
            fn = STRATEGIES.get(name)
            if not fn:
                print(f"Unknown strategy: {name}", file=sys.stderr)
                continue
            print(f"{'='*60}", file=sys.stderr)
            print(f"Strategy: {name}", file=sys.stderr)
            result = await fn(client, prompt)
            results.append(result)
            print(f"  → {result['elapsed']:.1f}s, {len(result['answer'])} chars", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
        
        # Print results
        print("\n" + "="*60)
        for r in results:
            print(f"\n### {r['strategy']} ({r['elapsed']:.1f}s)")
            print(r['answer'][:500])
            if len(r['answer']) > 500: print("...")
            print()
        
        # Summary table
        print("="*60)
        print(f"{'Strategy':<20} {'Time':>8} {'Chars':>8}")
        print("-"*40)
        for r in results:
            print(f"{r['strategy']:<20} {r['elapsed']:>7.1f}s {len(r['answer']):>7}")


if __name__ == "__main__":
    asyncio.run(main())
