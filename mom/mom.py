#!/usr/bin/env python3
"""
Mixture of Models (MoM) — NSED protocol experiment.

Runs on Studio (206GB Mac). Uses:
  - MiniMax-M2.5 on :64144 (already running via mesh-llm)
  - Qwen3-30B-A3B on :8081 (started by this script)
  - Qwen3.5-27B on :8082 (started by this script)

Usage:
    python3 mom.py                    # run all test cases
    python3 mom.py "your question"    # single query
"""
from __future__ import annotations

import asyncio, json, math, re, signal, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

# ── Config ──

LLAMA_SERVER = Path.home() / "mesh-bundle" / "llama-server"
MODELS_DIR = Path.home() / ".models"
MAX_ROUNDS = 3
TIMEOUT = 180.0


@dataclass
class Agent:
    name: str
    model: str
    base_url: str
    role: str = ""
    tokens: int = 0
    time_s: float = 0.0
    wins: int = 0


@dataclass
class Round:
    num: int
    proposals: dict
    votes: dict
    scores: dict
    winner: str
    answer: str
    elapsed: float


# ── Generation (streaming to avoid timeouts) ──

async def gen(agent: Agent, messages: list, client: httpx.AsyncClient,
              max_tokens: int = 4096, temperature: float = 0.7,
              tools: list = None) -> dict:
    body = dict(model=agent.model, messages=messages,
                max_tokens=max_tokens, temperature=temperature, stream=True,
                chat_template_kwargs={"enable_thinking": False})
    if tools:
        body["tools"] = tools
    url = f"{agent.base_url}/chat/completions"
    t0 = time.monotonic()

    content, reasoning = "", ""
    tool_calls_acc = {}
    model_name, usage, finish_reason = agent.model, {}, None

    async with client.stream("POST", url, json=body, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            model_name = chunk.get("model", model_name)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content += delta["content"]
                if delta.get("reasoning_content"):
                    reasoning += delta["reasoning_content"]
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc["index"]
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": tc.get("id", ""), "function": {"name": "", "arguments": ""}}
                        if tc.get("id"):
                            tool_calls_acc[idx]["id"] = tc["id"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_calls_acc[idx]["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    agent.time_s += time.monotonic() - t0
    agent.tokens += usage.get("total_tokens", 0)

    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls_acc:
        msg["tool_calls"] = [{"id": v["id"], "type": "function", "function": v["function"]}
                             for v in sorted(tool_calls_acc.values(), key=lambda x: x["id"])]
    return {"choices": [{"message": msg, "finish_reason": finish_reason}],
            "model": model_name, "usage": usage}


def content_of(resp: dict) -> str:
    msg = resp["choices"][0]["message"]
    parts = []
    if msg.get("reasoning_content"):
        parts.append(f"<think>{msg['reasoning_content'][:500]}</think>")
    if msg.get("content"):
        parts.append(msg["content"])
    for tc in msg.get("tool_calls", []):
        parts.append(f'[tool:{tc["function"]["name"]}({tc["function"]["arguments"]})]')
    return "\n".join(parts) or "(empty)"


def answer_of(resp: dict) -> str:
    """Extract the answer. Prefers content, falls back to reasoning_content."""
    msg = resp["choices"][0]["message"]
    parts = []
    if msg.get("content"):
        parts.append(msg["content"])
    elif msg.get("reasoning_content"):
        # Qwen3 thinking mode: all output in reasoning_content, content empty
        parts.append(msg["reasoning_content"])
    for tc in msg.get("tool_calls", []):
        parts.append(f'[tool:{tc["function"]["name"]}({tc["function"]["arguments"]})]')
    return "\n".join(parts) or "(empty)"


def tool_calls_of(resp: dict):
    return resp["choices"][0]["message"].get("tool_calls") or None


def parse_score(text: str) -> float:
    try:
        return float(json.loads(text.strip()).get("score", 50))
    except Exception:
        pass
    m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
    if m: return float(m.group(1))
    m = re.search(r'\b(\d{1,3})\b', text)
    if m: return float(m.group(1))
    return 50.0


# ── NSED Protocol ──

async def evaluate_proposal(evaluator: Agent, proposal: str, label: str,
                            query: str, client: httpx.AsyncClient) -> float:
    msgs = [
        {"role": "system", "content":
         "Score this proposal 0-100. Be skeptical — look for errors. "
         'Respond ONLY with JSON: {"score": <number>, "reason": "<brief>"}'},
        {"role": "user", "content":
         f"Question: {query}\n\n--- {label} ---\n{proposal[:3000]}\n---\n\nScore (0-100):"},
    ]
    r = await gen(evaluator, msgs, client, max_tokens=200, temperature=0.3)
    return parse_score(answer_of(r))


def qv_aggregate(votes: dict) -> dict:
    proposers = {p for v in votes.values() for p in v}
    return {p: sum(math.copysign(math.sqrt(abs(v.get(p, 0))), v.get(p, 0))
                   for v in votes.values())
            for p in proposers}


async def deliberate(agents: list, query: str, client: httpx.AsyncClient,
                     tools: list = None, tool_exec: Callable = None) -> tuple:
    consensus = None
    rounds = []

    for rnd in range(1, MAX_ROUNDS + 1):
        t0 = time.monotonic()

        # Phase 1 — parallel proposals
        async def make_proposal(agent):
            msgs = [{"role": "system", "content": f"You are {agent.role}. Think step by step."}]
            if consensus:
                msgs += [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": f"Previous best answer:\n{consensus}"},
                    {"role": "user", "content": "Peer review found issues. Improve or take a different approach."},
                ]
            else:
                msgs.append({"role": "user", "content": query})
            resp = await gen(agent, msgs, client, tools=tools)
            return answer_of(resp), resp

        results = await asyncio.gather(*[make_proposal(a) for a in agents], return_exceptions=True)

        proposals, raw = {}, {}
        for a, r in zip(agents, results):
            if isinstance(r, Exception):
                proposals[a.name] = f"(error: {r})"
                raw[a.name] = {}
                print(f"    ⚠ {a.name}: {r}")
            else:
                proposals[a.name], raw[a.name] = r

        for name, text in proposals.items():
            print(f"    {name}: {text[:120].replace(chr(10),' ')}...")

        # Phase 2 — cross-evaluation (diagonal mask: can't score own)
        eval_tasks, eval_idx = [], []
        labels = {a.name: f"Proposal {chr(65+i)}" for i, a in enumerate(agents)}
        for ev in agents:
            for pn, pt in proposals.items():
                if pn != ev.name:
                    eval_tasks.append(evaluate_proposal(ev, pt, labels[pn], query, client))
                    eval_idx.append((ev.name, pn))

        raw_scores = await asyncio.gather(*eval_tasks, return_exceptions=True)
        votes = {a.name: {} for a in agents}
        for (ev, pn), s in zip(eval_idx, raw_scores):
            votes[ev][pn] = s if not isinstance(s, Exception) else 50.0

        # Phase 3 — quadratic aggregation
        scores = qv_aggregate(votes)
        winner = max(scores, key=scores.get)
        answer = proposals[winner]

        if tool_exec and raw.get(winner):
            tcs = tool_calls_of(raw[winner])
            if tcs:
                for tc in tcs:
                    tr = tool_exec(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                    answer += f"\n[result: {tr}]"

        elapsed = time.monotonic() - t0
        rounds.append(Round(rnd, proposals, votes, scores, winner, answer, elapsed))

        score_str = ", ".join(f"{k}:{v:.1f}" for k, v in sorted(scores.items(), key=lambda x: -x[1]))
        print(f"  R{rnd}: winner={winner} [{score_str}] {elapsed:.1f}s")

        if consensus and answer.strip() == consensus.strip():
            print(f"  ✅ Converged")
            break
        consensus = answer

    for a in agents:
        a.wins += sum(1 for r in rounds if r.winner == a.name)
    return consensus or "", rounds


# ── Test Harness ──

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "units": {"type": "string", "enum": ["celsius","fahrenheit"]}
        }, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculate", "description": "Evaluate a math expression",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}
        }, "required": ["expression"]}}},
]

def mock_tools(name, args):
    if name == "get_weather":
        return json.dumps({"city": args.get("city","?"), "temp_c": 22, "condition": "partly cloudy"})
    if name == "calculate":
        try: return json.dumps({"result": eval(args.get("expression","0"), {"__builtins__":{}}, {})})
        except Exception as e: return json.dumps({"error": str(e)})
    return json.dumps({"error": "unknown tool"})

TESTS = [
    ("math", "Find the sum of all positive integers n such that n^2 + 12n - 2007 is a perfect square.", None),
    ("code", "Write a Python function `lis_length(arr)` returning the length of the longest strictly increasing subsequence in O(n log n). Include the algorithm explanation.", None),
    ("tool_use", "What's the weather in Sydney right now? Also calculate 10000 * (1 + 0.055/12)**36 for me.", TOOLS),
    ("reasoning", "Compare TCP vs QUIC for: (a) mobile connections, (b) high-latency links, (c) multiplexed streams. Be specific.", None),
]


async def run_solo(agent, query, client, tools=None):
    msgs = [{"role": "system", "content": "Think step by step."}, {"role": "user", "content": query}]
    t0 = time.monotonic()
    r = await gen(agent, msgs, client, tools=tools)
    return answer_of(r), time.monotonic() - t0


# ── Server Management ──

def start_server(model: str, port: int, ctx: int = 8192):
    cmd = [str(LLAMA_SERVER), "-m", model, "--port", str(port), "-c", str(ctx),
           "-ngl", "99", "-np", "1", "--no-webui", "--host", "127.0.0.1"]
    print(f"  Starting :{port} — {Path(model).stem}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


async def wait_ready(url: str, timeout: float = 120.0):
    async with httpx.AsyncClient() as c:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            try:
                r = await c.get(f"{url}/models", timeout=3.0)
                if r.status_code == 200: return
            except Exception: pass
            await asyncio.sleep(1.0)
        raise TimeoutError(f"{url} not ready after {timeout}s")


async def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    # MiniMax already running on :64144 (via mesh-llm)
    MINIMAX_URL = "http://127.0.0.1:64144/v1"

    # Verify MiniMax is reachable
    print("Checking MiniMax on :64144...")
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{MINIMAX_URL}/models", timeout=5.0)
            r.raise_for_status()
            print(f"  ✅ MiniMax ready")
    except Exception as e:
        print(f"  ❌ MiniMax not reachable: {e}")
        sys.exit(1)

    # Models already running (started externally):
    #   :64144  MiniMax-M2.5 (253B MoE)
    #   :8081   Qwen3-8B
    #   :8082   Qwen3-30B-A3B (30B MoE, 3B active)
    BETA_URL = "http://127.0.0.1:8081/v1"
    GAMMA_URL = "http://127.0.0.1:8082/v1"

    print("Checking Qwen3-8B on :8081...")
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{BETA_URL}/models", timeout=5.0)
            r.raise_for_status()
            print(f"  ✅ ready")
    except Exception as e:
        print(f"  ❌ Not reachable: {e}"); sys.exit(1)

    print("Checking Qwen3-30B on :8082...")
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{GAMMA_URL}/models", timeout=5.0)
            r.raise_for_status()
            print(f"  ✅ ready")
    except Exception as e:
        print(f"  ❌ Not reachable: {e}"); sys.exit(1)

    procs = []
    def cleanup(*_):
        print("(no servers to stop — all managed externally)")
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))

    try:
        print("✅ All 3 models ready\n")

        agents = [
            Agent("Alpha", "MiniMax-M2.5-Q4_K_M", MINIMAX_URL,
                  "a thorough analytical reasoner who checks work carefully"),
            Agent("Beta", "Qwen3-8B-Q4_K_M", BETA_URL,
                  "a creative problem solver who considers multiple approaches"),
            Agent("Gamma", "Qwen3-30B-A3B-Q4_K_M", GAMMA_URL,
                  "a precise and methodical thinker who values correctness"),
        ]

        async with httpx.AsyncClient() as client:
            if query:
                print(f"{'='*60}\nQuery: {query}\n{'='*60}")
                print("\n--- Solo baselines ---")
                for a in agents:
                    try:
                        txt, t = await run_solo(a, query, client)
                        print(f"  {a.name} ({a.model}) [{t:.1f}s]:\n    {txt[:300].replace(chr(10), chr(10)+'    ')}\n")
                    except Exception as e:
                        print(f"  {a.name}: ERROR {e}")
                print("--- MoM Ensemble ---")
                answer, rounds = await deliberate(agents, query, client)
                print(f"\n{'='*60}\nFinal ({rounds[-1].winner}):\n{answer[:1000]}\n{'='*60}")
            else:
                results = []
                for cat, q, tools in TESTS:
                    print(f"\n{'#'*60}\n{cat.upper()}: {q[:80]}...\n{'#'*60}")

                    solos = {}
                    for a in agents:
                        try:
                            txt, t = await run_solo(a, q, client, tools)
                            solos[a.name] = {"text": txt[:1000], "time": t}
                            print(f"  Solo {a.name}: {t:.1f}s — {txt[:150].replace(chr(10),' ')}")
                        except Exception as e:
                            print(f"  Solo {a.name}: ERROR {e}")
                            solos[a.name] = {"text": str(e), "time": 0}

                    print()
                    answer, rounds = await deliberate(agents, q, client, tools,
                                                      mock_tools if tools else None)
                    results.append(dict(
                        cat=cat, query=q, solos=solos,
                        ensemble=answer[:1000],
                        rounds=len(rounds),
                        total_s=sum(r.elapsed for r in rounds),
                        winner=rounds[-1].winner if rounds else "?",
                        round_winners=[r.winner for r in rounds],
                    ))

                # Summary
                print(f"\n\n{'='*60}\nSUMMARY\n{'='*60}")
                for r in results:
                    print(f"\n{r['cat']}: {r['rounds']} rounds, {r['total_s']:.1f}s, winners={r['round_winners']}")
                    for name, s in r['solos'].items():
                        print(f"  solo {name}: {s['time']:.1f}s")
                print(f"\nAgent totals:")
                for a in agents:
                    print(f"  {a.name} ({a.model}): {a.tokens} tok, {a.time_s:.1f}s, {a.wins} wins")

                out = Path(__file__).parent / "results"
                out.mkdir(exist_ok=True)
                fname = out / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(fname, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\nSaved: {fname}")
    finally:
        cleanup()


if __name__ == "__main__":
    asyncio.run(main())
