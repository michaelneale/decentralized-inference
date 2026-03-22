#!/usr/bin/env python3
"""Compare Claude vs local Qwen3-8B on a code review task."""
from __future__ import annotations
import asyncio, json, os, time
import httpx

CODE = open("/tmp/mom-agent-test/stats.py").read()

TASK = f"""Review this Python code. Find and fix all bugs, add error handling for edge cases (empty lists, single elements, non-numeric values). 

For each issue found, state:
1. The bug or missing handling
2. The fix
3. A test case that would catch it

Be specific and thorough.

```python
{CODE}
```"""

CLAUDE_KEY = os.environ["ANTHROPIC_API_KEY"]
LOCAL_URL = "http://localhost:8081/v1"


async def call_claude(client: httpx.AsyncClient) -> tuple[str, float]:
    t0 = time.monotonic()
    r = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": CLAUDE_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 3000,
            "messages": [{"role": "user", "content": TASK}],
        },
        timeout=120.0,
    )
    r.raise_for_status()
    d = r.json()
    elapsed = time.monotonic() - t0
    text = d["content"][0]["text"]
    tokens = d.get("usage", {})
    return text, elapsed, tokens


async def call_qwen(client: httpx.AsyncClient) -> tuple[str, float]:
    t0 = time.monotonic()
    # Stream to collect full response
    content = ""
    async with client.stream(
        "POST", f"{LOCAL_URL}/chat/completions",
        json={
            "model": "Qwen3-8B-Q4_K_M.gguf",
            "messages": [{"role": "user", "content": TASK}],
            "max_tokens": 3000,
            "temperature": 0.7,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180.0,
    ) as resp:
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
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content += delta["content"]
    elapsed = time.monotonic() - t0
    return content, elapsed, {}


async def main():
    async with httpx.AsyncClient() as client:
        print("=" * 60)
        print("CODE REVIEW COMPARISON: Claude vs Qwen3-8B")
        print("=" * 60)

        # Run both
        print("\nRunning Claude Sonnet 4...")
        claude_text, claude_time, claude_usage = await call_claude(client)
        print(f"  Done: {claude_time:.1f}s, {len(claude_text)} chars")

        print("\nRunning Qwen3-8B (local)...")
        qwen_text, qwen_time, _ = await call_qwen(client)
        print(f"  Done: {qwen_time:.1f}s, {len(qwen_text)} chars")

        # Output
        print(f"\n{'=' * 60}")
        print(f"CLAUDE SONNET 4 ({claude_time:.1f}s)")
        print(f"{'=' * 60}")
        print(claude_text)

        print(f"\n{'=' * 60}")
        print(f"QWEN3-8B LOCAL ({qwen_time:.1f}s)")
        print(f"{'=' * 60}")
        print(qwen_text)

        # Save
        with open("/tmp/mom-agent-test/comparison.json", "w") as f:
            json.dump({
                "claude": {"text": claude_text, "time_s": claude_time, "usage": claude_usage},
                "qwen": {"text": qwen_text, "time_s": qwen_time},
            }, f, indent=2)
        print(f"\nSaved to /tmp/mom-agent-test/comparison.json")


if __name__ == "__main__":
    asyncio.run(main())
