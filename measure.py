#!/usr/bin/env python3
"""
Measure TTFT and total time from a streaming llama-server request.

Outputs JSON: {"ttft_ms": ..., "total_ms": ..., "tokens": ..., "tok_s": ...}

TTFT = wall-clock time from request sent until first SSE chunk with content.
Total = wall-clock time from request sent until stream completes.
tok/s = tokens / (total - ttft), i.e. generation speed after first token.
"""

import argparse
import json
import time
import sys
import urllib.request


def measure(url, prompt, max_tokens):
    body = json.dumps({
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.monotonic()
    t_first_token = None
    token_count = 0

    with urllib.request.urlopen(req, timeout=120) as resp:
        buf = b""
        while True:
            chunk = resp.read(1024)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line or not line.startswith(b"data: "):
                    continue
                data_str = line[6:].decode()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    # Check both content and reasoning_content (thinking models)
                    has_content = bool(delta.get("content") or delta.get("reasoning_content"))
                    if has_content:
                        if t_first_token is None:
                            t_first_token = time.monotonic()
                        token_count += 1
                except json.JSONDecodeError:
                    pass

    t_end = time.monotonic()

    ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else None
    total_ms = (t_end - t_start) * 1000
    # Generation speed: tokens after TTFT
    if t_first_token and token_count > 1:
        gen_ms = (t_end - t_first_token) * 1000
        tok_s = token_count / gen_ms * 1000
    else:
        tok_s = 0

    print(json.dumps({
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "total_ms": round(total_ms, 1),
        "tokens": token_count,
        "tok_s": round(tok_s, 2),
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    parser.add_argument("--prompt", default="Say hello in 5 languages")
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()
    measure(args.url, args.prompt, args.max_tokens)
