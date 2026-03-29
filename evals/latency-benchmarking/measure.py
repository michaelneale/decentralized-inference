#!/usr/bin/env python3
"""
Measure TTFT and total time from an OpenAI-compatible request.

For streaming-capable backends, this measures:
- TTFT from request send until first SSE chunk with visible content
- total wall-clock time until stream completion
- exact completion_tokens via a paired deterministic non-streaming request

Outputs JSON: {"ttft_ms": ..., "total_ms": ..., "tokens": ..., "tok_s": ...}
"""

import argparse
import json
import time
import sys
import urllib.request
import urllib.error


def make_body(model, prompt, max_tokens, stream, messages_json=None):
    if messages_json:
        messages = json.loads(messages_json)
    else:
        messages = [{"role": "user", "content": prompt}]
    return json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }).encode()


def extract_visible_delta(obj):
    choice = (obj.get("choices") or [{}])[0]
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if content:
            return str(content)
        reasoning = delta.get("reasoning_content")
        if reasoning:
            return str(reasoning)
    text = choice.get("text")
    if text:
        return str(text)
    return ""


def measure_completion_usage(url, model, prompt, max_tokens, messages_json=None):
    body = make_body(model, prompt, max_tokens, False, messages_json=messages_json)
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)

    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None:
        return int(completion_tokens)

    choices = payload.get("choices") or []
    if not choices:
        return 0

    message = choices[0].get("message") or {}
    text = message.get("content")
    if text is None:
        text = choices[0].get("text") or ""
    if isinstance(text, list):
        text = json.dumps(text, ensure_ascii=False)
    return max(1, len(str(text).split())) if str(text).strip() else 0


def measure_streaming(url, model, prompt, max_tokens, messages_json=None):
    body = make_body(model, prompt, max_tokens, True, messages_json=messages_json)

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.monotonic()
    t_first_token = None
    saw_visible_content = False
    saw_error = None

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
                    if obj.get("error"):
                        saw_error = obj.get("error")
                        continue
                    if extract_visible_delta(obj):
                        if t_first_token is None:
                            t_first_token = time.monotonic()
                        saw_visible_content = True
                except json.JSONDecodeError:
                    pass

    t_end = time.monotonic()

    if saw_error:
        raise RuntimeError(f"streaming error: {saw_error}")

    ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else None
    total_ms = (t_end - t_start) * 1000
    completion_tokens = measure_completion_usage(
        url, model, prompt, max_tokens, messages_json=messages_json
    )
    if t_first_token and completion_tokens > 0:
        gen_ms = (t_end - t_first_token) * 1000
        tok_s = completion_tokens / gen_ms * 1000
    else:
        tok_s = 0

    return {
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "total_ms": round(total_ms, 1),
        "tokens": completion_tokens,
        "tok_s": round(tok_s, 2),
        "streaming": True,
        "visible_content": saw_visible_content,
    }


def measure_non_streaming(url, model, prompt, max_tokens, messages_json=None):
    body = make_body(model, prompt, max_tokens, False, messages_json=messages_json)
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    t_end = time.monotonic()

    total_ms = (t_end - t_start) * 1000
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = 0

    tok_s = completion_tokens / total_ms * 1000 if total_ms > 0 else 0
    return {
        "ttft_ms": None,
        "total_ms": round(total_ms, 1),
        "tokens": completion_tokens,
        "tok_s": round(tok_s, 2),
        "streaming": False,
        "visible_content": completion_tokens > 0,
    }


def measure(url, model, prompt, max_tokens, messages_json=None):
    try:
        result = measure_streaming(
            url, model, prompt, max_tokens, messages_json=messages_json
        )
    except urllib.error.HTTPError as err:
        if err.code != 400:
            raise
        result = measure_non_streaming(
            url, model, prompt, max_tokens, messages_json=messages_json
        )

    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    parser.add_argument("--model", default="test")
    parser.add_argument("--prompt", default="Say hello in 5 languages")
    parser.add_argument("--messages-json")
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()
    measure(
        args.url,
        args.model,
        args.prompt,
        args.max_tokens,
        messages_json=args.messages_json,
    )
