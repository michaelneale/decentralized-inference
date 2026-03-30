#!/usr/bin/env python3
"""LiteLLM smoke against a mesh-llm OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
from typing import Any, Iterable


def get_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def streamed_text(chunks: Iterable[object]) -> str:
    parts: list[str] = []
    saw_choice = False
    for chunk in chunks:
        choices = get_field(chunk, "choices") or []
        for choice in choices:
            saw_choice = True
            delta = get_field(choice, "delta")
            if delta is None:
                continue
            content = get_field(delta, "content")
            if isinstance(content, str) and content:
                parts.append(content)
    if not saw_choice:
        raise RuntimeError("stream returned no choices")
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError("stream returned no content")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    try:
        from litellm import completion
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "litellm package not installed; run `python -m pip install litellm` first"
        ) from exc

    provider_model = f"openai/{args.model}"
    print(f"Using model: {provider_model}")

    response = completion(
        model=provider_model,
        api_base=args.base_url,
        api_key="mesh-llm-ci",
        messages=[
            {"role": "user", "content": "Say hello in exactly 4 words."},
        ],
        max_tokens=32,
        temperature=0,
    )
    first_choice = (get_field(response, "choices") or [None])[0]
    message = get_field(get_field(first_choice, "message"), "content")
    if not isinstance(message, str) or not message.strip():
        raise RuntimeError("non-streaming chat returned empty content")
    print(f"Non-streaming response: {message.strip()}")

    stream = completion(
        model=provider_model,
        api_base=args.base_url,
        api_key="mesh-llm-ci",
        messages=[
            {"role": "user", "content": "Count from one to three."},
        ],
        max_tokens=32,
        temperature=0,
        stream=True,
    )
    text = streamed_text(stream)
    print(f"Streaming response: {text}")


if __name__ == "__main__":
    main()
