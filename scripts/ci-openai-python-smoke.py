#!/usr/bin/env python3
"""Official openai-python smoke against a mesh-llm OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
from typing import Iterable


def streamed_text(chunks: Iterable[object]) -> str:
    parts: list[str] = []
    saw_choice = False
    for chunk in chunks:
        choices = getattr(chunk, "choices", None) or []
        for choice in choices:
            saw_choice = True
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
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
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "openai package not installed; run `python -m pip install openai` first"
        ) from exc

    client = OpenAI(
        api_key="mesh-llm-ci",
        base_url=args.base_url,
    )

    models = client.models.list()
    if not models.data:
        raise RuntimeError("models.list returned no models")
    model = models.data[0].id
    print(f"Using model: {model}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Say hello in exactly 4 words."},
        ],
        max_tokens=32,
        temperature=0,
    )
    message = response.choices[0].message.content
    if not message or not message.strip():
        raise RuntimeError("non-streaming chat returned empty content")
    print(f"Non-streaming response: {message.strip()}")

    stream = client.chat.completions.create(
        model=model,
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
