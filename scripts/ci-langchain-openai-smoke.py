#!/usr/bin/env python3
"""langchain-openai smoke against a mesh-llm OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
from typing import Any, Iterable


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def streamed_text(chunks: Iterable[object]) -> str:
    parts: list[str] = []
    saw_chunk = False
    for chunk in chunks:
        saw_chunk = True
        text = content_text(getattr(chunk, "content", ""))
        if text:
            parts.append(text)
    if not saw_chunk:
        raise RuntimeError("stream returned no chunks")
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
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "langchain-openai package not installed; run `python -m pip install langchain-openai` first"
        ) from exc

    llm = ChatOpenAI(
        model=args.model,
        api_key="mesh-llm-ci",
        base_url=args.base_url,
        temperature=0,
        max_tokens=32,
        stream_usage=False,
    )

    print(f"Using model: {args.model}")

    response = llm.invoke([("human", "Say hello in exactly 4 words.")])
    message = content_text(response.content).strip()
    if not message:
        raise RuntimeError("non-streaming chat returned empty content")
    print(f"Non-streaming response: {message}")

    stream = llm.stream([("human", "Count from one to three.")])
    text = streamed_text(stream)
    print(f"Streaming response: {text}")


if __name__ == "__main__":
    main()
