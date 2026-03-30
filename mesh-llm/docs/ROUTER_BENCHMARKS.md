# Model Router Benchmarks

For the separate prefix-affinity routing benchmark, see [PREFIX_AFFINITY_BENCHMARKS.md](./PREFIX_AFFINITY_BENCHMARKS.md).

## Test Setup

- **Machine**: M4 Max, 52GB VRAM
- **Context**: 8192 tokens
- **Prompts**: chat, code, reasoning, tool_call, creative, long_code
- **Metrics**: tok/s (completion tokens / wall time), quality (direct answer vs thinking)

## Results

| Model | tok/s | Size | Direct Answers | Tool Call | Notes |
|-------|------:|------|:-:|:-:|-------|
| Qwen3-30B-A3B (MoE) | 103.2 | 17GB | ❌ (thinking) | ✅ | Fastest, but wastes tokens on reasoning |
| Hermes-2-Pro-Mistral-7B | 77.3 | 4GB | ✅ | ✅ | Best agent model, concise |
| Qwen3-8B | 77.2 | 5GB | ❌ (thinking) | ❌ | Thinks instead of answering |
| GLM-4.7-Flash (MoE) | 71.2 | 17GB | ❌ (thinking) | ✅ | Fast MoE but thinking-heavy |
| Qwen2.5-Coder-7B | 71.1 | 4GB | ✅ | ✅ | Strong code, fast |
| Mistral-Small-3.1-24B | 25.9 | 13GB | ✅ | ✅ | Good quality, slower |
| Qwen2.5-32B-Instruct | 17.8 | 18GB | ✅ | ✅ | Best quality, slowest |

## Key Findings

1. **Thinking models waste tokens**: Qwen3 and GLM-4.7-Flash spend all tokens reasoning and often don't produce a final answer within the token budget. They need `--reasoning-format deepseek` plus much higher `max_tokens`. Not ideal for routing.

2. **7B models are the sweet spot for speed**: Hermes-7B and Qwen2.5-Coder-7B give direct answers at 71-83 tok/s. Combined they're only 8GB VRAM.

3. **Qwen2.5-32B for quality**: 18 tok/s is slow but it gives the best prose, reasoning, and code. Worth routing hard problems here.

4. **All models handle tool calls**: Every model correctly emitted `bash({"command":"ls"})` when given tools.

5. **MoE speed advantage is real**: 103 tok/s for Qwen3-30B-A3B vs 18 tok/s for similarly-sized Qwen2.5-32B. But the thinking overhead negates it for short responses.

## Recommended Router Config

For a single machine with 52GB VRAM, run 2 models:

| Slot | Model | VRAM | Purpose |
|------|-------|------|---------|
| Fast | Hermes-2-Pro-Mistral-7B | 4GB | Chat, tool calls, simple questions |
| Strong | Qwen2.5-32B-Instruct | 18GB | Reasoning, code, creative, hard problems |

22GB total, leaving 30GB for KV cache. Route simple → fast, complex → strong.

For the Studio (206GB), add a frontier tier:
- Qwen3-235B-A22B or Qwen3-Coder-Next as the heavy model
- Plus a fast 7B for routing/simple tasks

## Not Yet Tested

- Qwen3.5-27B (needs newer llama.cpp for rope format)
- Qwen3-Coder-Next (48GB, downloading)
- Qwen3-235B-A22B (142GB, downloading)
- MiniMax-M2.5 (138GB, downloading)
