# Model Router Design

## Overview

mesh-llm becomes a smart routing layer: multiple models across the mesh, requests classified and routed to the best model for the job.

## Core Idea

Instead of every node serving one model and the proxy blindly forwarding, the proxy **classifies** each incoming request and **routes** it to the most appropriate model available in the mesh. Different nodes can serve different models — a big machine runs a 142GB MoE, a small one runs a 7B. The router picks based on task type, model capabilities, and observed speed.

## Request Classification

Categories (kept simple, extensible later):

| Category | Signal | Example |
|----------|--------|---------|
| `code` | Tools defined, code fences in history, system prompt mentions code/dev | "Write a Python script that..." |
| `reasoning` | Math, logic, multi-step, "think step by step", "explain why" | "Prove that √2 is irrational" |
| `chat` | Short messages, conversational tone, no tools | "What's the capital of France?" |
| `tool_call` | Tools in request, function-calling pattern | Agent loop with tool schemas |
| `creative` | Writing, stories, poetry | "Write a short story about..." |

### Classification Method

**Phase 1: Heuristic (no LLM needed)**
- Has `tools` array → `tool_call`
- System prompt contains "code", "developer", "programming" → `code`
- Contains math symbols, "prove", "calculate", "step by step" → `reasoning`
- Message count > 10 or short last message → `chat`
- Default → `chat`

**Phase 2: Small model classifier (later)**
- Route to a tiny model (0.6B) with a fixed prompt: "Classify this request as one of: code, reasoning, chat, tool_call, creative. Output only the category."
- <100ms latency, highly accurate.

## Model Capabilities

Each catalog model gets a capability profile:

```rust
struct ModelProfile {
    /// What this model is good at, in priority order
    strengths: &'static [Category],
    /// Relative quality tier (1=draft, 2=good, 3=strong, 4=frontier)
    tier: u8,
    /// Is this an MoE model? (faster tok/s for its total param count)
    is_moe: bool,
}
```

Examples:

| Model | Tier | Strengths | Notes |
|-------|------|-----------|-------|
| Qwen3-235B-A22B | 4 | code, reasoning, chat, creative | Frontier MoE, fast for its size |
| Qwen2.5-72B | 3 | chat, reasoning, code | Strong dense all-rounder |
| Qwen3-30B-A3B | 3 | chat, reasoning, code | Fast MoE, good quality |
| Qwen3-32B | 3 | reasoning, code, chat | Best dense mid-size |
| DeepSeek-R1-Distill-70B | 3 | reasoning | Specialist reasoner |
| Qwen2.5-Coder-32B | 3 | code | Specialist coder |
| Mistral-Small-3.1-24B | 2 | chat, tool_call | Good tool calling |
| Hermes-2-Pro-Mistral-7B | 2 | tool_call, chat | Proven agent model |
| Qwen3-8B | 2 | chat, code | Fast, good enough for simple |
| GLM-4.7-Flash | 2 | chat, tool_call | Very fast MoE |

## Routing Logic

```
request comes in
  → classify(request) → category
  → find models in mesh that list category as a strength
  → rank by: tier (higher better), then observed_tok_per_sec (faster better)
  → pick top candidate
  → forward request to that model's host
```

### Tiebreaking

When multiple models can handle a category equally:
1. **Highest tier wins** — always prefer quality
2. **Fastest observed speed wins** — track actual tok/s from recent responses
3. **Lowest current load wins** — prefer idle nodes (use `inflight_requests`)

### Escalation (Phase 2)

After getting a response, optionally check quality:
- If response is very short for a complex question → re-route to higher tier
- If response contains "I don't know" / "I can't" → try another model
- Keep it simple: one retry max, not a loop

### Fan-out (Phase 3)

For high-value requests (long prompts, complex tasks):
- Send to 2 models in parallel
- Return first response that's "good enough" (or wait for both and pick better)
- Judge with heuristics first (length, presence of code blocks, confidence markers)

## Speed Tracking

The proxy already sees every response. Add per-model tracking:

```rust
struct ModelStats {
    /// Exponential moving average of tokens/sec
    avg_tok_per_sec: f64,
    /// Recent response times
    avg_latency_ms: f64,
    /// Number of requests served
    request_count: u64,
    /// Last time this model was used
    last_used: Instant,
}
```

Populated from response headers (`x-tokens-per-second`) or by counting SSE `data:` chunks / timing.

## Multi-Model Per Node

For large VRAM machines, allow running multiple models simultaneously:

- Election assigns multiple models to a node based on VRAM budget
- Each model gets its own llama-server on a different port
- VRAM allocated proportionally (e.g., 142GB for heavy + 17GB for fast on 206GB)
- Gossip announces all served models per node

This is **Phase 2** — start with one model per node, route across nodes.

## Implementation Plan

### Phase 1: Heuristic Router (this branch)

1. Add `ModelProfile` to the curated model metadata in `models/mod.rs`
2. Add `classify_request()` in new `router.rs` — heuristic only
3. Add `ModelStats` tracking in `proxy.rs` — observe tok/s from responses
4. Modify proxy to pick model based on classification + available models
5. Announce model capabilities in gossip (already have model name)

Changes are entirely in the proxy layer. No changes to election, mesh, or llama-server.

### Phase 2: Escalation + Multi-Model

6. Add response quality check (heuristic: length, completeness)
7. Retry with higher-tier model on low quality
8. Multi-model per node in election

### Phase 3: Fan-out + LLM Classifier

9. Parallel request to 2 models
10. Judge with small model
11. Replace heuristic classifier with tiny model

## Wire Format

No gossip changes needed for Phase 1. The model name is already in `PeerAnnouncement`. The proxy can look up the profile from the catalog by name.

For Phase 2 (multi-model per node), `PeerAnnouncement.model` becomes a `Vec<String>` — but that's a wire format change. Alternatively, announce multiple `PeerAnnouncement` entries per node, one per model. Simpler, no format change.

## What This Is Not

- Not a load balancer (we already have that)
- Not model fine-tuning or LoRA routing
- Not a full agent orchestrator — just smart model selection
- Not speculative decoding (that's within a single model)
