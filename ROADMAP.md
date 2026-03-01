# Roadmap

High-level directions for mesh-llm. Not promises — just things we're thinking about.

## Production relay infrastructure

Currently mesh-llm uses iroh's default public relays for NAT traversal. These work but we don't control them — availability, latency, and capacity are someone else's problem. For production use we need our own relays, either via iroh's paid relay service or self-hosted on something like Fly.io (with proper end-to-end TLS — a plain reverse proxy won't work since iroh needs TLS termination at the relay itself). Dedicated relays in key regions (US, EU, AU) would improve connectivity reliability and let us tune for our traffic patterns. The mesh binary would ship with these as default relays instead of iroh's public ones.

## Agent launcher

`mesh-llm run` as a one-command way to launch popular AI agents talking to the mesh. Similar to how `ollama run` gives you a model, but for agents:

```bash
mesh-llm run goose          # launch goose session with mesh backend
mesh-llm run pi             # launch pi with --provider mesh
mesh-llm run claude         # claude code with mesh as provider  
mesh-llm run opencode       # opencode pointed at mesh API
mesh-llm run openclaw       # openclaw with mesh endpoint
```

Each agent gets configured with the mesh's OpenAI-compatible API (`localhost:9337`) and the best available model. We already print launch commands for pi and goose when the mesh is ready — this just makes it a single command. The mesh handles model selection, routing, and failover transparently.

Similar to how ollama is becoming an agent launcher (`ollama launch openclaw --model pshohel/kimi-k2.5`) — the local inference backend is a natural place to own the "run an agent" workflow.

## Mobile client

The QUIC transport and relay infrastructure already handle NAT traversal — a phone could join a mesh as a client the same way the Fly.io web app does. Start with client-only mode: discover a mesh, connect via relay, chat with models running on real hardware elsewhere. No GPU needed on the device. iOS and Android both have good QUIC support. The existing `--client --auto` code path is exactly what this would use under the hood.

## Single binary distribution

Currently mesh-llm ships alongside `llama-server` and `rpc-server` as separate binaries (the `just bundle` tarball). [llama-cpp-2](https://crates.io/crates/llama-cpp-2) demonstrates static linking of llama.cpp into a Rust binary at build time. We could do the same — compile llama.cpp (with Metal/CUDA) directly into `mesh-llm` so the entire thing is one binary. No bundle, no `--bin-dir`, just `mesh-llm`.

## Mesh as a library (`mesh-llm` crate)

Extract the mesh layer into a `lib.rs` published as a crate. Other Rust projects could embed mesh-llm in-process — join a mesh, serve models, route requests — without shelling out. The degenerate case (single node, one model) becomes local serving with no mesh overhead, so projects don't need duplicate llama.cpp dependencies. Combined with static linking above, any Rust binary could `use mesh_llm` and get distributed inference for free.

## Medusa-style speculative decoding

We already use tree-based speculative decoding with a separate draft model (e.g. Qwen2.5-0.5B drafting for Qwen2.5-32B). [Medusa](https://arxiv.org/abs/2401.10774) takes a different approach: instead of a separate draft model, it trains lightweight prediction heads directly on the base model. Each head predicts a different future token position, forming a candidate tree verified in one forward pass — similar speedup (~2-3x) but no second model to manage.

This is interesting for mesh-llm because:
- **No draft model download/VRAM**: Medusa heads are tiny (<100MB) vs a separate 0.5-1B model
- **Better for tensor splits**: Draft models must run locally (cross-mesh adds 2 RTTs for negligible compute), but Medusa heads ride on the same forward pass that's already distributed
- **Simpler catalog**: No need to pair every model family with a compatible draft

The blocker is GGUF/llama.cpp support — Medusa heads need to be quantized and integrated into the inference loop. Worth watching [llama.cpp#3137](https://github.com/ggml-org/llama.cpp/issues/3137) and the broader ecosystem.

## MoE expert parallelism — zero cross-node traffic ✅

MoE models (Mixtral, Qwen3-MoE, DeepSeek, GLM, OLMoE) have huge total parameters but small active parameters per token. Standard expert parallelism uses all-to-all dispatch — a non-starter over mesh networks. mesh-llm takes a different approach: **overlapping expert shards with zero cross-node inference traffic**.

### How it works

Each node gets a standalone GGUF containing the full trunk (attention, norms, embeddings) plus a **subset of experts**. The router naturally adapts — it can only select from the experts present in its GGUF. No masking at runtime, no remote calls, no coordination between nodes during inference.

The key insight: experts have wildly unequal importance. A **shared core** of the hottest experts (by gate mass) is replicated on every node. Remaining experts are distributed uniquely. With sufficient overlap (~68%), both nodes produce excellent output on all tested prompts.

```
Node A: [shared core: 46 experts] + [unique-A: 41 experts] = 87 experts
Node B: [shared core: 46 experts] + [unique-B: 41 experts] = 87 experts
                                                    Total coverage: 128/128
```

### What's built

- **Auto-detection**: reads `expert_count` from GGUF header (~1ms). Any MoE model works — no catalog entry needed.
- **`llama-moe-split`**: slices expert tensors along the expert dimension, gathers router gate rows, clamps `expert_used_count`. Produces a self-contained GGUF that loads in unmodified llama-server.
- **Automatic deployment**: `mesh-llm --model <moe-model> --split` → detects MoE, computes overlapping assignments, each node splits locally (cached), starts its own llama-server. Session-sticky routing via user/session hash.
- **Pre-computed rankings**: expert importance rankings baked into catalog for known models. Unknown models use conservative 50% shared core default.
- **Integration tested**: OLMoE-1B-7B across Sydney ↔ Sydney (225ms WAN), Qwen3-30B-A3B locally. Both shards produce coherent output.

### Only splits when needed

Same logic as tensor split: if the model fits on one machine, it runs normally. MoE splitting only kicks in when the model doesn't fit or `--split` is forced. GLM-4.7-Flash (MoE, 17GB) on a 64GB machine → no split, full speed.

### Scale path

| Model | Total | Active | Experts | Status |
|---|---|---|---|---|
| OLMoE-1B-7B | 7B | 1B | 64 | ✅ Tested (2 nodes, WAN) |
| Qwen3-30B-A3B | 30B | 3B | 128 | ✅ Tested (local, quality validated) |
| Mixtral 8×22B | 141B | 44B | 8 | Next — actually needs distribution |
| Qwen3-235B-A22B | 235B | 22B | 128 | Target — needs 3+ nodes |


