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

## MoE expert parallelism (non-all-to-all)

Current tensor splitting distributes layers across nodes, which works for dense models. MoE models have a different shape: huge total parameters but small active parameters per token, with the bulk in expert FFN weights. The standard expert parallelism (EP) approach uses all-to-all dispatch — any token can route to any expert on any host — which is a non-starter over mesh networks with real latency.

**The goal**: distribute MoE experts across mesh nodes with bounded network fanout (≤1 remote host per token per MoE layer). Never all-to-all.

### Why this matters

| Model | Total params | Active/token | Expert count |
|---|---|---|---|
| [Mixtral 8×7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 47B | ~13B | 8 |
| [Mixtral 8×22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) | 141B | ~44B | 8 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 236B | ~21B | 160 |
| [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | 235B | ~22B | 128 |

These models can't fit on one machine but only activate a fraction of parameters per token. If you can place experts on different nodes and route tokens to the right one, you get the capacity of the full model with the compute cost of the active subset.

### Approach: bounded-fanout expert routing

Most EP literature assumes NVLink/InfiniBand where all-to-all is cheap. Mesh-llm operates over WiFi, WAN, and QUIC tunnels — all-to-all is not an option. The key insight is that non-all-to-all routing is achievable by construction, not by hoping for good luck:

- **Group-masked routing**: assign experts to host-local groups. Token first picks a group (or is assigned one deterministically), then picks experts *only within that group*. The router logits for experts outside the group are masked to -inf. Result: at most 1 remote destination per MoE layer, by construction — not a scatter
- **Same-host top-2**: both selected experts must be on the same host. Mask enforces this. Keeps top-2 behaviour without fanning out
- **Sticky session routing**: `hash(session_id) % num_hosts` picks a host group, fixed for the whole conversation. No ping-pong between hosts across tokens
- **Replicated hot experts**: the 1–4 most frequently chosen "generalist" experts replicated on every host. Remote becomes the exception (cache miss), not the common path
- **Top-1 as fallback**: if top-2 same-host still has quality issues, top-1 routing (Switch Transformer style) guarantees one expert = one host with zero fanout

The network payload per MoE layer is just activations (a few KB per token), not weights. With sticky routing + hot replication, most tokens never leave the local host. When they do, it's one bounded RPC to one known destination — similar latency profile to the current HTTP tunnel, not the latency-multiplied RPC problem that tensor splits have.

**Per-host expert shards**: each node stores the shared trunk (embeddings, attention, norms, router, output head — replicated everywhere) plus only its assigned experts (unique per host). Much less storage than the full model on every machine.

### What it takes to build

This is not a simple extension of the current RPC-based tensor split. Key work:

**Rust side (mesh-llm)**:
- Expert placement logic: assign expert groups to nodes based on VRAM, replication policy
- New RPC protocol for expert dispatch: send activations to expert host, get results back. One destination per MoE layer, not scatter
- Router modification: intercept MoE routing decisions, apply group mask before expert selection
- Per-host model packaging: tooling to split a checkpoint into trunk + expert shards

**llama.cpp side**:
- Hook into MoE layer evaluation to call external expert backends instead of (or alongside) local compute
- llama.cpp currently has no expert parallelism — it splits by [whole layers, not experts](https://github.com/ggml-org/llama.cpp/discussions/11784). This would be new
- Good news: GGUF stores experts as [separate per-expert tensors](https://deepwiki.com/ggml-org/llama.cpp/3.2-model-loading-and-management) (`blk.{layer}.ffn_gate_exps.{expert}.weight`), so extracting individual experts and building per-host GGUF shards is feasible at the file format level

**Quality tradeoff**: the hard part isn't the file format — it's the routing. These models were trained with unconstrained top-2 routing across all experts. Group-masking constrains that at inference time, which means a token might not get its "best" expert if it lives on another host. Quality impact depends on expert diversity, group size, and how specialised vs general the experts are. Mitigated by larger groups, replicated generalists, and same-host top-2. Needs empirical tuning per model family — the open question is how much quality you actually lose in practice.

### Starter model

[Mixtral 8×7B Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) — 8 experts, popular, well-tooled, small enough to iterate on. Available as both HF safetensors (clean expert separation) and [GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF). Scale to 8×22B and DeepSeek-V2 once the mechanics work.

This would be a new execution mode alongside the current layer-based tensor split — chosen automatically based on model architecture.


