# Roadmap

High-level directions for mesh-llm. Not promises — just things we're thinking about.

## Production relay infrastructure

Currently mesh-llm uses iroh's default public relays for NAT traversal. These work but we don't control them. We have a self-hosted iroh-relay on Fly.io ready (`relay/`) but it's not shipped as the default yet. Dedicated relays in key regions (US, EU, AU) would improve connectivity reliability and let us tune for our traffic patterns.

## Agent launcher

`mesh-llm run` as a one-command way to launch popular AI agents talking to the mesh:

```bash
mesh-llm run goose          # launch goose session with mesh backend
mesh-llm run pi             # launch pi with --provider mesh
mesh-llm run opencode       # opencode pointed at mesh API
```

We already print launch commands for pi and goose when the mesh is ready, and the web console shows them in the API popover. This just makes it a single command. There's also a native Goose provider (`mesh` provider in `block/goose` on `micn/mesh-provider` branch) that auto-downloads and daemonises mesh-llm.

## Mobile client

The QUIC transport and relay infrastructure already handle NAT traversal — a phone could join a mesh as a client the same way the Fly.io web app does. The existing `--client --auto` code path is exactly what this would use. The React web console already works on mobile viewports.

## Single binary distribution

Currently mesh-llm ships alongside `llama-server` and `rpc-server` as separate binaries (the `just bundle` tarball). [llama-cpp-2](https://crates.io/crates/llama-cpp-2) demonstrates static linking of llama.cpp into a Rust binary at build time. We could compile llama.cpp (with Metal/CUDA) directly into `mesh-llm` — one binary, no bundle, no `--bin-dir`.

## Mesh as a library (`mesh-llm` crate)

Extract the mesh layer into a `lib.rs` published as a crate. Other Rust projects could embed mesh-llm in-process — join a mesh, serve models, route requests — without shelling out.

## Medusa-style speculative decoding

We already use tree-based speculative decoding with a separate draft model. [Medusa](https://arxiv.org/abs/2401.10774) trains lightweight prediction heads directly on the base model — no second model to manage, rides on the same forward pass. Blocked on GGUF/llama.cpp support.

## MoE expert sharding ✅

Implemented. Auto-detects MoE from GGUF metadata, computes overlapping expert assignments (shared core + unique shards), splits locally, each node runs its own llama-server with session-sticky routing. Zero cross-node traffic during inference. See [MoE_PLAN.md](MoE_PLAN.md) for design details and [MoE_SPLIT_REPORT.md](MoE_SPLIT_REPORT.md) for validation results.

Remaining work tracked in [mesh-llm/TODO.md](mesh-llm/TODO.md):
- Phase 4: optimized rankings for unknown models
- Phase 5: scale testing on models that actually need distribution (Mixtral 8×22B, Qwen3-235B)

## Demand-based rebalancing

Partially done. Unified demand map propagates via gossip, standby nodes promote to serve unserved models, Nostr `wanted` lists reflect active demand. Next: large-VRAM hosts serving small models should auto-upgrade when demand warrants it.

## Resilience

Done: Nostr re-discovery when all peers are lost (v0.26.1). `--auto` nodes re-discover and rejoin after 90s with 0 peers. Next: multi-node tensor split recovery when one split peer dies.
