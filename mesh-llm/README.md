# mesh-llm

Rust sidecar for distributed llama.cpp inference over QUIC. See the [project README](../README.md) for usage.

```
src/
├── main.rs        CLI, orchestration, startup flows (auto, idle, passive)
├── mesh.rs        QUIC endpoint, gossip, peer management, request rate sharing
├── election.rs    Per-model host election, latency-aware split, llama-server lifecycle
├── proxy.rs       HTTP proxy plumbing: request parsing, model routing, response helpers
├── api.rs         Mesh management API (:3131): status, events, discover, join, console HTML
├── tunnel.rs      TCP ↔ QUIC relay (RPC + HTTP), B2B rewrite map
├── rewrite.rs     REGISTER_PEER interception and endpoint rewriting
├── launch.rs      rpc-server and llama-server process management
├── console.html   Embedded dashboard with topology view and chat
├── download.rs    Model catalog and HuggingFace download (reqwest, resume support)
├── nostr.rs       Nostr publish/discover: mesh listings, smart auto-join, publish watchdog
```

## Design

**mesh-llm owns :9337** — never llama-server directly. The API proxy peeks at the `model` field and routes to the right node.

**One model per node** — multi-model = different nodes serving different things. No VRAM double-commitment.

**Solo by default** — if a model fits (VRAM ≥ size × 1.1), it runs solo. Tensor split only when necessary.

**Latency-aware splitting** — when splitting is needed, peers sorted by RTT ascending. Take just enough for the VRAM shortfall. 80ms hard cap — high-latency nodes participate as API clients, not split partners.

**Demand-aware rebalancing** — request rates tracked per model, gossipped every 60s. Standby nodes promote when a model is hot (≥10 req/min, ≥3x imbalance vs coldest) or has zero servers.

**Event-driven** — death detected via tunnel failure + 60s heartbeat. Dead peers broadcast, not re-added by gossip. Cost proportional to topology changes, not node count.

**Passive scaling** — clients don't gossip, use routing tables only. Zero per-client state on servers.

## Nostr discovery

Opt-in. Without `--publish`, nothing touches Nostr.

```bash
# Publish
mesh-llm --model Qwen2.5-3B --publish --mesh-name "Sydney Lab" --region AU

# Discover
mesh-llm discover
mesh-llm discover --model GLM --region AU

# Auto-join (discover + join + serve)
mesh-llm --auto
```

Smart auto-join scores meshes by: region match, node count, model overlap, VRAM, overload. QUIC health probe before committing. Publish watchdog auto-takes-over if the original publisher dies.
