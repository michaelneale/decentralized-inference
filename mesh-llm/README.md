# mesh-llm

Rust sidecar for distributed llama.cpp inference over QUIC. See the [project README](../README.md) for usage.

Docs:
- [docs/DESIGN.md](docs/DESIGN.md) — internals: stream types, B2B rewriting, tunnel manager
- [docs/MULTI-MODEL.md](docs/MULTI-MODEL.md) — multi-model serving: routing, election groups, gossip
- [docs/TESTING.md](docs/TESTING.md) — test scenarios and permutations

```
src/
├── main.rs        CLI, startup, API proxy (owns :9337), model routing
├── mesh.rs        iroh QUIC endpoint, gossip, peer management, routing table
├── election.rs    Per-model host election, solo/split mode, llama-server lifecycle
├── tunnel.rs      TCP ↔ QUIC relay (RPC + HTTP), B2B rewrite map
├── rewrite.rs     REGISTER_PEER interception and endpoint rewriting
├── launch.rs      rpc-server and llama-server process management
├── console.rs     Web console: status, model list, chat proxy (--console flag)
├── console.html   Embedded dashboard with model picker and topology view
├── download.rs    Model catalog and HuggingFace download
└── nostr.rs       Nostr publish/discover: mesh listings on public relays
```

## Key design

- **mesh-llm owns the API port** (:9337) — never llama-server directly
- **Model-aware routing** — API proxy peeks at request body, routes by `model` field to the right node
- **One model per node** — each node loads exactly one model. Multi-model = different nodes serving different things
- **No accidental split** — if a model fits on one node, it runs solo. Tensor split only when the model doesn't fit or `--split` is forced
- **Every mesh change = re-evaluate** — but skip restart if election result unchanged (no gossip storms)
- **Event-driven mesh** — death detected on use (tunnel failure) + 60s heartbeat fallback. Dead peers broadcast to mesh, not re-added by gossip. Scales better than aggressive polling.
- **Rejoin loop** — reconnects to bootstrap token every 60s if connection drops
- **Ephemeral client keys** — `--client` gets a unique identity, works alongside GPU nodes on the same machine
- **Reactive rebalancing** — standby nodes auto-promote when a model loses its last host
- **Passive scaling** — clients/standby nodes don't gossip, use routing table only, zero per-client server state

## Nostr discovery

Meshes can be published to Nostr relays so anyone can find and join them — no out-of-band token exchange needed.

### Publishing

```bash
# Publish your mesh with a name and region
mesh-llm --model Qwen2.5-3B --publish --mesh-name "Sydney Lab" --region AU

# With a client cap (delists when full, re-publishes when clients drop)
mesh-llm --model Qwen2.5-3B --publish --mesh-name "Sydney Lab" --max-clients 5
```

The listing includes: invite token, served models, wanted models, total VRAM, node count, name, region. Refreshes every 60s. Uses a persistent Nostr key stored in `~/.mesh-llm/nostr.nsec`.

Names are just display text — no collision risk. Each publisher has one listing (Nostr replaceable events, keyed by pubkey).

### Discovering

```bash
# List available meshes
mesh-llm discover

# Filter by model, region, or minimum VRAM
mesh-llm discover --model GLM --region AU --min-vram 50

# Auto-join the best match
mesh-llm discover --auto
```

### Auto-join (shorthand)

```bash
# Discover and join in one command
mesh-llm --auto

# With a model to serve
mesh-llm --auto --model Qwen2.5-3B

# As a client (no GPU needed)
mesh-llm --auto --client
```

`--auto` is equivalent to `mesh-llm --join $(mesh-llm discover --auto)` — discovers the best mesh via Nostr and joins it.
