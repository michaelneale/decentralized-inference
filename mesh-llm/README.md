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
