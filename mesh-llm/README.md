# mesh-llm

Rust sidecar for distributed llama.cpp inference over QUIC. See the [project README](../README.md) for usage.

See [DESIGN.md](DESIGN.md) for internals — stream types, B2B rewriting, tunnel manager, etc.
See [MULTI-MODEL.md](MULTI-MODEL.md) for multi-model serving design and status.

```
src/
├── main.rs        CLI, startup, API proxy (owns :9337), model routing
├── mesh.rs        iroh QUIC endpoint, gossip, peer management, health check
├── election.rs    Per-model host election, solo/split mode, llama-server lifecycle
├── tunnel.rs      TCP ↔ QUIC relay (RPC + HTTP), B2B rewrite map
├── rewrite.rs     REGISTER_PEER interception and endpoint rewriting
├── launch.rs      rpc-server and llama-server process management
├── console.rs     Web console: status, model list, chat proxy (--console flag)
├── console.html   Embedded dashboard with model picker and topology view
└── download.rs    Model catalog and HuggingFace download
```

## Key design

- **mesh-llm owns the API port** (:9337) — never llama-server directly
- **Model-aware routing** — API proxy peeks at request body, routes by `model` field to the right node
- **One model per node** — each node loads exactly one model. Multi-model = different nodes serving different things
- **No accidental split** — if a model fits on one node, it runs solo. Tensor split only when the model doesn't fit or `--split` is forced
- **Every mesh change = re-evaluate** — but skip restart if election result unchanged (no gossip storms)
- **Health check** — periodic gossip probes detect dead peers in ~15s
- **Ephemeral client keys** — `--client` gets a unique identity, works alongside GPU nodes on the same machine
