# mesh-inference

Rust sidecar for distributed llama.cpp inference over QUIC. See the [project README](../README.md) for usage.

See [DESIGN.md](DESIGN.md) for internals — stream types, B2B rewriting, tunnel manager, etc.

```
src/
├── main.rs        CLI, startup, API proxy (owns :9337)
├── mesh.rs        iroh QUIC endpoint, gossip, peer management, STUN
├── election.rs    Auto host election by VRAM, llama-server lifecycle
├── tunnel.rs      TCP ↔ QUIC relay (RPC + HTTP), B2B rewrite map
├── rewrite.rs     REGISTER_PEER interception and endpoint rewriting
├── launch.rs      rpc-server and llama-server process management
├── console.rs     Web console: status viewer over HTTP/SSE (--console flag)
└── console.html   Embedded single-page dashboard
```

## Key design

- **mesh-inference owns the API port** (:9337) — never llama-server directly
- **llama-server always uses --rpc** — even solo, the host's own rpc-server is in the list
- **Ephemeral ports for llama-server** — no port conflicts on restart
- **Every mesh change = kill + re-elect + fresh start** — no special restart logic
- **API proxy routes dynamically** — host proxies to local llama-server, worker proxies via QUIC to host
