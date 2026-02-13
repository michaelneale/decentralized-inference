# mesh-inference

A Rust sidecar that turns llama.cpp RPC into a peer-to-peer mesh. Nodes find
each other over QUIC (via [iroh](https://iroh.computer)), form a full mesh of
tunnels, and llama.cpp runs unmodified on top — rpc-server and llama-server
just see local TCP sockets.

## What it does

```
┌──────────────────────────────────────────────────────────────────────┐
│  Worker / Host node                                                  │
│                                                                      │
│  ┌────────────┐         ┌──────────────────────────────────────┐     │
│  │ rpc-server │◄──TCP──▶│          mesh-inference              │     │
│  │ (localhost) │         │                                      │     │
│  └────────────┘         │  - iroh QUIC endpoint (NodeId)       │     │
│                          │  - full mesh of peer connections     │     │
│  ┌──────────────┐        │  - local TCP tunnel port per peer   │     │
│  │ llama-server │◄──TCP──│  - REGISTER_PEER rewriting (B2B)   │     │
│  │ (host only)  │        │  - HTTP tunnel for lite clients     │     │
│  └──────────────┘        └──────────────────────────────────────┘     │
│         │                          ▲   ▲   ▲                         │
│         ▼                          │   │   │  QUIC                   │
│  http://localhost:8090             │   │   │                         │
│  (OpenAI-compatible API)           ▼   ▼   ▼                         │
│                              other mesh-inference nodes               │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  Lite client node                                                    │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐    │
│  │ localhost:8080    │    │          mesh-inference --client     │    │
│  │ (your apps talk  │◄──▶│                                      │    │
│  │  to this)        │    │  - discovers host via gossip         │    │
│  └──────────────────┘    │  - tunnels HTTP to host via QUIC    │    │
│                           └──────────────────────────────────────┘    │
│  No GPU / No model / No rpc-server / No llama.cpp binaries           │
└──────────────────────────────────────────────────────────────────────┘
```

## Node Roles

```rust
enum NodeRole {
    Worker,                      // rpc-server, provides GPU compute
    Host { http_port: u16 },     // llama-server + rpc-server, serves HTTP API
    Client,                      // no compute, just API access via tunnel
}
```

Roles are exchanged via gossip. Clients discover hosts automatically.

## QUIC Stream Types

All communication uses a single QUIC connection per peer, multiplexed by
a 1-byte stream type prefix:

| Byte | Type | Purpose |
|------|------|---------|
| 0x01 | GOSSIP | Peer list + role exchange |
| 0x02 | TUNNEL (RPC) | TCP relay to remote rpc-server |
| 0x03 | TUNNEL_MAP | B2B tunnel port map exchange |
| 0x04 | TUNNEL (HTTP) | TCP relay to remote llama-server HTTP |

## B2B Direct Transfer Support

When the orchestrator splits the model across multiple workers, activation
tensors must flow between workers at graph split boundaries. Without B2B,
these route through the orchestrator (2 network hops). With B2B, workers
push tensors directly to each other (1 hop, or 0 if on the same machine).

### How it works

1. Each node broadcasts its `{EndpointId → tunnel_port}` map to all peers
   via `STREAM_TUNNEL_MAP` (0x03).
2. The orchestrator's llama-server sends `REGISTER_PEER` commands through
   the mesh to tell each worker about its peers.
3. `rewrite.rs` intercepts these commands on the inbound tunnel path
   (QUIC→TCP to local rpc-server) and rewrites the endpoint port from the
   orchestrator's tunnel port to the worker's own local tunnel port for that
   peer.
4. When llama.cpp does a cross-backend `cpy_tensor`, the source worker
   pushes data directly to the destination worker via `PUSH_TENSOR_TO_PEER`.

### Data flow (3 nodes, B2B)

```
Alice (orchestrator)       Bob (worker)               Carol (worker)
────────────────────       ────────────               ──────────────

llama-server               rpc-server                 rpc-server
  ↕ TCP                      ↕ TCP                      ↕ TCP
sidecar                    sidecar                     sidecar
  │                          │                           │
  │──── QUIC ───────────────▶│ (inference commands)      │
  │──── QUIC ────────────────────────────────────────────▶│
  │                          │                           │
  │                          │──── QUIC ────────────────▶│ (B2B tensor push)
  │                          │  (direct, bypasses Alice) │
```

## Lite Client Mode

A lite client joins the mesh with no GPU, no model, and no llama.cpp binaries.
It discovers a host node via role-aware gossip, then accepts local TCP
connections and tunnels each through a QUIC bi-stream (type 0x04) to the host's
llama-server HTTP port.

The tunnel is a raw byte relay — no HTTP or SSE protocol awareness needed.
Chunked transfer encoding and SSE streaming work transparently because the
relay just moves bytes.

## Architecture

```
mesh-inference/
├── Cargo.toml
├── DESIGN.md              ← this file
├── README.md
└── src/
    ├── main.rs            ← CLI, startup, role branching (worker/host/client)
    ├── mesh.rs            ← iroh endpoint, gossip with roles, tunnel map exchange
    ├── tunnel.rs          ← TCP ↔ QUIC relay (RPC + HTTP), B2B rewrite map
    ├── rewrite.rs         ← REGISTER_PEER interception and endpoint rewriting
    └── launch.rs          ← rpc-server and llama-server process management
```

### main.rs
CLI entry point. Determines role from flags. Workers and hosts start
rpc-server + mesh + tunnel manager. Hosts also launch llama-server and
coordinate B2B tunnel map exchange. Clients skip all compute setup and
run a simple TCP listener → QUIC tunnel → host loop.

### mesh.rs
Manages the iroh endpoint and peer connections:
- Gossip protocol exchanges `PeerAnnouncement` (addr + role)
- Tracks peer roles for host discovery
- Multiplexes 4 stream types on a single QUIC connection
- Tunnel map broadcast/receive for B2B coordination

### tunnel.rs
Manages TCP ↔ QUIC tunnels:
- Per peer: allocates a local TCP listen port (outbound RPC tunnel)
- Inbound RPC streams: relayed to local rpc-server with REGISTER_PEER
  rewriting via `rewrite.rs`
- Inbound HTTP streams: relayed to local llama-server (plain byte relay)
- Port rewrite map updated from received tunnel maps

### rewrite.rs
Intercepts `RPC_CMD_REGISTER_PEER` (command byte 18) in the QUIC→TCP path:
- Parses port from the endpoint string (e.g. `127.0.0.1:49502`)
- Maps orchestrator port → local tunnel port via the rewrite map
- Rewrites the 128-byte endpoint field
- All other RPC commands stream through verbatim

### launch.rs
Process management for llama.cpp binaries:
- Starts rpc-server with `--gguf` for zero-transfer loading
- Starts llama-server with `--rpc`, `--no-mmap`, `-fit off`
- Health check polling, log file management

## Build Dependencies

- **iroh** (0.96): QUIC transport, endpoint identity, relay
- **tokio**: async runtime
- **clap**: CLI parsing
- **serde/serde_json**: gossip serialization
- **hex/base64**: key and token encoding
