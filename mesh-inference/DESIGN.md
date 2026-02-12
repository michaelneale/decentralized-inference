# mesh-inference

A Rust sidecar that turns llama.cpp RPC into a peer-to-peer mesh. Nodes find
each other over QUIC (via [iroh](https://iroh.computer)), form a full mesh of
tunnels, and llama.cpp runs unmodified on top — rpc-server and llama-server
just see local TCP sockets.

## What it does

```
┌──────────────────────────────────────────────────────────────────────┐
│  Any machine                                                         │
│                                                                      │
│  ┌────────────┐         ┌──────────────────────────────────────┐     │
│  │ rpc-server │◄──TCP──▶│          mesh-inference              │     │
│  │ (localhost) │         │                                      │     │
│  └────────────┘         │  - iroh QUIC endpoint (NodeId)       │     │
│                          │  - full mesh of peer connections     │     │
│  ┌──────────────┐        │  - local TCP tunnel port per peer   │     │
│  │ llama-server │◄──TCP──│  - REGISTER_PEER rewriting (B2B)   │     │
│  │ (optional)   │        │                                      │     │
│  └──────────────┘        └──────────────────────────────────────┘     │
│         │                          ▲   ▲   ▲                         │
│         ▼                          │   │   │  QUIC                   │
│  http://localhost:8080             │   │   │                         │
│  (OpenAI-compatible API)           ▼   ▼   ▼                         │
│                              other mesh-inference nodes               │
└──────────────────────────────────────────────────────────────────────┘
```

Every node runs the same binary. Every node offers its compute via a local
rpc-server. Any node can optionally also run llama-server to orchestrate
inference and expose an HTTP API.

## Concepts

**Node**: A machine running `mesh-inference`. Has an iroh identity (Ed25519
keypair → NodeId), a local rpc-server, and tunnel ports to every other node
in the mesh.

**Mesh**: The set of nodes that know about each other. Fully connected — every
node has a QUIC connection to every other node. Nodes join and leave
dynamically.

**Tunnel port**: A local TCP port on a node that transparently tunnels to
another node's rpc-server over QUIC. llama.cpp connects to these as if they
were local rpc-servers.

**Node ticket**: An iroh `NodeTicket` — encodes the node's public key, relay
URL, and direct addresses. Shared out-of-band to let a new node join.

## UX

### Worker node (offers compute, no inference server)

```bash
# First node — starts a new mesh
mesh-inference
# prints: Node ticket: <ticket_A>
# prints: Waiting for peers...

# Subsequent nodes — join via any existing member
mesh-inference --join <ticket_A>
# prints: Node ticket: <ticket_B>
# prints: Connected to peer A
```

### Inference node (orchestrates + serves HTTP)

```bash
mesh-inference --join <ticket_A> --serve 8080 --model ~/.models/big.gguf
# prints: Node ticket: <ticket_C>
# prints: Connected to peer A
# prints: Connected to peer B (learned via gossip from A)
# prints: llama-server listening on http://localhost:8080
```

Then use it:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}]}'
```

### Dynamic membership

```bash
# Check mesh status
mesh-inference status
# Mesh: 4 peers connected
#   alice   (node_abc...) connected
#   bob     (node_def...) connected
#   carol   (node_ghi...) connected
#   dave    (node_jkl...) connected
```

Nodes can join or leave at any time. The mesh adapts:
- New node joins → existing nodes learn about it via gossip, open new tunnel
  ports, new node becomes available for the next inference session
- Node leaves → tunnel dies, in-flight inference sessions using that node will
  error, mesh continues with remaining nodes

## How it works

### Joining the mesh

1. Node starts, creates an iroh endpoint (Ed25519 keypair, QUIC listener)
2. If `--join <ticket>` is provided, connects to that peer over QUIC
3. On connect, both sides exchange their full peer list
4. Node connects to any peers it didn't already know about
5. Those peers in turn learn about the new node and connect back
6. Within seconds, full mesh is established

Only one ticket is needed to join — you bootstrap off any single member and
discover everyone else through gossip.

### Tunnel setup

For each peer in the mesh, the sidecar:
1. Allocates a local TCP listen port
2. On inbound TCP connection (from local llama-server or rpc-server B2B),
   opens a QUIC stream to the peer's sidecar
3. Peer's sidecar connects the QUIC stream to its local rpc-server via TCP
4. Bidirectional byte relay: TCP ↔ QUIC ↔ TCP

The tunnel is a dumb byte pipe — no protocol parsing — except for one command
(see below).

### B2B direct transfer support

The [B2B fork](https://github.com/jh-block/llama.cpp/tree/rpc-backend-to-backend-comms)
enables direct worker-to-worker tensor transfer. The orchestrator tells each
worker about its peers via `RPC_CMD_REGISTER_PEER` (command byte 17), passing
an endpoint string like `127.0.0.1:60052`.

Problem: the orchestrator sends its own local tunnel port in the endpoint
string, but the rpc-server receiving this command is on a different machine
where that port means something else (or nothing).

Solution: the sidecar intercepts `RPC_CMD_REGISTER_PEER` on the inbound tunnel
(orchestrator → worker). It reads the `peer_id` from the message, looks up
which local tunnel port on **this** machine corresponds to that peer, and
rewrites the endpoint field before forwarding to the local rpc-server.

This is the only RPC command the sidecar understands. Everything else passes
through as raw bytes.

**Wire format of REGISTER_PEER:**
```
| cmd: 1 byte (17) | payload_size: 8 bytes LE | peer_id: 4 bytes LE | endpoint: 128 bytes |
```

The sidecar rewrites the 128-byte endpoint field. That's it.

### Data flow example (3 nodes, B2B)

```
Alice (orchestrator)       Bob (worker)               Carol (worker)
────────────────────       ────────────               ──────────────

llama-server               rpc-server                 rpc-server
  ↕ TCP                      ↕ TCP                      ↕ TCP
sidecar                    sidecar                     sidecar
  │                          │                           │
  │──── QUIC ───────────────▶│ (inference commands)      │
  │──── QUIC ────────────────────────────────────────────▶│ (inference commands)
  │                          │                           │
  │ REGISTER_PEER(carol) ──▶ │                           │
  │ (sidecar rewrites        │                           │
  │  endpoint to Bob's       │                           │
  │  local tunnel to Carol)  │                           │
  │                          │                           │
  │                          │──── QUIC ────────────────▶│ (B2B tensor push)
  │                          │  (Bob's sidecar tunnels   │
  │                          │   to Carol's rpc-server)  │
```

Activation tensors flow Bob → Carol directly. Alice is not in the data path.

## Architecture

```
mesh-inference/
├── Cargo.toml
├── DESIGN.md              ← this file
└── src/
    ├── main.rs            ← CLI parsing, startup orchestration
    ├── mesh.rs            ← iroh endpoint, peer gossip, membership tracking
    ├── tunnel.rs          ← TCP ↔ QUIC bidirectional relay
    ├── rewrite.rs         ← REGISTER_PEER interception and endpoint rewriting
    └── launch.rs          ← rpc-server and llama-server process management
```

### main.rs
CLI entry point. Parses args, starts the mesh, starts rpc-server, optionally
starts llama-server. Writes mesh status to stdout / status file.

### mesh.rs
Manages the iroh endpoint and peer connections:
- Creates iroh `Endpoint` with Ed25519 identity
- Accepts inbound QUIC connections
- Dials peers from tickets
- Gossip protocol: on new connection, exchange peer lists
- Tracks peer liveness (QUIC connection state)
- Notifies tunnel.rs when peers join/leave

### tunnel.rs
Manages TCP ↔ QUIC tunnels:
- Per peer: allocates a local TCP listen port
- On inbound TCP connection: opens a QUIC stream to peer, starts bidirectional
  relay with rewrite.rs in the pipeline
- On inbound QUIC stream: connects to local rpc-server TCP port, starts
  bidirectional relay
- Cleans up when peers disconnect

### rewrite.rs
Minimal RPC protocol awareness:
- Sits in the tunnel pipeline on the inbound side (orchestrator → local
  rpc-server)
- Reads the 1-byte command from each RPC message
- If not 17 (REGISTER_PEER): pass through verbatim
- If 17: read the 132-byte payload, extract peer_id (bytes 0-3 LE), look up
  the local tunnel port for that peer, write `127.0.0.1:<port>` into the
  128-byte endpoint field (bytes 4-131), forward

### launch.rs
Process management for llama.cpp binaries:
- Starts `rpc-server` on a localhost port, picks device automatically
- Optionally starts `llama-server` with:
  - `--rpc` pointing at all peer tunnel ports
  - `-m` with the user's model path
  - `--port` for the HTTP API
  - `-ngl 99` to offload everything
- Monitors processes, logs to files

## Build dependencies

- **iroh** (`0.96+`): QUIC transport, NodeId/tickets, connection management,
  relay fallback, hole punching
- **tokio**: async runtime (iroh requires it)
- **clap**: CLI parsing

No llama.cpp build dependency — mesh-inference launches the pre-built binaries
from the `llama.cpp/build/bin/` or `llama.cpp-rpc-b2b/build/bin/` directory.

## Future work (not in scope now)

- **Capacity metadata in gossip**: nodes announce free memory, GPU type, so
  the orchestrator can auto-compute tensor splits
- **Auto tensor split**: pick workers and splits based on available capacity
- **Session resilience**: detect worker failure, restart llama-server against
  remaining mesh
- **Multiple concurrent sessions**: multiple orchestrators using the same mesh
- **Auth**: signed peer messages, capability tokens
- **mDNS discovery**: auto-discover peers on LAN without exchanging tickets
