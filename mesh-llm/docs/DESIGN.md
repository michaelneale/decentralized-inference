# mesh-llm Design

A Rust sidecar that turns llama.cpp RPC into a peer-to-peer mesh. Nodes find
each other over QUIC (via [iroh](https://iroh.computer)), form a mesh of
tunnels, and llama.cpp runs unmodified on top — rpc-server and llama-server
just see local TCP sockets.

## Architecture

```
src/
├── main.rs        CLI, orchestration, startup flows (auto, idle, passive)
├── mesh.rs        QUIC endpoint, gossip, peer management, mesh identity, request rates
├── election.rs    Per-model host election, latency-aware tensor split, llama-server lifecycle
├── proxy.rs       HTTP proxy plumbing: request parsing, model routing, response helpers
├── api.rs         Mesh management API (:3131): status, events, discover, join, console HTML
├── tunnel.rs      TCP ↔ QUIC relay (RPC + HTTP), B2B rewrite map
├── rewrite.rs     REGISTER_PEER interception and endpoint rewriting
├── launch.rs      rpc-server and llama-server process management
├── console.html   Embedded dashboard with topology view and chat
├── download.rs    Model catalog and HuggingFace download (reqwest, resume support)
├── nostr.rs       Nostr publish/discover: mesh listings, smart auto-join, publish watchdog
```

## Node Roles

```rust
enum NodeRole {
    Worker,                      // rpc-server, provides GPU compute
    Host { http_port: u16 },     // llama-server + rpc-server, serves HTTP API
    Client,                      // no compute, just API access via tunnel
}
```

Roles exchanged via gossip. A node transitions Worker → Host when elected.

## QUIC Stream Types

Single QUIC connection per peer, multiplexed by 1-byte prefix:

| Byte | Type | Purpose |
|------|------|---------|
| 0x01 | GOSSIP | Peer announcements (role, serving, VRAM, models, demand, mesh_id) |
| 0x02 | TUNNEL_RPC | TCP relay to remote rpc-server |
| 0x03 | TUNNEL_MAP | B2B tunnel port map exchange |
| 0x04 | TUNNEL_HTTP | TCP relay to remote llama-server HTTP |
| 0x05 | ROUTE_REQUEST | Routing table for passive nodes (hosts + models) |
| 0x06 | PEER_DOWN | Death broadcast (immediate, from any node that detects a death) |
| 0x07 | PEER_LEAVING | Clean shutdown broadcast (ctrl-c) |

## Multi-Model

Different nodes serve different models. The API proxy on each node peeks at
the `model` field in POST bodies and routes to the correct host via QUIC tunnel.

- **One model per node** — no VRAM double-commitment
- **Solo by default** — if VRAM ≥ model_size × 1.1, run solo
- **Per-model election groups** — nodes serving the same model elect a host independently
- **Auto-assignment** — joiners without `--model` get assigned based on mesh needs and what's on disk

## Mesh Identity

Every mesh has a stable `mesh_id`:
- **Named mesh**: `hash(name + originator_nostr_pubkey)` — deterministic, unique per creator
- **Unnamed mesh**: random UUID, persisted to `~/.mesh-llm/mesh-id`

Propagated via gossip (`PeerAnnouncement.mesh_id`) and routing table (`RoutingTable.mesh_id`).
Published in Nostr listings (`MeshListing.mesh_id`).
Saved to `~/.mesh-llm/last-mesh` on successful join for sticky preference scoring.

## Bootstrap Proxy

When joining an existing mesh, a tunnel-only API proxy starts immediately on the
local port — before rpc-server or llama-server are ready. Requests are tunneled to
mesh hosts via QUIC. When the real `api_proxy` is ready, it takes over the listener.

This gives instant API access (within seconds of `mesh-llm --join`) while the local
GPU loads its model in the background.

## Passive Mode

Two flavors, one code path (`run_passive()`):
- **`--client`**: pure consumer, ephemeral key, no gossip, routing table only
- **Standby GPU**: has VRAM + models on disk, watches for topology changes, promotes when needed

Passive nodes get routing tables via `STREAM_ROUTE_REQUEST` (0x05), not full gossip.
Scales to hundreds of clients without O(n²) gossip cost.

## Demand-Aware Rebalancing

- `record_request(model)` increments per-model counter on every API proxy request
- `snapshot_request_rates()` computes delta each gossip cycle (requests/min)
- Rates gossipped in `PeerAnnouncement.request_rates`
- Standby nodes check on 60s timer + topology changes via `tokio::select!`
- Promotion triggers: (1) model with 0 servers, (2) ≥3x demand imbalance + ≥10 req/min, (3) single hot model ≥10 req/min

## Latency-Aware Tensor Split

When a model requires splitting across nodes:
1. Filter candidates by `rtt_ms < 80ms`
2. Sort by RTT ascending (unknown RTT sorts last)
3. Greedily accumulate VRAM until `≥ model_size × 1.1`
4. Stop — don't add unnecessary high-latency peers

## Event-Driven Peer Management

- **60s heartbeat** with 2-consecutive-failure threshold
- **Death broadcasts** (`STREAM_PEER_DOWN`) for immediate notification
- **Clean shutdown** (`STREAM_PEER_LEAVING`) on ctrl-c
- **Dead peers set** prevents gossip from re-adding killed nodes
- **Tunnel failure detection** triggers immediate death broadcast

## B2B Direct Transfer

When the model is split across workers, activation tensors flow directly
between workers (1 hop) instead of through the host (2 hops):
1. Each node broadcasts `{EndpointId → tunnel_port}` via `STREAM_TUNNEL_MAP`
2. `rewrite.rs` intercepts `REGISTER_PEER` and rewrites ports for local tunnels
3. llama.cpp's `PUSH_TENSOR_TO_PEER` goes directly between workers

## Management API (port 3131)

Separate from the inference API (port 9337). Serves mesh management endpoints
and an optional HTML console.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/status` | GET | Live mesh state (JSON): node, peers, models, targets |
| `/api/events` | GET | SSE stream of status updates (2s interval + on change) |
| `/api/discover` | GET | Browse Nostr-published meshes |
| `/api/join` | POST | Join a mesh by invite token `{"token":"..."}` |
| `/api/chat` | POST | Proxy to inference API (`/v1/chat/completions`) |
| `/` | GET | Console HTML dashboard |

The console HTML is a thin client — everything it shows comes from `/api/status`
and `/api/events`. Mesh management works without the HTML via curl/scripts.

Enabled by default. `--no-console` disables (overridden in idle mode where the
management API is required for discover/join).

## Idle Mode

`mesh-llm` with no arguments starts in idle mode:
1. Starts node + management API (port 3131)
2. Inference port (9337) returns 503 until joined
3. User browses meshes via console or `/api/discover`
4. `/api/join` triggers: connect → gossip → assign model → download if needed → serve

All join paths converge: `--auto`, `--join TOKEN`, and idle→console join end up
in the same connect → assign → serve flow.

## Nostr Discovery

Opt-in mesh advertisement via Nostr relays (NIP-89, kind 31990):
- `--publish`: republish listing every 60s (TTL 120s)
- `--auto`: discover meshes, score them, health-probe, join best
- Publish watchdog: if publisher dies, another node takes over
- `score_mesh()`: region match (+200), capacity, node count, VRAM, sticky preference (+500)
- `smart_auto()`: picks best mesh or recommends starting new one with models for your VRAM
