# Passive Mode: Scalable Mesh Architecture

## The Problem
Current architecture: every node (GPU and client) is a full gossip peer. O(n²) gossip,
health checks on every peer every 15s, full peer announcements. Doesn't scale past ~20 nodes.

## Design: Two Modes

### Active
- Running llama-server (or rpc-server for tensor split)
- Full gossip with other active nodes
- Health checks, election, tunnel management
- Small group: 2-10 nodes typically

### Passive
- Gets routing table from any active node periodically (every 30s)
- Routes requests by tunneling to active hosts
- No gossip, no health check, no election
- Can be hundreds of nodes
- Two flavors:
  - **`--client`**: pure consumer, never promotes. No GPU.
  - **Standby GPU**: has VRAM + models on disk. Can self-promote to active when needed.

### Routing Table
Simple payload from any active node:
```json
{
  "hosts": [
    {"model": "Qwen2.5-3B-Instruct-Q4_K_M", "node_id": "ce4362...", "vram_gb": 103},
    {"model": "GLM-4.7-Flash-Q4_K_M", "node_id": "8cfe80...", "vram_gb": 52}
  ]
}
```
Passive node picks host by `hash(my_id) % hosts_for_model.len()` — consistent, globally distributed.

### Check-in Flow (passive nodes)
Every 30s:
1. Open a lightweight QUIC stream to any known active node
2. Send: "give me routing table" (+ my VRAM, models on disk if standby GPU)
3. Receive: current routing table
4. If active node unreachable, try next one from last known table

### Promotion (standby → active)
During check-in, standby node sees:
- A model it has on disk with no host → promote, serve it
- A model it has on disk with overloaded hosts → promote (future: load metrics)
- Requested model it has on disk → promote

Promotion = start rpc-server + llama-server, join full gossip, become active.

### Bootstrap Resilience
- Passive node initially connects to bootstrap (from `--join` token)
- Routing table lists all active hosts — these are fallback check-in targets
- If bootstrap dies, check in with any other active host
- If ALL active hosts die, standby GPUs self-promote (nobody responded → I'm needed)
- `--client` nodes: if all hosts gone, nothing to do (no GPU = no inference)

## Implementation Plan

### Phase 1: Lightweight client (no gossip)
- New QUIC protocol: ALPN `mesh-llm-route/0` (separate from gossip)
- Active nodes handle route requests: return routing table
- `--client` uses this instead of full gossip
- Client keeps list of known active nodes, refreshes every 30s
- Random host selection: `hash(client_id + model) % hosts.len()`

### Phase 2: Unified passive mode
- GPU nodes joining without `--model` start passive (not idle GPU hack)
- Same networking as client: get routing table, proxy requests
- But also report VRAM + available models during check-in

### Phase 3: Self-promotion
- Standby checks routing table: any unserved models I can handle?
- If yes: start rpc-server, llama-server, join gossip, become active
- Demotion: if idle for N minutes and other hosts exist, go back to passive

### Phase 4: Don't download what won't fit
- Before downloading a model, check if node VRAM >= model_size * 1.1
- If not, skip (don't waste bandwidth on a model we can't serve solo)
- Exception: if joining a tensor split group, download anyway (host coordinates)
