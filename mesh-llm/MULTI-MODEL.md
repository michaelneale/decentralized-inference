# Multi-model mesh

## Status: implemented on `multi-model` branch

The mesh serves multiple models simultaneously. Different nodes load different models. The API proxy routes requests by model name. A single `localhost:9337` endpoint serves all models.

## How it works

### Starting a mesh

The first node seeds the mesh with one or more model names:

```bash
# Serve one model (backward compatible, identical to before)
mesh-llm --model Qwen2.5-32B

# Seed two models — this node serves the first, the second waits for capacity
mesh-llm --model Qwen2.5-32B --model Qwen2.5-3B
```

### Joining nodes

Nodes can join with a specific model or let the mesh assign one:

```bash
# Explicit: "I want to serve this model"
mesh-llm --model GLM-4.7-Flash --join <token>

# Auto-assign: mesh picks based on what's needed and what's on disk
mesh-llm --join <token>
```

Auto-assignment priority:
1. Unserved model already on disk (no download, fills a gap)
2. Unserved model, needs download (fills a gap)
3. Least-served model on disk (add capacity)
4. Least-served model overall

### Lite clients

Clients use an ephemeral key (unique identity), so they work even on the same machine as a GPU node:

```bash
mesh-llm --client --join <token> --port 9555
```

The client sees all models via gossip and routes to any host.

## API

Standard OpenAI-compatible API on every node's port:

```bash
# List all served models
curl http://localhost:9337/v1/models

# Chat with a specific model — routed to the right node automatically
curl http://localhost:9337/v1/chat/completions \
  -d '{"model": "Qwen2.5-3B-Instruct-Q4_K_M", "messages": [...]}'
```

If the requested model is on this node, it's served locally. If it's on another node, the request is tunneled via QUIC to that node's llama-server.

### Drop a model

```bash
mesh-llm drop <model-name>
```

Sends a control request to the running instance. The node clears its serving state, kills llama-server, and exits. Other nodes are unaffected.

## Architecture

### No accidental tensor split

Each node runs its own llama-server independently (solo mode) as long as the model fits in its VRAM. Two nodes both serving Qwen2.5-3B = two independent llama-servers, not a tensor split.

Tensor split only happens when:
- The model genuinely doesn't fit on any single node, OR
- `--split` is explicitly passed

### Per-model election groups

Nodes serving the same model form an election group. Within a group, the highest-VRAM node becomes host, others become RPC workers (only relevant in split mode).

Different model groups are independent — their elections don't interfere.

### Gossip

`PeerAnnouncement` carries:
- `serving: Option<String>` — model currently loaded
- `available_models: Vec<String>` — GGUFs on disk (>500MB, scanned from `~/.models/`)
- `requested_models: Vec<String>` — from `--model` flags (the mesh catalog seed)
- `role: NodeRole` — Worker / Host { http_port } / Client

The **mesh catalog** is the union of all nodes' `available_models` and `requested_models`.

State changes (becoming host, setting serving) trigger a regossip to all peers so they learn immediately rather than waiting for the next connection event.

### Health check

Every 15s, each node probes all peers via gossip with a 5s timeout. Dead peers are removed immediately rather than waiting for QUIC idle timeout (30s). This ensures:
- Dead models go cold within ~15s
- Rejoining nodes are discovered cleanly
- The console and `/v1/models` stay accurate

### Routing

The API proxy on each node peeks at the HTTP request body to extract the `model` field, then routes:
- **Local model** → forward to local llama-server
- **Remote model** → QUIC tunnel to the host for that model
- **Unknown model** → fall back to first available target

## Console

The web console (`--console <port>`) shows:
- **Models list** — all mesh models with warm (green) / cold (gray) status and node count
- **Model picker** — dropdown in chat header to select which model to talk to
- **Node highlight** — switching models highlights the node box serving it
- **Total mesh VRAM** — aggregate across all GPU nodes
- **Per-node model name** — each node box shows what it's serving

## What's not implemented

- **Load balancing across multiple hosts for the same model** — if two nodes both serve Qwen2.5-3B independently, the proxy picks one deterministically. No round-robin.
- **`--models-dir`** — serve everything on disk automatically. Currently you list models explicitly with `--model`.
- **Usage-aware rebalancing** — no automatic unloading of idle models or reassignment based on traffic.
- **Admin/trust gating on drop** — anyone on localhost can drop a model.
