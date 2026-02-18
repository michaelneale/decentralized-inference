# Multi-model mesh design

## Overview

The mesh serves multiple models simultaneously. Different nodes load different models. Nodes serving the same model group together for tensor split and election. The mesh routes requests by model name via a single API endpoint. The degenerate case (one model) behaves identically to today.

## Key principles

- **No VRAM double-commitment.** Each node loads exactly one model at a time. VRAM is never split between models on a single node — it either goes entirely to one model (solo) or contributes layers to a tensor split for that model.
- **No llama-server multi-model.** Each node runs one rpc-server + one llama-server (if host), same as today. Multi-model comes from different nodes running different things, not from any single process serving multiple models.
- **Degenerate case is today's behavior.** One model in the mesh = exactly how it works now. Adding more models is additive.
- **Smart packing.** Nodes are assigned models based on what the mesh needs, what they already have on disk, and their VRAM. No wasted capacity.

## Node lifecycle

### First node (seeder)

Starts the mesh and declares what models it wants to serve. Could be one or many:

```bash
mesh-llm --model Qwen2.5-32B                              # one model, same as today
mesh-llm --model Qwen2.5-32B --model Qwen2.5-3B           # seed with two models
mesh-llm --models-dir ~/.models/                           # serve everything on disk
```

The node itself only loads one model (the one it's best suited for given its VRAM). The others go into the mesh catalog — available but waiting for capacity.

### Joining nodes

Join without specifying a model, same as today:

```bash
mesh-llm --join <token>
```

The node gossips its VRAM and what GGUFs it has in `~/.models/`. The mesh assigns it a model using the allocation heuristic (see below). The node downloads if needed, starts rpc-server, joins that model's election group.

A node can also explicitly request a model: `mesh-llm --model Qwen2.5-3B --join <token>`. This overrides the assignment — "I want to serve this specifically."

## Gossip changes

`PeerAnnouncement` grows:

```rust
struct PeerAnnouncement {
    addr: EndpointAddr,
    role: NodeRole,           // Worker { model } | Host { model, http_port } | Client
    vram_bytes: u64,
    available_models: Vec<String>,   // GGUFs on disk (catalog contribution)
    serving: Option<String>,         // model currently loaded in VRAM (None = not assigned yet)
    model_source: Option<String>,    // how to get the model (for auto-download)
}
```

The **mesh catalog** is the union of `available_models` from all peers, plus any models from `--model` flags that haven't been downloaded yet (they exist in the catalog as "downloadable"). A model is **servable** if enough VRAM exists across nodes that have it on disk (or are willing to download it).

## Mesh catalog

A new concept: the mesh maintains a catalog of models, derived from gossip:

```rust
struct MeshModel {
    name: String,                    // e.g. "Qwen2.5-32B-Instruct-Q4_K_M"
    source: String,                  // catalog name / HF URL for downloading
    file_size_bytes: u64,            // from GGUF metadata or filesystem
    nodes_on_disk: Vec<EndpointId>,  // nodes that have the file
    nodes_serving: Vec<EndpointId>,  // nodes currently loaded and serving
    host: Option<EndpointId>,        // elected host for this model (runs llama-server)
    status: ModelStatus,             // Unloaded | Loading | Ready | NeedsCapacity
}
```

The catalog is computed locally by each node from gossip state. No central authority — everyone derives the same view from the same information.

## Model allocation heuristic

When a node joins (or a new model is added), the mesh decides what model the node should serve. Priority order:

1. **Model that can't run without this node.** A model needs tensor split and doesn't have enough VRAM yet. This node's VRAM would put it over the threshold. Prefer models the node already has on disk (no download wait).

2. **Model with no nodes serving it.** A model is in the catalog but nobody is loaded. Assign it if the node has enough VRAM and has it on disk. Gets a new model online with zero download.

3. **Model that would benefit from more capacity.** A model is being served but tensor split could use more VRAM (faster split, more headroom). Node already has it on disk.

4. **Download and serve.** Same as above but the node needs to download first. Deprioritized because of download latency.

5. **Join the biggest model.** If everything is covered, join the largest model's group (most likely to benefit from extra tensor split capacity, and handles the degenerate single-model case).

Tie-breaking: prefer models already on disk > larger models > alphabetical.

## Per-model election

Today's election: one global host, highest VRAM wins.

New: election runs **per model group**. For each model, the highest-VRAM node *serving that model* becomes its host. That host runs llama-server with `--rpc` pointing at other nodes in the same model group.

```rust
// Pseudo-code
for model in mesh_catalog.models_with_serving_nodes() {
    let group: Vec<PeerInfo> = peers_serving(model);
    let host = group.max_by_key(|p| (p.vram_bytes, p.id));
    if i_am(host) {
        start_llama_server(model, rpc_ports_for(group));
    }
}
```

Each node only participates in one model group, so it only runs one rpc-server and at most one llama-server. The election loop iterates over model groups but each node only acts on its own group.

Re-election triggers remain the same: any peer join/leave causes all model groups to re-evaluate. A node leaving a model group might leave it short on VRAM, which could trigger reassignment of unassigned nodes.

## API routing

The API proxy on `:9337` becomes model-aware:

```rust
// Parse model from request body
let model_name = parse_model_from_request(&body);

// Look up target for this model
let target = model_targets.get(&model_name);

match target {
    Some(InferenceTarget::Local(port)) => proxy_to_local(port),
    Some(InferenceTarget::Remote(peer)) => proxy_via_quic(peer),
    None => return_503("model not available"),
}
```

`/v1/models` returns all models in the mesh catalog with their status (ready, loading, needs capacity).

Every node in the mesh can serve as the API entry point — workers proxy to the right host via QUIC, same as today but per-model. A client node proxies all requests through the mesh.

## Unloading and contention

When VRAM is tight or models are unused:

**Tracking usage:** Each model host tracks last-request time. This propagates via gossip (a timestamp or "last used N seconds ago" field). Nodes can see which models are hot and which are cold.

**When to unload:** A model becomes a candidate for unloading when:
- It hasn't received a request in N minutes (configurable, default 30?)
- Another model needs the VRAM (new model added to catalog, or existing model needs more split capacity)
- The node is explicitly told to switch (future: admin command)

**How unloading works:** The node kills its rpc-server + llama-server (if host), updates gossip to `serving: None`, and the allocation heuristic runs again. It might get assigned a different model, or stay idle until needed. From the model group's perspective, a node left — same as today when a peer disconnects. The group re-elects and adjusts tensor split.

**No preemption for v1.** A node doesn't get yanked from a model mid-request. Unloading only happens during idle periods. Usage-aware rebalancing is an optimisation — v1 can just use static assignments (whatever you were assigned at join time, you keep).

## What changes per file

- **mesh.rs**: `PeerAnnouncement` grows `available_models` (all GGUFs on disk) and `serving` (currently loaded model). New helper methods: `models_catalog()`, `peers_serving(model)`, `peer_for_model_host(model)`.
- **election.rs**: `election_loop` becomes `election_loop_for_model` or iterates model groups. `should_be_host` takes a model filter. `InferenceTarget` map becomes `HashMap<String, InferenceTarget>`.
- **main.rs**: `--model` becomes repeatable or `--models-dir` added. API proxy parses model name from request body. `/v1/models` endpoint returns mesh catalog. Model allocation heuristic on join.
- **launch.rs**: No changes — still starts one rpc-server and one llama-server per node.
- **tunnel.rs**: No changes — tunnels are per-peer, model-agnostic.
- **rewrite.rs**: No changes.

## Degenerate cases

- **One model, one node**: identical to today. Single model group, solo election, no split.
- **One model, multiple nodes**: identical to today. Single model group, tensor split.
- **Multiple models, all fit solo**: each node loads a different model, each is its own host. No tensor split. Mesh is just a routing layer.
- **Mix of big and small models**: big model gets tensor split across multiple nodes, small models each on one node. Natural packing.

## Open questions

- Should nodes be able to serve more than one model if they have enough VRAM? (Probably not for v1 — keep it simple, one model per node.)
- How does the console show multi-model state? (Model list with status, per-model cluster bar?)
- Should the seeder's `--model` list be a hard requirement ("mesh must serve these") or a suggestion ("serve these if capacity allows")?
- Speculative decoding: draft model is always local to the host. Does it count as a second model? (No — it's small and co-located, not a mesh concern.)
