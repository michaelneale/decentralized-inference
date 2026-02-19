# mesh-llm TODO

## Console: VRAM usage bar per node
Currently each node box shows a bar for VRAM proportion (node's share of total mesh VRAM).
Add a second bar showing how much of that node's VRAM is in use by the model it's serving.

**Plan:**
1. Add `size_gb: f64` to `MeshModelPayload` in `console.rs`
2. Get size from `~/.models/<name>.gguf` via `fs::metadata`. If not on our disk, use 0
3. In console HTML, match `node.serving` to `mesh_models` to get `size_gb`
4. Compute `usage_pct = model.size_gb / node.vram * 100`
5. Draw a second thinner bar or inner fill in a different color (e.g. brighter green/blue)

Data is already in gossip — no new network calls needed.

## Smart model assignment for publisher
When publisher runs `--model A --model B`, assignment is arbitrary (HashSet iteration order).
Should assign biggest model to biggest VRAM node. The publisher picks first (no peers yet)
so it should sort its own models by size descending and serve the largest one, leaving
smaller ones for joining nodes.

## Load balancing across multiple hosts for same model
When multiple nodes independently serve the same model, the proxy picks one deterministically.
Should round-robin or least-connections when multiple hosts available.

## `--models-dir` flag
Serve everything on disk automatically instead of listing models explicitly.

## Usage-aware rebalancing
Track per-model request rates, unload idle models, reassign nodes.

## P2P model transfer
Nodes serve GGUF chunks over QUIC to new joiners instead of downloading from HuggingFace.
