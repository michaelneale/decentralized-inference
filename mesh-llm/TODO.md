# mesh-llm TODO

## First-Time Experience (fast `--auto`)
- [x] **Mesh identity**: Stable `mesh_id`, gossipped, in Nostr listings. Named: `hash(name+pubkey)`, unnamed: UUID.
- [x] **Sticky mesh preference**: `~/.mesh-llm/last-mesh` → +500 scoring bonus on `--auto`.
- [x] **API proxy during GPU bootstrap**: Tunnel-only proxy on `:9337` while GPU loads. Hands off to full proxy when ready.
- [x] **Idle mode**: `mesh-llm` with no args → read-only console + getting started instructions. Dormant QUIC. Use CLI to start/join.
- [ ] **Uptime signal**: Add `started_at: u64` to `MeshListing`. Score bonus for meshes that have been running longer — a 24h mesh beats a 10-minute test.
- [ ] **Solo fallback — fast starter model**: When `--auto` finds no mesh, download a small starter model first (Qwen2.5-3B, 2GB, ~1 min), start serving it immediately, then background-download the "real" model for the node's VRAM tier. User is chatting in <2 minutes.
- [ ] **Score mesh by model quality**: `smart_auto` should weight model quality — a mesh serving Qwen3-32B scores higher than one serving Qwen2.5-3B, all else equal. Use `MODEL_TIERS` VRAM requirements as a proxy for quality.

## Model Catalog Curation
- [ ] **Opinionated model tiers**: Curate recommended instruct models per VRAM tier. Current `MODEL_TIERS` and `MODEL_CATALOG` are ad-hoc — need a principled "if you have X GB, run Y" recommendation that considers quality, speed, and family diversity.
- [ ] **Draft model completeness**: Ensure every recommended main model has a draft pairing. Currently GLM-4.7 and DeepSeek have no draft.
- [ ] **Model quality metadata**: Add quality/benchmark scores to catalog entries so scoring can prefer better models, not just bigger ones.
- [ ] **Auto-upgrade path**: When a node is solo-serving a starter model and finishes downloading a better one, gracefully switch (stop llama-server, restart with new model). No impact to other mesh nodes.

## Bugs to Investigate
- [x] **Draft model leaking into served models**: Qwen2.5-0.5B-Instruct showing in `/v1/models`. Investigated — external node (Canada) explicitly listed it via `--model`. Not a code bug, deliberate choice by that node.
- [x] **Hermes disappearing during Mini WiFi flap**: Investigated. Two issues found and fixed:
  1. Strike 1 added peer to `dead_peers` which blocked incoming gossip — too aggressive. Fixed: only add to `dead_peers` on confirmed death (2 strikes).
  2. Reconnect path didn't trigger gossip — peer reconnected but sat invisible for up to 60s until next heartbeat. Fixed: immediately initiate gossip exchange on reconnect of previously-dead peer.

## Experiments
- [ ] **SOTA split: Qwen3.5-397B-A17B across 128GB M4 Max + second machine**: [Unsloth GGUF quants](https://unsloth.ai/docs/models/qwen3.5) — 4-bit (Q4_K_XL) is ~219GB, fits across 128GB + 64GB with tensor split. MoE model (397B total, 17B active) so should be fast despite size. Try 2-bit (~149GB) for single-machine fit on 128GB.
- [ ] **SOTA split: Qwen3.5-122B-A10B**: Smaller MoE, 4-bit should fit on 128GB solo. Good baseline before attempting 397B.
- [ ] **SOTA dense: try largest dense models that need 2+ machines**: Llama-3.3-70B, Qwen2.5-72B — already have 72B on disk. Benchmark split performance at scale.

## MoE Expert Sharding
See [MoE_PLAN.md](../MoE_PLAN.md) for full plan. Distribute MoE experts across mesh nodes with masked expert groups — non-all-to-all routing for interactive chat.
- [x] **Phase 1a: routing analysis tool** (`llama-moe-analyze`): observe MoE router decisions, measure group capture ratios. Initial results on Qwen3-30B-A3B (128 experts, top-8): best-group captures **99.3%** of unrestricted top-8 mass even with 8 groups (16 experts/group). Very promising.
- [x] **Phase 1b: expert masking in llama.cpp**: `llama_model_set_expert_mask()` API + logprob comparison. Best group (of 4) loses only -0.1 logprob; worst loses -2.3. Confirms probe-based placement is critical — some groups are nearly lossless while others degrade heavily.
- [ ] **Phase 2: per-node GGUF packaging**: tooling to split safetensors checkpoint into trunk + expert group bundles, convert to GGUF per node
- [ ] **Phase 3: mesh integration**: session placement with probe-and-pin, masked routing per node

## Nice to Have
- [ ] Don't download what won't fit: check VRAM before downloading via `--model`
- [x] Request rates in `/api/status` JSON for external tooling (demand map: `request_count`, `last_active_secs_ago` per model)
- [ ] `mesh-llm recommend`: CLI subcommand to suggest models for your hardware
- [ ] **Revisit `--publish` flag experience**: Bare `--publish` without `--mesh-name` is vestigial — publishes an unnamed mesh to Nostr that's hard to discover/filter. Consider: require `--mesh-name` with `--publish`, or auto-generate a name, or just document that `--mesh-name` is the intended way.

## Future
- [ ] **Public named meshes**: `--mesh-name "cool-mesh" --publish` currently gets -200 penalty for random `--auto` users (treated as private group). If someone explicitly passes both `--mesh-name` and `--publish`, add a `public: true` field to the Nostr listing so it scores like an unnamed mesh (no penalty). Lets people give their mesh a fun name without hiding it from discovery.
- [x] Demand-based Nostr listings: `wanted` list now derived from unified demand map (active demand, not served)
- [ ] Multi-node tensor split recovery: if one split peer dies, re-split across remaining
