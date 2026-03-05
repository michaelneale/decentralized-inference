# mesh-llm TODO

## First-Time Experience
- [ ] **Uptime signal**: Add `started_at: u64` to `MeshListing`. Score bonus for longer-running meshes.
- [ ] **Solo fallback — fast starter model**: When `--auto` finds no mesh, download a small starter model first (Qwen2.5-3B, 2GB, ~1 min), start serving immediately, then background-download a better model for the node's VRAM tier.
- [ ] **Score mesh by model quality**: `smart_auto` should weight model quality — use `MODEL_TIERS` VRAM requirements as a proxy.

## Model Catalog
- [ ] **Opinionated model tiers**: Principled "if you have X GB, run Y" recommendation per VRAM tier.
- [ ] **Draft model completeness**: GLM-4.7 and DeepSeek have no draft pairing.
- [ ] **Model quality metadata**: Benchmark scores in catalog entries for scoring.
- [ ] **Auto-upgrade path**: Solo node finishes downloading a better model → gracefully switch.
- [ ] **Don't download what won't fit**: Check VRAM before downloading via `--model`.
- [ ] `mesh-llm recommend`: CLI subcommand to suggest models for your hardware.

## MoE Expert Sharding

Design: [MoE_PLAN.md](../MoE_PLAN.md) · Auto-deploy: [MoE_DEPLOY_DESIGN.md](../MoE_DEPLOY_DESIGN.md) · Validation: [MoE_SPLIT_REPORT.md](../MoE_SPLIT_REPORT.md)

- [x] Phase 1a: routing analysis tool (`llama-moe-analyze`)
- [x] Phase 1b: expert masking in llama.cpp (`llama_model_set_expert_mask()`)
- [x] Phase 2: per-node GGUF packaging (`llama-moe-split`)
- [x] Phase 3: mesh integration — auto-detect, split, session-sticky routing. Tested OLMoE-1B-7B over WAN.
- [ ] **Phase 4: optimized rankings** — run `moe-analyze` lazily for unknown MoE models, cache rankings. Current fallback uses conservative 50% shared core.
- [ ] **Phase 5: scale testing** — Mixtral 8×22B (~80GB), Qwen3-235B-A22B (~130GB) — models that actually need distribution.

## Resilience
- [x] Nostr re-discovery on peer loss (v0.26.1): `--auto` nodes re-discover after 90s with 0 peers.
- [ ] **Demand-based model upgrade**: Large-VRAM host serving a small model should upgrade when demand exists for a bigger model nobody is serving.
- [ ] **Multi-node tensor split recovery**: If one split peer dies, re-split across remaining.

## Discovery & Publishing
- [ ] **Revisit `--publish` flag**: Bare `--publish` without `--mesh-name` is vestigial. Consider requiring `--mesh-name` or auto-generating a name.
- [ ] **Public named meshes**: `--mesh-name "cool-mesh" --publish` gets -200 penalty. Add `public: true` field so named meshes aren't penalized in discovery.

## Experiments
- [ ] Qwen3.5-397B-A17B across 128GB M4 Max + second machine (MoE, ~219GB Q4)
- [ ] Qwen3.5-122B-A10B solo on 128GB (smaller MoE baseline)
- [ ] Largest dense models across 2+ machines (Llama-3.3-70B, Qwen2.5-72B)
