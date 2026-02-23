# mesh-llm TODO

## First-Time Experience (fast `--auto`)
- [ ] **Mesh identity**: Stable `mesh_id` (UUID) generated on mesh creation, persisted in `~/.mesh-llm/mesh-id`, gossipped to all peers. Named meshes: `mesh_id = sha256(name + originator_nostr_pubkey)`. Unnamed: random UUID. Included in `MeshListing` on Nostr.
- [ ] **Sticky mesh preference**: Save last successful mesh ID to `~/.mesh-llm/last-mesh`. On `--auto`, score bonus (+500) for meshes matching saved ID. Not a hard lock — dead meshes expire from Nostr, degraded meshes lose on other scoring.
- [ ] **Uptime signal**: Add `started_at: u64` to `MeshListing`. Score bonus for meshes that have been running longer — a 24h mesh beats a 10-minute test.
- [ ] **API proxy during GPU bootstrap**: In `run_auto()`, start passive API proxy immediately after joining mesh (before model loads). User can hit `localhost:8080` and get inference via tunnel while their GPU is still loading. Transition to local serving when ready.
- [ ] **Solo fallback — fast starter model**: When `--auto` finds no mesh, download a small starter model first (Qwen2.5-3B, 2GB, ~1 min), start serving it immediately, then background-download the "real" model for the node's VRAM tier. User is chatting in <2 minutes.
- [ ] **Score mesh by model quality**: `smart_auto` should weight model quality — a mesh serving Qwen3-32B scores higher than one serving Qwen2.5-3B, all else equal. Use `MODEL_TIERS` VRAM requirements as a proxy for quality.

## Model Catalog Curation
- [ ] **Opinionated model tiers**: Curate recommended instruct models per VRAM tier. Current `MODEL_TIERS` and `MODEL_CATALOG` are ad-hoc — need a principled "if you have X GB, run Y" recommendation that considers quality, speed, and family diversity.
- [ ] **Draft model completeness**: Ensure every recommended main model has a draft pairing. Currently GLM-4.7 and DeepSeek have no draft.
- [ ] **Model quality metadata**: Add quality/benchmark scores to catalog entries so scoring can prefer better models, not just bigger ones.
- [ ] **Auto-upgrade path**: When a node is solo-serving a starter model and finishes downloading a better one, gracefully switch (stop llama-server, restart with new model). No impact to other mesh nodes.

## Nice to Have
- [ ] Don't download what won't fit: check VRAM before downloading via `--model`
- [ ] Demand tracking in console: show req/min per model in TUI
- [ ] Request rates in `/api/status` JSON for external tooling
- [ ] `mesh-llm recommend`: CLI subcommand to suggest models for your hardware

## Future
- [ ] Demand-based Nostr listings: include request rates so `--auto` joiners can see what's hot
- [ ] Multi-node tensor split recovery: if one split peer dies, re-split across remaining
