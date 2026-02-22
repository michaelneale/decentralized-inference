# mesh-llm TODO

## Ready to ship (on `smart-auto` branch, needs merge + release)
- [x] Smart auto-join: `score_mesh()`, region detection, QUIC health probe
- [x] Publish watchdog: auto-takeover if mesh publisher disappears
- [x] Reqwest downloads: native Rust, resume via Range header, no curl dependency
- [x] Linux support: VRAM detection (nvidia-smi, rocm-smi), device detection
- [x] Solo-only model assignment: nodes won't pick models they can't run solo
- [x] Deterministic model spreading: hash-based so concurrent joiners pick different models
- [x] Underserved model rebalancing at join time (pick least-served model)
- [x] Draft model auto-download on promotion
- [x] Demand tracking: per-model request counting, rates gossipped, demand-based promotion
- [x] Periodic demand check (60s timer alongside topology-change trigger)
- [x] Nostr field naming cleanup (`serving`/`wanted`/`on_disk`)
- [x] `--auto --client` clean error handling

## Nice to have
- [ ] Don't download what won't fit: check VRAM before downloading via `--model`
- [ ] Demand tracking in console: show req/min per model in TUI
- [ ] `mesh-llm recommend`: CLI subcommand to suggest models for your hardware
- [ ] Request rates in `/api/status` JSON for external tooling

## Future
- [ ] Demand-based Nostr listings: include request rates so `--auto` joiners can see what's hot
- [ ] Model catalog: embedded JSON of popular GGUF models with VRAM estimates
- [ ] Speculative decoding across mesh: draft model on small node, verify on big node
- [ ] Multi-node tensor split recovery: if one split peer dies, re-split across remaining
- [ ] Bandwidth-aware split: factor in measured RTT when deciding tensor split ratios
