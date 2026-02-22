# mesh-llm TODO

## Nice to have
- [ ] Don't download what won't fit: check VRAM before downloading via `--model`
- [ ] Demand tracking in console: show req/min per model in TUI
- [ ] Request rates in `/api/status` JSON for external tooling
- [ ] `mesh-llm recommend`: CLI subcommand to suggest models for your hardware

## Future
- [ ] Demand-based Nostr listings: include request rates so `--auto` joiners can see what's hot
- [ ] Model catalog: embedded JSON of popular GGUF models with VRAM estimates
- [ ] Speculative decoding across mesh: draft model on small node, verify on big node
- [ ] Multi-node tensor split recovery: if one split peer dies, re-split across remaining
