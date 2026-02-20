# mesh-llm TODO

## Done
- [x] Event-driven mesh (60s heartbeat + death broadcasts)
- [x] Dead peers set, rejoin loop, routing table protocol
- [x] Unified passive mode (`run_passive()`)
- [x] Hash-based host selection
- [x] `--auto` Nostr discovery
- [x] Passive node scaling (no gossip, routing table only, zero server state)
- [x] `MESH_LLM_EPHEMERAL_KEY=1` for single-machine testing
- [x] Tensor split tested locally (split, worker death → solo recovery)
- [x] Tensor split tested cross-network (Brad+Local+Mini, 3-way, Sydney↔QLD)
- [x] Passive client tested (no gossip, routing table, tunnel inference)

## Nice to have

### Reactive rebalancing
When a host dies, standby nodes with that model on disk could self-promote.
- On death broadcast: check if dead node's model is now unserved, promote if capable
- On join: check routing table for gaps
- Logic mostly exists in `pick_model_assignment()`, needs topology event trigger

### Don't download what won't fit
Before downloading via `--model`, check if node VRAM >= model_size * 1.1.
Skip download if model won't fit and no split peers available.

### Console VRAM usage bar
Second bar per node showing model_size / node_vram. Data already in gossip, just UI.
