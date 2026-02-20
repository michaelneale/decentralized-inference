# mesh-llm TODO

## Done
- [x] Event-driven mesh (60s heartbeat + death broadcasts)
- [x] Dead peers set, rejoin loop, routing table protocol
- [x] Unified passive mode (`run_passive()`)
- [x] Hash-based host selection
- [x] `--auto` Nostr discovery
- [x] Passive node scaling (no gossip, routing table only, zero server state)
- [x] `MESH_LLM_EPHEMERAL_KEY=1` for single-machine testing
- [x] Tensor split tested locally + cross-network (3-way Brad+Local+Mini)
- [x] Passive client tested (no gossip, routing table, tunnel inference)
- [x] Console VRAM usage bar (model_size / node_vram per node)

## Nice to have

### Reactive rebalancing
When a host dies, standby nodes with that model on disk could self-promote.
- On death broadcast: check if dead node's model is now unserved, promote if capable
- On join: check routing table for gaps
- Logic mostly exists in `pick_model_assignment()`, needs topology event trigger

### Don't download what won't fit
Before downloading via `--model`, check if node VRAM >= model_size * 1.1.
Skip download if model won't fit and no split peers available.
