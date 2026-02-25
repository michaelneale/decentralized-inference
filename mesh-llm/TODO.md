# mesh-llm TODO

## Crate API experiment (`crate-api` branch)

**Status**: Working but incomplete. In-process inference via safe Rust wrappers. Mesh binary unchanged.

### What this branch does

Adds `src/lib.rs` as a Rust crate API with three modes:
- `Engine::solo(path)` — loads GGUF in-process via `llama-cpp-2`, direct Metal GPU inference
- `Engine::connect(url)` — HTTP client to any OpenAI-compatible endpoint (mesh or standalone)
- `Engine::auto(path)` — probes localhost:9337 for running mesh, falls back to solo

Also adds `Engine::chat_with_tools()` for native tool calling via `apply_chat_template_with_tools_oaicompat()`.

### What this branch does NOT do

- Mesh binary (`src/main.rs`) is **unchanged** — still spawns llama-server + rpc-server as child processes
- Still 3 binaries to distribute (mesh-llm, llama-server, rpc-server) — not a single binary
- Goose has NOT been migrated to use `Engine` API — only a Cargo.toml dep swap to share the llama.cpp build

### Dependency chain

```
mesh-llm (this crate)
  └── llama-cpp-2 (our fork: michaelneale/llama-cpp-rs, branch mesh-llm-fork)
        └── llama-cpp-sys-2
              └── llama.cpp submodule → our fork (michaelneale/llama.cpp)
                    with 3 RPC patches rebased onto upstream pin 1051ecd28

goose (local Cargo.toml change on branch mesh-llm-integration)
  ├── mesh-llm (path dep → this crate)
  └── llama-cpp-2 (path dep → same fork, so one native lib, no conflict)
```

### How to reconstruct this setup

```bash
# 1. Clone the llama-cpp-rs fork
cd /path/to/deez
git clone git@github.com:michaelneale/llama-cpp-rs.git
cd llama-cpp-rs
git checkout mesh-llm-fork
git submodule update --init --recursive

# 2. Checkout this branch
cd /path/to/deez
git checkout crate-api

# 3. Build + test
cd mesh-llm
cargo build --release --example solo
./target/release/examples/solo ~/.models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

# 4. (Optional) Point goose at the fork
# In goose/crates/goose/Cargo.toml, replace:
#   llama-cpp-2 = { version = "0.1.133", ... }
# with:
#   llama-cpp-2 = { path = "/path/to/deez/llama-cpp-rs/llama-cpp-2", ... }
#   mesh-llm = { path = "/path/to/deez/mesh-llm" }
```

### Build impact

- Mesh binary size: **unchanged** (28MB) — llama symbols stripped by linker
- Clean build time: **+23s** (47s → 70s) — llama-cpp-sys-2 compiles llama.cpp C++ even though mesh binary doesn't use it
- Could be fixed with a cargo feature flag (`solo = ["llama-cpp-2"]`) to make it optional for mesh-only builds

### What was deleted

- `src/llama_ffi.rs` (150 lines raw unsafe FFI) — replaced by llama-cpp-2 safe wrappers
- `csrc/chat_shim.h` — abandoned earlier experiment

### Related branches

| Repo | Branch | What |
|------|--------|------|
| `michaelneale/decentralized-inference` | `crate-api` | This branch — Engine crate API |
| `michaelneale/llama-cpp-rs` | `mesh-llm-fork` | llama-cpp-rs with submodule pointing at our llama.cpp fork |
| `block/goose` (local) | `mesh-llm-integration` | Cargo.toml pointing llama-cpp-2 + mesh-llm at local paths |

---

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

## Nice to Have
- [ ] Don't download what won't fit: check VRAM before downloading via `--model`
- [ ] Demand tracking in console: show req/min per model in TUI
- [ ] Request rates in `/api/status` JSON for external tooling
- [ ] `mesh-llm recommend`: CLI subcommand to suggest models for your hardware

## Future
- [ ] **Public named meshes**: `--mesh-name "cool-mesh" --publish` currently gets -200 penalty for random `--auto` users (treated as private group). If someone explicitly passes both `--mesh-name` and `--publish`, add a `public: true` field to the Nostr listing so it scores like an unnamed mesh (no penalty). Lets people give their mesh a fun name without hiding it from discovery.
- [ ] Demand-based Nostr listings: include request rates so `--auto` joiners can see what's hot
- [ ] Multi-node tensor split recovery: if one split peer dies, re-split across remaining
