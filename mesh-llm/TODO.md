# mesh-llm TODO

## Mixture of Models (MoM)

Route different requests to specialized models based on task type. Instead of one "best" model, the mesh becomes smarter about which model handles what.

**Paper:** [Mixture of Models: An Intra-Model Ensemble Approach](https://arxiv.org/pdf/2601.16863)

The paper shows ensemble routing across heterogeneous models outperforms any single model. Our mesh already has the ingredients — multiple models, a router that classifies requests. The gap is making the router model-aware (which models are good at what) and potentially splitting complex requests across models.

**Relates to:** Smart Router (below), Vision routing (vision requests → vision model), Multi-Model Per Host.

## Multi-Model Per Host

Currently each host runs one llama-server serving one model. Hosts with spare VRAM could serve multiple simultaneously.

**Options:**
1. **Multiple llama-server processes** — each on a different port, proxy routes by model. Simple but duplicates KV cache overhead.
2. **llama-server native multi-model** — newer versions support `--model` multiple times. Single process, shared infrastructure.

**Why it matters:**
- Studio (206GB) could serve MiniMax (130GB) + a vision model (20GB)
- Mini (16GB) could serve Qwen3.5-9B (5.5GB) + draft model
- Enables MoM routing across models on the same host

## Peer-to-Peer Model Transfer

Fetch model files directly from mesh peers instead of HuggingFace. Peers already have QUIC connections — add a new stream type where the requester sends a filename and offset, the responder streams the file back.

**Why:** LAN transfers are massively faster than HuggingFace downloads. Two machines on the same network could transfer a 47GB model in minutes instead of an hour. Also works when HF is slow, rate-limited, or down.

**Design:**
- New bi-stream type (`STREAM_FILE_TRANSFER`): requester sends filename + resume offset, responder reads from `~/.models/` and streams back
- Only serve files from `~/.models/` — no path traversal
- Resume support via byte offset
- Prefer low-RTT peers (LAN) over high-RTT (relay)
- Download logic tries peers first, falls back to HuggingFace
- Extend gossip to include filenames on disk so peers know what's fetchable

## SSD Expert Streaming

Run giant MoE models on a single node by streaming active experts from NVMe instead of fitting everything in RAM.

[flash-moe](https://github.com/danveloper/flash-moe) already does this — runs Qwen3.5-397B-A17B at 5.5 tok/s on a 48GB M3 Max with 6GB resident memory. See [ROADMAP.md](../ROADMAP.md).

**Plan:** Use flash-moe as an alternative backend. Mesh-llm spawns it like llama-server. Needs HTTP/SSE endpoint (currently CLI only) and OpenAI-compatible `/v1/chat/completions`.

## MoE Expert Sharding

Design: [MoE_PLAN.md](docs/MoE_PLAN.md) · Auto-deploy: [MoE_DEPLOY_DESIGN.md](docs/MoE_DEPLOY_DESIGN.md) · Validation: [MoE_SPLIT_REPORT.md](docs/MoE_SPLIT_REPORT.md)

- [ ] **Lazy `moe-analyze`** — auto-run ranking for unknown MoE models.
- [ ] **Scale testing** — Mixtral 8×22B, Qwen3-235B-A22B across multi-node.

## Thinking / Reasoning Budget ✅

Shipped in v0.40.0. `--reasoning-budget 0` is set on all llama-servers — thinking off by default. API users can opt in per-request with `chat_template_kwargs: {"enable_thinking": true}`.

See `tests/test_reasoning_compat.sh` for the contract tests validating this behavior.

**Why off by default:** Qwen3.5-9B is broken with thinking (reasoning burns entire token budget, content always empty). MiniMax-M2.5 is 2-27x slower with thinking for no quality gain. Agentic eval with pi confirmed: same 3/4 accuracy, half the total time with thinking off.

## Smart Router
- [ ] **Context-aware routing**: Hosts advertise `n_ctx` in gossip. Router estimates request token count and skips hosts that can't fit it. Today a long chat routed to a small-context host returns 400 with no fallback.
- [ ] **Retry on 400**: If a host returns 400 (context overflow, bad request), try the next host instead of forwarding the error. Requires reading the response status before committing to the byte-pipe tunnel. Non-trivial — the current `relay_tcp_via_quic` is a blind bidirectional copy.
- [ ] **Static speed estimates**: `tok_s: f64` on ModelProfile. Quick tasks prefer fast models.
- [ ] **Response quality checks**: Detect empty/repetitive/truncated responses, retry with different model.
- [ ] **MoM-aware routing**: Route by task type to best-suited model (see Mixture of Models above).
- [ ] **Vision-aware routing**: Auto-route image requests to vision-capable models.

## Resilience
- [ ] **Multi-node tensor split recovery**: If one split peer dies, re-split across remaining.

## Distributed Agent Sandboxes (kodak integration)

Route coding agent tasks across the mesh to sandboxed environments (Tart VMs, Docker containers). See [kodak](https://github.com/michaelneale/kodak) for the sandbox manager.

**Core idea:** Agents pull work off the mesh and execute in isolated sandboxes. The mesh handles task routing, context transfer, and result streaming. Editors talk to a local ACP/stdio shim that posts tasks to the mesh — they don't know the work happened remotely.

**Task lifecycle:**
1. Task posted to blackboard (prompt + requirements + context manifest)
2. Capable nodes evaluate and claim (platform, toolchains, agents, repo familiarity)
3. First valid claim wins, poster acks
4. Context transferred over QUIC (repo bundle, files, error logs)
5. Agent runs in sandbox, streams results back over QUIC
6. Output artifacts (diffs, test results) published to mesh

**Key design constraint:** Agents need rich context to be useful — not just a prompt. A remote agent needs the repo (or relevant subset), build state, project conventions, error logs. This is the hard part.

**Node affinity / warm state:**
- Nodes track what repos they've cloned, what builds they've cached, what tasks they've completed
- Claiming priority favors nodes that already have the repo checked out and warm
- First task for a new repo is slow (clone), subsequent ones are fast
- Affinity is soft — any capable node can claim, warm nodes just claim faster

**What exists today:**
- Blackboard (gossip-based message store) → task metadata
- QUIC direct channels → context transfer + result streaming
- kodak justfile → Tart VM lifecycle, agent installation, headless execution
- Claim protocol design in kodak README

**What's needed:**
- [ ] Task schema on blackboard (structured JSON: task_id, prompt, requirements, context refs)
- [ ] Node capability advertisement (OS, toolchains, agents, warm repos)
- [ ] Context bundling: poster packages relevant files, transfers over QUIC
- [ ] Agent launcher on claiming node: spin up sandbox, inject context, run agent, stream output
- [ ] ACP stdio bridge: local shim speaks ACP to editor, translates to mesh tasks
- [ ] Affinity tracking: per-node state of repos/builds/history for smart claiming
- [ ] Artifact exchange: gossip the pointer (hash, size, location), QUIC the bytes

**Sandbox options:**
| Platform | Sandbox | Boot | Disk |
|----------|---------|------|------|
| macOS | Tart VM | ~12s | ~28GB |
| Linux | Docker | ~1-2s | ~0.5-2GB |
| Linux | Firecracker microVM | ~125ms | configurable |
| Windows | WSL2 / Hyper-V | varies | varies |

**Relates to:** Blackboard, QUIC tunnels, model routing (agents can use mesh-llm for inference)

## Vision — Future
- [ ] **More catalog entries**: Gemma-3-12B, Pixtral-12B, larger Qwen3.5 (35B-A3B MoE, 122B-A10B MoE)
- [ ] **Image generation**: Not supported by llama.cpp (transformers only), but could add diffusion backend later.
