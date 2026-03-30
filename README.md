# Mesh LLM

![Mesh LLM logo](docs/mesh-llm-logo.svg)

![Mesh LLM](mesh.png)

> ⚠️ **Built with caffeine and anger.** Harnesses used: [Goose](https://github.com/block/goose), [pi](https://github.com/mariozechner/pi-coding-agent), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview). Models: Opus, GPT 5.x, some MiniMax M2.5 and GLM 4.7 Flash.

Pool spare GPU capacity to run LLMs at larger scale. Models that don't fit on one machine are automatically distributed — dense models via pipeline parallelism, MoE models via expert sharding with zero cross-node inference traffic. Have your agents gossip across the mesh — share status, findings, and questions without a central server.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash
```

The installer probes your machine, recommends a flavor, and asks what you want to install.

If you want it to run as a per-user background service, see [Background service](#background-service).

The installer currently targets macOS and Linux release bundles. Windows is supported through source builds and published `.zip` release assets instead.

For non-interactive installs, set the flavor explicitly:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | MESH_LLM_INSTALL_FLAVOR=vulkan bash
```

Release bundles install flavor-specific llama.cpp binaries:

- macOS: `rpc-server-metal`, `llama-server-metal`
- Linux CPU: `rpc-server-cpu`, `llama-server-cpu`
- Linux CUDA: `rpc-server-cuda`, `llama-server-cuda`
- Linux ROCm: `rpc-server-rocm`, `llama-server-rocm`
- Linux Vulkan: `rpc-server-vulkan`, `llama-server-vulkan`

If you keep more than one flavor installed in the same `bin` directory, select the one you want explicitly:

```bash
mesh-llm --llama-flavor vulkan --model Qwen2.5-32B
```

If you want a local GPU build from source instead:

```bash
git clone https://github.com/michaelneale/mesh-llm
cd mesh-llm
just build
```

Requires: `just`, `cmake`, Rust toolchain, Node.js 24 + npm. NVIDIA GPU builds need `nvcc` (CUDA toolkit). AMD GPU builds need ROCm/HIP. Vulkan GPU builds need the Vulkan development files plus `glslc`. CPU-only and Jetson/Tegra also work. For source builds, `just build` auto-detects CUDA vs ROCm vs Vulkan on Linux, or you can force `backend=rocm` or `backend=vulkan`. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Windows source builds are also supported for `cuda`, `rocm`/`hip`, `vulkan`, and `cpu` via `just build`. Metal remains macOS-only. Tagged GitHub releases now publish Windows `.zip` bundles for `cpu`, `cuda`, `rocm`, and `vulkan`, and you can generate the same artifacts locally with `just release-build-windows`, `just release-build-cuda-windows`, `just release-build-amd-windows`, `just release-build-vulkan-windows`, and the matching `release-bundle-*-windows` recipes.

## Run
Once installed, you can run:

```bash
mesh-llm --auto                            # join the best public mesh, start serving
```

That's it. Downloads a model for your hardware, connects to other nodes, and gives you an OpenAI-compatible API at `http://localhost:9337`.

Or start your own:
```bash
mesh-llm --model Qwen2.5-32B              # downloads model (~20GB), starts API + web console
mesh-llm --model Qwen2.5-3B               # or a small model first (~2GB)
```

Add another machine:
```bash
mesh-llm --join <token>                    # token printed by the first machine
```

Or discover and join public meshes:
```bash
mesh-llm --auto                            # find and join the best mesh
mesh-llm --client --auto                   # join as API-only client (no GPU)
```

## How it works

Every node gets an OpenAI-compatible API at `http://localhost:9337/v1`. Distribution is automatic — you just say `mesh-llm --model X` and the mesh figures out the best strategy:

- **Model fits on one machine?** → runs solo, full speed, no network overhead
- **Dense model too big?** → pipeline parallelism — layers split across nodes
- **MoE model too big?** → expert parallelism — experts split across nodes, zero cross-node traffic

If a node has enough VRAM, it always runs the full model. Splitting only happens when it has to.
Currently using a lightly forked version of llama.cpp (see the Justfile for where it pulls branch from).

**Pipeline parallelism** — for dense models that don't fit on one machine, layers are distributed across nodes proportional to VRAM. llama-server runs on the highest-VRAM node and coordinates via RPC. Each rpc-server loads only its assigned layers from local disk. Latency-aware: peers are selected by lowest RTT first, with an 80ms hard cap — high-latency nodes stay in the mesh as API clients but don't participate in splits.

**MoE expert parallelism** — Mixture-of-Experts models (Qwen3-MoE, GLM, OLMoE, Mixtral, DeepSeek — increasingly the best-performing architectures) are auto-detected from the GGUF header. The mesh reads expert routing statistics to identify which experts matter most, then assigns each node an overlapping shard: a shared core of critical experts replicated everywhere, plus unique experts distributed across nodes. Each node gets a standalone GGUF with the full trunk + its expert subset and runs its own independent llama-server — zero cross-node traffic during inference. Sessions are hash-routed to nodes for KV cache locality.

**Multi-model** — different nodes serve different models simultaneously. The API proxy peeks at the `model` field in each request and routes to the right node via QUIC tunnel. `/v1/models` lists everything available.

**Demand-aware rebalancing** — a unified demand map tracks which models the mesh wants (from `--model` flags, API requests, and gossip). Demand signals propagate infectiously across all nodes and decay naturally via TTL. Standby nodes auto-promote to serve unserved models with active demand, or rebalance when one model is significantly hotter than others. When a model loses its last server, standby nodes detect it within ~60s.

**Latency design** — the key insight is that HTTP streaming is latency-tolerant while RPC is latency-multiplied. llama-server always runs on the same box as the GPU. The mesh tunnels HTTP, so cross-network latency only affects time-to-first-token, not per-token throughput. RPC only crosses the network for pipeline splits where the model physically doesn't fit on one machine.

### Network optimizations

- **Zero-transfer GGUF loading** — `SET_TENSOR_GGUF` tells rpc-server to read weights from local disk. Dropped model load from 111s → 5s.
- **RPC round-trip reduction** — cached `get_alloc_size`, skip GGUF lookups for intermediates. Per-token round-trips: 558 → 8.
- **Direct server-to-server transfers** — intermediate tensors pushed directly between rpc-servers via TCP, not relayed through the client.
- **Speculative decoding** — draft model runs locally on the host, proposes tokens verified in one batched forward pass. +38% throughput on code (75% acceptance).

## Usage

### Start a mesh
```bash
mesh-llm --model Qwen2.5-32B
```
Starts serving a model and prints an invite token. This mesh is **private** — only people you share the token with can join.

To make it **public** (discoverable by others via `--auto`):
```bash
mesh-llm --model Qwen2.5-32B --publish
```

### Join a mesh
```bash
mesh-llm --join <token>                    # join with invite token (GPU node)
mesh-llm --client --join <token>           # join as API-only client (no GPU)
```

### Named mesh (buddy mode)
```bash
mesh-llm --auto --model GLM-4.7-Flash-Q4_K_M --mesh-name "poker-night"
```
Everyone runs the same command. First person creates it, everyone else discovers "poker-night" and joins automatically. `--mesh-name` implies `--publish` — named meshes are always published to the directory.

### Auto-discover
```bash
mesh-llm --auto                            # discover, join, and serve a model
mesh-llm --client --auto                   # join as API-only client (no GPU)
mesh-llm discover                          # browse available meshes
```

### Multi-model
```bash
mesh-llm --model Qwen2.5-32B --model GLM-4.7-Flash

# Route by model name
curl localhost:9337/v1/chat/completions -d '{"model":"GLM-4.7-Flash-Q4_K_M", ...}'
```
Different nodes serve different models. The API proxy routes by the `model` field.

### Idle mode
```bash
mesh-llm                                   # no args — shows instructions + console
```
Opens a read-only console on `:3131`. Use the CLI to start or join a mesh.

## Background service

To install it as a per-user background service:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash -s -- --service
```

To seed the service with a custom startup command on first install:

```bash
curl -fsSL https://raw.githubusercontent.com/michaelneale/mesh-llm/main/install.sh | bash -s -- --service --service-args '--model Qwen2.5-3B'
```

Service installs are user-scoped:

- macOS installs a `launchd` agent at `~/Library/LaunchAgents/com.mesh-llm.mesh-llm.plist`
- Linux installs a `systemd --user` unit at `~/.config/systemd/user/mesh-llm.service`
- Shared environment config lives in `~/.config/mesh-llm/service.env`

The two platforms handle launch args differently:

- macOS: `launchd` runs `~/.config/mesh-llm/run-service.sh`, which reads `~/.config/mesh-llm/service.args`. `service.args` is one `mesh-llm` CLI argument per line. The installer creates it with `--auto` by default and preserves your edits on reinstall unless you pass `--service-args` again.
- Linux: the installer writes the `mesh-llm` argv directly into `ExecStart=` in `~/.config/systemd/user/mesh-llm.service`. If you pass `--service-args`, those replace the current unit args; otherwise the installer preserves the existing unit args on reinstall.

`service.env` is optional and shared by both platforms. Use plain `KEY=value` lines, for example:

```text
MESH_LLM_NO_SELF_UPDATE=1
```

If you edit the Linux unit manually, reload and restart it:

```bash
systemctl --user daemon-reload
systemctl --user restart mesh-llm.service
```

On Linux this is a user service, so if you want it to keep running after reboot before login, enable lingering once:

```bash
sudo loginctl enable-linger "$USER"
```

## Web console

```bash
mesh-llm --model Qwen2.5-32B    # dashboard at http://localhost:3131
```

Live topology, VRAM bars per node, model picker, built-in chat. Everything comes from `/api/status` (JSON) and `/api/events` (SSE).

Model management is also available on the console port:

```bash
curl 'http://localhost:3131/api/models'
curl 'http://localhost:3131/api/models/search?q=qwen+coder+30b'
curl 'http://localhost:3131/api/models/show?ref=Qwen3-8B-Q4_K_M'
curl -X POST http://localhost:3131/api/models/download \
  -H 'Content-Type: application/json' \
  -d '{"ref":"Qwen3-8B-Q4_K_M","draft":true}'
curl 'http://localhost:3131/api/models/download?ref=Qwen3-8B-Q4_K_M&draft=true'
curl -X DELETE 'http://localhost:3131/api/models/download?ref=Qwen3-8B-Q4_K_M&draft=true'
```

### Development

Build-from-source and UI development instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Using with agents

mesh-llm exposes an OpenAI-compatible API on `localhost:9337`. Any tool that supports custom OpenAI endpoints works. `/v1/models` lists available models; the `model` field in requests routes to the right node.

For built-in launcher integrations (`goose`, `claude`):

- If a mesh is already running locally on `--port`, it is reused.
- If not, `mesh-llm` auto-starts a background client node that auto-joins the mesh.
- If `--model` is omitted, the launcher picks the strongest tool-capable model available on the mesh.
- When the harness exits (e.g. `claude` quits), the auto-started node is cleaned up automatically.

### goose

[Goose](https://github.com/block/goose) is available as both CLI (`goose session`) and desktop app (Goose.app).

```bash
mesh-llm goose
```

Use a specific model (example: MiniMax):

```bash
mesh-llm goose --model MiniMax-M2.5-Q4_K_M
```

This command writes/updates `~/.config/goose/custom_providers/mesh.json` and launches Goose.

### pi

1. Start a mesh client:
```bash
mesh-llm --client --auto --port 9337
```

2. Check what models are available:
```bash
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

3. Add a `mesh` provider to `~/.pi/agent/models.json` (adjust model IDs to match your mesh):

```json
{
  "providers": {
    "mesh": {
      "api": "openai-completions",
      "apiKey": "mesh",
      "baseUrl": "http://localhost:9337/v1",
      "models": [
        {
          "id": "MiniMax-M2.5-Q4_K_M",
          "name": "MiniMax M2.5 (Mesh)",
          "contextWindow": 65536,
          "maxTokens": 8192,
          "reasoning": true,
          "input": ["text"],
          "compat": {
            "maxTokensField": "max_tokens",
            "supportsDeveloperRole": false,
            "supportsUsageInStreaming": false
          }
        }
      ]
    }
  }
}
```

4. Run pi:
```bash
pi --model mesh/MiniMax-M2.5-Q4_K_M
```

Or switch models interactively with Ctrl+M inside pi.

### opencode

```bash
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://localhost:9337/v1 opencode -m openai/GLM-4.7-Flash-Q4_K_M
```

### claude code

Claude Code can be launched directly through mesh-llm (no proxy required):

```bash
mesh-llm claude
```

Use a specific model (example: MiniMax):

```bash
mesh-llm claude --model MiniMax-M2.5-Q4_K_M
```

### curl / any OpenAI client

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-4.7-Flash-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

## Blackboard

The mesh doesn't just share compute — it shares knowledge. Agents and people post status updates, findings, and questions to a shared blackboard that propagates across the mesh.

Works standalone — you don't need to run models through the mesh. Using your own API keys or a cloud provider? Just run `mesh-llm --client` to give your agents a gossip layer. No GPU needed, no model needed.

```bash
mesh-llm --client

# Install the agent skill (works with pi, Goose, others)
mesh-llm blackboard install-skill

# Post what you're working on
mesh-llm blackboard "STATUS: [org/repo branch:main] refactoring billing module"

# Search the blackboard
mesh-llm blackboard --search "billing refactor"

# Check for unanswered questions
mesh-llm blackboard --search "QUESTION"
```

With the skill installed, agents proactively search before starting work, post their status, share findings, and answer each other's questions — all through the mesh.

Messages are ephemeral (48h), PII is auto-scrubbed, and everything stays within the mesh — no cloud, no external services.

### MCP Server

The blackboard is available as an MCP server for agent integration. Any MCP-compatible agent (pi, Claude Code, Goose, etc.) can post, search, and read the feed directly:

```bash
# Run as MCP server over stdio
mesh-llm blackboard --mcp
```

Configure in your agent's MCP settings:

```json
{
  "mcpServers": {
    "mesh-blackboard": {
      "command": "mesh-llm",
      "args": ["blackboard", "--mcp"]
    }
  }
}
```

Tools exposed: `blackboard_post`, `blackboard_search`, `blackboard_feed`.

## Benchmarks

GLM-4.7-Flash-Q4_K_M (17GB), M4 Max + Mac Mini M4, WiFi:

| Configuration | tok/s |
|---|---|
| Solo (no mesh) | 68 |
| 2-node split (85/15) | 21 |
| 3-node split (62/31/8) | 12-13 |

Cross-network (Sydney ↔ Queensland, ~20ms RTT): 10-25 tok/s. Overhead dominated by per-token RPC latency.

Stock llama.cpp RPC transfers 16.88GB on connect. This fork: **0 bytes, ~9 seconds**.

## Model discovery

```bash
mesh-llm models                     # list curated mesh-llm models
mesh-llm search qwen coder 30b      # search Hugging Face GGUFs
mesh-llm search --curated qwen      # search curated metadata only
mesh-llm show Qwen3-8B-Q4_K_M       # inspect one exact model ref
mesh-llm download Qwen3-8B-Q4_K_M --draft
```

Draft pairings for speculative decoding:

| Model | Size | Draft | Draft size |
|-------|------|-------|------------|
| Qwen2.5 (3B/7B/14B/32B/72B) | 2-47GB | Qwen2.5-0.5B | 491MB |
| Qwen3-32B | 20GB | Qwen3-0.6B | 397MB |
| Llama-3.3-70B | 43GB | Llama-3.2-1B | 760MB |
| Gemma-3-27B | 17GB | Gemma-3-1B | 780MB |

## Specifying models

`--model` accepts several formats. Models are auto-downloaded to your configured model directory on first use (defaults to `~/.models/`).

```bash
# Curated name (fuzzy match — finds Qwen3-8B-Q4_K_M)
mesh-llm --model Qwen3-8B

# Full curated model id
mesh-llm --model Qwen3-8B-Q4_K_M

# Hugging Face URL (any GGUF)
mesh-llm --model https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Hugging Face shorthand (org/repo/file.gguf)
mesh-llm --model bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Local file path
mesh-llm --model ~/my-models/custom-model.gguf
```

Downloads have resume support, so interrupted transfers pick up where they left off.

## Manifest sidecars

Downloaded models get a sibling manifest file with the suffix `.manifest.json`.

- `Qwen3-8B-Q4_K_M.gguf` writes `Qwen3-8B-Q4_K_M.gguf.manifest.json`
- `Qwen3-30B-A3B-UD-Q4_K_XL-00001-of-00002.gguf` writes `Qwen3-30B-A3B-UD-Q4_K_XL-00001-of-00002.gguf.manifest.json`
- directory-style model bundles write a sibling manifest next to the directory, for example `MyModel/` writes `MyModel.manifest.json`

Current manifest schema:

```json
{
  "version": 1,
  "source": {
    "provider": "huggingface",
    "repo": "Qwen/Qwen3-8B-GGUF",
    "revision": "abc123",
    "file": "Qwen3-8B-Q4_K_M.gguf",
    "resolved_url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/abc123/Qwen3-8B-Q4_K_M.gguf"
  },
  "identity": {
    "canonical_id": "huggingface:Qwen/Qwen3-8B-GGUF@abc123/Qwen3-8B-Q4_K_M.gguf",
    "display_name": "Qwen3-8B-Q4_K_M.gguf",
    "family": "qwen3",
    "architecture": null,
    "format": "gguf",
    "quantization": "Q4_K_M"
  },
  "compatibility": {
    "tokenizer_hash": null,
    "chat_template_hash": null,
    "base_model": null
  },
  "local": {
    "downloaded_at": "1743321600",
    "sha256": null,
    "size_bytes": 4921161728
  }
}
```

Field notes:

- `version` is the manifest schema version. Bump it if the on-disk format changes.
- `source` is the authoritative origin record. Current writers emit `provider` values `huggingface` and `direct_url`.
- `source.revision` is the resolved Hugging Face revision when known. If the user did not pin a revision, mesh-llm resolves the current repo SHA when it can.
- `identity.canonical_id` is the stable mesh-llm identity key used for cross-format and cross-location comparisons.
- `identity.family` and `identity.quantization` are best-effort hints inferred from the filename.
- `identity.format` comes from the file extension, or `directory` for model directories.
- `compatibility` is reserved for stronger tokenizer and chat-template matching. These fields are present now but usually null.
- `local.downloaded_at` is currently stored as Unix epoch seconds in a string.
- `local.sha256` is reserved for future hashing and is currently null.
- `local.size_bytes` is the local file size when mesh-llm writes the manifest.

If older downloads are missing manifests, rebuild them with:

```bash
mesh-llm provenance repair --source huggingface
```

## Config

Create `~/.mesh-llm/config.toml` to customize storage and Hugging Face auth:

```toml
[huggingface]
token = "hf_..."

[models]
dirs = [
  "/Volumes/LLM/models",
  "~/.models",
]
```

- `HF_TOKEN` overrides the config token when set.
- `models.dir` also works for a single directory.
- The first configured model directory is the default download destination.
- Without a Hugging Face token, search and download still work, but requests may be slower or throttled.

## Community

Join the [#mesh-llm channel on the Goose Discord](https://discord.gg/goose-oss) for discussion, support, and development chat.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build and development workflows.

## Project Structure

| Path | Purpose |
|---|---|
| `llama.cpp/` | [Fork](https://github.com/michaelneale/llama.cpp/tree/rpc-local-gguf) with zero-transfer RPC patches |
| `mesh-llm/` | Rust QUIC mesh ([internals](mesh-llm/README.md)) |

## [Roadmap](ROADMAP.md)

---

> *"You are all a bunch of dirty hackers"*
> — Author's CompSci 101 professor
