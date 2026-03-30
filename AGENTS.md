# Agent Notes

## Repo Overview

This repo (`mesh-llm`) contains mesh-llm — a Rust binary that pools GPUs over QUIC for distributed LLM inference using llama.cpp.

## Key Docs

| Doc | What it covers |
|---|---|
| `README.md` | Usage, install, CLI flags, examples |
| `CONTRIBUTING.md` | Build from source, dev workflow, UI dev |
| `RELEASE.md` | Release process (build, bundle, tag, GitHub release) |
| `ROADMAP.md` | Future directions |
| `PLAN.md` | Historical design notes and benchmarks |
| `mesh-llm/TODO.md` | Current work items and backlog |
| `mesh-llm/README.md` | Rust crate overview and file map |
| `mesh-llm/docs/DESIGN.md` | Architecture, protocols, features |
| `mesh-llm/docs/TESTING.md` | Test playbook, scenarios, remote deploy |
| `mesh-llm/docs/MoE_PLAN.md` | MoE expert sharding design |
| `mesh-llm/docs/MoE_DEPLOY_DESIGN.md` | MoE auto-deploy UX |
| `mesh-llm/docs/MoE_SPLIT_REPORT.md` | MoE splitting validation results |
| `fly/README.md` | Fly.io deployment (console + API apps) |
| `relay/README.md` | Self-hosted iroh relay on Fly |

## Building

Always use `just`. Never build manually.

```bash
just build    # llama.cpp fork + mesh-llm + UI
just bundle   # portable tarball
just stop     # kill mesh/rpc/llama processes
just test     # quick inference test against :9337
just auto     # build + stop + start with --auto
just ui-dev   # vite dev server with HMR
```

See `CONTRIBUTING.md` for full dev workflow.

## Project Structure

- `mesh-llm/src/` — Rust source
- `mesh-llm/ui/` — React web console (shadcn/ui patterns, see https://ui.shadcn.com/llms.txt)
- `mesh-llm/docs/` — Design and testing docs
- `fly/` — Fly.io deployment (console + API client apps)
- `relay/` — Self-hosted iroh relay
- `evals/` — Benchmarking and evaluation scripts

## Key Source Files

- `mesh-llm/src/main.rs` — CLI args, orchestration: `run_auto()`, `run_idle()`, `run_passive()`
- `mesh-llm/src/mesh.rs` — `Node` struct, gossip, mesh_id, peer management
- `mesh-llm/src/election.rs` — Host election, tensor split calculation
- `mesh-llm/src/proxy.rs` — HTTP proxy: request parsing, model routing, response helpers
- `mesh-llm/src/api.rs` — Management API (:3131): `/api/status`, `/api/events`, `/api/discover`, `/api/join`
- `mesh-llm/src/nostr.rs` — Nostr discovery, `score_mesh()`, `smart_auto()`
- `mesh-llm/src/download.rs` — Model catalog (`MODEL_CATALOG`), HuggingFace downloads
- `mesh-llm/src/moe.rs` — MoE detection, expert rankings, split orchestration
- `mesh-llm/src/launch.rs` — llama-server/rpc-server process management

## Plugin Protocol Compatibility

When iterating on the plugin protocol, always consider protocol compatibility.

- If a protocol change may be breaking, explicitly ask the developer whether the change is intended to be breaking.
- If the change is not intended to be breaking, the previous version of the plugin protocol must continue to be supported.
- Do not silently ship plugin protocol changes that strand older plugins or hosts without confirming that outcome is acceptable.

## UI Notes

For changes in `mesh-llm/ui/`, use components and compose interfaces consistently with shadcn/ui patterns. Prefer extending existing primitives in `ui/src/components/ui/` over ad-hoc markup.

## Testing

Read `mesh-llm/docs/TESTING.md` before running tests. It has all test scenarios, remote deploy instructions, and cleanup commands.

## Formatting

Before committing Rust changes, format only the changed Rust files from the repo root, for example with `cargo fmt --all -- path/to/file.rs`, and include those formatting changes in the commit.

### Deploy to Remote

```bash
just bundle
# scp bundle to remote, tar xzf, codesign -s - the three binaries
```

### Cleanup

```bash
pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server
```

## Deploy Checklist — MANDATORY

**Every deploy to test machines MUST follow this checklist.**

### Before starting nodes
1. **Bump VERSION** in `main.rs` so you can verify the running binary is new code.
2. `just build && just bundle`
3. Kill ALL processes on ALL nodes — `pkill -9 -f mesh-llm; pkill -9 -f llama-server; pkill -9 -f rpc-server`
4. Verify clean — `ps -eo pid,args | grep -E 'mesh-llm|llama-server|rpc-server' | grep -v grep` must be empty.
5. Deploy bundle — scp + tar + codesign on remote nodes.
6. Verify version — `mesh-llm --version` on every node.

### After starting nodes
7. Verify exactly 1 mesh-llm process per node.
8. Verify child processes (at most 1 rpc-server + 1 llama-server per mesh-llm).
9. `curl -s http://localhost:3131/api/status` returns valid JSON on every node.
10. Check `/api/status` peers for new version string.
11. Verify expected peer count.
12. Test inference through every model in `/v1/models`.
13. Test `/v1/` passthrough on port 3131.

### Common failures
- **nohup over SSH doesn't stick** — use `bash -c "nohup ... & disown"`, verify process survives disconnect.
- **Duplicate processes** — always kill-verify-start.
- **codesign changes the hash** — don't compare local vs codesigned remote.

## Releasing

See `RELEASE.md` for the full process.

Current release flow:

1. Build and verify locally:
   ```bash
   just build
   just bundle
   ```
2. Release from a clean local `main` branch:
   ```bash
   just release v0.X.Y
   ```
   This bumps the version, refreshes `Cargo.lock` without upgrading dependencies, commits as `v0.X.Y: release`, pushes `main`, and then pushes only the new release tag.
3. Pushing a `v*` tag triggers `.github/workflows/release.yml`, which builds the release artifacts on Linux CPU, Linux CUDA, and macOS and creates the GitHub release automatically.

## Credentials

Test machine IPs, SSH details, and passwords are in `~/Documents/private-note.txt` (outside the repo). **Never commit credentials to any tracked file.**

## What NOT to add

- **No `api_key_token` feature** — explicitly rejected, removed in v0.26.0
- **No credentials in tracked files** — IPs, passwords, SSH commands belong in `~/Documents/private-note.txt` only
