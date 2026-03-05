# mesh-llm — Agent Notes

## UI Composition (React UI)

For changes in `ui/`, use components and compose interfaces consistently with shadcn/ui patterns where possible.
Prefer extending existing shadcn-style primitives in `ui/src/components/ui/` over ad-hoc custom markup/styling.
Reference shadcn LLM instructions: https://ui.shadcn.com/llms.txt

## Project Structure

- `src/` — Rust source (see `README.md` for file map)
- `docs/TESTING.md` — **Test playbook** with all test scenarios and remote test setup
- `docs/DESIGN.md` — Architecture, protocols, all features
- `TODO.md` — Current work items

**Read `docs/TESTING.md` and `docs/DESIGN.md` before making changes.**

## Building

Always use `just`:

```bash
just build    # builds llama.cpp fork, mesh-llm binary, and UI
just bundle   # creates portable /tmp/mesh-bundle.tar.gz (~18MB)
just stop     # kill mesh/rpc/llama processes
just test     # quick inference test against :9337
just auto     # build + stop + start with --auto
just ui-dev   # vite dev server with HMR
```

See `CONTRIBUTING.md` for full dev workflow.

## Testing

**Read `docs/TESTING.md` before running any tests.** It has:
- All test scenarios (solo, split, multi-model, client, resilience)
- Remote deploy instructions (bundle, codesign, xattr)
- Single-machine testing with ephemeral keys
- Cleanup commands

Test machine details (IPs, credentials, SSH commands) are in `~/Documents/private-note.txt` (outside the repo) — **never commit credentials to any tracked file.**

### Deploy to Remote

```bash
just bundle
# scp bundle to remote, tar xzf, codesign -s - the three binaries
```

### Cleanup (always do this)

```bash
pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server
```

## Deploy Checklist — MANDATORY

**Every deploy to test machines MUST follow this checklist. No shortcuts.**

### Before starting nodes
1. **Bump VERSION** — Change `VERSION` in `main.rs` to a new tentative version so you can verify the running binary is actually the new code.
2. **Build and bundle** — `just build && just bundle`
3. **Kill ALL processes on ALL nodes** — `pkill -9 -f mesh-llm; pkill -9 -f llama-server; pkill -9 -f rpc-server` on every node.
4. **Verify clean** — `ps -eo pid,args | grep -E 'mesh-llm|llama-server|rpc-server' | grep -v grep` on every node. Must return empty.
5. **Deploy bundle** — scp + tar + codesign on remote nodes.
6. **Verify version on disk** — `mesh-llm --version` on every node. Must show the new version.

### After starting nodes
7. **Verify exactly 1 mesh-llm process per node**.
8. **Verify child processes** — Each mesh-llm should have at most 1 rpc-server and 1 llama-server as children.
9. **Verify API responds on every node** — `curl -s http://localhost:3131/api/status` must return valid JSON.
10. **Verify version in gossip** — Check `/api/status` peers list for the new version string.
11. **Verify peer count** — Every node should see the expected number of peers.
12. **Test inference through every model** — Send a request to every model in `/v1/models`.
13. **Test `/v1/` passthrough on 3131** — Verify `/v1/models` and `/v1/chat/completions` both work on port 3131, not just 9337.

### Common failures to watch for
- **nohup over SSH doesn't stick** — Use `bash -c "nohup ... & disown"` pattern. Always verify the process is still running after SSH disconnects.
- **Duplicate processes** — If a previous run wasn't killed cleanly, you get 2 mesh-llm processes fighting for ports. Always kill-verify-start.
- **codesign changes the hash** — Don't compare MD5 of local build vs codesigned remote binary.

## Releasing

See `RELEASE.md` for the full release process (build, verify, bundle, tag, `gh release create`).

## Key Files to Read

- `src/main.rs` — CLI args, orchestration: `run_auto()`, `run_idle()`, `run_passive()`
- `src/mesh.rs` — `Node` struct, gossip, mesh_id, peer management
- `src/election.rs` — Host election, tensor split calculation
- `src/proxy.rs` — HTTP proxy plumbing: request parsing, model routing, response helpers
- `src/api.rs` — Management API (:3131): `/api/status`, `/api/events`, `/api/discover`, `/api/join`
- `src/nostr.rs` — Nostr discovery, `score_mesh()`, `smart_auto()`
- `src/download.rs` — Model catalog (`MODEL_CATALOG`), HuggingFace downloads
- `src/moe.rs` — MoE detection, expert rankings, split orchestration
- `src/launch.rs` — llama-server/rpc-server process management
