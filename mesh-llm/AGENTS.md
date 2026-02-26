# mesh-llm — Agent Notes

## Console Changes — MANDATORY REVIEW

**NEVER push changes to `console.html` without the user seeing the result first.**
The console is a visual UI — code review is not sufficient. Changes must be:
1. Built and running locally
2. Shown to the user (leave it running, tell them to check localhost:3131)
3. Explicitly approved before commit/push

This applies to ANY change touching `console.html` or `api.rs` status payloads that affect what the console displays.

## Project Structure

- `src/` — Rust source (see `README.md` for file map)
- `docs/TESTING.md` — **Test playbook** with all test scenarios and remote test setup
- `docs/DESIGN.md` — Architecture, protocols, all features
- `TODO.md` — Current work items

**Read `docs/TESTING.md` and `docs/DESIGN.md` before making changes.**

## Testing

**Read `docs/TESTING.md` before running any tests.** It has:
- All test scenarios (solo, split, multi-model, client, resilience)
- Remote deploy instructions (bundle, codesign, xattr)
- Single-machine testing with ephemeral keys
- Cleanup commands

### Test Machines

| Machine | SSH | VRAM | Notes |
|---|---|---|---|
| **Local** (Sydney) | n/a (this machine) | 51.5GB (M4 Max 64GB) | Binary: `mesh-llm/target/release/mesh-llm` |
| **Mini** (Sydney) | `sshpass -p "spankychat2000" ssh michaelneale@192.168.86.60` | 12.9GB (M4 16GB) | Bundle at `/tmp/mesh-bundle/` |
| **Brad** (QLD) | `ssh mic@home.dwyer.au -p 23632` | 103GB (M4 Max 128GB) | Must use `--bind-port 7842`, broken SSL/NAT |

### Deploy to Remote

```bash
just bundle                  # builds /tmp/mesh-bundle.tar.gz (~17MB)
scp /tmp/mesh-bundle.tar.gz michaelneale@192.168.86.60:/tmp/
ssh michaelneale@192.168.86.60 "cd /tmp && tar xzf mesh-bundle.tar.gz && codesign -s - mesh-bundle/mesh-llm mesh-bundle/rpc-server mesh-bundle/llama-server"
```

### Common Test Pattern

```bash
# Local: start originator
nohup ./target/release/mesh-llm --model Qwen2.5-3B-Instruct-Q4_K_M --port 8090 > /tmp/local-node.log 2>&1 &

# Extract token
TOKEN=$(grep "Invite token:" /tmp/local-node.log | awk '{print $NF}')

# Mini: join
ssh mini "cd /tmp/mesh-bundle && MESH_LLM_EPHEMERAL_KEY=1 nohup ./mesh-llm --model Qwen2.5-3B-Instruct-Q4_K_M --port 8091 --join $TOKEN > /tmp/mini-node.log 2>&1 &"

# Verify
curl http://localhost:8090/v1/models
ssh mini "curl http://localhost:8091/v1/models"
```

### Cleanup (always do this)

```bash
pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server
ssh mini "pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server"
```

## Deploy Checklist — MANDATORY

**Every deploy to test machines MUST follow this checklist. No shortcuts.**

### Before starting nodes
1. **Bump VERSION** — Change `VERSION` in `main.rs` to a new tentative version so you can verify the running binary is actually the new code.
2. **Build and bundle** — `cargo build --release && just bundle`
3. **Kill ALL processes on ALL nodes** — `pkill -9 -f mesh-llm; pkill -9 -f llama-server; pkill -9 -f rpc-server` on Local, Mini, AND Brad.
4. **Verify clean** — Run `ps -eo pid,args | grep -E 'mesh-llm|llama-server|rpc-server' | grep -v grep` on EVERY node. Must return empty.
5. **Deploy bundle** — scp + tar + codesign on Mini and Brad.
6. **Verify version on disk** — Run `mesh-llm --version` on EVERY node. Must show the new version.

### After starting nodes
7. **Verify exactly 1 mesh-llm process per node** — `ps -eo pid,ppid,args | grep mesh-llm | grep -v grep` on EVERY node. Must show exactly 1 process, properly parented.
8. **Verify child processes** — Each mesh-llm should have at most 1 rpc-server and 1 llama-server as children (check ppid matches mesh-llm pid). No orphans.
9. **Verify API responds on every node** — `curl -s http://localhost:3131/api/status` must return valid JSON on EVERY node. Don't assume — check.
10. **Verify version in gossip** — Check `/api/status` peers list. New-code nodes must show the new version string. Old-code nodes show null/missing (that's expected).
11. **Verify peer count** — Every node should see the expected number of peers. If a node is missing, investigate immediately — don't assume "it'll show up".
12. **Test inference through every model** — Send a request to EVERY model listed in `/v1/models`. Verify non-empty, valid response. Empty responses must be investigated (check `reasoning_content` for reasoning models like GLM).
13. **Test `/v1/` passthrough on 3131** — Verify `/v1/models` and `/v1/chat/completions` both work on port 3131, not just 9337.

### Common failures to watch for
- **nohup over SSH doesn't stick** — Use `bash -c "nohup ... & disown"` pattern. Always verify the process is still running 5+ seconds after SSH disconnects.
- **Stale binary on Brad** — Brad copies from `/tmp/mesh-bundle/` to `~/bin/`. If you forget `cp`, Brad runs the old binary.
- **Duplicate processes** — If a previous run wasn't killed cleanly, you get 2 mesh-llm processes fighting for ports. Always kill-verify-start.
- **codesign changes the hash** — Don't compare MD5 of Local build vs codesigned remote binary. Compare Mini vs Brad (both codesigned from same bundle).

### Brad Constraints

- **Must be originator** (NAT prevents inbound connections when joining)
- **Must use `--bind-port 7842`** (port forwarding configured for this port)
- **Broken SSL trust store** — can't verify iroh relay HTTPS certs
- Only works as publisher/initiator, not as joiner

## Key Files to Read

- `src/main.rs` — CLI args, orchestration: `run_auto()`, `run_idle()`, `run_passive()`
- `src/mesh.rs` — `Node` struct, gossip, mesh_id, peer management
- `src/election.rs` — Host election, tensor split calculation
- `src/proxy.rs` — HTTP proxy plumbing: request parsing, model routing, response helpers
- `src/api.rs` — Management API (:3131): `/api/status`, `/api/events`, `/api/discover`, `/api/join`
- `src/nostr.rs` — Nostr discovery, `score_mesh()`, `smart_auto()`
- `src/download.rs` — Model catalog (`MODEL_CATALOG`), HuggingFace downloads
