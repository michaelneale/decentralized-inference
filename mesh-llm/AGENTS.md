# mesh-llm — Agent Notes

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
