# Deploy mesh-inference to a remote macOS node

## Build the bundle locally

```bash
cd /path/to/deez
just bundle    # creates /tmp/mesh-bundle.tar.gz
```

## Copy to remote

```bash
scp -P <SSH_PORT> /tmp/mesh-bundle.tar.gz user@host:
```

## Install on remote

```bash
ssh -p <SSH_PORT> user@host
mkdir -p ~/bin && tar xzf mesh-bundle.tar.gz -C ~/bin --strip-components=1
```

The bundle contains: `mesh-inference`, `rpc-server`, `llama-server`, `*.dylib`.

## Fix macOS quarantine

Files transferred via scp get `com.apple.provenance` xattr which causes macOS to SIGKILL (exit 137) on launch. **Always run after scp:**

```bash
codesign -s - ~/bin/mesh-inference
codesign -s - ~/bin/rpc-server
codesign -s - ~/bin/llama-server
xattr -cr ~/bin/
```

To verify: `xattr ~/bin/mesh-inference` should return nothing. If you see `com.apple.provenance` or `com.apple.quarantine`, the binary will be killed on launch.

## Download a model

```bash
~/bin/mesh-inference download 32b --draft    # downloads to ~/.models/
```

Or list all available models:
```bash
~/bin/mesh-inference download
```

Models go in `~/.models/` by convention. Both nodes need the same GGUF file for distributed inference.

## Start the node

### As first node (creates mesh)
```bash
nohup ~/bin/mesh-inference --model Qwen2.5-32B --bind-port 7842 > /tmp/mesh.log 2>&1 &
```

- `--bind-port` pins QUIC to a fixed UDP port for NAT port forwarding
- The invite token is printed to stderr (captured in the log)

Get the token:
```bash
grep "Invite token:" /tmp/mesh.log | tail -1 | sed "s/Invite token: //"
```

### As joining node
```bash
nohup ~/bin/mesh-inference --model Qwen2.5-32B --join <TOKEN> > /tmp/mesh.log 2>&1 &
```

### As lite client (no GPU, no model, API access only)
```bash
nohup ~/bin/mesh-inference --client --join <TOKEN> > /tmp/mesh.log 2>&1 &
```

## Networking

- **Only one side needs port forwarding.** Forward the `--bind-port` UDP port on the router of whichever node creates the mesh.
- The joining side does not need port forwarding.
- Check connectivity: the invite token embeds the creator's addresses. If the joiner can reach any of them over UDP, it works.
- If iroh relays are blocked on the remote network (DNS sinkhole), use `--relay <url>` to specify a reachable relay, or rely on direct UDP with port forwarding.

## Verifying it works

```bash
# Check processes are running
pgrep -la "mesh-inference|rpc-server|llama-server"

# Check API
curl -s http://localhost:9337/v1/models

# Test inference
curl -s http://localhost:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

## Stopping

```bash
pkill -f mesh-inference; pkill -f rpc-server; pkill -f llama-server
```

rpc-server and llama-server are child processes of mesh-inference, but killing the parent doesn't always kill them (they can become orphans with ppid=1). Always kill all three explicitly.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Exit 137 immediately | macOS quarantine xattr | `codesign -s - ~/bin/*; xattr -cr ~/bin/` |
| Empty reply from API | llama-server still loading | Wait. Check `/tmp/mesh-inference-llama-server.log` |
| "No inference server available" | Election in progress or llama-server crashed | Check `/tmp/mesh.log` for errors |
| Timeout waiting for tunnel maps | Peer disconnected during model load | Will auto-recover on next mesh change |
| Orphan rpc-server holding GPU memory | Parent mesh-inference was killed | `pkill -f rpc-server` |
| `*.n0.iroh-canary.iroh.link` DNS fails | Network has DNS sinkhole | Use `--bind-port` + UDP port forwarding instead of relays |

## Log locations

- `~/.mesh-inference/key` — persistent node identity
- `/tmp/mesh.log` — main process output (if started with `> /tmp/mesh.log 2>&1`)
- `/tmp/mesh-inference-llama-server.log` — llama-server stdout/stderr
- `/tmp/mesh-inference-rpc-<PORT>.log` — rpc-server stdout/stderr
