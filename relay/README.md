# iroh-relay on Fly.io

Self-hosted [iroh-relay](https://github.com/n0-computer/iroh/tree/main/iroh-relay) for mesh-llm. Provides relay connectivity for peers that can't connect directly (symmetric NAT, firewalls, etc).

**Live**: `https://mesh-llm-relay.fly.dev/`

## Architecture

```
  iroh client                    Fly.io edge                  container
  ─────────                      ──────────                   ─────────
  https://mesh-llm-relay.fly.dev ──► TLS termination ──► plain HTTP :8080
       WebSocket upgrade         ◄── passes through  ◄── iroh-relay (no TLS)
       binary relay framing
```

Fly terminates TLS at the edge and forwards plain HTTP to iroh-relay inside the container. The relay runs **without TLS** — Fly owns the cert. This works because:

- iroh clients connect via `https://` — Fly's TLS satisfies that
- The relay protocol is WebSocket-based — Fly passes upgrades through cleanly
- Authentication falls back to challenge-response (one extra round-trip vs end-to-end TLS keying material export). This is normal and handled by iroh's handshake protocol.

**QUIC address discovery is disabled** — it requires server-side TLS on a UDP port, which can't go through Fly's proxy. Doesn't matter: mesh-llm nodes already use raw STUN (Google/Cloudflare) to discover their public IP.

## Use with mesh-llm

```bash
mesh-llm --model Qwen2.5-32B --relay https://mesh-llm-relay.fly.dev./
```

Note the trailing `./` — iroh relay URL convention.

## Deploy

```bash
cd relay/
fly deploy
```

That's it. The Dockerfile pulls a pre-built `iroh-relay` binary from [GitHub releases](https://github.com/n0-computer/iroh/releases) — no Rust compilation needed. Image is ~32MB.

## Update iroh-relay version

Change `IROH_VERSION` in the Dockerfile, then `fly deploy`.

## Verify

```bash
# Root page
curl https://mesh-llm-relay.fly.dev/
# → <h1>Iroh Relay</h1>

# Health check
curl https://mesh-llm-relay.fly.dev/healthz
# → {"status":"ok","version":"0.96.0",...}

# Captive portal probe
curl -o /dev/null -w "%{http_code}" https://mesh-llm-relay.fly.dev/generate_204
# → 204

# Relay endpoint (needs WebSocket upgrade, 400 is correct for plain GET)
curl https://mesh-llm-relay.fly.dev/relay
# → missing header: upgrade
```

## Configuration

See `iroh-relay.toml`. Key settings:

| Setting | Value | Notes |
|---------|-------|-------|
| `http_bind_addr` | `[::]:8080` | Fly forwards here after TLS termination |
| `enable_relay` | `true` | The actual relay service |
| `enable_quic_addr_discovery` | `false` | Can't work behind Fly's proxy, not needed |
| `access` | `everyone` | Open relay — restrict with allowlist/denylist if needed |
| Rate limit | 2 MB/s per client | Generous for relay traffic (activations are ~10KB/token) |

## Infra

- Region: `syd`
- 2 machines (Fly default HA), auto-stop when idle
- shared-cpu-1x, 512MB RAM
- No persistent storage needed (no TLS certs to manage — Fly handles that)
