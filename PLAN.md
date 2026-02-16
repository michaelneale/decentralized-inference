# Design Notes & Benchmarks

Historical notes from development.

## Transport Benchmarks (WiFi, same LAN)

| Transport | RTT | Throughput | Notes |
|-----------|-----|------------|-------|
| Raw ICMP ping | 30-80ms | — | WiFi baseline, jittery |
| Raw TCP | — | 30 MB/s | WiFi ceiling |
| iroh QUIC direct | 10-14ms | 3-5 MB/s | |
| iroh QUIC relay | ~400ms | 2 MB/s | Rate-limited, unusable for interactive inference |
| Direct UDP (WAN, port forwarded) | ~20ms | — | Measured AU east coast ↔ AU east coast |

QUIC throughput is lower than raw TCP, but activations are ~10KB/token — throughput doesn't matter. **QUIC mesh adds zero measurable overhead to inference tok/s** vs raw TCP.

The real bottleneck is per-token RPC round-trip latency: ~40ms × 2 RTTs = ~80ms/token network cost.

## The Core Insight

Both machines have the model file on disk. **Zero need to transfer model weights over the network.** Workers load tensors from their own local GGUF at NVMe speed.

What crosses the network:
- **Control plane** (~KB): graph compute commands, tensor metadata
- **Activations** (~10KB/token): intermediate results at graph split boundaries
- **NOT weights** (~17GB): these come from local disk

## llama.cpp Patches

Branch: `rpc-local-gguf` in `michaelneale/llama.cpp`

| Patch | Effect |
|-------|--------|
| SET_TENSOR_GGUF | Client sends tensor name, server reads from local GGUF. 17GB loads in ~9s vs 14+ min. |
| Probing skip | Short-circuits `weight_buft_supported()` for RPC. Eliminates hundreds of alloc/free round-trips at startup. |
| get_alloc_size cache | FNV hash cache: 558 → 8 calls per token. 2-18x tok/s improvement. |
| B2B direct transfers | Workers push activations directly to each other via PUSH_TENSOR_TO_PEER, bypassing orchestrator. |

## Inference Benchmarks

2-node mesh: M4 Max (55GB VRAM) + Mac Mini M4 (12GB VRAM), WiFi, GLM-4.7-Flash-Q4_K_M (17GB).

| Configuration | Generation |
|---|---|
| Mini orchestrator, 85% remote on M4 Max | **21 tok/s** |
| M4 Max orchestrator, 82% local + 18% remote | **16 tok/s** |
| 3-node: Mini orchestrator + 2 workers (40/40/20) | **12-13 tok/s** |
| Local only (M4 Max, no mesh) | 68 tok/s |

### Per-token steady state

- 7 RPC calls per token (down from ~290 before chatter reduction)
- 2 blocking network round-trips per token (irreducible with current protocol)

## Key Design Decisions

- **iroh over libp2p**: simpler API, built-in NAT traversal
- **Same binary everywhere**: worker/host/client role determined by flags or auto-election
- **Single ALPN + single QUIC connection per peer**: multiplexed by first byte
- **Model weights never transfer**: both sides have the GGUF locally
- **`--no-mmap` always**: prevents mmap crash on unified memory Macs with less VRAM than model
- **llama-server always uses --rpc**: even solo, host's own rpc-server is always in the list. Same code path always.
- **mesh-inference owns the API port**: llama-server on ephemeral ports, no port conflicts on restart
- **Every mesh change = kill + re-elect + fresh start**: no special restart/rebalance logic
- **Mini as orchestrator > M4 Max as orchestrator**: bulk compute (85%) runs on M4 Max with zero network cost
- **STUN fallback for public IP discovery**: if iroh relay STUN doesn't work (DNS sinkhole etc), raw STUN to Google/Cloudflare discovers the public IP for the invite token
- **`--bind-port` for NAT port forwarding**: pins QUIC to a fixed UDP port so router rules work

## WAN Testing Notes

Tested: MacBook Pro M4 Max (local, symmetric NAT) ↔ Mac Studio M4 Max 128GB (remote, `home.dwyer.au:23632`, EIM NAT).

- Remote network has DNS sinkhole: `*.n0.iroh-canary.iroh.link` → bogus IP, all UDP/53 transparently intercepted
- iroh relays unreachable from remote — relay connections drop after 10s
- **Solution**: forward UDP port 7842 on remote router, use `--bind-port 7842`
- STUN discovers public IP automatically, invite token includes it
- Direct UDP at ~20ms RTT, inference works end-to-end

Verified scenarios:
- Brad solo → `localhost:9337` works (solo inference, llama-server uses own rpc-server)
- Local solo → `localhost:9337` works
- Brad host + local worker → both `localhost:9337` work (distributed, tensor split 0.67/0.33, 12 tok/s gen)
- Brad host + local lite client → `localhost:9337` works (proxied via QUIC)
- Mesh change (local joins) → brad kills llama-server, re-elects, restarts with updated --rpc and tensor split

Test script: `mesh-inference/test-mesh.sh`
