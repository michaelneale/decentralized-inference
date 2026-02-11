#!/usr/bin/env python3
"""
TCP latency proxy for llama.cpp RPC servers.

Sits between llama-server and an rpc-server, forwarding all traffic
with configurable artificial latency to simulate network conditions.

Usage:
  python3 latency-proxy.py --listen-port 60052 --target-port 50052 --latency-ms 50
  python3 latency-proxy.py --listen-port 60053 --target-port 50053 --latency-ms 100

Architecture:
  llama-server  →  latency-proxy(:60052)  →→  rpc-server(:50052)
                   (delay on inference ops)

Latency modes:
  --mode compute    Delay only GRAPH_COMPUTE/RECOMPUTE (original behavior)
  --mode transfer   Delay GRAPH_COMPUTE/RECOMPUTE + GET_TENSOR + SET_TENSOR
                    This properly simulates network latency for the full
                    client-relay path (activations bounced through orchestrator).
  --mode all        Delay every command (useful for debugging, very slow)

The key insight: in upstream llama.cpp, cross-server tensor copies go through
the client (orchestrator) via GET_TENSOR from server A → SET_TENSOR to server B.
The b2b fork eliminates this relay with direct server-to-server pushes. To fairly
benchmark both, we need to add latency on the full transfer path, not just compute.
"""

import argparse
import socket
import struct
import threading
import time
import sys

# RPC command IDs (from ggml-rpc.cpp)
RPC_CMD_SET_TENSOR = 6
RPC_CMD_SET_TENSOR_HASH = 7
RPC_CMD_GET_TENSOR = 8
RPC_CMD_GRAPH_COMPUTE = 10
RPC_CMD_GRAPH_RECOMPUTE = 16

RPC_CMD_NAMES = {
    0: "ALLOC_BUFFER", 1: "GET_ALIGNMENT", 2: "GET_MAX_SIZE",
    3: "BUFFER_GET_BASE", 4: "FREE_BUFFER", 5: "BUFFER_CLEAR",
    6: "SET_TENSOR", 7: "SET_TENSOR_HASH", 8: "GET_TENSOR",
    9: "COPY_TENSOR", 10: "GRAPH_COMPUTE", 11: "GET_DEVICE_MEMORY",
    12: "INIT_TENSOR", 13: "GET_ALLOC_SIZE", 14: "HELLO",
    15: "DEVICE_COUNT", 16: "GRAPH_RECOMPUTE",
    17: "REGISTER_PEER", 18: "PUSH_TENSOR_TO_PEER", 19: "PEER_TENSOR_DATA",
}

# Commands that have NO response (fire-and-forget from client)
NO_RESPONSE_CMDS = {RPC_CMD_SET_TENSOR, RPC_CMD_GRAPH_COMPUTE, RPC_CMD_GRAPH_RECOMPUTE}

# Latency sets for each mode
LATENCY_SETS = {
    "compute":  {RPC_CMD_GRAPH_COMPUTE, RPC_CMD_GRAPH_RECOMPUTE},
    "transfer": {RPC_CMD_GRAPH_COMPUTE, RPC_CMD_GRAPH_RECOMPUTE,
                 RPC_CMD_GET_TENSOR, RPC_CMD_SET_TENSOR},
    "all":      set(range(20)),
}

# Track whether model loading is done (first GRAPH_COMPUTE signals inference start)
loading_done = False
loading_lock = threading.Lock()

stats_lock = threading.Lock()
stats = {
    "bytes_forwarded": 0,
    "delays_injected": 0,
    "total_latency_ms": 0.0,
    "cmd_counts": {},
}


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[proxy {ts}] {msg}", flush=True)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf.extend(chunk)
    return bytes(buf)


def proxy_connection(client, target_host, target_port, latency_s, latency_cmds,
                     conn_id, listen_port, skip_during_loading):
    global loading_done
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.connect((target_host, target_port))
    except Exception as e:
        log(f"[conn {conn_id}] Cannot connect to {target_host}:{target_port}: {e}")
        client.close()
        return

    try:
        while True:
            cmd_byte = recv_exact(client, 1)
            cmd = cmd_byte[0]
            cmd_name = RPC_CMD_NAMES.get(cmd, f"UNK({cmd})")

            size_bytes = recv_exact(client, 8)
            payload_size = struct.unpack("<Q", size_bytes)[0]
            payload = recv_exact(client, payload_size) if payload_size > 0 else b""

            with stats_lock:
                stats["bytes_forwarded"] += 1 + 8 + payload_size
                stats["cmd_counts"][cmd_name] = stats["cmd_counts"].get(cmd_name, 0) + 1

            # Mark loading as done when we see first GRAPH_COMPUTE
            if cmd == RPC_CMD_GRAPH_COMPUTE:
                with loading_lock:
                    loading_done = True

            # Inject latency if:
            # 1. This command is in the latency set
            # 2. Latency is configured
            # 3. Either we don't skip during loading, or loading is done
            should_delay = (
                cmd in latency_cmds
                and latency_s > 0
                and (not skip_during_loading or loading_done)
            )

            if should_delay:
                time.sleep(latency_s)
                with stats_lock:
                    stats["delays_injected"] += 1
                    stats["total_latency_ms"] += latency_s * 1000

            server.sendall(cmd_byte + size_bytes + payload)

            if cmd not in NO_RESPONSE_CMDS:
                resp_size_bytes = recv_exact(server, 8)
                resp_size = struct.unpack("<Q", resp_size_bytes)[0]
                resp_data = recv_exact(server, resp_size) if resp_size > 0 else b""
                client.sendall(resp_size_bytes + resp_data)
                with stats_lock:
                    stats["bytes_forwarded"] += 8 + resp_size

    except ConnectionError:
        pass
    except Exception as e:
        log(f"[conn {conn_id}] Error: {e}")
    finally:
        client.close()
        server.close()


def print_stats(interval, listen_port, target_port):
    while True:
        time.sleep(interval)
        with stats_lock:
            mb = stats["bytes_forwarded"] / (1024 * 1024)
            delays = stats["delays_injected"]
            lat = stats["total_latency_ms"]
            cmds = dict(stats["cmd_counts"])
        log(f":{listen_port}→:{target_port} | {mb:.1f}MB | {delays} delays ({lat:.0f}ms) | {cmds}")


def main():
    parser = argparse.ArgumentParser(
        description="TCP latency proxy for llama.cpp RPC")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, required=True)
    parser.add_argument("--latency-ms", type=float, default=0)
    parser.add_argument("--mode", choices=["compute", "transfer", "all"],
                        default="transfer",
                        help="Which commands to delay (default: transfer)")
    parser.add_argument("--stats-interval", type=float, default=10)
    args = parser.parse_args()

    latency_s = args.latency_ms / 1000.0
    latency_cmds = LATENCY_SETS[args.mode]
    # Skip delays during loading for SET_TENSOR (bulk weight transfer)
    skip_during_loading = (args.mode in ("transfer", "all"))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.bind(("127.0.0.1", args.listen_port))
    sock.listen(5)

    log(f"Listening on :{args.listen_port} → :{args.target_port}")
    log(f"Latency: {args.latency_ms}ms, mode: {args.mode}")
    log(f"Delayed commands: {', '.join(RPC_CMD_NAMES.get(c, str(c)) for c in sorted(latency_cmds))}")
    if skip_during_loading:
        log(f"SET_TENSOR delays deferred until first GRAPH_COMPUTE (skip during model loading)")

    threading.Thread(
        target=print_stats,
        args=(args.stats_interval, args.listen_port, args.target_port),
        daemon=True,
    ).start()

    conn_id = 0
    try:
        while True:
            client, addr = sock.accept()
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn_id += 1
            threading.Thread(
                target=proxy_connection,
                args=(client, args.target_host, args.target_port, latency_s,
                      latency_cmds, conn_id, args.listen_port, skip_during_loading),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        log("Shutting down")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
