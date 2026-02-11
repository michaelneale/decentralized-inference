#!/usr/bin/env python3
"""
TCP latency proxy for llama.cpp RPC servers.

Sits between llama-server and an rpc-server, forwarding all traffic
with configurable artificial latency injected ONLY on compute commands
(GRAPH_COMPUTE and GRAPH_RECOMPUTE). This simulates network latency
for distributed inference without slowing down model loading.

Usage:
  python3 latency-proxy.py --listen-port 60052 --target-port 50052 --latency-ms 50
  python3 latency-proxy.py --listen-port 60053 --target-port 50053 --latency-ms 100

Architecture:
  llama-server  →  latency-proxy(:60052)  →→  rpc-server(:50052)
                   (delay on compute ops)

The RPC protocol is:
  client→server: | cmd (1 byte) | size (8 bytes) | payload (size bytes) |
  server→client: | size (8 bytes) | payload (size bytes) |   (for commands with responses)

We parse the command byte from the client stream to identify GRAPH_COMPUTE (10)
and GRAPH_RECOMPUTE (16), and inject latency only on those. Everything else
(SET_TENSOR, ALLOC_BUFFER, etc.) passes through at full speed.
"""

import argparse
import socket
import struct
import threading
import time
import sys

# RPC command IDs (from ggml-rpc.cpp)
RPC_CMD_GRAPH_COMPUTE = 10
RPC_CMD_GRAPH_RECOMPUTE = 16
RPC_CMD_SET_TENSOR = 6

RPC_CMD_NAMES = {
    0: "ALLOC_BUFFER", 1: "GET_ALIGNMENT", 2: "GET_MAX_SIZE",
    3: "BUFFER_GET_BASE", 4: "FREE_BUFFER", 5: "BUFFER_CLEAR",
    6: "SET_TENSOR", 7: "SET_TENSOR_HASH", 8: "GET_TENSOR",
    9: "COPY_TENSOR", 10: "GRAPH_COMPUTE", 11: "GET_DEVICE_MEMORY",
    12: "INIT_TENSOR", 13: "GET_ALLOC_SIZE", 14: "HELLO",
    15: "DEVICE_COUNT", 16: "GRAPH_RECOMPUTE",
}

# Commands that have NO response (fire-and-forget from client)
# SET_TENSOR, GRAPH_COMPUTE, GRAPH_RECOMPUTE send data but don't wait for a reply
NO_RESPONSE_CMDS = {RPC_CMD_SET_TENSOR, RPC_CMD_GRAPH_COMPUTE, RPC_CMD_GRAPH_RECOMPUTE}

# Commands where we inject latency
LATENCY_CMDS = {RPC_CMD_GRAPH_COMPUTE, RPC_CMD_GRAPH_RECOMPUTE}

stats_lock = threading.Lock()
stats = {
    "bytes_forwarded": 0,
    "compute_ops": 0,
    "total_latency_ms": 0.0,
    "cmd_counts": {},
}


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[proxy {ts}] {msg}", flush=True)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes or raise."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf.extend(chunk)
    return bytes(buf)


def forward_exact(src_data: bytes, dst: socket.socket):
    """Send all bytes to dst."""
    dst.sendall(src_data)


def proxy_connection(
    client: socket.socket,
    target_host: str,
    target_port: int,
    latency_s: float,
    conn_id: int,
    listen_port: int,
):
    """
    Proxy a single RPC connection, parsing the command stream.

    The client (llama-server) sends commands; we parse the 1-byte command ID
    and 8-byte payload size, forward the command + payload to the rpc-server,
    then (if the command expects a response) forward the response back.

    Latency is injected only before GRAPH_COMPUTE / GRAPH_RECOMPUTE.
    """
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.connect((target_host, target_port))
    except Exception as e:
        log(f"[conn {conn_id}] Cannot connect to {target_host}:{target_port}: {e}")
        client.close()
        return

    log(f"[conn {conn_id}] :{listen_port} → :{target_port} established")

    try:
        while True:
            # Read command byte
            cmd_byte = recv_exact(client, 1)
            cmd = cmd_byte[0]
            cmd_name = RPC_CMD_NAMES.get(cmd, f"UNKNOWN({cmd})")

            # Read payload size (8 bytes, little-endian uint64)
            size_bytes = recv_exact(client, 8)
            payload_size = struct.unpack("<Q", size_bytes)[0]

            # Read payload
            payload = recv_exact(client, payload_size) if payload_size > 0 else b""

            # Track stats
            with stats_lock:
                stats["bytes_forwarded"] += 1 + 8 + payload_size
                stats["cmd_counts"][cmd_name] = stats["cmd_counts"].get(cmd_name, 0) + 1

            # Inject latency on compute commands
            if cmd in LATENCY_CMDS and latency_s > 0:
                time.sleep(latency_s)
                with stats_lock:
                    stats["compute_ops"] += 1
                    stats["total_latency_ms"] += latency_s * 1000

            # Forward command to server
            server.sendall(cmd_byte + size_bytes + payload)

            # If the command expects a response, forward it back
            if cmd not in NO_RESPONSE_CMDS:
                # Response format: | size (8 bytes) | data (size bytes) |
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
        log(f"[conn {conn_id}] Closed")


def print_stats(interval: float, listen_port: int, target_port: int):
    """Periodically print stats."""
    while True:
        time.sleep(interval)
        with stats_lock:
            mb = stats["bytes_forwarded"] / (1024 * 1024)
            ops = stats["compute_ops"]
            lat = stats["total_latency_ms"]
            cmds = dict(stats["cmd_counts"])
        log(f":{listen_port}→:{target_port} | {mb:.1f} MB fwd | {ops} compute ops | {lat:.0f}ms injected | cmds: {cmds}")


def main():
    parser = argparse.ArgumentParser(
        description="TCP latency proxy for llama.cpp RPC (injects delay on compute ops only)")
    parser.add_argument("--listen-port", type=int, required=True,
                        help="Port to listen on (llama-server connects here)")
    parser.add_argument("--target-host", default="127.0.0.1",
                        help="RPC server host (default: 127.0.0.1)")
    parser.add_argument("--target-port", type=int, required=True,
                        help="RPC server port to forward to")
    parser.add_argument("--latency-ms", type=float, default=0,
                        help="Latency in ms to inject per GRAPH_COMPUTE/RECOMPUTE (default: 0)")
    parser.add_argument("--stats-interval", type=float, default=10,
                        help="Print stats every N seconds (default: 10)")
    args = parser.parse_args()

    latency_s = args.latency_ms / 1000.0

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.bind(("127.0.0.1", args.listen_port))
    sock.listen(5)

    log(f"Listening on :{args.listen_port} → forwarding to :{args.target_port}")
    log(f"Latency: {args.latency_ms}ms on GRAPH_COMPUTE/GRAPH_RECOMPUTE only")
    if latency_s == 0:
        log("No latency injection (passthrough mode)")

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
            log(f"[conn {conn_id}] Accepted from {addr}")
            threading.Thread(
                target=proxy_connection,
                args=(client, args.target_host, args.target_port, latency_s, conn_id, args.listen_port),
                daemon=True,
            ).start()
    except KeyboardInterrupt:
        log("Shutting down")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
