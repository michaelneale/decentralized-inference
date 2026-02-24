# mesh-llm web console — runs as a client, proxies to the mesh
#
# Build from repo root:
#   docker build -f web/Dockerfile -t mesh-llm-web .
#   docker run -p 3131:3131 -p 9337:9337 mesh-llm-web
#
# Fly.io (from repo root):
#   fly deploy --config web/fly.toml --dockerfile web/Dockerfile
#
# Or run natively (no Docker):
#   mesh-llm --client --auto

FROM rust:1.85-bookworm AS builder

RUN apt-get update && apt-get install -y cmake pkg-config && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY mesh-llm/ mesh-llm/

WORKDIR /src/mesh-llm
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/mesh-llm/target/release/mesh-llm /usr/local/bin/mesh-llm

EXPOSE 3131 9337

CMD ["mesh-llm", "--client", "--auto", "--port", "9337", "--console", "3131", "--listen-all"]
