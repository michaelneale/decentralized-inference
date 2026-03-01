# mesh-llm Fly.io apps — shared builder, different configs
#
# Console (dashboard + API):
#   fly deploy --config fly/console/fly.toml --dockerfile fly/Dockerfile
#
# API only (rate-limited public endpoint):
#   fly deploy --config fly/api/fly.toml --dockerfile fly/Dockerfile --build-arg CMD=api
#
# Or run natively (no Docker):
#   mesh-llm --client --auto

FROM rust:latest AS builder

RUN apt-get update && apt-get install -y cmake pkg-config && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY mesh-llm/ mesh-llm/

WORKDIR /src/mesh-llm
RUN cargo build --release

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/mesh-llm/target/release/mesh-llm /usr/local/bin/mesh-llm

ARG CMD=console
ENV APP_MODE=${CMD}

EXPOSE 3131 9337

COPY <<'EOF' /entrypoint.sh
#!/bin/sh
if [ "$APP_MODE" = "api" ]; then
  exec mesh-llm --client --auto --port 9337 --listen-all
else
  exec mesh-llm --client --auto --port 9337 --console 3131 --listen-all
fi
EOF
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
