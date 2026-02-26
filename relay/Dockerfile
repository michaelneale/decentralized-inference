# iroh-relay server — pre-built binary from GitHub releases
#
# Fly.io terminates TLS, relay runs plain HTTP.
#
#   fly deploy --config relay/fly.toml --dockerfile relay/Dockerfile

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates curl && rm -rf /var/lib/apt/lists/*

ARG IROH_VERSION=0.96.0
RUN curl -fsSL "https://github.com/n0-computer/iroh/releases/download/v${IROH_VERSION}/iroh-relay-v${IROH_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz -C /usr/local/bin/ && \
    chmod +x /usr/local/bin/iroh-relay

COPY iroh-relay.toml /etc/iroh-relay.toml

EXPOSE 8080 9090

CMD ["iroh-relay", "--config-path", "/etc/iroh-relay.toml"]
