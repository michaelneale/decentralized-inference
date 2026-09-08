# Build the skippy benchmark/debug telemetry collector.
[unix]
metrics-server-build:
    just with-lld cargo build -p metrics-server

[windows]
metrics-server-build:
    @just with-lld cargo build -p metrics-server

# Build the binaries copied into the Skippy WAN Docker lab image.
[linux]
skippy-wan-lab-build-bins:
    cargo build --release --locked -p skippy-server -p skippy-prompt -p metrics-server -p skippy-model-package

# Build the resumable GGUF conversion/quantization replacement CLI.
[unix]
skippy-quantize-build:
    just with-lld cargo build -p skippy-quantize

[windows]
skippy-quantize-build:
    @just with-lld cargo build -p skippy-quantize

# Build the release binary used in HF conversion/quantization job images.
[unix]
skippy-quantize-release-build:
    just with-lld cargo build --release --locked -p skippy-quantize

[windows]
skippy-quantize-release-build:
    @just with-lld cargo build --release --locked -p skippy-quantize

# Build skippy-quantize as a standalone quantization binary with the pinned
# llama.cpp quantization ABI linked into the executable.
[unix]
skippy-quantize-standalone-build backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static scripts/build-llama.sh
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static just with-lld cargo build -p skippy-quantize --no-default-features

[unix]
skippy-quantize-standalone-release-build backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static scripts/build-llama.sh
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static just with-lld cargo build --release --locked -p skippy-quantize --no-default-features

# Build and execute the opt-in private native Skippy application probes.
[unix]
skippy-native-tests backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static LLAMA_STAGE_BUILD_TESTS=ON scripts/build-llama.sh

# Replay every retained and current native Skippy gate without broad upstream tests.
[unix]
skippy-native-full-replay backend="cpu":
    scripts/prepare-llama.sh pinned
    LLAMA_STAGE_BACKEND="{{ backend }}" LLAMA_STAGE_LINK_MODE=static LLAMA_STAGE_BUILD_DIR="$PWD/.deps/llama-build/build-stage-full-replay-static-{{ backend }}" LLAMA_STAGE_FULL_REPLAY=ON LLAMA_STAGE_FORCE_BUILD=1 scripts/build-llama.sh

# Generate a reproducible benchmark corpus for skippy bench tooling.
bench-corpus tier="smoke" *ARGS="":
    scripts/generate-bench-corpus.py "{{ tier }}" {{ ARGS }}

# Run skippy family certification checks.
family-certify *ARGS:
    just with-lld scripts/family-certify.sh {{ ARGS }}

# Run target/draft speculative compatibility checks.
spec-bench target draft *ARGS:
    just with-lld env LLAMA_STAGE_BUILD_DIR=".deps/llama-build/build-stage-abi-static" cargo build -p llama-spec-bench
    LLAMA_STAGE_BUILD_DIR=".deps/llama-build/build-stage-abi-static" target/debug/llama-spec-bench --target-model-path "{{ target }}" --draft-model-path "{{ draft }}" {{ ARGS }}

# Smoke a standalone skippy OpenAI frontend stage.
skippy-openai-smoke *ARGS:
    just with-lld scripts/skippy-openai-smoke.sh {{ ARGS }}

# Build the OpenAI benchmark server against an explicitly selected native ABI.
# Nightly runners provide LLAMA_STAGE_BUILD_DIR from their pre-baked image.
competitive-benchmark-build:
    just with-lld cargo build --release --locked -p skippy-server

# Run the skippy benchmark/debug telemetry collector.
metrics-server db="/tmp/mesh-metrics.duckdb" http_addr="127.0.0.1:18080" otlp_addr="127.0.0.1:14317" *ARGS="": metrics-server-build
    target/debug/metrics-server serve --db "{{ db }}" --http-addr "{{ http_addr }}" --otlp-grpc-addr "{{ otlp_addr }}" {{ ARGS }}
