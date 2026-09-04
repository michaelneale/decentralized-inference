# Contributing

Join the [#mesh-llm channel on the Goose Discord](https://discord.gg/goose-oss) for discussion and questions.

This file covers local build and development workflows for this repository.

## Prerequisites

- `just`
- `cmake`
- Rust toolchain (`cargo`)
- Node.js 24 + pnpm 10 or newer (for UI development). The UI lockfile keeps
  `overrides` in `pnpm-workspace.yaml`, which pnpm 9 does not read, so pnpm 9
  cannot install it. `corepack pnpm@10` is enough if your host pnpm is older.

**macOS**: Apple Silicon. Metal is used automatically.

**Linux NVIDIA**: x86_64 with an NVIDIA GPU. Requires the CUDA toolkit (`nvcc` in your `PATH`). On Arch Linux, CUDA is typically at `/opt/cuda`; on Ubuntu/Debian it's at `/usr/local/cuda`. Auto-detection finds the right SM architecture for your GPU.

**Linux AMD**: ROCm/HIP is supported when ROCm is installed. Typical installs expose `hipcc`, `hipconfig`, and `rocm-smi` under `/opt/rocm/bin`.

**Linux Vulkan**: Vulkan is supported when the Vulkan development files and `glslc` are installed. On Ubuntu/Debian, install `libvulkan-dev glslc`. On Arch Linux, install `vulkan-headers shaderc`.

**Windows**: native runtime builds support `cuda`, `hip`/`rocm`, `vulkan`, or
`cpu`. Metal is not supported on Windows.

## Build from source

Build the normal debug product: a backend-neutral dynamic host, its adjacent
locally packaged native runtime, and the UI:

```bash
just build
```

Release and packaging use the same host/runtime boundary. The only lower-level
static compilation primitive is runtime packaging; it never builds a host.
Build a release host once:

```bash
just release-host-build
```

Then build the backend runtime you are changing:

```bash
just release-runtime-build cpu
just release-runtime-build metal
just release-runtime-build cuda
just release-runtime-build rocm
just release-runtime-build vulkan
```

Backend toolchains must be available for the corresponding runtime build. For
NVIDIA on Linux, put `nvcc` on `PATH`; runtime packaging detects the selected
CUDA major and architecture from the toolchain/environment.

```bash
PATH=/opt/cuda/bin:$PATH just release-runtime-build cuda
# or
PATH=/usr/local/cuda/bin:$PATH just release-runtime-build cuda
```

Exercise the exact release discovery boundary with an isolated cache:

```bash
runtime_cache="$(mktemp -d)"
MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$PWD/dist/native-runtimes" \
MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$runtime_cache" \
  ./target/release/mesh-llm runtime list
```

The resolver validates version, Skippy ABI, OS, architecture, and backend. It
does not search the current working directory or copy a matching bundled
runtime into the user cache.

Create a portable product bundle after building both layers:

```bash
just release-bundle v0.X.0 dist
```

## UI development workflow

The React console and embedded asset crate live in `crates/mesh-llm-ui/`.
The host binary serves the built assets through the management API.

Use this two-terminal flow for UI development.

Terminal A (run `mesh-llm` yourself):

```bash
mesh-llm --port 9337 --console 3131
```

If `mesh-llm` is not on your `PATH`:

```bash
./target/release/mesh-llm --port 9337 --console 3131
```

Terminal B (run Vite with HMR):

```bash
just ui-dev
```

Open:

```text
http://127.0.0.1:5173
```

`ui-dev` defaults:

- Serves on `127.0.0.1:5173`
- Proxies `/api/*` to `http://127.0.0.1:3131`

Overrides:

```bash
# Different backend API origin for /api proxy
just ui-dev http://127.0.0.1:4141

# Different Vite dev port
just ui-dev http://127.0.0.1:3131 5174
```

## Useful commands

```bash
just stop             # stop mesh/rpc/llama processes
just test             # quick test against :9337
just check-release    # release-target/docs/workflow parity check
just compat-smoke ~/.cache/huggingface/hub/<model>.gguf   # optional 2-node + 1-client Python/Node/LiteLLM smoke
just --list           # list all recipes
```

On macOS and Linux, local Cargo artifacts are bounded independently from the sccache
compiler-object cache, which here keeps sccache's own 10 GiB local default. That default is
a developer-machine limit only -- CI does not use it, and pins a 2 GiB trusted seed instead
(`SCCACHE_CACHE_SIZE=2G`, see `.github/actions/restore-sccache-seed`). Inspect and prune the
local Cargo artifacts with:

```bash
just cache-status
just cache-prune-dry-run max_size=80GiB max_age=14
just cache-prune max_size=80GiB max_age=14
```

Pruning evicts the oldest incremental sessions first, then uses
`cargo clean -p` for old or size-dominant workspace packages. It is scoped to
this worktree's `target/` and reports before/after bytes. On macOS and Linux,
`just build` and `just build-dev` hold a shared lock for their full build while
cache status and dry-run pruning take the same lock in shared mode. Executed
pruning requires the corresponding exclusive lock before measuring or deleting
artifacts. Direct Cargo and lower-level build commands do not share that lock,
so pruning also refuses to run when it detects an active Cargo or Rust compiler
process as a best-effort safeguard. Cargo configurations that separate
`build.build-dir` from `target-dir` are rejected because the cache manager does
not report, lock, or clean a second artifact tree.

On native Windows, `just check-release` runs the host-safe Rust/doc invariant subset and skips the Bash-only `install.sh` / `package-release.sh` parity checks. Run it on macOS or Linux when you need full shell parity coverage.

### Testing crates on native Windows

A bare Windows checkout cannot build the test targets of crates that pull in
`skippy-ffi`'s static link mode — `mesh-llm-system` does, through
`mesh-llm-runtime-install`, which depends on `skippy-ffi` with
`default-features = false` — because `skippy-ffi/build.rs` then requires
prepared llama.cpp ABI archives
(`automatic native preparation is not supported for Windows from build.rs yet`).
`just test-all` needs the same native preparation, through its Bash pipeline.

For crate suites that do not exercise the native runtime, enable
`dynamic-native-runtime`: feature unification turns on `skippy-ffi`'s
`dynamic-runtime`, whose build script returns early. The `mesh-llm-system`
gates then run on a machine without a prepared native build:

```powershell
cargo test --locked -p mesh-llm-system --lib --features dynamic-native-runtime
cargo clippy --no-deps -p mesh-llm-system --all-targets --features dynamic-native-runtime -- -D warnings
cargo fmt --check -p mesh-llm-system
cargo run -p xtask -- repo-consistency no-console-print
```

`--no-deps` keeps Clippy scoped to the package you are changing. Some
`cfg`-gated code is dead only on Windows and currently fails `-D warnings`
there (`download_optional_url` in `autoupdate/release_fetch.rs`, two
`SignalOutcome` variants in `backend.rs`, the lock file field in
`skippy-runtime`'s `materialized_cache.rs`), so compare your run against the
same command on `main` before attributing an error to your change. Running the
native-runtime suites still needs a prepared build (`LLAMA_STAGE_BUILD_DIR`
or `SKIPPY_LLAMA_BUILD_DIR` pointing at one), which this section does not cover.
The Rust MSVC toolchain needs the Visual Studio Build Tools with the
"Desktop development with C++" workload installed.

## CI / GitHub Actions

For the current PR and main topology, read [`ci/ci.md`](ci/ci.md), the
[optimization spec](.omo/specs/pr-ci-optimization.md), and the canonical
[`manage-ci` skill](.agents/skills/manage-ci/SKILL.md) before editing CI.
`.github/AGENTS.md` enforces that sequence.

The current five-way PR/main topology, manual controller, planner profiles,
and runner/provider/cache policy are documented in [`ci/ci.md`](ci/ci.md),
especially [Planner and profiles](ci/ci.md#planner-and-profiles) and [Provider
and cache policy](ci/ci.md#provider-and-cache-policy). The normative rules for
editing workflows and CI scripts live in the [`manage-ci` skill](.agents/skills/manage-ci/SKILL.md).

Linux CI uses prebuilt public and self-hosted images from
[`Mesh-LLM/mesh-llm-runner-images`](https://github.com/Mesh-LLM/mesh-llm-runner-images).
CPU, Vulkan, versioned CUDA, and versioned ROCm images share a core environment,
warm dependencies from MeshLLM's checked-in manifests, and allow container
runtimes to reuse cached layers where the runner host or K3s node retains them
instead of reinstalling host packages in every job.

If CI is missing a dependency, update the appropriate MeshLLM manifest and
lockfile or the runner image's YAML profile/installer, then publish the image
and pin its OCI digest. Do not add a one-off `apt-get`, `pip`, global `npm`,
`cargo install`, downloaded binary, or similar setup step to an individual
workflow. Existing workflow-local setup is migration debt, not a pattern for
new jobs.

### Local validation and extensions

Run the narrow checks that match the change, plus the full contract suite for
workflow changes:

```bash
just ci-validate
```

The canonical complete local gate is `just test-all`; it includes repository
consistency, Rust formatting, Clippy, Rust tests, UI/docs builds, and E2E smoke.
Use `just ci-shellcheck <changed-script>...` for changed shell scripts and
`just check-release` when release-target consistency is in scope. These are the
complete CI-definition and worktree checks; narrow checks do not replace
`just test-all` when full repository validation is required.

Planner fixtures and the CI repository-consistency recipes are included in
`just ci-validate`. Use `just ci-crate-lists`, `just check-release`, or
`just publish-crates` when iterating on the corresponding narrower contract.

To add coverage, extend one typed reusable slice or local action, update the
ownership/dependency/row catalog and planner fixtures, preserve immutable
producer-to-consumer reachability, and add the slice to its lane's stable
summary. Keep the controller's bounded projection and aggregate check contract
in sync. Never copy a PR job into an entrypoint or accept a raw runner label.
Validate the GitHub fallback before any provider rollout.

## GPU benchmark execution

GPU bandwidth benchmarks are launched through the `mesh-llm` binary itself rather than standalone benchmark executables. The public command remains:

```bash
mesh-llm gpus detect
```

Internally, mesh-llm runs a hidden `benchmark` subcommand in a narrow subprocess boundary so native backend hangs and stdout capture stay isolated from the main process.

Standard builds support benchmark execution only for the backends wired into the normal build flow:

- macOS Apple Silicon: Metal
- Linux / Windows NVIDIA: CUDA
- Linux / Windows AMD: HIP / ROCm

Intel GPU benchmark execution is not currently supported in standard `just build` flows, so runtime benchmark selection intentionally skips Intel GPUs.

## Protocol Backward Compatibility

Any change to `crates/mesh-llm-host-runtime/src/protocol/` or `crates/mesh-client/src/protocol/` requires backward-compatibility tests before merging.

Embedded clients (iOS, macOS, Android) are permanently supported. Protocol changes that break embedded client compatibility are breaking changes.

Run the protocol compatibility tests after any protocol change:

```bash
cargo test -p mesh-llm --test protocol_compat_v0_client
cargo test -p mesh-llm --test protocol_convert_matrix
```

See [`docs/design/EMBEDDED_CLIENT_ADR.md`](docs/design/EMBEDDED_CLIENT_ADR.md) for the full compatibility policy and rationale.
