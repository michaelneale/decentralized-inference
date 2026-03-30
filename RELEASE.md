# Releasing mesh-llm

## Prerequisites

- `just` installed (`brew install just`)
- `cmake` installed (`brew install cmake`)
- `cargo` installed (packaged with rust)
- `gh` CLI authenticated (`gh auth status`)
- llama.cpp fork cloned (`just build` does this automatically)

## Steps

### 1. Build everything fresh

```bash
just build
```

On macOS, this clones/updates the llama.cpp fork if needed, builds with `-DGGML_METAL=ON -DGGML_RPC=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF`, and builds the Rust mesh-llm binary. Linux release workflows build CPU, CUDA, ROCm, and Vulkan variants separately.

### 2. Verify no homebrew dependencies

```bash
otool -L llama.cpp/build/bin/llama-server | grep -v /System | grep -v /usr/lib
otool -L llama.cpp/build/bin/rpc-server | grep -v /System | grep -v /usr/lib
otool -L target/release/mesh-llm | grep -v /System | grep -v /usr/lib
```

Each should only show the binary name — no `/opt/homebrew/` paths.

### 3. Create the bundle

```bash
just bundle
```

Creates `/tmp/mesh-bundle.tar.gz` containing `mesh-llm`, flavor-specific llama.cpp runtime binaries, and `llama-moe-split` for MoE shard generation.

Bundle naming now follows the same convention everywhere:

- macOS bundles package `rpc-server-metal` and `llama-server-metal`
- generic Linux bundles package `rpc-server-cpu` and `llama-server-cpu`
- CUDA Linux bundles package `rpc-server-cuda` and `llama-server-cuda`
- ROCm Linux bundles package `rpc-server-rocm` and `llama-server-rocm`
- Vulkan Linux bundles package `rpc-server-vulkan` and `llama-server-vulkan`

### 4. Smoke test the bundle

```bash
mkdir /tmp/test-bundle && tar xzf /tmp/mesh-bundle.tar.gz -C /tmp/test-bundle --strip-components=1
/tmp/test-bundle/mesh-llm --model Qwen2.5-3B
# Should download model, start solo, API on :9337, console on :3131
# Hit http://localhost:9337/v1/chat/completions to verify inference works
# Ctrl+C to stop
rm -rf /tmp/test-bundle
```

### 5. Commit, tag, push

```bash
just release-version v0.X.0
git add -A && git commit -m "v0.X.0: <summary>"
git tag v0.X.0
git push origin main --tags
```

### 6. Let GitHub Actions build and publish the release

Pushing a `v*` tag triggers `.github/workflows/release.yml`, which:

- builds release bundles on macOS, Linux CPU, Linux CUDA, Linux ROCm, and Linux Vulkan
- uploads versioned assets such as `mesh-llm-v0.X.0-aarch64-apple-darwin.tar.gz`
- uploads stable `latest` assets such as `mesh-llm-x86_64-unknown-linux-gnu.tar.gz`
- uploads CUDA-specific Linux assets such as `mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz`
- uploads ROCm-specific Linux assets such as `mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz`
- uploads Vulkan-specific Linux assets such as `mesh-llm-x86_64-unknown-linux-gnu-vulkan.tar.gz`
- keeps the legacy macOS `mesh-bundle.tar.gz` asset available for direct archive installs
- creates the GitHub release automatically with generated notes

### 7. Verify the release assets

After the workflow finishes, verify:

- `mesh-bundle.tar.gz` still exists for direct macOS archive installs
- `mesh-llm-aarch64-apple-darwin.tar.gz` exists
- `mesh-llm-x86_64-unknown-linux-gnu.tar.gz` exists
- `mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz` exists
- `mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz` exists
- `mesh-llm-x86_64-unknown-linux-gnu-vulkan.tar.gz` exists

## Notes

- The unversioned asset name `mesh-bundle.tar.gz` is still kept for compatibility with direct archive installs.
- The default Linux release bundle is a generic CPU build.
- Release bundles use flavor-specific `rpc-server-<flavor>` and `llama-server-<flavor>` names so multiple flavors can coexist in one install directory. Use `mesh-llm --llama-flavor <flavor>` to force a specific pair.
- The CUDA Linux release bundle is built in CI with an explicit multi-arch `CMAKE_CUDA_ARCHITECTURES` list and is not runtime-tested during the workflow.
- The ROCm and Vulkan Linux release bundles are compile-tested in CI, but not runtime-tested against real GPUs during the workflow.
- `codesign` and `xattr` may be needed on the receiving machine if macOS Gatekeeper blocks unsigned binaries:
  ```bash
  codesign -s - /usr/local/bin/mesh-llm /usr/local/bin/rpc-server /usr/local/bin/llama-server /usr/local/bin/llama-moe-split
  xattr -cr /usr/local/bin/mesh-llm /usr/local/bin/rpc-server /usr/local/bin/llama-server /usr/local/bin/llama-moe-split
  ```
