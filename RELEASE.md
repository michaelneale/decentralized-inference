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

On Windows, use the release-specific recipes directly:

```powershell
just release-build-windows
just release-build-cuda-windows
just release-build-amd-windows
just release-build-vulkan-windows
```

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

On Windows, create release archives directly:

```powershell
just release-bundle-windows v0.X.0
just release-bundle-cuda-windows v0.X.0
just release-bundle-amd-windows v0.X.0
just release-bundle-vulkan-windows v0.X.0
```

Those commands emit `.zip` assets in `dist/` with `mesh-llm.exe`, plus flavor-specific `rpc-server-<flavor>.exe` and `llama-server-<flavor>.exe`.
If optional Windows benchmark binaries such as `membench-fingerprint-cuda.exe` or `membench-fingerprint-hip.exe` are present in `mesh-llm/target/release/`, the PowerShell packager also includes them in the `.zip`.

### 4. Smoke test the bundle

```bash
mkdir /tmp/test-bundle && tar xzf /tmp/mesh-bundle.tar.gz -C /tmp/test-bundle --strip-components=1
/tmp/test-bundle/mesh-llm --model Qwen2.5-3B
# Should download model, start solo, API on :9337, console on :3131
# Hit http://localhost:9337/v1/chat/completions to verify inference works
# Ctrl+C to stop
rm -rf /tmp/test-bundle
```

### 5. Release

```bash
just release v0.X.0
```

Run this from a clean local `main` branch. It bumps the version in source + Cargo manifests, refreshes `Cargo.lock` without upgrading dependencies, commits as `v0.X.0: release`, pushes `main`, and then pushes only the new release tag.

### 6. Let GitHub Actions build and publish the release

Pushing a `v*` tag triggers `.github/workflows/release.yml`, which:

- builds release bundles on macOS, Linux CPU, Linux CUDA, Linux ROCm, Linux Vulkan, and Windows CPU/CUDA/ROCm/Vulkan
- uses hosted `windows-2022` runners for Windows and installs the needed SDKs during the workflow
- uploads versioned assets such as `mesh-llm-v0.X.0-aarch64-apple-darwin.tar.gz`
- uploads stable `latest` assets such as `mesh-llm-x86_64-unknown-linux-gnu.tar.gz`
- uploads CUDA-specific Linux assets such as `mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz`
- uploads ROCm-specific Linux assets such as `mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz`
- uploads Vulkan-specific Linux assets such as `mesh-llm-x86_64-unknown-linux-gnu-vulkan.tar.gz`
- uploads Windows CPU assets such as `mesh-llm-x86_64-pc-windows-msvc.zip`
- uploads Windows CUDA assets such as `mesh-llm-x86_64-pc-windows-msvc-cuda.zip`
- uploads Windows ROCm assets such as `mesh-llm-x86_64-pc-windows-msvc-rocm.zip`
- uploads Windows Vulkan assets such as `mesh-llm-x86_64-pc-windows-msvc-vulkan.zip`
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
- `mesh-llm-x86_64-pc-windows-msvc.zip` exists
- `mesh-llm-x86_64-pc-windows-msvc-cuda.zip` exists
- `mesh-llm-x86_64-pc-windows-msvc-rocm.zip` exists
- `mesh-llm-x86_64-pc-windows-msvc-vulkan.zip` exists

## Notes

- The unversioned asset name `mesh-bundle.tar.gz` is still kept for compatibility with direct archive installs.
- The default Linux release bundle is a generic CPU build.
- Windows source builds exist, and tagged releases now publish Windows CPU/CUDA/ROCm/Vulkan `.zip` assets.
- Windows release artifacts can still be generated locally with the `*-windows` release recipes in `Justfile`.
- Release bundles use flavor-specific `rpc-server-<flavor>` and `llama-server-<flavor>` names so multiple flavors can coexist in one install directory. Use `mesh-llm --llama-flavor <flavor>` to force a specific pair.
- The CUDA Linux release bundle is built in CI with an explicit multi-arch `CMAKE_CUDA_ARCHITECTURES` list and is not runtime-tested during the workflow.
- The ROCm and Vulkan Linux release bundles are compile-tested in CI, but not runtime-tested against real GPUs during the workflow.
- The Windows release workflows are compile-and-package only. They do not run inference tests against real GPUs during the workflow.
- `codesign` and `xattr` may be needed on the receiving machine if macOS Gatekeeper blocks unsigned binaries:
  ```bash
  codesign -s - /usr/local/bin/mesh-llm /usr/local/bin/rpc-server /usr/local/bin/llama-server /usr/local/bin/llama-moe-split
  xattr -cr /usr/local/bin/mesh-llm /usr/local/bin/rpc-server /usr/local/bin/llama-server /usr/local/bin/llama-moe-split
  ```
