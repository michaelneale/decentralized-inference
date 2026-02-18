# Releasing mesh-llm

## Prerequisites

- `just` installed (`brew install just`)
- `gh` CLI authenticated (`gh auth status`)
- llama.cpp fork cloned (`just build` does this automatically)

## Steps

### 1. Build everything fresh

```bash
just build
```

This clones/updates the llama.cpp fork if needed, builds with `-DGGML_METAL=ON -DGGML_RPC=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF`, and builds the Rust mesh-llm binary.

### 2. Verify no homebrew dependencies

```bash
otool -L llama.cpp/build/bin/llama-server | grep -v /System | grep -v /usr/lib
otool -L llama.cpp/build/bin/rpc-server | grep -v /System | grep -v /usr/lib
otool -L mesh-llm/target/release/mesh-llm | grep -v /System | grep -v /usr/lib
```

Each should only show the binary name — no `/opt/homebrew/` paths.

### 3. Create the bundle

```bash
just bundle
```

Creates `/tmp/mesh-bundle.tar.gz` containing `mesh-llm`, `rpc-server`, `llama-server`.

### 4. Smoke test the bundle

```bash
mkdir /tmp/test-bundle && tar xzf /tmp/mesh-bundle.tar.gz -C /tmp/test-bundle --strip-components=1
/tmp/test-bundle/mesh-llm --model Qwen2.5-3B --console
# Should download model, start solo, API on :9337, console on :3131
# Hit http://localhost:9337/v1/chat/completions to verify inference works
# Ctrl+C to stop
rm -rf /tmp/test-bundle
```

### 5. Commit, tag, push

```bash
git add -A && git commit -m "v0.X.0: <summary>"
git tag v0.X.0
git push origin main --tags
```

### 6. Create GitHub release

```bash
VERSION=v0.X.0
cp /tmp/mesh-bundle.tar.gz /tmp/mesh-llm-${VERSION}-aarch64-apple-darwin.tar.gz
cp /tmp/mesh-bundle.tar.gz /tmp/mesh-llm-aarch64-apple-darwin.tar.gz

gh release create ${VERSION} \
  /tmp/mesh-llm-${VERSION}-aarch64-apple-darwin.tar.gz \
  /tmp/mesh-llm-aarch64-apple-darwin.tar.gz \
  --title "mesh-llm ${VERSION}" \
  --notes "## What's new

- <changelog here>

### Install (macOS Apple Silicon)

\`\`\`bash
curl -fsSL https://github.com/michaelneale/decentralized-inference/releases/latest/download/mesh-llm-aarch64-apple-darwin.tar.gz | tar xz && sudo mv mesh-bundle/* /usr/local/bin/
\`\`\`
"
```

Two assets are uploaded: one with the version in the name (for pinning), one without (for the `latest` URL used in the README).

### 7. Verify the install one-liner works

```bash
curl -fsSL https://github.com/michaelneale/decentralized-inference/releases/latest/download/mesh-llm-aarch64-apple-darwin.tar.gz | tar xz && sudo mv mesh-bundle/* /usr/local/bin/
mesh-llm --model Qwen2.5-3B --console
```

## Notes

- The unversioned asset name (`mesh-llm-aarch64-apple-darwin.tar.gz`) is what the README's install one-liner uses via the `/latest/download/` URL. It must be uploaded with every release.
- `codesign` and `xattr` may be needed on the receiving machine if macOS Gatekeeper blocks unsigned binaries:
  ```bash
  codesign -s - /usr/local/bin/mesh-llm /usr/local/bin/rpc-server /usr/local/bin/llama-server
  xattr -cr /usr/local/bin/mesh-llm /usr/local/bin/rpc-server /usr/local/bin/llama-server
  ```
