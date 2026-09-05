#!/usr/bin/env bash
set -euo pipefail

# affected-crates.sh: Compute which Rust crates are affected by changed files
# Usage:
#   bash scripts/affected-crates.sh --stdin < changed_files.txt
#   bash scripts/affected-crates.sh file1.rs file2.rs ...

# Hardcoded workspace members (fallback for fail-open)
WORKSPACE_MEMBERS=(
  "mesh-llm"
  "mesh-llm-build-info"
  "mesh-llm-cli"
  "mesh-llm-commands"
  "mesh-llm-config"
  "mesh-llm-events"
  "mesh-llm-gpu-bench"
  "mesh-llm-host-runtime"
  "mesh-llm-hardware-profile"
  "mesh-llm-identity"
  "mesh-llm-log-store"
  "mesh-llm-native-runtime"
  "mesh-llm-protocol"
  "mesh-llm-release-footer"
  "mesh-llm-routing"
  "mesh-llm-runtime-install"
  "mesh-llm-sdk"
  "mesh-llm-guardrails"
  "mesh-llm-system"
  "mesh-llm-types"
  "mesh-llm-console-server"
  "mesh-llm-embedded-runtime"
  "mesh-llm-tui"
  "mesh-llm-ui"
  "mesh-llm-plugin"
  "mesh-llm-skills"
  "mesh-llm-plugin-manager"
  "mesh-llm-client"
  "mesh-mixture-of-agents"
  "mesh-native-serving-plugin-api"
  "mesh-native-serving-plugin-host"
  "mesh-llm-api-client"
  "mesh-llm-api-server"
  "mesh-llm-node"
  "mesh-llm-ffi"
  "mesh-llm-nodejs"
  "mesh-llm-test-harness"
  "model-ref"
  "model-artifact"
  "model-hf"
  "model-resolver"
  "skippy-protocol"
  "skippy-tokenizer"
  "skippy-coordinator"
  "skippy-topology"
  "skippy-cache"
  "skippy-metrics"
  "openai-frontend"
  "skippy-ffi"
  "skippy-model"
  "skippy-package-format"
  "skippy-runtime"
  "skippy-scheduler"
  "skippy-server"
  "metrics-server"
  "skippy-model-package"
  "skippy-quantize"
  "model-package"
  "skippy-correctness"
  "llama-quant-ffi"
  "llama-spec-bench"
  "skippy-bench"
  "skippy-prompt"
  "xtask"
)

is_website_input() {
  local file="$1"

  [[ "$file" =~ ^website/ ]] || \
    [[ "$file" =~ ^install\.sh$ ]] || \
    [[ "$file" =~ ^install\.ps1$ ]] || \
    [[ "$file" =~ ^docs/(index\.html|CNAME|install\.sh|install\.ps1|mesh-llm-logo\.svg)$ ]] || \
    [[ "$file" =~ ^docs/(assets|catalog|docs|pagefind)(/|$) ]]
}

FAIL_OPEN_UI_CHANGED=false
FAIL_OPEN_WEBSITE_CHANGED=false

# Fail-open handler: emit all_rust=true with full workspace
fail_open() {
  local exit_code=$?
  echo "WARNING: affected-crates.sh encountered an error (exit=$exit_code), falling back to all_rust=true" >&2

  # Build full workspace list as JSON array
  local all_crates_json="["
  for i in "${!WORKSPACE_MEMBERS[@]}"; do
    if [ "$i" -gt 0 ]; then
      all_crates_json+=","
    fi
    all_crates_json+="\"${WORKSPACE_MEMBERS[$i]}\""
  done
  all_crates_json+="]"

  # Emit fallback JSON
  cat <<EOF
{
  "affected": $all_crates_json,
  "test_crates": [],
  "all_rust": true,
  "ui_changed": $([[ "$FAIL_OPEN_UI_CHANGED" == true ]] && echo "true" || echo "false"),
  "website_changed": $([[ "$FAIL_OPEN_WEBSITE_CHANGED" == true ]] && echo "true" || echo "false")
}
EOF
  exit 0
}

trap 'fail_open' ERR

array_contains() {
  local needle="$1"
  shift

  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done

  return 1
}

main() {
  # Parse input: --stdin or positional args
  local -a changed_files=()

  if [[ "${1:-}" == "--stdin" ]]; then
    while IFS= read -r line; do
      [[ -n "$line" ]] && changed_files+=("$line")
    done
  else
    changed_files=("$@")
  fi

  # Check for escalation paths or __force_all__ sentinel
  local escalate=false
  local ui_changed=false
  local website_changed=false
  FAIL_OPEN_UI_CHANGED=false
  FAIL_OPEN_WEBSITE_CHANGED=false

  for file in "${changed_files[@]}"; do
    # Public website changed detection. These paths are build inputs for the
    # Eleventy/Tailwind/Pagefind site or generated website outputs under docs/.
    if is_website_input "$file"; then
      website_changed=true
      FAIL_OPEN_WEBSITE_CHANGED=true
    fi

    # UI changed detection
    if [[ "$file" =~ ^crates/mesh-llm-ui/ ]]; then
      ui_changed=true
      FAIL_OPEN_UI_CHANGED=true
    fi

    # __force_all__ sentinel
    if [[ "$file" == "__force_all__" ]]; then
      escalate=true
      continue
    fi

    # Escalation patterns. Most native llama.cpp inputs are routed by
    # .github/actions/compute-changes as backend builds, not as all-Rust crate
    # test fanout. The upstream pin and the patch queue are the exception: they
    # can change which archives the native build emits, and the static link
    # line in crates/skippy-ffi/build.rs is hand-maintained, so only the Rust
    # test batches can prove the link still closes. Advancing the pin without
    # this escalation is how an undefined hash_sha256_hex reached main.
    if [[ "$file" =~ ^third_party/llama\.cpp/upstream\.txt$ ]] || \
       [[ "$file" =~ ^third_party/llama\.cpp/patches/ ]] || \
       [[ "$file" =~ ^Cargo\.lock$ ]] || \
       [[ "$file" =~ ^Cargo\.toml$ ]] || \
            [[ "$file" =~ ^scripts/(build-llama|prepare-llama|build-linux|build-linux-rocm|build-mac|build-windows|skippy-ci-smoke|ci-install-native-runtime|ci-prepare-native-runtime|ci-smoke-test|ci-compat-smoke|ci-client-auto-test|ci-two-node-client-serving-smoke|ci-two-node-split-smoke)\. ]] || \
       [[ "$file" =~ ^\.github/cache-version\.txt$ ]] || \
       [[ "$file" =~ ^rust-toolchain(\.toml)?$ ]]; then
      escalate=true
    fi
  done

  # If escalation, return all_rust=true
  if [[ "$escalate" == true ]]; then
    local all_crates_json="["
    for i in "${!WORKSPACE_MEMBERS[@]}"; do
      if [ "$i" -gt 0 ]; then
        all_crates_json+=","
      fi
      all_crates_json+="\"${WORKSPACE_MEMBERS[$i]}\""
    done
    all_crates_json+="]"

    cat <<EOF
{
  "affected": $all_crates_json,
  "test_crates": [],
  "all_rust": true,
  "ui_changed": $([[ "$ui_changed" == true ]] && echo "true" || echo "false"),
  "website_changed": $([[ "$website_changed" == true ]] && echo "true" || echo "false")
}
EOF
    return 0
  fi

  # Normal path: use cargo metadata to build crate map and reverse-dep graph

  # Step 1: Get crate → manifest_path mapping (no deps)
  local metadata_no_deps
  metadata_no_deps=$(cargo metadata --format-version=1 --no-deps 2>/dev/null) || fail_open

  local workspace_root
  workspace_root=$(echo "$metadata_no_deps" | jq -r '.workspace_root') || fail_open

  local -A crate_to_dir=()
  while IFS='|' read -r crate_name manifest_path; do
    [[ -z "$crate_name" ]] && continue
    dir="${manifest_path%/Cargo.toml}"
    crate_to_dir["$crate_name"]="$dir"
  done < <(echo "$metadata_no_deps" | jq -r '.packages[] | "\(.name)|\(.manifest_path)"')

  # Step 2: Build reverse-dep graph: for each crate, list which crates depend on it
  local -A reverse_deps=()
  local metadata_json
  metadata_json=$(cargo metadata --format-version=1 2>/dev/null) || fail_open

  # Build name→id mapping
  local -A name_to_id=()
  while IFS='|' read -r name id; do
    [[ -z "$name" ]] && continue
    name_to_id["$name"]="$id"
  done < <(echo "$metadata_json" | jq -r '.packages[] | "\(.name)|\(.id)"')

  # Build reverse deps: for each node, add it as a reverse dep of its dependencies
  while IFS='|' read -r node_id dep_id; do
    [[ -z "$node_id" ]] && continue
    # Find crate name for dep_id
    local dep_name=""
    for name in "${!name_to_id[@]}"; do
      if [[ "${name_to_id[$name]}" == "$dep_id" ]]; then
        dep_name="$name"
        break
      fi
    done
    [[ -z "$dep_name" ]] && continue

    # Find crate name for node_id
    local node_name=""
    for name in "${!name_to_id[@]}"; do
      if [[ "${name_to_id[$name]}" == "$node_id" ]]; then
        node_name="$name"
        break
      fi
    done
    [[ -z "$node_name" ]] && continue

    # Add node_name as reverse dep of dep_name
    if [[ -z "${reverse_deps[$dep_name]:-}" ]]; then
      reverse_deps["$dep_name"]="$node_name"
    else
      reverse_deps["$dep_name"]="${reverse_deps[$dep_name]} $node_name"
    fi
  done < <(echo "$metadata_json" | jq -r '.resolve.nodes[] | "\(.id)|\(.dependencies[]?)"')

  # Step 3: Match changed files to owning crates
  local -a test_crates=()

  for file in "${changed_files[@]}"; do
    # Skip non-Rust files (docs, config, etc.)
    if [[ ! "$file" =~ ^crates/ ]] && [[ ! "$file" =~ ^tools/ ]]; then
      continue
    fi

    # Skip UI crate files (they don't affect Rust builds)
    if [[ "$file" =~ ^crates/mesh-llm-ui/ ]]; then
      continue
    fi

    # Find longest-prefix match in crate_to_dir
    local best_crate=""
    local best_len=0

    for crate_name in "${!crate_to_dir[@]}"; do
      local crate_dir="${crate_to_dir[$crate_name]}"
      # Convert absolute manifest dirs to paths relative to the cargo workspace root.
      local crate_rel="$crate_dir"
      if [[ "$crate_rel" == "$workspace_root" ]]; then
        crate_rel=""
      elif [[ "$crate_rel" == "$workspace_root/"* ]]; then
        crate_rel="${crate_rel#"$workspace_root/"}"
      fi

      if [[ -n "$crate_rel" ]] && { [[ "$file" == "$crate_rel" ]] || [[ "$file" == "$crate_rel/"* ]]; }; then
        local len=${#crate_rel}
        if [[ $len -gt $best_len ]]; then
          best_crate="$crate_name"
          best_len=$len
        fi
      fi
    done

    if [[ -n "$best_crate" ]] && ! array_contains "$best_crate" "${test_crates[@]}"; then
      test_crates+=("$best_crate")
    fi
  done

  # Step 4: BFS from test_crates through reverse-dep graph
  local -a affected=()
  local -a queue=("${test_crates[@]}")
  local -A visited=()

  while [[ ${#queue[@]} -gt 0 ]]; do
    local current="${queue[0]}"
    queue=("${queue[@]:1}")

    if [[ -n "${visited[$current]:-}" ]]; then
      continue
    fi
    visited[$current]=1
    affected+=("$current")

    # Get reverse deps of current
    local deps="${reverse_deps[$current]:-}"
    for dep in $deps; do
      [[ -z "$dep" ]] && continue
      if [[ -z "${visited[$dep]:-}" ]]; then
        queue+=("$dep")
      fi
    done
  done

  # Step 5: Emit JSON output using jq
  jq -n \
    --argjson affected "$(printf '%s\n' "${affected[@]}" | jq -Rs 'split("\n") | map(select(length > 0))')" \
    --argjson test_crates "$(printf '%s\n' "${test_crates[@]}" | jq -Rs 'split("\n") | map(select(length > 0))')" \
    --arg ui_changed "$([[ "$ui_changed" == true ]] && echo "true" || echo "false")" \
    --arg website_changed "$([[ "$website_changed" == true ]] && echo "true" || echo "false")" \
    '{
      affected: $affected,
      test_crates: $test_crates,
      all_rust: false,
      ui_changed: ($ui_changed == "true"),
      website_changed: ($website_changed == "true")
    }'
}

main "$@"
