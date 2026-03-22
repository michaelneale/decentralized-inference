#!/bin/bash
set -euo pipefail

# MoM Agent Eval: Compare Claude vs mesh model on agentic coding tasks
# Uses pi -p as the agent harness, same approach as Open Model Gym

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/agent-eval-$(date +%Y%m%d_%H%M%S)"
PI_CONFIG_DIR="$RESULTS_DIR/.pi-config"
LLAMA_SERVER="$SCRIPT_DIR/../llama.cpp/build/bin/llama-server"
MODEL_PATH="$HOME/.models/Qwen3-8B-Q4_K_M.gguf"
LLAMA_PORT=8081
MESH_PORT=9337
TIMEOUT=300  # 5 min per run

mkdir -p "$RESULTS_DIR" "$PI_CONFIG_DIR"

# ── Setup scenarios ──

setup_file_editing() {
    local workdir="$1"
    mkdir -p "$workdir/src/models" "$workdir/src/utils"
    
    cat > "$workdir/Cargo.toml" << 'EOF'
[package]
name = "user-service"
version = "0.1.0"
edition = "2021"
EOF

    cat > "$workdir/src/main.rs" << 'EOF'
mod models;
mod utils;

use models::user::User;

fn main() {
    let user = User::new("Alice", "Smith", "alice@example.com");
    println!("Created user: {}", user.email());
}
EOF

    cat > "$workdir/src/models/mod.rs" << 'EOF'
pub mod user;
EOF

    cat > "$workdir/src/models/user.rs" << 'EOF'
pub struct User {
    first_name: String,
    last_name: String,
    email: String,
}

impl User {
    pub fn new(first_name: &str, last_name: &str, email: &str) -> Self {
        Self {
            first_name: first_name.to_string(),
            last_name: last_name.to_string(),
            email: email.to_string(),
        }
    }

    pub fn email(&self) -> &str {
        &self.email
    }

    pub fn first_name(&self) -> &str {
        &self.first_name
    }

    pub fn last_name(&self) -> &str {
        &self.last_name
    }
}
EOF

    cat > "$workdir/src/utils/mod.rs" << 'EOF'
pub mod formatting;
EOF

    cat > "$workdir/src/utils/formatting.rs" << 'EOF'
pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
EOF
}

validate_file_editing() {
    local workdir="$1"
    local pass=0
    local fail=0
    
    # Check display_name method exists
    if grep -qE 'fn\s+display_name' "$workdir/src/models/user.rs" 2>/dev/null; then
        echo "  ✅ display_name() added"
        ((pass++))
    else
        echo "  ❌ display_name() NOT found"
        ((fail++))
    fi

    # Check return type
    if grep -qE 'display_name.*->.*String|display_name.*->.*str' "$workdir/src/models/user.rs" 2>/dev/null; then
        echo "  ✅ has return type"
        ((pass++))
    else
        echo "  ❌ missing return type"
        ((fail++))
    fi

    # Check original methods preserved
    if grep -q 'pub fn email' "$workdir/src/models/user.rs" 2>/dev/null; then
        echo "  ✅ email() preserved"
        ((pass++))
    else
        echo "  ❌ email() missing"
        ((fail++))
    fi

    if grep -q 'pub fn first_name' "$workdir/src/models/user.rs" 2>/dev/null; then
        echo "  ✅ first_name() preserved"
        ((pass++))
    else
        echo "  ❌ first_name() missing"
        ((fail++))
    fi

    # Check it compiles
    if (cd "$workdir" && cargo build 2>/dev/null); then
        echo "  ✅ cargo build succeeds"
        ((pass++))
    else
        echo "  ❌ cargo build FAILS"
        ((fail++))
    fi

    echo "  Score: $pass/$((pass + fail))"
    return $fail
}

PROMPT='The User struct in user.rs is missing a display_name() method. Add a method that returns the full name formatted as "first_name last_name".'

# ── Pi config for mesh model ──

setup_mesh_pi_config() {
    # models.json for the local llama-server
    cat > "$PI_CONFIG_DIR/models.json" << EOF
{
  "providers": {
    "mesh-local": {
      "baseUrl": "http://localhost:$LLAMA_PORT/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "compat": {
        "supportsUsageInStreaming": false,
        "maxTokensField": "max_tokens",
        "supportsDeveloperRole": false
      },
      "models": [{
        "id": "Qwen3-8B-Q4_K_M.gguf",
        "name": "Qwen3-8B",
        "reasoning": false,
        "input": ["text"],
        "contextWindow": 16384,
        "maxTokens": 4096
      }]
    },
    "mesh": {
      "baseUrl": "http://localhost:$MESH_PORT/v1",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "compat": {
        "supportsUsageInStreaming": false,
        "maxTokensField": "max_tokens",
        "supportsDeveloperRole": false
      },
      "models": [
        {
          "id": "MiniMax-M2.5-Q4_K_M",
          "name": "MiniMax-253B",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 32768,
          "maxTokens": 4096
        },
        {
          "id": "moa",
          "name": "MoA (mixture)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 32768,
          "maxTokens": 4096
        }
      ]
    }
  }
}
EOF

    # Copy auth.json for API keys (Claude needs this)
    if [ -f "$HOME/.pi/agent/auth.json" ]; then
        cp "$HOME/.pi/agent/auth.json" "$PI_CONFIG_DIR/auth.json"
    fi

    # Write clean settings.json WITHOUT pi-mcp-adapter (npm install hangs)
    cat > "$PI_CONFIG_DIR/settings.json" << 'SETEOF'
{
  "packages": [],
  "defaultProvider": "anthropic",
  "defaultModel": "claude-sonnet-4-20250514",
  "defaultThinkingLevel": "off"
}
SETEOF

    # Empty mcp.json (no MCP servers needed for this test)
    cat > "$PI_CONFIG_DIR/mcp.json" << 'EOF'
{"mcpServers": {}}
EOF
}

# ── Run a single eval ──

run_eval() {
    local name="$1"
    local provider="$2" 
    local model="$3"
    local workdir="$RESULTS_DIR/$name"
    
    mkdir -p "$workdir"
    setup_file_editing "$workdir"
    
    # Write prompt to file (avoid shell escaping issues)
    echo "$PROMPT" > "$workdir/.prompt.txt"
    
    echo ""
    echo "━━━ $name: $provider/$model ━━━"
    echo ""
    
    local start=$(date +%s)
    
    # Run pi with isolated config
    cd "$workdir"
    PI_CODING_AGENT_DIR="$PI_CONFIG_DIR" \
    timeout "$TIMEOUT" \
    pi -p --no-session --no-skills \
        --provider "$provider" --model "$model" \
        "$(cat "$workdir/.prompt.txt")" \
        2>"$workdir/stderr.log" \
        > "$workdir/output.log" \
    || true  # Don't fail on non-zero exit
    cd "$SCRIPT_DIR"
    
    local end=$(date +%s)
    local elapsed=$((end - start))
    
    echo "  Time: ${elapsed}s"
    echo "  Output: $(wc -c < "$workdir/output.log" | tr -d ' ') bytes"
    
    # Show first few lines of output
    echo "  Preview:"
    head -5 "$workdir/output.log" | sed 's/^/    /'
    echo ""
    
    # Validate
    echo "  Validation:"
    validate_file_editing "$workdir"
    
    # Save metadata
    cat > "$workdir/meta.json" << EOF
{"name": "$name", "provider": "$provider", "model": "$model", "elapsed_s": $elapsed}
EOF
}

# ── Main ──

echo "============================================================"
echo "MoM Agent Eval: Claude vs Mesh Model"
echo "Results: $RESULTS_DIR"
echo "============================================================"

# Set up pi config
setup_mesh_pi_config

# Start local llama-server for mesh model
echo ""
echo "Starting Qwen3-8B on :$LLAMA_PORT..."
"$LLAMA_SERVER" -m "$MODEL_PATH" --port "$LLAMA_PORT" -c 16384 -ngl 99 -np 1 --no-webui --host 127.0.0.1 > "$RESULTS_DIR/llama-server.log" 2>&1 &
LLAMA_PID=$!

cleanup() {
    kill "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
    echo "llama-server stopped."
}
trap cleanup EXIT

# Wait for server
for i in $(seq 1 60); do
    if curl -s "http://localhost:$LLAMA_PORT/v1/models" >/dev/null 2>&1; then
        echo "  ✅ Qwen3-8B ready"
        break
    fi
    sleep 1
done

# Check mesh is available
if curl -s "http://localhost:$MESH_PORT/v1/models" | grep -q "moa"; then
    MESH_AVAILABLE=1
    echo "  ✅ Mesh available with MoA"
else
    MESH_AVAILABLE=0
    echo "  ⚠️  Mesh not available — skipping MiniMax and MoA evals"
fi

# Run evals
run_eval "claude" "anthropic" "claude-sonnet-4-20250514"

if [ "$MESH_AVAILABLE" = "1" ]; then
    run_eval "minimax-solo" "mesh" "MiniMax-M2.5-Q4_K_M"
    run_eval "moa" "mesh" "moa"
    run_eval "best-of-n" "mesh" "best-of-n"
fi

# Summary
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
for d in "$RESULTS_DIR"/claude "$RESULTS_DIR"/minimax-solo "$RESULTS_DIR"/moa "$RESULTS_DIR"/best-of-n; do
    name=$(basename "$d")
    if [ -f "$d/meta.json" ]; then
        elapsed=$(python3 -c "import json; print(json.load(open('$d/meta.json'))['elapsed_s'])")
        output_size=$(wc -c < "$d/output.log" | tr -d ' ')
        echo "  $name: ${elapsed}s, ${output_size} bytes output"
    fi
done
