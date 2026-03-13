#!/bin/bash
# Multi-turn eval runner using tmux
#
# Usage:
#   ./evals/run-multi.sh mesh scenario-name
#   ./evals/run-multi.sh opus scenario-name
#
# Scenarios are directories under evals/scenarios/ containing a turns.txt file.
# Each line in turns.txt is a message to send. Lines starting with #wait:N
# pause for N seconds (default wait between turns: 30s).
#
# Results go to evals/results/<provider>/<scenario>/

set -e

PROVIDER="${1:?Usage: $0 <mesh|opus> <scenario>}"
SCENARIO="${2:?Usage: $0 <mesh|opus> <scenario>}"

EVALS_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENARIOS_DIR="$EVALS_DIR/scenarios"
RESULTS_DIR="$EVALS_DIR/results"
SESSION="pi-eval-$$"
DEFAULT_WAIT=30

scenario_dir="$SCENARIOS_DIR/$SCENARIO"
result_dir="$RESULTS_DIR/$PROVIDER/$SCENARIO"
turns_file="$scenario_dir/turns.txt"

if [ ! -f "$turns_file" ]; then
    echo "ERROR: No turns.txt in $scenario_dir"
    exit 1
fi

# Fresh working copy
rm -rf "$result_dir"
mkdir -p "$result_dir"

# Copy scenario files (except turns.txt and prompt.txt)
for f in "$scenario_dir"/*; do
    base="$(basename "$f")"
    [ "$base" = "turns.txt" ] && continue
    [ "$base" = "prompt.txt" ] && continue
    cp "$f" "$result_dir/"
done

echo "═══════════════════════════════════════════════════"
echo "  Provider: $PROVIDER | Scenario: $SCENARIO"
echo "  Multi-turn eval via tmux"
echo "═══════════════════════════════════════════════════"

# Build pi command
if [ "$PROVIDER" = "mesh" ]; then
    PI_CMD="pi --provider mesh --model auto --working-dir $result_dir --no-session"
elif [ "$PROVIDER" = "opus" ]; then
    PI_CMD="pi --provider anthropic --model claude-sonnet-4-20250514 --working-dir $result_dir --no-session"
else
    echo "Unknown provider: $PROVIDER"
    exit 1
fi

start_time=$(date +%s)

# Launch pi in tmux — cd to result dir first so file paths resolve
tmux new-session -d -s "$SESSION" -x 200 -y 50
tmux send-keys -t "$SESSION" "cd $result_dir && $PI_CMD" Enter
sleep 5  # let pi start up

# Send each turn
turn_num=0
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [ -z "$line" ] && continue
    [[ "$line" =~ ^#.* ]] && {
        # Handle #wait:N directive
        if [[ "$line" =~ ^#wait:([0-9]+) ]]; then
            wait_secs="${BASH_REMATCH[1]}"
            echo "  ⏳ Waiting ${wait_secs}s..."
            sleep "$wait_secs"
        fi
        continue
    }

    turn_num=$((turn_num + 1))
    echo "  → Turn $turn_num: ${line:0:60}..."
    tmux send-keys -t "$SESSION" "$line" Enter

    # Wait for response
    sleep "$DEFAULT_WAIT"

    # Capture current screen state
    tmux capture-pane -t "$SESSION" -p > "$result_dir/_screen_turn${turn_num}.txt" 2>/dev/null

done < "$turns_file"

# Final capture — get the full scrollback
sleep 5
tmux capture-pane -t "$SESSION" -p -S - > "$result_dir/_output.txt" 2>/dev/null

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# Kill the session
tmux kill-session -t "$SESSION" 2>/dev/null

echo ""
echo "⏱  Completed in ${elapsed}s ($turn_num turns)"
echo "$elapsed" > "$result_dir/_time.txt"
echo "$PROVIDER" > "$result_dir/_provider.txt"
echo "$SCENARIO" > "$result_dir/_scenario.txt"
echo "$turn_num" > "$result_dir/_turns.txt"

echo "📁 Files in result dir:"
ls -la "$result_dir/" | grep -v "^total" | grep -v "^\."
echo ""
echo "═══════════════════════════════════════════════════"
