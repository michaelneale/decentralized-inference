#!/usr/bin/env bash
# detect-cuda-arch.sh — detect CUDA compute capability for CMAKE_CUDA_ARCHITECTURES
#
# Outputs a semicolon-separated list of SM values for all installed GPUs
# (e.g. "120;86" for a system with RTX 5090 + RTX 3080).
# Compute cap 8.7 → 87,  9.0 → 90,  12.0 → 120  (dot removed, no sm_ prefix).
#
# Exit codes:
#   0 — detected; SM list printed to stdout
#   1 — unable to detect; error with manual override instructions on stderr

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Collect unique SM values into an array, then join with semicolons.
SM_VALUES=()

add_sm() {
    local sm="$1"
    for existing in "${SM_VALUES[@]:-}"; do
        [[ "$existing" == "$sm" ]] && return
    done
    SM_VALUES+=("$sm")
}

# ── Check if any NVIDIA GPU exists first ───────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null && [[ ! -d /proc/driver/nvidia/gpus ]]; then
    echo "cpu"
    exit 0
fi

# ── Strategy 1: nvidia-smi (fastest and most reliable) ────────────────────────
if command -v nvidia-smi &>/dev/null; then
    # Query ALL GPUs (remove head -1 to capture every device)
    while IFS= read -r RAW; do
        RAW="${RAW//[[:space:]]/}"
        if [[ -n "$RAW" && "$RAW" =~ ^[0-9]+\.[0-9]+$ ]]; then
            add_sm "${RAW//./}"
        fi
    done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)

    if [[ ${#SM_VALUES[@]} -gt 0 ]]; then
        (IFS=';'; echo "${SM_VALUES[*]}")
        exit 0
    fi

    echo "WARN: nvidia-smi present but compute_cap query returned nothing" >&2
fi

# ── Strategy 2: CUDA deviceQuery binary ───────────────────────────────────────
for DQ in \
    /usr/local/cuda/extras/demo_suite/deviceQuery \
    /usr/local/cuda/samples/bin/aarch64/linux/release/deviceQuery \
    /usr/local/cuda/samples/bin/x86_64/linux/release/deviceQuery \
    deviceQuery; do
    if [[ -x "$DQ" ]] || command -v "$DQ" &>/dev/null 2>&1; then
        # Collect ALL capability lines, not just the first
        while IFS= read -r CAP; do
            CAP="${CAP//[[:space:]]/}"
            if [[ -n "$CAP" ]]; then
                add_sm "${CAP//./}"
            fi
        done < <("$DQ" 2>/dev/null \
              | grep 'CUDA Capability Major/Minor version number' \
              | grep -oP '\d+\.\d+')

        if [[ ${#SM_VALUES[@]} -gt 0 ]]; then
            (IFS=';'; echo "${SM_VALUES[*]}")
            exit 0
        fi
    fi
done

# ── Strategy 3: /proc driver GPU model lookup ──────────────────────────────────
# Source: https://developer.nvidia.com/cuda-gpus (verified 2026-03-25)
# Consumer Blackwell (SM 12.0) and datacenter Blackwell (SM 10.0) are
# architecturally distinct — kernels are NOT cross-compatible between them.
if [[ -d /proc/driver/nvidia/gpus ]]; then
    declare -A SM_MAP=(
        # Blackwell Ultra datacenter (SM 10.3)
        ["GB300"]="103" ["B300"]="103"
        # Blackwell datacenter (SM 10.0)
        ["GB200"]="100" ["B200"]="100" ["B100"]="100"
        # Blackwell consumer + pro (SM 12.0)
        ["RTX 5090"]="120" ["RTX 5080"]="120"
        ["RTX 5070"]="120" ["RTX 5060"]="120" ["RTX 5050"]="120"
        ["RTX PRO 6000 Blackwell"]="120"
        # Grace Blackwell Edge (SM 12.1)
        ["GB10"]="121"
        # Hopper (SM 9.0)
        ["H200"]="90" ["H100"]="90" ["GH200"]="90"
        # Ada Lovelace datacenter (SM 8.9)
        ["L40S"]="89" ["L40"]="89" ["L4"]="89"
        # Ada Lovelace consumer + pro (SM 8.9)
        ["RTX 4090"]="89" ["RTX 4080"]="89"
        ["RTX 4070"]="89" ["RTX 4060"]="89" ["RTX 4050"]="89"
        ["RTX 6000 Ada"]="89" ["RTX 5000 Ada"]="89"
        ["RTX 4500 Ada"]="89" ["RTX 4000 Ada"]="89"
        # Ampere embedded / Jetson Orin (SM 8.7)
        ["Orin"]="87"
        # Ampere datacenter (SM 8.6)
        ["A40"]="86" ["A16"]="86" ["A10"]="86" ["A2"]="86"
        # Ampere consumer + pro (SM 8.6)
        ["RTX 3090"]="86" ["RTX 3080"]="86"
        ["RTX 3070"]="86" ["RTX 3060"]="86" ["RTX 3050"]="86"
        ["RTX A6000"]="86" ["RTX A5000"]="86"
        ["RTX A4000"]="86" ["RTX A3000"]="86" ["RTX A2000"]="86"
        # Ampere datacenter (SM 8.0)
        ["A100"]="80" ["A30"]="80"
        # Turing consumer + datacenter (SM 7.5)
        ["RTX 2080"]="75" ["RTX 2070"]="75"
        ["RTX 2060"]="75" ["TITAN RTX"]="75" ["T4"]="75"
        # Volta datacenter (SM 7.0)
        ["V100"]="70"
        # Jetson Xavier (SM 7.2)
        ["Xavier"]="72"
    )

    # Iterate ALL GPU info files (not just the first)
    for INFO in /proc/driver/nvidia/gpus/*/information; do
        [[ -f "$INFO" ]] || continue
        MODEL=$(grep -m1 '^Model:' "$INFO" | sed 's/^Model:[[:space:]]*//')
        [[ -z "$MODEL" ]] && continue

        echo "Detected GPU model: $MODEL" >&2
        MATCHED=0
        for SUBSTR in "${!SM_MAP[@]}"; do
            if [[ "$MODEL" == *"$SUBSTR"* ]]; then
                SM="${SM_MAP[$SUBSTR]}"
                echo "Matched '${SUBSTR}' → SM ${SM}" >&2
                add_sm "$SM"
                MATCHED=1
                break
            fi
        done
        if [[ "$MATCHED" -eq 0 ]]; then
            echo "WARN: GPU '$MODEL' not in lookup table" >&2
        fi
    done

    if [[ ${#SM_VALUES[@]} -gt 0 ]]; then
        (IFS=';'; echo "${SM_VALUES[*]}")
        exit 0
    fi
fi

# ── All strategies exhausted ──────────────────────────────────────────────────
die "Could not detect CUDA compute capability automatically.
Pass the arch explicitly:
  just build cuda_arch=<SM>   e.g.  just build cuda_arch=87

For multi-GPU systems, separate with semicolons:
  just build cuda_arch='120;86'

Common values (source: https://developer.nvidia.com/cuda-gpus):
  120  RTX 5090/5080/5070/5060 (consumer Blackwell)
  103  GB300/B300 (Blackwell Ultra datacenter)
  100  B200/B100/GB200 (datacenter Blackwell — NOT compatible with SM 120)
   90  H100/H200/GH200 (Hopper)
   89  RTX 4090/4080/4070/4060, L4, L40, L40S (Ada Lovelace)
   87  Jetson AGX Orin / IGX Orin (Ampere embedded)
   86  RTX 3090/3080/3070/3060, A10, A40 (Ampere)
   80  A100, A30 (Ampere datacenter)
   75  T4, RTX 2080/2070/2060 (Turing)
   70  V100 (Volta)"
