#!/usr/bin/env python3
"""Require native model-offload evidence after the caller's inference probes.

Read the exact process's raw native log, not filtered JSON (which can discard
load_tensors summaries), and never search another PID or a previous smoke run.
This is a functional offload check, not a kernel-utilization benchmark.
"""
import argparse
import json
from pathlib import Path
import re
import sys


# llama-model.cpp logs these INFO lines after buffer allocation. CPU_Host CUDA
# staging buffers and device discovery are deliberately not GPU weight evidence.
OFFLOAD = re.compile(r"^(?:llm_)?load_tensors: offloaded (\d+)/(\d+) layers to GPU$")
CUDA_BUFFER = re.compile(
    r"^(?:llm_)?load_tensors:\s+CUDA0 model buffer size =\s+([0-9]+(?:\.[0-9]+)?) MiB$"
)


def verify_cuda_offload(lines):
    """Fail closed on absent, zero, malformed or contradictory load evidence."""
    counts = []
    buffers = []
    evidence = []
    for raw in lines:
        line = raw.strip()
        match = OFFLOAD.fullmatch(line)
        if match:
            loaded, total = map(int, match.groups())
            if not 0 < loaded <= total:
                raise ValueError("zero or invalid GPU layer-offload count")
            counts.append((loaded, total))
            evidence.append(line)
        match = CUDA_BUFFER.fullmatch(line)
        if match:
            size = float(match[1])
            if size <= 0:
                raise ValueError("zero CUDA0 model buffer")
            buffers.append(size)
            evidence.append(line)
    if not counts or not buffers:
        raise ValueError("missing positive GPU layer-offload and CUDA0 model-buffer evidence")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("CPU", "CUDA0", "Vulkan0", "ROCm0", "MTL0"))
    parser.add_argument("--native-log", required=True, type=Path)
    args = parser.parse_args()
    if args.device != "CUDA0":
        print(f"{args.device} smoke: CUDA offload qualification not requested")
        return 0
    try:
        with args.native_log.open(encoding="utf-8") as log:
            evidence = verify_cuda_offload(log)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"CUDA smoke offload failed ({args.native_log}): {error}", file=sys.stderr)
        return 1
    print(json.dumps({"check": "cuda_model_offload", "device": args.device,
                      "native_log": str(args.native_log), "evidence": evidence}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
