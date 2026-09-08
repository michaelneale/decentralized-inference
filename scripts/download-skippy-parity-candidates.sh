#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${SKIPPY_PARITY_MANIFEST:-$ROOT/docs/skippy/llama-parity-candidates.json}"
MODEL_MANIFEST="${SKIPPY_PARITY_MODEL_MANIFEST:-$ROOT/ci/model-artifacts/manifests/skippy-parity.json}"
STATUSES="${SKIPPY_PARITY_DOWNLOAD_STATUSES:-needs_candidate,candidate_multimodal,package_or_remote_only}"
PRIORITIES="${SKIPPY_PARITY_DOWNLOAD_PRIORITIES:-p0,p1}"
DRY_RUN=0

usage() {
  cat >&2 <<'EOF'
usage: scripts/download-skippy-parity-candidates.sh [--dry-run] [--status CSV] [--priority CSV]

Downloads Hugging Face GGUF/package artifacts for Skippy parity rows that still
need certification evidence. By default this includes:

  needs_candidate,candidate_multimodal,package_or_remote_only

and only P0/P1 popularity-priority rows.

Candidate rows may carry an immutable revision and per-file integrity records.
The downloader pins those revisions and verifies every downloaded model and
projector file before returning success.

Environment:
  SKIPPY_PARITY_MANIFEST=/path/to/llama-parity-candidates.json
  SKIPPY_PARITY_DOWNLOAD_STATUSES=needs_candidate,candidate_multimodal
  SKIPPY_PARITY_DOWNLOAD_PRIORITIES=p0,p1

Examples:
  scripts/download-skippy-parity-candidates.sh --dry-run
  scripts/download-skippy-parity-candidates.sh
  scripts/download-skippy-parity-candidates.sh --status needs_candidate
  scripts/download-skippy-parity-candidates.sh --priority p0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --status|--statuses)
      STATUSES="$2"
      shift 2
      ;;
    --priority|--priorities)
      PRIORITIES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is required. Install from https://hf.co/cli/install.sh" >&2
  exit 1
fi

python3 - "$MANIFEST" "$MODEL_MANIFEST" "$STATUSES" "$PRIORITIES" "$DRY_RUN" <<'PY'
from __future__ import annotations

import json
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
model_manifest = Path(sys.argv[2])
statuses = {status.strip() for status in sys.argv[3].split(",") if status.strip()}
priorities = {priority.strip() for priority in sys.argv[4].split(",") if priority.strip()}
dry_run = sys.argv[5] == "1"

data = json.loads(manifest.read_text())
registered = {
    row["id"]: row
    for row in json.loads(model_manifest.read_text()).get("artifacts", [])
}
priority_lookup = {}
for priority in ("p0", "p1", "p2"):
    group = data.get("support_priority", {}).get(priority, {})
    for llama_model in group.get("llama_models", []):
        priority_lookup[("llama_model", llama_model)] = priority
    for family in group.get("families", []):
        priority_lookup[("family", family)] = priority

rows = []
missing_repos = []
for item in data.get("candidates", []):
    if item.get("status") not in statuses:
        continue
    priority = (
        priority_lookup.get(("family", item.get("family", "")))
        or priority_lookup.get(("llama_model", item.get("llama_model", "")))
        or "p2"
    )
    if priorities and priority not in priorities:
        continue
    artifact_id = item.get("artifact_id")
    artifact = registered.get(artifact_id) if artifact_id else None
    if artifact_id and artifact is None:
        raise SystemExit(f"registered parity artifact is missing: {artifact_id}")
    if artifact and "manual" not in artifact.get("cadences", []):
        raise SystemExit(
            f"registered parity artifact is not allowed for manual downloads: {artifact_id}"
        )
    repo = artifact.get("repo") if artifact else item.get("repo")
    revision = artifact.get("revision") if artifact else item.get("revision")
    include = artifact.get("files") if artifact else item.get("include", "*.gguf")
    if not repo:
        missing_repos.append(
            {
                "llama_model": item.get("llama_model", ""),
                "family": item.get("family", ""),
                "status": item.get("status", ""),
                "priority": priority,
            }
        )
        continue
    includes = include if isinstance(include, list) else [include]
    rows.append(
        {
            "llama_model": item.get("llama_model", ""),
            "family": item.get("family", ""),
            "status": item.get("status", ""),
            "priority": priority,
            "repo": repo,
            "revision": revision,
            "includes": includes,
            "integrity": (
                artifact.get("file_integrity")
                if artifact
                else item.get("file_integrity")
            ),
        }
    )

rows.sort(key=lambda row: (row["priority"], row["status"], row["llama_model"], row["family"], row["repo"]))

if not rows:
    print(f"No manifest rows with repos matched statuses: {', '.join(sorted(statuses))}")
    if missing_repos:
        print()
        print("Rows still needing a repo/include target:")
        for row in sorted(
            missing_repos,
            key=lambda row: (row["priority"], row["status"], row["llama_model"], row["family"]),
        ):
            print(
                f"  - {row['llama_model']} / {row['family']} "
                f"({row['priority']}, {row['status']})"
            )
    raise SystemExit(0)

print(f"Manifest: {manifest}")
print(f"Statuses: {', '.join(sorted(statuses))}")
print(f"Priorities: {', '.join(sorted(priorities)) if priorities else 'all'}")
print(f"Download targets: {len(rows)}")
if missing_repos:
    print(f"Rows still needing a repo/include target: {len(missing_repos)}")
print()

skipped = []
for row in rows:
    cmd = ["hf", "download", row["repo"]]
    if row["revision"]:
        cmd.extend(row["includes"])
        cmd.extend(["--revision", row["revision"]])
    else:
        for pattern in row["includes"]:
            cmd.extend(["--include", pattern])
    label = f"{row['llama_model']} / {row['family']} ({row['priority']}, {row['status']})"
    print(f"# {label}")
    print(" ".join(shlex.quote(part) for part in cmd))
    if not dry_run:
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, text=True)
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.returncode != 0:
            skipped.append((label, row["repo"], row["includes"], result.returncode))
            print(
                f"skipping {label}: hf download failed with exit {result.returncode}",
                file=sys.stderr,
            )
        elif row["integrity"]:
            candidates = [Path(line.removeprefix("path=")) for line in result.stdout.splitlines()]
            resolved = next((path for path in reversed(candidates) if path.exists()), None)
            if resolved is None:
                raise SystemExit(f"unable to resolve downloaded path for registered artifact: {label}")
            root = resolved if resolved.is_dir() else resolved.parent
            for name in row["includes"]:
                path = root / name
                record = row["integrity"][name]
                if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                    raise SystemExit(f"registered artifact size mismatch: {path}")
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != record["blob_id"]:
                    raise SystemExit(f"registered artifact SHA-256 mismatch: {path}")
                print(f"verified {path} ({record['size_bytes']} bytes)")
    print()

if skipped:
    print("Needs ungating or replacement:", file=sys.stderr)
    for label, repo, includes, code in skipped:
        patterns = ", ".join(includes)
        print(f"  - {label}: {repo} --include {patterns} (exit {code})", file=sys.stderr)

if missing_repos:
    print("Rows still needing a repo/include target:", file=sys.stderr)
    for row in sorted(
        missing_repos,
        key=lambda row: (row["priority"], row["status"], row["llama_model"], row["family"]),
    ):
        print(
            f"  - {row['llama_model']} / {row['family']} "
            f"({row['priority']}, {row['status']})",
            file=sys.stderr,
        )
PY
