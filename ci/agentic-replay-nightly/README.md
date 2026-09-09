---
pretty_name: MeshLLM Agentic Replay Nightly (micstudio)
license: apache-2.0
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/runs/**/*.jsonl
---

# MeshLLM Agentic Replay Nightly

Append-only, machine-readable results from the nightly agentic-replay coding
benchmark on the pinned `micstudio` runner (Apple M3 Ultra, 256 GB, Metal 4).

Every run replays the pinned Thoughtworks agentic-coding trajectory sample
through `mesh-llm serve` for each model in the matrix and appends one immutable
JSONL shard per model/concurrency cohort under `data/runs/<date>/`. The schema
is versioned in `schema.json`. A companion `report.md` shard carries the
human-readable card-format report for each run.

Each model entry pins a Hugging Face commit and SHA-256 digest. The workflow
resolves those exact revisions from the pre-warmed cache and hashes every GGUF
before it starts the replay, so upstream repository changes cannot alter a
benchmark cohort silently.

The dataset contains performance metrics and content-addressed provenance
only: no prompts, completions, model weights, credentials, or local paths.

Regression reports compare only exact cohort keys and require at least three
prior complete runs before classifying drift (bootstrap-then-gate).
