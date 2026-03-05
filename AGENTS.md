# Agent Notes

## Repo Overview

This repo (`decentralized-inference`) contains mesh-llm — a Rust binary that pools GPUs over QUIC for distributed LLM inference using llama.cpp.

## Key Docs

| Doc | What it covers |
|---|---|
| `README.md` | Usage, install, CLI flags, examples |
| `CONTRIBUTING.md` | Build from source, dev workflow, UI dev |
| `RELEASE.md` | Release process (build, bundle, tag, GitHub release) |
| `ROADMAP.md` | Future directions |
| `PLAN.md` | Historical design notes and benchmarks |
| `mesh-llm/AGENTS.md` | **Start here for code changes** — project structure, key files, build, test, deploy checklist |
| `mesh-llm/TODO.md` | Current work items and backlog |
| `mesh-llm/README.md` | Rust crate overview and file map |
| `mesh-llm/docs/DESIGN.md` | Architecture, protocols, features |
| `mesh-llm/docs/TESTING.md` | Test playbook, scenarios, remote deploy |
| `MoE_PLAN.md` | MoE expert sharding design |
| `MoE_DEPLOY_DESIGN.md` | MoE auto-deploy UX |
| `MoE_SPLIT_REPORT.md` | MoE splitting validation results |
| `fly/README.md` | Fly.io deployment (console + API apps) |
| `relay/README.md` | Self-hosted iroh relay on Fly |

## Building

Always use `just`. Never build manually.

```bash
just build    # llama.cpp fork + mesh-llm + UI
just bundle   # portable tarball
just auto     # build + stop + start with --auto
```

See `CONTRIBUTING.md` for details.

## Credentials

Test machine IPs, SSH details, and passwords are in `~/Documents/private-note.txt` (outside the repo). **Never commit credentials to any tracked file.**

## What NOT to add

- **No `api_key_token` feature** — explicitly rejected, removed in v0.26.0
- **No credentials in tracked files** — IPs, passwords, SSH commands go in `private-note.txt` only
