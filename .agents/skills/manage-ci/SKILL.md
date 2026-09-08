---
name: manage-ci
description: Use this skill as the mandatory starting point whenever inspecting, running, debugging, defining, editing, reviewing, or documenting MeshLLM CI/CD. It governs GitHub Actions workflows and local actions, triggers and routing, runners, caches, artifacts, permissions, releases, deployments, and CI infrastructure.
---

# Manage CI

This is the normative CI rule source for MeshLLM. Read it completely before
every CI task.

Read these companion documents before changing CI:

1. `references/current-inventory.md` for the checked-in workflow, runner,
   variable, secret-name, and environment inventory.
2. `ci/ci.md` for the current topology and artifact flow.
3. `.omo/specs/pr-ci-optimization.md` when changing PR/main composition,
   routing, fan-out, or runner-provider policy.

The specification records design, status and acceptance criteria. When
implementation and documentation disagree, inspect the implementation, fix the
owning source, and update the inventory and topology in the same change.

## Required procedure

1. Inspect `git status`, applicable `AGENTS.md` files, the complete workflows,
   reusable workflows, local actions, scripts, and manifests in scope.
2. Classify the change as entrypoint, planner/routing, reusable slice, runner
   policy, cache, artifact producer/consumer, smoke, main, release, deployment,
   maintenance, or infrastructure.
3. Identify the event and trust context, every producer and consumer, required
   permissions and secrets, runner class, cache authority, required check, and
   cancellation behavior.
4. Inspect live GitHub configuration only when the task depends on it. Treat
   repository variables as strings and report inaccessible organization state
   as unverified, never absent.
5. Make the smallest coherent change. Update this skill first if a normative
   rule changes; update the inventory and `ci/ci.md` when facts or topology
   change.
6. Validate the changed contracts and report repository changes separately
   from external GitHub, Depot, runner, secret, or deployment state changes.

## CI architecture contract

### Entrypoints and reusable slices

- Event entrypoints own triggers, concurrency, and top-level permissions.
  Pull requests use separate Quality, Website, Linux, macOS, and Windows
  workflows. Each entry computes the canonical plan, then calls only its
  matching reusable lane so jobs remain attached to a small topic/platform
  run. PR callers use the protected default-branch lane definition; main
  callers use the same-commit local lane definition. Explicit manual-full
  diagnostics alone may use the protected default-branch dispatch controller.
  No path owns duplicated build implementations.
- New PR and main behavior must be implemented in a typed reusable workflow or
  local action consumed by both entrypoints. Do not copy a job between PR and
  main and do not add a third implementation to release.
- A PR-selected slice must run the same commands, build profile, artifact
  contract, and verification as that row on main. PRs reduce work by selecting
  fewer main-representative slices, not by changing what a selected slice
  means.
- Reusable workflow inputs must be typed, bounded, documented, and validated.
  Accept semantic inputs such as platform, architecture, backend, profile, or
  smoke kind. Never accept arbitrary shell, arbitrary workflow path, raw
  `runs-on` JSON, a secret name, or an independent cache-permission flag.
- Pass secrets by name only to the workflow and step that consumes them. Never
  use `secrets: inherit` in CI slice calls.
- Keep reusable-workflow nesting shallow and the slice catalog small. Separate
  topic/platform graphs may be dispatched with native, bounded inputs from the
  protected controller. Each lane owns a platform-local static superset of
  typed calls gated by the checked plan; a platform lane must not call a
  cross-platform reusable workflow with empty placeholder matrices. Do not
  generate workflow YAML at run time.
- Prefer local composite actions for repeated steps inside a job and reusable
  workflows for repeated jobs or job graphs.

### PR workflow visibility and split invariant

- PR validation has exactly five event entrypoints:
  `pr_quality.yml`, `pr_website.yml`, `pr_linux.yml`, `pr_macos.yml`, and
  `pr_windows.yml`. Keep this topic/platform split unless a maintainer
  explicitly changes the architecture contract.
- Each PR entrypoint may call only its matching protected default-branch lane.
  It owns an independent concurrency group and stable `PR / <lane>` result.
  Compose additional work inside that lane's typed reusable graph; do not add
  cross-lane jobs to an entrypoint.
- Native PR visibility is an operability requirement. Reviewers must be able
  to open the PR checks view, select Quality, Website, Linux, macOS, or Windows,
  and drill directly into that lane's jobs and logs. A custom `dispatched`
  placeholder or a separate workflow-dispatch run does not satisfy this
  requirement for PRs.
- Never introduce a monolithic PR workflow or reusable PR composer that calls
  all five lanes in one run. Do not replace the five entrypoints with one
  matrix, one giant dependency graph, or one aggregate workflow whose only
  visible PR result is a dispatch/controller job.
- A reusable-only, no-op filename shim is temporarily allowed when the
  protected pre-migration runner contract requires a deleted entrypoint to be
  present while the migration PR is open. It must have no event trigger, call
  no lane or producer, be documented as removable after merge, and must never
  regain the behavior implied by its legacy filename.
- Do not add `paths` filters to the five PR entrypoints. All five stable results
  must be created for every relevant PR synchronization; the checked planner
  makes an unselected lane skip its expensive work and lets its stable result
  succeed. This prevents required checks from remaining absent or pending.

### Main workflow visibility and split invariant

- Routine main validation has exactly five push entrypoints:
  `main_quality.yml`, `main_website.yml`, `main_linux.yml`, `main_macos.yml`,
  and `main_windows.yml`. Each calls only its matching same-commit reusable
  lane and owns one stable `Main / <lane>` result.
- Native main visibility is an operability requirement. A maintainer must be
  able to open a main push run for one topic/platform and drill directly into
  its jobs and logs. Do not route routine main pushes through a dispatch-only
  controller, synthetic check graph, monolithic matrix, or all-lanes composer.
- The five main workflows must not use path filters or cancel one another or
  older main revisions. Main is exhaustive evidence; supersession cancellation
  remains PR-only. Planner-owned skips and matrix fail-fast behavior must not
  make the top-level main result disappear.
- `ci-control.yml` is reserved for an explicit default-branch manual-full run.
  It may dispatch the five lanes with correlated synthetic checks because the
  operator deliberately chose the detached diagnostic surface. It must not be
  triggered by `push`, `pull_request`, or `workflow_run`.

### Planning and routing

- Compute changed files and the CI plan once per entry graph. Each independent
  PR topic/platform workflow projects only its own lane and calls that protected
  default-branch lane natively, so every nested job and log stays visible from
  the matching PR run without a monolithic matrix. Each main topic/platform
  workflow computes the same exhaustive plan and calls its same-commit lane
  natively. Only manual-full passes digest-bound lane projections through
  native workflow-dispatch inputs. Do not use an artifact as dispatch state.
  Fork PRs use the same
  no-secret reusable path after fetching the immutable head SHA through the
  base repository. All lane conditions and summaries consume the same plan or
  its bounded projection.
- Keep direct semantic ownership separate from Cargo reverse dependencies:
  direct paths/crates select product, SDK, backend, platform, protocol, model,
  UI, website, and infrastructure slices; affected crates select Rust compile,
  Clippy, and tests.
- Store path/domain ownership and slice dependencies in checked-in manifests.
  The planner must emit a versioned machine-readable plan containing the
  profile, source identity, direct domains, affected crates, required slice
  IDs, bounded matrices, dependency reasons, and fan-out budgets.
- Supported profiles are semantic and closed: draft PR, ready PR, exhaustive
  main, and explicit full/manual validation. Do not add ad hoc boolean inputs
  when a profile or ownership rule expresses the decision.
- CI control-plane changes fail open to the affected slices and their
  consumers, except paths that map only to documentation plus `ci-control`;
  those retain limited documentation routing. Unknown paths, unknown plan
  fields, duplicate slice IDs, invalid matrices, or an ownership/schema
  mismatch fail planning.
- Pull requests test affected crates plus reverse dependents. Main tests every
  workspace member exactly once. Workspace discovery belongs to Cargo metadata,
  not a workflow-maintained allowlist.
- Use measured workload data to rebalance deterministic shards, but keep the
  checked-in algorithm reproducible. Use one Cargo invocation per shard unless
  a documented package-isolation check requires otherwise.

### Dependencies, summaries, and cancellation

- If a consumer downloads an artifact, its producer must be reachable under
  the same plan and declared through `needs`. Consumers never rebuild missing
  producer inputs.
- Every lane has one unique, stable, non-matrix summary. Each PR/main entry has
  one stable native result; the manual-only controller owns one stable
  synthetic aggregate check.
- Each lane summary directly needs every top-level slice call in that lane,
  runs with `if: ${{ !cancelled() }}`, and validates results against its
  digest-bound plan projection. The aggregate completes only after every
  correlated lane check is terminal.
  A skipped slice is valid only when absent from the plan. Reject required
  skips, failures, cancellations, unknown results, duplicate plan entries, and
  planned IDs outside the static needs graph.
- Cancel superseded PR runs. Do not cancel releases, deployments, cache
  warming, cleanup, or publishing unless their rollback semantics explicitly
  permit it.
- Keep the five focused PR workflows as separate visible checks, but stop
  wasting capacity after the first definitive job failure for one exact PR
  revision. The protected default-branch sibling monitor may cancel the other
  queued or in-progress Quality, Website, Linux, macOS, and Windows workflow
  runs for the same PR number, head SHA, and event epoch. It must preserve the
  workflow containing the first failure so that lane can publish its stable
  diagnostic result. This policy is PR-only; main, manual, release,
  deployment, cleanup, cache-warming, and newer PR revisions are never targets.
- Within a PR platform lane, compilation, functional-test, product, and smoke
  matrices fail fast so a required row failure cancels their queued and
  in-progress sibling rows. Keep this profile-derived: exhaustive main/manual
  runs continue sibling rows to retain complete diagnostics, and Quality
  matrices continue all rows because they are independent diagnostics rather
  than producer/consumer gates.
- Prefer declared `needs` edges to suppress impossible consumers after a
  producer failure. Cross-workflow PR cancellation is owned only by the
  protected `workflow_run` monitor: ordinary PR-controlled workflows and
  checked-out code must never receive `actions: write`. Cancelled sibling
  checks are expected terminal evidence after another lane has already failed;
  the preserved failed lane remains the root diagnostic.
- Control fan-out in the graph. Use bounded matrices and `max-parallel` after
  eliminating irrelevant work; do not use additional runner capacity as a
  substitute for routing and composition.

## Trust, runners, and providers

- Use `pull_request`, never `pull_request_target`, to build or execute PR
  content. Metadata-only and positively matched cleanup workflows may use
  `pull_request_target` but must never execute PR-controlled content.
- Treat GitHub and Depot as placement providers, not different build graphs.
  Slices request a semantic runner role; a centralized, event-derived policy
  maps that role to a provider label and cache mode.
- A caller may not choose a privileged runner label or independently grant
  remote-cache access. The workflow that owns `runs-on` must derive both from
  repository, event, ref, trust profile, architecture, and a bounded size.
- Ordinary PR jobs remain GitHub-hosted unless the bounded same-repository
  Depot risk exception below is active; an explicitly approved ephemeral,
  uncredentialed hardware runner may own a documented GPU smoke exception.
  Eligible trusted main jobs may use
  Depot only through the exact-string `DEPOT_RUNNERS_ENABLED` gate and retain a
  GitHub-hosted fallback. Tags, feature refs, external callers, credentialed
  smokes, macOS, Windows, and hardware-qualified GPU work stay on their
  explicitly approved provider until separately migrated.
- Product-integration inference maps CUDA and Vulkan to the approved ephemeral
  `gpu-nvidia` runner because that host provides both backends. ROCm maps only
  to the repository-scoped `gpu-amd` role and must remain skipped unless
  `MESH_ROCM_INFERENCE_RUNNER_ENABLED` is exactly `true`; an unset or different
  value means no approved ROCm inference runner is available.
- Never route untrusted code to a persistent self-hosted runner. Public-repo
  self-hosted execution requires ephemeral runners, restricted credentials and
  network access, and a runner group limited to the repository and exact
  protected default-branch reusable-workflow refs.
- A checked-out selector is defense in depth, not a PR trust boundary. PRs can
  modify local workflows and actions. Any future PR-on-Depot path must use a
  default-branch-pinned runner-owning workflow and a GitHub runner-group
  selected-workflow restriction.
- Provider changes must not alter source checkout, commands, profile,
  artifacts, tests, required checks, or plan membership.
- Shared Linux Clippy/test/host/runtime compiler-cache publication has one
  trusted GitHub-hosted warmer. It
  publishes one bounded, exact-key sccache seed after successful main Quality;
  GitHub-hosted PR jobs may restore it and write only to their job-local copy.
  Depot selections must never restore this seed through Depot's repository-
  scoped Actions-cache proxy. Do not restore per-shard Cargo target archives
  or enable per-object GHA publication for Linux Clippy, Rust tests, host, or
  runtime jobs.
- Every seeded compiler job records whether the exact seed was warm or cold.
  Enforce a measured minimum hit rate only for an exact warm restore; an
  intentional cache miss is classified cold and must not fail the build.

### Bounded Depot PR cache-risk exception

Depot Cache currently scopes entries by repository but not by branch and
automatically connects GitHub-cache API consumers on Depot runners. Cache-key
prefixes, job-local sccache, and a trusted caller do not prevent checked-out PR
code from using injected cache authority directly.

The provider-isolation requirements below remain the desired end state:

1. Automatic Depot Cache connectivity is disabled for the PR runner context,
   or Depot supplies a comparably strong per-PR namespace and read/write policy.
2. A fork PR receives no Depot cache token, WebDAV credential, transparent
   Actions-cache write authority, registry credential, or repository secret.
3. A hostile PR can neither read a trusted sentinel cache entry nor publish an
   entry later restored by trusted main/release jobs. Cache-key conventions are
   not proof.
4. The ephemeral Depot runner group is restricted to this repository and exact
   protected default-branch runner-owning workflows.
5. CI/workflow changes force the GitHub-hosted validation path so a PR cannot
   replace the protected executor that grants Depot placement.
6. GitHub-hosted rollback remains one checked provider-policy change and does
   not change the plan or artifact graph.

Until the provider supplies that end state, maintainers may deliberately accept
the documented repository-wide cache risk for eligible same-repository PRs
under all of these controls:

- the exception has a checked-in UTC expiry and fails hosted after it;
- `DEPOT_PR_RUNNERS_ENABLED` is exactly `true`;
- the head repository is exactly `Mesh-LLM/mesh-llm`; forks remain hosted;
- CI-control, workflow, runner-policy and cache-policy changes force hosted;
- the protected default-branch runner-owning workflows remain the executor;
- PR jobs receive no repository secrets or registry credentials; and
- rollback deletes `DEPOT_PR_CANARY_REF` and deletes or sets
  `DEPOT_PR_RUNNERS_ENABLED=false`, so PR jobs return to hosted placement.

This is an explicit speed-versus-isolation decision, not evidence that Depot
cache is isolated. While active, GitHub Actions cache API consumers on eligible
Depot PR and trusted-main jobs may share Depot's repository-wide namespace.
Treat that namespace as attacker-controlled and keep release, publishing,
deployment and credential-bearing jobs off it. Record the rationale, known
failure modes, owner, start, expiry and rollback in
`ci/DEPOT_PR_RISK_EXCEPTION.md`.

GitHub's `all_external_contributors` workflow-approval policy does not cover a
same-repository branch pushed by a collaborator. Do not describe that setting
as the approval boundary for this exception. The repository-wide PR gate and
checked-in expiry are the maintainer-controlled approval boundary.

## Cache contract

- Declare cache mode per slice: no remote cache, PR restore-only/isolated, or
  trusted read-write. Derive it from the same policy as runner placement.
- GitHub-hosted PR writes remain isolated from trusted main: normal protected
  same-repository and fork lanes explicitly force restore-only cache mode even
  though their workflow ref is the default branch. The bounded Depot exception
  is the documented cross-branch deviation. Trusted
  main/release/warmers own shared publication. Do not save large shared
  Rust/native caches from PRs.
- GitHub scopes caches created by `pull_request` runs to that PR's merge ref.
  Small exact caches may publish in that isolated scope when their key covers
  every compatibility boundary and their contents are verified before use;
  this is the approved mechanism for accelerating reruns of the same PR.
  Never use a restore prefix for these PR-produced native caches.
- Keep multi-gigabyte Cargo target caches restore-only on PRs. Per-PR copies
  multiply by matrix row, consume repository cache quota, and can evict the
  trusted main caches that every PR can restore. Prefer small exact native ABI
  caches and package-manager download stores with lockfile-exact keys.
- Assign one publisher for a shared package-manager key. Other focused PR
  workflows may restore it but must not race to upload the same cache entry.
- Keep PR sccache job-local unless a provider proves safe isolation. A cache
  miss must remain a correctness-preserving miss, never a reason to rebuild a
  producer secretly in a consumer.
- Cache keys include every compatibility boundary: provider where necessary,
  OS, architecture, backend/toolchain, profile, image/toolchain epoch,
  lockfiles, recipe inputs, and `.github/cache-version.txt`. Do not use broad
  restore prefixes across trust or ABI boundaries.
- Treat restored cache contents as untrusted. Verify native build stamps,
  manifests, checksums, target/backend identity, and required link closure
  before reuse. Never store credentials or private endpoints in a cache.
- Artifacts move outputs within a run; caches accelerate regeneration across
  runs. Do not use a cache as the producer/consumer correctness contract.
- PR and smoke-only artifacts default to one-day retention. Release evidence
  follows the release policy.

## Product and artifact contract

- Model every executable product as a backend-neutral host, one separately
  packaged native runtime per OS/architecture/backend, and a composition-only
  product. A backend matrix belongs to runtime/product rows, never host rows.
- Build prepared UI assets once per selected platform lane and feed that
  immutable artifact to every host producer in the lane. Host producers must
  not rerun UI tests. Cross-workflow artifact sharing is an explicit timing
  tradeoff, not a correctness dependency.
- Host artifacts include the executable checksum and import-policy report.
  Runtime artifacts include manifest, checksums, ABI/version, platform, and
  backend identity. Product manifests record exact host and runtime digests.
- Use the owning preparation and composition actions. A composer verifies and
  copies exact producer bytes; it never compiles, links, re-stamps, or silently
  substitutes an input.
- No-driver `--version`, runtime discovery, and noninteractive client readiness
  are mandatory for every composed alias. Hardware serving qualification is
  additional coverage, not a replacement.
- Restore smoke inputs through the shared restore action and reusable smoke
  workflows. Every CI invocation of `mesh-llm` includes `--log-format json`.
- Release and packaging wrap the same producers with signing, attestation,
  publication, and retention. They do not fork the underlying build commands.

## Workflow definition and dependencies

- Declare least-privilege `permissions` explicitly. Default to
  `contents: read`; grant write or OIDC permissions only to the owning job.
- Set `persist-credentials: false` on checkout unless a narrowly scoped job
  must push. Never expose secrets to fork PRs or interpolate untrusted values
  into shell commands.
- Pin new third-party actions to a full commit SHA with a release comment.
- Use stable job/step IDs, explicit shells, `set -euo pipefail` for nontrivial
  Bash, PowerShell-native error handling on Windows, and realistic timeouts.
- Project dependencies belong in checked-in manifests/lockfiles. Shared Linux
  tools belong in `mesh-llm-runner-images`. Do not add one-off `apt`, `pip`,
  global npm, `cargo install`, downloaded binaries, or `curl | sh` to a job.
- Use digest-pinned multi-architecture runner images. A backend image provides
  a toolchain, not GPU hardware. GPU execution requires a restricted matching
  device runner.

## Operational safety

- Inspection, log reads, syntax validation, and dry-run planning are read-only.
  Dispatching, rerunning, cancelling, approving, deleting, changing variables
  or secrets, editing runner groups, changing Depot settings, publishing,
  deploying, or resetting caches are external mutations and require scope from
  the user.
- Never print or persist secret values. Report names, scope, and presence only.
- Diagnose failures as product, dependency, runner image, capacity, cache,
  artifact, permission, secret/variable, external service, or workflow logic.
  Fix the owning source; do not weaken a gate or add retries to hide a
  deterministic failure.
- Validate with the narrowest safe workflow. A run is not successful until all
  required jobs reach a terminal successful conclusion; state expected skips.

## Validation contract

For every workflow or local-action edit:

```bash
just ci-validate
```

Also run what applies:

- `just ci-shellcheck <changed-script>...` for changed shell scripts or
  substantial embedded Bash.
- `just ci-crate-lists` for routing,
  workspace, Clippy, or test-plan changes.
- `just check-release` for release/product target changes.
- `just publish-crates` for publishing.
- Owning planner/action tests covering selected, skipped, invalid, PR, main,
  GitHub, and Depot-denied branches as applicable.
- A non-required dual-run or authorized dispatch before migrating required
  checks or external runner policy.

Finish with changed files, validation results, expected skips, unresolved
risks, external state changes, and live configuration that could not be
verified.
