# PR and main CI composition plan

Status: implemented on this branch. The shared planner, focused PR and main
entrypoints, manual controller, topic/platform lane workflows, platform-pure reusable
graphs, contracts and documentation are checked in. Ruleset migration remains
separate work. Exact same-repository PR revisions may use Depot under the
time-bounded exception in `ci/DEPOT_PR_RISK_EXCEPTION.md`.

Owners: MeshLLM maintainers

## Outcome

PR and main CI use one versioned planner and one catalog of typed reusable
workflow slices. A PR selects representative rows from the main catalog; a
selected row uses the same workflow, commands, build profile, artifact
contract and verification that main uses. Faster PRs come from precise routing,
immutable producer reuse and bounded fan-out, not from a second lightweight
implementation or more runner capacity.

GitHub-hosted and Depot-managed runners are placement providers. Provider
selection cannot change plan membership, commands, artifacts, required results
or cache identity.

## Implemented on this branch

- Five `pr_*.yml` entrypoints independently plan and call their matching
  protected default-branch reusable lane. Five `main_*.yml` entrypoints
  independently plan the exhaustive profile and call their matching
  same-commit reusable lane. PRs and routine main pushes therefore keep nested
  jobs in focused native topic/platform runs; only explicit manual-full runs
  use protected detached lane dispatch.
- ci/ownership.yml and ci/slices.yml define the checked ownership, dependency,
  row, runner-role, cache-mode and worker-budget catalog.
- Protected PR planning extracts only those two manifests from the validated
  immutable source SHA and passes their unique runner-temp root to the
  default-branch planner. The source manifests are data only. Planner code,
  Cargo workspace discovery, and affected-crate operations remain rooted in
  the protected checkout. The source ownership and slice catalogs must match
  the protected catalogs, and missing manifests fail closed.
- scripts/plan-ci.py emits the versioned plan described by
  ci/ci-plan.schema.json, including direct domains, affected crates, signals,
  reasons, dependencies, matrices and budgets.
- Separate Quality, Website, Linux, macOS and Windows workflows receive bounded
  native inputs from one plan and own typed platform-local static slice
  supersets without unrelated platform placeholders. They support native
  reusable PR/main calls and protected manual dispatch.
- PRs and routine main pushes expose native lane jobs and five stable
  topic/platform results; dispatched manual-full lanes retain correlated checks.
- One protected default-branch `workflow_run` monitor observes the five focused
  runs for one exact PR event epoch. After the first definitive job failure it
  preserves that workflow and cancels the other queued or in-progress sibling
  lanes; PR-controlled jobs never receive Actions-write permission.
- Existing static ABI, native SDK, Swift, smoke and HF workflows remain
  lower-level reusable producers/consumers.
- Current PR routing may use Depot for eligible same-repository executor jobs
  under the checked-in cache-risk deadline and repository gate. Forks,
  control-plane jobs, credential-bearing smokes, and the documented CUDA smoke
  keep their approved placements; trusted-main Depot selection remains behind
  the existing exact-string policy gate.
- `ci-quality-slice.yml` contains an additive protected authority-sentinel
  diagnostic selected by separate `DEPOT_PR_SENTINEL_REF` and
  `DEPOT_PR_SENTINEL_ID` variables. It does not add a PR entrypoint, planner
  row, build command, matrix, artifact, producer/consumer edge or required
  summary; normal Quality jobs continue using the existing provider selector.
- Current CI docs, inventory, skill and agent instructions describe this graph.
- This cleanup retires the obsolete `pr_builds.yml` and `ci-orchestrator.yml`
  migration shims now that the five focused PR entrypoints own validation.
- `ci.yml` remains a reusable-only, no-op filename shim while the protected-main
  runner contract is updated. It must not regain a push trigger, dispatch
  behavior, or lane calls, and is removable in the follow-up cleanup once that
  contract is active on protected main.

This branch does not change branch rulesets, required checks, Depot settings,
runner groups, secrets or external capacity.

The five independent PR workflows are an acceptance invariant. The PR UI must
show distinct Quality, Website, Linux, macOS, and Windows runs with direct job
and log drill-down. A single all-lanes PR graph or detached dispatch-only check
is a regression even if its jobs execute successfully. Future work belongs in
the matching reusable lane and must preserve these five entry boundaries.
Routine main validation has the same acceptance invariant and exposes
`Main / Quality`, `Main / Website`, `Main / Linux`, `Main / macOS`, and
`Main / Windows` from five focused native runs.

## Graph

  PR Quality entry --> plan --> protected Quality lane --> PR / Quality
  PR Website entry --> plan --> protected Website lane --> PR / Website
  PR Linux entry --> plan --> protected Linux lane --> PR / Linux
  PR macOS entry --> plan --> protected macOS lane --> PR / macOS
  PR Windows entry --> plan --> protected Windows lane --> PR / Windows
  PR Quality in-progress --> protected sibling monitor --> cancel other exact-revision lanes after first failure

  Main Quality/Website/Linux/macOS/Windows entries --> same-commit matching lanes
  Explicit manual-full entry --> protected controller --> five dispatched lanes

Each PR entry calls one protected default-branch workflow as a nested reusable
job, and each main entry calls its same-commit workflow. Both preserve native
run/log visibility without a monolithic graph. The manual-only controller
dispatches the same list with bounded JSON inputs. PR planning also uses the
immutable source SHA to read only `ci/ownership.yml` and `ci/slices.yml` as
inert routing data. Product workflows use it for source checkout. Each lane
contains a platform-local static superset of typed reusable calls. Workflow
YAML is never generated, and lanes do not download a planner artifact or
allocate a planner. Fork heads are fetched through the base repository while
planner, action, and workflow definitions remain protected on the default
branch.

## Planner contract

scripts/plan-ci.py is the only eligibility implementation. It reads the
JSON-compatible manifests and validates their schema and dependency graph.
Unknown paths and malformed inputs fail closed. CI-control and
runner-infrastructure changes fail open to control rows and all supported
product rows. Paths that map only to documentation plus `ci-control` are the
exception. They keep the profile base and add only `runner-contract`, without
forcing product rows.

The planner accepts a separate manifest root for the two catalog files. It
defaults to the workspace root. Protected PR callers provide a unique
runner-temp directory populated from the immutable source SHA; push and manual
callers keep the protected workspace default. The manifest root never changes
workspace package discovery or affected-crate command paths. Missing source
manifests fail closed.

Each plan contains:

- event/profile and source/base identities;
- direct crates and affected Cargo reverse-dependency crates separately;
- ordered semantic domains and planner-owned signals;
- selected slice IDs, reasons and dependency closure;
- bounded Clippy, Rust-test, host, runtime/product, platform, smoke and SDK
  matrices;
- runner roles, cache modes and Linux/macOS/Windows/total worker budgets.

Profiles are closed and event-derived:

| Profile | Selection |
| --- | --- |
| pr-draft | No build slices; stable planner/gate results only, except CI-control or runner-infrastructure fail-open |
| pr-ready | Complete targeted rows for direct domains and affected Rust dependents |
| main | Every workspace crate and every supported product/platform/backend/SDK row |
| manual-full | Main-equivalent non-publishing dispatch |

The selected ready-PR row is semantically identical to main. Draft PRs select
no build row. Only plan membership, bounded parallelism, worker budgets
(`total_max_workers` is 10 for `pr-ready` and 18 for `main`), trust-derived
cache mode, short-lived artifact namespace, provider label and optional trusted
credentials may differ.

## Slice catalog

- ci-quality-slice.yml: contracts, formatting, bounded Clippy and CLI docs.
- ci-web-slice.yml: console lint/type/test and public website build.
- ci-ui-artifact-slice.yml: one immutable console distribution producer.
- static-abi-artifact.yml: one verified portable static llama ABI producer.
- ci-rust-tests-slice.yml: deterministic affected/all-workspace Cargo batches.
- ci-{linux,macos,windows}-host-slice.yml: platform-pure neutral hosts.
- ci-{linux,macos,windows}-runtime-slice.yml: one native runtime per selected
  backend, running in parallel with the matching host.
- ci-{linux,macos,windows}-product-slice.yml: platform-local composition after
  matching host and runtime producers succeed.
- ci-platform-checks-slice.yml: macOS portable/unit and Windows checks.
- ci-linux-product-smoke-slice.yml and ci-macos-product-smoke-slice.yml:
  platform-local inference, backend, two-node, Metal and model-download
  consumers using only composed artifacts. One Linux KV caching smoke job runs
  a fixed dense SmolLM2 leg followed by a recurrent Qwen3.5 leg; both must pass.
- ci-linux-sdk-slice.yml and ci-macos-sdk-slice.yml: platform-local
  Rust/Kotlin/Swift consumers. Swift and Kotlin SDK artifacts are independent
  producers that start from the plan and static ABI respectively, before
  product composition completes.
- ci-runner-contract-slice.yml: plan/provider/cache trust checks plus trusted
  main runner-image contracts.

Existing native-sdk-artifact.yml, swift-sdk-artifact.yml, smoke.yml,
scripted-binary-smoke.yml, sdk-smoke.yml and hf-download-smoke.yml remain
lower-level typed building blocks. Consumers never rebuild missing producers.
In the full Swift producer, the seven Apple Rust target libraries are immutable
matrix outputs assembled by one downstream XCFramework job. The matrix obeys
the lane's bounded macOS parallelism; host-only Swift production stays on its
single-job path.

## Product and artifact contract

Every executable product has three layers:

1. prepared UI assets;
2. one release-profile backend-neutral host per OS/architecture;
3. one native runtime per OS/architecture/backend.

The catalog covers Linux CPU/CUDA/ROCm/Vulkan, macOS Metal and Windows
CPU/CUDA/ROCm/Vulkan where supported. Unsupported combinations are omitted by
the planner, not represented as permanent skipped jobs.

Host actions emit executable, checksum and import-policy evidence. The native
runtime action emits a manifested archive. compose-product-input verifies exact
producer bytes and performs no compilation, relinking, restamping or
substitution. Smoke and SDK consumers download those artifacts only. Linux CPU
composition readiness additionally executes the shared SDK runtime reader on
the real host's available-runtime JSON with fallback building disabled, even
when full SDK rows are not selected. The #1675 path regression protects its
existing host/runtime/product reachability without changing the catalogs.

Selected PR and main product rows both use release-profile hosts. Native
runtime compilation does not depend on host production, so each platform lane
starts those immutable producers concurrently and only joins them for product
composition.

## Fan-out and timing controls

Initial planner budgets are:

| Profile | Linux cap | macOS cap | Windows cap | Total planned |
| --- | ---: | ---: | ---: | ---: |
| PR draft/ready | 7 | 2 | 1 | 10 |
| main/manual | 12 | 4 | 2 | 18 |

The lane projections pass smaller PR max-parallel values to Clippy, tests,
hosts, runtimes and platform checks and wider bounded values to main. Host, ABI
and runtime producer identities are not duplicated. The fixed two-row
split-model matrix is serialized, so it adds runner-minutes without increasing
peak workers. The full Swift target matrix is a deliberate fan-out of seven
distinct architecture/platform inputs,
bounded to two concurrent jobs for PR profiles and four for main/manual and
release; a single assembly job restores those libraries and publishes the
verified XCFramework. Separate run-scoped graphs build one prepared UI artifact
per active platform lane; that is the accepted readability tradeoff, while UI
tests remain owned by the Website graph. Every heavy job has a timeout and a
deterministic row identity.

PR platform matrices for compilation, Rust tests, products, functional
platform checks, and full Swift targets fail fast. Main/manual and release
matrices continue all rows for exhaustive diagnostics, and Quality remains
non-fail-fast within its own matrix. Declared
producer dependencies suppress consumers that can no longer run. Across the
five focused PR workflows, the protected sibling monitor preserves the lane
with the first definitive job failure and cancels the other exact-PR,
exact-SHA, same-epoch runs. Main/manual and unrelated workflows are never
targets. The monitor is the only owner of Actions-write permission; checked-out
PR code cannot invoke the cancellation API.

PR caching is selective. Linux Clippy, Rust tests, host, and runtime restore one
bounded trusted sccache seed on GitHub-hosted runners and no longer restore
per-row Cargo target archives. PR writes stay job-local; only the protected
post-Main-Quality warmer publishes the exact 2 GiB seed, and Depot selections
cannot restore it. Exact verified static,
Swift, Metal-unit and Windows ABI caches may publish into the PR merge-ref
scope for same-PR reruns. Website owns the single pnpm publisher and its npm
store cache, while platform UI producers are restore-only for the shared pnpm
key. Artifacts remain run-scoped correctness inputs, never rerun caches.

scripts/collect-ci-metrics.py is the read-only measurement tool. Schema-v3
reports keep queue, measured dependency wait, execution and wall-clock timing
separate; they also record provider/OS/architecture/role/size, sample counts,
runner-minutes, cancellation, peak workers and capacity contamination. Use
`--compare-input` for provider-separated historical PR cohort comparisons.
Its date-independent queue p95 heuristics are measurement gates, not dated run
conclusions. Timing experiments must use a new worktree from main. Keep raw
evidence outside authoritative topology docs. Capacity is not the optimization.

## Required summary

Each lane owns one stable non-matrix summary that directly needs its complete
static job superset and validates required work against its bounded plan.
Each PR entry owns one native non-matrix `PR / <lane>` job that directly needs
its reusable lane call and validates required success or an unplanned skip.
Each main entry owns one native non-matrix `Main / <lane>` job. ci-control.yml
creates correlated lane checks only for explicit manual-full runs;
the final lane completes that aggregate only when all expected checks are
terminal. Existing branch-protection rules are not edited by this
implementation.

## GitHub and Depot

select-ci-runners resolves semantic runner roles. Eligible same-repository PR
executor jobs may use Depot during the bounded exception. Forks, feature refs,
tags, credential-bearing smokes and hardware-qualified work stay on their
approved placement. Trusted main Linux work may use Depot only when
DEPOT_RUNNERS_ENABLED is exactly true, with a GitHub-hosted fallback. Callers
never provide raw labels or independent remote-cache permission.

Permanent Depot PR execution is not enabled. The selector has a bounded
`DEPOT_PR_CANARY_REF` hook for one exact same-repository merge ref. The separate
temporary exception requires `DEPOT_PR_RUNNERS_ENABLED` and the checked-in
2026-09-14 UTC deadline. The Quality slice
also has a separate `DEPOT_PR_SENTINEL_REF` selector and
`DEPOT_PR_SENTINEL_ID` validation for one no-checkout authority diagnostic;
the ordinary Quality jobs still use `DEPOT_PR_CANARY_REF`, and a global PR gate
alone cannot run the sentinel. The diagnostic runs only when the actual and
original event are `pull_request`, the exact configured merge ref selects
Depot, and the head repository is this repository. Fork heads,
`pull_request_target`, dispatches, planner-forced hosted paths, missing or
non-matching refs remain hosted/no-Depot; malformed selector configuration is
rejected by the central selector. Depot's documented GitHub cache path is
repository-scoped and not branch-isolated; automatic cache redirection can
expose repository-wide cache authority to PR code. Cache-key prefixes are not
isolation. The central selector emits
`allow_depot_remote_cache=false` for every Depot selection. Outside the bounded
exception, native Actions-cache consumers are disabled. During the exception,
eligible same-repository PR and trusted-main Depot jobs emit
`allow_native_github_cache=true`, deliberately sharing Depot's repository-wide,
cross-branch Actions-cache namespace for iteration speed. Hosted release and
cache-warmer paths retain their existing GitHub cache behavior. This is
accepted risk, not an ambient authority proof; the completed sentinel has shown
unsafe repository-scoped cross-trust access, so a provider-isolation redesign
and a new successful sentinel remain required for permanent activation.

The admin-verified organization switches have a narrower effect than the
checked-in consumer policy: they remove direct `DEPOT_CACHE_TOKEN`/WebDAV
build-tool preconfiguration and Registry Actions authentication from fresh
runners, but they do not document or enforce a per-connection/job/ref disable
or ACL for the GitHub Actions cache proxy/runtime-token path. The sentinel
proved that path remains repository-scoped across the trusted-main/PR boundary.
The required provider contract is therefore a supported, server-enforced
per-connection/job/ref control that either leaves PR jobs on GitHub-native
branch-scoped Actions endpoints/token with no Depot proxy or direct cache
token, or issues a PR-isolated namespace/token whose ACL permits reads and
writes only within that PR, denying reads and writes from trusted main/release
and every other PR namespace, without exposing `DEPOT_CACHE_TOKEN`.

The sentinel is deliberately outside the planner/build graph and does not
invoke `audit-depot-pr-isolation`: that audit rejects the ambient endpoint
before cache access, whereas this diagnostic must exercise the actual restore
and save authority without executing PR-controlled code. Its job has empty
permissions, no checkout, no secrets and only the fixed
`.depot-authority-sentinel` path. Before each cache action it requires the
provider-injected `ACTIONS_CACHE_URL` and `ACTIONS_RESULTS_URL` to be present
and structurally attested as HTTP endpoints with a nonempty, non-GitHub,
non-loopback authority (including all IPv4 `127/8` and IPv4-mapped IPv6
loopback spellings), numeric port and explicit path; it reports only
variable/reason classes and never endpoint values. The shell attestation does
not require ambient `ACTIONS_RUNTIME_TOKEN`: GitHub's
`NodeScriptActionHandler` injects that credential into the pinned cache
actions, while the shell `ScriptHandler` does not. Their successful full
restore/save calls provide the credential/token proof. Manual seed/verify
inputs bind to both the configured sentinel ID and the exact merge ref. Every
attestation uses the fixed runner's Python 3.8+ stdlib `ipaddress` classifier
for each bracketed IPv6 spelling; parser absence/version/invalidity fails
closed. The PR probe restores the trusted seed (not lookup-only),
validates exact seed marker content on a hit, replaces it with a deterministic
non-secret poison marker, saves the exact Stage 1 poison key, clears and fully
restores that key, and requires a cache hit and exact marker bytes before the
seed decision; only then does it fail if the seed was readable. This same-job
restore/save proves the PR Node token/write path. A seed miss passes with the
trusted-main `verify-pr-write` phase pending; main verify's poison miss remains
the cross-scope proof. Fork PRs remain hosted and
provide the no-Depot-authority evidence; only the exact same-repository
sentinel ref exercises the Depot diagnostic.

Controlled evidence now records the authority result: trusted-main seed
[run 31816775585](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31816775585)
succeeded at `main` commit `9e977e246`; the same-repository PR sentinel
[run 31816869128 / job 94821057215](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31816869128/job/94821057215)
restored and exactly validated the trusted seed, saved/cleared/restored and
exactly validated the poison, then failed its intended seed-isolation gate;
the enclosing PR run was later cancelled during cleanup. Trusted-main verify
[run 31817111471 / job 94821343605](https://github.com/Mesh-LLM/mesh-llm/actions/runs/31817111471/job/94821343605)
restored and exactly validated that poison, then failed its intended expected-
miss gate. This is unsafe repository-scoped cross-trust authority, not a
successful isolation result. The temporary exception knowingly accepts it only
when `DEPOT_PR_RUNNERS_ENABLED=true` and the 2026-09-14 UTC deadline is still
active. A provider-isolation redesign and a new successful sentinel are
required before that exception can become permanent. The exact-SHA five-lane
candidate, provider-separated comparison, and identical-SHA hosted rollback
are recorded in `.omo/specs/depot-pr-rollout-evidence.md`; Quality and Linux
had favorable queue observations but remain unclassified because execution
was cache-confounded, Website had insufficient samples, and macOS/Windows hit
the capacity rollback threshold. Fork validation and namespace purge/expiry
confirmation remain pending.

Before a permanent PR Depot path is enabled, an administrator must prove:

1. the provider's documented per-connection/job/ref control selects either
   GitHub-native branch-scoped Actions cache endpoints/token with no Depot
   cache token, or a PR-isolated namespace/token whose ACL permits reads and
   writes only within that PR, denying reads and writes from trusted
   main/release and every other PR namespace;
2. same-repository and fork PRs receive no cache, registry or repository secret
   authority;
3. hostile PRs cannot read a trusted sentinel or publish an entry restored by
   trusted main;
4. the ephemeral runner group is restricted to this repository and exact
   protected default-branch runner-owning workflow refs;
5. CI-control changes force the GitHub path and rollback is tested;
6. provider parity passes on comparable non-CI-change PRs, using
   provider-separated queue, execution and critical-path metrics.

The investigation and canary are defined in ci/DEPOT_MIGRATION.md. Do not
change Depot settings or runner groups in this graph PR. Start a later rollout
with remote cache disabled, then canary one non-secret Linux slice, one Rust
test slice and the selected Linux product graph before expanding to equivalent
Depot macOS 15 and Windows 2022 build/test rows. Keep control-plane
planning/required summaries, credential-bearing smokes, `gpu-nvidia` hardware,
and any Intel macOS row without a Depot-equivalent on their approved hosted
providers.

## Extension pattern

1. Read the manage-ci skill, current inventory, ci/ci.md and this spec.
2. Classify the owner and trust context.
3. Extend one typed reusable slice or local action; keep entrypoints thin.
4. Update ownership, dependencies, rows and budgets when routing changes.
5. Preserve immutable producer reachability and add the top-level call to its
   lane summary; update the bounded controller projection when necessary.
6. Keep runner and cache decisions in the central policy action.
7. Update current docs and fixtures in the same change.

Required local checks are actionlint with the repository config, git diff
--check, the full scripts/tests unittest suite, applicable shellcheck, and
the relevant xtask repo-consistency checks.

## Acceptance checklist

- docs-only, website-only, UI-only, ordinary Rust, runtime, protocol,
  split-serving, model, each SDK, each backend, macOS-only, Windows-only,
  mixed and CI-control changes produce expected plan rows and reasons;
- draft-to-ready changes select the correct closed profile;
- main covers every workspace member exactly once and all supported rows;
- each selected PR product uses the same slice and artifact contract as main;
- no duplicate host, ABI, or native-runtime producer identity is built for one
  source plan;
- every consumer has a reachable producer and no consumer fallback build;
- every lane summary passes unplanned skips and fails planned skips;
- a failed PR lane retains its stable diagnostic while sibling runs become
  terminal cancellations, with no required result stuck running;
- GitHub fallback passes all slice fixtures;
- no branch ruleset, runner group, Depot setting or external capacity change
  is included in this implementation PR.

Required-check migration and Depot PR enablement are separate follow-up changes
after these acceptance cases pass. The final PR-Depot contract preserves the
five native PR checks and every selected row's commands, build profile,
artifacts, tests, summaries, fail-fast behavior and producer/consumer edges.
It is not a direct runner-label replacement: protected workflow refs,
cache authority, source-SHA checkout, fork/no-secret behavior, CI-control
fallback and provider exceptions must all be enforced by the owning runner
policy.
