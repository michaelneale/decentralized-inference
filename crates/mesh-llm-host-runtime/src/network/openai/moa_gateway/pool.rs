//! MoA worker-pool assembly.
//!
//! Owns the discovery-and-assembly side of the MoA gateway: turning the
//! node's mesh-wide model view into a concrete `(backends, models)` worker
//! pool. `build_moa_config` (in [`super::workers`]) is the orchestrator that
//! calls [`assemble_worker_pool`] and [`compute_actor_candidates`] here.

use super::context_selection;
use super::self_fill::self_fill_from_extra_instances;
use super::workers::{LocalModelBackend, RemoteModelBackend};
use crate::inference::election;
use crate::mesh;
use mesh_mixture_of_agents as moa;
use std::collections::HashMap;

/// Boundary between small- and big-tier in billions of parameters.
///
/// Taken from the engine so host admission/capping and MoA's own role
/// assignment can never disagree about where the tier boundary sits.
const SMALL_TIER_MAX_B: f64 = moa::SMALL_TIER_MAX_B;

/// Model size class used for the destructive admission/cap decisions.
///
/// Only a verified gossiped GGUF size can make a model `Big`. A model with no
/// verified size is `Small` — the weakest — so an unverifiable label can never
/// masquerade as strong and displace a real big worker (i386 P1).
#[derive(Clone, Copy, PartialEq, Eq)]
enum SizeTier {
    Small,
    Big,
}

/// Map of canonical base name → verified size (billions of params) as gossiped
/// by peers through `ServedModelMetadata.parameter_count_b`. In MoA the
/// orchestrator has no peer GGUFs — this is the only authoritative size source.
async fn gossiped_sizes(node: &mesh::Node) -> HashMap<String, f64> {
    let mut by_base: HashMap<String, f64> = HashMap::new();
    for descriptor in node.all_served_model_descriptors().await {
        if let Some(b) = descriptor
            .metadata
            .as_ref()
            .and_then(|m| m.parameter_count_b)
        {
            let base = canonical_base_name(&descriptor.identity.model_name);
            by_base
                .entry(base)
                .and_modify(|e| {
                    if b > *e {
                        *e = b;
                    }
                })
                .or_insert(b);
        }
    }
    by_base
}

/// Tier a model: verified gossiped size first, model-name parse as a
/// lower-confidence fallback, `Unknown` when neither yields a size.
fn tier_for(name: &str, sizes: &HashMap<String, f64>) -> SizeTier {
    // Verified gossiped GGUF size is the ONLY tiering signal (per i386 review).
    // No name-based fallback: a model with no gossiped size is treated as the
    // weakest (Small), so an unverifiable label can never masquerade as big and
    // displace a real strong worker. `SMALL_TIER_MAX_B` splits small/big.
    match sizes.get(&canonical_base_name(name)) {
        Some(b) if *b >= SMALL_TIER_MAX_B => SizeTier::Big,
        _ => SizeTier::Small,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum AvailabilityRank {
    Healthy,
    Deprioritized,
}

/// Best currently routable health and advertised throughput for each canonical
/// model. A local model is healthy by definition; paused remote peers are not
/// routable and therefore do not contribute.
async fn model_routing_hints(
    node: &mesh::Node,
) -> HashMap<String, (AvailabilityRank, Option<u64>)> {
    use crate::proto::node::InferenceAdmissionState;

    let mut hints = HashMap::new();
    for local in node.hosted_models().await {
        hints.insert(
            canonical_base_name(&local),
            (AvailabilityRank::Healthy, None),
        );
    }
    for peer in node.peers().await {
        let availability = match peer.inference_admission_state {
            Some(InferenceAdmissionState::RemotePaused | InferenceAdmissionState::AllPaused) => {
                continue;
            }
            Some(InferenceAdmissionState::AcceptingDeprioritized) => {
                AvailabilityRank::Deprioritized
            }
            _ => AvailabilityRank::Healthy,
        };
        for model in peer.http_routable_models() {
            let model_base = canonical_base_name(&model);
            let throughput = peer
                .advertised_model_throughput
                .iter()
                .filter(|hint| canonical_base_name(&hint.model_name) == model_base)
                .map(|hint| hint.avg_tokens_per_second_milli)
                .max();
            hints
                .entry(model_base)
                .and_modify(|(best_availability, best_throughput)| {
                    match availability.cmp(best_availability) {
                        std::cmp::Ordering::Less => {
                            *best_availability = availability;
                            *best_throughput = throughput;
                        }
                        std::cmp::Ordering::Equal => {
                            *best_throughput = (*best_throughput).max(throughput);
                        }
                        std::cmp::Ordering::Greater => {}
                    }
                })
                .or_insert((availability, throughput));
        }
    }
    hints
}

/// Try each alias in `aliases` until one resolves to a backend, then stop.
///
/// Aliases are pre-sorted by `group_aliases_by_canonical_base` so the most
/// preferred (locally-served first, then shortest) is tried first. Falls
/// back to longer aliases when the preferred one's peer is unreachable.
#[allow(clippy::too_many_arguments)]
async fn resolve_one_worker_from_aliases(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    http: &reqwest::Client,
    aliases: &[String],
    required_tokens: Option<u32>,
    sizes: &HashMap<String, f64>,
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    local_count: &mut usize,
) {
    let resolution = WorkerBackendResolution {
        node,
        targets,
        http,
        required_tokens,
        sizes,
    };
    for name in aliases {
        if add_worker_backend(&resolution, name, backends, models, local_count).await {
            return;
        }
    }
}

/// Group all advertised model names by their canonical base so each
/// canonical model contributes exactly one worker, but the resolver gets
/// to pick the alias that actually has a reachable backend.
///
/// The earlier shape committed to a single alias per base *before* trying
/// to resolve a backend. Two failure modes:
///
///   1. The chosen alias is advertised only by a peer that drops between
///      gossip refresh and orchestration — `hosts_for_model` returns
///      empty, the worker is dropped, and longer-form aliases for the
///      same canonical model from still-reachable peers are rejected as
///      duplicates.
///   2. The local node advertises a longer convention
///      (e.g. `unsloth/Qwen3-8B-GGUF:Q4_K_M`) while a peer advertises a
///      shorter variant (e.g. `Qwen3-8B-Q4_K_M`). The shortest-name rule
///      picks the peer alias, `add_worker_backend` looks for a local port
///      under that specific string, finds nothing, and forces a
///      QUIC-tunnel backend even though the model is right here.
///
/// Both failure modes are fixed by grouping first and resolving second.
/// Within each group the aliases are ordered so the most likely
/// optimization wins first try: locally-served name (skippy-port fast
/// path) before remote names, then shortest first as a tiebreaker.
fn group_aliases_by_canonical_base(
    names: Vec<String>,
    targets: Option<&election::ModelTargets>,
) -> Vec<Vec<String>> {
    let mut by_base: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for name in names {
        by_base
            .entry(canonical_base_name(&name))
            .or_default()
            .push(name);
    }
    // Deterministic group order so the worker list is stable across
    // builds even though HashMap iteration is not. Sort group entries
    // (locally-served first, then shortest), then sort groups by their
    // first ("best") alias.
    let mut groups: Vec<Vec<String>> = by_base
        .into_values()
        .map(|mut aliases| {
            aliases.sort_by(|a, b| {
                let la = is_locally_served(a, targets);
                let lb = is_locally_served(b, targets);
                lb.cmp(&la) // local (true) before remote (false)
                    .then_with(|| a.len().cmp(&b.len()))
                    .then_with(|| a.cmp(b))
            });
            aliases
        })
        .collect();
    groups.sort_by(|a, b| a[0].cmp(&b[0]));
    groups
}

/// Does the local routing table have a backend port for this exact name?
fn is_locally_served(name: &str, targets: Option<&election::ModelTargets>) -> bool {
    targets
        .and_then(|t| {
            t.targets.get(name).map(|tv| {
                tv.iter()
                    .any(|t| matches!(t, election::InferenceTarget::Local(_)))
            })
        })
        .unwrap_or(false)
}

/// Resolve `name` to a backend (local skippy port if available, else first
/// remote host) and append it to `backends`/`models`. Returns true if a
/// backend was added.
struct WorkerBackendResolution<'a> {
    node: &'a mesh::Node,
    targets: Option<&'a election::ModelTargets>,
    http: &'a reqwest::Client,
    required_tokens: Option<u32>,
    /// Verified sizes by canonical base, so each worker carries its real size
    /// into the engine instead of leaving the engine to parse the name.
    sizes: &'a HashMap<String, f64>,
}

impl WorkerBackendResolution<'_> {
    /// Gossiped size for `name`, keyed by canonical base so any alias of the
    /// same model resolves to the same figure. `None` when no peer has
    /// advertised a verified size, which the engine treats as unknown.
    fn verified_size(&self, name: &str) -> Option<f64> {
        self.sizes.get(&canonical_base_name(name)).copied()
    }
}

async fn add_worker_backend(
    resolution: &WorkerBackendResolution<'_>,
    name: &str,
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    local_count: &mut usize,
) -> bool {
    // Prefer local skippy port when this node serves the model.
    let local_port = resolution.targets.and_then(|t| {
        t.targets.get(name).and_then(|tv| {
            tv.iter().find_map(|t| match t {
                election::InferenceTarget::Local(p) => Some(*p),
                _ => None,
            })
        })
    });
    if let Some(port) = local_port {
        let context_length = resolution.node.local_model_context_length(name).await;
        if context_selection::context_can_satisfy(resolution.required_tokens, context_length) {
            let backend_idx = backends.len();
            backends.push(std::sync::Arc::new(LocalModelBackend {
                port,
                http: resolution.http.clone(),
            }));
            models.push(
                moa::ModelEntry::new(name, backend_idx)
                    .with_parameter_count_b(resolution.verified_size(name)),
            );
            *local_count += 1;
            return true;
        } else {
            tracing::info!(
                "MoA: skipping local worker {name}; context {:?} cannot fit {:?} required tokens",
                context_length,
                resolution.required_tokens
            );
        }
    }

    // Otherwise find ranked remote replicas. The backend retains one standby
    // so a draft/reducer call is not pinned to a single peer for the life of
    // the committee. `hosts_for_model` is origin-stable rendezvous order and
    // the context filter preserves that order, so failover does not create a
    // static herd on one large replica.
    let remote_hosts = context_selection::eligible_remote_hosts(
        resolution.node,
        name,
        resolution.required_tokens,
        resolution.node.hosts_for_model(name).await,
    )
    .await;
    if !remote_hosts.is_empty() {
        let backend_idx = backends.len();
        backends.push(std::sync::Arc::new(RemoteModelBackend {
            node: resolution.node.clone(),
            peer_ids: remote_hosts
                .into_iter()
                .take(super::workers::MAX_REMOTE_REPLICAS_PER_WORKER)
                .collect(),
        }));
        models.push(
            moa::ModelEntry::new(name, backend_idx)
                .with_parameter_count_b(resolution.verified_size(name)),
        );
        return true;
    }
    false
}

/// Discover and assemble the MoA worker pool: resolve one worker per distinct
/// model, apply admission control, then self-fill same-model instances.
///
/// Returns parallel `(backends, models)` vecs linked by `backend_index`.
pub(super) async fn assemble_worker_pool(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    required_tokens: Option<u32>,
    http: &reqwest::Client,
    affinity: Option<&crate::network::affinity::AffinityRouter>,
) -> (
    Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    Vec<moa::ModelEntry>,
) {
    let mut backends: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut models: Vec<moa::ModelEntry> = Vec::new();
    let mut local_count = 0usize;

    // Full mesh-wide model list (local + every peer's advertised routable
    // models).
    let all_models: Vec<String> = node
        .models_being_served()
        .await
        .into_iter()
        .filter(|n| n != moa::VIRTUAL_MODEL_NAME)
        .collect();

    // Verified sizes gossiped by peers (metadata.parameter_count_b). The
    // orchestrator has no peer GGUFs, so this is the only authoritative size
    // source — for the destructive admission/cap decisions below and for the
    // per-worker size each `ModelEntry` carries into the engine.
    let sizes = gossiped_sizes(node).await;

    // Group aliases by canonical base and resolve one worker per base, trying
    // aliases in order so a longer-named reachable alias still resolves when
    // the shortest one is offline (PR #566).
    for aliases in group_aliases_by_canonical_base(all_models, targets) {
        resolve_one_worker_from_aliases(
            node,
            targets,
            http,
            &aliases,
            required_tokens,
            &sizes,
            &mut backends,
            &mut models,
            &mut local_count,
        )
        .await;
    }

    // Admission control preserves its measured quality rule while healthy big
    // capacity exists. If fewer than two big models are currently healthy,
    // retain small/local workers as spillover rather than deleting the only
    // responsive route.
    let routing_hints = model_routing_hints(node).await;
    let healthy_big_count = models
        .iter()
        .filter(|model| tier_for(&model.name, &sizes) == SizeTier::Big)
        .filter(|model| {
            routing_hints
                .get(&canonical_base_name(&model.name))
                .map(|(availability, _)| *availability == AvailabilityRank::Healthy)
                .unwrap_or(true)
        })
        .count();
    if healthy_big_count >= 2 {
        apply_admission_control(&mut backends, &mut models, &sizes);
    }

    // Same-model fill: if only one model resolved but it is served by >=2
    // DISTINCT physical endpoints, form a committee from them. Self-MoA shows
    // repeated sampling of one model ensembles as well as different models, so
    // a same-model mesh should still get MoA. Iron law: a single physical
    // endpoint must never become a fake 2-worker committee — one node stays
    // single-model.
    if models.len() == 1 {
        self_fill_from_extra_instances(
            node,
            targets,
            required_tokens,
            http,
            &mut backends,
            &mut models,
            affinity,
        )
        .await;
    }

    // All-small pools do not convene a committee.
    //
    // Measured through the shipped path (evals/moa-openrouter/RESULTS.md),
    // 8B-class peers with an 8B reducer, vs the pool's best member alone:
    //   2x 8B:  0W/43T/37L   4x 8B: see RESULTS   6x 8B: 5W/52T/23L (p=0.0009)
    // The committee never won and lost roughly a third of decided trials, with
    // consistently shorter answers. A capable pool is the opposite (71W/8T/1L,
    // p<0.0001), so this is a statement about *this* configuration — a weak
    // reducer synthesizing weak drafts — not about small-model MoA in general.
    // The untested cell is small peers with a strong reducer; if a mesh gains a
    // big-tier model the pool stops being all-small and MoA engages again.
    //
    // So: keep only the best member rather than ship a measured committee
    // regression. The Mesh gateway still owns `model=mesh` with this single
    // worker, including gateway-only rescue semantics.
    if !models.is_empty()
        && models
            .iter()
            .all(|m| tier_for(&m.name, &sizes) == SizeTier::Small)
    {
        keep_best_member_only(&mut backends, &mut models, &sizes);
        return (backends, models);
    }

    // Committee cap: fan-out cost is ~2N+1 model calls per turn (N drafts + N
    // refines + 1 synthesis), and measured quality is flat past ~4 workers
    // while latency and spend keep climbing. On a big shared mesh (say 20
    // nodes) an uncapped pool would fan out to all of them — 41 calls for no
    // quality gain. Keep the best MAX_COMMITTEE_WORKERS by capability ranking;
    // the rest are standbys (they still serve direct traffic, just not this
    // committee).
    cap_committee(node, &mut backends, &mut models).await;

    (backends, models)
}

/// Committee width caps, by pool tier. Measured (evals/moa-openrouter,
/// aggregator = qwen3-8b, 8B-class peers):
///
/// - all-small pool: 6x diverse 8B beats solo (12W/2L, p=0.013); 4 is only
///   marginal (5W/0L, p=0.06); 2 is null. Small, weak drafts need WIDTH — more
///   independent proposals — before aggregation clears the best member.
/// - big-tier present: a 24-32B pair already wins (49W/6L, p=2e-9); extra
///   workers past ~4 are latency/cost with no measured gain.
///
/// So the cap scales with size: wide for small pools, tight when a big model is
/// present.
const COMMITTEE_CAP_SMALL: usize = 6;
const COMMITTEE_CAP_BIG: usize = 4;

/// Reduce an all-small pool to its single strongest member so the Mesh gateway
/// avoids a committee that measured worse than the member alone.
fn keep_best_member_only(
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    sizes: &HashMap<String, f64>,
) {
    // Largest verified size wins; unsized models rank last, stable index
    // breaks ties (same ordering rule as `cap_committee`).
    let best = (0..models.len())
        .max_by(|&a, &b| {
            let key = |i: usize| {
                sizes
                    .get(&canonical_base_name(&models[i].name))
                    .copied()
                    .unwrap_or(0.0)
            };
            key(a)
                .partial_cmp(&key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.cmp(&a))
        })
        .unwrap_or(0);

    tracing::info!(
        "MoA: all-small pool ({} workers) — serving best member {} directly \
         (committee measured worse than the member alone)",
        models.len(),
        models[best].name,
    );

    let backend = backends[models[best].backend_index].clone();
    let mut kept = models[best].clone();
    kept.backend_index = 0;
    *backends = vec![backend];
    *models = vec![kept];
}

fn committee_cap(models: &[moa::ModelEntry], sizes: &HashMap<String, f64>) -> usize {
    let has_big = models
        .iter()
        .any(|m| tier_for(&m.name, sizes) == SizeTier::Big);
    if has_big {
        COMMITTEE_CAP_BIG
    } else {
        COMMITTEE_CAP_SMALL
    }
}

/// Trim the pool to the best workers for its tier (see [`committee_cap`]).
async fn cap_committee(
    node: &mesh::Node,
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
) {
    let sizes = gossiped_sizes(node).await;
    let cap = committee_cap(models, &sizes);
    if models.len() <= cap {
        return;
    }
    let routing_hints = model_routing_hints(node).await;
    // Rank by verified size, NOT the tool-actor ranking. The committee serves
    // ordinary answer turns where `tool_use` is irrelevant; ranking by it
    // (i386 P1) could evict a 32B/70B model with `tool_use=None` in favour of
    // four small models whose metadata advertises tool use — the opposite of
    // the admission goal. Keep the largest verified models; a model with no
    // verified size ranks as weakest, and stable index breaks ties.
    //
    // Availability sorts *within* a tier, never across one. Admission control
    // already ran, so this is the last stage that can drop a worker: without
    // the availability term, enough deprioritized bases of the same tier fill
    // the cap by index alone and evict the only healthy worker, leaving
    // `compute_actor_candidates` to rank a pool with nothing healthy left in
    // it. Keeping it below the tier key preserves the measured admission rule
    // — a healthy small model still never displaces a big one.
    let mut ranked: Vec<usize> = (0..models.len()).collect();
    ranked.sort_by(|&a, &b| {
        let tier_key = |i: usize| match tier_for(&models[i].name, &sizes) {
            SizeTier::Big => 0,
            SizeTier::Small => 1,
        };
        let availability = |i: usize| {
            routing_hints
                .get(&canonical_base_name(&models[i].name))
                .map_or(AvailabilityRank::Healthy, |(rank, _)| *rank)
        };
        tier_key(a)
            .cmp(&tier_key(b))
            .then_with(|| availability(a).cmp(&availability(b)))
            .then_with(|| a.cmp(&b))
    });
    let keep: std::collections::HashSet<usize> = ranked.into_iter().take(cap).collect();

    let mut kept_backends: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut kept_models: Vec<moa::ModelEntry> = Vec::new();
    for (i, m) in models.iter().enumerate() {
        if !keep.contains(&i) {
            tracing::info!("MoA: capping committee, dropping worker {}", m.name);
            continue;
        }
        let new_idx = kept_backends.len();
        kept_backends.push(backends[m.backend_index].clone());
        kept_models.push(moa::ModelEntry {
            backend_index: new_idx,
            ..m.clone()
        });
    }
    *backends = kept_backends;
    *models = kept_models;
}

/// Drop small-tier workers when any big-tier worker is present.
///
/// A weak draft can contaminate synthesis, and aggregation quality tracks
/// proposal quality (Self-MoA, arXiv:2502.00674), so a modest node must not be
/// admitted into a committee that already has a stronger member. When the pool
/// is mixed we keep only the big-tier workers; an all-small or all-big pool is
/// left untouched. `backends` and `models` are parallel vecs linked by
/// `backend_index`, so we rebuild both and reindex.
fn apply_admission_control(
    backends: &mut Vec<std::sync::Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    sizes: &HashMap<String, f64>,
) {
    // Only *verified* big-tier models count as strong, and only *verified*
    // small-tier models are eligible for exclusion. An `Unknown` size is never
    // treated as strong (so it can't anchor the ">=2 big" gate) and is never
    // filtered out (so an unverified label can't get a worker dropped). This
    // is the i386 P1 fix: a destructive admission decision must rest on
    // verified size, not a guessed tier.
    let big_count = models
        .iter()
        .filter(|m| tier_for(&m.name, sizes) == SizeTier::Big)
        .count();
    let has_small = models
        .iter()
        .any(|m| tier_for(&m.name, sizes) == SizeTier::Small);
    // Only exclude small-tier workers when doing so still leaves a committee
    // (>=2 big-tier). Measured:
    //   * 32B x2 + 8B  -> dropping the 8B leaves 32B x2, and the 8B added
    //     nothing (arm C: no upside, losses 2->5) — so drop it.
    //   * 32B + 8B     -> dropping the 8B collapses to a solo 32B, but the
    //     mixed committee beats solo decisively (47W/27T/5L, p=1e-9) — so
    //     KEEP the 8B. Admission must not throw away MoA to protect a pool
    //     that no longer exists.
    // See `evals/moa-openrouter/RESULTS.md`.
    if !(has_small && big_count >= 2) {
        return;
    }

    let mut kept_backends: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
    let mut kept_models: Vec<moa::ModelEntry> = Vec::new();
    for m in models.iter() {
        if tier_for(&m.name, sizes) == SizeTier::Small {
            tracing::info!(
                "MoA: excluding verified small-tier worker {} (>=2 big-tier present)",
                m.name
            );
            continue;
        }
        let new_idx = kept_backends.len();
        kept_backends.push(backends[m.backend_index].clone());
        kept_models.push(moa::ModelEntry {
            backend_index: new_idx,
            ..m.clone()
        });
    }
    *backends = kept_backends;
    *models = kept_models;
}

/// Rank the pool best-tool-caller-first (indices into `models`) for the actor.
///
/// Ordering: gossiped `tool_use` (`Supported` > `Likely` > `None`), then size
/// tier, then stable index. Capabilities match pool entries by canonical base
/// name (so `unsloth/Qwen3-8B-GGUF:Q4_K_M` supplies `Qwen3-8B-Q4_K_M`). Always
/// returns a full ranking; the engine reads an empty vec as "no host guidance".
pub(super) async fn compute_actor_candidates(
    node: &mesh::Node,
    models: &[moa::ModelEntry],
) -> Vec<usize> {
    // canonical base name -> best tool_use level seen across the mesh.
    let mut tool_use_by_base: std::collections::HashMap<String, crate::models::CapabilityLevel> =
        std::collections::HashMap::new();
    for descriptor in node.all_served_model_descriptors().await {
        let base = canonical_base_name(&descriptor.identity.model_name);
        let level = descriptor.capabilities.tool_use;
        tool_use_by_base
            .entry(base)
            .and_modify(|existing| {
                if level > *existing {
                    *existing = level;
                }
            })
            .or_insert(level);
    }

    let routing_hints = model_routing_hints(node).await;
    let mut ranked: Vec<usize> = (0..models.len()).collect();
    ranked.sort_by(|&a, &b| {
        let ma = &models[a];
        let mb = &models[b];
        let base_a = canonical_base_name(&ma.name);
        let base_b = canonical_base_name(&mb.name);
        let tool_a = tool_use_by_base
            .get(&base_a)
            .copied()
            .unwrap_or(crate::models::CapabilityLevel::None);
        let tool_b = tool_use_by_base
            .get(&base_b)
            .copied()
            .unwrap_or(crate::models::CapabilityLevel::None);
        let (availability_a, throughput_a) = routing_hints
            .get(&base_a)
            .copied()
            .unwrap_or((AvailabilityRank::Healthy, None));
        let (availability_b, throughput_b) = routing_hints
            .get(&base_b)
            .copied()
            .unwrap_or((AvailabilityRank::Healthy, None));
        // 1) higher tool_use first
        tool_b
            .cmp(&tool_a)
            // 2) healthy before explicitly deprioritized. This is deliberately
            // ahead of size so a healthy local/small model can absorb spillover.
            .then_with(|| availability_a.cmp(&availability_b))
            // 3) big-tier before small-tier when health is equal
            .then_with(|| {
                let small_a = moa::entry_is_small_tier(ma);
                let small_b = moa::entry_is_small_tier(mb);
                small_a.cmp(&small_b) // false (big) sorts before true (small)
            })
            // 4) faster advertised service rate within the same health/tier.
            // This is a historical capability hint, not live load pressure.
            .then_with(|| throughput_b.cmp(&throughput_a))
            // 5) stable index order
            .then_with(|| a.cmp(&b))
    });
    ranked
}

/// Canonical name used for cross-peer dedup. Different peers advertise the
/// same model under different conventions (`unsloth/Qwen3-8B-GGUF:Q4_K_M`
/// vs `Qwen3-8B-Q4_K_M`); normalize before comparing.
///
/// Strategy: strip the publisher prefix, the `-gguf` suffix, any `@branch`
/// suffix, then keep only `[a-z0-9]` characters so `:` vs `-` separators
/// don't matter.
pub(super) fn canonical_base_name(name: &str) -> String {
    let lower = name.to_lowercase();
    // Drop an `@branch` segment if present, keeping anything after the
    // next `:` so quant tags survive (e.g. `repo@main:q4_k_m` → `repo:q4_k_m`).
    let no_branch = match lower.find('@') {
        Some(at) => {
            let after = &lower[at + 1..];
            let rest = after.find(':').map(|c| &after[c..]).unwrap_or("");
            format!("{}{}", &lower[..at], rest)
        }
        None => lower,
    };
    let stripped = no_branch
        .replace("-gguf", "")
        .replace("unsloth/", "")
        .replace("meshllm/", "");
    stripped
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal backend stub for admission-control tests.
    struct FakeBackend;
    #[async_trait::async_trait]
    impl moa::ModelBackend for FakeBackend {
        async fn chat_completion(
            &self,
            _model: &str,
            _messages: &[serde_json::Value],
            _tools: Option<&serde_json::Value>,
            _max_tokens: u32,
            _timeout: std::time::Duration,
            _sampling: moa::SamplingParams,
        ) -> Result<serde_json::Value, String> {
            Ok(serde_json::json!({"choices":[{"message":{"content":"x"}}]}))
        }
    }

    fn pool(
        names: &[&str],
    ) -> (
        Vec<std::sync::Arc<dyn moa::ModelBackend>>,
        Vec<moa::ModelEntry>,
    ) {
        let mut b: Vec<std::sync::Arc<dyn moa::ModelBackend>> = Vec::new();
        let mut m = Vec::new();
        for name in names {
            m.push(moa::ModelEntry::new(*name, b.len()));
            b.push(std::sync::Arc::new(FakeBackend));
        }
        (b, m)
    }

    /// Build a verified-size map keyed by canonical base name, as if gossiped.
    fn sizes_of(entries: &[(&str, f64)]) -> HashMap<String, f64> {
        entries
            .iter()
            .map(|(n, b)| (canonical_base_name(n), *b))
            .collect()
    }

    #[test]
    fn all_small_pool_keeps_only_the_best_member() {
        // Measured: an all-small committee never beat its best member and lost
        // ~a third of decided trials (2x8B 0W/37L; 6x8B 5W/23L p=0.0009). So an
        // all-small pool collapses to the single strongest gateway worker.
        let (mut b, mut m) = pool(&["Qwen3-8B", "Llama-3.1-8B", "Qwen3.5-9B"]);
        let sizes = sizes_of(&[
            ("Qwen3-8B", 8.0),
            ("Llama-3.1-8B", 8.0),
            ("Qwen3.5-9B", 9.0),
        ]);
        keep_best_member_only(&mut b, &mut m, &sizes);
        assert_eq!(m.len(), 1, "all-small pool must collapse to one member");
        assert_eq!(m[0].name, "Qwen3.5-9B", "largest verified size wins");
        assert_eq!(b.len(), 1);
        assert_eq!(m[0].backend_index, 0, "backend index reindexed");
    }

    /// The verified size must travel with the entry into the engine — that is
    /// the whole point of resolving it host-side. Re-indexing steps (admission
    /// control, committee capping, the all-small collapse) rebuild entries, so
    /// they must carry the size across rather than reconstruct a bare entry.
    #[test]
    fn reindexing_preserves_the_verified_size() {
        // Two verified bigs plus a small, the shape where admission actually
        // drops the small worker and rebuilds the surviving entries.
        let (mut b, mut m) = pool(&["Qwen3-8B", "Llama-3.3-70B-Instruct-4bit", "Qwen3-32B"]);
        m[0].parameter_count_b = Some(8.2);
        m[1].parameter_count_b = Some(70.6);
        m[2].parameter_count_b = Some(32.8);
        let sizes = sizes_of(&[
            ("Qwen3-8B", 8.2),
            ("Llama-3.3-70B-Instruct-4bit", 70.6),
            ("Qwen3-32B", 32.8),
        ]);

        apply_admission_control(&mut b, &mut m, &sizes);

        assert_eq!(m.len(), 2, "the verified small worker is excluded");
        let quantised = m
            .iter()
            .find(|entry| entry.name == "Llama-3.3-70B-Instruct-4bit")
            .expect("the verified 70B must survive admission");
        assert_eq!(
            quantised.parameter_count_b,
            Some(70.6),
            "re-indexing must preserve the verified size, or the engine falls \
             back to parsing '-4bit' as a size and calls this model small"
        );
        for (i, entry) in m.iter().enumerate() {
            assert_eq!(
                entry.backend_index, i,
                "backend indices are rebuilt densely"
            );
        }
    }

    /// The all-small collapse also rebuilds its single entry.
    #[test]
    fn best_member_collapse_preserves_the_verified_size() {
        let (mut b, mut m) = pool(&["Qwen3-8B", "Qwen3.5-9B"]);
        m[0].parameter_count_b = Some(8.0);
        m[1].parameter_count_b = Some(9.0);
        let sizes = sizes_of(&[("Qwen3-8B", 8.0), ("Qwen3.5-9B", 9.0)]);

        keep_best_member_only(&mut b, &mut m, &sizes);

        assert_eq!(m[0].name, "Qwen3.5-9B");
        assert_eq!(m[0].parameter_count_b, Some(9.0));
    }

    #[test]
    fn best_member_falls_back_to_stable_order_when_unsized() {
        // No verified sizes: every member ranks equal, so the stable first
        // entry is kept rather than an arbitrary one.
        let (mut b, mut m) = pool(&["alpha-model", "beta-model"]);
        let sizes = sizes_of(&[]);
        keep_best_member_only(&mut b, &mut m, &sizes);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "alpha-model");
    }

    #[test]
    fn admission_drops_small_when_two_big_remain() {
        // Dropping the small workers still leaves a committee (2x 32B), and the
        // small drafts add nothing there — so exclude them.
        let (mut b, mut m) = pool(&["Qwen3-32B", "Qwen3-32B", "Qwen3-8B", "Ministral-8B"]);
        let sizes = sizes_of(&[
            ("Qwen3-32B", 32.0),
            ("Qwen3-8B", 8.0),
            ("Ministral-8B", 8.0),
        ]);
        apply_admission_control(&mut b, &mut m, &sizes);
        assert_eq!(m.len(), 2);
        assert!(m.iter().all(|e| e.name == "Qwen3-32B"));
        assert_eq!(b.len(), 2);
        // backends stay aligned and reindexed
        assert_eq!(m[0].backend_index, 0);
        assert_eq!(m[1].backend_index, 1);
    }

    #[test]
    fn admission_keeps_mix_when_dropping_would_collapse_to_solo() {
        // One strong + one weak: dropping the 8B leaves a solo 32B, but the
        // mixed committee beats solo (47W/27T/5L) — so keep the mix.
        let (mut b, mut m) = pool(&["Qwen3-32B", "Qwen3-8B"]);
        let sizes = sizes_of(&[("Qwen3-32B", 32.0), ("Qwen3-8B", 8.0)]);
        apply_admission_control(&mut b, &mut m, &sizes);
        assert_eq!(m.len(), 2, "must not collapse a lone-strong pool to solo");
    }

    #[test]
    fn admission_keeps_all_small_pool() {
        let (mut b, mut m) = pool(&["Qwen3-8B", "Llama-3.1-8B", "Ministral-8B"]);
        let sizes = sizes_of(&[
            ("Qwen3-8B", 8.0),
            ("Llama-3.1-8B", 8.0),
            ("Ministral-8B", 8.0),
        ]);
        apply_admission_control(&mut b, &mut m, &sizes);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn admission_keeps_all_big_pool() {
        let (mut b, mut m) = pool(&["Qwen3-32B", "Mistral-Small-24B"]);
        let sizes = sizes_of(&[("Qwen3-32B", 32.0), ("Mistral-Small-24B", 24.0)]);
        apply_admission_control(&mut b, &mut m, &sizes);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn admission_keeps_homogeneous_big_pool() {
        let (mut b, mut m) = pool(&["Qwen3-32B", "Qwen3-32B"]);
        let sizes = sizes_of(&[("Qwen3-32B", 32.0)]);
        apply_admission_control(&mut b, &mut m, &sizes);
        assert_eq!(m.len(), 2);
    }

    /// i386: a model with NO verified GGUF size is the weakest (Small), so
    /// next to two verified big models it is excluded like any other small
    /// worker. An unverifiable label can never masquerade as big.
    #[test]
    fn admission_treats_unsized_worker_as_weakest() {
        let (mut b, mut m) = pool(&["Qwen3-32B", "Qwen3-32B", "my-assistant"]);
        // Only the two big models have gossiped sizes; "my-assistant" has none.
        let sizes = sizes_of(&[("Qwen3-32B", 32.0)]);
        apply_admission_control(&mut b, &mut m, &sizes);
        assert_eq!(m.len(), 2, "unsized worker ranks weakest and is excluded");
        assert!(m.iter().all(|e| e.name == "Qwen3-32B"));
    }

    /// No gossiped sizes at all: every worker is weakest (Small), so it is an
    /// all-small pool and admission excludes nothing (there is no verified big
    /// to protect). Name is never consulted.
    #[test]
    fn admission_keeps_all_when_no_verified_sizes() {
        let (mut b, mut m) = pool(&["Qwen3-32B", "Qwen3-32B", "Qwen3-8B"]);
        let sizes: HashMap<String, f64> = HashMap::new();
        apply_admission_control(&mut b, &mut m, &sizes);
        assert_eq!(
            m.len(),
            3,
            "no verified big => all-small pool => nothing excluded"
        );
    }

    #[test]
    fn committee_cap_is_wide_for_small_pools_tight_for_big() {
        // All-small pool: cap is wide (6) so a 6× 8B mesh keeps its width —
        // measured 12W/2L p=0.013 at width 6, only marginal at 4.
        let small: Vec<moa::ModelEntry> = (0..8)
            .map(|i| moa::ModelEntry::new("Qwen3-8B".to_string(), i))
            .collect();
        let sizes = sizes_of(&[("Qwen3-8B", 8.0)]);
        assert_eq!(committee_cap(&small, &sizes), COMMITTEE_CAP_SMALL);

        // A verified big model present -> tight cap (4); extra workers past a
        // 24–32B pair buy nothing.
        let mixed = vec![
            moa::ModelEntry::new("Qwen3-32B".to_string(), 0),
            moa::ModelEntry::new("Qwen3-8B".to_string(), 1),
        ];
        let sizes = sizes_of(&[("Qwen3-32B", 32.0), ("Qwen3-8B", 8.0)]);
        assert_eq!(committee_cap(&mixed, &sizes), COMMITTEE_CAP_BIG);
    }

    #[test]
    fn canonical_base_dedupes_unsloth_and_gguf_variants() {
        assert_eq!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF:Q4_K_M"),
            canonical_base_name("Qwen3-8B-Q4_K_M")
        );
        assert_eq!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF@main:Q4_K_M"),
            canonical_base_name("Qwen3-8B-Q4_K_M")
        );
    }

    #[test]
    fn canonical_base_keeps_distinct_models_distinct() {
        assert_ne!(
            canonical_base_name("unsloth/Qwen3-8B-GGUF:Q4_K_M"),
            canonical_base_name("unsloth/Qwen3-32B-GGUF:Q4_K_M")
        );
        assert_ne!(
            canonical_base_name("unsloth/Qwen3-32B-GGUF:Q4_K_M"),
            canonical_base_name("unsloth/MiniMax-M2.5-GGUF:Q4_K_M")
        );
    }
    fn make_targets(local_names: &[&str]) -> election::ModelTargets {
        let mut t = election::ModelTargets::default();
        for (i, name) in local_names.iter().enumerate() {
            t.targets.insert(
                (*name).to_string(),
                vec![election::InferenceTarget::Local(50000 + i as u16)],
            );
        }
        t
    }

    #[test]
    fn group_aliases_keeps_all_aliases_per_canonical_base() {
        // Regression for PR #566 review (item #10): the dedup-then-resolve
        // shape committed to a single alias per base before checking
        // backend reachability. Now every alias is retained so the
        // resolver can fall back if the preferred alias is unreachable.
        let groups = group_aliases_by_canonical_base(
            vec![
                "Qwen3-8B-Q4_K_M".to_string(),
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 1, "both names share a canonical base");
        assert_eq!(groups[0].len(), 2, "both aliases retained");
    }

    #[test]
    fn group_aliases_prefers_locally_served_alias_even_when_longer() {
        // Without a targets table, length-order wins and the shorter peer
        // alias would be tried first — forcing an unnecessary QUIC hop
        // when the model is right here under a different alias.
        // With targets, the local-served alias must come first.
        let local = "unsloth/Qwen3-8B-GGUF:Q4_K_M";
        let peer = "Qwen3-8B-Q4_K_M";
        let targets = make_targets(&[local]);
        let groups = group_aliases_by_canonical_base(
            vec![peer.to_string(), local.to_string()],
            Some(&targets),
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].first().map(String::as_str),
            Some(local),
            "locally-served alias must win even though it's longer"
        );
    }

    #[test]
    fn group_aliases_falls_back_to_shortest_when_no_local() {
        // No targets table at all (pure --client --auto node) — shortest
        // alias should win, but the longer alias is still in the group so
        // it can be tried if the shortest one is unreachable.
        let groups = group_aliases_by_canonical_base(
            vec![
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
                "Qwen3-8B-Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].first().map(String::as_str),
            Some("Qwen3-8B-Q4_K_M")
        );
        assert_eq!(groups[0].len(), 2, "longer alias kept as fallback");
    }

    #[test]
    fn group_aliases_distinct_models_stay_in_separate_groups() {
        let groups = group_aliases_by_canonical_base(
            vec![
                "unsloth/Qwen3-8B-GGUF:Q4_K_M".to_string(),
                "unsloth/Qwen3-32B-GGUF:Q4_K_M".to_string(),
                "unsloth/MiniMax-M2.5-GGUF:Q4_K_M".to_string(),
            ],
            None,
        );
        assert_eq!(groups.len(), 3);
    }
}
