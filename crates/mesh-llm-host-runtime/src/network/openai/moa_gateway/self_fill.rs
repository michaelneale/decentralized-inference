//! Same-model committees: distinct physical clones, reserved for the turn.
use super::workers::{LocalModelBackend, RemoteModelBackend, ReservedModelBackend};
use crate::inference::election::{InferenceTarget, ModelTargets};
use crate::mesh;
use crate::network::affinity::AffinityRouter;
use crate::network::openai::routing_rank::rank_targets_by_context;
use crate::network::reservations::RoutingReservation;
use mesh_mixture_of_agents as moa;
use std::sync::Arc;

/// Measured self-MoA width; fleet capacity must not increase fan-out cost.
const SELF_FILL_TARGET_WORKERS: usize = 2;

async fn select_clones(
    node: &mesh::Node,
    name: &str,
    required_tokens: Option<u32>,
    mut candidates: Vec<InferenceTarget>,
    affinity: Option<&AffinityRouter>,
) -> Vec<(InferenceTarget, Option<RoutingReservation>)> {
    let mut selected = Vec::with_capacity(SELF_FILL_TARGET_WORKERS);
    while selected.len() < SELF_FILL_TARGET_WORKERS {
        // Re-rank the remaining endpoints so slot two can use the next tier
        // only when slot one exhausted the best tier, never due to load alone.
        let ranked = rank_targets_by_context(node, name, required_tokens, &candidates).await;
        let Some(preferred) = ranked.ordered.first() else {
            break;
        };
        let (target, reservation) = affinity
            .and_then(|router| {
                router.reserve_route(
                    name,
                    &ranked.ordered,
                    ranked.equivalent_prefix,
                    preferred,
                    false,
                )
            })
            .map(|(target, guard)| (target, Some(guard)))
            .unwrap_or_else(|| (preferred.clone(), None));
        // Selection+reservation is atomic per slot; removing the endpoint
        // prevents duplicate workers even when other turns interleave slots.
        candidates.retain(|candidate| candidate != &target);
        selected.push((target, reservation));
    }
    selected
}

pub(super) async fn self_fill_from_extra_instances(
    node: &mesh::Node,
    targets: Option<&ModelTargets>,
    required_tokens: Option<u32>,
    http: &reqwest::Client,
    backends: &mut Vec<Arc<dyn moa::ModelBackend>>,
    models: &mut Vec<moa::ModelEntry>,
    affinity: Option<&AffinityRouter>,
) {
    let Some(existing) = models.first().cloned() else {
        return;
    };
    let name = &existing.name;
    let mut candidates = Vec::new();
    if let Some(local) = targets
        .and_then(|targets| targets.targets.get(name))
        .and_then(|targets| {
            targets
                .iter()
                .find(|t| matches!(t, InferenceTarget::Local(_)))
        })
    {
        candidates.push(local.clone());
    }
    candidates.extend(
        node.hosts_for_model(name)
            .await
            .into_iter()
            .map(InferenceTarget::Remote),
    );
    if candidates.len() < 2 {
        return;
    }
    let selected = select_clones(node, name, required_tokens, candidates, affinity).await;
    if selected.len() < 2 {
        return; // Context filtering must not fabricate a second worker.
    }
    *backends = selected
        .into_iter()
        .map(|(target, reservation)| {
            let inner: Arc<dyn moa::ModelBackend> = match target {
                InferenceTarget::Local(port) => Arc::new(LocalModelBackend {
                    port,
                    http: http.clone(),
                }),
                // No failover onto a sibling slot: every worker is a distinct sample.
                InferenceTarget::Remote(peer_id) => Arc::new(RemoteModelBackend {
                    node: node.clone(),
                    peer_ids: vec![peer_id],
                }),
                InferenceTarget::None => unreachable!("self-fill only collects physical endpoints"),
            };
            match reservation {
                Some(reservation) => Arc::new(ReservedModelBackend {
                    inner,
                    _reservation: reservation,
                }) as Arc<dyn moa::ModelBackend>,
                None => inner,
            }
        })
        .collect();
    *models = (0..backends.len())
        .map(|backend_index| moa::ModelEntry {
            backend_index,
            ..existing.clone()
        })
        .collect();
}

#[cfg(test)]
mod tests;
