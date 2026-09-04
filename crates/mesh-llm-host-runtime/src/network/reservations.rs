//! Bounded process-local reservations for in-flight inference routes.

use crate::inference::election;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const RESERVATION_TTL: Duration = Duration::from_secs(10 * 60);
const RESERVATION_MAX_ACTIVE: usize = 4096;

#[derive(Clone, Debug, Default)]
pub(crate) struct ReservationStatsSnapshot {
    pub(crate) active: usize,
    pub(crate) active_models: usize,
    pub(crate) created: u64,
    pub(crate) spread_selections: u64,
    pub(crate) transferred: u64,
    pub(crate) released: u64,
    pub(crate) expired: u64,
    pub(crate) capacity_evictions: u64,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReservationKey {
    model: String,
    target: election::InferenceTarget,
}

struct ReservationEntry {
    key: ReservationKey,
    expires_at: Instant,
}

#[derive(Default)]
struct ReservationState {
    entries: HashMap<u64, ReservationEntry>,
    counts: HashMap<ReservationKey, usize>,
    next_id: u64,
    created: u64,
    spread_selections: u64,
    transferred: u64,
    released: u64,
    expired: u64,
    capacity_evictions: u64,
}

impl ReservationState {
    fn count(&self, model: &str, target: &election::InferenceTarget) -> usize {
        self.counts
            .get(&ReservationKey {
                model: model.to_string(),
                target: target.clone(),
            })
            .copied()
            .unwrap_or(0)
    }

    fn increment(&mut self, key: ReservationKey) {
        *self.counts.entry(key).or_insert(0) += 1;
    }

    fn decrement(&mut self, key: &ReservationKey) {
        let Some(count) = self.counts.get_mut(key) else {
            return;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.counts.remove(key);
        }
    }

    fn remove_entry(&mut self, id: u64) -> Option<ReservationEntry> {
        let entry = self.entries.remove(&id)?;
        self.decrement(&entry.key);
        Some(entry)
    }

    fn prune_expired(&mut self, now: Instant) {
        let expired = self
            .entries
            .iter()
            .filter_map(|(id, entry)| (entry.expires_at <= now).then_some(*id))
            .collect::<Vec<_>>();
        for id in expired {
            if self.remove_entry(id).is_some() {
                self.expired = self.expired.saturating_add(1);
            }
        }
    }

    fn ensure_capacity(&mut self) {
        let Some((&oldest_id, _)) = self
            .entries
            .iter()
            .min_by_key(|(id, entry)| (entry.expires_at, **id))
        else {
            return;
        };
        if self.remove_entry(oldest_id).is_some() {
            self.capacity_evictions = self.capacity_evictions.saturating_add(1);
        }
    }

    fn next_id(&mut self) -> u64 {
        loop {
            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1);
            if !self.entries.contains_key(&id) {
                return id;
            }
        }
    }

    fn snapshot(&self) -> ReservationStatsSnapshot {
        ReservationStatsSnapshot {
            active: self.entries.len(),
            active_models: self
                .counts
                .keys()
                .map(|key| key.model.as_str())
                .collect::<HashSet<_>>()
                .len(),
            created: self.created,
            spread_selections: self.spread_selections,
            transferred: self.transferred,
            released: self.released,
            expired: self.expired,
            capacity_evictions: self.capacity_evictions,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RoutingReservations {
    inner: Arc<Mutex<ReservationState>>,
    ttl: Duration,
    max_active: usize,
}

impl Default for RoutingReservations {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ReservationState::default())),
            ttl: RESERVATION_TTL,
            max_active: RESERVATION_MAX_ACTIVE,
        }
    }
}

impl RoutingReservations {
    #[cfg(test)]
    fn with_limits(ttl: Duration, max_active: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ReservationState::default())),
            ttl,
            max_active,
        }
    }

    /// Choose and reserve a target in one critical section. Existing affinity
    /// stays authoritative; otherwise the current picker result is the stable
    /// tie-breaker among targets with the fewest local reservations.
    pub(crate) fn reserve(
        &self,
        model: &str,
        candidates: &[election::InferenceTarget],
        preferred: &election::InferenceTarget,
        affinity_applied: bool,
    ) -> Option<(election::InferenceTarget, RoutingReservation)> {
        if candidates.is_empty() || self.max_active == 0 {
            return None;
        }
        let now = Instant::now();
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(now);

        let preferred_index = candidates
            .iter()
            .position(|candidate| candidate == preferred)
            .unwrap_or(0);
        let target = if affinity_applied {
            candidates[preferred_index].clone()
        } else {
            let minimum = candidates
                .iter()
                .map(|candidate| state.count(model, candidate))
                .min()
                .unwrap_or(0);
            (0..candidates.len())
                .map(|offset| (preferred_index + offset) % candidates.len())
                .find(|index| state.count(model, &candidates[*index]) == minimum)
                .map(|index| candidates[index].clone())
                .unwrap_or_else(|| candidates[preferred_index].clone())
        };

        if state.entries.len() >= self.max_active {
            state.ensure_capacity();
        }
        let id = state.next_id();
        let key = ReservationKey {
            model: model.to_string(),
            target: target.clone(),
        };
        state.increment(key.clone());
        state.entries.insert(
            id,
            ReservationEntry {
                key,
                expires_at: now + self.ttl,
            },
        );
        state.created = state.created.saturating_add(1);
        if target != *preferred {
            state.spread_selections = state.spread_selections.saturating_add(1);
        }
        drop(state);

        Some((
            target,
            RoutingReservation {
                reservations: self.clone(),
                id: Some(id),
            },
        ))
    }

    pub(crate) fn stats_snapshot(&self) -> ReservationStatsSnapshot {
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(Instant::now());
        state.snapshot()
    }

    fn transfer(&self, id: u64, target: &election::InferenceTarget) {
        let now = Instant::now();
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(now);
        let Some(mut entry) = state.entries.remove(&id) else {
            return;
        };
        if entry.key.target == *target {
            state.entries.insert(id, entry);
            return;
        }
        state.decrement(&entry.key);
        entry.key.target = target.clone();
        entry.expires_at = now + self.ttl;
        state.increment(entry.key.clone());
        state.entries.insert(id, entry);
        state.transferred = state.transferred.saturating_add(1);
    }

    fn release(&self, id: u64) {
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(Instant::now());
        if state.remove_entry(id).is_some() {
            state.released = state.released.saturating_add(1);
        }
    }

    #[cfg(test)]
    fn active_count(&self, model: &str, target: &election::InferenceTarget) -> usize {
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(Instant::now());
        state.count(model, target)
    }
}

pub(crate) struct RoutingReservation {
    reservations: RoutingReservations,
    id: Option<u64>,
}

impl RoutingReservation {
    pub(crate) fn transfer_to(&mut self, target: &election::InferenceTarget) {
        if let Some(id) = self.id {
            self.reservations.transfer(id, target);
        }
    }
}

impl Drop for RoutingReservation {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            self.reservations.release(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Barrier;

    fn local(port: u16) -> election::InferenceTarget {
        election::InferenceTarget::Local(port)
    }

    fn concurrent_burst(
        targets: Vec<election::InferenceTarget>,
        requests: usize,
    ) -> BTreeMap<u16, usize> {
        let reservations = RoutingReservations::default();
        let barrier = Arc::new(Barrier::new(requests));
        let selected = Arc::new(Mutex::new(Vec::new()));
        std::thread::scope(|scope| {
            for _ in 0..requests {
                let reservations = reservations.clone();
                let barrier = Arc::clone(&barrier);
                let selected = Arc::clone(&selected);
                let targets = targets.clone();
                scope.spawn(move || {
                    let (target, reservation) = reservations
                        .reserve("model", &targets, &targets[0], false)
                        .expect("reservation");
                    selected.lock().unwrap().push(target);
                    barrier.wait();
                    drop(reservation);
                });
            }
        });
        let mut counts = BTreeMap::new();
        for target in selected.lock().unwrap().drain(..) {
            let election::InferenceTarget::Local(port) = target else {
                panic!("expected local target");
            };
            *counts.entry(port).or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn concurrent_new_requests_spread_across_two_targets() {
        assert_eq!(
            concurrent_burst(vec![local(1), local(2)], 8),
            BTreeMap::from([(1, 4), (2, 4)])
        );
    }

    #[test]
    fn concurrent_new_requests_spread_across_three_targets() {
        assert_eq!(
            concurrent_burst(vec![local(1), local(2), local(3)], 9),
            BTreeMap::from([(1, 3), (2, 3), (3, 3)])
        );
    }

    #[test]
    fn single_target_routing_is_unchanged() {
        assert_eq!(
            concurrent_burst(vec![local(1)], 8),
            BTreeMap::from([(1, 8)])
        );
    }

    #[test]
    fn established_affinity_ignores_reservation_pressure() {
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        let (_, _first) = reservations
            .reserve("model", &targets, &targets[0], false)
            .expect("first reservation");
        let (selected, _sticky) = reservations
            .reserve("model", &targets, &targets[0], true)
            .expect("sticky reservation");

        assert_eq!(selected, targets[0]);
        assert_eq!(reservations.active_count("model", &targets[0]), 2);
        assert_eq!(reservations.active_count("model", &targets[1]), 0);
    }

    #[test]
    fn reservation_pressure_is_isolated_per_model() {
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        let (_, _model_a) = reservations
            .reserve("model-a", &targets, &targets[0], false)
            .expect("model-a reservation");
        let (selected, _model_b) = reservations
            .reserve("model-b", &targets, &targets[0], false)
            .expect("model-b reservation");

        assert_eq!(selected, targets[0]);
        assert_eq!(reservations.active_count("model-a", &targets[0]), 1);
        assert_eq!(reservations.active_count("model-b", &targets[0]), 1);
    }

    #[test]
    fn failover_transfers_and_drop_releases_reservation() {
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        let (_, mut reservation) = reservations
            .reserve("model", &targets, &targets[0], false)
            .expect("reservation");

        reservation.transfer_to(&targets[1]);
        assert_eq!(reservations.active_count("model", &targets[0]), 0);
        assert_eq!(reservations.active_count("model", &targets[1]), 1);
        drop(reservation);

        let stats = reservations.stats_snapshot();
        assert_eq!(stats.active, 0);
        assert_eq!(stats.transferred, 1);
        assert_eq!(stats.released, 1);
    }

    #[tokio::test]
    async fn task_cancellation_releases_reservation() {
        let reservations = RoutingReservations::default();
        let target = local(1);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let task_reservations = reservations.clone();
        let task_target = target.clone();
        let task = tokio::spawn(async move {
            let (_, reservation) = task_reservations
                .reserve(
                    "model",
                    std::slice::from_ref(&task_target),
                    &task_target,
                    false,
                )
                .expect("reservation");
            ready_tx.send(()).expect("signal active reservation");
            std::future::pending::<()>().await;
            drop(reservation);
        });

        ready_rx.await.expect("reservation became active");
        assert_eq!(reservations.active_count("model", &target), 1);
        task.abort();
        assert!(
            task.await
                .expect_err("task must be cancelled")
                .is_cancelled()
        );

        let stats = reservations.stats_snapshot();
        assert_eq!(stats.active, 0);
        assert_eq!(stats.released, 1);
    }

    #[test]
    fn abandoned_reservations_expire_defensively() {
        let reservations = RoutingReservations::with_limits(Duration::ZERO, 4);
        let target = local(1);
        let (_selected, reservation) = reservations
            .reserve("model", std::slice::from_ref(&target), &target, false)
            .expect("reservation");

        let stats = reservations.stats_snapshot();
        assert_eq!(stats.active, 0);
        assert_eq!(stats.expired, 1);
        drop(reservation);
        assert_eq!(reservations.stats_snapshot().released, 0);
    }

    #[test]
    fn active_reservations_stay_within_the_capacity_bound() {
        let reservations = RoutingReservations::with_limits(Duration::from_secs(60), 2);
        let target = local(1);
        let mut guards = Vec::new();
        for _ in 0..3 {
            let (_, reservation) = reservations
                .reserve("model", std::slice::from_ref(&target), &target, false)
                .expect("reservation");
            guards.push(reservation);
        }

        let stats = reservations.stats_snapshot();
        assert_eq!(stats.active, 2);
        assert_eq!(stats.capacity_evictions, 1);
        drop(guards);
        assert_eq!(reservations.stats_snapshot().active, 0);
    }
}
