//! Process-local in-flight counts used to spread concurrent new sessions
//! across equivalent inference targets.

use crate::inference::election;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReservationKey {
    model: String,
    target: election::InferenceTarget,
}

type ReservationCounts = Arc<Mutex<HashMap<ReservationKey, usize>>>;

fn decrement(counts: &mut HashMap<ReservationKey, usize>, key: &ReservationKey) {
    if let Some(count) = counts.get_mut(key) {
        *count = count.saturating_sub(1);
        if *count == 0 {
            counts.remove(key);
        }
    }
}

/// Per-process counts of requests currently in flight to each `(model,
/// target)` pair. Entries exist only while a [`RoutingReservation`] guard is
/// alive, so the map stays bounded by the number of in-flight requests.
#[derive(Clone, Default)]
pub(crate) struct RoutingReservations {
    counts: ReservationCounts,
}

impl RoutingReservations {
    /// Choose and reserve a target in one critical section. Existing affinity
    /// stays authoritative; otherwise the current picker result is the stable
    /// tie-breaker among targets with the fewest local in-flight requests.
    pub(crate) fn reserve(
        &self,
        model: &str,
        candidates: &[election::InferenceTarget],
        preferred: &election::InferenceTarget,
        affinity_applied: bool,
    ) -> Option<(election::InferenceTarget, RoutingReservation)> {
        if candidates.is_empty() {
            return None;
        }
        let mut counts = self.counts.lock().unwrap();
        let count = |counts: &HashMap<ReservationKey, usize>,
                     target: &election::InferenceTarget| {
            counts
                .get(&ReservationKey {
                    model: model.to_string(),
                    target: target.clone(),
                })
                .copied()
                .unwrap_or(0)
        };
        let preferred_index = candidates
            .iter()
            .position(|candidate| candidate == preferred)
            .unwrap_or(0);
        let target = if affinity_applied {
            candidates[preferred_index].clone()
        } else {
            let minimum = candidates
                .iter()
                .map(|candidate| count(&counts, candidate))
                .min()
                .unwrap_or(0);
            (0..candidates.len())
                .map(|offset| (preferred_index + offset) % candidates.len())
                .find(|index| count(&counts, &candidates[*index]) == minimum)
                .map(|index| candidates[index].clone())
                .unwrap_or_else(|| candidates[preferred_index].clone())
        };
        let key = ReservationKey {
            model: model.to_string(),
            target: target.clone(),
        };
        *counts.entry(key.clone()).or_insert(0) += 1;
        drop(counts);
        Some((
            target,
            RoutingReservation {
                counts: Arc::clone(&self.counts),
                key,
            },
        ))
    }

    /// Total in-flight reservations across all models and targets.
    pub(crate) fn active_total(&self) -> usize {
        self.counts.lock().unwrap().values().sum()
    }

    #[cfg(test)]
    fn active_count(&self, model: &str, target: &election::InferenceTarget) -> usize {
        self.counts
            .lock()
            .unwrap()
            .get(&ReservationKey {
                model: model.to_string(),
                target: target.clone(),
            })
            .copied()
            .unwrap_or(0)
    }
}

/// RAII guard for one in-flight request. Dropping it — on success, failure,
/// cancellation, or panic — releases the reservation.
pub(crate) struct RoutingReservation {
    counts: ReservationCounts,
    key: ReservationKey,
}

impl RoutingReservation {
    /// Move the reservation to the failover target so retries keep the
    /// in-flight count on the target actually serving the request.
    pub(crate) fn transfer_to(&mut self, target: &election::InferenceTarget) {
        if self.key.target == *target {
            return;
        }
        let mut counts = self.counts.lock().unwrap();
        decrement(&mut counts, &self.key);
        self.key.target = target.clone();
        *counts.entry(self.key.clone()).or_insert(0) += 1;
    }
}

impl Drop for RoutingReservation {
    fn drop(&mut self) {
        decrement(&mut self.counts.lock().unwrap(), &self.key);
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
        assert_eq!(reservations.active_total(), 0);
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

        assert_eq!(reservations.active_total(), 0);
    }
}
