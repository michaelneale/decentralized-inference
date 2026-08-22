use std::collections::VecDeque;
use std::sync::Mutex;

use skippy_protocol::binary::StageWireMessage;

/// Bounds so a misbehaving upstream cannot grow the registry without limit.
const MAX_TRACKED_REQUESTS: usize = 32;
const MAX_RANGES_PER_REQUEST: usize = 8;

#[derive(Debug)]
struct RequestDiscards {
    request_id: u64,
    session_id: u64,
    ranges: VecDeque<(i32, i32)>,
}

/// Discarded verify-window id ranges, recorded by the connection's reader
/// thread the moment a `DiscardStaleWindows` message is read and consulted by
/// the executor before running each buffered verify window. This is what lets
/// a divergence cancel the stale run-ahead tail instead of executing it.
#[derive(Debug, Default)]
pub(super) struct StaleDiscardRegistry {
    requests: Mutex<VecDeque<RequestDiscards>>,
}

impl StaleDiscardRegistry {
    /// Records the range carried by a `DiscardStaleWindows` message
    /// (`tokens = [min_window_id, max_window_id]`). Malformed messages are
    /// ignored: a discard is an optimization, never a correctness dependency.
    pub(super) fn record_message(&self, message: &StageWireMessage) {
        let (Some(&min_id), Some(&max_id)) = (message.tokens.first(), message.tokens.get(1)) else {
            return;
        };
        if min_id > max_id {
            return;
        }
        self.record(message.request_id, message.session_id, min_id, max_id);
    }

    pub(super) fn record(&self, request_id: u64, session_id: u64, min_id: i32, max_id: i32) {
        let mut requests = self.requests.lock().expect("stale discard lock poisoned");
        if let Some(entry) = requests
            .iter_mut()
            .find(|entry| entry.request_id == request_id && entry.session_id == session_id)
        {
            if entry.ranges.len() >= MAX_RANGES_PER_REQUEST {
                entry.ranges.pop_front();
            }
            entry.ranges.push_back((min_id, max_id));
            return;
        }
        if requests.len() >= MAX_TRACKED_REQUESTS {
            requests.pop_front();
        }
        let mut ranges = VecDeque::new();
        ranges.push_back((min_id, max_id));
        requests.push_back(RequestDiscards {
            request_id,
            session_id,
            ranges,
        });
    }

    pub(super) fn is_discarded(&self, request_id: u64, session_id: u64, window_id: i32) -> bool {
        let requests = self.requests.lock().expect("stale discard lock poisoned");
        requests
            .iter()
            .filter(|entry| entry.request_id == request_id && entry.session_id == session_id)
            .any(|entry| {
                entry
                    .ranges
                    .iter()
                    .any(|&(min_id, max_id)| (min_id..=max_id).contains(&window_id))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_matches_ranges_per_request() {
        let registry = StaleDiscardRegistry::default();
        registry.record(7, 9, 10, 14);

        assert!(registry.is_discarded(7, 9, 10));
        assert!(registry.is_discarded(7, 9, 14));
        assert!(!registry.is_discarded(7, 9, 15));
        assert!(!registry.is_discarded(7, 9, 9));
        // Other requests and sessions are unaffected.
        assert!(!registry.is_discarded(8, 9, 12));
        assert!(!registry.is_discarded(7, 10, 12));
    }

    #[test]
    fn range_count_is_bounded_per_request() {
        let registry = StaleDiscardRegistry::default();
        for index in 0..(MAX_RANGES_PER_REQUEST as i32 + 4) {
            registry.record(1, 1, index * 10, index * 10 + 1);
        }
        // The oldest ranges were evicted; the newest still match.
        assert!(!registry.is_discarded(1, 1, 0));
        let newest = (MAX_RANGES_PER_REQUEST as i32 + 3) * 10;
        assert!(registry.is_discarded(1, 1, newest));
    }

    #[test]
    fn tracked_request_count_is_bounded() {
        let registry = StaleDiscardRegistry::default();
        for request in 0..(MAX_TRACKED_REQUESTS as u64 + 4) {
            registry.record(request, 1, 0, 10);
        }
        assert!(!registry.is_discarded(0, 1, 5));
        assert!(registry.is_discarded(MAX_TRACKED_REQUESTS as u64 + 3, 1, 5));
    }

    #[test]
    fn malformed_discard_messages_are_ignored() {
        use skippy_protocol::binary::{StageStateHeader, WireActivationDType, WireMessageKind};
        let registry = StaleDiscardRegistry::default();
        let mut message = StageWireMessage {
            kind: WireMessageKind::DiscardStaleWindows,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(
                WireMessageKind::DiscardStaleWindows,
                WireActivationDType::F32,
            ),
            request_id: 1,
            session_id: 1,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        registry.record_message(&message);
        assert!(!registry.is_discarded(1, 1, 0));

        message.tokens = vec![9, 3];
        registry.record_message(&message);
        assert!(!registry.is_discarded(1, 1, 5));

        message.tokens = vec![3, 9];
        registry.record_message(&message);
        assert!(registry.is_discarded(1, 1, 5));
    }
}
