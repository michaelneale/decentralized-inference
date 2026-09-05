//! Per-client-IP reconnect rate limiting, from the frozen bounds table:
//! 10 connects per client key per 60 seconds; client key is peer IP.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use crate::runtime_events::config::{
    RECONNECT_KEY_CAP, RECONNECT_LIMIT_PER_WINDOW, RECONNECT_WINDOW,
};

static ATTEMPTS: LazyLock<Mutex<HashMap<Option<IpAddr>, Vec<Instant>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static LAST_CAP_WARNING: LazyLock<Mutex<Option<Instant>>> = LazyLock::new(|| Mutex::new(None));

fn warn_global_key_cap(now: Instant) {
    let mut last_warning = LAST_CAP_WARNING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if last_warning.is_some_and(|previous| now.duration_since(previous) < RECONNECT_WINDOW) {
        return;
    }
    *last_warning = Some(now);
    tracing::warn!(
        reconnect_key_cap = RECONNECT_KEY_CAP,
        "runtime event reconnect key capacity reached; rejecting a new distinct client key"
    );
}

/// Record a connect attempt for `key` (the caller's peer IP, or `None` when
/// unavailable) and report whether it is within the frozen rate limit.
/// `None` is bucketed together and rate-limited the same as any other key
/// rather than exempted, since an unavailable peer address is not proof of
/// a distinct caller.
pub(super) fn record_attempt(key: Option<IpAddr>) -> bool {
    record_attempt_at(key, Instant::now())
}

fn record_attempt_at(key: Option<IpAddr>, now: Instant) -> bool {
    let mut attempts = ATTEMPTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // Prune expired keys before considering a new one. Existing active keys
    // retain their full attempt history; when the distinct-key cap is full,
    // reject a new key instead of evicting an active key and resetting its
    // rate limit.
    attempts.retain(|_, history| {
        history.retain(|instant| now.duration_since(*instant) < RECONNECT_WINDOW);
        !history.is_empty()
    });
    if !attempts.contains_key(&key) && attempts.len() >= RECONNECT_KEY_CAP {
        warn_global_key_cap(now);
        return false;
    }
    let entry = attempts.entry(key).or_default();
    entry.retain(|instant| now.duration_since(*instant) < RECONNECT_WINDOW);
    if entry.len() >= RECONNECT_LIMIT_PER_WINDOW {
        return false;
    }
    entry.push(now);
    true
}

#[cfg(test)]
pub(super) fn clear() {
    ATTEMPTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    *LAST_CAP_WARNING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial(runtime_events_reconnect)]
    fn allows_up_to_the_frozen_limit_then_rejects() {
        clear();
        let key = Some(IpAddr::from([127, 0, 0, 1]));
        let start = Instant::now();
        for _ in 0..RECONNECT_LIMIT_PER_WINDOW {
            assert!(record_attempt_at(key, start));
        }
        assert!(!record_attempt_at(key, start));
        clear();
    }

    #[test]
    #[serial_test::serial(runtime_events_reconnect)]
    fn resets_after_the_window_elapses() {
        clear();
        let key = Some(IpAddr::from([127, 0, 0, 2]));
        let start = Instant::now();
        for _ in 0..RECONNECT_LIMIT_PER_WINDOW {
            assert!(record_attempt_at(key, start));
        }
        assert!(record_attempt_at(key, start + RECONNECT_WINDOW));
        clear();
    }

    #[test]
    #[serial_test::serial(runtime_events_reconnect)]
    fn distinct_keys_have_independent_budgets() {
        clear();
        let a = Some(IpAddr::from([127, 0, 0, 3]));
        let b = Some(IpAddr::from([127, 0, 0, 4]));
        let start = Instant::now();
        for _ in 0..RECONNECT_LIMIT_PER_WINDOW {
            assert!(record_attempt_at(a, start));
        }
        assert!(!record_attempt_at(a, start));
        assert!(record_attempt_at(b, start));
        clear();
    }

    #[test]
    #[serial_test::serial(runtime_events_reconnect)]
    fn distinct_key_cap_rejects_new_keys_without_evicting_active_history() {
        clear();
        let start = Instant::now();
        for index in 0..RECONNECT_KEY_CAP {
            let key = Some(IpAddr::V6(std::net::Ipv6Addr::new(
                0x2001,
                0xdb8,
                0,
                0,
                0,
                0,
                0,
                index as u16,
            )));
            assert!(
                record_attempt_at(key, start),
                "key {index} must fit the cap"
            );
        }

        let new_key = Some(IpAddr::V6(std::net::Ipv6Addr::new(
            0x2001, 0xdb8, 0, 0, 0, 0, 1, 0,
        )));
        assert!(!record_attempt_at(new_key, start));

        let active_key = Some(IpAddr::V6(std::net::Ipv6Addr::new(
            0x2001, 0xdb8, 0, 0, 0, 0, 0, 0,
        )));
        for _ in 1..RECONNECT_LIMIT_PER_WINDOW {
            assert!(record_attempt_at(active_key, start));
        }
        assert!(!record_attempt_at(active_key, start));
        clear();
    }

    #[test]
    #[serial_test::serial(runtime_events_reconnect)]
    fn expired_keys_free_distinct_key_capacity() {
        clear();
        let start = Instant::now();
        for index in 0..RECONNECT_KEY_CAP {
            let key = Some(IpAddr::V6(std::net::Ipv6Addr::new(
                0x2001,
                0xdb8,
                0,
                0,
                0,
                0,
                0,
                index as u16,
            )));
            assert!(record_attempt_at(key, start));
        }
        let replacement = Some(IpAddr::V6(std::net::Ipv6Addr::new(
            0x2001, 0xdb8, 0, 0, 0, 0, 2, 0,
        )));
        assert!(record_attempt_at(replacement, start + RECONNECT_WINDOW));
        clear();
    }

    #[test]
    #[serial_test::serial(runtime_events_reconnect)]
    fn distinct_key_cap_warning_is_rate_limited_without_client_identity() {
        clear();
        let start = Instant::now();
        warn_global_key_cap(start);
        assert!(
            LAST_CAP_WARNING
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_some()
        );
        let warning_at = LAST_CAP_WARNING
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .expect("warning timestamp");
        warn_global_key_cap(start + RECONNECT_WINDOW - std::time::Duration::from_nanos(1));
        assert_eq!(
            LAST_CAP_WARNING
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .expect("warning timestamp"),
            warning_at
        );
        warn_global_key_cap(start + RECONNECT_WINDOW);
        assert_eq!(
            LAST_CAP_WARNING
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .expect("warning timestamp"),
            start + RECONNECT_WINDOW
        );
        clear();
    }
}
