use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use openai_frontend::{OpenAiError, OpenAiResult};
use skippy_metrics::attr as attr_key;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifyWindowPipelineConfig {
    depth: usize,
    runahead_max_tokens: usize,
}

impl VerifyWindowPipelineConfig {
    pub(super) fn new(depth: usize) -> Self {
        Self {
            depth: depth.max(1),
            runahead_max_tokens: 0,
        }
    }

    /// Run-ahead mode: admission is bounded by a speculative-token budget
    /// instead of a fixed window count. The window count stays capped at the
    /// native checkpoint-retention bound so downstream recovery state cannot
    /// outgrow what the runtime can restore.
    pub(super) fn with_runahead(max_tokens: usize) -> Self {
        Self {
            depth: skippy_protocol::MAX_VERIFY_WINDOW_PIPELINE_DEPTH,
            runahead_max_tokens: max_tokens.max(1),
        }
    }

    pub(super) fn depth(self) -> usize {
        self.depth
    }

    pub(super) fn runahead_max_tokens(self) -> usize {
        self.runahead_max_tokens
    }

    pub(super) fn is_runahead(self) -> bool {
        self.runahead_max_tokens > 0
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct VerifyWindowPipelineStats {
    depth: usize,
    runahead_max_tokens: usize,
    max_in_flight_tokens: usize,
    direct_prediction_return: bool,
    direct_prediction_return_upstream_opened: bool,
    direct_prediction_return_reverse_fallback: bool,
    opened_windows: usize,
    max_in_flight: usize,
    recovery_epochs: usize,
    stale_marked: usize,
    stale_discarded: usize,
    stale_drain_ms: f64,
    stale_stage0_compute_ms: f64,
    stale_forward_write_ms: f64,
    stale_downstream_wait_ms: f64,
    stale_verify_elapsed_ms: f64,
    horizon_refill_attempts: usize,
    horizon_refill_successes: usize,
    horizon_refill_tokens: usize,
    horizon_refill_misses: usize,
    occupancy_ms_by_depth: Vec<f64>,
    occupancy_total_ms: f64,
    occupancy_parallel_ms: f64,
    occupancy_full_ms: f64,
    occupancy_average_in_flight: f64,
}

impl VerifyWindowPipelineStats {
    pub(super) fn insert_response_timings(
        &self,
        timings: &mut BTreeMap<String, serde_json::Value>,
    ) {
        timings.insert(
            "verify_window_depth".to_string(),
            serde_json::json!(self.depth),
        );
        timings.insert(
            "verify_window_direct_prediction_return".to_string(),
            serde_json::json!(self.direct_prediction_return),
        );
        timings.insert(
            "verify_window_direct_prediction_return_upstream_opened".to_string(),
            serde_json::json!(self.direct_prediction_return_upstream_opened),
        );
        timings.insert(
            "verify_window_direct_prediction_return_reverse_fallback".to_string(),
            serde_json::json!(self.direct_prediction_return_reverse_fallback),
        );
        timings.insert(
            "verify_window_opened".to_string(),
            serde_json::json!(self.opened_windows),
        );
        timings.insert(
            "verify_window_max_in_flight".to_string(),
            serde_json::json!(self.max_in_flight),
        );
        timings.insert(
            "verify_window_recovery_epochs".to_string(),
            serde_json::json!(self.recovery_epochs),
        );
        timings.insert(
            "verify_window_stale_marked".to_string(),
            serde_json::json!(self.stale_marked),
        );
        timings.insert(
            "verify_window_stale_discarded".to_string(),
            serde_json::json!(self.stale_discarded),
        );
        timings.insert(
            "verify_window_stale_drain_ms".to_string(),
            serde_json::json!(self.stale_drain_ms),
        );
        timings.insert(
            "verify_window_stale_stage0_compute_ms".to_string(),
            serde_json::json!(self.stale_stage0_compute_ms),
        );
        timings.insert(
            "verify_window_stale_forward_write_ms".to_string(),
            serde_json::json!(self.stale_forward_write_ms),
        );
        timings.insert(
            "verify_window_stale_downstream_wait_ms".to_string(),
            serde_json::json!(self.stale_downstream_wait_ms),
        );
        timings.insert(
            "verify_window_stale_verify_elapsed_ms".to_string(),
            serde_json::json!(self.stale_verify_elapsed_ms),
        );
        timings.insert(
            "verify_window_horizon_refill_attempts".to_string(),
            serde_json::json!(self.horizon_refill_attempts),
        );
        timings.insert(
            "verify_window_horizon_refill_successes".to_string(),
            serde_json::json!(self.horizon_refill_successes),
        );
        timings.insert(
            "verify_window_horizon_refill_tokens".to_string(),
            serde_json::json!(self.horizon_refill_tokens),
        );
        timings.insert(
            "verify_window_horizon_refill_misses".to_string(),
            serde_json::json!(self.horizon_refill_misses),
        );
        timings.insert(
            "verify_window_occupancy_ms_by_depth".to_string(),
            serde_json::json!(self.occupancy_ms_by_depth),
        );
        timings.insert(
            "verify_window_occupancy_total_ms".to_string(),
            serde_json::json!(self.occupancy_total_ms),
        );
        timings.insert(
            "verify_window_occupancy_parallel_ms".to_string(),
            serde_json::json!(self.occupancy_parallel_ms),
        );
        timings.insert(
            "verify_window_occupancy_parallel_fraction".to_string(),
            serde_json::json!(fraction(
                self.occupancy_parallel_ms,
                self.occupancy_total_ms
            )),
        );
        timings.insert(
            "verify_window_occupancy_full_ms".to_string(),
            serde_json::json!(self.occupancy_full_ms),
        );
        timings.insert(
            "verify_window_occupancy_full_fraction".to_string(),
            serde_json::json!(fraction(self.occupancy_full_ms, self.occupancy_total_ms)),
        );
        timings.insert(
            "verify_window_occupancy_average_in_flight".to_string(),
            serde_json::json!(self.occupancy_average_in_flight),
        );
        timings.insert(
            "verify_window_runahead_max_tokens".to_string(),
            serde_json::json!(self.runahead_max_tokens),
        );
        timings.insert(
            "verify_window_max_in_flight_tokens".to_string(),
            serde_json::json!(self.max_in_flight_tokens),
        );
    }
}

fn fraction(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct VerifyWindow {
    pub(super) id: i32,
    pub(super) base_position: usize,
    pub(super) decode_step: usize,
    pub(super) token_count: usize,
}

#[derive(Debug)]
pub(super) struct VerifyWindowScheduler {
    config: VerifyWindowPipelineConfig,
    next_id: i32,
    in_flight: VecDeque<VerifyWindow>,
    in_flight_tokens: usize,
    stats: VerifyWindowPipelineStats,
    occupancy_ms_by_depth: Vec<f64>,
    occupancy_changed: Instant,
}

impl VerifyWindowScheduler {
    pub(super) fn new(config: VerifyWindowPipelineConfig) -> Self {
        Self {
            config,
            next_id: 1,
            in_flight: VecDeque::new(),
            in_flight_tokens: 0,
            stats: VerifyWindowPipelineStats {
                depth: config.depth(),
                runahead_max_tokens: config.runahead_max_tokens(),
                ..VerifyWindowPipelineStats::default()
            },
            occupancy_ms_by_depth: vec![0.0; config.depth().saturating_add(1)],
            occupancy_changed: Instant::now(),
        }
    }

    pub(super) fn has_capacity(&self) -> bool {
        if self.in_flight.len() >= self.config.depth() {
            return false;
        }
        if self.config.is_runahead() {
            return self.in_flight_tokens < self.config.runahead_max_tokens();
        }
        true
    }

    pub(super) fn depth(&self) -> usize {
        self.config.depth()
    }

    pub(super) fn is_runahead(&self) -> bool {
        self.config.is_runahead()
    }

    pub(super) fn mark_direct_prediction_return(&mut self, upstream_opened: bool) {
        self.stats.direct_prediction_return = true;
        self.stats.direct_prediction_return_upstream_opened = upstream_opened;
        self.stats.direct_prediction_return_reverse_fallback = !upstream_opened;
    }

    pub(super) fn supports_pipelining(&self, width: usize) -> bool {
        self.config.depth() > 1 && width > 0
    }

    pub(super) fn record_horizon_refill(&mut self, appended_tokens: usize) {
        self.stats.horizon_refill_attempts = self.stats.horizon_refill_attempts.saturating_add(1);
        if appended_tokens == 0 {
            self.stats.horizon_refill_misses = self.stats.horizon_refill_misses.saturating_add(1);
            return;
        }
        self.stats.horizon_refill_successes = self.stats.horizon_refill_successes.saturating_add(1);
        self.stats.horizon_refill_tokens = self
            .stats
            .horizon_refill_tokens
            .saturating_add(appended_tokens);
    }

    pub(super) fn insert_pipeline_telemetry_attrs(
        &self,
        attrs: &mut BTreeMap<String, serde_json::Value>,
    ) {
        attrs.insert(
            attr_key::VERIFY_WINDOW_DIRECT_RETURN_UPSTREAM_OPENED.to_string(),
            serde_json::json!(self.stats.direct_prediction_return_upstream_opened),
        );
        attrs.insert(
            attr_key::VERIFY_WINDOW_DIRECT_RETURN_REVERSE_FALLBACK.to_string(),
            serde_json::json!(self.stats.direct_prediction_return_reverse_fallback),
        );
        attrs.insert(
            "llama_stage.verify_window.horizon_refill_attempts".to_string(),
            serde_json::json!(self.stats.horizon_refill_attempts),
        );
        attrs.insert(
            "llama_stage.verify_window.horizon_refill_successes".to_string(),
            serde_json::json!(self.stats.horizon_refill_successes),
        );
        attrs.insert(
            "llama_stage.verify_window.horizon_refill_tokens".to_string(),
            serde_json::json!(self.stats.horizon_refill_tokens),
        );
        attrs.insert(
            "llama_stage.verify_window.horizon_refill_misses".to_string(),
            serde_json::json!(self.stats.horizon_refill_misses),
        );
    }

    pub(super) fn open(
        &mut self,
        base_position: usize,
        decode_step: usize,
        token_count: usize,
    ) -> OpenAiResult<VerifyWindow> {
        if !self.has_capacity() {
            return Err(OpenAiError::backend(
                "verify window pipeline depth exceeded",
            ));
        }
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| OpenAiError::backend("verify window id overflow"))?;
        let window = VerifyWindow {
            id,
            base_position,
            decode_step,
            token_count,
        };
        self.record_occupancy();
        self.in_flight.push_back(window.clone());
        self.in_flight_tokens = self.in_flight_tokens.saturating_add(token_count);
        self.stats.opened_windows = self.stats.opened_windows.saturating_add(1);
        self.stats.max_in_flight = self.stats.max_in_flight.max(self.in_flight.len());
        self.stats.max_in_flight_tokens =
            self.stats.max_in_flight_tokens.max(self.in_flight_tokens);
        Ok(window)
    }

    pub(super) fn complete_next(&mut self, reply_window_id: i32) -> OpenAiResult<VerifyWindow> {
        let Some(window) = self.in_flight.front() else {
            return Err(OpenAiError::backend(
                "verify window reply arrived with no in-flight window",
            ));
        };
        if window.id != reply_window_id {
            return Err(OpenAiError::backend(format!(
                "verify window reply out of order: got {reply_window_id}, expected {}",
                window.id
            )));
        }
        self.record_occupancy();
        let completed = self.in_flight.pop_front().expect("checked non-empty queue");
        self.in_flight_tokens = self.in_flight_tokens.saturating_sub(completed.token_count);
        Ok(completed)
    }

    #[cfg(test)]
    pub(super) fn discard_stale(&mut self) -> usize {
        let discarded = self.in_flight.len();
        self.record_occupancy();
        self.in_flight.clear();
        self.in_flight_tokens = 0;
        self.stats.stale_discarded = self.stats.stale_discarded.saturating_add(discarded);
        discarded
    }

    pub(super) fn record_stale_discarded(&mut self, count: usize, drain_ms: f64) {
        self.stats.stale_discarded = self.stats.stale_discarded.saturating_add(count);
        self.stats.stale_drain_ms += drain_ms;
    }

    pub(super) fn mark_recovery_epoch(&mut self, stale_count: usize) {
        self.stats.recovery_epochs = self.stats.recovery_epochs.saturating_add(1);
        self.mark_stale(stale_count);
    }

    pub(super) fn mark_stale(&mut self, stale_count: usize) {
        self.stats.stale_marked = self.stats.stale_marked.saturating_add(stale_count);
    }

    pub(super) fn record_stale_execution(
        &mut self,
        drain_ms: f64,
        stage0_compute_ms: f64,
        forward_write_ms: f64,
        downstream_wait_ms: f64,
        verify_elapsed_ms: f64,
    ) {
        self.record_stale_discarded(1, drain_ms);
        self.stats.stale_stage0_compute_ms += stage0_compute_ms.max(0.0);
        self.stats.stale_forward_write_ms += forward_write_ms.max(0.0);
        self.stats.stale_downstream_wait_ms += downstream_wait_ms.max(0.0);
        self.stats.stale_verify_elapsed_ms += verify_elapsed_ms.max(0.0);
    }

    pub(super) fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    pub(super) fn stale_discard_count(&self) -> usize {
        self.stats.stale_discarded
    }

    pub(super) fn stats(&self) -> VerifyWindowPipelineStats {
        let mut stats = self.stats.clone();
        let mut occupancy_ms_by_depth = self.occupancy_ms_by_depth.clone();
        let current_interval_ms = self.occupancy_changed.elapsed().as_secs_f64() * 1_000.0;
        if let Some(bucket) = occupancy_ms_by_depth.get_mut(self.in_flight.len()) {
            *bucket += current_interval_ms;
        }
        let occupancy_total_ms = occupancy_ms_by_depth.iter().sum::<f64>();
        let occupancy_parallel_ms = occupancy_ms_by_depth.iter().skip(2).sum::<f64>();
        let occupancy_full_ms = occupancy_ms_by_depth
            .get(self.config.depth())
            .copied()
            .unwrap_or_default();
        let weighted_ms = occupancy_ms_by_depth
            .iter()
            .enumerate()
            .map(|(depth, elapsed_ms)| depth as f64 * elapsed_ms)
            .sum::<f64>();
        stats.occupancy_ms_by_depth = occupancy_ms_by_depth;
        stats.occupancy_total_ms = occupancy_total_ms;
        stats.occupancy_parallel_ms = occupancy_parallel_ms;
        stats.occupancy_full_ms = occupancy_full_ms;
        stats.occupancy_average_in_flight = fraction(weighted_ms, occupancy_total_ms);
        stats
    }

    fn record_occupancy(&mut self) {
        let elapsed_ms = self.occupancy_changed.elapsed().as_secs_f64() * 1_000.0;
        if let Some(bucket) = self.occupancy_ms_by_depth.get_mut(self.in_flight.len()) {
            *bucket += elapsed_ms;
        }
        self.occupancy_changed = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_preferred_and_reverse_direct_return_paths() {
        let mut preferred = VerifyWindowScheduler::new(VerifyWindowPipelineConfig::new(2));
        preferred.mark_direct_prediction_return(true);
        assert!(preferred.stats().direct_prediction_return);
        assert!(preferred.stats().direct_prediction_return_upstream_opened);
        assert!(!preferred.stats().direct_prediction_return_reverse_fallback);

        let mut reverse = VerifyWindowScheduler::new(VerifyWindowPipelineConfig::new(2));
        reverse.mark_direct_prediction_return(false);
        let mut timings = BTreeMap::new();
        reverse.stats().insert_response_timings(&mut timings);
        assert_eq!(
            timings["verify_window_direct_prediction_return_reverse_fallback"],
            true
        );
        assert_eq!(
            timings["verify_window_direct_prediction_return_upstream_opened"],
            false
        );
    }

    #[test]
    fn bounds_depth_and_requires_fifo_reply_ids() {
        let config = VerifyWindowPipelineConfig::new(2);
        let mut scheduler = VerifyWindowScheduler::new(config);
        let first = scheduler.open(10, 0, 1).unwrap();
        let second = scheduler.open(11, 1, 1).unwrap();

        assert!(scheduler.open(12, 2, 1).is_err());
        assert!(scheduler.complete_next(second.id).is_err());
        assert_eq!(scheduler.in_flight_len(), 2);
        assert_eq!(scheduler.complete_next(first.id).unwrap(), first);
        assert_eq!(scheduler.complete_next(second.id).unwrap(), second);
        assert_eq!(first.id, 1);
        assert_eq!(scheduler.stats().depth, 2);
        assert_eq!(scheduler.stats().opened_windows, 2);
        assert_eq!(scheduler.stats().max_in_flight, 2);
        assert!(!scheduler.stats().direct_prediction_return);
    }

    #[test]
    fn depth_nine_keeps_the_first_window_restorable() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig::new(9));
        let windows: Vec<_> = (0..9)
            .map(|step| scheduler.open(10 + step, step, 1).unwrap())
            .collect();

        assert_eq!(scheduler.in_flight_len(), 9);
        assert!(scheduler.open(19, 9, 1).is_err());
        assert_eq!(scheduler.stats().max_in_flight, 9);

        // The first window must still be completable after the ninth opens.
        // Native checkpoint retention has to cover every in-flight window, so
        // a partially accepted first window stays restorable.
        for window in windows {
            assert_eq!(scheduler.complete_next(window.id).unwrap(), window);
        }
        assert_eq!(scheduler.in_flight_len(), 0);
    }

    #[test]
    fn discards_stale_windows_after_divergence() {
        let config = VerifyWindowPipelineConfig::new(3);
        let mut scheduler = VerifyWindowScheduler::new(config);
        scheduler.open(10, 0, 1).unwrap();
        scheduler.open(11, 1, 1).unwrap();
        scheduler.open(12, 2, 1).unwrap();

        assert_eq!(scheduler.discard_stale(), 3);
        assert_eq!(scheduler.stale_discard_count(), 3);
        assert_eq!(scheduler.in_flight_len(), 0);
        assert_eq!(scheduler.stats().stale_discarded, 3);
    }

    #[test]
    fn stale_recovery_tracks_marked_and_completed_work_separately() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig::new(3));
        let first = scheduler.open(10, 0, 1).unwrap();
        let second = scheduler.open(11, 1, 1).unwrap();
        let third = scheduler.open(12, 2, 1).unwrap();

        scheduler.complete_next(first.id).unwrap();
        scheduler.mark_recovery_epoch(2);
        scheduler.complete_next(second.id).unwrap();
        scheduler.record_stale_execution(3.0, 4.0, 5.0, 6.0, 15.0);
        scheduler.complete_next(third.id).unwrap();
        scheduler.record_stale_execution(7.0, 8.0, 9.0, 10.0, 27.0);

        let stats = scheduler.stats();
        assert_eq!(stats.recovery_epochs, 1);
        assert_eq!(stats.stale_marked, 2);
        assert_eq!(stats.stale_discarded, 2);
        assert_eq!(stats.stale_drain_ms, 10.0);
        assert_eq!(stats.stale_stage0_compute_ms, 12.0);
        assert_eq!(stats.stale_forward_write_ms, 14.0);
        assert_eq!(stats.stale_downstream_wait_ms, 16.0);
        assert_eq!(stats.stale_verify_elapsed_ms, 42.0);
    }

    #[test]
    fn configured_depth_is_the_fill_target() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig::new(8));

        assert!(scheduler.supports_pipelining(4));
        assert_eq!(scheduler.depth(), 8);

        for position in 0..8 {
            scheduler.open(100 + position, position, 1).unwrap();
        }
        assert!(!scheduler.has_capacity());
        assert!(scheduler.open(108, 8, 1).is_err());
    }

    #[test]
    fn pipeline_depth_one_never_admits_dependent_work() {
        let scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig::new(1));

        assert!(!scheduler.supports_pipelining(2));
    }

    #[test]
    fn fixed_fill_counters_are_exposed_in_response_timings() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig::new(2));
        assert!(scheduler.supports_pipelining(2));
        scheduler.record_horizon_refill(7);
        scheduler.record_horizon_refill(0);
        let mut timings = BTreeMap::new();
        scheduler.stats().insert_response_timings(&mut timings);

        assert_eq!(timings["verify_window_horizon_refill_attempts"], 2);
        assert_eq!(timings["verify_window_horizon_refill_successes"], 1);
        assert_eq!(timings["verify_window_horizon_refill_tokens"], 7);
        assert_eq!(timings["verify_window_horizon_refill_misses"], 1);
        let mut attrs = BTreeMap::new();
        scheduler.insert_pipeline_telemetry_attrs(&mut attrs);
        assert_eq!(
            attrs["llama_stage.verify_window.horizon_refill_attempts"],
            2
        );
        assert_eq!(
            attrs["llama_stage.verify_window.horizon_refill_successes"],
            1
        );
        assert_eq!(attrs["llama_stage.verify_window.horizon_refill_tokens"], 7);
        assert_eq!(attrs["llama_stage.verify_window.horizon_refill_misses"], 1);
    }

    #[test]
    fn occupancy_timings_measure_parallel_and_full_depth_time() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig::new(2));
        scheduler.occupancy_changed = Instant::now() - std::time::Duration::from_millis(2);
        let first = scheduler.open(10, 0, 1).unwrap();
        scheduler.occupancy_changed = Instant::now() - std::time::Duration::from_millis(3);
        let second = scheduler.open(11, 1, 1).unwrap();
        scheduler.occupancy_changed = Instant::now() - std::time::Duration::from_millis(4);

        let stats = scheduler.stats();
        assert_eq!(stats.occupancy_ms_by_depth.len(), 3);
        assert!(stats.occupancy_ms_by_depth[0] >= 1.5);
        assert!(stats.occupancy_ms_by_depth[1] >= 2.5);
        assert!(stats.occupancy_ms_by_depth[2] >= 3.5);
        assert!(stats.occupancy_parallel_ms >= 3.5);
        assert!(stats.occupancy_full_ms >= 3.5);
        assert!(stats.occupancy_average_in_flight > 1.0);

        assert_eq!(scheduler.complete_next(first.id).unwrap(), first);
        assert_eq!(scheduler.complete_next(second.id).unwrap(), second);
    }

    #[test]
    fn runahead_budget_bounds_admission_by_tokens() {
        let mut scheduler =
            VerifyWindowScheduler::new(VerifyWindowPipelineConfig::with_runahead(100));
        assert!(scheduler.supports_pipelining(4));
        assert_eq!(
            scheduler.depth(),
            skippy_protocol::MAX_VERIFY_WINDOW_PIPELINE_DEPTH
        );

        let first = scheduler.open(10, 0, 48).unwrap();
        assert!(scheduler.has_capacity());
        let _second = scheduler.open(58, 1, 48).unwrap();
        // 96 tokens in flight, budget 100: one more window may still open.
        assert!(scheduler.has_capacity());
        let _third = scheduler.open(106, 2, 48).unwrap();
        assert!(!scheduler.has_capacity());
        assert!(scheduler.open(154, 3, 1).is_err());

        // Completing the head window frees its share of the budget.
        assert_eq!(scheduler.complete_next(first.id).unwrap(), first);
        assert!(scheduler.has_capacity());
        assert_eq!(scheduler.stats().max_in_flight_tokens, 144);
        assert_eq!(scheduler.stats().runahead_max_tokens, 100);
    }

    #[test]
    fn runahead_window_count_stays_within_native_retention() {
        let mut scheduler =
            VerifyWindowScheduler::new(VerifyWindowPipelineConfig::with_runahead(1_000_000));
        for step in 0..skippy_protocol::MAX_VERIFY_WINDOW_PIPELINE_DEPTH {
            scheduler.open(10 + step, step, 1).unwrap();
        }
        // The token budget is nowhere near spent, but the native checkpoint
        // retention bound still caps the number of in-flight windows.
        assert!(!scheduler.has_capacity());
        assert!(
            scheduler
                .open(
                    10 + skippy_protocol::MAX_VERIFY_WINDOW_PIPELINE_DEPTH,
                    64,
                    1
                )
                .is_err()
        );
    }
}
