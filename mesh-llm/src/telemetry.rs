use crate::{benchmark, tunnel};
use anyhow::Result;
use rusqlite::{params, Connection};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct Telemetry {
    inner: Arc<Inner>,
}

struct Inner {
    node_id: String,
    db_path: PathBuf,
    state: Mutex<State>,
}

struct State {
    current_minute: i64,
    requests: u64,
    errors: u64,
    requests_local: u64,
    requests_remote: u64,
    request_time_ms_total: u64,
    active_requests: u32,
    active_requests_peak: u32,
    latency_ms_samples: Vec<u32>,
    tunnel_bytes_minute_start: u64,
}

#[derive(Clone, Copy)]
pub enum RouteKind {
    Local,
    Remote,
}

pub struct RequestSpan {
    telemetry: Telemetry,
    route: RouteKind,
    started: Instant,
    finished: bool,
}

#[derive(Serialize, Clone)]
pub struct LiveSnapshot {
    pub ts_minute: i64,
    pub node_id: String,
    pub active_requests: u32,
    pub active_requests_peak: u32,
    pub requests: u64,
    pub errors: u64,
    pub requests_local: u64,
    pub requests_remote: u64,
    pub request_time_ms_total: u64,
    pub utilization_pct: f64,
    pub tunnel_bytes_total: u64,
}

#[derive(Serialize, Clone)]
pub struct NodeMetricRow {
    pub ts_minute: i64,
    pub source_node_id: String,
    pub requests: u64,
    pub errors: u64,
    pub requests_local: u64,
    pub requests_remote: u64,
    pub request_time_ms_total: u64,
    pub utilization_pct: f64,
    pub active_requests_peak: u32,
    pub latency_p50_ms: Option<f64>,
    pub latency_p95_ms: Option<f64>,
    pub latency_p99_ms: Option<f64>,
    pub tunnel_bytes_total: u64,
    pub observed_at: i64,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct NodeMetricSummary {
    pub ts_minute: i64,
    pub source_node_id: String,
    pub requests: u64,
    pub errors: u64,
    pub requests_local: u64,
    pub requests_remote: u64,
    pub request_time_ms_total: u64,
    pub utilization_pct: f64,
    pub active_requests_peak: u32,
    pub latency_p50_ms: Option<f64>,
    pub latency_p95_ms: Option<f64>,
    pub latency_p99_ms: Option<f64>,
    pub tunnel_bytes_total: u64,
    pub observed_at: i64,
}

#[derive(Serialize, Clone)]
pub struct RollupMetricRow {
    pub ts_minute: i64,
    pub node_count: u64,
    pub requests: u64,
    pub errors: u64,
    pub requests_local: u64,
    pub requests_remote: u64,
    pub request_time_ms_total: u64,
    pub utilization_pct_avg_nodes: f64,
    pub active_requests_peak_max: u32,
    pub tunnel_bytes_total: u64,
    pub latency_p95_ms_avg_nodes: Option<f64>,
    pub latency_p95_ms_max_nodes: Option<f64>,
}

#[derive(Serialize, Clone)]
pub struct BenchmarkRunRow {
    pub ts: i64,
    pub source_node_id: String,
    pub model: String,
    pub probe_name: String,
    pub probe_type: String,
    pub stream: bool,
    pub success: bool,
    pub status_code: Option<u16>,
    pub latency_ms: Option<u32>,
    pub ttft_ms: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub tokens_per_sec: Option<f64>,
    pub error_kind: Option<String>,
    pub observed_at: i64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BenchmarkRunInput {
    pub ts: i64,
    pub mesh_id: Option<String>,
    pub target_node_id: Option<String>,
    pub model: String,
    pub probe_name: String,
    pub probe_type: String,
    pub stream: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: u32,
    pub prompt_hash: Option<String>,
    pub route_kind: Option<String>,
    pub success: bool,
    pub status_code: Option<u16>,
    pub latency_ms: Option<u32>,
    pub ttft_ms: Option<u32>,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub tokens_per_sec: Option<f64>,
    pub error_kind: Option<String>,
    pub error_message: Option<String>,
    pub response_shape_ok: Option<bool>,
}

struct FlushRow {
    ts_minute: i64,
    source_node_id: String,
    requests: u64,
    errors: u64,
    requests_local: u64,
    requests_remote: u64,
    request_time_ms_total: u64,
    utilization_pct: f64,
    active_requests_peak: u32,
    latency_p50_ms: Option<f64>,
    latency_p95_ms: Option<f64>,
    latency_p99_ms: Option<f64>,
    tunnel_bytes_total: u64,
    observed_at: i64,
}

impl Telemetry {
    pub async fn new(node_id: String) -> Result<Self> {
        let db_path = telemetry_db_path();
        let db_path_for_init = db_path.clone();
        tokio::task::spawn_blocking(move || init_db(&db_path_for_init)).await??;

        let now_minute = unix_minute_now();
        let tunnel_bytes = tunnel::bytes_transferred();
        let telemetry = Self {
            inner: Arc::new(Inner {
                node_id,
                db_path,
                state: Mutex::new(State {
                    current_minute: now_minute,
                    requests: 0,
                    errors: 0,
                    requests_local: 0,
                    requests_remote: 0,
                    request_time_ms_total: 0,
                    active_requests: 0,
                    active_requests_peak: 0,
                    latency_ms_samples: Vec::new(),
                    tunnel_bytes_minute_start: tunnel_bytes,
                }),
            }),
        };
        telemetry.start_flush_loop();
        Ok(telemetry)
    }

    pub fn start_request(&self, route: RouteKind) -> RequestSpan {
        let mut state = self.inner.state.lock().expect("telemetry state mutex poisoned");
        state.active_requests = state.active_requests.saturating_add(1);
        state.active_requests_peak = state.active_requests_peak.max(state.active_requests);
        RequestSpan {
            telemetry: self.clone(),
            route,
            started: Instant::now(),
            finished: false,
        }
    }

    fn finish_request(&self, route: RouteKind, latency: Duration, success: bool) {
        let mut state = self.inner.state.lock().expect("telemetry state mutex poisoned");
        state.requests = state.requests.saturating_add(1);
        if !success {
            state.errors = state.errors.saturating_add(1);
        }
        match route {
            RouteKind::Local => state.requests_local = state.requests_local.saturating_add(1),
            RouteKind::Remote => state.requests_remote = state.requests_remote.saturating_add(1),
        }
        let ms = latency.as_millis().min(u128::from(u32::MAX)) as u32;
        state.request_time_ms_total = state.request_time_ms_total.saturating_add(ms as u64);
        state.latency_ms_samples.push(ms);
        state.active_requests = state.active_requests.saturating_sub(1);
    }

    pub fn snapshot(&self) -> LiveSnapshot {
        let tunnel_now = tunnel::bytes_transferred();
        let state = self.inner.state.lock().expect("telemetry state mutex poisoned");
        LiveSnapshot {
            ts_minute: state.current_minute,
            node_id: self.inner.node_id.clone(),
            active_requests: state.active_requests,
            active_requests_peak: state.active_requests_peak,
            requests: state.requests,
            errors: state.errors,
            requests_local: state.requests_local,
            requests_remote: state.requests_remote,
            request_time_ms_total: state.request_time_ms_total,
            utilization_pct: state.request_time_ms_total as f64 / 60_000.0 * 100.0,
            tunnel_bytes_total: tunnel_now.saturating_sub(state.tunnel_bytes_minute_start),
        }
    }

    pub fn current_minute_summary(&self) -> NodeMetricSummary {
        let tunnel_now = tunnel::bytes_transferred();
        let state = self.inner.state.lock().expect("telemetry state mutex poisoned");
        let row = flush_row_from_state(&self.inner.node_id, &state, tunnel_now);
        NodeMetricSummary::from(row)
    }

    pub async fn node_history(&self, minutes: u32) -> Result<Vec<NodeMetricRow>> {
        self.node_history_for(self.inner.node_id.clone(), minutes).await
    }

    pub async fn node_history_for(&self, node_id: String, minutes: u32) -> Result<Vec<NodeMetricRow>> {
        let db_path = self.inner.db_path.clone();
        tokio::task::spawn_blocking(move || query_node_history(&db_path, &node_id, minutes))
            .await?
    }

    pub async fn all_nodes_latest(&self, minutes: u32) -> Result<Vec<NodeMetricRow>> {
        let db_path = self.inner.db_path.clone();
        tokio::task::spawn_blocking(move || query_all_nodes_latest(&db_path, minutes)).await?
    }

    pub async fn rollup_history(&self, minutes: u32) -> Result<Vec<RollupMetricRow>> {
        let db_path = self.inner.db_path.clone();
        tokio::task::spawn_blocking(move || query_rollup_history(&db_path, minutes)).await?
    }

    pub async fn benchmark_history(&self, minutes: u32, limit: u32) -> Result<Vec<BenchmarkRunRow>> {
        let db_path = self.inner.db_path.clone();
        tokio::task::spawn_blocking(move || query_benchmark_history(&db_path, minutes, limit)).await?
    }

    pub async fn latest_summaries_for_gossip(&self) -> Result<Vec<NodeMetricSummary>> {
        let db_path = self.inner.db_path.clone();
        let current = self.current_minute_summary();
        let current_minute = current.ts_minute;
        let node_id = self.inner.node_id.clone();
        let mut rows = tokio::task::spawn_blocking(move || {
            query_node_history_exact_minutes(&db_path, &node_id, &[current_minute - 1])
        })
        .await??;
        rows.push(current);
        rows.sort_by_key(|r| r.ts_minute);
        rows.dedup_by_key(|r| r.ts_minute);
        Ok(rows)
    }

    pub async fn upsert_peer_summaries(&self, summaries: &[NodeMetricSummary]) -> Result<()> {
        if summaries.is_empty() {
            return Ok(());
        }
        let db_path = self.inner.db_path.clone();
        let rows = summaries.to_vec();
        tokio::task::spawn_blocking(move || {
            for summary in &rows {
                insert_node_metric_summary(&db_path, summary)?;
            }
            Ok::<(), anyhow::Error>(())
        })
        .await??;
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn insert_benchmark_run(&self, run: BenchmarkRunInput) -> Result<()> {
        let db_path = self.inner.db_path.clone();
        let source_node_id = self.inner.node_id.clone();
        tokio::task::spawn_blocking(move || insert_benchmark_run(&db_path, &source_node_id, &run))
            .await??;
        Ok(())
    }

    fn start_flush_loop(&self) {
        let telemetry = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                if let Err(e) = telemetry.maybe_flush().await {
                    tracing::debug!("Telemetry flush failed: {e}");
                }
            }
        });
    }

    async fn maybe_flush(&self) -> Result<()> {
        let now_minute = unix_minute_now();
        let tunnel_now = tunnel::bytes_transferred();
        let row = {
            let mut state = self.inner.state.lock().expect("telemetry state mutex poisoned");
            if now_minute <= state.current_minute {
                return Ok(());
            }
            let row = flush_row_from_state(&self.inner.node_id, &state, tunnel_now);
            state.current_minute = now_minute;
            state.requests = 0;
            state.errors = 0;
            state.requests_local = 0;
            state.requests_remote = 0;
            state.request_time_ms_total = 0;
            state.active_requests_peak = state.active_requests;
            state.latency_ms_samples.clear();
            state.tunnel_bytes_minute_start = tunnel_now;
            row
        };

        let db_path = self.inner.db_path.clone();
        tokio::task::spawn_blocking(move || insert_node_metric(&db_path, &row)).await??;
        Ok(())
    }
}

impl RequestSpan {
    pub fn finish(mut self, success: bool) {
        if !self.finished {
            self.telemetry
                .finish_request(self.route, self.started.elapsed(), success);
            self.finished = true;
        }
    }
}

impl From<FlushRow> for NodeMetricSummary {
    fn from(row: FlushRow) -> Self {
        Self {
            ts_minute: row.ts_minute,
            source_node_id: row.source_node_id,
            requests: row.requests,
            errors: row.errors,
            requests_local: row.requests_local,
            requests_remote: row.requests_remote,
            request_time_ms_total: row.request_time_ms_total,
            utilization_pct: row.utilization_pct,
            active_requests_peak: row.active_requests_peak,
            latency_p50_ms: row.latency_p50_ms,
            latency_p95_ms: row.latency_p95_ms,
            latency_p99_ms: row.latency_p99_ms,
            tunnel_bytes_total: row.tunnel_bytes_total,
            observed_at: row.observed_at,
        }
    }
}

impl Drop for RequestSpan {
    fn drop(&mut self) {
        if !self.finished {
            self.telemetry
                .finish_request(self.route, self.started.elapsed(), false);
            self.finished = true;
        }
    }
}

fn flush_row_from_state(node_id: &str, state: &State, tunnel_now: u64) -> FlushRow {
    let mut sorted = state.latency_ms_samples.clone();
    sorted.sort_unstable();
    FlushRow {
        ts_minute: state.current_minute,
        source_node_id: node_id.to_string(),
        requests: state.requests,
        errors: state.errors,
        requests_local: state.requests_local,
        requests_remote: state.requests_remote,
        request_time_ms_total: state.request_time_ms_total,
        utilization_pct: state.request_time_ms_total as f64 / 60_000.0 * 100.0,
        active_requests_peak: state.active_requests_peak,
        latency_p50_ms: percentile(&sorted, 0.50),
        latency_p95_ms: percentile(&sorted, 0.95),
        latency_p99_ms: percentile(&sorted, 0.99),
        tunnel_bytes_total: tunnel_now.saturating_sub(state.tunnel_bytes_minute_start),
        observed_at: unix_ts_now(),
    }
}

fn percentile(samples: &[u32], q: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let idx = ((samples.len() - 1) as f64 * q).round() as usize;
    samples.get(idx).map(|v| *v as f64)
}

fn telemetry_db_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let dir = home.join(".meshllm");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("metrics.db")
}

fn unix_ts_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn unix_minute_now() -> i64 {
    unix_ts_now() / 60
}

fn init_db(db_path: &PathBuf) -> Result<()> {
    let conn = Connection::open(db_path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS node_metrics_1m (
            ts_minute INTEGER NOT NULL,
            source_node_id TEXT NOT NULL,
            requests INTEGER NOT NULL,
            errors INTEGER NOT NULL,
            requests_local INTEGER NOT NULL,
            requests_remote INTEGER NOT NULL,
            request_time_ms_total INTEGER NOT NULL,
            utilization_pct REAL NOT NULL,
            active_requests_peak INTEGER NOT NULL,
            latency_p50_ms REAL,
            latency_p95_ms REAL,
            latency_p99_ms REAL,
            tunnel_bytes_total INTEGER NOT NULL,
            observed_at INTEGER NOT NULL,
            PRIMARY KEY (ts_minute, source_node_id)
        );
        CREATE INDEX IF NOT EXISTS idx_node_metrics_time ON node_metrics_1m (ts_minute);
        CREATE INDEX IF NOT EXISTS idx_node_metrics_node_time ON node_metrics_1m (source_node_id, ts_minute);

        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            source_node_id TEXT NOT NULL,
            mesh_id TEXT,
            target_node_id TEXT,
            model TEXT NOT NULL,
            probe_name TEXT NOT NULL,
            probe_type TEXT NOT NULL,
            stream INTEGER NOT NULL,
            temperature REAL NOT NULL,
            top_p REAL NOT NULL,
            max_tokens INTEGER NOT NULL,
            prompt_hash TEXT,
            route_kind TEXT,
            success INTEGER NOT NULL,
            status_code INTEGER,
            latency_ms INTEGER,
            ttft_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            tokens_per_sec REAL,
            error_kind TEXT,
            error_message TEXT,
            response_shape_ok INTEGER,
            settings_json TEXT,
            observed_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_benchmark_runs_time ON benchmark_runs (ts);
        CREATE INDEX IF NOT EXISTS idx_benchmark_runs_node_time ON benchmark_runs (source_node_id, ts);
        CREATE INDEX IF NOT EXISTS idx_benchmark_runs_probe_time ON benchmark_runs (probe_name, ts);
        "#,
    )?;
    Ok(())
}

fn insert_node_metric(db_path: &PathBuf, row: &FlushRow) -> Result<()> {
    let conn = Connection::open(db_path)?;
    conn.execute(
        r#"
        INSERT INTO node_metrics_1m (
            ts_minute, source_node_id, requests, errors, requests_local, requests_remote,
            request_time_ms_total, utilization_pct, active_requests_peak,
            latency_p50_ms, latency_p95_ms, latency_p99_ms, tunnel_bytes_total, observed_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
        ON CONFLICT(ts_minute, source_node_id) DO UPDATE SET
            requests = excluded.requests,
            errors = excluded.errors,
            requests_local = excluded.requests_local,
            requests_remote = excluded.requests_remote,
            request_time_ms_total = excluded.request_time_ms_total,
            utilization_pct = excluded.utilization_pct,
            active_requests_peak = excluded.active_requests_peak,
            latency_p50_ms = excluded.latency_p50_ms,
            latency_p95_ms = excluded.latency_p95_ms,
            latency_p99_ms = excluded.latency_p99_ms,
            tunnel_bytes_total = excluded.tunnel_bytes_total,
            observed_at = excluded.observed_at
        "#,
        params![
            row.ts_minute,
            row.source_node_id,
            row.requests,
            row.errors,
            row.requests_local,
            row.requests_remote,
            row.request_time_ms_total,
            row.utilization_pct,
            row.active_requests_peak,
            row.latency_p50_ms,
            row.latency_p95_ms,
            row.latency_p99_ms,
            row.tunnel_bytes_total,
            row.observed_at,
        ],
    )?;
    Ok(())
}

fn insert_node_metric_summary(db_path: &PathBuf, row: &NodeMetricSummary) -> Result<()> {
    let conn = Connection::open(db_path)?;
    conn.execute(
        r#"
        INSERT INTO node_metrics_1m (
            ts_minute, source_node_id, requests, errors, requests_local, requests_remote,
            request_time_ms_total, utilization_pct, active_requests_peak,
            latency_p50_ms, latency_p95_ms, latency_p99_ms, tunnel_bytes_total, observed_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
        ON CONFLICT(ts_minute, source_node_id) DO UPDATE SET
            requests = excluded.requests,
            errors = excluded.errors,
            requests_local = excluded.requests_local,
            requests_remote = excluded.requests_remote,
            request_time_ms_total = excluded.request_time_ms_total,
            utilization_pct = excluded.utilization_pct,
            active_requests_peak = excluded.active_requests_peak,
            latency_p50_ms = excluded.latency_p50_ms,
            latency_p95_ms = excluded.latency_p95_ms,
            latency_p99_ms = excluded.latency_p99_ms,
            tunnel_bytes_total = excluded.tunnel_bytes_total,
            observed_at = excluded.observed_at
        "#,
        params![
            row.ts_minute,
            row.source_node_id,
            row.requests,
            row.errors,
            row.requests_local,
            row.requests_remote,
            row.request_time_ms_total,
            row.utilization_pct,
            row.active_requests_peak,
            row.latency_p50_ms,
            row.latency_p95_ms,
            row.latency_p99_ms,
            row.tunnel_bytes_total,
            row.observed_at,
        ],
    )?;
    Ok(())
}

fn insert_benchmark_run(db_path: &PathBuf, source_node_id: &str, run: &BenchmarkRunInput) -> Result<()> {
    let conn = Connection::open(db_path)?;
    let settings_json = serde_json::to_string(&benchmark::BenchmarkSettings {
        temperature: run.temperature,
        top_p: run.top_p,
        max_tokens: run.max_tokens,
        stream: run.stream,
    })?;
    conn.execute(
        r#"
        INSERT INTO benchmark_runs (
            ts, source_node_id, mesh_id, target_node_id, model, probe_name, probe_type,
            stream, temperature, top_p, max_tokens, prompt_hash, route_kind, success, status_code,
            latency_ms, ttft_ms, prompt_tokens, completion_tokens, tokens_per_sec, error_kind,
            error_message, response_shape_ok, settings_json, observed_at
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7,
            ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
            ?16, ?17, ?18, ?19, ?20, ?21,
            ?22, ?23, ?24, ?25
        )
        "#,
        params![
            run.ts,
            source_node_id,
            run.mesh_id,
            run.target_node_id,
            run.model,
            run.probe_name,
            run.probe_type,
            if run.stream { 1 } else { 0 },
            run.temperature,
            run.top_p,
            run.max_tokens,
            run.prompt_hash,
            run.route_kind,
            if run.success { 1 } else { 0 },
            run.status_code,
            run.latency_ms,
            run.ttft_ms,
            run.prompt_tokens,
            run.completion_tokens,
            run.tokens_per_sec,
            run.error_kind,
            run.error_message,
            run.response_shape_ok.map(|v| if v { 1 } else { 0 }),
            settings_json,
            unix_ts_now(),
        ],
    )?;
    Ok(())
}

fn query_node_history(db_path: &PathBuf, node_id: &str, minutes: u32) -> Result<Vec<NodeMetricRow>> {
    let conn = Connection::open(db_path)?;
    let cutoff = unix_minute_now() - minutes as i64;
    let mut stmt = conn.prepare(
        r#"
        SELECT
            ts_minute, source_node_id, requests, errors, requests_local, requests_remote,
            request_time_ms_total, utilization_pct, active_requests_peak,
            latency_p50_ms, latency_p95_ms, latency_p99_ms, tunnel_bytes_total, observed_at
        FROM node_metrics_1m
        WHERE source_node_id = ?1 AND ts_minute >= ?2
        ORDER BY ts_minute ASC
        "#,
    )?;
    let rows = stmt
        .query_map(params![node_id, cutoff], map_node_row)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn query_node_history_exact_minutes(
    db_path: &PathBuf,
    node_id: &str,
    minutes: &[i64],
) -> Result<Vec<NodeMetricSummary>> {
    if minutes.is_empty() {
        return Ok(Vec::new());
    }
    let conn = Connection::open(db_path)?;
    let mut out = Vec::new();
    for minute in minutes {
        let mut stmt = conn.prepare(
            r#"
            SELECT
                ts_minute, source_node_id, requests, errors, requests_local, requests_remote,
                request_time_ms_total, utilization_pct, active_requests_peak,
                latency_p50_ms, latency_p95_ms, latency_p99_ms, tunnel_bytes_total, observed_at
            FROM node_metrics_1m
            WHERE source_node_id = ?1 AND ts_minute = ?2
            LIMIT 1
            "#,
        )?;
        let mut rows = stmt.query(params![node_id, minute])?;
        if let Some(row) = rows.next()? {
            out.push(NodeMetricSummary {
                ts_minute: row.get(0)?,
                source_node_id: row.get(1)?,
                requests: row.get(2)?,
                errors: row.get(3)?,
                requests_local: row.get(4)?,
                requests_remote: row.get(5)?,
                request_time_ms_total: row.get(6)?,
                utilization_pct: row.get(7)?,
                active_requests_peak: row.get::<_, u32>(8)?,
                latency_p50_ms: row.get(9)?,
                latency_p95_ms: row.get(10)?,
                latency_p99_ms: row.get(11)?,
                tunnel_bytes_total: row.get(12)?,
                observed_at: row.get(13)?,
            });
        }
    }
    Ok(out)
}

fn query_all_nodes_latest(db_path: &PathBuf, minutes: u32) -> Result<Vec<NodeMetricRow>> {
    let conn = Connection::open(db_path)?;
    let cutoff = unix_minute_now() - minutes as i64;
    let mut stmt = conn.prepare(
        r#"
        SELECT m.ts_minute, m.source_node_id, m.requests, m.errors, m.requests_local, m.requests_remote,
               m.request_time_ms_total, m.utilization_pct, m.active_requests_peak,
               m.latency_p50_ms, m.latency_p95_ms, m.latency_p99_ms, m.tunnel_bytes_total, m.observed_at
        FROM node_metrics_1m m
        INNER JOIN (
            SELECT source_node_id, MAX(ts_minute) AS max_ts
            FROM node_metrics_1m
            WHERE ts_minute >= ?1
            GROUP BY source_node_id
        ) latest
        ON m.source_node_id = latest.source_node_id AND m.ts_minute = latest.max_ts
        ORDER BY m.source_node_id ASC
        "#,
    )?;
    let rows = stmt
        .query_map(params![cutoff], map_node_row)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn query_rollup_history(db_path: &PathBuf, minutes: u32) -> Result<Vec<RollupMetricRow>> {
    let conn = Connection::open(db_path)?;
    let cutoff = unix_minute_now() - minutes as i64;
    let mut stmt = conn.prepare(
        r#"
        SELECT
            ts_minute,
            COUNT(*) AS node_count,
            COALESCE(SUM(requests), 0) AS requests,
            COALESCE(SUM(errors), 0) AS errors,
            COALESCE(SUM(requests_local), 0) AS requests_local,
            COALESCE(SUM(requests_remote), 0) AS requests_remote,
            COALESCE(SUM(request_time_ms_total), 0) AS request_time_ms_total,
            COALESCE(AVG(utilization_pct), 0.0) AS utilization_pct_avg_nodes,
            COALESCE(MAX(active_requests_peak), 0) AS active_requests_peak_max,
            COALESCE(SUM(tunnel_bytes_total), 0) AS tunnel_bytes_total,
            AVG(latency_p95_ms) AS latency_p95_ms_avg_nodes,
            MAX(latency_p95_ms) AS latency_p95_ms_max_nodes
        FROM node_metrics_1m
        WHERE ts_minute >= ?1
        GROUP BY ts_minute
        ORDER BY ts_minute ASC
        "#,
    )?;
    let rows = stmt
        .query_map(params![cutoff], |row| {
            Ok(RollupMetricRow {
                ts_minute: row.get(0)?,
                node_count: row.get(1)?,
                requests: row.get(2)?,
                errors: row.get(3)?,
                requests_local: row.get(4)?,
                requests_remote: row.get(5)?,
                request_time_ms_total: row.get(6)?,
                utilization_pct_avg_nodes: row.get(7)?,
                active_requests_peak_max: row.get::<_, u32>(8)?,
                tunnel_bytes_total: row.get(9)?,
                latency_p95_ms_avg_nodes: row.get(10)?,
                latency_p95_ms_max_nodes: row.get(11)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn query_benchmark_history(db_path: &PathBuf, minutes: u32, limit: u32) -> Result<Vec<BenchmarkRunRow>> {
    let conn = Connection::open(db_path)?;
    let cutoff = unix_ts_now() - (minutes as i64 * 60);
    let mut stmt = conn.prepare(
        r#"
        SELECT ts, source_node_id, model, probe_name, probe_type, stream, success, status_code,
               latency_ms, ttft_ms, completion_tokens, tokens_per_sec, error_kind, observed_at
        FROM benchmark_runs
        WHERE ts >= ?1
        ORDER BY ts DESC
        LIMIT ?2
        "#,
    )?;
    let rows = stmt.query_map(params![cutoff, limit], |row| {
        let status_code_i: Option<i64> = row.get(7)?;
        let status_code = status_code_i.and_then(|v| u16::try_from(v).ok());
        Ok(BenchmarkRunRow {
            ts: row.get(0)?,
            source_node_id: row.get(1)?,
            model: row.get(2)?,
            probe_name: row.get(3)?,
            probe_type: row.get(4)?,
            stream: row.get::<_, i64>(5)? != 0,
            success: row.get::<_, i64>(6)? != 0,
            status_code,
            latency_ms: row.get::<_, Option<u32>>(8)?,
            ttft_ms: row.get::<_, Option<u32>>(9)?,
            completion_tokens: row.get::<_, Option<u32>>(10)?,
            tokens_per_sec: row.get::<_, Option<f64>>(11)?,
            error_kind: row.get(12)?,
            observed_at: row.get(13)?,
        })
    })?
    .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}

fn map_node_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<NodeMetricRow> {
    Ok(NodeMetricRow {
        ts_minute: row.get(0)?,
        source_node_id: row.get(1)?,
        requests: row.get(2)?,
        errors: row.get(3)?,
        requests_local: row.get(4)?,
        requests_remote: row.get(5)?,
        request_time_ms_total: row.get(6)?,
        utilization_pct: row.get(7)?,
        active_requests_peak: row.get::<_, u32>(8)?,
        latency_p50_ms: row.get(9)?,
        latency_p95_ms: row.get(10)?,
        latency_p99_ms: row.get(11)?,
        tunnel_bytes_total: row.get(12)?,
        observed_at: row.get(13)?,
    })
}
