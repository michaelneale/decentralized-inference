// Trial execution and lifecycle management.

use super::streaming;
use super::trial_config;
use super::{
    TuneBenchmarkCandidate, TuneBenchmarkRunRequest, TuneBenchmarkTimingStats, TuneBenchmarkTrial,
    TuneBenchmarkTrialStatus,
};
use crate::gpus::tune_apply::PreparedTunePlan;
#[cfg(unix)]
use nix::sys::signal::{Signal, kill};
#[cfg(unix)]
use nix::unistd::Pid;

const MAX_TRIAL_PORT_RETRIES: usize = 3;
const PORT_BIND_ERROR_HINTS: [&str; 6] = [
    "address already in use",
    "failed to bind",
    "os error 98",
    "os error 10048",
    "address in use",
    "could not bind",
];

pub(crate) fn run_trial(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &PreparedTunePlan,
    index: usize,
    candidate: TuneBenchmarkCandidate,
) -> TuneBenchmarkTrial {
    match run_trial_inner(request, prepared, index, &candidate) {
        Ok(success) => success,
        Err(error) => TuneBenchmarkTrial {
            candidate,
            status: TuneBenchmarkTrialStatus::Failed,
            completion_tokens: None,
            elapsed_ms: None,
            decode_tok_s: None,
            ttft_ms: None,
            decode_only_tok_s: None,
            timings: None,
            log_path: None,
            error: Some(error.to_string()),
        },
    }
}

pub(crate) fn run_trial_inner(
    request: &TuneBenchmarkRunRequest<'_>,
    prepared: &PreparedTunePlan,
    index: usize,
    candidate: &TuneBenchmarkCandidate,
) -> anyhow::Result<TuneBenchmarkTrial> {
    anyhow::ensure!(
        request.max_tokens > 0,
        "--max-tokens must be greater than zero"
    );
    let mut timings = TrialTimingRecorder::new();
    let setup_started = std::time::Instant::now();
    let trial_dir = create_trial_dir(prepared, index)?;
    let config_path = trial_dir.join("config.toml");
    let log_path = trial_dir.join("serve.log");
    std::fs::write(
        &config_path,
        trial_config(request.config, prepared, candidate)?,
    )?;

    let request_timeout = std::time::Duration::from_secs(request.request_timeout_secs.max(1));
    let client = reqwest::blocking::Client::builder()
        .timeout(request_timeout)
        .build()?;
    timings.setup_ms = elapsed_ms_since(setup_started);

    let mut attempts = 0;
    loop {
        let port = reserve_local_port()?;
        let console = reserve_local_port()?;
        let mut child = TrialChild::spawn(
            &config_path,
            &log_path,
            port,
            console,
            request.debug_telemetry,
        )?;

        let readiness_started = std::time::Instant::now();
        let readiness_result = wait_for_trial_ready(TrialReadinessWait {
            client: &client,
            child: &mut child,
            log_path: &log_path,
            port,
            prompt: request.prompt,
            startup_timeout_secs: request.startup_timeout_secs,
            request_timeout,
            readiness_attempts: &mut timings.readiness_attempts,
        });
        timings.readiness_ms = elapsed_ms_since(readiness_started);
        if let Err(error) = readiness_result {
            if should_retry_with_new_ports(error.as_ref(), &log_path)
                && attempts + 1 < MAX_TRIAL_PORT_RETRIES
            {
                record_shutdown(&mut child, &mut timings);
                attempts += 1;
                continue;
            }
            return Ok(finish_failed_trial(
                candidate,
                &log_path,
                &mut timings,
                &mut child,
                error,
            ));
        }

        let started = std::time::Instant::now();
        let response_result = send_chat_request_with_watchdog(
            &client,
            &mut child,
            port,
            request.prompt,
            request.max_tokens,
            request_timeout,
        );
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        timings.request_ms = Some(elapsed_ms);
        let outcome = match response_result {
            Ok(outcome) => outcome,
            Err(error) => {
                return Ok(finish_failed_trial(
                    candidate,
                    &log_path,
                    &mut timings,
                    &mut child,
                    error,
                ));
            }
        };
        let completion_tokens = outcome.completion_tokens;
        if completion_tokens == 0 {
            return Ok(finish_failed_trial(
                candidate,
                &log_path,
                &mut timings,
                &mut child,
                anyhow::anyhow!("chat completion returned zero completion tokens"),
            ));
        }
        let decode_tok_s = completion_tokens as f64 / (elapsed_ms / 1000.0);
        let decode_only_tok_s =
            streaming::decode_only_tok_s(completion_tokens, elapsed_ms, outcome.ttft_ms);
        record_shutdown(&mut child, &mut timings);

        return Ok(TuneBenchmarkTrial {
            candidate: candidate.clone(),
            status: TuneBenchmarkTrialStatus::Succeeded,
            completion_tokens: Some(completion_tokens),
            elapsed_ms: Some(elapsed_ms),
            decode_tok_s: Some(decode_tok_s),
            ttft_ms: outcome.ttft_ms,
            decode_only_tok_s,
            timings: Some(timings.snapshot()),
            log_path: Some(log_path.display().to_string()),
            error: None,
        });
    }
}

pub(crate) struct TrialTimingRecorder {
    trial_started: std::time::Instant,
    setup_ms: f64,
    readiness_ms: f64,
    request_ms: Option<f64>,
    shutdown_ms: Option<f64>,
    readiness_attempts: u32,
}

impl TrialTimingRecorder {
    fn new() -> Self {
        Self {
            trial_started: std::time::Instant::now(),
            setup_ms: 0.0,
            readiness_ms: 0.0,
            request_ms: None,
            shutdown_ms: None,
            readiness_attempts: 0,
        }
    }

    fn snapshot(&self) -> TuneBenchmarkTimingStats {
        TuneBenchmarkTimingStats {
            total_ms: elapsed_ms_since(self.trial_started),
            setup_ms: self.setup_ms,
            readiness_ms: self.readiness_ms,
            request_ms: self.request_ms,
            shutdown_ms: self.shutdown_ms,
            readiness_attempts: self.readiness_attempts,
        }
    }
}

pub(crate) fn elapsed_ms_since(started: std::time::Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

pub(crate) fn record_shutdown(child: &mut TrialChild, timings: &mut TrialTimingRecorder) {
    let shutdown_started = std::time::Instant::now();
    child.shutdown();
    timings.shutdown_ms = Some(elapsed_ms_since(shutdown_started));
}

pub(crate) fn finish_failed_trial(
    candidate: &TuneBenchmarkCandidate,
    log_path: &std::path::Path,
    timings: &mut TrialTimingRecorder,
    child: &mut TrialChild,
    error: impl std::fmt::Display,
) -> TuneBenchmarkTrial {
    record_shutdown(child, timings);
    failed_trial_with_evidence(candidate, log_path, timings.snapshot(), error)
}

pub(crate) fn failed_trial_with_evidence(
    candidate: &TuneBenchmarkCandidate,
    log_path: &std::path::Path,
    timings: TuneBenchmarkTimingStats,
    error: impl std::fmt::Display,
) -> TuneBenchmarkTrial {
    TuneBenchmarkTrial {
        candidate: candidate.clone(),
        status: TuneBenchmarkTrialStatus::Failed,
        completion_tokens: None,
        elapsed_ms: timings.request_ms,
        decode_tok_s: None,
        ttft_ms: None,
        decode_only_tok_s: None,
        timings: Some(timings),
        log_path: Some(log_path.display().to_string()),
        error: Some(error.to_string()),
    }
}

pub(crate) struct TrialChild {
    child: std::process::Child,
}

impl TrialChild {
    fn spawn(
        config_path: &std::path::Path,
        log_path: &std::path::Path,
        port: u16,
        console: u16,
        debug_telemetry: bool,
    ) -> anyhow::Result<Self> {
        let exe = std::env::current_exe()?;
        let log = std::fs::File::create(log_path)?;
        let stderr = log.try_clone()?;
        let child = build_trial_child_command(&exe, config_path, port, console, debug_telemetry)
            .stdout(std::process::Stdio::from(log))
            .stderr(std::process::Stdio::from(stderr))
            .spawn()?;
        Ok(Self { child })
    }

    fn shutdown(&mut self) {
        terminate_child(&mut self.child);
    }
}

impl Drop for TrialChild {
    fn drop(&mut self) {
        terminate_child(&mut self.child);
    }
}

pub(crate) fn build_trial_child_command(
    exe: &std::path::Path,
    config_path: &std::path::Path,
    port: u16,
    console: u16,
    debug_telemetry: bool,
) -> std::process::Command {
    let mut command = std::process::Command::new(exe);
    if debug_telemetry {
        command.arg("--debug").env("SKIPPY_TELEMETRY_STDERR", "1");
    }
    command
        .arg("--config")
        .arg(config_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--console")
        .arg(console.to_string())
        .arg("--log-format")
        .arg("json")
        .arg("--headless")
        .arg("serve");
    command
}

pub(crate) fn terminate_child(child: &mut std::process::Child) {
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }
    #[cfg(unix)]
    {
        if let Ok(pid) = i32::try_from(child.id()) {
            let pid = Pid::from_raw(pid);
            if kill(pid, Signal::SIGTERM).is_err() {
                let _ = child.kill();
            }
        } else {
            let _ = child.kill();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while std::time::Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    let _ = child.kill();
    let _ = child.wait();
}

pub(crate) struct TrialReadinessWait<'a> {
    client: &'a reqwest::blocking::Client,
    child: &'a mut TrialChild,
    log_path: &'a std::path::Path,
    port: u16,
    prompt: &'a str,
    startup_timeout_secs: u64,
    request_timeout: std::time::Duration,
    readiness_attempts: &'a mut u32,
}

pub(crate) fn wait_for_trial_ready(wait: TrialReadinessWait<'_>) -> anyhow::Result<()> {
    let deadline = std::time::Instant::now()
        + std::time::Duration::from_secs(wait.startup_timeout_secs.max(1));
    let mut last_error = String::new();
    while std::time::Instant::now() < deadline {
        if let Some(status) = wait.child.child.try_wait()? {
            if let Some(error) = trial_startup_failure_from_log(wait.log_path) {
                anyhow::bail!("trial startup failed: {error}");
            }
            anyhow::bail!("trial server exited before readiness: {status}");
        }
        if let Some(error) = trial_startup_failure_from_log(wait.log_path) {
            anyhow::bail!("trial startup failed: {error}");
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let attempt_timeout = std::cmp::min(wait.request_timeout, remaining);
        *wait.readiness_attempts += 1;
        match send_chat_request_with_watchdog(
            wait.client,
            wait.child,
            wait.port,
            wait.prompt,
            1,
            attempt_timeout,
        ) {
            Ok(_) => return Ok(()),
            Err(error) => last_error = error.to_string(),
        }
        if let Some(error) = trial_startup_failure_from_log(wait.log_path) {
            anyhow::bail!("trial startup failed: {error}");
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    anyhow::bail!("trial server did not become ready: {last_error}");
}

pub(crate) fn trial_startup_failure_from_log(log_path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(log_path).ok()?;
    contents
        .lines()
        .rev()
        .take(200)
        .find_map(trial_startup_failure_from_log_line)
}

pub(crate) fn trial_startup_failure_from_log_line(line: &str) -> Option<String> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(line)
        && let Some(message) = value.get("message").and_then(|value| value.as_str())
        && message.contains("Failed to start model")
    {
        return Some(message.to_string());
    }
    line.contains("Failed to start model")
        .then(|| line.trim().to_string())
}

pub(crate) fn send_chat_request_with_watchdog(
    client: &reqwest::blocking::Client,
    child: &mut TrialChild,
    port: u16,
    prompt: &str,
    max_tokens: u32,
    timeout: std::time::Duration,
) -> anyhow::Result<streaming::TrialChatOutcome> {
    let client = client.clone();
    let prompt = prompt.to_string();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = sender.send(streaming::send_chat_request(
            &client, port, &prompt, max_tokens,
        ));
    });

    match receiver.recv_timeout(timeout.max(std::time::Duration::from_secs(1))) {
        Ok(result) => result,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            child.shutdown();
            anyhow::bail!(
                "chat completion exceeded request timeout of {}s",
                timeout.as_secs().max(1)
            );
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            anyhow::bail!("chat completion worker exited without a response")
        }
    }
}

fn should_retry_with_new_ports(
    error: &(dyn std::error::Error + 'static),
    log_path: &std::path::Path,
) -> bool {
    if error_string_is_port_bind_error(&error.to_string()) {
        return true;
    }
    trial_startup_failure_from_log(log_path)
        .is_some_and(|line| error_string_is_port_bind_error(&line))
}

fn error_string_is_port_bind_error(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    PORT_BIND_ERROR_HINTS
        .iter()
        .any(|hint| lower.contains(hint))
}

pub(crate) fn create_trial_dir(
    prepared: &PreparedTunePlan,
    index: usize,
) -> anyhow::Result<std::path::PathBuf> {
    let base = std::env::temp_dir().join("mesh-llm-tune");
    let mut dir = base.join(sanitize_path_component(
        &prepared.target.canonical_model_ref,
    ));
    dir.push(format!(
        "{}-{}-{index}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
        std::process::id(),
    ));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub(crate) fn reserve_local_port() -> anyhow::Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

pub(crate) fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}
