//! Shared backend-adjacent helpers that are still needed outside model serving.

use anyhow::Result;
use clap::ValueEnum;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BinaryFlavor {
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
}

impl BinaryFlavor {
    pub const ALL: [BinaryFlavor; 5] = [
        BinaryFlavor::Cpu,
        BinaryFlavor::Cuda,
        BinaryFlavor::Rocm,
        BinaryFlavor::Vulkan,
        BinaryFlavor::Metal,
    ];

    pub fn suffix(self) -> &'static str {
        match self {
            BinaryFlavor::Cpu => "cpu",
            BinaryFlavor::Cuda => "cuda",
            BinaryFlavor::Rocm => "rocm",
            BinaryFlavor::Vulkan => "vulkan",
            BinaryFlavor::Metal => "metal",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BinaryBackendDeviceProbe {
    pub path: PathBuf,
    pub flavor: Option<BinaryFlavor>,
    pub available_devices: Vec<String>,
}

static RUNTIME_SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

pub fn mark_runtime_shutting_down() {
    RUNTIME_SHUTTING_DOWN.store(true, Ordering::SeqCst);
}

pub fn runtime_shutting_down() -> bool {
    RUNTIME_SHUTTING_DOWN.load(Ordering::SeqCst)
}

pub fn clear_runtime_shutting_down() {
    RUNTIME_SHUTTING_DOWN.store(false, Ordering::SeqCst);
}

pub fn platform_bin_name(name: &str) -> String {
    #[cfg(windows)]
    {
        if Path::new(name)
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("exe"))
        {
            name.to_string()
        } else {
            format!("{name}.exe")
        }
    }

    #[cfg(not(windows))]
    {
        name.to_string()
    }
}

pub fn backend_device_for_flavor(index: usize, binary_flavor: BinaryFlavor) -> Option<String> {
    match binary_flavor {
        BinaryFlavor::Cpu => None,
        BinaryFlavor::Cuda => Some(format!("CUDA{index}")),
        BinaryFlavor::Rocm => Some(format!("ROCm{index}")),
        BinaryFlavor::Vulkan => Some(format!("Vulkan{index}")),
        BinaryFlavor::Metal => Some(format!("MTL{index}")),
    }
}

pub fn resolve_requested_device_from_available(
    available: &[String],
    binary: &Path,
    requested: &str,
) -> Result<String> {
    if !available.is_empty() {
        if let Some(candidate) = available
            .iter()
            .find(|candidate| backend_device_names_match(candidate, requested))
        {
            return Ok(candidate.clone());
        }

        anyhow::bail!(
            "requested device {requested} is not supported by {}. Available devices: {}",
            binary.display(),
            available.join(", ")
        );
    }

    Ok(requested.to_string())
}

/// Matches backend device identifiers while treating ROCm and HIP prefixes as
/// aliases for the same numbered AMD device.
pub fn backend_device_names_match(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
        || amd_backend_ordinal(left)
            .zip(amd_backend_ordinal(right))
            .is_some_and(|(left_ordinal, right_ordinal)| left_ordinal == right_ordinal)
}

fn amd_backend_ordinal(device: &str) -> Option<&str> {
    let ordinal = if device
        .get(..4)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("rocm"))
    {
        device.get(4..)?
    } else if device
        .get(..3)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("hip"))
    {
        device.get(3..)?
    } else {
        return None;
    };

    (!ordinal.is_empty() && ordinal.chars().all(|ch| ch.is_ascii_digit())).then_some(ordinal)
}

#[cfg(test)]
mod backend_device_tests {
    use super::*;

    #[test]
    fn rocm_and_hip_device_names_are_numbered_aliases() {
        assert!(backend_device_names_match("ROCm0", "HIP0"));
        assert!(backend_device_names_match("hip12", "rocm12"));
        assert!(!backend_device_names_match("ROCm0", "HIP1"));
        assert!(!backend_device_names_match("HIPster", "ROCm0"));
    }

    #[test]
    fn requested_amd_alias_resolves_to_runtime_emitted_name() {
        let binary = Path::new("mesh-llm");

        assert_eq!(
            resolve_requested_device_from_available(&["ROCm0".into()], binary, "HIP0").unwrap(),
            "ROCm0"
        );
        assert_eq!(
            resolve_requested_device_from_available(&["HIP1".into()], binary, "ROCm1").unwrap(),
            "HIP1"
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProcessSignal {
    Terminate,
    Kill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SignalOutcome {
    Sent,
    // Only the non-Windows pid/name verification path constructs these.
    #[cfg_attr(windows, allow(dead_code))]
    AlreadyDead,
    #[cfg_attr(windows, allow(dead_code))]
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationOutcome {
    NotRunning,
    Graceful,
    Killed,
    Failed,
}

impl TerminationOutcome {
    pub fn is_success(self) -> bool {
        !matches!(self, TerminationOutcome::Failed)
    }
}

pub fn is_safe_kill_target(pid: u32) -> bool {
    pid > 1 && pid <= i32::MAX as u32
}

pub fn terminate_process_blocking(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
) -> TerminationOutcome {
    match send_signal_if_matches(
        pid,
        expected_comm,
        expected_start_time,
        ProcessSignal::Terminate,
    ) {
        SignalOutcome::Sent => {}
        SignalOutcome::AlreadyDead => return TerminationOutcome::NotRunning,
        SignalOutcome::Skipped | SignalOutcome::Failed => return TerminationOutcome::Failed,
    }

    for _ in 0..20 {
        std::thread::sleep(Duration::from_millis(250));
        if crate::process::process_liveness(pid) == crate::process::Liveness::Dead {
            return TerminationOutcome::Graceful;
        }
    }

    match send_signal_if_matches(pid, expected_comm, expected_start_time, ProcessSignal::Kill) {
        SignalOutcome::Sent => TerminationOutcome::Killed,
        SignalOutcome::AlreadyDead => TerminationOutcome::Graceful,
        SignalOutcome::Skipped | SignalOutcome::Failed => TerminationOutcome::Failed,
    }
}

fn send_signal_if_matches(
    pid: u32,
    expected_comm: &str,
    expected_start_time: Option<i64>,
    signal: ProcessSignal,
) -> SignalOutcome {
    if !is_safe_kill_target(pid) {
        tracing::error!("BUG: attempted to signal unsafe pid {pid} - refusing");
        return SignalOutcome::Failed;
    }

    #[cfg(not(windows))]
    {
        let matches = if let Some(expected_t) = expected_start_time {
            crate::process::validate_pid_matches(pid, expected_comm, expected_t)
        } else {
            crate::process::process_name_matches(pid, expected_comm)
        };
        if !matches {
            if crate::process::process_liveness(pid) == crate::process::Liveness::Dead {
                return SignalOutcome::AlreadyDead;
            }
            tracing::warn!("pid {pid} no longer matches {expected_comm}, skipping signal");
            return SignalOutcome::Skipped;
        }
    }

    #[cfg(windows)]
    {
        let _ = (expected_comm, expected_start_time);
    }

    #[cfg(unix)]
    unsafe {
        let ret = libc::kill(
            pid as libc::pid_t,
            match signal {
                ProcessSignal::Terminate => libc::SIGTERM,
                ProcessSignal::Kill => libc::SIGKILL,
            },
        );
        if ret == 0 {
            return SignalOutcome::Sent;
        }

        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            return SignalOutcome::AlreadyDead;
        }

        tracing::warn!(pid, error = %err, ?signal, "failed to signal process");
        SignalOutcome::Failed
    }

    #[cfg(windows)]
    {
        let pid_str = pid.to_string();
        let mut command = std::process::Command::new("taskkill");
        command.args(["/PID", &pid_str, "/T"]);
        if signal == ProcessSignal::Kill {
            command.arg("/F");
        }
        match command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
        {
            Ok(status) if status.success() => SignalOutcome::Sent,
            Ok(status) => {
                tracing::warn!(pid, exit_code = status.code(), ?signal, "taskkill failed");
                SignalOutcome::Failed
            }
            Err(err) => {
                tracing::warn!(pid, error = %err, ?signal, "failed to run taskkill");
                SignalOutcome::Failed
            }
        }
    }
}
