use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::hardware::HardwareSurvey;

pub enum BenchmarkKind {
    MemoryBandwidth,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkOutput {
    pub device: String,
    pub buffer_mb: u32,
    pub runs: u32,
    pub p50_gbps: f64,
    pub p90_gbps: f64,
    pub noise_pct: f64,
    pub runtime_s: f64,
    pub rated_gbps: Option<f64>,
    pub rated_estimated: Option<bool>,
    pub efficiency_pct: Option<f64>,
    pub bus_width_bits: Option<u32>,
    pub mem_clock_mhz: Option<u64>,
    pub gcn_arch: Option<String>,
    pub hbm: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuBandwidth {
    pub name: String,
    pub vram_bytes: u64,
    pub p50_gbps: f64,
    pub p90_gbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkFingerprint {
    pub gpus: Vec<GpuBandwidth>, // per-GPU identity + bandwidth, in device order
    pub is_soc: bool,
    pub timestamp_secs: u64,
}

/// Returns true if the current hardware differs from the fingerprint's recorded hardware.
/// Compares GPU names, VRAM sizes (by index), and the is_soc flag.
pub fn hardware_changed(fingerprint: &BenchmarkFingerprint, hw: &HardwareSurvey) -> bool {
    if fingerprint.is_soc != hw.is_soc {
        return true;
    }
    let hw_names: Vec<&str> = hw
        .gpu_name
        .as_deref()
        .map(|s| s.split(", ").map(str::trim).collect())
        .unwrap_or_default();
    if fingerprint.gpus.len() != hw_names.len() || fingerprint.gpus.len() != hw.gpu_vram.len() {
        return true;
    }
    for (i, cached) in fingerprint.gpus.iter().enumerate() {
        if cached.name != hw_names[i] || cached.vram_bytes != hw.gpu_vram[i] {
            return true;
        }
    }
    false
}

/// Returns `~/.mesh-llm/benchmark-fingerprint.json`.
/// Falls back to `/tmp/mesh-llm-benchmark-fingerprint.json` if home dir is unavailable.
pub fn fingerprint_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".mesh-llm")
        .join("benchmark-fingerprint.json")
}

/// Load a `BenchmarkFingerprint` from disk.  Returns `None` on any error.
pub fn load_fingerprint(path: &Path) -> Option<BenchmarkFingerprint> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Atomically write a `BenchmarkFingerprint` to disk.
/// Uses a `.json.tmp` staging file + rename for crash safety.
/// Logs a warning on failure — never panics.
pub fn save_fingerprint(path: &Path, fp: &BenchmarkFingerprint) {
    let tmp = path.with_extension("json.tmp");

    if let Err(e) = std::fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new("."))) {
        tracing::warn!("benchmark: failed to create cache dir: {e}");
        return;
    }

    let json = match serde_json::to_string_pretty(fp) {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!("benchmark: failed to serialize fingerprint: {e}");
            return;
        }
    };

    if let Err(e) = std::fs::write(&tmp, &json) {
        tracing::warn!("benchmark: failed to write tmp fingerprint: {e}");
        return;
    }

    if let Err(e) = std::fs::rename(&tmp, path) {
        tracing::warn!("benchmark: failed to rename fingerprint into place: {e}");
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Determine which benchmark binary to use for the current hardware platform.
///
/// Returns `None` (soft failure) if:
/// - No GPUs are present
/// - The binary is not found on disk
/// - The platform/GPU combination is unrecognised
///
/// Never panics or hard-fails with `ensure!`.
pub fn detect_benchmark_binary(hw: &HardwareSurvey, bin_dir: &Path) -> Option<PathBuf> {
    if hw.gpu_count == 0 {
        tracing::debug!("no GPUs detected — skipping benchmark");
        return None;
    }

    let gpu_upper = hw.gpu_name.as_deref().unwrap_or("").to_uppercase();

    let candidate = if cfg!(target_os = "macos") && hw.is_soc {
        bin_dir.join("membench-fingerprint")
    } else if cfg!(target_os = "linux") {
        if gpu_upper.contains("NVIDIA") {
            bin_dir.join("membench-fingerprint-cuda")
        } else if gpu_upper.contains("AMD") || gpu_upper.contains("RADEON") {
            bin_dir.join("membench-fingerprint-hip")
        } else if gpu_upper.contains("INTEL") || gpu_upper.contains("ARC") {
            tracing::info!("Intel Arc benchmark is unvalidated — results may be inaccurate");
            bin_dir.join("membench-fingerprint-intel")
        } else if hw.is_soc {
            tracing::warn!("Jetson benchmark is unvalidated for ARM CUDA — attempting");
            bin_dir.join("membench-fingerprint-cuda")
        } else {
            tracing::warn!(
                "could not identify benchmark binary for this GPU platform: {:?}",
                hw.gpu_name
            );
            return None;
        }
    } else {
        tracing::warn!(
            "could not identify benchmark binary for this GPU platform: {:?}",
            hw.gpu_name
        );
        return None;
    };

    if candidate.exists() {
        return Some(candidate);
    }

    // Fallback: try the directory containing the mesh-llm executable itself.
    // In dev builds, benchmark binaries are built to `target/release/` (adjacent
    // to `mesh-llm`) while `bin_dir` points to `llama.cpp/build/bin/`.
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let filename = candidate.file_name()?;
            let fallback = exe_dir.join(filename);
            if fallback.exists() {
                return Some(fallback);
            }
        }
    }

    tracing::warn!(
        "{} not found in {:?} or adjacent to mesh-llm executable",
        candidate
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("benchmark binary"),
        bin_dir
    );
    None
}

/// Parse raw stdout bytes from a benchmark run into a vec of per-device outputs.
///
/// Expects a JSON array of [`BenchmarkOutput`].  Returns `None` on any parse
/// failure or if the device list is empty.
pub fn parse_benchmark_output(stdout: &[u8]) -> Option<Vec<BenchmarkOutput>> {
    match serde_json::from_slice::<Vec<BenchmarkOutput>>(stdout) {
        Ok(outputs) if !outputs.is_empty() => Some(outputs),
        Ok(_) => {
            tracing::debug!("benchmark returned empty device list");
            None
        }
        Err(err) => {
            if let Ok(val) = serde_json::from_slice::<serde_json::Value>(stdout) {
                if let Some(msg) = val.get("error").and_then(|v| v.as_str()) {
                    tracing::warn!("benchmark reported error: {msg}");
                    return None;
                }
            }
            tracing::warn!("failed to parse benchmark output: {err}");
            None
        }
    }
}

/// Run the benchmark binary synchronously and return per-device outputs.
///
/// Designed to be called inside `tokio::task::spawn_blocking` — never `async`.
pub fn run_benchmark(binary: &Path) -> Option<Vec<BenchmarkOutput>> {
    let output = match std::process::Command::new(binary).arg("--json").output() {
        Ok(o) => o,
        Err(e) => {
            tracing::error!("failed to spawn {binary:?}: {e}");
            return None;
        }
    };

    if !output.status.success() {
        tracing::warn!("benchmark exited with {:?}", output.status);
        return None;
    }

    parse_benchmark_output(&output.stdout)
}

/// Load a cached fingerprint if hardware is unchanged, otherwise run the
/// benchmark binary and persist the result.
///
/// Not `async` — intended for use inside `tokio::task::spawn_blocking`.
pub fn run_or_load(hw: &HardwareSurvey, bin_dir: &Path) -> Option<Vec<f64>> {
    let path = fingerprint_path();

    // Cache-hit path
    if let Some(ref cached) = load_fingerprint(&path) {
        if !hardware_changed(cached, hw) {
            let per_gpu: Vec<f64> = cached.gpus.iter().map(|g| g.p90_gbps).collect();
            tracing::info!("Using cached bandwidth fingerprint: {} GPUs", per_gpu.len());
            return Some(per_gpu);
        }
    }

    tracing::info!("Hardware changed or no cache — running memory bandwidth benchmark");

    let binary = detect_benchmark_binary(hw, bin_dir)?;
    let outputs = run_benchmark(&binary)?;

    // Build per-GPU entries by zipping benchmark output with hw survey data.
    // Both are in device order (nvidia-smi / CUDA enumerate the same order).
    let hw_names: Vec<&str> = hw
        .gpu_name
        .as_deref()
        .map(|s| s.split(", ").map(str::trim).collect())
        .unwrap_or_default();

    let count = outputs
        .len()
        .min(hw.gpu_vram.len())
        .min(if hw_names.is_empty() {
            usize::MAX
        } else {
            hw_names.len()
        });

    let gpus: Vec<GpuBandwidth> = (0..count)
        .map(|i| GpuBandwidth {
            name: hw_names.get(i).copied().unwrap_or("").trim().to_owned(),
            vram_bytes: hw.gpu_vram.get(i).copied().unwrap_or(0),
            p50_gbps: outputs[i].p50_gbps,
            p90_gbps: outputs[i].p90_gbps,
        })
        .collect();

    let per_gpu: Vec<f64> = gpus.iter().map(|g| g.p90_gbps).collect();

    let fingerprint = BenchmarkFingerprint {
        gpus,
        is_soc: hw.is_soc,
        timestamp_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };

    save_fingerprint(&path, &fingerprint);
    Some(per_gpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_survey(
        gpu_count: u8,
        gpu_vram: Vec<u64>,
        gpu_name: Option<&str>,
        is_soc: bool,
    ) -> HardwareSurvey {
        HardwareSurvey {
            gpu_count,
            gpu_vram,
            gpu_name: gpu_name.map(str::to_owned),
            is_soc,
            ..Default::default()
        }
    }

    fn make_fingerprint(gpus: Vec<GpuBandwidth>, is_soc: bool) -> BenchmarkFingerprint {
        BenchmarkFingerprint {
            gpus,
            is_soc,
            timestamp_secs: 0,
        }
    }

    // 1. Same hardware → false
    #[test]
    fn test_hardware_changed_same() {
        let hw = make_survey(1, vec![80_000_000_000], Some("A100"), false);
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
            }],
            false,
        );
        assert!(!hardware_changed(&fp, &hw));
    }

    // 2. VRAM differs → true
    #[test]
    fn test_hardware_changed_vram() {
        let hw = make_survey(1, vec![40_000_000_000], Some("A100"), false);
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
            }],
            false,
        );
        assert!(hardware_changed(&fp, &hw));
    }

    // 3. GPU count differs → true
    #[test]
    fn test_hardware_changed_gpu_count() {
        let hw = make_survey(
            2,
            vec![80_000_000_000, 80_000_000_000],
            Some("A100, A100"),
            false,
        );
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
            }],
            false,
        );
        assert!(hardware_changed(&fp, &hw));
    }

    // 4. is_soc differs → true
    #[test]
    fn test_hardware_changed_soc_flag() {
        let hw = make_survey(1, vec![16_000_000_000], None, false);
        let fp = make_fingerprint(vec![], true); // is_soc: true vs false
        assert!(hardware_changed(&fp, &hw));
    }

    // 5. Parse single CUDA GPU JSON — assert p90_gbps == 1948.7
    #[test]
    fn test_benchmark_output_deserialize_cuda_single() {
        let json_str = r#"[{"device":"NVIDIA A100-SXM4-80GB","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json_str).expect("should parse");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].p90_gbps, 1948.7);
    }

    // 6. Parse 2-device JSON — assert both entries deserialize
    #[test]
    fn test_benchmark_output_deserialize_multi_gpu() {
        let json_str = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215},{"device":"NVIDIA A6000","buffer_mb":512,"runs":20,"p50_gbps":768.0,"p90_gbps":780.1,"noise_pct":0.6,"runtime_s":1.15,"rated_gbps":768,"rated_estimated":false,"efficiency_pct":100.0,"bus_width_bits":384,"mem_clock_mhz":2000}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json_str).expect("should parse");
        assert_eq!(outputs.len(), 2);
    }

    // 7. Error JSON (object, not array) → Err, no panic
    #[test]
    fn test_benchmark_output_deserialize_error_json() {
        let json_str = r#"{"error":"No CUDA-capable device found"}"#;
        let result = serde_json::from_str::<Vec<BenchmarkOutput>>(json_str);
        assert!(result.is_err(), "expected Err, got Ok");
    }

    // 8. parse_benchmark_output: single GPU → Some(vec with 1 entry, p90 == 1948.7)
    #[test]
    fn test_parse_benchmark_output_single_gpu() {
        let json = r#"[{"device":"NVIDIA A100-SXM4-80GB","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let result = parse_benchmark_output(json.as_bytes()).expect("should return Some");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].p90_gbps, 1948.7);
    }

    // 9. parse_benchmark_output: two GPUs → Some(vec with 2 entries), sum ~2728.8
    #[test]
    fn test_parse_benchmark_output_multi_gpu_sum() {
        let json = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215},{"device":"NVIDIA A6000","buffer_mb":512,"runs":20,"p50_gbps":768.0,"p90_gbps":780.1,"noise_pct":0.6,"runtime_s":1.15,"rated_gbps":768,"rated_estimated":false,"efficiency_pct":100.0,"bus_width_bits":384,"mem_clock_mhz":2000}]"#;
        let outputs = parse_benchmark_output(json.as_bytes()).expect("should return Some");
        assert_eq!(outputs.len(), 2);
        let sum: f64 = outputs.iter().map(|o| o.p90_gbps).sum();
        assert!(
            (sum - 2728.8_f64).abs() < 0.01,
            "expected ~2728.8, got {sum}"
        );
    }

    // 10. parse_benchmark_output: error object → None
    #[test]
    fn test_parse_benchmark_output_error_json() {
        let json = r#"{"error": "No CUDA devices found"}"#;
        let result = parse_benchmark_output(json.as_bytes());
        assert!(result.is_none());
    }

    // 11. parse_benchmark_output: empty array → None
    #[test]
    fn test_parse_benchmark_output_empty_array() {
        let result = parse_benchmark_output(b"[]");
        assert!(result.is_none());
    }

    // 12. detect_benchmark_binary: gpu_count == 0 → None (no process spawned)
    #[test]
    fn test_detect_benchmark_binary_gpu_count_zero() {
        let hw = HardwareSurvey {
            gpu_count: 0,
            ..Default::default()
        };
        let result = detect_benchmark_binary(&hw, Path::new("/tmp"));
        assert!(result.is_none());
    }

    // 13. hardware_changed: same VRAM, different GPU name → true
    #[test]
    fn test_hardware_changed_gpu_name() {
        let hw = make_survey(1, vec![80_000_000_000], Some("NVIDIA A6000"), false);
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "NVIDIA A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.0,
                p90_gbps: 1948.7,
            }],
            false,
        );
        assert!(
            hardware_changed(&fp, &hw),
            "name change should trigger hardware_changed"
        );
    }

    // 14. Cache round-trip: save → load → hardware_changed returns false for same hw
    #[test]
    fn test_fingerprint_cache_roundtrip() {
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-roundtrip.json");
        let fp = make_fingerprint(
            vec![GpuBandwidth {
                name: "NVIDIA A100".into(),
                vram_bytes: 80_000_000_000,
                p50_gbps: 1935.2,
                p90_gbps: 1948.7,
            }],
            false,
        );
        save_fingerprint(&path, &fp);
        let loaded = load_fingerprint(&path).expect("fingerprint should round-trip");
        let _ = std::fs::remove_file(&path);

        let hw = make_survey(1, vec![80_000_000_000], Some("NVIDIA A100"), false);
        assert!(
            !hardware_changed(&loaded, &hw),
            "same hardware should not trigger hardware_changed after round-trip"
        );
    }

    // 15. Old cache format (hardware_key field) fails to parse → load_fingerprint returns None
    #[test]
    fn test_old_cache_format_fails_parse() {
        let old_json = r#"{
            "hardware_key": {
                "gpu_count": 1,
                "gpu_vram": [80000000000],
                "gpu_name": "NVIDIA A100",
                "is_soc": false
            },
            "mem_bandwidth_gbps": 1948.7,
            "p50_gbps": 1935.2,
            "timestamp_secs": 1700000000
        }"#;
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-old-format.json");
        std::fs::write(&path, old_json).expect("write should succeed");
        let result = load_fingerprint(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_none(),
            "old cache format should fail to parse and return None"
        );
    }
}
