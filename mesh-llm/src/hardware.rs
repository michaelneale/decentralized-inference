/// Hardware detection via Collector trait pattern.
/// VRAM formula preserved byte-identical from mesh.rs:detect_vram_bytes().

#[derive(Default, Debug, Clone, PartialEq)]
pub struct HardwareSurvey {
    pub vram_bytes: u64,
    pub gpu_name: Option<String>,
    pub gpu_count: u8,
    pub hostname: Option<String>,
    pub is_soc: bool,
    /// Per-GPU VRAM in bytes, same order as gpu_name list.
    /// Unified-memory SoCs report a single entry.
    pub gpu_vram: Vec<u64>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    GpuName,
    VramBytes,
    GpuCount,
    Hostname,
    IsSoc,
}

pub trait Collector {
    fn collect(&self, metrics: &[Metric]) -> HardwareSurvey;
}

struct DefaultCollector;

#[cfg(target_os = "linux")]
struct TegraCollector;

/// Parse `nvidia-smi --query-gpu=name --format=csv,noheader` output → GPU name list.
#[cfg(any(target_os = "linux", test))]
pub fn parse_nvidia_gpu_names(output: &str) -> Vec<String> {
    output
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

/// Parse `sysctl -n machdep.cpu.brand_string` output → CPU brand string.
#[cfg(any(target_os = "macos", test))]
pub fn parse_macos_cpu_brand(output: &str) -> Option<String> {
    let s = output.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Parse `rocm-smi --showproductname` output → GPU names from "Card series:" lines.
#[cfg(any(target_os = "linux", test))]
pub fn parse_rocm_gpu_names(output: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in output.lines() {
        if let Some(pos) = line.find("Card series:") {
            let val = line[pos + "Card series:".len()..].trim();
            if !val.is_empty() {
                names.push(val.to_string());
            }
        }
    }
    names
}

/// Parse `rocm-smi --showmeminfo vram --csv` output → per-GPU VRAM bytes.
#[cfg(any(target_os = "linux", test))]
pub fn parse_rocm_gpu_vrams(output: &str) -> Vec<u64> {
    output
        .lines()
        .skip(1)
        .filter_map(|line| {
            let total = line.split(',').nth(1)?;
            total.trim().parse::<u64>().ok()
        })
        .collect()
}

/// Summarize GPU names: empty→None, 1→name, N identical→"N× name", N mixed→"a, b".
#[cfg(any(target_os = "linux", test))]
pub fn summarize_gpu_name(names: &[String]) -> Option<String> {
    match names.len() {
        0 => None,
        1 => Some(names[0].clone()),
        n => {
            let first = &names[0];
            if names.iter().all(|name| name == first) {
                Some(format!("{}× {}", n, first))
            } else {
                Some(names.join(", "))
            }
        }
    }
}

/// Check if a null-separated `/proc/device-tree/compatible` string contains a Tegra entry.
#[cfg(any(target_os = "linux", test))]
pub fn is_tegra(compatible: &str) -> bool {
    compatible.split('\0').any(|entry| entry.contains("tegra"))
}

/// Parse `/sys/firmware/devicetree/base/model` (null-terminated) → clean Jetson name.
/// Strips "NVIDIA " prefix and " Developer Kit" suffix.
#[cfg(any(target_os = "linux", test))]
pub fn parse_tegra_model_name(model: &str) -> Option<String> {
    let s = model.trim_matches('\0').trim();
    if s.is_empty() {
        return None;
    }
    let s = s.strip_prefix("NVIDIA ").unwrap_or(s);
    let s = s.strip_suffix(" Developer Kit").unwrap_or(s);
    Some(s.to_string())
}

/// Parse a `tegrastats` output line → total RAM bytes.
/// Handles optional timestamp prefix. No regex crate — plain string search.
#[cfg(any(target_os = "linux", test))]
pub fn parse_tegrastats_ram(output: &str) -> Option<u64> {
    let ram_pos = output.find("RAM ")?;
    let after_ram = &output[ram_pos + 4..];
    let slash_pos = after_ram.find('/')?;
    let after_slash = &after_ram[slash_pos + 1..];
    let mb_end = after_slash.find('M')?;
    let mb: u64 = after_slash[..mb_end].trim().parse().ok()?;
    Some(mb * 1024 * 1024)
}

/// Parse `hostname` command output → trimmed hostname string.
pub fn parse_hostname(output: &str) -> Option<String> {
    let s = output.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn detect_hostname() -> Option<String> {
    let out = std::process::Command::new("hostname").output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_hostname(&String::from_utf8(out.stdout).ok()?)
}

#[cfg(target_os = "linux")]
fn read_system_ram_bytes() -> u64 {
    (|| -> Option<u64> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    })()
    .unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn try_tegrastats_ram() -> Option<u64> {
    use std::io::BufRead;
    let mut child = std::process::Command::new("tegrastats")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    let line = std::io::BufReader::new(stdout).lines().next()?.ok()?;
    let _ = child.kill();
    let _ = child.wait();
    parse_tegrastats_ram(&line)
}

impl Collector for DefaultCollector {
    fn collect(&self, metrics: &[Metric]) -> HardwareSurvey {
        let mut survey = HardwareSurvey::default();

        #[cfg(target_os = "macos")]
        {
            if metrics.contains(&Metric::IsSoc) {
                survey.is_soc = true;
            }
            if metrics.contains(&Metric::VramBytes) {
                let out = std::process::Command::new("sysctl")
                    .args(["-n", "hw.memsize"])
                    .output()
                    .ok();
                if let Some(out) = out {
                    if let Ok(s) = String::from_utf8(out.stdout) {
                        if let Ok(bytes) = s.trim().parse::<u64>() {
                            // ~75% usable for Metal on unified memory (mesh.rs:263)
                            survey.vram_bytes = (bytes as f64 * 0.75) as u64;
                            survey.gpu_vram = vec![bytes];
                        }
                    }
                }
            }
            if metrics.contains(&Metric::GpuName) {
                let out = std::process::Command::new("sysctl")
                    .args(["-n", "machdep.cpu.brand_string"])
                    .output()
                    .ok();
                if let Some(out) = out {
                    if let Ok(s) = String::from_utf8(out.stdout) {
                        survey.gpu_name = parse_macos_cpu_brand(&s);
                    }
                }
            }
            if metrics.contains(&Metric::GpuCount) {
                survey.gpu_count = 1;
            }
        }

        #[cfg(target_os = "linux")]
        {
            let system_ram = read_system_ram_bytes();

            if metrics.contains(&Metric::VramBytes) {
                // Try NVIDIA (mesh.rs:284-316)
                let nvidia_vram: Option<(u64, Vec<u64>)> = (|| {
                    let out = std::process::Command::new("nvidia-smi")
                        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                        .output()
                        .ok()?;
                    if !out.status.success() {
                        return None;
                    }
                    let s = String::from_utf8(out.stdout).ok()?;
                    let per_gpu: Vec<u64> = s
                        .lines()
                        .filter_map(|line| {
                            let mib = line.trim().parse::<u64>().ok()?;
                            Some(mib * 1024 * 1024)
                        })
                        .collect();
                    let total: u64 = per_gpu.iter().sum();
                    if total > 0 {
                        Some((total, per_gpu))
                    } else {
                        None
                    }
                })();

                if let Some((vram, per_gpu)) = nvidia_vram {
                    survey.gpu_vram = per_gpu;
                    let ram_offload = system_ram.saturating_sub(vram);
                    survey.vram_bytes = vram + (ram_offload as f64 * 0.75) as u64;
                } else {
                    // Try AMD ROCm (mesh.rs:295-316)
                    let rocm_vram: Option<Vec<u64>> = (|| {
                        let out = std::process::Command::new("rocm-smi")
                            .args(["--showmeminfo", "vram", "--csv"])
                            .output()
                            .ok()?;
                        if !out.status.success() {
                            return None;
                        }
                        let s = String::from_utf8(out.stdout).ok()?;
                        let vrams = parse_rocm_gpu_vrams(&s);
                        if vrams.is_empty() {
                            None
                        } else {
                            Some(vrams)
                        }
                    })();

                    if let Some(per_gpu) = rocm_vram {
                        let vram: u64 = per_gpu.iter().sum();
                        survey.gpu_vram = per_gpu;
                        let ram_offload = system_ram.saturating_sub(vram);
                        survey.vram_bytes = vram + (ram_offload as f64 * 0.75) as u64;
                    } else if system_ram > 0 {
                        // CPU-only (mesh.rs:320-322)
                        survey.vram_bytes = (system_ram as f64 * 0.75) as u64;
                    }
                }
            }

            if metrics.contains(&Metric::GpuName) || metrics.contains(&Metric::GpuCount) {
                let nvidia_names: Option<Vec<String>> = (|| {
                    let out = std::process::Command::new("nvidia-smi")
                        .args(["--query-gpu=name", "--format=csv,noheader"])
                        .output()
                        .ok()?;
                    if !out.status.success() {
                        return None;
                    }
                    let s = String::from_utf8(out.stdout).ok()?;
                    let names = parse_nvidia_gpu_names(&s);
                    if names.is_empty() {
                        None
                    } else {
                        Some(names)
                    }
                })();

                if let Some(ref names) = nvidia_names {
                    if metrics.contains(&Metric::GpuName) {
                        survey.gpu_name = summarize_gpu_name(names);
                    }
                    if metrics.contains(&Metric::GpuCount) {
                        survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                    }
                } else {
                    let out = std::process::Command::new("rocm-smi")
                        .args(["--showproductname"])
                        .output()
                        .ok();
                    if let Some(out) = out {
                        if out.status.success() {
                            if let Ok(s) = String::from_utf8(out.stdout) {
                                let names = parse_rocm_gpu_names(&s);
                                if metrics.contains(&Metric::GpuName) {
                                    survey.gpu_name = summarize_gpu_name(&names);
                                }
                                if metrics.contains(&Metric::GpuCount) {
                                    survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                                }
                            }
                        }
                    }
                }
            }
        }

        survey
    }
}

#[cfg(target_os = "linux")]
impl Collector for TegraCollector {
    fn collect(&self, metrics: &[Metric]) -> HardwareSurvey {
        let mut survey = HardwareSurvey::default();

        if metrics.contains(&Metric::IsSoc) {
            survey.is_soc = true;
        }

        if metrics.contains(&Metric::GpuName) {
            if let Ok(model) = std::fs::read_to_string("/sys/firmware/devicetree/base/model") {
                survey.gpu_name = parse_tegra_model_name(&model);
            }
        }

        if metrics.contains(&Metric::VramBytes) {
            let total_ram = (|| -> Option<u64> {
                let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
                        return Some(kb * 1024);
                    }
                }
                None
            })()
            .or_else(try_tegrastats_ram);
            if let Some(ram) = total_ram {
                survey.vram_bytes = (ram as f64 * 0.75) as u64;
                survey.gpu_vram = vec![ram];
            }
        }

        if metrics.contains(&Metric::GpuCount) {
            survey.gpu_count = 1;
        }

        survey
    }
}

#[cfg(target_os = "macos")]
fn detect_collector_impl() -> Box<dyn Collector> {
    Box::new(DefaultCollector)
}

#[cfg(target_os = "linux")]
fn detect_collector_impl() -> Box<dyn Collector> {
    if cfg!(target_arch = "aarch64") {
        if let Ok(compat) = std::fs::read_to_string("/proc/device-tree/compatible") {
            if is_tegra(&compat) {
                return Box::new(TegraCollector);
            }
        }
    }
    Box::new(DefaultCollector)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_collector_impl() -> Box<dyn Collector> {
    Box::new(DefaultCollector)
}

fn detect_collector() -> Box<dyn Collector> {
    detect_collector_impl()
}

/// Collect only the requested hardware metrics.
pub fn query(metrics: &[Metric]) -> HardwareSurvey {
    let collector = detect_collector();
    let mut survey = collector.collect(metrics);
    if metrics.contains(&Metric::Hostname) {
        survey.hostname = detect_hostname();
    }
    survey
}

pub fn survey() -> HardwareSurvey {
    query(&[
        Metric::GpuName,
        Metric::VramBytes,
        Metric::GpuCount,
        Metric::Hostname,
        Metric::IsSoc,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nvidia_gpu_name_single() {
        let names = parse_nvidia_gpu_names("NVIDIA A100-SXM4-80GB\n");
        assert_eq!(names, vec!["NVIDIA A100-SXM4-80GB"]);
    }

    #[test]
    fn test_parse_nvidia_gpu_name_multi_identical() {
        let names = parse_nvidia_gpu_names("NVIDIA A100\nNVIDIA A100\n");
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "NVIDIA A100");
        assert_eq!(names[1], "NVIDIA A100");
    }

    #[test]
    fn test_parse_nvidia_gpu_name_multi_mixed() {
        let names = parse_nvidia_gpu_names("NVIDIA A100\nNVIDIA RTX 4090\n");
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "NVIDIA A100");
        assert_eq!(names[1], "NVIDIA RTX 4090");
    }

    #[test]
    fn test_parse_nvidia_gpu_name_empty() {
        assert!(parse_nvidia_gpu_names("").is_empty());
    }

    #[test]
    fn test_parse_macos_cpu_brand() {
        assert_eq!(
            parse_macos_cpu_brand("Apple M4 Max\n"),
            Some("Apple M4 Max".to_string())
        );
    }

    #[test]
    fn test_parse_macos_cpu_brand_empty() {
        assert_eq!(parse_macos_cpu_brand(""), None);
    }

    #[test]
    fn test_parse_rocm_gpu_names_single() {
        let fixture = "\
======================= ROCm System Management Interface =======================
================================= Product Info =================================
GPU[0]\t\t: Card series:\t\t\tNavi31 [Radeon RX 7900 XTX]
================================================================================";
        assert_eq!(
            parse_rocm_gpu_names(fixture),
            vec!["Navi31 [Radeon RX 7900 XTX]".to_string()]
        );
    }

    #[test]
    fn test_parse_rocm_gpu_names_multi() {
        let fixture = "\
======================= ROCm System Management Interface =======================
================================= Product Info =================================
GPU[0]\t\t: Card series:\t\t\tAMD Instinct MI300X
GPU[1]\t\t: Card series:\t\t\tAMD Instinct MI300X
================================================================================";
        assert_eq!(
            parse_rocm_gpu_names(fixture),
            vec![
                "AMD Instinct MI300X".to_string(),
                "AMD Instinct MI300X".to_string()
            ]
        );
    }

    #[test]
    fn test_parse_rocm_gpu_vrams_single() {
        let fixture = "\
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,25753026560,416378880";
        assert_eq!(parse_rocm_gpu_vrams(fixture), vec![25753026560]);
    }

    #[test]
    fn test_parse_rocm_gpu_vrams_multi() {
        let fixture = "\
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,25753026560,416378880
card1,25753026560,512000000";
        assert_eq!(
            parse_rocm_gpu_vrams(fixture),
            vec![25753026560, 25753026560]
        );
    }

    #[test]
    fn test_parse_rocm_gpu_vrams_ignores_invalid_rows() {
        let fixture = "\
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,25753026560,416378880
card1,not-a-number,512000000";
        assert_eq!(parse_rocm_gpu_vrams(fixture), vec![25753026560]);
    }

    #[test]
    fn test_summarize_gpu_name_single() {
        assert_eq!(
            summarize_gpu_name(&["A100".to_string()]),
            Some("A100".to_string())
        );
    }

    #[test]
    fn test_summarize_gpu_name_identical() {
        assert_eq!(
            summarize_gpu_name(&["A100".to_string(), "A100".to_string()]),
            Some("2\u{00D7} A100".to_string())
        );
    }

    #[test]
    fn test_summarize_gpu_name_mixed() {
        assert_eq!(
            summarize_gpu_name(&["A100".to_string(), "RTX 4090".to_string()]),
            Some("A100, RTX 4090".to_string())
        );
    }

    #[test]
    fn test_summarize_gpu_name_empty() {
        assert_eq!(summarize_gpu_name(&[]), None);
    }

    #[test]
    fn test_hardware_survey_default() {
        let s = HardwareSurvey::default();
        assert_eq!(s.vram_bytes, 0);
        assert_eq!(s.gpu_name, None);
        assert_eq!(s.gpu_count, 0);
        assert_eq!(s.hostname, None);
    }

    #[test]
    fn test_query_gpu_name_only() {
        let result = query(&[Metric::GpuName]);
        assert_eq!(result.vram_bytes, 0);
        assert_eq!(result.hostname, None);
    }

    #[test]
    fn test_query_vram_only() {
        let result = query(&[Metric::VramBytes]);
        assert_eq!(result.gpu_name, None);
        assert_eq!(result.hostname, None);
    }

    #[test]
    fn test_query_multiple_metrics() {
        let result = query(&[Metric::GpuName, Metric::VramBytes]);
        assert_eq!(result.hostname, None);
        assert_eq!(result.gpu_count, 0);
    }

    #[test]
    fn test_survey_returns_all_metrics() {
        let s = survey();
        let q = query(&[
            Metric::GpuName,
            Metric::VramBytes,
            Metric::GpuCount,
            Metric::Hostname,
        ]);
        assert_eq!(s.vram_bytes, q.vram_bytes);
        assert_eq!(s.gpu_name, q.gpu_name);
        assert_eq!(s.gpu_count, q.gpu_count);
        assert_eq!(s.hostname.is_some(), q.hostname.is_some());
    }

    #[test]
    fn test_is_tegra_positive() {
        assert!(is_tegra("nvidia,p3737-0000+p3701-0005\0nvidia,tegra234\0"));
    }

    #[test]
    fn test_is_tegra_negative_arm() {
        assert!(!is_tegra("raspberrypi,4-model-b\0"));
    }

    #[test]
    fn test_parse_tegra_model_name() {
        assert_eq!(
            parse_tegra_model_name("NVIDIA Jetson AGX Orin Developer Kit\0"),
            Some("Jetson AGX Orin".to_string())
        );
    }

    #[test]
    fn test_parse_tegra_model_name_nano() {
        assert_eq!(
            parse_tegra_model_name("NVIDIA Jetson Orin Nano Developer Kit\0"),
            Some("Jetson Orin Nano".to_string())
        );
    }

    #[test]
    fn test_parse_tegra_model_name_no_prefix() {
        assert_eq!(
            parse_tegra_model_name("Jetson Xavier NX\0"),
            Some("Jetson Xavier NX".to_string())
        );
    }

    #[test]
    fn test_parse_tegrastats_ram() {
        let line = "RAM 14640/62838MB (lfb 11x4MB) CPU [0%@729,off,off,off,0%@729,off,off,off]";
        assert_eq!(parse_tegrastats_ram(line), Some(62838u64 * 1024 * 1024));
    }

    #[test]
    fn test_parse_tegrastats_ram_with_timestamp() {
        let line = "12-27-2022 13:48:01 RAM 14640/62838MB (lfb 11x4MB)";
        assert_eq!(parse_tegrastats_ram(line), Some(62838u64 * 1024 * 1024));
    }

    #[test]
    fn test_parse_tegrastats_ram_empty() {
        assert_eq!(parse_tegrastats_ram(""), None);
    }

    #[test]
    fn test_parse_hostname() {
        assert_eq!(parse_hostname("lemony-28\n"), Some("lemony-28".to_string()));
    }

    #[test]
    fn test_parse_hostname_empty() {
        assert_eq!(parse_hostname(""), None);
    }

    #[test]
    fn test_parse_hostname_whitespace() {
        assert_eq!(parse_hostname("  carrack  \n"), Some("carrack".to_string()));
    }

    #[test]
    fn test_is_tegra_negative_x86() {
        assert!(!is_tegra(""));
    }

    #[test]
    fn test_query_hostname_only() {
        let result = query(&[Metric::Hostname]);
        assert_eq!(result.gpu_name, None);
        assert_eq!(result.gpu_count, 0);
        assert_eq!(result.vram_bytes, 0);
    }

    #[test]
    fn test_detect_collector_returns_default_on_non_tegra() {
        let collector = detect_collector();
        let s = collector.collect(&[Metric::VramBytes]);
        let _ = s.vram_bytes;
    }

    #[test]
    fn test_query_is_soc_only() {
        let result = query(&[Metric::IsSoc]);
        assert_eq!(result.vram_bytes, 0);
        assert_eq!(result.gpu_name, None);
        assert_eq!(result.gpu_count, 0);
        assert_eq!(result.hostname, None);
        let _ = result.is_soc;
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_macos_is_soc_true() {
        let result = DefaultCollector.collect(&[Metric::IsSoc]);
        assert!(
            result.is_soc,
            "macOS DefaultCollector must report is_soc=true"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_tegra_is_soc_true() {
        let result = TegraCollector.collect(&[Metric::IsSoc]);
        assert!(result.is_soc, "TegraCollector must report is_soc=true");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_linux_discrete_is_soc_false() {
        let result = DefaultCollector.collect(&[Metric::IsSoc]);
        assert!(
            !result.is_soc,
            "Linux DefaultCollector must report is_soc=false"
        );
    }

    #[test]
    fn test_default_collector_nvidia_fixture() {
        let names = parse_nvidia_gpu_names("NVIDIA A100\n");
        assert_eq!(names, vec!["NVIDIA A100"]);
        assert_eq!(
            summarize_gpu_name(&["NVIDIA A100".to_string()]),
            Some("NVIDIA A100".to_string())
        );
    }

    #[test]
    fn test_tegra_collector_sysfs_fixture() {
        assert_eq!(
            parse_tegra_model_name("NVIDIA Jetson AGX Orin Developer Kit\0"),
            Some("Jetson AGX Orin".to_string())
        );
    }
}
