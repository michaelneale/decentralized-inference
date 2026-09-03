/// Parse `nvidia-smi --query-gpu=name --format=csv,noheader` output → GPU name list.
#[cfg(any(target_os = "linux", target_os = "windows", test))]
pub fn parse_nvidia_gpu_names(output: &str) -> Vec<String> {
    output
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect()
}

/// Parse `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` → per-GPU VRAM bytes.
#[cfg(any(target_os = "windows", test))]
pub fn parse_nvidia_gpu_memory(output: &str) -> Vec<u64> {
    output
        .lines()
        .filter_map(|line| {
            let mib = line.trim().parse::<u64>().ok()?;
            Some(mib * 1024 * 1024)
        })
        .collect()
}

#[cfg(any(target_os = "linux", target_os = "windows", test))]
pub fn parse_nvidia_gpu_memory_and_reserved(output: &str) -> Vec<(u64, Option<u64>)> {
    output
        .lines()
        .filter_map(|line| {
            let mut parts = line.split(',').map(str::trim);
            let total_mib = parts.next()?.parse::<u64>().ok()?;
            let reserved_mib = parts.next().and_then(|value| value.parse::<u64>().ok());
            Some((
                total_mib * 1024 * 1024,
                reserved_mib.map(|mib| mib * 1024 * 1024),
            ))
        })
        .collect()
}

/// Sanitizes a macOS Metal `MTLDevice.name` query result into `gpu_name`.
/// Rejects anything shaped like a CPU brand string (`sysctl
/// machdep.cpu.brand_string` output always embeds " CPU @ " and a GHz clock
/// speed on Intel Macs) so a regression that re-wires CPU output into this
/// path degrades to labeled-absent instead of silently mislabeling the CPU
/// as the GPU.
#[cfg(any(target_os = "macos", test))]
pub fn sanitize_macos_gpu_name(name: Option<String>) -> Option<String> {
    let trimmed = name?.trim().to_string();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.contains("cpu") || lower.contains("ghz") {
        return None;
    }
    Some(trimmed)
}

#[cfg(any(target_os = "macos", test))]
pub(super) fn macos_metal_gpu_budget(
    metal_recommended_bytes: Option<u64>,
) -> Option<(u64, Option<u64>)> {
    metal_recommended_bytes
        .filter(|bytes| *bytes > 0)
        .map(|bytes| (bytes, None))
}

/// Parse `rocm-smi --showproductname` output → GPU names from "Card series:" lines.
#[cfg(any(target_os = "linux", test))]
pub fn parse_rocm_gpu_names(output: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in output.lines() {
        let lower = line.to_ascii_lowercase();
        if let Some(pos) = lower.find("card series:") {
            let val = line[pos + "card series:".len()..].trim();
            if !val.is_empty() {
                names.push(val.to_string());
            }
        }
    }
    names
}

/// Parse `rocm-smi --showmeminfo vram --csv` output into per-GPU total bytes
/// and live used bytes. The used column is a utilization metric, not a
/// reserved/system-memory metric, so callers must not surface it as
/// `reserved_bytes`.
#[cfg(any(target_os = "linux", test))]
pub fn parse_rocm_gpu_memory_and_used(output: &str) -> Vec<(u64, Option<u64>)> {
    let mut rows = output.lines();
    let _header = rows.find(|line| {
        let lower = line.to_ascii_lowercase();
        lower.contains("total") && lower.contains("memory")
    });

    rows.filter_map(|line| {
        let mut columns = line.split(',').map(str::trim);
        let _device = columns.next()?;
        let total = columns.next()?.parse::<u64>().ok()?;
        let used = columns.next().and_then(|value| value.parse::<u64>().ok());
        Some((total, used))
    })
    .collect()
}

#[cfg(all(
    any(target_os = "linux", test),
    any(
        not(feature = "skippy-devices"),
        feature = "dynamic-native-runtime",
        test
    )
))]
const ROCM_UNIFIED_VRAM_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
#[cfg(all(
    any(target_os = "linux", test),
    any(
        not(feature = "skippy-devices"),
        feature = "dynamic-native-runtime",
        test
    )
))]
const ROCM_UNIFIED_MIN_GTT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[cfg(all(
    any(target_os = "linux", test),
    any(
        not(feature = "skippy-devices"),
        feature = "dynamic-native-runtime",
        test
    )
))]
pub(super) fn rocm_unified_memory_usable_bytes(
    vram_totals: &[u64],
    gtt_totals: &[u64],
    system_ram: u64,
) -> Option<u64> {
    if vram_totals.len() != 1 || gtt_totals.len() != 1 {
        return None;
    }
    let vram = vram_totals[0];
    let gtt = gtt_totals[0];
    if vram == 0
        || vram > ROCM_UNIFIED_VRAM_MAX_BYTES
        || gtt < ROCM_UNIFIED_MIN_GTT_BYTES
        || gtt < vram.saturating_mul(8)
    {
        return None;
    }

    let unified_total = if system_ram > 0 {
        gtt.min(system_ram)
    } else {
        gtt
    };
    Some((unified_total as f64 * 0.90) as u64)
}

#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XpuSmiGpuInfo {
    pub name: String,
    pub total_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
}

#[cfg(any(target_os = "linux", test))]
fn xpu_json_string(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match map.get(*key) {
        Some(Value::String(value)) if !value.trim().is_empty() => Some(value.trim().to_string()),
        _ => None,
    })
}

#[cfg(any(target_os = "linux", test))]
fn xpu_json_u64(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| match map.get(*key) {
        Some(Value::Number(value)) => value.as_u64(),
        Some(Value::String(value)) => value.trim().parse::<u64>().ok(),
        _ => None,
    })
}

#[cfg(any(target_os = "linux", test))]
fn collect_xpu_smi_devices(value: &Value, devices: &mut Vec<XpuSmiGpuInfo>) {
    match value {
        Value::Object(map) => {
            let name = xpu_json_string(map, &["device_name", "deviceName", "name"]);
            let total_bytes = xpu_json_u64(
                map,
                &[
                    "memory_physical_size_byte",
                    "memoryPhysicalSizeByte",
                    "memory_total_bytes",
                    "memoryTotalBytes",
                    "memory_size_byte",
                    "memorySizeByte",
                    "lmem_total_bytes",
                    "lmemTotalBytes",
                ],
            );
            let used_bytes = xpu_json_u64(
                map,
                &[
                    "memory_used_byte",
                    "memoryUsedByte",
                    "memory_used_bytes",
                    "memoryUsedBytes",
                    "lmem_used_bytes",
                    "lmemUsedBytes",
                ],
            );
            if let Some(name) = name.filter(|_| total_bytes.is_some() || used_bytes.is_some()) {
                devices.push(XpuSmiGpuInfo {
                    name,
                    total_bytes,
                    used_bytes,
                });
            }
            for child in map.values() {
                collect_xpu_smi_devices(child, devices);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_xpu_smi_devices(value, devices);
            }
        }
        _ => {}
    }
}

#[cfg(any(target_os = "linux", test))]
pub fn parse_xpu_smi_discovery_json(output: &str) -> Vec<XpuSmiGpuInfo> {
    let Ok(value) = serde_json::from_str::<Value>(output) else {
        return Vec::new();
    };
    let mut devices = Vec::new();
    collect_xpu_smi_devices(&value, &mut devices);
    devices
}

/// Summarize GPU names: empty→None, 1→name, N identical→"N× name", N mixed→"a, b".
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

/// Expand a summarized GPU name string into per-device names.
/// - Splits comma-separated mixed GPU names.
/// - Expands summarized forms like `2× NVIDIA A100`.
/// - Falls back to repeating the raw summary to match `expected_count`.
pub fn expand_gpu_names(summary: Option<&str>, expected_count: usize) -> Vec<String> {
    let Some(raw) = summary.map(str::trim) else {
        return Vec::new();
    };
    if raw.is_empty() {
        return Vec::new();
    }

    let mut names = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let counted_name = part.split_once('×').and_then(|(count_str, name)| {
            let name = name.trim();
            if name.is_empty() {
                return None;
            }
            count_str
                .trim()
                .parse::<usize>()
                .ok()
                .map(|count| (count, name))
        });
        if let Some((count, name)) = counted_name {
            for _ in 0..count {
                names.push(name.to_string());
            }
            continue;
        }
        names.push(part.to_string());
    }

    if expected_count > 0 && names.len() != expected_count {
        return vec![raw.to_string(); expected_count];
    }
    names
}

#[cfg(any(target_os = "linux", target_os = "windows", test))]
pub fn parse_nvidia_gpu_identity(output: &str) -> Vec<(Option<String>, Option<String>)> {
    fn normalize_identity_field(part: &str) -> Option<&str> {
        let part = part.trim();
        if part.is_empty() || part.eq_ignore_ascii_case("n/a") || part == "[N/A]" {
            None
        } else {
            Some(part)
        }
    }

    output
        .lines()
        .map(|line| {
            let mut parts = line.split(',').map(str::trim);
            let pci_bdf = parts
                .next()
                .and_then(normalize_identity_field)
                .map(|part| part.to_ascii_lowercase());
            let vendor_uuid = parts
                .next()
                .and_then(normalize_identity_field)
                .map(str::to_string);
            (pci_bdf, vendor_uuid)
        })
        .collect()
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

/// Parse PowerShell `Win32_VideoController | ConvertTo-Json` output → `(name, adapter_ram_bytes)`.
#[cfg(any(target_os = "windows", test))]
pub fn parse_windows_video_controller_json(output: &str) -> Vec<(String, u64)> {
    fn parse_u64(value: &Value) -> Option<u64> {
        match value {
            Value::Number(n) => n.as_u64(),
            Value::String(s) => s.trim().parse::<u64>().ok(),
            _ => None,
        }
    }

    fn parse_entry(value: &Value) -> Option<(String, u64)> {
        let name = value.get("Name")?.as_str()?.trim();
        if name.is_empty() {
            return None;
        }
        let adapter_ram = value.get("AdapterRAM").and_then(parse_u64).unwrap_or(0);
        Some((name.to_string(), adapter_ram))
    }

    let Ok(value) = serde_json::from_str::<Value>(output) else {
        return Vec::new();
    };

    match value {
        Value::Array(values) => values.iter().filter_map(parse_entry).collect(),
        Value::Object(_) => parse_entry(&value).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Parse `TotalPhysicalMemory` output from PowerShell/CIM.
#[cfg(any(target_os = "windows", test))]
pub fn parse_windows_total_physical_memory(output: &str) -> Option<u64> {
    output.trim().parse::<u64>().ok()
}
fn parse_vulkan_gpu_header(line: &str) -> Option<usize> {
    let trimmed = line.trim();
    let suffix = trimmed.strip_prefix("GPU")?.strip_suffix(':')?;
    suffix.parse().ok()
}

/// Parse `vulkaninfo --summary` device sections.
pub fn parse_vulkaninfo_summary_devices(output: &str) -> Vec<VulkanGpuFacts> {
    let mut devices = Vec::new();
    let mut current: Option<VulkanGpuFacts> = None;

    for line in output.lines() {
        if let Some(index) = parse_vulkan_gpu_header(line) {
            if let Some(device) = current.take() {
                devices.push(device);
            }
            current = Some(VulkanGpuFacts {
                index,
                ..Default::default()
            });
            continue;
        }

        let Some(device) = current.as_mut() else {
            continue;
        };
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().to_string();
        match key {
            "vendorID" => device.vendor_id = Some(value),
            "deviceID" => device.device_id = Some(value),
            "deviceType" => device.device_type = value,
            "deviceName" => device.display_name = value,
            "deviceUUID" => device.device_uuid = Some(value),
            _ => {}
        }
    }

    if let Some(device) = current {
        devices.push(device);
    }
    devices
}
use super::VulkanGpuFacts;

#[cfg(any(target_os = "windows", target_os = "linux", test))]
use serde_json::Value;
