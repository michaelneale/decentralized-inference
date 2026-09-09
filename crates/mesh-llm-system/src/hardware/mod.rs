//! Hardware detection via Collector trait pattern.
//! VRAM formula preserved byte-identical from mesh.rs:detect_vram_bytes().

#[cfg(feature = "skippy-devices")]
mod enrichers;
mod parsers;
#[cfg(feature = "skippy-devices")]
mod skippy_devices;
#[cfg(test)]
mod tests;

#[cfg(any(target_os = "macos", test))]
use parsers::macos_metal_gpu_budget;
pub use parsers::*;

#[derive(Default, Debug, Clone, PartialEq, serde::Serialize)]
pub struct GpuFacts {
    pub index: usize,
    pub display_name: String,
    pub backend_device: Option<String>,
    pub vram_bytes: u64,
    pub reserved_bytes: Option<u64>,
    pub mem_bandwidth_gbps: Option<f64>,
    pub compute_tflops_fp32: Option<f64>,
    pub compute_tflops_fp16: Option<f64>,
    pub unified_memory: bool,
    pub stable_id: Option<String>,
    pub pci_bdf: Option<String>,
    pub vendor_uuid: Option<String>,
    pub metal_registry_id: Option<String>,
    pub dxgi_luid: Option<String>,
    pub pnp_instance_id: Option<String>,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct VulkanGpuFacts {
    pub index: usize,
    pub display_name: String,
    pub device_type: String,
    pub vendor_id: Option<String>,
    pub device_id: Option<String>,
    pub device_uuid: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinnedGpuResolverError {
    MissingConfiguredId {
        available_pinnable_ids: Vec<String>,
    },
    NonPinnableConfiguredId {
        configured_id: String,
        available_pinnable_ids: Vec<String>,
    },
    NoPinnableGpus {
        configured_id: String,
        available_pinnable_ids: Vec<String>,
    },
    NoMatch {
        configured_id: String,
        available_pinnable_ids: Vec<String>,
    },
    AmbiguousMatch {
        configured_id: String,
        available_pinnable_ids: Vec<String>,
        match_indexes: Vec<usize>,
    },
}

impl std::fmt::Display for PinnedGpuResolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingConfiguredId {
                available_pinnable_ids,
            } => write!(
                f,
                "missing configured gpu_id; available pinnable GPU IDs: {}",
                format_pinnable_gpu_ids(available_pinnable_ids)
            ),
            Self::NonPinnableConfiguredId {
                configured_id,
                available_pinnable_ids,
            } => write!(
                f,
                "configured gpu_id '{}' is not pinnable; available pinnable GPU IDs: {}",
                configured_id,
                format_pinnable_gpu_ids(available_pinnable_ids)
            ),
            Self::NoPinnableGpus {
                configured_id,
                available_pinnable_ids,
            } => write!(
                f,
                "configured gpu_id '{}' could not be resolved because this host has no pinnable GPUs; available pinnable GPU IDs: {}",
                configured_id,
                format_pinnable_gpu_ids(available_pinnable_ids)
            ),
            Self::NoMatch {
                configured_id,
                available_pinnable_ids,
            } => write!(
                f,
                "configured gpu_id '{}' did not match any available pinnable GPU; available pinnable GPU IDs: {}",
                configured_id,
                format_pinnable_gpu_ids(available_pinnable_ids)
            ),
            Self::AmbiguousMatch {
                configured_id,
                available_pinnable_ids,
                match_indexes,
            } => write!(
                f,
                "configured gpu_id '{}' matched multiple GPUs at indexes {:?}; available pinnable GPU IDs: {}",
                configured_id,
                match_indexes,
                format_pinnable_gpu_ids(available_pinnable_ids)
            ),
        }
    }
}

impl std::error::Error for PinnedGpuResolverError {}

#[derive(Default, Debug, Clone, PartialEq, serde::Serialize)]
pub struct HardwareSurvey {
    pub vram_bytes: u64,
    /// GPU name as reported by the OS/driver (e.g. Metal, nvidia-smi, ROCm).
    /// Best-effort and OS-reported, not an independently verified measurement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_name: Option<String>,
    /// Collection mechanism behind `gpu_name`. A source, not a verification —
    /// it names which probe produced the string, not that the string is a
    /// confirmed-accurate GPU identifier. `None` when no source is available
    /// for `gpu_name`. Note this is not the same as `gpu_name` being `None`:
    /// `hydrate_gpu_facts_with_identities` backfills placeholder `"GPU N"`
    /// names with no naming probe behind them, tagged `GpuNameSource::Unknown`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_name_source: Option<GpuNameSource>,
    pub gpu_count: u8,
    pub hostname: Option<String>,
    pub is_soc: bool,
    /// Per-GPU VRAM in bytes, same order as gpu_name list.
    /// Unified-memory SoCs report a single entry.
    pub gpu_vram: Vec<u64>,
    /// Per-GPU reserved or otherwise unavailable bytes when the platform
    /// reports a true reserved/unavailable value. Do not populate this from
    /// live used-memory counters.
    pub gpu_reserved: Vec<Option<u64>>,
    /// Per-GPU facts in device-enumeration order.
    pub gpus: Vec<GpuFacts>,
}

/// Where a `HardwareSurvey.gpu_name` value came from. Each variant names a
/// collection mechanism, not a claim that the resulting string is a verified
/// GPU identifier.
///
/// `HardwareSurvey` and `GpuFacts` are serialize-only (`serde::Serialize`).
/// This enum therefore derives only `Serialize` — a `Deserialize` impl would
/// be unreachable through any serializable struct and is omitted to avoid a
/// dead, untestable code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuNameSource {
    /// A Metal device name from `MTLDevice.name` (macOS). Assigned when the
    /// native-runtime backend reports a device whose backend name starts with
    /// `"MTL"`, or when `MTLCreateSystemDefaultDevice` is queried directly
    /// via the DefaultCollector macOS path. On switchable-graphics Macs the
    /// default device is a moment-in-time fact: it can differ between
    /// collections as the OS switches GPUs. Best-effort, OS-reported, not a
    /// verified GPU identifier.
    MetalDefaultDevice,
    /// macOS `sysctl -n machdep.cpu.brand_string`, used before upstream
    /// commit 6e16b84a2 (`fix(system): report the real macOS GPU name`).
    /// **Not assigned by new collections.** Kept in the vocabulary so that
    /// consumers reading surveys recorded before 6e16b84a2 can deserialise
    /// the field; it will never appear in a freshly collected survey.
    /// (`HardwareSurvey` has no `Deserialize` impl today, so this variant
    /// is currently write-only; it is preserved for when a `Deserialize` impl
    /// is added rather than forcing a breaking vocabulary change at that point.)
    CpuBrandString,
    /// The skippy native-runtime's backend device enumeration reporting a
    /// non-Metal accelerator (CUDA, ROCm, Vulkan, SYCL, ...). Names whichever
    /// accelerator the loaded runtime enumerated; does not by itself say
    /// which of those backends answered.
    NativeRuntimeDevice,
    /// `nvidia-smi --query-gpu=name` output.
    NvidiaSmi,
    /// `rocm-smi --showproductname` output.
    RocmSmi,
    /// `xpu-smi discovery` JSON output (Intel GPUs).
    XpuSmi,
    /// Windows `Win32_VideoController` CIM/WMI query (`Name` field).
    WindowsVideoController,
    /// A device-tree model string read from sysfs
    /// (`/sys/firmware/devicetree/base/model`), used on Tegra/Jetson boards.
    Sysfs,
    /// A name is present but was not produced by any naming probe above —
    /// e.g. a placeholder ("GPU 0") backfilled from GPU count/VRAM data
    /// alone. Never treat this as identifying real hardware.
    Unknown,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    GpuName,
    VramBytes,
    GpuCount,
    Hostname,
    IsSoc,
    GpuFacts,
}

pub trait Collector {
    fn collect(&self, metrics: &[Metric]) -> HardwareSurvey;
}

struct DefaultCollector;

#[cfg(all(
    target_os = "linux",
    any(
        not(feature = "skippy-devices"),
        feature = "dynamic-native-runtime",
        test
    )
))]
struct TegraCollector;

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

#[cfg(target_os = "windows")]
fn read_system_ram_bytes() -> u64 {
    read_windows_total_ram_bytes().unwrap_or(0)
}

#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    any(feature = "skippy-devices", test)
))]
fn apply_cpu_only_runtime_budget(
    survey: &mut HardwareSurvey,
    metrics: &[Metric],
    system_ram: impl FnOnce() -> u64,
) {
    if !metrics.contains(&Metric::VramBytes) {
        return;
    }
    let system_ram = system_ram();
    if system_ram > 0 {
        survey.vram_bytes = (system_ram as f64 * 0.75) as u64;
    }
}

#[cfg(all(
    target_os = "linux",
    any(
        not(feature = "skippy-devices"),
        feature = "dynamic-native-runtime",
        test
    )
))]
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

#[cfg(target_os = "windows")]
fn powershell_output(script: &str) -> Option<String> {
    let output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

#[cfg(target_os = "windows")]
fn read_windows_total_ram_bytes() -> Option<u64> {
    let output = powershell_output(
        "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory",
    )?;
    parse_windows_total_physical_memory(&output)
}

/// Per-GPU VRAM in adapter order.
///
/// Adapters that report no VRAM (virtual and headless displays commonly report
/// none) keep a zero entry so this stays index-aligned with the adapter name
/// list that `hydrate_gpu_facts_with_identities` indexes into. Dropping those
/// entries shifts every later adapter's VRAM onto the wrong device.
#[cfg(any(target_os = "windows", test))]
fn windows_per_gpu_vram(controllers: &[(String, u64)]) -> Vec<u64> {
    controllers.iter().map(|(_, ram)| *ram).collect()
}

#[cfg(target_os = "windows")]
fn read_windows_video_controllers() -> Vec<(String, u64)> {
    let Some(output) = powershell_output(
        "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
    ) else {
        return Vec::new();
    };
    parse_windows_video_controller_json(&output)
}

/// Owns an Objective-C object retained via the "create rule" (e.g.
/// `MTLCreateSystemDefaultDevice`, `NS_RETURNS_RETAINED`) and releases it on
/// drop, so every return path -- including early `?`/null-check returns --
/// balances the retain instead of leaking the object.
#[cfg(target_os = "macos")]
struct RetainedObjcObject(*mut std::ffi::c_void);

#[cfg(target_os = "macos")]
impl Drop for RetainedObjcObject {
    /// Releases the retained object, balancing the "create rule" retain.
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        #[link(name = "objc")]
        unsafe extern "C" {
            fn objc_release(obj: *mut std::ffi::c_void);
        }
        unsafe { objc_release(self.0) };
    }
}

/// Queries the Metal-recommended working-set size in bytes for the default
/// device — best-effort, OS-reported, not a verified measurement.
#[cfg(target_os = "macos")]
fn query_metal_recommended_working_set_bytes() -> Option<u64> {
    use std::ffi::{c_char, c_void};

    #[link(name = "objc")]
    unsafe extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend(receiver: *mut c_void, selector: *mut c_void, ...) -> usize;
    }

    unsafe {
        let metal =
            libloading::Library::new("/System/Library/Frameworks/Metal.framework/Versions/A/Metal")
                .ok()?;
        let create_device = metal
            .get::<unsafe extern "C" fn() -> *mut c_void>(b"MTLCreateSystemDefaultDevice")
            .ok()?;
        let device = create_device();
        if device.is_null() {
            return None;
        }
        let _device = RetainedObjcObject(device);
        let selector = c"recommendedMaxWorkingSetSize";
        let selector = sel_registerName(selector.as_ptr());
        if selector.is_null() {
            return None;
        }
        let bytes = objc_msgSend(device, selector) as u64;
        (bytes > 0).then_some(bytes)
    }
}

/// Queries the GPU name as reported by the OS via `MTLDevice.name` (e.g.
/// "Apple M4 Max" or "AMD Radeon Pro 5500M") — best-effort, not a verified
/// measurement, but sourced from the GPU device rather than the CPU.
#[cfg(target_os = "macos")]
#[cfg_attr(
    all(feature = "skippy-devices", not(feature = "dynamic-native-runtime")),
    allow(dead_code)
)]
fn query_metal_device_name() -> Option<String> {
    use std::ffi::{CStr, c_char, c_void};

    // `objc_msgSend` is declared to return `usize` here (matching the other
    // FFI declaration of the same linked symbol above) and the pointer
    // results below are recovered with `as *mut/*const _` casts, to avoid a
    // `clashing_extern_declarations` warning from two conflicting return
    // types for one symbol.
    #[link(name = "objc")]
    unsafe extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend(receiver: *mut c_void, selector: *mut c_void, ...) -> usize;
    }

    unsafe {
        let metal =
            libloading::Library::new("/System/Library/Frameworks/Metal.framework/Versions/A/Metal")
                .ok()?;
        let create_device = metal
            .get::<unsafe extern "C" fn() -> *mut c_void>(b"MTLCreateSystemDefaultDevice")
            .ok()?;
        let device = create_device();
        if device.is_null() {
            return None;
        }
        let _device = RetainedObjcObject(device);
        let name_sel = sel_registerName(c"name".as_ptr());
        if name_sel.is_null() {
            return None;
        }
        let name_obj = objc_msgSend(device, name_sel) as *mut c_void;
        if name_obj.is_null() {
            return None;
        }
        let utf8_sel = sel_registerName(c"UTF8String".as_ptr());
        if utf8_sel.is_null() {
            return None;
        }
        let utf8_ptr = objc_msgSend(name_obj, utf8_sel) as *const c_char;
        if utf8_ptr.is_null() {
            return None;
        }
        Some(CStr::from_ptr(utf8_ptr).to_string_lossy().into_owned())
    }
}

#[cfg(feature = "skippy-devices")]
fn apply_skippy_backend_devices_to_survey(survey: &mut HardwareSurvey, metrics: &[Metric]) -> bool {
    let wants_gpu_data = metrics.contains(&Metric::GpuName)
        || metrics.contains(&Metric::GpuCount)
        || metrics.contains(&Metric::VramBytes)
        || metrics.contains(&Metric::GpuFacts)
        || metrics.contains(&Metric::IsSoc);
    if !wants_gpu_data {
        return false;
    }
    #[cfg(feature = "dynamic-native-runtime")]
    if !skippy_runtime::native_runtime_loaded() {
        return false;
    }

    apply_gpu_probe_outcome_to_survey(
        survey,
        metrics,
        skippy_devices::gpu_facts(),
        survey_system_ram,
    )
}

/// System RAM the survey may credit (CPU-only budgets and the discrete-GPU
/// RAM-offload credit). Real read on Linux and Windows; zero elsewhere, which
/// leaves those platforms' VRAM budgets untouched.
#[cfg(any(feature = "skippy-devices", test))]
#[cfg_attr(
    not(any(feature = "skippy-devices", target_os = "linux", target_os = "windows")),
    allow(dead_code)
)]
fn survey_system_ram() -> u64 {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        read_system_ram_bytes()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        0
    }
}

/// Applies a GPU probe outcome to the survey. The real probe (skippy device
/// enumeration, /proc/meminfo, the Windows CIM query) stays in the callers so
/// this decision can be exercised with injected values on every platform.
/// The system RAM closure feeds the CPU-only fallback branches and the
/// discrete-GPU RAM-offload credit; it runs only when VramBytes is requested,
/// and never for a unified-memory survey, so probes that skip VramBytes never
/// pay for it. Platform gating lives in the production RAM source
/// (`survey_system_ram`), which keeps this seam platform-pure.
#[cfg(any(feature = "skippy-devices", test))]
fn apply_gpu_probe_outcome_to_survey<E>(
    survey: &mut HardwareSurvey,
    metrics: &[Metric],
    probe: Result<Vec<GpuFacts>, E>,
    system_ram: impl FnOnce() -> u64,
) -> bool {
    let gpus = match probe {
        Ok(gpus) => gpus,
        Err(_) => {
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            apply_cpu_only_runtime_budget(survey, metrics, system_ram);
            return true;
        }
    };
    if gpus.is_empty() {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        apply_cpu_only_runtime_budget(survey, metrics, system_ram);
        return true;
    }

    if metrics.contains(&Metric::GpuName) {
        let names: Vec<String> = gpus.iter().map(|gpu| gpu.display_name.clone()).collect();
        // Derive the source from the actual backend device names the
        // runtime reported, not from the target OS: a macOS host running
        // MoltenVK/Vulkan stamps non-MTL device names and must not be
        // labelled MetalDefaultDevice.
        let is_metal = gpus.iter().any(|gpu| {
            gpu.backend_device
                .as_deref()
                .map(|d| d.starts_with("MTL"))
                .unwrap_or(false)
        });
        let name = summarize_gpu_name(&names);
        let source = if is_metal {
            GpuNameSource::MetalDefaultDevice
        } else {
            GpuNameSource::NativeRuntimeDevice
        };
        survey.gpu_name_source = name.is_some().then_some(source);
        survey.gpu_name = name;
    }
    if metrics.contains(&Metric::GpuCount) {
        survey.gpu_count = u8::try_from(gpus.len()).unwrap_or(u8::MAX);
    }
    // Decided from the probe itself, not from the requested metrics: a
    // VramBytes-only query must still route unified memory through the
    // reserved-bytes path instead of crediting system RAM as offload.
    let unified_memory = gpus.iter().any(|gpu| gpu.unified_memory);
    if metrics.contains(&Metric::IsSoc) {
        survey.is_soc = unified_memory;
    }
    if metrics.contains(&Metric::VramBytes) {
        survey.gpu_vram = gpus.iter().map(|gpu| gpu.vram_bytes).collect();
        survey.gpu_reserved = gpus.iter().map(|gpu| gpu.reserved_bytes).collect();
        let vram: u64 = survey.gpu_vram.iter().sum();
        if unified_memory {
            let reserved: u64 = survey.gpu_reserved.iter().flatten().copied().sum();
            survey.vram_bytes = vram.saturating_sub(reserved);
        } else {
            let ram_offload = system_ram().saturating_sub(vram);
            survey.vram_bytes = vram + (ram_offload as f64 * 0.90) as u64;
        }
    }
    if metrics.contains(&Metric::GpuFacts) {
        survey.gpus = gpus;
    }

    true
}

impl Collector for DefaultCollector {
    fn collect(&self, metrics: &[Metric]) -> HardwareSurvey {
        let mut survey = HardwareSurvey::default();

        #[cfg(feature = "skippy-devices")]
        if apply_skippy_backend_devices_to_survey(&mut survey, metrics) {
            return survey;
        }

        #[cfg(all(
            target_os = "macos",
            any(not(feature = "skippy-devices"), feature = "dynamic-native-runtime")
        ))]
        {
            if metrics.contains(&Metric::IsSoc) {
                survey.is_soc = true;
            }
            let metal_budget = if metrics.contains(&Metric::VramBytes) {
                macos_metal_gpu_budget(query_metal_recommended_working_set_bytes())
            } else {
                None
            };
            if let Some((vram_bytes, reserved_bytes)) = metal_budget {
                survey.vram_bytes = vram_bytes;
                survey.gpu_vram = vec![vram_bytes];
                survey.gpu_reserved = vec![reserved_bytes];
            }
            if metrics.contains(&Metric::GpuName) {
                let name = sanitize_macos_gpu_name(query_metal_device_name());
                survey.gpu_name_source =
                    name.is_some().then_some(GpuNameSource::MetalDefaultDevice);
                survey.gpu_name = name;
            }
            if metrics.contains(&Metric::GpuCount) {
                survey.gpu_count = 1;
            }
        }

        #[cfg(all(
            target_os = "linux",
            any(not(feature = "skippy-devices"), feature = "dynamic-native-runtime")
        ))]
        {
            let system_ram = read_system_ram_bytes();

            if metrics.contains(&Metric::VramBytes) {
                // Try NVIDIA (mesh.rs:284-316)
                let nvidia_vram: Option<(u64, Vec<u64>)> = (|| {
                    let out = std::process::Command::new("nvidia-smi")
                        .args([
                            "--query-gpu=memory.total,memory.reserved",
                            "--format=csv,noheader,nounits",
                        ])
                        .output()
                        .ok();
                    match out {
                        Some(out) if out.status.success() => {
                            let s = String::from_utf8(out.stdout).ok()?;
                            let parsed = parse_nvidia_gpu_memory_and_reserved(&s);
                            if !parsed.is_empty() {
                                survey.gpu_reserved =
                                    parsed.iter().map(|(_, reserved)| *reserved).collect();
                                let per_gpu: Vec<u64> =
                                    parsed.iter().map(|(total, _)| *total).collect();
                                let total: u64 = per_gpu.iter().sum();
                                if total > 0 {
                                    return Some((total, per_gpu));
                                }
                            }
                        }
                        Some(_) | None => {}
                    }
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
                        survey.gpu_reserved = vec![None; per_gpu.len()];
                        Some((total, per_gpu))
                    } else {
                        None
                    }
                })();

                if let Some((vram, per_gpu)) = nvidia_vram {
                    survey.gpu_vram = per_gpu;
                    let ram_offload = system_ram.saturating_sub(vram);
                    survey.vram_bytes = vram + (ram_offload as f64 * 0.90) as u64;
                } else {
                    // Try AMD ROCm (mesh.rs:295-316)
                    let rocm_vram: Option<(Vec<u64>, bool)> = (|| {
                        let out = std::process::Command::new("rocm-smi")
                            .args(["--showmeminfo", "vram", "--csv"])
                            .output()
                            .ok()?;
                        if !out.status.success() {
                            return None;
                        }
                        let s = String::from_utf8(out.stdout).ok()?;
                        let parsed = parse_rocm_gpu_memory_and_used(&s);
                        // ROCm exposes total and live used VRAM here, not a
                        // true reserved/unavailable metric, so leave
                        // reserved_bytes unavailable for this backend.
                        survey.gpu_reserved = vec![None; parsed.len()];
                        let vrams: Vec<u64> = parsed.iter().map(|(total, _)| *total).collect();
                        if vrams.is_empty() {
                            None
                        } else {
                            let gtt_totals = std::process::Command::new("rocm-smi")
                                .args(["--showmeminfo", "gtt", "--csv"])
                                .output()
                                .ok()
                                .filter(|out| out.status.success())
                                .and_then(|out| String::from_utf8(out.stdout).ok())
                                .map(|stdout| {
                                    parse_rocm_gpu_memory_and_used(&stdout)
                                        .into_iter()
                                        .map(|(total, _)| total)
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default();
                            if let Some(usable_bytes) =
                                rocm_unified_memory_usable_bytes(&vrams, &gtt_totals, system_ram)
                            {
                                Some((vec![usable_bytes], true))
                            } else {
                                Some((vrams, false))
                            }
                        }
                    })();

                    if let Some((per_gpu, unified_memory)) = rocm_vram {
                        let vram: u64 = per_gpu.iter().sum();
                        survey.gpu_vram = per_gpu;
                        if unified_memory {
                            survey.is_soc = true;
                            survey.vram_bytes = vram;
                        } else {
                            let ram_offload = system_ram.saturating_sub(vram);
                            survey.vram_bytes = vram + (ram_offload as f64 * 0.90) as u64;
                        }
                    } else {
                        let intel_gpus: Option<Vec<XpuSmiGpuInfo>> = (|| {
                            for args in [["discovery", "--json"], ["discovery", "-j"]] {
                                let out = std::process::Command::new("xpu-smi")
                                    .args(args)
                                    .output()
                                    .ok()?;
                                if !out.status.success() {
                                    continue;
                                }
                                let stdout = String::from_utf8(out.stdout).ok()?;
                                let gpus = parse_xpu_smi_discovery_json(&stdout);
                                if !gpus.is_empty() {
                                    return Some(gpus);
                                }
                            }
                            None
                        })();

                        if let Some(intel_gpus) = intel_gpus {
                            // xpu-smi discovery reports capacity plus used
                            // bytes, but not a true reserved/unavailable
                            // metric, so leave reserved_bytes unavailable.
                            survey.gpu_reserved = vec![None; intel_gpus.len()];
                            let per_gpu: Vec<u64> = intel_gpus
                                .iter()
                                .map(|gpu| gpu.total_bytes.unwrap_or(0))
                                .collect();
                            let total: u64 = per_gpu.iter().sum();
                            survey.gpu_vram = per_gpu;
                            if total > 0 {
                                let ram_offload = system_ram.saturating_sub(total);
                                survey.vram_bytes = total + (ram_offload as f64 * 0.90) as u64;
                            } else if system_ram > 0 {
                                survey.vram_bytes = (system_ram as f64 * 0.90) as u64;
                            }
                        } else if system_ram > 0 {
                            // CPU-only (mesh.rs:320-322)
                            survey.vram_bytes = (system_ram as f64 * 0.90) as u64;
                        }
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
                    if names.is_empty() { None } else { Some(names) }
                })();

                if let Some(ref names) = nvidia_names {
                    if metrics.contains(&Metric::GpuName) {
                        let name = summarize_gpu_name(names);
                        survey.gpu_name_source = name.is_some().then_some(GpuNameSource::NvidiaSmi);
                        survey.gpu_name = name;
                    }
                    if metrics.contains(&Metric::GpuCount) {
                        survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                    }
                } else {
                    let out = std::process::Command::new("rocm-smi")
                        .args(["--showproductname"])
                        .output()
                        .ok();
                    match out {
                        Some(out) if out.status.success() => {
                            if let Ok(s) = String::from_utf8(out.stdout) {
                                let names = parse_rocm_gpu_names(&s);
                                if metrics.contains(&Metric::GpuName) {
                                    let name = summarize_gpu_name(&names);
                                    survey.gpu_name_source =
                                        name.is_some().then_some(GpuNameSource::RocmSmi);
                                    survey.gpu_name = name;
                                }
                                if metrics.contains(&Metric::GpuCount) {
                                    survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                                }
                            }
                        }
                        None => {
                            for args in [["discovery", "--json"], ["discovery", "-j"]] {
                                let out = std::process::Command::new("xpu-smi")
                                    .args(args)
                                    .output()
                                    .ok();
                                if let Some(out) = out {
                                    if !out.status.success() {
                                        continue;
                                    }
                                    let Ok(stdout) = String::from_utf8(out.stdout) else {
                                        continue;
                                    };
                                    let gpus = parse_xpu_smi_discovery_json(&stdout);
                                    if !gpus.is_empty() {
                                        let names: Vec<String> =
                                            gpus.iter().map(|gpu| gpu.name.clone()).collect();
                                        if metrics.contains(&Metric::GpuName) {
                                            let name = summarize_gpu_name(&names);
                                            survey.gpu_name_source =
                                                name.is_some().then_some(GpuNameSource::XpuSmi);
                                            survey.gpu_name = name;
                                        }
                                        if metrics.contains(&Metric::GpuCount) {
                                            survey.gpu_count =
                                                u8::try_from(names.len()).unwrap_or(u8::MAX);
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        Some(_) => {}
                    }
                }
            }
        }

        #[cfg(all(
            target_os = "windows",
            any(not(feature = "skippy-devices"), feature = "dynamic-native-runtime")
        ))]
        {
            let system_ram = read_windows_total_ram_bytes().unwrap_or(0);
            let want_gpu_info =
                metrics.contains(&Metric::GpuName) || metrics.contains(&Metric::GpuCount);
            let want_vram = metrics.contains(&Metric::VramBytes);

            let nvidia_names = if want_gpu_info {
                std::process::Command::new("nvidia-smi")
                    .args(["--query-gpu=name", "--format=csv,noheader"])
                    .output()
                    .ok()
                    .and_then(|out| {
                        if !out.status.success() {
                            return None;
                        }
                        let s = String::from_utf8(out.stdout).ok()?;
                        let names = parse_nvidia_gpu_names(&s);
                        if names.is_empty() { None } else { Some(names) }
                    })
            } else {
                None
            };

            let nvidia_vram = if want_vram {
                std::process::Command::new("nvidia-smi")
                    .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                    .output()
                    .ok()
                    .and_then(|out| {
                        if !out.status.success() {
                            return None;
                        }
                        let s = String::from_utf8(out.stdout).ok()?;
                        let per_gpu = parse_nvidia_gpu_memory(&s);
                        if per_gpu.is_empty() {
                            None
                        } else {
                            Some(per_gpu)
                        }
                    })
            } else {
                None
            };

            let windows_gpus = if want_gpu_info || want_vram {
                read_windows_video_controllers()
            } else {
                Vec::new()
            };

            if want_vram {
                if let Some(per_gpu) = nvidia_vram {
                    let total: u64 = per_gpu.iter().sum();
                    if total > 0 {
                        survey.gpu_vram = per_gpu;
                        let ram_offload = system_ram.saturating_sub(total);
                        survey.vram_bytes = total + (ram_offload as f64 * 0.90) as u64;
                    }
                } else {
                    let per_gpu: Vec<u64> = windows_per_gpu_vram(&windows_gpus);
                    let total: u64 = per_gpu.iter().sum();
                    survey.gpu_vram = per_gpu;
                    if total > 0 {
                        let ram_offload = system_ram.saturating_sub(total);
                        survey.vram_bytes = total + (ram_offload as f64 * 0.90) as u64;
                    } else if system_ram > 0 {
                        survey.vram_bytes = (system_ram as f64 * 0.90) as u64;
                    }
                }
            }

            if want_gpu_info {
                if let Some(ref names) = nvidia_names {
                    if metrics.contains(&Metric::GpuName) {
                        let name = summarize_gpu_name(names);
                        survey.gpu_name_source = name.is_some().then_some(GpuNameSource::NvidiaSmi);
                        survey.gpu_name = name;
                    }
                    if metrics.contains(&Metric::GpuCount) {
                        survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                    }
                } else {
                    let names: Vec<String> =
                        windows_gpus.iter().map(|(name, _)| name.clone()).collect();
                    if metrics.contains(&Metric::GpuName) {
                        let name = summarize_gpu_name(&names);
                        survey.gpu_name_source = name
                            .is_some()
                            .then_some(GpuNameSource::WindowsVideoController);
                        survey.gpu_name = name;
                    }
                    if metrics.contains(&Metric::GpuCount) {
                        survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                    }
                }
            }
        }

        survey
    }
}

/// Read the Tegra/Jetson model name from `model_path` and record it (with its
/// `Sysfs` source) on `survey`. Leaves both `gpu_name` and `gpu_name_source`
/// untouched when the path is absent or unparseable — never a guessed source
/// for a name that was not actually read. Split out from `collect` so the path
/// can be driven deterministically in tests rather than depending on host
/// filesystem state.
#[cfg(all(
    target_os = "linux",
    any(
        not(feature = "skippy-devices"),
        feature = "dynamic-native-runtime",
        test
    )
))]
fn tegra_gpu_name_from_model_path(survey: &mut HardwareSurvey, model_path: &std::path::Path) {
    let name = std::fs::read_to_string(model_path)
        .ok()
        .and_then(|model| parse_tegra_model_name(&model));
    survey.gpu_name_source = name.is_some().then_some(GpuNameSource::Sysfs);
    survey.gpu_name = name;
}

#[cfg(all(
    target_os = "linux",
    any(
        not(feature = "skippy-devices"),
        feature = "dynamic-native-runtime",
        test
    )
))]
impl Collector for TegraCollector {
    fn collect(&self, metrics: &[Metric]) -> HardwareSurvey {
        let mut survey = HardwareSurvey::default();

        if metrics.contains(&Metric::IsSoc) {
            survey.is_soc = true;
        }

        if metrics.contains(&Metric::GpuName) {
            tegra_gpu_name_from_model_path(
                &mut survey,
                std::path::Path::new("/sys/firmware/devicetree/base/model"),
            );
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
                survey.vram_bytes = (ram as f64 * 0.90) as u64;
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

#[cfg(all(
    target_os = "linux",
    feature = "skippy-devices",
    not(feature = "dynamic-native-runtime")
))]
fn detect_collector_impl() -> Box<dyn Collector> {
    Box::new(DefaultCollector)
}

#[cfg(all(
    target_os = "linux",
    any(not(feature = "skippy-devices"), feature = "dynamic-native-runtime")
))]
fn detect_collector_impl() -> Box<dyn Collector> {
    if is_tegra_host() {
        return Box::new(TegraCollector);
    }
    Box::new(DefaultCollector)
}

#[cfg(all(
    target_os = "linux",
    any(not(feature = "skippy-devices"), feature = "dynamic-native-runtime")
))]
fn is_tegra_host() -> bool {
    if !cfg!(target_arch = "aarch64") {
        return false;
    }
    match std::fs::read_to_string("/proc/device-tree/compatible") {
        Ok(compat) => is_tegra(&compat),
        Err(_) => false,
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_collector_impl() -> Box<dyn Collector> {
    Box::new(DefaultCollector)
}

fn detect_collector() -> Box<dyn Collector> {
    detect_collector_impl()
}

#[cfg(any(
    not(feature = "skippy-devices"),
    feature = "dynamic-native-runtime",
    test
))]
fn backend_device_for_name(name: &str, index: usize, is_soc: bool) -> Option<String> {
    backend_device_for_name_for_platform(name, index, is_soc, cfg!(target_os = "macos"))
}

#[cfg(any(
    not(feature = "skippy-devices"),
    feature = "dynamic-native-runtime",
    test
))]
fn backend_device_for_name_for_platform(
    name: &str,
    index: usize,
    is_soc: bool,
    soc_backend_is_metal: bool,
) -> Option<String> {
    if soc_backend_is_metal && is_soc {
        return Some(format!("MTL{index}"));
    }
    let upper = name.to_ascii_uppercase();
    if upper.contains("NVIDIA")
        || (is_soc
            && (upper.contains("JETSON")
                || upper.contains("TEGRA")
                || upper.contains("NVGPU")
                || upper.contains("ORIN")))
    {
        Some(format!("CUDA{index}"))
    } else if upper.contains("AMD")
        || upper.contains("RADEON")
        || upper.contains("INSTINCT")
        || upper.starts_with("MI")
    {
        Some(format!("ROCm{index}"))
    } else {
        None
    }
}

#[cfg(all(
    any(
        not(feature = "skippy-devices"),
        feature = "dynamic-native-runtime",
        test
    ),
    any(target_os = "linux", target_os = "windows")
))]
fn detect_nvidia_identities() -> Vec<(Option<String>, Option<String>)> {
    let out = match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"])
        .output()
    {
        Ok(out) if out.status.success() => out,
        _ => return Vec::new(),
    };
    let Ok(stdout) = String::from_utf8(out.stdout) else {
        return Vec::new();
    };
    parse_nvidia_gpu_identity(&stdout)
}

#[cfg(all(
    any(
        not(feature = "skippy-devices"),
        feature = "dynamic-native-runtime",
        test
    ),
    not(any(target_os = "linux", target_os = "windows"))
))]
fn detect_nvidia_identities() -> Vec<(Option<String>, Option<String>)> {
    Vec::new()
}

#[cfg(any(
    not(feature = "skippy-devices"),
    feature = "dynamic-native-runtime",
    test
))]
fn inferred_gpu_name_count(name: Option<&str>) -> usize {
    let Some(name) = name.map(str::trim).filter(|name| !name.is_empty()) else {
        return 0;
    };

    name.split_once('×')
        .or_else(|| name.split_once('x'))
        .or_else(|| name.split_once('X'))
        .and_then(|(count, _)| count.trim().parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(1)
}

fn is_pinnable_gpu_stable_id(stable_id: &str) -> bool {
    stable_id.starts_with("pci:")
        || stable_id.starts_with("uuid:")
        || stable_id.starts_with("metal:")
}

fn is_placeholder_pci_bdf(pci_bdf: &str) -> bool {
    matches!(
        pci_bdf.trim().to_ascii_lowercase().as_str(),
        "0000:00:00.0" | "00000000:00:00.0"
    )
}

fn push_unique_pinnable_id(ids: &mut Vec<String>, id: String) {
    if is_pinnable_gpu_stable_id(&id) && !ids.iter().any(|existing| existing == &id) {
        ids.push(id);
    }
}

fn gpu_pinnable_ids(gpu: &GpuFacts) -> Vec<String> {
    let mut ids = Vec::new();
    if let Some(stable_id) = gpu.stable_id.as_deref() {
        push_unique_pinnable_id(&mut ids, stable_id.to_string());
    }
    if let Some(pci_bdf) = gpu
        .pci_bdf
        .as_deref()
        .filter(|pci_bdf| !is_placeholder_pci_bdf(pci_bdf))
    {
        push_unique_pinnable_id(&mut ids, format!("pci:{pci_bdf}"));
    }
    if let Some(vendor_uuid) = gpu.vendor_uuid.as_deref() {
        push_unique_pinnable_id(&mut ids, format!("uuid:{vendor_uuid}"));
    }
    ids
}

pub fn pinnable_gpu_stable_ids(gpus: &[GpuFacts]) -> Vec<String> {
    gpus.iter().flat_map(gpu_pinnable_ids).collect()
}

fn format_pinnable_gpu_ids(ids: &[String]) -> String {
    if ids.is_empty() {
        "none".to_string()
    } else {
        ids.join(", ")
    }
}

pub fn resolve_pinned_gpu<'a>(
    configured_id: Option<&str>,
    gpus: &'a [GpuFacts],
) -> Result<&'a GpuFacts, PinnedGpuResolverError> {
    resolve_pinned_gpu_with_compatibility(configured_id, gpus, true)
}

pub fn resolve_pinned_gpu_strict<'a>(
    configured_id: Option<&str>,
    gpus: &'a [GpuFacts],
) -> Result<&'a GpuFacts, PinnedGpuResolverError> {
    resolve_pinned_gpu_with_compatibility(configured_id, gpus, false)
}

fn resolve_pinned_gpu_with_compatibility<'a>(
    configured_id: Option<&str>,
    gpus: &'a [GpuFacts],
    accept_single_pinnable_gpu_fallback: bool,
) -> Result<&'a GpuFacts, PinnedGpuResolverError> {
    let available_pinnable_ids = pinnable_gpu_stable_ids(gpus);
    let Some(configured_id) = configured_id.map(str::trim).filter(|id| !id.is_empty()) else {
        return Err(PinnedGpuResolverError::MissingConfiguredId {
            available_pinnable_ids,
        });
    };
    let configured_id = configured_id.to_string();

    if !is_pinnable_gpu_stable_id(&configured_id) {
        return Err(PinnedGpuResolverError::NonPinnableConfiguredId {
            configured_id,
            available_pinnable_ids,
        });
    }

    if available_pinnable_ids.is_empty() {
        return Err(PinnedGpuResolverError::NoPinnableGpus {
            configured_id,
            available_pinnable_ids,
        });
    }

    let matches = gpus
        .iter()
        .enumerate()
        .filter(|(_, gpu)| gpu_pinnable_ids(gpu).iter().any(|id| id == &configured_id))
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [(_, gpu)] => Ok(*gpu),
        [] => {
            let pinnable_gpus = gpus
                .iter()
                .filter(|gpu| !gpu_pinnable_ids(gpu).is_empty())
                .collect::<Vec<_>>();
            if let (true, [gpu]) = (
                accept_single_pinnable_gpu_fallback,
                pinnable_gpus.as_slice(),
            ) {
                tracing::warn!(
                    "configured gpu_id '{}' did not match the single available pinnable GPU; accepting '{}' for compatibility",
                    configured_id,
                    gpu_pinnable_ids(gpu).join(", ")
                );
                return Ok(*gpu);
            }

            Err(PinnedGpuResolverError::NoMatch {
                configured_id,
                available_pinnable_ids,
            })
        }
        _ => Err(PinnedGpuResolverError::AmbiguousMatch {
            configured_id,
            available_pinnable_ids,
            match_indexes: matches.iter().map(|(index, _)| *index).collect(),
        }),
    }
}

#[cfg(any(
    not(feature = "skippy-devices"),
    feature = "dynamic-native-runtime",
    test
))]
fn hydrate_gpu_facts(survey: &mut HardwareSurvey, metrics: &[Metric]) {
    let expected_count = survey
        .gpu_vram
        .len()
        .max(usize::from(survey.gpu_count))
        .max(inferred_gpu_name_count(survey.gpu_name.as_deref()));
    let mut names = expand_gpu_names(survey.gpu_name.as_deref(), expected_count);
    if names.is_empty() && expected_count > 0 {
        names = (0..expected_count)
            .map(|index| format!("GPU {index}"))
            .collect();
    }

    let needs_nvidia_identities = metrics.contains(&Metric::GpuName);
    let nvidia_identities = if needs_nvidia_identities {
        detect_nvidia_identities()
    } else {
        Vec::new()
    };
    hydrate_gpu_facts_with_identities(
        survey,
        metrics,
        &nvidia_identities,
        names,
        expected_count,
        cfg!(target_os = "macos"),
    );
}

#[cfg(any(
    not(feature = "skippy-devices"),
    feature = "dynamic-native-runtime",
    test
))]
fn hydrate_gpu_facts_with_identities(
    survey: &mut HardwareSurvey,
    metrics: &[Metric],
    nvidia_identities: &[(Option<String>, Option<String>)],
    names: Vec<String>,
    expected_count: usize,
    soc_backend_is_metal: bool,
) {
    let count = expected_count.max(names.len());
    survey.gpus = (0..count)
        .map(|index| {
            let display_name = names
                .get(index)
                .cloned()
                .unwrap_or_else(|| format!("GPU {index}"));
            let backend_device = if soc_backend_is_metal == cfg!(target_os = "macos") {
                backend_device_for_name(&display_name, index, survey.is_soc)
            } else {
                backend_device_for_name_for_platform(
                    &display_name,
                    index,
                    survey.is_soc,
                    soc_backend_is_metal,
                )
            };
            let (pci_bdf, vendor_uuid) = nvidia_identities.get(index).cloned().unwrap_or_default();
            let stable_id = if survey.is_soc && soc_backend_is_metal {
                Some(format!("metal:{index}"))
            } else if let Some(pci_bdf) = pci_bdf
                .as_deref()
                .filter(|pci_bdf| !is_placeholder_pci_bdf(pci_bdf))
            {
                Some(format!("pci:{pci_bdf}"))
            } else if let Some(ref vendor_uuid) = vendor_uuid {
                Some(format!("uuid:{vendor_uuid}"))
            } else if let Some(ref backend_device) = backend_device {
                Some(backend_device.to_ascii_lowercase())
            } else {
                Some(format!("index:{index}"))
            };

            GpuFacts {
                index,
                display_name,
                backend_device,
                vram_bytes: survey.gpu_vram.get(index).copied().unwrap_or(0),
                reserved_bytes: survey.gpu_reserved.get(index).cloned().flatten(),
                mem_bandwidth_gbps: None,
                compute_tflops_fp32: None,
                compute_tflops_fp16: None,
                unified_memory: survey.is_soc,
                stable_id,
                pci_bdf,
                vendor_uuid,
                metal_registry_id: None,
                dxgi_luid: None,
                pnp_instance_id: None,
            }
        })
        .collect();

    debug_assert!(
        pinnable_gpu_stable_ids(&survey.gpus)
            .into_iter()
            .all(|stable_id| resolve_pinned_gpu(Some(&stable_id), &survey.gpus).is_ok())
    );

    if metrics.contains(&Metric::GpuCount) && survey.gpu_count == 0 {
        survey.gpu_count = u8::try_from(survey.gpus.len()).unwrap_or(u8::MAX);
    }
    if metrics.contains(&Metric::GpuName) && survey.gpu_name.is_none() {
        let names: Vec<String> = survey
            .gpus
            .iter()
            .map(|gpu| gpu.display_name.clone())
            .collect();
        let name = summarize_gpu_name(&names);
        survey.gpu_name_source = name.is_some().then_some(GpuNameSource::Unknown);
        survey.gpu_name = name;
    }
}

/// Collect only the requested hardware metrics.
pub fn query(metrics: &[Metric]) -> HardwareSurvey {
    let collector = detect_collector();
    let mut survey = collector.collect(metrics);
    if metrics.contains(&Metric::Hostname) {
        survey.hostname = detect_hostname();
    }
    #[cfg(any(not(feature = "skippy-devices"), feature = "dynamic-native-runtime"))]
    if metrics.contains(&Metric::GpuFacts) && survey.gpus.is_empty() {
        hydrate_gpu_facts(&mut survey, metrics);
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
        Metric::GpuFacts,
    ])
}
