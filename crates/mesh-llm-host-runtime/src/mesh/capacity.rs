use crate::system::hardware::HardwareSurvey;

pub(super) fn mesh_capacity_bytes(hw: &HardwareSurvey) -> u64 {
    if unified_memory_only(hw) {
        return hw.vram_bytes;
    }

    let gpu_capacity = hw
        .gpus
        .iter()
        .map(|gpu| mesh_llm_system::vram::allocatable_bytes(gpu.vram_bytes, gpu.reserved_bytes))
        .sum();
    if gpu_capacity > 0 {
        return gpu_capacity;
    }

    let legacy_gpu_capacity = hw
        .gpu_vram
        .iter()
        .enumerate()
        .map(|(index, &vram)| {
            mesh_llm_system::vram::allocatable_bytes(
                vram,
                hw.gpu_reserved.get(index).copied().flatten(),
            )
        })
        .sum();
    if legacy_gpu_capacity > 0 {
        legacy_gpu_capacity
    } else {
        // A non-SoC node without enumerated accelerator memory cannot host a
        // GPU stage. Keep the broader RAM/offload budget local-only instead
        // of advertising it as accelerator capacity.
        0
    }
}

/// A host whose accelerator memory is the system memory: its budget comes
/// from a platform working set (Metal) or a platform policy (the Tegra
/// collector), not from device VRAM minus a driver reserve.
fn unified_memory_only(hw: &HardwareSurvey) -> bool {
    hw.is_soc && (hw.gpus.is_empty() || hw.gpus.iter().all(|gpu| gpu.unified_memory))
}

pub(super) fn capped_capacity_bytes(capacity_bytes: u64, max_vram_gb: Option<f64>) -> u64 {
    max_vram_gb
        .map(|cap| capacity_bytes.min((cap * 1e9) as u64))
        .unwrap_or(capacity_bytes)
}

pub(super) fn advertised_capacity_bytes(hw: &HardwareSurvey, max_vram_gb: Option<f64>) -> u64 {
    let detected = mesh_capacity_bytes(hw);
    match (detected, max_vram_gb) {
        (0, Some(cap)) => hw.vram_bytes.min((cap * 1e9) as u64),
        _ => capped_capacity_bytes(detected, max_vram_gb),
    }
}

/// Itemized view of the capacity a node advertises. The announcement's
/// `vram_bytes` stays the placement budget; this block explains how that
/// number was derived. Invariant: `total_bytes == reserved_bytes +
/// platform_reserve_bytes + configured_reserve_bytes + usable_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdvertisedMemory {
    /// Enumerated accelerator memory: the sum of device VRAM, or the unified
    /// working set on SoCs.
    pub total_bytes: u64,
    /// Driver/runtime reserved or unavailable bytes when the platform reports
    /// a true value.
    pub reserved_bytes: u64,
    /// Withheld by platform policy before the owner configures anything: the
    /// share of unified memory the platform keeps for the system (the Tegra
    /// collector budgets 90% of physical RAM). Zero for discrete GPUs, whose
    /// budget is the device memory minus the driver reserve, and zero on
    /// Metal while the survey reports the working set as the device memory.
    pub platform_reserve_bytes: u64,
    /// Withheld by the node owner: the effective safety margin plus whatever
    /// a `max_vram_gb` cap leaves out.
    pub configured_reserve_bytes: u64,
    /// What remains for mesh placement after both reserves.
    pub usable_bytes: u64,
    /// Total system RAM when the platform reports it.
    pub system_ram_bytes: Option<u64>,
    /// Portion of the local fit budget backed by system RAM, after any
    /// `max_vram_gb` cap. Never advertised as accelerator capacity.
    pub ram_offload_bytes: u64,
}

pub(super) fn advertised_memory(
    hw: &HardwareSurvey,
    max_vram_gb: Option<f64>,
    safety_margin_bytes: u64,
) -> AdvertisedMemory {
    let (device_vram, driver_reserved) = enumerated_device_memory(hw);
    let budget = advertised_capacity_bytes(hw, max_vram_gb);
    // A host without enumerated accelerator memory only reaches a non-zero
    // budget through an explicit `max_vram_gb` cap on its CPU budget; that
    // bounded budget is then the whole of what it offers.
    let total_bytes = if device_vram == 0 {
        budget
    } else {
        device_vram
    };
    let reserved_bytes = driver_reserved.min(total_bytes);
    // On a unified-memory host the platform's own budget is a policy (Metal's
    // working set, Tegra's 90% of RAM): whatever it keeps back from the
    // enumerated memory is a platform reserve, not something the owner set.
    // Discrete hosts budget the device memory minus the driver reserve, so
    // nothing lands here.
    let platform_reserve_bytes = if unified_memory_only(hw) {
        total_bytes
            .saturating_sub(reserved_bytes)
            .saturating_sub(mesh_capacity_bytes(hw))
    } else {
        0
    };
    let owner_ceiling = total_bytes
        .saturating_sub(reserved_bytes)
        .saturating_sub(platform_reserve_bytes);
    // The budget never exceeds what the platform leaves to the owner; the
    // clamp only keeps the invariant if a survey ever reports otherwise.
    let usable_bytes = budget
        .saturating_sub(safety_margin_bytes)
        .min(owner_ceiling);
    let configured_reserve_bytes = owner_ceiling.saturating_sub(usable_bytes);
    // The survey derives its RAM-backed share from the uncapped budget. A
    // `max_vram_gb` cap shrinks the local budget first, and only what that
    // capped budget still carries beyond the device memory is RAM.
    let local_budget = capped_capacity_bytes(hw.vram_bytes, max_vram_gb);
    let ram_offload_bytes = hw
        .ram_offload_bytes
        .min(local_budget.saturating_sub(device_vram));
    AdvertisedMemory {
        total_bytes,
        reserved_bytes,
        platform_reserve_bytes,
        configured_reserve_bytes,
        usable_bytes,
        system_ram_bytes: hw.system_ram_bytes,
        ram_offload_bytes,
    }
}

/// Sum of the enumerated device memory and of the reserved bytes the platform
/// reported for it, with the same precedence as `mesh_capacity_bytes`: the
/// per-device facts first, the legacy per-GPU lists otherwise.
fn enumerated_device_memory(hw: &HardwareSurvey) -> (u64, u64) {
    if !hw.gpus.is_empty() {
        let vram = hw.gpus.iter().map(|gpu| gpu.vram_bytes).sum();
        let reserved = hw.gpus.iter().filter_map(|gpu| gpu.reserved_bytes).sum();
        return (vram, reserved);
    }
    let vram = hw.gpu_vram.iter().sum();
    let reserved = hw.gpu_reserved.iter().flatten().sum();
    (vram, reserved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{NodeRole, node::hardware_snapshot_for_start};
    use crate::system::hardware::GpuFacts;

    fn gpu(vram_bytes: u64, reserved_bytes: Option<u64>, unified_memory: bool) -> GpuFacts {
        GpuFacts {
            vram_bytes,
            reserved_bytes,
            unified_memory,
            ..GpuFacts::default()
        }
    }

    #[test]
    fn discrete_gpu_mesh_capacity_excludes_host_ram_offload_budget() {
        let hw = HardwareSurvey {
            vram_bytes: 491_000_000_000,
            gpu_vram: vec![40_000_000_000],
            gpu_reserved: vec![Some(1_000_000_000)],
            gpus: vec![gpu(40_000_000_000, Some(1_000_000_000), false)],
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None, 0);

        assert_eq!(snapshot.vram_bytes, 39_000_000_000);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 491_000_000_000);
    }

    #[test]
    fn unified_memory_mesh_capacity_keeps_recommended_working_set() {
        let hw = HardwareSurvey {
            vram_bytes: 96_000_000_000,
            is_soc: true,
            gpu_vram: vec![128_000_000_000],
            gpu_reserved: vec![Some(16_000_000_000)],
            gpus: vec![gpu(128_000_000_000, Some(16_000_000_000), true)],
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None, 0);

        assert_eq!(snapshot.vram_bytes, 96_000_000_000);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 96_000_000_000);
    }

    #[test]
    fn missing_discrete_gpu_facts_do_not_advertise_host_ram_as_stage_capacity() {
        let hw = HardwareSurvey {
            vram_bytes: 491_000_000_000,
            is_soc: false,
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None, 0);

        assert_eq!(snapshot.vram_bytes, 0);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 491_000_000_000);
    }

    #[test]
    fn explicit_cpu_budget_advertises_bounded_stage_capacity() {
        let hw = HardwareSurvey {
            vram_bytes: 16_000_000_000,
            is_soc: false,
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, Some(1.0), 0);

        assert_eq!(snapshot.vram_bytes, 1_000_000_000);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 1_000_000_000);
    }

    #[test]
    fn max_vram_caps_mesh_and_local_runtime_capacities() {
        let hw = HardwareSurvey {
            vram_bytes: 491_000_000_000,
            gpu_vram: vec![40_000_000_000],
            gpu_reserved: vec![Some(1_000_000_000)],
            gpus: vec![gpu(40_000_000_000, Some(1_000_000_000), false)],
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, Some(32.0), 0);

        assert_eq!(snapshot.vram_bytes, 32_000_000_000);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 32_000_000_000);
    }

    #[test]
    fn discrete_gpu_breakdown_itemizes_driver_reserve_margin_and_offload() {
        // 12 GB device, 0.5 GB driver reserve, 2 GB margin, 32 GB host:
        // 9.5 GB usable, and the 18 GB RAM credit stays a local-only item.
        let hw = HardwareSurvey {
            vram_bytes: 30_000_000_000,
            gpu_vram: vec![12_000_000_000],
            gpu_reserved: vec![Some(500_000_000)],
            gpus: vec![gpu(12_000_000_000, Some(500_000_000), false)],
            system_ram_bytes: Some(32_000_000_000),
            ram_offload_bytes: 18_000_000_000,
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, None, 2_000_000_000);

        assert_eq!(
            memory,
            AdvertisedMemory {
                total_bytes: 12_000_000_000,
                reserved_bytes: 500_000_000,
                platform_reserve_bytes: 0,
                configured_reserve_bytes: 2_000_000_000,
                usable_bytes: 9_500_000_000,
                system_ram_bytes: Some(32_000_000_000),
                ram_offload_bytes: 18_000_000_000,
            }
        );
        assert_breakdown_adds_up(&memory);
    }

    #[test]
    fn max_vram_cap_remainder_lands_in_the_configured_reserve() {
        // 40 GB device, 1 GB driver reserve, capped at 32 GB, 2 GB margin:
        // 30 GB usable, and the owner withholds the 7 GB cap remainder plus
        // the 2 GB margin.
        let hw = HardwareSurvey {
            vram_bytes: 491_000_000_000,
            gpu_vram: vec![40_000_000_000],
            gpu_reserved: vec![Some(1_000_000_000)],
            gpus: vec![gpu(40_000_000_000, Some(1_000_000_000), false)],
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, Some(32.0), 2_000_000_000);

        assert_eq!(memory.total_bytes, 40_000_000_000);
        assert_eq!(memory.reserved_bytes, 1_000_000_000);
        assert_eq!(memory.configured_reserve_bytes, 9_000_000_000);
        assert_eq!(memory.usable_bytes, 30_000_000_000);
        assert_breakdown_adds_up(&memory);
    }

    #[test]
    fn max_vram_cap_bounds_the_ram_backed_share_of_the_local_budget() {
        // 12 GB device credited to 30 GB with RAM: a 20 GB cap leaves 8 GB of
        // the capped local budget beyond the device, a cap under the device
        // memory leaves none.
        let hw = HardwareSurvey {
            vram_bytes: 30_000_000_000,
            gpu_vram: vec![12_000_000_000],
            gpu_reserved: vec![None],
            gpus: vec![gpu(12_000_000_000, None, false)],
            system_ram_bytes: Some(32_000_000_000),
            ram_offload_bytes: 18_000_000_000,
            ..HardwareSurvey::default()
        };

        assert_eq!(
            advertised_memory(&hw, None, 0).ram_offload_bytes,
            18_000_000_000
        );
        assert_eq!(
            advertised_memory(&hw, Some(20.0), 0).ram_offload_bytes,
            8_000_000_000
        );
        assert_eq!(advertised_memory(&hw, Some(8.0), 0).ram_offload_bytes, 0);
    }

    #[test]
    fn unified_memory_breakdown_reports_the_working_set_as_total() {
        // The Metal survey reports the recommended working set as the device
        // memory with no driver reserve, so only the margin is withheld.
        let hw = HardwareSurvey {
            vram_bytes: 96_000_000_000,
            is_soc: true,
            gpu_vram: vec![96_000_000_000],
            gpu_reserved: vec![None],
            gpus: vec![gpu(96_000_000_000, None, true)],
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, None, 2_000_000_000);

        assert_eq!(memory.total_bytes, 96_000_000_000);
        assert_eq!(memory.reserved_bytes, 0);
        assert_eq!(memory.platform_reserve_bytes, 0);
        assert_eq!(memory.configured_reserve_bytes, 2_000_000_000);
        assert_eq!(memory.usable_bytes, 94_000_000_000);
        assert_eq!(memory.ram_offload_bytes, 0);
        assert_breakdown_adds_up(&memory);
    }

    #[test]
    fn tegra_shaped_survey_reports_the_platform_share_as_platform_reserve() {
        // The Tegra collector reports physical RAM as the device memory and
        // budgets 90% of it without a driver reserve: the 10% it keeps back
        // is platform policy, not an owner setting.
        let hw = HardwareSurvey {
            vram_bytes: 57_600_000_000,
            is_soc: true,
            gpu_vram: vec![64_000_000_000],
            gpu_reserved: Vec::new(),
            system_ram_bytes: Some(64_000_000_000),
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, None, 0);

        assert_eq!(memory.total_bytes, 64_000_000_000);
        assert_eq!(memory.reserved_bytes, 0);
        assert_eq!(memory.platform_reserve_bytes, 6_400_000_000);
        assert_eq!(memory.configured_reserve_bytes, 0);
        assert_eq!(memory.usable_bytes, 57_600_000_000);
        assert_eq!(memory.ram_offload_bytes, 0);
        assert_breakdown_adds_up(&memory);

        // The owner's margin and cap still land in the configured reserve,
        // on top of the platform share.
        let memory = advertised_memory(&hw, Some(32.0), 2_000_000_000);

        assert_eq!(memory.platform_reserve_bytes, 6_400_000_000);
        assert_eq!(memory.usable_bytes, 30_000_000_000);
        assert_eq!(memory.configured_reserve_bytes, 27_600_000_000);
        assert_breakdown_adds_up(&memory);
    }

    #[test]
    fn cpu_only_host_without_cap_advertises_an_empty_breakdown() {
        // No accelerator memory to itemize; the RAM-backed local budget is
        // still reported, as an informational item.
        let hw = HardwareSurvey {
            vram_bytes: 24_000_000_000,
            system_ram_bytes: Some(32_000_000_000),
            ram_offload_bytes: 24_000_000_000,
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, None, 2_000_000_000);

        assert_eq!(
            memory,
            AdvertisedMemory {
                total_bytes: 0,
                reserved_bytes: 0,
                platform_reserve_bytes: 0,
                configured_reserve_bytes: 0,
                usable_bytes: 0,
                system_ram_bytes: Some(32_000_000_000),
                ram_offload_bytes: 24_000_000_000,
            }
        );
    }

    #[test]
    fn explicit_cpu_budget_is_the_whole_total_of_a_cpu_only_host() {
        // A 1 GB bounded budget under a 2 GB margin: nothing usable, and the
        // whole bounded budget counts as withheld by the owner.
        let hw = HardwareSurvey {
            vram_bytes: 16_000_000_000,
            ..HardwareSurvey::default()
        };

        let memory = advertised_memory(&hw, Some(1.0), 2_000_000_000);

        assert_eq!(memory.total_bytes, 1_000_000_000);
        assert_eq!(memory.usable_bytes, 0);
        assert_eq!(memory.configured_reserve_bytes, 1_000_000_000);
        assert_breakdown_adds_up(&memory);
    }

    #[test]
    fn snapshot_carries_the_breakdown_next_to_the_budget() {
        let hw = HardwareSurvey {
            vram_bytes: 30_000_000_000,
            gpu_vram: vec![12_000_000_000],
            gpu_reserved: vec![Some(500_000_000)],
            gpus: vec![gpu(12_000_000_000, Some(500_000_000), false)],
            ..HardwareSurvey::default()
        };

        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, None, 2_000_000_000);

        assert_eq!(snapshot.vram_bytes, 11_500_000_000);
        assert_eq!(snapshot.memory.total_bytes, 12_000_000_000);
        assert_eq!(snapshot.memory.usable_bytes, 9_500_000_000);
    }

    fn assert_breakdown_adds_up(memory: &AdvertisedMemory) {
        assert_eq!(
            memory.total_bytes,
            memory.reserved_bytes
                + memory.platform_reserve_bytes
                + memory.configured_reserve_bytes
                + memory.usable_bytes
        );
    }
}
