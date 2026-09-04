use super::*;
use serial_test::serial;

fn synthetic_gpu(index: usize, stable_id: Option<&str>) -> GpuFacts {
    GpuFacts {
        index,
        display_name: format!("GPU {index}"),
        backend_device: Some(format!("CUDA{index}")),
        vram_bytes: 24_000_000_000,
        reserved_bytes: None,
        mem_bandwidth_gbps: None,
        compute_tflops_fp32: None,
        compute_tflops_fp16: None,
        unified_memory: false,
        stable_id: stable_id.map(str::to_string),
        pci_bdf: None,
        vendor_uuid: None,
        metal_registry_id: None,
        dxgi_luid: None,
        pnp_instance_id: None,
    }
}

fn synthetic_gpu_with_ids(
    index: usize,
    stable_id: Option<&str>,
    pci_bdf: Option<&str>,
    vendor_uuid: Option<&str>,
) -> GpuFacts {
    GpuFacts {
        pci_bdf: pci_bdf.map(str::to_string),
        vendor_uuid: vendor_uuid.map(str::to_string),
        ..synthetic_gpu(index, stable_id)
    }
}

#[test]
fn test_parse_vulkaninfo_summary_devices() {
    let fixture = r#"
Devices:
========
GPU0:
    vendorID           = 0x10de
    deviceID           = 0x2c02
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = NVIDIA GeForce RTX 5080
    deviceUUID         = 459fc93e-aae5-c491-2080-5b93901cdcce
GPU1:
    vendorID           = 0x1002
    deviceID           = 0x164e
    deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    deviceName         = AMD Ryzen 7 7800X3D 8-Core Processor (RADV RAPHAEL_MENDOCINO)
    deviceUUID         = 00000000-0c00-0000-0000-000000000000
GPU2:
    deviceType         = PHYSICAL_DEVICE_TYPE_CPU
    deviceName         = llvmpipe
"#;

    let devices = parse_vulkaninfo_summary_devices(fixture);

    assert_eq!(devices.len(), 3);
    assert_eq!(devices[0].index, 0);
    assert_eq!(devices[0].display_name, "NVIDIA GeForce RTX 5080");
    assert_eq!(devices[1].index, 1);
    assert_eq!(
        devices[1].device_uuid.as_deref(),
        Some("00000000-0c00-0000-0000-000000000000")
    );
}

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
fn test_parse_nvidia_gpu_memory() {
    assert_eq!(
        parse_nvidia_gpu_memory("81920\n24576\n"),
        vec![81_920u64 * 1024 * 1024, 24_576u64 * 1024 * 1024]
    );
}

#[test]
fn test_parse_nvidia_gpu_memory_and_reserved() {
    assert_eq!(
        parse_nvidia_gpu_memory_and_reserved("81920,1024\n24576,0\n"),
        vec![
            (81_920u64 * 1024 * 1024, Some(1_024u64 * 1024 * 1024)),
            (24_576u64 * 1024 * 1024, Some(0)),
        ]
    );
}

/// `sanitize_macos_gpu_name` passes through real GPU names untouched (modulo
/// trimming), for both Apple Silicon and AMD/Intel discrete/integrated parts.
#[test]
fn test_sanitize_macos_gpu_name_accepts_real_gpu_names() {
    assert_eq!(
        sanitize_macos_gpu_name(Some("Apple M4 Max\n".to_string())),
        Some("Apple M4 Max".to_string())
    );
    assert_eq!(
        sanitize_macos_gpu_name(Some("AMD Radeon Pro 5500M".to_string())),
        Some("AMD Radeon Pro 5500M".to_string())
    );
    assert_eq!(
        sanitize_macos_gpu_name(Some("Intel(R) UHD Graphics 630".to_string())),
        Some("Intel(R) UHD Graphics 630".to_string())
    );
}

/// Empty or missing input degrades to labeled-absent (`None`), not an empty
/// string masquerading as a GPU name.
#[test]
fn test_sanitize_macos_gpu_name_empty_is_labeled_absent() {
    assert_eq!(sanitize_macos_gpu_name(Some(String::new())), None);
    assert_eq!(sanitize_macos_gpu_name(None), None);
}

/// Guards against the regression this fix removed: a CPU brand string routed
/// into the GPU-name path must degrade to `None`, never surface mislabeled.
#[test]
fn test_sanitize_macos_gpu_name_flags_cpu_brand_string_mislabel() {
    // The exact `sysctl -n machdep.cpu.brand_string` shape this fix removed —
    // must never surface as a GPU name again.
    assert_eq!(
        sanitize_macos_gpu_name(Some("Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz".to_string())),
        None
    );
}

#[test]
fn test_macos_gpu_budget_uses_metal_working_set_as_vram() {
    let metal_bytes = 107_u64 * 1024 * 1024 * 1024;

    assert_eq!(
        macos_metal_gpu_budget(Some(metal_bytes)),
        Some((metal_bytes, None))
    );
}

#[test]
fn test_macos_gpu_budget_is_unavailable_without_metal_working_set() {
    assert_eq!(macos_metal_gpu_budget(Some(0)), None);
    assert_eq!(macos_metal_gpu_budget(None), None);
}

#[test]
fn test_parse_rocm_gpu_names_single() {
    let fixture = "\
======================= ROCm System Management Interface =======================
================================= Product Info =================================
GPU[0]\t\t: Card Series:\t\t\tNavi31 [Radeon RX 7900 XTX]
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
fn test_rocm_tiny_vram_large_gtt_is_unified_memory() {
    let vram = vec![536_870_912];
    let gtt = vec![137_438_953_472];
    let system_ram = 137_438_953_472;

    assert_eq!(
        rocm_unified_memory_usable_bytes(&vram, &gtt, system_ram),
        Some(123_695_058_124)
    );
}

#[test]
fn test_rocm_discrete_vram_large_gtt_is_not_unified_memory() {
    let vram = vec![24 * 1024 * 1024 * 1024];
    let gtt = vec![137_438_953_472];
    let system_ram = 137_438_953_472;

    assert_eq!(
        rocm_unified_memory_usable_bytes(&vram, &gtt, system_ram),
        None
    );
}

#[test]
fn test_parse_rocm_gpu_memory_and_used() {
    let fixture = "\
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,25753026560,416378880
card1,25753026560,512000000";
    assert_eq!(
        parse_rocm_gpu_memory_and_used(fixture),
        vec![
            (25_753_026_560, Some(416_378_880)),
            (25_753_026_560, Some(512_000_000)),
        ]
    );
}

#[test]
fn test_parse_rocm_gpu_memory_and_used_ignores_warning_preamble() {
    let fixture = "\
WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status

device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,206158430208,0";
    assert_eq!(
        parse_rocm_gpu_memory_and_used(fixture),
        vec![(206_158_430_208, Some(0))]
    );
}

#[test]
fn test_parse_xpu_smi_discovery_json() {
    let fixture = r#"{
          "devices": [
            {
              "device_name": "Intel Arc A770",
              "memory_physical_size_byte": 17179869184,
              "memory_used_byte": 536870912
            },
            {
              "device_name": "Intel Arc B580",
              "memory_physical_size_byte": "12884901888",
              "memory_used_byte": "268435456"
            }
          ]
        }"#;
    assert_eq!(
        parse_xpu_smi_discovery_json(fixture),
        vec![
            XpuSmiGpuInfo {
                name: "Intel Arc A770".to_string(),
                total_bytes: Some(17_179_869_184),
                used_bytes: Some(536_870_912),
            },
            XpuSmiGpuInfo {
                name: "Intel Arc B580".to_string(),
                total_bytes: Some(12_884_901_888),
                used_bytes: Some(268_435_456),
            },
        ]
    );
}

#[test]
fn test_rocm_used_memory_does_not_surface_as_reserved_bytes() {
    let fixture = "\
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,25753026560,416378880
card1,25753026560,512000000";
    let parsed = parse_rocm_gpu_memory_and_used(fixture);
    let mut survey = HardwareSurvey {
        gpu_vram: parsed.iter().map(|(total, _)| *total).collect(),
        gpu_reserved: vec![None; parsed.len()],
        ..Default::default()
    };

    hydrate_gpu_facts(&mut survey, &[Metric::GpuFacts]);

    assert_eq!(survey.gpus.len(), 2);
    assert!(survey.gpus.iter().all(|gpu| gpu.reserved_bytes.is_none()));
}

#[test]
fn test_xpu_used_memory_does_not_surface_as_reserved_bytes() {
    let fixture = r#"{
          "devices": [
            {
              "device_name": "Intel Arc A770",
              "memory_physical_size_byte": 17179869184,
              "memory_used_byte": 536870912
            },
            {
              "device_name": "Intel Arc B580",
              "memory_physical_size_byte": "12884901888",
              "memory_used_byte": "268435456"
            }
          ]
        }"#;
    let gpus = parse_xpu_smi_discovery_json(fixture);
    let mut survey = HardwareSurvey {
        gpu_vram: gpus
            .iter()
            .map(|gpu| gpu.total_bytes.unwrap_or(0))
            .collect(),
        gpu_reserved: vec![None; gpus.len()],
        ..Default::default()
    };

    hydrate_gpu_facts(&mut survey, &[Metric::GpuFacts]);

    assert_eq!(survey.gpus.len(), 2);
    assert!(survey.gpus.iter().all(|gpu| gpu.reserved_bytes.is_none()));
}

#[test]
fn test_hydrate_gpu_facts_uses_uuid_and_cuda_for_tegra_soc() {
    let mut survey = HardwareSurvey {
        gpu_name: Some("Jetson AGX Orin".to_string()),
        gpu_count: 1,
        gpu_vram: vec![65_890_271_232],
        is_soc: true,
        ..Default::default()
    };
    let identities = vec![(
        None,
        Some("ddae9891-aaa8-5edd-bbf3-3a33c5adc75f".to_string()),
    )];
    let expected_count = survey
        .gpu_vram
        .len()
        .max(usize::from(survey.gpu_count))
        .max(inferred_gpu_name_count(survey.gpu_name.as_deref()));
    let names = expand_gpu_names(survey.gpu_name.as_deref(), expected_count);

    hydrate_gpu_facts_with_identities(
        &mut survey,
        &[Metric::GpuFacts],
        &identities,
        names,
        expected_count,
        false,
    );

    assert_eq!(survey.gpus.len(), 1);
    assert_eq!(survey.gpus[0].display_name, "Jetson AGX Orin");
    assert_eq!(survey.gpus[0].backend_device.as_deref(), Some("CUDA0"));
    assert_eq!(
        survey.gpus[0].stable_id.as_deref(),
        Some("uuid:ddae9891-aaa8-5edd-bbf3-3a33c5adc75f")
    );
    assert_eq!(survey.gpus[0].pci_bdf, None);
    assert_eq!(
        survey.gpus[0].vendor_uuid.as_deref(),
        Some("ddae9891-aaa8-5edd-bbf3-3a33c5adc75f")
    );
    assert!(survey.gpus[0].unified_memory);
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
fn test_expand_gpu_names_identical_summary() {
    assert_eq!(
        expand_gpu_names(Some("2× NVIDIA A100"), 2),
        vec!["NVIDIA A100".to_string(), "NVIDIA A100".to_string()]
    );
}

#[test]
fn test_expand_gpu_names_mixed_summary() {
    assert_eq!(
        expand_gpu_names(Some("NVIDIA A100, NVIDIA RTX 4090"), 2),
        vec!["NVIDIA A100".to_string(), "NVIDIA RTX 4090".to_string()]
    );
}

#[test]
fn test_parse_nvidia_gpu_identity_rows() {
    let identities =
        parse_nvidia_gpu_identity("00000000:65:00.0, GPU-abc\n00000000:b3:00.0, GPU-def\n");
    assert_eq!(
        identities,
        vec![
            (
                Some("00000000:65:00.0".to_string()),
                Some("GPU-abc".to_string())
            ),
            (
                Some("00000000:b3:00.0".to_string()),
                Some("GPU-def".to_string())
            )
        ]
    );
}

#[test]
fn test_parse_nvidia_gpu_identity_ignores_not_available_placeholders() {
    let identities = parse_nvidia_gpu_identity("[N/A], ddae9891-aaa8-5edd-bbf3-3a33c5adc75f\n");
    assert_eq!(
        identities,
        vec![(
            None,
            Some("ddae9891-aaa8-5edd-bbf3-3a33c5adc75f".to_string())
        )]
    );
}

#[test]
fn test_hydrate_prefers_uuid_over_placeholder_pci() {
    let mut survey = HardwareSurvey {
        is_soc: true,
        gpu_vram: vec![64 * 1024 * 1024 * 1024],
        gpu_reserved: vec![None],
        gpu_count: 1,
        gpu_name: Some("Jetson AGX Orin".into()),
        ..Default::default()
    };

    hydrate_gpu_facts_with_identities(
        &mut survey,
        &[Metric::GpuFacts],
        &[(
            Some("00000000:00:00.0".to_string()),
            Some("ddae9891-aaa8-5edd-bbf3-3a33c5adc75f".to_string()),
        )],
        vec!["Jetson AGX Orin".to_string()],
        1,
        false,
    );

    assert_eq!(
        survey.gpus[0].stable_id.as_deref(),
        Some("uuid:ddae9891-aaa8-5edd-bbf3-3a33c5adc75f")
    );
}

#[test]
fn test_backend_device_for_name_recognizes_jetson_soc_names() {
    assert_eq!(
        backend_device_for_name_for_platform("Jetson AGX Orin", 0, true, false),
        Some("CUDA0".to_string())
    );
}

#[test]
fn test_backend_device_for_name_recognizes_nvgpu_soc_names() {
    assert_eq!(
        backend_device_for_name_for_platform("Orin (nvgpu)", 1, true, false),
        Some("CUDA1".to_string())
    );
}

#[test]
fn pinned_gpu_runtime_resolver_accepts_single_match() {
    let gpus = vec![
        synthetic_gpu(0, Some("pci:0000:65:00.0")),
        synthetic_gpu(1, Some("uuid:GPU-def")),
    ];

    let resolved = resolve_pinned_gpu(Some("pci:0000:65:00.0"), &gpus).unwrap();

    assert_eq!(resolved.index, 0);
    assert_eq!(resolved.stable_id.as_deref(), Some("pci:0000:65:00.0"));
}

#[test]
fn pinned_gpu_runtime_resolver_missing_configured_id_fails() {
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"))];

    let err = resolve_pinned_gpu(None, &gpus).unwrap_err();

    assert_eq!(
        err,
        PinnedGpuResolverError::MissingConfiguredId {
            available_pinnable_ids: vec!["pci:0000:65:00.0".to_string()],
        }
    );
    assert!(
        err.to_string()
            .contains("available pinnable GPU IDs: pci:0000:65:00.0")
    );
}

#[test]
fn pinned_gpu_runtime_resolver_no_match_lists_available_ids() {
    let gpus = vec![
        synthetic_gpu(0, Some("pci:0000:65:00.0")),
        synthetic_gpu(1, Some("uuid:GPU-def")),
    ];

    let err = resolve_pinned_gpu(Some("pci:0000:b3:00.0"), &gpus).unwrap_err();

    assert_eq!(
        err,
        PinnedGpuResolverError::NoMatch {
            configured_id: "pci:0000:b3:00.0".to_string(),
            available_pinnable_ids: vec![
                "pci:0000:65:00.0".to_string(),
                "uuid:GPU-def".to_string(),
            ],
        }
    );
    assert!(err.to_string().contains("pci:0000:b3:00.0"));
    assert!(err.to_string().contains("pci:0000:65:00.0, uuid:GPU-def"));
}

#[test]
fn pinned_gpu_runtime_resolver_matches_vendor_uuid_alias() {
    let gpus = vec![synthetic_gpu_with_ids(
        0,
        Some("pci:0000:65:00.0"),
        Some("0000:65:00.0"),
        Some("GPU-def"),
    )];

    let resolved = resolve_pinned_gpu(Some("uuid:GPU-def"), &gpus).unwrap();

    assert_eq!(resolved.index, 0);
}

#[test]
fn pinned_gpu_runtime_resolver_accepts_single_pinnable_gpu_for_legacy_alias() {
    let gpus = vec![synthetic_gpu_with_ids(
        0,
        Some("pci:00000000:00:00.0"),
        Some("00000000:00:00.0"),
        None,
    )];

    let resolved =
        resolve_pinned_gpu(Some("uuid:ddae9891-aaa8-5edd-bbf3-3a33c5adc75f"), &gpus).unwrap();

    assert_eq!(resolved.index, 0);
}

#[test]
fn pinned_gpu_runtime_resolver_duplicate_match_fails() {
    let gpus = vec![
        synthetic_gpu(0, Some("uuid:GPU-shared")),
        synthetic_gpu(1, Some("uuid:GPU-shared")),
    ];

    let err = resolve_pinned_gpu(Some("uuid:GPU-shared"), &gpus).unwrap_err();

    assert_eq!(
        err,
        PinnedGpuResolverError::AmbiguousMatch {
            configured_id: "uuid:GPU-shared".to_string(),
            available_pinnable_ids: vec![
                "uuid:GPU-shared".to_string(),
                "uuid:GPU-shared".to_string(),
            ],
            match_indexes: vec![0, 1],
        }
    );
    assert!(err.to_string().contains("indexes [0, 1]"));
}

#[test]
fn pinned_gpu_runtime_resolver_rejects_index_fallback_ids() {
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"))];

    let err = resolve_pinned_gpu(Some("index:0"), &gpus).unwrap_err();

    assert_eq!(
        err,
        PinnedGpuResolverError::NonPinnableConfiguredId {
            configured_id: "index:0".to_string(),
            available_pinnable_ids: vec!["pci:0000:65:00.0".to_string()],
        }
    );
    assert!(err.to_string().contains("not pinnable"));
}

#[test]
fn pinned_gpu_runtime_resolver_rejects_backend_device_fallback_ids() {
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"))];

    let err = resolve_pinned_gpu(Some("cuda0"), &gpus).unwrap_err();

    assert_eq!(
        err,
        PinnedGpuResolverError::NonPinnableConfiguredId {
            configured_id: "cuda0".to_string(),
            available_pinnable_ids: vec!["pci:0000:65:00.0".to_string()],
        }
    );
}

#[test]
fn pinned_gpu_runtime_resolver_fails_when_host_has_no_pinnable_gpus() {
    let gpus = vec![
        synthetic_gpu(0, Some("cuda0")),
        synthetic_gpu(1, Some("index:1")),
    ];

    let err = resolve_pinned_gpu(Some("pci:0000:65:00.0"), &gpus).unwrap_err();

    assert_eq!(
        err,
        PinnedGpuResolverError::NoPinnableGpus {
            configured_id: "pci:0000:65:00.0".to_string(),
            available_pinnable_ids: vec![],
        }
    );
    assert!(err.to_string().contains("available pinnable GPU IDs: none"));
}

#[test]
fn test_hardware_survey_default() {
    let s = HardwareSurvey::default();
    assert_eq!(s.vram_bytes, 0);
    assert_eq!(s.gpu_name, None);
    assert_eq!(s.gpu_count, 0);
    assert_eq!(s.hostname, None);
    assert!(s.gpu_vram.is_empty());
    assert!(s.gpu_reserved.is_empty());
    assert!(s.gpus.is_empty());
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[test]
fn test_cpu_only_runtime_budget_uses_system_ram_when_vram_requested() {
    let mut survey = HardwareSurvey::default();

    apply_cpu_only_runtime_budget(&mut survey, &[Metric::VramBytes], || 16_000_000_000);

    assert_eq!(survey.vram_bytes, 12_000_000_000);
    assert!(survey.gpu_vram.is_empty());
    assert!(survey.gpus.is_empty());
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[test]
fn test_cpu_only_runtime_budget_respects_requested_metrics() {
    let mut survey = HardwareSurvey::default();

    apply_cpu_only_runtime_budget(&mut survey, &[Metric::GpuName], || {
        panic!("system RAM must not be probed when VramBytes is not requested")
    });

    assert_eq!(survey.vram_bytes, 0);
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[test]
fn test_probe_error_applies_cpu_only_budget_only_when_vram_requested() {
    let mut survey = HardwareSurvey::default();
    let handled = apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::VramBytes],
        Err::<Vec<GpuFacts>, ()>(()),
        || 16_000_000_000,
    );
    assert!(handled);
    assert_eq!(survey.vram_bytes, 12_000_000_000);
    assert!(survey.gpu_vram.is_empty());
    assert!(survey.gpus.is_empty());

    let mut survey = HardwareSurvey::default();
    let handled = apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::GpuName],
        Err::<Vec<GpuFacts>, ()>(()),
        || 16_000_000_000,
    );
    assert!(handled);
    assert_eq!(survey.vram_bytes, 0);
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[test]
fn test_empty_gpu_probe_applies_cpu_only_budget_only_when_vram_requested() {
    let mut survey = HardwareSurvey::default();
    let handled = apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::VramBytes],
        Ok::<Vec<GpuFacts>, ()>(Vec::new()),
        || 16_000_000_000,
    );
    assert!(handled);
    assert_eq!(survey.vram_bytes, 12_000_000_000);
    assert!(survey.gpu_vram.is_empty());
    assert!(survey.gpus.is_empty());

    let mut survey = HardwareSurvey::default();
    let handled = apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::GpuName],
        Ok::<Vec<GpuFacts>, ()>(Vec::new()),
        || 16_000_000_000,
    );
    assert!(handled);
    assert_eq!(survey.vram_bytes, 0);
}

#[test]
fn test_healthy_gpu_probe_credits_ram_offload_from_injected_source() {
    // 12 GB dGPU in a 32 GB machine → 12 + 0.90 × 20 = 30 GB advertised,
    // matching both the Linux behavior of this seam and the legacy Windows
    // collector path.
    let mut survey = HardwareSurvey::default();
    let mut gpu = synthetic_gpu(0, None);
    gpu.vram_bytes = 12_000_000_000;
    let handled = apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::VramBytes],
        Ok::<Vec<GpuFacts>, ()>(vec![gpu]),
        || 32_000_000_000,
    );
    assert!(handled);
    assert_eq!(survey.vram_bytes, 30_000_000_000);
    assert_eq!(survey.gpu_vram, vec![12_000_000_000]);
}

#[test]
fn test_healthy_gpu_probe_with_zero_ram_source_advertises_bare_vram() {
    // A zero RAM source (platforms without a reader) keeps the pre-existing
    // behavior: bare VRAM, no offload credit.
    let mut survey = HardwareSurvey::default();
    let mut gpu = synthetic_gpu(0, None);
    gpu.vram_bytes = 12_000_000_000;
    let handled = apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::VramBytes],
        Ok::<Vec<GpuFacts>, ()>(vec![gpu]),
        || 0,
    );
    assert!(handled);
    assert_eq!(survey.vram_bytes, 12_000_000_000);
}

#[test]
fn test_healthy_gpu_probe_ram_below_vram_saturates_to_bare_vram() {
    let mut survey = HardwareSurvey::default();
    let mut gpu = synthetic_gpu(0, None);
    gpu.vram_bytes = 12_000_000_000;
    let handled = apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::VramBytes],
        Ok::<Vec<GpuFacts>, ()>(vec![gpu]),
        || 8_000_000_000,
    );
    assert!(handled);
    assert_eq!(survey.vram_bytes, 12_000_000_000);
}

#[test]
fn test_healthy_soc_probe_does_not_probe_system_ram() {
    let mut survey = HardwareSurvey::default();
    let mut gpu = synthetic_gpu(0, None);
    gpu.vram_bytes = 16_000_000_000;
    gpu.reserved_bytes = Some(2_000_000_000);
    gpu.unified_memory = true;
    let handled = apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::IsSoc, Metric::VramBytes],
        Ok::<Vec<GpuFacts>, ()>(vec![gpu]),
        || panic!("system RAM must not be probed for a unified-memory survey"),
    );
    assert!(handled);
    assert_eq!(survey.vram_bytes, 14_000_000_000);
}

#[test]
fn test_healthy_soc_probe_without_is_soc_metric_still_skips_ram_offload() {
    let mut survey = HardwareSurvey::default();
    let mut gpu = synthetic_gpu(0, None);
    gpu.vram_bytes = 16_000_000_000;
    gpu.reserved_bytes = Some(2_000_000_000);
    gpu.unified_memory = true;
    let handled = apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::VramBytes],
        Ok::<Vec<GpuFacts>, ()>(vec![gpu]),
        || panic!("system RAM must not be probed for unified memory, even without IsSoc"),
    );
    assert!(handled);
    assert!(!survey.is_soc);
    assert_eq!(survey.vram_bytes, 14_000_000_000);
}

#[test]
fn test_healthy_gpu_probe_without_vram_metric_does_not_probe_system_ram() {
    let mut survey = HardwareSurvey::default();
    let handled = apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::GpuName],
        Ok::<Vec<GpuFacts>, ()>(vec![synthetic_gpu(0, None)]),
        || panic!("system RAM must not be probed when VramBytes is not requested"),
    );
    assert!(handled);
    assert_eq!(survey.vram_bytes, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn test_probe_fallback_leaves_vram_untouched_on_macos() {
    let mut survey = HardwareSurvey::default();
    assert!(apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::VramBytes],
        Err::<Vec<GpuFacts>, ()>(()),
        || 16_000_000_000,
    ));
    assert_eq!(survey.vram_bytes, 0);

    let mut survey = HardwareSurvey::default();
    assert!(apply_gpu_probe_outcome_to_survey(
        &mut survey,
        &[Metric::VramBytes],
        Ok::<Vec<GpuFacts>, ()>(Vec::new()),
        || 16_000_000_000,
    ));
    assert_eq!(survey.vram_bytes, 0);
}

// Keep real-collector tests in one group so the Linux-only process-global
// skippy probe override cannot be consumed by an unrelated query test.
#[cfg(all(target_os = "linux", feature = "skippy-devices"))]
#[test]
#[serial(real_collector)]
fn test_skippy_backend_error_uses_cpu_only_budget_without_legacy_fallback() {
    skippy_devices::set_test_gpu_facts_result(Err(anyhow::anyhow!("boom")));

    let mut survey = HardwareSurvey::default();
    let handled = apply_skippy_backend_devices_to_survey(
        &mut survey,
        &[Metric::GpuName, Metric::GpuCount, Metric::VramBytes],
    );

    skippy_devices::clear_test_gpu_facts_result();

    assert!(handled);
    assert_eq!(survey.gpu_name, None);
    assert_eq!(survey.gpu_count, 0);
    assert!(survey.gpu_vram.is_empty());
    assert!(survey.gpus.is_empty());
    assert!(survey.vram_bytes > 0);
}

#[cfg(all(target_os = "linux", feature = "skippy-devices"))]
#[test]
#[serial(real_collector)]
fn test_skippy_backend_empty_result_uses_cpu_only_budget_without_legacy_fallback() {
    skippy_devices::set_test_gpu_facts_result(Ok(vec![]));

    let mut survey = HardwareSurvey::default();
    let handled = apply_skippy_backend_devices_to_survey(
        &mut survey,
        &[
            Metric::GpuName,
            Metric::GpuCount,
            Metric::VramBytes,
            Metric::GpuFacts,
        ],
    );

    skippy_devices::clear_test_gpu_facts_result();

    assert!(handled);
    assert_eq!(survey.gpu_name, None);
    assert_eq!(survey.gpu_count, 0);
    assert!(survey.gpu_vram.is_empty());
    assert!(survey.gpus.is_empty());
    assert!(survey.vram_bytes > 0);
}

#[test]
#[serial(real_collector)]
fn test_query_gpu_name_only() {
    let result = query(&[Metric::GpuName]);
    assert_eq!(result.vram_bytes, 0);
    assert_eq!(result.hostname, None);
}

#[test]
#[serial(real_collector)]
fn test_query_vram_only() {
    let result = query(&[Metric::VramBytes]);
    assert_eq!(result.gpu_name, None);
    assert_eq!(result.hostname, None);
}

#[test]
#[serial(real_collector)]
fn test_query_multiple_metrics() {
    let result = query(&[Metric::GpuName, Metric::VramBytes]);
    assert_eq!(result.hostname, None);
    assert_eq!(result.gpu_count, 0);
}

#[test]
#[serial(real_collector)]
fn test_survey_returns_all_metrics() {
    let s = survey();
    let q = query(&[
        Metric::GpuName,
        Metric::VramBytes,
        Metric::GpuCount,
        Metric::Hostname,
        Metric::IsSoc,
        Metric::GpuFacts,
    ]);
    assert_eq!(s.vram_bytes, q.vram_bytes);
    assert_eq!(s.gpu_name, q.gpu_name);
    assert_eq!(s.gpu_count, q.gpu_count);
    assert_eq!(s.hostname.is_some(), q.hostname.is_some());
    assert_eq!(s.is_soc, q.is_soc);
    assert_eq!(s.gpu_vram, q.gpu_vram);
    assert_eq!(s.gpu_reserved, q.gpu_reserved);
    assert_eq!(s.gpus, q.gpus);
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
fn test_parse_windows_video_controller_json_array() {
    let json = r#"[{"Name":"NVIDIA RTX 4090","AdapterRAM":25769803776},{"Name":"AMD Radeon PRO","AdapterRAM":"8589934592"}]"#;
    assert_eq!(
        parse_windows_video_controller_json(json),
        vec![
            ("NVIDIA RTX 4090".to_string(), 25_769_803_776),
            ("AMD Radeon PRO".to_string(), 8_589_934_592),
        ]
    );
}

#[test]
fn test_parse_windows_video_controller_json_single_object() {
    let json = r#"{"Name":"NVIDIA RTX 5090","AdapterRAM":34359738368}"#;
    assert_eq!(
        parse_windows_video_controller_json(json),
        vec![("NVIDIA RTX 5090".to_string(), 34_359_738_368)]
    );
}

#[test]
fn test_windows_per_gpu_vram_keeps_zero_vram_adapters_aligned() {
    let controllers = vec![
        ("Virtual Display Adapter".to_string(), 0),
        ("AMD Radeon RX 9070 XT".to_string(), 4_293_918_720),
    ];

    assert_eq!(
        windows_per_gpu_vram(&controllers),
        vec![0, 4_293_918_720],
        "adapters reporting no VRAM must keep a slot so later adapters stay aligned"
    );
}

#[test]
fn test_windows_per_gpu_vram_keeps_all_zero_adapters_aligned() {
    let controllers = vec![
        ("Virtual Display Adapter".to_string(), 0),
        ("Headless Display Adapter".to_string(), 0),
    ];

    assert_eq!(windows_per_gpu_vram(&controllers), vec![0, 0]);
}

#[test]
fn test_windows_zero_vram_adapter_does_not_shift_vram_onto_wrong_gpu() {
    let controllers = vec![
        ("Virtual Display Adapter".to_string(), 0),
        ("AMD Radeon RX 9070 XT".to_string(), 4_293_918_720),
    ];
    let names: Vec<String> = controllers.iter().map(|(name, _)| name.clone()).collect();
    let mut survey = HardwareSurvey {
        gpu_count: 2,
        gpu_vram: windows_per_gpu_vram(&controllers),
        ..Default::default()
    };

    hydrate_gpu_facts_with_identities(&mut survey, &[Metric::GpuFacts], &[], names, 2, false);

    assert_eq!(survey.gpus.len(), 2);
    assert_eq!(survey.gpus[0].display_name, "Virtual Display Adapter");
    assert_eq!(survey.gpus[0].vram_bytes, 0);
    assert_eq!(survey.gpus[1].display_name, "AMD Radeon RX 9070 XT");
    assert_eq!(survey.gpus[1].vram_bytes, 4_293_918_720);
}

#[test]
fn test_parse_windows_total_physical_memory() {
    assert_eq!(
        parse_windows_total_physical_memory("68719476736\r\n"),
        Some(68_719_476_736)
    );
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
#[serial(real_collector)]
fn test_detect_collector_returns_default_on_non_tegra() {
    let collector = detect_collector();
    let s = collector.collect(&[Metric::VramBytes]);
    let _ = s.vram_bytes;
}

#[test]
#[serial(real_collector)]
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
#[serial(real_collector)]
fn test_macos_is_soc_true() {
    let result = DefaultCollector.collect(&[Metric::IsSoc]);
    assert!(
        result.is_soc,
        "macOS DefaultCollector must report is_soc=true"
    );
}

#[cfg(target_os = "linux")]
#[test]
#[serial(real_collector)]
fn test_tegra_is_soc_true() {
    let result = TegraCollector.collect(&[Metric::IsSoc]);
    assert!(result.is_soc, "TegraCollector must report is_soc=true");
}

#[cfg(target_os = "linux")]
#[test]
#[serial(real_collector)]
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
