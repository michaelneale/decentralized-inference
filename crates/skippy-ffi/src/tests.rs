use std::mem::{offset_of, size_of};

use crate::{
    ABI_VERSION_MAJOR, ABI_VERSION_MINOR, ABI_VERSION_PATCH, AbiVersion, ActivationBoundaryDesc,
    StagePlanDescV1, StagePlanProfileDescV1, StagePlanStateDescV1, StagePlanStateKind,
    StagePlanStringRefV1, StagePlanValueDescV1, StagePlannerConfigV1, StagePlannerProfileV1,
    StagePlannerTensorV1, runtime_abi_supported,
};

#[cfg(target_pointer_width = "64")]
use crate::{
    ModelImatrixEntryV1, ModelTensorSourceV1, MtmdContextParams, MtmdHelperBitmapWrapper,
    MtmdHelperInitOpt, MtmdHelperVideoInitParams, MtmdInputText,
};

#[cfg(not(feature = "dynamic-runtime"))]
use crate::mtmd_context_params_default;

const fn version(major: u32, minor: u32, patch: u32) -> AbiVersion {
    AbiVersion {
        major,
        minor,
        patch,
    }
}

#[test]
fn accepts_current_patch_runtime() {
    assert!(runtime_abi_supported(version(
        ABI_VERSION_MAJOR,
        ABI_VERSION_MINOR,
        ABI_VERSION_PATCH,
    )));
}

#[test]
fn rejects_other_patch_runtimes() {
    assert!(!runtime_abi_supported(version(
        ABI_VERSION_MAJOR,
        ABI_VERSION_MINOR,
        ABI_VERSION_PATCH + 1,
    )));
    if let Some(lower_patch) = ABI_VERSION_PATCH.checked_sub(1) {
        assert!(!runtime_abi_supported(version(
            ABI_VERSION_MAJOR,
            ABI_VERSION_MINOR,
            lower_patch,
        )));
    }
}

#[test]
fn rejects_major_and_minor_mismatches() {
    assert!(!runtime_abi_supported(version(
        ABI_VERSION_MAJOR + 1,
        ABI_VERSION_MINOR,
        ABI_VERSION_PATCH,
    )));
    assert!(!runtime_abi_supported(version(
        ABI_VERSION_MAJOR,
        ABI_VERSION_MINOR + 1,
        ABI_VERSION_PATCH,
    )));
}

#[test]
fn activation_boundary_descriptor_matches_native_layout() {
    assert_eq!(size_of::<ActivationBoundaryDesc>(), 48);
    assert_eq!(offset_of!(ActivationBoundaryDesc, version), 0);
    assert_eq!(offset_of!(ActivationBoundaryDesc, ggml_type), 4);
    assert_eq!(offset_of!(ActivationBoundaryDesc, layout), 8);
    assert_eq!(offset_of!(ActivationBoundaryDesc, reserved), 12);
    assert_eq!(offset_of!(ActivationBoundaryDesc, elements_per_token), 16);
    assert_eq!(offset_of!(ActivationBoundaryDesc, bytes_per_token), 24);
    assert_eq!(offset_of!(ActivationBoundaryDesc, required_frame_flags), 32);
    assert_eq!(offset_of!(ActivationBoundaryDesc, required_sidebands), 40);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn stage_plan_types_match_native_layout() {
    assert_eq!(StagePlanStateKind::DerivedPersistent as i32, 4);
    assert_eq!(size_of::<StagePlanStringRefV1>(), 16);
    assert_eq!(size_of::<StagePlannerTensorV1>(), 88);
    assert_eq!(offset_of!(StagePlannerTensorV1, dimensions), 32);
    assert_eq!(offset_of!(StagePlannerTensorV1, stored_length), 80);

    assert_eq!(size_of::<StagePlannerProfileV1>(), 32);
    assert_eq!(offset_of!(StagePlannerProfileV1, profile_id), 8);
    assert_eq!(size_of::<StagePlannerConfigV1>(), 80);
    assert_eq!(offset_of!(StagePlannerConfigV1, shard_paths), 16);
    assert_eq!(offset_of!(StagePlannerConfigV1, profiles), 48);
    assert_eq!(offset_of!(StagePlannerConfigV1, graph_configuration_id), 64);
    assert_eq!(offset_of!(StagePlannerConfigV1, backend_id), 72);

    assert_eq!(size_of::<StagePlanDescV1>(), 72);
    assert_eq!(offset_of!(StagePlanDescV1, profile_count), 56);
    assert_eq!(size_of::<StagePlanProfileDescV1>(), 168);
    assert_eq!(offset_of!(StagePlanProfileDescV1, n_tokens), 120);
    assert_eq!(offset_of!(StagePlanProfileDescV1, state_effect_count), 160);
    assert_eq!(size_of::<StagePlanValueDescV1>(), 24);
    assert_eq!(size_of::<StagePlanStateDescV1>(), 48);
    assert_eq!(offset_of!(StagePlanStateDescV1, write_ordinal), 40);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn model_tensor_source_types_match_native_layout() {
    assert_eq!(size_of::<ModelImatrixEntryV1>(), 24);
    assert_eq!(offset_of!(ModelImatrixEntryV1, tensor_name), 0);
    assert_eq!(offset_of!(ModelImatrixEntryV1, values), 8);
    assert_eq!(offset_of!(ModelImatrixEntryV1, value_count), 16);

    assert_eq!(size_of::<ModelTensorSourceV1>(), 40);
    assert_eq!(offset_of!(ModelTensorSourceV1, abi_version), 0);
    assert_eq!(offset_of!(ModelTensorSourceV1, struct_size), 4);
    assert_eq!(offset_of!(ModelTensorSourceV1, read_tensor_f32), 8);
    assert_eq!(offset_of!(ModelTensorSourceV1, user_data), 16);
    assert_eq!(offset_of!(ModelTensorSourceV1, imatrix), 24);
    assert_eq!(offset_of!(ModelTensorSourceV1, imatrix_count), 32);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn mtmd_context_params_matches_native_layout() {
    // Mirrors `struct mtmd_context_params` in tools/mtmd/mtmd.h. `device` sits
    // second, right after `use_gpu`; leaving it out shifts everything below it
    // by 8 bytes and makes the struct 16 bytes short, so the C side reads
    // `progress_callback` from past the end of what Rust allocated.
    assert_eq!(size_of::<MtmdContextParams>(), 96);
    assert_eq!(offset_of!(MtmdContextParams, device), 8);
    assert_eq!(offset_of!(MtmdContextParams, batch_max_tokens), 72);
    assert_eq!(offset_of!(MtmdContextParams, progress_callback), 80);
    assert_eq!(
        offset_of!(MtmdContextParams, progress_callback_user_data),
        88
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn mtmd_input_text_matches_native_layout() {
    // Mirrors `struct mtmd_input_text` in tools/mtmd/mtmd.h. `mtmd_tokenize`
    // reads `text_len` to size the prompt, so a missing field here hands the
    // native side a length built from uninitialised padding.
    assert_eq!(size_of::<MtmdInputText>(), 24);
    assert_eq!(offset_of!(MtmdInputText, text_len), 8);
    assert_eq!(offset_of!(MtmdInputText, add_special), 16);
    assert_eq!(offset_of!(MtmdInputText, parse_special), 17);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn mtmd_helper_bitmap_types_match_native_layout() {
    // Mirrors `mtmd_helper_bitmap_wrapper`, `mtmd_helper_init_opt` and
    // `mtmd_helper_video_init_params` in tools/mtmd/mtmd-helper.h.
    // `mtmd_helper_bitmap_init_from_buf` returns the wrapper by value and takes
    // the opt by value, so both layouts are part of the calling convention.
    assert_eq!(size_of::<MtmdHelperBitmapWrapper>(), 16);
    assert_eq!(offset_of!(MtmdHelperBitmapWrapper, bitmap), 0);
    assert_eq!(offset_of!(MtmdHelperBitmapWrapper, video_ctx), 8);

    assert_eq!(size_of::<MtmdHelperVideoInitParams>(), 24);
    assert_eq!(offset_of!(MtmdHelperVideoInitParams, fps_target), 0);
    assert_eq!(offset_of!(MtmdHelperVideoInitParams, ffmpeg_bin_dir), 8);
    assert_eq!(
        offset_of!(MtmdHelperVideoInitParams, timestamp_interval_ms),
        16
    );

    assert_eq!(size_of::<MtmdHelperInitOpt>(), 24);
    assert_eq!(offset_of!(MtmdHelperInitOpt, video_params), 0);
}

#[test]
#[cfg(not(feature = "dynamic-runtime"))]
fn native_mtmd_defaults_cross_the_ffi_boundary() {
    let params = unsafe { mtmd_context_params_default() };

    assert_eq!(params.batch_max_tokens, 1024);
    assert!(params.progress_callback.is_none());
    assert!(params.progress_callback_user_data.is_null());
}
