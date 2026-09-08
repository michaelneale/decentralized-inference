use anyhow::{Context, Result, bail};
use skippy_protocol::binary::{
    WireMessageKind, activation_payload_multiplier_from_state_flags,
    activation_state_flags_from_frame_flags,
};
use skippy_runtime::ActivationFrame;

pub(in crate::binary_transport) fn executable_prefill_start(
    kind: WireMessageKind,
    restored_tokens: usize,
    token_count: usize,
    layer_start: u32,
    has_downstream: bool,
) -> usize {
    let partial_restore = restored_tokens > 0 && restored_tokens < token_count;
    if kind.is_prefill() && partial_restore && (layer_start == 0 || !has_downstream) {
        restored_tokens
    } else {
        0
    }
}

pub(in crate::binary_transport) fn suffix_activation_frame(
    input: Option<ActivationFrame>,
    token_start: usize,
) -> Result<Option<ActivationFrame>> {
    let Some(frame) = input else {
        return Ok(None);
    };
    if token_start == 0 {
        return Ok(Some(frame));
    }
    let token_count =
        usize::try_from(frame.desc.token_count).context("activation token count overflow")?;
    if token_start >= token_count {
        bail!("suffix activation start {token_start} exceeds frame token count {token_count}");
    }
    let plane_count = activation_payload_multiplier_from_state_flags(
        activation_state_flags_from_frame_flags(frame.desc.flags),
    );
    if !frame.payload.len().is_multiple_of(plane_count) {
        bail!(
            "activation payload is not divisible by plane count: payload={} planes={plane_count}",
            frame.payload.len()
        );
    }
    let plane_bytes = frame.payload.len() / plane_count;
    if !plane_bytes.is_multiple_of(token_count) {
        bail!(
            "activation plane is not divisible by token count: plane={plane_bytes} tokens={token_count}",
        );
    }
    let row_bytes = plane_bytes / token_count;
    let suffix_offset = token_start * row_bytes;
    let suffix_plane_bytes = plane_bytes - suffix_offset;
    let mut payload = Vec::with_capacity(suffix_plane_bytes * plane_count);
    if plane_bytes > 0 {
        for plane in frame.payload.chunks_exact(plane_bytes) {
            payload.extend_from_slice(&plane[suffix_offset..]);
        }
    }
    let suffix_tokens = token_count - token_start;
    let mut desc = frame.desc;
    desc.token_count = u32::try_from(suffix_tokens).context("suffix token count overflow")?;
    desc.sequence_count = if suffix_tokens > 0 { 1 } else { 0 };
    desc.payload_bytes = payload.len() as u64;
    Ok(Some(ActivationFrame { desc, payload }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_non_first_stage_executes_only_suffix_after_partial_restore() {
        assert_eq!(
            executable_prefill_start(WireMessageKind::PrefillEmbd, 3, 5, 8, false),
            3
        );
    }

    #[test]
    fn intermediate_non_first_stage_preserves_full_activation_range() {
        assert_eq!(
            executable_prefill_start(WireMessageKind::PrefillEmbd, 3, 5, 8, true),
            0
        );
    }

    fn frame(token_count: u32, row_bytes: usize) -> ActivationFrame {
        use skippy_runtime::{ActivationDesc, RuntimeActivationDType, RuntimeActivationLayout};
        let mut payload = Vec::with_capacity(row_bytes * token_count as usize);
        for row_idx in 0..token_count {
            // Each row is filled with a distinct byte value (the row index)
            payload.extend(vec![row_idx as u8; row_bytes]);
        }
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 8,
                token_count,
                sequence_count: 1,
                payload_bytes: payload.len() as u64,
                flags: 0,
            },
            payload,
        }
    }

    #[test]
    fn suffix_frame_slices_payload_rows_and_rebuilds_descriptor() {
        let sliced = suffix_activation_frame(Some(frame(5, 8)), 3)
            .unwrap()
            .unwrap();
        assert_eq!(sliced.desc.token_count, 2);
        assert_eq!(sliced.desc.sequence_count, 1);
        assert_eq!(sliced.payload.len(), 16);
        assert_eq!(sliced.desc.payload_bytes, 16);
        // Verify the payload contains rows 3 and 4 from the original frame
        // Row 3: 8 bytes of 0x03, Row 4: 8 bytes of 0x04
        let mut expected = vec![3_u8; 8];
        expected.extend(vec![4_u8; 8]);
        assert_eq!(&sliced.payload, &expected);
    }

    #[test]
    fn suffix_frame_slices_each_sideband_plane_by_token() {
        use skippy_protocol::binary::ACTIVATION_FLAG_RWKV7_V_FIRST;

        let mut input = frame(5, 4);
        input.desc.flags = ACTIVATION_FLAG_RWKV7_V_FIRST;
        input.payload.clear();
        for plane in [0_u8, 10] {
            for row_idx in 0..5 {
                input.payload.extend(vec![plane + row_idx; 4]);
            }
        }
        input.desc.payload_bytes = input.payload.len() as u64;

        let sliced = suffix_activation_frame(Some(input), 3).unwrap().unwrap();
        let mut expected = vec![3_u8; 4];
        expected.extend(vec![4_u8; 4]);
        expected.extend(vec![13_u8; 4]);
        expected.extend(vec![14_u8; 4]);
        assert_eq!(sliced.payload, expected);
        assert_eq!(sliced.desc.token_count, 2);
        assert_eq!(sliced.desc.payload_bytes, 16);
        assert_eq!(sliced.desc.flags, ACTIVATION_FLAG_RWKV7_V_FIRST);
    }

    #[test]
    fn suffix_frame_start_zero_is_identity() {
        let original = frame(5, 8);
        let sliced = suffix_activation_frame(Some(original.clone()), 0)
            .unwrap()
            .unwrap();
        assert_eq!(sliced.desc.token_count, original.desc.token_count);
        assert_eq!(sliced.payload.len(), original.payload.len());
    }

    #[test]
    fn suffix_frame_none_passes_through() {
        assert!(suffix_activation_frame(None, 3).unwrap().is_none());
    }

    #[test]
    fn suffix_frame_rejects_start_beyond_token_count() {
        assert!(suffix_activation_frame(Some(frame(5, 8)), 5).is_err());
        assert!(suffix_activation_frame(Some(frame(5, 8)), 6).is_err());
    }
}
