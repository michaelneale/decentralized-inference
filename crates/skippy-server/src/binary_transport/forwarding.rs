use std::time::Instant;

use anyhow::{Context, Result, bail};
use skippy_protocol::{
    StageActivationCodec, StageActivationCodecPolicy, StageConfig,
    binary::{
        StageWireMessage, activation_state_flags_from_frame_flags,
        select_lossless_activation_codec_with_state_flags,
    },
};
use skippy_runtime::{ActivationFrame, RuntimeActivationDType};

pub(crate) fn forwarded_stage_message(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    activation_width: i32,
) -> Result<StageWireMessage> {
    Ok(forwarded_stage_message_timed(config, incoming, output, activation_width)?.message)
}

pub(crate) struct ForwardedStageMessage {
    pub message: StageWireMessage,
    pub activation_encode_ms: f64,
}

pub(crate) fn forwarded_stage_message_timed(
    config: &StageConfig,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    activation_width: i32,
) -> Result<ForwardedStageMessage> {
    // A stage with a downstream consumer must compute the *full* incoming token
    // range unless it owns layer 0. Suffix-only execution after a partial cache
    // hit is also safe on a final stage with no downstream consumer (see
    // `prefill_execution.rs`). The forwarded header keeps
    // `incoming.token_count` and downstream stages attend over the whole
    // range. If a later stage ever emitted a short frame, the next stage would
    // attend over a prefix it never received and produce plausible-looking but
    // wrong tokens.
    //
    // Today the payload-size check inside the encoder happens to catch this,
    // but it fails for the wrong reason. Assert the
    // invariant directly so a future change to the restore path fails loudly
    // and specifically instead of silently corrupting output.
    if config.layer_start != 0
        && i64::from(output.desc.token_count) != i64::from(incoming.token_count)
    {
        bail!(
            "stage {} (layers {}..{}) produced {} activation tokens for {} incoming tokens; \
             non-first stages must execute the full range",
            config.stage_index,
            config.layer_start,
            config.layer_end,
            output.desc.token_count,
            incoming.token_count,
        );
    }
    let mut state = incoming.state;
    state.source_stage_index = config.stage_index as i32;
    state.flags |= activation_state_flags_from_frame_flags(output.desc.flags);
    let activation_codec = select_output_activation_codec(
        config.activation_codec,
        config.activation_codec_policy,
        incoming,
        output,
        activation_width,
        state.flags,
    )?;
    state.activation_codec = activation_codec;
    let encode_started = Instant::now();
    let activation =
        encode_output_activation_payload(
            activation_codec,
            incoming,
            output,
            activation_width,
            state.flags,
        )
            .with_context(|| {
                format!(
                    "encode f32 output activation payload; frame_dtype={:?} incoming_tokens={} output_tokens={} activation_width={} payload_bytes={} frame_payload_bytes={} state_flags={}",
                    output.desc.dtype,
                    incoming.token_count,
                    output.desc.token_count,
                    activation_width,
                    output.payload.len(),
                    output.desc.payload_bytes,
                    state.flags,
                )
            })?;
    Ok(ForwardedStageMessage {
        message: StageWireMessage {
            kind: incoming.kind,
            pos_start: incoming.pos_start,
            token_count: incoming.token_count,
            state,
            request_id: incoming.request_id,
            session_id: incoming.session_id,
            sampling: incoming.sampling.clone(),
            chat_sampling_metadata: None,
            tokens: incoming.tokens.clone(),
            positions: incoming.positions.clone(),
            activation,
            raw_bytes: Vec::new(),
        },
        activation_encode_ms: encode_started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn select_output_activation_codec(
    configured_codec: StageActivationCodec,
    policy: StageActivationCodecPolicy,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    activation_width: i32,
    state_flags: i32,
) -> Result<StageActivationCodec> {
    if !policy.compatible(configured_codec) {
        bail!(
            "activation codec policy {policy:?} is incompatible with configured codec {configured_codec:?}"
        );
    }
    match policy {
        StageActivationCodecPolicy::Fixed => Ok(configured_codec),
        StageActivationCodecPolicy::AutoLosslessV1 => match output.desc.dtype {
            RuntimeActivationDType::F32 => Ok(select_lossless_activation_codec_with_state_flags(
                incoming.token_count,
                activation_width,
                &output.payload,
                state_flags,
                &[
                    StageActivationCodec::RawF32V1,
                    StageActivationCodec::Bf16RneV1,
                    StageActivationCodec::F16RneV1,
                ],
            )?),
            dtype => bail!("unsupported activation dtype selection: {dtype:?}"),
        },
    }
}

fn encode_output_activation_payload(
    codec: skippy_protocol::StageActivationCodec,
    incoming: &StageWireMessage,
    output: &ActivationFrame,
    activation_width: i32,
    state_flags: i32,
) -> Result<Vec<u8>> {
    match output.desc.dtype {
        RuntimeActivationDType::F32 => Ok(
            skippy_protocol::binary::encode_activation_payload_with_state_flags(
                codec,
                incoming.token_count,
                activation_width,
                &output.payload,
                state_flags,
            )?,
        ),
        dtype => {
            bail!("unsupported activation dtype conversion: {dtype:?} to f32")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{
        FlashAttentionType, LoadMode, PeerConfig, StageDevice, StageKvCacheConfig,
        binary::{
            StageStateHeader, WireMessageKind, activation_frame_flags_from_state_flags, state_flags,
        },
    };
    use skippy_runtime::{ActivationDesc, RuntimeActivationDType, RuntimeActivationLayout};

    fn stage_config() -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "model".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: None,
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            layer_start: 4,
            layer_end: 8,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: true,
            selected_device: None::<StageDevice>,
            kv_cache: None::<StageKvCacheConfig>,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: Some(PeerConfig {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                endpoint: "tcp://127.0.0.1:19000".to_string(),
            }),
            downstream: None,
            ..StageConfig::default()
        }
    }

    fn incoming_message() -> StageWireMessage {
        StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 7,
            token_count: 1,
            state: StageStateHeader::new(WireMessageKind::DecodeEmbd),
            request_id: 42,
            session_id: 99,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![11],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn f32_frame(flags: u64, token_count: u32, values: &[f32]) -> ActivationFrame {
        let mut payload = Vec::new();
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 1,
                layer_start: 4,
                layer_end: 8,
                token_count,
                sequence_count: 1,
                payload_bytes: payload.len() as u64,
                flags,
            },
            payload,
        }
    }

    fn rwkv7_sideband_frame() -> ActivationFrame {
        f32_frame(
            skippy_protocol::binary::ACTIVATION_FLAG_RWKV7_V_FIRST,
            1,
            &[1.0_f32, 2.0, 3.0, 4.0],
        )
    }

    #[test]
    fn forwarded_stage_message_preserves_rwkv7_sideband_shape() {
        let mut config = stage_config();
        config.activation_codec = skippy_protocol::StageActivationCodec::F16RneV1;
        let forwarded =
            forwarded_stage_message_timed(&config, &incoming_message(), &rwkv7_sideband_frame(), 2)
                .unwrap();

        assert_eq!(
            forwarded.message.state.activation_codec,
            skippy_protocol::StageActivationCodec::F16RneV1
        );
        assert_eq!(forwarded.message.activation.len(), 8);
        assert_ne!(
            forwarded.message.state.flags & state_flags::RWKV7_V_FIRST_SIDEBAND,
            0
        );
        assert_eq!(
            activation_frame_flags_from_state_flags(forwarded.message.state.flags),
            skippy_protocol::binary::ACTIVATION_FLAG_RWKV7_V_FIRST
        );

        let mut wire = Vec::new();
        skippy_protocol::binary::write_stage_message(&mut wire, &forwarded.message).unwrap();
        let decoded = skippy_protocol::binary::read_stage_message_for_codec(
            std::io::Cursor::new(wire),
            2,
            skippy_protocol::StageActivationCodec::F16RneV1,
        )
        .unwrap();
        assert_eq!(decoded.activation.len(), 16);
    }

    #[test]
    fn auto_lossless_selects_bf16_when_both_half_formats_are_exact() {
        let mut config = stage_config();
        config.activation_codec = StageActivationCodec::RawF32V1;
        config.activation_codec_policy = StageActivationCodecPolicy::AutoLosslessV1;
        let forwarded = forwarded_stage_message_timed(
            &config,
            &incoming_message(),
            &f32_frame(0, 1, &[1.0, 2.0]),
            2,
        )
        .unwrap();

        assert_eq!(
            forwarded.message.state.activation_codec,
            StageActivationCodec::Bf16RneV1
        );
        assert_eq!(forwarded.message.activation.len(), 4);
    }

    #[test]
    fn auto_lossless_selects_f16_when_only_f16_is_exact() {
        let mut config = stage_config();
        config.activation_codec = StageActivationCodec::RawF32V1;
        config.activation_codec_policy = StageActivationCodecPolicy::AutoLosslessV1;
        let forwarded = forwarded_stage_message_timed(
            &config,
            &incoming_message(),
            &f32_frame(0, 1, &[1.000_976_6, 2.0]),
            2,
        )
        .unwrap();

        assert_eq!(
            forwarded.message.state.activation_codec,
            StageActivationCodec::F16RneV1
        );
    }

    #[test]
    fn auto_lossless_uses_raw_for_mixed_primary_and_sideband_requirements() {
        let mut config = stage_config();
        config.activation_codec = StageActivationCodec::RawF32V1;
        config.activation_codec_policy = StageActivationCodecPolicy::AutoLosslessV1;
        let forwarded = forwarded_stage_message_timed(
            &config,
            &incoming_message(),
            &f32_frame(
                skippy_protocol::binary::ACTIVATION_FLAG_RWKV7_V_FIRST,
                1,
                &[1.000_976_6, 2.0, 65_536.0, 4.0],
            ),
            2,
        )
        .unwrap();

        assert_eq!(
            forwarded.message.state.activation_codec,
            StageActivationCodec::RawF32V1
        );
        assert_eq!(forwarded.message.activation.len(), 16);
    }

    #[test]
    fn auto_lossless_rejects_a_non_raw_fallback() {
        let mut config = stage_config();
        config.activation_codec = StageActivationCodec::F16RneV1;
        config.activation_codec_policy = StageActivationCodecPolicy::AutoLosslessV1;
        let error = forwarded_stage_message_timed(
            &config,
            &incoming_message(),
            &f32_frame(0, 1, &[1.0, 2.0]),
            2,
        )
        .err()
        .expect("invalid AutoLosslessV1 fallback must fail");

        assert!(format!("{error:#}").contains("incompatible with configured codec"));
    }

    #[test]
    fn first_stage_applies_auto_lossless_selection() {
        let mut config = stage_config();
        config.stage_index = 0;
        config.layer_start = 0;
        config.activation_codec = StageActivationCodec::RawF32V1;
        config.activation_codec_policy = StageActivationCodecPolicy::AutoLosslessV1;
        let forwarded = forwarded_stage_message_timed(
            &config,
            &incoming_message(),
            &f32_frame(0, 1, &[1.0, 2.0]),
            2,
        )
        .unwrap();

        assert_eq!(forwarded.message.state.source_stage_index, 0);
        assert_eq!(
            forwarded.message.state.activation_codec,
            StageActivationCodec::Bf16RneV1
        );
    }

    #[test]
    fn compact_activation_encode_failure_does_not_fall_back_to_raw() {
        let mut config = stage_config();
        config.activation_codec = skippy_protocol::StageActivationCodec::F16RneV1;
        let error = forwarded_stage_message_timed(
            &config,
            &incoming_message(),
            &f32_frame(0, 1, &[f32::MAX, 1.0]),
            2,
        )
        .err()
        .expect("F16 overflow must fail the stage connection");

        assert!(format!("{error:#}").contains("F16 activation value is out of range"));
    }

    /// A non-first stage must execute the full incoming token range.
    ///
    /// Suffix-only execution after a partial cache hit is legal only on the
    /// stage owning layer 0. If a later stage emitted a short frame, the next
    /// stage would attend over a prefix it never received -- plausible-looking
    /// but wrong output. This must fail loudly and by name.
    #[test]
    fn non_first_stage_must_not_emit_a_short_activation_frame() {
        let config = stage_config();
        assert_ne!(config.layer_start, 0, "fixture must be a non-first stage");
        let mut incoming = incoming_message();
        incoming.token_count = 4;
        // Frame covers only 1 of the 4 incoming tokens, as a suffix-only
        // execution after a 3-token restore would produce.
        let output = f32_frame(0, 1, &[1.0, 2.0, 3.0, 4.0]);

        let error = forwarded_stage_message_timed(&config, &incoming, &output, 4)
            .err()
            .expect("short frame from a non-first stage must be rejected");

        let text = format!("{error:#}");
        assert!(
            text.contains("must execute the full range"),
            "expected the named invariant, got: {text}"
        );
    }

    /// The first stage is explicitly allowed to emit a short frame: that is
    /// how suffix-only prefill after a cache hit works.
    #[test]
    fn first_stage_may_emit_a_short_activation_frame() {
        let mut config = stage_config();
        config.stage_index = 0;
        config.layer_start = 0;
        let mut incoming = incoming_message();
        incoming.token_count = 4;

        // Keep the encoded payload valid for the four-token wire header while
        // the frame descriptor exercises the first-stage short-frame branch.
        assert!(
            forwarded_stage_message_timed(&config, &incoming, &f32_frame(0, 1, &[1.0; 16]), 4,)
                .is_ok()
        );
    }
}
