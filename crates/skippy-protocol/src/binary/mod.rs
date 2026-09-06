mod activation;
mod activation_codec;
mod codec;
mod types;

pub use activation::{
    activation_payload_multiplier_from_state_flags, activation_wire_bytes,
    activation_wire_bytes_for_codec_with_state_flags, activation_wire_bytes_with_state_flags,
    encode_activation_payload_with_state_flags, encode_f32_activation_payload,
    encode_f32_activation_payload_with_state_flags,
    select_lossless_activation_codec_with_state_flags,
};
pub use codec::{
    read_stage_message, read_stage_message_for_codec, read_stage_message_for_codec_policy,
    recv_ready, recv_reply, send_ready, send_reply_ack, send_reply_ack_with_stats,
    send_reply_message, send_reply_predicted, send_reply_predicted_tokens_with_stats,
    send_reply_predicted_tokens_with_window_and_stats, send_reply_predicted_with_stats,
    send_reply_predicted_with_tokens_and_stats, send_reply_predicted_with_tokens_window_and_stats,
    write_stage_message,
};
pub use types::sampling_flags;
pub use types::{
    ACTIVATION_FLAG_GEMMA3N_ALTUP, ACTIVATION_FLAG_INKLING_MTP_EMBD, ACTIVATION_FLAG_RWKV7_V_FIRST,
    LLAMA_TOKEN_NULL, MAX_STAGE_ACTIVATION_BYTES, MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES,
    MAX_STAGE_DECODED_ACTIVATION_BYTES, MAX_STAGE_DRY_SEQUENCE_BREAKERS, MAX_STAGE_LOGIT_BIAS,
    MAX_STAGE_PREDICTED_TOKENS, MAX_STAGE_SAMPLERS, MAX_STAGE_SAMPLING_STRING_BYTES,
    MAX_STAGE_SIDEBAND_VALUES, MAX_STAGE_STATE_IMPORT_BYTES, READY_MAGIC,
    STAGE_LOGIT_BIAS_WIRE_BYTES, STAGE_SAMPLING_CONFIG_BASE_BYTES, STAGE_STATE_HEADER_BYTES,
    STAGE_STATE_VERSION, STAGE_WIRE_FIXED_HEADER_BYTES, StageLogitBias, StageNativeMtpDraft,
    StageReply, StageReplyStats, StageReplyWindow, StageRequestEpoch, StageSamplingConfig,
    StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind, WireStagePhase,
    activation_frame_flags_from_state_flags, activation_state_flags_from_frame_flags, state_flags,
};

pub(crate) fn invalid_data(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

pub(crate) fn invalid_input(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::io::{Cursor, Read, Write};
    use std::rc::Rc;

    fn push_i32(bytes: &mut Vec<u8>, value: i32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_state_header(bytes: &mut Vec<u8>, state: StageStateHeader) {
        push_i32(bytes, state.version);
        push_i32(bytes, state.seq_id);
        push_i32(bytes, state.phase);
        push_i32(bytes, state.flags);
        push_i32(bytes, state.checkpoint_generation);
        push_i32(bytes, state.prompt_token_count);
        push_i32(bytes, state.decode_step);
        push_i32(bytes, state.current_token);
        push_i32(bytes, state.source_stage_index);
        push_i32(bytes, state.activation_codec.binary_wire_id());
    }

    fn stage_frame_prefix(
        kind: WireMessageKind,
        token_count: i32,
        token_sideband_count: i32,
        position_sideband_count: i32,
        state: StageStateHeader,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_i32(&mut bytes, kind as i32);
        push_i32(&mut bytes, 0);
        push_i32(&mut bytes, token_count);
        push_i32(&mut bytes, token_sideband_count);
        push_i32(&mut bytes, position_sideband_count);
        push_state_header(&mut bytes, state);
        push_i32(&mut bytes, 0);
        push_u64(&mut bytes, 7);
        push_u64(&mut bytes, 11);
        bytes
    }

    fn assert_invalid_data<T: std::fmt::Debug>(result: std::io::Result<T>, expected: &str) {
        let error = result.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), expected);
    }

    struct CountingReader {
        inner: Cursor<Vec<u8>>,
        calls: Rc<Cell<usize>>,
    }

    impl Read for CountingReader {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            self.calls.set(self.calls.get() + 1);
            self.inner.read(output)
        }
    }

    struct CountingWriter {
        bytes: Vec<u8>,
        calls: Rc<Cell<usize>>,
    }

    impl Write for CountingWriter {
        fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
            self.calls.set(self.calls.get() + 1);
            self.bytes.extend_from_slice(input);
            Ok(input.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn ready_round_trips() {
        let mut bytes = Vec::new();
        send_ready(&mut bytes).unwrap();
        recv_ready(Cursor::new(bytes)).unwrap();
    }

    #[test]
    fn reply_round_trips() {
        let mut bytes = Vec::new();
        send_reply_predicted(&mut bytes, 42).unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();
        assert_eq!(reply.kind, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
        assert_eq!(reply.predicted_tokens, vec![42]);
        assert_eq!(reply.native_mtp_draft, None);
    }

    #[test]
    fn reply_round_trips_typed_native_mtp_draft() {
        let reply = StageReply {
            kind: WireReplyKind::PredictedToken,
            predicted: 42,
            predicted_tokens: vec![42],
            native_mtp_draft: Some(StageNativeMtpDraft {
                token_ids: vec![43, 44],
                proposal_compute_us: 12_345,
            }),
            window: StageReplyWindow::default(),
            stats: StageReplyStats::default(),
        };
        let mut bytes = Vec::new();
        send_reply_message(&mut bytes, &reply).unwrap();

        assert_eq!(recv_reply(Cursor::new(bytes)).unwrap(), reply);
    }

    #[test]
    fn predicted_token_reply_preserves_sideband_tokens() {
        let mut bytes = Vec::new();
        send_reply_predicted_with_tokens_and_stats(
            &mut bytes,
            42,
            &[42, 43, 123],
            StageReplyStats::default(),
        )
        .unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();
        assert_eq!(reply.kind, WireReplyKind::PredictedToken);
        assert_eq!(reply.predicted, 42);
        assert_eq!(reply.predicted_tokens, vec![42, 43, 123]);
    }

    #[test]
    fn reply_stats_preserve_prefill_edge_transport() {
        let mut stats = StageReplyStats::default();
        stats.observe_prefill_edge_transport(2, 12_000, 3_000, 1_048_576);
        stats.observe_prefill_edge_transport(1, 4_000, 40_000, 524_288);
        stats.observe_prefill_compute(2, 18_000, 128);
        stats.observe_prefill_compute(1, 42_000, 128);
        stats.observe_prefill_compute(3, 25_000, 64);

        let mut bytes = Vec::new();
        send_reply_predicted_with_stats(&mut bytes, 42, stats).unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();

        assert_eq!(reply.stats.prefill_edge_write_us_max, 12_000);
        assert_eq!(reply.stats.prefill_edge_wait_us_max, 40_000);
        assert_eq!(reply.stats.prefill_edge_total_us_max, 44_000);
        assert_eq!(reply.stats.prefill_edge_stage_index, 1);
        assert_eq!(reply.stats.prefill_edge_activation_bytes_max, 524_288);
        assert_eq!(reply.stats.prefill_edge_observation_count, 2);
        assert_eq!(reply.stats.prefill_compute_us_at_slowest_rate, 42_000);
        assert_eq!(reply.stats.prefill_compute_stage_index, 1);
        assert_eq!(reply.stats.prefill_compute_token_count_at_slowest_rate, 128);
        assert_eq!(reply.stats.prefill_compute_observation_count, 3);
    }

    #[test]
    fn prefill_compute_calibration_ignores_short_tail_overhead() {
        let mut stats = StageReplyStats::default();
        stats.observe_prefill_compute(1, 48_000, 128);
        stats.observe_prefill_compute(1, 30_000, 9);

        assert_eq!(stats.prefill_compute_us_at_slowest_rate, 48_000);
        assert_eq!(stats.prefill_compute_stage_index, 1);
        assert_eq!(stats.prefill_compute_token_count_at_slowest_rate, 128);
        assert_eq!(stats.prefill_compute_observation_count, 2);
    }

    #[test]
    fn token_vector_reply_round_trips() {
        let mut bytes = Vec::new();
        send_reply_predicted_tokens_with_stats(&mut bytes, &[1, 2, 3], StageReplyStats::default())
            .unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();
        assert_eq!(reply.kind, WireReplyKind::PredictedTokens);
        assert_eq!(reply.predicted, 1);
        assert_eq!(reply.predicted_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn reply_window_metadata_round_trips() {
        let mut bytes = Vec::new();
        send_reply_predicted_tokens_with_window_and_stats(
            &mut bytes,
            &[1, 2, 3],
            StageReplyWindow { window_id: 42 },
            StageReplyStats::default(),
        )
        .unwrap();
        let reply = recv_reply(Cursor::new(bytes)).unwrap();

        assert_eq!(reply.kind, WireReplyKind::PredictedTokens);
        assert_eq!(reply.predicted_tokens, vec![1, 2, 3]);
        assert_eq!(reply.window.window_id, 42);
    }

    #[test]
    fn reply_rejects_predicted_token_count_over_limit() {
        let mut bytes = Vec::new();
        push_i32(&mut bytes, WireReplyKind::PredictedTokens as i32);
        push_i32(&mut bytes, 1);
        push_i32(
            &mut bytes,
            i32::try_from(MAX_STAGE_PREDICTED_TOKENS + 1).unwrap(),
        );

        assert_invalid_data(
            recv_reply(Cursor::new(bytes)),
            "predicted token count exceeds maximum",
        );
    }

    #[test]
    fn stage_message_round_trips_f32() {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        state.checkpoint_generation = 3;
        state.prompt_token_count = 1;
        state.decode_step = 0;
        state.current_token = 11;
        state.source_stage_index = 0;
        state.activation_codec = crate::StageActivationCodec::RawF32V1;
        let activation: Vec<u8> = [1.0_f32, 2.0_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 1,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 11,
            sampling: Some(StageSamplingConfig {
                flags: 1,
                seed: 42,
                temperature: 0.8,
                top_p: 0.9,
                top_k: 40,
                logit_bias: vec![StageLogitBias {
                    token_id: 123,
                    bias: -50.0,
                }],
                ..StageSamplingConfig::default()
            }),
            chat_sampling_metadata: None,
            tokens: vec![11],
            positions: Vec::new(),
            activation: activation.clone(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::DecodeEmbd);
        assert_eq!(decoded.tokens, vec![11]);
        assert_eq!(decoded.activation, activation);
        assert_eq!(decoded.state.source_stage_index, 0);
        assert_eq!(decoded.request_id, 7);
        assert_eq!(decoded.session_id, 11);
        assert_eq!(
            decoded.request_epoch(),
            StageRequestEpoch {
                request_id: 7,
                session_id: 11,
                checkpoint_generation: 3,
                prompt_token_count: 1,
                decode_step: 0,
            }
        );
        assert_ne!(decoded.state.flags & state_flags::SAMPLING, 0);
        assert_eq!(decoded.state.flags & state_flags::CHAT_SAMPLING_METADATA, 0);
        assert_eq!(decoded.chat_sampling_metadata, None);
        let sampling = decoded.sampling.expect("sampling extension round-tripped");
        assert_eq!(sampling.seed, 42);
        assert_eq!(sampling.top_k, 40);
        assert_eq!(sampling.logit_bias.len(), 1);
        assert_eq!(sampling.logit_bias[0].token_id, 123);
        assert_eq!(sampling.logit_bias[0].bias, -50.0);
    }

    #[test]
    fn stage_message_round_trips_every_activation_codec_with_sideband_rows() {
        let values = [1.0_f32, -1.0, 0.5, -0.5];
        let f32_payload = values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        for codec in [
            crate::StageActivationCodec::RawF32V1,
            crate::StageActivationCodec::F16RneV1,
            crate::StageActivationCodec::Bf16RneV1,
            crate::StageActivationCodec::S8RowF32RneV1,
        ] {
            let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
            state.source_stage_index = 0;
            state.flags |= state_flags::RWKV7_V_FIRST_SIDEBAND;
            state.activation_codec = codec;
            let activation =
                encode_activation_payload_with_state_flags(codec, 1, 2, &f32_payload, state.flags)
                    .unwrap();
            assert_eq!(
                activation.len(),
                activation_wire_bytes_for_codec_with_state_flags(codec, 1, 2, state.flags).unwrap()
            );
            let message = StageWireMessage {
                kind: WireMessageKind::DecodeEmbd,
                pos_start: 0,
                token_count: 1,
                state,
                request_id: 7,
                session_id: 9,
                sampling: None,
                chat_sampling_metadata: None,
                tokens: vec![42],
                positions: Vec::new(),
                activation,
                raw_bytes: Vec::new(),
            };
            let mut bytes = Vec::new();
            write_stage_message(&mut bytes, &message).unwrap();
            let decoded = read_stage_message_for_codec(Cursor::new(bytes), 2, codec).unwrap();
            assert_eq!(decoded.state.activation_codec, codec);
            assert_eq!(decoded.activation.len(), f32_payload.len());
            if codec == crate::StageActivationCodec::RawF32V1 {
                assert_eq!(decoded.activation, f32_payload);
            }
        }
    }

    #[test]
    fn stage_message_rejects_codec_mismatch_and_malformed_compact_header() {
        let codec = crate::StageActivationCodec::F16RneV1;
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        state.source_stage_index = 0;
        state.activation_codec = codec;
        let f32_payload = [1.0_f32, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let activation =
            encode_activation_payload_with_state_flags(codec, 1, 2, &f32_payload, 0).unwrap();
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation,
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let fixed_header_only = bytes[..STAGE_WIRE_FIXED_HEADER_BYTES].to_vec();
        assert_invalid_data(
            read_stage_message_for_codec(
                Cursor::new(fixed_header_only),
                2,
                crate::StageActivationCodec::RawF32V1,
            ),
            "stage activation codec mismatch",
        );

        let mut wrong_size = bytes.clone();
        wrong_size[60..64].copy_from_slice(&5_i32.to_le_bytes());
        assert_invalid_data(
            read_stage_message(Cursor::new(wrong_size), 2),
            "activation payload size mismatch",
        );

        bytes[56..60].copy_from_slice(&99_i32.to_le_bytes());
        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2),
            "unknown stage activation codec",
        );
    }

    #[test]
    fn stage_message_auto_lossless_policy_admits_only_exact_codec_set() {
        let f32_payload = [1.0_f32, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        for codec in [
            crate::StageActivationCodec::RawF32V1,
            crate::StageActivationCodec::Bf16RneV1,
            crate::StageActivationCodec::F16RneV1,
        ] {
            let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
            state.source_stage_index = 0;
            state.activation_codec = codec;
            let message = StageWireMessage {
                kind: WireMessageKind::DecodeEmbd,
                pos_start: 0,
                token_count: 1,
                state,
                request_id: 7,
                session_id: 9,
                sampling: None,
                chat_sampling_metadata: None,
                tokens: Vec::new(),
                positions: Vec::new(),
                activation: encode_activation_payload_with_state_flags(
                    codec,
                    1,
                    2,
                    &f32_payload,
                    0,
                )
                .unwrap(),
                raw_bytes: Vec::new(),
            };
            let mut bytes = Vec::new();
            write_stage_message(&mut bytes, &message).unwrap();
            let decoded = read_stage_message_for_codec_policy(
                Cursor::new(bytes),
                2,
                crate::StageActivationCodec::RawF32V1,
                crate::StageActivationCodecPolicy::AutoLosslessV1,
            )
            .unwrap();
            assert_eq!(decoded.state.activation_codec, codec);
        }

        let codec = crate::StageActivationCodec::S8RowF32RneV1;
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        state.source_stage_index = 0;
        state.activation_codec = codec;
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: encode_activation_payload_with_state_flags(codec, 1, 2, &f32_payload, 0)
                .unwrap(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        assert_invalid_data(
            read_stage_message_for_codec_policy(
                Cursor::new(bytes),
                2,
                crate::StageActivationCodec::RawF32V1,
                crate::StageActivationCodecPolicy::AutoLosslessV1,
            ),
            "stage activation codec mismatch",
        );
    }

    #[test]
    fn verify_window_message_round_trips_window_metadata() {
        let mut state = StageStateHeader::new(WireMessageKind::VerifyWindow);
        state.seq_id = 42;
        state.prompt_token_count = 128;
        state.decode_step = 7;
        state.current_token = 1001;
        let message = StageWireMessage {
            kind: WireMessageKind::VerifyWindow,
            pos_start: 135,
            token_count: 4,
            state,
            request_id: 7,
            session_id: 11,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![1001, 1002, 1003, 1004],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };

        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();

        assert_eq!(decoded.kind, WireMessageKind::VerifyWindow);
        assert_eq!(decoded.verify_window_id(), Some(42));
        assert_eq!(decoded.verify_window_base_position(), Some(135));
        assert_eq!(decoded.verify_window_token_count(), Some(4));
        assert_eq!(decoded.authoritative_session_position(), Some(135));
        assert_eq!(decoded.tokens, vec![1001, 1002, 1003, 1004]);
        assert_eq!(decoded.state.decode_step, 7);
    }

    #[test]
    fn only_decode_messages_carry_authoritative_session_positions() {
        let mut decode = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 17,
            token_count: 1,
            state: StageStateHeader::new(WireMessageKind::DecodeEmbd),
            request_id: 1,
            session_id: 2,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![3],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        assert_eq!(decode.authoritative_session_position(), Some(17));

        decode.pos_start = -1;
        assert_eq!(decode.authoritative_session_position(), None);

        decode.kind = WireMessageKind::PrefillEmbd;
        assert_eq!(decode.authoritative_session_position(), None);
    }

    #[test]
    fn stage_message_rejects_old_state_version() {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        state.version = STAGE_STATE_VERSION - 1;
        let bytes = stage_frame_prefix(WireMessageKind::DecodeEmbd, 1, 0, 0, state);

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2),
            "unsupported stage state version",
        );
    }

    #[test]
    fn stage_message_rejects_legacy_kind_10() {
        let mut bytes = Vec::new();
        push_i32(&mut bytes, 10);
        push_i32(&mut bytes, 0);
        push_i32(&mut bytes, 1);
        push_i32(&mut bytes, 0);
        push_i32(&mut bytes, 0);

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2),
            "unknown stage message kind",
        );
    }

    #[test]
    fn stage_message_estimates_full_wire_transfer_bytes() {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.source_stage_index = 0;
        let message = StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: 2,
            state,
            request_id: 7,
            session_id: 11,
            sampling: Some(StageSamplingConfig {
                flags: 1,
                logit_bias: vec![
                    StageLogitBias {
                        token_id: 1,
                        bias: -1.0,
                    },
                    StageLogitBias {
                        token_id: 2,
                        bias: 1.0,
                    },
                ],
                ..StageSamplingConfig::default()
            }),
            chat_sampling_metadata: Some("{}".to_string()),
            tokens: vec![1, 2],
            positions: vec![0],
            activation: vec![0; 16],
            raw_bytes: Vec::new(),
        };

        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        assert_eq!(message.estimated_wire_bytes(), bytes.len());
    }

    #[test]
    fn request_epoch_orders_only_matching_flows() {
        let older = StageRequestEpoch {
            request_id: 7,
            session_id: 11,
            checkpoint_generation: 1,
            prompt_token_count: 8,
            decode_step: 2,
        };
        let newer = StageRequestEpoch {
            request_id: 7,
            session_id: 11,
            checkpoint_generation: 1,
            prompt_token_count: 8,
            decode_step: 3,
        };
        let different_session = StageRequestEpoch {
            session_id: 12,
            ..newer
        };

        assert!(older.same_flow(newer));
        assert!(older.is_stale_for(newer));
        assert!(!newer.is_stale_for(older));
        assert!(!older.same_flow(different_session));
        assert!(!older.is_stale_for(different_session));
    }

    #[test]
    fn request_epoch_staleness_orders_generation_before_prompt_before_decode() {
        let base = StageRequestEpoch {
            request_id: 7,
            session_id: 11,
            checkpoint_generation: 1,
            prompt_token_count: 8,
            decode_step: 3,
        };
        let newer_checkpoint = StageRequestEpoch {
            checkpoint_generation: 2,
            prompt_token_count: 0,
            decode_step: 0,
            ..base
        };
        let newer_prompt = StageRequestEpoch {
            prompt_token_count: 9,
            decode_step: 0,
            ..base
        };
        let newer_decode = StageRequestEpoch {
            decode_step: 4,
            ..base
        };

        assert!(base.same_flow(newer_checkpoint));
        assert!(base.is_stale_for(newer_checkpoint));
        assert!(!newer_checkpoint.is_stale_for(base));
        assert!(base.is_stale_for(newer_prompt));
        assert!(!newer_prompt.is_stale_for(base));
        assert!(base.is_stale_for(newer_decode));
        assert!(!newer_decode.is_stale_for(base));
    }

    #[test]
    fn generation_config_round_trips_sampling_metadata() {
        let message = StageWireMessage::configure_generation(
            7,
            11,
            123,
            Some(StageSamplingConfig {
                flags: 1,
                seed: 42,
                temperature: 0.8,
                top_p: 0.9,
                top_k: 40,
                ..StageSamplingConfig::default()
            }),
            Some("{\"grammar\":\"root ::= \\\"x\\\"\"}".to_string()),
        );
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::ConfigureGeneration);
        assert_eq!(decoded.token_count, 0);
        assert_eq!(decoded.tokens, Vec::<i32>::new());
        assert_eq!(decoded.activation, Vec::<u8>::new());
        assert_eq!(decoded.request_id, 7);
        assert_eq!(decoded.session_id, 11);
        assert_eq!(decoded.state.prompt_token_count, 123);
        assert_ne!(decoded.state.flags & state_flags::SAMPLING, 0);
        assert_ne!(decoded.state.flags & state_flags::CHAT_SAMPLING_METADATA, 0);
        assert_eq!(
            decoded.chat_sampling_metadata.as_deref(),
            Some("{\"grammar\":\"root ::= \\\"x\\\"\"}")
        );
        let sampling = decoded.sampling.expect("sampling extension round-tripped");
        assert_eq!(sampling.seed, 42);
        assert_eq!(sampling.top_k, 40);
    }

    #[test]
    fn stage_message_rejects_sampling_metadata_length_over_limit() {
        let mut state = StageStateHeader::new(WireMessageKind::ConfigureGeneration);
        state.flags |= state_flags::CHAT_SAMPLING_METADATA;
        let mut bytes = stage_frame_prefix(WireMessageKind::ConfigureGeneration, 0, 0, 0, state);
        push_u32(
            &mut bytes,
            u32::try_from(MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES + 1).unwrap(),
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "chat sampling metadata length exceeds maximum",
        );
    }

    #[test]
    fn driver_origin_message_round_trips_without_activation() {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.prompt_token_count = 2;
        state.current_token = 22;
        state.source_stage_index = -1;
        let message = StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: 2,
            state,
            request_id: 13,
            session_id: 17,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![11, 22],
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.tokens, vec![11, 22]);
        assert!(decoded.activation.is_empty());
        assert_eq!(decoded.state.source_stage_index, -1);
        assert_eq!(decoded.request_id, 13);
        assert_eq!(decoded.session_id, 17);
        assert_eq!(decoded.state.flags & state_flags::SAMPLING, 0);
        assert!(decoded.sampling.is_none());
    }

    #[test]
    fn stage_message_bulk_reads_sidebands() {
        let value_count = 4_096;
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.source_stage_index = -1;
        let mut bytes = stage_frame_prefix(
            WireMessageKind::PrefillEmbd,
            value_count,
            value_count,
            value_count,
            state,
        );
        for value in 0..value_count {
            push_i32(&mut bytes, value);
        }
        for value in 0..value_count {
            push_i32(&mut bytes, value + 10_000);
        }
        let calls = Rc::new(Cell::new(0));
        let reader = CountingReader {
            inner: Cursor::new(bytes),
            calls: calls.clone(),
        };

        let decoded = read_stage_message(reader, 2_048).unwrap();

        assert_eq!(decoded.tokens.len(), value_count as usize);
        assert_eq!(decoded.positions.len(), value_count as usize);
        assert!(
            calls.get() < 128,
            "sideband decoding used {} reads",
            calls.get()
        );
    }

    #[test]
    fn stage_message_bulk_writes_sidebands() {
        let value_count = 4_096;
        let kind = WireMessageKind::TrimSession;
        let message = StageWireMessage {
            kind,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(kind),
            request_id: 23,
            session_id: 29,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: (0..value_count).collect(),
            positions: (10_000..10_000 + value_count).collect(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let calls = Rc::new(Cell::new(0));
        let writer = CountingWriter {
            bytes: Vec::new(),
            calls: calls.clone(),
        };

        write_stage_message(writer, &message).unwrap();

        assert!(
            calls.get() < 128,
            "sideband encoding used {} writes",
            calls.get()
        );
    }

    #[test]
    fn stage_message_rejects_token_sideband_count_over_limit() {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.source_stage_index = -1;
        let bytes = stage_frame_prefix(
            WireMessageKind::PrefillEmbd,
            0,
            i32::try_from(MAX_STAGE_SIDEBAND_VALUES + 1).unwrap(),
            0,
            state,
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "token sideband count exceeds maximum",
        );
    }

    #[test]
    fn stage_message_rejects_position_sideband_count_over_limit() {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.source_stage_index = -1;
        let bytes = stage_frame_prefix(
            WireMessageKind::PrefillEmbd,
            0,
            0,
            i32::try_from(MAX_STAGE_SIDEBAND_VALUES + 1).unwrap(),
            state,
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "position sideband count exceeds maximum",
        );
    }

    #[test]
    fn prefill_wire_overhead_is_fixed_and_bounded() {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.prompt_token_count = 128;
        state.current_token = 127;
        state.source_stage_index = -1;
        let tokens: Vec<i32> = (0..128).collect();
        let message = StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: tokens.len() as i32,
            state,
            request_id: u64::MAX - 1,
            session_id: u64::MAX,
            sampling: None,
            chat_sampling_metadata: None,
            tokens,
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();

        assert_eq!(STAGE_STATE_HEADER_BYTES, 40);
        assert_eq!(STAGE_SAMPLING_CONFIG_BASE_BYTES, 108);
        assert_eq!(STAGE_WIRE_FIXED_HEADER_BYTES, 80);
        assert_eq!(
            bytes.len(),
            STAGE_WIRE_FIXED_HEADER_BYTES + message.tokens.len() * 4
        );
        const { assert!(STAGE_WIRE_FIXED_HEADER_BYTES <= 80) };
    }

    #[test]
    fn verify_retirement_round_trips_exact_identity() {
        let kind = WireMessageKind::RetireVerifyWindow;
        let message = StageWireMessage {
            kind,
            pos_start: 128,
            token_count: 8,
            state: StageStateHeader::new(kind),
            request_id: 23,
            session_id: 29,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();

        assert_eq!(decoded.kind, kind);
        assert_eq!(decoded.pos_start, 128);
        assert_eq!(decoded.token_count, 8);
        assert!(decoded.state.matches_kind(kind));
    }

    #[test]
    fn session_control_messages_are_fixed_header_only() {
        let kind = WireMessageKind::TrimSession;
        let message = StageWireMessage {
            kind,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(kind),
            request_id: 23,
            session_id: 29,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        assert_eq!(bytes.len(), STAGE_WIRE_FIXED_HEADER_BYTES);
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.kind, kind);
        assert_eq!(decoded.request_id, 23);
        assert_eq!(decoded.session_id, 29);
        assert!(decoded.tokens.is_empty());
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn state_import_message_round_trips_raw_bytes() {
        let state = StageStateHeader::new(WireMessageKind::StateImport);
        let message = StageWireMessage {
            kind: WireMessageKind::StateImport,
            pos_start: 0,
            token_count: 4,
            state,
            request_id: 31,
            session_id: 37,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: vec![1, 2, 3, 4],
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::StateImport);
        assert_eq!(decoded.raw_bytes, vec![1, 2, 3, 4]);
        assert!(decoded.tokens.is_empty());
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn state_import_rejects_raw_byte_count_over_limit() {
        let state = StageStateHeader::new(WireMessageKind::StateImport);
        let bytes = stage_frame_prefix(
            WireMessageKind::StateImport,
            i32::try_from(MAX_STAGE_STATE_IMPORT_BYTES + 1).unwrap(),
            0,
            0,
            state,
        );

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 2048),
            "state import byte count exceeds maximum",
        );
    }

    #[test]
    fn state_import_writer_rejects_raw_byte_count_mismatch() {
        let state = StageStateHeader::new(WireMessageKind::StateImport);
        let message = StageWireMessage {
            kind: WireMessageKind::StateImport,
            pos_start: 0,
            token_count: 8,
            state,
            request_id: 31,
            session_id: 37,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: vec![1, 2, 3, 4],
        };
        let mut bytes = Vec::new();
        let error = write_stage_message(&mut bytes, &message)
            .expect_err("mismatched state import byte count should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(error.to_string(), "state import raw byte count mismatch");
    }

    #[test]
    fn state_export_message_round_trips_without_payload() {
        let state = StageStateHeader::new(WireMessageKind::StateExport);
        let message = StageWireMessage {
            kind: WireMessageKind::StateExport,
            pos_start: 0,
            token_count: 0,
            state,
            request_id: 41,
            session_id: 43,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
        assert_eq!(decoded.kind, WireMessageKind::StateExport);
        assert!(decoded.raw_bytes.is_empty());
        assert!(decoded.tokens.is_empty());
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn stage_message_rejects_activation_payload_over_limit() {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        state.source_stage_index = 0;
        state.flags |= state_flags::GEMMA3N_ALTUP_SIDEBAND;
        let token_count = i32::try_from(MAX_STAGE_ACTIVATION_BYTES / 2 / 1024 + 1).unwrap();
        let bytes = stage_frame_prefix(WireMessageKind::DecodeEmbd, token_count, 0, 0, state);

        assert_invalid_data(
            read_stage_message(Cursor::new(bytes), 1024),
            "activation payload byte count exceeds maximum",
        );
    }

    #[test]
    fn activation_encoding_rejects_decoded_payload_over_limit_before_compression() {
        let n_embd = 65_536;
        let token_count =
            i32::try_from(MAX_STAGE_DECODED_ACTIVATION_BYTES / 4 / n_embd as usize + 1).unwrap();

        assert_invalid_data(
            encode_f32_activation_payload(token_count, n_embd, &[]),
            "decoded activation payload byte count exceeds maximum",
        );
    }

    #[test]
    fn rwkv7_sideband_activation_round_trips() {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        state.source_stage_index = 0;
        state.activation_codec = crate::StageActivationCodec::RawF32V1;
        state.flags |= state_flags::RWKV7_V_FIRST_SIDEBAND;
        let mut activation_f32 = Vec::new();
        for value in [1.0_f32, 2.0, 3.0, 4.0] {
            activation_f32.extend_from_slice(&value.to_le_bytes());
        }
        let activation =
            encode_f32_activation_payload_with_state_flags(1, 2, &activation_f32, state.flags)
                .unwrap();
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![42],
            positions: Vec::new(),
            activation,
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();
        assert_eq!(decoded.activation.len(), 16);
        assert_eq!(
            activation_frame_flags_from_state_flags(decoded.state.flags),
            ACTIVATION_FLAG_RWKV7_V_FIRST
        );
        assert_eq!(decoded.activation_f32_payload().unwrap(), activation_f32);
    }

    #[test]
    fn inkling_mtp_embedding_sideband_activation_round_trips() {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        state.source_stage_index = 0;
        state.activation_codec = crate::StageActivationCodec::RawF32V1;
        state.flags |= state_flags::INKLING_MTP_EMBD_SIDEBAND;
        let mut activation_f32 = Vec::new();
        for value in [1.0_f32, 2.0, 3.0, 4.0] {
            activation_f32.extend_from_slice(&value.to_le_bytes());
        }
        let activation =
            encode_f32_activation_payload_with_state_flags(1, 2, &activation_f32, state.flags)
                .unwrap();
        let message = StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation,
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 2).unwrap();
        assert_eq!(decoded.activation.len(), 16);
        assert_eq!(
            activation_frame_flags_from_state_flags(decoded.state.flags),
            ACTIVATION_FLAG_INKLING_MTP_EMBD
        );
        assert_eq!(
            activation_state_flags_from_frame_flags(ACTIVATION_FLAG_INKLING_MTP_EMBD),
            state_flags::INKLING_MTP_EMBD_SIDEBAND
        );
        assert_eq!(decoded.activation_f32_payload().unwrap(), activation_f32);
    }

    #[test]
    fn f32_activation_payload_can_be_taken() {
        let state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        let f32_payload = [1.0_f32, -1.0_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let activation = encode_f32_activation_payload(1, 2, &f32_payload).unwrap();
        let mut message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![42],
            positions: Vec::new(),
            activation: activation.clone(),
            raw_bytes: Vec::new(),
        };

        let payload = message.take_activation_f32_payload().unwrap();

        assert_eq!(payload, f32_payload);
        assert!(message.activation.is_empty());
    }

    #[test]
    fn f32_activation_payload_helper_preserves_wire_payload() {
        let state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        let f32_payload = [1.0_f32, -1.0_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let activation = encode_f32_activation_payload(1, 2, &f32_payload).unwrap();
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![42],
            positions: Vec::new(),
            activation: activation.clone(),
            raw_bytes: Vec::new(),
        };

        let payload = message.activation_f32_payload().unwrap();

        assert_eq!(payload, f32_payload);
        assert_eq!(message.activation, activation);
    }

    #[test]
    fn gemma3n_altup_sideband_activation_round_trips() {
        let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
        state.source_stage_index = 0;
        state.activation_codec = crate::StageActivationCodec::RawF32V1;
        state.flags |= state_flags::GEMMA3N_ALTUP_SIDEBAND;
        let mut activation_f32 = Vec::new();
        for value in 0..8 {
            activation_f32.extend_from_slice(&(value as f32).to_le_bytes());
        }
        let activation =
            encode_f32_activation_payload_with_state_flags(1, 8, &activation_f32, state.flags)
                .unwrap();
        let message = StageWireMessage {
            kind: WireMessageKind::DecodeEmbd,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 7,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![42],
            positions: Vec::new(),
            activation,
            raw_bytes: Vec::new(),
        };
        let mut bytes = Vec::new();
        write_stage_message(&mut bytes, &message).unwrap();
        let decoded = read_stage_message(Cursor::new(bytes), 8).unwrap();
        assert_eq!(decoded.activation.len(), 32);
        assert_eq!(
            activation_frame_flags_from_state_flags(decoded.state.flags),
            ACTIVATION_FLAG_GEMMA3N_ALTUP
        );
        assert_eq!(decoded.activation_f32_payload().unwrap(), activation_f32);
    }
}
