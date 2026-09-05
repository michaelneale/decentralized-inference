use std::io::{self, Read, Write};

use crate::StageActivationCodec;

use super::{
    MAX_STAGE_ACTIVATION_BYTES, MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES,
    MAX_STAGE_DECODED_ACTIVATION_BYTES, MAX_STAGE_DRY_SEQUENCE_BREAKERS, MAX_STAGE_LOGIT_BIAS,
    MAX_STAGE_PREDICTED_TOKENS, MAX_STAGE_SAMPLERS, MAX_STAGE_SAMPLING_STRING_BYTES,
    MAX_STAGE_SIDEBAND_VALUES, MAX_STAGE_STATE_IMPORT_BYTES, READY_MAGIC, STAGE_STATE_VERSION,
    StageLogitBias, StageNativeMtpDraft, StageReply, StageReplyStats, StageReplyWindow,
    StageSamplingConfig, StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind,
    activation::{
        activation_decoded_f32_bytes_with_state_flags,
        activation_wire_bytes_for_codec_with_state_flags,
        decode_activation_payload_with_state_flags,
    },
    invalid_data, invalid_input,
};

pub fn send_ready(mut writer: impl Write) -> io::Result<()> {
    write_i32(&mut writer, READY_MAGIC)
}

pub fn recv_ready(mut reader: impl Read) -> io::Result<()> {
    let magic = read_i32(&mut reader)?;
    if magic != READY_MAGIC {
        return Err(invalid_data("stage ready magic mismatch"));
    }
    Ok(())
}

pub fn send_reply_ack(mut writer: impl Write) -> io::Result<()> {
    send_reply_ack_with_stats(&mut writer, StageReplyStats::default())
}

pub fn send_reply_ack_with_stats(mut writer: impl Write, stats: StageReplyStats) -> io::Result<()> {
    send_reply_message(
        &mut writer,
        &StageReply {
            kind: WireReplyKind::Ack,
            predicted: 0,
            predicted_tokens: Vec::new(),
            native_mtp_draft: None,
            window: StageReplyWindow::default(),
            stats,
        },
    )
}

pub fn send_reply_predicted(mut writer: impl Write, predicted: i32) -> io::Result<()> {
    send_reply_predicted_with_stats(&mut writer, predicted, StageReplyStats::default())
}

pub fn send_reply_predicted_with_stats(
    mut writer: impl Write,
    predicted: i32,
    stats: StageReplyStats,
) -> io::Result<()> {
    send_reply_predicted_with_tokens_window_and_stats(
        &mut writer,
        predicted,
        &[predicted],
        StageReplyWindow::default(),
        stats,
    )
}

pub fn send_reply_predicted_with_tokens_and_stats(
    mut writer: impl Write,
    predicted: i32,
    predicted_tokens: &[i32],
    stats: StageReplyStats,
) -> io::Result<()> {
    send_reply_predicted_with_tokens_window_and_stats(
        &mut writer,
        predicted,
        predicted_tokens,
        StageReplyWindow::default(),
        stats,
    )
}

pub fn send_reply_predicted_with_tokens_window_and_stats(
    mut writer: impl Write,
    predicted: i32,
    predicted_tokens: &[i32],
    window: StageReplyWindow,
    stats: StageReplyStats,
) -> io::Result<()> {
    if predicted_tokens.len() > MAX_STAGE_PREDICTED_TOKENS {
        return Err(invalid_input("too many predicted tokens"));
    }
    write_reply_header(
        &mut writer,
        WireReplyKind::PredictedToken,
        predicted,
        predicted_tokens,
        window,
    )?;
    write_native_mtp_draft(&mut writer, None)?;
    write_reply_stats(&mut writer, stats)
}

pub fn send_reply_predicted_tokens_with_stats(
    mut writer: impl Write,
    predicted_tokens: &[i32],
    stats: StageReplyStats,
) -> io::Result<()> {
    send_reply_predicted_tokens_with_window_and_stats(
        &mut writer,
        predicted_tokens,
        StageReplyWindow::default(),
        stats,
    )
}

pub fn send_reply_predicted_tokens_with_window_and_stats(
    mut writer: impl Write,
    predicted_tokens: &[i32],
    window: StageReplyWindow,
    stats: StageReplyStats,
) -> io::Result<()> {
    if predicted_tokens.len() > MAX_STAGE_PREDICTED_TOKENS {
        return Err(invalid_input("too many predicted tokens"));
    }
    let predicted = predicted_tokens.first().copied().unwrap_or(0);
    write_reply_header(
        &mut writer,
        WireReplyKind::PredictedTokens,
        predicted,
        predicted_tokens,
        window,
    )?;
    write_native_mtp_draft(&mut writer, None)?;
    write_reply_stats(&mut writer, stats)
}

pub fn send_reply_message(mut writer: impl Write, reply: &StageReply) -> io::Result<()> {
    if reply.predicted_tokens.len() > MAX_STAGE_PREDICTED_TOKENS {
        return Err(invalid_input("too many predicted tokens"));
    }
    write_reply_header(
        &mut writer,
        reply.kind,
        reply.predicted,
        &reply.predicted_tokens,
        reply.window,
    )?;
    write_native_mtp_draft(&mut writer, reply.native_mtp_draft.as_ref())?;
    write_reply_stats(&mut writer, reply.stats)
}

pub fn recv_reply(mut reader: impl Read) -> io::Result<StageReply> {
    let kind = WireReplyKind::try_from(read_i32(&mut reader)?)?;
    let predicted = read_i32(&mut reader)?;
    let predicted_count = checked_i32_len(
        read_i32(&mut reader)?,
        MAX_STAGE_PREDICTED_TOKENS,
        "negative predicted token count",
        "predicted token count exceeds maximum",
    )?;
    let mut predicted_tokens = Vec::with_capacity(predicted_count);
    for _ in 0..predicted_count {
        predicted_tokens.push(read_i32(&mut reader)?);
    }
    let window = read_reply_window(&mut reader)?;
    let native_mtp_draft = read_native_mtp_draft(&mut reader)?;
    let stats = read_reply_stats(&mut reader)?;
    Ok(StageReply {
        kind,
        predicted,
        predicted_tokens,
        native_mtp_draft,
        window,
        stats,
    })
}

fn write_reply_header(
    mut writer: impl Write,
    kind: WireReplyKind,
    predicted: i32,
    predicted_tokens: &[i32],
    window: StageReplyWindow,
) -> io::Result<()> {
    write_i32(&mut writer, kind as i32)?;
    write_i32(&mut writer, predicted)?;
    write_i32(
        &mut writer,
        i32::try_from(predicted_tokens.len())
            .map_err(|_| invalid_input("too many predicted tokens"))?,
    )?;
    for token in predicted_tokens {
        write_i32(&mut writer, *token)?;
    }
    write_reply_window(&mut writer, window)
}

fn write_native_mtp_draft(
    mut writer: impl Write,
    draft: Option<&StageNativeMtpDraft>,
) -> io::Result<()> {
    let Some(draft) = draft else {
        return write_i32(&mut writer, 0);
    };
    if draft.token_ids.len() > MAX_STAGE_PREDICTED_TOKENS {
        return Err(invalid_input("too many native MTP draft tokens"));
    }
    write_i32(&mut writer, 1)?;
    write_i32(
        &mut writer,
        i32::try_from(draft.token_ids.len())
            .map_err(|_| invalid_input("too many native MTP draft tokens"))?,
    )?;
    for token in &draft.token_ids {
        write_i32(&mut writer, *token)?;
    }
    write_i64(&mut writer, draft.proposal_compute_us)
}

fn read_native_mtp_draft(mut reader: impl Read) -> io::Result<Option<StageNativeMtpDraft>> {
    match read_i32(&mut reader)? {
        0 => Ok(None),
        1 => {
            let token_count = checked_i32_len(
                read_i32(&mut reader)?,
                MAX_STAGE_PREDICTED_TOKENS,
                "negative native MTP draft token count",
                "native MTP draft token count exceeds maximum",
            )?;
            let mut token_ids = Vec::with_capacity(token_count);
            for _ in 0..token_count {
                token_ids.push(read_i32(&mut reader)?);
            }
            Ok(Some(StageNativeMtpDraft {
                token_ids,
                proposal_compute_us: read_i64(&mut reader)?,
            }))
        }
        _ => Err(invalid_data("unknown native MTP draft reply marker")),
    }
}

pub fn write_stage_message(mut writer: impl Write, message: &StageWireMessage) -> io::Result<()> {
    // Wire v4 fixed prefix, little-endian:
    // kind, pos_start, token_count, token_sideband_count, position_sideband_count (5 x i32);
    // StageStateHeader (10 x i32); activation wire byte count (1 x i32);
    // request_id, session_id (2 x u64);
    // optional StageSamplingConfig follows when state_flags::SAMPLING is set.
    // Token sideband, raw StateImport bytes, or activation bytes follow this
    // prefix, so prefill overhead stays independent of ID string length.
    write_i32(&mut writer, message.kind as i32)?;
    write_i32(&mut writer, message.pos_start)?;
    write_i32(&mut writer, message.token_count)?;
    if message.tokens.len() > MAX_STAGE_SIDEBAND_VALUES {
        return Err(invalid_input("too many tokens"));
    }
    write_i32(
        &mut writer,
        i32::try_from(message.tokens.len()).map_err(|_| invalid_input("too many tokens"))?,
    )?;
    if message.positions.len() > MAX_STAGE_SIDEBAND_VALUES {
        return Err(invalid_input("too many position sideband values"));
    }
    write_i32(
        &mut writer,
        i32::try_from(message.positions.len())
            .map_err(|_| invalid_input("too many position sideband values"))?,
    )?;

    let mut state = message.state;
    if message.sampling.is_some() {
        state.flags |= super::state_flags::SAMPLING;
    } else {
        state.flags &= !super::state_flags::SAMPLING;
    }
    if message.chat_sampling_metadata.is_some() {
        state.flags |= super::state_flags::CHAT_SAMPLING_METADATA;
    } else {
        state.flags &= !super::state_flags::CHAT_SAMPLING_METADATA;
    }
    let activation_wire_byte_count = if message.kind == WireMessageKind::StateImport
        || state.source_stage_index < 0
        || message.kind.is_activationless_prefix_cache_control()
    {
        if !message.activation.is_empty() {
            return Err(invalid_input(
                "activationless stage message contains activation payload",
            ));
        }
        0
    } else {
        if message.activation.len() > MAX_STAGE_ACTIVATION_BYTES {
            return Err(invalid_input(
                "activation payload byte count exceeds maximum",
            ));
        }
        i32::try_from(message.activation.len())
            .map_err(|_| invalid_input("activation payload byte count exceeds maximum"))?
    };
    write_state_header(&mut writer, state)?;
    write_i32(&mut writer, activation_wire_byte_count)?;
    write_u64(&mut writer, message.request_id)?;
    write_u64(&mut writer, message.session_id)?;
    if let Some(sampling) = message.sampling.as_ref() {
        write_sampling_config(&mut writer, sampling)?;
    }
    if let Some(metadata) = message.chat_sampling_metadata.as_ref() {
        let bytes = metadata.as_bytes();
        if bytes.len() > MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES {
            return Err(invalid_input("chat sampling metadata is too large"));
        }
        write_u32(
            &mut writer,
            u32::try_from(bytes.len())
                .map_err(|_| invalid_input("chat sampling metadata is too large"))?,
        )?;
        writer.write_all(bytes)?;
    }

    if message.kind == WireMessageKind::StateImport {
        let raw_byte_count = usize::try_from(message.token_count)
            .map_err(|_| invalid_input("state import raw byte count mismatch"))?;
        if raw_byte_count != message.raw_bytes.len() {
            return Err(invalid_input("state import raw byte count mismatch"));
        }
        if raw_byte_count > MAX_STAGE_STATE_IMPORT_BYTES {
            return Err(invalid_input("state import raw byte count exceeds maximum"));
        }
        writer.write_all(&message.raw_bytes)?;
        return Ok(());
    }
    write_i32_slice(&mut writer, &message.tokens)?;
    write_i32_slice(&mut writer, &message.positions)?;
    writer.write_all(&message.activation)?;
    Ok(())
}

pub fn read_stage_message(reader: impl Read, n_embd: i32) -> io::Result<StageWireMessage> {
    read_stage_message_inner(reader, n_embd, None)
}

pub fn read_stage_message_for_codec(
    reader: impl Read,
    n_embd: i32,
    expected_codec: StageActivationCodec,
) -> io::Result<StageWireMessage> {
    read_stage_message_inner(reader, n_embd, Some(expected_codec))
}

fn read_stage_message_inner(
    mut reader: impl Read,
    n_embd: i32,
    expected_codec: Option<StageActivationCodec>,
) -> io::Result<StageWireMessage> {
    let kind = WireMessageKind::try_from(read_i32(&mut reader)?)?;
    let pos_start = read_i32(&mut reader)?;
    let token_count = read_i32(&mut reader)?;
    let token_sideband_count = read_i32(&mut reader)?;
    let position_sideband_count = read_i32(&mut reader)?;
    let state = read_state_header(&mut reader)?;
    if state.version != STAGE_STATE_VERSION {
        return Err(invalid_data("unsupported stage state version"));
    }
    let activation_wire_byte_count = checked_i32_len(
        read_i32(&mut reader)?,
        MAX_STAGE_ACTIVATION_BYTES,
        "negative activation payload byte count",
        "activation payload byte count exceeds maximum",
    )?;
    let request_id = read_u64(&mut reader)?;
    let session_id = read_u64(&mut reader)?;
    let activation_bytes = if kind == WireMessageKind::StateImport
        || kind == WireMessageKind::Stop
        || state.source_stage_index < 0
        || kind.is_activationless_prefix_cache_control()
    {
        if activation_wire_byte_count != 0 {
            return Err(invalid_data(
                "activationless stage message contains activation payload",
            ));
        }
        0
    } else {
        if expected_codec.is_some_and(|expected| expected != state.activation_codec) {
            return Err(invalid_data("stage activation codec mismatch"));
        }
        let expected_bytes = activation_wire_bytes_for_codec_with_state_flags(
            state.activation_codec,
            token_count,
            n_embd,
            state.flags,
        )?;
        if activation_wire_byte_count != expected_bytes {
            return Err(invalid_data("activation payload size mismatch"));
        }
        let decoded_activation_bytes =
            activation_decoded_f32_bytes_with_state_flags(token_count, n_embd, state.flags)?;
        if decoded_activation_bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
            return Err(invalid_data(
                "decoded activation payload byte count exceeds maximum",
            ));
        }
        expected_bytes
    };
    let sampling = if (state.flags & super::state_flags::SAMPLING) != 0 {
        Some(read_sampling_config(&mut reader)?)
    } else {
        None
    };
    let chat_sampling_metadata = if (state.flags & super::state_flags::CHAT_SAMPLING_METADATA) != 0
    {
        let len = checked_u32_len(
            read_u32(&mut reader)?,
            MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES,
            "chat sampling metadata length exceeds maximum",
        )?;
        let mut bytes = vec![0_u8; len];
        reader.read_exact(&mut bytes)?;
        Some(
            String::from_utf8(bytes)
                .map_err(|_| invalid_data("chat sampling metadata is not UTF-8"))?,
        )
    } else {
        None
    };
    if kind == WireMessageKind::Stop {
        return Ok(StageWireMessage {
            kind,
            pos_start,
            token_count,
            state,
            request_id,
            session_id,
            sampling,
            chat_sampling_metadata,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        });
    }
    if token_count < 0 || token_sideband_count < 0 || position_sideband_count < 0 {
        return Err(invalid_data("negative wire count"));
    }
    let token_sideband_count = checked_i32_len(
        token_sideband_count,
        MAX_STAGE_SIDEBAND_VALUES,
        "negative wire count",
        "token sideband count exceeds maximum",
    )?;
    let position_sideband_count = checked_i32_len(
        position_sideband_count,
        MAX_STAGE_SIDEBAND_VALUES,
        "negative wire count",
        "position sideband count exceeds maximum",
    )?;
    if kind == WireMessageKind::StateImport {
        let raw_byte_count = checked_i32_len(
            token_count,
            MAX_STAGE_STATE_IMPORT_BYTES,
            "negative wire count",
            "state import byte count exceeds maximum",
        )?;
        let mut raw_bytes = vec![0; raw_byte_count];
        reader.read_exact(&mut raw_bytes)?;
        return Ok(StageWireMessage {
            kind,
            pos_start,
            token_count,
            state,
            request_id,
            session_id,
            sampling,
            chat_sampling_metadata,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes,
        });
    }

    let tokens = read_i32_values(&mut reader, token_sideband_count)?;
    let positions = read_i32_values(&mut reader, position_sideband_count)?;
    let mut wire_activation = vec![0; activation_bytes];
    if activation_bytes > 0 {
        reader.read_exact(&mut wire_activation)?;
    }
    let activation = if wire_activation.is_empty() {
        Vec::new()
    } else {
        decode_activation_payload_with_state_flags(
            state.activation_codec,
            token_count,
            n_embd,
            &wire_activation,
            state.flags,
        )?
    };
    Ok(StageWireMessage {
        kind,
        pos_start,
        token_count,
        state,
        request_id,
        session_id,
        sampling,
        chat_sampling_metadata,
        tokens,
        positions,
        activation,
        raw_bytes: Vec::new(),
    })
}

fn checked_i32_len(
    value: i32,
    max: usize,
    negative_message: &'static str,
    too_large_message: &'static str,
) -> io::Result<usize> {
    if value < 0 {
        return Err(invalid_data(negative_message));
    }
    let value = usize::try_from(value).map_err(|_| invalid_data(too_large_message))?;
    if value > max {
        return Err(invalid_data(too_large_message));
    }
    Ok(value)
}

fn checked_u32_len(value: u32, max: usize, too_large_message: &'static str) -> io::Result<usize> {
    let value = usize::try_from(value).map_err(|_| invalid_data(too_large_message))?;
    if value > max {
        return Err(invalid_data(too_large_message));
    }
    Ok(value)
}

fn write_state_header(mut writer: impl Write, state: StageStateHeader) -> io::Result<()> {
    write_i32(&mut writer, state.version)?;
    write_i32(&mut writer, state.seq_id)?;
    write_i32(&mut writer, state.phase)?;
    write_i32(&mut writer, state.flags)?;
    write_i32(&mut writer, state.checkpoint_generation)?;
    write_i32(&mut writer, state.prompt_token_count)?;
    write_i32(&mut writer, state.decode_step)?;
    write_i32(&mut writer, state.current_token)?;
    write_i32(&mut writer, state.source_stage_index)?;
    write_i32(&mut writer, state.activation_codec.binary_wire_id())
}

fn read_state_header(mut reader: impl Read) -> io::Result<StageStateHeader> {
    let version = read_i32(&mut reader)?;
    let seq_id = read_i32(&mut reader)?;
    let phase = read_i32(&mut reader)?;
    let flags = read_i32(&mut reader)?;
    let checkpoint_generation = read_i32(&mut reader)?;
    let prompt_token_count = read_i32(&mut reader)?;
    let decode_step = read_i32(&mut reader)?;
    let current_token = read_i32(&mut reader)?;
    let source_stage_index = read_i32(&mut reader)?;
    let activation_codec = StageActivationCodec::from_binary_wire_id(read_i32(&mut reader)?)
        .ok_or_else(|| invalid_data("unknown stage activation codec"))?;
    Ok(StageStateHeader {
        version,
        seq_id,
        phase,
        flags,
        checkpoint_generation,
        prompt_token_count,
        decode_step,
        current_token,
        source_stage_index,
        activation_codec,
    })
}

fn write_sampling_config(mut writer: impl Write, sampling: &StageSamplingConfig) -> io::Result<()> {
    write_u32(&mut writer, sampling.flags)?;
    write_u32(&mut writer, sampling.seed)?;
    write_f32(&mut writer, sampling.temperature)?;
    write_f32(&mut writer, sampling.top_p)?;
    write_i32(&mut writer, sampling.top_k)?;
    write_f32(&mut writer, sampling.min_p)?;
    write_f32(&mut writer, sampling.presence_penalty)?;
    write_f32(&mut writer, sampling.frequency_penalty)?;
    write_f32(&mut writer, sampling.repeat_penalty)?;
    write_i32(&mut writer, sampling.penalty_last_n)?;
    let count = sampling.logit_bias.len().min(MAX_STAGE_LOGIT_BIAS);
    write_u32(&mut writer, count as u32)?;
    for bias in sampling.logit_bias.iter().take(count) {
        write_i32(&mut writer, bias.token_id)?;
        write_f32(&mut writer, bias.bias)?;
    }
    write_f32(&mut writer, sampling.typical_p)?;
    write_f32(&mut writer, sampling.top_nsigma)?;
    write_f32(&mut writer, sampling.dynatemp_range)?;
    write_f32(&mut writer, sampling.dynatemp_exponent)?;
    write_f32(&mut writer, sampling.dry_multiplier)?;
    write_f32(&mut writer, sampling.dry_base)?;
    write_i32(&mut writer, sampling.dry_allowed_length)?;
    write_i32(&mut writer, sampling.dry_penalty_last_n)?;
    write_string_list(
        &mut writer,
        &sampling.dry_sequence_breakers,
        MAX_STAGE_DRY_SEQUENCE_BREAKERS,
    )?;
    write_f32(&mut writer, sampling.xtc_probability)?;
    write_f32(&mut writer, sampling.xtc_threshold)?;
    write_i32(&mut writer, sampling.mirostat_mode)?;
    write_f32(&mut writer, sampling.mirostat_entropy)?;
    write_f32(&mut writer, sampling.mirostat_learning_rate)?;
    write_string_list(&mut writer, &sampling.samplers, MAX_STAGE_SAMPLERS)?;
    write_u32(&mut writer, u32::from(sampling.ignore_eos))?;
    Ok(())
}

fn read_sampling_config(mut reader: impl Read) -> io::Result<StageSamplingConfig> {
    let mut sampling = StageSamplingConfig {
        flags: read_u32(&mut reader)?,
        seed: read_u32(&mut reader)?,
        temperature: read_f32(&mut reader)?,
        top_p: read_f32(&mut reader)?,
        top_k: read_i32(&mut reader)?,
        min_p: read_f32(&mut reader)?,
        presence_penalty: read_f32(&mut reader)?,
        frequency_penalty: read_f32(&mut reader)?,
        repeat_penalty: read_f32(&mut reader)?,
        penalty_last_n: read_i32(&mut reader)?,
        logit_bias: Vec::new(),
        ..StageSamplingConfig::default()
    };
    let logit_bias_count = usize::try_from(read_u32(&mut reader)?)
        .map_err(|_| invalid_data("logit bias count overflows usize"))?;
    if logit_bias_count > MAX_STAGE_LOGIT_BIAS {
        return Err(invalid_data("logit bias count exceeds maximum"));
    }
    sampling.logit_bias.reserve(logit_bias_count);
    for _ in 0..logit_bias_count {
        sampling.logit_bias.push(StageLogitBias {
            token_id: read_i32(&mut reader)?,
            bias: read_f32(&mut reader)?,
        });
    }
    sampling.typical_p = read_f32(&mut reader)?;
    sampling.top_nsigma = read_f32(&mut reader)?;
    sampling.dynatemp_range = read_f32(&mut reader)?;
    sampling.dynatemp_exponent = read_f32(&mut reader)?;
    sampling.dry_multiplier = read_f32(&mut reader)?;
    sampling.dry_base = read_f32(&mut reader)?;
    sampling.dry_allowed_length = read_i32(&mut reader)?;
    sampling.dry_penalty_last_n = read_i32(&mut reader)?;
    sampling.dry_sequence_breakers =
        read_string_list(&mut reader, MAX_STAGE_DRY_SEQUENCE_BREAKERS)?;
    sampling.xtc_probability = read_f32(&mut reader)?;
    sampling.xtc_threshold = read_f32(&mut reader)?;
    sampling.mirostat_mode = read_i32(&mut reader)?;
    sampling.mirostat_entropy = read_f32(&mut reader)?;
    sampling.mirostat_learning_rate = read_f32(&mut reader)?;
    sampling.samplers = read_string_list(&mut reader, MAX_STAGE_SAMPLERS)?;
    sampling.ignore_eos = read_u32(&mut reader)? != 0;
    Ok(sampling)
}

fn write_string_list(
    mut writer: impl Write,
    values: &[String],
    maximum_count: usize,
) -> io::Result<()> {
    let count = values.len().min(maximum_count);
    write_u32(&mut writer, count as u32)?;
    for value in values.iter().take(count) {
        let bytes = value.as_bytes();
        if bytes.len() > MAX_STAGE_SAMPLING_STRING_BYTES {
            return Err(invalid_data("sampling string exceeds maximum length"));
        }
        write_u32(&mut writer, bytes.len() as u32)?;
        writer.write_all(bytes)?;
    }
    Ok(())
}

fn read_string_list(mut reader: impl Read, maximum_count: usize) -> io::Result<Vec<String>> {
    let count = usize::try_from(read_u32(&mut reader)?)
        .map_err(|_| invalid_data("sampling string count overflows usize"))?;
    if count > maximum_count {
        return Err(invalid_data("sampling string count exceeds maximum"));
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let length = usize::try_from(read_u32(&mut reader)?)
            .map_err(|_| invalid_data("sampling string length overflows usize"))?;
        if length > MAX_STAGE_SAMPLING_STRING_BYTES {
            return Err(invalid_data("sampling string exceeds maximum length"));
        }
        let mut bytes = vec![0_u8; length];
        reader.read_exact(&mut bytes)?;
        values.push(
            String::from_utf8(bytes).map_err(|_| invalid_data("sampling string is not UTF-8"))?,
        );
    }
    Ok(values)
}

const REPLY_STATS_FIELD_COUNT: usize = 27;
const REPLY_STATS_WIRE_BYTES: usize = REPLY_STATS_FIELD_COUNT * std::mem::size_of::<i64>();

fn write_reply_stats(mut writer: impl Write, stats: StageReplyStats) -> io::Result<()> {
    let fields = reply_stats_fields(stats);
    let mut bytes = [0_u8; REPLY_STATS_WIRE_BYTES];
    for (chunk, value) in bytes
        .as_chunks_mut::<{ std::mem::size_of::<i64>() }>()
        .0
        .iter_mut()
        .zip(fields)
    {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    writer.write_all(&bytes)
}

fn read_reply_stats(mut reader: impl Read) -> io::Result<StageReplyStats> {
    let mut bytes = [0_u8; REPLY_STATS_WIRE_BYTES];
    reader.read_exact(&mut bytes)?;
    let mut fields = [0_i64; REPLY_STATS_FIELD_COUNT];
    for (field, chunk) in fields
        .iter_mut()
        .zip(bytes.as_chunks::<{ std::mem::size_of::<i64>() }>().0.iter())
    {
        *field = i64::from_le_bytes(*chunk);
    }
    Ok(reply_stats_from_fields(fields))
}

fn write_reply_window(mut writer: impl Write, window: StageReplyWindow) -> io::Result<()> {
    write_i32(&mut writer, window.window_id)
}

fn read_reply_window(mut reader: impl Read) -> io::Result<StageReplyWindow> {
    Ok(StageReplyWindow {
        window_id: read_i32(&mut reader)?,
    })
}

fn reply_stats_fields(stats: StageReplyStats) -> [i64; REPLY_STATS_FIELD_COUNT] {
    [
        stats.kv_lookup_hits,
        stats.kv_lookup_misses,
        stats.kv_lookup_errors,
        stats.kv_imported_pages,
        stats.kv_imported_tokens,
        stats.kv_recorded_pages,
        stats.kv_recorded_bytes,
        stats.kv_hit_stage_mask,
        stats.kv_record_stage_mask,
        stats.verify_window_compute_us,
        stats.verify_window_forward_write_us,
        stats.verify_window_downstream_wait_us,
        stats.verify_window_total_us,
        stats.verify_window_stage_count,
        stats.verify_window_request_count,
        stats.verify_window_token_count,
        stats.verify_window_max_tokens,
        stats.prefill_edge_write_us_max,
        stats.prefill_edge_wait_us_max,
        stats.prefill_edge_total_us_max,
        stats.prefill_edge_stage_index,
        stats.prefill_edge_activation_bytes_max,
        stats.prefill_edge_observation_count,
        stats.prefill_compute_us_at_slowest_rate,
        stats.prefill_compute_stage_index,
        stats.prefill_compute_token_count_at_slowest_rate,
        stats.prefill_compute_observation_count,
    ]
}

fn reply_stats_from_fields(fields: [i64; REPLY_STATS_FIELD_COUNT]) -> StageReplyStats {
    StageReplyStats {
        kv_lookup_hits: fields[0],
        kv_lookup_misses: fields[1],
        kv_lookup_errors: fields[2],
        kv_imported_pages: fields[3],
        kv_imported_tokens: fields[4],
        kv_recorded_pages: fields[5],
        kv_recorded_bytes: fields[6],
        kv_hit_stage_mask: fields[7],
        kv_record_stage_mask: fields[8],
        verify_window_compute_us: fields[9],
        verify_window_forward_write_us: fields[10],
        verify_window_downstream_wait_us: fields[11],
        verify_window_total_us: fields[12],
        verify_window_stage_count: fields[13],
        verify_window_request_count: fields[14],
        verify_window_token_count: fields[15],
        verify_window_max_tokens: fields[16],
        prefill_edge_write_us_max: fields[17],
        prefill_edge_wait_us_max: fields[18],
        prefill_edge_total_us_max: fields[19],
        prefill_edge_stage_index: fields[20],
        prefill_edge_activation_bytes_max: fields[21],
        prefill_edge_observation_count: fields[22],
        prefill_compute_us_at_slowest_rate: fields[23],
        prefill_compute_stage_index: fields[24],
        prefill_compute_token_count_at_slowest_rate: fields[25],
        prefill_compute_observation_count: fields[26],
    }
}

const I32_WIRE_BYTES: usize = std::mem::size_of::<i32>();

fn read_i32_values(mut reader: impl Read, count: usize) -> io::Result<Vec<i32>> {
    const VALUES_PER_CHUNK: usize = 4_096;
    let mut values = Vec::with_capacity(count);
    let mut bytes = [0_u8; VALUES_PER_CHUNK * I32_WIRE_BYTES];
    let mut remaining = count;
    while remaining > 0 {
        let chunk_values = remaining.min(VALUES_PER_CHUNK);
        let chunk_bytes = chunk_values * I32_WIRE_BYTES;
        reader.read_exact(&mut bytes[..chunk_bytes])?;
        values.extend(
            bytes[..chunk_bytes]
                .as_chunks::<I32_WIRE_BYTES>()
                .0
                .iter()
                .map(|chunk| i32::from_le_bytes(*chunk)),
        );
        remaining -= chunk_values;
    }
    Ok(values)
}

fn write_i32_slice(mut writer: impl Write, values: &[i32]) -> io::Result<()> {
    const VALUES_PER_CHUNK: usize = 4_096;
    let mut bytes = [0_u8; VALUES_PER_CHUNK * I32_WIRE_BYTES];
    for values in values.chunks(VALUES_PER_CHUNK) {
        let chunk_bytes = std::mem::size_of_val(values);
        for (bytes, value) in bytes[..chunk_bytes]
            .as_chunks_mut::<I32_WIRE_BYTES>()
            .0
            .iter_mut()
            .zip(values)
        {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes[..chunk_bytes])?;
    }
    Ok(())
}

fn read_i32(mut reader: impl Read) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn write_i32(mut writer: impl Write, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_i64(mut reader: impl Read) -> io::Result<i64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn write_i64(mut writer: impl Write, value: i64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(mut reader: impl Read) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn write_u32(mut writer: impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_f32(mut reader: impl Read) -> io::Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn write_f32(mut writer: impl Write, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u64(mut reader: impl Read) -> io::Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_u64(mut writer: impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
