use std::io;

use crate::StageActivationCodec;

use super::{
    activation_codec::{
        ActivationShape, decode_activation, encode_activation, encoded_len,
        select_lossless_activation_codec_from_f32_payload,
    },
    invalid_data, state_flags,
    types::MAX_STAGE_DECODED_ACTIVATION_BYTES,
};

pub fn activation_wire_bytes(token_count: i32, n_embd: i32) -> io::Result<usize> {
    activation_wire_bytes_with_state_flags(token_count, n_embd, 0)
}

pub fn activation_wire_bytes_with_state_flags(
    token_count: i32,
    n_embd: i32,
    state_flag_bits: i32,
) -> io::Result<usize> {
    if token_count < 0 || n_embd < 0 {
        return Err(invalid_data("negative activation dimensions"));
    }
    let token_count = (token_count as usize)
        .checked_mul(activation_payload_multiplier_from_state_flags(
            state_flag_bits,
        ))
        .ok_or_else(|| invalid_data("activation token count overflow"))?;
    let n_embd = n_embd as usize;
    let elements = token_count
        .checked_mul(n_embd)
        .ok_or_else(|| invalid_data("activation element count overflow"))?;
    elements
        .checked_mul(4)
        .ok_or_else(|| invalid_data("activation byte count overflow"))
}

pub fn activation_wire_bytes_for_codec_with_state_flags(
    codec: StageActivationCodec,
    token_count: i32,
    n_embd: i32,
    state_flag_bits: i32,
) -> io::Result<usize> {
    encoded_len(
        codec,
        activation_shape(token_count, n_embd, state_flag_bits)?,
    )
}

pub(crate) fn activation_decoded_f32_bytes_with_state_flags(
    token_count: i32,
    n_embd: i32,
    state_flag_bits: i32,
) -> io::Result<usize> {
    activation_wire_bytes_with_state_flags(token_count, n_embd, state_flag_bits)
}

pub fn encode_f32_activation_payload(
    token_count: i32,
    n_embd: i32,
    f32_payload: &[u8],
) -> io::Result<Vec<u8>> {
    encode_f32_activation_payload_with_state_flags(token_count, n_embd, f32_payload, 0)
}

pub fn encode_f32_activation_payload_with_state_flags(
    token_count: i32,
    n_embd: i32,
    f32_payload: &[u8],
    state_flag_bits: i32,
) -> io::Result<Vec<u8>> {
    let expected_f32_bytes =
        activation_decoded_f32_bytes_with_state_flags(token_count, n_embd, state_flag_bits)?;
    if expected_f32_bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
        return Err(invalid_data(
            "decoded activation payload byte count exceeds maximum",
        ));
    }
    if f32_payload.len() != expected_f32_bytes {
        return Err(invalid_data("F32 activation payload size mismatch"));
    }
    encode_activation_payload_with_state_flags(
        StageActivationCodec::RawF32V1,
        token_count,
        n_embd,
        f32_payload,
        state_flag_bits,
    )
}

pub fn encode_activation_payload_with_state_flags(
    codec: StageActivationCodec,
    token_count: i32,
    n_embd: i32,
    f32_payload: &[u8],
    state_flag_bits: i32,
) -> io::Result<Vec<u8>> {
    let shape = activation_shape(token_count, n_embd, state_flag_bits)?;
    let expected_f32_bytes = shape
        .primary_rows
        .checked_add(shape.sideband_rows)
        .and_then(|rows| rows.checked_mul(shape.columns))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| invalid_data("decoded activation byte count overflow"))?;
    if expected_f32_bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
        return Err(invalid_data(
            "decoded activation payload byte count exceeds maximum",
        ));
    }
    if f32_payload.len() != expected_f32_bytes {
        return Err(invalid_data("F32 activation payload size mismatch"));
    }
    let values = f32_payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    encode_activation(codec, shape, &values)
}

pub fn select_lossless_activation_codec_with_state_flags(
    token_count: i32,
    n_embd: i32,
    f32_payload: &[u8],
    state_flag_bits: i32,
    permitted_codecs: &[StageActivationCodec],
) -> io::Result<StageActivationCodec> {
    let shape = activation_shape(token_count, n_embd, state_flag_bits)?;
    let expected_f32_bytes = shape
        .primary_rows
        .checked_add(shape.sideband_rows)
        .and_then(|rows| rows.checked_mul(shape.columns))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| invalid_data("decoded activation byte count overflow"))?;
    if expected_f32_bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
        return Err(invalid_data(
            "decoded activation payload byte count exceeds maximum",
        ));
    }
    if f32_payload.len() != expected_f32_bytes {
        return Err(invalid_data("F32 activation payload size mismatch"));
    }
    select_lossless_activation_codec_from_f32_payload(shape, f32_payload, permitted_codecs)
}

pub(crate) fn decode_activation_payload_with_state_flags(
    codec: StageActivationCodec,
    token_count: i32,
    n_embd: i32,
    payload: &[u8],
    state_flag_bits: i32,
) -> io::Result<Vec<u8>> {
    let values = decode_activation(
        codec,
        activation_shape(token_count, n_embd, state_flag_bits)?,
        payload,
    )?;
    let mut decoded = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        decoded.extend_from_slice(&value.to_le_bytes());
    }
    Ok(decoded)
}

fn activation_shape(
    token_count: i32,
    n_embd: i32,
    state_flag_bits: i32,
) -> io::Result<ActivationShape> {
    if token_count < 0 || n_embd <= 0 {
        return Err(invalid_data("negative activation dimensions"));
    }
    let primary_rows = token_count as usize;
    let multiplier = activation_payload_multiplier_from_state_flags(state_flag_bits);
    let sideband_rows = primary_rows
        .checked_mul(multiplier.saturating_sub(1))
        .ok_or_else(|| invalid_data("activation token count overflow"))?;
    Ok(ActivationShape::new(
        primary_rows,
        sideband_rows,
        n_embd as usize,
    ))
}

pub fn activation_payload_multiplier_from_state_flags(state_flag_bits: i32) -> usize {
    // Generation 7 graph boundaries report the complete Gemma3n AltUp tensor
    // width. Its flag describes the tensor semantics; it must not multiply an
    // already multidimensional payload a second time.
    if (state_flag_bits
        & (state_flags::INKLING_MTP_EMBD_SIDEBAND | state_flags::RWKV7_V_FIRST_SIDEBAND))
        != 0
    {
        2
    } else {
        1
    }
}
