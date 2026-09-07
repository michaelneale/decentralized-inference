use std::io;

use crate::StageActivationCodec;

use super::{
    invalid_data,
    types::{MAX_STAGE_ACTIVATION_BYTES, MAX_STAGE_DECODED_ACTIVATION_BYTES},
};

/// Pure activation payload codecs. This module intentionally does not define a
/// frame header: framing, codec negotiation, and server wiring own that seam.
/// Logical activation dimensions. Sideband rows are encoded after primary
/// rows and use the same width and codec rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActivationShape {
    pub primary_rows: usize,
    pub sideband_rows: usize,
    pub columns: usize,
}

impl ActivationShape {
    pub(crate) const fn new(primary_rows: usize, sideband_rows: usize, columns: usize) -> Self {
        Self {
            primary_rows,
            sideband_rows,
            columns,
        }
    }

    fn rows(self) -> io::Result<usize> {
        self.primary_rows
            .checked_add(self.sideband_rows)
            .ok_or_else(|| invalid_data("activation row count overflow"))
    }

    fn elements(self) -> io::Result<usize> {
        let rows = self.rows()?;
        if self.columns == 0 {
            return Err(invalid_data("activation columns must be positive"));
        }
        rows.checked_mul(self.columns)
            .ok_or_else(|| invalid_data("activation element count overflow"))
    }

    fn decoded_bytes(self) -> io::Result<usize> {
        self.elements()?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid_data("decoded activation byte count overflow"))
    }

    fn validate_decoded_limit(self) -> io::Result<usize> {
        let bytes = self.decoded_bytes()?;
        if bytes > MAX_STAGE_DECODED_ACTIVATION_BYTES {
            return Err(invalid_data(
                "decoded activation payload byte count exceeds maximum",
            ));
        }
        Ok(bytes)
    }
}

pub(crate) fn encode_activation(
    codec: StageActivationCodec,
    shape: ActivationShape,
    values: &[f32],
) -> io::Result<Vec<u8>> {
    let elements = shape.elements()?;
    shape.validate_decoded_limit()?;
    if values.len() != elements {
        return Err(invalid_data("activation value count does not match shape"));
    }
    match codec {
        StageActivationCodec::RawF32V1 => encode_raw_f32(values),
        StageActivationCodec::F16RneV1 => {
            validate_finite(values)?;
            encode_f16(values)
        }
        StageActivationCodec::Bf16RneV1 => {
            validate_finite(values)?;
            encode_bf16(values)
        }
        StageActivationCodec::S8RowF32RneV1 => {
            validate_finite(values)?;
            encode_s8_rows(shape, values)
        }
    }
}

pub(crate) fn decode_activation(
    codec: StageActivationCodec,
    shape: ActivationShape,
    payload: &[u8],
) -> io::Result<Vec<f32>> {
    let elements = shape.elements()?;
    shape.validate_decoded_limit()?;
    match codec {
        StageActivationCodec::RawF32V1 => decode_raw_f32(payload, elements),
        StageActivationCodec::F16RneV1 => decode_f16(payload, elements),
        StageActivationCodec::Bf16RneV1 => decode_bf16(payload, elements),
        StageActivationCodec::S8RowF32RneV1 => decode_s8_rows(shape, payload),
    }
}

pub(crate) fn encoded_len(
    codec: StageActivationCodec,
    shape: ActivationShape,
) -> io::Result<usize> {
    let rows = shape.rows()?;
    match codec {
        StageActivationCodec::RawF32V1 => checked_wire_bytes(shape.elements()?, 4),
        StageActivationCodec::F16RneV1 | StageActivationCodec::Bf16RneV1 => {
            checked_wire_bytes(shape.elements()?, 2)
        }
        StageActivationCodec::S8RowF32RneV1 => {
            let row_bytes = shape
                .columns
                .checked_add(4)
                .ok_or_else(|| invalid_data("S8 row byte count overflow"))?;
            checked_wire_bytes(rows, row_bytes)
        }
    }
}

/// Selects the smallest permitted activation encoding that reconstructs the
/// realized F32 values bit-for-bit. RawF32 is mandatory so the caller always
/// has an admitted lossless fallback. BF16 wins an equal-size exact tie with
/// F16; otherwise F16 is selected when it alone is exact. A compact codec is
/// selected only when the complete payload is smaller.
#[cfg(test)]
pub(crate) fn select_lossless_activation_codec(
    shape: ActivationShape,
    values: &[f32],
    permitted_codecs: &[StageActivationCodec],
) -> io::Result<StageActivationCodec> {
    let elements = shape.elements()?;
    shape.validate_decoded_limit()?;
    if values.len() != elements {
        return Err(invalid_data("activation value count does not match shape"));
    }
    select_lossless_activation_codec_from_values(shape, values.iter().copied(), permitted_codecs)
}

pub(crate) fn select_lossless_activation_codec_from_f32_payload(
    shape: ActivationShape,
    f32_payload: &[u8],
    permitted_codecs: &[StageActivationCodec],
) -> io::Result<StageActivationCodec> {
    let elements = shape.elements()?;
    shape.validate_decoded_limit()?;
    validate_payload_bytes(f32_payload.len(), checked_wire_bytes(elements, 4)?)?;
    select_lossless_activation_codec_from_values(
        shape,
        f32_payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes)),
        permitted_codecs,
    )
}

fn select_lossless_activation_codec_from_values(
    shape: ActivationShape,
    values: impl Iterator<Item = f32>,
    permitted_codecs: &[StageActivationCodec],
) -> io::Result<StageActivationCodec> {
    if !permitted_codecs.contains(&StageActivationCodec::RawF32V1) {
        return Err(invalid_data(
            "lossless activation selection requires RawF32 fallback",
        ));
    }

    let mut bf16_exact = permitted_codecs.contains(&StageActivationCodec::Bf16RneV1);
    let mut f16_exact = permitted_codecs.contains(&StageActivationCodec::F16RneV1);
    for value in values {
        if !value.is_finite() {
            bf16_exact = false;
            f16_exact = false;
            continue;
        }
        bf16_exact &= value.to_bits() & 0xffff == 0;
        f16_exact &= f16_bits_to_f32(f32_to_f16_bits(value)).to_bits() == value.to_bits();
    }

    let raw_len = encoded_len(StageActivationCodec::RawF32V1, shape)?;
    if bf16_exact && encoded_len(StageActivationCodec::Bf16RneV1, shape)? < raw_len {
        return Ok(StageActivationCodec::Bf16RneV1);
    }
    if f16_exact && encoded_len(StageActivationCodec::F16RneV1, shape)? < raw_len {
        return Ok(StageActivationCodec::F16RneV1);
    }
    Ok(StageActivationCodec::RawF32V1)
}

fn encode_raw_f32(values: &[f32]) -> io::Result<Vec<u8>> {
    let wire_bytes = checked_wire_bytes(values.len(), std::mem::size_of::<f32>())?;
    let mut out = Vec::with_capacity(wire_bytes);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

fn decode_raw_f32(payload: &[u8], elements: usize) -> io::Result<Vec<f32>> {
    validate_payload_bytes(payload.len(), checked_wire_bytes(elements, 4)?)?;
    let mut out = Vec::with_capacity(elements);
    for bytes in payload.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*bytes));
    }
    Ok(out)
}

fn encode_f16(values: &[f32]) -> io::Result<Vec<u8>> {
    let wire_bytes = checked_wire_bytes(values.len(), 2)?;
    let mut out = Vec::with_capacity(wire_bytes);
    for value in values {
        let bits = f32_to_f16_bits(*value);
        let rounded = f16_bits_to_f32(bits);
        if !rounded.is_finite() {
            return Err(invalid_data("F16 activation value is out of range"));
        }
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(out)
}

fn decode_f16(payload: &[u8], elements: usize) -> io::Result<Vec<f32>> {
    validate_payload_bytes(payload.len(), checked_wire_bytes(elements, 2)?)?;
    let mut out = Vec::with_capacity(elements);
    for bytes in payload.as_chunks::<2>().0 {
        let value = f16_bits_to_f32(u16::from_le_bytes(*bytes));
        if !value.is_finite() {
            return Err(invalid_data("activation payload contains non-finite value"));
        }
        out.push(value);
    }
    Ok(out)
}

fn encode_bf16(values: &[f32]) -> io::Result<Vec<u8>> {
    let wire_bytes = checked_wire_bytes(values.len(), 2)?;
    let mut out = Vec::with_capacity(wire_bytes);
    for value in values {
        let bits = f32_to_bf16_bits(*value);
        if !f32::from_bits(u32::from(bits) << 16).is_finite() {
            return Err(invalid_data("BF16 activation value is out of range"));
        }
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(out)
}

fn decode_bf16(payload: &[u8], elements: usize) -> io::Result<Vec<f32>> {
    validate_payload_bytes(payload.len(), checked_wire_bytes(elements, 2)?)?;
    let mut out = Vec::with_capacity(elements);
    for bytes in payload.as_chunks::<2>().0 {
        let value = f32::from_bits(u32::from(u16::from_le_bytes(*bytes)) << 16);
        if !value.is_finite() {
            return Err(invalid_data("activation payload contains non-finite value"));
        }
        out.push(value);
    }
    Ok(out)
}

// Each row is self-contained: little-endian finite positive f32 scale followed
// by `columns` signed i8 values. This makes concatenating independently encoded
// row chunks byte-identical to encoding the complete shape at once.
fn encode_s8_rows(shape: ActivationShape, values: &[f32]) -> io::Result<Vec<u8>> {
    let rows = shape.rows()?;
    let row_bytes = shape
        .columns
        .checked_add(4)
        .ok_or_else(|| invalid_data("S8 row byte count overflow"))?;
    let wire_bytes = checked_wire_bytes(rows, row_bytes)?;
    let mut out = Vec::with_capacity(wire_bytes);

    for row in values.chunks_exact(shape.columns) {
        let max_abs = row
            .iter()
            .fold(0.0_f32, |max_abs, value| max_abs.max(value.abs()));
        let scale = if max_abs == 0.0 {
            1.0
        } else {
            ((f64::from(max_abs) / 127.0) as f32).max(f32::from_bits(1))
        };
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid_data(
                "S8 activation row scale is not finite and positive",
            ));
        }
        out.extend_from_slice(&scale.to_le_bytes());
        for value in row {
            let quantized = round_to_even_i8(f64::from(*value) / f64::from(scale));
            if !(f32::from(quantized) * scale).is_finite() {
                return Err(invalid_data("S8 activation value is out of range"));
            }
            out.push(quantized as u8);
        }
    }
    Ok(out)
}

fn decode_s8_rows(shape: ActivationShape, payload: &[u8]) -> io::Result<Vec<f32>> {
    let rows = shape.rows()?;
    let row_bytes = shape
        .columns
        .checked_add(4)
        .ok_or_else(|| invalid_data("S8 row byte count overflow"))?;
    validate_payload_bytes(payload.len(), checked_wire_bytes(rows, row_bytes)?)?;

    let mut out = Vec::with_capacity(shape.elements()?);
    for row in payload.chunks_exact(row_bytes) {
        let scale = f32::from_le_bytes(row[..4].try_into().expect("S8 scale prefix"));
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid_data(
                "S8 activation row scale is not finite and positive",
            ));
        }
        for value in &row[4..] {
            let quantized = *value as i8;
            if quantized == i8::MIN {
                return Err(invalid_data(
                    "S8 activation payload contains reserved -128 value",
                ));
            }
            let decoded = f32::from(quantized) * scale;
            if !decoded.is_finite() {
                return Err(invalid_data("activation payload contains non-finite value"));
            }
            out.push(decoded);
        }
    }
    Ok(out)
}

fn validate_finite(values: &[f32]) -> io::Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_data("activation values must be finite"));
    }
    Ok(())
}

fn checked_wire_bytes(count: usize, bytes_per_value: usize) -> io::Result<usize> {
    let bytes = count
        .checked_mul(bytes_per_value)
        .ok_or_else(|| invalid_data("activation wire byte count overflow"))?;
    if bytes > MAX_STAGE_ACTIVATION_BYTES {
        return Err(invalid_data(
            "activation payload byte count exceeds maximum",
        ));
    }
    Ok(bytes)
}

fn validate_payload_bytes(actual: usize, expected: usize) -> io::Result<()> {
    if actual != expected {
        return Err(invalid_data("activation payload size mismatch"));
    }
    Ok(())
}

fn round_to_even_i8(value: f64) -> i8 {
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else if (floor as i64) & 1 == 0 {
        floor
    } else {
        floor + 1.0
    };
    rounded.clamp(-127.0, 127.0) as i8
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if exponent == 0 {
        return sign;
    }
    if exponent == 0xff {
        return sign | 0x7c00 | u16::from(mantissa != 0);
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = 14 - half_exponent;
        let mut half_mantissa = mantissa >> shift;
        let remainder = mantissa & ((1_u32 << shift) - 1);
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && (half_mantissa & 1) != 0) {
            half_mantissa += 1;
        }
        return sign | half_mantissa as u16;
    }

    let mut half_mantissa = mantissa >> 13;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half_mantissa & 1) != 0) {
        half_mantissa += 1;
        if half_mantissa == 0x400 {
            let rounded_exponent = half_exponent + 1;
            if rounded_exponent >= 31 {
                return sign | 0x7c00;
            }
            return sign | ((rounded_exponent as u16) << 10);
        }
    }
    sign | ((half_exponent as u16) << 10) | half_mantissa as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let f32_bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut mantissa = mantissa;
            let mut exponent = -14_i32;
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: ActivationShape = ActivationShape::new(2, 1, 2);

    fn assert_invalid(result: io::Result<impl std::fmt::Debug>, expected: &str) {
        let error = result.expect_err("operation should fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), expected);
    }

    #[test]
    fn raw_f32_is_exact_and_shape_aware() {
        let values = [-0.0, 1.0, -2.5, f32::MIN_POSITIVE, 7.0, -9.0];
        let payload = encode_activation(StageActivationCodec::RawF32V1, SHAPE, &values).unwrap();
        assert_eq!(payload.len(), 24);
        assert_eq!(
            decode_activation(StageActivationCodec::RawF32V1, SHAPE, &payload).unwrap(),
            values
        );
    }

    #[test]
    fn f16_and_bf16_use_round_to_nearest_even_golden_bytes() {
        let shape = ActivationShape::new(1, 0, 4);
        let f16_values = [1.0, 1.000_488_3, 1.001_464_8, -0.0];
        assert_eq!(
            encode_activation(StageActivationCodec::F16RneV1, shape, &f16_values).unwrap(),
            vec![0x00, 0x3c, 0x00, 0x3c, 0x02, 0x3c, 0x00, 0x80]
        );

        let bf16_values = [1.0, 1.003_906_3, 1.011_718_8, -0.0];
        assert_eq!(
            encode_activation(StageActivationCodec::Bf16RneV1, shape, &bf16_values).unwrap(),
            vec![0x80, 0x3f, 0x80, 0x3f, 0x82, 0x3f, 0x00, 0x80]
        );
    }

    #[test]
    fn s8_rows_have_deterministic_golden_bytes_and_zero_rows_use_scale_one() {
        let shape = ActivationShape::new(2, 0, 4);
        let values = [0.0, 0.0, 0.0, 0.0, -1.0, -0.5, 0.5, 1.0];
        let payload =
            encode_activation(StageActivationCodec::S8RowF32RneV1, shape, &values).unwrap();
        assert_eq!(
            payload,
            vec![
                0x00, 0x00, 0x80, 0x3f, 0, 0, 0, 0, // zero row
                0x04, 0x02, 0x01, 0x3c, 129, 192, 64, 127,
            ]
        );
        let decoded =
            decode_activation(StageActivationCodec::S8RowF32RneV1, shape, &payload).unwrap();
        assert_eq!(&decoded[..4], &[0.0; 4]);
        assert_eq!(decoded[4], -1.0);
        assert_eq!(decoded[5], -64.0 / 127.0);
        assert_eq!(decoded[6], 64.0 / 127.0);
        assert_eq!(decoded[7], 1.0);
    }

    #[test]
    fn s8_uses_symmetric_range_zero_point_and_round_to_even() {
        let shape = ActivationShape::new(1, 0, 5);
        let values = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let payload =
            encode_activation(StageActivationCodec::S8RowF32RneV1, shape, &values).unwrap();
        assert_eq!(&payload[4..], &[129, 192, 0, 64, 127]);

        assert_eq!(round_to_even_i8(-2.5), -2);
        assert_eq!(round_to_even_i8(-1.5), -2);
        assert_eq!(round_to_even_i8(1.5), 2);
        assert_eq!(round_to_even_i8(2.5), 2);
    }

    #[test]
    fn s8_encoder_only_emits_frames_the_decoder_can_reconstruct() {
        let shape = ActivationShape::new(1, 0, 1);
        for value in [f32::MAX, -f32::MAX] {
            assert_invalid(
                encode_activation(StageActivationCodec::S8RowF32RneV1, shape, &[value]),
                "S8 activation value is out of range",
            );
        }
    }

    #[test]
    fn s8_decoder_rejects_reserved_negative_128() {
        let shape = ActivationShape::new(1, 0, 1);
        let mut payload = 1.0_f32.to_le_bytes().to_vec();
        payload.push(i8::MIN as u8);
        assert_invalid(
            decode_activation(StageActivationCodec::S8RowF32RneV1, shape, &payload),
            "S8 activation payload contains reserved -128 value",
        );
    }

    #[test]
    fn s8_tiny_rows_use_the_minimum_positive_subnormal_scale() {
        let shape = ActivationShape::new(1, 0, 2);
        let values = [f32::from_bits(1), f32::from_bits(63)];
        let payload =
            encode_activation(StageActivationCodec::S8RowF32RneV1, shape, &values).unwrap();
        assert_eq!(&payload[..4], &f32::from_bits(1).to_le_bytes());
        assert_eq!(&payload[4..], &[1, 63]);
        assert_eq!(
            decode_activation(StageActivationCodec::S8RowF32RneV1, shape, &payload).unwrap(),
            values
        );
    }

    #[test]
    fn s8_quantization_uses_f64_quotients_before_rounding() {
        let shape = ActivationShape::new(1, 0, 2);
        let values = [1.0, f32::from_bits(0x3e0d_1a34)];
        let payload =
            encode_activation(StageActivationCodec::S8RowF32RneV1, shape, &values).unwrap();
        assert_eq!(&payload[4..], &[127, 17]);
    }

    #[test]
    fn sideband_rows_are_included_in_shape_and_s8_chunks_are_invariant() {
        let values = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        let all = encode_activation(StageActivationCodec::S8RowF32RneV1, SHAPE, &values).unwrap();
        let mut chunks = Vec::new();
        for row in values.as_chunks::<2>().0 {
            chunks.extend(
                encode_activation(
                    StageActivationCodec::S8RowF32RneV1,
                    ActivationShape::new(1, 0, 2),
                    row,
                )
                .unwrap(),
            );
        }
        assert_eq!(all, chunks);
        assert_eq!(
            decode_activation(StageActivationCodec::S8RowF32RneV1, SHAPE, &all)
                .unwrap()
                .len(),
            6
        );
    }

    #[test]
    fn raw_f32_preserves_non_finite_bit_patterns() {
        let shape = ActivationShape::new(1, 0, 3);
        let values = [
            f32::from_bits(0x7fc0_1234),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let payload = encode_activation(StageActivationCodec::RawF32V1, shape, &values).unwrap();
        let decoded = decode_activation(StageActivationCodec::RawF32V1, shape, &payload).unwrap();

        assert_eq!(
            decoded
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            select_lossless_activation_codec(
                shape,
                &values,
                &[
                    StageActivationCodec::RawF32V1,
                    StageActivationCodec::F16RneV1,
                    StageActivationCodec::Bf16RneV1,
                ],
            )
            .unwrap(),
            StageActivationCodec::RawF32V1
        );
    }

    #[test]
    fn compressed_codecs_reject_non_finite_values_and_malformed_payloads() {
        let shape = ActivationShape::new(1, 0, 1);
        for codec in [
            StageActivationCodec::F16RneV1,
            StageActivationCodec::Bf16RneV1,
            StageActivationCodec::S8RowF32RneV1,
        ] {
            assert_invalid(
                encode_activation(codec, shape, &[f32::NAN]),
                "activation values must be finite",
            );
            assert_invalid(
                encode_activation(codec, shape, &[f32::INFINITY]),
                "activation values must be finite",
            );
            assert_invalid(
                decode_activation(codec, shape, &[]),
                "activation payload size mismatch",
            );
        }
        assert_invalid(
            decode_activation(StageActivationCodec::RawF32V1, shape, &[]),
            "activation payload size mismatch",
        );
        assert_invalid(
            decode_activation(StageActivationCodec::S8RowF32RneV1, shape, &[0, 0, 0, 0, 0]),
            "S8 activation row scale is not finite and positive",
        );
        assert_invalid(
            decode_activation(StageActivationCodec::F16RneV1, shape, &[0, 0x7c]),
            "activation payload contains non-finite value",
        );
        assert_invalid(
            decode_activation(StageActivationCodec::Bf16RneV1, shape, &[0x80, 0x7f]),
            "activation payload contains non-finite value",
        );
        let mut overflowing_s8 = f32::MAX.to_le_bytes().to_vec();
        overflowing_s8.push(127);
        assert_invalid(
            decode_activation(StageActivationCodec::S8RowF32RneV1, shape, &overflowing_s8),
            "activation payload contains non-finite value",
        );
        assert_invalid(
            encode_activation(StageActivationCodec::F16RneV1, shape, &[f32::MAX]),
            "F16 activation value is out of range",
        );
        assert_invalid(
            encode_activation(StageActivationCodec::Bf16RneV1, shape, &[f32::MAX]),
            "BF16 activation value is out of range",
        );
    }

    #[test]
    fn checked_limits_reject_overflow_and_payloads_over_protocol_maximum() {
        let overflow = ActivationShape::new(usize::MAX, 1, 1);
        assert_invalid(
            encode_activation(StageActivationCodec::RawF32V1, overflow, &[]),
            "activation row count overflow",
        );

        let too_large = ActivationShape::new(MAX_STAGE_DECODED_ACTIVATION_BYTES / 4 + 1, 0, 1);
        assert_invalid(
            decode_activation(StageActivationCodec::RawF32V1, too_large, &[]),
            "decoded activation payload byte count exceeds maximum",
        );

        let s8_wire_too_large = ActivationShape::new(MAX_STAGE_ACTIVATION_BYTES / 5 + 1, 0, 1);
        assert_invalid(
            decode_activation(StageActivationCodec::S8RowF32RneV1, s8_wire_too_large, &[]),
            "activation payload byte count exceeds maximum",
        );
    }

    #[test]
    fn lossless_selector_uses_bf16_only_for_exactly_representable_values() {
        let shape = ActivationShape::new(1, 1, 3);
        let exact = [1.0, -2.5, 0.0, -0.0, f32::MIN_POSITIVE, 8.0];
        let permitted = [
            StageActivationCodec::RawF32V1,
            StageActivationCodec::Bf16RneV1,
        ];

        let codec = select_lossless_activation_codec(shape, &exact, &permitted).unwrap();
        assert_eq!(codec, StageActivationCodec::Bf16RneV1);
        let encoded = encode_activation(codec, shape, &exact).unwrap();
        let decoded = decode_activation(codec, shape, &encoded).unwrap();
        assert_eq!(
            decoded
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            exact
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        let mut mixed = exact;
        mixed[4] = 1.000_001;
        assert_eq!(
            select_lossless_activation_codec(shape, &mixed, &permitted).unwrap(),
            StageActivationCodec::RawF32V1
        );
    }

    #[test]
    fn lossless_selector_round_trips_every_finite_bf16_value_exactly() {
        let values = (0_u16..=u16::MAX)
            .map(|bits| f32::from_bits(u32::from(bits) << 16))
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let shape = ActivationShape::new(values.len(), 0, 1);
        let permitted = [
            StageActivationCodec::RawF32V1,
            StageActivationCodec::Bf16RneV1,
        ];

        let codec = select_lossless_activation_codec(shape, &values, &permitted).unwrap();
        assert_eq!(codec, StageActivationCodec::Bf16RneV1);
        let encoded = encode_activation(codec, shape, &values).unwrap();
        let decoded = decode_activation(codec, shape, &encoded).unwrap();
        assert!(
            decoded
                .iter()
                .zip(&values)
                .all(|(decoded, original)| decoded.to_bits() == original.to_bits())
        );
    }

    #[test]
    fn lossless_selector_round_trips_every_finite_f16_value_exactly() {
        let values = (0_u16..=u16::MAX)
            .map(f16_bits_to_f32)
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let shape = ActivationShape::new(values.len(), 0, 1);
        let permitted = [
            StageActivationCodec::RawF32V1,
            StageActivationCodec::F16RneV1,
        ];

        let codec = select_lossless_activation_codec(shape, &values, &permitted).unwrap();
        assert_eq!(codec, StageActivationCodec::F16RneV1);
        let encoded = encode_activation(codec, shape, &values).unwrap();
        let decoded = decode_activation(codec, shape, &encoded).unwrap();
        assert_eq!(
            decoded
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn lossless_selector_uses_bf16_tie_break_and_f16_when_it_alone_is_exact() {
        let shape = ActivationShape::new(1, 0, 1);
        let permitted = [
            StageActivationCodec::RawF32V1,
            StageActivationCodec::F16RneV1,
            StageActivationCodec::Bf16RneV1,
        ];

        assert_eq!(
            select_lossless_activation_codec(shape, &[1.0], &permitted).unwrap(),
            StageActivationCodec::Bf16RneV1
        );
        assert_eq!(
            select_lossless_activation_codec(shape, &[1.0 + 2.0_f32.powi(-10)], &permitted)
                .unwrap(),
            StageActivationCodec::F16RneV1
        );
        assert_eq!(
            select_lossless_activation_codec(shape, &[65_536.0], &permitted).unwrap(),
            StageActivationCodec::Bf16RneV1
        );
    }

    #[test]
    fn lossless_selector_requires_one_codec_to_cover_the_whole_frame() {
        let shape = ActivationShape::new(1, 1, 1);
        let permitted = [
            StageActivationCodec::RawF32V1,
            StageActivationCodec::F16RneV1,
            StageActivationCodec::Bf16RneV1,
        ];
        let f16_only_primary = 1.0 + 2.0_f32.powi(-10);
        let bf16_only_sideband = 65_536.0;

        assert_eq!(
            select_lossless_activation_codec(
                shape,
                &[f16_only_primary, bf16_only_sideband],
                &permitted,
            )
            .unwrap(),
            StageActivationCodec::RawF32V1
        );
    }

    #[test]
    fn lossless_selector_requires_raw_and_never_selects_unpermitted_lossy_codecs() {
        let shape = ActivationShape::new(1, 0, 2);
        let values = [1.0, -2.0];

        assert_invalid(
            select_lossless_activation_codec(shape, &values, &[StageActivationCodec::Bf16RneV1]),
            "lossless activation selection requires RawF32 fallback",
        );
        assert_eq!(
            select_lossless_activation_codec(
                shape,
                &values,
                &[
                    StageActivationCodec::RawF32V1,
                    StageActivationCodec::S8RowF32RneV1,
                ],
            )
            .unwrap(),
            StageActivationCodec::RawF32V1
        );
    }

    #[test]
    fn lossless_selector_validates_shape_values_and_empty_payload_cost() {
        let permitted = [
            StageActivationCodec::RawF32V1,
            StageActivationCodec::Bf16RneV1,
        ];
        assert_invalid(
            select_lossless_activation_codec(ActivationShape::new(1, 0, 2), &[1.0], &permitted),
            "activation value count does not match shape",
        );
        assert_eq!(
            select_lossless_activation_codec(
                ActivationShape::new(1, 0, 1),
                &[f32::NAN],
                &permitted,
            )
            .unwrap(),
            StageActivationCodec::RawF32V1
        );
        assert_eq!(
            select_lossless_activation_codec(ActivationShape::new(0, 0, 4), &[], &permitted,)
                .unwrap(),
            StageActivationCodec::RawF32V1
        );
    }
}
