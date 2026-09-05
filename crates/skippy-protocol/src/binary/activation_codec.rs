use std::io;

use super::{
    invalid_data,
    types::{MAX_STAGE_ACTIVATION_BYTES, MAX_STAGE_DECODED_ACTIVATION_BYTES},
};

/// Pure activation payload codecs. This module intentionally does not define a
/// frame header: framing, codec negotiation, and server wiring own that seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActivationCodec {
    RawF32,
    F16Rne,
    Bf16Rne,
    S8RowF32RneV1,
}

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
    codec: ActivationCodec,
    shape: ActivationShape,
    values: &[f32],
) -> io::Result<Vec<u8>> {
    let elements = shape.elements()?;
    shape.validate_decoded_limit()?;
    if values.len() != elements {
        return Err(invalid_data("activation value count does not match shape"));
    }
    validate_finite(values)?;

    match codec {
        ActivationCodec::RawF32 => encode_raw_f32(values),
        ActivationCodec::F16Rne => encode_f16(values),
        ActivationCodec::Bf16Rne => encode_bf16(values),
        ActivationCodec::S8RowF32RneV1 => encode_s8_rows(shape, values),
    }
}

pub(crate) fn decode_activation(
    codec: ActivationCodec,
    shape: ActivationShape,
    payload: &[u8],
) -> io::Result<Vec<f32>> {
    let elements = shape.elements()?;
    shape.validate_decoded_limit()?;
    match codec {
        ActivationCodec::RawF32 => decode_raw_f32(payload, elements),
        ActivationCodec::F16Rne => decode_f16(payload, elements),
        ActivationCodec::Bf16Rne => decode_bf16(payload, elements),
        ActivationCodec::S8RowF32RneV1 => decode_s8_rows(shape, payload),
    }
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
    for bytes in payload.chunks_exact(4) {
        out.push(f32::from_le_bytes(
            bytes.try_into().expect("exact f32 chunk"),
        ));
    }
    validate_finite(&out)?;
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
    for bytes in payload.chunks_exact(2) {
        let value = f16_bits_to_f32(u16::from_le_bytes(
            bytes.try_into().expect("exact f16 chunk"),
        ));
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
    for bytes in payload.chunks_exact(2) {
        let value = f32::from_bits(
            u32::from(u16::from_le_bytes(
                bytes.try_into().expect("exact bf16 chunk"),
            )) << 16,
        );
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
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid_data(
                "S8 activation row scale is not finite and positive",
            ));
        }
        out.extend_from_slice(&scale.to_le_bytes());
        for value in row {
            out.push(round_to_even_i8(*value / scale) as u8);
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
            let decoded = (*value as i8) as f32 * scale;
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

fn round_to_even_i8(value: f32) -> i8 {
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else if (floor as i32) & 1 == 0 {
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
        let payload = encode_activation(ActivationCodec::RawF32, SHAPE, &values).unwrap();
        assert_eq!(payload.len(), 24);
        assert_eq!(
            decode_activation(ActivationCodec::RawF32, SHAPE, &payload).unwrap(),
            values
        );
    }

    #[test]
    fn f16_and_bf16_use_round_to_nearest_even_golden_bytes() {
        let shape = ActivationShape::new(1, 0, 4);
        let f16_values = [1.0, 1.000_488_3, 1.001_464_8, -0.0];
        assert_eq!(
            encode_activation(ActivationCodec::F16Rne, shape, &f16_values).unwrap(),
            vec![0x00, 0x3c, 0x00, 0x3c, 0x02, 0x3c, 0x00, 0x80]
        );

        let bf16_values = [1.0, 1.003_906_3, 1.011_718_8, -0.0];
        assert_eq!(
            encode_activation(ActivationCodec::Bf16Rne, shape, &bf16_values).unwrap(),
            vec![0x80, 0x3f, 0x80, 0x3f, 0x82, 0x3f, 0x00, 0x80]
        );
    }

    #[test]
    fn s8_rows_have_deterministic_golden_bytes_and_zero_rows_use_scale_one() {
        let shape = ActivationShape::new(2, 0, 4);
        let values = [0.0, 0.0, 0.0, 0.0, -1.0, -0.5, 0.5, 1.0];
        let payload = encode_activation(ActivationCodec::S8RowF32RneV1, shape, &values).unwrap();
        assert_eq!(
            payload,
            vec![
                0x00, 0x00, 0x80, 0x3f, 0, 0, 0, 0, // zero row
                0x04, 0x02, 0x01, 0x3c, 129, 192, 64, 127,
            ]
        );
        let decoded = decode_activation(ActivationCodec::S8RowF32RneV1, shape, &payload).unwrap();
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
        let payload = encode_activation(ActivationCodec::S8RowF32RneV1, shape, &values).unwrap();
        assert_eq!(&payload[4..], &[129, 192, 0, 64, 127]);

        assert_eq!(round_to_even_i8(-2.5), -2);
        assert_eq!(round_to_even_i8(-1.5), -2);
        assert_eq!(round_to_even_i8(1.5), 2);
        assert_eq!(round_to_even_i8(2.5), 2);
    }

    #[test]
    fn sideband_rows_are_included_in_shape_and_s8_chunks_are_invariant() {
        let values = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        let all = encode_activation(ActivationCodec::S8RowF32RneV1, SHAPE, &values).unwrap();
        let mut chunks = Vec::new();
        for row in values.chunks_exact(2) {
            chunks.extend(
                encode_activation(
                    ActivationCodec::S8RowF32RneV1,
                    ActivationShape::new(1, 0, 2),
                    row,
                )
                .unwrap(),
            );
        }
        assert_eq!(all, chunks);
        assert_eq!(
            decode_activation(ActivationCodec::S8RowF32RneV1, SHAPE, &all)
                .unwrap()
                .len(),
            6
        );
    }

    #[test]
    fn codecs_reject_non_finite_values_and_malformed_payloads() {
        let shape = ActivationShape::new(1, 0, 1);
        for codec in [
            ActivationCodec::RawF32,
            ActivationCodec::F16Rne,
            ActivationCodec::Bf16Rne,
            ActivationCodec::S8RowF32RneV1,
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

        let nan_payload = f32::NAN.to_le_bytes();
        assert_invalid(
            decode_activation(ActivationCodec::RawF32, shape, &nan_payload),
            "activation values must be finite",
        );
        assert_invalid(
            decode_activation(ActivationCodec::S8RowF32RneV1, shape, &[0, 0, 0, 0, 0]),
            "S8 activation row scale is not finite and positive",
        );
        assert_invalid(
            decode_activation(ActivationCodec::F16Rne, shape, &[0, 0x7c]),
            "activation payload contains non-finite value",
        );
        assert_invalid(
            decode_activation(ActivationCodec::Bf16Rne, shape, &[0x80, 0x7f]),
            "activation payload contains non-finite value",
        );
        let mut overflowing_s8 = f32::MAX.to_le_bytes().to_vec();
        overflowing_s8.push(127);
        assert_invalid(
            decode_activation(ActivationCodec::S8RowF32RneV1, shape, &overflowing_s8),
            "activation payload contains non-finite value",
        );
        assert_invalid(
            encode_activation(ActivationCodec::F16Rne, shape, &[f32::MAX]),
            "F16 activation value is out of range",
        );
        assert_invalid(
            encode_activation(ActivationCodec::Bf16Rne, shape, &[f32::MAX]),
            "BF16 activation value is out of range",
        );
    }

    #[test]
    fn checked_limits_reject_overflow_and_payloads_over_protocol_maximum() {
        let overflow = ActivationShape::new(usize::MAX, 1, 1);
        assert_invalid(
            encode_activation(ActivationCodec::RawF32, overflow, &[]),
            "activation row count overflow",
        );

        let too_large = ActivationShape::new(MAX_STAGE_DECODED_ACTIVATION_BYTES / 4 + 1, 0, 1);
        assert_invalid(
            decode_activation(ActivationCodec::RawF32, too_large, &[]),
            "decoded activation payload byte count exceeds maximum",
        );

        let s8_wire_too_large = ActivationShape::new(MAX_STAGE_ACTIVATION_BYTES / 5 + 1, 0, 1);
        assert_invalid(
            decode_activation(ActivationCodec::S8RowF32RneV1, s8_wire_too_large, &[]),
            "activation payload byte count exceeds maximum",
        );
    }
}
