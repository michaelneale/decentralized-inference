use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IecSizeParseError {
    message: String,
}

impl fmt::Display for IecSizeParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for IecSizeParseError {}

/// Parse a positive integer size with an explicit IEC suffix.
///
/// Bare numbers, decimal values, SI suffixes, zero, and overflow are rejected
/// so a public cache setting can never silently change units or mean
/// "unbounded".
pub fn parse_iec_size(input: &str) -> Result<u64, IecSizeParseError> {
    let value = input.trim();
    let (digits, multiplier) = [
        ("KiB", 1024_u64),
        ("MiB", 1024_u64.pow(2)),
        ("GiB", 1024_u64.pow(3)),
        ("TiB", 1024_u64.pow(4)),
    ]
    .into_iter()
    .find_map(|(suffix, multiplier)| {
        value
            .strip_suffix(suffix)
            .map(|digits| (digits, multiplier))
    })
    .ok_or_else(|| error("size must use an explicit IEC suffix: KiB, MiB, GiB, or TiB"))?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(error(
            "size must be a positive whole number with an IEC suffix",
        ));
    }
    let units = digits
        .parse::<u64>()
        .map_err(|_| error("size is too large"))?;
    if units == 0 {
        return Err(error("size must be greater than zero"));
    }
    units
        .checked_mul(multiplier)
        .ok_or_else(|| error("size is too large"))
}

fn error(message: &str) -> IecSizeParseError {
    IecSizeParseError {
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_explicit_iec_sizes() {
        assert_eq!(parse_iec_size("1KiB").unwrap(), 1024);
        assert_eq!(parse_iec_size("32MiB").unwrap(), 32 * 1024 * 1024);
        assert_eq!(parse_iec_size("32GiB").unwrap(), 32 * 1024 * 1024 * 1024);
        assert_eq!(parse_iec_size("1TiB").unwrap(), 1024_u64.pow(4));
    }

    #[test]
    fn rejects_ambiguous_or_unbounded_sizes() {
        for value in ["", "0", "0GiB", "32", "32GB", "1.5GiB", "-1GiB"] {
            assert!(parse_iec_size(value).is_err(), "accepted {value:?}");
        }
    }
}
