//! How readings are written in the info panel.

/// A reading with a stable layout: 4 decimals from 0.001 to 1000, else scientific.
///
/// With a fixed number of decimals the decimal point stays in place as the value
/// changes (the info panel right-aligns numbers in fixed-width columns).
pub fn number(value: f64, signed: bool) -> String {
    let magnitude = value.abs();
    if magnitude == 0.0 || (1e-3..1e3).contains(&magnitude) {
        return if signed { format!("{value:+.4}") } else { format!("{value:.4}") };
    }
    // Python's "{:.2e}": mantissa with 2 decimals, exponent signed and at least 2 digits.
    let text = format!("{value:.2e}");
    let (mantissa, exponent) = text.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let sign = if signed && value >= 0.0 { "+" } else { "" };
    let exp_sign = if exponent < 0 { '-' } else { '+' };
    format!("{sign}{mantissa}e{exp_sign}{:02}", exponent.abs())
}

/// Whether a field reads as a number (the info panel right-aligns those).
pub fn is_number(text: &str) -> bool {
    let body = text.strip_prefix(['+', '-']).unwrap_or(text);
    let (mantissa, exponent) = match body.split_once('e') {
        Some((m, e)) => (m, Some(e)),
        None => (body, None),
    };
    let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    let mantissa_ok = match mantissa.split_once('.') {
        Some((int, frac)) => (digits(int) && (frac.is_empty() || digits(frac))) || (int.is_empty() && digits(frac)),
        None => digits(mantissa),
    };
    let exponent_ok = exponent.is_none_or(|e| digits(e.strip_prefix(['+', '-']).unwrap_or(e)));
    mantissa_ok && exponent_ok
}

/// The widest number `number` produces; sizes the panel's number columns.
pub const WIDEST_NUMBER: &str = "-0.0000e-00";

pub fn plural(count: usize, noun: &str) -> String {
    format!("{count} {noun}{}", if count == 1 { "" } else { "s" })
}

pub fn describe_frequency(frequency: f64) -> String {
    if frequency == 0.0 { "DC".into() } else { format!("AC {frequency} cycles/time") }
}

/// Direction of B_z in words. Screen y points down, so +z points into the screen.
pub fn describe_b(bz: f64) -> &'static str {
    if bz == 0.0 {
        ""
    } else if bz > 0.0 {
        "into the screen"
    } else {
        "out of the screen"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_fit_their_fixed_column() {
        let values = [0.0, 1e-9, -3.2e-5, 0.000999, 0.001, -0.0417, 0.5, 9.99999, -123.456, 999.9, 4.2e6];
        for value in values {
            for signed in [true, false] {
                let text = number(value, signed);
                // Recognised as a number: right-aligned in a column.
                assert!(is_number(&text), "{text}");
                assert!(text.len() <= WIDEST_NUMBER.len(), "{text}");
            }
        }
    }

    #[test]
    fn numbers_match_python_formatting() {
        assert_eq!(number(0.5, true), "+0.5000");
        assert_eq!(number(-3.2e-5, true), "-3.20e-05");
        assert_eq!(number(4.2e6, false), "4.20e+06");
        assert_eq!(number(1e-9, true), "+1.00e-09");
    }
}
