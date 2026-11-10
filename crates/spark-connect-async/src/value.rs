//! Values of Spark's types as Rust values: [`Decimal`], [`Value`] and [`Row`].

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Utc};
use serde::de::{self, Deserialize, Deserializer, SeqAccess, Visitor};

use crate::error::{Error, Result};

/// An exact decimal number of Spark's `DECIMAL(precision, scale)`: `unscaled × 10^-scale`.
///
/// It holds every value of Spark's decimals, whose precision is at most 38 digits. The scale may
/// be negative, as with `spark.sql.legacy.allowNegativeScaleOfDecimal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Decimal {
    unscaled: i128,
    precision: u8,
    scale: i8,
}

impl Decimal {
    /// The largest precision of a Spark decimal.
    pub const MAX_PRECISION: u8 = 38;

    /// The decimal `unscaled × 10^-scale` of type `DECIMAL(precision, scale)`.
    ///
    /// Fails with [`ErrorKind::InvalidArgument`](crate::ErrorKind::InvalidArgument) unless
    /// `1 <= precision <= 38` and `unscaled` has at most `precision` digits.
    pub fn new(unscaled: i128, precision: u8, scale: i8) -> Result<Decimal> {
        if !(1..=Self::MAX_PRECISION).contains(&precision) {
            return Err(Error::invalid_argument(format!(
                "the precision of a decimal must be between 1 and 38, not {precision}"
            )));
        }
        if digits(unscaled) > u32::from(precision) {
            return Err(Error::invalid_argument(format!(
                "{unscaled} has more than {precision} digits"
            )));
        }
        Ok(Decimal {
            unscaled,
            precision,
            scale,
        })
    }

    /// The unscaled value.
    pub fn unscaled(&self) -> i128 {
        self.unscaled
    }

    /// The number of digits of the type.
    pub fn precision(&self) -> u8 {
        self.precision
    }

    /// The number of digits after the decimal point; negative for a multiple of a power of ten.
    pub fn scale(&self) -> i8 {
        self.scale
    }

    /// The nearest `f64`.
    pub fn to_f64(&self) -> f64 {
        // Through the text, which `f64::from_str` rounds correctly, unlike a division.
        self.to_string().parse().unwrap_or(f64::NAN)
    }
}

fn digits(value: i128) -> u32 {
    value
        .unsigned_abs()
        .checked_ilog10()
        .map_or(1, |log| log + 1)
}

impl fmt::Display for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text =
            spark_connect_core::decode::i128_to_decimal_string(self.unscaled, self.scale.into());
        f.write_str(&text)
    }
}

/// Parses decimal text such as `-12.50` or `1.5E+3`. The precision is the number of digits, and
/// at least the scale.
impl FromStr for Decimal {
    type Err = Error;

    fn from_str(text: &str) -> Result<Decimal> {
        let invalid = || Error::invalid_argument(format!("{text:?} is not a decimal number"));
        let (mantissa, exponent) = match text.find(['e', 'E']) {
            Some(at) => (
                &text[..at],
                text[at + 1..].parse::<i32>().map_err(|_| invalid())?,
            ),
            None => (text, 0),
        };
        let (negative, mantissa) = match mantissa.as_bytes().first() {
            Some(b'-') => (true, &mantissa[1..]),
            Some(b'+') => (false, &mantissa[1..]),
            _ => (false, mantissa),
        };
        let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let all_digits = |part: &str| part.bytes().all(|b| b.is_ascii_digit());
        if whole.len() + fraction.len() == 0 || !all_digits(whole) || !all_digits(fraction) {
            return Err(invalid());
        }
        let significant = format!("{whole}{fraction}");
        let significant = significant.trim_start_matches('0');
        let unscaled: i128 = if significant.is_empty() {
            0
        } else {
            significant.parse().map_err(|_| invalid())?
        };
        let scale = i32::try_from(fraction.len()).map_err(|_| invalid())? - exponent;
        let scale = i8::try_from(scale).map_err(|_| invalid())?;
        let precision = digits(unscaled).max(u32::try_from(scale).unwrap_or(0));
        let precision = u8::try_from(precision).map_err(|_| invalid())?;
        Decimal::new(
            if negative { -unscaled } else { unscaled },
            precision,
            scale,
        )
    }
}

/// The name under which [`Decimal`] asks a deserializer of this crate for its exact parts.
pub(crate) const DECIMAL_PARTS: &str = "$spark_connect_async::Decimal";

/// Deserializes from decimal text, from an integer, or, from a result of a query, exactly with
/// the precision and scale of its column.
impl<'de> Deserialize<'de> for Decimal {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_newtype_struct(DECIMAL_PARTS, DecimalVisitor)
    }
}

struct DecimalVisitor;

impl<'de> Visitor<'de> for DecimalVisitor {
    type Value = Decimal;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a decimal number")
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Decimal, D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_str<E: de::Error>(self, text: &str) -> std::result::Result<Decimal, E> {
        text.parse().map_err(E::custom)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<Decimal, E> {
        self.visit_i128(value.into())
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<Decimal, E> {
        self.visit_i128(value.into())
    }

    fn visit_i128<E: de::Error>(self, value: i128) -> std::result::Result<Decimal, E> {
        let precision = u8::try_from(digits(value)).map_err(E::custom)?;
        Decimal::new(value, precision, 0).map_err(E::custom)
    }

    /// The exact parts: the unscaled value, the precision and the scale.
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Decimal, A::Error> {
        let missing = || de::Error::custom("the parts of a decimal are missing");
        let unscaled: i128 = seq.next_element()?.ok_or_else(missing)?;
        let precision: u8 = seq.next_element()?.ok_or_else(missing)?;
        let scale: i8 = seq.next_element()?.ok_or_else(missing)?;
        Decimal::new(unscaled, precision, scale).map_err(de::Error::custom)
    }
}

/// A value of any Spark type, as a row of a result holds it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Value {
    /// `NULL`, of any type.
    Null,
    /// `BOOLEAN`.
    Boolean(bool),
    /// `TINYINT`.
    Byte(i8),
    /// `SMALLINT`.
    Short(i16),
    /// `INT`.
    Int(i32),
    /// `BIGINT`.
    Long(i64),
    /// `FLOAT`.
    Float(f32),
    /// `DOUBLE`.
    Double(f64),
    /// `DECIMAL(precision, scale)`.
    Decimal(Decimal),
    /// `STRING`.
    String(String),
    /// `BINARY`.
    Binary(Vec<u8>),
    /// `DATE`.
    Date(NaiveDate),
    /// `TIMESTAMP`: an instant, which Spark shows in the session's time zone.
    Timestamp(DateTime<Utc>),
    /// `TIMESTAMP_NTZ`: a date and a time of day, in no time zone.
    TimestampNtz(NaiveDateTime),
    /// `TIME`.
    Time(NaiveTime),
    /// `INTERVAL DAY TO SECOND`, and its narrower forms.
    DayTimeInterval(TimeDelta),
    /// `INTERVAL YEAR TO MONTH`, and its narrower forms, in months.
    YearMonthInterval(i32),
    /// `VARIANT`, as its encoded value and metadata.
    Variant {
        /// The encoded value.
        value: Vec<u8>,
        /// The encoded metadata.
        metadata: Vec<u8>,
    },
    /// `ARRAY`.
    Array(Vec<Value>),
    /// `MAP`, with its entries in the server's order.
    Map(Vec<(Value, Value)>),
    /// `STRUCT`.
    Struct(Row),
}

/// A row of a result, or a value of a `STRUCT`: values by position, with the names of their
/// fields.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    names: Arc<[String]>,
    values: Vec<Value>,
}

impl Row {
    pub(crate) fn new(names: Arc<[String]>, values: Vec<Value>) -> Row {
        Row { names, values }
    }

    /// The number of values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the row has no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The value at `index`.
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }

    /// The value of the first field named `name`.
    pub fn get_by_name(&self, name: &str) -> Option<&Value> {
        let index = self.names.iter().position(|field| field == name)?;
        self.values.get(index)
    }

    /// The names of the fields.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The values, by position.
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Consumes the row, returning its values.
    pub fn into_values(self) -> Vec<Value> {
        self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    #[test]
    fn decimal_text_parses_into_its_parts_and_displays_exactly() {
        let max = "9".repeat(38);
        let smallest = format!("-0.{}1", "0".repeat(37));
        let cases: [(&str, i128, u8, i8, &str); 9] = [
            ("0", 0, 1, 0, "0"),
            ("-1.50", -150, 3, 2, "-1.50"),
            ("+0.05", 5, 2, 2, "0.05"),
            ("007", 7, 1, 0, "7"),
            ("1.5E+3", 15, 2, -2, "15E+2"),
            ("2.5e-2", 25, 3, 3, "0.025"),
            (&max, max.parse().unwrap(), 38, 0, &max),
            (&smallest, -1, 38, 38, &smallest),
            ("-0", 0, 1, 0, "0"),
        ];
        for (text, unscaled, precision, scale, shown) in cases {
            let decimal: Decimal = text.parse().unwrap();
            assert_eq!(
                (decimal.unscaled(), decimal.precision(), decimal.scale()),
                (unscaled, precision, scale),
                "{text}"
            );
            assert_eq!(decimal.to_string(), shown, "{text}");
        }
    }

    #[test]
    fn a_decimal_outside_spark_bounds_or_not_a_number_is_rejected() {
        let too_long = "1".repeat(39);
        for text in ["", "-", "1.2.3", "1e", "abc", "1,5", &too_long] {
            let error = text.parse::<Decimal>().unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{text:?}");
        }
        assert!(Decimal::new(10i128.pow(38) - 1, 38, 0).is_ok());
        for (unscaled, precision) in [(10i128.pow(38), 38), (100, 2), (1, 0), (1, 39)] {
            assert!(
                Decimal::new(unscaled, precision, 0).is_err(),
                "{unscaled} of precision {precision}"
            );
        }
    }

    #[test]
    fn a_decimal_converts_to_the_nearest_f64() {
        for (text, expected) in [("0.1", 0.1), ("-12.50", -12.5), ("15E+2", 1500.0)] {
            assert_eq!(text.parse::<Decimal>().unwrap().to_f64(), expected);
        }
    }
}
