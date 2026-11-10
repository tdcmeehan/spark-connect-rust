//! Values of Spark's types as Rust values: [`Decimal`], [`Value`] and [`Row`].

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Timelike, Utc};
use serde::de::{self, Deserialize, Deserializer, SeqAccess, Visitor};
use spark_connect_core::row::{StructValue, Value as CoreValue};

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

/// A row of the named values, such as a value of a `STRUCT` to send in local rows.
impl<N: Into<String>> FromIterator<(N, Value)> for Row {
    fn from_iter<I: IntoIterator<Item = (N, Value)>>(fields: I) -> Row {
        let (names, values): (Vec<String>, Vec<Value>) = fields
            .into_iter()
            .map(|(name, value)| (name.into(), value))
            .unzip();
        Row::new(names.into(), values)
    }
}

/// The values of the row, by position.
impl IntoIterator for Row {
    type Item = Value;
    type IntoIter = std::vec::IntoIter<Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl Value {
    /// The value as the encoder of local rows that the crates share takes it.
    pub(crate) fn into_core(self) -> Result<CoreValue> {
        let unsupported = |kind: &str| {
            Error::invalid_argument(format!("{kind} values cannot be sent in local rows"))
        };
        Ok(match self {
            Value::Null => CoreValue::Null,
            Value::Boolean(value) => CoreValue::Bool(value),
            Value::Byte(value) => CoreValue::Byte(value),
            Value::Short(value) => CoreValue::Short(value),
            Value::Int(value) => CoreValue::Integer(value),
            Value::Long(value) => CoreValue::Long(value),
            Value::Float(value) => CoreValue::Float(value),
            Value::Double(value) => CoreValue::Double(value),
            Value::Decimal(decimal) => CoreValue::Decimal {
                value: decimal.to_string(),
                precision: Some(decimal.precision().into()),
                scale: Some(decimal.scale().into()),
            },
            Value::String(value) => CoreValue::String(value),
            Value::Binary(value) => CoreValue::Binary(value),
            Value::Date(date) => CoreValue::Date(date.to_epoch_days()),
            Value::Timestamp(instant) => CoreValue::Timestamp(instant.timestamp_micros()),
            Value::TimestampNtz(time) => CoreValue::TimestampNtz(time.and_utc().timestamp_micros()),
            Value::Time(time) => CoreValue::Time(nanos_of_day(time)),
            Value::DayTimeInterval(interval) => CoreValue::DayTimeInterval(
                interval
                    .num_microseconds()
                    .ok_or_else(|| unsupported("Out-of-range INTERVAL DAY TO SECOND"))?,
            ),
            Value::YearMonthInterval(_) => return Err(unsupported("INTERVAL YEAR TO MONTH")),
            Value::Variant { value, metadata } => CoreValue::Variant { value, metadata },
            Value::Array(values) => CoreValue::List(into_core_values(values)?),
            Value::Map(entries) => CoreValue::Map(
                entries
                    .into_iter()
                    .map(|(key, value)| Ok((key.into_core()?, value.into_core()?)))
                    .collect::<Result<_>>()?,
            ),
            Value::Struct(row) => {
                CoreValue::Struct(StructValue::new(row.names, into_core_values(row.values)?))
            }
        })
    }
}

/// The nanoseconds of `time` since midnight, as Spark holds a `TIME`.
pub(crate) fn nanos_of_day(time: NaiveTime) -> i64 {
    i64::from(time.num_seconds_from_midnight()) * 1_000_000_000 + i64::from(time.nanosecond())
}

fn into_core_values(values: Vec<Value>) -> Result<Vec<CoreValue>> {
    values.into_iter().map(Value::into_core).collect()
}

/// The row of the metrics that the server observed for an observation.
pub(crate) fn row_of_metrics<'a>(
    keys: impl IntoIterator<Item = &'a String>,
    values: impl IntoIterator<Item = &'a spark_connect_proto::expression::Literal>,
) -> Row {
    let values = values.into_iter().map(value_of_literal).collect();
    Row::new(keys.into_iter().cloned().collect(), values)
}

/// The value of a literal of the server. A literal of a type that results do not have here,
/// such as a calendar interval, is `Value::Null`.
fn value_of_literal(literal: &spark_connect_proto::expression::Literal) -> Value {
    use spark_connect_proto::data_type::Kind;
    use spark_connect_proto::expression::literal::LiteralType as L;
    let data_type = literal
        .data_type
        .as_ref()
        .and_then(|data_type| data_type.kind.as_ref());
    let Some(literal) = &literal.literal_type else {
        return Value::Null;
    };
    match literal {
        L::Null(_) => Value::Null,
        L::Binary(bytes) => Value::Binary(bytes.to_vec()),
        L::Boolean(value) => Value::Boolean(*value),
        L::Byte(value) => i8::try_from(*value).map_or(Value::Int(*value), Value::Byte),
        L::Short(value) => i16::try_from(*value).map_or(Value::Int(*value), Value::Short),
        L::Integer(value) => Value::Int(*value),
        L::Long(value) => Value::Long(*value),
        L::Float(value) => Value::Float(*value),
        L::Double(value) => Value::Double(*value),
        L::Decimal(decimal) => decimal
            .value
            .parse()
            .map_or_else(|_| Value::String(decimal.value.clone()), Value::Decimal),
        L::String(value) => Value::String(value.clone()),
        L::Date(days) => NaiveDate::from_epoch_days(*days).map_or(Value::Null, Value::Date),
        L::Timestamp(micros) => {
            DateTime::from_timestamp_micros(*micros).map_or(Value::Null, Value::Timestamp)
        }
        L::TimestampNtz(micros) => DateTime::from_timestamp_micros(*micros)
            .map_or(Value::Null, |value| Value::TimestampNtz(value.naive_utc())),
        L::Time(time) => {
            let seconds = u32::try_from(time.nano / 1_000_000_000).unwrap_or(u32::MAX);
            let nanos = u32::try_from(time.nano % 1_000_000_000).unwrap_or(0);
            NaiveTime::from_num_seconds_from_midnight_opt(seconds, nanos)
                .map_or(Value::Null, Value::Time)
        }
        L::DayTimeInterval(micros) => Value::DayTimeInterval(TimeDelta::microseconds(*micros)),
        L::YearMonthInterval(months) => Value::YearMonthInterval(*months),
        L::Array(array) => Value::Array(array.elements.iter().map(value_of_literal).collect()),
        L::Map(map) => Value::Map(
            map.keys
                .iter()
                .zip(&map.values)
                .map(|(key, value)| (value_of_literal(key), value_of_literal(value)))
                .collect(),
        ),
        L::Struct(fields) => {
            let names: Arc<[String]> = match data_type {
                Some(Kind::Struct(struct_type)) => struct_type
                    .fields
                    .iter()
                    .map(|field| field.name.clone())
                    .collect(),
                _ => (1..=fields.elements.len())
                    .map(|i| format!("col{i}"))
                    .collect(),
            };
            let values = fields.elements.iter().map(value_of_literal).collect();
            Value::Struct(Row::new(names, values))
        }
        _ => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    #[test]
    fn observed_metrics_are_a_row_of_their_values() {
        use spark_connect_proto::expression::literal::LiteralType as L;
        let literal = |literal| spark_connect_proto::expression::Literal {
            literal_type: Some(literal),
            ..Default::default()
        };
        let observed = spark_connect_proto::execute_plan_response::ObservedMetrics {
            keys: ["rows", "total", "first", "day"].map(String::from).to_vec(),
            values: vec![
                literal(L::Long(3)),
                literal(L::Decimal(
                    spark_connect_proto::expression::literal::Decimal {
                        value: "1.50".to_string(),
                        precision: Some(10),
                        scale: Some(2),
                    },
                )),
                literal(L::String("a".to_string())),
                literal(L::Date(1)),
            ],
            ..Default::default()
        };
        let row = row_of_metrics(&observed.keys, &observed.values);
        assert_eq!(row.names(), ["rows", "total", "first", "day"]);
        assert_eq!(row.get(0), Some(&Value::Long(3)));
        assert_eq!(row.get(1), Some(&Value::Decimal("1.50".parse().unwrap())));
        assert_eq!(
            row.get_by_name("first"),
            Some(&Value::String("a".to_string()))
        );
        assert_eq!(
            row.get(3),
            Some(&Value::Date(NaiveDate::from_ymd_opt(1970, 1, 2).unwrap()))
        );
    }

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
