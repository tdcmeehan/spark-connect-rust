//! Columns, and literals of Rust values.

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Timelike, Utc};
use spark_connect_core::expression::{Expression, LiteralExpression};
use spark_connect_core::types::{DataType, DayTimeIntervalType};

use crate::value::Decimal;

#[doc(no_inline)]
pub use spark_connect_core::column::Column;

/// A column of the literal `value`, of the Spark type of its Rust type: `i32` is an `INT`, `i64`
/// a `BIGINT`, `&str` a `STRING`, [`NaiveDate`] a `DATE`, `None::<T>` a `NULL` of `T`'s type.
///
/// ```
/// use spark_connect_async::column::lit;
///
/// let answer = lit(42);
/// let label = lit("answer");
/// let missing = lit(None::<i64>);
/// ```
pub fn lit(value: impl IntoLiteral) -> Column {
    Column::new(Expression::literal(value.into_literal()))
}

/// A Rust value that [`lit`] turns into a literal.
pub trait IntoLiteral {
    #[doc(hidden)]
    fn into_literal(self) -> LiteralExpression;

    /// The Spark type of a `NULL` of this Rust type.
    #[doc(hidden)]
    fn null_type() -> DataType;
}

macro_rules! into_literal {
    ($($rust:ty => |$value:ident| $literal:expr, $null_type:expr;)*) => {
        $(impl IntoLiteral for $rust {
            fn into_literal(self) -> LiteralExpression {
                let $value = self;
                $literal
            }

            fn null_type() -> DataType {
                $null_type
            }
        })*
    };
}

fn string_type() -> DataType {
    DataType::String {
        collation: "UTF8_BINARY".to_string(),
    }
}

const TIME_PRECISION: i32 = 6;

fn microseconds(delta: TimeDelta) -> i64 {
    delta
        .num_microseconds()
        .unwrap_or(if delta < TimeDelta::zero() {
            i64::MIN
        } else {
            i64::MAX
        })
}

into_literal! {
    bool => |v| LiteralExpression::Boolean(v), DataType::Boolean;
    i8 => |v| LiteralExpression::Byte(v.into()), DataType::Byte;
    i16 => |v| LiteralExpression::Short(v.into()), DataType::Short;
    i32 => |v| LiteralExpression::Integer(v), DataType::Integer;
    i64 => |v| LiteralExpression::Long(v), DataType::Long;
    f32 => |v| LiteralExpression::Float(v), DataType::Float;
    f64 => |v| LiteralExpression::Double(v), DataType::Double;
    &str => |v| LiteralExpression::String(v.to_string()), string_type();
    String => |v| LiteralExpression::String(v), string_type();
    &[u8] => |v| LiteralExpression::Binary(v.to_vec()), DataType::Binary;
    Vec<u8> => |v| LiteralExpression::Binary(v), DataType::Binary;
    Decimal => |v| LiteralExpression::Decimal {
        value: v.to_string(),
        precision: v.precision().into(),
        scale: v.scale().into(),
    }, DataType::Decimal { precision: 10, scale: 0 };
    NaiveDate => |v| LiteralExpression::Date(v.to_epoch_days()), DataType::Date;
    DateTime<Utc> => |v| LiteralExpression::Timestamp(v.timestamp_micros()), DataType::Timestamp;
    NaiveDateTime => |v| LiteralExpression::TimestampNtz(v.and_utc().timestamp_micros()),
        DataType::TimestampNtz;
    NaiveTime => |v| LiteralExpression::Time {
        nano: i64::from(v.num_seconds_from_midnight()) * 1_000_000_000 + i64::from(v.nanosecond()),
        precision: TIME_PRECISION,
    }, DataType::Time { precision: TIME_PRECISION };
    TimeDelta => |v| LiteralExpression::DayTimeInterval(microseconds(v)), DataType::DayTimeInterval {
        start_field: DayTimeIntervalType::DAY,
        end_field: DayTimeIntervalType::SECOND,
    };
}

impl<T: IntoLiteral> IntoLiteral for Option<T> {
    fn into_literal(self) -> LiteralExpression {
        match self {
            Some(value) => value.into_literal(),
            None => LiteralExpression::Null(T::null_type()),
        }
    }

    fn null_type() -> DataType {
        T::null_type()
    }
}

/// An `ARRAY` of the type of its elements, or of `T`'s type when it has none. An array of
/// decimals has the narrowest decimal type that holds each of them.
impl<T: IntoLiteral> IntoLiteral for Vec<T> {
    fn into_literal(self) -> LiteralExpression {
        let elements: Vec<LiteralExpression> =
            self.into_iter().map(IntoLiteral::into_literal).collect();
        let element_type = elements
            .iter()
            .map(type_of)
            .reduce(wider)
            .unwrap_or_else(T::null_type);
        LiteralExpression::Array {
            element_type: Box::new(element_type),
            elements,
        }
    }

    fn null_type() -> DataType {
        DataType::Array {
            element_type: Box::new(T::null_type()),
            contains_null: true,
        }
    }
}

/// The type of both decimal types `a` and `b`, or `a` for other types.
fn wider(a: DataType, b: DataType) -> DataType {
    match (&a, &b) {
        (
            DataType::Decimal {
                precision: pa,
                scale: sa,
            },
            DataType::Decimal {
                precision: pb,
                scale: sb,
            },
        ) => {
            let scale = (*sa).max(*sb);
            let digits = (pa - sa).max(pb - sb);
            DataType::Decimal {
                precision: (digits + scale).min(Decimal::MAX_PRECISION.into()),
                scale,
            }
        }
        _ => a,
    }
}

fn type_of(literal: &LiteralExpression) -> DataType {
    match literal {
        LiteralExpression::Null(data_type) => data_type.clone(),
        LiteralExpression::Boolean(_) => DataType::Boolean,
        LiteralExpression::Byte(_) => DataType::Byte,
        LiteralExpression::Short(_) => DataType::Short,
        LiteralExpression::Integer(_) => DataType::Integer,
        LiteralExpression::Long(_) => DataType::Long,
        LiteralExpression::Float(_) => DataType::Float,
        LiteralExpression::Double(_) => DataType::Double,
        LiteralExpression::Decimal {
            precision, scale, ..
        } => DataType::Decimal {
            precision: *precision,
            scale: *scale,
        },
        LiteralExpression::String(_) => string_type(),
        LiteralExpression::Binary(_) => DataType::Binary,
        LiteralExpression::Date(_) => DataType::Date,
        LiteralExpression::Timestamp(_) => DataType::Timestamp,
        LiteralExpression::TimestampNtz(_) => DataType::TimestampNtz,
        LiteralExpression::DayTimeInterval(_) => TimeDelta::null_type(),
        LiteralExpression::Time { precision, .. } => DataType::Time {
            precision: *precision,
        },
        LiteralExpression::Array { element_type, .. } => DataType::Array {
            element_type: element_type.clone(),
            contains_null: true,
        },
    }
}
