//! Columns, and literals of Rust values.

use std::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Rem, Sub};

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Timelike, Utc};
use spark_connect_core::column::Column as CoreColumn;
use spark_connect_core::expression::{
    Alias, Expression, ExpressionNode, LiteralExpression, SubqueryExpression,
};
use spark_connect_core::types::{DataType, DayTimeIntervalType};

use crate::dataframe::DataFrame;
use crate::value::Decimal;
use crate::window::Window;

/// An expression over the columns of a DataFrame: a column, a literal, or an operation on
/// other columns. Its methods build new columns; nothing reaches the server until an action
/// runs a plan that uses them.
///
/// Wherever a column is taken (`impl Into<Column>`), a `&str` or `String` is the column of that
/// name, and a Rust value is a literal through [`lit`]:
///
/// ```
/// use spark_connect_async::{col, lit};
///
/// let adult = col("age").ge(lit(18));
/// let total = col("price") * col("quantity");
/// let renamed = col("name").alias("label");
/// let labeled = col("name").starts_with(lit("A")).and(adult);
/// ```
///
/// Code that also uses the `spark_connect` crate converts its columns with `.into()`, both
/// ways.
#[derive(Debug, Clone)]
pub struct Column(CoreColumn);

impl From<CoreColumn> for Column {
    fn from(column: CoreColumn) -> Column {
        Column(column)
    }
}

impl From<Column> for CoreColumn {
    fn from(column: Column) -> CoreColumn {
        column.0
    }
}

impl From<&str> for Column {
    fn from(name: &str) -> Column {
        col(name)
    }
}

impl From<String> for Column {
    fn from(name: String) -> Column {
        Column::new(Expression::column(name))
    }
}

impl From<&String> for Column {
    fn from(name: &String) -> Column {
        col(name)
    }
}

pub(crate) fn to_core(column: impl Into<Column>) -> CoreColumn {
    column.into().0
}

impl Column {
    pub(crate) fn new(expression: Expression) -> Column {
        Column(CoreColumn::new(expression))
    }

    pub(crate) fn expression(&self) -> &Expression {
        self.0.expression()
    }

    /// The column under another name: one name, several names for the columns of a generator,
    /// or one name with metadata (see [`AliasSpec`]).
    ///
    /// ```
    /// use spark_connect_async::col;
    /// use spark_connect_async::column::AliasSpec;
    ///
    /// let one = col("id").alias("key");
    /// let entries = col("m").alias(["k", "v"]);
    /// let described = col("id").alias(AliasSpec::new("key").metadata(r#"{"comment": "the key"}"#));
    /// ```
    pub fn alias(self, alias: impl Into<AliasSpec>) -> Column {
        let AliasSpec { names, metadata } = alias.into();
        let Some(metadata) = metadata else {
            return Column(self.0.aliases(names));
        };
        let alias = Alias::with_names(self.expression().clone(), names).with_metadata(metadata);
        Column::new(ExpressionNode::Alias(Box::new(alias)).into())
    }

    /// The column converted to `data_type`: a [`DataType`], or a type in Spark SQL's syntax
    /// such as `"int"` or `"array<string>"`. A value that does not convert fails the query,
    /// unless the session is not in ANSI mode, where it is `NULL`.
    pub fn cast(self, data_type: impl IntoDataType) -> Column {
        Column(data_type.cast_column(self.0))
    }

    /// The column converted to `data_type`, as [`cast`](Self::cast) converts it, but `NULL`
    /// where a value does not convert.
    pub fn try_cast(self, data_type: impl IntoDataType) -> Column {
        Column(data_type.try_cast_column(self.0))
    }

    /// Whether the value is `NULL`.
    pub fn is_null(self) -> Column {
        Column(self.0.is_null())
    }

    /// Whether the value is not `NULL`.
    pub fn is_not_null(self) -> Column {
        Column(self.0.is_not_null())
    }

    /// Whether the value is a floating-point NaN.
    pub fn is_nan(self) -> Column {
        Column(self.0.is_nan())
    }

    /// Whether the value equals `other`'s; `NULL` if either is `NULL`.
    pub fn eq(self, other: impl Into<Column>) -> Column {
        Column(self.0.eq(to_core(other)))
    }

    /// Whether the value differs from `other`'s; `NULL` if either is `NULL`.
    pub fn ne(self, other: impl Into<Column>) -> Column {
        Column(self.0.ne(to_core(other)))
    }

    /// Whether the value is greater than `other`'s.
    pub fn gt(self, other: impl Into<Column>) -> Column {
        Column(self.0.gt(to_core(other)))
    }

    /// Whether the value is less than `other`'s.
    pub fn lt(self, other: impl Into<Column>) -> Column {
        Column(self.0.lt(to_core(other)))
    }

    /// Whether the value is greater than or equal to `other`'s.
    pub fn ge(self, other: impl Into<Column>) -> Column {
        Column(self.0.ge(to_core(other)))
    }

    /// Whether the value is less than or equal to `other`'s.
    pub fn le(self, other: impl Into<Column>) -> Column {
        Column(self.0.le(to_core(other)))
    }

    /// Whether the value equals `other`'s, where two `NULL`s are equal and a `NULL` differs
    /// from any other value; never `NULL`.
    pub fn eq_null_safe(self, other: impl Into<Column>) -> Column {
        Column(self.0.eq_null_safe(to_core(other)))
    }

    /// Whether the value is between `lower` and `upper`, both included.
    pub fn between(self, lower: impl Into<Column>, upper: impl Into<Column>) -> Column {
        Column(self.0.between(to_core(lower), to_core(upper)))
    }

    /// Whether the value equals one of `values`.
    pub fn is_in(self, values: impl IntoIterator<Item = impl Into<Column>>) -> Column {
        Column(self.0.isin(values.into_iter().map(to_core)))
    }

    /// Whether the value is in the rows of `query`, which has one column. Of a `struct` of
    /// columns, whether their values are in the rows of `query`, which has as many columns.
    pub fn is_in_subquery(self, query: &DataFrame) -> Column {
        Column(self.0.isin_subquery(query))
    }

    /// The logical AND of the value and `other`'s; also `&`.
    pub fn and(self, other: impl Into<Column>) -> Column {
        Column(self.0.and(to_core(other)))
    }

    /// The logical OR of the value and `other`'s; also `|`.
    pub fn or(self, other: impl Into<Column>) -> Column {
        Column(self.0.or(to_core(other)))
    }

    /// The bitwise AND of the value and `other`'s.
    pub fn bitwise_and(self, other: impl Into<Column>) -> Column {
        Column(self.0.bitwise_and(to_core(other)))
    }

    /// The bitwise OR of the value and `other`'s.
    pub fn bitwise_or(self, other: impl Into<Column>) -> Column {
        Column(self.0.bitwise_or(to_core(other)))
    }

    /// The bitwise exclusive OR of the value and `other`'s.
    pub fn bitwise_xor(self, other: impl Into<Column>) -> Column {
        Column(self.0.bitwise_xor(to_core(other)))
    }

    /// Whether the string matches the SQL `LIKE` pattern `pattern`, where `_` is any
    /// character and `%` any characters.
    pub fn like(self, pattern: &str) -> Column {
        Column(self.0.like(pattern))
    }

    /// Whether the string matches the `LIKE` pattern `pattern`, ignoring case.
    pub fn ilike(self, pattern: &str) -> Column {
        Column(self.0.ilike(pattern))
    }

    /// Whether the string contains a match of the Java regular expression `pattern`.
    pub fn rlike(self, pattern: &str) -> Column {
        Column(self.0.rlike(pattern))
    }

    /// Whether the string contains `other`'s; a string to look for is `lit("...")`.
    pub fn contains(self, other: impl Into<Column>) -> Column {
        Column(self.0.contains(to_core(other)))
    }

    /// Whether the string starts with `prefix`'s; a fixed prefix is `lit("...")`.
    pub fn starts_with(self, prefix: impl Into<Column>) -> Column {
        Column(self.0.startswith(to_core(prefix)))
    }

    /// Whether the string ends with `suffix`'s; a fixed suffix is `lit("...")`.
    pub fn ends_with(self, suffix: impl Into<Column>) -> Column {
        Column(self.0.endswith(to_core(suffix)))
    }

    /// The part of the string or binary value that starts at `start`, counted from 1, and
    /// has `length` characters or bytes.
    pub fn substr(self, start: impl Into<Column>, length: impl Into<Column>) -> Column {
        Column(self.0.substr(to_core(start), to_core(length)))
    }

    /// The field `name` of a struct.
    pub fn get_field(self, name: &str) -> Column {
        Column(self.0.get_field(name))
    }

    /// The element at `key`: of an array, at the index `key`, from 0; of a map, of the key
    /// `key`; of a struct, the field of the name `key`.
    pub fn get_item(self, key: impl Into<Column>) -> Column {
        Column(self.0.get_item(to_core(key)))
    }

    /// The struct with its field `name` set to `value`, added if it has no such field.
    pub fn with_field(self, name: &str, value: impl Into<Column>) -> Column {
        Column(self.0.with_field(name, to_core(value)))
    }

    /// The struct without its fields `names`.
    pub fn drop_fields(self, names: impl IntoIterator<Item = impl AsRef<str>>) -> Column {
        let names: Vec<_> = names.into_iter().collect();
        Column(
            self.0
                .drop_fields(names.iter().map(|name| name.as_ref()).collect()),
        )
    }

    /// The ascending order of the column, `NULL`s first.
    pub fn asc(self) -> Column {
        Column(self.0.asc())
    }

    /// The ascending order of the column, `NULL`s first.
    pub fn asc_nulls_first(self) -> Column {
        Column(self.0.asc_nulls_first())
    }

    /// The ascending order of the column, `NULL`s last.
    pub fn asc_nulls_last(self) -> Column {
        Column(self.0.asc_nulls_last())
    }

    /// The descending order of the column, `NULL`s last.
    pub fn desc(self) -> Column {
        Column(self.0.desc())
    }

    /// The descending order of the column, `NULL`s first.
    pub fn desc_nulls_first(self) -> Column {
        Column(self.0.desc_nulls_first())
    }

    /// The descending order of the column, `NULL`s last.
    pub fn desc_nulls_last(self) -> Column {
        Column(self.0.desc_nulls_last())
    }

    /// Adds the branch of `condition` to the `CASE WHEN` that [`when`] starts: its value is
    /// that of the first branch whose condition holds. Of a column that is not a `CASE WHEN`,
    /// it is a `CASE WHEN` of this branch, as [`when`] makes it.
    pub fn when(self, condition: impl Into<Column>, value: impl Into<Column>) -> Column {
        Column(self.0.when(to_core(condition), to_core(value)))
    }

    /// The value of the window function or aggregate for each row, over the rows of `window`.
    pub fn over(self, window: Window) -> Column {
        Column(self.0.over(window.into_core()))
    }

    /// The value of the `CASE WHEN` where no condition holds, which is `NULL` without it. A
    /// column that is not a `CASE WHEN` is returned unchanged.
    pub fn otherwise(self, value: impl Into<Column>) -> Column {
        Column(self.0.otherwise(to_core(value)))
    }
}

/// A `CASE WHEN` whose value is `value` where `condition` holds; [`Column::when`] adds
/// branches and [`Column::otherwise`] the value where none holds.
///
/// ```
/// use spark_connect_async::{col, lit, when};
///
/// let size = when(col("n").lt(lit(10)), lit("small"))
///     .when(col("n").lt(lit(100)), lit("medium"))
///     .otherwise(lit("large"));
/// ```
pub fn when(condition: impl Into<Column>, value: impl Into<Column>) -> Column {
    Column(spark_connect_core::column::when(
        to_core(condition),
        to_core(value),
    ))
}

macro_rules! binary_operators {
    ($($trait:ident :: $method:ident => $core:ident;)*) => {
        $(impl<T: Into<Column>> $trait<T> for Column {
            type Output = Column;

            fn $method(self, other: T) -> Column {
                Column(self.0.$core(to_core(other)))
            }
        })*
    };
}

binary_operators! {
    Add::add => add;
    Sub::sub => sub;
    Mul::mul => mul;
    Div::div => div;
    Rem::rem => modulo;
    BitAnd::bitand => and;
    BitOr::bitor => or;
}

impl Neg for Column {
    type Output = Column;

    fn neg(self) -> Column {
        Column(self.0.neg())
    }
}

impl Not for Column {
    type Output = Column;

    fn not(self) -> Column {
        Column(!self.0)
    }
}

/// The name or names that [`Column::alias`] gives a column. A `&str` or `String` is one name;
/// an array or a `Vec` of names names each column of a generator in order, such as the key and
/// the value of an exploded map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasSpec {
    names: Vec<String>,
    metadata: Option<String>,
}

impl AliasSpec {
    /// The name `name`.
    pub fn new(name: impl Into<String>) -> Self {
        AliasSpec::names([name])
    }

    /// The names `names`, one for each column of a generator.
    pub fn names(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        AliasSpec {
            names: names.into_iter().map(Into::into).collect(),
            metadata: None,
        }
    }

    /// The metadata of the column, a JSON object such as `{"comment": "..."}`, as a field of a
    /// schema has it. The server takes metadata only with one name.
    pub fn metadata(mut self, json: impl Into<String>) -> Self {
        self.metadata = Some(json.into());
        self
    }
}

impl From<&str> for AliasSpec {
    fn from(name: &str) -> Self {
        AliasSpec::new(name)
    }
}

impl From<String> for AliasSpec {
    fn from(name: String) -> Self {
        AliasSpec::new(name)
    }
}

impl<S: Into<String>, const N: usize> From<[S; N]> for AliasSpec {
    fn from(names: [S; N]) -> Self {
        AliasSpec::names(names)
    }
}

impl<S: Into<String>> From<Vec<S>> for AliasSpec {
    fn from(names: Vec<S>) -> Self {
        AliasSpec::names(names)
    }
}

/// A type that [`Column::cast`] and [`Column::try_cast`] take: a [`DataType`], or a type in
/// Spark SQL's syntax as a `&str` or `String`, such as `"decimal(10, 2)"`, which the server
/// parses.
pub trait IntoDataType: sealed::Sealed {
    #[doc(hidden)]
    fn cast_column(self, column: CoreColumn) -> CoreColumn;

    #[doc(hidden)]
    fn try_cast_column(self, column: CoreColumn) -> CoreColumn;
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::DataType {}
    impl Sealed for &str {}
    impl Sealed for String {}
}

impl IntoDataType for DataType {
    fn cast_column(self, column: CoreColumn) -> CoreColumn {
        column.cast(self)
    }

    fn try_cast_column(self, column: CoreColumn) -> CoreColumn {
        column.try_cast(self)
    }
}

impl IntoDataType for &str {
    fn cast_column(self, column: CoreColumn) -> CoreColumn {
        column.cast_str(self)
    }

    fn try_cast_column(self, column: CoreColumn) -> CoreColumn {
        column.try_cast_str(self)
    }
}

impl IntoDataType for String {
    fn cast_column(self, column: CoreColumn) -> CoreColumn {
        column.cast_str(&self)
    }

    fn try_cast_column(self, column: CoreColumn) -> CoreColumn {
        column.try_cast_str(&self)
    }
}

/// The column `name`, which the server resolves against the input of the operation that uses
/// it. A `&str` or `String` converts into the same column, wherever a column is taken.
/// [`DataFrame::col`](crate::DataFrame::col) refers to the column of one DataFrame, as a join
/// of two DataFrames with the same column names needs.
pub fn col(name: &str) -> Column {
    Column(spark_connect_core::column::col(name))
}

/// A value of an argument that is a literal or a column: a Rust value of [`IntoLiteral`],
/// as [`lit`] makes it, or a [`Column`]. A `&str` is a string, not a column name.
pub trait IntoArgument {
    #[doc(hidden)]
    fn into_argument(self) -> Column;
}

impl<T: IntoLiteral> IntoArgument for T {
    fn into_argument(self) -> Column {
        lit(self)
    }
}

impl IntoArgument for Column {
    fn into_argument(self) -> Column {
        self
    }
}

pub(crate) fn expressions(columns: impl IntoIterator<Item = impl Into<Column>>) -> Vec<Expression> {
    columns
        .into_iter()
        .map(|column| column.into().expression().clone())
        .collect()
}

pub(crate) fn subquery_column(subquery: SubqueryExpression) -> Column {
    Column::new(ExpressionNode::Subquery(Box::new(subquery)).into())
}

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
