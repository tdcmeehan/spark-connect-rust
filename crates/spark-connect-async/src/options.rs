//! The options of DataFrame operations: each is a value with the defaults of its operation,
//! and builder methods for the rest.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::ops::Range;

use spark_connect_core::error::SparkError;
use spark_connect_core::expression::{Expression, LiteralExpression};
use spark_connect_core::row::Value as CoreValue;

use crate::column::{expressions, Column, IntoLiteral};
use crate::error::Result;

/// The integers of [`Session::range`](crate::Session::range): from `start`, inclusive, to
/// `end`, exclusive, by `step`. An integer `n` is `0..n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeSpec {
    pub(crate) start: i64,
    pub(crate) end: i64,
    pub(crate) step: i64,
    pub(crate) partitions: Option<i32>,
}

impl RangeSpec {
    /// The integers of `range`, by 1, in the server's default number of partitions.
    pub fn new(range: Range<i64>) -> Self {
        RangeSpec {
            start: range.start,
            end: range.end,
            step: 1,
            partitions: None,
        }
    }

    /// The distance between two integers. A negative step counts down from `start`.
    pub fn step(mut self, step: i64) -> Self {
        self.step = step;
        self
    }

    /// The number of partitions of the DataFrame.
    pub fn partitions(mut self, partitions: i32) -> Self {
        self.partitions = Some(partitions);
        self
    }
}

impl From<Range<i64>> for RangeSpec {
    fn from(range: Range<i64>) -> Self {
        RangeSpec::new(range)
    }
}

impl From<Range<i32>> for RangeSpec {
    fn from(range: Range<i32>) -> Self {
        RangeSpec::new(range.start.into()..range.end.into())
    }
}

impl From<i64> for RangeSpec {
    fn from(end: i64) -> Self {
        RangeSpec::new(0..end)
    }
}

impl From<i32> for RangeSpec {
    fn from(end: i32) -> Self {
        RangeSpec::new(0..end.into())
    }
}

/// A random seed from 0 to `i64::MAX`, the range of PySpark's. The plan keeps it, so that every
/// execution of a sample, a split or a seeded function makes the same random choices.
pub(crate) fn random_seed() -> i64 {
    (RandomState::new().hash_one(()) >> 1) as i64
}

/// The sample of [`DataFrame::sample`](crate::DataFrame::sample): each row is chosen with the
/// probability `fraction`. A fraction `f` is `Sample::fraction(f)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub(crate) fraction: f64,
    pub(crate) with_replacement: bool,
    pub(crate) seed: i64,
}

impl Sample {
    /// A sample without replacement of about `fraction` of the rows, of a random seed.
    pub fn fraction(fraction: f64) -> Self {
        Sample {
            fraction,
            with_replacement: false,
            seed: random_seed(),
        }
    }

    /// Sample with replacement, so a row can be chosen more than once; `fraction` is then the
    /// expected number of times that each row is chosen.
    pub fn with_replacement(mut self) -> Self {
        self.with_replacement = true;
        self
    }

    /// The seed of the random choice.
    pub fn seed(mut self, seed: i64) -> Self {
        self.seed = seed;
        self
    }
}

impl From<f64> for Sample {
    fn from(fraction: f64) -> Self {
        Sample::fraction(fraction)
    }
}

/// The split of [`DataFrame::random_split`](crate::DataFrame::random_split): each part has
/// about the share of the rows that its weight has of the total.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomSplit {
    pub(crate) shares: Vec<f64>,
    pub(crate) seed: i64,
}

impl RandomSplit {
    /// A split of `weights`, of a random seed.
    ///
    /// Fails with [`ErrorKind::InvalidArgument`](crate::ErrorKind::InvalidArgument) and the
    /// condition `VALUE_NOT_POSITIVE` for a negative weight, or when the weights do not add up
    /// to more than zero.
    pub fn try_new(weights: impl IntoIterator<Item = f64>) -> Result<Self> {
        let mut weights: Vec<f64> = weights.into_iter().collect();
        if let Some(weight) = weights.iter().find(|weight| **weight < 0.0) {
            return Err(not_positive("weights", *weight).into());
        }
        // A sum from -0.0, as `Iterator::sum` starts, would report an empty list's sum as "-0".
        let total = weights.iter().fold(0.0, |sum, weight| sum + weight);
        if total <= 0.0 {
            return Err(not_positive("sum(weights)", total).into());
        }
        weights.iter_mut().for_each(|weight| *weight /= total);
        Ok(RandomSplit {
            shares: weights,
            seed: random_seed(),
        })
    }

    /// The seed of the random choice.
    pub fn seed(mut self, seed: i64) -> Self {
        self.seed = seed;
        self
    }
}

fn not_positive(name: &str, value: f64) -> SparkError {
    SparkError::value(
        "VALUE_NOT_POSITIVE",
        &[("arg_name", name), ("arg_value", &value.to_string())],
    )
}

/// The partitioning of [`DataFrame::repartition`](crate::DataFrame::repartition). An integer
/// `n` is `Partitioning::round_robin(n)`.
#[derive(Debug, Clone)]
pub struct Partitioning {
    pub(crate) kind: PartitioningKind,
    pub(crate) partitions: Option<i32>,
}

#[derive(Debug, Clone)]
pub(crate) enum PartitioningKind {
    RoundRobin,
    Hash(Vec<Expression>),
    Range(Vec<Expression>),
}

impl Partitioning {
    /// `partitions` partitions of about the same size.
    pub fn round_robin(partitions: i32) -> Self {
        Partitioning {
            kind: PartitioningKind::RoundRobin,
            partitions: Some(partitions),
        }
    }

    /// Partitions by the hash of `columns`, in the server's default number of partitions.
    pub fn hash(columns: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        Partitioning {
            kind: PartitioningKind::Hash(expressions(columns)),
            partitions: None,
        }
    }

    /// Partitions of ranges of the sort order of `columns`, in the server's default number of
    /// partitions. A column sorts ascending unless it is a sort order, such as `col.desc()`.
    pub fn range(columns: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        Partitioning {
            kind: PartitioningKind::Range(expressions(columns)),
            partitions: None,
        }
    }

    /// Partitions in which each row goes to the partition that `partition_id`, an integer
    /// column, gives.
    pub fn by_id(partition_id: impl Into<Column>) -> Self {
        use spark_connect_core::expression::ExpressionNode;
        let id = partition_id.into().expression().clone();
        Partitioning {
            kind: PartitioningKind::Hash(vec![ExpressionNode::DirectShufflePartitionId(id).into()]),
            partitions: None,
        }
    }

    /// The number of partitions.
    pub fn partitions(mut self, partitions: i32) -> Self {
        self.partitions = Some(partitions);
        self
    }
}

impl From<i32> for Partitioning {
    fn from(partitions: i32) -> Self {
        Partitioning::round_robin(partitions)
    }
}

/// How [`DataFrame::union_by_name`](crate::DataFrame::union_by_name) matches the columns of
/// both DataFrames. By default, both must have the same names.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnionByName {
    pub(crate) allow_missing_columns: bool,
}

impl UnionByName {
    /// Allow a column that only one DataFrame has: its values are `NULL` in the rows of the
    /// other.
    pub fn allow_missing_columns(mut self) -> Self {
        self.allow_missing_columns = true;
        self
    }
}

/// The type of a join.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum JoinType {
    /// The pairs of rows that match.
    #[default]
    Inner,
    /// The pairs of rows that match, and each other row of the left DataFrame with `NULL`s.
    LeftOuter,
    /// The pairs of rows that match, and each other row of the right DataFrame with `NULL`s.
    RightOuter,
    /// The pairs of rows that match, and each other row of either DataFrame with `NULL`s.
    FullOuter,
    /// The rows of the left DataFrame that match a row of the right one.
    LeftSemi,
    /// The rows of the left DataFrame that match no row of the right one.
    LeftAnti,
    /// Every pair of rows.
    Cross,
}

impl JoinType {
    pub(crate) fn plan(self) -> spark_connect_core::plan::JoinType {
        use spark_connect_core::plan::JoinType as Plan;
        match self {
            JoinType::Inner => Plan::Inner,
            JoinType::LeftOuter => Plan::LeftOuter,
            JoinType::RightOuter => Plan::RightOuter,
            JoinType::FullOuter => Plan::FullOuter,
            JoinType::LeftSemi => Plan::LeftSemi,
            JoinType::LeftAnti => Plan::LeftAnti,
            JoinType::Cross => Plan::Cross,
        }
    }
}

/// The as-of join of [`DataFrame::as_of_join`](crate::DataFrame::as_of_join): each row of
/// the left DataFrame is joined with the row of the right one whose as-of value is the
/// nearest in its [`AsOfDirection`]. By default an inner join, which also matches equal
/// values, looking backward.
#[derive(Debug, Clone)]
pub struct AsOfJoin {
    pub(crate) left_as_of: Column,
    pub(crate) right_as_of: Column,
    pub(crate) on: Option<Column>,
    pub(crate) using_columns: Vec<String>,
    pub(crate) left_outer: bool,
    pub(crate) tolerance: Option<Column>,
    pub(crate) allow_exact_matches: bool,
    pub(crate) direction: AsOfDirection,
}

impl AsOfJoin {
    /// The join on the as-of columns `left_as_of`, of the left DataFrame, and `right_as_of`,
    /// of the right one.
    pub fn new(left_as_of: impl Into<Column>, right_as_of: impl Into<Column>) -> Self {
        AsOfJoin {
            left_as_of: left_as_of.into(),
            right_as_of: right_as_of.into(),
            on: None,
            using_columns: Vec::new(),
            left_outer: false,
            tolerance: None,
            allow_exact_matches: true,
            direction: AsOfDirection::Backward,
        }
    }

    /// Join only rows for which `condition` holds, besides the as-of match.
    pub fn on(mut self, condition: impl Into<Column>) -> Self {
        self.on = Some(condition.into());
        self
    }

    /// Join only rows with equal values in the columns `names`, which both DataFrames have.
    pub fn using(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.using_columns = names.into_iter().map(Into::into).collect();
        self
    }

    /// A left outer join, which also keeps each row of the left DataFrame that has no match,
    /// with `NULL`s, rather than an inner join.
    pub fn left_outer(mut self) -> Self {
        self.left_outer = true;
        self
    }

    /// The largest distance between the as-of values of a match.
    pub fn tolerance(mut self, tolerance: impl Into<Column>) -> Self {
        self.tolerance = Some(tolerance.into());
        self
    }

    /// Match only rows whose as-of values differ.
    pub fn exclude_exact_matches(mut self) -> Self {
        self.allow_exact_matches = false;
        self
    }

    /// The direction in which to look for the nearest as-of value.
    pub fn direction(mut self, direction: AsOfDirection) -> Self {
        self.direction = direction;
        self
    }
}

/// Where an [`AsOfJoin`] looks for the nearest as-of value of the right DataFrame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum AsOfDirection {
    /// The largest value that is not greater than the left one.
    #[default]
    Backward,
    /// The smallest value that is not less than the left one.
    Forward,
    /// The value nearest to the left one, in either direction.
    Nearest,
}

impl AsOfDirection {
    pub(crate) fn name(self) -> &'static str {
        match self {
            AsOfDirection::Backward => "backward",
            AsOfDirection::Forward => "forward",
            AsOfDirection::Nearest => "nearest",
        }
    }
}

/// The nearest-by join of [`DataFrame::nearest_by_join`](crate::DataFrame::nearest_by_join):
/// each row of the left DataFrame is joined with the rows of the right one that rank first
/// by a ranking expression of both. By default an inner join of the approximately nearest
/// rows by distance.
#[derive(Debug, Clone)]
pub struct NearestByJoin {
    pub(crate) ranking: Column,
    pub(crate) num_results: i32,
    pub(crate) left_outer: bool,
    pub(crate) exact: bool,
    pub(crate) similarity: bool,
}

impl NearestByJoin {
    /// The join with the `num_results` right rows that rank first by `ranking`.
    pub fn new(ranking: impl Into<Column>, num_results: i32) -> Self {
        NearestByJoin {
            ranking: ranking.into(),
            num_results,
            left_outer: false,
            exact: false,
            similarity: false,
        }
    }

    /// Rank every pair of rows, rather than approximately.
    pub fn exact(mut self) -> Self {
        self.exact = true;
        self
    }

    /// Rank the rows of the greatest ranking values first, as of a similarity, rather than
    /// of the least, as of a distance.
    pub fn by_similarity(mut self) -> Self {
        self.similarity = true;
        self
    }

    /// A left outer join, which also keeps each row of the left DataFrame that has no match,
    /// with `NULL`s, rather than an inner join.
    pub fn left_outer(mut self) -> Self {
        self.left_outer = true;
        self
    }
}

/// The pivot of [`GroupedData::pivot`](crate::group::GroupedData::pivot): a column of the
/// result for each value of a column, in each group. Without values, the server finds them,
/// which runs a query. A column or a column name is a pivot without values.
#[derive(Debug, Clone)]
pub struct Pivot {
    pub(crate) column: Column,
    pub(crate) values: Vec<Expression>,
}

impl Pivot {
    /// The pivot on the values of `column` that the server finds.
    pub fn new(column: impl Into<Column>) -> Self {
        Pivot {
            column: column.into(),
            values: Vec::new(),
        }
    }

    /// Pivots on `values` only, in their order.
    pub fn values(mut self, values: impl IntoIterator<Item = impl IntoLiteral>) -> Self {
        self.values = values
            .into_iter()
            .map(|value| Expression::literal(value.into_literal()))
            .collect();
        self
    }
}

impl From<&str> for Pivot {
    fn from(name: &str) -> Self {
        Pivot::new(name)
    }
}

impl From<String> for Pivot {
    fn from(name: String) -> Self {
        Pivot::new(name)
    }
}

impl From<Column> for Pivot {
    fn from(column: Column) -> Self {
        Pivot::new(column)
    }
}

/// A value that replaces `NULL`s: an integer, a floating-point number, a boolean or a string.
/// The server fills only the columns of the value's type; an integer fills as a `BIGINT`.
pub trait FillScalar: sealed::Sealed {
    #[doc(hidden)]
    fn into_fill(self) -> CoreValue;
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f64 {}
    impl Sealed for bool {}
    impl Sealed for &str {}
    impl Sealed for String {}
}

macro_rules! fill_scalar {
    ($($rust:ty => |$value:ident| $fill:expr;)*) => {
        $(impl FillScalar for $rust {
            fn into_fill(self) -> CoreValue {
                let $value = self;
                $fill
            }
        })*
    };
}

fill_scalar! {
    i32 => |v| CoreValue::Long(v.into());
    i64 => |v| CoreValue::Long(v);
    f64 => |v| CoreValue::Double(v);
    bool => |v| CoreValue::Bool(v);
    &str => |v| CoreValue::String(v.to_string());
    String => |v| CoreValue::String(v);
}

/// What [`NaFunctions::fill`](crate::stat::NaFunctions::fill) puts in place of `NULL`s: one
/// value, in every column of its type or in some columns, or a value of each column
/// ([`FillColumns`]). A [`FillScalar`] value converts into it.
///
/// ```
/// use spark_connect_async::options::{FillColumns, FillValue};
///
/// let zero = FillValue::from(0);
/// let unknown = FillValue::from("unknown").subset(["name", "city"]);
/// let each = FillColumns::new().column("age", 0).column("name", "unknown");
/// ```
#[derive(Debug, Clone)]
pub struct FillValue {
    pub(crate) fill: Fill,
}

#[derive(Debug, Clone)]
pub(crate) enum Fill {
    One {
        value: CoreValue,
        columns: Vec<String>,
    },
    PerColumn(Vec<(String, CoreValue)>),
}

impl FillValue {
    /// Fills only the columns `columns`, of the value's type. A value of each column
    /// ([`FillColumns`]) keeps its own columns.
    pub fn subset(self, columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        match self.fill {
            Fill::One { value, .. } => FillValue {
                fill: Fill::One {
                    value,
                    columns: columns.into_iter().map(Into::into).collect(),
                },
            },
            per_column @ Fill::PerColumn(_) => FillValue { fill: per_column },
        }
    }
}

impl<T: FillScalar> From<T> for FillValue {
    fn from(value: T) -> Self {
        FillValue {
            fill: Fill::One {
                value: value.into_fill(),
                columns: Vec::new(),
            },
        }
    }
}

/// The value of each column that [`NaFunctions::fill`](crate::stat::NaFunctions::fill) puts in
/// place of its `NULL`s.
#[derive(Debug, Clone, Default)]
pub struct FillColumns(Vec<(String, CoreValue)>);

impl FillColumns {
    /// No column.
    pub fn new() -> Self {
        FillColumns::default()
    }

    /// Fills the column `name` with `value`.
    pub fn column(mut self, name: impl Into<String>, value: impl FillScalar) -> Self {
        self.0.push((name.into(), value.into_fill()));
        self
    }
}

impl From<FillColumns> for FillValue {
    fn from(columns: FillColumns) -> Self {
        FillValue {
            fill: Fill::PerColumn(columns.0),
        }
    }
}

/// Which rows [`NaFunctions::drop`](crate::stat::NaFunctions::drop) drops: by default each
/// row with a `NULL` or NaN in any column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropNa {
    pub(crate) rows: DropRows,
    pub(crate) columns: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DropRows {
    Any,
    All,
    AtLeast(i32),
}

impl DropNa {
    /// Drops each row with a `NULL` or NaN in any column.
    pub fn any() -> Self {
        DropNa {
            rows: DropRows::Any,
            columns: Vec::new(),
        }
    }

    /// Drops each row whose columns are all `NULL` or NaN.
    pub fn all() -> Self {
        DropNa {
            rows: DropRows::All,
            ..DropNa::any()
        }
    }

    /// Drops each row with fewer than `count` values that are neither `NULL` nor NaN.
    pub fn at_least(count: i32) -> Self {
        DropNa {
            rows: DropRows::AtLeast(count),
            ..DropNa::any()
        }
    }

    /// Looks only at the columns `columns`.
    pub fn subset(mut self, columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.columns = columns.into_iter().map(Into::into).collect();
        self
    }
}

impl Default for DropNa {
    fn default() -> Self {
        DropNa::any()
    }
}

/// The replacements of [`NaFunctions::replace`](crate::stat::NaFunctions::replace): each value
/// and the value that replaces it, which may be `None`, in every column of the value's type
/// or in some columns. The server replaces booleans, floating-point numbers and strings; an
/// array of pairs, or a `Vec`, converts into it.
///
/// ```
/// use spark_connect_async::options::Replace;
///
/// let names = Replace::from([("Bob", "Robert"), ("Al", "Albert")]).subset(["name"]);
/// let mixed = Replace::new().pair(0.0, None::<f64>).pair("n/a", None::<&str>);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Replace {
    pub(crate) replacements: Vec<(LiteralExpression, LiteralExpression)>,
    pub(crate) columns: Vec<String>,
}

impl Replace {
    /// No replacement.
    pub fn new() -> Self {
        Replace::default()
    }

    /// Replaces `value` with `replacement`.
    pub fn pair(mut self, value: impl IntoLiteral, replacement: impl IntoLiteral) -> Self {
        self.replacements
            .push((value.into_literal(), replacement.into_literal()));
        self
    }

    /// Replaces only in the columns `columns`.
    pub fn subset(mut self, columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.columns = columns.into_iter().map(Into::into).collect();
        self
    }
}

impl<V: IntoLiteral, R: IntoLiteral, const N: usize> From<[(V, R); N]> for Replace {
    fn from(pairs: [(V, R); N]) -> Self {
        Replace::from(Vec::from(pairs))
    }
}

impl<V: IntoLiteral, R: IntoLiteral> From<Vec<(V, R)>> for Replace {
    fn from(pairs: Vec<(V, R)>) -> Self {
        pairs
            .into_iter()
            .fold(Replace::new(), |replace, (value, replacement)| {
                replace.pair(value, replacement)
            })
    }
}
