//! The groupings of a DataFrame's rows, and their aggregations.

use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use spark_connect_core::error::{SparkError, SparkErrorKind};
use spark_connect_core::expression::Expression;
use spark_connect_core::group::{numeric_aggregates, NumericAggregates};
use spark_connect_core::plan::{self, AggregateGroupType};

use crate::column::{expressions, lit, to_core, Column};
use crate::dataframe::DataFrame;
use crate::error::Result;
use crate::options::Pivot;

/// The grouping of [`DataFrame::group_by`], the only one that
/// [`GroupedData::pivot`] pivots.
#[derive(Debug, Clone, Copy)]
pub enum ByColumns {}

/// A grouping that does not pivot: of a rollup, a cube, grouping sets, or a pivot.
#[derive(Debug, Clone, Copy)]
pub enum Grouped {}

/// The rows of a DataFrame in groups, which an aggregation turns into one row each. `K` is
/// [`ByColumns`] for the grouping of [`DataFrame::group_by`], which pivots, and [`Grouped`]
/// for the others.
///
/// ```
/// # async fn example(df: spark_connect_async::DataFrame) -> spark_connect_async::Result<()> {
/// use spark_connect_async::options::Pivot;
///
/// let counts = df.group_by(["city"]).count();
/// let sums = df.rollup(["country", "city"]).sum(["sales"]).await?;
/// let by_year = df.group_by(["city"]).pivot(Pivot::new("year").values([2025, 2026]));
/// let pivoted = by_year.sum(["sales"]).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct GroupedData<K = Grouped> {
    dataframe: DataFrame,
    columns: Vec<Expression>,
    grouping: Grouping,
    kind: PhantomData<K>,
}

#[derive(Clone)]
pub(crate) enum Grouping {
    By,
    Rollup,
    Cube,
    Sets(Vec<Vec<Expression>>),
    Pivot(Expression, Vec<Expression>),
}

impl<K> GroupedData<K> {
    pub(crate) fn new(dataframe: DataFrame, columns: Vec<Expression>, grouping: Grouping) -> Self {
        GroupedData {
            dataframe,
            columns,
            grouping,
            kind: PhantomData,
        }
    }

    /// A row of each group with the grouping columns and the aggregations `aggregates`, such
    /// as `sum(col("sales")).alias("total")`.
    pub fn agg(&self, aggregates: impl IntoIterator<Item = impl Into<Column>>) -> DataFrame {
        aggregated(
            &self.dataframe,
            &self.columns,
            &self.grouping,
            expressions(aggregates),
        )
    }

    /// The number of rows of each group, in a column `count`.
    pub fn count(&self) -> DataFrame {
        let count = spark_connect_core::functions::count(to_core(lit(1))).alias("count");
        self.agg([Column::from(count)])
    }

    /// The sums of the numeric columns `columns` in each group, named `sum(<column>)`; of
    /// every numeric column without `columns`. It reads the schema of the DataFrame, and
    /// fails if a column of `columns` is not a numeric one, as a struct is not.
    pub fn sum(
        &self,
        columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> impl Future<Output = Result<DataFrame>> + Send + 'static {
        self.numeric("sum", columns)
    }

    /// The averages of the numeric columns `columns` in each group, as [`sum`](Self::sum)
    /// takes them.
    pub fn avg(
        &self,
        columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> impl Future<Output = Result<DataFrame>> + Send + 'static {
        self.numeric("avg", columns)
    }

    /// The least values of the numeric columns `columns` in each group, as
    /// [`sum`](Self::sum) takes them.
    pub fn min(
        &self,
        columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> impl Future<Output = Result<DataFrame>> + Send + 'static {
        self.numeric("min", columns)
    }

    /// The greatest values of the numeric columns `columns` in each group, as
    /// [`sum`](Self::sum) takes them.
    pub fn max(
        &self,
        columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> impl Future<Output = Result<DataFrame>> + Send + 'static {
        self.numeric("max", columns)
    }

    fn numeric(
        &self,
        function: &'static str,
        columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> impl Future<Output = Result<DataFrame>> + Send + 'static {
        let (dataframe, grouping_columns, grouping) = (
            self.dataframe.clone(),
            self.columns.clone(),
            self.grouping.clone(),
        );
        let columns: Vec<String> = columns.into_iter().map(Into::into).collect();
        async move {
            let schema = dataframe.schema().await?;
            let aggregates = numeric_columns(numeric_aggregates(function, &schema, &columns))?;
            Ok(aggregated(
                &dataframe,
                &grouping_columns,
                &grouping,
                aggregates,
            ))
        }
    }
}

/// The aggregations of numeric columns, or the error of the given columns that are not numeric.
fn numeric_columns(aggregates: NumericAggregates) -> Result<Vec<Expression>> {
    let invalid = match aggregates {
        NumericAggregates::Aggregates(aggregates) => return Ok(aggregates),
        NumericAggregates::NotNumeric(invalid) => invalid,
        NumericAggregates::Struct { column, .. } => vec![column],
    };
    Err(SparkError::classed(
        SparkErrorKind::TypeError,
        "NOT_NUMERIC_COLUMNS",
        &[("invalid_columns", &format!("{invalid:?}"))],
    )
    .into())
}

fn aggregated(
    dataframe: &DataFrame,
    columns: &[Expression],
    grouping: &Grouping,
    aggregates: Vec<Expression>,
) -> DataFrame {
    let input = Arc::clone(dataframe.plan());
    let columns = columns.to_vec();
    dataframe.with_plan(match grouping {
        Grouping::By => plan::aggregate(input, AggregateGroupType::GroupBy, columns, aggregates),
        Grouping::Rollup => plan::aggregate(input, AggregateGroupType::Rollup, columns, aggregates),
        Grouping::Cube => plan::aggregate(input, AggregateGroupType::Cube, columns, aggregates),
        Grouping::Sets(sets) => {
            plan::aggregate_with_grouping_sets(input, columns, aggregates, sets.clone())
        }
        Grouping::Pivot(column, values) => plan::aggregate_with_pivot(
            input,
            AggregateGroupType::Pivot,
            columns,
            aggregates,
            column.clone(),
            values.clone(),
        ),
    })
}

impl GroupedData<ByColumns> {
    /// Each group with a column of each value of the pivot's column, for each aggregation,
    /// named `<value>_<aggregation>` when there are several aggregations and `<value>`
    /// otherwise.
    pub fn pivot(&self, pivot: impl Into<Pivot>) -> GroupedData {
        let pivot = pivot.into();
        GroupedData::new(
            self.dataframe.clone(),
            self.columns.clone(),
            Grouping::Pivot(pivot.column.expression().clone(), pivot.values),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_struct_column_is_not_numeric() {
        let error = numeric_columns(NumericAggregates::Struct {
            column: "st".to_string(),
            error: Box::new(SparkError::value_msg("list index out of range")),
        })
        .err()
        .unwrap();
        assert_eq!(error.error_class(), Some("NOT_NUMERIC_COLUMNS"));
        assert!(error.message().contains(r#"["st"]"#), "{error}");
    }
}
