//! The statistics of a DataFrame's columns, and the handling of their missing values.

use std::future::Future;

use arrow::record_batch::RecordBatch;
use futures::{FutureExt, TryFutureExt};
use std::sync::Arc;

use spark_connect_core::expression::Expression;
use spark_connect_core::plan::{self, LogicalPlan};

use crate::column::{Column, IntoLiteral};
use crate::dataframe::{strings, DataFrame};
use crate::error::{Error, Result};
use crate::options::{random_seed, DropNa, DropRows, Fill, FillValue, Replace};

/// The statistics of [`DataFrame::stat`].
#[derive(Clone)]
pub struct StatFunctions {
    dataframe: DataFrame,
}

impl StatFunctions {
    pub(crate) fn new(dataframe: DataFrame) -> Self {
        StatFunctions { dataframe }
    }

    fn with_input(&self, plan: impl FnOnce(Arc<LogicalPlan>) -> LogicalPlan) -> DataFrame {
        self.dataframe
            .with_plan(plan(Arc::clone(self.dataframe.plan())))
    }

    /// The table of how often each pair of values of the columns `column1` and `column2`
    /// occurs: a row of each value of `column1`, and a column of each value of `column2`.
    pub fn crosstab(&self, column1: impl Into<String>, column2: impl Into<String>) -> DataFrame {
        self.with_input(|input| LogicalPlan::StatCrosstab {
            input,
            col1: column1.into(),
            col2: column2.into(),
        })
    }

    /// The values of each of the columns `columns` that occur in at least the fraction
    /// `support` of the rows, possibly with false positives, in a column `<column>_freqItems`
    /// of an array.
    pub fn freq_items(
        &self,
        columns: impl IntoIterator<Item = impl Into<String>>,
        support: f64,
    ) -> DataFrame {
        self.with_input(|input| LogicalPlan::StatFreqItems {
            input,
            columns: strings(columns),
            support,
        })
    }

    /// A sample of the rows without replacement whose values of `column` are a stratum of
    /// `fractions`, each with the probability of its fraction; a value of no stratum is not
    /// chosen. `seed` is optional: `None` draws a random seed, which the plan keeps.
    pub fn sample_by(
        &self,
        column: impl Into<Column>,
        fractions: impl IntoIterator<Item = (impl IntoLiteral, f64)>,
        seed: Option<i64>,
    ) -> DataFrame {
        let fractions = fractions
            .into_iter()
            .map(|(stratum, fraction)| (Expression::literal(stratum.into_literal()), fraction))
            .collect();
        let seed = seed.unwrap_or_else(random_seed);
        self.with_input(|input| LogicalPlan::StatSampleBy {
            input,
            col: column.into().expression().clone(),
            fractions,
            seed: Some(seed),
        })
    }

    /// The Pearson correlation coefficient of the numeric columns `column1` and `column2`,
    /// counting a `NULL` as 0. Where it is undefined, as with fewer than two rows or a column of
    /// one value, the server answers NaN. An answer that is `NULL` or missing, which a 4.3.0
    /// server doesn't send, is an [`ErrorKind::Decode`](crate::ErrorKind::Decode) error.
    pub fn corr(
        &self,
        column1: impl Into<String>,
        column2: impl Into<String>,
    ) -> impl Future<Output = Result<f64>> + Send + 'static {
        let answer = self.with_input(|input| LogicalPlan::StatCorr {
            input,
            col1: column1.into(),
            col2: column2.into(),
        });
        first_value(answer, "correlation")
    }

    /// The sample covariance of the numeric columns `column1` and `column2`, counting a `NULL` as
    /// 0. With fewer than two rows, where it is undefined, the server answers 0.0. An answer that
    /// is `NULL` or missing, which a 4.3.0 server doesn't send, is an
    /// [`ErrorKind::Decode`](crate::ErrorKind::Decode) error.
    pub fn cov(
        &self,
        column1: impl Into<String>,
        column2: impl Into<String>,
    ) -> impl Future<Output = Result<f64>> + Send + 'static {
        let answer = self.with_input(|input| LogicalPlan::StatCov {
            input,
            col1: column1.into(),
            col2: column2.into(),
        });
        first_value(answer, "covariance")
    }

    /// The approximate quantiles at `probabilities`, each from 0 to 1, of each of the numeric
    /// columns `columns`, leaving out `NULL`s and NaNs: one `Vec` of each column, in order, empty
    /// for a column with no other values. The rank of each is within `relative_error` times the
    /// number of values of the exact one; 0 computes exact quantiles, at a greater cost.
    pub fn approx_quantile(
        &self,
        columns: impl IntoIterator<Item = impl Into<String>>,
        probabilities: impl IntoIterator<Item = f64>,
        relative_error: f64,
    ) -> impl Future<Output = Result<Vec<Vec<f64>>>> + Send + 'static {
        self.with_input(|input| LogicalPlan::StatApproxQuantile {
            input,
            columns: strings(columns),
            probabilities: probabilities.into_iter().collect(),
            relative_error,
        })
        .collect_as::<(Vec<Vec<f64>>,)>()
        .map_ok(|rows| {
            rows.into_iter()
                .next()
                .map(|(quantiles,)| quantiles)
                .unwrap_or_default()
        })
    }
}

fn first_value(
    dataframe: DataFrame,
    what: &'static str,
) -> impl Future<Output = Result<f64>> + Send + 'static {
    dataframe
        .limit(1)
        .collect()
        .map(move |batches| value_of(&batches?, what))
}

/// The value of the first row of `batches`, the answer of a `what`. A `NULL` or no row is an
/// error.
fn value_of(batches: &[RecordBatch], what: &str) -> Result<f64> {
    for batch in batches {
        let rows = crate::de::from_record_batch::<(f64,)>(batch)?;
        if let Some(&(value,)) = rows.first() {
            return Ok(value);
        }
    }
    Err(Error::decode(format!("the server returned no {what}")))
}

/// The handling of the missing values of [`DataFrame::na`]: `NULL`s, and NaNs where noted.
#[derive(Clone)]
pub struct NaFunctions {
    dataframe: DataFrame,
}

impl NaFunctions {
    pub(crate) fn new(dataframe: DataFrame) -> Self {
        NaFunctions { dataframe }
    }

    /// The rows without those that `drop` selects, such as `DropNa::all()`.
    pub fn drop(&self, drop: DropNa) -> DataFrame {
        let (how, min_non_null) = match drop.rows {
            DropRows::Any => ("any", None),
            DropRows::All => ("all", None),
            DropRows::AtLeast(count) => ("any", Some(count)),
        };
        let input = Arc::clone(self.dataframe.plan());
        self.dataframe
            .with_plan(plan::na_drop(input, how, min_non_null, drop.columns))
    }

    /// The rows with the `NULL`s and NaNs of the columns that `value` fills replaced by its
    /// value.
    ///
    /// ```
    /// # fn example(df: spark_connect_async::DataFrame) {
    /// use spark_connect_async::options::{FillColumns, FillValue};
    ///
    /// let zeros = df.na().fill(0);
    /// let names = df.na().fill(FillValue::from("unknown").subset(["name"]));
    /// let each = df.na().fill(FillColumns::new().column("age", 0).column("name", "?"));
    /// # }
    /// ```
    pub fn fill(&self, value: impl Into<FillValue>) -> DataFrame {
        let input = Arc::clone(self.dataframe.plan());
        self.dataframe.with_plan(match value.into().fill {
            Fill::One { value, columns } => plan::na_fill(input, value, columns),
            Fill::PerColumn(pairs) => {
                let (cols, values) = pairs.into_iter().unzip();
                LogicalPlan::NAFillColumns {
                    input,
                    cols,
                    values,
                }
            }
        })
    }

    /// The rows with the values of `replace` replaced.
    pub fn replace(&self, replace: impl Into<Replace>) -> DataFrame {
        let replace = replace.into();
        self.dataframe.with_plan(LogicalPlan::NAReplace {
            input: Arc::clone(self.dataframe.plan()),
            replacements: replace.replacements,
            columns: replace.columns,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    use super::value_of;
    use crate::ErrorKind;

    /// An answer of `values` in its one column, `corr`.
    fn answer(values: Vec<Option<f64>>) -> RecordBatch {
        let schema = Schema::new(vec![Field::new("corr", DataType::Float64, true)]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Float64Array::from(values))]).unwrap()
    }

    #[test]
    fn a_statistic_is_the_value_of_the_first_row_of_its_answer() {
        for (name, batches, expected) in [
            ("a value", vec![answer(vec![Some(0.5)])], 0.5),
            ("NaN", vec![answer(vec![Some(f64::NAN)])], f64::NAN),
            (
                "an empty batch, then a value",
                vec![answer(vec![]), answer(vec![Some(-1.0)])],
                -1.0,
            ),
        ] {
            let value = value_of(&batches, "correlation").unwrap();
            assert_eq!(value.to_bits(), expected.to_bits(), "{name}: {value}");
        }
    }

    #[test]
    fn a_null_or_missing_answer_of_a_statistic_is_an_error_of_decoding() {
        for (name, batches, message) in [
            (
                "a NULL",
                vec![answer(vec![None])],
                "a NULL can only be read into an Option (row 0, column `corr`)",
            ),
            (
                "an empty batch",
                vec![answer(vec![])],
                "the server returned no correlation",
            ),
            ("no batch", vec![], "the server returned no correlation"),
        ] {
            let error = value_of(&batches, "correlation").unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Decode, "{name}");
            assert_eq!(error.message(), message, "{name}");
        }
    }
}
