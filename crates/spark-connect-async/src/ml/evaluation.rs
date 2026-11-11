//! The evaluators of predictions.

use std::future::Future;

use spark_connect_core::ml::OperatorType;

use super::{Evaluator, Operator};
use crate::dataframe::DataFrame;
use crate::error::Result;

/// Measures the predictions of numbers: by default their root mean squared error, `rmse`.
#[derive(Debug, Clone)]
pub struct RegressionEvaluator(Operator);

impl RegressionEvaluator {
    /// An evaluator of the column `prediction` against the column `label`, by `rmse`.
    pub fn new() -> Self {
        RegressionEvaluator(Operator::new(
            "org.apache.spark.ml.evaluation.RegressionEvaluator",
            OperatorType::Evaluator,
        ))
    }

    /// The column of the true numbers; by default `label`.
    pub fn label_col(self, name: impl Into<String>) -> Self {
        RegressionEvaluator(self.0.string("labelCol", name))
    }

    /// The column of the predictions; by default `prediction`.
    pub fn prediction_col(self, name: impl Into<String>) -> Self {
        RegressionEvaluator(self.0.string("predictionCol", name))
    }

    /// The metric: `rmse`, `mse`, `mae`, `r2` or `var`, the variance that the predictions
    /// explain.
    pub fn metric_name(self, metric: impl Into<String>) -> Self {
        RegressionEvaluator(self.0.string("metricName", metric))
    }
}

impl Default for RegressionEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for RegressionEvaluator {
    fn evaluate(&self, dataset: &DataFrame) -> impl Future<Output = Result<f64>> + Send + 'static {
        self.0.evaluate(dataset)
    }

    /// Whether the metric is `r2` or `var`; the errors are better smaller.
    fn is_larger_better(&self) -> bool {
        matches!(self.0.text("metricName"), Some("r2" | "var"))
    }
}

/// Measures the predictions of two classes, 0 and 1, by their raw predictions: by default the
/// area under the ROC curve, `areaUnderROC`.
#[derive(Debug, Clone)]
pub struct BinaryClassificationEvaluator(Operator);

impl BinaryClassificationEvaluator {
    /// An evaluator of the column `rawPrediction` against the column `label`, by
    /// `areaUnderROC`.
    pub fn new() -> Self {
        BinaryClassificationEvaluator(Operator::new(
            "org.apache.spark.ml.evaluation.BinaryClassificationEvaluator",
            OperatorType::Evaluator,
        ))
    }

    /// The column of the true classes; by default `label`.
    pub fn label_col(self, name: impl Into<String>) -> Self {
        BinaryClassificationEvaluator(self.0.string("labelCol", name))
    }

    /// The column of the raw predictions, vectors of a score of each class or numbers; by
    /// default `rawPrediction`.
    pub fn raw_prediction_col(self, name: impl Into<String>) -> Self {
        BinaryClassificationEvaluator(self.0.string("rawPredictionCol", name))
    }

    /// The metric: `areaUnderROC` or `areaUnderPR`.
    pub fn metric_name(self, metric: impl Into<String>) -> Self {
        BinaryClassificationEvaluator(self.0.string("metricName", metric))
    }
}

impl Default for BinaryClassificationEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for BinaryClassificationEvaluator {
    fn evaluate(&self, dataset: &DataFrame) -> impl Future<Output = Result<f64>> + Send + 'static {
        self.0.evaluate(dataset)
    }

    /// Always: both metrics are areas.
    fn is_larger_better(&self) -> bool {
        true
    }
}

/// Measures the predictions of classes: by default their `f1` score.
#[derive(Debug, Clone)]
pub struct MulticlassClassificationEvaluator(Operator);

impl MulticlassClassificationEvaluator {
    /// An evaluator of the column `prediction` against the column `label`, by `f1`.
    pub fn new() -> Self {
        MulticlassClassificationEvaluator(Operator::new(
            "org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator",
            OperatorType::Evaluator,
        ))
    }

    /// The column of the true classes; by default `label`.
    pub fn label_col(self, name: impl Into<String>) -> Self {
        MulticlassClassificationEvaluator(self.0.string("labelCol", name))
    }

    /// The column of the predicted classes; by default `prediction`.
    pub fn prediction_col(self, name: impl Into<String>) -> Self {
        MulticlassClassificationEvaluator(self.0.string("predictionCol", name))
    }

    /// The metric, such as `f1`, `accuracy`, `weightedPrecision`, `weightedRecall`, `logLoss`
    /// or `hammingLoss`.
    pub fn metric_name(self, metric: impl Into<String>) -> Self {
        MulticlassClassificationEvaluator(self.0.string("metricName", metric))
    }
}

impl Default for MulticlassClassificationEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for MulticlassClassificationEvaluator {
    fn evaluate(&self, dataset: &DataFrame) -> impl Future<Output = Result<f64>> + Send + 'static {
        self.0.evaluate(dataset)
    }

    /// Whether the metric is not a rate of false positives or a loss.
    fn is_larger_better(&self) -> bool {
        !matches!(
            self.0.text("metricName"),
            Some(
                "weightedFalsePositiveRate"
                    | "falsePositiveRateByLabel"
                    | "logLoss"
                    | "hammingLoss"
            )
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_evaluator_tells_whether_a_larger_metric_is_better_by_its_metric() {
        let regression = RegressionEvaluator::new;
        let multiclass = MulticlassClassificationEvaluator::new;
        let cases: [(&str, &dyn Fn() -> bool, bool); 12] = [
            (
                "rmse by default",
                &|| regression().is_larger_better(),
                false,
            ),
            (
                "mse",
                &|| regression().metric_name("mse").is_larger_better(),
                false,
            ),
            (
                "mae",
                &|| regression().metric_name("mae").is_larger_better(),
                false,
            ),
            (
                "r2",
                &|| regression().metric_name("r2").is_larger_better(),
                true,
            ),
            (
                "var",
                &|| regression().metric_name("var").is_larger_better(),
                true,
            ),
            (
                "areaUnderPR",
                &|| {
                    BinaryClassificationEvaluator::new()
                        .metric_name("areaUnderPR")
                        .is_larger_better()
                },
                true,
            ),
            ("f1 by default", &|| multiclass().is_larger_better(), true),
            (
                "accuracy",
                &|| multiclass().metric_name("accuracy").is_larger_better(),
                true,
            ),
            (
                "logLoss",
                &|| multiclass().metric_name("logLoss").is_larger_better(),
                false,
            ),
            (
                "hammingLoss",
                &|| multiclass().metric_name("hammingLoss").is_larger_better(),
                false,
            ),
            (
                "weightedFalsePositiveRate",
                &|| {
                    multiclass()
                        .metric_name("weightedFalsePositiveRate")
                        .is_larger_better()
                },
                false,
            ),
            (
                "falsePositiveRateByLabel",
                &|| {
                    multiclass()
                        .metric_name("falsePositiveRateByLabel")
                        .is_larger_better()
                },
                false,
            ),
        ];
        for (case, larger_is_better, expected) in cases {
            assert_eq!(larger_is_better(), expected, "{case}");
        }
    }
}
