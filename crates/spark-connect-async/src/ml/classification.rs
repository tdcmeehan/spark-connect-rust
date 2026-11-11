//! The estimators of classifications.

use std::future::Future;

use spark_connect_core::ml::OperatorType;

use super::linalg::{count, double_of, int_of, matrix_of, vector_of};
use super::{Estimator, Fitted, Matrix, Operator, Transformer, Vector};
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};

/// A count of a model, which Spark sends as an `INT`.
fn count_of(value: &spark_connect_proto::expression::Literal) -> Result<usize> {
    count(int_of(value)?).map_err(Error::decode)
}

/// Classifies the rows by a logistic regression of a vector of features, into two classes or
/// more: its model adds a prediction, the probability of each class and the raw prediction to
/// each row.
#[derive(Debug, Clone)]
pub struct LogisticRegression(Operator);

impl LogisticRegression {
    /// A logistic regression of the column `features` to the column `label`, with Spark's
    /// defaults.
    pub fn new() -> Self {
        LogisticRegression(Operator::new(
            "org.apache.spark.ml.classification.LogisticRegression",
            OperatorType::Estimator,
        ))
    }

    /// The column of the vectors of features; by default `features`.
    pub fn features_col(self, name: impl Into<String>) -> Self {
        LogisticRegression(self.0.string("featuresCol", name))
    }

    /// The column of the classes, numbers from 0; by default `label`.
    pub fn label_col(self, name: impl Into<String>) -> Self {
        LogisticRegression(self.0.string("labelCol", name))
    }

    /// The column of the predicted classes; by default `prediction`.
    pub fn prediction_col(self, name: impl Into<String>) -> Self {
        LogisticRegression(self.0.string("predictionCol", name))
    }

    /// The largest number of iterations of the optimizer; by default 100.
    pub fn max_iter(self, iterations: i32) -> Self {
        LogisticRegression(self.0.long("maxIter", iterations))
    }
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LogisticRegression {
    type Model = LogisticRegressionModel;

    fn fit(
        &self,
        dataset: &DataFrame,
    ) -> impl Future<Output = Result<LogisticRegressionModel>> + Send + 'static {
        let fit = self.0.fit(dataset);
        async move { Ok(LogisticRegressionModel(fit.await?)) }
    }
}

/// A [`LogisticRegression`] that the server fitted.
#[derive(Debug, Clone)]
pub struct LogisticRegressionModel(Fitted);

impl LogisticRegressionModel {
    /// The coefficient of each feature, of a model of two classes. The server fails for more
    /// classes, which have [`coefficient_matrix`](Self::coefficient_matrix).
    pub fn coefficients(&self) -> impl Future<Output = Result<Vector>> + Send + 'static {
        self.0.attribute("coefficients", vector_of)
    }

    /// The intercept of a model of two classes. The server fails for more classes, which have
    /// [`intercept_vector`](Self::intercept_vector).
    pub fn intercept(&self) -> impl Future<Output = Result<f64>> + Send + 'static {
        self.0.attribute("intercept", double_of)
    }

    /// The coefficients, a row for each class, or one row for two classes.
    pub fn coefficient_matrix(&self) -> impl Future<Output = Result<Matrix>> + Send + 'static {
        self.0.attribute("coefficientMatrix", matrix_of)
    }

    /// The intercept of each class, or one intercept for two classes.
    pub fn intercept_vector(&self) -> impl Future<Output = Result<Vector>> + Send + 'static {
        self.0.attribute("interceptVector", vector_of)
    }

    /// The number of classes.
    pub fn num_classes(&self) -> impl Future<Output = Result<usize>> + Send + 'static {
        self.0.attribute("numClasses", count_of)
    }

    /// The number of features.
    pub fn num_features(&self) -> impl Future<Output = Result<usize>> + Send + 'static {
        self.0.attribute("numFeatures", count_of)
    }
}

impl Transformer for LogisticRegressionModel {
    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        self.0.transform(dataset)
    }
}

/// Classifies the rows into two classes, 0 and 1, by a linear support vector machine of a
/// vector of features: its model adds a prediction and the raw prediction, the margin, to each
/// row.
#[derive(Debug, Clone)]
pub struct LinearSVC(Operator);

impl LinearSVC {
    /// A classifier of the column `features` to the column `label`, with Spark's defaults.
    pub fn new() -> Self {
        LinearSVC(Operator::new(
            "org.apache.spark.ml.classification.LinearSVC",
            OperatorType::Estimator,
        ))
    }

    /// The column of the vectors of features; by default `features`.
    pub fn features_col(self, name: impl Into<String>) -> Self {
        LinearSVC(self.0.string("featuresCol", name))
    }

    /// The column of the classes, 0 or 1; by default `label`.
    pub fn label_col(self, name: impl Into<String>) -> Self {
        LinearSVC(self.0.string("labelCol", name))
    }

    /// The column of the predicted classes; by default `prediction`.
    pub fn prediction_col(self, name: impl Into<String>) -> Self {
        LinearSVC(self.0.string("predictionCol", name))
    }

    /// The column of the margins; by default `rawPrediction`.
    pub fn raw_prediction_col(self, name: impl Into<String>) -> Self {
        LinearSVC(self.0.string("rawPredictionCol", name))
    }

    /// The largest number of iterations of the optimizer; by default 100.
    pub fn max_iter(self, iterations: i32) -> Self {
        LinearSVC(self.0.long("maxIter", iterations))
    }

    /// The strength of the L2 regularization; by default 0.
    pub fn reg_param(self, strength: f64) -> Self {
        LinearSVC(self.0.double("regParam", strength))
    }

    /// The tolerance of the convergence; by default 1e-6.
    pub fn tol(self, tolerance: f64) -> Self {
        LinearSVC(self.0.double("tol", tolerance))
    }

    /// Whether to fit an intercept; by default `true`.
    pub fn fit_intercept(self, fit: bool) -> Self {
        LinearSVC(self.0.bool("fitIntercept", fit))
    }

    /// Whether to standardize the features before the fit; by default `true`.
    pub fn standardization(self, standardize: bool) -> Self {
        LinearSVC(self.0.bool("standardization", standardize))
    }

    /// The margin above which a row is of class 1; by default 0.
    pub fn threshold(self, threshold: f64) -> Self {
        LinearSVC(self.0.double("threshold", threshold))
    }

    /// The column of the weights of the rows; by default every row weighs 1.
    pub fn weight_col(self, name: impl Into<String>) -> Self {
        LinearSVC(self.0.string("weightCol", name))
    }

    /// The depth of the tree aggregation of the optimizer; by default 2.
    pub fn aggregation_depth(self, depth: i32) -> Self {
        LinearSVC(self.0.long("aggregationDepth", depth))
    }

    /// The largest size, in MB, of the blocks of rows that the optimizer stacks; by default 0,
    /// for a size that the server chooses.
    pub fn max_block_size_in_mb(self, size: f64) -> Self {
        LinearSVC(self.0.double("maxBlockSizeInMB", size))
    }
}

impl Default for LinearSVC {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LinearSVC {
    type Model = LinearSVCModel;

    fn fit(
        &self,
        dataset: &DataFrame,
    ) -> impl Future<Output = Result<LinearSVCModel>> + Send + 'static {
        let fit = self.0.fit(dataset);
        async move { Ok(LinearSVCModel(fit.await?)) }
    }
}

/// A [`LinearSVC`] that the server fitted.
#[derive(Debug, Clone)]
pub struct LinearSVCModel(Fitted);

impl LinearSVCModel {
    /// The coefficient of each feature.
    pub fn coefficients(&self) -> impl Future<Output = Result<Vector>> + Send + 'static {
        self.0.attribute("coefficients", vector_of)
    }

    /// The intercept.
    pub fn intercept(&self) -> impl Future<Output = Result<f64>> + Send + 'static {
        self.0.attribute("intercept", double_of)
    }

    /// The number of classes, 2.
    pub fn num_classes(&self) -> impl Future<Output = Result<usize>> + Send + 'static {
        self.0.attribute("numClasses", count_of)
    }

    /// The number of features.
    pub fn num_features(&self) -> impl Future<Output = Result<usize>> + Send + 'static {
        self.0.attribute("numFeatures", count_of)
    }
}

impl Transformer for LinearSVCModel {
    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        self.0.transform(dataset)
    }
}
