//! The estimators of regressions.

use std::future::Future;

use spark_connect_core::ml::OperatorType;

use super::linalg::{count, double_of, int_of, vector_of};
use super::{Estimator, Fitted, Operator, Transformer, Vector};
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};

/// Predicts a number by a linear regression of a vector of features: its model adds the
/// prediction to each row.
#[derive(Debug, Clone)]
pub struct LinearRegression(Operator);

impl LinearRegression {
    /// A regression of the column `features` to the column `label`, with Spark's defaults.
    pub fn new() -> Self {
        LinearRegression(Operator::new(
            "org.apache.spark.ml.regression.LinearRegression",
            OperatorType::Estimator,
        ))
    }

    /// The column of the vectors of features; by default `features`.
    pub fn features_col(self, name: impl Into<String>) -> Self {
        LinearRegression(self.0.string("featuresCol", name))
    }

    /// The column of the numbers to predict; by default `label`.
    pub fn label_col(self, name: impl Into<String>) -> Self {
        LinearRegression(self.0.string("labelCol", name))
    }

    /// The column of the predictions; by default `prediction`.
    pub fn prediction_col(self, name: impl Into<String>) -> Self {
        LinearRegression(self.0.string("predictionCol", name))
    }

    /// The largest number of iterations of the optimizer; by default 100.
    pub fn max_iter(self, iterations: i32) -> Self {
        LinearRegression(self.0.int("maxIter", iterations))
    }

    /// The strength of the regularization; by default 0.
    pub fn reg_param(self, strength: f64) -> Self {
        LinearRegression(self.0.double("regParam", strength))
    }

    /// The mix of the regularization, from 0 for L2 to 1 for L1; by default 0.
    pub fn elastic_net_param(self, mix: f64) -> Self {
        LinearRegression(self.0.double("elasticNetParam", mix))
    }

    /// The tolerance of the convergence; by default 1e-6.
    pub fn tol(self, tolerance: f64) -> Self {
        LinearRegression(self.0.double("tol", tolerance))
    }

    /// Whether to fit an intercept; by default `true`.
    pub fn fit_intercept(self, fit: bool) -> Self {
        LinearRegression(self.0.bool("fitIntercept", fit))
    }

    /// Whether to standardize the features before the fit; by default `true`.
    pub fn standardization(self, standardize: bool) -> Self {
        LinearRegression(self.0.bool("standardization", standardize))
    }

    /// The solver: `auto` (the default), `normal` or `l-bfgs`.
    pub fn solver(self, solver: impl Into<String>) -> Self {
        LinearRegression(self.0.string("solver", solver))
    }

    /// The column of the weights of the rows; by default every row weighs 1.
    pub fn weight_col(self, name: impl Into<String>) -> Self {
        LinearRegression(self.0.string("weightCol", name))
    }

    /// The depth of the tree aggregation of the optimizer; by default 2.
    pub fn aggregation_depth(self, depth: i32) -> Self {
        LinearRegression(self.0.int("aggregationDepth", depth))
    }

    /// The loss to minimize: `squaredError` (the default) or `huber`.
    pub fn loss(self, loss: impl Into<String>) -> Self {
        LinearRegression(self.0.string("loss", loss))
    }

    /// The shape of the `huber` loss, above 1; by default 1.35.
    pub fn epsilon(self, epsilon: f64) -> Self {
        LinearRegression(self.0.double("epsilon", epsilon))
    }

    /// The largest size, in MB, of the blocks of rows that the optimizer stacks; by default 0,
    /// for a size that the server chooses.
    pub fn max_block_size_in_mb(self, size: f64) -> Self {
        LinearRegression(self.0.double("maxBlockSizeInMB", size))
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LinearRegression {
    type Model = LinearRegressionModel;

    fn fit(
        &self,
        dataset: &DataFrame,
    ) -> impl Future<Output = Result<LinearRegressionModel>> + Send + 'static {
        let fit = self.0.fit(dataset);
        async move { Ok(LinearRegressionModel(fit.await?)) }
    }
}

/// A [`LinearRegression`] that the server fitted.
#[derive(Debug, Clone)]
pub struct LinearRegressionModel(Fitted);

impl LinearRegressionModel {
    /// The coefficient of each feature.
    pub fn coefficients(&self) -> impl Future<Output = Result<Vector>> + Send + 'static {
        self.0.attribute("coefficients", vector_of)
    }

    /// The intercept.
    pub fn intercept(&self) -> impl Future<Output = Result<f64>> + Send + 'static {
        self.0.attribute("intercept", double_of)
    }

    /// The scale of the `huber` loss, or 1 for the squared error.
    pub fn scale(&self) -> impl Future<Output = Result<f64>> + Send + 'static {
        self.0.attribute("scale", double_of)
    }

    /// The number of features.
    pub fn num_features(&self) -> impl Future<Output = Result<usize>> + Send + 'static {
        self.0.attribute("numFeatures", |value| {
            count(int_of(value)?).map_err(Error::decode)
        })
    }
}

impl Transformer for LinearRegressionModel {
    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        self.0.transform(dataset)
    }
}
