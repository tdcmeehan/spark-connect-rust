//! The transformers and estimators of features.

use std::future::Future;

use spark_connect_core::ml::OperatorType;

use super::linalg::{strings_of, vector_of};
use super::{Estimator, Fitted, Operator, Transformer, Vector};
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};

/// Puts the values of several columns, numbers or vectors, into one vector column, such as the
/// features of an estimator.
#[derive(Debug, Clone)]
pub struct VectorAssembler(Operator);

impl VectorAssembler {
    /// An assembler of no columns yet.
    pub fn new() -> Self {
        VectorAssembler(Operator::new(
            "org.apache.spark.ml.feature.VectorAssembler",
            OperatorType::Transformer,
        ))
    }

    /// The columns to put into the vector, in its order.
    pub fn input_cols(self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        VectorAssembler(self.0.strings("inputCols", names))
    }

    /// The column of the vectors. By default the server names it after the assembler.
    pub fn output_col(self, name: impl Into<String>) -> Self {
        VectorAssembler(self.0.string("outputCol", name))
    }
}

impl Default for VectorAssembler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for VectorAssembler {
    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        self.0.transform(dataset)
    }
}

/// Scales vectors of features by the statistics of the rows it is fitted to: each feature to a
/// standard deviation of 1.
#[derive(Debug, Clone)]
pub struct StandardScaler(Operator);

impl StandardScaler {
    /// A scaler of no column yet: its input column is required.
    pub fn new() -> Self {
        StandardScaler(Operator::new(
            "org.apache.spark.ml.feature.StandardScaler",
            OperatorType::Estimator,
        ))
    }

    /// The column of the vectors to scale.
    pub fn input_col(self, name: impl Into<String>) -> Self {
        StandardScaler(self.0.string("inputCol", name))
    }

    /// The column of the scaled vectors. By default the server names it after the scaler.
    pub fn output_col(self, name: impl Into<String>) -> Self {
        StandardScaler(self.0.string("outputCol", name))
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StandardScaler {
    type Model = StandardScalerModel;

    fn fit(
        &self,
        dataset: &DataFrame,
    ) -> impl Future<Output = Result<StandardScalerModel>> + Send + 'static {
        let fit = self.0.fit(dataset);
        async move { Ok(StandardScalerModel(fit.await?)) }
    }
}

/// A [`StandardScaler`] that the server fitted.
#[derive(Debug, Clone)]
pub struct StandardScalerModel(Fitted);

impl StandardScalerModel {
    /// The mean of each feature of the rows of the fit.
    pub fn mean(&self) -> impl Future<Output = Result<Vector>> + Send + 'static {
        self.0.attribute("mean", vector_of)
    }

    /// The standard deviation of each feature of the rows of the fit.
    pub fn std(&self) -> impl Future<Output = Result<Vector>> + Send + 'static {
        self.0.attribute("std", vector_of)
    }
}

impl Transformer for StandardScalerModel {
    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        self.0.transform(dataset)
    }
}

/// Scales vectors of features by the rows it is fitted to: each feature into `[-1, 1]`, by its
/// largest absolute value.
#[derive(Debug, Clone)]
pub struct MaxAbsScaler(Operator);

impl MaxAbsScaler {
    /// A scaler of no column yet: its input column is required.
    pub fn new() -> Self {
        MaxAbsScaler(Operator::new(
            "org.apache.spark.ml.feature.MaxAbsScaler",
            OperatorType::Estimator,
        ))
    }

    /// The column of the vectors to scale.
    pub fn input_col(self, name: impl Into<String>) -> Self {
        MaxAbsScaler(self.0.string("inputCol", name))
    }

    /// The column of the scaled vectors. By default the server names it after the scaler.
    pub fn output_col(self, name: impl Into<String>) -> Self {
        MaxAbsScaler(self.0.string("outputCol", name))
    }
}

impl Default for MaxAbsScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MaxAbsScaler {
    type Model = MaxAbsScalerModel;

    fn fit(
        &self,
        dataset: &DataFrame,
    ) -> impl Future<Output = Result<MaxAbsScalerModel>> + Send + 'static {
        let fit = self.0.fit(dataset);
        async move { Ok(MaxAbsScalerModel(fit.await?)) }
    }
}

/// A [`MaxAbsScaler`] that the server fitted.
#[derive(Debug, Clone)]
pub struct MaxAbsScalerModel(Fitted);

impl MaxAbsScalerModel {
    /// The largest absolute value of each feature of the rows of the fit.
    pub fn max_abs(&self) -> impl Future<Output = Result<Vector>> + Send + 'static {
        self.0.attribute("maxAbs", vector_of)
    }
}

impl Transformer for MaxAbsScalerModel {
    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        self.0.transform(dataset)
    }
}

/// Indexes the values of a column of strings by the rows it is fitted to: the most frequent
/// value is 0, the next 1, and so on.
#[derive(Debug, Clone)]
pub struct StringIndexer(Operator);

impl StringIndexer {
    /// An indexer of no column yet: its input column is required.
    pub fn new() -> Self {
        StringIndexer(Operator::new(
            "org.apache.spark.ml.feature.StringIndexer",
            OperatorType::Estimator,
        ))
    }

    /// The column of the values to index.
    pub fn input_col(self, name: impl Into<String>) -> Self {
        StringIndexer(self.0.string("inputCol", name))
    }

    /// The column of the indices, as `DOUBLE`s. By default the server names it after the
    /// indexer.
    pub fn output_col(self, name: impl Into<String>) -> Self {
        StringIndexer(self.0.string("outputCol", name))
    }
}

impl Default for StringIndexer {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StringIndexer {
    type Model = StringIndexerModel;

    fn fit(
        &self,
        dataset: &DataFrame,
    ) -> impl Future<Output = Result<StringIndexerModel>> + Send + 'static {
        let fit = self.0.fit(dataset);
        async move { Ok(StringIndexerModel(fit.await?)) }
    }
}

/// A [`StringIndexer`] that the server fitted.
#[derive(Debug, Clone)]
pub struct StringIndexerModel(Fitted);

impl StringIndexerModel {
    /// The values of the rows of the fit, in the order of their indices.
    pub fn labels(&self) -> impl Future<Output = Result<Vec<String>>> + Send + 'static {
        self.0.attribute("labelsArray", |value| {
            let mut columns = super::linalg::arrays_of(value, strings_of)?;
            match columns.len() {
                1 => Ok(columns.remove(0)),
                count => Err(Error::decode(format!("the labels of {count} columns"))),
            }
        })
    }
}

impl Transformer for StringIndexerModel {
    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        self.0.transform(dataset)
    }
}
