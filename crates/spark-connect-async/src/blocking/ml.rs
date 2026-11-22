//! The blocking forms of the fits, the evaluations and the fitted models of Spark ML.

use super::wait;
use crate::error::Result;
use crate::ml::{self, Matrix, Vector};

/// The fit of an estimator of [`crate::ml`], which blocks until the server has fitted the
/// model: the blocking form of [`ml::Estimator`].
///
/// The estimators are the async API's: their params are set there, and this trait gives each a
/// [`fit`](Estimator::fit) with the name and the arguments of [`ml::Estimator::fit`]. A file
/// imports one of the two traits; where both are in scope, a call names its trait, as
/// `ml::Estimator::fit(&estimator, &dataset)` does. An estimator of another crate fits as a
/// stage of a [`ml::Pipeline`].
///
/// ```no_run
/// # fn run(session: spark_connect_async::blocking::Session) -> spark_connect_async::Result<()> {
/// use spark_connect_async::blocking::{Estimator, Evaluator};
/// use spark_connect_async::ml::{BinaryClassificationEvaluator, LogisticRegression, Transformer};
///
/// let training = session.as_async().table("training");
/// let model = LogisticRegression::new().max_iter(20).fit(&training)?;
/// println!("intercept {}", model.intercept()?);
/// let predictions = model.as_async().transform(&session.as_async().table("test"));
/// let area = BinaryClassificationEvaluator::new().evaluate(&predictions)?;
/// # Ok(())
/// # }
/// ```
pub trait Estimator {
    /// The model that a fit returns: the blocking form of the async model, for a model with
    /// attributes that the server reads.
    type Model;

    /// Fits this estimator to the rows of `dataset` on the server, as [`ml::Estimator::fit`]
    /// does, and blocks until the server has fitted the model. For a blocking `dataset`, pass
    /// its [`as_async`](super::DataFrame::as_async).
    fn fit(&self, dataset: &crate::DataFrame) -> Result<Self::Model>;
}

/// `Estimator` for estimators whose model is `$model` in its blocking form.
macro_rules! fits {
    ($($estimator:ty => $model:ty;)*) => {$(
        impl Estimator for $estimator {
            type Model = $model;

            fn fit(&self, dataset: &crate::DataFrame) -> Result<$model> {
                wait(ml::Estimator::fit(self, dataset)).map(<$model>::from)
            }
        }
    )*};
}

fits! {
    ml::LinearRegression => LinearRegressionModel;
    ml::LinearSVC => LinearSVCModel;
    ml::LogisticRegression => LogisticRegressionModel;
    ml::MaxAbsScaler => MaxAbsScalerModel;
    ml::StandardScaler => StandardScalerModel;
    ml::StringIndexer => StringIndexerModel;
    ml::Pipeline => ml::PipelineModel;
}

impl<E: ml::Estimator, V: ml::Evaluator> Estimator for ml::CrossValidator<E, V> {
    type Model = ml::CrossValidatorModel<E::Model>;

    fn fit(&self, dataset: &crate::DataFrame) -> Result<Self::Model> {
        wait(ml::Estimator::fit(self, dataset))
    }
}

/// The evaluation of an evaluator, which blocks until the server has computed the metric: the
/// blocking form of [`ml::Evaluator`], for every evaluator of the async API.
///
/// As with [`Estimator`], a file imports one of the two traits; the
/// [`is_larger_better`](ml::Evaluator::is_larger_better) of an evaluator is the async trait's.
pub trait Evaluator {
    /// The metric of the rows of `dataset`, as [`ml::Evaluator::evaluate`] computes it, which
    /// blocks until the server answers. For a blocking `dataset`, pass its
    /// [`as_async`](super::DataFrame::as_async).
    fn evaluate(&self, dataset: &crate::DataFrame) -> Result<f64>;
}

impl<V: ml::Evaluator> Evaluator for V {
    fn evaluate(&self, dataset: &crate::DataFrame) -> Result<f64> {
        wait(ml::Evaluator::evaluate(self, dataset))
    }
}

/// A model with attributes that block, holding its async model.
macro_rules! model {
    ($(#[$doc:meta])* $model:ident) => {
        $(#[$doc])*
        ///
        /// Its [`transform`](ml::Transformer::transform) builds a DataFrame and sends nothing:
        /// it is the async model's, which [`as_async`](Self::as_async) borrows. The blocking model
        /// holds the async one, so the server keeps the model until the last handle to it, async
        /// or blocking, and the last DataFrame that uses it are dropped.
        #[derive(Clone)]
        pub struct $model(ml::$model);

        impl $model {
            /// This model as the async model it holds, borrowed: its transform is called on it.
            pub fn as_async(&self) -> &ml::$model {
                &self.0
            }

            /// This model as the async model it holds, as [`From`] converts it.
            pub fn into_async(self) -> ml::$model {
                self.0
            }
        }

        impl From<ml::$model> for $model {
            fn from(model: ml::$model) -> Self {
                $model(model)
            }
        }

        impl From<$model> for ml::$model {
            fn from(model: $model) -> Self {
                model.0
            }
        }

        impl ml::$model {
            /// The blocking form of this model, the same model, whose attributes block.
            pub fn blocking(&self) -> $model {
                $model(self.clone())
            }
        }
    };
}

model! {
    /// A fitted [`ml::LogisticRegression`], whose attributes block until the server answers.
    LogisticRegressionModel
}

impl LogisticRegressionModel {
    /// The coefficient of each feature, of a model of two classes, as
    /// [`ml::LogisticRegressionModel::coefficients`] reads it.
    pub fn coefficients(&self) -> Result<Vector> {
        wait(self.0.coefficients())
    }

    /// The intercept of a model of two classes, as [`ml::LogisticRegressionModel::intercept`]
    /// reads it.
    pub fn intercept(&self) -> Result<f64> {
        wait(self.0.intercept())
    }

    /// The coefficients, a row for each class, or one row for two classes.
    pub fn coefficient_matrix(&self) -> Result<Matrix> {
        wait(self.0.coefficient_matrix())
    }

    /// The intercept of each class, or one intercept for two classes.
    pub fn intercept_vector(&self) -> Result<Vector> {
        wait(self.0.intercept_vector())
    }

    /// The number of classes.
    pub fn num_classes(&self) -> Result<usize> {
        wait(self.0.num_classes())
    }

    /// The number of features.
    pub fn num_features(&self) -> Result<usize> {
        wait(self.0.num_features())
    }
}

model! {
    /// A fitted [`ml::LinearSVC`], whose attributes block until the server answers.
    LinearSVCModel
}

impl LinearSVCModel {
    /// The coefficient of each feature.
    pub fn coefficients(&self) -> Result<Vector> {
        wait(self.0.coefficients())
    }

    /// The intercept.
    pub fn intercept(&self) -> Result<f64> {
        wait(self.0.intercept())
    }

    /// The number of classes, 2.
    pub fn num_classes(&self) -> Result<usize> {
        wait(self.0.num_classes())
    }

    /// The number of features.
    pub fn num_features(&self) -> Result<usize> {
        wait(self.0.num_features())
    }
}

model! {
    /// A fitted [`ml::LinearRegression`], whose attributes block until the server answers.
    LinearRegressionModel
}

impl LinearRegressionModel {
    /// The coefficient of each feature.
    pub fn coefficients(&self) -> Result<Vector> {
        wait(self.0.coefficients())
    }

    /// The intercept.
    pub fn intercept(&self) -> Result<f64> {
        wait(self.0.intercept())
    }

    /// The scale of the `huber` loss, or 1 for the squared error.
    pub fn scale(&self) -> Result<f64> {
        wait(self.0.scale())
    }

    /// The number of features.
    pub fn num_features(&self) -> Result<usize> {
        wait(self.0.num_features())
    }
}

model! {
    /// A fitted [`ml::StandardScaler`], whose attributes block until the server answers.
    StandardScalerModel
}

impl StandardScalerModel {
    /// The mean of each feature of the rows of the fit.
    pub fn mean(&self) -> Result<Vector> {
        wait(self.0.mean())
    }

    /// The standard deviation of each feature of the rows of the fit.
    pub fn std(&self) -> Result<Vector> {
        wait(self.0.std())
    }
}

model! {
    /// A fitted [`ml::MaxAbsScaler`], whose attribute blocks until the server answers.
    MaxAbsScalerModel
}

impl MaxAbsScalerModel {
    /// The largest absolute value of each feature of the rows of the fit.
    pub fn max_abs(&self) -> Result<Vector> {
        wait(self.0.max_abs())
    }
}

model! {
    /// A fitted [`ml::StringIndexer`], whose attribute blocks until the server answers.
    StringIndexerModel
}

impl StringIndexerModel {
    /// The values of the rows of the fit, in the order of their indices.
    pub fn labels(&self) -> Result<Vec<String>> {
        wait(self.0.labels())
    }
}
