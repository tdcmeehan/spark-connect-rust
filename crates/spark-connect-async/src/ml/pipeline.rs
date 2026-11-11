//! Pipelines: stages that fit and transform in order.

use std::any::Any;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use futures::future::BoxFuture;

use super::{
    CrossValidator, CrossValidatorModel, Estimator, Evaluator, LinearRegression,
    LinearRegressionModel, LinearSVC, LinearSVCModel, LogisticRegression, LogisticRegressionModel,
    MaxAbsScaler, MaxAbsScalerModel, StandardScaler, StandardScalerModel, StringIndexer,
    StringIndexerModel, Transformer, VectorAssembler,
};
use crate::dataframe::DataFrame;
use crate::error::Result;

/// A stage of a [`Pipeline`]: an estimator, which the pipeline fits to the rows that the stages
/// before it transformed, or a transformer, which it applies.
///
/// The operators of this module convert into stages; another estimator or transformer becomes
/// one with [`PipelineStage::estimator`] or [`PipelineStage::transformer`].
#[derive(Clone)]
pub struct PipelineStage(Stage);

#[derive(Clone)]
enum Stage {
    Estimator(Arc<dyn FitStage>),
    Transformer(Arc<dyn Transformer>),
}

/// An estimator whose model is a transformer of any type.
trait FitStage: Send + Sync {
    fn fit_stage(&self, dataset: &DataFrame) -> BoxFuture<'static, Result<Arc<dyn Transformer>>>;
}

impl<E: Estimator> FitStage for E {
    fn fit_stage(&self, dataset: &DataFrame) -> BoxFuture<'static, Result<Arc<dyn Transformer>>> {
        let fit = self.fit(dataset);
        Box::pin(async move { Ok(Arc::new(fit.await?) as Arc<dyn Transformer>) })
    }
}

impl PipelineStage {
    /// The stage that fits `estimator`.
    pub fn estimator(estimator: impl Estimator) -> Self {
        PipelineStage(Stage::Estimator(Arc::new(estimator)))
    }

    /// The stage that applies `transformer`.
    pub fn transformer(transformer: impl Transformer) -> Self {
        PipelineStage(Stage::Transformer(Arc::new(transformer)))
    }
}

impl fmt::Debug for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self.0 {
            Stage::Estimator(_) => "PipelineStage::Estimator",
            Stage::Transformer(_) => "PipelineStage::Transformer",
        })
    }
}

macro_rules! stages_of {
    ($kind:ident: $($operator:ty),* $(,)?) => {
        $(
            impl From<$operator> for PipelineStage {
                fn from(stage: $operator) -> Self {
                    PipelineStage::$kind(stage)
                }
            }
        )*
    };
}

stages_of!(estimator: StandardScaler, MaxAbsScaler, StringIndexer, LogisticRegression, LinearSVC,
    LinearRegression, Pipeline);
stages_of!(transformer: VectorAssembler, StandardScalerModel, MaxAbsScalerModel,
    StringIndexerModel, LogisticRegressionModel, LinearSVCModel, LinearRegressionModel,
    PipelineModel);

impl<E: Estimator, V: Evaluator> From<CrossValidator<E, V>> for PipelineStage {
    fn from(stage: CrossValidator<E, V>) -> Self {
        PipelineStage::estimator(stage)
    }
}

impl<M: Transformer + Clone> From<CrossValidatorModel<M>> for PipelineStage {
    fn from(stage: CrossValidatorModel<M>) -> Self {
        PipelineStage::transformer(stage)
    }
}

/// Stages that a fit runs in order, on the client: it fits each estimator to the rows that the
/// stages before it transformed, and its model applies the transformers and the fitted models in
/// order.
#[derive(Clone, Debug, Default)]
pub struct Pipeline {
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    /// A pipeline of no stages, whose model returns its input.
    pub fn new() -> Self {
        Pipeline::default()
    }

    /// The pipeline with `stage` after its stages.
    pub fn stage(mut self, stage: impl Into<PipelineStage>) -> Self {
        self.stages.push(stage.into());
        self
    }
}

impl Estimator for Pipeline {
    type Model = PipelineModel;

    /// Fits the estimators in order. The rows go through the stages before the last estimator:
    /// those after it are only applied by the model.
    fn fit(
        &self,
        dataset: &DataFrame,
    ) -> impl Future<Output = Result<PipelineModel>> + Send + 'static {
        let stages = self.stages.clone();
        let mut dataset = dataset.clone();
        async move {
            let last = stages
                .iter()
                .rposition(|stage| matches!(stage.0, Stage::Estimator(_)));
            let mut fitted = Vec::with_capacity(stages.len());
            for (index, stage) in stages.into_iter().enumerate() {
                let transformer = match stage.0 {
                    Stage::Estimator(estimator) => estimator.fit_stage(&dataset).await?,
                    Stage::Transformer(transformer) => transformer,
                };
                if last.is_some_and(|last| index < last) {
                    dataset = transformer.transform(&dataset);
                }
                fitted.push(transformer);
            }
            Ok(PipelineModel {
                stages: fitted.into(),
            })
        }
    }
}

/// A [`Pipeline`] that was fitted: the transformers and the fitted models of its stages, in
/// order.
#[derive(Clone)]
pub struct PipelineModel {
    stages: Arc<[Arc<dyn Transformer>]>,
}

impl PipelineModel {
    /// The stages: the fitted model of each estimator, and the transformers.
    pub fn stages(&self) -> &[Arc<dyn Transformer>] {
        &self.stages
    }

    /// The stage `index` as a `T`, if it is one, such as the
    /// [`LogisticRegressionModel`] of a [`LogisticRegression`] stage.
    pub fn stage<T: Transformer>(&self, index: usize) -> Option<&T> {
        let stage: &dyn Any = self.stages.get(index)?.as_ref();
        stage.downcast_ref()
    }
}

impl fmt::Debug for PipelineModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineModel")
            .field("stages", &self.stages.len())
            .finish()
    }
}

impl Transformer for PipelineModel {
    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        self.stages
            .iter()
            .fold(dataset.clone(), |dataset, stage| stage.transform(&dataset))
    }
}

#[cfg(test)]
mod tests {
    use super::super::mock::{self, plan_of, Fits, Log, Marks};
    use super::*;

    #[tokio::test]
    async fn a_pipeline_fits_each_estimator_to_the_rows_that_the_stages_before_it_transformed() {
        let session = mock::session().await;
        let data = session.range(3);
        let marked = |names: &[&'static str]| {
            names.iter().fold(data.clone(), |dataset, &name| {
                Marks(name).transform(&dataset)
            })
        };
        let log = Log::default();
        let model = Pipeline::new()
            .stage(PipelineStage::transformer(Marks("t1")))
            .stage(PipelineStage::estimator(Fits::new("e1", &log)))
            .stage(PipelineStage::transformer(Marks("t2")))
            .stage(PipelineStage::estimator(Fits::new("e2", &log)))
            .stage(PipelineStage::transformer(Marks("t3")))
            .fit(&data)
            .await
            .unwrap();
        assert_eq!(
            log.take(),
            [
                format!("fit e1 on {}", plan_of(&marked(&["t1"]))),
                format!("fit e2 on {}", plan_of(&marked(&["t1", "e1", "t2"]))),
            ]
        );
        assert_eq!(
            plan_of(&model.transform(&data)),
            plan_of(&marked(&["t1", "e1", "t2", "e2", "t3"]))
        );
        assert_eq!(model.stages().len(), 5);
        assert_eq!(model.stage::<Marks>(3), Some(&Marks("e2")));
        assert!(model.stage::<VectorAssembler>(3).is_none());
        assert!(model.stage::<Marks>(5).is_none());

        let transformers = Pipeline::new()
            .stage(PipelineStage::transformer(Marks("t")))
            .fit(&data)
            .await
            .unwrap();
        assert!(log.take().is_empty(), "a pipeline of no estimator fitted");
        assert_eq!(
            plan_of(&transformers.transform(&data)),
            plan_of(&marked(&["t"]))
        );
    }
}
