//! Cross validation: the choice of the best of several settings of an estimator.

use std::future::Future;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use spark_connect_core::expression::ExpressionNode;
use spark_connect_core::ml::{MlOperator, OperatorType};

use super::{Estimator, Evaluator, Transformer};
use crate::column::{lit, Column};
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::functions::rand;
use crate::options::random_seed;

/// Chooses the best of several candidates, settings of one estimator, by a k-fold cross
/// validation, on the client: the rows are split into folds at random; each candidate is fitted
/// to the rows of all folds but one and its model evaluated on that one, for each fold; the
/// candidate of the best mean metric is then fitted to every row.
///
/// The folds are not cached: the server computes each split of the rows again, the same each
/// time for a seed.
#[derive(Debug)]
pub struct CrossValidator<E, V> {
    candidates: Arc<[E]>,
    evaluator: Arc<V>,
    folds: usize,
    parallelism: usize,
    seed: Option<i64>,
    id: String,
}

impl<E, V> Clone for CrossValidator<E, V> {
    fn clone(&self) -> Self {
        CrossValidator {
            candidates: Arc::clone(&self.candidates),
            evaluator: Arc::clone(&self.evaluator),
            folds: self.folds,
            parallelism: self.parallelism,
            seed: self.seed,
            id: self.id.clone(),
        }
    }
}

impl<E: Estimator, V: Evaluator> CrossValidator<E, V> {
    /// A cross validation of `candidates`, each a setting of the estimator to try, whose models
    /// `evaluator` measures, in 3 folds, one fit at a time.
    pub fn new(candidates: impl IntoIterator<Item = E>, evaluator: V) -> Self {
        CrossValidator {
            candidates: candidates.into_iter().collect(),
            evaluator: Arc::new(evaluator),
            folds: 3,
            parallelism: 1,
            seed: None,
            id: MlOperator::new("CrossValidator", OperatorType::Estimator).uid,
        }
    }

    /// The number of folds, at least 2.
    pub fn num_folds(mut self, folds: usize) -> Self {
        self.folds = folds;
        self
    }

    /// How many fits and evaluations of a fold run at a time, at least 1.
    pub fn parallelism(mut self, fits: usize) -> Self {
        self.parallelism = fits;
        self
    }

    /// The seed of the split into folds. Without one, each fit draws a seed.
    pub fn seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn checked(&self) -> Result<()> {
        if self.candidates.is_empty() {
            return Err(Error::invalid_argument(
                "a cross validation needs a candidate",
            ));
        }
        if self.folds < 2 {
            return Err(Error::invalid_argument(format!(
                "a cross validation needs 2 folds or more, not {}",
                self.folds
            )));
        }
        if self.parallelism == 0 {
            return Err(Error::invalid_argument(
                "a cross validation needs a parallelism of 1 or more",
            ));
        }
        Ok(())
    }
}

impl<E: Estimator, V: Evaluator> Estimator for CrossValidator<E, V> {
    type Model = CrossValidatorModel<E::Model>;

    fn fit(
        &self,
        dataset: &DataFrame,
    ) -> impl Future<Output = Result<CrossValidatorModel<E::Model>>> + Send + 'static {
        let checked = self.checked();
        let validation = self.clone();
        let dataset = dataset.clone();
        async move {
            checked?;
            let seed = validation.seed.unwrap_or_else(random_seed);
            let metrics = validation.fold_metrics(&dataset, seed).await?;
            let (avg_metrics, std_metrics) = mean_and_deviation(&metrics);
            let best = best(&avg_metrics, validation.evaluator.is_larger_better());
            let best_model = validation.candidates[best].fit(&dataset).await?;
            Ok(CrossValidatorModel {
                best_model,
                avg_metrics,
                std_metrics,
            })
        }
    }
}

impl<E: Estimator, V: Evaluator> CrossValidator<E, V> {
    /// The metric of each candidate on each fold, by fold.
    async fn fold_metrics(&self, dataset: &DataFrame, seed: i64) -> Result<Vec<Vec<f64>>> {
        let fold_col = format!("{}_rand", self.id);
        let star = Column::new(ExpressionNode::UnresolvedStar(None, None).into());
        let split = dataset.select([star, rand(Some(seed)).alias(fold_col.as_str())]);
        let width = 1.0 / self.folds as f64;
        let mut metrics = Vec::with_capacity(self.folds);
        for fold in 0..self.folds {
            let in_fold = split
                .col(&fold_col)
                .ge(lit(fold as f64 * width))
                .and(split.col(&fold_col).lt(lit((fold + 1) as f64 * width)));
            let validation = split.filter(in_fold.clone());
            let training = split.filter(!in_fold);
            let mut runs = Vec::with_capacity(self.candidates.len());
            for (index, candidate) in self.candidates.iter().enumerate() {
                let fit = candidate.fit(&training);
                let validation = validation.clone();
                let evaluator = Arc::clone(&self.evaluator);
                runs.push(async move {
                    let model = fit.await?;
                    let metric = evaluator.evaluate(&model.transform(&validation)).await?;
                    Ok::<_, Error>((index, metric))
                });
            }
            let mut fold_metrics = vec![0.0; self.candidates.len()];
            let mut measured = futures::stream::iter(runs).buffer_unordered(self.parallelism);
            while let Some((index, metric)) = measured.try_next().await? {
                fold_metrics[index] = metric;
            }
            metrics.push(fold_metrics);
        }
        Ok(metrics)
    }
}

/// The mean and the standard deviation, over the folds, of the metrics of each candidate.
fn mean_and_deviation(metrics: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let folds = metrics.len() as f64;
    let candidates = metrics.first().map_or(0, Vec::len);
    (0..candidates)
        .map(|candidate| {
            let mean = metrics.iter().map(|fold| fold[candidate]).sum::<f64>() / folds;
            let variance = metrics
                .iter()
                .map(|fold| (fold[candidate] - mean).powi(2))
                .sum::<f64>()
                / folds;
            (mean, variance.sqrt())
        })
        .unzip()
}

/// The first candidate of the largest mean metric, or of the smallest one.
fn best(means: &[f64], larger_is_better: bool) -> usize {
    let mut best = 0;
    for (index, &mean) in means.iter().enumerate() {
        let better = if larger_is_better {
            mean > means[best]
        } else {
            mean < means[best]
        };
        if better {
            best = index;
        }
    }
    best
}

/// A [`CrossValidator`] that was fitted: the model of the best candidate, fitted to every row,
/// and the metrics of the candidates.
#[derive(Debug, Clone)]
pub struct CrossValidatorModel<M> {
    best_model: M,
    avg_metrics: Vec<f64>,
    std_metrics: Vec<f64>,
}

impl<M> CrossValidatorModel<M> {
    /// The model of the best candidate, fitted to every row.
    pub fn best_model(&self) -> &M {
        &self.best_model
    }

    /// The mean metric of each candidate over the folds, in the order of the candidates.
    pub fn avg_metrics(&self) -> &[f64] {
        &self.avg_metrics
    }

    /// The standard deviation of the metric of each candidate over the folds.
    pub fn std_metrics(&self) -> &[f64] {
        &self.std_metrics
    }
}

impl<M: Transformer> Transformer for CrossValidatorModel<M> {
    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        self.best_model.transform(dataset)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;

    use super::super::mock::{self, plan_of, Answers, Fits, Log, Marks};
    use super::*;
    use crate::ErrorKind;

    fn answers(log: &Log, metrics: &[f64], larger_is_better: bool) -> Answers {
        Answers {
            log: log.clone(),
            metrics: Arc::new(Mutex::new(VecDeque::from(metrics.to_vec()))),
            larger_is_better,
        }
    }

    #[tokio::test]
    async fn each_candidate_is_fitted_and_evaluated_on_each_fold_and_the_best_mean_refitted() {
        let session = mock::session().await;
        let data = session.range(10);
        let cases = [
            (
                "larger is better",
                [0.25, 0.75, 0.5, 1.0],
                true,
                "c1",
                [0.375, 0.875],
            ),
            (
                "smaller is better",
                [0.25, 0.75, 0.5, 1.0],
                false,
                "c0",
                [0.375, 0.875],
            ),
            ("a tie", [0.5, 0.25, 0.5, 0.75], true, "c0", [0.5, 0.5]),
        ];
        for (case, metrics, larger_is_better, best, means) in cases {
            let log = Log::default();
            let candidates = [Fits::new("c0", &log), Fits::new("c1", &log)];
            let validation =
                CrossValidator::new(candidates, answers(&log, &metrics, larger_is_better))
                    .num_folds(2)
                    .seed(7);
            let model = validation.fit(&data).await.unwrap();
            assert_eq!(model.best_model(), &Marks(best), "{case}");
            assert_eq!(model.avg_metrics(), means, "{case}");
            if case != "larger is better" {
                continue;
            }
            assert_eq!(model.std_metrics(), [0.125, 0.125]);
            let fold_col = format!("{}_rand", validation.id);
            let star = Column::new(ExpressionNode::UnresolvedStar(None, None).into());
            let split = data.select([star, rand(Some(7)).alias(fold_col.as_str())]);
            let in_fold = |lower: f64, upper: f64| {
                split
                    .col(&fold_col)
                    .ge(lit(lower))
                    .and(split.col(&fold_col).lt(lit(upper)))
            };
            let mut expected = Vec::new();
            for (lower, upper) in [(0.0, 0.5), (0.5, 1.0)] {
                let training = split.filter(!in_fold(lower, upper));
                let held_out = split.filter(in_fold(lower, upper));
                for candidate in ["c0", "c1"] {
                    expected.push(format!("fit {candidate} on {}", plan_of(&training)));
                    let predicted = Marks(candidate).transform(&held_out);
                    expected.push(format!("evaluate {}", plan_of(&predicted)));
                }
            }
            expected.push(format!("fit c1 on {}", plan_of(&data)));
            assert_eq!(log.take(), expected);
        }
    }

    #[tokio::test]
    async fn the_fits_of_a_fold_run_as_many_at_a_time_as_its_parallelism() {
        let session = mock::session().await;
        for (parallelism, most) in [(1, 1), (2, 2), (5, 3)] {
            let log = Log::default();
            let fits = Fits::new("c", &log);
            let validation = CrossValidator::new(
                [fits.clone(), fits.clone(), fits.clone()],
                answers(&log, &[0.0; 6], true),
            )
            .num_folds(2)
            .parallelism(parallelism);
            validation.fit(&session.range(10)).await.unwrap();
            assert_eq!(fits.most.load(Ordering::SeqCst), most, "{parallelism}");
        }
    }

    #[tokio::test]
    async fn a_cross_validation_without_a_seed_draws_one_for_each_fit() {
        let session = mock::session().await;
        let data = session.range(10);
        for (case, seed, same) in [("no seed", None, false), ("a seed", Some(3), true)] {
            let log = Log::default();
            let mut validation =
                CrossValidator::new([Fits::new("c", &log)], answers(&log, &[0.0; 6], true));
            if let Some(seed) = seed {
                validation = validation.seed(seed);
            }
            validation.fit(&data).await.unwrap();
            let first = log.take();
            validation.fit(&data).await.unwrap();
            assert_eq!(log.take() == first, same, "{case}");
        }
    }

    #[tokio::test]
    async fn invalid_settings_are_rejected_before_anything_runs() {
        let session = mock::session().await;
        let log = Log::default();
        let one = || [Fits::new("c", &log)];
        let cases = [
            (
                "no candidate",
                CrossValidator::new([], answers(&log, &[], true)),
            ),
            (
                "one fold",
                CrossValidator::new(one(), answers(&log, &[], true)).num_folds(1),
            ),
            (
                "no parallelism",
                CrossValidator::new(one(), answers(&log, &[], true)).parallelism(0),
            ),
        ];
        for (case, validation) in cases {
            let error = validation.fit(&session.range(3)).await.unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{case}: {error}");
        }
        assert!(log.take().is_empty());
    }
}
