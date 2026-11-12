//! Machine learning with Spark ML: estimators that the server fits into models, transformers
//! that add columns to DataFrames, evaluators that measure predictions, and the pipelines and
//! cross validations that combine them.
//!
//! A fit and an evaluation run on the server when they are awaited; a transform only builds the
//! plan of a DataFrame, which runs at its actions. The server keeps each fitted model for its
//! session until the last handle to it is dropped: the model, its clones, and the DataFrames and
//! calls that use it.
//!
//! ```no_run
//! # async fn run(session: spark_connect_async::Session) -> spark_connect_async::Result<()> {
//! use spark_connect_async::ml::{
//!     BinaryClassificationEvaluator, Estimator, Evaluator, LogisticRegression, Pipeline,
//!     Transformer, Vector, VectorAssembler,
//! };
//!
//! let training = session.table("training");
//! let pipeline = Pipeline::new()
//!     .stage(VectorAssembler::new().input_cols(["age", "income"]).output_col("features"))
//!     .stage(LogisticRegression::new().max_iter(20));
//! let model = pipeline.fit(&training).await?;
//! let predictions = model.transform(&session.table("test"));
//! let probabilities: Vec<(f64, Vector)> = predictions
//!     .select(["prediction", "probability"])
//!     .collect_as()
//!     .await?;
//! let area = BinaryClassificationEvaluator::new().evaluate(&predictions).await?;
//! # Ok(())
//! # }
//! ```

mod classification;
mod evaluation;
mod feature;
mod linalg;
mod pipeline;
mod regression;
mod tuning;

use std::any::Any;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use spark_connect_core::actions::ml::{fetched, fitted, metric};
use spark_connect_core::ml::{self as core_ml, MlOperator, OperatorType, Params};
use spark_connect_core::plan::{LogicalPlan, RelationOwner};
use spark_connect_proto as proto;

use crate::column::IntoLiteral;
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::session::Session;
use crate::stream;

pub use classification::{LinearSVC, LinearSVCModel, LogisticRegression, LogisticRegressionModel};
pub use evaluation::{
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator,
};
pub use feature::{
    MaxAbsScaler, MaxAbsScalerModel, StandardScaler, StandardScalerModel, StringIndexer,
    StringIndexerModel, VectorAssembler,
};
pub use linalg::{Matrix, Vector};
pub use pipeline::{Pipeline, PipelineModel, PipelineStage};
pub use regression::{LinearRegression, LinearRegressionModel};
pub use tuning::{CrossValidator, CrossValidatorModel};

/// What transforms a DataFrame into another, with more columns: a fitted model, which adds its
/// predictions, or a transformer such as [`VectorAssembler`].
pub trait Transformer: Any + Send + Sync {
    /// The DataFrame of the rows of `dataset` with the columns that this transformer adds. It
    /// only builds the plan, which the server checks when it runs it: a column that does not
    /// exist, or a model of another session, fails at the DataFrame's action.
    fn transform(&self, dataset: &DataFrame) -> DataFrame;
}

/// What the server fits to a DataFrame into a model.
pub trait Estimator: Send + Sync + 'static {
    /// The model that a fit returns.
    type Model: Transformer + Clone;

    /// Fits this estimator to the rows of `dataset` on the server. Dropping the future before the
    /// server accepts the fit interrupts it, and dropping it after releases it, which stops it
    /// too.
    fn fit(
        &self,
        dataset: &DataFrame,
    ) -> impl Future<Output = Result<Self::Model>> + Send + 'static;
}

/// What measures the predictions of a DataFrame.
pub trait Evaluator: Send + Sync + 'static {
    /// The metric of the rows of `dataset`, which the server computes. Dropping the future stops
    /// the evaluation, as dropping a fit does.
    fn evaluate(&self, dataset: &DataFrame) -> impl Future<Output = Result<f64>> + Send + 'static;

    /// Whether a larger metric is better, as for an accuracy, rather than a smaller one, as for
    /// an error.
    fn is_larger_better(&self) -> bool;
}

/// An operator of Spark ML, as the server knows it: its class, an id of this instance, and the
/// params set on it.
#[derive(Debug, Clone)]
struct Operator {
    operator: MlOperator,
    params: Params,
}

impl Operator {
    fn new(class: &str, kind: OperatorType) -> Self {
        Operator {
            operator: MlOperator::new(class, kind),
            params: Params::new(),
        }
    }

    fn string(mut self, name: &str, value: impl Into<String>) -> Self {
        self.params = self.params.set_param_string(name, &value.into());
        self
    }

    fn strings(mut self, name: &str, values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let values: Vec<String> = values.into_iter().map(Into::into).collect();
        self.params
            .params
            .insert(name.to_string(), values.into_literal().to_proto());
        self
    }

    /// The integer param `name`, sent as a `BIGINT`, as `apache-spark-connect` sends it; the
    /// server narrows it to the `INT` of the param.
    fn long(mut self, name: &str, value: i32) -> Self {
        self.params = self.params.set_param_int(name, i64::from(value));
        self
    }

    fn int(mut self, name: &str, value: i32) -> Self {
        self.params = self.params.set_param_int32(name, value);
        self
    }

    fn double(mut self, name: &str, value: f64) -> Self {
        self.params = self.params.set_param_double(name, value);
        self
    }

    fn bool(mut self, name: &str, value: bool) -> Self {
        self.params = self.params.set_param_bool(name, value);
        self
    }

    /// Fits this estimator to `dataset`, and returns the model that the server keeps.
    fn fit(&self, dataset: &DataFrame) -> impl Future<Output = Result<Fitted>> + Send + 'static {
        let command = core_ml::fit(&self.operator, &self.params, dataset.to_relation());
        let session = dataset.session().clone();
        let params = self.params.clone();
        dataset.holding(async move {
            let info = stream::command_result(session.clone(), command, fitted).await?;
            match info.and_then(|info| info.r#type) {
                Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(reference))
                    if !reference.id.is_empty() =>
                {
                    Ok(Fitted::new(reference, session, params))
                }
                _ => Err(Error::decode("the server returned no model")),
            }
        })
    }

    /// The metric of this evaluator of `dataset`.
    fn evaluate(&self, dataset: &DataFrame) -> impl Future<Output = Result<f64>> + Send + 'static {
        let command = core_ml::evaluate(&self.operator, &self.params, dataset.to_relation());
        let session = dataset.session().clone();
        dataset.holding(async move {
            stream::command_result(session, command, metric)
                .await?
                .ok_or_else(|| Error::decode("the server returned no metric"))
        })
    }

    /// `dataset` transformed by this transformer.
    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        let operator =
            proto::ml_relation::transform::Operator::Transformer(self.operator.to_proto());
        transformed(dataset, operator, &self.params, None)
    }

    /// The text of the param `name`, if it is set.
    fn text(&self, name: &str) -> Option<&str> {
        match self.params.get_param(name)?.expr_type.as_ref()? {
            proto::expression::ExprType::Literal(proto::expression::Literal {
                literal_type: Some(proto::expression::literal::LiteralType::String(text)),
                ..
            }) => Some(text),
            _ => None,
        }
    }
}

/// The DataFrame of `dataset` transformed by `operator` with `params`, whose plan keeps `owner`.
fn transformed(
    dataset: &DataFrame,
    operator: proto::ml_relation::transform::Operator,
    params: &Params,
    owner: Option<RelationOwner>,
) -> DataFrame {
    let relation = proto::Relation {
        common: None,
        rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
            core_ml::transform(operator, params, dataset.to_relation()),
        ))),
    };
    dataset.with_plan(LogicalPlan::MlTransform {
        ml_relation: relation,
        input: Arc::clone(dataset.plan()),
        owner,
    })
}

/// A model that the server fitted and keeps for the session of its fit, with the params of its
/// estimator, which its transforms send. Its clones share the model, which the last handle to
/// drop deletes.
#[derive(Clone)]
struct Fitted {
    reference: proto::ObjectRef,
    kept: Arc<stream::RemoveOnDrop>,
    params: Params,
}

impl fmt::Debug for Fitted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fitted")
            .field("id", &self.reference.id)
            .finish_non_exhaustive()
    }
}

impl Fitted {
    fn new(reference: proto::ObjectRef, session: Session, params: Params) -> Self {
        let delete = core_ml::delete(vec![reference.clone()]);
        Fitted {
            reference,
            kept: Arc::new(stream::RemoveOnDrop::new(session, delete)),
            params,
        }
    }

    fn transform(&self, dataset: &DataFrame) -> DataFrame {
        let operator = proto::ml_relation::transform::Operator::ObjRef(self.reference.clone());
        let owner = RelationOwner(Arc::clone(&self.kept) as Arc<dyn Any + Send + Sync>);
        transformed(dataset, operator, &self.params, Some(owner))
    }

    /// The attribute `name` of the model, read by `read`. The future keeps the model.
    fn attribute<T: Send + 'static>(
        &self,
        name: &'static str,
        read: fn(&proto::expression::Literal) -> Result<T>,
    ) -> impl Future<Output = Result<T>> + Send + 'static {
        let command = core_ml::fetch(self.reference.clone(), name);
        let session = self.kept.session().clone();
        let kept = Arc::clone(&self.kept);
        async move {
            let value = stream::command_result(session, command, fetched).await;
            drop(kept);
            let value = value?.ok_or_else(|| Error::decode("the server returned no value"));
            value
                .and_then(|value| read(&value))
                .map_err(|error| error.in_field(name))
        }
    }
}

/// Operators that run on the client, for the tests of what combines operators.
#[cfg(test)]
pub(crate) mod mock {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::*;

    /// What the operators did, in order.
    #[derive(Clone, Default)]
    pub(crate) struct Log(Arc<Mutex<Vec<String>>>);

    impl Log {
        fn push(&self, entry: String) {
            self.0.lock().unwrap().push(entry);
        }

        pub(crate) fn take(&self) -> Vec<String> {
            std::mem::take(&mut self.0.lock().unwrap())
        }
    }

    /// The relation of `dataset` without its plan ids, which differ between plans built apart.
    pub(crate) fn plan_of(dataset: &DataFrame) -> String {
        let text = format!("{:?}", dataset.to_relation());
        let mut out = String::with_capacity(text.len());
        let mut rest = text.as_str();
        while let Some(at) = rest.find("plan_id: Some(") {
            let (before, after) = rest.split_at(at + "plan_id: Some(".len());
            out.push_str(before);
            rest = after.trim_start_matches(|c: char| c.is_ascii_digit());
        }
        out.push_str(rest);
        out
    }

    /// An estimator whose fit logs its dataset, and counts the fits that run at once.
    #[derive(Clone)]
    pub(crate) struct Fits {
        pub(crate) name: &'static str,
        pub(crate) log: Log,
        pub(crate) running: Arc<AtomicUsize>,
        pub(crate) most: Arc<AtomicUsize>,
    }

    impl Fits {
        pub(crate) fn new(name: &'static str, log: &Log) -> Self {
            Fits {
                name,
                log: log.clone(),
                running: Arc::default(),
                most: Arc::default(),
            }
        }
    }

    impl Estimator for Fits {
        type Model = Marks;

        fn fit(&self, dataset: &DataFrame) -> impl Future<Output = Result<Marks>> + Send + 'static {
            let fits = self.clone();
            let entry = format!("fit {} on {}", self.name, plan_of(dataset));
            async move {
                let running = fits.running.fetch_add(1, Ordering::SeqCst) + 1;
                fits.most.fetch_max(running, Ordering::SeqCst);
                tokio::task::yield_now().await;
                fits.running.fetch_sub(1, Ordering::SeqCst);
                fits.log.push(entry);
                Ok(Marks(fits.name))
            }
        }
    }

    /// A transformer that puts its rows under its name, as an alias of the plan.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) struct Marks(pub(crate) &'static str);

    impl Transformer for Marks {
        fn transform(&self, dataset: &DataFrame) -> DataFrame {
            dataset.alias(self.0)
        }
    }

    /// An evaluator that logs its dataset and answers the metrics given, in order.
    #[derive(Clone)]
    pub(crate) struct Answers {
        pub(crate) log: Log,
        pub(crate) metrics: Arc<Mutex<VecDeque<f64>>>,
        pub(crate) larger_is_better: bool,
    }

    impl Evaluator for Answers {
        fn evaluate(
            &self,
            dataset: &DataFrame,
        ) -> impl Future<Output = Result<f64>> + Send + 'static {
            self.log.push(format!("evaluate {}", plan_of(dataset)));
            let metric = self.metrics.lock().unwrap().pop_front().unwrap_or_default();
            async move { Ok(metric) }
        }

        fn is_larger_better(&self) -> bool {
            self.larger_is_better
        }
    }

    /// A session of no server: what builds plans needs none.
    pub(crate) async fn session() -> Session {
        Session::builder()
            .remote("sc://localhost:1")
            .connect()
            .await
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Weak;

    use futures::future::BoxFuture;
    use futures::FutureExt;

    use super::*;
    use crate::column::col;

    type Action = fn(&DataFrame) -> BoxFuture<'static, ()>;

    #[tokio::test]
    async fn the_actions_of_a_dataframe_keep_what_its_plan_refers_to_until_they_are_done() {
        let session = mock::session().await;
        let actions: [(&str, Action); 25] = [
            ("stream", |df| df.stream().map(drop).boxed()),
            ("collect", |df| df.collect().map(drop).boxed()),
            ("collect_rows", |df| df.collect_rows().map(drop).boxed()),
            ("collect_as", |df| {
                df.collect_as::<(i64,)>().map(drop).boxed()
            }),
            ("count", |df| df.count().map(drop).boxed()),
            ("take", |df| df.take(1).map(drop).boxed()),
            ("first", |df| df.first().map(drop).boxed()),
            ("is_empty", |df| df.is_empty().map(drop).boxed()),
            ("show", |df| df.show().to_string().map(drop).boxed()),
            ("schema", |df| df.schema().map(drop).boxed()),
            ("explain", |df| {
                df.explain(Default::default()).map(drop).boxed()
            }),
            ("persist", |df| df.cache().map(drop).boxed()),
            ("unpersist", |df| df.unpersist(false).map(drop).boxed()),
            ("storage_level", |df| df.storage_level().map(drop).boxed()),
            ("create_temp_view", |df| {
                df.create_temp_view("v", Default::default())
                    .map(drop)
                    .boxed()
            }),
            ("checkpoint", |df| {
                df.checkpoint(Default::default()).map(drop).boxed()
            }),
            ("input_files", |df| df.input_files().map(drop).boxed()),
            ("same_semantics", |df| {
                df.same_semantics(&df.limit(1)).map(drop).boxed()
            }),
            ("same_semantics of another", |df| {
                df.session().range(1).same_semantics(df).map(drop).boxed()
            }),
            ("semantic_hash", |df| df.semantic_hash().map(drop).boxed()),
            ("is_streaming", |df| df.is_streaming().map(drop).boxed()),
            ("write_stream", |df| {
                df.write_stream().start_without_path().map(drop).boxed()
            }),
            ("write", |df| df.write().save("/out").map(drop).boxed()),
            ("write_to", |df| df.write_to("t").create().map(drop).boxed()),
            ("merge_into", |df| {
                use std::future::IntoFuture;
                df.merge_into("t", col("t.id").eq(col("s.id")))
                    .when_matched(None)
                    .delete()
                    .into_future()
                    .map(drop)
                    .boxed()
            }),
        ];
        for (action, run) in actions {
            let owner = Arc::new(());
            let kept: Weak<()> = Arc::downgrade(&owner);
            let dataframe = session.range(3).with_plan(LogicalPlan::Raw {
                relation: Box::new(session.range(3).to_relation()),
                owner: Some(RelationOwner(owner)),
            });
            let pending = run(&dataframe);
            drop(dataframe);
            assert!(kept.upgrade().is_some(), "{action}: let go while pending");
            drop(pending);
            assert!(kept.upgrade().is_none(), "{action}: kept after");
        }
    }
}
