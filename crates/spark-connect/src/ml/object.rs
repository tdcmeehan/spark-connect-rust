// SPDX-License-Identifier: Apache-2.0
//! The objects of Spark Connect ML that the server holds, and the commands that read, save and
//! load them, as PySpark 4.3.0 sends them over Spark Connect:
//!
//! - [`MlRef`]: a fitted model or another server-side object that a fetch returns (a submodel, a
//!   tree of an ensemble), released on the server when its last owner drops, as PySpark's
//!   `RemoteModelRef`;
//! - [`MlTarget`] and [`fetch`]: a method of an object, or of an object that a method chain reaches
//!   (`<model>.summary`), whose value the server returns (`try_remote_call` of `pyspark.ml.util`);
//! - [`attribute_relation`]: a method whose value is a DataFrame
//!   (`invoke_remote_attribute_relation`);
//! - [`MlValue`]: the value of a fetch, as PySpark's `deserialize_param` reads it;
//! - [`write_operator`], [`write_object`] and [`read`]: the `Write` and `Read` commands of
//!   `pyspark.ml.connect.readwrite`.

use super::*;
use spark_connect_core::ml::read_params;
pub use spark_connect_core::ml::{MlValue, ML_CONNECT_HELPER_ID};

impl MlRef {
    /// The id of the object on the server.
    pub fn id(&self) -> &str {
        &self.0.model_ref.id
    }

    /// The session that holds the object.
    pub fn session(&self) -> &SparkSession {
        &self.0.session
    }

    /// An object that the server returned by its id (an `operator_info` of a fetch or a read), of
    /// which this client is the first owner.
    pub(crate) fn from_id(session: &SparkSession, id: &str) -> Self {
        Self(Arc::new(CachedModelRef {
            model_ref: proto::ObjectRef { id: id.to_string() },
            session: session.clone(),
        }))
    }

    pub(crate) fn object_ref(&self) -> proto::ObjectRef {
        self.0.model_ref.clone()
    }
}

/// The object of a fetch: a server-side object, or the ML helper, and the chain of methods that
/// reaches the object of which a method is fetched, as `_extract_id_methods` of
/// `pyspark.ml.connect.util` splits `"<id>.summary"`. A summary target keeps the dataset that
/// recreates the summary when the server has offloaded the model (`CreateSummary`).
#[derive(Clone)]
pub struct MlTarget {
    root: TargetRoot,
    chain: Vec<String>,
    summary_dataset: Option<DataFrame>,
    /// A target that this one keeps alive, as PySpark's summary of an `evaluate` keeps its model
    /// (`__source_transformer__`).
    keep_alive: Option<Box<MlTarget>>,
}

#[derive(Clone)]
enum TargetRoot {
    Object(MlRef),
    Helper(SparkSession),
    /// An object of an id that this client does not release, as PySpark does not release the
    /// summary that `evaluate` returns.
    Unowned(SparkSession, String),
}

impl std::fmt::Debug for MlTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlTarget")
            .field("id", &self.id())
            .field("summary_dataset", &self.summary_dataset.is_some())
            .finish()
    }
}

impl MlTarget {
    /// The object `object`.
    pub fn object(object: &MlRef) -> Self {
        Self {
            root: TargetRoot::Object(object.clone()),
            chain: Vec::new(),
            summary_dataset: None,
            keep_alive: None,
        }
    }

    /// The object of the id `id` of `session`, which this client does not release, as PySpark's
    /// summary of an `evaluate` (a `summary` result of a fetch).
    pub fn unowned(session: &SparkSession, id: &str) -> Self {
        Self {
            root: TargetRoot::Unowned(session.clone(), id.to_string()),
            chain: Vec::new(),
            summary_dataset: None,
            keep_alive: None,
        }
    }

    /// The ML helper of `session`.
    pub fn helper(session: &SparkSession) -> Self {
        Self {
            root: TargetRoot::Helper(session.clone()),
            chain: Vec::new(),
            summary_dataset: None,
            keep_alive: None,
        }
    }

    /// The object that the method `method` of this one returns, without arguments, as PySpark
    /// names a summary `"<id>.summary"`.
    pub fn child(&self, method: &str) -> Self {
        let mut child = self.clone();
        child.chain.push(method.to_string());
        child
    }

    /// The target, whose summary the server recreates of `dataset`, the summary dataset of the fit
    /// (PySpark's `_summary_dataset`, see [`training_summary`]), when it has lost it, and which the
    /// DataFrames of the summary read.
    /// The target, which keeps `other` alive (its object, which the server keeps while a target
    /// holds it), as PySpark's summary of an `evaluate` keeps its model.
    pub fn keeping(mut self, other: &MlTarget) -> Self {
        self.keep_alive = Some(Box::new(other.clone()));
        self
    }

    pub fn with_summary_dataset(mut self, dataset: &DataFrame) -> Self {
        self.summary_dataset = Some(dataset.clone());
        self
    }

    /// The dataset of the summary of this target, if any.
    pub fn summary_dataset(&self) -> Option<&DataFrame> {
        self.summary_dataset.as_ref()
    }

    /// The object at the root of the target, unless it is the helper.
    pub fn root(&self) -> Option<&MlRef> {
        match &self.root {
            TargetRoot::Object(object) => Some(object),
            TargetRoot::Helper(_) | TargetRoot::Unowned(..) => None,
        }
    }

    /// The id of the target as PySpark names it: the id of its root, and its methods, by dots.
    pub fn id(&self) -> String {
        let root = match &self.root {
            TargetRoot::Object(object) => object.id().to_string(),
            TargetRoot::Helper(_) => ML_CONNECT_HELPER_ID.to_string(),
            TargetRoot::Unowned(_, id) => id.clone(),
        };
        std::iter::once(root)
            .chain(self.chain.iter().cloned())
            .collect::<Vec<_>>()
            .join(".")
    }

    pub fn session(&self) -> &SparkSession {
        match &self.root {
            TargetRoot::Object(object) => object.session(),
            TargetRoot::Helper(session) | TargetRoot::Unowned(session, _) => session,
        }
    }

    fn object_ref(&self) -> proto::ObjectRef {
        match &self.root {
            TargetRoot::Object(object) => object.object_ref(),
            TargetRoot::Helper(_) => ml::helper(),
            TargetRoot::Unowned(_, id) => proto::ObjectRef { id: id.clone() },
        }
    }

    /// The `Fetch` of the method `method` of the target, of the arguments `args`.
    fn fetch_proto(
        &self,
        method: &str,
        args: Vec<FetchArg>,
    ) -> spark_connect_core::error::Result<proto::Fetch> {
        let args = args
            .into_iter()
            .map(|arg| {
                Ok(match arg {
                    FetchArg::Value(value) => ml::MethodArg::Param(value.to_literal()?),
                    FetchArg::Typed(value, data_type) => {
                        ml::MethodArg::Param(value.to_typed_literal(&data_type)?)
                    }
                    FetchArg::Literal(literal) => ml::MethodArg::Param(literal),
                    FetchArg::Input(df) => ml::MethodArg::Input(df.relation()?),
                })
            })
            .collect::<spark_connect_core::error::Result<_>>()?;
        Ok(ml::fetch(self.object_ref(), &self.chain, method, args))
    }
}

/// An argument of a method of a fetch, as `serialize` of `pyspark.ml.connect.serialize` sends it.
#[derive(Clone)]
pub enum FetchArg {
    /// A value, as `serialize_param` sends it.
    Value(MlValue),
    /// A value of the given type, as PySpark sends a `(value, DataType)` pair.
    Typed(MlValue, DataType),
    /// A literal, as the caller has built it.
    Literal(proto::expression::Literal),
    /// A DataFrame, sent as its relation.
    Input(DataFrame),
}

impl std::fmt::Debug for FetchArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(value) => f.debug_tuple("Value").field(value).finish(),
            Self::Typed(value, data_type) => f
                .debug_tuple("Typed")
                .field(value)
                .field(data_type)
                .finish(),
            Self::Literal(literal) => f.debug_tuple("Literal").field(literal).finish(),
            Self::Input(_) => f.write_str("Input(DataFrame)"),
        }
    }
}

/// The result of a fetch, of the result types of `MlCommandResult`.
#[derive(Debug, Clone)]
pub enum FetchOutput {
    /// A value (`param`).
    Value(MlValue),
    /// The ids of new server-side objects (`operator_info.obj_ref`), as PySpark reads a
    /// comma-joined id of the trees of an ensemble: each its own object.
    Objects(Vec<MlRef>),
    /// A string result (`summary`), as of `toString` or `toDebugString`.
    Summary(String),
}

impl FetchOutput {
    /// The value of the result, or an error of another result.
    pub fn into_value(self) -> spark_connect_core::error::Result<MlValue> {
        match self {
            Self::Value(value) => Ok(value),
            other => Err(SparkError::value_msg(format!(
                "expected a value of the fetch, got {other:?}"
            ))),
        }
    }

    /// The single object of the result, or an error of another result.
    pub fn into_object(self) -> spark_connect_core::error::Result<MlRef> {
        match self {
            Self::Objects(mut objects) if objects.len() == 1 => Ok(objects.remove(0)),
            other => Err(SparkError::value_msg(format!(
                "expected one object of the fetch, got {other:?}"
            ))),
        }
    }

    /// The objects of the result, or an error of another result.
    pub fn into_objects(self) -> spark_connect_core::error::Result<Vec<MlRef>> {
        match self {
            Self::Objects(objects) => Ok(objects),
            other => Err(SparkError::value_msg(format!(
                "expected objects of the fetch, got {other:?}"
            ))),
        }
    }

    /// The string of the result, or an error of another result.
    pub fn into_string(self) -> spark_connect_core::error::Result<String> {
        match self {
            Self::Summary(text) => Ok(text),
            Self::Value(MlValue::String(text)) => Ok(text),
            other => Err(SparkError::value_msg(format!(
                "expected a string of the fetch, got {other:?}"
            ))),
        }
    }
}

/// The error class of a summary that the server has lost with its offloaded model.
const MODEL_SUMMARY_LOST: &str = "CONNECT_ML.MODEL_SUMMARY_LOST";

/// Fetch the method `method` of `target`, of the arguments `args`, as `try_remote_call` of
/// `pyspark.ml.util` does: a `Fetch` command of the target and its method chain. When the server
/// has lost a summary (`CONNECT_ML.MODEL_SUMMARY_LOST`), it recreates it of the summary dataset of
/// the target (`CreateSummary`) and fetches again, once.
pub fn fetch(
    target: &MlTarget,
    method: &str,
    args: Vec<FetchArg>,
) -> spark_connect_core::error::Result<FetchOutput> {
    let fetch = target.fetch_proto(method, args)?;
    match run_fetch(target, fetch.clone()) {
        Err(error) if error.get_error_class().as_deref() == Some(MODEL_SUMMARY_LOST) => {
            let (Some(model), Some(dataset)) = (target.root(), target.summary_dataset()) else {
                return Err(error);
            };
            run_ml_command(
                target.session(),
                ml::create_summary(model.object_ref(), dataset.relation()?),
            )?;
            run_fetch(target, fetch)
        }
        result => result,
    }
}

fn run_fetch(
    target: &MlTarget,
    fetch: proto::Fetch,
) -> spark_connect_core::error::Result<FetchOutput> {
    use proto::ml_command_result::{ml_operator_info::Type, ResultType};
    let result = run_ml_command(target.session(), ml::fetch_command(fetch))?;
    match result.and_then(|r| r.result_type) {
        Some(ResultType::Param(literal)) => {
            Ok(FetchOutput::Value(MlValue::from_literal(&literal)?))
        }
        Some(ResultType::Summary(text)) => Ok(FetchOutput::Summary(text)),
        Some(ResultType::OperatorInfo(info)) => match info.r#type {
            Some(Type::ObjRef(object)) => Ok(FetchOutput::Objects(
                object
                    .id
                    .split(',')
                    .filter(|id| !id.is_empty())
                    .map(|id| MlRef::from_id(target.session(), id))
                    .collect(),
            )),
            _ => Err(SparkError::connect_msg(
                "the server returned an operator of no object of the fetch",
            )),
        },
        None => Err(SparkError::connect_msg(
            "the server returned no result of the fetch",
        )),
    }
}

/// Run the ML command `command`, and return its `MlCommandResult`, if the server sends one.
fn run_ml_command(
    session: &SparkSession,
    command: proto::command::CommandType,
) -> spark_connect_core::error::Result<Option<proto::MlCommandResult>> {
    let responses = crate::dataframe::execute_command_collect(session, command)?;
    Ok(responses
        .into_iter()
        .find_map(|response| match response.response_type {
            Some(proto::execute_plan_response::ResponseType::MlCommandResult(result)) => {
                Some(result)
            }
            _ => None,
        }))
}

/// The DataFrame of the method `method` of `target`, of the arguments `args`, as
/// `invoke_remote_attribute_relation` of `pyspark.ml.util` reads it: an `MlRelation` of the fetch,
/// whose `model_summary_dataset` is the summary dataset of a summary target. The DataFrame keeps
/// the object of the target alive, and the plans of the summary dataset and of the DataFrame
/// arguments, which may own other cached objects.
pub fn attribute_relation(
    target: &MlTarget,
    method: &str,
    args: Vec<FetchArg>,
) -> spark_connect_core::error::Result<DataFrame> {
    let inputs: Vec<Arc<LogicalPlan>> = args
        .iter()
        .filter_map(|arg| match arg {
            FetchArg::Input(df) => Some(df.plan.clone()),
            _ => None,
        })
        .collect();
    let fetch = target.fetch_proto(method, args)?;
    let relation = proto::Relation {
        rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
            ml::fetch_relation(
                fetch,
                target
                    .summary_dataset()
                    .map(|df| df.relation())
                    .transpose()?,
            ),
        ))),
        ..Default::default()
    };
    let owner: Arc<dyn std::any::Any + Send + Sync> =
        Arc::new((target.root().cloned(), inputs.clone()));
    match target.summary_dataset() {
        Some(dataset) => Ok(DataFrame::new(
            target.session().clone(),
            LogicalPlan::MlTransform {
                ml_relation: relation,
                input: dataset.plan.clone(),
                owner: Some(RelationOwner(owner)),
            },
        )),
        None => match inputs.first() {
            Some(input) => Ok(DataFrame::new(
                target.session().clone(),
                LogicalPlan::MlTransform {
                    ml_relation: relation,
                    input: input.clone(),
                    owner: Some(RelationOwner(owner)),
                },
            )),
            None => DataFrame::from_relation(target.session().clone(), relation, Some(owner)),
        },
    }
}

/// Save the operator `operator` (an estimator, a transformer or an evaluator, which is not fitted)
/// of the params `params` to `path`, as `RemoteMLWriter.saveInstance` of
/// `pyspark.ml.connect.readwrite` writes one: a `Write` of the operator.
pub fn write_operator(
    session: &SparkSession,
    operator: &MlOperator,
    params: &Params,
    path: &str,
    overwrite: bool,
    options: &HashMap<String, String>,
) -> spark_connect_core::error::Result<()> {
    run_ml_command(
        session,
        ml::write_operator(operator, params, path, overwrite, options),
    )?;
    Ok(())
}

/// Save the server-side object `object` (a fitted model) of the params `params` to `path`, as
/// `RemoteMLWriter.saveInstance` writes a model: a `Write` of its object reference.
pub fn write_object(
    object: &MlRef,
    params: &Params,
    path: &str,
    overwrite: bool,
    options: &HashMap<String, String>,
) -> spark_connect_core::error::Result<()> {
    run_ml_command(
        object.session(),
        ml::write_object(object.object_ref(), params, path, overwrite, options),
    )?;
    Ok(())
}

/// A writer of an estimator, transformer, evaluator or fitted model, as PySpark's `MLWriter`:
/// `.session(&spark)` sets the session that saves an operator (a model saves of its own session),
/// `.overwrite()` replaces a saved one, `.option(key, value)` passes an option of the server's
/// writer, and `.save(path)` sends the `Write`.
#[derive(Clone)]
pub struct MlWriter {
    target: WriterTarget,
    session: Option<SparkSession>,
    overwrite: bool,
    options: HashMap<String, String>,
}

#[derive(Clone)]
enum WriterTarget {
    Operator(MlOperator, Params),
    Object(MlRef, Params),
    /// A meta-algorithm of PySpark's Python format (a pipeline, a validator, their models), which
    /// its own save writes, of the session of the writer.
    Meta(std::sync::Arc<dyn MetaSave>),
}

/// The save of a meta-algorithm of PySpark's Python format to a path, of a session and the
/// writer's options, as its `MLWriter.saveImpl` (see [`MlWriter::meta`]).
pub trait MetaSave: Send + Sync {
    /// The uid of the meta-algorithm, of the writer's `Debug`.
    fn name(&self) -> String;

    /// Save the meta-algorithm to `path`, of `session`, of the writer's `options`.
    fn save_meta(
        &self,
        session: &SparkSession,
        path: &str,
        options: &HashMap<String, String>,
    ) -> spark_connect_core::error::Result<()>;
}

impl std::fmt::Debug for MlWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let target = match &self.target {
            WriterTarget::Operator(operator, _) => operator.name.clone(),
            WriterTarget::Object(object, _) => object.id().to_string(),
            WriterTarget::Meta(meta) => meta.name(),
        };
        f.debug_struct("MlWriter")
            .field("target", &target)
            .field("overwrite", &self.overwrite)
            .field("options", &self.options)
            .finish()
    }
}

impl MlWriter {
    /// The writer of the operator `operator` of the params `params`, which is not fitted.
    pub fn operator(operator: &MlOperator, params: &Params) -> Self {
        Self {
            target: WriterTarget::Operator(operator.clone(), params.clone()),
            session: None,
            overwrite: false,
            options: HashMap::new(),
        }
    }

    /// The writer of the fitted model `object` of the params `params`, of its session.
    pub fn object(object: &MlRef, params: &Params) -> Self {
        Self {
            target: WriterTarget::Object(object.clone(), params.clone()),
            session: Some(object.session().clone()),
            overwrite: false,
            options: HashMap::new(),
        }
    }

    /// The writer of a meta-algorithm of PySpark's Python format, which `meta` saves (see
    /// [`MetaSave`]), of a session that [`Self::session`] sets.
    pub fn meta(meta: std::sync::Arc<dyn MetaSave>) -> Self {
        Self {
            target: WriterTarget::Meta(meta),
            session: None,
            overwrite: false,
            options: HashMap::new(),
        }
    }

    /// The session that saves, as `MLWriter.session` of PySpark.
    pub fn session(mut self, session: &SparkSession) -> Self {
        self.session = Some(session.clone());
        self
    }

    /// Replace a saved operator or model of the path, as `MLWriter.overwrite`.
    pub fn overwrite(mut self) -> Self {
        self.overwrite = true;
        self
    }

    /// An option of the server's writer, as `MLWriter.option`, whose key is lowercased as
    /// PySpark's `optionMap[key.lower()]` does, so two spellings of a key are one option.
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_lowercase(), value.to_string());
        self
    }

    /// Save to `path`, as `MLWriter.save`.
    pub fn save(&self, path: &str) -> spark_connect_core::error::Result<()> {
        match &self.target {
            WriterTarget::Object(object, params) => {
                write_object(object, params, path, self.overwrite, &self.options)
            }
            WriterTarget::Meta(meta) => {
                let session = self.session.as_ref().ok_or_else(|| {
                    SparkError::value_msg(format!(
                        "{} needs a session to save with: call .session(&spark) on its writer",
                        meta.name()
                    ))
                })?;
                // As `RemoteMLWriter.saveInstance` of a meta-algorithm: the ML helper deletes the
                // path of an overwrite first.
                if self.overwrite {
                    handle_overwrite(session, path)?;
                }
                meta.save_meta(session, path, &self.options)
            }
            WriterTarget::Operator(operator, params) => {
                let session = self.session.as_ref().ok_or_else(|| {
                    SparkError::value_msg(format!(
                        "{} needs a session to save with: call .session(&spark) on its writer",
                        operator.name
                    ))
                })?;
                write_operator(
                    session,
                    operator,
                    params,
                    path,
                    self.overwrite,
                    &self.options,
                )
            }
        }
    }
}

/// What the server reads of a saved operator or model: the object of a model, which this client
/// owns, the uid and the params.
#[derive(Debug, Clone)]
pub struct ReadResult {
    /// The object of a model; None of an estimator, a transformer or an evaluator.
    pub object: Option<MlRef>,
    pub uid: String,
    pub params: Params,
}

/// Load the operator or model of the JVM class `jvm_class` of the type `op_type` from `path`, as
/// `RemoteMLReader.loadInstance` of `pyspark.ml.connect.readwrite` reads one: a `Read` of the
/// operator, whose result names the object of a model, its uid and its params.
pub fn read(
    session: &SparkSession,
    jvm_class: &str,
    op_type: OperatorType,
    path: &str,
) -> spark_connect_core::error::Result<ReadResult> {
    use proto::ml_command_result::{ml_operator_info::Type, ResultType};
    let result = run_ml_command(session, ml::read(jvm_class, op_type, path))?;
    let Some(ResultType::OperatorInfo(info)) = result.and_then(|r| r.result_type) else {
        return Err(SparkError::connect_msg(format!(
            "the server returned no operator of the read of {jvm_class}"
        )));
    };
    let object = match &info.r#type {
        Some(Type::ObjRef(object)) if op_type == OperatorType::Model => {
            Some(MlRef::from_id(session, &object.id))
        }
        _ => None,
    };
    Ok(ReadResult {
        object,
        uid: info.uid.clone().unwrap_or_default(),
        params: read_params(info.params.as_ref()),
    })
}

/// A model of a server-side object, of its operator and params, which transforms as every cached
/// model does: the model of an object that `read_model` loads, or that a fetch returns (a tree of
/// an ensemble), whose class this client does not otherwise model.
#[derive(Debug, Clone)]
pub struct ObjectModel {
    operator: MlOperator,
    params: Params,
    object: MlRef,
}

impl ObjectModel {
    /// The model of `object`, of the operator `operator` (of the model's JVM class and uid) and
    /// the params `params`.
    pub fn new(object: MlRef, operator: MlOperator, params: Params) -> Self {
        Self {
            operator,
            params,
            object,
        }
    }
}

impl Transformer for ObjectModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.object.transform(df, &self.params)
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.object.build_ml_relation(input, &self.params)
    }
}

impl Model for ObjectModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }

    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.object)
    }
}

/// The target of the training summary of the fitted model `object` (`"<id>.summary"`), of the
/// summary dataset `dataset`, as PySpark's `try_remote_fit` makes it: `dataset` is the training
/// dataset itself for the models whose `_summary_dataset` returns it (the classification models,
/// LinearRegression and GeneralizedLinearRegression), and [`summary_transform`] of it for the
/// others (the clustering models, of the default `_summary_dataset`).
pub fn training_summary(object: &MlRef, dataset: &DataFrame) -> MlTarget {
    MlTarget::object(object)
        .child("summary")
        .with_summary_dataset(dataset)
}

/// The transform of `dataset` by the fitted model `object`, of no params, as PySpark's default
/// `_summary_dataset` (`self.transform(train_dataset)`) transforms the training dataset before the
/// fit copies the estimator's params to the model: the server's model object transforms of its own
/// params.
pub fn summary_transform(
    object: &MlRef,
    dataset: &DataFrame,
) -> spark_connect_core::error::Result<DataFrame> {
    object.transform(dataset, &Params::new())
}

/// The target of the summary of an `evaluate` of the model `model`, of the id of the `summary`
/// result that the server returns, which this client does not release, as PySpark does not, and
/// which keeps the model alive, as PySpark's summary keeps its `__source_transformer__`.
pub fn evaluated_summary(
    model: &MlTarget,
    output: FetchOutput,
) -> spark_connect_core::error::Result<MlTarget> {
    let summary = MlTarget::unowned(model.session(), &output.into_string()?);
    Ok(summary.keeping(model))
}

/// The estimator, transformer or evaluator of the JVM class `jvm_class` that `read` loads from
/// `path`: its operator, of the uid that it was saved with, and its params.
pub fn read_operator(
    session: &SparkSession,
    jvm_class: &str,
    op_type: OperatorType,
    path: &str,
) -> spark_connect_core::error::Result<(MlOperator, Params)> {
    let read = read(session, jvm_class, op_type, path)?;
    Ok((
        MlOperator::with_uid(jvm_class, &read.uid, op_type),
        read.params,
    ))
}

/// The model of the JVM class `jvm_class` that `read` loads from `path`: its object, its operator,
/// of the uid that it was saved with, and its params.
pub fn read_model(
    session: &SparkSession,
    jvm_class: &str,
    path: &str,
) -> spark_connect_core::error::Result<(MlRef, MlOperator, Params)> {
    let read = read(session, jvm_class, OperatorType::Model, path)?;
    let object = read.object.ok_or_else(|| {
        SparkError::connect_msg(format!(
            "the server returned no object of the read of {jvm_class}"
        ))
    })?;
    Ok((
        object,
        MlOperator::with_uid(jvm_class, &read.uid, OperatorType::Model),
        read.params,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)] // The element and struct types of the literals of PySpark.
    fn read_params_are_typed_as_pyspark_sends_them_again() {
        use proto::expression::literal::LiteralType as L;
        let literal = |literal_type| proto::expression::Literal {
            literal_type: Some(literal_type),
            ..Default::default()
        };
        // Just past the 32 bits of an Integer, on either side.
        let (above, below) = (i64::from(i32::MAX) + 1, i64::from(i32::MIN) - 1);
        let cases = [
            ("seed", L::Long(7), L::Integer(7)),
            ("max", L::Long(i64::from(i32::MAX)), L::Integer(i32::MAX)),
            ("min", L::Long(i64::from(i32::MIN)), L::Integer(i32::MIN)),
            ("above", L::Long(above), L::Long(above)),
            ("below", L::Long(below), L::Long(below)),
            ("short", L::Short(3), L::Integer(3)),
            ("byte", L::Byte(-2), L::Integer(-2)),
            ("int", L::Integer(5), L::Integer(5)),
            ("float", L::Float(0.5), L::Double(0.5)),
            ("double", L::Double(0.25), L::Double(0.25)),
            ("string", L::String("s".into()), L::String("s".into())),
            ("bool", L::Boolean(true), L::Boolean(true)),
        ];
        let mut sent = proto::MlParams::default();
        for (name, read, _) in &cases {
            sent.params.insert(name.to_string(), literal(read.clone()));
        }
        // A typed Long: PySpark's literal of the Python int has no type of its own.
        sent.params.insert(
            "typed".into(),
            proto::expression::Literal {
                literal_type: Some(L::Long(9)),
                data_type: Some(DataType::Long.to_proto()),
            },
        );
        // An array is kept as it is, elements and all: an int array param is an Array[Int] on the
        // server, which sends Integer elements, as PySpark re-infers them (no pyspark.ml param is
        // an array of longs).
        let array = literal(L::Array(proto::expression::literal::Array {
            elements: vec![literal(L::Integer(1))],
            ..Default::default()
        }));
        sent.params.insert("array".into(), array.clone());
        let info = proto::ml_command_result::MlOperatorInfo {
            params: Some(sent),
            ..Default::default()
        };

        let params = read_params(info.params.as_ref()).to_proto();
        for (name, _, expected) in cases {
            assert_eq!(params.params[name], literal(expected), "{name}");
        }
        assert_eq!(params.params["typed"], literal(L::Integer(9)));
        // Serialized again as its elements infer it: the element type of the first one.
        let inferred = literal(L::Array(proto::expression::literal::Array {
            element_type: Some(DataType::Integer.to_proto()),
            elements: vec![literal(L::Integer(1))],
        }));
        assert_eq!(params.params["array"], inferred);
        assert_eq!(params.params.len(), 14);
        assert_eq!(read_params(None).to_proto(), proto::MlParams::default());
    }

    #[test]
    #[allow(deprecated)] // The element and struct types of the literals of PySpark.
    fn read_params_serialize_other_values_again_as_pyspark_infers_them() {
        use proto::expression::literal::{Array, LiteralType as L};
        let string = |word: &str| proto::expression::Literal {
            literal_type: Some(L::String(word.into())),
            ..Default::default()
        };
        // The stopWords of the StopWordsRemover golden (feature_models_persistence.json): the
        // server reads its string array without a collation, which PySpark's save sends as
        // UTF8_BINARY.
        let string_array = |collation: &str| proto::expression::Literal {
            literal_type: Some(L::Array(Array {
                element_type: Some(
                    DataType::String {
                        collation: collation.into(),
                    }
                    .to_proto(),
                ),
                elements: vec![string("the"), string("gamma")],
            })),
            ..Default::default()
        };
        // A dense vector that the server reads with its Python class, which PySpark's save does
        // not send.
        let mut vector = MlValue::Vector(MlVector::Dense(vec![1.0, 2.0]))
            .to_literal()
            .unwrap();
        let Some(L::Struct(vector_struct)) = vector.literal_type.as_mut() else {
            panic!("not a vector: {vector:?}");
        };
        let Some(proto::data_type::Kind::Udt(udt)) = vector_struct
            .struct_type
            .as_mut()
            .and_then(|data_type| data_type.kind.as_mut())
        else {
            panic!("not a UDT: {vector_struct:?}");
        };
        udt.python_class = Some("pyspark.ml.linalg.DenseVector".into());
        let mut sent = proto::MlParams::default();
        sent.params.insert("stopWords".into(), string_array(""));
        sent.params.insert("vector".into(), vector);
        let params = read_params(Some(&sent)).to_proto();
        assert_eq!(params.params["stopWords"], string_array("UTF8_BINARY"));
        let expected = MlValue::Vector(MlVector::Dense(vec![1.0, 2.0]))
            .to_literal()
            .unwrap();
        assert_eq!(params.params["vector"], expected);
    }

    #[test]
    fn writer_option_keys_are_lowercased_as_pyspark_does() {
        let operator = MlOperator::with_uid(
            "org.apache.spark.ml.feature.StopWordsRemover",
            "uid",
            OperatorType::Model,
        );
        let writer = MlWriter::operator(&operator, &Params::default())
            .option("Foo", "1")
            .option("foo", "2");
        assert_eq!(writer.options.len(), 1);
        assert_eq!(writer.options["foo"], "2");
    }

    #[test]
    fn ml_values_round_trip_through_their_literals() {
        let values = vec![
            MlValue::Null,
            MlValue::Bool(true),
            MlValue::Int(7),
            MlValue::Long(1 << 40),
            MlValue::Double(0.5),
            MlValue::String("s".into()),
            MlValue::Array(vec![MlValue::Double(1.0), MlValue::Double(2.5)]),
            MlValue::Array(vec![MlValue::String("a".into())]),
            MlValue::Vector(MlVector::Dense(vec![1.0, 2.0])),
            MlValue::Vector(MlVector::Sparse {
                size: 3,
                indices: vec![1],
                values: vec![2.0],
            }),
            MlValue::Matrix(MlMatrix::Dense {
                num_rows: 1,
                num_cols: 2,
                values: vec![0.0, -1.0],
                is_transposed: false,
            }),
        ];
        for value in values {
            let literal = value.to_literal().unwrap();
            assert_eq!(
                MlValue::from_literal(&literal).unwrap(),
                value,
                "{literal:?}"
            );
        }
    }

    #[test]
    fn decimal_values_are_sent_as_decimal_type_as_pyspark_sends_them() {
        use proto::expression::literal::LiteralType as L;
        // PySpark's `_infer_type` of a decimal.Decimal is DecimalType(): precision 10 and scale 0,
        // whatever the digits of the value, and the server widens it.
        for value in ["1.25", "-0.5", "12345678901.5", "100"] {
            let literal = MlValue::Decimal(value.into()).to_literal().unwrap();
            let Some(L::Decimal(decimal)) = literal.literal_type else {
                panic!("not a decimal: {literal:?}");
            };
            assert_eq!(decimal.value, value);
            assert_eq!((decimal.precision, decimal.scale), (Some(10), Some(0)));
        }
        // A `(value, DecimalType(p, s))` pair is the literal of its type.
        let typed = MlValue::Decimal("1.25".into())
            .to_typed_literal(&DataType::Decimal {
                precision: 12,
                scale: 3,
            })
            .unwrap();
        let Some(L::Decimal(decimal)) = typed.literal_type else {
            panic!("not a decimal: {typed:?}");
        };
        assert_eq!((decimal.precision, decimal.scale), (Some(12), Some(3)));
    }

    #[test]
    fn ml_values_type_arrays_by_their_first_element_or_as_given() {
        let error = MlValue::Array(Vec::new()).to_literal().unwrap_err();
        assert_eq!(
            error.get_error_class().as_deref(),
            Some("CANNOT_INFER_ARRAY_ELEMENT_TYPE")
        );
        let typed = MlValue::Array(Vec::new())
            .to_typed_literal(&DataType::Array {
                element_type: Box::new(DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                }),
                contains_null: true,
            })
            .unwrap();
        assert_eq!(
            MlValue::from_literal(&typed).unwrap(),
            MlValue::Array(Vec::new())
        );
        assert_eq!(MlValue::Int(3).as_f64(), Some(3.0));
        assert_eq!(MlValue::Long(4).as_i64(), Some(4));
    }

    #[test]
    fn ml_targets_chain_their_methods_as_pyspark_names_them() {
        let session = crate::session::SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let model = MlRef::from_id(&session, "abc");
        let summary = MlTarget::object(&model).child("summary");
        assert_eq!(summary.id(), "abc.summary");
        let fetch = summary
            .fetch_proto("accuracy", vec![FetchArg::Value(MlValue::Int(1))])
            .unwrap();
        assert_eq!(fetch.obj_ref.unwrap().id, "abc");
        let names: Vec<_> = fetch.methods.iter().map(|m| m.method.as_str()).collect();
        assert_eq!(names, ["summary", "accuracy"]);
        assert_eq!(MlTarget::helper(&session).id(), ML_CONNECT_HELPER_ID);
        std::mem::forget(model);
    }

    #[test]
    fn training_summaries_object_models_and_evaluated_summaries_are_of_their_objects() {
        let session = crate::session::SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let object = MlRef::from_id(&session, "fitted");
        let train = session.range(3).expect("range");

        // The training summary is the summary of the object, of the summary dataset.
        let summary = training_summary(&object, &train);
        assert_eq!(summary.id(), "fitted.summary");
        assert!(summary.summary_dataset().is_some());

        // A model of the object transforms of its params, as a cached model does.
        let operator = MlOperator::with_uid(
            "org.apache.spark.ml.clustering.KMeansModel",
            "KMeans_0123456789ab",
            OperatorType::Model,
        );
        let model = ObjectModel::new(
            object.clone(),
            operator,
            Params::new().set_param_int("k", 2),
        );
        assert_eq!(model.ml_ref().map(MlRef::id), Some("fitted"));
        assert_eq!(model.clone_box().operator().uid, "KMeans_0123456789ab");
        let relation = model.build_ml_relation(&train).expect("relation");
        let Some(proto::ml_relation::MlType::Transform(transform)) = relation.ml_type else {
            panic!("a transform: {relation:?}");
        };
        let Some(proto::ml_relation::transform::Operator::ObjRef(obj_ref)) = transform.operator
        else {
            panic!("of the object");
        };
        assert_eq!(obj_ref.id, "fitted");
        assert!(transform.params.expect("params").params.contains_key("k"));

        // The summary of an evaluate is the server's id; any other result is an error.
        let held = std::sync::Arc::strong_count(&object.0);
        let model_target = MlTarget::object(&object);
        let evaluated =
            evaluated_summary(&model_target, FetchOutput::Summary("evaluated".into())).unwrap();
        assert_eq!(evaluated.id(), "evaluated");
        assert!(evaluated_summary(&model_target, FetchOutput::Value(MlValue::Int(1))).is_err());
        // The summary keeps the model's object alive.
        drop(model_target);
        assert_eq!(std::sync::Arc::strong_count(&object.0), held + 1);
        std::mem::forget(evaluated);
        std::mem::forget((model, object));
    }
}
