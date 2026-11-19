//! The actions of Spark ML: the fit of an estimator, the evaluation of a dataset, the methods of
//! the objects that the server holds, and the other ML commands, such as those that save and load
//! them.

use std::sync::Arc;

use spark_connect_proto as proto;

use super::{command_result, ExecutionObserver};
use crate::client::SparkConnectClient;
use crate::error::{Result, SparkError};
use crate::ml::{self, MlValue};

use proto::execute_plan_response::ResponseType;
use proto::ml_command_result::ResultType;

/// Runs `command`, which fits the estimator `estimator_name` (see
/// [`ml::fit`]), and returns what the server tells of the fitted model.
pub async fn fit(
    client: Arc<SparkConnectClient>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    estimator_name: String,
    observer: impl ExecutionObserver,
) -> Result<proto::ml_command_result::MlOperatorInfo> {
    command_result(client, tags, command, observer, |result| match result {
        ResponseType::MlCommandResult(proto::MlCommandResult {
            result_type: Some(ResultType::OperatorInfo(info)),
        }) => Some(info),
        _ => None,
    })
    .await?
    .ok_or_else(|| {
        SparkError::connect_msg(format!(
            "{estimator_name}.fit: server returned no fitted model"
        ))
    })
}

/// Runs `command`, which evaluates a dataset (see [`ml::evaluate`]), and
/// returns the metric.
pub async fn evaluate(
    client: Arc<SparkConnectClient>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    observer: impl ExecutionObserver,
) -> Result<f64> {
    use proto::expression::literal::LiteralType;
    command_result(client, tags, command, observer, |result| match result {
        ResponseType::MlCommandResult(proto::MlCommandResult {
            result_type:
                Some(ResultType::Param(proto::expression::Literal {
                    literal_type: Some(LiteralType::Double(metric)),
                    ..
                })),
        }) => Some(metric),
        _ => None,
    })
    .await?
    .ok_or_else(|| SparkError::connect_msg("evaluate: server returned no metric"))
}

/// Runs the ML command `command`, and returns its result, if the server sends one.
pub async fn run(
    client: Arc<SparkConnectClient>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    observer: impl ExecutionObserver,
) -> Result<Option<proto::MlCommandResult>> {
    command_result(client, tags, command, observer, ml_command_result).await
}

/// The result of a fetch, of the result types of `MlCommandResult`.
#[derive(Debug, Clone, PartialEq)]
pub enum FetchResult {
    /// A value (`param`).
    Value(MlValue),
    /// The ids of new server-side objects (`operator_info.obj_ref`), of a comma-joined id, such
    /// as that of the trees of an ensemble: each its own object.
    Objects(Vec<String>),
    /// A string result (`summary`), as of `toString` or `toDebugString`.
    Summary(String),
}

/// Runs `fetch`, the fetch of a method of an object that the server holds, and returns its
/// result.
pub async fn fetch(
    client: Arc<SparkConnectClient>,
    tags: Vec<String>,
    fetch: proto::Fetch,
    observer: impl ExecutionObserver,
) -> Result<FetchResult> {
    use proto::ml_command_result::ml_operator_info::Type;
    let command = ml::fetch_command(fetch);
    let result = command_result(client, tags, command, observer, ml_command_result);
    match result.await?.and_then(|result| result.result_type) {
        Some(ResultType::Param(literal)) => {
            Ok(FetchResult::Value(MlValue::from_literal(&literal)?))
        }
        Some(ResultType::Summary(text)) => Ok(FetchResult::Summary(text)),
        Some(ResultType::OperatorInfo(info)) => match info.r#type {
            Some(Type::ObjRef(object)) => Ok(FetchResult::Objects(
                object
                    .id
                    .split(',')
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
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

/// Runs `command`, which loads an operator or a model of the JVM class `jvm_class` (see
/// [`ml::read`]), and returns what the server tells of it: the object of a
/// model, its uid and its params.
pub async fn read(
    client: Arc<SparkConnectClient>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    jvm_class: String,
    observer: impl ExecutionObserver,
) -> Result<proto::ml_command_result::MlOperatorInfo> {
    let result = command_result(client, tags, command, observer, ml_command_result);
    let Some(ResultType::OperatorInfo(info)) = result.await?.and_then(|r| r.result_type) else {
        return Err(SparkError::connect_msg(format!(
            "the server returned no operator of the read of {jvm_class}"
        )));
    };
    Ok(info)
}

/// The value of the method `method` of Spark Connect's ML helper, called with `args`: a fetch of
/// the helper, whose result is the literal of the value.
pub async fn helper_attribute(
    client: Arc<SparkConnectClient>,
    tags: Vec<String>,
    method: String,
    args: Vec<proto::expression::Literal>,
    observer: impl ExecutionObserver,
) -> Result<proto::expression::Literal> {
    let args = args.into_iter().map(ml::MethodArg::Param).collect();
    let command = ml::fetch_command(ml::fetch(ml::helper(), &[], &method, args));
    command_result(client, tags, command, observer, |result| match result {
        ResponseType::MlCommandResult(proto::MlCommandResult {
            result_type: Some(ResultType::Param(literal)),
        }) => Some(literal),
        _ => None,
    })
    .await?
    .ok_or_else(|| SparkError::connect_msg(format!("{method}: server returned no value")))
}

fn ml_command_result(result: ResponseType) -> Option<proto::MlCommandResult> {
    match result {
        ResponseType::MlCommandResult(result) => Some(result),
        _ => None,
    }
}
