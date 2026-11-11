//! The actions of Spark ML: the fit of an estimator and the evaluation of a dataset.

use std::sync::Arc;

use spark_connect_proto as proto;

use super::{command_result, ExecutionObserver};
use crate::client::SparkConnectClient;
use crate::error::{Result, SparkError};
use crate::plan::PlanIds;

use proto::execute_plan_response::ResponseType;
use proto::ml_command_result::ResultType;

/// Runs `command`, which fits the estimator `estimator_name` (see
/// [`ml::fit`](crate::ml::fit)), and returns what the server tells of the fitted model.
pub async fn fit(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    estimator_name: String,
    observer: impl ExecutionObserver,
) -> Result<proto::ml_command_result::MlOperatorInfo> {
    command_result(client, plan_ids, tags, command, observer, fitted)
        .await?
        .ok_or_else(|| {
            SparkError::connect_msg(format!(
                "{estimator_name}.fit: server returned no fitted model"
            ))
        })
}

/// What the server tells of a fitted model in `response`, if it is the result of a fit. Not a
/// stable API.
#[doc(hidden)]
pub fn fitted(response: ResponseType) -> Option<proto::ml_command_result::MlOperatorInfo> {
    match response {
        ResponseType::MlCommandResult(proto::MlCommandResult {
            result_type: Some(ResultType::OperatorInfo(info)),
        }) => Some(info),
        _ => None,
    }
}

/// Runs `command`, which evaluates a dataset (see [`ml::evaluate`](crate::ml::evaluate)), and
/// returns the metric.
pub async fn evaluate(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    observer: impl ExecutionObserver,
) -> Result<f64> {
    command_result(client, plan_ids, tags, command, observer, metric)
        .await?
        .ok_or_else(|| SparkError::connect_msg("evaluate: server returned no metric"))
}

/// The metric in `response`, if it is the result of an evaluation. Not a stable API.
#[doc(hidden)]
pub fn metric(response: ResponseType) -> Option<f64> {
    use proto::expression::literal::LiteralType;
    match response {
        ResponseType::MlCommandResult(proto::MlCommandResult {
            result_type:
                Some(ResultType::Param(proto::expression::Literal {
                    literal_type: Some(LiteralType::Double(metric)),
                    ..
                })),
        }) => Some(metric),
        _ => None,
    }
}

/// The value in `response`, if it is the result of a fetch (see [`ml::fetch`](crate::ml::fetch)).
/// Not a stable API.
#[doc(hidden)]
pub fn fetched(response: ResponseType) -> Option<proto::expression::Literal> {
    match response {
        ResponseType::MlCommandResult(proto::MlCommandResult {
            result_type: Some(ResultType::Param(value)),
        }) => Some(value),
        _ => None,
    }
}
