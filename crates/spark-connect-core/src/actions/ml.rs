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
    command_result(
        client,
        plan_ids,
        tags,
        command,
        observer,
        |result| match result {
            ResponseType::MlCommandResult(proto::MlCommandResult {
                result_type: Some(ResultType::OperatorInfo(info)),
            }) => Some(info),
            _ => None,
        },
    )
    .await?
    .ok_or_else(|| {
        SparkError::connect_msg(format!(
            "{estimator_name}.fit: server returned no fitted model"
        ))
    })
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
    use proto::expression::literal::LiteralType;
    command_result(
        client,
        plan_ids,
        tags,
        command,
        observer,
        |result| match result {
            ResponseType::MlCommandResult(proto::MlCommandResult {
                result_type:
                    Some(ResultType::Param(proto::expression::Literal {
                        literal_type: Some(LiteralType::Double(metric)),
                        ..
                    })),
            }) => Some(metric),
            _ => None,
        },
    )
    .await?
    .ok_or_else(|| SparkError::connect_msg("evaluate: server returned no metric"))
}
