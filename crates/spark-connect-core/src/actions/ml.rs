//! The actions of Spark ML: the fit of an estimator, the evaluation of a dataset, and the
//! deletion of the objects that the server caches.

use std::sync::Arc;

use spark_connect_proto as proto;

use super::{command_result, execute_command, ExecutionObserver};
use crate::client::SparkConnectClient;
use crate::error::{Result, SparkError};
use crate::ml;
use crate::plan::PlanIds;

use proto::execute_plan_response::ResponseType;
use proto::ml_command_result::ResultType;

/// Runs `command`, which fits the estimator `estimator_name` (see [`ml::fit`]), and returns
/// what the server tells of the fitted model.
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

/// Runs `command`, which evaluates a dataset (see [`ml::evaluate`]), and returns the metric.
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

/// Deletes the objects of `refs` that the server caches. Its errors, such as those of a stopped
/// session, are ignored, as PySpark ignores them when it releases a model.
pub async fn delete(client: Arc<SparkConnectClient>, refs: Vec<proto::ObjectRef>) {
    let plan_ids = Arc::new(PlanIds::new());
    let _ = execute_command(client, plan_ids, Vec::new(), ml::delete(refs), ()).await;
}
