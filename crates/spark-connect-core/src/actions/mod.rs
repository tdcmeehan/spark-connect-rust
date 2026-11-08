//! The actions of the client, as async functions: what a front end sends and how it reads the
//! responses. Each takes its inputs by value, so that its future is `Send + 'static`.
//!
//! An action that executes a plan chooses its operation id before it sends the request, and
//! reports the execution to an [`ExecutionObserver`] while the future runs.

pub mod session;

use std::sync::Arc;

use spark_connect_proto as proto;

use crate::client::SparkConnectClient;
use crate::error::Result;
use crate::execution::{self, ExecutionInfo};
use crate::plan::PlanIds;

/// What a front end does while an execution runs, such as calling its progress handlers. Its
/// methods run in the action's future, so on the thread that polls it.
pub trait ExecutionObserver: Send {
    /// Before the request is sent, with the operation id of the execution.
    fn on_start(&mut self, _operation_id: &str) {}

    /// With each response, before the action takes its fields.
    fn on_response(&mut self, _response: &proto::ExecutePlanResponse) {}

    /// With the observed metrics of a response, after the observations of the plan got theirs.
    fn on_observed_metrics(&mut self, _metrics: &[proto::execute_plan_response::ObservedMetrics]) {}

    /// When the execution has ended without an error, with what it reported.
    fn on_complete(&mut self, _info: &ExecutionInfo) {}
}

impl ExecutionObserver for () {}

/// The operation id of `request`, chosen now if it has none.
pub fn operation_id(request: &mut proto::ExecutePlanRequest) -> String {
    request
        .operation_id
        .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
        .clone()
}

/// Executes `command`, with the operation tags `tags`, to completion, and returns its responses
/// and what it reported. The observations of the plans of `plan_ids` get their metrics.
pub async fn execute_command(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    mut observer: impl ExecutionObserver,
) -> Result<(Vec<proto::ExecutePlanResponse>, ExecutionInfo)> {
    let mut request =
        execution::execute_plan_request(client.session_id(), tags, execution::command(command));
    observer.on_start(&operation_id(&mut request));
    let mut stream = client.execute_plan_reattachable(request).await?;
    let mut info = ExecutionInfo::default();
    let mut responses = Vec::new();
    while let Some(mut response) = stream.message().await? {
        observer.on_response(&response);
        let observed = info.capture(&mut response, &plan_ids);
        if !observed.is_empty() {
            observer.on_observed_metrics(observed);
        }
        responses.push(response);
    }
    observer.on_complete(&info);
    Ok((responses, info))
}
