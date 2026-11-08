//! The actions of the client, as async functions: what a front end sends and how it reads the
//! responses. Each takes its inputs by value, so that its future is `Send + 'static`.
//!
//! An action that executes a plan chooses its operation id before it sends the request, and
//! reports the execution to an [`ExecutionObserver`] while the future runs.

pub mod catalog;
pub mod dataframe;
pub mod session;
pub mod streaming;

use std::sync::Arc;

use spark_connect_proto as proto;

use crate::client::{ReattachableResponseStream, SparkConnectClient};
use crate::error::{Result, SparkError};
use crate::execution::{self, ExecutionInfo};
use crate::plan::PlanIds;
use crate::types::DataType;

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

    /// When an execution whose result is read as it arrives has failed after it started,
    /// with what it reported until then.
    fn on_failure(&mut self, _info: &ExecutionInfo, _error: &SparkError) {}
}

impl ExecutionObserver for () {}

/// The operation id of `request`, chosen now if it has none.
pub fn operation_id(request: &mut proto::ExecutePlanRequest) -> String {
    request
        .operation_id
        .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
        .clone()
}

/// An execution of a plan, whose responses are read as they arrive.
pub struct Execution<O> {
    stream: ReattachableResponseStream,
    plan_ids: Arc<PlanIds>,
    observer: O,
    info: ExecutionInfo,
    schema: Option<DataType>,
}

impl<O: ExecutionObserver> Execution<O> {
    /// Sends `request`, after its operation id is chosen.
    pub async fn start(
        client: Arc<SparkConnectClient>,
        plan_ids: Arc<PlanIds>,
        mut request: proto::ExecutePlanRequest,
        mut observer: O,
    ) -> Result<Self> {
        observer.on_start(&operation_id(&mut request));
        let stream = client.execute_plan_reattachable(request).await?;
        Ok(Execution {
            stream,
            plan_ids,
            observer,
            info: ExecutionInfo::default(),
            schema: None,
        })
    }

    /// What the execution reported so far.
    pub fn info(&self) -> &ExecutionInfo {
        &self.info
    }

    /// The next response, once the observer and the observations of the plan have what it
    /// reports, or `None` at the end of the execution.
    async fn next_response(&mut self) -> Result<Option<proto::ExecutePlanResponse>> {
        let response = self.stream.message().await;
        let Some(mut response) = self.failed(response)? else {
            self.observer.on_complete(&self.info);
            return Ok(None);
        };
        self.observer.on_response(&response);
        let observed = self.info.capture(&mut response, &self.plan_ids);
        if !observed.is_empty() {
            self.observer.on_observed_metrics(observed);
        }
        Ok(Some(response))
    }

    /// `result`, which the observer hears of if it is an error.
    fn failed<T>(&mut self, result: Result<T>) -> Result<T> {
        if let Err(error) = &result {
            self.observer.on_failure(&self.info, error);
        }
        result
    }
}

/// Executes `command`, with the operation tags `tags`, to completion, and returns its responses
/// and what it reported. The observations of the plans of `plan_ids` get their metrics.
pub async fn execute_command(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    observer: impl ExecutionObserver,
) -> Result<(Vec<proto::ExecutePlanResponse>, ExecutionInfo)> {
    let request =
        execution::execute_plan_request(client.session_id(), tags, execution::command(command));
    let mut execution = Execution::start(client, plan_ids, request, observer).await?;
    let mut responses = Vec::new();
    while let Some(response) = execution.next_response().await? {
        responses.push(response);
    }
    Ok((responses, execution.info))
}

/// Executes `command` to completion and returns the first of its responses that `pick` takes.
pub(crate) async fn command_result<T>(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    observer: impl ExecutionObserver,
    pick: impl FnMut(proto::execute_plan_response::ResponseType) -> Option<T>,
) -> Result<Option<T>> {
    let (responses, _) = execute_command(client, plan_ids, tags, command, observer).await?;
    Ok(responses
        .into_iter()
        .filter_map(|response| response.response_type)
        .find_map(pick))
}

/// The result of asking the server for `analyze`.
pub(crate) async fn analyze_result(
    client: &SparkConnectClient,
    analyze: proto::analyze_plan_request::Analyze,
) -> Result<Option<proto::analyze_plan_response::Result>> {
    let request = execution::analyze_plan_request(client.session_id(), analyze);
    Ok(client.analyze_plan(request).await?.result)
}

/// Asks the server for `analyze`, and returns what `take` reads from the result, or else an
/// error that the response did not contain `what`.
pub(crate) async fn analyze<T>(
    client: &SparkConnectClient,
    analyze: proto::analyze_plan_request::Analyze,
    what: &str,
    take: impl FnOnce(proto::analyze_plan_response::Result) -> Option<Result<T>>,
) -> Result<T> {
    let result = analyze_result(client, analyze).await?;
    result.and_then(take).unwrap_or_else(|| {
        Err(SparkError::connect_msg(format!(
            "AnalyzePlan response did not contain {what}"
        )))
    })
}
