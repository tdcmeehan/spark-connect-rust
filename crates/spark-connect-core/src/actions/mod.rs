//! The actions of the client, as async functions: what a front end sends and how it reads the
//! responses. Each takes its inputs by value, so that its future is `Send + 'static`.
//!
//! An action that executes a plan chooses its operation id before it sends the request, and
//! reports the execution to an [`ExecutionObserver`] while the future runs.

pub mod catalog;
pub mod dataframe;
pub mod session;
pub mod streaming;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use spark_connect_proto as proto;

use crate::arrow_chunks::ArrowChunkAssembler;
use crate::client::{ReattachableResponseStream, SparkConnectClient};
use crate::error::{Result, SparkError};
use crate::execution::{self, ExecutionInfo};
use crate::types::DataType;

/// What a front end does while an execution runs, such as calling its progress handlers. Its
/// methods run in the action's future, so on the thread that polls it. An error of one fails the
/// execution.
pub trait ExecutionObserver: Send {
    /// Before the request is sent, with the operation id of the execution.
    fn on_start(&mut self, _operation_id: &str) {}

    /// With each response, before the action takes its fields.
    fn on_response(&mut self, _response: &proto::ExecutePlanResponse) -> Result<()> {
        Ok(())
    }

    /// With the observed metrics of a response. An error fails the execution with its operation
    /// id.
    fn on_observed_metrics(
        &mut self,
        _metrics: &[proto::execute_plan_response::ObservedMetrics],
    ) -> Result<()> {
        Ok(())
    }

    /// When the execution has ended without an error, with what it reported.
    fn on_complete(&mut self, _info: &ExecutionInfo) -> Result<()> {
        Ok(())
    }

    /// When the execution fails after its start with `error`, with what it reported until then:
    /// the error that it fails with.
    fn on_failure(&mut self, _info: &ExecutionInfo, error: SparkError) -> SparkError {
        error
    }

    /// When the request that starts the execution fails with `error`, before any response: the
    /// error that it fails with. By default, the failure of an execution that reported nothing.
    fn on_start_failure(&mut self, error: SparkError) -> SparkError {
        self.on_failure(&ExecutionInfo::default(), error)
    }

    /// When a prefetch stops reading the execution before its end, as its rows are dropped or a
    /// batch fails to decode, with what it reported. The execution ends when it drops.
    fn on_abandoned(&mut self, _info: &ExecutionInfo) {}
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
    /// Dropped unfinished, it releases the execution, so it outlives the observer's done call.
    stream: ReattachableResponseStream,
    chunks: ArrowChunkAssembler,
    observer: O,
    info: ExecutionInfo,
    schema: Option<DataType>,
}

impl<O: ExecutionObserver> Execution<O> {
    /// Sends `request`, after its operation id is chosen.
    pub async fn start(
        client: Arc<SparkConnectClient>,
        mut request: proto::ExecutePlanRequest,
        mut observer: O,
    ) -> Result<Self> {
        observer.on_start(&operation_id(&mut request));
        let stream = match boxed(|| client.execute_plan_reattachable(request)).await {
            Ok(stream) => stream,
            Err(error) => return Err(observer.on_start_failure(error)),
        };
        Ok(Execution {
            stream,
            chunks: ArrowChunkAssembler::default(),
            observer,
            info: ExecutionInfo::default(),
            schema: None,
        })
    }

    /// What the execution reported so far.
    pub fn info(&self) -> &ExecutionInfo {
        &self.info
    }

    /// The response of `message`, the next message of the stream, with the chunks of an Arrow
    /// batch joined into the batch, once the observer has what it reports, or `None` at the end of
    /// the execution.
    // Sync, as all work on a response: debug builds keep an async fn's values on the polling stack.
    fn response(
        &mut self,
        message: Result<Option<proto::ExecutePlanResponse>>,
    ) -> Result<Option<proto::ExecutePlanResponse>> {
        let response = self.receive(message);
        let Some(response) = self.failed(response)? else {
            self.observer.on_complete(&self.info)?;
            return Ok(None);
        };
        Ok(Some(response))
    }

    fn receive(
        &mut self,
        message: Result<Option<proto::ExecutePlanResponse>>,
    ) -> Result<Option<proto::ExecutePlanResponse>> {
        let response = message?;
        if let Some(response) = &response {
            self.observer.on_response(response)?;
        }
        let Some(mut response) = self.chunks.join(response)? else {
            return Ok(None);
        };
        capture(&mut response, &mut self.info, &mut self.observer)?;
        Ok(Some(response))
    }

    /// `result`, or the error that the observer fails the execution with when it is an error.
    fn failed<T>(&mut self, result: Result<T>) -> Result<T> {
        result.map_err(|error| self.observer.on_failure(&self.info, error))
    }
}

/// Executes `command`, with the operation tags `tags`, to completion, and returns its responses,
/// with the chunks of each Arrow batch joined into the batch, and what it reported.
pub async fn execute_command(
    client: Arc<SparkConnectClient>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    observer: impl ExecutionObserver,
) -> Result<(Vec<proto::ExecutePlanResponse>, ExecutionInfo)> {
    let request =
        execution::execute_plan_request(client.session_id(), tags, execution::command(command));
    let mut execution = Execution::start(client, request, observer).await?;
    let mut responses = Vec::new();
    loop {
        let message = boxed(|| execution.stream.message()).await;
        let Some(response) = execution.response(message)? else {
            break;
        };
        responses.push(response);
    }
    Ok((responses, execution.info))
}

/// Executes `command` to completion and returns the first of its responses that `pick` takes.
pub(crate) async fn command_result<T>(
    client: Arc<SparkConnectClient>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    observer: impl ExecutionObserver,
    pick: impl FnMut(proto::execute_plan_response::ResponseType) -> Option<T>,
) -> Result<Option<T>> {
    let (responses, _) = execute_command(client, tags, command, observer).await?;
    Ok(responses
        .into_iter()
        .filter_map(|response| response.response_type)
        .find_map(pick))
}

/// The future that `make` makes, on the heap, so that the async fn that awaits it holds a pointer.
fn boxed<F: Future>(make: impl FnOnce() -> F) -> Pin<Box<F>> {
    Box::pin(make())
}

/// Takes what `response` reports into `info`, and gives its observed metrics to `observer`, whose
/// error is one of the execution, with its operation id, as PySpark's client raises it.
fn capture(
    response: &mut proto::ExecutePlanResponse,
    info: &mut ExecutionInfo,
    observer: &mut impl ExecutionObserver,
) -> Result<()> {
    let observed = info.capture(response);
    if observed.is_empty() {
        return Ok(());
    }
    let set = observer.on_observed_metrics(observed);
    set.map_err(|error| match &info.operation_id {
        Some(operation_id) => error.with_operation_id(operation_id.as_str()),
        None => error,
    })
}

/// The result of asking the server for `analyze`.
pub(crate) async fn analyze_result(
    client: &SparkConnectClient,
    analyze: proto::analyze_plan_request::Analyze,
) -> Result<Option<proto::analyze_plan_response::Result>> {
    let request = execution::analyze_plan_request(client.session_id(), analyze);
    Ok(boxed(|| client.analyze_plan(request)).await?.result)
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
