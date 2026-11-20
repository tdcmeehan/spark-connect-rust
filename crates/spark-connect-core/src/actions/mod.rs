//! The actions of the client, as async functions: what a front end sends and how it reads the
//! responses. Each takes its inputs by value, so that its future is `Send + 'static`.
//!
//! An action that executes a plan chooses its operation id before it sends the request, and
//! reports the execution to an [`ExecutionObserver`] while the future runs.

pub mod catalog;
pub mod dataframe;
pub mod ml;
pub mod pipelines;
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

/// The responses of an execution of a plan, as the server sends them. Its observer hears of the
/// start and of each response only, and the observations of the plan get no metrics.
pub struct Responses<O> {
    /// Dropped unfinished, it releases the execution, so it outlives the observer's done call.
    stream: ReattachableResponseStream,
    chunks: ArrowChunkAssembler,
    observer: O,
}

impl<O: ExecutionObserver> Responses<O> {
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
        Ok(Responses {
            stream,
            chunks: ArrowChunkAssembler::default(),
            observer,
        })
    }

    /// The next response, with the chunks of an Arrow batch joined into the batch, or `None` at
    /// the end of the execution: the response of a chunk that is not the last one has no batch.
    pub async fn next(&mut self) -> Result<Option<proto::ExecutePlanResponse>> {
        let message = boxed(|| self.stream.message()).await;
        self.receive(message)
    }

    /// The response of `message`, the next message of the stream, once the observer has heard of
    /// it, with its chunk joined into its batch.
    fn receive(
        &mut self,
        message: Result<Option<proto::ExecutePlanResponse>>,
    ) -> Result<Option<proto::ExecutePlanResponse>> {
        let response = message?;
        if let Some(response) = &response {
            self.observer.on_response(response)?;
        }
        self.chunks.join(response)
    }
}

/// An execution of a plan, whose responses are read as they arrive.
pub struct Execution<O> {
    responses: Responses<O>,
    info: ExecutionInfo,
    schema: Option<DataType>,
}

impl<O: ExecutionObserver> Execution<O> {
    /// Sends `request`, after its operation id is chosen.
    pub async fn start(
        client: Arc<SparkConnectClient>,
        request: proto::ExecutePlanRequest,
        observer: O,
    ) -> Result<Self> {
        Ok(Execution {
            responses: Responses::start(client, request, observer).await?,
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
            self.responses.observer.on_complete(&self.info)?;
            return Ok(None);
        };
        Ok(Some(response))
    }

    fn receive(
        &mut self,
        message: Result<Option<proto::ExecutePlanResponse>>,
    ) -> Result<Option<proto::ExecutePlanResponse>> {
        let Some(mut response) = self.responses.receive(message)? else {
            return Ok(None);
        };
        capture_response(&mut response, &mut self.info, &mut self.responses.observer)?;
        Ok(Some(response))
    }

    /// `result`, or the error that the observer fails the execution with when it is an error.
    fn failed<T>(&mut self, result: Result<T>) -> Result<T> {
        result.map_err(|error| self.responses.observer.on_failure(&self.info, error))
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
        let message = boxed(|| execution.responses.stream.message()).await;
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

/// Executes `command` to completion and ignores its responses and its errors, such as those of a
/// closed channel: for a command that the caller does not wait for, as PySpark sends the releases
/// of what the server caches.
pub async fn execute_command_ignoring_errors(
    client: Arc<SparkConnectClient>,
    command: proto::command::CommandType,
) {
    let _ = execute_command(client, Vec::new(), command, ()).await;
}

/// The future that `make` makes, on the heap, so that the async fn that awaits it holds a pointer.
/// Not a stable API.
#[doc(hidden)]
pub fn boxed<F: Future>(make: impl FnOnce() -> F) -> Pin<Box<F>> {
    Box::pin(make())
}

/// Takes what `response` reports into `info`, and gives its observed metrics to `observer`, whose
/// error is one of the execution, with its operation id, as PySpark's client raises it. Not a
/// stable API.
#[doc(hidden)]
pub fn capture_response(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_static<T: Send + 'static>(_: T) {}

    #[test]
    fn the_futures_of_the_actions_are_send_and_static() {
        // Type-checked, never called: it would need a connected client.
        let _ = |client: Arc<SparkConnectClient>| {
            let client = || Arc::clone(&client);
            let command = || proto::command::CommandType::SqlCommand(Default::default());
            let relation = proto::Relation::default;
            assert_send_static(Responses::start(client(), Default::default(), ()));
            assert_send_static(Execution::start(client(), Default::default(), ()));
            assert_send_static(execute_command(client(), vec![], command(), ()));
            assert_send_static(session::spark_version(client()));
            assert_send_static(session::parse_ddl(client(), String::new()));
            assert_send_static(session::json_to_ddl(client(), String::new()));
            assert_send_static(session::sql(client(), vec![], relation(), ()));
            assert_send_static(session::create_resource_profile(
                client(),
                vec![],
                Default::default(),
                (),
            ));
            assert_send_static(dataframe::collect_rows(
                client(),
                Default::default(),
                (),
                |_| Ok(()),
            ));
            assert_send_static(dataframe::collect_record_batches(
                client(),
                Default::default(),
                (),
                |_| Ok(()),
            ));
            assert_send_static(dataframe::schema(client(), relation()));
            assert_send_static(dataframe::explain(client(), relation(), Default::default()));
            assert_send_static(dataframe::input_files(client(), relation()));
            assert_send_static(dataframe::storage_level(client(), relation()));
            assert_send_static(dataframe::semantic_hash(client(), relation()));
            assert_send_static(dataframe::same_semantics(client(), relation(), relation()));
            assert_send_static(dataframe::is_streaming(client(), relation()));
            assert_send_static(dataframe::is_local(client(), relation()));
            assert_send_static(dataframe::persist(client(), relation(), Default::default()));
            assert_send_static(dataframe::unpersist(client(), relation(), false));
            assert_send_static(catalog::rows(client(), Default::default(), ()));
            assert_send_static(streaming::start(client(), vec![], Default::default(), ()));
            assert_send_static(streaming::query_command(
                client(),
                vec![],
                Default::default(),
                (),
            ));
            assert_send_static(streaming::manager_command(
                client(),
                vec![],
                Default::default(),
                (),
            ));
            assert_send_static(streaming::unsubscribe(client(), vec![], ()));
            assert_send_static(streaming::ListenerEvents::subscribe(client(), ()));
            assert_send_static(ml::fit(client(), vec![], command(), String::new(), ()));
            assert_send_static(ml::evaluate(client(), vec![], command(), ()));
            assert_send_static(ml::run(client(), vec![], command(), ()));
            assert_send_static(ml::fetch(client(), vec![], Default::default(), ()));
            assert_send_static(ml::read(client(), vec![], command(), String::new(), ()));
            assert_send_static(ml::helper_attribute(
                client(),
                vec![],
                String::new(),
                vec![],
                (),
            ));
            assert_send_static(execute_command_ignoring_errors(client(), command()));
            assert_send_static(pipelines::create_dataflow_graph(
                client(),
                vec![],
                None,
                None,
                Default::default(),
                (),
            ));
            assert_send_static(pipelines::start_run(client(), vec![], command(), ()));
        };
    }
}
