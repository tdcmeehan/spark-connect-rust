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

    /// When the execution has failed after it started, with what it reported until then.
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

/// The responses of an execution of a plan, as the server sends them. Its observer hears of the
/// start and of each response only, and the observations of the plan get no metrics.
pub struct Responses<O> {
    stream: ReattachableResponseStream,
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
        let stream = client.execute_plan_reattachable(request).await?;
        Ok(Responses { stream, observer })
    }

    /// The next response, or `None` at the end of the execution.
    pub async fn next(&mut self) -> Result<Option<proto::ExecutePlanResponse>> {
        let response = self.stream.message().await?;
        if let Some(response) = &response {
            self.observer.on_response(response);
        }
        Ok(response)
    }
}

/// An execution of a plan, whose responses are read as they arrive.
pub struct Execution<O> {
    responses: Responses<O>,
    plan_ids: Arc<PlanIds>,
    info: ExecutionInfo,
    schema: Option<DataType>,
}

impl<O: ExecutionObserver> Execution<O> {
    /// Sends `request`, after its operation id is chosen.
    pub async fn start(
        client: Arc<SparkConnectClient>,
        plan_ids: Arc<PlanIds>,
        request: proto::ExecutePlanRequest,
        observer: O,
    ) -> Result<Self> {
        Ok(Execution {
            responses: Responses::start(client, request, observer).await?,
            plan_ids,
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
        let response = self.responses.next().await;
        let Some(mut response) = self.failed(response)? else {
            self.complete();
            return Ok(None);
        };
        capture_response(
            &mut response,
            &mut self.info,
            &self.plan_ids,
            &mut self.responses.observer,
        );
        Ok(Some(response))
    }

    /// Tells the observer that the execution has ended without an error.
    fn complete(&mut self) {
        self.responses.observer.on_complete(&self.info);
    }

    /// `result`, which the observer hears of if it is an error.
    fn failed<T>(&mut self, result: Result<T>) -> Result<T> {
        if let Err(error) = &result {
            self.responses.observer.on_failure(&self.info, error);
        }
        result
    }
}

/// What an execution takes from each of its responses after the observer heard of it: `info`
/// gets what the response reports, the observations of the plans of `plan_ids` get their
/// metrics, and then the observer gets them too. Not a stable API.
#[doc(hidden)]
pub fn capture_response(
    response: &mut proto::ExecutePlanResponse,
    info: &mut ExecutionInfo,
    plan_ids: &PlanIds,
    observer: &mut impl ExecutionObserver,
) {
    let observed = info.capture(response, plan_ids);
    if !observed.is_empty() {
        observer.on_observed_metrics(observed);
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

/// Executes `command` to completion and ignores its responses and its errors, such as those of a
/// stopped session: for a command that the caller does not wait for, as PySpark sends the
/// releases of what the server caches.
pub async fn execute_command_ignoring_errors(
    client: Arc<SparkConnectClient>,
    command: proto::command::CommandType,
) {
    let plan_ids = Arc::new(PlanIds::new());
    let _ = execute_command(client, plan_ids, Vec::new(), command, ()).await;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_static<T: Send + 'static>(_: T) {}

    #[test]
    fn the_futures_of_the_actions_are_send_and_static() {
        // Type-checked, never called: it would need a connected client.
        let _ = |client: Arc<SparkConnectClient>, plan_ids: Arc<PlanIds>| {
            let client = || Arc::clone(&client);
            let plan_ids = || Arc::clone(&plan_ids);
            let command = || proto::command::CommandType::SqlCommand(Default::default());
            let relation = proto::Relation::default;
            assert_send_static(Responses::start(client(), Default::default(), ()));
            assert_send_static(Execution::start(
                client(),
                plan_ids(),
                Default::default(),
                (),
            ));
            assert_send_static(execute_command(client(), plan_ids(), vec![], command(), ()));
            assert_send_static(session::spark_version(client()));
            assert_send_static(session::parse_ddl(client(), String::new()));
            assert_send_static(session::json_to_ddl(client(), String::new()));
            assert_send_static(session::sql(client(), plan_ids(), vec![], relation(), ()));
            assert_send_static(session::create_resource_profile(
                client(),
                plan_ids(),
                vec![],
                Default::default(),
                (),
            ));
            assert_send_static(dataframe::collect_rows(
                client(),
                plan_ids(),
                Default::default(),
                (),
            ));
            assert_send_static(dataframe::collect_record_batches(
                client(),
                plan_ids(),
                Default::default(),
                (),
            ));
            assert_send_static(dataframe::schema(client(), relation()));
            assert_send_static(dataframe::explain(client(), relation(), Default::default()));
            assert_send_static(dataframe::input_files(client(), relation()));
            assert_send_static(dataframe::storage_level(client(), relation()));
            assert_send_static(dataframe::semantic_hash(client(), relation()));
            assert_send_static(dataframe::same_semantics(client(), relation(), relation()));
            assert_send_static(dataframe::is_streaming(client(), relation()));
            assert_send_static(dataframe::persist(client(), relation(), Default::default()));
            assert_send_static(dataframe::unpersist(client(), relation(), false));
            assert_send_static(catalog::rows(client(), plan_ids(), Default::default(), ()));
            assert_send_static(streaming::start(
                client(),
                plan_ids(),
                vec![],
                Default::default(),
                (),
            ));
            assert_send_static(streaming::query_command(
                client(),
                plan_ids(),
                vec![],
                Default::default(),
                (),
            ));
            assert_send_static(streaming::manager_command(
                client(),
                plan_ids(),
                vec![],
                Default::default(),
                (),
            ));
            assert_send_static(streaming::unsubscribe(client(), plan_ids(), vec![], ()));
            assert_send_static(streaming::ListenerEvents::subscribe(
                client(),
                plan_ids(),
                (),
            ));
            assert_send_static(ml::fit(
                client(),
                plan_ids(),
                vec![],
                command(),
                String::new(),
                (),
            ));
            assert_send_static(ml::evaluate(client(), plan_ids(), vec![], command(), ()));
            assert_send_static(execute_command_ignoring_errors(client(), command()));
            assert_send_static(pipelines::create_dataflow_graph(
                client(),
                plan_ids(),
                vec![],
                None,
                None,
                Default::default(),
                (),
            ));
            assert_send_static(pipelines::start_run(client(), vec![], command(), ()));
        };
        let _ = |execution: Execution<()>| assert_send_static(catalog::rows_of(execution));
    }
}
