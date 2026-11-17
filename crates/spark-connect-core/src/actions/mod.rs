//! The actions of the client, as async functions: what a front end sends and how it reads the
//! responses. Each takes its inputs by value, so that its future is `Send + 'static`.
//!
//! An action that executes a plan chooses its operation id before it sends the request, and
//! reports the execution to an [`ExecutionObserver`] while the future runs.

pub mod session;

use std::sync::Arc;

use spark_connect_proto as proto;

use crate::arrow_chunks::ArrowChunkAssembler;
use crate::client::SparkConnectClient;
use crate::error::{Result, SparkError};
use crate::execution::{self, ExecutionInfo};

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

    /// When the execution fails after its start with `error`: the error that it fails with.
    fn on_failure(&mut self, error: SparkError) -> SparkError {
        error
    }
}

impl ExecutionObserver for () {}

/// The operation id of `request`, chosen now if it has none.
pub fn operation_id(request: &mut proto::ExecutePlanRequest) -> String {
    request
        .operation_id
        .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
        .clone()
}

/// Executes `command`, with the operation tags `tags`, to completion, and returns its responses,
/// with the chunks of each Arrow batch joined into the batch, and what it reported.
pub async fn execute_command(
    client: Arc<SparkConnectClient>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    mut observer: impl ExecutionObserver,
) -> Result<(Vec<proto::ExecutePlanResponse>, ExecutionInfo)> {
    let mut request =
        execution::execute_plan_request(client.session_id(), tags, execution::command(command));
    observer.on_start(&operation_id(&mut request));
    let mut stream = match client.execute_plan_reattachable(request).await {
        Ok(stream) => stream,
        Err(error) => return Err(observer.on_failure(error)),
    };
    let mut chunks = ArrowChunkAssembler::default();
    let mut info = ExecutionInfo::default();
    let mut responses = Vec::new();
    // The stream outlives the observer's done call: dropped unfinished, it releases the execution.
    let drained = async {
        loop {
            let response = stream.message().await?;
            if let Some(response) = &response {
                observer.on_response(response)?;
            }
            let Some(mut response) = chunks.join(response)? else {
                return Ok(());
            };
            capture(&mut response, &mut info, &mut observer)?;
            responses.push(response);
        }
    }
    .await;
    match drained {
        Ok(()) => {
            observer.on_complete(&info)?;
            Ok((responses, info))
        }
        Err(error) => Err(observer.on_failure(error)),
    }
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
