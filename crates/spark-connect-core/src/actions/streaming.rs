//! The actions of structured streaming: the start of a query, the commands of a query and of
//! the query manager, and the events of the listener bus of a session.

use std::collections::VecDeque;
use std::sync::Arc;

use spark_connect_proto as proto;

use super::{command_result, execute_command, Execution, ExecutionObserver};
use crate::client::SparkConnectClient;
use crate::error::{Result, SparkError};
use crate::execution;
use crate::plan::PlanIds;

use proto::execute_plan_response::ResponseType;

/// A streaming query that the server started.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Started {
    pub query_id: String,
    pub run_id: String,
    /// The name the server gives the query, if any.
    pub name: Option<String>,
    /// The JSON of the started event of the query, which the server sends when the session has
    /// listeners of its own, for the client to post to them.
    pub started_event_json: Option<String>,
}

/// Starts the streaming query of `start`.
pub async fn start(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    start: proto::WriteStreamOperationStart,
    observer: impl ExecutionObserver,
) -> Result<Started> {
    let command = proto::command::CommandType::WriteStreamOperationStart(start);
    let (responses, _) = execute_command(client, plan_ids, tags, command, observer).await?;
    let mut started = Started::default();
    for response in responses {
        if let Some(ResponseType::WriteStreamOperationStartResult(result)) = response.response_type
        {
            if let Some(id) = result.query_id {
                started.query_id = id.id;
                started.run_id = id.run_id;
            }
            if !result.name.is_empty() {
                started.name = Some(result.name);
            }
            started.started_event_json = result.query_started_event_json;
        }
    }
    if started.query_id.is_empty() {
        return Err(SparkError::connect_msg(
            "writeStream.start: server returned no WriteStreamOperationStartResult",
        ));
    }
    Ok(started)
}

/// Runs `command`, a command of a streaming query, and returns its result, or the default one if
/// the server sends none.
pub async fn query_command(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    command: proto::StreamingQueryCommand,
    observer: impl ExecutionObserver,
) -> Result<proto::StreamingQueryCommandResult> {
    let command = proto::command::CommandType::StreamingQueryCommand(command);
    let result = command_result(
        client,
        plan_ids,
        tags,
        command,
        observer,
        |result| match result {
            ResponseType::StreamingQueryCommandResult(result) => Some(result),
            _ => None,
        },
    );
    Ok(result.await?.unwrap_or_default())
}

/// Runs `command`, a command of the streaming query manager, and returns its result, or the
/// default one if the server sends none.
pub async fn manager_command(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    command: proto::StreamingQueryManagerCommand,
    observer: impl ExecutionObserver,
) -> Result<proto::StreamingQueryManagerCommandResult> {
    let command = proto::command::CommandType::StreamingQueryManagerCommand(command);
    let result = command_result(
        client,
        plan_ids,
        tags,
        command,
        observer,
        |result| match result {
            ResponseType::StreamingQueryManagerCommandResult(result) => Some(result),
            _ => None,
        },
    );
    Ok(result.await?.unwrap_or_default())
}

/// Unsubscribes the session from the events of its listener bus.
pub async fn unsubscribe(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    observer: impl ExecutionObserver,
) -> Result<()> {
    let command = proto::StreamingQueryListenerBusCommand {
        command: Some(
            proto::streaming_query_listener_bus_command::Command::RemoveListenerBusListener(true),
        ),
    };
    let command = proto::command::CommandType::StreamingQueryListenerBusCommand(command);
    execute_command(client, plan_ids, tags, command, observer)
        .await
        .map(drop)
}

/// The events of the streaming query listener bus of a session, as the server sends them: each
/// is its type and its JSON.
pub struct ListenerEvents<O> {
    execution: Execution<O>,
    buffered: VecDeque<(i32, String)>,
    done: bool,
    cancel: Option<tokio::sync::oneshot::Receiver<()>>,
}

impl<O: ExecutionObserver> ListenerEvents<O> {
    /// Subscribes the session to the events of its listener bus.
    pub async fn subscribe(
        client: Arc<SparkConnectClient>,
        plan_ids: Arc<PlanIds>,
        observer: O,
    ) -> Result<Self> {
        let command = proto::StreamingQueryListenerBusCommand {
            command: Some(
                proto::streaming_query_listener_bus_command::Command::AddListenerBusListener(true),
            ),
        };
        let request = execution::execute_plan_request(
            client.session_id(),
            Vec::new(),
            execution::command(
                proto::command::CommandType::StreamingQueryListenerBusCommand(command),
            ),
        );
        Ok(ListenerEvents {
            execution: Execution::start(client, plan_ids, request, observer).await?,
            buffered: VecDeque::new(),
            done: false,
            cancel: None,
        })
    }

    /// Ends the events like the end of the stream once `cancel` completes, also while a
    /// response is awaited. A response that is ready when it completes is not read.
    pub fn cancel_on(&mut self, cancel: tokio::sync::oneshot::Receiver<()>) {
        self.cancel = Some(cancel);
    }

    /// Reads to the next listener-events response and keeps its events; `Some(added)` tells
    /// whether it carries `listener_bus_listener_added`, and `None` that the stream ended.
    async fn next_response(&mut self) -> Result<Option<bool>> {
        if self.done {
            return Ok(None);
        }
        loop {
            let response = match &mut self.cancel {
                Some(cancel) => tokio::select! {
                    biased;
                    _ = cancel => Ok(None),
                    response = self.execution.next_response() => response,
                },
                None => self.execution.next_response().await,
            };
            let response = match response {
                Ok(Some(response)) => response,
                Ok(None) => {
                    self.cancel = None;
                    self.done = true;
                    return Ok(None);
                }
                Err(error) => {
                    self.done = true;
                    return Err(error);
                }
            };
            if let Some(ResponseType::StreamingQueryListenerEventsResult(result)) =
                response.response_type
            {
                self.buffered.extend(
                    result
                        .events
                        .into_iter()
                        .map(|event| (event.event_type, event.event_json)),
                );
                return Ok(Some(result.listener_bus_listener_added.is_some()));
            }
        }
    }

    /// Reads the responses until the one that tells that the server added the listener of the
    /// session (`listener_bus_listener_added`), keeping their events, which [`Self::next`] gives
    /// first, as PySpark Connect's `_register_server_side_listener` waits for it.
    pub async fn wait_until_listener_added(&mut self) -> Result<()> {
        while let Some(added) = self.next_response().await? {
            if added {
                break;
            }
        }
        Ok(())
    }

    /// The next event, or `None` once the server ends the stream.
    pub async fn next(&mut self) -> Option<Result<(i32, String)>> {
        loop {
            if let Some(event) = self.buffered.pop_front() {
                return Some(Ok(event));
            }
            if self.done {
                return None;
            }
            match self.next_response().await {
                Ok(Some(_)) => {}
                Ok(None) => return None,
                Err(error) => return Some(Err(error)),
            }
        }
    }
}
