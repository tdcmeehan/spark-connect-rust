//! The executions of a session: result streams, and the start that a dropped future cancels.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::record_batch::RecordBatch;
use futures::future::BoxFuture;
use futures::{FutureExt, Stream, StreamExt};
use serde::de::DeserializeOwned;
use spark_connect_core::actions::{self, capture_response, Execution, Responses};
use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::error::SparkError;
use spark_connect_core::execution::{self as core_execution, ExecutionInfo as CoreInfo};
use spark_connect_core::types::DataType;
use spark_connect_proto as proto;

use crate::error::Result;
use crate::progress::{check_callback_depth, Observer};
use crate::session::Session;
use crate::value::Row;

/// Sends `Interrupt` for the operation if it is dropped armed: the caller stopped waiting for
/// the server to accept the execution.
struct InterruptOnDrop {
    client: Arc<SparkConnectClient>,
    operation_id: String,
    abandoned: Arc<AtomicBool>,
    runtime: tokio::runtime::Handle,
    armed: bool,
}

impl Drop for InterruptOnDrop {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.abandoned.store(true, Ordering::Release);
        let client = Arc::clone(&self.client);
        let operation_id = std::mem::take(&mut self.operation_id);
        self.runtime.spawn(async move {
            let _ = client.interrupt_operation(&operation_id).await;
        });
    }
}

/// The request of an execution of `plan` of `session`, with the handle's tags, a new operation
/// id and the session's hook applied; that operation id; and the observer of the execution,
/// which reports nothing once `abandoned` is set.
pub(crate) fn prepared(
    session: &Session,
    plan: proto::plan::OpType,
    abandoned: &Arc<AtomicBool>,
) -> Result<(proto::ExecutePlanRequest, String, Observer)> {
    check_callback_depth()?;
    let mut request = core_execution::execute_plan_request(
        session.session_id(),
        session.tags().to_vec(),
        proto::Plan {
            op_type: Some(plan),
        },
    );
    let operation_id = actions::operation_id(&mut request);
    let request = session.hooked(request)?;
    let observer = Observer::new(
        session.progress_handlers(),
        operation_id.clone(),
        Arc::clone(abandoned),
    );
    Ok((request, operation_id, observer))
}

/// Starts the execution of `plan` of `session`, its request with the handle's tags, a new
/// operation id and the session's hook applied, and returns what `start` returns once the
/// server has accepted it, with the operation id.
///
/// `start` runs as a task of its own, so that dropping the future before then interrupts the
/// operation, and the task, once the server answers, drops the response stream, which releases
/// the operation.
async fn accepted<T: Send + 'static, F>(
    session: &Session,
    plan: proto::plan::OpType,
    start: impl FnOnce(proto::ExecutePlanRequest, Observer) -> F,
) -> Result<(T, String)>
where
    F: Future<Output = spark_connect_core::error::Result<T>> + Send + 'static,
{
    let abandoned = Arc::new(AtomicBool::new(false));
    let (request, operation_id, observer) = prepared(session, plan, &abandoned)?;
    let runtime = tokio::runtime::Handle::current();
    let mut guard = InterruptOnDrop {
        client: session.client(),
        operation_id: operation_id.clone(),
        abandoned,
        runtime: runtime.clone(),
        armed: true,
    };
    let started = runtime.spawn(start(request, observer)).await;
    guard.armed = false;
    match started {
        Ok(started) => Ok((started?, operation_id)),
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => Err(SparkError::connect_msg(error.to_string()).into()),
    }
}

/// Starts the execution of the query `relation` of `session`. The future is boxed: the
/// futures of the actions nest it, and unboxed they overflowed the 2 MiB stack of a test
/// thread in a debug build.
pub(crate) fn query(
    session: Session,
    relation: proto::Relation,
) -> BoxFuture<'static, Result<RecordBatchStream>> {
    async move {
        let client = session.client();
        let plan_ids = Arc::clone(session.plan_ids());
        let (execution, operation_id) = accepted(
            &session,
            proto::plan::OpType::Root(relation),
            |request, observer| Execution::start(client, plan_ids, request, observer),
        )
        .await?;
        Ok(RecordBatchStream::new(execution, operation_id))
    }
    .boxed()
}

/// Executes the catalog operation `relation` of `session`, and returns the rows of its result.
/// The future is boxed, as that of [`query`].
pub(crate) fn catalog_rows(
    session: Session,
    relation: proto::Relation,
) -> BoxFuture<'static, Result<Vec<spark_connect_core::row::Row>>> {
    async move {
        let client = session.client();
        let plan_ids = Arc::clone(session.plan_ids());
        let (execution, _) = accepted(
            &session,
            proto::plan::OpType::Root(relation),
            |request, observer| Execution::start(client, plan_ids, request, observer),
        )
        .await?;
        Ok(actions::catalog::rows_of(execution).await?)
    }
    .boxed()
}

/// Starts the execution of `command` of `session`, and returns its responses once the server has
/// accepted it, with its operation id. The future is boxed, as that of [`query`].
pub(crate) fn command_responses(
    session: Session,
    command: proto::command::CommandType,
) -> BoxFuture<'static, Result<(Responses<Observer>, String)>> {
    async move {
        let plan = proto::plan::OpType::Command(proto::Command {
            command_type: Some(command),
        });
        let client = session.client();
        accepted(&session, plan, |request, observer| {
            Responses::start(client, request, observer)
        })
        .await
    }
    .boxed()
}

/// Executes `command` of `session` to its end, and returns the first response that `pick`
/// takes. The future is boxed, as that of [`query`].
pub(crate) fn command_result<T: Send + 'static>(
    session: Session,
    command: proto::command::CommandType,
    mut pick: impl FnMut(proto::execute_plan_response::ResponseType) -> Option<T> + Send + 'static,
) -> BoxFuture<'static, Result<Option<T>>> {
    async move {
        let (mut responses, _) = command_responses(session.clone(), command).await?;
        let mut info = CoreInfo::default();
        let mut picked = None;
        while let Some(mut response) = responses.next().await? {
            capture_response(&mut response, &mut info, session.plan_ids(), &mut ());
            if picked.is_none() {
                picked = response.response_type.and_then(&mut pick);
            }
        }
        Ok(picked)
    }
    .boxed()
}

/// Executes `command` of `session` to its end, and ignores its results.
pub(crate) fn command(
    session: Session,
    command: proto::command::CommandType,
) -> BoxFuture<'static, Result<()>> {
    command_result(session, command, |_| None::<()>)
        .map(|result| result.map(drop))
        .boxed()
}

/// Executes `command` of `session` from `runtime` without waiting for it, unless the session is
/// stopped, and ignores its result: what a value sends when it is dropped, such as the removal of
/// what the server keeps for it.
pub(crate) fn spawn_command(
    runtime: &tokio::runtime::Handle,
    session: Session,
    command: proto::command::CommandType,
) {
    if session.is_stopped() {
        return;
    }
    runtime.spawn(async move {
        let _ = self::command(session, command).await;
    });
}

type Step = (
    Box<Execution<Observer>>,
    spark_connect_core::error::Result<Option<Vec<RecordBatch>>>,
);

enum State {
    Idle(Box<Execution<Observer>>),
    Reading(BoxFuture<'static, Step>),
    Done,
}

/// The record batches of the result of a query, as the server sends them.
///
/// Dropping the stream before its end releases the operation on the server, which stops it.
/// The column names of each batch, nested ones included, are those of the schema that the
/// server sends with the result.
pub struct RecordBatchStream {
    state: State,
    batches: VecDeque<RecordBatch>,
    operation_id: String,
    schema: Option<DataType>,
    info: ExecutionInfo,
}

impl RecordBatchStream {
    fn new(execution: Execution<Observer>, operation_id: String) -> Self {
        RecordBatchStream {
            state: State::Idle(Box::new(execution)),
            batches: VecDeque::new(),
            operation_id,
            schema: None,
            info: ExecutionInfo::default(),
        }
    }

    /// The id of the operation on the server.
    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }

    /// The schema of the result, with the types of the columns as Spark has them, as of the
    /// last read of the stream that finished. It is `None` until a read has received it.
    pub fn schema(&self) -> Option<&DataType> {
        self.schema.as_ref()
    }

    /// What the execution reported up to the last read of the stream that finished, such as
    /// its metrics, which the server sends at the end. A read still in progress adds nothing.
    pub fn execution_info(&self) -> &ExecutionInfo {
        &self.info
    }

    fn finished_read(&mut self, execution: &Execution<Observer>) {
        if self.schema.as_ref() != execution.schema() {
            self.schema = execution.schema().cloned();
        }
        self.info = ExecutionInfo(execution.info().clone());
    }

    pub(crate) fn rows_as<T: DeserializeOwned + Send + 'static>(
        self,
    ) -> impl Stream<Item = Result<T>> + Send + 'static {
        self.flat_map(|batch| {
            match batch.and_then(|batch| crate::de::from_record_batch::<T>(&batch)) {
                Ok(rows) => futures::stream::iter(rows.into_iter().map(Ok)).left_stream(),
                Err(error) => futures::stream::iter([Err(error)]).right_stream(),
            }
        })
    }
}

impl Stream for RecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if let Some(batch) = this.batches.pop_front() {
                return Poll::Ready(Some(Ok(batch)));
            }
            match std::mem::replace(&mut this.state, State::Done) {
                State::Idle(mut execution) => {
                    this.state = State::Reading(
                        async move {
                            let step = execution.next_record_batches().await;
                            (execution, step)
                        }
                        .boxed(),
                    );
                }
                State::Reading(mut reading) => match reading.poll_unpin(cx) {
                    Poll::Pending => {
                        this.state = State::Reading(reading);
                        return Poll::Pending;
                    }
                    Poll::Ready((execution, step)) => {
                        this.finished_read(&execution);
                        match step {
                            Ok(Some(batches)) => {
                                for batch in batches {
                                    match crate::de::with_names_of(batch, this.schema.as_ref()) {
                                        Ok(batch) => this.batches.push_back(batch),
                                        Err(error) => return Poll::Ready(Some(Err(error))),
                                    }
                                }
                                this.state = State::Idle(execution);
                            }
                            Ok(None) => return Poll::Ready(None),
                            Err(error) => return Poll::Ready(Some(Err(error.into()))),
                        }
                    }
                },
                State::Done => return Poll::Ready(None),
            }
        }
    }
}

/// What an execution reported: its operation id, the metrics of its plan, and the metrics
/// that its observations collected.
#[derive(Debug, Clone, Default)]
pub struct ExecutionInfo(CoreInfo);

impl ExecutionInfo {
    /// The id of the operation on the server.
    pub fn operation_id(&self) -> Option<&str> {
        self.0.operation_id.as_deref()
    }

    /// The metrics of the plan's nodes, as the server sends them at the end of the execution.
    pub fn plan_metrics(&self) -> Option<&proto::execute_plan_response::Metrics> {
        self.0.metrics.as_ref()
    }

    /// The metrics of each observation of the plan, by its name, as a row of its metrics.
    pub fn observed_metrics(&self) -> Vec<(String, Row)> {
        self.0
            .observed_metrics
            .iter()
            .map(|observed| {
                (
                    observed.name.clone(),
                    crate::value::row_of_metrics(&observed.keys, &observed.values),
                )
            })
            .collect()
    }
}
