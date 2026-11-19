//! The actions of a DataFrame: the executions of its plan, whose result arrives as Arrow
//! batches, and the analyses of its plan.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use spark_connect_proto as proto;

use super::{analyze, analyze_result, boxed, Execution, ExecutionObserver};
use crate::client::{wait_aborted, InflightHandle, SparkConnectClient};
use crate::decode::{decode_arrow_batch_of, decode_arrow_record_batches, share_schema};
use crate::error::{Result, SparkError};
use crate::execution::{self, ExecutionInfo};
use crate::row::Row;
use crate::runtime::{until_abort, ABORT_RETRIES};
use crate::types::DataType;

use proto::analyze_plan_request::Analyze;
use proto::analyze_plan_response::Result as R;

impl<O: ExecutionObserver> Execution<O> {
    /// The schema of the result, as the server sends it before the rows, once received.
    /// Unlike the Arrow schema of the rows, it has the user-defined types of the columns.
    pub fn schema(&self) -> Option<&DataType> {
        self.schema.as_ref()
    }

    /// The next Arrow batch of the result, or `None` at its end. `check` runs on the schema of
    /// the result when the server sends it, before any later response is read. An error of the
    /// schema or of `check` is the inner error, which the observer has not heard of.
    async fn next_unchecked_batch(
        &mut self,
        check: &impl Fn(&DataType) -> Result<()>,
    ) -> Result<Option<Result<proto::execute_plan_response::ArrowBatch>>> {
        loop {
            let message = boxed(|| self.stream.message()).await;
            let Some(response) = self.response(message)? else {
                return Ok(None);
            };
            if let Some(batch) = self.take_batch(response, check) {
                return Ok(Some(batch));
            }
        }
    }

    /// Keeps the schema of `response`, if it has one, once `check` accepts it, and returns its Arrow
    /// batch, if it has one, or the error of a schema that fails to decode or that `check` rejects.
    fn take_batch(
        &mut self,
        mut response: proto::ExecutePlanResponse,
        check: &impl Fn(&DataType) -> Result<()>,
    ) -> Option<Result<proto::execute_plan_response::ArrowBatch>> {
        if let Some(data_type) = response.schema.take() {
            let schema = match DataType::from_proto(&data_type) {
                Ok(schema) => schema,
                Err(error) => return Some(Err(error)),
            };
            if let Err(error) = check(&schema) {
                return Some(Err(error));
            }
            self.schema = Some(schema);
        }
        if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
            response.response_type
        {
            return Some(Ok(batch));
        }
        None
    }

    /// As [`Execution::next_unchecked_batch`], with its inner error one of the execution.
    pub(super) async fn next_batch(
        &mut self,
        check: &impl Fn(&DataType) -> Result<()>,
    ) -> Result<Option<proto::execute_plan_response::ArrowBatch>> {
        match self.next_unchecked_batch(check).await? {
            Some(batch) => self.failed(batch).map(Some),
            None => Ok(None),
        }
    }

    /// The rows of the next Arrow batch of the result, or `None` at its end. A batch may have
    /// no rows.
    pub async fn next_rows(&mut self) -> Result<Option<Vec<Row>>> {
        let Some(batch) = self.next_batch(&|_| Ok(())).await? else {
            return Ok(None);
        };
        let rows = decode_arrow_batch_of(&batch, self.schema.as_ref());
        self.failed(rows).map(Some)
    }

    /// The record batches of the next Arrow batch of the result, with their schema when they
    /// were read with their fields nullable (see [`share_schema`]), or `None` at its end.
    pub async fn next_record_batches(
        &mut self,
    ) -> Result<Option<(Vec<RecordBatch>, Option<SchemaRef>)>> {
        self.next_checked_record_batches(&|_| Ok(())).await
    }

    async fn next_checked_record_batches(
        &mut self,
        check: &impl Fn(&DataType) -> Result<()>,
    ) -> Result<Option<(Vec<RecordBatch>, Option<SchemaRef>)>> {
        let Some(batch) = self.next_batch(check).await? else {
            return Ok(None);
        };
        let batches = decode_arrow_record_batches(&batch);
        self.failed(batches).map(Some)
    }
}

impl<O: ExecutionObserver + 'static> Execution<O> {
    /// Reads the result on a task of `runtime`, one batch of rows ahead of the caller, until
    /// the caller drops the returned rows. Each batch carries the schema of the result while it
    /// is new.
    pub fn prefetch(mut self, runtime: &tokio::runtime::Handle) -> PrefetchedRows {
        let wait = self.stream.inflight_handle();
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        let task = runtime.spawn(async move {
            let mut sent_schema = false;
            loop {
                // Dropping the rows must end the execution also while the server has not sent
                // another response: dropping the stream releases it.
                let next = tokio::select! {
                    _ = sender.closed() => break,
                    next = self.next_prefetched() => next,
                };
                match next {
                    Ok(Some(Ok(rows))) => {
                        let schema = if sent_schema {
                            None
                        } else {
                            self.schema.clone()
                        };
                        sent_schema |= schema.is_some();
                        if sender.send(Ok((schema, rows))).await.is_err() {
                            break;
                        }
                    }
                    Ok(Some(Err(error))) => {
                        let _ = sender.send(Err(error)).await;
                        break;
                    }
                    // The observer has heard of the end or of the failure.
                    Ok(None) => return,
                    Err(error) => {
                        let _ = sender.send(Err(error)).await;
                        return;
                    }
                }
            }
            self.observer.on_abandoned(&self.info);
        });
        PrefetchedRows {
            receiver,
            task: Some(task),
            wait,
        }
    }

    /// As [`Execution::next_rows`], but a schema or a batch that fails to decode is the inner
    /// error, which the observer does not hear of: the rows end with that error, and the
    /// execution ends unfinished, its done call made when it drops.
    async fn next_prefetched(&mut self) -> Result<Option<Result<Vec<Row>>>> {
        let batch = self.next_unchecked_batch(&|_| Ok(())).await?;
        Ok(batch.map(|batch| decode_arrow_batch_of(&batch?, self.schema.as_ref())))
    }
}

/// The batches of rows of an execution that a task reads ahead; see [`Execution::prefetch`].
pub struct PrefetchedRows {
    receiver: tokio::sync::mpsc::Receiver<Result<(Option<DataType>, Vec<Row>)>>,
    /// The task, to raise its panic (of a callback of the observer, say) in the caller once
    /// its channel closes, instead of ending the rows early.
    task: Option<tokio::task::JoinHandle<()>>,
    /// The execution, which the caller waits for while it waits for the task's batches.
    wait: InflightHandle,
}

impl PrefetchedRows {
    /// The next batch of rows, with the schema of the result when it is new, or `None` at the
    /// end of the result. A panic of the task resumes here.
    pub async fn next(&mut self) -> Option<Result<(Option<DataType>, Vec<Row>)>> {
        // The caller waits for the batch of the task, which is a wait for the execution.
        // The task has no wait of its own that a signal aborts, so the caller ends its
        // wait for the batch at the first signal, as PySpark's retry sleeps do, and
        // dropping the receiver stops the task.
        let batch = {
            let _waiting = self.wait.waiting();
            until_abort(ABORT_RETRIES, self.receiver.recv()).await
        };
        let Some(batch) = batch else {
            return Some(Err(wait_aborted()));
        };
        if batch.is_none() {
            if let Some(task) = self.task.take() {
                if let Err(error) = task.await {
                    if error.is_panic() {
                        std::panic::resume_unwind(error.into_panic());
                    }
                }
            }
        }
        batch
    }
}

/// The rows of the result of `request`, with its schema if the server sends it and what the
/// execution reported. `check` runs on
/// the schema as soon as the server sends it, before the rows and any later response are read:
/// its error ends the execution. A string that is not valid UTF-8 fails the result with the error
/// of the first column that has one, once the rest of the result is read, as PySpark converts a
/// result column by column.
pub async fn collect_rows(
    client: Arc<SparkConnectClient>,
    request: proto::ExecutePlanRequest,
    observer: impl ExecutionObserver,
    check: impl Fn(&DataType) -> Result<()>,
) -> Result<(Vec<Row>, Option<DataType>, ExecutionInfo)> {
    let mut execution = Execution::start(client, request, observer).await?;
    let mut rows = Vec::new();
    let mut invalid_utf8: Option<SparkError> = None;
    while let Some(batch) = execution.next_batch(&check).await? {
        match decode_arrow_batch_of(&batch, execution.schema.as_ref()) {
            Ok(batch_rows) => rows.extend(batch_rows),
            Err(error) if error.utf8_column().is_some() => {
                let first = invalid_utf8
                    .as_ref()
                    .is_some_and(|first| first.utf8_column() <= error.utf8_column());
                if !first {
                    invalid_utf8 = Some(error);
                }
            }
            Err(error) => return execution.failed(Err(error)),
        }
    }
    if let Some(error) = invalid_utf8 {
        return Err(error);
    }
    Ok((rows, execution.schema, execution.info))
}

/// The record batches of the result of `request`, which share one schema, with the schema of the
/// result if the server sends it and what the execution reported. `check` runs on the schema as
/// [`collect_rows`] runs it.
pub async fn collect_record_batches(
    client: Arc<SparkConnectClient>,
    request: proto::ExecutePlanRequest,
    observer: impl ExecutionObserver,
    check: impl Fn(&DataType) -> Result<()>,
) -> Result<(Vec<RecordBatch>, Option<DataType>, ExecutionInfo)> {
    let mut execution = Execution::start(client, request, observer).await?;
    let mut batches = Vec::new();
    let mut relaxed = None;
    while let Some((record_batches, read)) = execution.next_checked_record_batches(&check).await? {
        batches.extend(record_batches);
        relaxed = read.or(relaxed);
    }
    Ok((
        share_schema(batches, relaxed)?,
        execution.schema,
        execution.info,
    ))
}

/// The schema of the relation `relation`, a struct.
pub async fn schema(
    client: Arc<SparkConnectClient>,
    relation: proto::Relation,
) -> Result<DataType> {
    let result = analyze_result(
        &client,
        Analyze::Schema(proto::analyze_plan_request::Schema {
            plan: Some(execution::root(relation)),
        }),
    )
    .await?;
    let Some(R::Schema(schema)) = result else {
        return Err(SparkError::connect_msg(
            "Schema analyze failed: no schema in response",
        ));
    };
    let schema = DataType::from_proto(
        &schema
            .schema
            .ok_or_else(|| SparkError::connect_msg("Schema is missing"))?,
    )?;
    if !matches!(&schema, DataType::Struct { .. }) {
        return Err(SparkError::connect_msg("Schema is not a struct type"));
    }
    Ok(schema)
}

/// The plan of `relation` as the server explains it in `mode`.
pub async fn explain(
    client: Arc<SparkConnectClient>,
    relation: proto::Relation,
    mode: proto::analyze_plan_request::explain::ExplainMode,
) -> Result<String> {
    let request = Analyze::Explain(proto::analyze_plan_request::Explain {
        plan: Some(execution::root(relation)),
        explain_mode: mode as i32,
    });
    analyze(
        &client,
        request,
        "an explain result",
        |result| match result {
            R::Explain(explain) => Some(Ok(explain.explain_string)),
            _ => None,
        },
    )
    .await
}

/// The files that `relation` reads, or none if the server gives none.
pub async fn input_files(
    client: Arc<SparkConnectClient>,
    relation: proto::Relation,
) -> Result<Vec<String>> {
    let request = Analyze::InputFiles(proto::analyze_plan_request::InputFiles {
        plan: Some(execution::root(relation)),
    });
    Ok(match analyze_result(&client, request).await? {
        Some(R::InputFiles(files)) => files.files,
        _ => Vec::new(),
    })
}

/// The storage level that `relation` is cached at, or the default if the server gives none.
pub async fn storage_level(
    client: Arc<SparkConnectClient>,
    relation: proto::Relation,
) -> Result<proto::StorageLevel> {
    let request = Analyze::GetStorageLevel(proto::analyze_plan_request::GetStorageLevel {
        relation: Some(relation),
    });
    Ok(match analyze_result(&client, request).await? {
        Some(R::GetStorageLevel(level)) => level.storage_level.unwrap_or_default(),
        _ => proto::StorageLevel::default(),
    })
}

/// The semantic hash of the plan of `relation`.
pub async fn semantic_hash(
    client: Arc<SparkConnectClient>,
    relation: proto::Relation,
) -> Result<i32> {
    let request = Analyze::SemanticHash(proto::analyze_plan_request::SemanticHash {
        plan: Some(execution::root(relation)),
    });
    analyze(&client, request, "a semantic hash", |result| match result {
        R::SemanticHash(hash) => Some(Ok(hash.result)),
        _ => None,
    })
    .await
}

/// Whether the plans of `relation` and `other` have the same semantics.
pub async fn same_semantics(
    client: Arc<SparkConnectClient>,
    relation: proto::Relation,
    other: proto::Relation,
) -> Result<bool> {
    let request = Analyze::SameSemantics(proto::analyze_plan_request::SameSemantics {
        target_plan: Some(execution::root(relation)),
        other_plan: Some(execution::root(other)),
    });
    analyze(
        &client,
        request,
        "a sameSemantics result",
        |result| match result {
            R::SameSemantics(same) => Some(Ok(same.result)),
            _ => None,
        },
    )
    .await
}

/// Whether a source of the plan of `relation` returns data continuously.
pub async fn is_streaming(
    client: Arc<SparkConnectClient>,
    relation: proto::Relation,
) -> Result<bool> {
    let request = Analyze::IsStreaming(proto::analyze_plan_request::IsStreaming {
        plan: Some(execution::root(relation)),
    });
    analyze(
        &client,
        request,
        "an isStreaming result",
        |result| match result {
            R::IsStreaming(streaming) => Some(Ok(streaming.is_streaming)),
            _ => None,
        },
    )
    .await
}

/// Whether the server answers that the plan of `relation`, as it analyzes it, is local.
pub async fn is_local(client: Arc<SparkConnectClient>, relation: proto::Relation) -> Result<bool> {
    let request = Analyze::IsLocal(proto::analyze_plan_request::IsLocal {
        plan: Some(execution::root(relation)),
    });
    analyze(
        &client,
        request,
        "an isLocal result",
        |result| match result {
            R::IsLocal(local) => Some(Ok(local.is_local)),
            _ => None,
        },
    )
    .await
}

/// Caches `relation` at `storage_level`, or, of `None`, at the server's default.
pub async fn persist(
    client: Arc<SparkConnectClient>,
    relation: proto::Relation,
    storage_level: Option<proto::StorageLevel>,
) -> Result<()> {
    let request = persist_request(client.session_id(), relation, storage_level);
    boxed(|| client.analyze_plan(request)).await.map(drop)
}

/// The request of [`persist`] in the session `session_id`, of no storage level for `None`. Not
/// a stable API.
#[doc(hidden)]
pub fn persist_request(
    session_id: &str,
    relation: proto::Relation,
    storage_level: Option<proto::StorageLevel>,
) -> proto::AnalyzePlanRequest {
    let persist = proto::analyze_plan_request::Persist {
        relation: Some(relation),
        storage_level,
    };
    execution::analyze_plan_request(session_id, Analyze::Persist(persist))
}

/// Removes `relation` from the cache, waiting for it if `blocking`.
pub async fn unpersist(
    client: Arc<SparkConnectClient>,
    relation: proto::Relation,
    blocking: bool,
) -> Result<()> {
    let request = Analyze::Unpersist(proto::analyze_plan_request::Unpersist {
        relation: Some(relation),
        blocking: Some(blocking),
    });
    analyze_result(&client, request).await.map(drop)
}
