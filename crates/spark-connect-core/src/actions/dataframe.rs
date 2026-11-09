//! The actions of a DataFrame: the executions of its plan, whose result arrives as Arrow
//! batches, and the analyses of its plan.

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use spark_connect_proto as proto;

use super::{analyze, analyze_result, Execution, ExecutionObserver};
use crate::client::SparkConnectClient;
use crate::decode::{decode_arrow_batch_of, decode_arrow_record_batches};
use crate::error::{Result, SparkError};
use crate::execution;
use crate::plan::PlanIds;
use crate::row::Row;
use crate::types::DataType;

use proto::analyze_plan_request::Analyze;
use proto::analyze_plan_response::Result as R;

impl<O: ExecutionObserver> Execution<O> {
    /// The schema of the result, as the server sends it before the rows, once received.
    /// Unlike the Arrow schema of the rows, it has the user-defined types of the columns.
    pub fn schema(&self) -> Option<&DataType> {
        self.schema.as_ref()
    }

    /// The next Arrow batch of the result, or `None` at its end.
    pub(super) async fn next_batch(
        &mut self,
    ) -> Result<Option<proto::execute_plan_response::ArrowBatch>> {
        while let Some(mut response) = self.next_response().await? {
            if let Some(data_type) = response.schema.take() {
                let schema = DataType::from_proto(&data_type);
                self.schema = Some(self.failed(schema)?);
            }
            if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                response.response_type
            {
                return Ok(Some(batch));
            }
        }
        Ok(None)
    }

    /// The rows of the next Arrow batch of the result, or `None` at its end. A batch may have
    /// no rows.
    pub async fn next_rows(&mut self) -> Result<Option<Vec<Row>>> {
        let Some(batch) = self.next_batch().await? else {
            return Ok(None);
        };
        let rows = decode_arrow_batch_of(&batch, self.schema.as_ref());
        self.failed(rows).map(Some)
    }

    /// The record batches of the next Arrow batch of the result, or `None` at its end.
    pub async fn next_record_batches(&mut self) -> Result<Option<Vec<RecordBatch>>> {
        let Some(batch) = self.next_batch().await? else {
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
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        let task = runtime.spawn(async move {
            let mut sent_schema = false;
            loop {
                // Dropping the rows must end the execution also while the server has not sent
                // another response: dropping the stream releases it.
                let next = tokio::select! {
                    _ = sender.closed() => break,
                    next = self.next_rows() => next,
                };
                match next {
                    Ok(Some(rows)) => {
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
                    // The observer has heard of the end or of the failure.
                    Ok(None) => return,
                    Err(error) => {
                        let _ = sender.send(Err(error)).await;
                        return;
                    }
                }
            }
            // The caller has gone: the execution ends here for it.
            self.complete();
        });
        PrefetchedRows {
            receiver,
            task: Some(task),
        }
    }
}

/// The batches of rows of an execution that a task reads ahead; see [`Execution::prefetch`].
pub struct PrefetchedRows {
    receiver: tokio::sync::mpsc::Receiver<Result<(Option<DataType>, Vec<Row>)>>,
    /// The task, to raise its panic (of a callback of the observer, say) in the caller once
    /// its channel closes, instead of ending the rows early.
    task: Option<tokio::task::JoinHandle<()>>,
}

impl PrefetchedRows {
    /// The next batch of rows, with the schema of the result when it is new, or `None` at the
    /// end of the result. A panic of the task resumes here.
    pub async fn next(&mut self) -> Option<Result<(Option<DataType>, Vec<Row>)>> {
        let batch = self.receiver.recv().await;
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

/// The rows of the result of `request`, with its schema if the server sends it.
pub async fn collect_rows(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    request: proto::ExecutePlanRequest,
    observer: impl ExecutionObserver,
) -> Result<(Vec<Row>, Option<DataType>)> {
    let mut execution = Execution::start(client, plan_ids, request, observer).await?;
    let mut rows = Vec::new();
    while let Some(batch) = execution.next_rows().await? {
        rows.extend(batch);
    }
    Ok((rows, execution.schema))
}

/// The record batches of the result of `request`, with its schema if the server sends it.
pub async fn collect_record_batches(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    request: proto::ExecutePlanRequest,
    observer: impl ExecutionObserver,
) -> Result<(Vec<RecordBatch>, Option<DataType>)> {
    let mut execution = Execution::start(client, plan_ids, request, observer).await?;
    let mut batches = Vec::new();
    while let Some(batch) = execution.next_record_batches().await? {
        batches.extend(batch);
    }
    Ok((batches, execution.schema))
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

/// Caches `relation` at `storage_level`.
pub async fn persist(
    client: Arc<SparkConnectClient>,
    relation: proto::Relation,
    storage_level: proto::StorageLevel,
) -> Result<()> {
    let request = Analyze::Persist(proto::analyze_plan_request::Persist {
        relation: Some(relation),
        storage_level: Some(storage_level),
    });
    analyze_result(&client, request).await.map(drop)
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
