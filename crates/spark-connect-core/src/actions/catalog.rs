//! The actions of the catalog.

use std::sync::Arc;

use spark_connect_proto as proto;

use super::{Execution, ExecutionObserver};
use crate::client::SparkConnectClient;
use crate::decode::decode_arrow_batch;
use crate::error::Result;
use crate::execution;
use crate::plan::{LogicalPlan, PlanIds};
use crate::row::Row;

/// The rows of the catalog operation `catalog`, decoded by their Arrow schema rather than the
/// schema that the server sends with them.
pub async fn rows(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    catalog: proto::Catalog,
    observer: impl ExecutionObserver,
) -> Result<Vec<Row>> {
    let request = execution::execute_plan_request(
        client.session_id(),
        Vec::new(),
        execution::root(LogicalPlan::Catalog { catalog }.to_proto()),
    );
    let execution = Execution::start(client, plan_ids, request, observer).await?;
    rows_of(execution).await
}

/// The rows of `execution`, an execution of a catalog operation that has started, decoded as
/// [`rows`] decodes them. Not a stable API.
#[doc(hidden)]
pub async fn rows_of<O: ExecutionObserver>(mut execution: Execution<O>) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    while let Some(batch) = execution.next_batch().await? {
        let decoded = decode_arrow_batch(&batch);
        rows.extend(execution.failed(decoded)?);
    }
    Ok(rows)
}
