//! The actions of the catalog.

use std::sync::Arc;

use spark_connect_proto as proto;

use super::{Execution, ExecutionObserver};
use crate::client::SparkConnectClient;
use crate::decode::decode_arrow_batch;
use crate::error::Result;
use crate::execution;
use crate::plan::LogicalPlan;
use crate::row::Row;

/// The rows of the catalog operation `catalog`, decoded by their Arrow schema rather than the
/// schema that the server sends with them.
pub async fn rows(
    client: Arc<SparkConnectClient>,
    catalog: proto::Catalog,
    observer: impl ExecutionObserver,
) -> Result<Vec<Row>> {
    let request = execution::execute_plan_request(
        client.session_id(),
        Vec::new(),
        execution::root(LogicalPlan::Catalog { catalog }.to_proto()),
    );
    let mut execution = Execution::start(client, request, observer).await?;
    let mut rows = Vec::new();
    while let Some(batch) = execution.next_batch(&|_| Ok(())).await? {
        let decoded = decode_arrow_batch(&batch);
        rows.extend(execution.failed(decoded)?);
    }
    Ok(rows)
}
