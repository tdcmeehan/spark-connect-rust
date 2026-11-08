//! The actions of Spark Declarative Pipelines that the client reads the result of: the creation
//! of a dataflow graph, and the run of a pipeline. Its other commands run as any command.

use std::collections::HashMap;
use std::sync::Arc;

use spark_connect_proto as proto;

use super::{command_result, ExecutionObserver, Responses};
use crate::client::SparkConnectClient;
use crate::error::{Result, SparkError};
use crate::execution;
use crate::pipelines;
use crate::plan::PlanIds;

use proto::execute_plan_response::ResponseType;
use proto::pipeline_command_result::ResultType;

/// Creates a dataflow graph, with these defaults, and returns the id the server gave it.
pub async fn create_dataflow_graph(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    default_catalog: Option<String>,
    default_database: Option<String>,
    sql_conf: HashMap<String, String>,
    observer: impl ExecutionObserver,
) -> Result<String> {
    let command = pipelines::create_dataflow_graph(default_catalog, default_database, sql_conf);
    command_result(
        client,
        plan_ids,
        tags,
        command,
        observer,
        |result| match result {
            ResponseType::PipelineCommandResult(proto::PipelineCommandResult {
                result_type: Some(ResultType::CreateDataflowGraphResult(graph)),
            }) => graph.dataflow_graph_id,
            _ => None,
        },
    )
    .await?
    .ok_or_else(|| {
        SparkError::connect_msg("CreateDataflowGraph did not return a dataflow_graph_id")
    })
}

/// Starts the run of a pipeline that `command` describes (see [`pipelines::start_run`]), with
/// the operation tags `tags`, and returns its responses, which the server sends as the run goes.
pub async fn start_run<O: ExecutionObserver>(
    client: Arc<SparkConnectClient>,
    tags: Vec<String>,
    command: proto::command::CommandType,
    observer: O,
) -> Result<Responses<O>> {
    let request =
        execution::execute_plan_request(client.session_id(), tags, execution::command(command));
    Responses::start(client, request, observer).await
}
