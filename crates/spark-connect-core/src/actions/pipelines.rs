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
    command_result(client, plan_ids, tags, command, observer, graph_id)
        .await?
        .ok_or_else(|| {
            SparkError::connect_msg("CreateDataflowGraph did not return a dataflow_graph_id")
        })
}

/// The id of the dataflow graph that `result` reports the creation of. Not a stable API.
#[doc(hidden)]
pub fn graph_id(result: ResponseType) -> Option<String> {
    match result {
        ResponseType::PipelineCommandResult(proto::PipelineCommandResult {
            result_type: Some(ResultType::CreateDataflowGraphResult(graph)),
        }) => graph.dataflow_graph_id,
        _ => None,
    }
}

/// The identifier that `result` reports for the output or the flow that it defined. Not a stable
/// API.
#[doc(hidden)]
pub fn defined_identifier(result: ResponseType) -> Option<proto::ResolvedIdentifier> {
    let ResponseType::PipelineCommandResult(result) = result else {
        return None;
    };
    match result.result_type? {
        ResultType::DefineOutputResult(output) => output.resolved_identifier,
        ResultType::DefineFlowResult(flow) => flow.resolved_identifier,
        ResultType::CreateDataflowGraphResult(_) => None,
    }
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
