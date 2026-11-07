//! Spark Declarative Pipelines (SDP) command execution.
//!
//! Mirrors the Connect client's `pyspark.pipelines.spark_connect_pipeline` +
//! `spark_connect_graph_element_registry`, which build `PipelineCommand` protos and
//! execute them via the low-level client. Here the proto construction and execution
//! live in Rust; the thin Python `pyspark.pipelines` connect glue calls these instead
//! of assembling protos itself.

use std::collections::HashMap;

use spark_connect_core::client::ReattachableResponseStream;
use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::execution;
pub use spark_connect_core::pipelines::{OutputSpec, SinkDetailsSpec, TableDetailsSpec};
use spark_connect_core::runtime::block_on;
use spark_connect_proto as proto;

use crate::dataframe::{execute_command_collect, DataFrame};
use crate::session::SparkSession;

use proto::pipeline_command::define_flow;

/// `PipelineCommand.CreateDataflowGraph` — returns the server-assigned graph id.
pub fn create_dataflow_graph(
    session: &SparkSession,
    default_catalog: Option<String>,
    default_database: Option<String>,
    sql_conf: HashMap<String, String>,
) -> Result<String> {
    let command = spark_connect_core::pipelines::create_dataflow_graph(
        default_catalog,
        default_database,
        sql_conf,
    );
    let responses = execute_command_collect(session, command)?;
    for resp in responses {
        if let Some(proto::execute_plan_response::ResponseType::PipelineCommandResult(r)) =
            resp.response_type
        {
            if let Some(proto::pipeline_command_result::ResultType::CreateDataflowGraphResult(g)) =
                r.result_type
            {
                if let Some(id) = g.dataflow_graph_id {
                    return Ok(id);
                }
            }
        }
    }
    Err(SparkError::connect_msg(
        "CreateDataflowGraph did not return a dataflow_graph_id",
    ))
}

/// `PipelineCommand.DefineOutput` — register a table/view/sink in the graph.
pub fn define_output(
    session: &SparkSession,
    dataflow_graph_id: &str,
    spec: OutputSpec,
) -> Result<()> {
    let command = spark_connect_core::pipelines::define_output(dataflow_graph_id, spec);
    execute_command_collect(session, command)?;
    Ok(())
}

/// `PipelineCommand.DefineFlow` with a relation body — the common query-defined flow.
pub fn define_flow(
    session: &SparkSession,
    dataflow_graph_id: &str,
    flow_name: &str,
    target_dataset_name: &str,
    relation_df: &DataFrame,
    sql_conf: HashMap<String, String>,
    source_code_location: Option<proto::SourceCodeLocation>,
) -> Result<()> {
    let command = spark_connect_core::pipelines::define_flow(
        dataflow_graph_id,
        flow_name,
        target_dataset_name,
        relation_df.relation(),
        sql_conf,
        source_code_location,
    );
    execute_command_collect(session, command)?;
    Ok(())
}

/// `PipelineCommand.DefineFlow` carrying an Auto-CDC flow body.
#[allow(clippy::too_many_arguments)]
pub fn define_auto_cdc_flow(
    session: &SparkSession,
    dataflow_graph_id: &str,
    flow_name: &str,
    target_dataset_name: &str,
    details: define_flow::AutoCdcFlowDetails,
    sql_conf: HashMap<String, String>,
    source_code_location: Option<proto::SourceCodeLocation>,
) -> Result<()> {
    let command = spark_connect_core::pipelines::define_auto_cdc_flow(
        dataflow_graph_id,
        flow_name,
        target_dataset_name,
        details,
        sql_conf,
        source_code_location,
    );
    execute_command_collect(session, command)?;
    Ok(())
}

/// `PipelineCommand.DefineSqlGraphElements` — register raw SQL definitions.
pub fn define_sql_graph_elements(
    session: &SparkSession,
    dataflow_graph_id: &str,
    sql_text: &str,
    sql_file_path: &str,
) -> Result<()> {
    let command = spark_connect_core::pipelines::define_sql_graph_elements(
        dataflow_graph_id,
        sql_text,
        sql_file_path,
    );
    execute_command_collect(session, command)?;
    Ok(())
}

/// A lazily-consumed stream of a pipeline run's responses (`PipelineCommandResult` and
/// `PipelineEventResult`), so callers observe events — and any server error — as they
/// arrive, matching `handle_pipeline_events` iterating `execute_command_as_iterator`.
///
/// The gRPC stream is opened on the FIRST `next_response`, not at construction, so that a
/// fail-fast validation error (e.g. a dry run rejecting an invalid graph) surfaces during
/// iteration rather than from `start_run` itself — mirroring the reference client, whose
/// `execute_command_as_iterator` is a generator that starts the RPC lazily.
pub struct PipelineRunStream {
    session: SparkSession,
    request: Option<proto::ExecutePlanRequest>,
    stream: Option<ReattachableResponseStream>,
}

impl PipelineRunStream {
    /// The next response, or `None` when the run's stream completes. A server-side
    /// failure surfaces here as the mapped `SparkError`.
    pub fn next_response(&mut self) -> Result<Option<proto::ExecutePlanResponse>> {
        if self.stream.is_none() {
            let request = self
                .request
                .take()
                .expect("pipeline run stream already exhausted its request");
            self.stream = Some(block_on(
                self.session.client().execute_plan_reattachable(request),
            )?);
        }
        block_on(self.stream.as_mut().unwrap().message())
    }
}

/// `PipelineCommand.StartRun` — begin (or dry-run/validate) a pipeline run and return the
/// (lazily-opened) live response stream.
#[allow(clippy::too_many_arguments)]
pub fn start_run(
    session: &SparkSession,
    dataflow_graph_id: &str,
    full_refresh_selection: Vec<String>,
    full_refresh_all: bool,
    refresh_selection: Vec<String>,
    dry: bool,
    storage: Option<String>,
) -> Result<PipelineRunStream> {
    let command = spark_connect_core::pipelines::start_run(
        dataflow_graph_id,
        full_refresh_selection,
        full_refresh_all,
        refresh_selection,
        dry,
        storage,
    );
    let request = execution::execute_plan_request(
        session.client().session_id(),
        session.tags(),
        execution::command(command),
    );
    Ok(PipelineRunStream {
        session: session.clone(),
        request: Some(request),
        stream: None,
    })
}
