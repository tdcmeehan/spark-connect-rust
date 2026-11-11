//! The `PipelineCommand`s of Spark Declarative Pipelines.

use std::collections::HashMap;

use spark_connect_proto as proto;

use crate::types::DataType;
use proto::pipeline_command;
use proto::pipeline_command::{
    define_flow, define_output, CreateDataflowGraph, DefineFlow, DefineOutput,
    DefineSqlGraphElements, DropDataflowGraph, StartRun,
};

/// The details needed to define one pipeline output (table/materialized-view/view/sink).
pub struct OutputSpec {
    pub name: String,
    /// `spark.connect.OutputType` enum value (MATERIALIZED_VIEW=1, TABLE=2, TEMPORARY_VIEW=3, SINK=4).
    pub output_type: i32,
    pub comment: Option<String>,
    pub source_code_location: Option<proto::SourceCodeLocation>,
    /// Present for TABLE/MATERIALIZED_VIEW.
    pub table_details: Option<TableDetailsSpec>,
    /// Present for SINK.
    pub sink_details: Option<SinkDetailsSpec>,
}

/// Table/materialized-view details.
pub struct TableDetailsSpec {
    pub table_properties: HashMap<String, String>,
    pub partition_cols: Vec<String>,
    pub clustering_columns: Vec<String>,
    pub format: Option<String>,
    /// Schema as a DDL string, or a structured type, or neither.
    pub schema_string: Option<String>,
    pub schema_data_type: Option<DataType>,
}

/// External-sink details.
pub struct SinkDetailsSpec {
    pub options: HashMap<String, String>,
    pub format: Option<String>,
}

fn wrap(cmd: pipeline_command::CommandType) -> proto::command::CommandType {
    proto::command::CommandType::PipelineCommand(proto::PipelineCommand {
        command_type: Some(cmd),
    })
}

/// The `PipelineCommand.CreateDataflowGraph` that creates a dataflow graph.
pub fn create_dataflow_graph(
    default_catalog: Option<String>,
    default_database: Option<String>,
    sql_conf: HashMap<String, String>,
) -> proto::command::CommandType {
    let inner = CreateDataflowGraph {
        default_catalog,
        default_database,
        sql_conf,
    };
    wrap(pipeline_command::CommandType::CreateDataflowGraph(inner))
}

/// The `PipelineCommand.DefineOutput` that registers a table, view or sink in a graph.
pub fn define_output(dataflow_graph_id: &str, spec: OutputSpec) -> proto::command::CommandType {
    let details = if let Some(t) = spec.table_details {
        let schema = match (t.schema_data_type, t.schema_string) {
            (Some(dt), _) => Some(define_output::table_details::Schema::SchemaDataType(
                dt.to_proto(),
            )),
            (None, Some(s)) => Some(define_output::table_details::Schema::SchemaString(s)),
            (None, None) => None,
        };
        Some(define_output::Details::TableDetails(
            define_output::TableDetails {
                table_properties: t.table_properties,
                partition_cols: t.partition_cols,
                clustering_columns: t.clustering_columns,
                format: t.format,
                schema,
            },
        ))
    } else {
        spec.sink_details.map(|s| {
            define_output::Details::SinkDetails(define_output::SinkDetails {
                options: s.options,
                format: s.format,
            })
        })
    };
    let inner = DefineOutput {
        dataflow_graph_id: Some(dataflow_graph_id.to_string()),
        output_name: Some(spec.name),
        output_type: Some(spec.output_type),
        comment: spec.comment,
        source_code_location: spec.source_code_location,
        details,
    };
    wrap(pipeline_command::CommandType::DefineOutput(inner))
}

/// The `PipelineCommand.DefineFlow` of a flow that writes `relation`.
pub fn define_flow(
    dataflow_graph_id: &str,
    flow_name: &str,
    target_dataset_name: &str,
    relation: proto::Relation,
    sql_conf: HashMap<String, String>,
    source_code_location: Option<proto::SourceCodeLocation>,
) -> proto::command::CommandType {
    flow(
        dataflow_graph_id,
        flow_name,
        target_dataset_name,
        sql_conf,
        source_code_location,
        define_flow::Details::RelationFlowDetails(define_flow::WriteRelationFlowDetails {
            relation: Some(relation),
        }),
    )
}

/// The `PipelineCommand.DefineFlow` of an Auto-CDC flow.
pub fn define_auto_cdc_flow(
    dataflow_graph_id: &str,
    flow_name: &str,
    target_dataset_name: &str,
    details: define_flow::AutoCdcFlowDetails,
    sql_conf: HashMap<String, String>,
    source_code_location: Option<proto::SourceCodeLocation>,
) -> proto::command::CommandType {
    flow(
        dataflow_graph_id,
        flow_name,
        target_dataset_name,
        sql_conf,
        source_code_location,
        define_flow::Details::AutoCdcFlowDetails(details),
    )
}

fn flow(
    dataflow_graph_id: &str,
    flow_name: &str,
    target_dataset_name: &str,
    sql_conf: HashMap<String, String>,
    source_code_location: Option<proto::SourceCodeLocation>,
    details: define_flow::Details,
) -> proto::command::CommandType {
    let inner = DefineFlow {
        dataflow_graph_id: Some(dataflow_graph_id.to_string()),
        flow_name: Some(flow_name.to_string()),
        target_dataset_name: Some(target_dataset_name.to_string()),
        sql_conf,
        client_id: None,
        source_code_location,
        once: None,
        details: Some(details),
    };
    wrap(pipeline_command::CommandType::DefineFlow(inner))
}

/// The `PipelineCommand.DefineSqlGraphElements` that registers SQL definitions.
pub fn define_sql_graph_elements(
    dataflow_graph_id: &str,
    sql_text: &str,
    sql_file_path: &str,
) -> proto::command::CommandType {
    sql_graph_elements(dataflow_graph_id, sql_text, Some(sql_file_path))
}

/// [`define_sql_graph_elements`] with the file path left out when it is `None`. Not a stable
/// API.
#[doc(hidden)]
pub fn sql_graph_elements(
    dataflow_graph_id: &str,
    sql_text: &str,
    sql_file_path: Option<&str>,
) -> proto::command::CommandType {
    let inner = DefineSqlGraphElements {
        dataflow_graph_id: Some(dataflow_graph_id.to_string()),
        sql_file_path: sql_file_path.map(str::to_string),
        sql_text: Some(sql_text.to_string()),
    };
    wrap(pipeline_command::CommandType::DefineSqlGraphElements(inner))
}

/// The `PipelineCommand.DropDataflowGraph` that removes a dataflow graph from the server. Not a
/// stable API.
#[doc(hidden)]
pub fn drop_dataflow_graph(dataflow_graph_id: &str) -> proto::command::CommandType {
    wrap(pipeline_command::CommandType::DropDataflowGraph(
        DropDataflowGraph {
            dataflow_graph_id: Some(dataflow_graph_id.to_string()),
        },
    ))
}

/// The `PipelineCommand.StartRun` that starts, or validates, a run of a graph.
pub fn start_run(
    dataflow_graph_id: &str,
    full_refresh_selection: Vec<String>,
    full_refresh_all: bool,
    refresh_selection: Vec<String>,
    dry: bool,
    storage: Option<String>,
) -> proto::command::CommandType {
    let inner = StartRun {
        dataflow_graph_id: Some(dataflow_graph_id.to_string()),
        full_refresh_selection,
        full_refresh_all: Some(full_refresh_all),
        refresh_selection,
        dry: Some(dry),
        storage,
    };
    wrap(pipeline_command::CommandType::StartRun(inner))
}
