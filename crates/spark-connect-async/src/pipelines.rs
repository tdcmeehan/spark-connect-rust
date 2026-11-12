//! Spark Declarative Pipelines: a dataflow graph of datasets and of the flows that fill them,
//! which the server runs.
//!
//! [`Session::create_dataflow_graph`] creates a [`DataflowGraph`] on the server. Its datasets and
//! flows are defined from SQL text or from DataFrames, and [`DataflowGraph::start_run`] starts a
//! run, whose events stream back as the server runs the flows.
//!
//! ```no_run
//! # async fn run(session: spark_connect_async::Session) -> spark_connect_async::Result<()> {
//! use futures::StreamExt;
//!
//! let graph = session.create_dataflow_graph().await?;
//! graph
//!     .define_sql("CREATE MATERIALIZED VIEW numbers AS SELECT id FROM range(10)")
//!     .await?;
//! graph.define_materialized_view("first_numbers").await?;
//! let first = session.table("numbers").limit(3);
//! graph.define_flow("first_numbers", "first_numbers", &first).await?;
//! let mut run = graph.start_run("file:///tmp/pipeline-storage").await?;
//! while let Some(event) = run.next().await {
//!     let event = event?;
//!     println!("{} {}", event.timestamp, event.message);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::future::IntoFuture;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use futures::{FutureExt, Stream};
use spark_connect_core::actions::{pipelines as core_actions, Responses};
use spark_connect_core::pipelines::{self as core, OutputSpec, SinkDetailsSpec, TableDetailsSpec};
use spark_connect_core::plan::{relation_with_ids, LogicalPlan};
use spark_connect_proto as proto;

use crate::column::{expressions, Column, IntoDataType, TypeOrDdl};
use crate::dataframe::{strings, DataFrame};
use crate::error::{Error, Result};
use crate::progress::Observer;
use crate::readwriter::{insert_options, OptionValue};
use crate::session::Session;
use crate::stream;

use proto::execute_plan_response::ResponseType;
use proto::pipeline_command::define_flow::AutoCdcFlowDetails;

impl Session {
    /// A new dataflow graph on the server, to define the datasets and flows of a pipeline in;
    /// await it to create the graph.
    pub fn create_dataflow_graph(&self) -> CreateDataflowGraph {
        CreateDataflowGraph {
            session: self.clone(),
            default_catalog: None,
            default_database: None,
            sql_conf: HashMap::new(),
        }
    }
}

/// The creation of the graph of [`Session::create_dataflow_graph`]: await it to create the
/// graph.
#[must_use = "a dataflow graph is not created until it is awaited"]
pub struct CreateDataflowGraph {
    session: Session,
    default_catalog: Option<String>,
    default_database: Option<String>,
    sql_conf: HashMap<String, String>,
}

impl CreateDataflowGraph {
    /// The catalog of the datasets whose names name none; by default, the current catalog of the
    /// session. The server also makes it the current catalog of the session.
    pub fn default_catalog(mut self, catalog: impl Into<String>) -> Self {
        self.default_catalog = Some(catalog.into());
        self
    }

    /// The database of the datasets whose names name none; by default, the current database of
    /// the session. The server also makes it the current database of the session.
    pub fn default_database(mut self, database: impl Into<String>) -> Self {
        self.default_database = Some(database.into());
        self
    }

    /// Sets a SQL configuration for every flow of the graph.
    pub fn sql_conf(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.sql_conf.insert(key.into(), value.into());
        self
    }
}

impl IntoFuture for CreateDataflowGraph {
    type Output = Result<DataflowGraph>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let command =
            core::create_dataflow_graph(self.default_catalog, self.default_database, self.sql_conf);
        let session = self.session;
        async move {
            let id = stream::command_result(session.clone(), command, core_actions::graph_id)
                .await?
                .ok_or_else(|| Error::decode("the server returned no dataflow graph"))?;
            Ok(DataflowGraph::new(id, session))
        }
        .boxed()
    }
}

/// A dataflow graph on the server: the datasets of a pipeline and the flows that fill them.
///
/// A `DataflowGraph` is a handle: its clones share the graph. Once the last of them is dropped,
/// and the last of the definitions in progress and of the runs that keep the graph, the graph
/// is removed from the server, in the background on the runtime that created it. A server keeps
/// the graphs of a session at most until the session ends.
#[derive(Clone)]
pub struct DataflowGraph {
    kept: Arc<Kept>,
}

/// The graph that the server keeps, which the last handle to drop removes.
struct Kept {
    id: String,
    /// Dropped before `plans`, so the graph is removed before what its flows refer to.
    removal: stream::RemoveOnDrop,
    /// The plans of the DataFrames that the flows of the graph were defined from. The server
    /// reads them when a run runs the flows, so the graph keeps what they refer to, such as the
    /// fitted models and the checkpoints that the server keeps for them.
    plans: Mutex<Vec<Arc<LogicalPlan>>>,
}

impl fmt::Debug for DataflowGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataflowGraph")
            .field("id", &self.kept.id)
            .finish_non_exhaustive()
    }
}

impl DataflowGraph {
    fn new(id: String, session: Session) -> Self {
        DataflowGraph {
            kept: Arc::new(Kept {
                removal: stream::RemoveOnDrop::new(session, core::drop_dataflow_graph(&id)),
                id,
                plans: Mutex::new(Vec::new()),
            }),
        }
    }

    /// The id of the graph on the server.
    pub fn id(&self) -> &str {
        &self.kept.id
    }

    /// The session of the graph.
    pub fn session(&self) -> &Session {
        self.kept.removal.session()
    }

    fn keep(&self, plan: Arc<LogicalPlan>) {
        self.kept
            .plans
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(plan);
    }

    fn output(&self, name: impl Into<String>, output_type: proto::OutputType) -> OutputSpec {
        OutputSpec {
            name: name.into(),
            output_type: output_type as i32,
            comment: None,
            source_code_location: None,
            table_details: None,
            sink_details: None,
        }
    }

    /// The definition of the materialized view `name`, a table that the server recomputes from
    /// the flows that target it at each run; await it to define it. A name without a catalog or
    /// a database is in those of the graph.
    pub fn define_materialized_view(&self, name: impl Into<String>) -> DefineTable {
        DefineTable::new(self, self.output(name, proto::OutputType::MaterializedView))
    }

    /// The definition of the streaming table `name`, a table that the flows that target it
    /// append to with streaming queries; await it to define it. A name without a catalog or a
    /// database is in those of the graph.
    pub fn define_streaming_table(&self, name: impl Into<String>) -> DefineTable {
        DefineTable::new(self, self.output(name, proto::OutputType::Table))
    }

    /// The definition of the temporary view `name`, which the flows of the graph can read but
    /// which no catalog has; await it to define it.
    pub fn define_temporary_view(&self, name: impl Into<String>) -> DefineView {
        DefineView {
            graph: self.clone(),
            spec: self.output(name, proto::OutputType::TemporaryView),
        }
    }

    /// The definition of the sink `name`, which the flows that target it write to with a
    /// streaming query of the data source `format`, such as `kafka`; await it to define it. No
    /// catalog has a sink.
    pub fn define_sink(&self, name: impl Into<String>, format: impl Into<String>) -> DefineSink {
        DefineSink {
            graph: self.clone(),
            spec: self.output(name, proto::OutputType::Sink),
            details: SinkDetailsSpec {
                options: HashMap::new(),
                format: Some(format.into()),
            },
        }
    }

    /// The definition of the flow `name`, which writes the rows of `query` to the dataset
    /// `target` at each run; await it to define it. The name has one part, without a catalog or
    /// a database. The query is a [`DataFrame`], or the text of a SQL query, and it can read the
    /// other datasets of the graph by their names. The graph keeps the plan of the DataFrame, and
    /// so what the server keeps for it, such as fitted models, while the graph lives.
    pub fn define_flow(
        &self,
        name: impl Into<String>,
        target: impl Into<String>,
        query: impl Into<FlowQuery>,
    ) -> DefineFlow {
        DefineFlow {
            graph: self.clone(),
            name: name.into(),
            target: target.into(),
            query: query.into(),
            sql_conf: HashMap::new(),
            source_code_location: None,
        }
    }

    /// The definition of the flow `name` that applies the changes of the source `source` to
    /// the streaming table `target`: each change upserts or deletes the row of its `keys`, and
    /// `sequence_by` orders the changes of a row. Await it to define it.
    pub fn define_auto_cdc_flow(
        &self,
        name: impl Into<String>,
        target: impl Into<String>,
        source: impl Into<String>,
        keys: impl IntoIterator<Item = impl Into<Column>>,
        sequence_by: impl Into<Column>,
    ) -> DefineAutoCdcFlow {
        DefineAutoCdcFlow {
            graph: self.clone(),
            name: name.into(),
            target: target.into(),
            details: AutoCdcFlowDetails {
                source: Some(source.into()),
                keys: protos(expressions(keys)),
                sequence_by: Some(sequence_by.into().expression().to_proto()),
                ..Default::default()
            },
            sql_conf: HashMap::new(),
            source_code_location: None,
        }
    }

    /// The definition of the datasets and flows of the SQL text `sql_text`, whose statements
    /// create them, such as `CREATE MATERIALIZED VIEW`, `CREATE STREAMING TABLE`,
    /// `CREATE TEMPORARY VIEW` and `CREATE FLOW`; await it to define them.
    pub fn define_sql(&self, sql_text: impl Into<String>) -> DefineSql {
        DefineSql {
            graph: self.clone(),
            sql_text: sql_text.into(),
            file_path: None,
        }
    }

    /// A run of the graph, which updates each of its datasets from the flows that target it,
    /// and keeps its checkpoints and metadata in `storage`, a URI with a scheme, such as
    /// `file:///`; await it to start the run. Spark 4.3.0 runs a graph at most once: a second
    /// run of it fails.
    pub fn start_run(&self, storage: impl Into<String>) -> StartRun {
        StartRun {
            graph: self.clone(),
            storage: storage.into(),
            full_refresh: Vec::new(),
            full_refresh_all: false,
            refresh: Vec::new(),
            dry_run: false,
        }
    }
}

fn protos(expressions: Vec<spark_connect_core::expression::Expression>) -> Vec<proto::Expression> {
    expressions.iter().map(|e| e.to_proto()).collect()
}

/// The identifier that the server resolved the name of a dataset or of a flow to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Identifier {
    /// The catalog; `None` for a temporary view or a sink, which no catalog has.
    pub catalog: Option<String>,
    /// The namespace in the catalog, such as the database; empty for a temporary view or a sink.
    pub namespace: Vec<String>,
    /// The name in the namespace.
    pub name: String,
}

impl Identifier {
    fn of(identifier: proto::ResolvedIdentifier) -> Result<Identifier> {
        if identifier.table_name.is_empty() {
            return Err(Error::decode(
                "the server returned an identifier without a name",
            ));
        }
        Ok(Identifier {
            catalog: Some(identifier.catalog_name).filter(|catalog| !catalog.is_empty()),
            namespace: identifier.namespace,
            name: identifier.table_name,
        })
    }
}

/// Executes the definition `command` in `graph`, and returns the identifier that the server
/// resolved its name to. The future keeps the graph.
fn defined(graph: DataflowGraph, command: proto::command::CommandType) -> DefineFuture {
    async move {
        let identifier = stream::command_result(
            graph.session().clone(),
            command,
            core_actions::defined_identifier,
        )
        .await?;
        let identifier =
            identifier.ok_or_else(|| Error::decode("the server returned no identifier"))?;
        Identifier::of(identifier)
    }
    .boxed()
}

type DefineFuture = BoxFuture<'static, Result<Identifier>>;

/// Where a dataset or a flow is defined in the source code of a pipeline, which the server
/// reports with what it says of them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceCodeLocation {
    file_name: Option<String>,
    line_number: Option<i32>,
    definition_path: Option<String>,
}

impl SourceCodeLocation {
    /// A location in the file `file_name`.
    pub fn file(file_name: impl Into<String>) -> Self {
        SourceCodeLocation {
            file_name: Some(file_name.into()),
            ..SourceCodeLocation::default()
        }
    }

    /// The line of the location in its file.
    pub fn line_number(mut self, line_number: i32) -> Self {
        self.line_number = Some(line_number);
        self
    }

    /// The path of the definition in the pipeline, such as that of its function.
    pub fn definition_path(mut self, definition_path: impl Into<String>) -> Self {
        self.definition_path = Some(definition_path.into());
        self
    }

    fn to_proto(&self) -> proto::SourceCodeLocation {
        proto::SourceCodeLocation {
            file_name: self.file_name.clone(),
            line_number: self.line_number,
            definition_path: self.definition_path.clone(),
            extension: Vec::new(),
        }
    }
}

/// The definition of a materialized view or of a streaming table of
/// [`DataflowGraph::define_materialized_view`] and [`DataflowGraph::define_streaming_table`]:
/// await it to define it.
#[must_use = "a dataset is not defined until it is awaited"]
pub struct DefineTable {
    graph: DataflowGraph,
    spec: OutputSpec,
    details: TableDetailsSpec,
}

impl DefineTable {
    fn new(graph: &DataflowGraph, spec: OutputSpec) -> Self {
        DefineTable {
            graph: graph.clone(),
            spec,
            details: TableDetailsSpec {
                table_properties: HashMap::new(),
                partition_cols: Vec::new(),
                clustering_columns: Vec::new(),
                format: None,
                schema_string: None,
                schema_data_type: None,
            },
        }
    }

    /// The comment of the table.
    pub fn comment(mut self, comment: impl Into<String>) -> Self {
        self.spec.comment = Some(comment.into());
        self
    }

    /// Where the table is defined in the source code of the pipeline.
    pub fn source_code_location(mut self, location: SourceCodeLocation) -> Self {
        self.spec.source_code_location = Some(location.to_proto());
        self
    }

    /// Sets a property of the table.
    pub fn table_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details
            .table_properties
            .insert(key.into(), value.into());
        self
    }

    /// The columns that partition the table, by their names.
    pub fn partition_by(mut self, columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.details.partition_cols = strings(columns);
        self
    }

    /// The columns that cluster the table, by their names.
    pub fn cluster_by(mut self, columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.details.clustering_columns = strings(columns);
        self
    }

    /// The format of the table, such as `parquet`.
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.details.format = Some(format.into());
        self
    }

    /// The schema of the table: a struct [`DataType`](crate::types::DataType), or DDL text such
    /// as `"a INT, b STRING"`. Without one, the server takes that of the flows' rows.
    pub fn schema(mut self, schema: impl IntoDataType) -> Self {
        match schema.into_type_or_ddl() {
            TypeOrDdl::Type(data_type) => {
                self.details.schema_data_type = Some(data_type);
                self.details.schema_string = None;
            }
            TypeOrDdl::Text(text) => {
                self.details.schema_string = Some(text);
                self.details.schema_data_type = None;
            }
        }
        self
    }
}

impl IntoFuture for DefineTable {
    type Output = Result<Identifier>;
    type IntoFuture = DefineFuture;

    fn into_future(mut self) -> Self::IntoFuture {
        self.spec.table_details = Some(self.details);
        let command = core::define_output(self.graph.id(), self.spec);
        defined(self.graph, command)
    }
}

/// The definition of a temporary view of [`DataflowGraph::define_temporary_view`]: await it to
/// define it.
#[must_use = "a dataset is not defined until it is awaited"]
pub struct DefineView {
    graph: DataflowGraph,
    spec: OutputSpec,
}

impl DefineView {
    /// The comment of the view.
    pub fn comment(mut self, comment: impl Into<String>) -> Self {
        self.spec.comment = Some(comment.into());
        self
    }

    /// Where the view is defined in the source code of the pipeline.
    pub fn source_code_location(mut self, location: SourceCodeLocation) -> Self {
        self.spec.source_code_location = Some(location.to_proto());
        self
    }
}

impl IntoFuture for DefineView {
    type Output = Result<Identifier>;
    type IntoFuture = DefineFuture;

    fn into_future(self) -> Self::IntoFuture {
        let command = core::define_output(self.graph.id(), self.spec);
        defined(self.graph, command)
    }
}

/// The definition of a sink of [`DataflowGraph::define_sink`]: await it to define it.
#[must_use = "a dataset is not defined until it is awaited"]
pub struct DefineSink {
    graph: DataflowGraph,
    spec: OutputSpec,
    details: SinkDetailsSpec,
}

impl DefineSink {
    /// Sets an option of the streaming writes to the sink.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<OptionValue>) -> Self {
        insert_options(&mut self.details.options, [(key, value)]);
        self
    }

    /// Sets options of the streaming writes to the sink.
    pub fn options<K: Into<String>, V: Into<OptionValue>>(
        mut self,
        options: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        insert_options(&mut self.details.options, options);
        self
    }

    /// Where the sink is defined in the source code of the pipeline.
    pub fn source_code_location(mut self, location: SourceCodeLocation) -> Self {
        self.spec.source_code_location = Some(location.to_proto());
        self
    }
}

impl IntoFuture for DefineSink {
    type Output = Result<Identifier>;
    type IntoFuture = DefineFuture;

    fn into_future(mut self) -> Self::IntoFuture {
        self.spec.sink_details = Some(self.details);
        let command = core::define_output(self.graph.id(), self.spec);
        defined(self.graph, command)
    }
}

/// The query of a flow: a [`DataFrame`], or the text of a SQL query, which the server runs when
/// a run runs the flow.
pub struct FlowQuery(Query);

enum Query {
    DataFrame(DataFrame),
    Sql(String),
}

impl From<&DataFrame> for FlowQuery {
    fn from(dataframe: &DataFrame) -> Self {
        FlowQuery(Query::DataFrame(dataframe.clone()))
    }
}

impl From<DataFrame> for FlowQuery {
    fn from(dataframe: DataFrame) -> Self {
        FlowQuery(Query::DataFrame(dataframe))
    }
}

impl From<&str> for FlowQuery {
    fn from(sql: &str) -> Self {
        FlowQuery(Query::Sql(sql.to_string()))
    }
}

impl From<String> for FlowQuery {
    fn from(sql: String) -> Self {
        FlowQuery(Query::Sql(sql))
    }
}

impl From<&String> for FlowQuery {
    fn from(sql: &String) -> Self {
        FlowQuery(Query::Sql(sql.clone()))
    }
}

impl FlowQuery {
    /// The relation of the query, as the session `session` sends it, and the plan of its
    /// DataFrame.
    fn relation(self, session: &Session) -> (proto::Relation, Option<Arc<LogicalPlan>>) {
        match self.0 {
            Query::DataFrame(dataframe) => {
                (dataframe.to_relation(), Some(Arc::clone(dataframe.plan())))
            }
            Query::Sql(query) => {
                let plan = Arc::new(LogicalPlan::Sql {
                    query,
                    pos_args: Vec::new(),
                    named_args: HashMap::new(),
                });
                (relation_with_ids(&plan, session.plan_ids()), None)
            }
        }
    }
}

/// The definition of a flow of [`DataflowGraph::define_flow`]: await it to define it.
#[must_use = "a flow is not defined until it is awaited"]
pub struct DefineFlow {
    graph: DataflowGraph,
    name: String,
    target: String,
    query: FlowQuery,
    sql_conf: HashMap<String, String>,
    source_code_location: Option<proto::SourceCodeLocation>,
}

impl DefineFlow {
    /// Sets a SQL configuration for the runs of the flow.
    pub fn sql_conf(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.sql_conf.insert(key.into(), value.into());
        self
    }

    /// Where the flow is defined in the source code of the pipeline.
    pub fn source_code_location(mut self, location: SourceCodeLocation) -> Self {
        self.source_code_location = Some(location.to_proto());
        self
    }
}

impl IntoFuture for DefineFlow {
    type Output = Result<Identifier>;
    type IntoFuture = DefineFuture;

    fn into_future(self) -> Self::IntoFuture {
        let (relation, plan) = self.query.relation(self.graph.session());
        let command = core::define_flow(
            self.graph.id(),
            &self.name,
            &self.target,
            relation,
            self.sql_conf,
            self.source_code_location,
        );
        let graph = self.graph;
        async move {
            let identifier = defined(graph.clone(), command).await?;
            if let Some(plan) = plan {
                graph.keep(plan);
            }
            Ok(identifier)
        }
        .boxed()
    }
}

/// The type of slowly changing dimension that the table of an Auto CDC flow keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScdType {
    /// Type 1: each row has the values of its last change.
    Type1,
    /// Type 2: each row has the history of its values, a row for each of them.
    Type2,
}

/// The definition of an Auto CDC flow of [`DataflowGraph::define_auto_cdc_flow`]: await it to
/// define it.
#[must_use = "a flow is not defined until it is awaited"]
pub struct DefineAutoCdcFlow {
    graph: DataflowGraph,
    name: String,
    target: String,
    details: AutoCdcFlowDetails,
    sql_conf: HashMap<String, String>,
    source_code_location: Option<proto::SourceCodeLocation>,
}

impl DefineAutoCdcFlow {
    /// The condition of the changes that delete their row.
    pub fn apply_as_deletes(mut self, condition: impl Into<Column>) -> Self {
        self.details.apply_as_deletes = Some(condition.into().expression().to_proto());
        self
    }

    /// The columns of the source that the table has; by default, all of them.
    pub fn column_list(mut self, columns: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        self.details.column_list = protos(expressions(columns));
        self
    }

    /// The columns of the source that the table does not have.
    pub fn except_column_list(
        mut self,
        columns: impl IntoIterator<Item = impl Into<Column>>,
    ) -> Self {
        self.details.except_column_list = protos(expressions(columns));
        self
    }

    /// The type of slowly changing dimension that the table keeps; by default, the server's.
    pub fn stored_as_scd_type(mut self, scd_type: ScdType) -> Self {
        use proto::pipeline_command::define_flow::ScdType as Proto;
        self.details.stored_as_scd_type = match scd_type {
            ScdType::Type1 => Proto::ScdType1,
            ScdType::Type2 => Proto::ScdType2,
        } as i32;
        self
    }

    /// For [`ScdType::Type2`], the columns whose changes add a row to the history; by default,
    /// all of them.
    pub fn track_history_column_list(
        mut self,
        columns: impl IntoIterator<Item = impl Into<Column>>,
    ) -> Self {
        self.details.track_history_column_list = protos(expressions(columns));
        self
    }

    /// For [`ScdType::Type2`], the columns whose changes add no row to the history.
    pub fn track_history_except_column_list(
        mut self,
        columns: impl IntoIterator<Item = impl Into<Column>>,
    ) -> Self {
        self.details.track_history_except_column_list = protos(expressions(columns));
        self
    }

    /// Sets a SQL configuration for the runs of the flow.
    pub fn sql_conf(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.sql_conf.insert(key.into(), value.into());
        self
    }

    /// Where the flow is defined in the source code of the pipeline.
    pub fn source_code_location(mut self, location: SourceCodeLocation) -> Self {
        self.source_code_location = Some(location.to_proto());
        self
    }
}

impl IntoFuture for DefineAutoCdcFlow {
    type Output = Result<Identifier>;
    type IntoFuture = DefineFuture;

    fn into_future(self) -> Self::IntoFuture {
        let command = core::define_auto_cdc_flow(
            self.graph.id(),
            &self.name,
            &self.target,
            self.details,
            self.sql_conf,
            self.source_code_location,
        );
        defined(self.graph, command)
    }
}

/// The definition of the SQL text of [`DataflowGraph::define_sql`]: await it to define the
/// datasets and flows that its statements create.
#[must_use = "SQL text is not defined until it is awaited"]
pub struct DefineSql {
    graph: DataflowGraph,
    sql_text: String,
    file_path: Option<String>,
}

impl DefineSql {
    /// The path of the file of the SQL text, which the server reports with what it says of the
    /// datasets and flows of the text.
    pub fn file_path(mut self, path: impl Into<String>) -> Self {
        self.file_path = Some(path.into());
        self
    }
}

impl IntoFuture for DefineSql {
    type Output = Result<()>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let command =
            core::sql_graph_elements(self.graph.id(), &self.sql_text, self.file_path.as_deref());
        let graph = self.graph;
        async move {
            let defined = stream::command(graph.session().clone(), command).await;
            drop(graph);
            defined
        }
        .boxed()
    }
}

/// The start of a run of [`DataflowGraph::start_run`]: await it to start the run.
///
/// The future resolves once the server has accepted the run, with the stream of its events. An
/// error that the server reports before any event, such as a graph that it does not know or a
/// storage that it rejects, is the error of the future. Dropping the future before the server
/// accepts the run interrupts its operation; a Spark 4.3.0 server may still run the pipeline.
#[must_use = "a run does not start until it is awaited"]
pub struct StartRun {
    graph: DataflowGraph,
    storage: String,
    full_refresh: Vec<String>,
    full_refresh_all: bool,
    refresh: Vec<String>,
    dry_run: bool,
}

impl StartRun {
    /// Resets these datasets and computes them again from all of their data, rather than from
    /// what is new since the last run.
    pub fn full_refresh(mut self, datasets: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.full_refresh.extend(strings(datasets));
        self
    }

    /// Resets every dataset of the graph and computes them again from all of their data. The
    /// server rejects a run that also selects datasets to refresh.
    pub fn full_refresh_all(mut self) -> Self {
        self.full_refresh_all = true;
        self
    }

    /// Updates these datasets only, rather than all of them.
    pub fn refresh(mut self, datasets: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.refresh.extend(strings(datasets));
        self
    }

    /// Checks the graph and runs no flow: the run reports what is wrong with the graph, and
    /// changes no dataset.
    pub fn dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }
}

impl IntoFuture for StartRun {
    type Output = Result<PipelineRun>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let command = core::start_run(
            self.graph.id(),
            self.full_refresh,
            self.full_refresh_all,
            self.refresh,
            self.dry_run,
            Some(self.storage),
        );
        let graph = self.graph;
        async move {
            let (responses, operation_id) =
                stream::command_responses(graph.session().clone(), command).await?;
            Ok(PipelineRun {
                state: RunState::Idle(Box::new(responses)),
                operation_id,
                _graph: graph,
            })
        }
        .boxed()
    }
}

type Read = (
    Box<Responses<Observer>>,
    spark_connect_core::error::Result<Option<proto::ExecutePlanResponse>>,
);

enum RunState {
    Idle(Box<Responses<Observer>>),
    Reading(BoxFuture<'static, Read>),
    Done,
}

/// The events of a run of a dataflow graph, as the server sends them while it runs the flows.
///
/// The stream ends when the run does: after its last event when the run completes, a dry run
/// included, or after an error when it fails, such as that of a flow whose query fails or that
/// of the interruption of its operation. An event that cannot be read is an error of
/// [`ErrorKind::Decode`](crate::ErrorKind::Decode), and the events after it still come. Spark
/// 4.3.0 may leave out events, other than those that end a flow or the run, when they come
/// faster than it sends them.
///
/// Dropping the stream before its end releases the operation on the server, so that the
/// server sends no more events; a Spark 4.3.0 server still runs the pipeline to its end, as it
/// also does when the operation is interrupted or the session stops. The stream keeps its
/// graph.
pub struct PipelineRun {
    state: RunState,
    operation_id: String,
    _graph: DataflowGraph,
}

impl PipelineRun {
    /// The id of the operation of the run on the server.
    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }
}

impl fmt::Debug for PipelineRun {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineRun")
            .field("operation_id", &self.operation_id)
            .finish_non_exhaustive()
    }
}

impl Stream for PipelineRun {
    type Item = Result<PipelineEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            match std::mem::replace(&mut this.state, RunState::Done) {
                RunState::Idle(mut responses) => {
                    this.state = RunState::Reading(
                        async move {
                            let response = responses.next().await;
                            (responses, response)
                        }
                        .boxed(),
                    );
                }
                RunState::Reading(mut reading) => match reading.poll_unpin(cx) {
                    Poll::Pending => {
                        this.state = RunState::Reading(reading);
                        return Poll::Pending;
                    }
                    Poll::Ready((responses, Ok(Some(response)))) => {
                        this.state = RunState::Idle(responses);
                        if let Some(event) = event_of(response) {
                            return Poll::Ready(Some(event));
                        }
                    }
                    Poll::Ready((_, Ok(None))) => return Poll::Ready(None),
                    Poll::Ready((_, Err(error))) => return Poll::Ready(Some(Err(error.into()))),
                },
                RunState::Done => return Poll::Ready(None),
            }
        }
    }
}

/// An event of a run of a dataflow graph, such as a flow that starts, completes or fails, or
/// the run that completes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PipelineEvent {
    /// When the event occurred.
    pub timestamp: DateTime<Utc>,
    /// What the server says of the event, for people to read. It is all that the server sends
    /// of the event, such as the state of a flow and the error of a failed one.
    pub message: String,
}

/// The event of `response`, or `None` if it has none.
fn event_of(response: proto::ExecutePlanResponse) -> Option<Result<PipelineEvent>> {
    match response.response_type? {
        ResponseType::PipelineEventResult(result) => Some(PipelineEvent::of(result)),
        _ => None,
    }
}

impl PipelineEvent {
    fn of(result: proto::PipelineEventResult) -> Result<PipelineEvent> {
        let event = result.event.ok_or_else(|| {
            Error::decode("the server sent a pipeline event result without its event")
        })?;
        let timestamp = event
            .timestamp
            .ok_or_else(|| Error::decode("the server sent a pipeline event without a timestamp"))?;
        let timestamp = u32::try_from(timestamp.nanos)
            .ok()
            .and_then(|nanos| DateTime::from_timestamp(timestamp.seconds, nanos))
            .filter(|_| (0..1_000_000_000).contains(&timestamp.nanos))
            .ok_or_else(|| {
                Error::decode(format!(
                    "the server sent a pipeline event whose timestamp, {}s and {}ns, is out of range",
                    timestamp.seconds, timestamp.nanos
                ))
            })?;
        let message = event
            .message
            .ok_or_else(|| Error::decode("the server sent a pipeline event without a message"))?;
        Ok(PipelineEvent { timestamp, message })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    fn event(timestamp: Option<(i64, i32)>, message: Option<&str>) -> proto::ExecutePlanResponse {
        let mut event = proto::PipelineEvent {
            message: message.map(str::to_string),
            ..Default::default()
        };
        if let Some((seconds, nanos)) = timestamp {
            let at = event.timestamp.insert(Default::default());
            at.seconds = seconds;
            at.nanos = nanos;
        }
        response(ResponseType::PipelineEventResult(
            proto::PipelineEventResult { event: Some(event) },
        ))
    }

    fn response(response_type: ResponseType) -> proto::ExecutePlanResponse {
        proto::ExecutePlanResponse {
            response_type: Some(response_type),
            ..Default::default()
        }
    }

    type Read = Option<std::result::Result<PipelineEvent, String>>;

    #[test]
    fn the_events_of_a_run_are_read_with_their_nulls() {
        let at = |seconds, nanos| DateTime::from_timestamp(seconds, nanos).unwrap();
        let read = |message: &str| PipelineEvent {
            timestamp: at(1_794_431_318, 963_703_034),
            message: message.to_string(),
        };
        let out_of_range = |seconds, nanos| {
            Some(Err(format!(
                "the server sent a pipeline event whose timestamp, {seconds}s and {nanos}ns, is \
                 out of range"
            )))
        };
        let cases: Vec<(&str, proto::ExecutePlanResponse, Read)> = vec![
            (
                "an event",
                event(
                    Some((1_794_431_318, 963_703_034)),
                    Some("Flow f is QUEUED."),
                ),
                Some(Ok(read("Flow f is QUEUED."))),
            ),
            (
                "an empty message at the epoch",
                event(Some((0, 0)), Some("")),
                Some(Ok(PipelineEvent {
                    timestamp: at(0, 0),
                    message: String::new(),
                })),
            ),
            (
                "the last nanosecond before the epoch",
                event(Some((-1, 999_999_999)), Some("m")),
                Some(Ok(PipelineEvent {
                    timestamp: at(-1, 999_999_999),
                    message: "m".to_string(),
                })),
            ),
            (
                "no message",
                event(Some((1_794_431_318, 963_703_034)), None),
                Some(Err(
                    "the server sent a pipeline event without a message".to_string()
                )),
            ),
            (
                "no timestamp",
                event(None, Some("m")),
                Some(Err(
                    "the server sent a pipeline event without a timestamp".to_string()
                )),
            ),
            (
                "no event",
                response(ResponseType::PipelineEventResult(
                    proto::PipelineEventResult { event: None },
                )),
                Some(Err(
                    "the server sent a pipeline event result without its event".to_string(),
                )),
            ),
            (
                "negative nanoseconds",
                event(Some((5, -1)), Some("m")),
                out_of_range(5, -1),
            ),
            (
                "a second of nanoseconds, which chrono takes for a leap second",
                event(Some((59, 1_000_000_000)), Some("m")),
                out_of_range(59, 1_000_000_000),
            ),
            (
                "seconds after the last date",
                event(Some((i64::MAX, 0)), Some("m")),
                out_of_range(i64::MAX, 0),
            ),
            (
                "progress",
                response(ResponseType::ExecutionProgress(Default::default())),
                None,
            ),
            (
                "the result of the command",
                response(ResponseType::PipelineCommandResult(Default::default())),
                None,
            ),
            (
                "the end",
                response(ResponseType::ResultComplete(Default::default())),
                None,
            ),
            ("nothing", proto::ExecutePlanResponse::default(), None),
        ];
        for (name, response, expected) in cases {
            let event = event_of(response);
            if let Some(Err(error)) = &event {
                assert_eq!(error.kind(), ErrorKind::Decode, "{name}");
            }
            let event = event.map(|event| event.map_err(|error| error.to_string()));
            assert_eq!(event, expected, "{name}");
        }
    }

    #[test]
    fn the_identifiers_that_the_server_resolves_are_read_with_their_nulls() {
        let resolved = |catalog: &str, namespace: &[&str], name: &str| proto::ResolvedIdentifier {
            catalog_name: catalog.to_string(),
            namespace: namespace.iter().map(|part| part.to_string()).collect(),
            table_name: name.to_string(),
        };
        let identifier = |catalog: Option<&str>, namespace: &[&str], name: &str| Identifier {
            catalog: catalog.map(str::to_string),
            namespace: namespace.iter().map(|part| part.to_string()).collect(),
            name: name.to_string(),
        };
        let cases = [
            (
                "a table",
                resolved("spark_catalog", &["default"], "t"),
                Ok(identifier(Some("spark_catalog"), &["default"], "t")),
            ),
            (
                "a namespace of two parts",
                resolved("c", &["a", "b"], "t"),
                Ok(identifier(Some("c"), &["a", "b"], "t")),
            ),
            (
                "a temporary view",
                resolved("", &[], "v"),
                Ok(identifier(None, &[], "v")),
            ),
            (
                "no name",
                resolved("spark_catalog", &["default"], ""),
                Err("the server returned an identifier without a name".to_string()),
            ),
        ];
        for (name, resolved, expected) in cases {
            let read = Identifier::of(resolved);
            if let Err(error) = &read {
                assert_eq!(error.kind(), ErrorKind::Decode, "{name}");
            }
            assert_eq!(read.map_err(|error| error.to_string()), expected, "{name}");
        }
    }
}
