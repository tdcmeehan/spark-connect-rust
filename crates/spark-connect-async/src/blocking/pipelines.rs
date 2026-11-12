//! The blocking forms of the dataflow graphs of Spark Declarative Pipelines and of their runs.

use std::future::IntoFuture;

use futures::StreamExt;

use super::{runtime, wait, Session};
use crate::error::Result;
use crate::pipelines::{self, Identifier, PipelineEvent};

/// A dataflow graph on the server, from the [`run`](crate::pipelines::CreateDataflowGraph::run)
/// of a session's [`create_dataflow_graph`](crate::Session::create_dataflow_graph).
///
/// The definitions of its datasets and flows and its runs are builders of the async
/// [`crate::pipelines::DataflowGraph`], which [`as_async`](DataflowGraph::as_async) borrows:
/// [`run`](crate::pipelines::DefineTable::run) one to send it and block. The blocking graph holds
/// the async one, so the server removes the graph once the last handle to it, async or blocking,
/// and the last of the runs that keep it are dropped.
///
/// ```no_run
/// # fn run(session: spark_connect_async::blocking::Session) -> spark_connect_async::Result<()> {
/// let graph = session.as_async().create_dataflow_graph().run()?;
/// graph
///     .as_async()
///     .define_sql("CREATE MATERIALIZED VIEW numbers AS SELECT id FROM range(10)")
///     .run()?;
/// for event in graph.as_async().start_run("file:///tmp/pipeline-storage").run()? {
///     println!("{}", event?.message);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct DataflowGraph(pipelines::DataflowGraph);

impl DataflowGraph {
    /// This graph as the async [`crate::pipelines::DataflowGraph`] it holds, borrowed: its id,
    /// its definitions and its runs are made on it.
    pub fn as_async(&self) -> &pipelines::DataflowGraph {
        &self.0
    }

    /// This graph as the async [`crate::pipelines::DataflowGraph`] it holds, as [`From`]
    /// converts it.
    pub fn into_async(self) -> pipelines::DataflowGraph {
        self.0
    }

    /// The session of the graph.
    pub fn session(&self) -> Session {
        Session::from(self.0.session().clone())
    }
}

impl From<pipelines::DataflowGraph> for DataflowGraph {
    fn from(graph: pipelines::DataflowGraph) -> Self {
        DataflowGraph(graph)
    }
}

impl From<DataflowGraph> for pipelines::DataflowGraph {
    fn from(graph: DataflowGraph) -> Self {
        graph.0
    }
}

impl pipelines::DataflowGraph {
    /// The blocking form of this graph, the same graph.
    pub fn blocking(&self) -> DataflowGraph {
        DataflowGraph(self.clone())
    }
}

impl pipelines::CreateDataflowGraph {
    /// Creates the graph, as awaiting the creation does, and blocks until the server answers.
    pub fn run(self) -> Result<DataflowGraph> {
        wait(self.into_future()).map(DataflowGraph)
    }
}

/// `run()` for the definitions of a graph.
macro_rules! run {
    ($($(#[$doc:meta])* $builder:ident -> $output:ty;)*) => {$(
        impl pipelines::$builder {
            $(#[$doc])*
            pub fn run(self) -> Result<$output> {
                wait(self.into_future())
            }
        }
    )*};
}

run! {
    /// Defines the table, as awaiting the definition does, and blocks until the server answers.
    DefineTable -> Identifier;
    /// Defines the view, as awaiting the definition does, and blocks until the server answers.
    DefineView -> Identifier;
    /// Defines the sink, as awaiting the definition does, and blocks until the server answers.
    DefineSink -> Identifier;
    /// Defines the flow, as awaiting the definition does, and blocks until the server answers.
    DefineFlow -> Identifier;
    /// Defines the flow, as awaiting the definition does, and blocks until the server answers.
    DefineAutoCdcFlow -> Identifier;
    /// Defines the datasets and flows of the SQL text, as awaiting the definition does, and
    /// blocks until the server answers.
    DefineSql -> ();
}

impl pipelines::StartRun {
    /// Starts the run, as awaiting it does, and blocks until the server has accepted it; returns
    /// the iterator of its events.
    pub fn run(self) -> Result<PipelineRun> {
        wait(self.into_future()).map(PipelineRun)
    }
}

/// The events of a run of a dataflow graph, read as the iterator is advanced: each
/// [`next`](Iterator::next) blocks until the server sends the next event.
///
/// The events, the errors and the end are those of the async
/// [`crate::pipelines::PipelineRun`] that it reads, which [`as_async`](PipelineRun::as_async)
/// borrows for its operation id. Dropping the iterator before its end releases the operation on
/// the server, from the runtime that started it, and the iterator keeps its graph. A `next` can't
/// be given up: see [the waits](super#waits-that-only-the-server-ends) for what ends it.
pub struct PipelineRun(pipelines::PipelineRun);

impl PipelineRun {
    /// The run that this iterator reads, borrowed: its operation id.
    pub fn as_async(&self) -> &pipelines::PipelineRun {
        &self.0
    }

    /// The run that this iterator reads, with the events not read yet, as [`From`] converts it.
    pub fn into_async(self) -> pipelines::PipelineRun {
        self.0
    }
}

impl Iterator for PipelineRun {
    type Item = Result<PipelineEvent>;

    fn next(&mut self) -> Option<Result<PipelineEvent>> {
        match runtime() {
            Ok(runtime) => runtime.block_on(self.0.next()),
            Err(error) => Some(Err(error)),
        }
    }
}

impl From<pipelines::PipelineRun> for PipelineRun {
    fn from(run: pipelines::PipelineRun) -> Self {
        PipelineRun(run)
    }
}

impl From<PipelineRun> for pipelines::PipelineRun {
    fn from(run: PipelineRun) -> Self {
        run.0
    }
}

impl pipelines::PipelineRun {
    /// The blocking form of this run: an iterator whose events block until they arrive.
    pub fn blocking(self) -> PipelineRun {
        PipelineRun(self)
    }
}
