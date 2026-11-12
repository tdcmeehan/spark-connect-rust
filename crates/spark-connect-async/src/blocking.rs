//! The client with calls that block, for code that doesn't run async: a command-line tool, a
//! script, or a synchronous callback.
//!
//! Enable the `blocking` feature. A blocking [`Session`] holds a [`crate::Session`]. Its calls to
//! the server block the calling thread until the server answers, and return what the async
//! call's future returns. What sends nothing, such as the transformations of a query, is the
//! async API's, reached with `as_async()`:
//!
//! ```no_run
//! # fn run() -> spark_connect_async::Result<()> {
//! use spark_connect_async::readwriter::SaveMode;
//! use spark_connect_async::{col, lit, Session};
//!
//! let session = Session::builder()
//!     .remote("sc://localhost:15002")
//!     .blocking()
//!     .connect()?;
//! let people = session.as_async().sql("SELECT name, age FROM people").run()?;
//! println!("{} people", people.count()?);
//! let adults = people.as_async().filter(col("age").ge(lit(18))).blocking();
//! let names: Vec<(String,)> = adults.as_async().select(["name"]).blocking().collect_as()?;
//! adults.as_async().write().mode(SaveMode::Overwrite).blocking().parquet("/tmp/adults")?;
//! for table in session.catalog().as_async().list_tables().run()? {
//!     println!("{}", table.name);
//! }
//! session.stop()?;
//! # Ok(())
//! # }
//! ```
//!
//! In async code, await the async API instead: a blocking call holds its thread for as long as
//! the server takes.
//!
//! # The blocking forms of the async API
//!
//! - [`Session`], [`DataFrame`], [`Catalog`], [`RuntimeConfig`], [`GroupedData`],
//!   [`StatFunctions`], [`StreamingQueryManager`], [`StreamingQuery`], the fitted models of
//!   Spark ML, such as [`LogisticRegressionModel`], and the registration of Rust UDFs, with the
//!   `rust-udf` feature, have the methods of their async types that return futures, with the
//!   same names and arguments, and those block: `df.count()?` here is `df.count().await?`
//!   there. The [`conf`](Session::conf), [`catalog`](Session::catalog),
//!   [`streams`](Session::streams), `udf` and [`with_tag`](Session::with_tag) of a session, and
//!   the [`session`](DataFrame::session) of a DataFrame, a catalog or a dataflow graph, return
//!   blocking values too.
//! - The fit of an estimator and the evaluation of an evaluator of Spark ML, which the async API
//!   has as trait methods, block through the traits [`Estimator`] and [`Evaluator`].
//! - What the server sends in parts is an iterator, each of whose items blocks: the record
//!   batches of a query ([`RecordBatchIter`]), the events of streaming queries ([`QueryEvents`])
//!   and those of a run of a pipeline ([`PipelineRun`]).
//! - Everything else is the async API's alone, such as a DataFrame's transformations and
//!   columns, the builders of a session or a catalog, and a session's id: a blocking value
//!   doesn't dereference to its async one. `as_async()` borrows the async value, and
//!   `into_async()` returns it. A lazy call returns an async value, which `.blocking()` turns
//!   into its blocking form, so a lazy chain starts with `as_async()` and ends with
//!   `.blocking()`. [`From`] converts both ways too, so code can mix the two.
//! - A parameter that takes a DataFrame, such as [`crate::DataFrame::union`]'s,
//!   [`DataFrame::same_semantics`]'s or [`Estimator::fit`]'s, takes an async one: pass a
//!   blocking DataFrame's `as_async()`.
//! - A builder that async code awaits, such as [`crate::Session::sql`]'s,
//!   [`crate::catalog::Catalog::list_tables`]'s or [`crate::pipelines::DataflowGraph::start_run`]'s,
//!   has a `run()` method, which sends it and blocks.
//! - A builder whose last step is a method has a `.blocking()` form with that method, which
//!   blocks: a session builder's [`connect`](SessionBuilder::connect), a reader's
//!   [`parse`](DataFrameReader::parse), a writer's saves, the starts of a stream's
//!   [`DataStreamWriter`], and a show's [`to_string`](Show::to_string).
//!
//! # Waits that only the server ends
//!
//! Dropping a future of the async API gives up its wait. A blocking call can't be given up, so
//! these calls wait for as long as the server does:
//!
//! - [`StreamingQuery::await_termination`] and [`StreamingQueryManager::await_any_termination`]
//!   end at their timeout, or once a query terminates, as one does that fails or that another
//!   thread [stops](StreamingQuery::stop). [`StreamingQuery::process_all_available`] ends once
//!   the query has processed the data that its sources had, or once it terminates.
//! - Each of the three also ends when another thread interrupts the operations of the session
//!   with [`Session::interrupt_all`], or those of a tag of the handle that made the call with
//!   [`Session::interrupt_tag`]: the call fails with
//!   [`ErrorKind::Cancelled`](crate::ErrorKind::Cancelled), and a Spark 4.3.0 server also stops
//!   the queries that the interrupt reaches, those that the session, or that tag's handle,
//!   started.
//! - A `next` of a [`PipelineRun`] ends at the run's next event or at its end, or when another
//!   thread interrupts the run's operation, with [`Session::interrupt_all`] or with
//!   [`Session::interrupt_operation`] of its
//!   [`operation_id`](crate::pipelines::PipelineRun::operation_id): the iterator then gives the
//!   error and ends, while a Spark 4.3.0 server still runs the pipeline.
//! - A `next` of [`QueryEvents`] ends at the next event, or when the server ends the
//!   subscription, as it does when the session stops. An interrupt doesn't end it.
//! - [`StreamingQueryManager::events`] ends once the server acknowledges the subscription. While
//!   another client of the same session on the server holds one, a Spark 4.3.0 server doesn't
//!   answer, even once that client removes it: only the end of the session, such as another
//!   thread's [`Session::stop`], ends the call, which then fails.
//!
//! # Threads and runtimes
//!
//! The calls of this module share one Tokio runtime, which starts at the first call and lasts as
//! long as the process. Its threads drive the connections, the timers and the cleanup that a drop
//! sends. A call polls its future on the calling thread, so the progress handlers and the
//! execute-plan hook of a session run there, and may make blocking calls of their own.
//!
//! A call doesn't panic, outside a runtime or inside one of either flavor, in a task or in
//! [`spawn_blocking`](tokio::task::spawn_blocking). The exception is the thread of a
//! current-thread runtime that has entered the handle of a multi-thread runtime, which Tokio
//! can't tell apart from a worker of that runtime: there a call panics. A call blocks the thread
//! it runs on: on a worker of a multi-thread runtime, the worker's other tasks move to another
//! thread first; on the thread of a current-thread runtime, the runtime's other tasks wait until
//! the call returns.
//!
//! Dropping an iterator of results before its end releases its operation on the server, and
//! dropping the events of streaming queries removes their subscription. Dropping the last handle
//! of a checkpoint's DataFrame, of a fitted model or of a dataflow graph removes it from the
//! server. Each is sent on the runtime that made it, which for what a call of this module made is
//! the runtime of this module, whichever thread or runtime drops it.
//!
//! A session connected with the async API keeps its connection on the runtime it connected on,
//! so its blocking form needs that runtime to keep running. A blocking call on the thread of a
//! current-thread runtime that the session connected on stalls: the connection can't run until
//! the call returns.

mod catalog;
mod dataframe;
mod ml;
mod pipelines;
mod readwriter;
#[cfg(feature = "rust-udf")]
mod rust_udf;
mod streaming;

use std::collections::BTreeMap;
use std::future::{Future, IntoFuture};
use std::sync::OnceLock;

use spark_connect_blocking::Blocking;
use spark_connect_core::error::SparkError;

pub use catalog::Catalog;
pub use dataframe::{DataFrame, GroupedData, RecordBatchIter, Show, StatFunctions};
pub use ml::{
    Estimator, Evaluator, LinearRegressionModel, LinearSVCModel, LogisticRegressionModel,
    MaxAbsScalerModel, StandardScalerModel, StringIndexerModel,
};
pub use pipelines::{DataflowGraph, PipelineRun};
pub use readwriter::{DataFrameReader, DataFrameWriter, DataFrameWriterV2};
#[cfg(feature = "rust-udf")]
pub use rust_udf::UdfRegistration;
pub use streaming::{DataStreamWriter, QueryEvents, StreamingQuery, StreamingQueryManager};

use crate::error::{Error, Result};
use crate::session::{AddArtifacts, RuntimeConfig as AsyncConfig};
use crate::sql::Sql;

/// The runtime of every blocking call of the process, started by the first call. A runtime
/// that fails to start is an error of that call, and the next call tries again.
fn runtime() -> Result<&'static Blocking> {
    static RUNTIME: OnceLock<Blocking> = OnceLock::new();
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime);
    }
    let started = Blocking::new().map_err(|error| {
        Error::from(SparkError::connect_msg(format!(
            "the runtime of the blocking API failed to start: {error}"
        )))
    })?;
    Ok(RUNTIME.get_or_init(|| started))
}

/// Runs `future` to completion on the calling thread, with the runtime of the module. The
/// future is boxed: its state would otherwise sit on the caller's stack, and threads such as
/// those of `spawn_blocking` have small ones.
fn wait<T>(future: impl Future<Output = Result<T>>) -> Result<T> {
    runtime()?.block_on(Box::pin(future))
}

/// A session with a Spark Connect server, whose calls block until the server answers.
///
/// Connect one with [`crate::SessionBuilder::blocking`], or convert a [`crate::Session`] with
/// [`crate::Session::blocking`] or [`From`]. What sends nothing, such as
/// [`table`](crate::Session::table), [`sql`](crate::Session::sql)'s builder or
/// [`session_id`](crate::Session::session_id), is the async session's, which
/// [`as_async`](Session::as_async) borrows:
///
/// ```no_run
/// # fn run(session: spark_connect_async::blocking::Session) -> spark_connect_async::Result<()> {
/// let people = session.as_async().table("people").blocking();
/// println!("{} people", people.count()?);
/// println!("in session {}", session.as_async().session_id());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Session(crate::Session);

impl Session {
    /// This session as the async [`crate::Session`] it holds, borrowed: the calls that send
    /// nothing are made on it.
    pub fn as_async(&self) -> &crate::Session {
        &self.0
    }

    /// This session as the async [`crate::Session`] it holds, as [`From`] converts it.
    pub fn into_async(self) -> crate::Session {
        self.0
    }

    /// The runtime configuration of the session.
    pub fn conf(&self) -> RuntimeConfig {
        RuntimeConfig(self.0.conf())
    }

    /// A handle to the same session whose operations also carry `tag`, as
    /// [`crate::Session::with_tag`] makes one.
    pub fn with_tag(&self, tag: impl Into<String>) -> Result<Session> {
        self.0.with_tag(tag).map(Session)
    }

    /// Interrupts every running operation of the session, and returns the ids of those it
    /// interrupted.
    pub fn interrupt_all(&self) -> Result<Vec<String>> {
        wait(self.0.interrupt_all())
    }

    /// Interrupts the running operations of the session that carry `tag`, and returns the ids of
    /// those it interrupted.
    pub fn interrupt_tag(&self, tag: impl Into<String>) -> Result<Vec<String>> {
        wait(self.0.interrupt_tag(tag))
    }

    /// Interrupts the operation `operation_id` if it is running, and returns the ids of those it
    /// interrupted.
    pub fn interrupt_operation(&self, operation_id: impl Into<String>) -> Result<Vec<String>> {
        wait(self.0.interrupt_operation(operation_id))
    }

    /// The Spark version of the server. Only the first successful call asks the server.
    pub fn version(&self) -> Result<String> {
        wait(self.0.version())
    }

    /// Releases the session on the server and closes it, as [`crate::Session::stop`] does.
    pub fn stop(&self) -> Result<()> {
        wait(self.0.stop())
    }
}

impl From<crate::Session> for Session {
    fn from(session: crate::Session) -> Self {
        Session(session)
    }
}

impl From<Session> for crate::Session {
    fn from(session: Session) -> Self {
        session.0
    }
}

impl crate::Session {
    /// The blocking form of this session: the same session, whose calls block.
    ///
    /// Its connection stays on the runtime that this session connected on, which must keep
    /// running: see [`blocking`](crate::blocking).
    pub fn blocking(&self) -> Session {
        Session(self.clone())
    }
}

/// The connection of a [`crate::SessionBuilder`], from [`crate::SessionBuilder::blocking`].
#[derive(Clone)]
#[must_use = "a session builder connects when its connect is called"]
pub struct SessionBuilder(crate::SessionBuilder);

impl SessionBuilder {
    /// Connects to the server, as [`crate::SessionBuilder::connect`] does, and returns the
    /// blocking session. The connection runs on the runtime of the module, so the session
    /// keeps working whichever runtime, if any, the call was made in.
    pub fn connect(self) -> Result<Session> {
        wait(self.0.connect()).map(Session)
    }
}

impl From<crate::SessionBuilder> for SessionBuilder {
    fn from(builder: crate::SessionBuilder) -> Self {
        SessionBuilder(builder)
    }
}

impl From<SessionBuilder> for crate::SessionBuilder {
    fn from(builder: SessionBuilder) -> Self {
        builder.0
    }
}

impl crate::SessionBuilder {
    /// The blocking form of this builder, whose [`connect`](SessionBuilder::connect) blocks
    /// and returns a blocking [`Session`].
    pub fn blocking(self) -> SessionBuilder {
        SessionBuilder(self)
    }
}

/// The runtime configuration of a session, whose calls block until the server answers.
///
/// Each call of the async [`crate::session::RuntimeConfig`] has its blocking form here;
/// [`as_async`](RuntimeConfig::as_async) borrows the async one.
#[derive(Clone)]
pub struct RuntimeConfig(AsyncConfig);

impl RuntimeConfig {
    /// This configuration as the async [`crate::session::RuntimeConfig`] it holds, borrowed.
    pub fn as_async(&self) -> &AsyncConfig {
        &self.0
    }

    /// This configuration as the async [`crate::session::RuntimeConfig`] it holds, as [`From`]
    /// converts it.
    pub fn into_async(self) -> AsyncConfig {
        self.0
    }

    /// Sets `key` to `value`, and returns the server's warnings, such as of a deprecated key.
    pub fn set(&self, key: impl Into<String>, value: impl Into<String>) -> Result<Vec<String>> {
        wait(self.0.set(key, value))
    }

    /// Sets each key to its value, and returns the server's warnings.
    pub fn set_all(
        &self,
        pairs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Result<Vec<String>> {
        wait(self.0.set_all(pairs))
    }

    /// The value of `key`, or `None` if it has none.
    pub fn get(&self, key: impl Into<String>) -> Result<Option<String>> {
        wait(self.0.get(key))
    }

    /// Every key that has a value, with its value.
    pub fn get_all(&self) -> Result<BTreeMap<String, String>> {
        wait(self.0.get_all())
    }

    /// Removes the value of `key`, and returns the server's warnings.
    pub fn unset(&self, key: impl Into<String>) -> Result<Vec<String>> {
        wait(self.0.unset(key))
    }

    /// Whether `key` can be set on a running session.
    pub fn is_modifiable(&self, key: impl Into<String>) -> Result<bool> {
        wait(self.0.is_modifiable(key))
    }
}

impl From<AsyncConfig> for RuntimeConfig {
    fn from(config: AsyncConfig) -> Self {
        RuntimeConfig(config)
    }
}

impl From<RuntimeConfig> for AsyncConfig {
    fn from(config: RuntimeConfig) -> Self {
        config.0
    }
}

impl AsyncConfig {
    /// The blocking form of this configuration, whose calls block.
    pub fn blocking(&self) -> RuntimeConfig {
        RuntimeConfig(self.clone())
    }
}

impl Sql {
    /// Sends the statement, as awaiting it does, and blocks until the server answers: a
    /// command runs now, and a query returns a DataFrame that runs at its actions.
    pub fn run(self) -> Result<DataFrame> {
        wait(self.into_future()).map(DataFrame::from)
    }
}

impl AddArtifacts {
    /// Uploads the artifacts, as awaiting the upload does, and blocks until the server has
    /// them.
    pub fn run(self) -> Result<()> {
        wait(self.into_future())
    }
}

/// Each of these fails to compile, since a blocking value doesn't dereference to its async
/// value. [`DataFrame`]'s own documentation has its example.
///
/// ```compile_fail,E0308
/// use spark_connect_async::blocking;
///
/// fn session(session: &blocking::Session) -> &spark_connect_async::Session {
///     session
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, session};
///
/// fn conf(conf: &blocking::RuntimeConfig) -> &session::RuntimeConfig {
///     conf
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, catalog};
///
/// fn catalog(catalog: &blocking::Catalog) -> &catalog::Catalog {
///     catalog
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, group};
///
/// fn groups(groups: &blocking::GroupedData) -> &group::GroupedData {
///     groups
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, stat};
///
/// fn stat(stat: &blocking::StatFunctions) -> &stat::StatFunctions {
///     stat
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, RecordBatchStream};
///
/// fn batches(batches: &blocking::RecordBatchIter) -> &RecordBatchStream {
///     batches
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, streaming};
///
/// fn queries(queries: &blocking::StreamingQueryManager) -> &streaming::StreamingQueryManager {
///     queries
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, streaming};
///
/// fn query(query: &blocking::StreamingQuery) -> &streaming::StreamingQuery {
///     query
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, streaming};
///
/// fn events(events: &blocking::QueryEvents) -> &streaming::QueryEvents {
///     events
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, pipelines};
///
/// fn graph(graph: &blocking::DataflowGraph) -> &pipelines::DataflowGraph {
///     graph
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, pipelines};
///
/// fn run(run: &blocking::PipelineRun) -> &pipelines::PipelineRun {
///     run
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, ml};
///
/// fn model(model: &blocking::LogisticRegressionModel) -> &ml::LogisticRegressionModel {
///     model
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, ml};
///
/// fn model(model: &blocking::LinearSVCModel) -> &ml::LinearSVCModel {
///     model
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, ml};
///
/// fn model(model: &blocking::LinearRegressionModel) -> &ml::LinearRegressionModel {
///     model
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, ml};
///
/// fn model(model: &blocking::StandardScalerModel) -> &ml::StandardScalerModel {
///     model
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, ml};
///
/// fn model(model: &blocking::MaxAbsScalerModel) -> &ml::MaxAbsScalerModel {
///     model
/// }
/// ```
///
/// ```compile_fail,E0308
/// use spark_connect_async::{blocking, ml};
///
/// fn model(model: &blocking::StringIndexerModel) -> &ml::StringIndexerModel {
///     model
/// }
/// ```
#[cfg_attr(
    feature = "rust-udf",
    doc = "
```compile_fail,E0308
use spark_connect_async::{blocking, rust_udf};

fn udf(udf: &blocking::UdfRegistration) -> &rust_udf::UdfRegistration {
    udf
}
```"
)]
#[cfg(doctest)]
struct NoDeref;

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}

    #[test]
    fn the_blocking_types_are_send_and_sync_and_their_iterators_send() {
        assert_send_sync::<Session>();
        assert_send_sync::<SessionBuilder>();
        assert_send_sync::<RuntimeConfig>();
        assert_send_sync::<DataFrame>();
        assert_send_sync::<GroupedData>();
        assert_send_sync::<GroupedData<crate::group::ByColumns>>();
        assert_send_sync::<StatFunctions>();
        assert_send_sync::<Show>();
        assert_send_sync::<DataFrameReader>();
        assert_send_sync::<DataFrameWriter>();
        assert_send_sync::<DataFrameWriterV2>();
        assert_send_sync::<Catalog>();
        assert_send_sync::<StreamingQueryManager>();
        assert_send_sync::<StreamingQuery>();
        assert_send_sync::<DataStreamWriter>();
        assert_send_sync::<DataflowGraph>();
        assert_send_sync::<LogisticRegressionModel>();
        assert_send_sync::<LinearSVCModel>();
        assert_send_sync::<LinearRegressionModel>();
        assert_send_sync::<StandardScalerModel>();
        assert_send_sync::<MaxAbsScalerModel>();
        assert_send_sync::<StringIndexerModel>();
        #[cfg(feature = "rust-udf")]
        assert_send_sync::<UdfRegistration>();
        assert_send::<RecordBatchIter>();
        assert_send::<QueryEvents>();
        assert_send::<PipelineRun>();
        // Type-checked, never called: it would need a server.
        let _ = |df: DataFrame| {
            fn assert_send_static<T: Send + 'static>(_: T) {}
            assert_send_static(df.stream_as::<(i64,)>().unwrap());
        };
    }
}
