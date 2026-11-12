//! The client with calls that block, for code that doesn't run async: a command-line tool, a
//! script, or a synchronous callback.
//!
//! Enable the `blocking` feature. A blocking [`Session`] is a [`crate::Session`] whose calls to
//! the server block the calling thread until the server answers, and return the answer that
//! the async call's future returns:
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
//! let people = session.sql("SELECT name, age FROM people").run()?;
//! println!("{} people", people.count()?);
//! let adults = people.filter(col("age").ge(lit(18))).blocking();
//! let names: Vec<(String,)> = adults.select(["name"]).blocking().collect_as()?;
//! adults.write().mode(SaveMode::Overwrite).blocking().parquet("/tmp/adults")?;
//! for table in session.catalog().list_tables().run()? {
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
//! - [`Session`], [`DataFrame`], [`Catalog`], [`RuntimeConfig`], [`GroupedData`] and
//!   [`StatFunctions`] are the async types with the same methods, which block: `df.count()?`
//!   here is `df.count().await?` there.
//! - What sends nothing stays the async API's: a blocking type dereferences to its async type,
//!   so a DataFrame's transformations, and the builders of the session and the catalog, are the
//!   async API's, and return async values. `.blocking()` turns an async value into its blocking
//!   form, and [`From`] converts both ways, so code can mix the two.
//! - A builder that async code awaits, such as [`crate::Session::sql`]'s or
//!   [`crate::catalog::Catalog::list_tables`]'s, has a `run()` method, which sends it and blocks.
//! - A builder whose last step is a method has a `.blocking()` form with that method, which
//!   blocks: a session builder's [`connect`](SessionBuilder::connect), a reader's
//!   [`parse`](DataFrameReader::parse), a writer's saves, and a show's
//!   [`to_string`](Show::to_string).
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
//! dropping the last DataFrame of a checkpoint removes the checkpoint: both are sent on the
//! runtime of this module, whichever thread or runtime drops them.
//!
//! A session connected with the async API keeps its connection on the runtime it connected on,
//! so its blocking form needs that runtime to keep running. A blocking call on the thread of a
//! current-thread runtime that the session connected on stalls: the connection can't run until
//! the call returns.

mod catalog;
mod dataframe;
mod readwriter;

use std::collections::BTreeMap;
use std::future::{Future, IntoFuture};
use std::ops::Deref;
use std::sync::OnceLock;

use spark_connect_blocking::Blocking;
use spark_connect_core::error::SparkError;

pub use catalog::Catalog;
pub use dataframe::{DataFrame, GroupedData, RecordBatchIter, Show, StatFunctions};
pub use readwriter::{DataFrameReader, DataFrameWriter, DataFrameWriterV2};

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
/// [`crate::Session::blocking`] or [`From`]. It dereferences to the async session, whose
/// methods that send nothing, such as [`table`](crate::Session::table) or
/// [`sql`](crate::Session::sql)'s builder, it shares.
#[derive(Clone)]
pub struct Session(crate::Session);

impl Session {
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

impl Deref for Session {
    type Target = crate::Session;

    fn deref(&self) -> &crate::Session {
        &self.0
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
#[derive(Clone)]
pub struct RuntimeConfig(AsyncConfig);

impl RuntimeConfig {
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

impl Deref for RuntimeConfig {
    type Target = AsyncConfig;

    fn deref(&self) -> &AsyncConfig {
        &self.0
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
        assert_send::<RecordBatchIter>();
        // Type-checked, never called: it would need a server.
        let _ = |df: DataFrame| {
            fn assert_send_static<T: Send + 'static>(_: T) {}
            assert_send_static(df.stream_as::<(i64,)>().unwrap());
        };
    }
}
