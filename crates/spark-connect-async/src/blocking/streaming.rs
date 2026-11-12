//! The blocking forms of the streaming queries of a session, of their starts and of their
//! events.

use std::future::IntoFuture;
use std::time::Duration;

use futures::StreamExt;

use super::{runtime, wait, Session};
use crate::error::{Error, Result};
use crate::streaming::{self, QueryEvent, QueryProgress, QueryStatus};

/// The streaming queries of a session, whose calls block until the server answers, from
/// [`Session::streams`].
///
/// ```no_run
/// # fn run(session: spark_connect_async::blocking::Session) -> spark_connect_async::Result<()> {
/// for query in session.streams().active()? {
///     println!("{} is {}", query.as_async().id(), query.status()?.message);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct StreamingQueryManager(streaming::StreamingQueryManager);

impl StreamingQueryManager {
    /// These queries as the async [`crate::streaming::StreamingQueryManager`] it holds,
    /// borrowed.
    pub fn as_async(&self) -> &streaming::StreamingQueryManager {
        &self.0
    }

    /// These queries as the async [`crate::streaming::StreamingQueryManager`] it holds, as
    /// [`From`] converts it.
    pub fn into_async(self) -> streaming::StreamingQueryManager {
        self.0
    }

    /// The queries of the session that run.
    pub fn active(&self) -> Result<Vec<StreamingQuery>> {
        wait(self.0.active()).map(|queries| queries.into_iter().map(StreamingQuery).collect())
    }

    /// The query of the session with the id `id`, if it runs.
    pub fn get(&self, id: impl Into<String>) -> Result<Option<StreamingQuery>> {
        wait(self.0.get(id)).map(|query| query.map(StreamingQuery))
    }

    /// Waits until a query of the session terminates, or for `timeout` if it is given, as
    /// [`crate::streaming::StreamingQueryManager::await_any_termination`] does. The call can't
    /// be given up: see [the waits](super#waits-that-only-the-server-ends) for what ends it.
    pub fn await_any_termination(&self, timeout: Option<Duration>) -> Result<bool> {
        wait(self.0.await_any_termination(timeout))
    }

    /// Forgets the queries that have terminated, so that
    /// [`await_any_termination`](Self::await_any_termination) waits for another.
    pub fn reset_terminated(&self) -> Result<()> {
        wait(self.0.reset_terminated())
    }

    /// Subscribes the session to the events of its streaming queries, as
    /// [`crate::streaming::StreamingQueryManager::events`] does, and returns their iterator once
    /// the server has acknowledged the subscription. It fails with
    /// [`ErrorKind::Unsupported`](crate::ErrorKind::Unsupported) while the session has one.
    pub fn events(&self) -> Result<QueryEvents> {
        wait(self.0.events()).map(QueryEvents)
    }
}

impl From<streaming::StreamingQueryManager> for StreamingQueryManager {
    fn from(manager: streaming::StreamingQueryManager) -> Self {
        StreamingQueryManager(manager)
    }
}

impl From<StreamingQueryManager> for streaming::StreamingQueryManager {
    fn from(manager: StreamingQueryManager) -> Self {
        manager.0
    }
}

impl streaming::StreamingQueryManager {
    /// The blocking form of these queries, whose calls block.
    pub fn blocking(&self) -> StreamingQueryManager {
        StreamingQueryManager(self.clone())
    }
}

impl Session {
    /// The streaming queries of the session, and the events of those it starts.
    pub fn streams(&self) -> StreamingQueryManager {
        StreamingQueryManager(self.0.streams())
    }
}

/// The starts of a [`crate::streaming::DataStreamWriter`], from
/// [`crate::streaming::DataStreamWriter::blocking`]: each starts the query, and blocks until the
/// server has started it.
#[derive(Clone)]
#[must_use = "a writer starts nothing until a method starts its query"]
pub struct DataStreamWriter(streaming::DataStreamWriter);

impl DataStreamWriter {
    /// Starts the query, which writes to `path`.
    pub fn start(self, path: impl Into<String>) -> Result<StreamingQuery> {
        wait(self.0.start(path)).map(StreamingQuery)
    }

    /// Starts the query, which writes where the sink's options say, for a sink that takes no
    /// path, such as `memory`.
    pub fn start_without_path(self) -> Result<StreamingQuery> {
        wait(self.0.start_without_path()).map(StreamingQuery)
    }

    /// Starts the query, which writes to the table `name`.
    pub fn to_table(self, name: impl Into<String>) -> Result<StreamingQuery> {
        wait(self.0.to_table(name)).map(StreamingQuery)
    }
}

impl From<streaming::DataStreamWriter> for DataStreamWriter {
    fn from(writer: streaming::DataStreamWriter) -> Self {
        DataStreamWriter(writer)
    }
}

impl From<DataStreamWriter> for streaming::DataStreamWriter {
    fn from(writer: DataStreamWriter) -> Self {
        writer.0
    }
}

impl streaming::DataStreamWriter {
    /// The blocking form of this write, whose starts block: set its format, output mode,
    /// options, trigger and name first.
    pub fn blocking(self) -> DataStreamWriter {
        DataStreamWriter(self)
    }
}

/// A streaming query that runs on the server, whose calls block until the server answers.
///
/// Its id, run id and name are the async [`crate::streaming::StreamingQuery`]'s, which
/// [`as_async`](StreamingQuery::as_async) borrows, and so is its
/// [`explain`](crate::streaming::StreamingQuery::explain), whose builder has a
/// [`run`](crate::streaming::Explain::run). Dropping the handle leaves the query running.
///
/// ```no_run
/// # fn run(session: spark_connect_async::blocking::Session) -> spark_connect_async::Result<()> {
/// let numbers = session.as_async().read_stream().format("rate").load_without_path();
/// let query = numbers
///     .write_stream()
///     .format("memory")
///     .query_name("numbers")
///     .blocking()
///     .start_without_path()?;
/// query.process_all_available()?;
/// println!("{}", query.as_async().explain().run()?);
/// query.stop()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct StreamingQuery(streaming::StreamingQuery);

impl StreamingQuery {
    /// This query as the async [`crate::streaming::StreamingQuery`] it holds, borrowed: its id,
    /// run id, name and explain.
    pub fn as_async(&self) -> &streaming::StreamingQuery {
        &self.0
    }

    /// This query as the async [`crate::streaming::StreamingQuery`] it holds, as [`From`]
    /// converts it.
    pub fn into_async(self) -> streaming::StreamingQuery {
        self.0
    }

    /// Whether the query runs.
    pub fn is_active(&self) -> Result<bool> {
        wait(self.0.is_active())
    }

    /// What the query does now.
    pub fn status(&self) -> Result<QueryStatus> {
        wait(self.0.status())
    }

    /// Stops the query, and returns once it has stopped.
    pub fn stop(&self) -> Result<()> {
        wait(self.0.stop())
    }

    /// Waits until the query terminates, or for `timeout` if it is given, as
    /// [`crate::streaming::StreamingQuery::await_termination`] does. The call can't be given
    /// up: see [the waits](super#waits-that-only-the-server-ends) for what ends it.
    pub fn await_termination(&self, timeout: Option<Duration>) -> Result<bool> {
        wait(self.0.await_termination(timeout))
    }

    /// The progress of the last micro-batch of the query, if it ran one.
    pub fn last_progress(&self) -> Result<Option<QueryProgress>> {
        wait(self.0.last_progress())
    }

    /// The progress of the last micro-batches of the query that the server keeps, oldest first.
    pub fn recent_progress(&self) -> Result<Vec<QueryProgress>> {
        wait(self.0.recent_progress())
    }

    /// Waits until the query has processed all the data that its sources have now, as
    /// [`crate::streaming::StreamingQuery::process_all_available`] does. The call can't be given
    /// up: see [the waits](super#waits-that-only-the-server-ends) for what ends it.
    pub fn process_all_available(&self) -> Result<()> {
        wait(self.0.process_all_available())
    }

    /// The error that terminated the query, if it failed, as
    /// [`crate::streaming::StreamingQuery::exception`] reads it.
    pub fn exception(&self) -> Result<Option<Error>> {
        wait(self.0.exception())
    }
}

impl From<streaming::StreamingQuery> for StreamingQuery {
    fn from(query: streaming::StreamingQuery) -> Self {
        StreamingQuery(query)
    }
}

impl From<StreamingQuery> for streaming::StreamingQuery {
    fn from(query: StreamingQuery) -> Self {
        query.0
    }
}

impl streaming::StreamingQuery {
    /// The blocking form of this query, the same query, whose calls block.
    pub fn blocking(&self) -> StreamingQuery {
        StreamingQuery(self.clone())
    }
}

impl streaming::Explain {
    /// Asks the server for the plan, as awaiting the builder does, and blocks until it answers.
    pub fn run(self) -> Result<String> {
        wait(self.into_future())
    }
}

/// The events of the streaming queries of a session, read as the iterator is advanced: each
/// [`next`](Iterator::next) blocks until the server sends the next event, from
/// [`StreamingQueryManager::events`].
///
/// The events, the errors and the end are those of the async
/// [`crate::streaming::QueryEvents`] that it reads, which [`into_async`](QueryEvents::into_async)
/// returns. Dropping the iterator removes the subscription on the server, from the runtime that
/// it was made on: the runtime of this module, for the iterator of
/// [`StreamingQueryManager::events`]. A `next` can't be given up: see
/// [the waits](super#waits-that-only-the-server-ends) for what ends it.
///
/// ```no_run
/// # fn run(session: spark_connect_async::blocking::Session) -> spark_connect_async::Result<()> {
/// use spark_connect_async::streaming::QueryEvent;
///
/// for event in session.streams().events()? {
///     if let QueryEvent::Terminated(terminated) = event? {
///         println!("{} terminated", terminated.id);
///         break;
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct QueryEvents(streaming::QueryEvents);

impl QueryEvents {
    /// The stream of events that this iterator reads, borrowed.
    pub fn as_async(&self) -> &streaming::QueryEvents {
        &self.0
    }

    /// The stream of events that this iterator reads, with the events not read yet, as
    /// [`From`] converts it.
    pub fn into_async(self) -> streaming::QueryEvents {
        self.0
    }
}

impl Iterator for QueryEvents {
    type Item = Result<QueryEvent>;

    fn next(&mut self) -> Option<Result<QueryEvent>> {
        match runtime() {
            Ok(runtime) => runtime.block_on(self.0.next()),
            Err(error) => Some(Err(error)),
        }
    }
}

impl From<streaming::QueryEvents> for QueryEvents {
    fn from(events: streaming::QueryEvents) -> Self {
        QueryEvents(events)
    }
}

impl From<QueryEvents> for streaming::QueryEvents {
    fn from(events: QueryEvents) -> Self {
        events.0
    }
}

impl streaming::QueryEvents {
    /// The blocking form of these events: an iterator whose items block until they arrive.
    pub fn blocking(self) -> QueryEvents {
        QueryEvents(self)
    }
}
