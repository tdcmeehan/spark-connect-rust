//! Structured streaming: streams read from sources, the queries that write them to sinks, and
//! the events of those queries.
//!
//! [`Session::read_stream`] reads a stream as a [`DataFrame`], and [`DataFrame::write_stream`]
//! starts the [`StreamingQuery`] that writes it. The [`StreamingQueryManager`] of
//! [`Session::streams`] finds the queries of the session, and streams their events.
//!
//! ```no_run
//! # async fn run(session: spark_connect_async::Session) -> spark_connect_async::Result<()> {
//! let numbers = session
//!     .read_stream()
//!     .format("rate")
//!     .option("rowsPerSecond", 10)
//!     .load_without_path();
//! let query = numbers
//!     .write_stream()
//!     .format("memory")
//!     .query_name("numbers")
//!     .start_without_path()
//!     .await?;
//! query.process_all_available().await?;
//! if let Some(progress) = query.last_progress().await? {
//!     println!("batch {} read {} rows", progress.batch_id, progress.num_input_rows);
//! }
//! query.stop().await?;
//! # Ok(())
//! # }
//! ```

mod events;
mod progress;

use std::collections::HashMap;
use std::fmt;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::str::FromStr;
use std::time::Duration;

use spark_connect_core::actions::streaming as core_streaming;
use spark_connect_core::error::{ServerErrorInfo, SparkError, SparkErrorKind};
use spark_connect_core::streaming::{Trigger as CoreTrigger, WriteStreamSpec};
use spark_connect_proto as proto;
use spark_connect_proto::streaming_query_command::Command as QueryCommand;
use spark_connect_proto::streaming_query_command_result::ResultType as QueryResult;
use spark_connect_proto::streaming_query_manager_command::Command as ManagerCommand;
use spark_connect_proto::streaming_query_manager_command_result::ResultType as ManagerResult;

use crate::column::IntoDataType;
use crate::dataframe::{strings, DataFrame};
use crate::error::{Error, Result};
use crate::readwriter::{insert_options, DataFrameReader, OptionValue};
use crate::session::Session;
use crate::stream;

pub(crate) use events::Bus;
pub use events::QueryEvents;
pub use progress::{
    QueryEvent, QueryIdle, QueryProgress, QueryStarted, QueryTerminated, SinkProgress,
    SourceProgress, StateOperatorProgress,
};

impl Session {
    /// A read of a stream from a source, such as files that arrive in a directory, or Kafka.
    pub fn read_stream(&self) -> DataStreamReader {
        DataStreamReader(DataFrameReader::of_stream(self.clone()))
    }

    /// The streaming queries of the session.
    pub fn streams(&self) -> StreamingQueryManager {
        StreamingQueryManager {
            session: self.clone(),
        }
    }
}

impl DataFrame {
    /// A write of this DataFrame, a stream, by a streaming query to a sink.
    pub fn write_stream(&self) -> DataStreamWriter {
        DataStreamWriter {
            input: self.clone(),
            format: None,
            output_mode: None,
            options: HashMap::new(),
            partitioning_columns: Vec::new(),
            clustering_columns: Vec::new(),
            query_name: None,
            trigger: None,
        }
    }
}

/// A read of a stream from a source, from [`Session::read_stream`]: its format, schema and
/// options, then the method that names what to read and returns the DataFrame of the stream.
///
/// The read starts with the query that writes the DataFrame. A later option of the same key
/// replaces an earlier one.
#[derive(Clone)]
#[must_use = "a reader reads nothing until a method names what to read"]
pub struct DataStreamReader(DataFrameReader);

impl DataStreamReader {
    /// Read the format `source`, such as `rate`, `kafka` or `parquet`; by default, the server's
    /// default source.
    pub fn format(self, source: impl Into<String>) -> Self {
        DataStreamReader(self.0.format(source))
    }

    /// Read with this schema, rather than the one that the source infers: a
    /// [`DataType`](crate::types::DataType), or a type in Spark SQL's syntax or as JSON text.
    pub fn schema(self, schema: impl IntoDataType) -> Self {
        DataStreamReader(self.0.schema(schema))
    }

    /// Set the option `key` of the source.
    pub fn option(self, key: impl Into<String>, value: impl Into<OptionValue>) -> Self {
        DataStreamReader(self.0.option(key, value))
    }

    /// Set the options of the source.
    pub fn options<K: Into<String>, V: Into<OptionValue>>(
        self,
        options: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        DataStreamReader(self.0.options(options))
    }

    /// Name the source, so that the checkpoint of a query keeps the source's progress under
    /// the name rather than under the source's place in the query.
    pub fn name(self, name: impl Into<String>) -> Self {
        DataStreamReader(self.0.source_name(name.into()))
    }

    /// The stream of the data at `path`.
    pub fn load(self, path: impl Into<String>) -> DataFrame {
        self.0.load(path)
    }

    /// The stream of the data that the source's options name, for a source that takes no
    /// path, such as `rate`.
    pub fn load_without_path(self) -> DataFrame {
        self.0.load_without_path()
    }

    /// The stream of the CSV files that arrive at `path`.
    pub fn csv(self, path: impl Into<String>) -> DataFrame {
        self.0.csv(path)
    }

    /// The stream of the JSON Lines files that arrive at `path`.
    pub fn json(self, path: impl Into<String>) -> DataFrame {
        self.0.json(path)
    }

    /// The stream of the ORC files that arrive at `path`.
    pub fn orc(self, path: impl Into<String>) -> DataFrame {
        self.0.orc(path)
    }

    /// The stream of the Parquet files that arrive at `path`.
    pub fn parquet(self, path: impl Into<String>) -> DataFrame {
        self.0.parquet(path)
    }

    /// The stream of the lines of the text files that arrive at `path`, in a column `value`.
    pub fn text(self, path: impl Into<String>) -> DataFrame {
        self.0.text(path)
    }

    /// The stream of the rows that are added to the table `name`.
    pub fn table(self, name: impl Into<String>) -> DataFrame {
        self.0.table(name)
    }

    /// The stream of the changes of the table `name`, from where the options of this reader
    /// say, such as `startingVersion`.
    pub fn changes(self, name: impl Into<String>) -> DataFrame {
        self.0.changes(name)
    }
}

/// Which rows of the result of a streaming query each micro-batch writes to the sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputMode {
    /// The rows that are new in the result.
    Append,
    /// The whole result.
    Complete,
    /// The rows of the result that changed.
    Update,
}

impl OutputMode {
    fn as_str(self) -> &'static str {
        match self {
            OutputMode::Append => "append",
            OutputMode::Complete => "complete",
            OutputMode::Update => "update",
        }
    }
}

/// Parses `append`, `complete` and `update`, in any case.
impl FromStr for OutputMode {
    type Err = Error;

    fn from_str(mode: &str) -> Result<OutputMode> {
        [OutputMode::Append, OutputMode::Complete, OutputMode::Update]
            .into_iter()
            .find(|output_mode| output_mode.as_str().eq_ignore_ascii_case(mode))
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "{mode:?} is not an output mode: append, complete or update"
                ))
            })
    }
}

/// When a streaming query processes its new data. An interval is in Spark's syntax, such as
/// `"10 seconds"`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Trigger {
    /// A micro-batch at each interval, or at once after one that took longer.
    ProcessingTime(String),
    /// One micro-batch of the data that is available, then the query stops.
    Once,
    /// Micro-batches of the data that is available when the query starts, then the query
    /// stops.
    AvailableNow,
    /// Continuous processing, with a checkpoint at each interval.
    Continuous(String),
    /// Real-time processing, in batches of the interval.
    RealTime(String),
}

impl Trigger {
    fn core(self) -> CoreTrigger {
        match self {
            Trigger::ProcessingTime(interval) => CoreTrigger::ProcessingTime(interval),
            Trigger::Once => CoreTrigger::Once,
            Trigger::AvailableNow => CoreTrigger::AvailableNow,
            Trigger::Continuous(interval) => CoreTrigger::Continuous(interval),
            Trigger::RealTime(interval) => CoreTrigger::RealTime(interval),
        }
    }
}

/// A write of a stream to a sink by a streaming query, from [`DataFrame::write_stream`]: its
/// format, options, output mode and trigger, then the method that names the destination and
/// returns the future that starts the query.
#[derive(Clone)]
#[must_use = "a writer starts nothing until a method starts its query"]
pub struct DataStreamWriter {
    input: DataFrame,
    format: Option<String>,
    output_mode: Option<OutputMode>,
    options: HashMap<String, String>,
    partitioning_columns: Vec<String>,
    clustering_columns: Vec<String>,
    query_name: Option<String>,
    trigger: Option<Trigger>,
}

impl DataStreamWriter {
    /// Write to the sink `format`, such as `parquet`, `kafka`, `memory` or `console`; by
    /// default, the server's default source.
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Write the rows that `mode` says; by default, those that are new in the result.
    pub fn output_mode(mut self, mode: OutputMode) -> Self {
        self.output_mode = Some(mode);
        self
    }

    /// Set the option `key` of the sink, or of the query, such as `checkpointLocation`.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<OptionValue>) -> Self {
        insert_options(&mut self.options, [(key, value)]);
        self
    }

    /// Set the options of the sink and of the query.
    pub fn options<K: Into<String>, V: Into<OptionValue>>(
        mut self,
        options: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        insert_options(&mut self.options, options);
        self
    }

    /// Partition the output of a file sink by the columns `names`.
    pub fn partition_by(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.partitioning_columns = strings(names);
        self
    }

    /// Cluster the output by the columns `names`.
    pub fn cluster_by(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.clustering_columns = strings(names);
        self
    }

    /// Name the query: the name of the table of a `memory` sink, and of the query in its
    /// events. The session's active queries have distinct names.
    pub fn query_name(mut self, name: impl Into<String>) -> Self {
        self.query_name = Some(name.into());
        self
    }

    /// Process the data when `trigger` says; by default, a micro-batch as soon as the last one
    /// ended.
    pub fn trigger(mut self, trigger: Trigger) -> Self {
        self.trigger = Some(trigger);
        self
    }

    fn start_query(
        self,
        path: Option<String>,
        table_name: Option<String>,
    ) -> impl Future<Output = Result<StreamingQuery>> + Send + 'static {
        let session = self.input.session().clone();
        let name = self.query_name.clone();
        let start = WriteStreamSpec {
            format: self.format,
            output_mode: self.output_mode.map(|mode| mode.as_str().to_string()),
            options: self.options,
            partitioning_columns: self.partitioning_columns,
            clustering_columns: self.clustering_columns,
            query_name: self.query_name,
            trigger: self.trigger.map(Trigger::core),
            path,
            table_name,
            foreach_batch_payload: None,
            foreach_payload: None,
        }
        .to_proto(self.input.to_relation());
        self.input.holding(async move {
            let command = proto::command::CommandType::WriteStreamOperationStart(start);
            let result =
                stream::command_result(session.clone(), command, core_streaming::start_result)
                    .await?;
            let started = core_streaming::started(result)?;
            if let Some(json) = started.started_event_json {
                session.listener_bus().post_started(json);
            }
            Ok(StreamingQuery {
                session,
                id: started.query_id,
                run_id: started.run_id,
                name: started.name.or(name),
            })
        })
    }

    /// Starts the query, which writes to `path`, and returns it once the server has started it.
    pub fn start(
        self,
        path: impl Into<String>,
    ) -> impl Future<Output = Result<StreamingQuery>> + Send + 'static {
        self.start_query(Some(path.into()), None)
    }

    /// Starts the query, which writes where the sink's options say, for a sink that takes no
    /// path, such as `memory`, and returns it once the server has started it.
    pub fn start_without_path(
        self,
    ) -> impl Future<Output = Result<StreamingQuery>> + Send + 'static {
        self.start_query(None, None)
    }

    /// Starts the query, which writes to the table `name`, and returns it once the server has
    /// started it.
    pub fn to_table(
        self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<StreamingQuery>> + Send + 'static {
        self.start_query(None, Some(name.into()))
    }
}

/// The milliseconds of `timeout`, at most `i64::MAX`. The server waits at least a millisecond.
fn timeout_ms(timeout: Option<Duration>) -> Result<Option<i64>> {
    let Some(timeout) = timeout else {
        return Ok(None);
    };
    match i64::try_from(timeout.as_millis()) {
        Ok(0) => Err(Error::invalid_argument(format!(
            "the timeout {timeout:?} is shorter than a millisecond, the shortest that the server \
             waits"
        ))),
        Ok(millis) => Ok(Some(millis)),
        Err(_) => Ok(Some(i64::MAX)),
    }
}

/// Whether a wait for queries with the timeout of `waited` ended with them terminated, by
/// `result`, the server's answer.
fn terminated(waited: Option<i64>, result: Option<bool>) -> Result<bool> {
    match (result, waited) {
        (Some(terminated), _) => Ok(terminated),
        // Without a timeout, the server answers once the queries have terminated.
        (None, None) => Ok(true),
        (None, Some(_)) => Err(Error::decode("the server returned no answer")),
    }
}

/// A streaming query that runs on the server, which a session started.
///
/// The handle can be cloned, and dropping it leaves the query running: [`StreamingQuery::stop`]
/// stops it.
#[derive(Clone)]
pub struct StreamingQuery {
    session: Session,
    id: String,
    run_id: String,
    name: Option<String>,
}

impl fmt::Debug for StreamingQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamingQuery")
            .field("id", &self.id)
            .field("run_id", &self.run_id)
            .field("name", &self.name)
            .finish()
    }
}

impl StreamingQuery {
    /// The id of the query, which it keeps when it restarts from its checkpoint.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The id of this run of the query.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// The name of the query, if it has one.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn command(
        &self,
        command: QueryCommand,
    ) -> impl Future<Output = Result<Option<QueryResult>>> + Send + 'static {
        let command = proto::StreamingQueryCommand {
            query_id: Some(proto::StreamingQueryInstanceId {
                id: self.id.clone(),
                run_id: self.run_id.clone(),
            }),
            command: Some(command),
        };
        let command = proto::command::CommandType::StreamingQueryCommand(command);
        let result =
            stream::command_result(self.session.clone(), command, core_streaming::query_result);
        async move { Ok(result.await?.and_then(|result| result.result_type)) }
    }

    /// Whether the query runs.
    pub fn is_active(&self) -> impl Future<Output = Result<bool>> + Send + 'static {
        let status = self.status();
        async move { Ok(status.await?.is_active) }
    }

    /// What the query does now.
    pub fn status(&self) -> impl Future<Output = Result<QueryStatus>> + Send + 'static {
        let result = self.command(QueryCommand::Status(true));
        async move {
            match result.await? {
                Some(QueryResult::Status(status)) => Ok(QueryStatus {
                    message: status.status_message,
                    is_data_available: status.is_data_available,
                    is_trigger_active: status.is_trigger_active,
                    is_active: status.is_active,
                }),
                _ => Err(Error::decode("the server returned no status")),
            }
        }
    }

    /// Stops the query, and returns once it has stopped.
    pub fn stop(&self) -> impl Future<Output = Result<()>> + Send + 'static {
        let result = self.command(QueryCommand::Stop(true));
        async move { result.await.map(drop) }
    }

    /// Waits until the query terminates, or for `timeout` if it is given, and returns whether
    /// the query has terminated. A query that failed fails the wait with its error.
    ///
    /// Dropping the future stops the wait on the server. Fails with
    /// [`ErrorKind::InvalidArgument`](crate::ErrorKind::InvalidArgument) for a timeout shorter
    /// than a millisecond, before it sends anything.
    pub fn await_termination(
        &self,
        timeout: Option<Duration>,
    ) -> impl Future<Output = Result<bool>> + Send + 'static {
        let waited = timeout_ms(timeout);
        let result = waited.as_ref().ok().map(|&timeout_ms| {
            self.command(QueryCommand::AwaitTermination(
                proto::streaming_query_command::AwaitTerminationCommand { timeout_ms },
            ))
        });
        async move {
            let waited = waited?;
            let answer = match result {
                Some(result) => match result.await? {
                    Some(QueryResult::AwaitTermination(answer)) => Some(answer.terminated),
                    _ => None,
                },
                None => None,
            };
            terminated(waited, answer)
        }
    }

    /// The progress of the last micro-batch of the query, if it ran one.
    pub fn last_progress(
        &self,
    ) -> impl Future<Output = Result<Option<QueryProgress>>> + Send + 'static {
        let result = self.command(QueryCommand::LastProgress(true));
        async move {
            match result.await? {
                Some(QueryResult::RecentProgress(progress)) => progress
                    .recent_progress_json
                    .into_iter()
                    .last()
                    .map(progress::progress_of)
                    .transpose(),
                _ => Ok(None),
            }
        }
    }

    /// The progress of the last micro-batches of the query that the server keeps, oldest
    /// first.
    pub fn recent_progress(
        &self,
    ) -> impl Future<Output = Result<Vec<QueryProgress>>> + Send + 'static {
        let result = self.command(QueryCommand::RecentProgress(true));
        async move {
            match result.await? {
                Some(QueryResult::RecentProgress(progress)) => progress
                    .recent_progress_json
                    .into_iter()
                    .map(progress::progress_of)
                    .collect(),
                _ => Ok(Vec::new()),
            }
        }
    }

    /// Waits until the query has processed all the data that its sources have now. A query that
    /// failed fails the wait with its error.
    ///
    /// Dropping the future stops the wait on the server.
    pub fn process_all_available(&self) -> impl Future<Output = Result<()>> + Send + 'static {
        let result = self.command(QueryCommand::ProcessAllAvailable(true));
        async move { result.await.map(drop) }
    }

    /// The plan of the query, as text.
    pub fn explain(&self) -> Explain {
        Explain {
            query: self.clone(),
            extended: false,
        }
    }

    /// The error that terminated the query, if it failed, with its condition, such as
    /// `STREAM_FAILED`. The server sends no SQL state with it; the error of
    /// [`StreamingQuery::await_termination`] has one.
    pub fn exception(&self) -> impl Future<Output = Result<Option<Error>>> + Send + 'static {
        let result = self.command(QueryCommand::Exception(true));
        async move {
            match result.await? {
                Some(QueryResult::Exception(exception)) => Ok(query_exception(exception)),
                _ => Ok(None),
            }
        }
    }
}

/// The error of `exception`, as the server reports the error that terminated a query: its
/// message is the error's text without the JVM class that the server's text starts with.
fn query_exception(
    exception: proto::streaming_query_command_result::ExceptionResult,
) -> Option<Error> {
    let text = exception
        .exception_message
        .filter(|text| !text.is_empty())?;
    let (classes, message) = match text.split_once(": ") {
        Some((class, message)) if is_jvm_class(class) => {
            (vec![class.to_string()], message.to_string())
        }
        _ => (Vec::new(), text),
    };
    let mut error = SparkError::connect_msg(message);
    error.kind = SparkErrorKind::StreamingQuery;
    error.error_class = exception.error_class.unwrap_or_default();
    error.server_stacktrace = exception.stack_trace;
    error.server = Some(Box::new(ServerErrorInfo {
        classes,
        ..Default::default()
    }));
    Some(error.into())
}

/// Whether `name` is the name of a JVM class, such as `java.lang.Exception`.
fn is_jvm_class(name: &str) -> bool {
    name.contains('.')
        && name.split('.').all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '$')
        })
}

/// What a streaming query does, as [`StreamingQuery::status`] returns it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QueryStatus {
    /// What the query does, such as `Waiting for data to arrive`.
    pub message: String,
    /// Whether a source has new data.
    pub is_data_available: bool,
    /// Whether a trigger runs.
    pub is_trigger_active: bool,
    /// Whether the query runs.
    pub is_active: bool,
}

/// The plan of a streaming query, from [`StreamingQuery::explain`]. Awaiting it asks the
/// server.
#[must_use = "the plan is asked for only when the builder is awaited"]
pub struct Explain {
    query: StreamingQuery,
    extended: bool,
}

impl Explain {
    /// Also the logical plans of the query, before the physical one.
    pub fn extended(mut self) -> Self {
        self.extended = true;
        self
    }
}

impl IntoFuture for Explain {
    type Output = Result<String>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        let result = self.query.command(QueryCommand::Explain(
            proto::streaming_query_command::ExplainCommand {
                extended: self.extended,
            },
        ));
        Box::pin(async move {
            match result.await? {
                Some(QueryResult::Explain(explain)) => Ok(explain.result),
                _ => Ok(String::new()),
            }
        })
    }
}

/// The streaming queries of a session, from [`Session::streams`].
#[derive(Clone)]
pub struct StreamingQueryManager {
    session: Session,
}

impl StreamingQueryManager {
    fn command(
        &self,
        command: ManagerCommand,
    ) -> impl Future<Output = Result<Option<ManagerResult>>> + Send + 'static {
        let command = proto::StreamingQueryManagerCommand {
            command: Some(command),
        };
        let command = proto::command::CommandType::StreamingQueryManagerCommand(command);
        let result = stream::command_result(
            self.session.clone(),
            command,
            core_streaming::manager_result,
        );
        async move { Ok(result.await?.and_then(|result| result.result_type)) }
    }

    fn query(
        &self,
        instance: proto::streaming_query_manager_command_result::StreamingQueryInstance,
    ) -> Result<StreamingQuery> {
        let id = instance
            .id
            .ok_or_else(|| Error::decode("the server returned a query without an id"))?;
        Ok(StreamingQuery {
            session: self.session.clone(),
            id: id.id,
            run_id: id.run_id,
            name: instance.name,
        })
    }

    /// The queries of the session that run.
    pub fn active(&self) -> impl Future<Output = Result<Vec<StreamingQuery>>> + Send + 'static {
        let manager = self.clone();
        let result = self.command(ManagerCommand::Active(true));
        async move {
            match result.await? {
                Some(ManagerResult::Active(active)) => active
                    .active_queries
                    .into_iter()
                    .map(|instance| manager.query(instance))
                    .collect(),
                _ => Ok(Vec::new()),
            }
        }
    }

    /// The query of the session with the id `id`, if it runs.
    pub fn get(
        &self,
        id: impl Into<String>,
    ) -> impl Future<Output = Result<Option<StreamingQuery>>> + Send + 'static {
        let manager = self.clone();
        let result = self.command(ManagerCommand::GetQuery(id.into()));
        async move {
            match result.await? {
                Some(ManagerResult::Query(instance)) => manager.query(instance).map(Some),
                _ => Ok(None),
            }
        }
    }

    /// Waits until a query of the session terminates, or for `timeout` if it is given, and
    /// returns whether one has terminated since the session started or since
    /// [`StreamingQueryManager::reset_terminated`]. A query that failed fails the wait with its
    /// error.
    ///
    /// Dropping the future stops the wait on the server. Fails with
    /// [`ErrorKind::InvalidArgument`](crate::ErrorKind::InvalidArgument) for a timeout shorter
    /// than a millisecond, before it sends anything.
    pub fn await_any_termination(
        &self,
        timeout: Option<Duration>,
    ) -> impl Future<Output = Result<bool>> + Send + 'static {
        let waited = timeout_ms(timeout);
        let result = waited.as_ref().ok().map(|&timeout_ms| {
            self.command(ManagerCommand::AwaitAnyTermination(
                proto::streaming_query_manager_command::AwaitAnyTerminationCommand { timeout_ms },
            ))
        });
        async move {
            let waited = waited?;
            let answer = match result {
                Some(result) => match result.await? {
                    Some(ManagerResult::AwaitAnyTermination(answer)) => Some(answer.terminated),
                    _ => None,
                },
                None => None,
            };
            terminated(waited, answer)
        }
    }

    /// Forgets the queries that have terminated, so that
    /// [`StreamingQueryManager::await_any_termination`] waits for another.
    pub fn reset_terminated(&self) -> impl Future<Output = Result<()>> + Send + 'static {
        let result = self.command(ManagerCommand::ResetTerminated(true));
        async move { result.await.map(drop) }
    }

    /// Subscribes the session to the events of its streaming queries, and returns their stream
    /// once the server has acknowledged the subscription: the queries that start after that
    /// report their start, progress and end.
    ///
    /// The server keeps one subscription for each session, so this fails with
    /// [`ErrorKind::Unsupported`](crate::ErrorKind::Unsupported) while the session has one,
    /// and waits while the last one ends: while its removal is on its way to the server, or
    /// while a subscription whose future was dropped stops. While another client of the same session
    /// on the server holds the subscription, the server does not answer: the future waits until
    /// it is dropped, which ends the attempt. Dropping it once the server has answered removes
    /// the subscription.
    pub fn events(&self) -> impl Future<Output = Result<QueryEvents>> + Send + 'static {
        events::subscribe(self.session.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    #[test]
    fn a_timeout_is_sent_in_whole_milliseconds_from_one_to_the_largest() {
        let cases = [
            (None, Some(None)),
            (Some(Duration::from_millis(1)), Some(Some(1))),
            (Some(Duration::from_micros(1_500)), Some(Some(1))),
            (Some(Duration::from_secs(90)), Some(Some(90_000))),
            (Some(Duration::MAX), Some(Some(i64::MAX))),
            (Some(Duration::ZERO), None),
            (Some(Duration::from_micros(999)), None),
        ];
        for (timeout, expected) in cases {
            match (timeout_ms(timeout), expected) {
                (Ok(sent), Some(expected)) => assert_eq!(sent, expected, "{timeout:?}"),
                (Err(error), None) => {
                    assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{timeout:?}")
                }
                (result, _) => panic!("{timeout:?}: {result:?}"),
            }
        }
    }

    #[test]
    fn a_wait_has_terminated_as_the_server_answers_or_else_when_it_had_no_timeout() {
        let cases = [
            (None, Some(true), Some(true)),
            (Some(10), Some(false), Some(false)),
            (Some(10), Some(true), Some(true)),
            (None, None, Some(true)),
            (Some(10), None, None),
        ];
        for (waited, answer, expected) in cases {
            match (terminated(waited, answer), expected) {
                (Ok(terminated), Some(expected)) => {
                    assert_eq!(terminated, expected, "{waited:?} {answer:?}")
                }
                (Err(error), None) => assert_eq!(error.kind(), ErrorKind::Decode),
                (result, _) => panic!("{waited:?} {answer:?}: {result:?}"),
            }
        }
    }

    #[test]
    fn the_exception_of_a_query_is_an_error_of_its_condition_without_its_jvm_class() {
        use proto::streaming_query_command_result::ExceptionResult;
        // As a 4.3.0 server sends a query's failure: its message is the error's toString.
        let failed = ExceptionResult {
            exception_message: Some(
                "org.apache.spark.sql.streaming.StreamingQueryException: [USER_RAISED_EXCEPTION] \
                 boom SQLSTATE: P0001\n=== Streaming Query ==="
                    .to_string(),
            ),
            error_class: Some("STREAM_FAILED".to_string()),
            stack_trace: Some("at org.apache.spark.sql.execution.streaming".to_string()),
        };
        let error = query_exception(failed).unwrap();
        assert_eq!(error.kind(), ErrorKind::Execution);
        assert_eq!(error.error_class(), Some("STREAM_FAILED"));
        assert_eq!(
            error.message(),
            "[USER_RAISED_EXCEPTION] boom SQLSTATE: P0001\n=== Streaming Query ==="
        );
        assert_eq!(
            error.server_stack_trace(),
            Some("at org.apache.spark.sql.execution.streaming")
        );
        assert_eq!(error.sql_state(), None);
        let unclassed = ExceptionResult {
            exception_message: Some("Query [id = a.b] failed: no class".to_string()),
            ..Default::default()
        };
        let error = query_exception(unclassed).unwrap();
        assert_eq!(error.message(), "Query [id = a.b] failed: no class");
        assert_eq!(error.error_class(), None);
        for none in [
            ExceptionResult::default(),
            ExceptionResult {
                exception_message: Some(String::new()),
                ..Default::default()
            },
        ] {
            assert!(query_exception(none).is_none());
        }
    }

    #[test]
    fn output_modes_parse_in_any_case_and_other_text_is_rejected() {
        for (text, mode) in [
            ("append", OutputMode::Append),
            ("Complete", OutputMode::Complete),
            ("UPDATE", OutputMode::Update),
        ] {
            assert_eq!(text.parse::<OutputMode>().unwrap(), mode);
            assert!(mode.as_str().eq_ignore_ascii_case(text));
        }
        for text in ["", "overwrite", "appends"] {
            let error = text.parse::<OutputMode>().unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{text:?}");
        }
    }
}
