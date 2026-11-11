//! The progress and the events of streaming queries, read from the JSON that the server sends.

use std::collections::BTreeMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::{Map, Value};
use spark_connect_proto::StreamingQueryEventType;

use crate::error::{Error, Result};

/// The progress of a streaming query in one of its micro-batches.
///
/// A field that Spark leaves out or sends as `null` is `None`, or an empty map.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct QueryProgress {
    /// The id of the query, which it keeps when it restarts from its checkpoint.
    pub id: String,
    /// The id of this run of the query.
    pub run_id: String,
    /// The name of the query, if it has one.
    pub name: Option<String>,
    /// When the micro-batch started.
    pub timestamp: DateTime<Utc>,
    /// The number of the micro-batch, from 0.
    pub batch_id: i64,
    /// How long the micro-batch took.
    pub batch_duration: Duration,
    /// The number of rows that the micro-batch read from all the sources.
    pub num_input_rows: i64,
    /// How many rows arrived per second from all the sources, if Spark could tell for each.
    pub input_rows_per_second: Option<f64>,
    /// How many rows the micro-batch processed per second from all the sources, if Spark
    /// could tell for each.
    pub processed_rows_per_second: Option<f64>,
    /// How long each step of the micro-batch took, by the name of the step, such as
    /// `addBatch`.
    pub durations: BTreeMap<String, Duration>,
    /// The statistics of the event times of the rows, such as `max` and `watermark`, for a
    /// query with an event-time watermark.
    pub event_time: BTreeMap<String, DateTime<Utc>>,
    /// The progress of each stateful operator of the query.
    pub state_operators: Vec<StateOperatorProgress>,
    /// The progress of each source of the query.
    pub sources: Vec<SourceProgress>,
    /// The progress of the sink of the query.
    pub sink: SinkProgress,
    /// The metrics of each observation of the query, by its name, as the JSON of an object of
    /// the metrics.
    pub observed_metrics: BTreeMap<String, String>,
    /// The JSON that the progress was read from: the server's text for
    /// [`StreamingQuery::last_progress`](super::StreamingQuery::last_progress) and
    /// [`StreamingQuery::recent_progress`](super::StreamingQuery::recent_progress), and for a
    /// [`QueryEvent::Progress`] the event's member `progress`, written again with the keys of
    /// each object sorted.
    pub json: String,
}

/// The progress of a stateful operator of a streaming query, such as an aggregation, in a
/// micro-batch.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StateOperatorProgress {
    /// The name of the operator.
    pub operator_name: String,
    /// The number of rows in its state.
    pub num_rows_total: i64,
    /// The number of rows of its state that the micro-batch updated.
    pub num_rows_updated: i64,
    /// The number of rows that the micro-batch removed from its state.
    pub num_rows_removed: i64,
    /// How long the updates of its state took.
    pub all_updates_time: Duration,
    /// How long the removals from its state took.
    pub all_removals_time: Duration,
    /// How long the commit of its state took.
    pub commit_time: Duration,
    /// The memory that its state uses, in bytes.
    pub memory_used_bytes: i64,
    /// The number of rows that it dropped as later than the watermark.
    pub num_rows_dropped_by_watermark: i64,
    /// The number of partitions of its state.
    pub num_shuffle_partitions: i64,
    /// The number of instances of the store of its state.
    pub num_state_store_instances: i64,
    /// The metrics of the store of its state, by their names.
    pub custom_metrics: BTreeMap<String, i64>,
}

/// The progress of a source of a streaming query in a micro-batch.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SourceProgress {
    /// A description of the source.
    pub description: String,
    /// Where the micro-batch started reading, as the source writes its offsets, if it had
    /// read before: JSON text, or the text of an offset that is not JSON.
    pub start_offset: Option<String>,
    /// Where the micro-batch stopped reading, as [`SourceProgress::start_offset`] is written.
    pub end_offset: Option<String>,
    /// The latest offset of the source when the micro-batch started, if it has one.
    pub latest_offset: Option<String>,
    /// The number of rows that the micro-batch read from the source.
    pub num_input_rows: i64,
    /// How many rows arrived per second, if Spark could tell.
    pub input_rows_per_second: Option<f64>,
    /// How many rows the micro-batch processed per second, if Spark could tell.
    pub processed_rows_per_second: Option<f64>,
    /// The metrics of the source, by their names.
    pub metrics: BTreeMap<String, String>,
}

/// The progress of the sink of a streaming query in a micro-batch.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SinkProgress {
    /// A description of the sink.
    pub description: String,
    /// The number of rows that the micro-batch wrote, or -1 if the sink does not count them.
    pub num_output_rows: i64,
    /// The metrics of the sink, by their names.
    pub metrics: BTreeMap<String, String>,
}

/// An event of a streaming query of a session, from
/// [`QueryEvents`](super::QueryEvents).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QueryEvent {
    /// A query that the session started began to run.
    Started(QueryStarted),
    /// A query ended a micro-batch.
    Progress(Box<QueryProgress>),
    /// A query found no new data to process, and so ran no micro-batch.
    Idle(QueryIdle),
    /// A query stopped, or failed.
    Terminated(QueryTerminated),
}

impl QueryEvent {
    /// The id of the query of the event.
    pub fn id(&self) -> &str {
        match self {
            QueryEvent::Started(started) => &started.id,
            QueryEvent::Progress(progress) => &progress.id,
            QueryEvent::Idle(idle) => &idle.id,
            QueryEvent::Terminated(terminated) => &terminated.id,
        }
    }
}

/// The start of a streaming query.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QueryStarted {
    /// The id of the query.
    pub id: String,
    /// The id of this run of the query.
    pub run_id: String,
    /// The name of the query, if it has one.
    pub name: Option<String>,
    /// When the query started.
    pub timestamp: DateTime<Utc>,
    /// The tags of the jobs of the query.
    pub job_tags: Vec<String>,
    /// The JSON of the event, as the server sent it.
    pub json: String,
}

/// A trigger of a streaming query that found no new data.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QueryIdle {
    /// The id of the query.
    pub id: String,
    /// The id of this run of the query.
    pub run_id: String,
    /// When the query found no new data.
    pub timestamp: DateTime<Utc>,
    /// The JSON of the event, as the server sent it.
    pub json: String,
}

/// The end of a streaming query.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QueryTerminated {
    /// The id of the query.
    pub id: String,
    /// The id of this run of the query.
    pub run_id: String,
    /// The error that ended the query, as text, if it failed.
    pub exception: Option<String>,
    /// The condition of that error, such as `STREAM_FAILED`, if it has one.
    pub error_class: Option<String>,
    /// The JSON of the event, as the server sent it.
    pub json: String,
}

const PROGRESS: &str = "the progress of a streaming query";
const EVENT: &str = "an event of a streaming query";

/// The progress of `json`, a progress of a query as the server sends it.
pub(crate) fn progress_of(json: String) -> Result<QueryProgress> {
    let value = parsed(&json, PROGRESS)?;
    progress(&Object::of(&value, String::new(), PROGRESS)?, json)
}

/// The started event of a query, whose JSON the server sends with the result of its start.
pub(crate) fn started_of(json: String) -> Result<QueryEvent> {
    let value = parsed(&json, EVENT)?;
    let event = Object::of(&value, String::new(), EVENT)?;
    Ok(QueryEvent::Started(QueryStarted {
        id: event.text("id")?,
        run_id: event.text("runId")?,
        name: event.optional_text("name")?,
        timestamp: event.timestamp("timestamp")?,
        job_tags: event.texts("jobTags")?,
        json,
    }))
}

/// The event of the type `event_type` of the listener bus, with the JSON `json`.
pub(crate) fn event_of(event_type: i32, json: String) -> Result<QueryEvent> {
    let value = parsed(&json, EVENT)?;
    let event = Object::of(&value, String::new(), EVENT)?;
    match StreamingQueryEventType::try_from(event_type) {
        Ok(StreamingQueryEventType::QueryProgressEvent) => {
            let member = event.object("progress")?;
            let text = event.required("progress")?.to_string();
            Ok(QueryEvent::Progress(Box::new(progress(&member, text)?)))
        }
        Ok(StreamingQueryEventType::QueryIdleEvent) => Ok(QueryEvent::Idle(QueryIdle {
            id: event.text("id")?,
            run_id: event.text("runId")?,
            timestamp: event.timestamp("timestamp")?,
            json,
        })),
        Ok(StreamingQueryEventType::QueryTerminatedEvent) => {
            Ok(QueryEvent::Terminated(QueryTerminated {
                id: event.text("id")?,
                run_id: event.text("runId")?,
                exception: event.optional_text("exception")?,
                error_class: event.optional_text("errorClassOnException")?,
                json,
            }))
        }
        _ => Err(Error::decode(format!(
            "the server sent {EVENT} of the unknown type {event_type}"
        ))),
    }
}

fn parsed(json: &str, what: &str) -> Result<Value> {
    serde_json::from_str(json)
        .map_err(|error| Error::decode(format!("{what} is not JSON: {error}")))
}

fn progress(progress: &Object<'_>, json: String) -> Result<QueryProgress> {
    let sources = progress
        .objects("sources")?
        .iter()
        .map(source)
        .collect::<Result<Vec<_>>>()?;
    // Spark derives these from the sources, and leaves them out of some of its texts.
    let num_input_rows = match progress.optional_long("numInputRows")? {
        Some(rows) => rows,
        None => sources.iter().map(|source| source.num_input_rows).sum(),
    };
    let rate =
        |name: &str, of_source: fn(&SourceProgress) -> Option<f64>| match progress.value(name) {
            Some(_) => progress.rate(name),
            None => Ok(sources.iter().map(of_source).sum()),
        };
    Ok(QueryProgress {
        id: progress.text("id")?,
        run_id: progress.text("runId")?,
        name: progress.optional_text("name")?,
        timestamp: progress.timestamp("timestamp")?,
        batch_id: progress.long("batchId")?,
        batch_duration: progress.duration("batchDuration")?,
        num_input_rows,
        input_rows_per_second: rate("inputRowsPerSecond", |source| source.input_rows_per_second)?,
        processed_rows_per_second: rate("processedRowsPerSecond", |source| {
            source.processed_rows_per_second
        })?,
        durations: progress.map("durationMs", millis, "a number of milliseconds")?,
        event_time: progress.map("eventTime", timestamp, "a timestamp")?,
        state_operators: progress
            .objects("stateOperators")?
            .iter()
            .map(state_operator)
            .collect::<Result<_>>()?,
        sources,
        sink: sink(&progress.object("sink")?)?,
        observed_metrics: progress.map(
            "observedMetrics",
            |row| row.is_object().then(|| row.to_string()),
            "an object",
        )?,
        json,
    })
}

fn state_operator(operator: &Object<'_>) -> Result<StateOperatorProgress> {
    Ok(StateOperatorProgress {
        operator_name: operator.text("operatorName")?,
        num_rows_total: operator.long("numRowsTotal")?,
        num_rows_updated: operator.long("numRowsUpdated")?,
        num_rows_removed: operator.long("numRowsRemoved")?,
        all_updates_time: operator.duration("allUpdatesTimeMs")?,
        all_removals_time: operator.duration("allRemovalsTimeMs")?,
        commit_time: operator.duration("commitTimeMs")?,
        memory_used_bytes: operator.long("memoryUsedBytes")?,
        num_rows_dropped_by_watermark: operator.long("numRowsDroppedByWatermark")?,
        num_shuffle_partitions: operator.long("numShufflePartitions")?,
        num_state_store_instances: operator.long("numStateStoreInstances")?,
        custom_metrics: operator.map("customMetrics", Value::as_i64, "an integer")?,
    })
}

fn source(source: &Object<'_>) -> Result<SourceProgress> {
    Ok(SourceProgress {
        description: source.text("description")?,
        start_offset: source.offset("startOffset"),
        end_offset: source.offset("endOffset"),
        latest_offset: source.offset("latestOffset"),
        num_input_rows: source.long("numInputRows")?,
        input_rows_per_second: source.rate("inputRowsPerSecond")?,
        processed_rows_per_second: source.rate("processedRowsPerSecond")?,
        metrics: source.map("metrics", text, "a string")?,
    })
}

fn sink(sink: &Object<'_>) -> Result<SinkProgress> {
    Ok(SinkProgress {
        description: sink.text("description")?,
        num_output_rows: sink.long("numOutputRows")?,
        metrics: sink.map("metrics", text, "a string")?,
    })
}

fn millis(value: &Value) -> Option<Duration> {
    value.as_u64().map(Duration::from_millis)
}

fn timestamp(value: &Value) -> Option<DateTime<Utc>> {
    value.as_str()?.parse().ok()
}

fn text(value: &Value) -> Option<String> {
    value.as_str().map(str::to_string)
}

/// A JSON object of `what` that the server sent, at `path` in it.
struct Object<'a> {
    members: &'a Map<String, Value>,
    path: String,
    what: &'static str,
}

impl<'a> Object<'a> {
    fn of(value: &'a Value, path: String, what: &'static str) -> Result<Self> {
        match value {
            Value::Object(members) => Ok(Object {
                members,
                path,
                what,
            }),
            _ => Err(Error::decode(format!(
                "{} in {what} is not an object",
                if path.is_empty() { "the JSON" } else { &path }
            ))),
        }
    }

    fn path(&self, name: &str) -> String {
        if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{name}", self.path)
        }
    }

    fn invalid(&self, name: &str, expected: &str) -> Error {
        Error::decode(format!(
            "`{}` in {} is not {expected}",
            self.path(name),
            self.what
        ))
    }

    /// The member `name`, unless it is absent or `null`.
    fn value(&self, name: &str) -> Option<&'a Value> {
        self.members.get(name).filter(|value| !value.is_null())
    }

    fn required(&self, name: &str) -> Result<&'a Value> {
        self.value(name)
            .ok_or_else(|| Error::decode(format!("{} has no `{}`", self.what, self.path(name))))
    }

    fn text(&self, name: &str) -> Result<String> {
        text(self.required(name)?).ok_or_else(|| self.invalid(name, "a string"))
    }

    fn optional_text(&self, name: &str) -> Result<Option<String>> {
        self.value(name)
            .map(|value| text(value).ok_or_else(|| self.invalid(name, "a string")))
            .transpose()
    }

    fn texts(&self, name: &str) -> Result<Vec<String>> {
        let Some(value) = self.value(name) else {
            return Ok(Vec::new());
        };
        value
            .as_array()
            .and_then(|items| items.iter().map(text).collect())
            .ok_or_else(|| self.invalid(name, "an array of strings"))
    }

    fn long(&self, name: &str) -> Result<i64> {
        self.required(name)?
            .as_i64()
            .ok_or_else(|| self.invalid(name, "an integer"))
    }

    fn optional_long(&self, name: &str) -> Result<Option<i64>> {
        self.value(name)
            .map(|value| {
                value
                    .as_i64()
                    .ok_or_else(|| self.invalid(name, "an integer"))
            })
            .transpose()
    }

    fn duration(&self, name: &str) -> Result<Duration> {
        millis(self.required(name)?).ok_or_else(|| self.invalid(name, "a number of milliseconds"))
    }

    fn rate(&self, name: &str) -> Result<Option<f64>> {
        self.value(name)
            .map(|value| value.as_f64().ok_or_else(|| self.invalid(name, "a number")))
            .transpose()
    }

    fn timestamp(&self, name: &str) -> Result<DateTime<Utc>> {
        timestamp(self.required(name)?).ok_or_else(|| self.invalid(name, "a timestamp"))
    }

    /// An offset of a source, which Spark sends as the JSON that the source writes it as, or
    /// as a string if that is not JSON.
    fn offset(&self, name: &str) -> Option<String> {
        self.value(name).map(|value| match value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        })
    }

    /// The object `name`, whose values `read` reads; an absent or `null` object is empty.
    fn map<T>(
        &self,
        name: &str,
        read: impl Fn(&Value) -> Option<T>,
        expected: &str,
    ) -> Result<BTreeMap<String, T>> {
        let Some(value) = self.value(name) else {
            return Ok(BTreeMap::new());
        };
        let Value::Object(members) = value else {
            return Err(self.invalid(name, "an object"));
        };
        members
            .iter()
            .map(|(key, value)| {
                read(value)
                    .map(|value| (key.clone(), value))
                    .ok_or_else(|| self.invalid(&format!("{name}.{key}"), expected))
            })
            .collect()
    }

    fn object(&self, name: &str) -> Result<Object<'a>> {
        Object::of(self.required(name)?, self.path(name), self.what)
    }

    fn objects(&self, name: &str) -> Result<Vec<Object<'a>>> {
        let Value::Array(items) = self.required(name)? else {
            return Err(self.invalid(name, "an array"));
        };
        items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                Object::of(item, format!("{}[{index}]", self.path(name)), self.what)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    /// A progress as a 4.3.0 server sends it for `recentProgress`: without the totals of the
    /// sources, and with offsets as strings.
    const RECENT_PROGRESS: &str = r#"{"id":"4a97984d-3435-4258-9d5e-2d4182e5dc87","runId":"f7595c35-d758-436d-82fc-8602386d678f","name":"probe_p3","timestamp":"2026-11-11T17:57:25.657Z","batchId":1,"batchDuration":229,"durationMs":{"triggerExecution":229,"queryPlanning":7,"getBatch":0,"commitOffsets":28,"latestOffset":0,"addBatch":170,"walCommit":23},"eventTime":{},"stateOperators":[],"sources":[{"description":"RateStreamV2[rowsPerSecond=5, rampUpTimeSeconds=0, numPartitions=default","startOffset":"0","endOffset":"1","latestOffset":"1","numInputRows":5,"inputRowsPerSecond":454.54545454545456,"processedRowsPerSecond":21.83406113537118,"metrics":{}}],"sink":{"description":"MemorySink","numOutputRows":5,"metrics":{}},"observedMetrics":{}}"#;

    /// The same progress as a 4.3.0 server sends it in an event: with the totals, offsets as
    /// numbers, and no empty maps.
    const PROGRESS_EVENT: &str = r#"{"progress":{"id":"4a97984d-3435-4258-9d5e-2d4182e5dc87","runId":"f7595c35-d758-436d-82fc-8602386d678f","name":"probe_p3","timestamp":"2026-11-11T17:57:25.657Z","batchId":1,"batchDuration":229,"numInputRows":5,"inputRowsPerSecond":454.5,"processedRowsPerSecond":21.8,"durationMs":{"addBatch":170,"commitOffsets":28,"getBatch":0,"latestOffset":0,"queryPlanning":7,"triggerExecution":229,"walCommit":23},"stateOperators":[],"sources":[{"description":"RateStreamV2[rowsPerSecond=5, rampUpTimeSeconds=0, numPartitions=default","startOffset":0,"endOffset":1,"latestOffset":1,"numInputRows":5,"inputRowsPerSecond":454.5,"processedRowsPerSecond":21.8}],"sink":{"description":"MemorySink","numOutputRows":5}}}"#;

    /// A progress with nulls, a stateful operator, two sources and an observation.
    const PROGRESS_WITH_NULLS: &str = r#"{"id":"a","runId":"b","name":null,"timestamp":"2026-01-02T03:04:05.006Z","batchId":7,"batchDuration":12,"durationMs":{"addBatch":3},"eventTime":{"max":"2026-01-02T03:00:00.000Z","watermark":"1970-01-01T00:00:00.000Z"},"stateOperators":[{"operatorName":"stateStoreSave","numRowsTotal":4,"numRowsUpdated":2,"allUpdatesTimeMs":5,"numRowsRemoved":1,"allRemovalsTimeMs":6,"commitTimeMs":7,"memoryUsedBytes":800,"numRowsDroppedByWatermark":0,"numShufflePartitions":200,"numStateStoreInstances":200,"customMetrics":{"loadedMapCacheHitCount":3}}],"sources":[{"description":"KafkaV2[Subscribe[events]]","startOffset":null,"endOffset":{"events":{"0":42}},"numInputRows":3,"processedRowsPerSecond":1.5,"metrics":{"avgOffsetsBehindLatest":"0.0"}},{"description":"second","startOffset":"text offset","endOffset":"2","latestOffset":null,"numInputRows":4,"inputRowsPerSecond":2.0,"processedRowsPerSecond":0.5}],"sink":{"description":"ForeachBatchSink","numOutputRows":-1},"observedMetrics":{"checks":{"rows":7,"max":null}}}"#;

    const STARTED: &str = r#"{"id":"4a97984d-3435-4258-9d5e-2d4182e5dc87","runId":"f7595c35-d758-436d-82fc-8602386d678f","name":"probe_p3","timestamp":"2026-11-11T17:57:24.633Z","jobTags":["SparkConnect_OperationTag_User_spark_Session_1"]}"#;

    const TERMINATED: &str = r#"{"id":"4a97984d-3435-4258-9d5e-2d4182e5dc87","runId":"f7595c35-d758-436d-82fc-8602386d678f","exception":null,"errorClassOnException":null}"#;

    fn at(text: &str) -> DateTime<Utc> {
        text.parse().unwrap()
    }

    fn millis(entries: &[(&str, u64)]) -> BTreeMap<String, Duration> {
        entries
            .iter()
            .map(|(key, ms)| (key.to_string(), Duration::from_millis(*ms)))
            .collect()
    }

    fn rate_source(start: &str, end: &str, input: f64, processed: f64) -> SourceProgress {
        SourceProgress {
            description: "RateStreamV2[rowsPerSecond=5, rampUpTimeSeconds=0, numPartitions=default"
                .to_string(),
            start_offset: Some(start.to_string()),
            end_offset: Some(end.to_string()),
            latest_offset: Some(end.to_string()),
            num_input_rows: 5,
            input_rows_per_second: Some(input),
            processed_rows_per_second: Some(processed),
            metrics: BTreeMap::new(),
        }
    }

    fn live_progress(input: f64, processed: f64, json: &str) -> QueryProgress {
        QueryProgress {
            id: "4a97984d-3435-4258-9d5e-2d4182e5dc87".to_string(),
            run_id: "f7595c35-d758-436d-82fc-8602386d678f".to_string(),
            name: Some("probe_p3".to_string()),
            timestamp: at("2026-11-11T17:57:25.657Z"),
            batch_id: 1,
            batch_duration: Duration::from_millis(229),
            num_input_rows: 5,
            input_rows_per_second: Some(input),
            processed_rows_per_second: Some(processed),
            durations: millis(&[
                ("addBatch", 170),
                ("commitOffsets", 28),
                ("getBatch", 0),
                ("latestOffset", 0),
                ("queryPlanning", 7),
                ("triggerExecution", 229),
                ("walCommit", 23),
            ]),
            event_time: BTreeMap::new(),
            state_operators: Vec::new(),
            sources: vec![rate_source("0", "1", input, processed)],
            sink: SinkProgress {
                description: "MemorySink".to_string(),
                num_output_rows: 5,
                metrics: BTreeMap::new(),
            },
            observed_metrics: BTreeMap::new(),
            json: json.to_string(),
        }
    }

    fn progress_with_nulls() -> QueryProgress {
        QueryProgress {
            id: "a".to_string(),
            run_id: "b".to_string(),
            name: None,
            timestamp: at("2026-01-02T03:04:05.006Z"),
            batch_id: 7,
            batch_duration: Duration::from_millis(12),
            num_input_rows: 7,
            input_rows_per_second: None,
            processed_rows_per_second: Some(2.0),
            durations: millis(&[("addBatch", 3)]),
            event_time: BTreeMap::from([
                ("max".to_string(), at("2026-01-02T03:00:00Z")),
                ("watermark".to_string(), at("1970-01-01T00:00:00Z")),
            ]),
            state_operators: vec![StateOperatorProgress {
                operator_name: "stateStoreSave".to_string(),
                num_rows_total: 4,
                num_rows_updated: 2,
                num_rows_removed: 1,
                all_updates_time: Duration::from_millis(5),
                all_removals_time: Duration::from_millis(6),
                commit_time: Duration::from_millis(7),
                memory_used_bytes: 800,
                num_rows_dropped_by_watermark: 0,
                num_shuffle_partitions: 200,
                num_state_store_instances: 200,
                custom_metrics: BTreeMap::from([("loadedMapCacheHitCount".to_string(), 3)]),
            }],
            sources: vec![
                SourceProgress {
                    description: "KafkaV2[Subscribe[events]]".to_string(),
                    start_offset: None,
                    end_offset: Some(r#"{"events":{"0":42}}"#.to_string()),
                    latest_offset: None,
                    num_input_rows: 3,
                    input_rows_per_second: None,
                    processed_rows_per_second: Some(1.5),
                    metrics: BTreeMap::from([(
                        "avgOffsetsBehindLatest".to_string(),
                        "0.0".to_string(),
                    )]),
                },
                SourceProgress {
                    description: "second".to_string(),
                    start_offset: Some("text offset".to_string()),
                    end_offset: Some("2".to_string()),
                    latest_offset: None,
                    num_input_rows: 4,
                    input_rows_per_second: Some(2.0),
                    processed_rows_per_second: Some(0.5),
                    metrics: BTreeMap::new(),
                },
            ],
            sink: SinkProgress {
                description: "ForeachBatchSink".to_string(),
                num_output_rows: -1,
                metrics: BTreeMap::new(),
            },
            observed_metrics: BTreeMap::from([(
                "checks".to_string(),
                r#"{"max":null,"rows":7}"#.to_string(),
            )]),
            json: PROGRESS_WITH_NULLS.to_string(),
        }
    }

    #[test]
    fn progress_and_events_are_read_with_their_nulls() {
        let event_json: Value = serde_json::from_str(PROGRESS_EVENT).unwrap();
        let mut from_event = live_progress(454.5, 21.8, "");
        let progress_event = event_of(1, PROGRESS_EVENT.to_string()).unwrap();
        if let QueryEvent::Progress(progress) = &progress_event {
            let member: Value = serde_json::from_str(&progress.json).unwrap();
            assert_eq!(
                member, event_json["progress"],
                "the JSON of the event's progress"
            );
            from_event.json = progress.json.clone();
        }
        let cases: Vec<(&str, Result<QueryEvent>, QueryEvent)> = vec![
            (
                "a progress of recentProgress, with the totals of its sources",
                progress_of(RECENT_PROGRESS.to_string()).map(|p| QueryEvent::Progress(Box::new(p))),
                QueryEvent::Progress(Box::new(live_progress(
                    454.54545454545456,
                    21.83406113537118,
                    RECENT_PROGRESS,
                ))),
            ),
            (
                "a progress event",
                Ok(progress_event.clone()),
                QueryEvent::Progress(Box::new(from_event)),
            ),
            (
                "a progress with nulls",
                progress_of(PROGRESS_WITH_NULLS.to_string())
                    .map(|p| QueryEvent::Progress(Box::new(p))),
                QueryEvent::Progress(Box::new(progress_with_nulls())),
            ),
            (
                "a started event",
                started_of(STARTED.to_string()),
                QueryEvent::Started(QueryStarted {
                    id: "4a97984d-3435-4258-9d5e-2d4182e5dc87".to_string(),
                    run_id: "f7595c35-d758-436d-82fc-8602386d678f".to_string(),
                    name: Some("probe_p3".to_string()),
                    timestamp: at("2026-11-11T17:57:24.633Z"),
                    job_tags: vec!["SparkConnect_OperationTag_User_spark_Session_1".to_string()],
                    json: STARTED.to_string(),
                }),
            ),
            (
                "a started event without a name or tags",
                started_of(
                    r#"{"id":"a","runId":"b","name":null,"timestamp":"2026-01-02T03:04:05.006Z"}"#
                        .to_string(),
                ),
                QueryEvent::Started(QueryStarted {
                    id: "a".to_string(),
                    run_id: "b".to_string(),
                    name: None,
                    timestamp: at("2026-01-02T03:04:05.006Z"),
                    job_tags: Vec::new(),
                    json: r#"{"id":"a","runId":"b","name":null,"timestamp":"2026-01-02T03:04:05.006Z"}"#
                        .to_string(),
                }),
            ),
            (
                "a terminated event",
                event_of(2, TERMINATED.to_string()),
                QueryEvent::Terminated(QueryTerminated {
                    id: "4a97984d-3435-4258-9d5e-2d4182e5dc87".to_string(),
                    run_id: "f7595c35-d758-436d-82fc-8602386d678f".to_string(),
                    exception: None,
                    error_class: None,
                    json: TERMINATED.to_string(),
                }),
            ),
            (
                "a terminated event of a failure",
                event_of(
                    2,
                    r#"{"id":"a","runId":"b","exception":"boom","errorClassOnException":"STREAM_FAILED"}"#
                        .to_string(),
                ),
                QueryEvent::Terminated(QueryTerminated {
                    id: "a".to_string(),
                    run_id: "b".to_string(),
                    exception: Some("boom".to_string()),
                    error_class: Some("STREAM_FAILED".to_string()),
                    json: r#"{"id":"a","runId":"b","exception":"boom","errorClassOnException":"STREAM_FAILED"}"#
                        .to_string(),
                }),
            ),
            (
                "an idle event",
                event_of(
                    3,
                    r#"{"id":"a","runId":"b","timestamp":"2026-01-02T03:04:05.006Z"}"#.to_string(),
                ),
                QueryEvent::Idle(QueryIdle {
                    id: "a".to_string(),
                    run_id: "b".to_string(),
                    timestamp: at("2026-01-02T03:04:05.006Z"),
                    json: r#"{"id":"a","runId":"b","timestamp":"2026-01-02T03:04:05.006Z"}"#
                        .to_string(),
                }),
            ),
        ];
        let mut failures = Vec::new();
        for (name, read, expected) in cases {
            match read {
                Ok(read) if read == expected => {}
                Ok(read) => failures.push(format!("{name}:\n  read {read:?}\n  want {expected:?}")),
                Err(error) => failures.push(format!("{name}: {error}")),
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// `PROGRESS_WITH_NULLS` with `edit` applied, as JSON text.
    fn edited(edit: impl FnOnce(&mut Value)) -> String {
        let mut progress: Value = serde_json::from_str(PROGRESS_WITH_NULLS).unwrap();
        edit(&mut progress);
        progress.to_string()
    }

    #[test]
    fn json_that_does_not_fit_its_type_is_a_decode_error() {
        let progress = |edit: fn(&mut Value)| progress_of(edited(edit)).map(drop);
        let cases: Vec<(&str, Result<()>, &str)> = vec![
            ("text", progress_of("{".to_string()).map(drop), "is not JSON"),
            ("an array", progress_of("[]".to_string()).map(drop), "is not an object"),
            (
                "no id",
                progress(|p| drop(p.as_object_mut().unwrap().remove("id"))),
                "has no `id`",
            ),
            ("a null run id", progress(|p| p["runId"] = Value::Null), "has no `runId`"),
            (
                "a batch id as text",
                progress(|p| p["batchId"] = "7".into()),
                "`batchId` in the progress of a streaming query is not an integer",
            ),
            (
                "a negative duration",
                progress(|p| p["batchDuration"] = (-1).into()),
                "`batchDuration` in the progress of a streaming query is not a number of milliseconds",
            ),
            (
                "a time that is not a timestamp",
                progress(|p| p["timestamp"] = "yesterday".into()),
                "`timestamp` in the progress of a streaming query is not a timestamp",
            ),
            (
                "sources that are not an array",
                progress(|p| p["sources"] = Value::Object(Map::new())),
                "`sources` in the progress of a streaming query is not an array",
            ),
            (
                "a source without a description",
                progress(|p| drop(p["sources"][0].as_object_mut().unwrap().remove("description"))),
                "has no `sources[0].description`",
            ),
            (
                "a step that is not a duration",
                progress(|p| p["durationMs"]["addBatch"] = "fast".into()),
                "`durationMs.addBatch` in the progress of a streaming query is not a number of milliseconds",
            ),
            (
                "an operator's metric that is not an integer",
                progress(|p| p["stateOperators"][0]["customMetrics"]["loadedMapCacheHitCount"] = 1.5.into()),
                "`stateOperators[0].customMetrics.loadedMapCacheHitCount` in the progress of a streaming query is not an integer",
            ),
            (
                "no sink",
                progress(|p| drop(p.as_object_mut().unwrap().remove("sink"))),
                "has no `sink`",
            ),
            (
                "an event of an unknown type",
                event_of(0, TERMINATED.to_string()).map(drop),
                "of the unknown type 0",
            ),
            (
                "a progress event without its progress",
                event_of(1, r#"{"id":"a"}"#.to_string()).map(drop),
                "has no `progress`",
            ),
            (
                "a progress event with a progress that does not fit",
                event_of(1, format!(r#"{{"progress":{}}}"#, edited(|p| p["batchId"] = Value::Null)))
                    .map(drop),
                "has no `progress.batchId`",
            ),
            (
                "a started event with tags that are not text",
                started_of(r#"{"id":"a","runId":"b","timestamp":"2026-01-02T03:04:05.006Z","jobTags":[1]}"#.to_string())
                    .map(drop),
                "`jobTags` in an event of a streaming query is not an array of strings",
            ),
        ];
        let mut failures = Vec::new();
        for (name, read, expected) in cases {
            match read {
                Err(error)
                    if error.kind() == ErrorKind::Decode && error.message().contains(expected) => {}
                Err(error) => failures.push(format!("{name}: {:?} {error}", error.kind())),
                Ok(()) => failures.push(format!("{name}: read")),
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
