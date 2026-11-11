// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

//! The requests of the structured streaming of the async API, recorded by the old API's
//! recording server and compared with the old API's.

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;
mod support;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use futures::StreamExt;
use prost::Message;
use spark_connect::streaming::{
    StreamingQueryListener, StreamingQueryListenerEvent, Trigger as OldTrigger,
};
use spark_connect::SparkSession;
use spark_connect_async::streaming::{OutputMode, StreamingQuery, Trigger};
use spark_connect_async::types::{DataType, StructField};
use spark_connect_async::Session;
use spark_connect_proto as proto;

const SESSION_ID: &str = "00000000-0000-4000-8000-0000000000ba";

struct Case {
    name: &'static str,
    /// Whether the case subscribes to the listener bus and removes the subscription, which the
    /// recording server then keeps open until the removal.
    subscribes: bool,
    old: fn(&SparkSession) -> Option<proto::Relation>,
    new: fn(Session) -> BoxFuture<'static, Option<proto::Relation>>,
}

/// A case whose operations return DataFrames, compared by their relations.
macro_rules! read {
    ($name:expr, |$o:ident| $old:expr, |$n:ident| $new:expr) => {
        Case {
            name: $name,
            subscribes: false,
            old: |$o| Some($old.to_relation()),
            new: |$n| Box::pin(async move { Some($new.to_relation()) }),
        }
    };
}

/// A case of operations compared by their requests. Their results are not: the recording
/// server's answers are not those of a query.
macro_rules! run {
    ($name:expr, |$o:ident| $old:expr, |$n:ident| $new:expr) => {
        run!($name, false, |$o| $old, |$n| $new)
    };
    ($name:expr, $subscribes:expr, |$o:ident| $old:expr, |$n:ident| $new:expr) => {
        Case {
            name: $name,
            subscribes: $subscribes,
            old: |$o| {
                let _ = $old;
                None
            },
            new: |$n| {
                Box::pin(async move {
                    let _ = $new.await;
                    None
                })
            },
        }
    };
}

struct Ignored;

impl StreamingQueryListener for Ignored {
    fn on_event(&self, _: &StreamingQueryListenerEvent) {}
}

fn options() -> HashMap<String, String> {
    HashMap::from([
        ("a".to_string(), "1".to_string()),
        ("b".to_string(), "2".to_string()),
    ])
}

fn schema() -> DataType {
    DataType::Struct {
        fields: vec![StructField {
            name: "id".to_string(),
            data_type: DataType::Integer,
            nullable: true,
            metadata: BTreeMap::new(),
        }],
    }
}

fn old_range(s: &SparkSession) -> spark_connect::DataFrame {
    s.range(3).unwrap()
}

fn old_query(s: &SparkSession) -> spark_connect::streaming::StreamingQuery {
    old_range(s)
        .write_stream()
        .format("memory")
        .start("")
        .unwrap()
}

async fn new_query(s: &Session) -> StreamingQuery {
    s.range(3)
        .write_stream()
        .format("memory")
        .start_without_path()
        .await
        .unwrap()
}

fn cases() -> Vec<Case> {
    vec![
        read!(
            "DataStreamReader::load",
            |o| o
                .read_stream()
                .format("rate")
                .schema("a INT")
                .option("k", "v")
                .options(options())
                .name("src")
                .load(Some("/in")),
            |s| s
                .read_stream()
                .format("rate")
                .schema("a INT")
                .option("k", "v")
                .options(options())
                .name("src")
                .load("/in")
        ),
        read!(
            "DataStreamReader::load without a path",
            |o| o.read_stream().format("rate").load(None),
            |s| s.read_stream().format("rate").load_without_path()
        ),
        read!(
            "DataStreamReader::schema of a type",
            |o| o.read_stream().schema(schema().json()).load(None),
            |s| s.read_stream().schema(schema()).load_without_path()
        ),
        read!(
            "DataStreamReader::table",
            |o| o.read_stream().option("k", "v").table("db.t"),
            |s| s.read_stream().option("k", "v").table("db.t")
        ),
        read!(
            "DataStreamReader::changes",
            |o| o
                .read_stream()
                .option("startingVersion", "1")
                .changes("db.t"),
            |s| s.read_stream().option("startingVersion", 1).changes("db.t")
        ),
        read!(
            "DataStreamReader::json",
            |o| o.read_stream().json("/in"),
            |s| s.read_stream().json("/in")
        ),
        read!(
            "DataStreamReader::parquet",
            |o| o.read_stream().parquet("/in"),
            |s| s.read_stream().parquet("/in")
        ),
        read!(
            "DataStreamReader::csv",
            |o| o.read_stream().csv("/in"),
            |s| s.read_stream().csv("/in")
        ),
        read!(
            "DataStreamReader::orc",
            |o| o.read_stream().orc("/in"),
            |s| s.read_stream().orc("/in")
        ),
        read!(
            "DataStreamReader::text",
            |o| o.read_stream().text("/in"),
            |s| s.read_stream().text("/in")
        ),
        run!(
            "DataStreamWriter::start",
            |o| old_range(o)
                .write_stream()
                .format("parquet")
                .output_mode("append")
                .option("checkpointLocation", "/cp")
                .options(options())
                .partition_by(vec!["a"])
                .cluster_by(vec!["b"])
                .query_name("q")
                .start("/out"),
            |s| s
                .range(3)
                .write_stream()
                .format("parquet")
                .output_mode(OutputMode::Append)
                .option("checkpointLocation", "/cp")
                .options(options())
                .partition_by(["a"])
                .cluster_by(["b"])
                .query_name("q")
                .start("/out")
        ),
        run!(
            "DataStreamWriter::start without a path",
            |o| old_range(o).write_stream().format("memory").start(""),
            |s| s
                .range(3)
                .write_stream()
                .format("memory")
                .start_without_path()
        ),
        run!(
            "DataStreamWriter::to_table",
            |o| old_range(o).write_stream().to_table("db.t"),
            |s| s.range(3).write_stream().to_table("db.t")
        ),
        run!(
            "DataStreamWriter::output_mode complete",
            |o| old_range(o)
                .write_stream()
                .output_mode("complete")
                .start("/out"),
            |s| s
                .range(3)
                .write_stream()
                .output_mode(OutputMode::Complete)
                .start("/out")
        ),
        run!(
            "DataStreamWriter::output_mode update",
            |o| old_range(o)
                .write_stream()
                .output_mode("update")
                .start("/out"),
            |s| s
                .range(3)
                .write_stream()
                .output_mode(OutputMode::Update)
                .start("/out")
        ),
        run!(
            "DataStreamWriter::option of a number",
            |o| old_range(o)
                .write_stream()
                .option("maxFilesPerTrigger", "5")
                .start("/out"),
            |s| s
                .range(3)
                .write_stream()
                .option("maxFilesPerTrigger", 5)
                .start("/out")
        ),
        run!(
            "DataStreamWriter::trigger ProcessingTime",
            |o| old_range(o)
                .write_stream()
                .trigger(OldTrigger::ProcessingTime("5 seconds".to_string()))
                .start("/out"),
            |s| s
                .range(3)
                .write_stream()
                .trigger(Trigger::ProcessingTime("5 seconds".to_string()))
                .start("/out")
        ),
        run!(
            "DataStreamWriter::trigger Once",
            |o| old_range(o)
                .write_stream()
                .trigger(OldTrigger::Once)
                .start("/out"),
            |s| s
                .range(3)
                .write_stream()
                .trigger(Trigger::Once)
                .start("/out")
        ),
        run!(
            "DataStreamWriter::trigger AvailableNow",
            |o| old_range(o)
                .write_stream()
                .trigger(OldTrigger::AvailableNow)
                .start("/out"),
            |s| s
                .range(3)
                .write_stream()
                .trigger(Trigger::AvailableNow)
                .start("/out")
        ),
        run!(
            "DataStreamWriter::trigger Continuous",
            |o| old_range(o)
                .write_stream()
                .trigger(OldTrigger::Continuous("1 second".to_string()))
                .start("/out"),
            |s| s
                .range(3)
                .write_stream()
                .trigger(Trigger::Continuous("1 second".to_string()))
                .start("/out")
        ),
        run!(
            "DataStreamWriter::trigger RealTime",
            |o| old_range(o)
                .write_stream()
                .trigger(OldTrigger::RealTime("10 seconds".to_string()))
                .start("/out"),
            |s| s
                .range(3)
                .write_stream()
                .trigger(Trigger::RealTime("10 seconds".to_string()))
                .start("/out")
        ),
        run!(
            "StreamingQuery::is_active",
            |o| old_query(o).is_active(),
            |s| async move { new_query(&s).await.is_active().await }
        ),
        run!(
            "StreamingQuery::status",
            |o| old_query(o).status(),
            |s| async move { new_query(&s).await.status().await }
        ),
        run!(
            "StreamingQuery::stop",
            |o| old_query(o).stop(),
            |s| async move { new_query(&s).await.stop().await }
        ),
        run!(
            "StreamingQuery::await_termination",
            |o| old_query(o).await_termination(None),
            |s| async move { new_query(&s).await.await_termination(None).await }
        ),
        run!(
            "StreamingQuery::await_termination with a timeout",
            |o| old_query(o).await_termination(Some(1.5)),
            |s| async move {
                new_query(&s)
                    .await
                    .await_termination(Some(Duration::from_millis(1500)))
                    .await
            }
        ),
        run!(
            "StreamingQuery::last_progress",
            |o| old_query(o).last_progress(),
            |s| async move { new_query(&s).await.last_progress().await }
        ),
        run!(
            "StreamingQuery::recent_progress",
            |o| old_query(o).recent_progress(),
            |s| async move { new_query(&s).await.recent_progress().await }
        ),
        run!(
            "StreamingQuery::process_all_available",
            |o| old_query(o).process_all_available(),
            |s| async move { new_query(&s).await.process_all_available().await }
        ),
        run!(
            "StreamingQuery::explain",
            |o| old_query(o).explain(false),
            |s| async move { new_query(&s).await.explain().await }
        ),
        run!(
            "StreamingQuery::explain extended",
            |o| old_query(o).explain(true),
            |s| async move { new_query(&s).await.explain().extended().await }
        ),
        run!(
            "StreamingQuery::exception",
            |o| old_query(o).exception(),
            |s| async move { new_query(&s).await.exception().await }
        ),
        run!(
            "StreamingQueryManager::active",
            |o| o.streams().active(),
            |s| s.streams().active()
        ),
        run!(
            "StreamingQueryManager::get",
            |o| o.streams().get("query-id"),
            |s| s.streams().get("query-id")
        ),
        run!(
            "StreamingQueryManager::await_any_termination",
            |o| o.streams().await_any_termination(None),
            |s| s.streams().await_any_termination(None)
        ),
        run!(
            "StreamingQueryManager::await_any_termination with a timeout",
            |o| o.streams().await_any_termination(Some(2.0)),
            |s| s
                .streams()
                .await_any_termination(Some(Duration::from_secs(2)))
        ),
        run!(
            "StreamingQueryManager::reset_terminated",
            |o| o.streams().reset_terminated(),
            |s| s.streams().reset_terminated()
        ),
        run!(
            "StreamingQueryManager::add_listener and remove_listener, as events dropped",
            true,
            |o| {
                let streams = o.streams();
                let id = streams.add_listener(Arc::new(Ignored)).unwrap();
                streams.remove_listener(&id)
            },
            |s| async move { drop(s.streams().events().await) }
        ),
        run!(
            "StreamingQueryManager::listener_event_stream, as events read to their end",
            |o| o
                .streams()
                .listener_event_stream()
                .map(|events| events.count()),
            |s| async move {
                if let Ok(mut events) = s.streams().events().await {
                    while events.next().await.is_some() {}
                }
            }
        ),
    ]
}

fn is_removal(entry: &server::Entry) -> bool {
    use proto::streaming_query_listener_bus_command::Command as Bus;
    entry.kind == "ExecutePlanRequest"
        && matches!(
            proto::ExecutePlanRequest::decode(&entry.bytes[..])
                .ok()
                .and_then(|request| request.plan?.op_type),
            Some(proto::plan::OpType::Command(proto::Command {
                command_type: Some(
                    proto::command::CommandType::StreamingQueryListenerBusCommand(
                        proto::StreamingQueryListenerBusCommand {
                            command: Some(Bus::RemoveListenerBusListener(_)),
                        }
                    )
                ),
            }))
        )
}

/// Each entry as its message, without the operation id, which the client chooses at random.
fn messages(entries: Vec<server::Entry>) -> Vec<String> {
    entries
        .into_iter()
        .map(|entry| match entry.kind.as_str() {
            "ExecutePlanRequest" => {
                let mut request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
                request.operation_id = None;
                format!("{request:?}")
            }
            "AnalyzePlanRequest" => {
                format!(
                    "{:?}",
                    proto::AnalyzePlanRequest::decode(&entry.bytes[..]).unwrap()
                )
            }
            _ => format!("{entry:?}"),
        })
        .collect()
}

fn start() -> (Arc<server::Recorder>, String) {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let url = format!("sc://{address}/;session_id={SESSION_ID};user_id=snapshots");
    (recorder, url)
}

/// Runs the new operations of `case` with `session`, and waits for the removal of a
/// subscription that they drop, which is sent in the background.
async fn run_new(
    case: &Case,
    session: Session,
    recorder: &Arc<server::Recorder>,
) -> Option<proto::Relation> {
    let relation = (case.new)(session).await;
    if case.subscribes {
        let recorder = Arc::clone(recorder);
        let removed = tokio::task::spawn_blocking(move || {
            recorder.wait_for(Duration::from_secs(5), is_removal)
        });
        assert!(removed.await.unwrap(), "{}: no removal", case.name);
    }
    relation
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structured_streaming_sends_the_requests_of_the_old_api() {
    let (recorder, url) = start();
    let mut both = Vec::new();
    for case in cases() {
        recorder.hold_listener_bus(case.subscribes);
        let old_session = SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        let old_relation = (case.old)(&old_session);
        let old = messages(recorder.take());
        let session = Session::builder()
            .remote(url.clone())
            .connect()
            .await
            .unwrap();
        let new_relation = run_new(&case, session, &recorder).await;
        let new = messages(recorder.take());
        recorder.hold_listener_bus(false);
        assert!(
            !old.is_empty() || old_relation.is_some(),
            "{}: the old API sent and built nothing",
            case.name
        );
        both.push((case.name, (old, old_relation), (new, new_relation)));
    }
    let differences = support::differences(both);
    assert!(differences.is_empty(), "{differences}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_operation_sends_the_tags_of_its_handle() {
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let nightly = session.with_tag("nightly").unwrap();
    let mut sent = 0;
    let mut failures = Vec::new();
    for case in cases() {
        recorder.hold_listener_bus(case.subscribes);
        run_new(&case, nightly.clone(), &recorder).await;
        recorder.hold_listener_bus(false);
        for entry in recorder.take() {
            if entry.kind != "ExecutePlanRequest" {
                continue;
            }
            sent += 1;
            let request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
            if request.tags != ["nightly"] {
                failures.push(format!("{}: tags {:?}", case.name, request.tags));
            }
        }
    }
    assert!(sent > 40, "only {sent} requests");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
