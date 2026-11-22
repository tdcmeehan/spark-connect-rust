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

//! How the blocking API runs its calls: outside a runtime and inside runtimes of either flavor,
//! with its callbacks on the calling thread, its connection and its cleanup on the runtime of
//! the module; and what its iterators of events yield.

#![cfg(feature = "blocking")]

mod gated;
#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;

use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use futures::StreamExt;
use gated::{Received, RunResponse, Wait};
use prost::Message;
use spark_connect_async::actions::Checkpoint;
use spark_connect_async::blocking::{Estimator, Evaluator};
use spark_connect_async::ml::{self, LinearRegression, RegressionEvaluator};
use spark_connect_async::readwriter::SaveMode;
use spark_connect_async::streaming::QueryEvent;
use spark_connect_async::{blocking, ErrorKind, Session};
use spark_connect_proto as proto;
use spark_connect_proto::StreamingQueryEventType as EventType;
use tokio::runtime::{Builder, Runtime};

const SESSION_ID: &str = "00000000-0000-4000-8000-00000000b14a";
const WAIT: Duration = Duration::from_secs(10);

fn start() -> (Arc<server::Recorder>, String) {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let url = format!("sc://{address}/;session_id={SESSION_ID}");
    (recorder, url)
}

fn connect(url: &str) -> blocking::Session {
    Session::builder().remote(url).blocking().connect().unwrap()
}

/// Where a test makes a blocking call.
#[derive(Debug, Clone, Copy)]
enum Context {
    NoRuntime,
    CurrentThread,
    CurrentThreadTask,
    CurrentThreadSpawnBlocking,
    MultiThread,
    MultiThreadTask,
    MultiThreadSpawnBlocking,
}

const CONTEXTS: [Context; 7] = [
    Context::NoRuntime,
    Context::CurrentThread,
    Context::CurrentThreadTask,
    Context::CurrentThreadSpawnBlocking,
    Context::MultiThread,
    Context::MultiThreadTask,
    Context::MultiThreadSpawnBlocking,
];

// No I/O or time driver: the calls must use the runtime of the module.
fn current_thread() -> Runtime {
    Builder::new_current_thread().build().unwrap()
}

fn multi_thread() -> Runtime {
    Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap()
}

/// What `call` returns, called in `context`, whose runtime is dropped right after.
fn run_in<T: Send + 'static>(context: Context, call: impl FnOnce() -> T + Send + 'static) -> T {
    match context {
        Context::NoRuntime => call(),
        Context::CurrentThread => current_thread().block_on(async move { call() }),
        Context::CurrentThreadTask => current_thread()
            .block_on(async move { tokio::spawn(async move { call() }).await.unwrap() }),
        Context::CurrentThreadSpawnBlocking => current_thread()
            .block_on(async move { tokio::task::spawn_blocking(call).await.unwrap() }),
        Context::MultiThread => multi_thread().block_on(async move { call() }),
        Context::MultiThreadTask => multi_thread()
            .block_on(async move { tokio::spawn(async move { call() }).await.unwrap() }),
        Context::MultiThreadSpawnBlocking => {
            multi_thread().block_on(async move { tokio::task::spawn_blocking(call).await.unwrap() })
        }
    }
}

/// A session's calls of every kind, with what they returned.
fn calls(url: String) -> (i64, Vec<(i64,)>, usize, String, Option<String>) {
    let session = connect(&url);
    let three = session.as_async().sql(server::THREE_BATCHES).run().unwrap();
    let count = session.as_async().range(3).blocking().count().unwrap();
    let rows = three.collect_as::<(i64,)>().unwrap();
    let batches = three.stream().unwrap().map(Result::unwrap).count();
    let database = session.catalog().current_database().unwrap();
    three
        .as_async()
        .write()
        .mode(SaveMode::Overwrite)
        .blocking()
        .save("/out")
        .unwrap();
    let value = session.conf().get("k").unwrap();
    session.stop().unwrap();
    (count, rows, batches, database, value)
}

#[test]
fn blocking_calls_return_outside_a_runtime_and_inside_runtimes_of_either_flavor() {
    let (_, url) = start();
    for context in CONTEXTS {
        let url = url.clone();
        let returned = run_in(context, move || calls(url));
        assert_eq!(
            returned,
            (
                3,
                vec![(0,), (1,), (2,)],
                3,
                "spark_catalog".to_string(),
                None
            ),
            "{context:?}"
        );
    }
}

/// The count of a DataFrame by the old API and by the blocking API.
fn counts(url: String) -> (i64, i64) {
    let old = spark_connect::SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .unwrap();
    let old_count = old.range(3).unwrap().count().unwrap();
    let count = connect(&url)
        .as_async()
        .range(3)
        .blocking()
        .count()
        .unwrap();
    (old_count, count)
}

#[tokio::main(flavor = "current_thread")]
async fn counts_in_a_current_thread_main(url: String) -> (i64, i64) {
    counts(url)
}

#[tokio::main]
async fn counts_in_a_multi_thread_main(url: String) -> (i64, i64) {
    counts(url)
}

#[test]
fn the_old_api_and_the_blocking_api_return_inside_tokio_main_of_either_flavor() {
    let (_, url) = start();
    assert_eq!(counts_in_a_current_thread_main(url.clone()), (3, 3));
    assert_eq!(counts_in_a_multi_thread_main(url), (3, 3));
}

fn is_removal(entry: &server::Entry) -> bool {
    entry.kind == "ExecutePlanRequest"
        && format!(
            "{:?}",
            proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap()
        )
        .contains("RemoveCachedRemoteRelationCommand")
}

/// The `ReleaseAll` requests that the server has had.
fn releases(recorder: &server::Recorder) -> usize {
    recorder.wait_for_releases(Duration::ZERO, usize::MAX)
}

#[test]
fn a_result_dropped_anywhere_is_released_from_the_runtime_of_the_module() {
    let (recorder, url) = start();
    let session = connect(&url);
    let three = session.as_async().sql(server::THREE_BATCHES).run().unwrap();
    // That of the SQL command.
    assert_eq!(recorder.wait_for_releases(WAIT, 0), 1);
    for context in CONTEXTS {
        for created_in in [Context::NoRuntime, context] {
            let dropped_in = match created_in {
                Context::NoRuntime => context,
                _ => Context::NoRuntime,
            };
            let what = format!("created in {created_in:?}, dropped in {dropped_in:?}");

            let released = releases(&recorder);
            let df = three.clone();
            let mut batches = run_in(created_in, move || df.stream().unwrap());
            batches.next().unwrap().unwrap();
            assert_eq!(
                releases(&recorder),
                released,
                "batches {what}: before the drop"
            );
            run_in(dropped_in, move || drop(batches));
            assert_eq!(
                recorder.wait_for_releases(WAIT, released),
                released + 1,
                "batches {what}"
            );

            let released = releases(&recorder);
            let df = three.clone();
            let mut rows = run_in(created_in, move || df.stream_as::<i64>().unwrap());
            rows.next().unwrap().unwrap();
            assert_eq!(
                releases(&recorder),
                released,
                "rows {what}: before the drop"
            );
            run_in(dropped_in, move || drop(rows));
            assert_eq!(
                recorder.wait_for_releases(WAIT, released),
                released + 1,
                "rows {what}"
            );

            // The creation and the removal are commands, each released once it is done.
            let released = releases(&recorder);
            let df = three.clone();
            let checkpoint = run_in(created_in, move || {
                df.checkpoint(Checkpoint::default()).unwrap()
            });
            assert_eq!(recorder.wait_for_releases(WAIT, released), released + 1);
            recorder.take();
            run_in(dropped_in, move || drop(checkpoint));
            assert!(recorder.wait_for(WAIT, is_removal), "checkpoint {what}");
            assert_eq!(
                recorder.wait_for_releases(WAIT, released + 1),
                released + 2,
                "checkpoint {what}: the removal's release"
            );
        }
    }
}

#[test]
fn progress_handlers_and_the_hook_run_on_the_calling_thread() {
    let (_, url) = start();
    for context in CONTEXTS {
        let threads: Arc<Mutex<Vec<(&str, ThreadId)>>> = Arc::default();
        let (on_progress, on_hook) = (Arc::clone(&threads), Arc::clone(&threads));
        let builder = Session::builder()
            .remote(url.clone())
            .on_progress(move |_| {
                on_progress
                    .lock()
                    .unwrap()
                    .push(("progress", thread::current().id()));
            })
            .execute_plan_hook(move |request| {
                on_hook
                    .lock()
                    .unwrap()
                    .push(("hook", thread::current().id()));
                Ok(request)
            });
        let caller = run_in(context, move || {
            let session = builder.blocking().connect().unwrap();
            session
                .as_async()
                .sql(server::REPORTS_PROGRESS)
                .run()
                .unwrap();
            thread::current().id()
        });
        let threads = threads.lock().unwrap();
        assert_eq!(
            threads.iter().map(|(what, _)| *what).collect::<Vec<_>>(),
            ["hook", "progress", "progress"],
            "{context:?}"
        );
        assert!(
            threads.iter().all(|(_, thread)| *thread == caller),
            "{context:?}: {threads:?}, caller {caller:?}"
        );
    }
}

#[test]
fn a_session_connected_in_any_context_works_after_the_context_ends() {
    let (_, url) = start();
    for context in CONTEXTS {
        let url = url.clone();
        let session = run_in(context, move || connect(&url));
        let count = session.as_async().range(3).blocking().count();
        assert_eq!(count.unwrap(), 3, "{context:?}");
    }
}

/// The writer of a stream of the rate source to the memory sink, of `session`.
fn stream_writer(session: &blocking::Session, name: &str) -> blocking::DataStreamWriter {
    session
        .as_async()
        .read_stream()
        .format("rate")
        .load_without_path()
        .write_stream()
        .format("memory")
        .query_name(name)
        .blocking()
}

/// What the calls of streaming, Spark ML and pipelines returned: those of streaming on the
/// recording server at `recorded`, the others on the gated server at `gated`, which answers
/// them as Spark does.
#[allow(clippy::type_complexity)]
fn calls_of_the_areas(
    recorded: String,
    gated: String,
) -> (bool, bool, usize, String, f64, f64, String, usize) {
    let session = connect(&recorded);
    let query = stream_writer(&session, "q").start_without_path().unwrap();
    let active = query.is_active().unwrap();
    let terminated = query
        .await_termination(Some(Duration::from_secs(1)))
        .unwrap();
    query.process_all_available().unwrap();
    let queries = session.streams().active().unwrap().len();
    let plan = query.as_async().explain().run().unwrap();
    assert_eq!(session.streams().events().unwrap().count(), 0);
    session.stop().unwrap();

    let session = connect(&gated);
    let training = session.as_async().range(3);
    let model = LinearRegression::new().fit(&training).unwrap();
    let intercept = model.intercept().unwrap();
    let metric = RegressionEvaluator::new().evaluate(&training).unwrap();
    let graph = session.as_async().create_dataflow_graph().run().unwrap();
    let view = graph
        .as_async()
        .define_materialized_view("mv")
        .run()
        .unwrap()
        .name;
    let events = graph
        .as_async()
        .start_run("file:///storage")
        .run()
        .unwrap()
        .count();
    session.stop().unwrap();
    (
        active, terminated, queries, plan, intercept, metric, view, events,
    )
}

#[test]
fn the_calls_of_streaming_ml_and_pipelines_return_in_every_context() {
    let (_, recorded) = start();
    let gated = gated::Server::start();
    for context in CONTEXTS {
        let (recorded, gated) = (recorded.clone(), gated.url());
        let returned = run_in(context, move || calls_of_the_areas(recorded, gated));
        assert_eq!(
            returned,
            (
                true,
                true,
                1,
                "== Plan ==".to_string(),
                0.5,
                0.75,
                "mv".to_string(),
                2
            ),
            "{context:?}"
        );
    }
}

fn is_unsubscription(entry: &server::Entry) -> bool {
    use proto::streaming_query_listener_bus_command::Command as Bus;
    entry.kind == "ExecutePlanRequest"
        && format!(
            "{:?}",
            proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap()
        )
        .contains(&format!("{:?}", Bus::RemoveListenerBusListener(true)))
}

/// Whether the gated server receives a request that `found` accepts within [`WAIT`].
fn received_within(server: &gated::Server, found: impl Fn(&Received) -> bool) -> bool {
    let deadline = Instant::now() + WAIT;
    while Instant::now() < deadline {
        if server.received().iter().any(&found) {
            return true;
        }
        thread::sleep(Duration::from_millis(10));
    }
    false
}

fn deletes(server: &gated::Server) -> usize {
    let deletes = server.received().into_iter();
    deletes
        .filter(|received| matches!(received, Received::Delete(_)))
        .count()
}

#[test]
fn events_runs_models_and_graphs_dropped_anywhere_are_cleaned_up_from_the_runtime_of_the_module() {
    let (recorder, url) = start();
    recorder.hold_listener_bus(true);
    let session = connect(&url);
    let server = gated::Server::start();
    let on_gated = connect(&server.url());
    for context in CONTEXTS {
        for created_in in [Context::NoRuntime, context] {
            let dropped_in = match created_in {
                Context::NoRuntime => context,
                _ => Context::NoRuntime,
            };
            let what = format!("created in {created_in:?}, dropped in {dropped_in:?}");

            recorder.take();
            let streams = session.streams();
            let events = run_in(created_in, move || streams.events().unwrap());
            assert!(
                !recorder.wait_for(Duration::ZERO, is_unsubscription),
                "events {what}: before the drop"
            );
            run_in(dropped_in, move || drop(events));
            assert!(recorder.wait_for(WAIT, is_unsubscription), "events {what}");

            let deleted = deletes(&server);
            let training = on_gated.as_async().range(3);
            let model = run_in(created_in, move || {
                LinearRegression::new().fit(&training).unwrap()
            });
            assert_eq!(deletes(&server), deleted, "model {what}: before the drop");
            run_in(dropped_in, move || drop(model));
            assert!(
                received_within(&server, |_| deletes(&server) == deleted + 1),
                "model {what}"
            );

            let creator = on_gated.clone();
            let graph = run_in(created_in, move || {
                creator.as_async().create_dataflow_graph().run().unwrap()
            });
            let id = graph.as_async().id().to_string();
            let dropped_graph = |received: &Received| *received == Received::DropGraph(id.clone());
            assert!(
                !server.received().iter().any(dropped_graph),
                "graph {what}: before the drop"
            );
            run_in(dropped_in, move || drop(graph));
            assert!(received_within(&server, dropped_graph), "graph {what}");

            // A run read up to its first event. The server records no drop of a graph while
            // it waits after first batches, so this graph outlives the run.
            let graph = on_gated.as_async().create_dataflow_graph().run().unwrap();
            let kept = graph.clone();
            server.wait_at(Wait::AfterTheFirstBatch);
            let run = run_in(created_in, move || {
                let mut run = graph.as_async().start_run("file:///s").run().unwrap();
                run.next().unwrap().unwrap();
                run
            });
            let operation = run.as_async().operation_id().to_string();
            let released =
                |received: &Received| *received == Received::ReleaseAll(operation.clone());
            assert!(
                !server.received().iter().any(released),
                "run {what}: before the drop"
            );
            run_in(dropped_in, move || drop(run));
            assert!(received_within(&server, released), "run {what}");
            server.wait_at(Wait::Never);
            server.open_gate();
            drop(kept);
        }
    }
}

#[test]
fn a_model_and_a_graph_last_until_their_last_handle_of_either_api_drops() {
    let server = gated::Server::start();
    let session = connect(&server.url());
    let training = session.as_async().range(3);
    let model = LinearRegression::new().fit(&training).unwrap();
    let borrowed = model.as_async().clone();
    let converted = model.clone().into_async();
    let predictions = ml::Transformer::transform(model.as_async(), &training);
    for (dropped, handle) in [
        ("the blocking model", Box::new(model) as Box<dyn Send>),
        ("a clone of its async model", Box::new(borrowed)),
        ("its async model", Box::new(converted)),
    ] {
        drop(handle);
        thread::sleep(Duration::from_millis(100));
        assert_eq!(deletes(&server), 0, "{dropped}, while its transform lives");
    }
    drop(predictions);
    assert!(
        received_within(&server, |_| deletes(&server) == 1),
        "the transform dropped"
    );

    let graph = session.as_async().create_dataflow_graph().run().unwrap();
    let id = graph.as_async().id().to_string();
    let dropped_graph = |received: &Received| *received == Received::DropGraph(id.clone());
    let borrowed = graph.as_async().clone();
    let converted = graph.clone().into_async();
    drop(graph);
    drop(borrowed);
    thread::sleep(Duration::from_millis(100));
    assert!(
        !server.received().iter().any(dropped_graph),
        "the async graph lives"
    );
    drop(converted);
    assert!(
        received_within(&server, dropped_graph),
        "the last handle dropped"
    );
}

const PROGRESS_EVENT: &str = r#"{"progress":{"id":"00000000-0000-4000-8000-000000000001","runId":"00000000-0000-4000-8000-000000000002","name":"numbers","timestamp":"2026-11-11T17:57:25.657Z","batchId":1,"batchDuration":229,"numInputRows":5,"stateOperators":[],"sources":[{"description":"rate","startOffset":0,"endOffset":1,"latestOffset":1,"numInputRows":5}],"sink":{"description":"MemorySink","numOutputRows":5}}}"#;

const TERMINATED: &str = r#"{"id":"00000000-0000-4000-8000-000000000001","runId":"00000000-0000-4000-8000-000000000002","exception":null,"errorClassOnException":null}"#;

/// The next item of `items`, read on a thread of its own within [`WAIT`], so that a read that
/// doesn't end fails the test.
fn next_within<I>(items: &mut Option<I>) -> Option<I::Item>
where
    I: Iterator + Send + 'static,
    I::Item: Send + 'static,
{
    let mut taken = items.take().expect("the iterator");
    let (sender, received) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let item = taken.next();
        let _ = sender.send((taken, item));
    });
    let (taken, item) = received
        .recv_timeout(WAIT)
        .expect("no item within the deadline");
    *items = Some(taken);
    item
}

/// An event of the iterator, as the kind of the event or of the error.
fn kind_of(event: Option<spark_connect_async::Result<QueryEvent>>) -> String {
    match event {
        Some(Ok(QueryEvent::Started(started))) => format!("started {}", started.id),
        Some(Ok(QueryEvent::Progress(progress))) => format!("progress {}", progress.batch_id),
        Some(Ok(QueryEvent::Terminated(terminated))) => format!("terminated {}", terminated.id),
        Some(Ok(other)) => format!("{other:?}"),
        Some(Err(error)) => format!("error {:?}", error.kind()),
        None => "end".to_string(),
    }
}

#[test]
fn the_events_iterator_yields_what_the_server_sends_and_ends_as_the_stream_ends() {
    let server = gated::Server::start();
    let session = connect(&server.url());
    let streams = session.streams();
    let mut events = Some(streams.events().unwrap());
    let second = streams.events().map(drop).unwrap_err();
    assert_eq!(second.kind(), ErrorKind::Unsupported, "{second}");
    let query = stream_writer(&session, "numbers")
        .start_without_path()
        .unwrap();
    let mut yielded = vec![kind_of(next_within(&mut events))];
    server.send_event(EventType::QueryProgressEvent, PROGRESS_EVENT);
    yielded.push(kind_of(next_within(&mut events)));
    server.send_event(EventType::QueryProgressUnspecified, TERMINATED);
    yielded.push(kind_of(next_within(&mut events)));
    server.send_event(EventType::QueryTerminatedEvent, TERMINATED);
    yielded.push(kind_of(next_within(&mut events)));
    server.end_subscription();
    yielded.push(kind_of(next_within(&mut events)));
    yielded.push(kind_of(next_within(&mut events)));
    let id = query.as_async().id();
    assert_eq!(
        yielded,
        [
            format!("started {id}"),
            "progress 1".to_string(),
            "error Decode".to_string(),
            format!("terminated {id}"),
            "end".to_string(),
            "end".to_string(),
        ]
    );
    drop(events);
    let unsubscribed = |received: &Received| matches!(received, Received::Unsubscribe(_));
    assert!(
        !server.received().iter().any(unsubscribed),
        "removed a subscription that the server ended"
    );

    // A failed subscription: its error, then the end, and the client removes it.
    let mut events = Some(streams.events().unwrap());
    server.fail_subscription();
    assert_eq!(kind_of(next_within(&mut events)), "error Internal");
    assert_eq!(kind_of(next_within(&mut events)), "end");
    assert!(
        received_within(&server, unsubscribed),
        "a failed subscription"
    );
    drop(events);

    // An async stream of events, converted.
    let runtime = Runtime::new().unwrap();
    let (progress, events) = runtime.block_on(async {
        let mut events = streams.as_async().events().await.unwrap();
        server.send_event(EventType::QueryProgressEvent, PROGRESS_EVENT);
        (kind_of(events.next().await), events)
    });
    assert_eq!(progress, "progress 1");
    let mut events = Some(events.blocking());
    server.send_event(EventType::QueryProgressUnspecified, TERMINATED);
    assert_eq!(
        kind_of(next_within(&mut events)),
        "error Decode",
        "converted from async"
    );
}

/// An item of a run, as its message or the kind of its error.
fn event_of_run(
    event: spark_connect_async::Result<spark_connect_async::pipelines::PipelineEvent>,
) -> String {
    match event {
        Ok(event) => event.message,
        Err(error) => format!("error {:?}", error.kind()),
    }
}

#[test]
fn a_run_iterator_yields_what_the_async_run_yields_then_ends() {
    let server = gated::Server::start();
    let session = connect(&server.url());
    let graph = session.as_async().create_dataflow_graph().run().unwrap();
    let runtime = Runtime::new().unwrap();
    let failing = vec![
        RunResponse::Event(None, Some("no timestamp")),
        RunResponse::Event(Some((1_794_431_318, 0)), Some("Flow f is RUNNING.")),
        RunResponse::Failure,
    ];
    for responses in [gated::default_run(), failing] {
        server.run_with(responses);
        let mut run = Some(graph.as_async().start_run("file:///s").run().unwrap());
        let mut blocking = Vec::new();
        while let Some(event) = next_within(&mut run) {
            blocking.push(event_of_run(event));
        }
        assert!(
            next_within(&mut run).is_none(),
            "an ended run yields nothing more"
        );
        let awaited: Vec<String> = runtime.block_on(async {
            let run = graph.as_async().start_run("file:///s").await.unwrap();
            run.map(event_of_run).collect().await
        });
        assert_eq!(blocking, awaited);
    }
    let mut run = Some(graph.as_async().start_run("file:///s").run().unwrap());
    let first = next_within(&mut run).map(|event| event.map(|_| ()));
    let rest = run.unwrap().into_async().map(event_of_run).collect();
    let rest: Vec<String> = runtime.block_on(rest);
    assert!(matches!(first, Some(Err(_))), "{first:?}");
    assert_eq!(rest, ["Flow f is RUNNING.", "error Internal"]);
}
