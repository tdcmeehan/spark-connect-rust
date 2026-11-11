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

//! The graphs of pipelines, their definitions and their runs, against a server that answers as
//! Spark 4.3.0 does.

mod gated;

use std::future::IntoFuture;
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use futures::StreamExt;
use gated::{Received, RunResponse, Server, Wait};
use spark_connect_async::ml::{Estimator, StandardScaler, Transformer};
use spark_connect_async::pipelines::{DataflowGraph, Identifier, PipelineEvent};
use spark_connect_async::{ErrorKind, Session};

async fn connect(server: &Server) -> Session {
    Session::builder()
        .remote(server.url())
        .connect()
        .await
        .unwrap()
}

/// A runtime of `flavor`, a current-thread or a multi-thread one.
fn runtime(flavor: &str) -> tokio::runtime::Runtime {
    match flavor {
        "current-thread" => tokio::runtime::Builder::new_current_thread(),
        _ => tokio::runtime::Builder::new_multi_thread(),
    }
    .enable_all()
    .build()
    .unwrap()
}

fn drops(server: &Server) -> Vec<String> {
    server
        .received()
        .into_iter()
        .filter_map(|received| match received {
            Received::DropGraph(id) => Some(id),
            _ => None,
        })
        .collect()
}

fn executions(server: &Server) -> usize {
    server
        .received()
        .iter()
        .filter(|received| matches!(received, Received::Execute(_)))
        .count()
}

/// The parts of `identifier`.
fn parts(identifier: &Identifier) -> (Option<&str>, Vec<&str>, &str) {
    (
        identifier.catalog.as_deref(),
        identifier.namespace.iter().map(String::as_str).collect(),
        &identifier.name,
    )
}

/// The timestamps, in seconds and nanoseconds, and the messages of `events`.
fn texts(events: &[PipelineEvent]) -> Vec<(i64, u32, &str)> {
    events
        .iter()
        .map(|event| {
            (
                event.timestamp.timestamp(),
                event.timestamp.timestamp_subsec_nanos(),
                event.message.as_str(),
            )
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_definition_returns_the_identifier_that_the_server_resolves_its_name_to() {
    let server = Server::start();
    let session = connect(&server).await;
    let graph = session.create_dataflow_graph().await.unwrap();
    assert_eq!(graph.id(), "graph-1");
    let definitions: Vec<(
        &str,
        BoxFuture<'static, spark_connect_async::Result<Identifier>>,
    )> = vec![
        (
            "materialized view",
            Box::pin(graph.define_materialized_view("mv").into_future()),
        ),
        (
            "streaming table",
            Box::pin(graph.define_streaming_table("st").into_future()),
        ),
        (
            "temporary view",
            Box::pin(graph.define_temporary_view("v").into_future()),
        ),
        (
            "sink",
            Box::pin(graph.define_sink("s", "kafka").into_future()),
        ),
        (
            "flow",
            Box::pin(graph.define_flow("f", "mv", session.range(3)).into_future()),
        ),
        (
            "Auto CDC flow",
            Box::pin(
                graph
                    .define_auto_cdc_flow("c", "st", "changes", ["k"], "ts")
                    .into_future(),
            ),
        ),
    ];
    let mut read = Vec::new();
    for (name, definition) in definitions {
        let identifier = definition.await.unwrap();
        read.push((name, format!("{:?}", parts(&identifier))));
    }
    let expected = [
        (
            "materialized view",
            r#"(Some("spark_catalog"), ["default"], "mv")"#,
        ),
        (
            "streaming table",
            r#"(Some("spark_catalog"), ["default"], "st")"#,
        ),
        ("temporary view", r#"(None, [], "v")"#),
        ("sink", r#"(None, [], "s")"#),
        ("flow", r#"(Some("spark_catalog"), ["default"], "f")"#),
        (
            "Auto CDC flow",
            r#"(Some("spark_catalog"), ["default"], "c")"#,
        ),
    ];
    let expected: Vec<_> = expected
        .iter()
        .map(|(name, text)| (*name, text.to_string()))
        .collect();
    assert_eq!(read, expected);
    graph
        .define_sql("CREATE MATERIALIZED VIEW m AS SELECT 1")
        .await
        .unwrap();
    let other = session.create_dataflow_graph().await.unwrap();
    assert_eq!(other.id(), "graph-2");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_graph_whose_creation_has_no_id_in_its_answer_is_a_decode_error() {
    let server = Server::start();
    server.answer_pipelines_with_bare_results();
    let session = connect(&server).await;
    let error = session.create_dataflow_graph().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Decode);
    assert_eq!(error.message(), "the server returned no dataflow graph");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_events_of_a_run_come_until_it_completes_and_it_reports_its_progress() {
    let server = Server::start();
    let updates = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&updates);
    let session = Session::builder()
        .remote(server.url())
        .on_progress(move |update| {
            seen.lock()
                .unwrap()
                .push((update.operation_id.clone(), update.done));
        })
        .connect()
        .await
        .unwrap();
    let graph = session.create_dataflow_graph().await.unwrap();
    let run = graph.start_run("file:///storage").await.unwrap();
    let operation_id = run.operation_id().to_string();
    assert_eq!(operation_id, server.operation_id());
    let events: Vec<_> = run.collect().await;
    let events: Vec<PipelineEvent> = events.into_iter().map(Result::unwrap).collect();
    assert_eq!(
        texts(&events),
        [
            (1_794_431_318, 963_703_034, "Flow f is RUNNING."),
            (1_794_431_321, 984_787_258, "Run is COMPLETED."),
        ]
    );
    let updates = updates.lock().unwrap().clone();
    let of_the_run: Vec<_> = updates
        .iter()
        .filter(|(id, _)| *id == operation_id)
        .map(|(_, done)| *done)
        .collect();
    assert_eq!(of_the_run.first(), Some(&false), "{updates:?}");
    assert_eq!(of_the_run.last(), Some(&true), "{updates:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_run_is_an_error_after_its_events_then_the_end() {
    let server = Server::start();
    server.run_with(vec![
        RunResponse::Event(Some((10, 0)), Some("Flow f has FAILED.")),
        RunResponse::Failure,
    ]);
    let session = connect(&server).await;
    let graph = session.create_dataflow_graph().await.unwrap();
    let mut run = graph.start_run("file:///storage").await.unwrap();
    let event = run.next().await.unwrap().unwrap();
    assert_eq!(texts(&[event]), [(10, 0, "Flow f has FAILED.")]);
    let error = run.next().await.unwrap().unwrap_err();
    assert_ne!(error.kind(), ErrorKind::Decode);
    assert!(error.message().contains("the run failed"), "{error}");
    assert!(run.next().await.is_none());
    assert!(run.next().await.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_event_that_cannot_be_read_is_an_error_and_the_events_after_it_still_come() {
    let server = Server::start();
    server.run_with(vec![
        RunResponse::Event(None, Some("no timestamp")),
        RunResponse::Event(Some((20, 5)), None),
        RunResponse::Event(Some((30, 0)), Some("Run is COMPLETED.")),
    ]);
    let session = connect(&server).await;
    let graph = session.create_dataflow_graph().await.unwrap();
    let read: Vec<_> = graph
        .start_run("file:///storage")
        .await
        .unwrap()
        .map(|event| match event {
            Ok(event) => Ok(texts(&[event])[0].2.to_string()),
            Err(error) => Err((error.kind(), error.message())),
        })
        .collect()
        .await;
    assert_eq!(
        read,
        [
            Err((
                ErrorKind::Decode,
                "the server sent a pipeline event without a timestamp".to_string()
            )),
            Err((
                ErrorKind::Decode,
                "the server sent a pipeline event without a message".to_string()
            )),
            Ok("Run is COMPLETED.".to_string()),
        ]
    );
}

#[test]
fn a_dropped_graph_is_removed_from_the_runtime_that_created_it() {
    for flavor in ["current-thread", "multi-thread"] {
        for place in ["in its runtime", "outside a runtime", "in another runtime"] {
            let runtime = runtime(flavor);
            let server = Server::start();
            let (session, graph) = runtime.block_on(async {
                let session = connect(&server).await;
                let graph = session.create_dataflow_graph().await.unwrap();
                (session, graph)
            });
            match place {
                "in its runtime" => runtime.block_on(async move { drop(graph) }),
                "outside a runtime" => drop(graph),
                _ => self::runtime("current-thread").block_on(async move { drop(graph) }),
            }
            assert!(
                runtime.block_on(server.wait_for(|r| matches!(r, Received::DropGraph(_)))),
                "{flavor}, {place}: not removed"
            );
            assert_eq!(drops(&server), ["graph-1"], "{flavor}, {place}");
            runtime.block_on(async move { drop(session) });
        }
    }
}

/// What holds a graph, and lets it go when it is called.
type Holder<'a> = Box<dyn FnOnce() + 'a>;

#[test]
fn a_graph_is_removed_once_when_the_last_of_what_holds_it_lets_it_go() {
    let holders = [
        "a clone",
        "a definition not yet awaited",
        "a definition in flight",
        "a start of a run in flight",
        "a run",
    ];
    for holder in holders {
        // On a current-thread runtime that the test does not drive, a spawned removal stays a
        // live task of the runtime until the test drives it.
        let runtime = runtime("current-thread");
        let tasks = || runtime.metrics().num_alive_tasks();
        let server = Server::start();
        let (session, graph): (Session, DataflowGraph) = runtime.block_on(async {
            let session = connect(&server).await;
            let graph = session.create_dataflow_graph().await.unwrap();
            (session, graph)
        });
        let in_flight = |call: BoxFuture<'static, ()>| -> Holder<'_> {
            server.wait_at(Wait::BeforeAccepting);
            let before = executions(&server);
            let call = runtime.spawn(call);
            assert!(
                runtime.block_on(async {
                    for _ in 0..500 {
                        if executions(&server) > before {
                            return true;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                    false
                }),
                "{holder}: not sent"
            );
            Box::new(|| {
                server.wait_at(Wait::Never);
                server.open_gate();
                let _ = runtime.block_on(call);
            })
        };
        let release: Holder<'_> = match holder {
            "a clone" => {
                let clone = graph.clone();
                Box::new(move || drop(clone))
            }
            "a definition not yet awaited" => {
                let definition = graph.define_temporary_view("v");
                Box::new(move || drop(definition))
            }
            "a definition in flight" => {
                let definition = graph.define_materialized_view("mv");
                in_flight(Box::pin(async move { drop(definition.await) }))
            }
            "a start of a run in flight" => {
                let start = graph.start_run("file:///storage");
                in_flight(Box::pin(async move { drop(start.await) }))
            }
            _ => {
                let run = runtime.block_on(graph.start_run("file:///storage").into_future());
                let run = run.unwrap();
                Box::new(move || drop(run))
            }
        };
        let before = tasks();
        drop(graph);
        assert_eq!(tasks(), before, "{holder}: removed while held");
        release();
        assert!(
            runtime.block_on(server.wait_for(|r| matches!(r, Received::DropGraph(_)))),
            "{holder}: not removed after"
        );
        assert_eq!(drops(&server), ["graph-1"], "{holder}");
        runtime.block_on(async move { drop(session) });
    }
}

#[test]
fn the_graph_of_a_stopped_session_is_not_removed() {
    let runtime = runtime("current-thread");
    let server = Server::start();
    let (session, graph) = runtime.block_on(async {
        let session = connect(&server).await;
        let graph = session.create_dataflow_graph().await.unwrap();
        session.stop().await.unwrap();
        (session, graph)
    });
    let before = runtime.metrics().num_alive_tasks();
    drop(graph);
    assert_eq!(runtime.metrics().num_alive_tasks(), before);
    runtime.block_on(async move { drop(session) });
    assert!(drops(&server).is_empty());
}

#[test]
fn a_graph_keeps_what_the_queries_of_its_flows_refer_to() {
    let runtime = runtime("current-thread");
    let tasks = || runtime.metrics().num_alive_tasks();
    let server = Server::start();
    let (session, graph, model, scaled) = runtime.block_on(async {
        let session = connect(&server).await;
        let model = StandardScaler::new().fit(&session.range(3)).await.unwrap();
        let graph = session.create_dataflow_graph().await.unwrap();
        let scaled = model.transform(&session.range(3));
        graph.define_flow("f", "t", &scaled).await.unwrap();
        (session, graph, model, scaled)
    });
    let before = tasks();
    drop(scaled);
    drop(model);
    assert_eq!(
        tasks(),
        before,
        "the model deleted while its flow's graph lives"
    );
    drop(graph);
    let deleted = Received::Delete(vec!["model-1".to_string()]);
    assert!(
        runtime.block_on(server.wait_for(|r| *r == deleted)),
        "the model not deleted with the graph"
    );
    assert_eq!(drops(&server), ["graph-1"]);
    runtime.block_on(async move { drop(session) });
}
