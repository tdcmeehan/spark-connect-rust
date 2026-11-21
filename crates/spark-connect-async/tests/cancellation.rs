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

//! Dropping the future or the stream of an action at each point of an execution sends what
//! stops the operation on the server, once.

mod gated;

use std::future::IntoFuture;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use gated::{Received, Server, Wait, REJECTED_CONFIG};
use spark_connect_async::ml::{
    Estimator, Evaluator, LogisticRegression, LogisticRegressionModel,
    MulticlassClassificationEvaluator,
};
use spark_connect_async::Session;
use spark_connect_core::types::DataType;

async fn connect(server: &Server) -> Session {
    Session::builder()
        .remote(server.url())
        .connect()
        .await
        .unwrap()
}

/// The `ReleaseAll`s of `operation_id`, once one has come and the server has received a later
/// request of `session`.
async fn releases(server: &Server, session: &Session, operation_id: &str) -> usize {
    let release = Received::ReleaseAll(operation_id.to_string());
    server.wait_for(|received| *received == release).await;
    releases_by_now(server, session, operation_id).await
}

/// The `ReleaseAll`s of `operation_id` that the server received before a later request of
/// `session`.
async fn releases_by_now(server: &Server, session: &Session, operation_id: &str) -> usize {
    session.conf().set("spark.test.later", "1").await.unwrap();
    server
        .received()
        .iter()
        .filter(|received| **received == Received::ReleaseAll(operation_id.to_string()))
        .count()
}

fn interrupted(server: &Server, operation_id: &str) -> bool {
    server
        .received()
        .contains(&Received::Interrupt(operation_id.to_string()))
}

async fn dropped_before_the_server_accepts_it(server: &Server) {
    server.wait_at(Wait::BeforeAccepting);
    let updates = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&updates);
    let session = Session::builder()
        .remote(server.url())
        .on_progress(move |_| {
            counted.fetch_add(1, Ordering::SeqCst);
        })
        .connect()
        .await
        .unwrap();
    let collect = tokio::spawn(session.range(3).collect());
    assert!(server.wait_for(|r| matches!(r, Received::Execute(_))).await);
    let operation_id = server.operation_id();
    collect.abort();
    assert!(collect.await.unwrap_err().is_cancelled());
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::Interrupt(id.clone()))
            .await
    );
    assert_eq!(
        releases_by_now(server, &session, &operation_id).await,
        0,
        "released before it was accepted"
    );
    server.open_gate();
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::ReleaseAll(id.clone()))
            .await
    );
    assert_eq!(releases(server, &session, &operation_id).await, 1);
    assert_eq!(
        updates.load(Ordering::SeqCst),
        0,
        "progress of an abandoned execution"
    );
}

async fn dropped_after_the_first_batch(server: &Server) {
    server.wait_at(Wait::AfterTheFirstBatch);
    let session = connect(server).await;
    let mut batches = session.range(3).stream().await.unwrap();
    assert_eq!(batches.next().await.unwrap().unwrap().num_rows(), 1);
    let operation_id = batches.operation_id().to_string();
    drop(batches);
    assert_eq!(releases(server, &session, &operation_id).await, 1);
    assert!(!interrupted(server, &operation_id));
    server.open_gate();
}

async fn dropped_after_the_end(server: &Server) {
    server.wait_at(Wait::Never);
    let session = connect(server).await;
    let mut batches = session.range(3).stream().await.unwrap();
    let mut rows = 0;
    while let Some(batch) = batches.next().await {
        rows += batch.unwrap().num_rows();
    }
    assert_eq!(rows, 2);
    let operation_id = batches.operation_id().to_string();
    let before = releases(server, &session, &operation_id).await;
    drop(batches);
    assert_eq!(
        releases_by_now(server, &session, &operation_id).await,
        before
    );
    assert!(before <= 1);
    assert!(!interrupted(server, &operation_id));
}

async fn dropped_while_reattaching(server: &Server) {
    server.wait_at(Wait::BeforeReattaching);
    let session = connect(server).await;
    let mut batches = session.range(3).stream().await.unwrap();
    assert_eq!(batches.next().await.unwrap().unwrap().num_rows(), 1);
    let operation_id = batches.operation_id().to_string();
    let reading = tokio::spawn(async move { batches.next().await.map(|batch| batch.is_ok()) });
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::Reattach(id.clone()))
            .await
    );
    reading.abort();
    assert!(reading.await.unwrap_err().is_cancelled());
    assert_eq!(releases(server, &session, &operation_id).await, 1);
    assert!(!interrupted(server, &operation_id));
    server.open_gate();
}

#[test]
fn dropping_an_action_at_each_point_stops_the_operation_once_in_either_runtime() {
    for flavor in ["current-thread", "multi-thread"] {
        let mut runtime = match flavor {
            "current-thread" => tokio::runtime::Builder::new_current_thread(),
            _ => tokio::runtime::Builder::new_multi_thread(),
        };
        runtime.enable_all().build().unwrap().block_on(async {
            for (point, run) in [
                ("before accepting", 0),
                ("after the first batch", 1),
                ("after the end", 2),
                ("while reattaching", 3),
            ] {
                // A server of each case, so that no gate opened in one passes another.
                let server = Server::start();
                println!("{flavor}: {point}");
                match run {
                    0 => dropped_before_the_server_accepts_it(&server).await,
                    1 => dropped_after_the_first_batch(&server).await,
                    2 => dropped_after_the_end(&server).await,
                    _ => dropped_while_reattaching(&server).await,
                }
            }
        });
    }
}

#[tokio::test]
async fn a_stream_keeps_its_schema_and_execution_info_while_a_dropped_read_is_pending() {
    let server = Server::start();
    server.wait_at(Wait::AfterTheFirstBatch);
    let session = connect(&server).await;
    let mut batches = session.range(3).stream().await.unwrap();
    assert_eq!(batches.next().await.unwrap().unwrap().num_rows(), 1);
    let pending = tokio::time::timeout(Duration::from_millis(100), batches.next()).await;
    assert!(pending.is_err(), "the server sent past its gate");
    let schema = DataType::from_proto(&gated::schema()).unwrap();
    assert_eq!(batches.schema(), Some(&schema));
    assert_eq!(
        batches.execution_info().operation_id(),
        Some(batches.operation_id())
    );
    server.open_gate();
    assert_eq!(batches.next().await.unwrap().unwrap().num_rows(), 1);
}

#[tokio::test]
async fn a_stream_read_to_its_end_reports_the_metrics_sent_after_its_last_batch() {
    use spark_connect_async::{lit, Value};
    let server = Server::start();
    server.end_results_with_metrics();
    let session = connect(&server).await;
    let observed = session.range(2).observe("metrics", [lit(1).alias("m")]);
    let mut batches = observed.stream().await.unwrap();
    assert_eq!(batches.next().await.unwrap().unwrap().num_rows(), 1);
    let first = batches.execution_info();
    assert!(first.plan_metrics().is_none() && first.observed_metrics().is_empty());
    while let Some(batch) = batches.next().await {
        batch.unwrap();
    }
    let info = batches.execution_info();
    assert_eq!(info.operation_id(), Some(batches.operation_id()));
    assert_eq!(info.plan_metrics(), Some(&gated::plan_metrics()));
    let metrics = info.observed_metrics();
    assert_eq!(metrics.len(), 1, "{metrics:?}");
    assert_eq!(metrics[0].0, "metrics");
    assert_eq!(metrics[0].1.get_by_name("m"), Some(&Value::Int(1)));
}

#[tokio::test]
async fn a_command_of_an_observed_dataframe_sets_its_observation() {
    use spark_connect_async::{lit, Observation, Value};
    let server = Server::start();
    server.end_results_with_metrics();
    let session = connect(&server).await;
    let observation = Observation::new("metrics");
    let observed = session.range(2).observe(&observation, [lit(1).alias("m")]);
    observed
        .create_temp_view("v", Default::default())
        .await
        .unwrap();
    let metrics = observation.get().unwrap();
    assert_eq!(metrics.get_by_name("m"), Some(&Value::Int(1)));
}

#[tokio::test]
async fn nulls_of_a_column_declared_non_nullable_read_as_nullable_in_each_collected_batch() {
    let server = Server::start();
    server.send_a_null_in_the_second_batch();
    let session = connect(&server).await;
    let streamed: Vec<_> = session.range(2).stream().await.unwrap().collect().await;
    let nullable: Vec<_> = streamed
        .iter()
        .map(|batch| batch.as_ref().unwrap().schema().field(0).is_nullable())
        .collect();
    assert_eq!(nullable, [false, true]);
    let collected = session.range(2).collect().await.unwrap();
    assert_eq!(collected.len(), 2);
    assert!(
        collected
            .iter()
            .all(|batch| batch.schema().field(0).is_nullable()),
        "{collected:?}"
    );
    assert_eq!(collected[1].column(0).null_count(), 1);
}

#[tokio::test]
async fn a_session_whose_configuration_the_server_rejects_is_released() {
    let server = Server::start();
    let error = Session::builder()
        .remote(server.url())
        .config(REJECTED_CONFIG, "1")
        .connect()
        .await
        .err()
        .unwrap();
    assert!(error.message().contains("rejected"), "{error}");
    assert!(server.wait_for(|r| *r == Received::ReleaseSession).await);
}

/// The future of the wait `wait` of `query`.
fn wait_of(
    wait: &str,
    query: &spark_connect_async::streaming::StreamingQuery,
    session: &Session,
) -> futures::future::BoxFuture<'static, spark_connect_async::Result<()>> {
    use futures::FutureExt;
    match wait {
        "await_termination" => query.await_termination(None).map(|r| r.map(drop)).boxed(),
        "process_all_available" => query.process_all_available().boxed(),
        _ => session
            .streams()
            .await_any_termination(None)
            .map(|r| r.map(drop))
            .boxed(),
    }
}

async fn executions(server: &Server) -> usize {
    server
        .received()
        .iter()
        .filter(|received| matches!(received, Received::Execute(_)))
        .count()
}

async fn a_wait_dropped_before_the_server_accepts_it(server: &Server, wait: &str) {
    let session = connect(server).await;
    let query = session
        .range(3)
        .write_stream()
        .format("memory")
        .start_without_path()
        .await
        .unwrap();
    server.wait_at(Wait::BeforeAccepting);
    let waiting = tokio::spawn(wait_of(wait, &query, &session));
    while executions(server).await < 2 {
        assert!(server.wait_for(|r| matches!(r, Received::Execute(_))).await);
        tokio::task::yield_now().await;
    }
    let operation_id = server.operation_id();
    waiting.abort();
    assert!(waiting.await.unwrap_err().is_cancelled());
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::Interrupt(id.clone()))
            .await,
        "{wait}: not interrupted"
    );
    assert_eq!(
        releases_by_now(server, &session, &operation_id).await,
        0,
        "{wait}: released early"
    );
    server.open_gate();
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::ReleaseAll(id.clone()))
            .await,
        "{wait}: not released"
    );
    assert_eq!(releases(server, &session, &operation_id).await, 1);
}

async fn a_wait_dropped_after_the_server_accepted_it(server: &Server, wait: &str) {
    let session = connect(server).await;
    let query = session
        .range(3)
        .write_stream()
        .format("memory")
        .start_without_path()
        .await
        .unwrap();
    server.wait_at(Wait::AfterTheFirstBatch);
    let waiting = tokio::spawn(wait_of(wait, &query, &session));
    while executions(server).await < 2 {
        assert!(server.wait_for(|r| matches!(r, Received::Execute(_))).await);
        tokio::task::yield_now().await;
    }
    let operation_id = server.operation_id();
    // The client releases each response that it has read.
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::ReleaseUntil(id.clone()))
            .await
    );
    waiting.abort();
    assert!(waiting.await.unwrap_err().is_cancelled());
    assert_eq!(releases(server, &session, &operation_id).await, 1, "{wait}");
    assert!(!interrupted(server, &operation_id), "{wait}: interrupted");
    server.open_gate();
}

#[test]
fn dropping_a_wait_of_a_streaming_query_stops_it_once_in_either_runtime() {
    for flavor in ["current-thread", "multi-thread"] {
        let mut runtime = match flavor {
            "current-thread" => tokio::runtime::Builder::new_current_thread(),
            _ => tokio::runtime::Builder::new_multi_thread(),
        };
        runtime.enable_all().build().unwrap().block_on(async {
            for wait in [
                "await_termination",
                "process_all_available",
                "await_any_termination",
            ] {
                for accepted in [false, true] {
                    let server = Server::start();
                    println!("{flavor}: {wait}, accepted {accepted}");
                    if accepted {
                        a_wait_dropped_after_the_server_accepted_it(&server, wait).await;
                    } else {
                        a_wait_dropped_before_the_server_accepts_it(&server, wait).await;
                    }
                }
            }
        });
    }
}

/// The future of the call `call` of Spark ML: a fit, an evaluation, or an attribute of `model`.
fn ml_call(
    call: &str,
    session: &Session,
    model: &LogisticRegressionModel,
) -> futures::future::BoxFuture<'static, ()> {
    use futures::FutureExt;
    match call {
        "fit" => LogisticRegression::new()
            .fit(&session.range(3))
            .map(drop)
            .boxed(),
        "evaluate" => MulticlassClassificationEvaluator::new()
            .evaluate(&session.range(3))
            .map(drop)
            .boxed(),
        _ => model.coefficients().map(drop).boxed(),
    }
}

/// Starts the call `call` of Spark ML once the server waits at `wait`, and returns its task
/// once the server has received it, with its operation id.
async fn an_ml_call_sent(
    server: &Server,
    call: &str,
    wait: Wait,
) -> (tokio::task::JoinHandle<()>, String, Session) {
    let session = connect(server).await;
    let model = LogisticRegression::new()
        .fit(&session.range(3))
        .await
        .unwrap();
    server.wait_at(wait);
    let before = executions(server).await;
    let calling = tokio::spawn(ml_call(call, &session, &model));
    while executions(server).await <= before {
        assert!(server.wait_for(|r| matches!(r, Received::Execute(_))).await);
        tokio::task::yield_now().await;
    }
    (calling, server.operation_id(), session)
}

async fn an_ml_call_dropped_before_the_server_accepts_it(server: &Server, call: &str) {
    let (calling, operation_id, session) =
        an_ml_call_sent(server, call, Wait::BeforeAccepting).await;
    calling.abort();
    assert!(calling.await.unwrap_err().is_cancelled());
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::Interrupt(id.clone()))
            .await,
        "{call}: not interrupted"
    );
    assert_eq!(
        releases_by_now(server, &session, &operation_id).await,
        0,
        "{call}: released early"
    );
    server.open_gate();
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::ReleaseAll(id.clone()))
            .await,
        "{call}: not released"
    );
    assert_eq!(releases(server, &session, &operation_id).await, 1);
}

async fn an_ml_call_dropped_after_the_server_accepted_it(server: &Server, call: &str) {
    let (calling, operation_id, session) =
        an_ml_call_sent(server, call, Wait::AfterTheFirstBatch).await;
    // The client releases each response that it has read.
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::ReleaseUntil(id.clone()))
            .await
    );
    calling.abort();
    assert!(calling.await.unwrap_err().is_cancelled());
    assert_eq!(releases(server, &session, &operation_id).await, 1, "{call}");
    assert!(!interrupted(server, &operation_id), "{call}: interrupted");
    server.open_gate();
}

#[test]
fn dropping_a_fit_an_evaluation_or_an_attribute_stops_it_once_in_either_runtime() {
    for flavor in ["current-thread", "multi-thread"] {
        let mut runtime = match flavor {
            "current-thread" => tokio::runtime::Builder::new_current_thread(),
            _ => tokio::runtime::Builder::new_multi_thread(),
        };
        runtime.enable_all().build().unwrap().block_on(async {
            for call in ["fit", "evaluate", "attribute"] {
                for accepted in [false, true] {
                    let server = Server::start();
                    println!("{flavor}: {call}, accepted {accepted}");
                    if accepted {
                        an_ml_call_dropped_after_the_server_accepted_it(&server, call).await;
                    } else {
                        an_ml_call_dropped_before_the_server_accepts_it(&server, call).await;
                    }
                }
            }
        });
    }
}

async fn a_start_of_a_run_dropped_before_the_server_accepts_it(server: &Server) {
    let session = connect(server).await;
    let graph = session.create_dataflow_graph().await.unwrap();
    server.wait_at(Wait::BeforeAccepting);
    let before = executions(server).await;
    let starting = tokio::spawn(graph.start_run("file:///storage").into_future());
    while executions(server).await <= before {
        assert!(server.wait_for(|r| matches!(r, Received::Execute(_))).await);
        tokio::task::yield_now().await;
    }
    let operation_id = server.operation_id();
    starting.abort();
    assert!(starting.await.unwrap_err().is_cancelled());
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::Interrupt(id.clone()))
            .await,
        "not interrupted"
    );
    assert_eq!(
        releases_by_now(server, &session, &operation_id).await,
        0,
        "released early"
    );
    server.open_gate();
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::ReleaseAll(id.clone()))
            .await,
        "not released"
    );
    assert_eq!(releases(server, &session, &operation_id).await, 1);
}

async fn the_events_of_a_run_dropped_after_the_first(server: &Server) {
    let session = connect(server).await;
    let graph = session.create_dataflow_graph().await.unwrap();
    server.wait_at(Wait::AfterTheFirstBatch);
    let mut run = graph.start_run("file:///storage").await.unwrap();
    assert!(run.next().await.unwrap().is_ok());
    let operation_id = run.operation_id().to_string();
    drop(run);
    assert_eq!(releases(server, &session, &operation_id).await, 1);
    assert!(!interrupted(server, &operation_id), "interrupted");
    server.open_gate();
}

#[test]
fn dropping_the_start_or_the_events_of_a_run_stops_them_once_in_either_runtime() {
    for flavor in ["current-thread", "multi-thread"] {
        let mut runtime = match flavor {
            "current-thread" => tokio::runtime::Builder::new_current_thread(),
            _ => tokio::runtime::Builder::new_multi_thread(),
        };
        runtime.enable_all().build().unwrap().block_on(async {
            for accepted in [false, true] {
                let server = Server::start();
                println!("{flavor}: accepted {accepted}");
                if accepted {
                    the_events_of_a_run_dropped_after_the_first(&server).await;
                } else {
                    a_start_of_a_run_dropped_before_the_server_accepts_it(&server).await;
                }
            }
        });
    }
}

#[test]
fn a_checkpoint_dropped_after_the_server_answered_it_removes_its_relation() {
    for flavor in ["current-thread", "multi-thread"] {
        let mut runtime = match flavor {
            "current-thread" => tokio::runtime::Builder::new_current_thread(),
            _ => tokio::runtime::Builder::new_multi_thread(),
        };
        runtime.enable_all().build().unwrap().block_on(async {
            let server = Server::start();
            let session = connect(&server).await;
            server.wait_at(Wait::AfterTheFirstBatch);
            let checkpoint = spark_connect_async::actions::Checkpoint::default();
            let checkpointing = tokio::spawn(session.range(3).checkpoint(checkpoint));
            // The client releases each response that it has read.
            assert!(
                server
                    .wait_for(|r| matches!(r, Received::ReleaseUntil(_)))
                    .await,
                "{flavor}: the answer was not read"
            );
            server.wait_at(Wait::Never);
            checkpointing.abort();
            assert!(checkpointing.await.is_err_and(|error| error.is_cancelled()));
            let removal = Received::RemoveCachedRelation("checkpointed".to_string());
            assert!(
                server.wait_for(|r| *r == removal).await,
                "{flavor}: the relation of the dropped checkpoint was not removed"
            );
            server.open_gate();
        });
    }
}
