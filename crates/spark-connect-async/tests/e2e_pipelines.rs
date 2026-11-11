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

//! The pipelines of the async API against a Spark Connect server at `SPARK_REMOTE`; each test
//! passes without running when it is not set.

use std::future::{Future, IntoFuture};
use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::{FutureExt, StreamExt};
use spark_connect_async::functions::expr;
use spark_connect_async::pipelines::{DataflowGraph, PipelineEvent, PipelineRun};
use spark_connect_async::{ErrorKind, Session};
use tokio::time::timeout;

/// The longest that a step waits for the server.
const WAIT: Duration = Duration::from_secs(120);

/// The server, a suffix of names unique to this test, and the connection string of a session of
/// the server of its own.
fn remote(test: &str) -> Option<(String, String, String)> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let unique = format!("{:x}", now.as_nanos() & 0xffff_ffff_ffff);
    let id = format!("00000000-0000-4000-8000-{:0>12}", unique);
    let url = format!("{remote}/;session_id={id};user_id=e2e_pipelines");
    Some((remote, format!("{test}_{unique}"), url))
}

/// Runs `body` on a session of `url`, then stops the session, which the server answers once
/// the runs of the session have ended, and drops the datasets `datasets` with another session
/// of `remote`, also when `body` panics.
async fn cleaned_up<F: Future<Output = ()>>(
    remote: &str,
    url: &str,
    datasets: &[String],
    body: impl FnOnce(Session) -> F,
) {
    let session = timeout(WAIT, Session::builder().remote(url).connect())
        .await
        .unwrap()
        .unwrap();
    let outcome = AssertUnwindSafe(body(session.clone())).catch_unwind().await;
    let _ = timeout(WAIT, session.stop()).await;
    let cleaner = timeout(WAIT, Session::builder().remote(remote).connect())
        .await
        .unwrap()
        .unwrap();
    for dataset in datasets {
        for kind in ["MATERIALIZED VIEW", "TABLE", "VIEW"] {
            let statement = cleaner.sql(format!("DROP {kind} IF EXISTS {dataset}"));
            let _ = timeout(WAIT, statement.into_future()).await;
        }
    }
    let _ = timeout(WAIT, cleaner.stop()).await;
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
}

/// A pipeline storage of its own, which is removed when it is dropped, and its URI.
fn storage() -> (tempfile::TempDir, String) {
    let directory = tempfile::tempdir().unwrap();
    let uri = format!("file://{}", directory.path().display());
    (directory, uri)
}

/// The events of `run` to its end, each read within [`WAIT`].
async fn events(mut run: PipelineRun) -> Vec<spark_connect_async::Result<PipelineEvent>> {
    let mut events = Vec::new();
    while let Some(event) = timeout(WAIT, run.next()).await.unwrap() {
        events.push(event);
    }
    events
}

/// The error of the run that `start` starts: that of its start, or else the first of its
/// events.
async fn error_of(
    start: impl IntoFuture<Output = spark_connect_async::Result<PipelineRun>>,
) -> Option<spark_connect_async::Error> {
    match timeout(WAIT, start.into_future()).await.unwrap() {
        Ok(run) => events(run).await.into_iter().find_map(Result::err),
        Err(error) => Some(error),
    }
}

/// The ids of the rows of `dataset`, in order.
async fn ids(session: &Session, dataset: &str) -> Vec<i64> {
    let read = session.table(dataset).sort(["id"]).collect_as::<(i64,)>();
    let rows = timeout(WAIT, read).await.unwrap().unwrap();
    rows.into_iter().map(|(id,)| id).collect()
}

/// Whether the server keeps the graph `id` for the session of `url`, asked through the old API,
/// which defines a temporary view in it if it does.
async fn kept(url: &str, id: &str) -> bool {
    let (url, id) = (url.to_string(), id.to_string());
    tokio::task::spawn_blocking(move || {
        let session = spark_connect::SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        let view = spark_connect::pipelines::OutputSpec {
            name: format!("probe_{}", id.replace('-', "_")),
            output_type: spark_connect_proto::OutputType::TemporaryView as i32,
            comment: None,
            source_code_location: None,
            table_details: None,
            sink_details: None,
        };
        match spark_connect::pipelines::define_output(&session, &id, view) {
            Ok(()) => true,
            Err(error) if error.to_string().contains("DATAFLOW_GRAPH_NOT_FOUND") => false,
            Err(error) => panic!("{error}"),
        }
    })
    .await
    .unwrap()
}

/// Waits until the server no longer keeps the graph `id`, at most [`WAIT`].
async fn removed(url: &str, id: &str) -> bool {
    let deadline = Instant::now() + WAIT;
    while Instant::now() < deadline {
        if !kept(url, id).await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_of_datasets_defined_from_sql_and_from_a_dataframe_fills_them_and_streams_its_events()
{
    let Some((remote, unique, url)) = remote("sdp_run") else {
        return;
    };
    let numbers = format!("{unique}_numbers");
    let evens = format!("{unique}_evens");
    let tens = format!("{unique}_tens");
    let datasets = [numbers.clone(), evens.clone(), tens.clone()];
    cleaned_up(&remote, &url.clone(), &datasets, |session| async move {
        let (_directory, storage) = storage();
        let started = chrono::Utc::now() - chrono::TimeDelta::minutes(1);
        let graph: DataflowGraph = timeout(WAIT, session.create_dataflow_graph().into_future())
            .await
            .unwrap()
            .unwrap();
        let sql = format!("CREATE MATERIALIZED VIEW {numbers} AS SELECT id FROM range(6)");
        let defined = graph.define_sql(sql).file_path("numbers.sql");
        timeout(WAIT, defined.into_future()).await.unwrap().unwrap();
        let view = graph
            .define_materialized_view(&evens)
            .comment("the even numbers");
        let view = timeout(WAIT, view.into_future()).await.unwrap().unwrap();
        assert_eq!(
            (
                view.catalog.as_deref(),
                view.namespace.clone(),
                view.name.as_str()
            ),
            (
                Some("spark_catalog"),
                vec!["default".to_string()],
                evens.as_str()
            )
        );
        let query = session.table(&numbers).filter(expr("id % 2 = 0"));
        let flow = graph.define_flow(&evens, &evens, &query);
        let flow = timeout(WAIT, flow.into_future()).await.unwrap().unwrap();
        assert_eq!(flow.name, evens);
        timeout(WAIT, graph.define_materialized_view(&tens).into_future())
            .await
            .unwrap()
            .unwrap();
        let query = format!("SELECT id * 10 AS id FROM {numbers}");
        let flow = graph.define_flow(&tens, &tens, query);
        timeout(WAIT, flow.into_future()).await.unwrap().unwrap();
        let run = graph.start_run(&storage);
        let run = timeout(WAIT, run.into_future()).await.unwrap().unwrap();
        let events: Vec<PipelineEvent> = events(run)
            .await
            .into_iter()
            .map(|event| event.unwrap())
            .collect();
        let ended = chrono::Utc::now() + chrono::TimeDelta::minutes(1);
        assert!(
            events.iter().any(|event| event.message.contains(&evens)),
            "{events:?}"
        );
        for event in &events {
            assert!(
                started <= event.timestamp && event.timestamp <= ended,
                "{event:?}"
            );
        }
        assert_eq!(ids(&session, &numbers).await, [0, 1, 2, 3, 4, 5]);
        assert_eq!(ids(&session, &evens).await, [0, 2, 4]);
        assert_eq!(ids(&session, &tens).await, [0, 10, 20, 30, 40, 50]);
        // Spark 4.3.0 runs a graph at most once.
        let error = error_of(graph.start_run(&storage)).await;
        let error = error.expect("a second run of the graph");
        assert_eq!(
            error.error_class(),
            Some("SPARK_CONNECT_ILLEGAL_STATE.EXECUTION_STATE_EXECUTE_HOLDER_ALREADY_EXISTS_GRAPH"),
            "{error}"
        );
        let id = graph.id().to_string();
        assert!(kept(&url, &id).await, "the graph not kept while it is held");
        drop(graph);
        assert!(removed(&url, &id).await, "the dropped graph still kept");
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_whose_events_are_dropped_still_runs_to_its_end_on_the_server() {
    let Some((remote, unique, url)) = remote("sdp_dropped") else {
        return;
    };
    let slow = format!("{unique}_slow");
    let datasets = [slow.clone()];
    cleaned_up(&remote, &url.clone(), &datasets, |session| async move {
        let (_directory, storage) = storage();
        let graph = timeout(WAIT, session.create_dataflow_graph().into_future())
            .await
            .unwrap()
            .unwrap();
        // One row a second.
        let sql = format!(
            "CREATE MATERIALIZED VIEW {slow} AS \
             SELECT id, reflect('java.lang.Thread', 'sleep', 1000L) AS slept FROM range(3)"
        );
        timeout(WAIT, graph.define_sql(sql).into_future())
            .await
            .unwrap()
            .unwrap();
        let mut run = timeout(WAIT, graph.start_run(&storage).into_future())
            .await
            .unwrap()
            .unwrap();
        timeout(WAIT, run.next()).await.unwrap().unwrap().unwrap();
        drop(run);
        drop(graph);
        let deadline = Instant::now() + WAIT;
        loop {
            let count = session.table(&slow).count();
            if let Ok(Ok(3)) = timeout(WAIT, count).await {
                break;
            }
            assert!(Instant::now() < deadline, "the run did not fill its view");
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_run_is_an_error_with_the_condition_and_sql_state_of_the_server() {
    let Some((remote, unique, url)) = remote("sdp_failed") else {
        return;
    };
    let failing = format!("{unique}_failing");
    let datasets = [failing.clone()];
    cleaned_up(&remote, &url.clone(), &datasets, |session| async move {
        let (_directory, storage) = storage();
        let graph = session
            .create_dataflow_graph()
            .sql_conf("spark.sql.pipelines.maxFlowRetryAttempts", "0");
        let graph = timeout(WAIT, graph.into_future()).await.unwrap().unwrap();
        let sql = format!("CREATE MATERIALIZED VIEW {failing} AS SELECT raise_error('boom') AS a");
        timeout(WAIT, graph.define_sql(sql).into_future())
            .await
            .unwrap()
            .unwrap();
        let error = error_of(graph.start_run("relative/storage")).await;
        let error = error.expect("a run of a storage without a scheme");
        assert_eq!(
            error.error_class(),
            Some("PIPELINE_STORAGE_ROOT_INVALID"),
            "{error}"
        );
        let run = timeout(WAIT, graph.start_run(&storage).into_future())
            .await
            .unwrap()
            .unwrap();
        let events = events(run).await;
        let (last, before) = events.split_last().expect("no events");
        assert!(before.iter().all(Result::is_ok), "{events:?}");
        let error = last.as_ref().unwrap_err();
        assert_eq!(
            error.error_class(),
            Some("USER_RAISED_EXCEPTION"),
            "{error}"
        );
        assert_eq!(error.sql_state().as_deref(), Some("P0001"), "{error}");
        assert_eq!(error.kind(), ErrorKind::Execution, "{error}");
        assert!(!error.message().contains("Exception"), "{error}");
    })
    .await;
}
