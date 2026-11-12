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

//! The actions of the async API against a Spark Connect server at `SPARK_REMOTE`; each test
//! passes without running when it is not set.

use std::future::{Future, IntoFuture};
use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::{FutureExt, StreamExt};
use serde::Deserialize;
use spark_connect::functions::count;
use spark_connect_async::actions::{Checkpoint, ExplainMode, TempView};
use spark_connect_async::{col, lit, Column, Observation, Session};
use spark_connect_core::{ChannelBuilder, SparkConnectClient};
use spark_connect_proto as proto;
use tokio::sync::mpsc;

/// The longest that a call to the server, or a wait for it, may take.
const WAIT: Duration = Duration::from_secs(120);

/// The longest that the server may take to drop an execution that the client released.
const RELEASE: Duration = Duration::from_secs(30);

/// Waits for the call that `call` makes until the deadline. The call's future is made and boxed
/// in a frame of its own: made in the frame of a test, the futures of a debug build overflowed
/// the stack of its thread.
async fn within<F: IntoFuture>(call: impl FnOnce() -> F) -> F::Output {
    let call = Box::pin(call().into_future());
    tokio::time::timeout(WAIT, call)
        .await
        .expect("no answer before the deadline")
}

/// The connection string of a session of the server of its own, so that a second client can
/// reach it.
fn url() -> Option<String> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let id = format!(
        "00000000-0000-4000-8000-{:012x}",
        now.as_nanos() & 0xffff_ffff_ffff
    );
    Some(format!("{remote}/;session_id={id};user_id=e2e_actions"))
}

async fn session() -> Option<Session> {
    let url = url()?;
    Some(
        within(|| Session::builder().remote(url).connect())
            .await
            .unwrap(),
    )
}

/// Runs `body` until the deadline, then stops `session`, which ends what it runs and drops its
/// views, also when `body` panics or does not finish. The body is boxed: unboxed, a body that
/// holds the futures of its calls overflowed the stack of its thread in a debug build.
async fn cleaned_up(session: &Session, body: impl Future<Output = ()>) {
    let outcome = AssertUnwindSafe(tokio::time::timeout(WAIT, Box::pin(body)))
        .catch_unwind()
        .await;
    let _ = tokio::time::timeout(WAIT, session.stop()).await;
    match outcome {
        Err(panic) => std::panic::resume_unwind(panic),
        Ok(Err(_)) => panic!("the test did not end within {WAIT:?}"),
        Ok(Ok(())) => {}
    }
}

/// Waits until the server no longer has the operation `operation_id` of the session of `url`:
/// reattaching to it fails with `OPERATION_NOT_FOUND` once it is released, and replays its
/// responses until then.
async fn released(url: &str, operation_id: &str) {
    let client = SparkConnectClient::connect(&ChannelBuilder::parse(url).unwrap())
        .await
        .unwrap();
    let deadline = Instant::now() + RELEASE;
    loop {
        let request = proto::ReattachExecuteRequest {
            session_id: client.session_id().to_string(),
            user_context: Some(proto::UserContext {
                user_id: "e2e_actions".to_string(),
                ..Default::default()
            }),
            operation_id: operation_id.to_string(),
            ..Default::default()
        };
        let answer = match client.reattach_execute(request).await {
            Err(error) => error.to_string(),
            Ok(mut stream) => match stream.message().await {
                Err(status) => status.message().to_string(),
                Ok(_) => String::new(),
            },
        };
        if answer.contains("OPERATION_NOT_FOUND") {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the server still has {operation_id}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// A query that runs for minutes.
const LONG: &str = "SELECT count(*) FROM range(1000000) a CROSS JOIN range(1000000) b \
                    WHERE a.id + b.id = -1";

#[tokio::test]
async fn results_read_into_rust_types_with_the_server_names() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, read_into_rust_types(&session)).await;
}

async fn read_into_rust_types(session: &Session) {
    #[derive(Deserialize, Debug, PartialEq)]
    struct Point {
        x: i64,
    }
    #[derive(Deserialize, Debug, PartialEq)]
    struct Row {
        id: i64,
        point: Point,
        labels: Vec<(String, i64)>,
    }
    let df = session
        .sql("SELECT id, named_struct('x', id) AS point, map('k', id) AS labels FROM range(3)")
        .await
        .unwrap();
    let rows: Vec<Row> = df.collect_as().await.unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows[2],
        Row {
            id: 2,
            point: Point { x: 2 },
            labels: vec![("k".to_string(), 2)]
        }
    );
    assert_eq!(df.count().await.unwrap(), 3);
    assert_eq!(df.columns().await.unwrap(), ["id", "point", "labels"]);
    assert!(df.show().to_string().await.unwrap().contains("point"));
    assert!(df
        .explain(ExplainMode::Extended)
        .await
        .unwrap()
        .contains("Range"));
    assert_eq!(df.take(2).await.unwrap().len(), 2);
    assert!(df.filter(col("id").gt(lit(5))).is_empty().await.unwrap());
}

#[tokio::test]
async fn checkpoints_views_and_observations_work_on_the_server() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, checkpoints_views_and_observations(&session)).await;
}

async fn checkpoints_views_and_observations(session: &Session) {
    let df = session.range(5);
    let checkpoint = df.checkpoint(Checkpoint::default().local()).await.unwrap();
    assert_eq!(checkpoint.count().await.unwrap(), 5);
    df.create_temp_view("b7_numbers", TempView::default().or_replace())
        .await
        .unwrap();
    let viewed = session.sql("SELECT * FROM b7_numbers").await.unwrap();
    assert_eq!(viewed.count().await.unwrap(), 5);
    let observation = Observation::new("b7_metrics");
    let observed = df.observe(
        &observation,
        [Column::from(count(lit(1).into())).alias("rows")],
    );
    observed.collect().await.unwrap();
    let metrics = observation.get().unwrap();
    assert_eq!(
        metrics.get_by_name("rows"),
        Some(&spark_connect_async::Value::Long(5))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_a_running_action_stops_it_on_the_server() {
    let Some(url) = url() else {
        return;
    };
    // The operation of each progress that the server reports while tasks run.
    let (progress, mut running) = mpsc::unbounded_channel();
    let session = within(|| {
        Session::builder()
            .remote(&url)
            .on_progress(move |update| {
                if !update.done {
                    let _ = progress.send(update.operation_id.clone());
                }
            })
            .connect()
    })
    .await
    .unwrap();
    cleaned_up(&session, async {
        within(|| {
            session
                .conf()
                .set("spark.connect.progress.reportInterval", "10ms")
        })
        .await
        .unwrap();

        // The control: the tagged query runs until it is interrupted.
        let control = session.with_tag("b7-control").unwrap();
        let query = within(|| control.sql(LONG)).await.unwrap();
        let collecting = tokio::spawn(query.collect());
        let id = within(|| running.recv()).await.unwrap();
        assert_eq!(
            within(|| session.interrupt_tag("b7-control"))
                .await
                .unwrap(),
            [id]
        );
        let error = within(|| collecting).await.unwrap().unwrap_err();
        assert_eq!(error.error_class(), Some("OPERATION_CANCELED"), "{error}");
        while running.try_recv().is_ok() {}

        let dropped = session.with_tag("b7-dropped").unwrap();
        let query = within(|| dropped.sql(LONG)).await.unwrap();
        let collecting = tokio::spawn(query.collect());
        let id = within(|| running.recv()).await.unwrap();
        collecting.abort();
        assert!(collecting.await.unwrap_err().is_cancelled());
        released(&url, &id).await;
        assert!(within(|| session.interrupt_tag("b7-dropped"))
            .await
            .unwrap()
            .is_empty());
    })
    .await;
}

#[tokio::test]
async fn a_result_larger_than_a_message_or_than_a_stream_reads_whole() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, async {
        // One row of 5 MiB, more than a gRPC message of 4 MiB by default.
        let wide = within(|| session.sql("SELECT repeat('x', 5242880) AS s"))
            .await
            .unwrap();
        let rows: Vec<(String,)> = within(|| wide.collect_as()).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0.len(), 5_242_880);
        assert!(rows[0].0.bytes().all(|byte| byte == b'x'));

        // More than the server sends on one stream (CI's server ends each stream after 1 MiB),
        // so that the client reattaches to read the rest.
        let n: i64 = 300_000;
        let ids: Vec<(i64,)> = within(|| session.range(n).collect_as()).await.unwrap();
        assert_eq!(ids.len() as i64, n);
        assert_eq!(ids.iter().map(|(id,)| id).sum::<i64>(), n * (n - 1) / 2);
        let mut stream = within(|| session.range(n).stream()).await.unwrap();
        let mut read = 0;
        while let Some(batch) = within(|| stream.next()).await {
            read += batch.unwrap().num_rows();
        }
        assert_eq!(read as i64, n);
    })
    .await;
}

#[tokio::test]
async fn an_execution_is_released_once_its_stream_ends_or_is_dropped() {
    let Some(url) = url() else {
        return;
    };
    let session = within(|| Session::builder().remote(&url).connect())
        .await
        .unwrap();
    cleaned_up(&session, async {
        let one = within(|| session.sql("SELECT 1")).await.unwrap();
        let mut ended = within(|| one.stream()).await.unwrap();
        while let Some(batch) = within(|| ended.next()).await {
            batch.unwrap();
        }
        released(&url, ended.operation_id()).await;

        let many = within(|| session.sql("SELECT id FROM range(0, 10000000, 1, 10)"))
            .await
            .unwrap();
        let mut dropped = within(|| many.stream()).await.unwrap();
        within(|| dropped.next()).await.unwrap().unwrap();
        let id = dropped.operation_id().to_string();
        drop(dropped);
        released(&url, &id).await;
    })
    .await;
}
