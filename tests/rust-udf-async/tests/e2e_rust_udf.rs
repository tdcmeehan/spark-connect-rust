// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Rust UDFs of the async client on a live Spark Connect server, whose Python workers need
//! `pyarrow` and `wasmtime`. Run with `SPARK_REMOTE` set; skipped otherwise. Each call to the
//! server has a deadline, and each test stops the sessions it opens, which removes the functions
//! it registered, also when it fails.

use std::future::{Future, IntoFuture};
use std::time::Duration;

use rust_udf_async_tests::udf;
use spark_connect_async::actions::ExplainMode;
use spark_connect_async::functions::call_udf;
use spark_connect_async::{DataFrame, Error, ErrorKind, Session};

const DEADLINE: Duration = Duration::from_secs(300);

/// Ten rows in one partition: `id`; `x`, the id, but NULL for 9; `y`, ten times the id, but NULL
/// for multiples of 3; `s`, the id as text, but NULL for 4; and `xs`, the array of the id and 1.
const ROWS: &str = "SELECT id, IF(id = 9, NULL, id) AS x, IF(id % 3 = 0, NULL, id * 10) AS y, \
                    IF(id = 4, NULL, CAST(id AS STRING)) AS s, array(id, 1L) AS xs \
                    FROM range(0, 10, 1, 1)";

/// Waits for `call` until the deadline. The call is boxed: unboxed, the futures of a test
/// overflowed the stack of its thread in a debug build.
async fn within<T>(call: impl IntoFuture<Output = T>) -> T {
    tokio::time::timeout(DEADLINE, Box::pin(call.into_future()))
        .await
        .expect("no answer before the deadline")
}

/// A session whose Arrow batches have 4 rows, so that the ten rows are 3 batches.
async fn session(remote: &str) -> Session {
    let session = within(Session::builder().remote(remote).connect())
        .await
        .unwrap();
    let conf = session.conf();
    within(conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "4"))
        .await
        .unwrap();
    session
}

async fn rows(session: &Session) -> DataFrame {
    within(session.sql(ROWS)).await.unwrap()
}

/// Runs `body`, then stops `sessions`, also when `body` panics.
async fn cleaned_up(sessions: &[&Session], body: impl Future<Output = ()> + Send + 'static) {
    let outcome = tokio::spawn(body).await;
    for session in sessions {
        let _ = tokio::time::timeout(DEADLINE, session.stop()).await;
    }
    if let Err(error) = outcome {
        std::panic::resume_unwind(error.into_panic());
    }
}

/// The error of running `query` on `session`, wherever the server reports it.
async fn failure(session: &Session, query: &str) -> Error {
    match within(session.sql(query)).await {
        Ok(df) => within(df.collect_rows()).await.unwrap_err(),
        Err(error) => error,
    }
}

#[tokio::test]
async fn udfs_run_once_per_batch_or_once_per_row_on_the_server() {
    let Ok(remote) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let session = session(&remote).await;
    cleaned_up(&[&session], once_per_batch_or_row(session.clone())).await;
}

async fn once_per_batch_or_row(session: Session) {
    let rows = rows(&session).await;
    let per_batch = rows.select([
        udf::minus("id", "y").unwrap(),
        udf::label("s", "id").unwrap(),
    ]);
    let results: Vec<(Option<i64>, String)> = within(per_batch.collect_as()).await.unwrap();
    let expected: Vec<(Option<i64>, String)> = (0..10)
        .map(|id: i64| {
            let y = (id % 3 != 0).then_some(id * 10);
            let s = if id == 4 {
                String::new()
            } else {
                id.to_string()
            };
            (y.map(|y| id - y), format!("{s}#{id}"))
        })
        .collect();
    assert_eq!(results, expected);
    let per_row = rows.select([udf::total("xs").unwrap()]);
    let totals: Vec<(i64,)> = within(per_row.collect_as()).await.unwrap();
    assert_eq!(totals, (1..11).map(|total| (total,)).collect::<Vec<_>>());

    // The plan's eval type: 250, a scalar Arrow UDF, is called once per Arrow batch, and 101, an
    // Arrow-optimized Python UDF, once per row.
    for (name, df, eval_type) in [
        ("per batch", &per_batch, "250"),
        ("per row", &per_row, "101"),
    ] {
        let plan = within(df.explain(ExplainMode::Simple)).await.unwrap();
        let evals: Vec<&str> = plan
            .lines()
            .filter(|line| line.contains("ArrowEvalPython"))
            .collect();
        let suffix = format!(", {eval_type}");
        assert!(!evals.is_empty(), "{name}: {plan}");
        assert!(
            evals.iter().all(|line| line.trim_end().ends_with(&suffix)),
            "{name}: {plan}"
        );
    }

    // The batch export reports a NULL for an argument that is not an `Option` with its row in its
    // batch: the 10th row is the 2nd of the 3rd batch.
    let null = rows.select([udf::minus("x", "y").unwrap()]);
    let error = within(null.collect_rows()).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Execution, "{error}");
    assert_eq!(error.error_class(), Some("PYTHON_EXCEPTION"), "{error}");
    assert_eq!(error.sql_state().as_deref(), Some("38000"), "{error}");
    let message = "WASM UDF 'spark_batch_udf_minus': argument `x` is null in row 1 of the batch";
    assert!(error.message().contains(message), "{error}");
    within(session.stop()).await.unwrap();
}

#[tokio::test]
async fn a_registered_udf_is_a_temporary_function_of_its_session_alone() {
    let Ok(remote) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let (session, other) = (session(&remote).await, session(&remote).await);
    let body = registered_on_its_session(session.clone(), other.clone());
    cleaned_up(&[&session, &other], body).await;
}

async fn registered_on_its_session(session: Session, other: Session) {
    let registration = session.udf();
    within(registration.register("rust_udf_f", &udf::label_udf()))
        .await
        .unwrap();
    let in_sql = within(session.sql(format!("SELECT rust_udf_f(s, id) FROM ({ROWS})")))
        .await
        .unwrap();
    let labels: Vec<(String,)> = within(in_sql.collect_as()).await.unwrap();
    let called = rows(&session)
        .await
        .select([call_udf("rust_udf_f", ["s", "id"])]);
    let by_name: Vec<(String,)> = within(called.collect_as()).await.unwrap();
    assert_eq!(labels, by_name);
    assert_eq!(labels[3..5], [("3#3".to_string(),), ("#4".to_string(),)]);

    // Registering another UDF under the name replaces it.
    within(registration.register("rust_udf_f", &udf::minus_udf()))
        .await
        .unwrap();
    let in_sql = within(session.sql(format!("SELECT rust_udf_f(id, y) FROM ({ROWS})")))
        .await
        .unwrap();
    let differences: Vec<(Option<i64>,)> = within(in_sql.collect_as()).await.unwrap();
    assert_eq!(differences[..3], [(None,), (Some(-9),), (Some(-18),)]);

    let error = failure(&other, "SELECT rust_udf_f(1, 2)").await;
    assert_eq!(error.kind(), ErrorKind::Analysis, "{error}");
    assert_eq!(error.error_class(), Some("UNRESOLVED_ROUTINE"), "{error}");
    assert_eq!(error.sql_state().as_deref(), Some("42883"), "{error}");
    within(other.stop()).await.unwrap();
    within(session.stop()).await.unwrap();
}
