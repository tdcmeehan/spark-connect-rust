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

//! The session of the async API against a Spark Connect server at `SPARK_REMOTE`: its errors,
//! statements, configuration, interrupts, progress and end; each test passes without running
//! when it is not set.

use std::future::{Future, IntoFuture};
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::{FutureExt, StreamExt};
use spark_connect_async::progress::ProgressUpdate;
use spark_connect_async::{DataFrame, Error, ErrorKind, Result, Session};

/// The longest that a call to the server may take.
const WAIT: Duration = Duration::from_secs(120);

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
    Some(format!("{remote}/;session_id={id};user_id=e2e_session"))
}

async fn session() -> Option<Session> {
    let url = url()?;
    Some(
        within(|| Session::builder().remote(url).connect())
            .await
            .unwrap(),
    )
}

fn unique(prefix: &str) -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{prefix}_{}", now.as_nanos())
}

/// Runs `body`, then stops `session`, which ends what it runs and drops its views, also when
/// `body` panics.
async fn cleaned_up(session: &Session, body: impl Future<Output = ()>) {
    let outcome = AssertUnwindSafe(body).catch_unwind().await;
    let _ = tokio::time::timeout(WAIT, session.stop()).await;
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
}

/// The error of `result`, which must have failed.
fn error_of<T>(result: Result<T>) -> Error {
    match result {
        Ok(_) => panic!("no error"),
        Err(error) => error,
    }
}

/// A query that runs for minutes.
const LONG: &str = "SELECT count(*) FROM range(1000000) a CROSS JOIN range(1000000) b \
                    WHERE a.id + b.id = -1";

#[tokio::test]
async fn the_errors_of_the_server_carry_their_condition_sql_state_parameters_and_query_context() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, async {
        let parse = error_of(within(|| session.sql("SELEC 1")).await);
        assert_eq!(parse.kind(), ErrorKind::Parse, "{parse}");
        assert_eq!(parse.error_class(), Some("PARSE_SYNTAX_ERROR"));
        assert_eq!(parse.sql_state().as_deref(), Some("42601"));
        assert_eq!(
            parse.message_parameters(),
            [("error", "'SELEC'"), ("hint", "")]
        );
        assert_eq!(parse.message().matches("[PARSE_SYNTAX_ERROR]").count(), 1);
        let trace = parse.server_stack_trace().unwrap_or_default();
        assert!(
            trace.starts_with("org.apache.spark.sql.catalyst.parser.ParseException"),
            "{trace}"
        );

        // A query runs at its actions, so that its errors come from them.
        let divide = within(|| session.sql("SELECT 1/0 AS x")).await.unwrap();
        let arithmetic = error_of(within(|| divide.collect()).await);
        assert_eq!(arithmetic.kind(), ErrorKind::Execution, "{arithmetic}");
        assert_eq!(arithmetic.error_class(), Some("DIVIDE_BY_ZERO"));
        assert_eq!(arithmetic.sql_state().as_deref(), Some("22012"));
        assert_eq!(
            arithmetic.message_parameters(),
            [("config", "\"spark.sql.ansi.enabled\"")]
        );
        let contexts = arithmetic.query_contexts();
        assert_eq!(contexts.len(), 1, "{contexts:?}");
        let context = &contexts[0];
        assert_eq!(
            (
                &context.fragment[..],
                context.start_index,
                context.stop_index
            ),
            ("1/0", 7, 9)
        );
        assert!(
            context.summary.contains("== SQL (line 1, position 8) =="),
            "{}",
            context.summary
        );

        let missing = error_of(within(|| session.table("b15_missing").count()).await);
        assert_eq!(missing.kind(), ErrorKind::Analysis, "{missing}");
        assert_eq!(missing.error_class(), Some("TABLE_OR_VIEW_NOT_FOUND"));
        assert_eq!(missing.sql_state().as_deref(), Some("42P01"));
        assert_eq!(
            missing.message_parameters()[0],
            ("relationName", "`b15_missing`")
        );

        let conf = session.conf();
        let fixed = error_of(within(|| conf.set("spark.driver.memory", "1g")).await);
        assert_eq!(fixed.kind(), ErrorKind::Analysis, "{fixed}");
        assert_eq!(fixed.error_class(), Some("CANNOT_MODIFY_CONFIG"));
        assert_eq!(fixed.sql_state().as_deref(), Some("46110"));
        assert_eq!(
            fixed.message_parameters()[0],
            ("key", "\"spark.driver.memory\"")
        );
    })
    .await;
}

#[tokio::test]
async fn statements_run_when_awaited_and_queries_with_their_arguments_and_references() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, async {
        let catalog = session.catalog();
        let view = unique("b15_view");
        within(|| session.sql(format!("CREATE TEMPORARY VIEW {view} AS SELECT 1 AS id")))
            .await
            .unwrap();
        assert!(within(|| catalog.table_exists(&view)).await.unwrap());
        within(|| session.sql("SET spark.sql.shuffle.partitions = 7"))
            .await
            .unwrap();
        assert_eq!(
            within(|| session.conf().get("spark.sql.shuffle.partitions"))
                .await
                .unwrap()
                .as_deref(),
            Some("7")
        );

        let positional = within(|| session.sql("SELECT ? + 1 AS a, ? AS b").arg(41).arg("x"))
            .await
            .unwrap();
        let rows: Vec<(i32, String)> = within(|| positional.collect_as()).await.unwrap();
        assert_eq!(rows, [(42, "x".to_string())]);
        let named = within(|| {
            session
                .sql("SELECT :n * 2 AS a, :s AS b")
                .named_arg("n", 21)
                .named_arg("s", "y")
        })
        .await
        .unwrap();
        let rows: Vec<(i32, String)> = within(|| named.collect_as()).await.unwrap();
        assert_eq!(rows, [(42, "y".to_string())]);

        let ids = session.range(3);
        let name = unique("b15_ids");
        let joined: DataFrame = within(|| {
            session
                .sql(format!(
                    "SELECT count(*) FROM {name} JOIN {name} AS other USING (id)"
                ))
                .reference(&name, &ids)
        })
        .await
        .unwrap();
        let counts: Vec<(i64,)> = within(|| joined.collect_as()).await.unwrap();
        assert_eq!(counts, [(3,)]);
        let positive = within(|| {
            session
                .sql(format!("SELECT id FROM {name} WHERE id > 0 ORDER BY id"))
                .reference(&name, &ids)
        })
        .await
        .unwrap();
        let rows: Vec<(i64,)> = within(|| positive.collect_as()).await.unwrap();
        assert_eq!(rows, [(1,), (2,)]);
        assert!(!within(|| catalog.table_exists(&name)).await.unwrap());
    })
    .await;
}

#[tokio::test]
async fn the_configuration_version_and_interrupts_of_a_session_are_the_servers() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, async {
        assert!(within(|| session.version())
            .await
            .unwrap()
            .starts_with("4."));

        let conf = session.conf();
        let partitions = "spark.sql.shuffle.partitions";
        within(|| conf.set(partitions, "7")).await.unwrap();
        assert_eq!(
            within(|| conf.get(partitions)).await.unwrap().as_deref(),
            Some("7")
        );
        within(|| conf.set_all([("spark.b15.first", "1"), ("spark.b15.second", "2")]))
            .await
            .unwrap();
        let all = within(|| conf.get_all()).await.unwrap();
        assert_eq!(
            (
                all.get(partitions).map(String::as_str),
                all.get("spark.b15.first").map(String::as_str),
                all.get("spark.b15.second").map(String::as_str)
            ),
            (Some("7"), Some("1"), Some("2"))
        );
        within(|| conf.unset(partitions)).await.unwrap();
        assert_eq!(
            within(|| conf.get(partitions)).await.unwrap().as_deref(),
            Some("200")
        );
        assert_eq!(within(|| conf.get("spark.b15.never")).await.unwrap(), None);
        for (name, key, value) in [
            (
                "a configuration without a default",
                "spark.sql.streaming.checkpointLocation",
                None,
            ),
            (
                "a static configuration",
                "spark.app.name",
                Some("Spark Connect server"),
            ),
        ] {
            assert_eq!(
                within(|| conf.get(key)).await.unwrap().as_deref(),
                value,
                "{name}"
            );
        }
        assert!(within(|| conf.is_modifiable(partitions)).await.unwrap());
        assert!(!within(|| conf.is_modifiable("spark.driver.memory"))
            .await
            .unwrap());

        let long = within(|| session.sql(LONG)).await.unwrap();
        let mut running = within(|| long.stream()).await.unwrap();
        let id = running.operation_id().to_string();
        assert_eq!(
            within(|| session.interrupt_operation(&id)).await.unwrap(),
            std::slice::from_ref(&id)
        );
        let interrupted = within(|| running.next()).await.unwrap().unwrap_err();
        assert_eq!(interrupted.error_class(), Some("OPERATION_CANCELED"));
        assert_eq!(interrupted.sql_state().as_deref(), Some("HY008"));
        assert_eq!(interrupted.kind(), ErrorKind::Cancelled, "{interrupted}");
        assert!(within(|| session.interrupt_all()).await.unwrap().is_empty());
        assert!(within(|| session.interrupt_operation(&id))
            .await
            .unwrap()
            .is_empty());
    })
    .await;
}

#[tokio::test]
async fn the_progress_of_an_execution_reaches_the_handlers_of_the_session() {
    let Some(url) = url() else {
        return;
    };
    let updates: Arc<Mutex<Vec<ProgressUpdate>>> = Arc::default();
    let seen = Arc::clone(&updates);
    let session = within(|| {
        Session::builder()
            .remote(&url)
            .on_progress(move |update| seen.lock().unwrap().push(update.clone()))
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
        let slow = within(|| {
            session.sql(
                "SELECT id, reflect('java.lang.Thread', 'sleep', 200L) AS slept \
                 FROM range(0, 4, 1, 4)",
            )
        })
        .await
        .unwrap();
        let mut stream = within(|| slow.stream()).await.unwrap();
        let mut rows = 0;
        while let Some(batch) = within(|| stream.next()).await {
            rows += batch.unwrap().num_rows();
        }
        assert_eq!(rows, 4);

        // Each execution of the session reports to the handlers, the statement's as well.
        let updates = updates.lock().unwrap();
        let updates: Vec<&ProgressUpdate> = updates
            .iter()
            .filter(|update| update.operation_id == stream.operation_id())
            .collect();
        let (last, running) = updates.split_last().expect("no progress");
        assert!(last.done, "{last:?}");
        assert!(
            running.iter().any(
                |update| !update.done && update.stages.iter().any(|stage| stage.num_tasks == 4)
            ),
            "{updates:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn a_stopped_session_is_released_on_the_server_and_every_handle_fails() {
    let Some(url) = url() else {
        return;
    };
    let session = within(|| Session::builder().remote(&url).connect())
        .await
        .unwrap();
    let tagged = session.with_tag("b15").unwrap();
    assert_eq!(within(|| tagged.range(3).count()).await.unwrap(), 3);
    within(|| session.stop()).await.unwrap();
    assert!(session.is_stopped() && tagged.is_stopped());
    within(|| tagged.stop()).await.unwrap();
    let stopped = error_of(within(|| tagged.range(1).count()).await);
    assert_eq!(
        stopped.error_class(),
        Some("NO_ACTIVE_SESSION"),
        "{stopped}"
    );
    assert_eq!(stopped.sql_state().as_deref(), Some("08003"));
    assert_eq!(stopped.kind(), ErrorKind::Transport, "{stopped}");

    let again = within(|| Session::builder().remote(&url).connect()).await;
    let closed = match again {
        Err(error) => error,
        Ok(again) => {
            let error = error_of(within(|| again.range(1).count()).await);
            let _ = within(|| again.stop()).await;
            error
        }
    };
    assert_eq!(
        closed.error_class(),
        Some("INVALID_HANDLE.SESSION_CLOSED"),
        "{closed}"
    );
    assert_eq!(closed.kind(), ErrorKind::Transport, "{closed}");
}
