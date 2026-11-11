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

//! The structured streaming of the async API against a Spark Connect server at
//! `SPARK_REMOTE`; each test passes without running when it is not set.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::{FutureExt, StreamExt};
use spark_connect_async::streaming::{OutputMode, QueryEvent, QueryEvents, Trigger};
use spark_connect_async::{ErrorKind, Session};
use tokio::time::timeout;

/// The longest that a step waits for the server.
const WAIT: Duration = Duration::from_secs(60);

async fn session() -> Option<Session> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    Some(Session::builder().remote(remote).connect().await.unwrap())
}

fn unique(prefix: &str) -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{prefix}_{}", now.as_nanos())
}

/// Runs `body`, then stops the queries of `session` and the session, also when `body` panics.
async fn cleaned_up(session: &Session, body: impl Future<Output = ()>) {
    let outcome = AssertUnwindSafe(body).catch_unwind().await;
    if let Ok(Ok(queries)) = timeout(WAIT, session.streams().active()).await {
        for query in queries {
            let _ = timeout(WAIT, query.stop()).await;
        }
    }
    let _ = timeout(WAIT, session.stop()).await;
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
}

/// The next event of the query `id` that `wanted` accepts, skipping the others.
async fn event_of(
    events: &mut QueryEvents,
    id: &str,
    wanted: fn(&QueryEvent) -> bool,
) -> QueryEvent {
    let found = async {
        while let Some(event) = events.next().await {
            let event = event.unwrap();
            if event.id() == id && wanted(&event) {
                return event;
            }
        }
        panic!("the events ended");
    };
    timeout(WAIT, found).await.expect("no such event")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_query_reports_its_progress_and_events_until_it_stops() {
    let Some(session) = session().await else {
        return;
    };
    let checkpoint = tempfile::tempdir().unwrap();
    let name = unique("b10_numbers");
    cleaned_up(&session, async {
        let streams = session.streams();
        let mut events = timeout(WAIT, streams.events()).await.unwrap().unwrap();
        let query = session
            .read_stream()
            .format("rate")
            .option("rowsPerSecond", 10)
            .load_without_path()
            .write_stream()
            .format("memory")
            .query_name(&name)
            .output_mode(OutputMode::Append)
            .trigger(Trigger::ProcessingTime("100 milliseconds".to_string()))
            .option("checkpointLocation", checkpoint.path().to_str().unwrap())
            .start_without_path()
            .await
            .unwrap();
        assert_eq!(query.name(), Some(name.as_str()));
        match event_of(&mut events, query.id(), |e| {
            matches!(e, QueryEvent::Started(_))
        })
        .await
        {
            QueryEvent::Started(started) => {
                assert_eq!(started.run_id, query.run_id());
                assert_eq!(started.name.as_deref(), Some(name.as_str()));
            }
            other => panic!("{other:?}"),
        }
        assert!(query.is_active().await.unwrap());
        let found = streams.get(query.id()).await.unwrap().unwrap();
        assert_eq!(found.run_id(), query.run_id());
        let active = streams.active().await.unwrap();
        assert!(
            active.iter().any(|active| active.id() == query.id()),
            "{active:?}"
        );

        let progress = timeout(WAIT, async {
            loop {
                query.process_all_available().await.unwrap();
                if let Some(progress) = query.last_progress().await.unwrap() {
                    if progress.num_input_rows > 0 {
                        return progress;
                    }
                }
            }
        })
        .await
        .expect("no batch read rows");
        assert_eq!(progress.id, query.id());
        assert_eq!(progress.run_id, query.run_id());
        assert_eq!(progress.name.as_deref(), Some(name.as_str()));
        assert!(progress.batch_id >= 1, "{progress:?}");
        assert_eq!(progress.sources.len(), 1, "{progress:?}");
        assert!(
            progress.sources[0].description.contains("RateStream"),
            "{progress:?}"
        );
        assert!(progress.sources[0].end_offset.is_some(), "{progress:?}");
        assert_eq!(progress.sink.description, "MemorySink");
        assert!(
            progress.durations.contains_key("triggerExecution"),
            "{progress:?}"
        );
        let recent = query.recent_progress().await.unwrap();
        assert!(recent
            .iter()
            .any(|recent| recent.batch_id == progress.batch_id));
        let status = query.status().await.unwrap();
        assert!(status.is_active, "{status:?}");
        assert!(query.explain().await.unwrap().contains("Physical Plan"));
        assert!(query
            .explain()
            .extended()
            .await
            .unwrap()
            .contains("Logical Plan"));
        assert!(session.table(&name).count().await.unwrap() > 0);
        match event_of(&mut events, query.id(), |e| {
            matches!(e, QueryEvent::Progress(_))
        })
        .await
        {
            QueryEvent::Progress(event) => assert_eq!(event.run_id, query.run_id()),
            other => panic!("{other:?}"),
        }

        query.stop().await.unwrap();
        assert!(query
            .await_termination(Some(Duration::from_secs(10)))
            .await
            .unwrap());
        match event_of(&mut events, query.id(), |e| {
            matches!(e, QueryEvent::Terminated(_))
        })
        .await
        {
            QueryEvent::Terminated(terminated) => {
                assert_eq!(terminated.exception, None);
                assert_eq!(terminated.error_class, None);
            }
            other => panic!("{other:?}"),
        }
        assert!(!query.is_active().await.unwrap());
        assert!(query.exception().await.unwrap().is_none());

        // The server holds a subscription that was released but not removed, and then answers
        // no other: this one is acknowledged only if the dropped stream removed its own.
        drop(events);
        let again = timeout(WAIT, streams.events())
            .await
            .expect("no subscription after the stream was dropped")
            .unwrap();
        drop(again);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_query_is_an_error_with_the_condition_and_sql_state_of_the_server() {
    let Some(session) = session().await else {
        return;
    };
    let checkpoint = tempfile::tempdir().unwrap();
    let view = unique("b10_rate");
    cleaned_up(&session, async {
        let mut events = timeout(WAIT, session.streams().events())
            .await
            .unwrap()
            .unwrap();
        session
            .read_stream()
            .format("rate")
            .load_without_path()
            .create_temp_view(&view, Default::default())
            .await
            .unwrap();
        let query = session
            .sql(format!(
                "SELECT CAST(raise_error('b10 failure') AS INT) AS x FROM {view}"
            ))
            .await
            .unwrap()
            .write_stream()
            .format("memory")
            .query_name(format!("{view}_failed"))
            .option("checkpointLocation", checkpoint.path().to_str().unwrap())
            .start_without_path()
            .await
            .unwrap();

        let error = query.await_termination(Some(WAIT)).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Execution, "{error}");
        assert_eq!(error.error_class(), Some("STREAM_FAILED"), "{error}");
        assert_eq!(error.sql_state().as_deref(), Some("XXKST"), "{error}");
        assert!(error.message().contains("b10 failure"), "{error}");
        let error = query.process_all_available().await.unwrap_err();
        assert_eq!(error.error_class(), Some("STREAM_FAILED"), "{error}");

        let exception = query.exception().await.unwrap().expect("no exception");
        assert_eq!(exception.kind(), ErrorKind::Execution);
        assert_eq!(exception.error_class(), Some("STREAM_FAILED"));
        assert!(
            exception
                .message()
                .starts_with("[USER_RAISED_EXCEPTION] b10 failure"),
            "{exception}"
        );
        assert!(!query.status().await.unwrap().is_active);
        match event_of(&mut events, query.id(), |e| {
            matches!(e, QueryEvent::Terminated(_))
        })
        .await
        {
            QueryEvent::Terminated(terminated) => {
                let exception = terminated.exception.expect("no exception in the event");
                assert!(exception.contains("b10 failure"), "{exception}");
                assert!(terminated.error_class.is_some());
            }
            other => panic!("{other:?}"),
        }
        session.streams().reset_terminated().await.unwrap();
        assert!(!session
            .streams()
            .await_any_termination(Some(Duration::from_millis(10)))
            .await
            .unwrap());
    })
    .await;
}
