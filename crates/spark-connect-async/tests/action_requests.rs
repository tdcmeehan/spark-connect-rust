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

//! The actions of the async API send the requests of `apache-spark-connect` for the same
//! actions, as the in-process server of its request snapshots records them; and they read its
//! results and report their progress.

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;
mod support;

use std::future::IntoFuture;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::ThreadId;

use futures::future::BoxFuture;
use futures::StreamExt;
use prost::Message;
use spark_connect::SparkSession;
use spark_connect_async::actions::{Checkpoint, ExplainMode, StorageLevel, TempView};
use spark_connect_async::progress::ProgressUpdate;
use spark_connect_async::{DataFrame, ErrorKind, Session, SessionBuilder};
use spark_connect_proto as proto;

const SESSION_ID: &str = "00000000-0000-4000-8000-00000000005e";

struct Case {
    name: &'static str,
    old: fn(&SparkSession),
    new: fn(Session) -> BoxFuture<'static, ()>,
}

fn old_three(s: &SparkSession) -> spark_connect::DataFrame {
    s.sql(server::THREE_BATCHES).unwrap()
}

async fn new_three(s: &Session) -> DataFrame {
    s.sql(server::THREE_BATCHES).await.unwrap()
}

fn old_level() -> proto::StorageLevel {
    proto::StorageLevel {
        use_disk: true,
        use_memory: false,
        use_off_heap: false,
        deserialized: false,
        replication: 2,
    }
}

macro_rules! case {
    ($name:expr, |$o:ident| $old:expr, |$n:ident| $new:expr) => {
        Case {
            name: $name,
            old: |$o| {
                let _ = $old;
            },
            new: |$n| {
                Box::pin(async move {
                    let _ = $new;
                })
            },
        }
    };
}

fn cases() -> Vec<Case> {
    vec![
        case!(
            "collect",
            |s| old_three(s).collect_record_batches().unwrap(),
            |s| new_three(&s).await.collect().await.unwrap()
        ),
        case!("collect_rows", |s| old_three(s).collect().unwrap(), |s| {
            new_three(&s).await.collect_rows().await.unwrap()
        }),
        case!(
            "stream",
            |s| old_three(s).collect_record_batches().unwrap(),
            |s| {
                let batches = new_three(&s).await.stream().await.unwrap();
                batches.collect::<Vec<_>>().await
            }
        ),
        case!("count", |s| s.range(3).unwrap().count().unwrap(), |s| s
            .range(3)
            .count()
            .await
            .unwrap()),
        case!(
            "show",
            |s| s.range(3).unwrap().show_string(20, 20, false).unwrap(),
            |s| s.range(3).show().to_string().await.unwrap()
        ),
        case!(
            "show of options",
            |s| s.range(3).unwrap().show_string(5, 0, true).unwrap(),
            |s| {
                s.range(3)
                    .show()
                    .rows(5)
                    .truncate(0)
                    .vertical()
                    .to_string()
                    .await
                    .unwrap()
            }
        ),
        case!("take", |s| s.range(3).unwrap().take(2).unwrap(), |s| s
            .range(3)
            .take(2)
            .await
            .unwrap()),
        case!("first", |s| s.range(3).unwrap().first().unwrap(), |s| s
            .range(3)
            .first()
            .await
            .unwrap()),
        case!(
            "is_empty",
            |s| s.range(3).unwrap().is_empty().unwrap(),
            |s| s.range(3).is_empty().await.unwrap()
        ),
        case!(
            "schema and columns",
            |s| {
                let df = s.range(3).unwrap();
                (df.schema().unwrap(), df.columns().unwrap())
            },
            |s| {
                let df = s.range(3);
                (df.schema().await.unwrap(), df.columns().await.unwrap())
            }
        ),
        case!(
            "explain",
            |s| {
                for mode in ["simple", "extended", "codegen", "cost", "formatted"] {
                    s.range(3).unwrap().explain_string(mode).unwrap();
                }
            },
            |s| {
                use ExplainMode::*;
                for mode in [Simple, Extended, Codegen, Cost, Formatted] {
                    s.range(3).explain(mode).await.unwrap();
                }
            }
        ),
        case!("cache", |s| s.range(3).unwrap().cache().unwrap(), |s| s
            .range(3)
            .cache()
            .await
            .unwrap()),
        case!(
            "persist",
            |s| s.range(3).unwrap().persist(old_level()).unwrap(),
            |s| { s.range(3).persist(StorageLevel::DISK_ONLY_2).await.unwrap() }
        ),
        case!(
            "unpersist",
            |s| s.range(3).unwrap().unpersist(true).unwrap(),
            |s| s.range(3).unpersist(true).await.unwrap()
        ),
        case!(
            "storage_level",
            |s| s.range(3).unwrap().storage_level().unwrap(),
            |s| s.range(3).storage_level().await.unwrap()
        ),
        case!(
            "temp views",
            |s| {
                let df = s.range(3).unwrap();
                df.create_temp_view("a").unwrap();
                df.create_or_replace_temp_view("b").unwrap();
                df.create_global_temp_view("c").unwrap();
                df.create_or_replace_global_temp_view("d").unwrap();
            },
            |s| {
                let df = s.range(3);
                df.create_temp_view("a", TempView::default()).await.unwrap();
                df.create_temp_view("b", TempView::default().or_replace())
                    .await
                    .unwrap();
                df.create_temp_view("c", TempView::default().global())
                    .await
                    .unwrap();
                df.create_temp_view("d", TempView::default().or_replace().global())
                    .await
                    .unwrap();
            }
        ),
        case!(
            "checkpoint",
            |s| std::mem::forget(s.range(3).unwrap().checkpoint().unwrap()),
            |s| { std::mem::forget(s.range(3).checkpoint(Checkpoint::default()).await.unwrap()) }
        ),
        case!(
            "lazy checkpoint",
            |s| std::mem::forget(s.range(3).unwrap().checkpoint_with(false).unwrap()),
            |s| {
                std::mem::forget(
                    s.range(3)
                        .checkpoint(Checkpoint::default().lazy())
                        .await
                        .unwrap(),
                )
            }
        ),
        case!(
            "local checkpoint",
            |s| {
                let df = s
                    .range(3)
                    .unwrap()
                    .local_checkpoint_with(false, Some(old_level()))
                    .unwrap();
                std::mem::forget(df)
            },
            |s| {
                let checkpoint = Checkpoint::default()
                    .local()
                    .lazy()
                    .storage_level(StorageLevel::DISK_ONLY_2);
                std::mem::forget(s.range(3).checkpoint(checkpoint).await.unwrap())
            }
        ),
        case!(
            "input_files",
            |s| s.range(3).unwrap().input_files().unwrap(),
            |s| s.range(3).input_files().await.unwrap()
        ),
        case!(
            "same_semantics",
            |s| s
                .range(3)
                .unwrap()
                .same_semantics(&s.range(4).unwrap())
                .unwrap(),
            |s| { s.range(3).same_semantics(&s.range(4)).await.unwrap() }
        ),
        case!(
            "semantic_hash",
            |s| s.range(3).unwrap().semantic_hash().unwrap(),
            |s| s.range(3).semantic_hash().await.unwrap()
        ),
        case!(
            "is_streaming",
            |s| s.range(3).unwrap().is_streaming().unwrap(),
            |s| s.range(3).is_streaming().await.unwrap()
        ),
        case!("to_json", |s| s.range(3).unwrap().to_json().unwrap(), |s| s
            .range(3)
            .to_json()
            .await
            .unwrap()),
        case!(
            "an action of a tagged session",
            |s| {
                s.add_tag("etl").unwrap();
                let rows = s.range(3).unwrap().collect().unwrap();
                s.clear_tags();
                rows
            },
            |s| s.with_tag("etl").unwrap().range(3).collect().await.unwrap()
        ),
    ]
}

/// Each entry as its message, with the operation ids, which the client chooses at random,
/// removed.
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

#[tokio::test]
async fn actions_send_the_requests_of_the_old_api() {
    let (recorder, url) = start();
    let mut both = Vec::new();
    for case in cases() {
        let old_session = SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        (case.old)(&old_session);
        let old = messages(recorder.take());
        let session = Session::builder()
            .remote(url.clone())
            .connect()
            .await
            .unwrap();
        (case.new)(session).await;
        let new = messages(recorder.take());
        assert!(!old.is_empty(), "{}: the old API sent nothing", case.name);
        both.push((case.name, old, new));
    }
    let differences = support::differences(both);
    assert!(differences.is_empty(), "{differences}");
}

#[tokio::test]
async fn results_read_into_rust_types_and_a_failure_ends_the_stream() {
    let (_, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let three = session.sql(server::THREE_BATCHES).await.unwrap();
    let ids: Vec<(i64,)> = three.collect_as().await.unwrap();
    assert_eq!(ids, [(0,), (1,), (2,)]);
    let streamed: Vec<i64> = three
        .stream_as::<i64>()
        .await
        .unwrap()
        .map(Result::unwrap)
        .collect()
        .await;
    assert_eq!(streamed, [0, 1, 2]);
    assert_eq!(session.range(3).count().await.unwrap(), 3);
    assert!(!session.range(3).is_empty().await.unwrap());
    assert_eq!(
        session.range(3).storage_level().await.unwrap(),
        StorageLevel::MEMORY_AND_DISK_DESER
    );

    let failing = session.sql(server::A_BATCH_THEN_A_FAILURE).await.unwrap();
    let mut batches = failing.stream().await.unwrap();
    assert_eq!(batches.next().await.unwrap().unwrap().num_rows(), 1);
    assert!(batches.schema().is_some());
    let error = batches.next().await.unwrap().unwrap_err();
    assert!(error.message().contains("the execution failed"), "{error}");
    assert!(batches.next().await.is_none());
}

#[tokio::test]
async fn a_dropped_checkpoint_removes_its_relation_from_the_server() {
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let checkpoint = session
        .range(3)
        .checkpoint(Checkpoint::default())
        .await
        .unwrap();
    let kept = checkpoint.limit(1);
    recorder.take();
    drop(checkpoint);
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(
        recorder.take().is_empty(),
        "removed while a DataFrame of it is alive"
    );
    drop(kept);
    let removed = |entry: &server::Entry| {
        entry.kind == "ExecutePlanRequest"
            && format!(
                "{:?}",
                proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap()
            )
            .contains("RemoveCachedRemoteRelationCommand")
    };
    let recorder_for_wait = Arc::clone(&recorder);
    let found = tokio::task::spawn_blocking(move || {
        recorder_for_wait.wait_for(std::time::Duration::from_secs(5), removed)
    })
    .await
    .unwrap();
    assert!(found);
}

/// A session of `builder` on the recording server, and the server's recorder.
async fn connected(builder: SessionBuilder) -> (Arc<server::Recorder>, Session) {
    let (recorder, url) = start();
    (recorder, builder.remote(url).connect().await.unwrap())
}

#[test]
fn progress_handlers_and_hooks_run_on_the_thread_that_polls_in_either_runtime() {
    for multi_thread in [false, true] {
        let mut runtime = if multi_thread {
            tokio::runtime::Builder::new_multi_thread()
        } else {
            tokio::runtime::Builder::new_current_thread()
        };
        let runtime = runtime.enable_all().build().unwrap();
        let threads: Arc<Mutex<Vec<(&str, ThreadId)>>> = Arc::default();
        let updates: Arc<Mutex<Vec<ProgressUpdate>>> = Arc::default();
        let (on_progress, on_hook, seen) = (
            Arc::clone(&threads),
            Arc::clone(&threads),
            Arc::clone(&updates),
        );
        let builder = Session::builder()
            .on_progress(move |update| {
                on_progress
                    .lock()
                    .unwrap()
                    .push(("progress", std::thread::current().id()));
                seen.lock().unwrap().push(update.clone());
            })
            .execute_plan_hook(move |mut request| {
                on_hook
                    .lock()
                    .unwrap()
                    .push(("hook", std::thread::current().id()));
                request.operation_id = Some("of the hook".to_string());
                Ok(request)
            });
        let (polling, sent) = std::thread::spawn(move || {
            let sent = runtime.block_on(async move {
                let (recorder, session) = connected(builder).await;
                session.sql(server::REPORTS_PROGRESS).await.unwrap();
                recorder.take()
            });
            (std::thread::current().id(), sent)
        })
        .join()
        .unwrap();
        let sent = proto::ExecutePlanRequest::decode(&sent[0].bytes[..]).unwrap();
        let threads = threads.lock().unwrap();
        assert_eq!(
            threads.iter().map(|(what, _)| *what).collect::<Vec<_>>(),
            ["hook", "progress", "progress"],
            "multi-thread {multi_thread}"
        );
        assert!(
            threads.iter().all(|(_, thread)| *thread == polling),
            "multi-thread {multi_thread}"
        );
        let updates = updates.lock().unwrap();
        assert_eq!(
            updates.iter().map(|update| update.done).collect::<Vec<_>>(),
            [false, true]
        );
        assert_eq!(updates[0].stages[0].num_tasks, 1);
        assert_eq!(updates[0].inflight_tasks, 1);
        assert!(!updates[0].operation_id.is_empty());
        assert_eq!(updates[0].operation_id, updates[1].operation_id);
        assert_eq!(sent.operation_id.as_ref(), Some(&updates[0].operation_id));
    }
}

/// Polls `future` on this thread until it is ready, also within another call of its own, as a
/// blocking API that a callback calls does.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    struct Unpark(std::thread::Thread);
    impl std::task::Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        std::thread::park();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_hook_may_run_an_execution_whose_hook_runs_one_but_no_deeper() {
    static SESSION: OnceLock<Session> = OnceLock::new();
    let nested: Arc<Mutex<Vec<Option<ErrorKind>>>> = Arc::default();
    let results = Arc::clone(&nested);
    let builder = Session::builder().execute_plan_hook(move |request| {
        for _ in 0..2 {
            let statement = SESSION.get().unwrap().sql("SELECT 1");
            let result = block_on(statement.into_future());
            results
                .lock()
                .unwrap()
                .push(result.err().map(|error| error.kind()));
        }
        Ok(request)
    });
    let (_, session) = connected(builder).await;
    SESSION.set(session.clone()).ok().unwrap();
    let outer = session.clone();
    tokio::task::spawn_blocking(move || {
        block_on(outer.sql("SELECT 1").into_future()).unwrap();
    })
    .await
    .unwrap();
    // The outer hook's first execution runs, and its hook's two fail; once one has failed,
    // the outer hook's second fails too, without running its hook.
    let internal = Some(ErrorKind::Internal);
    assert_eq!(
        *nested.lock().unwrap(),
        [internal, internal, None, internal]
    );
}

#[cfg(all(feature = "datafusion", feature = "polars"))]
#[tokio::test]
async fn a_result_converts_into_a_polars_or_a_datafusion_dataframe() {
    let (_, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let three = session.sql(server::THREE_BATCHES).await.unwrap();
    assert_eq!(three.to_polars().await.unwrap().shape(), (3, 1));
    let context = datafusion::prelude::SessionContext::new();
    assert_eq!(
        three
            .to_datafusion(&context)
            .await
            .unwrap()
            .count()
            .await
            .unwrap(),
        3
    );
}
