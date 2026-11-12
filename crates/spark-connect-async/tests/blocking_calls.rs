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
//! the module.

#![cfg(feature = "blocking")]

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;

use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::Duration;

use prost::Message;
use spark_connect_async::actions::Checkpoint;
use spark_connect_async::readwriter::SaveMode;
use spark_connect_async::{blocking, Session};
use spark_connect_proto as proto;
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
    let three = session.sql(server::THREE_BATCHES).run().unwrap();
    let count = session.range(3).blocking().count().unwrap();
    let rows = three.collect_as::<(i64,)>().unwrap();
    let batches = three.stream().unwrap().map(Result::unwrap).count();
    let database = session.catalog().current_database().unwrap();
    three
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
    let count = connect(&url).range(3).blocking().count().unwrap();
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
    let three = session.sql(server::THREE_BATCHES).run().unwrap();
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
            session.sql(server::REPORTS_PROGRESS).run().unwrap();
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
        let count = session.range(3).blocking().count();
        assert_eq!(count.unwrap(), 3, "{context:?}");
    }
}
