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

//! The Rust UDFs of the async API send the requests of `apache-spark-connect` for the same
//! registrations and calls, as the in-process server of its request snapshots records them, and
//! a registration is cancelled as every operation is.
//!
//! Both APIs build their UDFs with the Python of `SPARK_CONNECT_PYTHON`, which every test sets
//! first to a script that answers with the spec it is given, so the payload is the spec, and that
//! logs the entrypoint of each build.
#![cfg(all(feature = "rust-udf", unix))]

mod gated;
#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;
mod support;

use std::os::unix::fs::PermissionsExt;
use std::sync::{Arc, Once};

use futures::future::BoxFuture;
use futures::FutureExt;
use gated::{Received, Server, Wait};
use prost::Message;
use spark_connect::rust_udf as old;
use spark_connect::SparkSession;
use spark_connect_async::rust_udf::{udf, AbiType, UserDefinedFunction};
use spark_connect_async::{col, ErrorKind, Session};
use spark_connect_proto as proto;

const SESSION_ID: &str = "00000000-0000-4000-8000-0000000000d1";

fn scratch() -> &'static std::path::Path {
    std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
}

/// The script that answers with the spec it is given, after it logs the spec's entrypoint to
/// `BUILDS`. It fails for the entrypoint `failing`, and for an entrypoint `e` while the file
/// `e.fail` is beside it. A run without a spec is the version check, which it answers with
/// nothing.
const ECHO: &str = r#"#!/bin/sh
spec=$(cat)
[ -n "$spec" ] || exit 0
dir=$(dirname "$0")
entrypoint=$(printf '%s' "$spec" | sed -n 's/.*"entrypoint":"\([^"]*\)".*/\1/p')
echo "$entrypoint" >> "$dir/rust-udf-requests-builds"
if [ "$entrypoint" = failing ] || [ -e "$dir/$entrypoint.fail" ]; then
  echo 'the packer failed' >&2
  exit 1
fi
printf '%s' "$spec"
"#;

/// Points `SPARK_CONNECT_PYTHON` at `ECHO`, with an empty log. Every test calls it before anything
/// else, so no other thread reads the environment while it is set.
fn echo_python() {
    static SET: Once = Once::new();
    SET.call_once(|| {
        let path = scratch().join("rust-udf-requests-python");
        let staged = scratch().join(format!("rust-udf-requests-python.{}", std::process::id()));
        std::fs::write(&staged, ECHO).unwrap();
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::rename(&staged, &path).unwrap();
        std::fs::write(scratch().join("rust-udf-requests-builds"), "").unwrap();
        std::env::set_var("SPARK_CONNECT_PYTHON", &path);
    });
}

/// The builds of UDFs whose entrypoint is `entrypoint`, as `ECHO` logged them.
fn builds(entrypoint: &str) -> usize {
    let log = std::fs::read_to_string(scratch().join("rust-udf-requests-builds")).unwrap();
    log.lines().filter(|line| *line == entrypoint).count()
}

const MODULE: [u8; 4] = [0, 97, 115, 109];

fn arg_types() -> Vec<AbiType> {
    vec![AbiType::I64, AbiType::Nullable(Box::new(AbiType::Str))]
}

fn ret_type() -> AbiType {
    AbiType::Array(Box::new(AbiType::F64))
}

fn old_label() -> old::UserDefinedFunction {
    old::udf("label", MODULE, "label", arg_types(), ret_type())
}

fn label() -> UserDefinedFunction {
    udf("label", MODULE, "label", arg_types(), ret_type())
}

fn old_plus() -> old::UserDefinedFunction {
    let types = vec![AbiType::I64, AbiType::I64];
    old::udf("plus", MODULE, "plus", types, AbiType::I64).with_batch_entrypoint("plus_batch")
}

fn plus() -> UserDefinedFunction {
    let types = vec![AbiType::I64, AbiType::I64];
    udf("plus", MODULE, "plus", types, AbiType::I64).with_batch_entrypoint("plus_batch")
}

struct Case {
    name: &'static str,
    old: fn(&SparkSession),
    new: fn(Session) -> BoxFuture<'static, ()>,
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
            "register a UDF called per row",
            |s| s.udf().register("label", &old_label()).unwrap(),
            |s| s.udf().register("label", &label()).await.unwrap()
        ),
        case!(
            "register a UDF called per batch",
            |s| s.udf().register("plus", &old_plus()).unwrap(),
            |s| s.udf().register("plus", &plus()).await.unwrap()
        ),
        case!(
            "register a nondeterministic UDF",
            |s| s
                .udf()
                .register("label", &old_label().as_nondeterministic())
                .unwrap(),
            |s| s
                .udf()
                .register("label", &label().as_nondeterministic())
                .await
                .unwrap()
        ),
        case!(
            "call a UDF called per row on column names",
            |s| s
                .range(3)
                .unwrap()
                .select([old_label()
                    .call([spark_connect::col("id"), spark_connect::col("s")])
                    .unwrap()])
                .collect_record_batches()
                .unwrap(),
            |s| s
                .range(3)
                .select([label().call(["id", "s"]).unwrap()])
                .collect()
                .await
                .unwrap()
        ),
        case!(
            "call a UDF called per batch on columns",
            |s| s
                .range(3)
                .unwrap()
                .select([old_plus()
                    .call([spark_connect::col("id"), spark_connect::col("id")])
                    .unwrap()])
                .collect_record_batches()
                .unwrap(),
            |s| s
                .range(3)
                .select([plus().call([col("id"), col("id")]).unwrap()])
                .collect()
                .await
                .unwrap()
        ),
        case!(
            "call one UDF twice and register it",
            |s| {
                let label = old_label();
                let columns = || [spark_connect::col("id"), spark_connect::col("s")];
                for _ in 0..2 {
                    let column = label.call(columns()).unwrap();
                    s.range(3)
                        .unwrap()
                        .select([column])
                        .collect_record_batches()
                        .unwrap();
                }
                s.udf().register("label", &label).unwrap()
            },
            |s| {
                let label = label();
                for _ in 0..2 {
                    let column = label.call(["id", "s"]).unwrap();
                    s.range(3).select([column]).collect().await.unwrap();
                }
                s.udf().register("label", &label).await.unwrap()
            }
        ),
        case!(
            "call a nondeterministic UDF",
            |s| s
                .range(3)
                .unwrap()
                .select([old_plus()
                    .as_nondeterministic()
                    .call([spark_connect::col("id"), spark_connect::col("id")])
                    .unwrap()])
                .collect_record_batches()
                .unwrap(),
            |s| s
                .range(3)
                .select([plus().as_nondeterministic().call(["id", "id"]).unwrap()])
                .collect()
                .await
                .unwrap()
        ),
    ]
}

/// Each entry as its message, without the operation id, which the client chooses at random.
fn messages(entries: Vec<server::Entry>) -> Vec<String> {
    entries
        .into_iter()
        .map(|entry| match entry.kind.as_str() {
            "ExecutePlanRequest" => {
                let mut request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
                request.operation_id = None;
                format!("{request:?}")
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
async fn rust_udfs_send_the_requests_of_the_old_api() {
    echo_python();
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
async fn every_registration_and_call_sends_the_tags_of_its_handle() {
    echo_python();
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let tagged = session.with_tag("udfs").unwrap();
    for case in cases() {
        (case.new)(tagged.clone()).await;
        let mut sent = 0;
        for entry in recorder.take() {
            if entry.kind != "ExecutePlanRequest" {
                continue;
            }
            sent += 1;
            let request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
            assert_eq!(request.tags, ["udfs"], "{}", case.name);
        }
        assert!(sent > 0, "{}: sent nothing", case.name);
    }
}

#[tokio::test]
async fn a_registration_whose_udf_cannot_be_built_fails_and_sends_nothing() {
    echo_python();
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let array = vec![AbiType::Array(Box::new(AbiType::I64))];
    for (name, udf, kind, message) in [
        (
            "Python fails",
            udf("f", MODULE, "failing", arg_types(), ret_type()),
            ErrorKind::Internal,
            "the packer failed",
        ),
        (
            "an array per batch",
            udf("f", MODULE, "f", array, AbiType::I64).with_batch_entrypoint("f_batch"),
            ErrorKind::InvalidArgument,
            "so it can't be called once per batch",
        ),
    ] {
        recorder.take();
        let error = session.udf().register("f", &udf).await.unwrap_err();
        assert_eq!(error.kind(), kind, "{name}");
        assert!(error.message().contains(message), "{name}: {error}");
        assert!(recorder.take().is_empty(), "{name}: sent");
    }
}

#[tokio::test]
async fn a_udf_value_builds_its_payload_once_for_its_calls_and_registrations() {
    echo_python();
    let (_, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let types = vec![AbiType::I64, AbiType::I64];
    let counted = udf("counted", MODULE, "counted", types, AbiType::I64);
    assert_eq!(builds("counted"), 0, "built when created");
    for _ in 0..3 {
        counted.call(["a", "b"]).unwrap();
    }
    counted.clone().call(["a", "b"]).unwrap();
    let nondeterministic = counted.clone().as_nondeterministic();
    nondeterministic.call(["a", "b"]).unwrap();
    session.udf().register("counted", &counted).await.unwrap();
    session
        .udf()
        .register("n", &nondeterministic)
        .await
        .unwrap();
    assert_eq!(
        builds("counted"),
        1,
        "the calls and registrations of a value"
    );
    let batched = counted.with_batch_entrypoint("counted_batch");
    session.udf().register("b", &batched).await.unwrap();
    batched.call(["a", "b"]).unwrap();
    assert_eq!(builds("counted"), 2, "a value with a batch entrypoint");
}

#[test]
fn a_failed_build_is_tried_again_at_the_next_call() {
    echo_python();
    let fail = scratch().join("flaky.fail");
    std::fs::write(&fail, "").unwrap();
    let flaky = udf("flaky", MODULE, "flaky", arg_types(), ret_type());
    let error = flaky.call(["a", "b"]).unwrap_err();
    std::fs::remove_file(&fail).unwrap();
    assert_eq!(error.kind(), ErrorKind::Internal, "{error}");
    flaky.call(["a", "b"]).unwrap();
    flaky.call(["a", "b"]).unwrap();
    assert_eq!(builds("flaky"), 2);
}

/// Starts a registration once the gated server waits at `wait`, and returns its task once the
/// server has received it, with its operation id.
async fn a_registration_sent(server: &Server, wait: Wait) -> (tokio::task::JoinHandle<()>, String) {
    let session = Session::builder()
        .remote(server.url())
        .connect()
        .await
        .unwrap();
    server.wait_at(wait);
    let registering = tokio::spawn(
        session
            .udf()
            .register("label", &label())
            .map(|result| result.unwrap()),
    );
    assert!(server.wait_for(|r| matches!(r, Received::Execute(_))).await);
    (registering, server.operation_id())
}

async fn a_registration_dropped_before_the_server_accepts_it(server: &Server) {
    let (registering, operation_id) = a_registration_sent(server, Wait::BeforeAccepting).await;
    registering.abort();
    assert!(registering.await.unwrap_err().is_cancelled());
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::Interrupt(id.clone()))
            .await,
        "not interrupted"
    );
    server.open_gate();
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::ReleaseAll(id.clone()))
            .await,
        "not released"
    );
}

async fn a_registration_dropped_after_the_server_accepted_it(server: &Server) {
    let (registering, operation_id) = a_registration_sent(server, Wait::AfterTheFirstBatch).await;
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::ReleaseUntil(id.clone()))
            .await,
        "the first response not read"
    );
    registering.abort();
    assert!(registering.await.unwrap_err().is_cancelled());
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::ReleaseAll(id.clone()))
            .await,
        "not released"
    );
    let interrupt = Received::Interrupt(operation_id);
    assert!(!server.received().contains(&interrupt), "interrupted");
    server.open_gate();
}

#[test]
fn dropping_a_registration_stops_it_in_either_runtime() {
    echo_python();
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
                    a_registration_dropped_after_the_server_accepted_it(&server).await;
                } else {
                    a_registration_dropped_before_the_server_accepts_it(&server).await;
                }
            }
        });
    }
}

/// The blocking registrations of the blocking API.
#[cfg(feature = "blocking")]
mod blocking_registrations {
    use spark_connect_async::blocking;
    use tokio::runtime::{Builder, Runtime};

    use super::*;

    fn connect(url: &str) -> blocking::Session {
        Session::builder().remote(url).blocking().connect().unwrap()
    }

    type BlockingCase = (
        &'static str,
        fn(&blocking::Session),
        fn(Session) -> BoxFuture<'static, ()>,
    );

    #[test]
    fn a_blocking_registration_sends_the_requests_of_the_async_one() {
        echo_python();
        let (recorder, url) = start();
        let runtime = Runtime::new().unwrap();
        let cases: [BlockingCase; 4] = [
            (
                "register a UDF called per row",
                |b| {
                    b.with_tag("etl")
                        .unwrap()
                        .udf()
                        .register("label", &label())
                        .unwrap()
                },
                |a| {
                    Box::pin(async move {
                        let tagged = a.with_tag("etl").unwrap();
                        tagged.udf().register("label", &label()).await.unwrap()
                    })
                },
            ),
            (
                "register a UDF called per batch",
                |b| b.udf().register("plus", &plus()).unwrap(),
                |a| Box::pin(async move { a.udf().register("plus", &plus()).await.unwrap() }),
            ),
            (
                "register a nondeterministic UDF",
                |b| {
                    let label = label().as_nondeterministic();
                    b.udf().register("label", &label).unwrap()
                },
                |a| {
                    Box::pin(async move {
                        let label = label().as_nondeterministic();
                        a.udf().register("label", &label).await.unwrap()
                    })
                },
            ),
            (
                "an async registration converted both ways",
                |b| {
                    let registration = Session::from(b.clone()).udf().blocking();
                    let registration = registration.into_async().blocking();
                    registration.register("plus", &plus()).unwrap()
                },
                |a| Box::pin(async move { a.udf().register("plus", &plus()).await.unwrap() }),
            ),
        ];
        let mut both = Vec::new();
        for (name, blocking, awaited) in cases {
            blocking(&connect(&url));
            let sent = messages(recorder.take());
            runtime.block_on(async {
                awaited(
                    Session::builder()
                        .remote(url.clone())
                        .connect()
                        .await
                        .unwrap(),
                )
                .await
            });
            let awaited = messages(recorder.take());
            assert!(!awaited.is_empty(), "{name}: the async API sent nothing");
            both.push((name, awaited, sent));
        }
        let differences = support::differences(both);
        assert!(differences.is_empty(), "{differences}");
    }

    #[test]
    fn a_udf_value_builds_its_payload_once_for_both_apis() {
        echo_python();
        let (_, url) = start();
        let session = connect(&url);
        let runtime = Runtime::new().unwrap();
        let shared = udf("shared", MODULE, "shared", vec![AbiType::I64], AbiType::I64);
        session.udf().register("shared", &shared).unwrap();
        shared.call(["id"]).unwrap();
        runtime
            .block_on(session.as_async().udf().register("again", &shared))
            .unwrap();
        session.udf().register("twice", &shared.clone()).unwrap();
        assert_eq!(
            builds("shared"),
            1,
            "built first by a blocking registration"
        );

        let called = udf("called", MODULE, "called", vec![AbiType::I64], AbiType::I64);
        called.call(["id"]).unwrap();
        session.udf().register("called", &called).unwrap();
        assert_eq!(builds("called"), 1, "built first by a call");
    }

    /// Where a test makes a blocking call, as in blocking_calls.rs.
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

    /// What `call` returns, called in `context`, whose runtime has no I/O or time driver.
    fn run_in<T: Send + 'static>(context: Context, call: impl FnOnce() -> T + Send + 'static) -> T {
        let current_thread = || Builder::new_current_thread().build().unwrap();
        let multi_thread = || {
            Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .unwrap()
        };
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
            Context::MultiThreadSpawnBlocking => multi_thread()
                .block_on(async move { tokio::task::spawn_blocking(call).await.unwrap() }),
        }
    }

    #[test]
    fn a_blocking_registration_returns_in_every_context() {
        echo_python();
        let (recorder, url) = start();
        let registered = udf("contexts", MODULE, "contexts", arg_types(), ret_type());
        for context in CONTEXTS {
            let (url, registered) = (url.clone(), registered.clone());
            run_in(context, move || {
                connect(&url)
                    .udf()
                    .register("contexts", &registered)
                    .unwrap()
            });
        }
        let registrations = messages(recorder.take())
            .iter()
            .filter(|message| message.contains("RegisterFunction"))
            .count();
        assert_eq!(registrations, CONTEXTS.len());
        assert_eq!(builds("contexts"), 1);
    }
}
