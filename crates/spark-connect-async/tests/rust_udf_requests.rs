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
//! first to a script that answers with the spec it is given, so the payload is the spec.
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

/// Points `SPARK_CONNECT_PYTHON` at a script that answers with the spec it is given, and fails
/// for the entrypoint `failing`. Every test calls it before anything else, so no other thread
/// reads the environment while it is set.
fn echo_python() {
    static SET: Once = Once::new();
    SET.call_once(|| {
        let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
        let path = dir.join("rust-udf-requests-python");
        let staged = dir.join(format!("rust-udf-requests-python.{}", std::process::id()));
        let script = "#!/bin/sh\nspec=$(cat)\ncase \"$spec\" in\n  \
                      *'\"entrypoint\":\"failing\"'*) echo 'the packer failed' >&2; exit 1 ;;\n\
                      esac\nprintf '%s' \"$spec\"\n";
        std::fs::write(&staged, script).unwrap();
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::rename(&staged, &path).unwrap();
        std::env::set_var("SPARK_CONNECT_PYTHON", &path);
    });
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
    let mut sent = 0;
    for case in cases() {
        (case.new)(tagged.clone()).await;
        for entry in recorder.take() {
            if entry.kind != "ExecutePlanRequest" {
                continue;
            }
            sent += 1;
            let request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
            assert_eq!(request.tags, ["udfs"], "{}", case.name);
        }
    }
    assert_eq!(sent, cases().len());
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
            "so it cannot be called once per batch",
        ),
    ] {
        recorder.take();
        let error = session.udf().register("f", &udf).await.unwrap_err();
        assert_eq!(error.kind(), kind, "{name}");
        assert!(error.message().contains(message), "{name}: {error}");
        assert!(recorder.take().is_empty(), "{name}: sent");
    }
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
