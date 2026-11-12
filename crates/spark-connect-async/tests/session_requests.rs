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

//! The requests of the session API equal those of `apache-spark-connect` for the same calls,
//! as the in-process server of its request snapshots records them.

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;

use std::sync::Arc;

use futures::future::BoxFuture;
use prost::Message;
use spark_connect::session::SparkSession;
use spark_connect_async::session::ArtifactKind;
use spark_connect_async::Session;
use spark_connect_proto as proto;

const SESSION_ID: &str = "00000000-0000-4000-8000-00000000005e";

struct Case {
    name: &'static str,
    old: fn(&SparkSession),
    /// Connects to the URL of the old session itself, and makes the same calls.
    new: fn(String) -> BoxFuture<'static, ()>,
}

async fn connect(url: String) -> Session {
    Session::builder().remote(url).connect().await.unwrap()
}

/// A file named `name` in the scratch directory of the tests, the same for both APIs.
fn artifact(name: &str) -> String {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("session-requests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, name.as_bytes()).unwrap();
    path.to_str().unwrap().to_string()
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "conf set",
            old: |s| drop(s.conf().set("spark.a", "1").unwrap()),
            new: |url| {
                Box::pin(async move {
                    connect(url).await.conf().set("spark.a", "1").await.unwrap();
                })
            },
        },
        Case {
            name: "conf set_all",
            old: |s| {
                let pairs = [("a", "1"), ("b", "2")].map(|(k, v)| (k.to_string(), v.to_string()));
                drop(s.conf().set_all(&pairs, false).unwrap())
            },
            new: |url| {
                Box::pin(async move {
                    let conf = connect(url).await.conf();
                    conf.set_all([("a", "1"), ("b", "2")]).await.unwrap();
                })
            },
        },
        Case {
            name: "conf get",
            old: |s| drop(s.conf().get_with_default("spark.a", None).unwrap()),
            new: |url| {
                Box::pin(async move {
                    connect(url).await.conf().get("spark.a").await.unwrap();
                })
            },
        },
        Case {
            name: "conf get_all",
            old: |s| drop(s.conf().get_all().unwrap()),
            new: |url| {
                Box::pin(async move {
                    connect(url).await.conf().get_all().await.unwrap();
                })
            },
        },
        Case {
            name: "conf unset",
            old: |s| drop(s.conf().unset("spark.a").unwrap()),
            new: |url| {
                Box::pin(async move {
                    connect(url).await.conf().unset("spark.a").await.unwrap();
                })
            },
        },
        Case {
            name: "conf is_modifiable",
            old: |s| {
                s.conf().is_modifiable("spark.a").unwrap();
            },
            new: |url| {
                Box::pin(async move {
                    connect(url)
                        .await
                        .conf()
                        .is_modifiable("spark.a")
                        .await
                        .unwrap();
                })
            },
        },
        Case {
            name: "builder config",
            old: |s| {
                let pairs = [("spark.a", "1")].map(|(k, v)| (k.to_string(), v.to_string()));
                drop(s.conf().set_all(&pairs, false).unwrap())
            },
            new: |url| {
                Box::pin(async move {
                    let builder = Session::builder().remote(url).config("spark.a", "1");
                    builder.connect().await.unwrap();
                })
            },
        },
        Case {
            name: "interrupt_all",
            old: |s| drop(s.interrupt_all().unwrap()),
            new: |url| {
                Box::pin(async move {
                    connect(url).await.interrupt_all().await.unwrap();
                })
            },
        },
        Case {
            name: "interrupt_tag",
            old: |s| drop(s.interrupt_tag("etl").unwrap()),
            new: |url| {
                Box::pin(async move {
                    connect(url).await.interrupt_tag("etl").await.unwrap();
                })
            },
        },
        Case {
            name: "interrupt_operation",
            old: |s| drop(s.interrupt_operation("op-1").unwrap()),
            new: |url| {
                Box::pin(async move {
                    connect(url)
                        .await
                        .interrupt_operation("op-1")
                        .await
                        .unwrap();
                })
            },
        },
        Case {
            name: "add_artifacts classpath",
            old: |s| s.add_artifacts(&[&artifact("a.jar")]).unwrap(),
            new: |url| {
                Box::pin(async move {
                    let session = connect(url).await;
                    session.add_artifacts([artifact("a.jar")]).await.unwrap();
                })
            },
        },
        Case {
            name: "add_artifacts file",
            old: |s| {
                s.add_artifacts_with(&[&artifact("b.csv")], false, false, true)
                    .unwrap()
            },
            new: |url| {
                Box::pin(async move {
                    let session = connect(url).await;
                    let upload = session.add_artifacts([artifact("b.csv")]);
                    upload.kind(ArtifactKind::File).await.unwrap();
                })
            },
        },
        Case {
            name: "add_artifacts archive",
            old: |s| {
                s.add_artifacts_with(&[&artifact("c.zip")], false, true, false)
                    .unwrap()
            },
            new: |url| {
                Box::pin(async move {
                    let session = connect(url).await;
                    let upload = session.add_artifacts([artifact("c.zip")]);
                    upload.kind(ArtifactKind::Archive).await.unwrap();
                })
            },
        },
        Case {
            name: "version, twice",
            old: |s| {
                s.version().unwrap();
                s.version().unwrap();
            },
            new: |url| {
                Box::pin(async move {
                    let session = connect(url).await;
                    session.version().await.unwrap();
                    session.version().await.unwrap();
                })
            },
        },
        Case {
            name: "stop, twice, then a call",
            old: |s| {
                s.stop().unwrap();
                s.stop().unwrap();
                s.interrupt_all().unwrap_err();
            },
            new: |url| {
                Box::pin(async move {
                    let session = connect(url).await;
                    session.stop().await.unwrap();
                    session.stop().await.unwrap();
                    let error = session.interrupt_all().await.unwrap_err();
                    assert_eq!(error.error_class(), Some("NO_ACTIVE_SESSION"));
                    assert!(session.is_stopped());
                })
            },
        },
    ]
}

#[tokio::test]
async fn the_session_api_sends_the_requests_of_the_old_api() {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let url = format!("sc://{address}/;session_id={SESSION_ID};user_id=snapshots");
    let mut differ = Vec::new();
    for case in cases() {
        let old = SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        (case.old)(&old);
        let old = recorder.take();
        (case.new)(url.clone()).await;
        let new = recorder.take();
        assert!(!old.is_empty(), "{}: the old API sent nothing", case.name);
        if !same_requests(&old, &new) {
            differ.push(format!("{}: old {old:?}\nnew {new:?}", case.name));
        }
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// Whether two recordings hold the same requests. prost encodes the entries of a map field in the
/// order of its `HashMap`, which differs from map to map, so encodings that differ may still be
/// the same message.
fn same_requests(old: &[server::Entry], new: &[server::Entry]) -> bool {
    old.len() == new.len()
        && old.iter().zip(new).all(|(old, new)| {
            old.kind == new.kind
                && (old.bytes == new.bytes || same_message(&old.kind, &old.bytes, &new.bytes))
        })
}

/// Whether two encodings of a request of `kind` decode to the same message.
fn same_message(kind: &str, old: &[u8], new: &[u8]) -> bool {
    fn decoded<M: Message + Default + PartialEq>(old: &[u8], new: &[u8]) -> bool {
        M::decode(old).unwrap() == M::decode(new).unwrap()
    }
    match kind {
        "ExecutePlanRequest" => decoded::<proto::ExecutePlanRequest>(old, new),
        "AnalyzePlanRequest" => decoded::<proto::AnalyzePlanRequest>(old, new),
        "ConfigRequest" => decoded::<proto::ConfigRequest>(old, new),
        "AddArtifactsRequest" => decoded::<proto::AddArtifactsRequest>(old, new),
        "ArtifactStatusesRequest" => decoded::<proto::ArtifactStatusesRequest>(old, new),
        "InterruptRequest" => decoded::<proto::InterruptRequest>(old, new),
        "ReattachExecuteRequest" => decoded::<proto::ReattachExecuteRequest>(old, new),
        "ReleaseSessionRequest" => decoded::<proto::ReleaseSessionRequest>(old, new),
        "CloneSessionRequest" => decoded::<proto::CloneSessionRequest>(old, new),
        "FetchErrorDetailsRequest" => decoded::<proto::FetchErrorDetailsRequest>(old, new),
        other => panic!("no message type for requests of kind {other}"),
    }
}

#[tokio::test]
async fn the_parameters_of_the_builder_are_those_of_the_connection_string() {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let url = format!("sc://{address}/;session_id={SESSION_ID};user_id=other;user_agent=agent");
    SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .unwrap()
        .version()
        .unwrap();
    let old = recorder.take();
    let session = Session::builder()
        .remote(format!("sc://{address}"))
        .session_id(SESSION_ID)
        .user_id("other")
        .user_agent("agent")
        .connect()
        .await
        .unwrap();
    session.version().await.unwrap();
    assert_eq!(session.session_id(), SESSION_ID);
    assert_eq!(old, recorder.take());
}

/// A request to read with one option, `key` set to `value`.
fn read_with_option(key: &str, value: &str) -> proto::ExecutePlanRequest {
    let source = proto::read::DataSource {
        format: Some("csv".to_string()),
        options: [(key.to_string(), value.to_string())].into(),
        ..Default::default()
    };
    let read = proto::Read {
        read_type: Some(proto::read::ReadType::DataSource(source)),
        ..Default::default()
    };
    proto::ExecutePlanRequest {
        plan: Some(proto::Plan {
            op_type: Some(proto::plan::OpType::Root(proto::Relation {
                rel_type: Some(proto::relation::RelType::Read(read)),
                ..Default::default()
            })),
        }),
        ..Default::default()
    }
}

#[test]
fn recorded_requests_compare_as_messages_whatever_the_order_of_their_map_entries() {
    // Protobuf merges the occurrences of a field, so requests encoded one after the other decode
    // as one request with all of their options: each order of the entries, without a HashMap.
    let recorded = |kind: &str, options: &[(&str, &str)]| {
        let bytes = options
            .iter()
            .flat_map(|(key, value)| read_with_option(key, value).encode_to_vec())
            .collect();
        vec![server::Entry {
            kind: kind.to_string(),
            bytes,
        }]
    };
    let old = recorded("ExecutePlanRequest", &[("a", "1"), ("b", "2")]);
    let reordered = recorded("ExecutePlanRequest", &[("b", "2"), ("a", "1")]);
    assert_ne!(old[0].bytes, reordered[0].bytes);
    let cases = [
        ("the options in another order", reordered, true),
        (
            "another value",
            recorded("ExecutePlanRequest", &[("a", "1"), ("b", "3")]),
            false,
        ),
        (
            "another kind",
            recorded("AnalyzePlanRequest", &[("a", "1"), ("b", "2")]),
            false,
        ),
        ("no request", Vec::new(), false),
    ];
    for (case, new, same) in cases {
        assert_eq!(same_requests(&old, &new), same, "{case}");
    }
}
