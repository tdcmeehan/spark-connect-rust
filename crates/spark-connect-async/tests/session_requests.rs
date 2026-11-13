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
            old: |s| drop(s.conf().get("spark.a").unwrap()),
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

#[tokio::test]
async fn a_get_answers_the_value_the_session_uses_and_none_where_spark_has_none() {
    let conf = connect(conf_server::start().await).await.conf();
    let cases = [
        ("a key that the session set", "spark.set", Ok(Some("7"))),
        (
            "a configuration with a default",
            "spark.defaulted",
            Ok(Some("200")),
        ),
        (
            "a configuration without a default",
            "spark.optional",
            Ok(None),
        ),
        ("a key that is no configuration", "spark.unknown", Ok(None)),
        (
            "another error",
            "spark.failing",
            Err(Some("INVALID_HANDLE.SESSION_CLOSED")),
        ),
    ];
    for (case, key, expected) in cases {
        let answer = conf.get(key).await;
        let answer = match &answer {
            Ok(value) => Ok(value.as_deref()),
            Err(error) => Err(error.error_class()),
        };
        assert_eq!(answer, expected, "{case}");
    }
}

/// A server whose `Config` answers reads of the keys of
/// `a_get_answers_the_value_the_session_uses_and_none_where_spark_has_none` as the 4.3.0 server
/// does: a `Get` answers the value set in the session, else the configuration's default, and
/// fails with `SQL_CONF_NOT_FOUND` for a key that is no configuration, while a `GetWithDefault`
/// without a default answers only the value set.
mod conf_server {
    use std::collections::HashMap;
    use std::convert::Infallible;
    use std::task::{Context, Poll};

    use futures::stream;
    use prost::Message;
    use spark_connect_core::bytes_codec::BytesCodec;
    use spark_connect_proto as proto;
    use tonic::body::Body;
    use tonic::codegen::{http, BoxFuture, Bytes, Service};
    use tonic::server::{Grpc, NamedService, UnaryService};
    use tonic::{Code, Request, Response, Status};

    /// Starts the server on the test's runtime, and returns its connection string.
    pub async fn start() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("sc://{}", listener.local_addr().unwrap());
        let incoming = stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(ConfServer)
                .serve_with_incoming(incoming),
        );
        url
    }

    #[derive(Clone)]
    struct ConfServer;

    impl NamedService for ConfServer {
        const NAME: &'static str = "spark.connect.SparkConnectService";
    }

    impl Service<http::Request<Body>> for ConfServer {
        type Response = http::Response<Body>;
        type Error = Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;

        fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, request: http::Request<Body>) -> Self::Future {
            let config = request.uri().path() == "/spark.connect.SparkConnectService/Config";
            Box::pin(async move {
                Ok(if config {
                    Grpc::new(BytesCodec).unary(Answers, request).await
                } else {
                    Status::unimplemented(request.uri().path().to_string()).into_http()
                })
            })
        }
    }

    struct Answers;

    impl UnaryService<Vec<u8>> for Answers {
        type Response = Vec<u8>;
        type Future = BoxFuture<Response<Vec<u8>>, Status>;

        fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
            let answer = proto::ConfigRequest::decode(&request.into_inner()[..])
                .map_err(|error| Status::invalid_argument(error.to_string()))
                .and_then(answer)
                .map(|response| Response::new(response.encode_to_vec()));
            Box::pin(async move { answer })
        }
    }

    fn answer(request: proto::ConfigRequest) -> Result<proto::ConfigResponse, Status> {
        use proto::config_request::operation::OpType;
        let (keys, with_default) = match request.operation.and_then(|operation| operation.op_type) {
            Some(OpType::Get(get)) => (get.keys, false),
            Some(OpType::GetWithDefault(get)) => {
                (get.pairs.into_iter().map(|pair| pair.key).collect(), true)
            }
            _ => return Err(Status::unimplemented("not a read of configurations")),
        };
        let mut pairs = Vec::new();
        for key in keys {
            let value = match (key.as_str(), with_default) {
                ("spark.set", _) => Some("7"),
                ("spark.defaulted", false) => Some("200"),
                ("spark.unknown", false) => {
                    return Err(server_error(
                        "org.apache.spark.SparkNoSuchElementException",
                        "SQL_CONF_NOT_FOUND",
                        "42K0I",
                        &format!(
                            "[SQL_CONF_NOT_FOUND] The SQL config \"{key}\" cannot be found. \
                             Please verify that the config exists. SQLSTATE: 42K0I"
                        ),
                    ))
                }
                ("spark.failing", _) => return Err(server_error(
                    "org.apache.spark.sql.SparkSQLException",
                    "INVALID_HANDLE.SESSION_CLOSED",
                    "08003",
                    "[INVALID_HANDLE.SESSION_CLOSED] The handle is invalid. Session was closed. \
                         SQLSTATE: 08003",
                )),
                _ => None,
            };
            pairs.push(proto::KeyValue {
                key,
                value: value.map(str::to_string),
            });
        }
        Ok(proto::ConfigResponse {
            session_id: request.session_id,
            pairs,
            ..Default::default()
        })
    }

    /// An error that the server raised: an internal status whose details hold its `ErrorInfo`.
    fn server_error(class: &str, condition: &str, sql_state: &str, message: &str) -> Status {
        let info = ErrorInfo {
            reason: class.to_string(),
            domain: "org.apache.spark".to_string(),
            metadata: HashMap::from([
                ("classes".to_string(), format!("[\"{class}\"]")),
                ("errorClass".to_string(), condition.to_string()),
                ("sqlState".to_string(), sql_state.to_string()),
            ]),
        };
        let status = RpcStatus {
            code: Code::Internal as i32,
            message: message.to_string(),
            details: vec![RpcAny {
                type_url: "type.googleapis.com/google.rpc.ErrorInfo".to_string(),
                value: info.encode_to_vec(),
            }],
        };
        Status::with_details(Code::Internal, message, Bytes::from(status.encode_to_vec()))
    }

    #[derive(Clone, PartialEq, Message)]
    struct RpcStatus {
        #[prost(int32, tag = "1")]
        code: i32,
        #[prost(string, tag = "2")]
        message: String,
        #[prost(message, repeated, tag = "3")]
        details: Vec<RpcAny>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct RpcAny {
        #[prost(string, tag = "1")]
        type_url: String,
        #[prost(bytes = "vec", tag = "2")]
        value: Vec<u8>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct ErrorInfo {
        #[prost(string, tag = "1")]
        reason: String,
        #[prost(string, tag = "2")]
        domain: String,
        #[prost(map = "string, string", tag = "3")]
        metadata: HashMap<String, String>,
    }
}

// Above the futures that hold no execution of the core inline, and below those that do.
const FUTURE_SIZE_BOUND: usize = 16 * 1024;

#[test]
fn the_future_of_version_is_at_most_16_kib() {
    fn size<R>(_: impl FnOnce(&Session) -> R) -> usize {
        std::mem::size_of::<R>()
    }
    let version = size(|s| s.version());
    assert!(version <= FUTURE_SIZE_BOUND, "{version}");
}
