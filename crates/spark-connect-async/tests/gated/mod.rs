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

//! An in-process Spark Connect server whose executions wait at a gate that the test opens, so
//! that a test can drop a client future at a chosen point of an execution.
//!
//! Its listener bus acts as Spark 4.3.0's: it acknowledges a subscription and keeps its stream
//! open until a removal, and it never answers a subscription while it holds another.

// Each test binary uses a part of the server.
#![allow(dead_code)]

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use futures::channel::mpsc;
use futures::{stream, Stream};
use prost::Message;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_proto as proto;
use tokio::sync::Semaphore;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{Grpc, NamedService, ServerStreamingService, UnaryService};
use tonic::{Request, Response, Status};

use proto::execute_plan_response::ResponseType;

/// Where an execution waits for the gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    /// Before the server answers the ExecutePlan call, so before the client has a stream.
    BeforeAccepting,
    /// After the first batch, before the rest of the result.
    AfterTheFirstBatch,
    /// Nowhere: the whole result at once.
    Never,
    /// The first response ends after a batch without the end of the result, so the client
    /// reattaches; the server waits before it answers the reattach.
    BeforeReattaching,
}

/// Where the listener bus waits for its gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BusWait {
    /// Nowhere.
    Never,
    /// Before it answers a subscription, so before the client has a stream.
    BeforeAcknowledging,
    /// After a first response to a subscription, a progress of one task rather than its
    /// acknowledgement, before the rest: the client has accepted the stream, but not read the
    /// acknowledgement.
    AfterAccepting,
    /// Before it removes the subscription and answers the removal.
    BeforeRemoving,
}

/// What the server received.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Received {
    Execute(String),
    Subscribe(String),
    Unsubscribe(String),
    Reattach(String),
    Interrupt(String),
    ReleaseAll(String),
    ReleaseUntil(String),
    ReleaseSession,
    Config,
}

pub struct Server {
    pub address: std::net::SocketAddr,
    state: Arc<State>,
}

struct State {
    wait: Mutex<Wait>,
    gate: Semaphore,
    received: Mutex<Vec<Received>>,
    bus_wait: Mutex<BusWait>,
    bus_gate: Semaphore,
    subscription: Mutex<Option<Subscription>>,
}

/// The subscription that the listener bus holds, and the stream of its responses.
struct Subscription {
    sender: mpsc::UnboundedSender<Result<Vec<u8>, Status>>,
    session_id: String,
    operation_id: String,
    next: usize,
}

impl Subscription {
    fn send(&mut self, response_type: proto::execute_plan_response::ResponseType) {
        let response = respond(
            &self.session_id,
            &self.operation_id,
            self.next,
            response_type,
        );
        self.next += 1;
        let _ = self.sender.unbounded_send(Ok(response.encode_to_vec()));
    }
}

/// The id, run id and name of the queries that the server starts.
pub const QUERY_ID: &str = "00000000-0000-4000-8000-000000000001";
pub const QUERY_RUN_ID: &str = "00000000-0000-4000-8000-000000000002";

/// The JSON of the started event of a query named `name`.
pub fn started_json(name: &str) -> String {
    format!(
        r#"{{"id":"{QUERY_ID}","runId":"{QUERY_RUN_ID}","name":"{name}","timestamp":"2026-11-11T17:57:24.633Z","jobTags":[]}}"#
    )
}

/// The key of a configuration that the server rejects.
pub const REJECTED_CONFIG: &str = "spark.test.rejected";

impl Server {
    pub fn start() -> Server {
        let state = Arc::new(State {
            wait: Mutex::new(Wait::Never),
            gate: Semaphore::new(0),
            received: Mutex::new(Vec::new()),
            bus_wait: Mutex::new(BusWait::Never),
            bus_gate: Semaphore::new(0),
            subscription: Mutex::new(None),
        });
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let service = Gated(Arc::clone(&state));
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                let incoming = stream::unfold(listener, |listener| async {
                    Some((listener.accept().await.map(|(socket, _)| socket), listener))
                });
                tonic::transport::Server::builder()
                    .add_service(service)
                    .serve_with_incoming(incoming)
                    .await
                    .unwrap();
            });
        });
        Server { address, state }
    }

    pub fn url(&self) -> String {
        format!("sc://{}", self.address)
    }

    pub fn wait_at(&self, wait: Wait) {
        *self.state.wait.lock().unwrap() = wait;
    }

    pub fn open_gate(&self) {
        self.state.gate.add_permits(1);
    }

    pub fn received(&self) -> Vec<Received> {
        self.state.received.lock().unwrap().clone()
    }

    pub fn bus_wait_at(&self, wait: BusWait) {
        *self.state.bus_wait.lock().unwrap() = wait;
    }

    pub fn open_bus_gate(&self) {
        self.state.bus_gate.add_permits(1);
    }

    /// Holds the subscription of another client of the session, so that the bus answers no
    /// other until it is removed.
    pub fn hold_other_subscription(&self) {
        let (sender, _) = mpsc::unbounded();
        *self.state.subscription.lock().unwrap() = Some(Subscription {
            sender,
            session_id: String::new(),
            operation_id: "other".to_string(),
            next: 0,
        });
    }

    /// Whether the bus holds a subscription.
    pub fn subscribed(&self) -> bool {
        self.state.subscription.lock().unwrap().is_some()
    }

    /// Sends an event of the type `event_type` with the JSON `json` to the subscription.
    pub fn send_event(&self, event_type: proto::StreamingQueryEventType, json: &str) {
        let mut subscription = self.state.subscription.lock().unwrap();
        let subscription = subscription.as_mut().expect("no subscription");
        subscription.send(ResponseType::StreamingQueryListenerEventsResult(
            proto::StreamingQueryListenerEventsResult {
                events: vec![proto::StreamingQueryListenerEvent {
                    event_json: json.to_string(),
                    event_type: event_type as i32,
                }],
                listener_bus_listener_added: None,
            },
        ));
    }

    /// Ends the subscription as a removal by another client of the session does.
    pub fn end_subscription(&self) {
        let mut subscription = self
            .state
            .subscription
            .lock()
            .unwrap()
            .take()
            .expect("no subscription");
        subscription.send(ResponseType::ResultComplete(Default::default()));
    }

    /// Fails the stream of the subscription; the bus still holds the subscription.
    pub fn fail_subscription(&self) {
        let subscription = self.state.subscription.lock().unwrap();
        let subscription = subscription.as_ref().expect("no subscription");
        let _ = subscription
            .sender
            .unbounded_send(Err(Status::internal("the subscription failed")));
    }

    /// Waits up to five seconds for a request that `found` accepts, and returns whether one
    /// came.
    pub async fn wait_for(&self, found: impl Fn(&Received) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if self.received().iter().any(&found) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }

    /// The operation id of the last execution.
    pub fn operation_id(&self) -> String {
        self.received()
            .into_iter()
            .rev()
            .find_map(|received| match received {
                Received::Execute(operation_id) => Some(operation_id),
                _ => None,
            })
            .expect("no execution")
    }
}

impl State {
    fn receive(&self, received: Received) {
        self.received.lock().unwrap().push(received);
    }

    async fn pass_the_gate(&self) {
        self.gate.acquire().await.unwrap().forget();
    }

    async fn pass_the_bus_gate(&self) {
        self.bus_gate.acquire().await.unwrap().forget();
    }

    /// Holds a subscription, and returns its stream, which starts with the acknowledgement
    /// at `index`; or `None` while the bus holds another.
    fn register(&self, session_id: &str, operation_id: &str, index: usize) -> Option<ByteStream> {
        let mut held = self.subscription.lock().unwrap();
        if held.is_some() {
            return None;
        }
        let (sender, receiver) = mpsc::unbounded();
        let mut subscription = Subscription {
            sender,
            session_id: session_id.to_string(),
            operation_id: operation_id.to_string(),
            next: index,
        };
        subscription.send(ResponseType::StreamingQueryListenerEventsResult(
            proto::StreamingQueryListenerEventsResult {
                events: Vec::new(),
                listener_bus_listener_added: Some(true),
            },
        ));
        *held = Some(subscription);
        Some(Box::pin(receiver))
    }

    /// The responses to a subscription, or `None` while the bus holds another, which Spark
    /// never answers.
    async fn subscribe(
        self: &Arc<Self>,
        session_id: String,
        operation_id: String,
    ) -> Option<ByteStream> {
        let wait = *self.bus_wait.lock().unwrap();
        if wait == BusWait::BeforeAcknowledging {
            self.pass_the_bus_gate().await;
        }
        if wait != BusWait::AfterAccepting {
            return self.register(&session_id, &operation_id, 0);
        }
        let first = respond(
            &session_id,
            &operation_id,
            0,
            ResponseType::ExecutionProgress(proto::execute_plan_response::ExecutionProgress {
                stages: vec![
                    proto::execute_plan_response::execution_progress::StageInfo {
                        num_tasks: 1,
                        ..Default::default()
                    },
                ],
                num_inflight_tasks: 1,
            }),
        );
        let state = Arc::clone(self);
        let rest = stream::once(async move {
            state.pass_the_bus_gate().await;
            state
                .register(&session_id, &operation_id, 1)
                .unwrap_or_else(|| Box::pin(stream::pending()))
        });
        Some(Box::pin(futures::StreamExt::chain(
            stream::iter([Ok(first.encode_to_vec())]),
            futures::StreamExt::flatten(rest),
        )))
    }

    /// Removes the subscription, ending its stream.
    async fn unsubscribe(&self) {
        if *self.bus_wait.lock().unwrap() == BusWait::BeforeRemoving {
            self.pass_the_bus_gate().await;
        }
        if let Some(mut subscription) = self.subscription.lock().unwrap().take() {
            subscription.send(ResponseType::ResultComplete(Default::default()));
        }
    }
}

fn respond(
    session_id: &str,
    operation_id: &str,
    index: usize,
    response_type: proto::execute_plan_response::ResponseType,
) -> proto::ExecutePlanResponse {
    proto::ExecutePlanResponse {
        session_id: session_id.to_string(),
        operation_id: operation_id.to_string(),
        response_id: format!("response-{index}"),
        response_type: Some(response_type),
        ..Default::default()
    }
}

/// The command of `request`, if it executes one.
fn command_of(request: &proto::ExecutePlanRequest) -> Option<&proto::command::CommandType> {
    match request.plan.as_ref()?.op_type.as_ref()? {
        proto::plan::OpType::Command(command) => command.command_type.as_ref(),
        _ => None,
    }
}

#[derive(Clone)]
struct Gated(Arc<State>);

impl NamedService for Gated {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

type ByteStream = Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>>;

impl Service<http::Request<Body>> for Gated {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let rpc = Rpc {
            state: Arc::clone(&self.0),
            method: request
                .uri()
                .path()
                .trim_start_matches("/spark.connect.SparkConnectService/")
                .to_string(),
        };
        Box::pin(async move {
            let mut grpc = Grpc::new(BytesCodec);
            Ok(match rpc.method.as_str() {
                "ExecutePlan" | "ReattachExecute" => grpc.server_streaming(rpc, request).await,
                _ => grpc.unary(rpc, request).await,
            })
        })
    }
}

struct Rpc {
    state: Arc<State>,
    method: String,
}

impl UnaryService<Vec<u8>> for Rpc {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let bytes = request.into_inner();
        let state = Arc::clone(&self.state);
        let reply = match self.method.as_str() {
            "Interrupt" => {
                let request = proto::InterruptRequest::decode(&bytes[..]).unwrap();
                let operation_id = match request.interrupt {
                    Some(proto::interrupt_request::Interrupt::OperationId(id)) => id,
                    _ => String::new(),
                };
                state.receive(Received::Interrupt(operation_id));
                Ok(proto::InterruptResponse {
                    session_id: request.session_id,
                    ..Default::default()
                }
                .encode_to_vec())
            }
            "ReleaseExecute" => {
                let request = proto::ReleaseExecuteRequest::decode(&bytes[..]).unwrap();
                use proto::release_execute_request::Release;
                let operation_id = request.operation_id.clone();
                state.receive(match request.release {
                    Some(Release::ReleaseAll(_)) => Received::ReleaseAll(operation_id),
                    _ => Received::ReleaseUntil(operation_id),
                });
                Ok(proto::ReleaseExecuteResponse {
                    session_id: request.session_id,
                    operation_id: Some(request.operation_id),
                    ..Default::default()
                }
                .encode_to_vec())
            }
            "ReleaseSession" => {
                let request = proto::ReleaseSessionRequest::decode(&bytes[..]).unwrap();
                state.receive(Received::ReleaseSession);
                Ok(proto::ReleaseSessionResponse {
                    session_id: request.session_id,
                    ..Default::default()
                }
                .encode_to_vec())
            }
            "Config" => {
                let request = proto::ConfigRequest::decode(&bytes[..]).unwrap();
                state.receive(Received::Config);
                if format!("{:?}", request.operation).contains(REJECTED_CONFIG) {
                    Err(Status::invalid_argument("the configuration is rejected"))
                } else {
                    Ok(proto::ConfigResponse {
                        session_id: request.session_id,
                        ..Default::default()
                    }
                    .encode_to_vec())
                }
            }
            other => Err(Status::unimplemented(other.to_string())),
        };
        Box::pin(async move { reply.map(Response::new) })
    }
}

impl ServerStreamingService<Vec<u8>> for Rpc {
    type Response = Vec<u8>;
    type ResponseStream = ByteStream;
    type Future = BoxFuture<Response<ByteStream>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let bytes = request.into_inner();
        let state = Arc::clone(&self.state);
        let reattach = self.method == "ReattachExecute";
        Box::pin(async move {
            let wait = *state.wait.lock().unwrap();
            let (session_id, operation_id) = if reattach {
                let request = proto::ReattachExecuteRequest::decode(&bytes[..]).unwrap();
                state.receive(Received::Reattach(request.operation_id.clone()));
                (request.session_id, request.operation_id)
            } else {
                let request = proto::ExecutePlanRequest::decode(&bytes[..]).unwrap();
                let operation_id = request.operation_id.clone().unwrap_or_default();
                state.receive(Received::Execute(operation_id.clone()));
                use proto::command::CommandType as C;
                use proto::streaming_query_listener_bus_command::Command as Bus;
                match command_of(&request) {
                    Some(C::StreamingQueryListenerBusCommand(bus)) => match bus.command {
                        Some(Bus::AddListenerBusListener(_)) => {
                            state.receive(Received::Subscribe(operation_id.clone()));
                            let session_id = request.session_id.clone();
                            match state.subscribe(session_id, operation_id.clone()).await {
                                Some(responses) => return Ok(Response::new(responses)),
                                None => std::future::pending::<()>().await,
                            }
                        }
                        _ => {
                            state.receive(Received::Unsubscribe(operation_id.clone()));
                            state.unsubscribe().await;
                            let complete = respond(
                                &request.session_id,
                                &operation_id,
                                0,
                                ResponseType::ResultComplete(Default::default()),
                            );
                            return Ok(Response::new(encoded(vec![complete])));
                        }
                    },
                    Some(C::WriteStreamOperationStart(start)) => {
                        let started = proto::WriteStreamOperationStartResult {
                            query_id: Some(proto::StreamingQueryInstanceId {
                                id: QUERY_ID.to_string(),
                                run_id: QUERY_RUN_ID.to_string(),
                            }),
                            name: start.query_name.clone(),
                            query_started_event_json: state
                                .subscription
                                .lock()
                                .unwrap()
                                .is_some()
                                .then(|| started_json(&start.query_name)),
                        };
                        let respond = |index, response_type| {
                            respond(&request.session_id, &operation_id, index, response_type)
                        };
                        return Ok(Response::new(encoded(vec![
                            respond(0, ResponseType::WriteStreamOperationStartResult(started)),
                            respond(1, ResponseType::ResultComplete(Default::default())),
                        ])));
                    }
                    _ => {}
                }
                (request.session_id, operation_id)
            };
            let respond = |index: usize, response_type| proto::ExecutePlanResponse {
                session_id: session_id.clone(),
                operation_id: operation_id.clone(),
                response_id: format!("response-{index}"),
                response_type: Some(response_type),
                ..Default::default()
            };
            use proto::execute_plan_response::{ResponseType, ResultComplete};
            let complete = |index| respond(index, ResponseType::ResultComplete(ResultComplete {}));
            let responses: ByteStream = if reattach {
                if wait == Wait::BeforeReattaching {
                    state.pass_the_gate().await;
                }
                encoded(vec![complete(1)])
            } else {
                if wait == Wait::BeforeAccepting {
                    state.pass_the_gate().await;
                }
                let first = proto::ExecutePlanResponse {
                    schema: Some(schema()),
                    ..respond(0, ResponseType::ArrowBatch(arrow_batch(0)))
                };
                match wait {
                    Wait::BeforeAccepting | Wait::Never => encoded(vec![
                        first,
                        respond(1, ResponseType::ArrowBatch(arrow_batch(1))),
                        complete(2),
                    ]),
                    Wait::BeforeReattaching => encoded(vec![first]),
                    Wait::AfterTheFirstBatch => {
                        let rest = vec![
                            respond(1, ResponseType::ArrowBatch(arrow_batch(1))),
                            complete(2),
                        ];
                        let gated = stream::once(async move {
                            state.pass_the_gate().await;
                            stream::iter(rest.into_iter().map(|r| Ok(r.encode_to_vec())))
                        });
                        Box::pin(futures::StreamExt::chain(
                            stream::iter([Ok(first.encode_to_vec())]),
                            futures::StreamExt::flatten(gated),
                        ))
                    }
                }
            };
            Ok(Response::new(responses))
        })
    }
}

fn encoded(responses: Vec<proto::ExecutePlanResponse>) -> ByteStream {
    Box::pin(stream::iter(
        responses
            .into_iter()
            .map(|response| Ok(response.encode_to_vec())),
    ))
}

/// The schema of the result, as the server sends it with the first batch.
pub fn schema() -> proto::DataType {
    spark_connect_core::types::DataType::from_ddl("id BIGINT NOT NULL")
        .unwrap()
        .to_proto()
}

/// An Arrow batch of one row of `id`.
fn arrow_batch(id: i64) -> proto::execute_plan_response::ArrowBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![id]))],
    )
    .unwrap();
    let mut data = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut data, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    proto::execute_plan_response::ArrowBatch {
        row_count: 1,
        data: data.into(),
        ..Default::default()
    }
}
