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

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use futures::{stream, Stream};
use prost::Message;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_proto as proto;
use tokio::sync::Semaphore;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{Grpc, NamedService, ServerStreamingService, UnaryService};
use tonic::{Request, Response, Status};

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

/// What the server received.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Received {
    Execute(String),
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
    ends_with_metrics: Mutex<bool>,
    null_in_the_second_batch: Mutex<bool>,
    received: Mutex<Vec<Received>>,
}

/// The key of a configuration that the server rejects.
pub const REJECTED_CONFIG: &str = "spark.test.rejected";

impl Server {
    pub fn start() -> Server {
        let state = Arc::new(State {
            wait: Mutex::new(Wait::Never),
            gate: Semaphore::new(0),
            ends_with_metrics: Mutex::new(false),
            null_in_the_second_batch: Mutex::new(false),
            received: Mutex::new(Vec::new()),
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

    /// Ends each result that the server sends at once from now on as Spark 4.3.0 ends it: the
    /// last batch is followed by the metrics of the plan, then by those of the observation at
    /// the root of the relation that it runs, each in a response of its own.
    pub fn end_results_with_metrics(&self) {
        *self.state.ends_with_metrics.lock().unwrap() = true;
    }

    /// Sends a null `id` as the second batch of each result that the server sends at once from
    /// now on, which the schema of the batch still declares non-nullable.
    pub fn send_a_null_in_the_second_batch(&self) {
        *self.state.null_in_the_second_batch.lock().unwrap() = true;
    }

    pub fn received(&self) -> Vec<Received> {
        self.state.received.lock().unwrap().clone()
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
                    ..respond(0, ResponseType::ArrowBatch(arrow_batch(Some(0))))
                };
                match wait {
                    Wait::BeforeAccepting | Wait::Never => {
                        let second =
                            (!*state.null_in_the_second_batch.lock().unwrap()).then_some(1);
                        let mut responses = vec![
                            first,
                            respond(1, ResponseType::ArrowBatch(arrow_batch(second))),
                        ];
                        if *state.ends_with_metrics.lock().unwrap() {
                            let request = proto::ExecutePlanRequest::decode(&bytes[..]).unwrap();
                            for (index, metrics) in (2..).zip(metrics_at_the_end(&request)) {
                                responses.push(proto::ExecutePlanResponse {
                                    session_id: session_id.clone(),
                                    operation_id: operation_id.clone(),
                                    response_id: format!("response-{index}"),
                                    ..metrics
                                });
                            }
                        }
                        responses.push(complete(responses.len()));
                        encoded(responses)
                    }
                    Wait::BeforeReattaching => encoded(vec![first]),
                    Wait::AfterTheFirstBatch => {
                        let rest = vec![
                            respond(1, ResponseType::ArrowBatch(arrow_batch(Some(1)))),
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

/// An Arrow batch of one row of `id`, whose schema declares `id` non-nullable even when it is
/// null, as Spark declares a column whose nulls it does not foresee.
fn arrow_batch(id: Option<i64>) -> proto::execute_plan_response::ArrowBatch {
    let declared = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)])),
        vec![Arc::new(Int64Array::from(vec![id]))],
    )
    .unwrap();
    let mut data = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut data, &declared).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    proto::execute_plan_response::ArrowBatch {
        row_count: 1,
        data: data.into(),
        ..Default::default()
    }
}

/// The metrics of the plan that the server sends after the last batch of a result, once
/// [`Server::end_results_with_metrics`] is called: those of a range of the two rows.
pub fn plan_metrics() -> proto::execute_plan_response::Metrics {
    use proto::execute_plan_response::metrics::{MetricObject, MetricValue};
    let output_rows = MetricValue {
        name: "number of output rows".to_string(),
        value: 2,
        metric_type: "sum".to_string(),
    };
    proto::execute_plan_response::Metrics {
        metrics: vec![MetricObject {
            name: "Range".to_string(),
            plan_id: 1,
            parent: 0,
            execution_metrics: [("numOutputRows".to_string(), output_rows)].into(),
        }],
    }
}

/// What Spark 4.3.0 sends after the last batch of the result of `request`, each in a response
/// without a type: the metrics of the plan, then those of the observation at the root of the
/// relation that it runs. Its metrics here are literals, `lit(value).alias(key)` each, observed
/// as `key` = `value`.
fn metrics_at_the_end(request: &proto::ExecutePlanRequest) -> Vec<proto::ExecutePlanResponse> {
    use proto::expression::ExprType;
    let plan = proto::ExecutePlanResponse {
        metrics: Some(plan_metrics()),
        ..Default::default()
    };
    let Some(root) = executed_relation(request) else {
        return vec![plan];
    };
    let Some(proto::relation::RelType::CollectMetrics(collect)) = &root.rel_type else {
        return vec![plan];
    };
    let literal = |metric: &proto::Expression| match &metric.expr_type {
        Some(ExprType::Alias(alias)) => match &alias.expr.as_ref()?.expr_type {
            Some(ExprType::Literal(value)) => Some((alias.name.first()?.clone(), value.clone())),
            _ => None,
        },
        _ => None,
    };
    let (keys, values) = collect.metrics.iter().filter_map(literal).unzip();
    let observed = proto::execute_plan_response::ObservedMetrics {
        name: collect.name.clone(),
        keys,
        values,
        plan_id: root
            .common
            .as_ref()
            .and_then(|c| c.plan_id)
            .unwrap_or_default(),
        ..Default::default()
    };
    let observed = proto::ExecutePlanResponse {
        observed_metrics: vec![observed],
        ..Default::default()
    };
    vec![plan, observed]
}

/// The relation that `request` runs: its query, or the input of the view that it creates.
fn executed_relation(request: &proto::ExecutePlanRequest) -> Option<&proto::Relation> {
    use proto::command::CommandType;
    match request.plan.as_ref()?.op_type.as_ref()? {
        proto::plan::OpType::Root(root) => Some(root),
        proto::plan::OpType::Command(command) => match command.command_type.as_ref()? {
            CommandType::CreateDataframeView(view) => view.input.as_ref(),
            _ => None,
        },
        _ => None,
    }
}
