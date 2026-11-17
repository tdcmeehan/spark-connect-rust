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

//! An in-process Spark Connect server that records the request bytes of every RPC and
//! answers each with the smallest response the client accepts.

use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll};

use futures::{stream, Stream, StreamExt};
use prost::Message;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_proto as proto;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{
    ClientStreamingService, Grpc, NamedService, ServerStreamingService, UnaryService,
};
use tonic::{Request, Response, Status, Streaming};

/// One request as the server received it, or one value a case recorded itself.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// The protobuf message type of `bytes`.
    pub kind: String,
    pub bytes: Vec<u8>,
}

#[derive(Default)]
pub struct Recorder {
    entries: Mutex<Vec<Entry>>,
    pushed: Condvar,
    holds_listener_bus: AtomicBool,
    held_subscription_end: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    /// The references to ML objects that the server has given since the last `take`.
    ml_refs_given: AtomicUsize,
}

impl Recorder {
    pub fn push(&self, kind: &str, bytes: Vec<u8>) {
        self.entries.lock().unwrap().push(Entry {
            kind: kind.to_string(),
            bytes,
        });
        self.pushed.notify_all();
    }

    /// Waits up to `timeout` for an entry that `found` accepts, and returns whether one came.
    pub fn wait_for(&self, timeout: std::time::Duration, found: impl Fn(&Entry) -> bool) -> bool {
        let entries = self.entries.lock().unwrap();
        let (entries, _) = self
            .pushed
            .wait_timeout_while(entries, timeout, |entries| !entries.iter().any(&found))
            .unwrap();
        entries.iter().any(found)
    }

    pub fn take(&self) -> Vec<Entry> {
        let mut entries = self.entries.lock().unwrap();
        self.ml_refs_given.store(0, Ordering::SeqCst);
        std::mem::take(&mut entries)
    }

    /// Waits up to `timeout` until the client has deleted each ML object whose reference the
    /// server gave since the last `take`, which it does in the background once the last owner
    /// of the object drops, and returns how many it has not deleted.
    pub fn wait_for_ml_deletes(&self, timeout: std::time::Duration) -> usize {
        let undeleted = |entries: &Vec<Entry>| {
            self.ml_refs_given
                .load(Ordering::SeqCst)
                .saturating_sub(deleted_ml_refs(entries))
        };
        let entries = self.entries.lock().unwrap();
        let (entries, _) = self
            .pushed
            .wait_timeout_while(entries, timeout, |entries| undeleted(entries) > 0)
            .unwrap();
        undeleted(&entries)
    }

    /// Whether the server keeps each subscription to the listener bus open, after its
    /// acknowledgement, until the client removes its listener, as Spark does.
    #[allow(dead_code)]
    pub fn hold_listener_bus(&self, hold: bool) {
        self.holds_listener_bus.store(hold, Ordering::SeqCst);
    }

    fn hold_subscription(&self) -> Option<tokio::sync::oneshot::Receiver<()>> {
        if !self.holds_listener_bus.load(Ordering::SeqCst) {
            return None;
        }
        let (end, ended) = tokio::sync::oneshot::channel();
        *self.held_subscription_end.lock().unwrap() = Some(end);
        Some(ended)
    }

    fn end_held_subscription(&self) {
        if let Some(end) = self.held_subscription_end.lock().unwrap().take() {
            let _ = end.send(());
        }
    }
}

/// Starts the server on its own runtime thread, as the client under test blocks on the
/// shared runtime of the crate, and returns its address.
pub fn start(recorder: Arc<Recorder>) -> std::net::SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
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
                .add_service(FakeService(recorder))
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
    });
    address
}

#[derive(Clone)]
struct FakeService(Arc<Recorder>);

impl NamedService for FakeService {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

type ByteStream = Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>>;

impl Service<http::Request<Body>> for FakeService {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let recorder = self.0.clone();
        let method = request
            .uri()
            .path()
            .trim_start_matches("/spark.connect.SparkConnectService/")
            .to_string();
        Box::pin(async move {
            let mut grpc = Grpc::new(BytesCodec);
            let rpc = Rpc { recorder, method };
            Ok(match rpc.method.as_str() {
                "ExecutePlan" | "ReattachExecute" => grpc.server_streaming(rpc, request).await,
                "AddArtifacts" => grpc.client_streaming(rpc, request).await,
                _ => grpc.unary(rpc, request).await,
            })
        })
    }
}

struct Rpc {
    recorder: Arc<Recorder>,
    method: String,
}

impl Rpc {
    fn record(&self, kind: &str, bytes: &[u8]) {
        // Releases are sent in the background, after the call that read the stream returned.
        if kind != "ReleaseExecuteRequest" {
            self.recorder.push(kind, bytes.to_vec());
        }
    }
}

impl UnaryService<Vec<u8>> for Rpc {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let bytes = request.into_inner();
        let reply = match self.method.as_str() {
            "AnalyzePlan" => {
                self.record("AnalyzePlanRequest", &bytes);
                Ok(analyze(proto::AnalyzePlanRequest::decode(&bytes[..]).unwrap()).encode_to_vec())
            }
            "Config" => {
                self.record("ConfigRequest", &bytes);
                Ok(config(proto::ConfigRequest::decode(&bytes[..]).unwrap()).encode_to_vec())
            }
            "ArtifactStatus" => {
                self.record("ArtifactStatusesRequest", &bytes);
                let request = proto::ArtifactStatusesRequest::decode(&bytes[..]).unwrap();
                Ok(proto::ArtifactStatusesResponse {
                    session_id: request.session_id,
                    statuses: request
                        .names
                        .into_iter()
                        .map(|name| (name, Default::default()))
                        .collect(),
                    ..Default::default()
                }
                .encode_to_vec())
            }
            "Interrupt" => {
                self.record("InterruptRequest", &bytes);
                let request = proto::InterruptRequest::decode(&bytes[..]).unwrap();
                Ok(proto::InterruptResponse {
                    session_id: request.session_id,
                    ..Default::default()
                }
                .encode_to_vec())
            }
            "ReleaseExecute" => {
                self.record("ReleaseExecuteRequest", &bytes);
                let request = proto::ReleaseExecuteRequest::decode(&bytes[..]).unwrap();
                Ok(proto::ReleaseExecuteResponse {
                    session_id: request.session_id,
                    operation_id: Some(request.operation_id),
                    ..Default::default()
                }
                .encode_to_vec())
            }
            "ReleaseSession" => {
                self.record("ReleaseSessionRequest", &bytes);
                let request = proto::ReleaseSessionRequest::decode(&bytes[..]).unwrap();
                Ok(proto::ReleaseSessionResponse {
                    session_id: request.session_id,
                    ..Default::default()
                }
                .encode_to_vec())
            }
            "CloneSession" => {
                self.record("CloneSessionRequest", &bytes);
                let request = proto::CloneSessionRequest::decode(&bytes[..]).unwrap();
                Ok(proto::CloneSessionResponse {
                    session_id: request.session_id,
                    new_session_id: request
                        .new_session_id
                        .unwrap_or_else(|| "00000000-0000-0000-0000-00000000c10e".to_string()),
                    ..Default::default()
                }
                .encode_to_vec())
            }
            "FetchErrorDetails" => {
                self.record("FetchErrorDetailsRequest", &bytes);
                Ok(proto::FetchErrorDetailsResponse::default().encode_to_vec())
            }
            other => {
                self.record(other, &bytes);
                Err(Status::unimplemented(other.to_string()))
            }
        };
        Box::pin(async move { reply.map(Response::new) })
    }
}

impl ServerStreamingService<Vec<u8>> for Rpc {
    type Response = Vec<u8>;
    type ResponseStream = ByteStream;
    type Future = BoxFuture<Response<ByteStream>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        use proto::streaming_query_listener_bus_command::Command as Bus;
        let bytes = request.into_inner();
        let responses = if self.method == "ReattachExecute" {
            self.record("ReattachExecuteRequest", &bytes);
            let request = proto::ReattachExecuteRequest::decode(&bytes[..]).unwrap();
            complete(&request.session_id, &request.operation_id, Vec::new())
        } else {
            self.record("ExecutePlanRequest", &bytes);
            let request = proto::ExecutePlanRequest::decode(&bytes[..]).unwrap();
            match listener_bus_command(&request) {
                Some(Bus::AddListenerBusListener(_)) => {
                    if let Some(ended) = self.recorder.hold_subscription() {
                        let stream = held_subscription(&request, ended);
                        return Box::pin(async move { Ok(Response::new(stream)) });
                    }
                }
                Some(Bus::RemoveListenerBusListener(_)) => self.recorder.end_held_subscription(),
                _ => {}
            }
            let responses = execute(request);
            self.recorder
                .ml_refs_given
                .fetch_add(given_ml_refs(&responses), Ordering::SeqCst);
            responses
        };
        let stream: ByteStream = Box::pin(stream::iter(
            responses.into_iter().map(|r| Ok(r.encode_to_vec())),
        ));
        Box::pin(async move { Ok(Response::new(stream)) })
    }
}

impl ClientStreamingService<Vec<u8>> for Rpc {
    type Response = Vec<u8>;
    type Future = Pin<Box<dyn Future<Output = Result<Response<Vec<u8>>, Status>> + Send>>;

    fn call(&mut self, request: Request<Streaming<Vec<u8>>>) -> Self::Future {
        let recorder = self.recorder.clone();
        Box::pin(async move {
            let mut requests = request.into_inner();
            let mut session_id = String::new();
            let mut names = Vec::new();
            while let Some(bytes) = requests.next().await {
                let bytes = bytes?;
                recorder.push("AddArtifactsRequest", bytes.clone());
                let request = proto::AddArtifactsRequest::decode(&bytes[..]).unwrap();
                session_id = request.session_id;
                use proto::add_artifacts_request::Payload;
                match request.payload {
                    Some(Payload::Batch(batch)) => {
                        names.extend(batch.artifacts.into_iter().map(|a| a.name))
                    }
                    Some(Payload::BeginChunk(begin)) => names.push(begin.name),
                    _ => {}
                }
            }
            Ok(Response::new(
                proto::AddArtifactsResponse {
                    session_id,
                    artifacts: names
                        .into_iter()
                        .map(|name| proto::add_artifacts_response::ArtifactSummary {
                            name,
                            is_crc_successful: true,
                        })
                        .collect(),
                    ..Default::default()
                }
                .encode_to_vec(),
            ))
        })
    }
}

/// The struct schema that the server reports for every plan.
pub fn schema() -> proto::DataType {
    use proto::data_type::{Kind, StructField};
    let field = |name: &str, kind: Kind| StructField {
        name: name.to_string(),
        data_type: Some(proto::DataType { kind: Some(kind) }),
        nullable: true,
        metadata: None,
    };
    proto::DataType {
        kind: Some(Kind::Struct(proto::data_type::Struct {
            fields: vec![
                field("id", Kind::Long(Default::default())),
                field("name", Kind::String(Default::default())),
                field("value", Kind::Double(Default::default())),
            ],
            type_variation_reference: 0,
        })),
    }
}

fn analyze(request: proto::AnalyzePlanRequest) -> proto::AnalyzePlanResponse {
    use proto::analyze_plan_request::Analyze as A;
    use proto::analyze_plan_response as r;
    let result = match request.analyze {
        Some(A::Schema(_)) => r::Result::Schema(r::Schema {
            schema: Some(schema()),
        }),
        Some(A::Explain(_)) => r::Result::Explain(r::Explain {
            explain_string: "== Physical Plan ==".to_string(),
        }),
        Some(A::TreeString(_)) => r::Result::TreeString(r::TreeString {
            tree_string: "root".to_string(),
        }),
        Some(A::IsLocal(_)) => r::Result::IsLocal(r::IsLocal { is_local: false }),
        Some(A::IsStreaming(_)) => r::Result::IsStreaming(r::IsStreaming {
            is_streaming: false,
        }),
        Some(A::InputFiles(_)) => r::Result::InputFiles(r::InputFiles { files: Vec::new() }),
        Some(A::SparkVersion(_)) => r::Result::SparkVersion(r::SparkVersion {
            version: "4.3.0".to_string(),
        }),
        Some(A::DdlParse(_)) => r::Result::DdlParse(r::DdlParse {
            parsed: Some(schema()),
        }),
        Some(A::SameSemantics(_)) => r::Result::SameSemantics(r::SameSemantics { result: true }),
        Some(A::SemanticHash(_)) => r::Result::SemanticHash(r::SemanticHash { result: 7 }),
        Some(A::Persist(_)) => r::Result::Persist(r::Persist {}),
        Some(A::Unpersist(_)) => r::Result::Unpersist(r::Unpersist {}),
        Some(A::GetStorageLevel(_)) => r::Result::GetStorageLevel(r::GetStorageLevel {
            storage_level: Some(proto::StorageLevel {
                use_disk: true,
                use_memory: true,
                use_off_heap: false,
                deserialized: true,
                replication: 1,
            }),
        }),
        Some(A::JsonToDdl(_)) => r::Result::JsonToDdl(r::JsonToDdl {
            ddl_string: "id BIGINT".to_string(),
        }),
        None => r::Result::IsLocal(r::IsLocal { is_local: false }),
    };
    proto::AnalyzePlanResponse {
        session_id: request.session_id,
        result: Some(result),
        ..Default::default()
    }
}

fn config(request: proto::ConfigRequest) -> proto::ConfigResponse {
    use proto::config_request::operation::OpType;
    let pairs = match request.operation.and_then(|op| op.op_type) {
        Some(OpType::Get(get)) => get.keys.into_iter().map(unset).collect(),
        Some(OpType::GetOption(get)) => get.keys.into_iter().map(unset).collect(),
        Some(OpType::GetWithDefault(get)) => get.pairs,
        Some(OpType::IsModifiable(get)) => get
            .keys
            .into_iter()
            .map(|key| proto::KeyValue {
                key,
                value: Some("true".to_string()),
            })
            .collect(),
        _ => Vec::new(),
    };
    proto::ConfigResponse {
        session_id: request.session_id,
        pairs,
        ..Default::default()
    }
}

/// The value of the configs that the client reads before it sends local data or its first plan
/// (Spark's defaults of the plan compression), else none.
fn unset(key: String) -> proto::KeyValue {
    let value = match key.as_str() {
        "spark.connect.session.planCompression.threshold" => Some("10485760"),
        "spark.connect.session.planCompression.defaultAlgorithm" => Some("ZSTD"),
        "spark.sql.timestampType" => Some("TIMESTAMP_LTZ"),
        "spark.sql.session.timeZone" => Some("UTC"),
        "spark.sql.execution.arrow.useLargeVarTypes" => Some("false"),
        "spark.sql.session.localRelationCacheThreshold" => Some("1024"),
        "spark.sql.session.localRelationSizeLimit" => Some("67108864"),
        "spark.sql.session.localRelationChunkSizeRows" => Some("1000"),
        "spark.sql.session.localRelationChunkSizeBytes" => Some("4096"),
        "spark.sql.session.localRelationBatchOfChunksSizeBytes" => Some("8192"),
        _ => None,
    };
    proto::KeyValue {
        key,
        value: value.map(str::to_string),
    }
}

fn response(
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

/// `results` followed by `ResultComplete`.
fn complete(
    session_id: &str,
    operation_id: &str,
    results: Vec<proto::execute_plan_response::ResponseType>,
) -> Vec<proto::ExecutePlanResponse> {
    use proto::execute_plan_response::{ResponseType, ResultComplete};
    results
        .into_iter()
        .chain([ResponseType::ResultComplete(ResultComplete {})])
        .enumerate()
        .map(|(index, result)| response(session_id, operation_id, index, result))
        .collect()
}

fn listener_bus_command(
    request: &proto::ExecutePlanRequest,
) -> Option<&proto::streaming_query_listener_bus_command::Command> {
    use proto::command::CommandType as C;
    match request.plan.as_ref()?.op_type.as_ref()? {
        proto::plan::OpType::Command(proto::Command {
            command_type: Some(C::StreamingQueryListenerBusCommand(bus)),
        }) => bus.command.as_ref(),
        _ => None,
    }
}

/// Spark's responses to a subscription to the listener bus: its acknowledgement, then the end of
/// the results once `ended` completes.
fn held_subscription(
    request: &proto::ExecutePlanRequest,
    ended: tokio::sync::oneshot::Receiver<()>,
) -> ByteStream {
    let operation_id = request.operation_id.clone().unwrap_or_default();
    let mut responses = complete(&request.session_id, &operation_id, vec![listener_added()]);
    let result_complete = responses.pop().unwrap().encode_to_vec();
    let acknowledgement = stream::iter(
        responses
            .into_iter()
            .map(|response| Ok(response.encode_to_vec())),
    );
    let end = stream::once(async move {
        let _ = ended.await;
        Ok(result_complete)
    });
    Box::pin(acknowledgement.chain(end))
}

/// Spark's acknowledgement of a subscription to the listener bus.
fn listener_added() -> proto::execute_plan_response::ResponseType {
    proto::execute_plan_response::ResponseType::StreamingQueryListenerEventsResult(
        proto::StreamingQueryListenerEventsResult {
            events: Vec::new(),
            listener_bus_listener_added: Some(true),
        },
    )
}

fn execute(request: proto::ExecutePlanRequest) -> Vec<proto::ExecutePlanResponse> {
    use proto::plan::OpType;
    let operation_id = request.operation_id.clone().unwrap_or_default();
    let mut observed = Vec::new();
    let results = match request.plan.and_then(|plan| plan.op_type) {
        Some(OpType::Root(relation)) => {
            observed = observed_literals(&relation);
            relation_results(&relation)
        }
        Some(OpType::Command(command)) => command_results(command),
        _ => Vec::new(),
    };
    let mut responses = complete(&request.session_id, &operation_id, results);
    responses[0].observed_metrics = observed;
    responses
}

/// The metrics observed for a root `CollectMetrics` relation whose metrics are all literals,
/// `lit(value).alias(key)` each, as `key` = `value`; none for any other relation.
fn observed_literals(
    relation: &proto::Relation,
) -> Vec<proto::execute_plan_response::ObservedMetrics> {
    use proto::expression::ExprType;
    let Some(proto::relation::RelType::CollectMetrics(collect)) = &relation.rel_type else {
        return Vec::new();
    };
    let literal = |metric: &proto::Expression| match &metric.expr_type {
        Some(ExprType::Alias(alias)) => match &alias.expr.as_ref()?.expr_type {
            Some(ExprType::Literal(value)) => Some((alias.name.first()?.clone(), value.clone())),
            _ => None,
        },
        _ => None,
    };
    let Some(metrics) = collect
        .metrics
        .iter()
        .map(literal)
        .collect::<Option<Vec<_>>>()
    else {
        return Vec::new();
    };
    let (keys, values) = metrics.into_iter().unzip();
    vec![proto::execute_plan_response::ObservedMetrics {
        name: collect.name.clone(),
        keys,
        values,
        plan_id: relation
            .common
            .as_ref()
            .and_then(|common| common.plan_id)
            .unwrap_or_default(),
        ..Default::default()
    }]
}

fn relation_results(relation: &proto::Relation) -> Vec<proto::execute_plan_response::ResponseType> {
    use proto::relation::RelType;
    match &relation.rel_type {
        Some(RelType::ShowString(_)) | Some(RelType::HtmlString(_)) => {
            vec![arrow_batch(string_batch("show_string", "+--+"))]
        }
        // The largest label, of which OneVsRest's fit takes its number of classes.
        Some(RelType::Aggregate(aggregate)) if is_max_of_cast(aggregate) => {
            vec![arrow_batch(double_batch("max", 1.0))]
        }
        Some(RelType::Aggregate(aggregate)) if aggregate.grouping_expressions.is_empty() => {
            vec![arrow_batch(long_batch("count", 3))]
        }
        Some(RelType::Catalog(catalog)) => {
            use proto::catalog::CatType;
            match catalog.cat_type {
                Some(CatType::CurrentCatalog(_)) | Some(CatType::CurrentDatabase(_)) => {
                    vec![arrow_batch(string_batch("value", "spark_catalog"))]
                }
                _ => Vec::new(),
            }
        }
        _ => Vec::new(),
    }
}

fn is_max_of_cast(aggregate: &proto::Aggregate) -> bool {
    use proto::expression::ExprType;
    let [proto::Expression {
        expr_type: Some(ExprType::UnresolvedFunction(function)),
        ..
    }] = aggregate.aggregate_expressions.as_slice()
    else {
        return false;
    };
    aggregate.grouping_expressions.is_empty()
        && function.function_name == "max"
        && matches!(
            function.arguments.as_slice(),
            [proto::Expression {
                expr_type: Some(ExprType::Cast(_)),
                ..
            }]
        )
}

fn command_results(command: proto::Command) -> Vec<proto::execute_plan_response::ResponseType> {
    use proto::command::CommandType as C;
    use proto::execute_plan_response::{ResponseType as R, SqlCommandResult};
    use proto::streaming_query_listener_bus_command::Command as Bus;
    let query_id = || proto::StreamingQueryInstanceId {
        id: "query-id".to_string(),
        run_id: "query-run-id".to_string(),
    };
    match command.command_type {
        Some(C::SqlCommand(sql)) => vec![R::SqlCommandResult(SqlCommandResult {
            relation: sql.input,
        })],
        Some(C::WriteStreamOperationStart(start)) => {
            vec![R::WriteStreamOperationStartResult(
                proto::WriteStreamOperationStartResult {
                    query_id: Some(query_id()),
                    name: start.query_name,
                    ..Default::default()
                },
            )]
        }
        Some(C::StreamingQueryCommand(command)) => {
            vec![R::StreamingQueryCommandResult(streaming_query_result(
                command,
            ))]
        }
        Some(C::StreamingQueryManagerCommand(command)) => {
            vec![R::StreamingQueryManagerCommandResult(
                streaming_manager_result(command),
            )]
        }
        Some(C::CheckpointCommand(_)) => {
            vec![R::CheckpointCommandResult(proto::CheckpointCommandResult {
                relation: Some(proto::CachedRemoteRelation {
                    relation_id: "checkpointed".to_string(),
                }),
            })]
        }
        Some(C::CreateResourceProfileCommand(_)) => vec![R::CreateResourceProfileCommandResult(
            proto::CreateResourceProfileCommandResult { profile_id: 1 },
        )],
        Some(C::GetResourcesCommand(_)) => vec![R::GetResourcesCommandResult(
            proto::GetResourcesCommandResult::default(),
        )],
        Some(C::MlCommand(command)) => vec![R::MlCommandResult(ml_result(command))],
        Some(C::PipelineCommand(command)) => {
            vec![R::PipelineCommandResult(pipeline_result(command))]
        }
        Some(C::StreamingQueryListenerBusCommand(proto::StreamingQueryListenerBusCommand {
            command: Some(Bus::AddListenerBusListener(_)),
        })) => vec![listener_added()],
        _ => Vec::new(),
    }
}

fn streaming_query_result(
    command: proto::StreamingQueryCommand,
) -> proto::StreamingQueryCommandResult {
    use proto::streaming_query_command::Command as C;
    use proto::streaming_query_command_result as r;
    let result_type = match command.command {
        Some(C::Status(_)) => Some(r::ResultType::Status(r::StatusResult {
            status_message: "Waiting".to_string(),
            is_data_available: false,
            is_trigger_active: false,
            is_active: true,
        })),
        Some(C::LastProgress(_)) | Some(C::RecentProgress(_)) => {
            Some(r::ResultType::RecentProgress(r::RecentProgressResult {
                recent_progress_json: vec!["{}".to_string()],
            }))
        }
        Some(C::Explain(_)) => Some(r::ResultType::Explain(r::ExplainResult {
            result: "== Plan ==".to_string(),
        })),
        Some(C::Exception(_)) => Some(r::ResultType::Exception(r::ExceptionResult::default())),
        Some(C::AwaitTermination(_)) => {
            Some(r::ResultType::AwaitTermination(r::AwaitTerminationResult {
                terminated: true,
            }))
        }
        _ => None,
    };
    proto::StreamingQueryCommandResult {
        query_id: command.query_id,
        result_type,
    }
}

fn streaming_manager_result(
    command: proto::StreamingQueryManagerCommand,
) -> proto::StreamingQueryManagerCommandResult {
    use proto::streaming_query_manager_command::Command as C;
    use proto::streaming_query_manager_command_result as r;
    let instance = || r::StreamingQueryInstance {
        id: Some(proto::StreamingQueryInstanceId {
            id: "query-id".to_string(),
            run_id: "query-run-id".to_string(),
        }),
        name: Some("query".to_string()),
    };
    let result_type = match command.command {
        Some(C::Active(_)) => r::ResultType::Active(r::ActiveResult {
            active_queries: vec![instance()],
        }),
        Some(C::GetQuery(_)) => r::ResultType::Query(instance()),
        Some(C::AwaitAnyTermination(_)) => {
            r::ResultType::AwaitAnyTermination(r::AwaitAnyTerminationResult { terminated: true })
        }
        Some(C::ResetTerminated(_)) => r::ResultType::ResetTerminated(true),
        Some(C::AddListener(_)) => r::ResultType::AddListener(true),
        Some(C::RemoveListener(_)) => r::ResultType::RemoveListener(true),
        Some(C::ListListeners(_)) => {
            r::ResultType::ListListeners(r::ListStreamingQueryListenerResult {
                listener_ids: Vec::new(),
            })
        }
        None => r::ResultType::ResetTerminated(true),
    };
    proto::StreamingQueryManagerCommandResult {
        result_type: Some(result_type),
    }
}

/// The references to ML objects that `responses` give the client.
fn given_ml_refs(responses: &[proto::ExecutePlanResponse]) -> usize {
    use proto::execute_plan_response::ResponseType;
    use proto::ml_command_result::{ml_operator_info::Type, ResultType};
    responses
        .iter()
        .filter(|response| {
            matches!(
                &response.response_type,
                Some(ResponseType::MlCommandResult(proto::MlCommandResult {
                    result_type: Some(ResultType::OperatorInfo(
                        proto::ml_command_result::MlOperatorInfo {
                            r#type: Some(Type::ObjRef(_)),
                            ..
                        }
                    )),
                }))
            )
        })
        .count()
}

/// The ML objects that the `MlCommand.Delete`s among `entries` delete.
fn deleted_ml_refs(entries: &[Entry]) -> usize {
    use proto::command::CommandType;
    entries
        .iter()
        .filter(|entry| entry.kind == "ExecutePlanRequest")
        .filter_map(|entry| proto::ExecutePlanRequest::decode(&entry.bytes[..]).ok())
        .map(|request| match request.plan.and_then(|plan| plan.op_type) {
            Some(proto::plan::OpType::Command(proto::Command {
                command_type:
                    Some(CommandType::MlCommand(proto::MlCommand {
                        command: Some(proto::ml_command::Command::Delete(delete)),
                    })),
            })) => delete.obj_refs.len(),
            _ => 0,
        })
        .sum()
}

fn ml_result(command: proto::MlCommand) -> proto::MlCommandResult {
    use proto::ml_command::Command as C;
    use proto::ml_command_result as r;
    let result_type = match command.command {
        // As Spark's: only a model is a cached object; another operator is named.
        Some(C::Read(proto::ml_command::Read {
            operator: Some(operator),
            ..
        })) if operator.r#type != proto::ml_operator::OperatorType::Model as i32 => {
            r::ResultType::OperatorInfo(r::MlOperatorInfo {
                r#type: Some(r::ml_operator_info::Type::Name(operator.name)),
                uid: Some("loaded-uid".to_string()),
                params: None,
                warning_message: None,
            })
        }
        // The ML helper's model constructors, such as `countVectorizerModelFromVocabulary`.
        Some(C::Fetch(fetch)) if fetch.methods.iter().any(|m| m.method.contains("ModelFrom")) => {
            r::ResultType::OperatorInfo(r::MlOperatorInfo {
                r#type: Some(r::ml_operator_info::Type::ObjRef(proto::ObjectRef {
                    id: "model-ref".to_string(),
                })),
                uid: None,
                params: None,
                warning_message: None,
            })
        }
        Some(C::Fit(_)) | Some(C::Read(_)) => r::ResultType::OperatorInfo(r::MlOperatorInfo {
            r#type: Some(r::ml_operator_info::Type::ObjRef(proto::ObjectRef {
                id: "model-ref".to_string(),
            })),
            uid: Some("model-uid".to_string()),
            params: None,
            warning_message: None,
        }),
        Some(C::Evaluate(_)) => r::ResultType::Param(proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Double(0.5)),
            ..Default::default()
        }),
        _ => r::ResultType::Summary("ok".to_string()),
    };
    proto::MlCommandResult {
        result_type: Some(result_type),
    }
}

fn pipeline_result(command: proto::PipelineCommand) -> proto::PipelineCommandResult {
    use proto::pipeline_command::CommandType as C;
    use proto::pipeline_command_result as r;
    let result_type = match command.command_type {
        Some(C::CreateDataflowGraph(_)) => Some(r::ResultType::CreateDataflowGraphResult(
            r::CreateDataflowGraphResult {
                dataflow_graph_id: Some("graph-id".to_string()),
            },
        )),
        _ => None,
    };
    proto::PipelineCommandResult { result_type }
}

fn arrow_batch(data: Vec<u8>) -> proto::execute_plan_response::ResponseType {
    proto::execute_plan_response::ResponseType::ArrowBatch(
        proto::execute_plan_response::ArrowBatch {
            row_count: 1,
            data: data.into(),
            start_offset: None,
            ..Default::default()
        },
    )
}

fn ipc(batch: arrow::record_batch::RecordBatch) -> Vec<u8> {
    let mut data = Vec::new();
    let mut writer = arrow::ipc::writer::StreamWriter::try_new(&mut data, &batch.schema()).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    drop(writer);
    data
}

fn one_value_batch(name: &str, column: arrow::array::ArrayRef) -> Vec<u8> {
    use arrow::datatypes::{Field, Schema};
    let field = Field::new(name, column.data_type().clone(), true);
    ipc(
        arrow::record_batch::RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![column])
            .unwrap(),
    )
}

fn string_batch(name: &str, value: &str) -> Vec<u8> {
    one_value_batch(name, Arc::new(arrow::array::StringArray::from(vec![value])))
}

fn long_batch(name: &str, value: i64) -> Vec<u8> {
    one_value_batch(name, Arc::new(arrow::array::Int64Array::from(vec![value])))
}

fn double_batch(name: &str, value: f64) -> Vec<u8> {
    one_value_batch(
        name,
        Arc::new(arrow::array::Float64Array::from(vec![value])),
    )
}
