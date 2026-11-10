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

//! The events of a listener bus, against a loopback gRPC service whose ExecutePlan stream never
//! sends a response.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::{stream, Stream};
use spark_connect_core::actions::streaming::ListenerEvents;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_core::plan::PlanIds;
use spark_connect_core::{ChannelBuilder, SparkConnectClient};
use tokio::net::TcpListener;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::transport::Server;
use tonic::{Request, Response, Status};

const SERVICE: &str = "/spark.connect.SparkConnectService/";
type Frames = Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>>;

#[derive(Clone)]
struct QuietService;

impl NamedService for QuietService {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

impl Service<http::Request<Body>> for QuietService {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let method = request.uri().path().trim_start_matches(SERVICE).to_string();
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(BytesCodec);
            Ok(match method.as_str() {
                "ExecutePlan" | "ReattachExecute" => grpc.server_streaming(Quiet, request).await,
                _ => grpc.unary(Empty, request).await,
            })
        })
    }
}

/// A stream that never sends a response.
struct Quiet;

impl ServerStreamingService<Vec<u8>> for Quiet {
    type Response = Vec<u8>;
    type ResponseStream = Frames;
    type Future = BoxFuture<Response<Frames>, Status>;

    fn call(&mut self, _: Request<Vec<u8>>) -> Self::Future {
        let frames: Frames = Box::pin(stream::pending());
        Box::pin(async { Ok(Response::new(frames)) })
    }
}

/// The release of an execution, answered with an empty message.
struct Empty;

impl UnaryService<Vec<u8>> for Empty {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, _: Request<Vec<u8>>) -> Self::Future {
        Box::pin(async { Ok(Response::new(Vec::new())) })
    }
}

async fn client_of_a_quiet_server() -> SparkConnectClient {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let incoming = stream::unfold(listener, |listener| async {
        Some((listener.accept().await.map(|(socket, _)| socket), listener))
    });
    tokio::spawn(async move {
        Server::builder()
            .add_service(QuietService)
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });
    let builder = ChannelBuilder::parse(&format!("sc://{address}")).unwrap();
    SparkConnectClient::connect(&builder).await.unwrap()
}

#[tokio::test]
async fn a_cancel_ends_the_events_while_no_response_comes() {
    let client = Arc::new(client_of_a_quiet_server().await);
    let mut events = ListenerEvents::subscribe(client, Arc::new(PlanIds::new()), ())
        .await
        .unwrap();
    let (cancel, canceled) = tokio::sync::oneshot::channel();
    events.cancel_on(canceled);
    cancel.send(()).unwrap();
    let within = Duration::from_secs(10);
    tokio::time::timeout(within, events.wait_until_listener_added())
        .await
        .unwrap()
        .unwrap();
    for _ in 0..2 {
        let next = tokio::time::timeout(within, events.next()).await.unwrap();
        assert!(next.is_none(), "{next:?}");
    }
}
