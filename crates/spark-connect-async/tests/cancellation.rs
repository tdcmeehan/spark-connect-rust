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

//! Dropping the future or the stream of an action at each point of an execution sends what
//! stops the operation on the server, once.

mod gated;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use gated::{Received, Server, Wait, REJECTED_CONFIG};
use spark_connect_async::Session;
use spark_connect_core::types::DataType;

async fn connect(server: &Server) -> Session {
    Session::builder()
        .remote(server.url())
        .connect()
        .await
        .unwrap()
}

/// The `ReleaseAll`s of `operation_id`, once the client has had time to send any more.
async fn releases(server: &Server, operation_id: &str) -> usize {
    tokio::time::sleep(Duration::from_millis(300)).await;
    server
        .received()
        .iter()
        .filter(|received| **received == Received::ReleaseAll(operation_id.to_string()))
        .count()
}

fn interrupted(server: &Server, operation_id: &str) -> bool {
    server
        .received()
        .contains(&Received::Interrupt(operation_id.to_string()))
}

async fn dropped_before_the_server_accepts_it(server: &Server) {
    server.wait_at(Wait::BeforeAccepting);
    let updates = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&updates);
    let session = Session::builder()
        .remote(server.url())
        .on_progress(move |_| {
            counted.fetch_add(1, Ordering::SeqCst);
        })
        .connect()
        .await
        .unwrap();
    let collect = tokio::spawn(session.range(3).collect());
    assert!(server.wait_for(|r| matches!(r, Received::Execute(_))).await);
    let operation_id = server.operation_id();
    collect.abort();
    assert!(collect.await.unwrap_err().is_cancelled());
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::Interrupt(id.clone()))
            .await
    );
    assert_eq!(
        releases(server, &operation_id).await,
        0,
        "released before it was accepted"
    );
    server.open_gate();
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::ReleaseAll(id.clone()))
            .await
    );
    assert_eq!(releases(server, &operation_id).await, 1);
    assert_eq!(
        updates.load(Ordering::SeqCst),
        0,
        "progress of an abandoned execution"
    );
}

async fn dropped_after_the_first_batch(server: &Server) {
    server.wait_at(Wait::AfterTheFirstBatch);
    let session = connect(server).await;
    let mut batches = session.range(3).stream().await.unwrap();
    assert_eq!(batches.next().await.unwrap().unwrap().num_rows(), 1);
    let operation_id = batches.operation_id().to_string();
    drop(batches);
    assert_eq!(releases(server, &operation_id).await, 1);
    assert!(!interrupted(server, &operation_id));
    server.open_gate();
}

async fn dropped_after_the_end(server: &Server) {
    server.wait_at(Wait::Never);
    let session = connect(server).await;
    let mut batches = session.range(3).stream().await.unwrap();
    let mut rows = 0;
    while let Some(batch) = batches.next().await {
        rows += batch.unwrap().num_rows();
    }
    assert_eq!(rows, 2);
    let operation_id = batches.operation_id().to_string();
    let before = releases(server, &operation_id).await;
    drop(batches);
    assert_eq!(releases(server, &operation_id).await, before);
    assert!(before <= 1);
    assert!(!interrupted(server, &operation_id));
}

async fn dropped_while_reattaching(server: &Server) {
    server.wait_at(Wait::BeforeReattaching);
    let session = connect(server).await;
    let mut batches = session.range(3).stream().await.unwrap();
    assert_eq!(batches.next().await.unwrap().unwrap().num_rows(), 1);
    let operation_id = batches.operation_id().to_string();
    let reading = tokio::spawn(async move { batches.next().await.map(|batch| batch.is_ok()) });
    let id = operation_id.clone();
    assert!(
        server
            .wait_for(|r| *r == Received::Reattach(id.clone()))
            .await
    );
    reading.abort();
    assert!(reading.await.unwrap_err().is_cancelled());
    assert_eq!(releases(server, &operation_id).await, 1);
    assert!(!interrupted(server, &operation_id));
    server.open_gate();
}

#[test]
fn dropping_an_action_at_each_point_stops_the_operation_once_in_either_runtime() {
    for flavor in ["current-thread", "multi-thread"] {
        let mut runtime = match flavor {
            "current-thread" => tokio::runtime::Builder::new_current_thread(),
            _ => tokio::runtime::Builder::new_multi_thread(),
        };
        runtime.enable_all().build().unwrap().block_on(async {
            for (point, run) in [
                ("before accepting", 0),
                ("after the first batch", 1),
                ("after the end", 2),
                ("while reattaching", 3),
            ] {
                // A server of each case, so that no gate opened in one passes another.
                let server = Server::start();
                println!("{flavor}: {point}");
                match run {
                    0 => dropped_before_the_server_accepts_it(&server).await,
                    1 => dropped_after_the_first_batch(&server).await,
                    2 => dropped_after_the_end(&server).await,
                    _ => dropped_while_reattaching(&server).await,
                }
            }
        });
    }
}

#[tokio::test]
async fn a_stream_keeps_its_schema_and_execution_info_while_a_dropped_read_is_pending() {
    let server = Server::start();
    server.wait_at(Wait::AfterTheFirstBatch);
    let session = connect(&server).await;
    let mut batches = session.range(3).stream().await.unwrap();
    assert_eq!(batches.next().await.unwrap().unwrap().num_rows(), 1);
    let pending = tokio::time::timeout(Duration::from_millis(100), batches.next()).await;
    assert!(pending.is_err(), "the server sent past its gate");
    let schema = DataType::from_proto(&gated::schema()).unwrap();
    assert_eq!(batches.schema(), Some(&schema));
    assert_eq!(
        batches.execution_info().operation_id(),
        Some(batches.operation_id())
    );
    server.open_gate();
    assert_eq!(batches.next().await.unwrap().unwrap().num_rows(), 1);
}

#[tokio::test]
async fn a_session_whose_configuration_the_server_rejects_is_released() {
    let server = Server::start();
    let error = Session::builder()
        .remote(server.url())
        .config(REJECTED_CONFIG, "1")
        .connect()
        .await
        .err()
        .unwrap();
    assert!(error.message().contains("rejected"), "{error}");
    assert!(server.wait_for(|r| *r == Received::ReleaseSession).await);
}
