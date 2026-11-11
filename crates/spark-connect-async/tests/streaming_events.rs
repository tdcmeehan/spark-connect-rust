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

//! The events of the streaming queries of a session, from a server whose listener bus acts as
//! Spark 4.3.0's: one subscription for each session, which it keeps until it is removed.

mod gated;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{poll, StreamExt};
use gated::{BusWait, Received, Server, QUERY_ID};
use spark_connect_async::streaming::{QueryEvent, QueryEvents};
use spark_connect_async::{ErrorKind, Session};
use spark_connect_proto as proto;
use spark_connect_proto::StreamingQueryEventType as EventType;

const WAIT: Duration = Duration::from_secs(5);

const PROGRESS_EVENT: &str = r#"{"progress":{"id":"00000000-0000-4000-8000-000000000001","runId":"00000000-0000-4000-8000-000000000002","name":"numbers","timestamp":"2026-11-11T17:57:25.657Z","batchId":1,"batchDuration":229,"numInputRows":5,"stateOperators":[],"sources":[{"description":"rate","startOffset":0,"endOffset":1,"latestOffset":1,"numInputRows":5}],"sink":{"description":"MemorySink","numOutputRows":5}}}"#;

const TERMINATED: &str = r#"{"id":"00000000-0000-4000-8000-000000000001","runId":"00000000-0000-4000-8000-000000000002","exception":null,"errorClassOnException":null}"#;

async fn connect(server: &Server) -> Session {
    Session::builder()
        .remote(server.url())
        .connect()
        .await
        .unwrap()
}

fn is_subscription(request: &proto::ExecutePlanRequest) -> bool {
    use proto::streaming_query_listener_bus_command::Command as Bus;
    matches!(
        request.plan.as_ref().and_then(|plan| plan.op_type.as_ref()),
        Some(proto::plan::OpType::Command(proto::Command {
            command_type: Some(
                proto::command::CommandType::StreamingQueryListenerBusCommand(
                    proto::StreamingQueryListenerBusCommand {
                        command: Some(Bus::AddListenerBusListener(_)),
                    },
                )
            ),
        }))
    )
}

/// A session, and the number of the subscriptions that it has sent, which its hook counts as
/// they are sent.
async fn counting(server: &Server) -> (Session, Arc<AtomicUsize>) {
    let subscriptions = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&subscriptions);
    let session = Session::builder()
        .remote(server.url())
        .execute_plan_hook(move |request| {
            if is_subscription(&request) {
                counted.fetch_add(1, Ordering::SeqCst);
            }
            Ok(request)
        })
        .connect()
        .await
        .unwrap();
    (session, subscriptions)
}

fn subscriptions(received: &[Received]) -> usize {
    received
        .iter()
        .filter(|received| matches!(received, Received::Subscribe(_)))
        .count()
}

fn removed(received: &[Received]) -> bool {
    received
        .iter()
        .any(|received| matches!(received, Received::Unsubscribe(_)))
}

/// Waits up to `WAIT` until what the server received satisfies `done`.
async fn until(server: &Server, done: impl Fn(&[Received]) -> bool) -> bool {
    let deadline = Instant::now() + WAIT;
    while Instant::now() < deadline {
        if done(&server.received()) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    false
}

async fn next(events: &mut QueryEvents) -> Option<spark_connect_async::Result<QueryEvent>> {
    tokio::time::timeout(WAIT, events.next())
        .await
        .expect("no event")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_has_one_subscription_and_the_next_waits_until_the_last_is_removed() {
    let server = Server::start();
    let (session, sent) = counting(&server).await;
    let first = session.streams().events().await.unwrap();
    for handle in [session.clone(), session.with_tag("etl").unwrap()] {
        let second = tokio::time::timeout(WAIT, handle.streams().events())
            .await
            .expect("a second subscription waits");
        let error = second.err().expect("a second subscription");
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
    }
    assert_eq!(
        sent.load(Ordering::SeqCst),
        1,
        "a second subscription was sent"
    );
    server.bus_wait_at(BusWait::BeforeRemoving);
    drop(first);
    assert!(
        until(&server, removed).await,
        "the dropped subscription was not removed"
    );
    let mut next = Box::pin(session.streams().events());
    assert!(poll!(next.as_mut()).is_pending());
    assert_eq!(
        sent.load(Ordering::SeqCst),
        1,
        "subscribed before the server answered the removal"
    );
    server.open_bus_gate();
    tokio::time::timeout(WAIT, next)
        .await
        .expect("no subscription after the removal")
        .unwrap();
    assert_eq!(sent.load(Ordering::SeqCst), 2);
}

#[test]
fn a_dropped_stream_is_removed_from_its_runtime_before_it_is_released() {
    for flavor in ["current-thread", "multi-thread"] {
        for place in ["in its runtime", "outside a runtime", "in another runtime"] {
            let runtime = match flavor {
                "current-thread" => tokio::runtime::Builder::new_current_thread(),
                _ => tokio::runtime::Builder::new_multi_thread(),
            }
            .enable_all()
            .build()
            .unwrap();
            let server = Server::start();
            let (session, events) = runtime.block_on(async {
                let session = connect(&server).await;
                let events = session.streams().events().await.unwrap();
                (session, events)
            });
            let subscription = server.operation_id();
            match place {
                "in its runtime" => runtime.block_on(async move { drop(events) }),
                "outside a runtime" => drop(events),
                _ => tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async move { drop(events) }),
            }
            let release = Received::ReleaseAll(subscription.clone());
            assert!(
                runtime.block_on(until(&server, |received| received.contains(&release))),
                "{flavor}, {place}: the subscription was not released"
            );
            let received = server.received();
            let removal = received
                .iter()
                .position(|received| matches!(received, Received::Unsubscribe(_)));
            let released = received.iter().position(|received| *received == release);
            assert!(
                removal.is_some() && removal < released,
                "{flavor}, {place}: not removed before it was released: {received:?}"
            );
            assert!(!server.subscribed(), "{flavor}, {place}");
            runtime.block_on(async move { drop(session) });
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn events_come_until_the_server_ends_the_subscription() {
    let server = Server::start();
    let session = connect(&server).await;
    let mut events = session.streams().events().await.unwrap();
    server.send_event(EventType::QueryProgressEvent, PROGRESS_EVENT);
    match next(&mut events).await {
        Some(Ok(QueryEvent::Progress(progress))) => assert_eq!(progress.batch_id, 1),
        other => panic!("not the progress: {other:?}"),
    }
    server.send_event(EventType::QueryProgressUnspecified, TERMINATED);
    let error = next(&mut events).await.unwrap().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Decode, "{error}");
    server.send_event(EventType::QueryTerminatedEvent, TERMINATED);
    match next(&mut events).await {
        Some(Ok(QueryEvent::Terminated(terminated))) => assert_eq!(terminated.id, QUERY_ID),
        other => panic!("no event after one that was not read: {other:?}"),
    }
    server.end_subscription();
    assert!(next(&mut events).await.is_none());
    assert!(next(&mut events).await.is_none());
    drop(events);
    tokio::time::timeout(WAIT, session.streams().events())
        .await
        .expect("no subscription after the end")
        .unwrap();
    assert!(
        !removed(&server.received()),
        "removed a subscription that the server had ended"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_subscription_is_an_error_then_the_end_and_is_removed() {
    let server = Server::start();
    let session = connect(&server).await;
    let mut events = session.streams().events().await.unwrap();
    server.fail_subscription();
    let error = next(&mut events).await.unwrap().unwrap_err();
    assert!(
        error.message().contains("the subscription failed"),
        "{error}"
    );
    assert!(next(&mut events).await.is_none());
    assert!(
        until(&server, removed).await,
        "the failed subscription was not removed"
    );
    tokio::time::timeout(WAIT, session.streams().events())
        .await
        .expect("no subscription after the removal")
        .unwrap();
}

#[tokio::test]
async fn an_unanswered_subscription_waits_and_its_drop_ends_it_without_a_removal() {
    let server = Server::start();
    server.hold_other_subscription();
    let session = connect(&server).await;
    let mut waiting = Box::pin(session.streams().events());
    assert!(poll!(waiting.as_mut()).is_pending());
    assert!(until(&server, |received| subscriptions(received) == 1).await);
    assert!(
        poll!(waiting.as_mut()).is_pending(),
        "answered while another client holds the subscription"
    );
    drop(waiting);
    let mut again = Box::pin(session.streams().events());
    assert!(
        poll!(again.as_mut()).is_pending(),
        "the dropped subscription still holds the session"
    );
    let again = tokio::spawn(again);
    assert!(until(&server, |received| subscriptions(received) == 2).await);
    assert!(
        !removed(&server.received()) && server.subscribed(),
        "removed the subscription of another client"
    );
    again.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_subscription_acknowledged_after_its_future_was_dropped_is_removed() {
    let server = Server::start();
    server.bus_wait_at(BusWait::AfterAccepting);
    let read = Arc::new(tokio::sync::Notify::new());
    let signal = Arc::clone(&read);
    let session = Session::builder()
        .remote(server.url())
        .on_progress(move |_| signal.notify_one())
        .connect()
        .await
        .unwrap();
    let waiting = tokio::spawn(session.streams().events());
    tokio::time::timeout(WAIT, read.notified())
        .await
        .expect("the first response was not read");
    waiting.abort();
    assert!(waiting.await.is_err_and(|error| error.is_cancelled()));
    server.open_bus_gate();
    assert!(
        until(&server, removed).await,
        "the subscription that the server acknowledged was not removed"
    );
    assert!(until(&server, |_| !server.subscribed()).await);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_started_event_of_a_query_of_the_session_comes_from_its_start() {
    let server = Server::start();
    let session = connect(&server).await;
    session
        .range(3)
        .write_stream()
        .format("memory")
        .query_name("before")
        .start_without_path()
        .await
        .unwrap();
    let mut events = session.streams().events().await.unwrap();
    let query = session
        .with_tag("etl")
        .unwrap()
        .range(3)
        .write_stream()
        .format("memory")
        .query_name("numbers")
        .start_without_path()
        .await
        .unwrap();
    match next(&mut events).await {
        Some(Ok(QueryEvent::Started(started))) => {
            assert_eq!(started.id, query.id());
            assert_eq!(started.name.as_deref(), Some("numbers"));
        }
        other => panic!("not the started event: {other:?}"),
    }
}
