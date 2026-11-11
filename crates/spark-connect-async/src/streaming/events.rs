//! The events of the streaming queries of a session, as a stream of the subscription of the
//! session to them.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures::future::BoxFuture;
use futures::{FutureExt, Stream};
use spark_connect_core::actions::streaming::{
    subscribe_command, unsubscribe_command, ListenerEvents,
};
use spark_connect_core::actions::Execution;
use spark_connect_core::error::SparkError;
use spark_connect_proto as proto;
use tokio::sync::{mpsc, oneshot, Notify};

use super::progress::{event_of, started_of, QueryEvent};
use crate::error::{Error, Result};
use crate::progress::Observer;
use crate::session::Session;
use crate::stream;

/// The subscription of a session to the events of its streaming queries. The server keeps one
/// subscription for each session: it does not answer another until the first is removed.
#[derive(Default)]
pub(crate) struct Bus {
    state: Mutex<BusState>,
    freed: Notify,
}

#[derive(Default)]
struct BusState {
    generation: u64,
    phase: Phase,
}

#[derive(Default)]
enum Phase {
    #[default]
    Free,
    /// A subscription is starting or streaming. Its channel takes the started events of the
    /// queries that the session starts, which the server sends with their start.
    Subscribed(mpsc::UnboundedSender<String>),
    /// The subscription is ending: its removal is on its way to the server, or the task of a
    /// start that its caller gave up is finishing.
    Ending,
}

impl Bus {
    /// Takes the bus for a new subscription, once the last one has ended, and returns its
    /// generation and the channel of its started events. Fails while another subscription of
    /// the session starts or streams.
    async fn claim(&self) -> Result<(u64, mpsc::UnboundedReceiver<String>)> {
        loop {
            let freed = self.freed.notified();
            tokio::pin!(freed);
            {
                let mut state = self.state.lock().unwrap();
                match state.phase {
                    Phase::Free => {
                        state.generation += 1;
                        let (started, receiver) = mpsc::unbounded_channel();
                        state.phase = Phase::Subscribed(started);
                        return Ok((state.generation, receiver));
                    }
                    Phase::Subscribed(_) => {
                        return Err(Error::unsupported(
                            "the session already streams the events of its queries: the server \
                             keeps one subscription to them for each session, so drop that \
                             stream before asking for another",
                        ))
                    }
                    Phase::Ending => {
                        freed.as_mut().enable();
                    }
                }
            }
            freed.await;
        }
    }

    /// Gives `json`, the started event of a query that the session started, to the session's
    /// subscription, if it has one.
    pub(crate) fn post_started(&self, json: String) {
        if let Phase::Subscribed(started) = &self.state.lock().unwrap().phase {
            let _ = started.send(json);
        }
    }

    fn ending(&self, generation: u64) {
        let mut state = self.state.lock().unwrap();
        if state.generation == generation && matches!(state.phase, Phase::Subscribed(_)) {
            state.phase = Phase::Ending;
        }
    }

    fn free(&self, generation: u64) {
        let mut state = self.state.lock().unwrap();
        if state.generation == generation && !matches!(state.phase, Phase::Free) {
            state.phase = Phase::Free;
            self.freed.notify_waiters();
        }
    }
}

/// A subscription's hold on the bus of its session. Dropped, it frees the bus.
struct Claim {
    session: Session,
    generation: u64,
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.session.listener_bus().free(self.generation);
    }
}

impl Claim {
    /// Removes the subscription on the server, from a task of `runtime`, and frees the bus once
    /// the server has answered; `subscription`, what is left of it, is dropped after that, so
    /// that the server ends its stream before the client releases it.
    fn remove(self, runtime: &tokio::runtime::Handle, subscription: impl Send + 'static) {
        self.session.listener_bus().ending(self.generation);
        let session = self.session.clone();
        runtime.spawn(async move {
            let _claim = self;
            let _ = stream::command(session, unsubscribe_command()).await;
            drop(subscription);
        });
    }
}

/// Tells the task of a subscription that the future that waits for it was dropped before the
/// task ended: its observer reports nothing more, and the session's next subscription waits for
/// the task.
struct Abandon {
    session: Session,
    generation: u64,
    abandoned: Arc<AtomicBool>,
    dropped: Option<oneshot::Sender<()>>,
}

impl Drop for Abandon {
    fn drop(&mut self) {
        if self.dropped.is_some() {
            self.abandoned.store(true, Ordering::Release);
            self.session.listener_bus().ending(self.generation);
        }
    }
}

/// Subscribes `session` to the events of its streaming queries, and returns them once the
/// server has acknowledged the subscription.
///
/// The subscription runs as a task of its own. Dropping the future before the server answers
/// ends it; once the server has answered, which it does only with its acknowledgement, the
/// task goes on, and the dropped stream removes the subscription.
pub(crate) async fn subscribe(session: Session) -> Result<QueryEvents> {
    let (generation, started) = session.listener_bus().claim().await?;
    let claim = Claim {
        session: session.clone(),
        generation,
    };
    let abandoned = Arc::new(AtomicBool::new(false));
    let plan = proto::plan::OpType::Command(proto::Command {
        command_type: Some(subscribe_command()),
    });
    let (request, _, observer) = stream::prepared(&session, plan, &abandoned)?;
    let (dropped, abandonment) = oneshot::channel::<()>();
    let mut abandon = Abandon {
        session: session.clone(),
        generation,
        abandoned,
        dropped: Some(dropped),
    };
    let client = session.client();
    let plan_ids = Arc::clone(session.plan_ids());
    let runtime = tokio::runtime::Handle::current();
    let task_runtime = runtime.clone();
    let subscription = runtime.spawn(async move {
        let execution = tokio::select! {
            biased;
            execution = Execution::start(client, plan_ids, request, observer) => execution?,
            _ = abandonment => return Ok(None),
        };
        let mut events = ListenerEvents::of(execution);
        match events.wait_until_listener_added().await {
            Ok(()) => Ok(Some(QueryEvents::new(claim, events, started, task_runtime))),
            Err(error) => {
                claim.remove(&task_runtime, events);
                Err(error)
            }
        }
    });
    let subscribed = subscription.await;
    abandon.dropped = None;
    match subscribed {
        Ok(Ok(Some(events))) => Ok(events),
        Ok(Ok(None)) => Err(SparkError::connect_msg("the subscription was abandoned").into()),
        Ok(Err(error)) => Err(error.into()),
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => Err(SparkError::connect_msg(error.to_string()).into()),
    }
}

type Next = (
    Box<ListenerEvents<Observer>>,
    Option<spark_connect_core::error::Result<(i32, String)>>,
);

enum State {
    Waiting(Box<ListenerEvents<Observer>>),
    Reading(BoxFuture<'static, Next>),
    Ended,
}

/// The events of the streaming queries of a session, as the server sends them, from
/// [`StreamingQueryManager::events`](super::StreamingQueryManager::events).
///
/// The started event of a query that the session starts comes when its start returns; the
/// other events come as the server sends them. An event that cannot be read is an error, and
/// the events after it still come.
///
/// The stream ends when the server ends the subscription, as it does when another client of
/// the same session removes it. If the subscription fails, the stream gives the error, then
/// ends, and the client removes the subscription, which the server may still hold. Dropping
/// the stream removes the subscription on the server, from the Tokio runtime that it was made
/// on; until the server has removed it, the next
/// [`events`](super::StreamingQueryManager::events) of the session waits.
pub struct QueryEvents {
    claim: Option<Claim>,
    started: mpsc::UnboundedReceiver<String>,
    state: State,
    runtime: tokio::runtime::Handle,
}

impl QueryEvents {
    fn new(
        claim: Claim,
        events: ListenerEvents<Observer>,
        started: mpsc::UnboundedReceiver<String>,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        QueryEvents {
            claim: Some(claim),
            started,
            state: State::Waiting(Box::new(events)),
            runtime,
        }
    }
}

impl Stream for QueryEvents {
    type Item = Result<QueryEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if matches!(this.state, State::Ended) {
            return Poll::Ready(None);
        }
        if let Poll::Ready(Some(json)) = this.started.poll_recv(cx) {
            return Poll::Ready(Some(started_of(json)));
        }
        loop {
            match std::mem::replace(&mut this.state, State::Ended) {
                State::Waiting(mut events) => {
                    this.state = State::Reading(
                        async move {
                            let next = events.next().await;
                            (events, next)
                        }
                        .boxed(),
                    );
                }
                State::Reading(mut reading) => match reading.poll_unpin(cx) {
                    Poll::Pending => {
                        this.state = State::Reading(reading);
                        return Poll::Pending;
                    }
                    Poll::Ready((events, Some(Ok((event_type, json))))) => {
                        this.state = State::Waiting(events);
                        return Poll::Ready(Some(event_of(event_type, json)));
                    }
                    Poll::Ready((events, Some(Err(error)))) => {
                        // The server may still hold a subscription whose stream failed.
                        if let Some(claim) = this.claim.take() {
                            claim.remove(&this.runtime, events);
                        }
                        return Poll::Ready(Some(Err(error.into())));
                    }
                    Poll::Ready((_, None)) => {
                        this.claim = None;
                        return Poll::Ready(None);
                    }
                },
                State::Ended => return Poll::Ready(None),
            }
        }
    }
}

impl Drop for QueryEvents {
    fn drop(&mut self) {
        if let Some(claim) = self.claim.take() {
            let subscription = std::mem::replace(&mut self.state, State::Ended);
            claim.remove(&self.runtime, subscription);
        }
    }
}
