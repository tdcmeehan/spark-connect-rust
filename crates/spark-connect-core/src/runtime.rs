//! The interruption of the waits for the server by a signal of the process: a thread sets the
//! watch that its waits poll ([`with_wait_watch`]), and the futures of a wait that
//! [`until_abort`] covers end at the level of abort that the wait reaches. The waits themselves
//! are those of a synchronous front end, such as `apache-spark-connect`.

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// What a wait for the server does at a poll of the signals of the process (see
/// [`with_wait_watch`]).
pub enum Watch {
    /// Keep waiting for the server.
    Continue,
    /// Interrupt the execution that the wait is for, by this future, which the wait drives to its
    /// end. The wait then ends with the server's cancellation of the execution.
    Interrupt(Pin<Box<dyn Future<Output = ()> + Send>>),
    /// Stop waiting, when no execution is in flight to interrupt. An abortable wait, for a call
    /// that nothing on the server stands behind, ends at once, without the output of its future;
    /// any other wait keeps waiting, as for [`Watch::Continue`].
    Abort,
}

/// The poll of the signals of the process of a wait, which the wait calls every
/// [`POLL_INTERVAL`] on the thread that waits.
pub type WaitWatch = dyn Fn() -> Watch + Send + Sync;

/// The level of [`until_abort`] that a wait reaches once the first signal of the process
/// arrives while an execution of the wait is in flight: the retries, the backoffs and the RPCs
/// that have not been answered end at once, as PySpark's retry sleeps raise KeyboardInterrupt,
/// while the responses of a streaming execution are still read, so that the server's
/// cancellation of it ends them.
pub const ABORT_RETRIES: u8 = 1;

/// The level of [`until_abort`] that every wait reaches, when a signal arrives while no execution
/// of the wait is in flight, or a second signal arrives after the interrupt of the execution.
pub const ABORT_ALL: u8 = 2;

thread_local! {
    static WAIT_WATCH: RefCell<Option<Arc<WaitWatch>>> = const { RefCell::new(None) };
    /// The abort level of the wait that the calling thread runs, see [`until_abort`].
    static WAIT_ABORT: RefCell<Option<tokio::sync::watch::Receiver<u8>>> =
        const { RefCell::new(None) };
}

/// Sets the abort level of a wait on this thread, which [`until_abort`] reads, until it drops.
#[doc(hidden)]
pub struct AbortScope(Option<tokio::sync::watch::Receiver<u8>>);

impl AbortScope {
    pub fn enter(receiver: tokio::sync::watch::Receiver<u8>) -> Self {
        AbortScope(WAIT_ABORT.with(|abort| abort.replace(Some(receiver))))
    }
}

impl Drop for AbortScope {
    fn drop(&mut self) {
        let previous = self.0.take();
        WAIT_ABORT.with(|abort| *abort.borrow_mut() = previous);
    }
}

/// `f`, which a wait of the calling thread ends at once, dropping it unfinished, when the wait
/// reaches `level` (see [`ABORT_RETRIES`] and [`ABORT_ALL`]): None then, else the output of `f`.
/// Outside a wait that a watch polls, `f` is awaited as it is.
pub async fn until_abort<F: std::future::Future>(level: u8, f: F) -> Option<F::Output> {
    let receiver = WAIT_ABORT.with(|abort| abort.borrow().clone());
    let Some(mut receiver) = receiver else {
        return Some(f.await);
    };
    tokio::pin!(f);
    tokio::select! {
        biased;
        _ = async {
            // A wait whose sender is gone is not aborted by it.
            if receiver.wait_for(|reached| *reached >= level).await.is_err() {
                std::future::pending::<()>().await;
            }
        } => None,
        output = &mut f => Some(output),
    }
}

/// Run `f` on this thread, with `watch` called by the waits for the server that `f` makes on this
/// thread: the waits poll `watch` every [`POLL_INTERVAL`] while the future of each waits. The
/// waits of other threads do not poll it. The watch of `f` ends with it, and the watch of an
/// enclosing call is restored.
pub fn with_wait_watch<R>(watch: Arc<WaitWatch>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Arc<WaitWatch>>);

    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            WAIT_WATCH.with(|current| *current.borrow_mut() = previous);
        }
    }

    let previous = WAIT_WATCH.with(|current| current.replace(Some(watch)));
    let _restore = Restore(previous);
    f()
}

/// The watch that [`with_wait_watch`] set on the calling thread, if any.
#[doc(hidden)]
pub fn wait_watch() -> Option<Arc<WaitWatch>> {
    WAIT_WATCH.with(|watch| watch.borrow().clone())
}

/// How often a wait for the server polls its [`WaitWatch`].
pub const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_wait_watch_restores_the_enclosing_watch() {
        let outer: Arc<WaitWatch> = Arc::new(|| Watch::Continue);
        let is_outer =
            |w: &Option<Arc<WaitWatch>>| w.as_ref().is_some_and(|w| Arc::ptr_eq(w, &outer));
        with_wait_watch(Arc::clone(&outer), || {
            with_wait_watch(Arc::new(|| Watch::Continue), || {});
            assert!(WAIT_WATCH.with(|w| is_outer(&w.borrow())));
        });
        assert!(WAIT_WATCH.with(|w| w.borrow().is_none()));
    }
}
