//! The runtime of the synchronous API: one process-wide [`Blocking`] runs its calls. Not a
//! stable API.

use std::future::Future;
use std::sync::OnceLock;

pub use spark_connect_blocking::Blocking;
use spark_connect_core::runtime;

/// The process-wide [`Blocking`] of the synchronous API.
pub fn shared() -> &'static Blocking {
    static SHARED: OnceLock<Blocking> = OnceLock::new();
    SHARED.get_or_init(|| {
        Blocking::new().expect("failed to create the runtime of the synchronous API")
    })
}

/// Runs `future` to completion on the calling thread, with [`shared`], also inside a Tokio
/// runtime of either flavor. The wait polls the watch of the thread, if
/// [`with_wait_watch`](runtime::with_wait_watch) set one: an interrupt that it returns is driven
/// to the end of the wait, and an abort ends the waits of `future` that
/// [`until_abort`](runtime::until_abort) covers.
pub fn block_on<F: Future>(future: F) -> F::Output {
    // Boxed first, so that the frames that move it on to the runtime hold only its pointer.
    let future = Box::pin(future);
    match runtime::wait_watch() {
        Some(watch) => {
            let (abort, receiver) = tokio::sync::watch::channel(0_u8);
            let _scope = runtime::AbortScope::enter(receiver);
            shared().block_on(runtime::watched(future, watch, abort))
        }
        None => shared().block_on(future),
    }
}

/// As [`block_on`], but the wait ends at once, with `None`, when the watch of the thread returns
/// [`Watch::Abort`](runtime::Watch::Abort) while the future runs: the future is dropped,
/// unfinished. If the future has ended and the wait only awaits the interrupt that a
/// [`Watch::Interrupt`](runtime::Watch::Interrupt) started, an Abort ends that wait instead,
/// cancelling the interrupt, and the output of the future is given. For a wait that nothing on
/// the server stands behind, such as the retries of a session release, which a signal of the
/// process stops as it stops PySpark's sleep between the retries.
pub fn block_on_abortable<F: Future>(future: F) -> Option<F::Output> {
    // Boxed first, as in `block_on`.
    let future = Box::pin(future);
    match runtime::wait_watch() {
        Some(watch) => {
            let (abort, receiver) = tokio::sync::watch::channel(0_u8);
            let _scope = runtime::AbortScope::enter(receiver);
            shared().block_on(runtime::watched_abortable(future, watch, abort))
        }
        None => Some(shared().block_on(future)),
    }
}
