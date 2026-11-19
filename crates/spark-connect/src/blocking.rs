//! The runtime of the synchronous API: one process-wide [`Blocking`] runs its calls. Not a
//! stable API.

use std::future::Future;
use std::sync::{Arc, OnceLock};

pub use spark_connect_blocking::Blocking;
pub use spark_connect_core::runtime::POLL_INTERVAL;
use spark_connect_core::runtime::{self, WaitWatch, Watch, ABORT_ALL, ABORT_RETRIES};

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
            shared().block_on(watched(future, watch, abort))
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
            shared().block_on(watched_abortable(future, watch, abort))
        }
        None => Some(shared().block_on(future)),
    }
}

/// `f`, polled as it is, and `watch` called every [`POLL_INTERVAL`]: `None` at once when it
/// returns [`Watch::Abort`], else the output of `f`. The first interrupt that `watch` returns is
/// spawned and awaited with `f`, as in [`watched`], and the wait reaches [`ABORT_RETRIES`], which
/// ends the backoffs of `f` (see [`until_abort`]), as PySpark's retry sleeps raise
/// KeyboardInterrupt at the first signal. The polls go on after it, so that a second signal ends
/// the wait even while the interrupt hangs, and the running interrupt is cancelled with the wait.
async fn watched_abortable<F: Future>(
    f: F,
    watch: Arc<WaitWatch>,
    abort: tokio::sync::watch::Sender<u8>,
) -> Option<F::Output> {
    tokio::pin!(f);
    let mut ticks = tokio::time::interval(POLL_INTERVAL);
    let mut interrupt: Option<tokio::task::JoinHandle<()>> = None;
    loop {
        tokio::select! {
            output = &mut f => {
                // `f` has ended, so the wait only awaits the interrupt; a second signal, polled
                // meanwhile, ends that too.
                if let Some(mut interrupt) = interrupt {
                    loop {
                        tokio::select! {
                            _ = &mut interrupt => break,
                            _ = ticks.tick() => {
                                if let Watch::Abort = watch() {
                                    interrupt.abort();
                                    break;
                                }
                            }
                        }
                    }
                }
                return Some(output);
            }
            _ = ticks.tick() => match watch() {
                Watch::Abort => {
                    if let Some(interrupt) = interrupt {
                        interrupt.abort();
                    }
                    return None;
                }
                Watch::Interrupt(sent) => {
                    abort.send_replace(ABORT_RETRIES);
                    if interrupt.is_none() {
                        interrupt = Some(tokio::spawn(sent));
                    }
                }
                Watch::Continue => {}
            },
        }
    }
}

/// `f`, polled as it is, and `watch` called every [`POLL_INTERVAL`]. The first interrupt that it
/// returns is spawned, and the wait reaches [`ABORT_RETRIES`]: `f` is driven on, and the interrupt
/// is awaited with it, so that the cleanup of `f` runs as the server ends the execution. A wait
/// that `watch` aborts reaches [`ABORT_ALL`], which ends the waits of `f` that [`until_abort`]
/// covers, and the interrupt that is still running. The polls go on after an interrupt, so that
/// a second signal aborts the wait.
async fn watched<F: Future>(
    f: F,
    watch: Arc<WaitWatch>,
    abort: tokio::sync::watch::Sender<u8>,
) -> F::Output {
    tokio::pin!(f);
    let mut ticks = tokio::time::interval(POLL_INTERVAL);
    let mut interrupt: Option<tokio::task::JoinHandle<()>> = None;
    // Whether `watch` aborted the wait, which a second signal after an interrupt does.
    let mut aborted = false;
    loop {
        tokio::select! {
            output = &mut f => {
                // The interrupt is awaited, and a second signal, polled meanwhile, ends the wait
                // of it: the interrupt is then cancelled. When that signal already aborted the
                // wait, which ended `f`, the interrupt is cancelled at once, as PySpark stops at
                // the second signal while an interrupt hangs or is retried.
                if let Some(mut interrupt) = interrupt {
                    if aborted {
                        interrupt.abort();
                        return output;
                    }
                    loop {
                        tokio::select! {
                            _ = &mut interrupt => break,
                            _ = ticks.tick() => {
                                if let Watch::Abort = watch() {
                                    abort.send_replace(ABORT_ALL);
                                    interrupt.abort();
                                    break;
                                }
                            }
                        }
                    }
                }
                return output;
            }
            _ = ticks.tick() => match watch() {
                Watch::Continue => {}
                Watch::Interrupt(sent) => {
                    abort.send_replace(ABORT_RETRIES);
                    if interrupt.is_none() {
                        interrupt = Some(tokio::spawn(sent));
                    }
                }
                Watch::Abort => {
                    abort.send_replace(ABORT_ALL);
                    aborted = true;
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::callbacks::{check_callback_depth, CallbackGuard, MAX_CALLBACK_DEPTH};
    use shared as get_runtime;
    use spark_connect_core::retries::{retrying, Backoff, RetryError, RetryPolicy, Retrying};
    use spark_connect_core::runtime::{until_abort, with_wait_watch};

    #[test]
    fn test_runtime_created_once() {
        let rt1 = get_runtime();
        let rt2 = get_runtime();
        // Both should point to the same runtime
        assert_eq!(rt1 as *const _, rt2 as *const _);
    }

    #[test]
    fn test_block_on_works() {
        let result = block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_block_on_nested_in_block_on() {
        // A callback of a running execution blocks on a nested query, from within the runtime.
        let result = block_on(async { block_on(async { 7 }) + 1 });
        assert_eq!(result, 8);
    }

    #[test]
    fn test_block_on_nested_on_a_worker_thread() {
        let result =
            block_on(async { tokio::spawn(async { block_on(async { 5 }) }).await.unwrap() });
        assert_eq!(result, 5);
    }

    #[test]
    fn test_block_on_async_task() {
        let result = block_on(async {
            let x = 10;
            let y = 20;
            x + y
        });
        assert_eq!(result, 30);
    }

    /// A future of a large state, as the future of a request is of the state of its gRPC call,
    /// is driven from the heap: a thread of a small stack, which holds it once, holds no other
    /// copy of it in the frames that drive it, whatever the compiler inlines. (The frames of a
    /// debug build copy more, so its thread has the stack of a default thread of Linux.)
    #[test]
    fn test_block_on_holds_no_copy_of_a_large_future_on_a_small_stack() {
        const STATE: usize = 192 << 10;
        let stack = if cfg!(debug_assertions) {
            8 << 20
        } else {
            512 << 10
        };
        let output = std::thread::Builder::new()
            .stack_size(stack)
            .spawn(|| {
                block_on(async {
                    let state = [7u8; STATE];
                    tokio::task::yield_now().await;
                    state[STATE - 1]
                })
            })
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(output, 7);
    }

    #[test]
    fn test_watch_is_called_while_the_future_waits() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let watch: Arc<WaitWatch> = Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Watch::Continue
        });
        let output = with_wait_watch(watch, || {
            block_on(async {
                tokio::time::sleep(POLL_INTERVAL * 5).await;
                7
            })
        });
        assert_eq!(output, 7);
        assert!(calls.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn test_interrupt_of_a_watch_is_driven_to_the_end_of_the_wait() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let fired = AtomicBool::new(false);
        let interrupted = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&interrupted);
        let watch: Arc<WaitWatch> = Arc::new(move || {
            if fired.swap(true, Ordering::SeqCst) {
                Watch::Continue
            } else {
                let flag = Arc::clone(&flag);
                Watch::Interrupt(Box::pin(async move { flag.store(true, Ordering::SeqCst) }))
            }
        });
        let output = with_wait_watch(watch, || {
            block_on(async {
                tokio::time::sleep(POLL_INTERVAL * 5).await;
                9
            })
        });
        assert_eq!(output, 9);
        assert!(interrupted.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_a_second_signal_ends_the_wait_without_awaiting_a_hung_interrupt() {
        // The first signal interrupts, and the interrupt never ends, as against a server that
        // does not answer it. The second aborts the wait, which ends its future, and the wait
        // returns at once, cancelling the interrupt, without the third signal that comes later.
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let polls = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let (counter, flag) = (Arc::clone(&polls), Arc::clone(&cancelled));
        let watch: Arc<WaitWatch> =
            Arc::new(move || match counter.fetch_add(1, Ordering::SeqCst) {
                0 => {
                    let dropped = Dropped(Arc::clone(&flag));
                    Watch::Interrupt(Box::pin(async move {
                        let _dropped = dropped;
                        std::future::pending::<()>().await
                    }))
                }
                1 => Watch::Abort,
                n if n >= 10 => Watch::Abort,
                _ => Watch::Continue,
            });
        let output = with_wait_watch(watch, || {
            block_on(until_abort(ABORT_ALL, std::future::pending::<u8>()))
        });
        assert_eq!(output, None);
        assert!(
            polls.load(Ordering::SeqCst) < 10,
            "the wait needed a third signal"
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !cancelled.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            cancelled.load(Ordering::SeqCst),
            "the interrupt was not cancelled"
        );
    }

    /// The output of `block_on_abortable(f)` under `watch`, run on a thread of its own, or None
    /// when the wait has not ended within 5 s: a wait that is not ended then fails the test,
    /// rather than hanging it.
    fn abortable_wait_under<F>(watch: Arc<WaitWatch>, f: F) -> Option<Option<F::Output>>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let output = with_wait_watch(watch, || block_on_abortable(f));
            let _ = sender.send(output);
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .ok()
    }

    #[test]
    fn test_a_second_signal_ends_an_abortable_wait_while_its_interrupt_hangs() {
        // As `watched`: the first signal interrupts, and the interrupt never ends. The second
        // signal aborts the wait, which ends at once without the output of its future, and the
        // interrupt is cancelled.
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let polls = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let (counter, flag) = (Arc::clone(&polls), Arc::clone(&cancelled));
        let watch: Arc<WaitWatch> =
            Arc::new(move || match counter.fetch_add(1, Ordering::SeqCst) {
                0 => {
                    let dropped = Dropped(Arc::clone(&flag));
                    Watch::Interrupt(Box::pin(async move {
                        let _dropped = dropped;
                        std::future::pending::<()>().await
                    }))
                }
                1 => Watch::Abort,
                _ => Watch::Continue,
            });
        let output = abortable_wait_under(watch, std::future::pending::<u8>());
        assert_eq!(
            output,
            Some(None),
            "the wait did not end at the second signal"
        );
        assert_eq!(polls.load(Ordering::SeqCst), 2);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !cancelled.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            cancelled.load(Ordering::SeqCst),
            "the interrupt was not cancelled"
        );
    }

    #[test]
    fn test_a_second_signal_ends_an_abortable_wait_while_its_future_waits_after_the_interrupt() {
        // The interrupt ends at once, and the future then waits, as a release that the server
        // does not answer: the first signal alone does not end the wait, the second one does.
        let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&polls);
        let watch: Arc<WaitWatch> =
            Arc::new(
                move || match counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
                    0 => Watch::Interrupt(Box::pin(async {})),
                    1 => Watch::Continue,
                    _ => Watch::Abort,
                },
            );
        let output = abortable_wait_under(watch, std::future::pending::<u8>());
        assert_eq!(
            output,
            Some(None),
            "the wait did not end at the second signal"
        );
        assert_eq!(polls.load(std::sync::atomic::Ordering::SeqCst), 3);

        // Without a second signal, the output of the future is still given after the interrupt.
        let watch: Arc<WaitWatch> = Arc::new({
            let first = std::sync::atomic::AtomicBool::new(true);
            move || {
                if first.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    Watch::Interrupt(Box::pin(async {}))
                } else {
                    Watch::Continue
                }
            }
        });
        let ended = abortable_wait_under(watch, async {
            tokio::time::sleep(POLL_INTERVAL * 2).await;
            7_u8
        });
        assert_eq!(ended, Some(Some(7)));
    }

    #[test]
    fn test_a_second_signal_after_the_future_ends_cancels_the_interrupt_and_gives_the_output() {
        // The first signal interrupts, and the interrupt never ends; the future then ends. The
        // second signal ends the wait for the interrupt, which is cancelled, and the wait gives the
        // output of the future, which has ended.
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let polls = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let (counter, flag) = (Arc::clone(&polls), Arc::clone(&cancelled));
        let watch: Arc<WaitWatch> =
            Arc::new(move || match counter.fetch_add(1, Ordering::SeqCst) {
                0 => {
                    let dropped = Dropped(Arc::clone(&flag));
                    Watch::Interrupt(Box::pin(async move {
                        let _dropped = dropped;
                        std::future::pending::<()>().await
                    }))
                }
                _ => Watch::Abort,
            });
        let output = abortable_wait_under(watch, async {
            tokio::time::sleep(POLL_INTERVAL / 2).await;
            7_u8
        });
        assert_eq!(
            output,
            Some(Some(7)),
            "the wait did not give the output of its ended future"
        );
        assert_eq!(polls.load(Ordering::SeqCst), 2);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !cancelled.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            cancelled.load(Ordering::SeqCst),
            "the interrupt was not cancelled"
        );
    }

    #[test]
    fn test_an_abortable_wait_ends_when_its_watch_aborts_and_another_keeps_waiting() {
        // A watch that aborts ends an abortable wait at once, without the output of its future.
        let aborted = with_wait_watch(Arc::new(|| Watch::Abort), || {
            block_on_abortable(std::future::pending::<u8>())
        });
        assert_eq!(aborted, None);
        // Without an abort, an abortable wait gives the output of its future.
        let ended = with_wait_watch(Arc::new(|| Watch::Continue), || {
            block_on_abortable(async {
                tokio::time::sleep(POLL_INTERVAL * 2).await;
                7
            })
        });
        assert_eq!(ended, Some(7));
        assert_eq!(block_on_abortable(async { 8 }), Some(8));
        // Any other wait keeps waiting for its future, whatever the watch returns.
        let waited = with_wait_watch(Arc::new(|| Watch::Abort), || {
            block_on(async {
                tokio::time::sleep(POLL_INTERVAL * 2).await;
                9
            })
        });
        assert_eq!(waited, 9);
    }

    #[test]
    fn test_until_abort_ends_the_waits_that_the_watch_aborts_at_their_level() {
        // A wait of the watch's abort at ABORT_ALL ends at once, dropping its future unfinished.
        let aborted = with_wait_watch(Arc::new(|| Watch::Abort), || {
            block_on(until_abort(ABORT_ALL, std::future::pending::<u8>()))
        });
        assert_eq!(aborted, None);
        // An interrupt reaches ABORT_RETRIES only: the retries end, and the interrupt is awaited.
        let interrupted = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&interrupted);
        let retried = with_wait_watch(
            Arc::new(move || {
                let flag = Arc::clone(&flag);
                Watch::Interrupt(Box::pin(async move {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                }))
            }),
            || block_on(until_abort(ABORT_RETRIES, std::future::pending::<u8>())),
        );
        assert_eq!(retried, None);
        assert!(interrupted.load(std::sync::atomic::Ordering::SeqCst));
        // Outside a watch, a future is awaited as it is.
        assert_eq!(block_on(until_abort(ABORT_ALL, async { 8 })), Some(8));
    }

    fn current_thread_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn test_block_on_inside_current_thread_runtime() {
        // The default flavor of `#[tokio::test]`: the sync API blocks the thread, which the
        // runtime cannot block in place.
        let result = current_thread_runtime().block_on(async { block_on(async { 42 }) });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_block_on_nested_inside_current_thread_runtime() {
        // A callback of the execution runs a query, which runs inside the current_thread runtime.
        let result = current_thread_runtime()
            .block_on(async { block_on(async { block_on(async { 7 }) + 1 }) });
        assert_eq!(result, 8);
    }

    #[test]
    fn test_block_on_inside_current_thread_runtime_keeps_caller_thread() {
        // The per-thread state (tags and extensions) of the caller is read as on its own thread.
        let caller = std::thread::current().id();
        let seen = current_thread_runtime()
            .block_on(async { block_on(async { std::thread::current().id() }) });
        assert_eq!(seen, caller);
        assert_eq!(std::thread::current().id(), caller);
    }

    #[test]
    fn test_block_on_inside_current_thread_runtime_keeps_callback_depth() {
        // A call from a callback at the bound fails with a RecursionError on the other thread too.
        let guards: Vec<CallbackGuard> = (0..MAX_CALLBACK_DEPTH)
            .map(|_| CallbackGuard::enter())
            .collect();
        let err = current_thread_runtime()
            .block_on(async { block_on(async { check_callback_depth() }) })
            .unwrap_err();
        assert_eq!(
            err.kind,
            spark_connect_core::error::SparkErrorKind::RecursionError
        );
        drop(guards);
        // The failure is released with the outermost callback, as on this thread.
        assert!(current_thread_runtime()
            .block_on(async { block_on(async { check_callback_depth() }) })
            .is_ok());
    }

    #[test]
    fn test_block_on_inside_current_thread_runtime_propagates_panic() {
        let result = std::panic::catch_unwind(|| {
            current_thread_runtime().block_on(async {
                block_on(async { panic!("from the future") });
            })
        });
        assert!(result.is_err());
    }

    /// A policy of `max_retries` retries with waits of 1 ms, so that a test does not wait.
    fn quick_policy(max_retries: u32) -> RetryPolicy {
        RetryPolicy {
            max_retries: Some(max_retries),
            initial_backoff_ms: 1,
            max_backoff_ms: Some(1),
            backoff_multiplier: 1.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
            max_retry_exception_elapsed_time_ms: None,
        }
    }

    fn unavailable<T>() -> Result<T, RetryError<&'static str>> {
        Err(RetryError::Status(tonic::Status::unavailable("restarting")))
    }

    /// A policy whose one backoff, of a minute, a test does not wait out.
    fn slow_policy() -> RetryPolicy {
        RetryPolicy {
            initial_backoff_ms: 60_000,
            max_backoff_ms: Some(60_000),
            ..quick_policy(1)
        }
    }

    #[test]
    fn test_a_signal_ends_the_backoff_of_retries_that_end_with_an_abort() {
        let down = tonic::Status::unavailable("down");
        let started = std::time::Instant::now();
        let ended = with_wait_watch(std::sync::Arc::new(|| Watch::Abort), || {
            block_on(async {
                let mut retrying = Retrying::new(&slow_policy()).ending_with_abort();
                retrying.backoff(&down).await
            })
        });
        assert_eq!(ended, Backoff::Aborted);
        assert!(started.elapsed() < std::time::Duration::from_secs(30));

        // The retry after the backoff is not made, and its check is not run.
        let mut attempts = 0;
        let result: Result<(), RetryError<&'static str>> =
            with_wait_watch(std::sync::Arc::new(|| Watch::Abort), || {
                block_on(retrying(
                    &mut Retrying::new(&slow_policy()).ending_with_abort(),
                    || panic!("no check before a retry that is not made"),
                    || {
                        attempts += 1;
                        async { unavailable() }
                    },
                ))
            });
        assert!(matches!(result, Err(RetryError::WaitAborted)), "{result:?}");
        assert_eq!(attempts, 1);
    }

    #[test]
    fn test_an_interrupt_ends_the_backoff_of_an_abortable_wait_as_the_first_signal_does() {
        // The release of `stop()` backs off inside `block_on_abortable`, which an interrupt of
        // the execution in flight leaves running. The first interrupt ends the backoff at once,
        // as PySpark's sleep raises KeyboardInterrupt, and a future without backoffs is not ended
        // by it. The wait runs on a thread of its own, so that a backoff that is not ended fails
        // the test at its timeout rather than after the whole minute.
        use std::sync::atomic::{AtomicBool, Ordering};
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let first = std::sync::Arc::new(AtomicBool::new(true));
            let watch: std::sync::Arc<WaitWatch> = std::sync::Arc::new(move || {
                if first.swap(false, Ordering::SeqCst) {
                    Watch::Interrupt(Box::pin(async {}))
                } else {
                    Watch::Continue
                }
            });
            let ended = with_wait_watch(watch, || {
                block_on_abortable(async {
                    let mut retrying = Retrying::new(&slow_policy()).ending_with_abort();
                    retrying.backoff(&tonic::Status::unavailable("down")).await
                })
            });
            let _ = sender.send(ended);
        });
        let ended = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the interrupt did not end the backoff");
        assert_eq!(ended, Some(Backoff::Aborted));

        // A future that waits without backoffs is given its output after the interrupt.
        let watch: std::sync::Arc<WaitWatch> = std::sync::Arc::new({
            let first = AtomicBool::new(true);
            move || {
                if first.swap(false, Ordering::SeqCst) {
                    Watch::Interrupt(Box::pin(async {}))
                } else {
                    Watch::Continue
                }
            }
        });
        let kept = with_wait_watch(watch, || {
            block_on_abortable(async {
                tokio::time::sleep(POLL_INTERVAL * 2).await;
                5
            })
        });
        assert_eq!(kept, Some(5));
    }

    #[test]
    fn test_a_signal_does_not_end_the_backoff_of_retries_that_are_waited_whole() {
        // Longer than the poll of the watch, which aborts at its first poll.
        let policy = RetryPolicy {
            initial_backoff_ms: (POLL_INTERVAL * 2).as_millis() as u64,
            max_backoff_ms: None,
            ..quick_policy(1)
        };
        let down = tonic::Status::unavailable("down");
        let ended = with_wait_watch(std::sync::Arc::new(|| Watch::Abort), || {
            block_on(async {
                let mut retrying = Retrying::new(&policy);
                retrying.backoff(&down).await
            })
        });
        assert_eq!(ended, Backoff::Retry);
    }
}
