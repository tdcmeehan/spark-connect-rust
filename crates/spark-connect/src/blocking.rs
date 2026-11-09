//! The runtime of the synchronous API: one process-wide [`Blocking`] runs its calls. Not a
//! stable API.

use std::future::Future;
use std::sync::OnceLock;

pub use spark_connect_blocking::Blocking;

/// The process-wide [`Blocking`] of the synchronous API.
pub fn shared() -> &'static Blocking {
    static SHARED: OnceLock<Blocking> = OnceLock::new();
    SHARED.get_or_init(|| {
        Blocking::new().expect("failed to create the runtime of the synchronous API")
    })
}

/// Runs `future` to completion on the calling thread, with [`shared`].
pub fn block_on<F: Future>(future: F) -> F::Output {
    shared().block_on(future)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::callbacks::{check_callback_depth, CallbackGuard, MAX_CALLBACK_DEPTH};
    use shared as get_runtime;

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
}
