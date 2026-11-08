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
}
