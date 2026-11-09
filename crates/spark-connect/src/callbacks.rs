//! The bound on the callbacks that run nested on a thread: the progress handlers and the hooks
//! of the executions of a session, each of which may run a query of its own.

use spark_connect_core::error;

/// How many progress handlers and hooks may run nested on a thread, each one running a query
/// that runs the next one. Past it, an execution fails with a `RecursionError` (as PySpark's
/// recursion limit raises it) instead of recursing until the native stack of the thread
/// overflows. A level takes about 100 KiB of native stack (Python frames of the handler, and
/// the calls of PyO3 between them): on a thread of 512 KiB, four levels crash, so two levels
/// leave room for the frames of the thread itself.
pub(crate) const MAX_CALLBACK_DEPTH: usize = 2;

thread_local! {
    static CALLBACK_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Set once an execution failed past the bound, until the outermost callback returns: the
    /// callbacks of the levels above then fail their queries at once, rather than each level
    /// running a query per progress update again, which would grow exponentially.
    static CALLBACK_RECURSION_FAILED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A progress handler or a hook that runs on this thread, for as long as it lives.
pub(crate) struct CallbackGuard(());

impl CallbackGuard {
    /// Enter a callback that may run a query of its own.
    pub(crate) fn enter() -> Self {
        CALLBACK_DEPTH.with(|depth| depth.set(depth.get() + 1));
        CallbackGuard(())
    }
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        let depth = CALLBACK_DEPTH.with(|depth| {
            depth.set(depth.get() - 1);
            depth.get()
        });
        if depth == 0 {
            CALLBACK_RECURSION_FAILED.with(|failed| failed.set(false));
        }
    }
}

/// Fail with a `RecursionError` when the executions started from this thread are nested past
/// [`MAX_CALLBACK_DEPTH`] callbacks, instead of the stack of the thread running out.
#[allow(clippy::result_large_err)] // SparkError is large, as in the other entry points
pub(crate) fn check_callback_depth() -> error::Result<()> {
    let too_deep = CALLBACK_DEPTH.with(|depth| depth.get()) >= MAX_CALLBACK_DEPTH;
    if too_deep || CALLBACK_RECURSION_FAILED.with(|failed| failed.get()) {
        CALLBACK_RECURSION_FAILED.with(|failed| failed.set(true));
        return Err(error::SparkError::recursion_msg(
            "maximum recursion depth exceeded in a progress handler or a hook of an execution",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callback_depth_is_bounded_and_released() {
        assert!(check_callback_depth().is_ok());
        let guards: Vec<CallbackGuard> = (0..MAX_CALLBACK_DEPTH)
            .map(|_| CallbackGuard::enter())
            .collect();
        let err = check_callback_depth().unwrap_err();
        assert_eq!(err.kind, error::SparkErrorKind::RecursionError);
        drop(guards);
        assert!(check_callback_depth().is_ok());
    }
}
