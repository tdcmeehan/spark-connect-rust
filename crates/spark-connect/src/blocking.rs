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
