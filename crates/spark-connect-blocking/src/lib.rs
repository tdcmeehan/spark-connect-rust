//! Runs the async actions of the Spark Connect client to completion from synchronous code.
//!
//! A [`Blocking`] owns a multi-thread Tokio runtime, whose threads drive the I/O and the timers
//! of the futures it runs. [`Blocking::block_on`] polls the future on the calling thread, inside
//! the runtime's context, and parks the thread between polls:
//!
//! - it never panics, outside any runtime or inside one of either flavor;
//! - the code a future runs, callbacks included, runs on the calling thread, so it sees that
//!   thread's thread-locals, and may call `block_on` again;
//! - the future may borrow from the caller and need not be `Send`.
//!
//! Blocking a thread of another runtime stops that runtime's other tasks until the call returns,
//! as any blocking call does.

use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

use tokio::runtime::{Builder, Handle, Runtime};

/// Runs futures to completion from synchronous code, on a runtime of its own.
#[derive(Debug)]
pub struct Blocking {
    runtime: Option<Runtime>,
}

impl Blocking {
    /// A new runtime for the futures this runs.
    pub fn new() -> std::io::Result<Self> {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .thread_name("spark-connect")
            .build()?;
        Ok(Blocking {
            runtime: Some(runtime),
        })
    }

    /// The handle of the runtime, to spawn on it the work that outlives a call.
    pub fn handle(&self) -> &Handle {
        self.runtime
            .as_ref()
            .expect("the runtime lives until drop")
            .handle()
    }

    /// Runs `future` to completion on the calling thread and returns its output.
    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        let _context = self.handle().enter();
        // A task of another runtime may have used up its budget: the resources this future
        // polls must not see it, or they would return pending until that task yields.
        let mut future = pin!(tokio::task::unconstrained(future));
        let wake = Arc::new(ThreadWake {
            thread: thread::current(),
            woken: AtomicBool::new(false),
        });
        let waker = Waker::from(Arc::clone(&wake));
        let mut context = Context::from_waker(&waker);
        loop {
            if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
                return output;
            }
            // A nested call on this thread may take the park token of this one's wake: the
            // flag keeps the wake.
            while !wake.woken.swap(false, Ordering::Acquire) {
                thread::park();
            }
        }
    }
}

impl Drop for Blocking {
    fn drop(&mut self) {
        // Shutting down in place would block, which panics inside a runtime.
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

struct ThreadWake {
    thread: Thread,
    woken: AtomicBool,
}

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.woken.store(true, Ordering::Release);
        self.thread.unpark();
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    /// A future that is woken from another thread: a timer of the helper's runtime.
    async fn after_a_timer<T>(value: T) -> T {
        tokio::time::sleep(Duration::from_millis(5)).await;
        value
    }

    #[test]
    fn runs_a_future_outside_any_runtime() {
        let blocking = Blocking::new().unwrap();
        assert_eq!(blocking.block_on(after_a_timer(7)), 7);
    }

    #[test]
    fn does_not_panic_inside_a_current_thread_runtime() {
        let blocking = Blocking::new().unwrap();
        // No I/O or time driver: the future's timer runs on the helper's runtime.
        let runtime = Builder::new_current_thread().build().unwrap();
        let value = runtime.block_on(async { blocking.block_on(after_a_timer(3)) });
        assert_eq!(value, 3);
    }

    #[test]
    fn does_not_panic_inside_a_multi_thread_runtime() {
        let blocking = Arc::new(Blocking::new().unwrap());
        let runtime = Builder::new_multi_thread().enable_all().build().unwrap();
        let on_block_on = runtime.block_on(async { blocking.block_on(after_a_timer(1)) });
        let in_task = runtime.block_on({
            let blocking = Arc::clone(&blocking);
            async move {
                tokio::spawn(async move { blocking.block_on(after_a_timer(2)) })
                    .await
                    .unwrap()
            }
        });
        assert_eq!((on_block_on, in_task), (1, 2));
    }

    #[test]
    fn code_after_a_wake_runs_on_the_calling_thread() {
        let blocking = Blocking::new().unwrap();
        let caller = thread::current().id();
        let ran_on = blocking.block_on(async {
            after_a_timer(()).await;
            thread::current().id()
        });
        assert_eq!(ran_on, caller);

        // Also when the caller is the thread of a current-thread runtime.
        let runtime = Builder::new_current_thread().build().unwrap();
        let ran_on = runtime.block_on(async {
            blocking.block_on(async {
                after_a_timer(()).await;
                thread::current().id()
            })
        });
        assert_eq!(ran_on, caller);
    }

    #[test]
    fn a_callback_can_block_on_a_nested_future() {
        let blocking = Blocking::new().unwrap();
        let callback = |value: i32| blocking.block_on(after_a_timer(value + 1));
        let value = blocking.block_on(async {
            let first = after_a_timer(1).await;
            let nested = callback(first);
            after_a_timer(nested + 1).await
        });
        assert_eq!(value, 3);
    }

    #[test]
    fn a_wake_taken_by_a_nested_call_is_not_lost() {
        let blocking = Blocking::new().unwrap();
        // The outer future waits on a timer that fires while the nested call is parked, so
        // the wake's unpark goes to the nested call's park.
        let value = blocking.block_on(async {
            let mut timer = pin!(tokio::time::sleep(Duration::from_millis(10)));
            let mut nested = None;
            std::future::poll_fn(|context| {
                if timer.as_mut().poll(context).is_ready() {
                    return Poll::Ready(nested.unwrap_or(0) + 10);
                }
                if nested.is_none() {
                    nested = Some(blocking.block_on(async {
                        tokio::time::sleep(Duration::from_millis(40)).await;
                        5
                    }));
                }
                Poll::Pending
            })
            .await
        });
        assert_eq!(value, 15);
    }

    #[test]
    fn takes_borrowed_futures_that_are_not_send() {
        let blocking = Blocking::new().unwrap();
        let borrowed = [1, 2, 3];
        let local = Rc::new(RefCell::new(0));
        let sum = blocking.block_on(async {
            *local.borrow_mut() += 1;
            after_a_timer(()).await;
            borrowed.iter().sum::<i32>() + *local.borrow()
        });
        assert_eq!(sum, 7);
    }

    #[test]
    fn runs_io_and_spawned_tasks_on_its_runtime() {
        let blocking = Blocking::new().unwrap();
        // Inside a current-thread runtime that has no I/O driver: the socket and the task
        // spawned by the future both use the helper's runtime.
        let runtime = Builder::new_current_thread().build().unwrap();
        let echoed = runtime.block_on(async {
            blocking.block_on(async {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let address = listener.local_addr().unwrap();
                let server = tokio::spawn(async move {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    let mut buffer = [0u8; 4];
                    socket.read_exact(&mut buffer).await.unwrap();
                    socket.write_all(&buffer).await.unwrap();
                });
                let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
                client.write_all(b"ping").await.unwrap();
                let mut buffer = [0u8; 4];
                client.read_exact(&mut buffer).await.unwrap();
                server.await.unwrap();
                buffer
            })
        });
        assert_eq!(&echoed, b"ping");
    }

    #[test]
    fn dropping_inside_a_runtime_does_not_panic() {
        let runtime = Builder::new_current_thread().build().unwrap();
        runtime.block_on(async {
            let blocking = Blocking::new().unwrap();
            assert_eq!(blocking.block_on(after_a_timer(4)), 4);
            drop(blocking);
        });
    }
}
