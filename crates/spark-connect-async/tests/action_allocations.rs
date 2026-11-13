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

//! An action whose answer the DataFrame keeps allocates, once it has the answer, only the copy that
//! it returns.

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use spark_connect_async::Session;

struct Counting;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

// Only this thread's allocations count: the server answers on threads of its own.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.try_with(Cell::get).unwrap_or(false) {
            ALLOCATIONS.with(|count| {
                let (allocations, bytes) = count.get();
                count.set((allocations + 1, bytes + layout.size()));
            });
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// What `f` returns, and the number and the total size of the allocations of this thread in it.
fn allocations<T>(f: impl FnOnce() -> T) -> (T, (usize, usize)) {
    ALLOCATIONS.with(|count| count.set((0, 0)));
    COUNTING.with(|on| on.set(true));
    let value = f();
    COUNTING.with(|on| on.set(false));
    (value, ALLOCATIONS.with(Cell::get))
}

#[tokio::test]
async fn a_schema_from_the_cache_allocates_only_the_copy_that_it_returns() {
    let address = server::start(Arc::new(server::Recorder::default()));
    let session = Session::builder()
        .remote(format!("sc://{address}"))
        .connect()
        .await
        .unwrap();
    let dataframe = session.range(3);
    let schema = dataframe.schema().await.unwrap();
    let (cached, cached_allocations) = allocations(|| {
        let mut call = pin!(dataframe.schema());
        match call.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(answer) => answer.unwrap(),
            Poll::Pending => panic!("the schema is not cached"),
        }
    });
    let (copy, copy_allocations) = allocations(|| schema.clone());
    assert_eq!(cached, copy);
    assert_eq!(cached_allocations, copy_allocations);
}
