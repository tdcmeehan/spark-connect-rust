// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! A Rust UDF of the async client registered and called through its blocking API, on a live
//! Spark Connect server whose Python workers need `pyarrow` and `wasmtime`. Run with
//! `SPARK_REMOTE` set; skipped otherwise. Each call has a deadline, and the test stops its
//! session, which removes the function it registered, also when it fails.

use std::panic::AssertUnwindSafe;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rust_udf_async_tests::udf;
use spark_connect_async::{blocking, Session};

const DEADLINE: Duration = Duration::from_secs(300);

/// What `call` returns, from a thread of its own; a failure if it takes longer than
/// [`DEADLINE`].
fn within<T: Send + 'static>(what: &str, call: impl FnOnce() -> T + Send + 'static) -> T {
    let (sender, returned) = mpsc::channel();
    let caller = thread::spawn(move || sender.send(call()).ok());
    match returned.recv_timeout(DEADLINE) {
        Ok(value) => value,
        Err(RecvTimeoutError::Disconnected) => match caller.join() {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(_) => unreachable!("{what}: the call returned without its value"),
        },
        Err(RecvTimeoutError::Timeout) => panic!("{what}: no answer within {DEADLINE:?}"),
    }
}

#[test]
fn a_udf_registered_through_the_blocking_api_runs_in_sql_and_on_columns() {
    let Ok(remote) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let session: blocking::Session = within("connect", move || {
        Session::builder()
            .remote(remote)
            .blocking()
            .connect()
            .unwrap()
    });
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let name = format!("blocking_minus_{:x}", now.as_nanos() & 0xffff_ffff_ffff);
    let caller = session.clone();
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(move || {
        let minus = udf::minus_udf();
        let (registrar, registered, function) = (caller.clone(), minus.clone(), name.clone());
        within("register", move || {
            registrar.udf().register(function, &registered).unwrap()
        });
        let query = format!("SELECT {name}(id, id * 10) FROM range(0, 3, 1, 1)");
        let reader = caller.clone();
        let in_sql = within("in SQL", move || {
            let rows = reader.as_async().sql(query).run().unwrap();
            rows.collect_as::<(Option<i64>,)>().unwrap()
        });
        assert_eq!(in_sql, [(Some(0),), (Some(-9),), (Some(-18),)]);
        let column = minus.call(["id", "id"]).unwrap();
        let on_columns = within("on columns", move || {
            let rows = caller.as_async().range(3).select([column]).blocking();
            rows.collect_as::<(Option<i64>,)>().unwrap()
        });
        assert_eq!(on_columns, [(Some(0),), (Some(0),), (Some(0),)]);
    }));
    let (sender, stopped) = mpsc::channel();
    thread::spawn(move || sender.send(session.stop()).ok());
    let _ = stopped.recv_timeout(DEADLINE);
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
}
