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

//! The blocking API against a Spark Connect server at `SPARK_REMOTE`; each test passes without
//! running when it is not set.

#![cfg(feature = "blocking")]

use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use serde::Deserialize;
use spark_connect_async::actions::TempView;
use spark_connect_async::readwriter::SaveMode;
use spark_connect_async::{blocking, col, lit, Session};

/// The longest that any call of these tests may take.
const DEADLINE: Duration = Duration::from_secs(120);

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

/// Stops its session when it drops, also when the test panics.
struct Stopped(blocking::Session);

impl Drop for Stopped {
    fn drop(&mut self) {
        let (session, (stopped, waiting)) = (self.0.clone(), mpsc::channel());
        thread::spawn(move || stopped.send(session.stop()).ok());
        let _ = waiting.recv_timeout(DEADLINE);
    }
}

fn session() -> Option<blocking::Session> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    Some(within("connect", move || {
        Session::builder()
            .remote(remote)
            .blocking()
            .connect()
            .unwrap()
    }))
}

#[derive(Deserialize, Debug, PartialEq)]
struct Person {
    id: i64,
    name: String,
}

#[test]
fn a_blocking_session_queries_writes_reads_back_and_lists_its_catalog() {
    let Some(session) = session() else {
        return;
    };
    let _stopped = Stopped(session.clone());
    let s = session.clone();
    let people = within("sql", move || {
        s.sql("SELECT id, concat('p', id) AS name FROM range(5)")
            .run()
            .unwrap()
    });
    let df = people.clone();
    let mut rows: Vec<Person> = within("collect_as", move || df.collect_as().unwrap());
    rows.sort_by_key(|person| person.id);
    assert_eq!(rows.len(), 5);
    assert_eq!(
        rows[4],
        Person {
            id: 4,
            name: "p4".to_string()
        }
    );
    let df = people.clone();
    let table = within("show", move || df.show().blocking().to_string().unwrap());
    assert!(table.contains("| id|name|"), "{table}");
    let df = people.clone();
    let adults = within("count", move || {
        df.filter(col("id").ge(lit(2))).blocking().count().unwrap()
    });
    assert_eq!(adults, 3);

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("people").display().to_string();
    let (df, out) = (people.clone(), path.clone());
    within("write", move || {
        df.write()
            .mode(SaveMode::Overwrite)
            .blocking()
            .parquet(out)
            .unwrap()
    });
    let s = session.clone();
    let mut read: Vec<Person> = within("read back", move || {
        s.read().parquet(path).blocking().collect_as().unwrap()
    });
    read.sort_by_key(|person| person.id);
    assert_eq!(read, rows);

    let df = people.clone();
    within("create_temp_view", move || {
        df.create_temp_view("b14a_people", TempView::default().or_replace())
            .unwrap()
    });
    let s = session.clone();
    let tables = within("list_tables", move || {
        s.catalog().list_tables().run().unwrap()
    });
    assert!(
        tables
            .iter()
            .any(|table| table.name == "b14a_people" && table.is_temporary),
        "{tables:?}"
    );
    within("stop", move || session.stop().unwrap());
}
