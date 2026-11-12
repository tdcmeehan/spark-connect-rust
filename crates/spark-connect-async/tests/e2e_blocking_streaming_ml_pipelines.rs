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

//! Structured streaming, Spark ML and pipelines through the blocking API, against a Spark
//! Connect server at `SPARK_REMOTE`; each test passes without running when it is not set. Each
//! call runs within a deadline, on a thread of its own, and each test gives what it creates names
//! of its own, then stops its queries and its session and drops its datasets, also when it fails.

#![cfg(feature = "blocking")]

use std::panic::AssertUnwindSafe;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use spark_connect_async::blocking::{self, Estimator, Evaluator};
use spark_connect_async::functions::expr;
use spark_connect_async::ml::{
    self, LinearRegression, Pipeline, RegressionEvaluator, StandardScaler, VectorAssembler,
};
use spark_connect_async::streaming::{QueryEvent, Trigger};
use spark_connect_async::types::DataType;
use spark_connect_async::{ErrorKind, Session, Value};

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

/// What `call` returns within [`DEADLINE`], if it returns: for cleaning up.
fn bounded<T: Send + 'static>(call: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (sender, returned) = mpsc::channel();
    thread::spawn(move || sender.send(call()).ok());
    returned.recv_timeout(DEADLINE).ok()
}

/// The server, a suffix of names unique to this test, and the connection string of a session of
/// the server of its own.
fn remote(test: &str) -> Option<(String, String, String)> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let unique = format!("{:x}", now.as_nanos() & 0xffff_ffff_ffff);
    let id = format!("00000000-0000-4000-8000-{unique:0>12}");
    let url = format!("{remote}/;session_id={id};user_id=e2e_blocking");
    Some((remote, format!("{test}_{unique}"), url))
}

/// Runs `body` with a blocking session of `url`, then stops the queries of the session and the
/// session, and drops the datasets `datasets` with another session of `remote`, also when `body`
/// panics.
fn cleaned_up(remote: &str, url: &str, datasets: &[String], body: impl FnOnce(blocking::Session)) {
    let url = url.to_string();
    let session = within("connect", move || {
        Session::builder().remote(url).blocking().connect().unwrap()
    });
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| body(session.clone())));
    let streams = session.streams();
    if let Some(Ok(queries)) = bounded(move || streams.active()) {
        for query in queries {
            bounded(move || query.stop());
        }
    }
    bounded(move || session.stop());
    let (remote, datasets) = (remote.to_string(), datasets.to_vec());
    bounded(move || {
        let cleaner = Session::builder().remote(remote).blocking().connect()?;
        for dataset in datasets {
            for kind in ["MATERIALIZED VIEW", "TABLE", "VIEW"] {
                let _ = cleaner
                    .as_async()
                    .sql(format!("DROP {kind} IF EXISTS {dataset}"))
                    .run();
            }
        }
        cleaner.stop()
    });
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
}

/// The next event of the query `id` that `wanted` accepts, skipping the others, and the events.
fn event_of(
    mut events: blocking::QueryEvents,
    id: &str,
    wanted: fn(&QueryEvent) -> bool,
) -> (blocking::QueryEvents, QueryEvent) {
    let id = id.to_string();
    within("an event", move || {
        while let Some(event) = events.next() {
            let event = event.unwrap();
            if event.id() == id && wanted(&event) {
                return (events, event);
            }
        }
        panic!("the events ended");
    })
}

#[test]
fn a_query_reports_its_events_and_a_wait_for_it_ends_when_another_thread_stops_it() {
    let Some((remote, name, url)) = remote("blocking_stream") else {
        return;
    };
    let checkpoint = tempfile::tempdir().unwrap();
    let location = checkpoint.path().display().to_string();
    cleaned_up(&remote, &url, &[], |session| {
        let streams = session.streams();
        let subscriber = streams.clone();
        let events = within("events", move || subscriber.events().unwrap());
        let subscriber = streams.clone();
        let second = within("a second events", move || subscriber.events().map(drop));
        assert_eq!(second.unwrap_err().kind(), ErrorKind::Unsupported);

        let writer = session
            .as_async()
            .read_stream()
            .format("rate")
            .option("rowsPerSecond", 10)
            .load_without_path()
            .write_stream()
            .format("memory")
            .query_name(&name)
            .trigger(Trigger::ProcessingTime("100 milliseconds".to_string()))
            .option("checkpointLocation", location.as_str())
            .blocking();
        let query = within("start", move || writer.start_without_path().unwrap());
        assert_eq!(query.as_async().name(), Some(name.as_str()));
        let id = query.as_async().id().to_string();
        let (events, started) = event_of(events, &id, |e| matches!(e, QueryEvent::Started(_)));
        match started {
            QueryEvent::Started(started) => assert_eq!(started.run_id, query.as_async().run_id()),
            other => panic!("{other:?}"),
        }
        let manager = streams.clone();
        let found = within("get", move || manager.get(&id).unwrap().unwrap());
        assert_eq!(found.as_async().run_id(), query.as_async().run_id());

        let reader = query.clone();
        let progress = within("a batch that read rows", move || {
            let deadline = Instant::now() + DEADLINE;
            while Instant::now() < deadline {
                reader.process_all_available().unwrap();
                if let Some(progress) = reader.last_progress().unwrap() {
                    if progress.num_input_rows > 0 {
                        return progress;
                    }
                }
            }
            panic!("no batch read rows");
        });
        assert_eq!(progress.name.as_deref(), Some(name.as_str()));
        let id = query.as_async().id().to_string();
        let (events, _) = event_of(events, &id, |e| matches!(e, QueryEvent::Progress(_)));

        let waiter = query.clone();
        let waited = within("a wait with a timeout", move || {
            waiter.await_termination(Some(Duration::from_millis(200)))
        });
        assert!(!waited.unwrap(), "the query terminated");
        // A blocking wait can't be given up: another thread ends it by stopping the query.
        let stopper = query.clone();
        let stopping = thread::spawn(move || {
            thread::sleep(Duration::from_millis(500));
            stopper.stop()
        });
        let waiter = query.clone();
        let terminated = within("a wait without a timeout", move || {
            waiter.await_termination(None)
        });
        assert!(
            terminated.unwrap(),
            "the wait ended without the query's end"
        );
        stopping.join().unwrap().unwrap();
        let checker = query.clone();
        assert!(!within("is_active", move || checker.is_active().unwrap()));
        let (events, terminated) =
            event_of(events, &id, |e| matches!(e, QueryEvent::Terminated(_)));
        match terminated {
            QueryEvent::Terminated(terminated) => assert_eq!(terminated.exception, None),
            other => panic!("{other:?}"),
        }
        let reader = session.clone();
        let rows = within("the rows of the sink", move || {
            reader.as_async().table(&name).blocking().count().unwrap()
        });
        assert!(rows > 0, "{rows} rows");

        // Dropping the events removes the subscription, so the session can subscribe again.
        drop(events);
        within("events after the drop", move || {
            drop(streams.events().unwrap())
        });
    });
}

/// Rows of a feature `x` and a label `y = 2x + 1`.
fn line(session: &blocking::Session) -> blocking::DataFrame {
    let schema = DataType::from_ddl("x DOUBLE, y DOUBLE").unwrap();
    let rows = (0..10).map(|i| {
        let x = f64::from(i);
        vec![Value::Double(x), Value::Double(2.0 * x + 1.0)]
    });
    let session = session.clone();
    within("create_dataframe", move || {
        session.create_dataframe(rows, schema).unwrap()
    })
}

#[test]
fn a_fit_reads_its_model_s_attributes_predicts_and_evaluates() {
    let Some((remote, _, url)) = remote("blocking_ml") else {
        return;
    };
    cleaned_up(&remote, &url, &[], |session| {
        let rows = line(&session);
        let pipeline = Pipeline::new()
            .stage(
                VectorAssembler::new()
                    .input_cols(["x"])
                    .output_col("features"),
            )
            .stage(LinearRegression::new().label_col("y"));
        let training = rows.clone();
        let fitted = within("fit", move || pipeline.fit(training.as_async()).unwrap());
        let model = fitted
            .stage::<ml::LinearRegressionModel>(1)
            .unwrap()
            .blocking();
        let reader = model.clone();
        let (coefficients, intercept) = within("the attributes", move || {
            (reader.coefficients().unwrap(), reader.intercept().unwrap())
        });
        let slope = coefficients.to_dense();
        assert_eq!(slope.len(), 1, "{slope:?}");
        assert!((slope[0] - 2.0).abs() < 1e-6, "{slope:?}");
        assert!((intercept - 1.0).abs() < 1e-6, "{intercept}");

        let predictions = ml::Transformer::transform(&fitted, rows.as_async());
        let evaluated = predictions.clone();
        let rmse = within("evaluate", move || {
            RegressionEvaluator::new()
                .label_col("y")
                .evaluate(&evaluated)
                .unwrap()
        });
        assert!(rmse < 1e-6, "{rmse}");
        let predicted = predictions.select(["prediction"]).blocking();
        let predicted = within("the predictions", move || {
            predicted.collect_as::<(f64,)>().unwrap()
        });
        assert_eq!(predicted.len(), 10);

        let features = VectorAssembler::new()
            .input_cols(["x"])
            .output_col("features");
        let assembled = ml::Transformer::transform(&features, rows.as_async());
        let scaler = StandardScaler::new().input_col("features");
        let scaled = within("fit a scaler", move || scaler.fit(&assembled).unwrap());
        let mean = within("its mean", move || scaled.mean().unwrap());
        assert_eq!(mean.to_dense(), [4.5]);
    });
}

#[test]
fn a_pipeline_defined_through_the_blocking_api_runs_to_its_end() {
    let Some((remote, unique, url)) = remote("blocking_sdp") else {
        return;
    };
    let numbers = format!("{unique}_numbers");
    let evens = format!("{unique}_evens");
    let datasets = [numbers.clone(), evens.clone()];
    cleaned_up(&remote, &url, &datasets, |session| {
        let directory = tempfile::tempdir().unwrap();
        let storage = format!("file://{}", directory.path().display());
        let creator = session.clone();
        let graph = within("create_dataflow_graph", move || {
            creator.as_async().create_dataflow_graph().run().unwrap()
        });
        let (definer, sql) = (
            graph.clone(),
            format!("CREATE MATERIALIZED VIEW {numbers} AS SELECT id FROM range(6)"),
        );
        within("define_sql", move || {
            definer.as_async().define_sql(sql).run().unwrap()
        });
        let (definer, name) = (graph.clone(), evens.clone());
        let view = within("define_materialized_view", move || {
            definer
                .as_async()
                .define_materialized_view(name)
                .run()
                .unwrap()
        });
        assert_eq!(view.name, evens);
        let query = session
            .as_async()
            .table(&numbers)
            .filter(expr("id % 2 = 0"));
        let (definer, name) = (graph.clone(), evens.clone());
        within("define_flow", move || {
            definer
                .as_async()
                .define_flow(&name, &name, &query)
                .run()
                .unwrap()
        });
        let runner = graph.clone();
        let events = within("the run to its end", move || {
            let run = runner.as_async().start_run(storage).run().unwrap();
            run.collect::<spark_connect_async::Result<Vec<_>>>()
                .unwrap()
        });
        let last = events.last().expect("no events");
        assert!(last.message.contains("COMPLETED"), "{events:?}");
        let reader = session.clone();
        let ids = within("the rows", move || {
            let rows = reader.as_async().table(&evens).sort(["id"]).blocking();
            rows.collect_as::<(i64,)>().unwrap()
        });
        assert_eq!(ids, [(0,), (2,), (4,)]);
        drop(directory);
    });
}
