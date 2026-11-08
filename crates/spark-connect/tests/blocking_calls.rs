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

//! How the synchronous API runs its calls: against the in-process server of the request
//! snapshots, inside runtimes of either flavor, with its callbacks on the calling thread.

#![allow(clippy::result_large_err)] // SparkError is large; the API returns it

#[path = "request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;

use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

use prost::Message;
use spark_connect::ml::{Estimator, Evaluator, LogisticRegression, RegressionEvaluator};
use spark_connect::pipelines;
use spark_connect::session::SparkSession;
use spark_connect::ExecutePlanHook;
use spark_connect_proto as proto;

fn session() -> SparkSession {
    session_recording(Arc::new(server::Recorder::default()))
}

fn session_recording(recorder: Arc<server::Recorder>) -> SparkSession {
    let address = server::start(recorder);
    SparkSession::builder()
        .remote(&format!(
            "sc://{address}/;session_id=00000000-0000-4000-8000-00000000005e"
        ))
        .get_or_create()
        .unwrap()
}

fn runtimes() -> [tokio::runtime::Runtime; 2] {
    [
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap(),
        tokio::runtime::Builder::new_multi_thread().build().unwrap(),
    ]
}

/// A hook that records the thread it runs on.
fn thread_hook(threads: &Arc<Mutex<Vec<ThreadId>>>) -> ExecutePlanHook {
    let threads = Arc::clone(threads);
    Arc::new(move |request| {
        threads.lock().unwrap().push(thread::current().id());
        Ok(request)
    })
}

#[test]
fn a_command_runs_inside_runtimes_of_either_flavor() {
    let spark = session();
    for runtime in runtimes() {
        let on_block_on = runtime.block_on(async { spark.sql("SELECT 1").map(drop) });
        let spark = spark.clone();
        let in_task = runtime.block_on(async move {
            tokio::spawn(async move { spark.sql("SELECT 2").map(drop) })
                .await
                .unwrap()
        });
        assert!(on_block_on.is_ok() && in_task.is_ok());
    }
}

#[test]
fn the_hook_of_a_command_or_a_pipeline_run_runs_on_the_calling_thread() {
    let spark = session();
    let threads = Arc::new(Mutex::new(Vec::new()));
    spark.set_execute_plan_hook(Some(thread_hook(&threads)));
    spark.sql("SELECT 1").unwrap();
    // Also when the calling thread is that of a current-thread runtime.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async { spark.sql("SELECT 2") }).unwrap();
    run_pipeline(&spark);
    let caller = thread::current().id();
    assert_eq!(*threads.lock().unwrap(), vec![caller; 3]);
}

/// Starts a run of a pipeline and reads its responses to the end.
fn run_pipeline(spark: &SparkSession) {
    let mut run =
        pipelines::start_run(spark, "graph-id", vec![], true, vec![], false, None).unwrap();
    while run.next_response().unwrap().is_some() {}
}

#[test]
fn a_hook_can_run_a_blocking_call_of_its_own() {
    let spark = session();
    let other = spark.clone();
    let nested = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&nested);
    let hook: ExecutePlanHook = Arc::new(move |request| {
        seen.lock()
            .unwrap()
            .push(other.conf().get("k").map_err(|e| e.to_string()));
        Ok(request)
    });
    spark.set_execute_plan_hook(Some(hook));
    spark.sql("SELECT 1").unwrap();
    assert_eq!(*nested.lock().unwrap(), vec![Ok(None)]);
}

#[test]
fn a_hook_that_runs_queries_without_end_fails_with_a_recursion_error() {
    let spark = session();
    let other = spark.clone();
    let hook: ExecutePlanHook = Arc::new(move |request| {
        other.sql("SELECT 1")?;
        Ok(request)
    });
    spark.set_execute_plan_hook(Some(hook));
    let Err(error) = spark.sql("SELECT 1") else {
        panic!("the hooks ran without end");
    };
    assert_eq!(error.kind, spark_connect::SparkErrorKind::RecursionError);
    // Once the hooks have returned, queries run again.
    spark.set_execute_plan_hook(None);
    spark.sql("SELECT 2").unwrap();
}

#[test]
fn the_actions_of_a_dataframe_run_inside_runtimes_of_either_flavor() {
    let spark = session();
    let df = spark.range(3).unwrap();
    for runtime in runtimes() {
        runtime.block_on(async {
            df.collect().unwrap();
            df.schema().unwrap();
            for prefetch in [false, true] {
                for row in df.to_local_iterator(prefetch).unwrap() {
                    row.unwrap();
                }
            }
        });
    }
}

#[test]
fn the_hook_of_a_dataframe_action_runs_on_the_calling_thread() {
    let spark = session();
    let threads = Arc::new(Mutex::new(Vec::new()));
    spark.set_execute_plan_hook(Some(thread_hook(&threads)));
    let df = spark.range(3).unwrap();
    df.collect().unwrap();
    for row in df.to_local_iterator(false).unwrap() {
        row.unwrap();
    }
    let caller = thread::current().id();
    assert_eq!(*threads.lock().unwrap(), vec![caller, caller]);
}

#[test]
fn a_prefetching_iterator_can_be_dropped_before_its_end() {
    let spark = session();
    let df = spark.range(3).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async {
        let iterator = df.to_local_iterator(true).unwrap();
        drop(iterator);
        df.collect().unwrap();
    });
}

#[test]
fn a_catalog_operation_and_a_write_run_inside_runtimes_of_either_flavor() {
    let spark = session();
    let df = spark.range(3).unwrap();
    for runtime in runtimes() {
        runtime.block_on(async {
            spark.catalog().current_database().unwrap();
            df.write().mode("overwrite").save(Some("/out")).unwrap();
        });
    }
}

#[test]
fn a_catalog_operation_records_no_execution_and_a_write_does() {
    let spark = session();
    spark.catalog().current_database().unwrap();
    assert!(spark.last_execution_info().is_none());
    spark.range(3).unwrap().write().save(Some("/out")).unwrap();
    assert!(spark.last_execution_info().is_some());
}

struct NoListener;

impl spark_connect::streaming::StreamingQueryListener for NoListener {
    fn on_event(&self, _: &spark_connect::streaming::StreamingQueryListenerEvent) {}
}

#[test]
fn streaming_runs_inside_runtimes_of_either_flavor() {
    let spark = session();
    let df = spark.range(3).unwrap();
    for runtime in runtimes() {
        runtime.block_on(async {
            let query = df
                .write_stream()
                .format("memory")
                .query_name("q")
                .start("")
                .unwrap();
            query.status().unwrap();
            spark.streams().active().unwrap();
            let id = spark.streams().add_listener(Arc::new(NoListener)).unwrap();
            spark.streams().remove_listener(&id).unwrap();
        });
    }
}

#[test]
fn ml_and_pipelines_run_inside_runtimes_of_either_flavor() {
    let spark = session();
    let df = spark.range(3).unwrap();
    for runtime in runtimes() {
        runtime.block_on(async {
            LogisticRegression::default().fit(&df).unwrap();
            RegressionEvaluator::new().evaluate(&df).unwrap();
            pipelines::create_dataflow_graph(&spark, None, None, Default::default()).unwrap();
            run_pipeline(&spark);
        });
    }
}

fn is_ml_delete(entry: &server::Entry) -> bool {
    use proto::command::CommandType;
    let Ok(request) = proto::ExecutePlanRequest::decode(&entry.bytes[..]) else {
        return false;
    };
    matches!(
        request.plan.and_then(|plan| plan.op_type),
        Some(proto::plan::OpType::Command(proto::Command {
            command_type: Some(CommandType::MlCommand(proto::MlCommand {
                command: Some(proto::ml_command::Command::Delete(_)),
            })),
        }))
    )
}

#[test]
fn a_model_dropped_inside_a_current_thread_runtime_is_deleted_on_the_server() {
    let recorder = Arc::new(server::Recorder::default());
    let spark = session_recording(Arc::clone(&recorder));
    let df = spark.range(3).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async { drop(LogisticRegression::default().fit(&df).unwrap()) });
    drop(runtime);
    assert!(recorder.wait_for(std::time::Duration::from_secs(10), is_ml_delete));
}
