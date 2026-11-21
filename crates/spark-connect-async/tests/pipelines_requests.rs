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

//! The requests of the pipelines of the async API, recorded by the old API's recording server
//! and compared with the old API's.

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;
mod support;

use std::collections::HashMap;
use std::future::IntoFuture;
use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use futures::StreamExt;
use prost::Message;
use spark_connect::pipelines as old;
use spark_connect::SparkSession;
use spark_connect_async::pipelines::{ScdType, SourceCodeLocation};
use spark_connect_async::types::DataType;
use spark_connect_async::{col, ErrorKind, Session};
use spark_connect_proto as proto;

use proto::pipeline_command::define_flow::AutoCdcFlowDetails;

const SESSION_ID: &str = "00000000-0000-4000-8000-0000000000cc";

/// The id of the graphs of the recording server.
const GRAPH: &str = "graph-id";

struct Case {
    name: &'static str,
    old: fn(&SparkSession),
    new: fn(Session) -> BoxFuture<'static, ()>,
}

fn conf() -> HashMap<String, String> {
    HashMap::from([("k".to_string(), "v".to_string())])
}

fn old_location() -> proto::SourceCodeLocation {
    proto::SourceCodeLocation {
        file_name: Some("flows.rs".to_string()),
        line_number: Some(4),
        definition_path: Some("flows::f".to_string()),
        ..Default::default()
    }
}

fn location() -> SourceCodeLocation {
    SourceCodeLocation::file("flows.rs")
        .line_number(4)
        .definition_path("flows::f")
}

fn schema() -> DataType {
    DataType::from_ddl("a INT, b STRING").unwrap()
}

fn old_table(
    schema_data_type: Option<DataType>,
    schema_string: Option<&str>,
) -> old::TableDetailsSpec {
    old::TableDetailsSpec {
        table_properties: conf(),
        partition_cols: vec!["p".to_string()],
        clustering_columns: vec!["c".to_string()],
        format: Some("parquet".to_string()),
        schema_string: schema_string.map(str::to_string),
        schema_data_type,
    }
}

fn old_output(name: &str, output_type: proto::OutputType) -> old::OutputSpec {
    old::OutputSpec {
        name: name.to_string(),
        output_type: output_type as i32,
        comment: None,
        source_code_location: None,
        table_details: None,
        sink_details: None,
    }
}

/// A graph of the old API's session `s`, as the recording server creates one.
fn old_graph(s: &SparkSession) -> String {
    let id = old::create_dataflow_graph(s, None, None, HashMap::new()).unwrap();
    assert_eq!(id, GRAPH);
    id
}

/// Reads the responses of the old run of `run` to their end.
fn read_old(mut run: old::PipelineRunStream) {
    while run.next_response().unwrap().is_some() {}
}

/// A case of the definition `$old` of the old API and `$new` of a graph of the async API, whose
/// answer the recording server leaves out: it sends no identifier.
macro_rules! define {
    ($name:expr, |$s:ident, $id:ident| $old:expr, |$graph:ident| $new:expr) => {
        Case {
            name: $name,
            old: |$s| {
                let $id = old_graph($s);
                $old.unwrap();
            },
            new: |session| {
                Box::pin(async move {
                    let $graph = session.create_dataflow_graph().await.unwrap();
                    let _ = $new.await;
                })
            },
        }
    };
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "create_dataflow_graph with its defaults",
            old: |s| {
                old::create_dataflow_graph(
                    s,
                    Some("cat".to_string()),
                    Some("db".to_string()),
                    conf(),
                )
                .unwrap();
            },
            new: |session| {
                Box::pin(async move {
                    session
                        .create_dataflow_graph()
                        .default_catalog("cat")
                        .default_database("db")
                        .sql_conf("k", "v")
                        .await
                        .unwrap();
                })
            },
        },
        Case {
            name: "create_dataflow_graph",
            old: |s| {
                old_graph(s);
            },
            new: |session| {
                Box::pin(async move {
                    session.create_dataflow_graph().await.unwrap();
                })
            },
        },
        define!(
            "define_output of a materialized view with every detail",
            |s, id| {
                let mut mv = old_output("mv", proto::OutputType::MaterializedView);
                mv.comment = Some("comment".to_string());
                mv.source_code_location = Some(old_location());
                mv.table_details = Some(old_table(Some(schema()), None));
                old::define_output(s, &id, mv)
            },
            |graph| graph
                .define_materialized_view("mv")
                .comment("comment")
                .source_code_location(location())
                .table_property("k", "v")
                .partition_by(["p"])
                .cluster_by(["c"])
                .format("parquet")
                .schema(schema())
        ),
        define!(
            "define_output of a streaming table with a schema in DDL",
            |s, id| {
                let mut table = old_output("t", proto::OutputType::Table);
                table.table_details = Some(old_table(None, Some("a INT")));
                old::define_output(s, &id, table)
            },
            |graph| graph
                .define_streaming_table("t")
                .table_property("k", "v")
                .partition_by(["p"])
                .cluster_by(["c"])
                .format("parquet")
                .schema("a INT")
        ),
        define!(
            "define_output of a materialized view of no detail",
            |s, id| {
                let mut mv = old_output("plain", proto::OutputType::MaterializedView);
                mv.table_details = Some(old::TableDetailsSpec {
                    table_properties: HashMap::new(),
                    partition_cols: Vec::new(),
                    clustering_columns: Vec::new(),
                    format: None,
                    schema_string: None,
                    schema_data_type: None,
                });
                old::define_output(s, &id, mv)
            },
            |graph| graph.define_materialized_view("plain")
        ),
        define!(
            "define_output of a temporary view",
            |s, id| old::define_output(s, &id, old_output("v", proto::OutputType::TemporaryView)),
            |graph| graph.define_temporary_view("v")
        ),
        define!(
            "define_output of a temporary view with a comment and a location",
            |s, id| {
                let mut view = old_output("v", proto::OutputType::TemporaryView);
                view.comment = Some("comment".to_string());
                view.source_code_location = Some(old_location());
                old::define_output(s, &id, view)
            },
            |graph| graph
                .define_temporary_view("v")
                .comment("comment")
                .source_code_location(location())
        ),
        define!(
            "define_output of a sink",
            |s, id| {
                let mut sink = old_output("s", proto::OutputType::Sink);
                sink.source_code_location = Some(old_location());
                sink.sink_details = Some(old::SinkDetailsSpec {
                    options: HashMap::from([
                        ("k".to_string(), "v".to_string()),
                        ("n".to_string(), "1".to_string()),
                    ]),
                    format: Some("kafka".to_string()),
                });
                old::define_output(s, &id, sink)
            },
            |graph| graph
                .define_sink("s", "kafka")
                .option("k", "v")
                .options([("n", 1)])
                .source_code_location(location())
        ),
        Case {
            name: "define_flow of a DataFrame",
            old: |s| {
                let id = old_graph(s);
                let data = s.range(3).unwrap();
                old::define_flow(s, &id, "f", "t", &data, conf(), Some(old_location())).unwrap();
            },
            new: |session| {
                Box::pin(async move {
                    let graph = session.create_dataflow_graph().await.unwrap();
                    let _ = graph
                        .define_flow("f", "t", session.range(3))
                        .sql_conf("k", "v")
                        .source_code_location(location())
                        .await;
                })
            },
        },
        Case {
            name: "define_flow of a filtered DataFrame",
            old: |s| {
                let id = old_graph(s);
                let data = s
                    .range(3)
                    .unwrap()
                    .filter(spark_connect::column::col("id").is_not_null());
                old::define_flow(s, &id, "g", "t", &data, HashMap::new(), None).unwrap();
            },
            new: |session| {
                Box::pin(async move {
                    let graph = session.create_dataflow_graph().await.unwrap();
                    let data = session.range(3).filter(col("id").is_not_null());
                    let _ = graph.define_flow("g", "t", data).await;
                })
            },
        },
        Case {
            name: "define_flow of SQL text",
            old: |s| {
                let id = old_graph(s);
                let query = "SELECT id FROM range(4)";
                let data = s.sql_without_running(query, Vec::new(), HashMap::new(), Vec::new());
                old::define_flow(s, &id, "f", "t", &data, HashMap::new(), None).unwrap();
            },
            new: |session| {
                Box::pin(async move {
                    let graph = session.create_dataflow_graph().await.unwrap();
                    let _ = graph.define_flow("f", "t", "SELECT id FROM range(4)").await;
                })
            },
        },
        define!(
            "define_auto_cdc_flow",
            |s, id| {
                let details = AutoCdcFlowDetails {
                    source: Some("src".to_string()),
                    keys: vec![spark_connect::column::col("k").to_proto()],
                    sequence_by: Some(spark_connect::column::col("ts").to_proto()),
                    stored_as_scd_type: 2,
                    ..Default::default()
                };
                old::define_auto_cdc_flow(s, &id, "cdc", "t", details, conf(), Some(old_location()))
            },
            |graph| graph
                .define_auto_cdc_flow("cdc", "t", "src", ["k"], "ts")
                .stored_as_scd_type(ScdType::Type2)
                .sql_conf("k", "v")
                .source_code_location(location())
        ),
        define!(
            "define_auto_cdc_flow of SCD type 1 with columns",
            |s, id| {
                let column = |name: &str| spark_connect::column::col(name).to_proto();
                let details = AutoCdcFlowDetails {
                    source: Some("src".to_string()),
                    keys: vec![column("k"), column("l")],
                    sequence_by: Some(column("ts")),
                    apply_as_deletes: Some(column("deleted")),
                    column_list: vec![column("a"), column("b")],
                    stored_as_scd_type: 1,
                    ..Default::default()
                };
                old::define_auto_cdc_flow(s, &id, "cdc", "t", details, HashMap::new(), None)
            },
            |graph| graph
                .define_auto_cdc_flow("cdc", "t", "src", ["k", "l"], col("ts"))
                .apply_as_deletes("deleted")
                .column_list(["a", "b"])
                .stored_as_scd_type(ScdType::Type1)
        ),
        define!(
            "define_auto_cdc_flow of SCD type 2 with the columns left out",
            |s, id| {
                let column = |name: &str| spark_connect::column::col(name).to_proto();
                let details = AutoCdcFlowDetails {
                    source: Some("src".to_string()),
                    keys: vec![column("k")],
                    sequence_by: Some(column("ts")),
                    except_column_list: vec![column("x")],
                    stored_as_scd_type: 2,
                    track_history_column_list: vec![column("a")],
                    track_history_except_column_list: vec![column("b")],
                    ..Default::default()
                };
                old::define_auto_cdc_flow(s, &id, "cdc", "t", details, HashMap::new(), None)
            },
            |graph| graph
                .define_auto_cdc_flow("cdc", "t", "src", ["k"], "ts")
                .except_column_list(["x"])
                .stored_as_scd_type(ScdType::Type2)
                .track_history_column_list(["a"])
                .track_history_except_column_list(["b"])
        ),
        Case {
            name: "define_sql_graph_elements",
            old: |s| {
                let id = old_graph(s);
                old::define_sql_graph_elements(
                    s,
                    &id,
                    "CREATE MATERIALIZED VIEW mv AS SELECT 1",
                    "defs.sql",
                )
                .unwrap();
            },
            new: |session| {
                Box::pin(async move {
                    let graph = session.create_dataflow_graph().await.unwrap();
                    graph
                        .define_sql("CREATE MATERIALIZED VIEW mv AS SELECT 1")
                        .file_path("defs.sql")
                        .await
                        .unwrap();
                })
            },
        },
        Case {
            name: "start_run of selected datasets, dry",
            old: |s| {
                let id = old_graph(s);
                let run = old::start_run(
                    s,
                    &id,
                    vec!["a".to_string(), "b".to_string()],
                    false,
                    vec!["c".to_string()],
                    true,
                    Some("file:///storage".to_string()),
                );
                read_old(run.unwrap());
            },
            new: |session| {
                Box::pin(async move {
                    let graph = session.create_dataflow_graph().await.unwrap();
                    let run = graph
                        .start_run("file:///storage")
                        .full_refresh(["a"])
                        .full_refresh(["b"])
                        .refresh(["c"])
                        .dry_run()
                        .await
                        .unwrap();
                    assert!(run.collect::<Vec<_>>().await.is_empty());
                })
            },
        },
        Case {
            name: "start_run of every dataset",
            old: |s| {
                let id = old_graph(s);
                let run = old::start_run(
                    s,
                    &id,
                    Vec::new(),
                    true,
                    Vec::new(),
                    false,
                    Some("file:///storage".to_string()),
                );
                read_old(run.unwrap());
            },
            new: |session| {
                Box::pin(async move {
                    let graph = session.create_dataflow_graph().await.unwrap();
                    let run = graph
                        .start_run("file:///storage")
                        .full_refresh_all()
                        .await
                        .unwrap();
                    assert!(run.collect::<Vec<_>>().await.is_empty());
                })
            },
        },
    ]
}

fn start() -> (Arc<server::Recorder>, String) {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let url = format!("sc://{address}/;session_id={SESSION_ID};user_id=snapshots");
    (recorder, url)
}

fn command_of(entry: &server::Entry) -> Option<proto::command::CommandType> {
    match proto::ExecutePlanRequest::decode(&entry.bytes[..])
        .ok()?
        .plan?
        .op_type?
    {
        proto::plan::OpType::Command(command) => command.command_type,
        _ => None,
    }
}

fn is_drop(entry: &server::Entry) -> bool {
    matches!(
        command_of(entry),
        Some(proto::command::CommandType::PipelineCommand(
            proto::PipelineCommand {
                command_type: Some(proto::pipeline_command::CommandType::DropDataflowGraph(_)),
            }
        ))
    )
}

/// Waits for the drop of a graph that the operations of a case dropped.
async fn dropped(recorder: &Arc<server::Recorder>) -> bool {
    let recorder = Arc::clone(recorder);
    tokio::task::spawn_blocking(move || recorder.wait_for(Duration::from_secs(5), is_drop))
        .await
        .unwrap()
}

/// Each entry as its message, without the operation id, which the client chooses at random.
fn messages(entries: Vec<server::Entry>) -> Vec<String> {
    entries
        .into_iter()
        .map(|entry| match entry.kind.as_str() {
            "ExecutePlanRequest" => {
                let mut request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
                request.operation_id = None;
                format!("{request:?}")
            }
            _ => format!("{entry:?}"),
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipelines_send_the_requests_of_the_old_api() {
    let (recorder, url) = start();
    let mut both = Vec::new();
    for case in cases() {
        let old_session = SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        (case.old)(&old_session);
        let old = messages(recorder.take());
        assert!(!old.is_empty(), "{}: the old API sent nothing", case.name);
        let session = Session::builder()
            .remote(url.clone())
            .connect()
            .await
            .unwrap();
        (case.new)(session).await;
        assert!(
            dropped(&recorder).await,
            "{}: the graph not dropped",
            case.name
        );
        let entries = recorder.take();
        let drops = entries.iter().filter(|entry| is_drop(entry)).count();
        assert_eq!(drops, 1, "{}: drops", case.name);
        let new = messages(
            entries
                .into_iter()
                .filter(|entry| !is_drop(entry))
                .collect(),
        );
        both.push((case.name, old, new));
    }
    let differences = support::differences(both);
    assert!(differences.is_empty(), "{differences}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_operation_sends_the_tags_of_its_handle() {
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let nightly = session.with_tag("nightly").unwrap();
    let mut entries = Vec::new();
    for case in cases() {
        (case.new)(nightly.clone()).await;
        assert!(dropped(&recorder).await, "{}", case.name);
        entries.extend(recorder.take());
    }
    let mut failures = Vec::new();
    let mut sent = 0;
    let mut drops = 0;
    for entry in entries {
        if entry.kind != "ExecutePlanRequest" {
            continue;
        }
        sent += 1;
        drops += usize::from(is_drop(&entry));
        let request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
        if request.tags != ["nightly"] {
            failures.push(format!("{:?}: tags {:?}", command_of(&entry), request.tags));
        }
    }
    assert_eq!(drops, cases().len());
    assert!(sent > 2 * cases().len(), "only {sent} requests");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_definition_whose_answer_has_no_identifier_is_a_decode_error() {
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let graph = session.create_dataflow_graph().await.unwrap();
    let definitions: Vec<(&str, BoxFuture<'static, spark_connect_async::Result<_>>)> = vec![
        (
            "a materialized view",
            Box::pin(graph.define_materialized_view("mv").into_future()),
        ),
        (
            "a temporary view",
            Box::pin(graph.define_temporary_view("v").into_future()),
        ),
        (
            "a sink",
            Box::pin(graph.define_sink("s", "kafka").into_future()),
        ),
        (
            "a flow",
            Box::pin(graph.define_flow("f", "t", session.range(3)).into_future()),
        ),
        (
            "an Auto CDC flow",
            Box::pin(
                graph
                    .define_auto_cdc_flow("c", "t", "s", ["k"], "ts")
                    .into_future(),
            ),
        ),
    ];
    for (name, definition) in definitions {
        let error = definition.await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Decode, "{name}");
        assert_eq!(
            error.message(),
            "the server returned no identifier",
            "{name}"
        );
    }
    drop(graph);
    assert!(dropped(&recorder).await);
}
