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

//! The reads, writes, merges and local data of the async API send the requests and build the
//! plans of `apache-spark-connect` for the same operations, as the in-process server of its
//! request snapshots records them.

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;
mod support;

use std::collections::HashMap;
use std::future::IntoFuture;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, NaiveDate, NaiveTime, TimeDelta};
use futures::future::BoxFuture;
use prost::Message;
use spark_connect::column::{col as oc, lit as olit};
use spark_connect::plan::ParseFormat as OldParseFormat;
use spark_connect::row::{Row as OldRow, StructValue, Value as Old};
use spark_connect::SparkSession;
use spark_connect_async::readwriter::{OptionValue, ParseFormat, SaveMode};
use spark_connect_async::types::DataType;
use spark_connect_async::{col, lit, CalendarInterval, Decimal, ErrorKind, Row, Session, Value};
use spark_connect_proto as proto;

const SESSION_ID: &str = "00000000-0000-4000-8000-00000000005e";

struct Case {
    name: &'static str,
    old: fn(&SparkSession) -> Option<proto::Relation>,
    new: fn(Session) -> BoxFuture<'static, Option<proto::Relation>>,
}

/// A case whose operations return DataFrames, compared by their relations and requests.
macro_rules! read {
    ($name:expr, |$o:ident| $old:expr, |$n:ident| $new:expr) => {
        Case {
            name: $name,
            old: |$o| Some($old.to_relation()),
            new: |$n| Box::pin(async move { Some($new.to_relation()) }),
        }
    };
}

/// A case whose operations are writes, compared by their requests.
macro_rules! write {
    ($name:expr, |$o:ident| $old:expr, |$n:ident| $new:expr) => {
        Case {
            name: $name,
            old: |$o| {
                $old.unwrap();
                None
            },
            new: |$n| {
                Box::pin(async move {
                    $new.await.unwrap();
                    None
                })
            },
        }
    };
}

fn one_option() -> HashMap<String, String> {
    HashMap::from([("a".to_string(), "1".to_string())])
}

fn old_range(s: &SparkSession) -> spark_connect::DataFrame {
    s.range(3).unwrap()
}

fn names() -> Vec<String> {
    vec!["id".to_string(), "name".to_string()]
}

fn schema() -> DataType {
    DataType::from_ddl("id BIGINT, name STRING").unwrap()
}

fn old_rows(count: i64) -> Vec<OldRow> {
    (0..count)
        .map(|i| {
            OldRow::new(
                names(),
                vec![Old::Long(i), Old::String(format!("name {i}"))],
            )
        })
        .collect()
}

fn new_rows(count: i64) -> Vec<Vec<Value>> {
    (0..count)
        .map(|i| vec![Value::Long(i), Value::String(format!("name {i}"))])
        .collect()
}

fn arrow_batch(rows: i64) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("name", ArrowType::Utf8, true),
    ]));
    let names: Vec<String> = (0..rows).map(|i| format!("name {i}")).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows)) as ArrayRef,
            Arc::new(StringArray::from(names)) as ArrayRef,
        ],
    )
    .unwrap()
}

/// A row of every type that local rows send, with the old API's values and the new API's.
fn every_type() -> (DataType, Vec<Old>, Vec<Value>) {
    let schema = DataType::from_ddl(
        "b BOOLEAN, y TINYINT, s SMALLINT, i INT, l BIGINT, f FLOAT, d DOUBLE, \
         m DECIMAL(10, 2), t STRING, x BINARY, dt DATE, ts TIMESTAMP, tn TIMESTAMP_NTZ, \
         tm TIME, iv INTERVAL DAY TO SECOND, v VARIANT, a ARRAY<INT>, mp MAP<STRING, INT>, \
         st STRUCT<p: INT, q: STRING>, n STRING",
    )
    .unwrap();
    let micros = 1_700_000_000_123_456;
    let old = vec![
        Old::Bool(true),
        Old::Byte(-1),
        Old::Short(2),
        Old::Integer(3),
        Old::Long(4),
        Old::Float(1.5),
        Old::Double(2.5),
        Old::Decimal {
            value: "12.30".to_string(),
            precision: Some(4),
            scale: Some(2),
        },
        Old::String("t".to_string()),
        Old::Binary(vec![1, 2]),
        Old::Date(19_000),
        Old::Timestamp(micros),
        Old::TimestampNtz(micros),
        Old::Time(3_600_000_001_000),
        Old::DayTimeInterval(90_061_000_001),
        Old::Variant {
            value: vec![12, 1],
            metadata: vec![1, 0, 0],
        },
        Old::List(vec![Old::Integer(1), Old::Null]),
        Old::Map(vec![(Old::String("k".to_string()), Old::Integer(1))]),
        Old::Struct(StructValue::new(
            Arc::from(["p".to_string(), "q".to_string()]),
            vec![Old::Integer(5), Old::String("w".to_string())],
        )),
        Old::Null,
    ];
    let instant = DateTime::from_timestamp_micros(micros).unwrap();
    let new = vec![
        Value::Boolean(true),
        Value::Byte(-1),
        Value::Short(2),
        Value::Int(3),
        Value::Long(4),
        Value::Float(1.5),
        Value::Double(2.5),
        Value::Decimal("12.30".parse().unwrap()),
        Value::String("t".to_string()),
        Value::Binary(vec![1, 2]),
        Value::Date(NaiveDate::from_epoch_days(19_000).unwrap()),
        Value::Timestamp(instant),
        Value::TimestampNtz(instant.naive_utc()),
        Value::Time(NaiveTime::from_num_seconds_from_midnight_opt(3_600, 1_000).unwrap()),
        Value::DayTimeInterval(TimeDelta::microseconds(90_061_000_001)),
        Value::Variant {
            value: vec![12, 1],
            metadata: vec![1, 0, 0],
        },
        Value::Array(vec![Value::Int(1), Value::Null]),
        Value::Map(vec![(Value::String("k".to_string()), Value::Int(1))]),
        Value::Struct(Row::from_iter([
            ("p", Value::Int(5)),
            ("q", Value::String("w".to_string())),
        ])),
        Value::Null,
    ];
    (schema, old, new)
}

fn cases() -> Vec<Case> {
    vec![
        read!("load", |s| s.read().load(Some("/data/x")), |s| s
            .read()
            .load("/data/x")),
        read!(
            "load without a path",
            |s| s.read().format("delta").load(None),
            |s| s.read().format("delta").load_without_path()
        ),
        read!(
            "format, schema in SQL syntax and option",
            |s| s
                .read()
                .format("parquet")
                .schema("a INT".to_string())
                .option("k", "v")
                .load(Some("/p")),
            |s| s
                .read()
                .format("parquet")
                .schema("a INT")
                .option("k", "v")
                .load("/p")
        ),
        read!(
            "schema as a type",
            |s| s.read().schema(schema().json()).load(Some("/p")),
            |s| s.read().schema(schema()).load("/p")
        ),
        read!(
            "options",
            |s| s
                .read()
                .options(one_option())
                .option("b", "true")
                .load(Some("/p")),
            |s| s.read().options([("a", 1)]).option("b", true).load("/p")
        ),
        read!(
            "load_paths",
            |s| s
                .read()
                .format("json")
                .load_paths(vec!["/a".to_string(), "/b".to_string()]),
            |s| s.read().format("json").load_paths(["/a", "/b"])
        ),
        read!("table", |s| s.read().option("k", "v").table("db.t"), |s| s
            .read()
            .option("k", "v")
            .table("db.t")),
        read!(
            "changes",
            |s| s.read().option("startingVersion", "1").changes("db.t"),
            |s| s.read().option("startingVersion", 1).changes("db.t")
        ),
        read!("json", |s| s.read().json("/a.json"), |s| s
            .read()
            .json("/a.json")),
        read!("parquet", |s| s.read().parquet("/a.parquet"), |s| s
            .read()
            .parquet("/a.parquet")),
        read!(
            "csv",
            |s| s.read().option("header", "true").csv("/a.csv"),
            |s| s.read().option("header", true).csv("/a.csv")
        ),
        read!("orc", |s| s.read().orc("/a.orc"), |s| s
            .read()
            .orc("/a.orc")),
        read!("text", |s| s.read().text("/a.txt"), |s| s
            .read()
            .text("/a.txt")),
        read!(
            "xml",
            |s| s.read().option("rowTag", "r").xml("/a.xml"),
            |s| s.read().option("rowTag", "r").xml("/a.xml")
        ),
        read!("jdbc", |s| s.read().jdbc("jdbc:x", "t", None), |s| s
            .read()
            .jdbc("jdbc:x", "t", None)),
        read!(
            "jdbc with predicates",
            |s| s.read().jdbc(
                "jdbc:x",
                "t",
                Some(vec!["a < 1".to_string(), "a >= 1".to_string()])
            ),
            |s| s.read().jdbc("jdbc:x", "t", Some(&["a < 1", "a >= 1"]))
        ),
        read!(
            "parse with a schema in SQL syntax",
            |s| s
                .read()
                .schema("id BIGINT".to_string())
                .option("sep", ";")
                .parse(OldParseFormat::Csv, &old_range(s))
                .unwrap(),
            |s| s
                .read()
                .schema("id BIGINT")
                .option("sep", ";")
                .parse(ParseFormat::Csv, &s.range(3))
                .await
                .unwrap()
        ),
        read!(
            "parse with a schema as JSON text",
            |s| s
                .read()
                .schema(schema().json())
                .parse(OldParseFormat::Json, &old_range(s))
                .unwrap(),
            |s| s
                .read()
                .schema(schema().json())
                .parse(ParseFormat::Json, &s.range(3))
                .await
                .unwrap()
        ),
        read!(
            "parse with a schema as a type, and without one",
            |s| {
                let typed = s
                    .read()
                    .schema(schema().json())
                    .parse(OldParseFormat::Xml, &old_range(s))
                    .unwrap();
                s.read().parse(OldParseFormat::Xml, &typed).unwrap()
            },
            |s| {
                let typed = s
                    .read()
                    .schema(schema())
                    .parse(ParseFormat::Xml, &s.range(3))
                    .await
                    .unwrap();
                s.read().parse(ParseFormat::Xml, &typed).await.unwrap()
            }
        ),
        write!(
            "save",
            |s| old_range(s)
                .write()
                .format("parquet")
                .mode("overwrite")
                .option("k", "v")
                .save(Some("/out")),
            |s| s
                .range(3)
                .write()
                .format("parquet")
                .mode(SaveMode::Overwrite)
                .option("k", "v")
                .save("/out")
        ),
        write!(
            "save of options",
            |s| old_range(s)
                .write()
                .options(one_option())
                .save(Some("/out")),
            |s| s.range(3).write().options([("a", 1)]).save("/out")
        ),
        write!(
            "save without a path",
            |s| old_range(s).write().save(None),
            |s| s.range(3).write().save_without_path()
        ),
        write!(
            "modes",
            |s| {
                for mode in ["append", "overwrite", "error", "ignore"] {
                    old_range(s).write().mode(mode).save(Some("/out")).unwrap();
                }
                Ok::<(), ()>(())
            },
            |s| async move {
                use SaveMode::*;
                for mode in [Append, Overwrite, ErrorIfExists, Ignore] {
                    s.range(3).write().mode(mode).save("/out").await?;
                }
                spark_connect_async::Result::<()>::Ok(())
            }
        ),
        write!(
            "partition_by, bucket_by and sort_by",
            |s| old_range(s)
                .write()
                .partition_by(["a", "b"])
                .bucket_by(4, ["c"])
                .sort_by(["d"])
                .save_as_table("db.t"),
            |s| s
                .range(3)
                .write()
                .partition_by(["a", "b"])
                .bucket_by(4, ["c"])
                .sort_by(["d"])
                .save_as_table("db.t")
        ),
        write!(
            "cluster_by",
            |s| old_range(s).write().cluster_by(["a"]).save_as_table("db.t"),
            |s| s.range(3).write().cluster_by(["a"]).save_as_table("db.t")
        ),
        write!(
            "save_as_table",
            |s| old_range(s).write().mode("append").save_as_table("db.t"),
            |s| s
                .range(3)
                .write()
                .mode(SaveMode::Append)
                .save_as_table("db.t")
        ),
        write!(
            "insert_into",
            |s| old_range(s).write().insert_into("db.t"),
            |s| s.range(3).write().insert_into("db.t")
        ),
        write!(
            "insert_into of overwrite",
            |s| old_range(s).write().mode("overwrite").insert_into("db.t"),
            |s| s
                .range(3)
                .write()
                .mode(SaveMode::Overwrite)
                .insert_into("db.t")
        ),
        write!("json", |s| old_range(s).write().json("/o.json"), |s| s
            .range(3)
            .write()
            .json("/o.json")),
        write!(
            "parquet",
            |s| old_range(s).write().parquet("/o.parquet"),
            |s| s.range(3).write().parquet("/o.parquet")
        ),
        write!(
            "csv",
            |s| old_range(s).write().option("header", "true").csv("/o.csv"),
            |s| s.range(3).write().option("header", true).csv("/o.csv")
        ),
        write!("orc", |s| old_range(s).write().orc("/o.orc"), |s| s
            .range(3)
            .write()
            .orc("/o.orc")),
        write!("text", |s| old_range(s).write().text("/o.txt"), |s| s
            .range(3)
            .write()
            .text("/o.txt")),
        write!("xml", |s| old_range(s).write().xml("/o.xml"), |s| s
            .range(3)
            .write()
            .xml("/o.xml")),
        write!(
            "create",
            |s| old_range(s)
                .write_to("db.t")
                .using("delta")
                .option("k", "v")
                .table_property("p", "q")
                .partition_by([oc("a")])
                .create(),
            |s| s
                .range(3)
                .write_to("db.t")
                .using("delta")
                .option("k", "v")
                .table_property("p", "q")
                .partitioned_by([col("a")])
                .create()
        ),
        write!(
            "create of options",
            |s| old_range(s).write_to("db.t").options(one_option()).create(),
            |s| s.range(3).write_to("db.t").options([("a", 1)]).create()
        ),
        write!(
            "replace",
            |s| old_range(s).write_to("db.t").replace(),
            |s| s.range(3).write_to("db.t").replace()
        ),
        write!(
            "create_or_replace",
            |s| old_range(s)
                .write_to("db.t")
                .cluster_by(["a"])
                .create_or_replace(),
            |s| s
                .range(3)
                .write_to("db.t")
                .cluster_by(["a"])
                .create_or_replace()
        ),
        write!("append", |s| old_range(s).write_to("db.t").append(), |s| s
            .range(3)
            .write_to("db.t")
            .append()),
        write!(
            "overwrite",
            |s| old_range(s).write_to("db.t").overwrite(oc("a").gt(olit(1))),
            |s| s.range(3).write_to("db.t").overwrite(col("a").gt(lit(1)))
        ),
        write!(
            "overwrite_partitions",
            |s| old_range(s).write_to("db.t").overwrite_partitions(),
            |s| s.range(3).write_to("db.t").overwrite_partitions()
        ),
        write!(
            "merge of every clause",
            |s| {
                let one = || HashMap::from([("x".to_string(), oc("s.x"))]);
                old_range(s)
                    .merge_into("db.t", oc("t.id").eq(oc("s.id")))
                    .when_matched(Some(oc("s.d")))
                    .delete()
                    .when_matched(None)
                    .update(one())
                    .when_matched(None)
                    .update_all()
                    .when_not_matched(Some(oc("s.n")))
                    .insert(one())
                    .when_not_matched(None)
                    .insert_all()
                    .when_not_matched_by_source(Some(oc("t.o")))
                    .delete()
                    .when_not_matched_by_source(None)
                    .update(one())
                    .when_not_matched_by_source(None)
                    .update_all()
                    .with_schema_evolution()
                    .merge()
            },
            |s| {
                let one = || [("x", col("s.x"))];
                s.range(3)
                    .merge_into("db.t", col("t.id").eq(col("s.id")))
                    .when_matched(Some(col("s.d")))
                    .delete()
                    .when_matched(None)
                    .update(one())
                    .when_matched(None)
                    .update_all()
                    .when_not_matched(Some(col("s.n")))
                    .insert(one())
                    .when_not_matched(None)
                    .insert_all()
                    .when_not_matched_by_source(Some(col("t.o")))
                    .delete()
                    .when_not_matched_by_source(None)
                    .update(one())
                    .when_not_matched_by_source(None)
                    .update_all()
                    .with_schema_evolution()
                    .into_future()
            }
        ),
        read!(
            "create_dataframe",
            |s| s.create_dataframe(old_rows(2), schema()).unwrap(),
            |s| s.create_dataframe(new_rows(2), schema()).await.unwrap()
        ),
        read!(
            "create_dataframe of no rows",
            |s| s.create_dataframe(Vec::new(), schema()).unwrap(),
            |s| s
                .create_dataframe(Vec::<Vec<Value>>::new(), schema())
                .await
                .unwrap()
        ),
        read!(
            "create_dataframe of large rows, uploaded",
            |s| s.create_dataframe(old_rows(5_000), schema()).unwrap(),
            |s| s.create_dataframe(new_rows(5_000), schema()).await.unwrap()
        ),
        read!(
            "create_dataframe of every type",
            |s| {
                let (schema, old, _) = every_type();
                let names = schema_names(&schema);
                s.create_dataframe(vec![OldRow::new(names, old)], schema)
                    .unwrap()
            },
            |s| {
                let (schema, _, new) = every_type();
                s.create_dataframe([new], schema).await.unwrap()
            }
        ),
        read!(
            "create_dataframe of collected rows",
            |s| s.create_dataframe(old_rows(1), schema()).unwrap(),
            |s| {
                let row: Row = names().into_iter().zip(new_rows(1).remove(0)).collect();
                s.create_dataframe([row], schema()).await.unwrap()
            }
        ),
        read!(
            "create_dataframe_from_arrow",
            |s| s
                .create_dataframe_from_arrow(vec![arrow_batch(2)], None)
                .unwrap(),
            |s| s
                .create_dataframe_from_arrow([arrow_batch(2)])
                .await
                .unwrap()
        ),
        read!(
            "create_dataframe_from_arrow with a schema",
            |s| s
                .create_dataframe_from_arrow(vec![arrow_batch(2)], schema())
                .unwrap(),
            |s| s
                .create_dataframe_from_arrow([arrow_batch(2)])
                .schema(schema())
                .await
                .unwrap()
        ),
        read!(
            "create_dataframe_from_arrow of large data, uploaded",
            |s| s
                .create_dataframe_from_arrow(vec![arrow_batch(2_000)], None)
                .unwrap(),
            |s| s
                .create_dataframe_from_arrow([arrow_batch(2_000)])
                .await
                .unwrap()
        ),
        read!(
            "create_dataframe_from_arrow of no batches, and of no rows",
            |s| {
                s.create_dataframe_from_arrow(Vec::new(), None).unwrap();
                s.create_dataframe_from_arrow(vec![arrow_batch(0)], schema())
                    .unwrap()
            },
            |s| {
                s.create_dataframe_from_arrow([]).await.unwrap();
                s.create_dataframe_from_arrow([arrow_batch(0)])
                    .schema(schema())
                    .await
                    .unwrap()
            }
        ),
    ]
}

fn schema_names(schema: &DataType) -> Vec<String> {
    match schema {
        DataType::Struct { fields } => fields.iter().map(|field| field.name.clone()).collect(),
        _ => unreachable!("a schema is a struct"),
    }
}

/// Each entry as its message, with the operation ids, which the client chooses at random,
/// removed.
fn messages(entries: Vec<server::Entry>) -> Vec<String> {
    entries
        .into_iter()
        .map(|entry| match entry.kind.as_str() {
            "ExecutePlanRequest" => {
                let mut request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
                request.operation_id = None;
                format!("{request:?}")
            }
            "AnalyzePlanRequest" => {
                format!(
                    "{:?}",
                    proto::AnalyzePlanRequest::decode(&entry.bytes[..]).unwrap()
                )
            }
            _ => format!("{entry:?}"),
        })
        .collect()
}

fn start() -> (Arc<server::Recorder>, String) {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let url = format!("sc://{address}/;session_id={SESSION_ID};user_id=snapshots");
    (recorder, url)
}

#[tokio::test]
async fn reads_writes_merges_and_local_data_send_the_requests_of_the_old_api() {
    let (recorder, url) = start();
    let mut both = Vec::new();
    for case in cases() {
        let old_session = SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        let old_relation = (case.old)(&old_session);
        let old = messages(recorder.take());
        let session = Session::builder()
            .remote(url.clone())
            .connect()
            .await
            .unwrap();
        let new_relation = (case.new)(session).await;
        let new = messages(recorder.take());
        assert!(
            !old.is_empty() || old_relation.is_some(),
            "{}: the old API sent and built nothing",
            case.name
        );
        both.push((case.name, (old, old_relation), (new, new_relation)));
    }
    let differences = support::differences(both);
    assert!(differences.is_empty(), "{differences}");
}

fn merge_command(entries: Vec<server::Entry>) -> proto::MergeIntoTableCommand {
    let [entry] = &entries[..] else {
        panic!("not one request: {entries:?}");
    };
    let request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
    match request.plan.and_then(|plan| plan.op_type) {
        Some(proto::plan::OpType::Command(proto::Command {
            command_type: Some(proto::command::CommandType::MergeIntoTableCommand(command)),
        })) => command,
        other => panic!("not a merge: {other:?}"),
    }
}

#[tokio::test]
async fn the_assignments_of_a_merge_are_sent_in_their_order() {
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    session
        .range(3)
        .merge_into("db.t", col("t.id").eq(col("s.id")))
        .when_matched(None)
        .update([("c", lit(3)), ("a", lit(1)), ("b", lit(2))])
        .when_not_matched(None)
        .insert([("z", col("s.z")), ("y", col("s.y"))])
        .await
        .unwrap();
    let command = merge_command(recorder.take());
    let targets = |actions: &[proto::Expression]| -> Vec<String> {
        let [action] = actions else {
            panic!("not one action: {actions:?}");
        };
        let Some(proto::expression::ExprType::MergeAction(action)) = &action.expr_type else {
            panic!("not a merge action: {action:?}");
        };
        action
            .assignments
            .iter()
            .map(|assignment| {
                match assignment
                    .key
                    .as_ref()
                    .and_then(|key| key.expr_type.as_ref())
                {
                    Some(proto::expression::ExprType::ExpressionString(key)) => {
                        key.expression.clone()
                    }
                    other => panic!("not a target column: {other:?}"),
                }
            })
            .collect()
    };
    assert_eq!(targets(&command.match_actions), ["c", "a", "b"]);
    assert_eq!(targets(&command.not_matched_actions), ["z", "y"]);
}

#[test]
fn save_modes_parse_in_any_case_and_other_text_is_rejected() {
    for (text, mode) in [
        ("append", SaveMode::Append),
        ("APPEND", SaveMode::Append),
        ("Overwrite", SaveMode::Overwrite),
        ("error", SaveMode::ErrorIfExists),
        ("errorifexists", SaveMode::ErrorIfExists),
        ("ignore", SaveMode::Ignore),
    ] {
        assert_eq!(text.parse::<SaveMode>().unwrap(), mode, "{text:?}");
    }
    for text in ["bad", "default", "", " append"] {
        let error = text.parse::<SaveMode>().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{text:?}");
    }
    assert_eq!(SaveMode::default(), SaveMode::ErrorIfExists);
}

#[tokio::test]
async fn options_are_sent_as_text_and_a_later_value_replaces_an_earlier_one() {
    let session = Session::builder()
        .remote("sc://localhost:1")
        .connect()
        .await
        .unwrap();
    let df = session
        .read()
        .option("s", "x")
        .option("b", true)
        .option("i", 3)
        .option("l", 4_000_000_000i64)
        .option("f", 1.5)
        .option("g", 2.0)
        .options([("s", "y"), ("o", "z")])
        .option("o", OptionValue::from(&"w".to_string()))
        .load("/p");
    let Some(proto::relation::RelType::Read(read)) = df.to_relation().rel_type else {
        panic!("not a read");
    };
    let Some(proto::read::ReadType::DataSource(source)) = read.read_type else {
        panic!("not a data source");
    };
    let expected: HashMap<String, String> = [
        ("s", "y"),
        ("b", "true"),
        ("i", "3"),
        ("l", "4000000000"),
        ("f", "1.5"),
        ("g", "2"),
        ("o", "w"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect();
    assert_eq!(source.options, expected);
}

#[tokio::test]
async fn local_rows_that_their_schema_cannot_hold_are_rejected() {
    let session = Session::builder()
        .remote("sc://localhost:1")
        .connect()
        .await
        .unwrap();
    let cases: Vec<(&str, Vec<Value>, DataType)> = vec![
        ("a row too short", vec![Value::Long(1)], schema()),
        (
            "a row too long",
            vec![Value::Long(1), Value::Null, Value::Null],
            schema(),
        ),
        (
            "a string for a BIGINT",
            vec![Value::String("1".to_string()), Value::Null],
            schema(),
        ),
        (
            "a decimal of more digits than its type's scale",
            vec![Value::Decimal(Decimal::new(1_234, 4, 3).unwrap())],
            DataType::from_ddl("m DECIMAL(10, 2)").unwrap(),
        ),
        (
            "a year-month interval, which is not sent as its months",
            vec![Value::YearMonthInterval(14)],
            DataType::from_ddl("i INT").unwrap(),
        ),
        (
            "a day-time interval beyond its microseconds",
            vec![Value::DayTimeInterval(TimeDelta::MAX)],
            DataType::from_ddl("iv INTERVAL DAY TO SECOND").unwrap(),
        ),
        (
            "a calendar interval, which is not sent as its text",
            vec![Value::CalendarInterval(CalendarInterval::new(1, 2, 3))],
            DataType::from_ddl("s STRING").unwrap(),
        ),
        (
            "a schema that is not a struct",
            vec![Value::Long(1)],
            DataType::Long,
        ),
    ];
    for (name, row, schema) in cases {
        let error = session
            .create_dataframe([row], schema)
            .await
            .err()
            .unwrap_or_else(|| panic!("{name}: accepted"));
        assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{name}: {error}");
    }
}

#[tokio::test]
async fn local_rows_are_sent_in_the_plan_until_they_are_large() {
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let small = session
        .create_dataframe(new_rows(2), schema())
        .await
        .unwrap();
    assert!(recorder.take().is_empty());
    assert!(matches!(
        small.to_relation().rel_type,
        Some(proto::relation::RelType::LocalRelation(
            proto::LocalRelation { data: Some(_), .. }
        ))
    ));
    let large = session
        .create_dataframe(new_rows(5_000), schema())
        .await
        .unwrap();
    let kinds: Vec<String> = recorder
        .take()
        .into_iter()
        .map(|entry| entry.kind)
        .collect();
    assert!(kinds.contains(&"ConfigRequest".to_string()), "{kinds:?}");
    assert!(
        kinds.contains(&"AddArtifactsRequest".to_string()),
        "{kinds:?}"
    );
    assert!(matches!(
        large.to_relation().rel_type,
        Some(proto::relation::RelType::ChunkedCachedLocalRelation(_))
    ));
}

/// Prints like a `HashMap`, but in the order given, so that the test doesn't depend on hashing.
struct MapPrintedInOrder<'a>(&'a [(&'a str, &'a dyn std::fmt::Debug)]);

impl std::fmt::Debug for MapPrintedInOrder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.0.iter().copied()).finish()
    }
}

#[test]
fn the_maps_of_a_request_are_compared_in_key_order() {
    type Entries<'a> = [(&'a str, &'a dyn std::fmt::Debug); 2];
    let id = oc("id").to_proto();
    let braces = spark_connect::lit_string("}, \"a\": {").to_proto();
    let yx: Entries = [("y", &"2"), ("x", &"1")];
    let xy: Entries = [("x", &"1"), ("y", &"2")];
    let (open, close) = ("{\"a\", ", "}, \"b\": {\\");
    let cases: [(&str, Entries, Entries); 4] = [
        (
            "strings",
            [("b", &"2"), ("a", &"1")],
            [("a", &"1"), ("b", &"2")],
        ),
        (
            "messages",
            [("b", &braces), ("a", &id)],
            [("a", &id), ("b", &braces)],
        ),
        (
            "maps",
            [("b", &MapPrintedInOrder(&yx)), ("a", &id)],
            [("a", &id), ("b", &MapPrintedInOrder(&xy))],
        ),
        (
            "strings with braces, quotes, commas and a backslash",
            [("b", &open), ("a", &close)],
            [("a", &close), ("b", &open)],
        ),
    ];
    for (values, unordered, ordered) in cases {
        let expected = format!("{:?}", MapPrintedInOrder(&ordered));
        for entries in [unordered, ordered] {
            let map = MapPrintedInOrder(&entries);
            assert_eq!(support::message_text(&map), expected, "{values}");
            let text = format!("{map:?}");
            let in_text = format!("{expected:?}");
            assert_eq!(support::message_text(&text), in_text, "{values}, in a text");
        }
        let in_a_text = format!("{values}, in a text");
        let (first, second) = (MapPrintedInOrder(&unordered), MapPrintedInOrder(&ordered));
        let texts = (vec![format!("{first:?}")], vec![format!("{second:?}")]);
        let both: [(&str, &dyn std::fmt::Debug, &dyn std::fmt::Debug); 2] =
            [(values, &first, &second), (&in_a_text, &texts.0, &texts.1)];
        let differences = support::differences(Vec::from(both));
        assert!(differences.is_empty(), "{differences}");
    }
}

// Above the futures that hold no execution of the core inline, and below those that do.
const FUTURE_SIZE_BOUND: usize = 16 * 1024;

#[test]
fn the_future_of_parse_is_at_most_16_kib() {
    fn size<R>(_: impl FnOnce(&Session, &spark_connect_async::DataFrame) -> R) -> usize {
        std::mem::size_of::<R>()
    }
    let parse = size(|s, d| s.read().parse(ParseFormat::Csv, d));
    assert!(parse <= FUTURE_SIZE_BOUND, "{parse}");
}
