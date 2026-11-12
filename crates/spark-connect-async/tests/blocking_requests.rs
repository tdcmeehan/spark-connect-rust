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

//! The blocking API sends the requests of the async API for the same calls, as the in-process
//! server of the old API's request snapshots records them, and its iterators yield what the
//! async streams yield.
//!
//! Each import of a Spark ML trait here is the blocking one, so the async calls name theirs.

#![cfg(feature = "blocking")]

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;
mod support;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{ArrayRef, Int64Array};
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::record_batch::RecordBatch;
use futures::future::BoxFuture;
use futures::StreamExt;
use prost::Message;
use spark_connect_async::actions::{Checkpoint, ExplainMode, StorageLevel, TempView};
use spark_connect_async::blocking::{self, Estimator, Evaluator};
use spark_connect_async::ml::{
    self, BinaryClassificationEvaluator, CrossValidator, LinearRegression, LinearSVC,
    LogisticRegression, MaxAbsScaler, MulticlassClassificationEvaluator, Pipeline,
    RegressionEvaluator, StandardScaler, StringIndexer, VectorAssembler,
};
use spark_connect_async::pipelines::{self, ScdType};
use spark_connect_async::readwriter::{ParseFormat, SaveMode};
use spark_connect_async::session::ArtifactKind;
use spark_connect_async::streaming::{self, OutputMode, Trigger};
use spark_connect_async::types::DataType;
use spark_connect_async::{col, lit, DataFrame, Session, Value};
use spark_connect_proto as proto;

const SESSION_ID: &str = "00000000-0000-4000-8000-000000000b14";

struct Case {
    name: &'static str,
    /// Whether the case subscribes to the events of streaming queries, which the server then
    /// holds until the client removes the subscription.
    subscribes: bool,
    blocking: fn(&blocking::Session),
    awaited: fn(Session) -> BoxFuture<'static, ()>,
}

/// A case of a call of both APIs. Their results are not compared: the recording server answers
/// most requests with no rows.
macro_rules! case {
    ($name:expr, |$b:ident| $blocking:expr, |$a:ident| $awaited:expr) => {
        case!($name, false, |$b| $blocking, |$a| $awaited)
    };
    ($name:expr, $subscribes:expr, |$b:ident| $blocking:expr, |$a:ident| $awaited:expr) => {
        Case {
            name: $name,
            subscribes: $subscribes,
            blocking: |$b| {
                let _ = $blocking;
            },
            awaited: |$a| {
                Box::pin(async move {
                    let _ = $awaited;
                })
            },
        }
    };
}

fn three(s: &blocking::Session) -> blocking::DataFrame {
    s.as_async().sql(server::THREE_BATCHES).run().unwrap()
}

async fn three_async(s: &Session) -> DataFrame {
    s.sql(server::THREE_BATCHES).await.unwrap()
}

fn ids() -> DataType {
    DataType::from_ddl("id BIGINT").unwrap()
}

fn batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int64, true)]));
    let ids = Arc::new(Int64Array::from_iter_values(0..3)) as ArrayRef;
    RecordBatch::try_new(schema, vec![ids]).unwrap()
}

fn rows() -> Vec<Vec<Value>> {
    vec![vec![Value::Long(1)], vec![Value::Long(2)]]
}

/// A file to upload, in the scratch directory of the tests.
fn artifact() -> String {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("blocking-requests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("notes.txt");
    std::fs::write(&path, "notes").unwrap();
    path.display().to_string()
}

fn session_cases() -> Vec<Case> {
    vec![
        case!("version", |b| b.version(), |a| a.version().await),
        case!("conf set", |b| b.conf().set("k", "v"), |a| a
            .conf()
            .set("k", "v")
            .await),
        case!(
            "conf set_all",
            |b| b.conf().set_all([("a", "1"), ("b", "2")]),
            |a| a.conf().set_all([("a", "1"), ("b", "2")]).await
        ),
        case!("conf get", |b| b.conf().get("k"), |a| a
            .conf()
            .get("k")
            .await),
        case!("conf get_all", |b| b.conf().get_all(), |a| a
            .conf()
            .get_all()
            .await),
        case!("conf unset", |b| b.conf().unset("k"), |a| a
            .conf()
            .unset("k")
            .await),
        case!("conf is_modifiable", |b| b.conf().is_modifiable("k"), |a| a
            .conf()
            .is_modifiable("k")
            .await),
        case!(
            "a SQL command",
            |b| b.as_async().sql("SET x = 1").run(),
            |a| a.sql("SET x = 1").await
        ),
        case!(
            "a SQL query with arguments and a DataFrame",
            |b| b
                .as_async()
                .sql("SELECT :a, ? FROM {t}")
                .named_arg("a", 1)
                .arg("two")
                .reference("t", &b.as_async().range(3))
                .run()
                .and_then(|df| df.collect()),
            |a| {
                let df = a
                    .sql("SELECT :a, ? FROM {t}")
                    .named_arg("a", 1)
                    .arg("two")
                    .reference("t", &a.range(3))
                    .await
                    .unwrap();
                df.collect().await
            }
        ),
        case!(
            "add_artifacts",
            |b| b
                .as_async()
                .add_artifacts([artifact()])
                .kind(ArtifactKind::File)
                .run(),
            |a| a.add_artifacts([artifact()]).kind(ArtifactKind::File).await
        ),
        case!("interrupt_all", |b| b.interrupt_all(), |a| a
            .interrupt_all()
            .await),
        case!("interrupt_tag", |b| b.interrupt_tag("etl"), |a| a
            .interrupt_tag("etl")
            .await),
        case!(
            "interrupt_operation",
            |b| b.interrupt_operation("00000000-0000-4000-8000-000000000001"),
            |a| a
                .interrupt_operation("00000000-0000-4000-8000-000000000001")
                .await
        ),
        case!(
            "with_tag",
            |b| three(&b.with_tag("etl").unwrap()).collect(),
            |a| three_async(&a.with_tag("etl").unwrap())
                .await
                .collect()
                .await
        ),
        case!(
            "create_dataframe",
            |b| b
                .create_dataframe(rows(), ids())
                .and_then(|df| df.collect()),
            |a| a
                .create_dataframe(rows(), ids())
                .await
                .unwrap()
                .collect()
                .await
        ),
        case!(
            "create_dataframe_from_arrow",
            |b| b
                .as_async()
                .create_dataframe_from_arrow([batch()])
                .schema(ids())
                .run()
                .and_then(|df| df.collect()),
            |a| a
                .create_dataframe_from_arrow([batch()])
                .schema(ids())
                .await
                .unwrap()
                .collect()
                .await
        ),
        case!(
            "a session from an async session",
            |b| crate_session(b).blocking().version(),
            |a| a.version().await
        ),
        case!("stop", |b| b.stop(), |a| a.stop().await),
    ]
}

/// The async session of `session`, through the conversion.
fn crate_session(session: &blocking::Session) -> Session {
    Session::from(session.clone())
}

fn dataframe_cases() -> Vec<Case> {
    vec![
        case!(
            "stream",
            |b| three(b).stream().unwrap().collect::<Vec<_>>(),
            |a| three_async(&a)
                .await
                .stream()
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        ),
        case!(
            "stream_as",
            |b| three(b).stream_as::<i64>().unwrap().collect::<Vec<_>>(),
            |a| three_async(&a)
                .await
                .stream_as::<i64>()
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        ),
        case!("collect", |b| three(b).collect(), |a| three_async(&a)
            .await
            .collect()
            .await),
        case!("collect_rows", |b| three(b).collect_rows(), |a| {
            three_async(&a).await.collect_rows().await
        }),
        case!("collect_as", |b| three(b).collect_as::<(i64,)>(), |a| {
            three_async(&a).await.collect_as::<(i64,)>().await
        }),
        case!("count", |b| three(b).count(), |a| three_async(&a)
            .await
            .count()
            .await),
        case!("take", |b| three(b).take(2), |a| three_async(&a)
            .await
            .take(2)
            .await),
        case!("first", |b| three(b).first(), |a| three_async(&a)
            .await
            .first()
            .await),
        case!("is_empty", |b| three(b).is_empty(), |a| three_async(&a)
            .await
            .is_empty()
            .await),
        case!("schema", |b| three(b).schema(), |a| three_async(&a)
            .await
            .schema()
            .await),
        case!("columns", |b| three(b).columns(), |a| three_async(&a)
            .await
            .columns()
            .await),
        case!(
            "explain",
            |b| three(b).explain(ExplainMode::Extended),
            |a| three_async(&a).await.explain(ExplainMode::Extended).await
        ),
        case!("cache", |b| three(b).cache(), |a| three_async(&a)
            .await
            .cache()
            .await),
        case!(
            "persist",
            |b| three(b).persist(StorageLevel::DISK_ONLY_2),
            |a| three_async(&a)
                .await
                .persist(StorageLevel::DISK_ONLY_2)
                .await
        ),
        case!("unpersist", |b| three(b).unpersist(true), |a| three_async(
            &a
        )
        .await
        .unpersist(true)
        .await),
        case!("storage_level", |b| three(b).storage_level(), |a| {
            three_async(&a).await.storage_level().await
        }),
        case!(
            "create_temp_view",
            |b| three(b).create_temp_view("v", TempView::default().global().or_replace()),
            |a| three_async(&a)
                .await
                .create_temp_view("v", TempView::default().global().or_replace())
                .await
        ),
        // Kept: the removal at its drop would reach the server after the case.
        case!(
            "checkpoint",
            |b| std::mem::forget(
                three(b)
                    .checkpoint(Checkpoint::default().local().lazy())
                    .unwrap()
            ),
            |a| std::mem::forget(
                three_async(&a)
                    .await
                    .checkpoint(Checkpoint::default().local().lazy())
                    .await
                    .unwrap()
            )
        ),
        case!("input_files", |b| three(b).input_files(), |a| three_async(
            &a
        )
        .await
        .input_files()
        .await),
        case!(
            "same_semantics",
            |b| three(b).same_semantics(&b.as_async().range(3)),
            |a| three_async(&a).await.same_semantics(&a.range(3)).await
        ),
        case!("semantic_hash", |b| three(b).semantic_hash(), |a| {
            three_async(&a).await.semantic_hash().await
        }),
        case!("is_streaming", |b| three(b).is_streaming(), |a| {
            three_async(&a).await.is_streaming().await
        }),
        case!("to_json", |b| three(b).to_json(), |a| three_async(&a)
            .await
            .to_json()
            .await),
        case!(
            "show",
            |b| three(b).as_async().show().rows(5).truncate(0).run(),
            |a| { three_async(&a).await.show().rows(5).truncate(0).await }
        ),
        case!(
            "show to_string",
            |b| three(b).as_async().show().vertical().blocking().to_string(),
            |a| three_async(&a).await.show().vertical().to_string().await
        ),
        case!(
            "a transformation, then an action",
            |b| three(b)
                .as_async()
                .filter(col("id").gt(lit(1)))
                .blocking()
                .count(),
            |a| three_async(&a)
                .await
                .filter(col("id").gt(lit(1)))
                .count()
                .await
        ),
        case!(
            "an async DataFrame converted",
            |b| blocking::DataFrame::from(b.as_async().range(3)).count(),
            |a| a.range(3).count().await
        ),
        case!(
            "a blocking DataFrame into_async",
            |b| three(b).into_async().limit(2).blocking().collect(),
            |a| three_async(&a).await.limit(2).collect().await
        ),
        case!(
            "group_by sum",
            |b| b
                .as_async()
                .range(3)
                .group_by(["id"])
                .blocking()
                .sum(["id"])
                .and_then(|df| df.collect()),
            |a| a
                .range(3)
                .group_by(["id"])
                .sum(["id"])
                .await
                .unwrap()
                .collect()
                .await
        ),
        case!(
            "group_by avg",
            |b| b
                .as_async()
                .range(3)
                .group_by(["id"])
                .blocking()
                .avg(["id"])
                .and_then(|df| df.collect()),
            |a| a
                .range(3)
                .group_by(["id"])
                .avg(["id"])
                .await
                .unwrap()
                .collect()
                .await
        ),
        case!(
            "rollup min",
            |b| b
                .as_async()
                .range(3)
                .rollup(["id"])
                .blocking()
                .min(["id"])
                .and_then(|df| df.collect()),
            |a| a
                .range(3)
                .rollup(["id"])
                .min(["id"])
                .await
                .unwrap()
                .collect()
                .await
        ),
        case!(
            "pivot max",
            |b| b
                .as_async()
                .range(3)
                .group_by(["id"])
                .pivot("id")
                .blocking()
                .max(["id"])
                .and_then(|df| df.collect()),
            |a| a
                .range(3)
                .group_by(["id"])
                .pivot("id")
                .max(["id"])
                .await
                .unwrap()
                .collect()
                .await
        ),
        case!(
            "stat corr",
            |b| b.as_async().range(3).stat().blocking().corr("id", "id"),
            |a| a.range(3).stat().corr("id", "id").await
        ),
        case!(
            "stat cov",
            |b| b.as_async().range(3).stat().blocking().cov("id", "id"),
            |a| a.range(3).stat().cov("id", "id").await
        ),
        case!(
            "stat approx_quantile",
            |b| b
                .as_async()
                .range(3)
                .stat()
                .blocking()
                .approx_quantile(["id"], [0.5, 0.9], 0.01),
            |a| a
                .range(3)
                .stat()
                .approx_quantile(["id"], [0.5, 0.9], 0.01)
                .await
        ),
        case!(
            "the session of a DataFrame",
            |b| {
                let session = b
                    .with_tag("etl")
                    .unwrap()
                    .as_async()
                    .range(3)
                    .blocking()
                    .session();
                (session.version(), session.as_async().sql("SET x = 1").run())
            },
            |a| {
                let df = a.with_tag("etl").unwrap().range(3);
                (
                    df.session().version().await,
                    df.session().sql("SET x = 1").await,
                )
            }
        ),
    ]
}

fn conversion_cases() -> Vec<Case> {
    vec![
        #[cfg(feature = "polars")]
        case!("to_polars", |b| three(b).to_polars(), |a| three_async(&a)
            .await
            .to_polars()
            .await),
        #[cfg(feature = "datafusion")]
        case!(
            "to_datafusion",
            |b| three(b).to_datafusion(&datafusion::prelude::SessionContext::new()),
            |a| three_async(&a)
                .await
                .to_datafusion(&datafusion::prelude::SessionContext::new())
                .await
        ),
    ]
}

fn read_write_cases() -> Vec<Case> {
    vec![
        case!(
            "parse",
            |b| b
                .as_async()
                .read()
                .schema("id INT")
                .option("header", true)
                .blocking()
                .parse(ParseFormat::Csv, &b.as_async().range(3))
                .and_then(|df| df.collect()),
            |a| a
                .read()
                .schema("id INT")
                .option("header", true)
                .parse(ParseFormat::Csv, &a.range(3))
                .await
                .unwrap()
                .collect()
                .await
        ),
        case!(
            "a load, then an action",
            |b| b
                .as_async()
                .read()
                .format("csv")
                .load("/in")
                .blocking()
                .count(),
            |a| a.read().format("csv").load("/in").count().await
        ),
        case!(
            "save",
            |b| b
                .as_async()
                .range(3)
                .write()
                .mode(SaveMode::Overwrite)
                .format("parquet")
                .option("k", "v")
                .partition_by(["id"])
                .blocking()
                .save("/out"),
            |a| a
                .range(3)
                .write()
                .mode(SaveMode::Overwrite)
                .format("parquet")
                .option("k", "v")
                .partition_by(["id"])
                .save("/out")
                .await
        ),
        case!(
            "save_without_path",
            |b| b
                .as_async()
                .range(3)
                .write()
                .format("noop")
                .blocking()
                .save_without_path(),
            |a| a.range(3).write().format("noop").save_without_path().await
        ),
        case!(
            "save_as_table",
            |b| b
                .as_async()
                .range(3)
                .write()
                .mode(SaveMode::Append)
                .blocking()
                .save_as_table("db.t"),
            |a| a
                .range(3)
                .write()
                .mode(SaveMode::Append)
                .save_as_table("db.t")
                .await
        ),
        case!(
            "insert_into",
            |b| b.as_async().range(3).write().blocking().insert_into("db.t"),
            |a| a.range(3).write().insert_into("db.t").await
        ),
        case!(
            "csv",
            |b| b.as_async().range(3).write().blocking().csv("/out"),
            |a| a.range(3).write().csv("/out").await
        ),
        case!(
            "json",
            |b| b.as_async().range(3).write().blocking().json("/out"),
            |a| a.range(3).write().json("/out").await
        ),
        case!(
            "orc",
            |b| b.as_async().range(3).write().blocking().orc("/out"),
            |a| a.range(3).write().orc("/out").await
        ),
        case!(
            "parquet",
            |b| b.as_async().range(3).write().blocking().parquet("/out"),
            |a| a.range(3).write().parquet("/out").await
        ),
        case!(
            "text",
            |b| b.as_async().range(3).write().blocking().text("/out"),
            |a| a.range(3).write().text("/out").await
        ),
        case!(
            "xml",
            |b| b.as_async().range(3).write().blocking().xml("/out"),
            |a| a.range(3).write().xml("/out").await
        ),
        case!(
            "create",
            |b| b
                .as_async()
                .range(3)
                .write_to("db.t")
                .using("parquet")
                .option("k", "v")
                .blocking()
                .create(),
            |a| a
                .range(3)
                .write_to("db.t")
                .using("parquet")
                .option("k", "v")
                .create()
                .await
        ),
        case!(
            "replace",
            |b| b.as_async().range(3).write_to("db.t").blocking().replace(),
            |a| a.range(3).write_to("db.t").replace().await
        ),
        case!(
            "create_or_replace",
            |b| b
                .as_async()
                .range(3)
                .write_to("db.t")
                .blocking()
                .create_or_replace(),
            |a| a.range(3).write_to("db.t").create_or_replace().await
        ),
        case!(
            "append",
            |b| b.as_async().range(3).write_to("db.t").blocking().append(),
            |a| a.range(3).write_to("db.t").append().await
        ),
        case!(
            "overwrite",
            |b| b
                .as_async()
                .range(3)
                .write_to("db.t")
                .blocking()
                .overwrite(col("id").gt(lit(1))),
            |a| a
                .range(3)
                .write_to("db.t")
                .overwrite(col("id").gt(lit(1)))
                .await
        ),
        case!(
            "overwrite_partitions",
            |b| b
                .as_async()
                .range(3)
                .write_to("db.t")
                .blocking()
                .overwrite_partitions(),
            |a| a.range(3).write_to("db.t").overwrite_partitions().await
        ),
        case!(
            "MERGE",
            |b| b
                .as_async()
                .range(3)
                .merge_into("db.t", col("t.id").eq(col("s.id")))
                .when_matched(Some(col("s.d")))
                .delete()
                .when_matched(None)
                .update_all()
                .when_not_matched(None)
                .insert_all()
                .run(),
            |a| a
                .range(3)
                .merge_into("db.t", col("t.id").eq(col("s.id")))
                .when_matched(Some(col("s.d")))
                .delete()
                .when_matched(None)
                .update_all()
                .when_not_matched(None)
                .insert_all()
                .await
        ),
    ]
}

fn catalog_cases() -> Vec<Case> {
    vec![
        case!("current_catalog", |b| b.catalog().current_catalog(), |a| a
            .catalog()
            .current_catalog()
            .await),
        case!(
            "set_current_catalog",
            |b| b.catalog().set_current_catalog("lake"),
            |a| a.catalog().set_current_catalog("lake").await
        ),
        case!(
            "list_catalogs",
            |b| b.catalog().as_async().list_catalogs().pattern("s*").run(),
            |a| a.catalog().list_catalogs().pattern("s*").await
        ),
        case!(
            "current_database",
            |b| b.catalog().current_database(),
            |a| a.catalog().current_database().await
        ),
        case!(
            "set_current_database",
            |b| b.catalog().set_current_database("db"),
            |a| a.catalog().set_current_database("db").await
        ),
        case!(
            "list_databases",
            |b| b.catalog().as_async().list_databases().pattern("d*").run(),
            |a| a.catalog().list_databases().pattern("d*").await
        ),
        case!("get_database", |b| b.catalog().get_database("db"), |a| a
            .catalog()
            .get_database("db")
            .await),
        case!(
            "database_exists",
            |b| b.catalog().database_exists("db"),
            |a| a.catalog().database_exists("db").await
        ),
        case!(
            "a blocking catalog into_async",
            |b| b.catalog().into_async().list_tables().run(),
            |a| a.catalog().list_tables().await
        ),
        case!(
            "list_tables",
            |b| b
                .catalog()
                .as_async()
                .list_tables()
                .database("db")
                .pattern("t*")
                .run(),
            |a| a.catalog().list_tables().database("db").pattern("t*").await
        ),
        case!(
            "get_table",
            |b| b.catalog().as_async().get_table("t").database("db").run(),
            |a| a.catalog().get_table("t").database("db").await
        ),
        case!(
            "table_exists",
            |b| b
                .catalog()
                .as_async()
                .table_exists("t")
                .database("db")
                .run(),
            |a| a.catalog().table_exists("t").database("db").await
        ),
        case!(
            "list_columns",
            |b| b
                .catalog()
                .as_async()
                .list_columns("t")
                .database("db")
                .run(),
            |a| a.catalog().list_columns("t").database("db").await
        ),
        case!(
            "list_functions",
            |b| b
                .catalog()
                .as_async()
                .list_functions()
                .database("db")
                .pattern("f*")
                .run(),
            |a| a
                .catalog()
                .list_functions()
                .database("db")
                .pattern("f*")
                .await
        ),
        case!(
            "get_function",
            |b| b
                .catalog()
                .as_async()
                .get_function("f")
                .database("db")
                .run(),
            |a| a.catalog().get_function("f").database("db").await
        ),
        case!(
            "function_exists",
            |b| b
                .catalog()
                .as_async()
                .function_exists("f")
                .database("db")
                .run(),
            |a| a.catalog().function_exists("f").database("db").await
        ),
        case!(
            "create_table",
            |b| b
                .catalog()
                .as_async()
                .create_table("db.t")
                .path("/p")
                .source("parquet")
                .schema(ids())
                .description("d")
                .option("k", "v")
                .run()
                .and_then(|df| df.collect()),
            |a| a
                .catalog()
                .create_table("db.t")
                .path("/p")
                .source("parquet")
                .schema(ids())
                .description("d")
                .option("k", "v")
                .await
                .unwrap()
                .collect()
                .await
        ),
        case!("drop_temp_view", |b| b.catalog().drop_temp_view("v"), |a| a
            .catalog()
            .drop_temp_view("v")
            .await),
        case!(
            "drop_global_temp_view",
            |b| b.catalog().drop_global_temp_view("v"),
            |a| a.catalog().drop_global_temp_view("v").await
        ),
        case!(
            "cache_table",
            |b| b
                .catalog()
                .as_async()
                .cache_table("t")
                .storage_level(StorageLevel::DISK_ONLY)
                .run(),
            |a| a
                .catalog()
                .cache_table("t")
                .storage_level(StorageLevel::DISK_ONLY)
                .await
        ),
        case!("uncache_table", |b| b.catalog().uncache_table("t"), |a| a
            .catalog()
            .uncache_table("t")
            .await),
        case!("is_cached", |b| b.catalog().is_cached("t"), |a| a
            .catalog()
            .is_cached("t")
            .await),
        case!("clear_cache", |b| b.catalog().clear_cache(), |a| a
            .catalog()
            .clear_cache()
            .await),
        case!("refresh_table", |b| b.catalog().refresh_table("t"), |a| a
            .catalog()
            .refresh_table("t")
            .await),
        case!(
            "refresh_by_path",
            |b| b.catalog().refresh_by_path("/p"),
            |a| a.catalog().refresh_by_path("/p").await
        ),
        case!(
            "recover_partitions",
            |b| b.catalog().recover_partitions("t"),
            |a| a.catalog().recover_partitions("t").await
        ),
        case!(
            "create_database",
            |b| b
                .catalog()
                .as_async()
                .create_database("db")
                .if_not_exists()
                .property("k", "v")
                .run(),
            |a| a
                .catalog()
                .create_database("db")
                .if_not_exists()
                .property("k", "v")
                .await
        ),
        case!(
            "drop_database",
            |b| b
                .catalog()
                .as_async()
                .drop_database("db")
                .if_exists()
                .cascade()
                .run(),
            |a| a.catalog().drop_database("db").if_exists().cascade().await
        ),
        case!(
            "drop_table",
            |b| b
                .catalog()
                .as_async()
                .drop_table("t")
                .if_exists()
                .purge()
                .run(),
            |a| a.catalog().drop_table("t").if_exists().purge().await
        ),
        case!(
            "drop_view",
            |b| b.catalog().as_async().drop_view("v").if_exists().run(),
            |a| a.catalog().drop_view("v").if_exists().await
        ),
        case!("truncate_table", |b| b.catalog().truncate_table("t"), |a| a
            .catalog()
            .truncate_table("t")
            .await),
        case!(
            "analyze_table",
            |b| b.catalog().as_async().analyze_table("t").no_scan().run(),
            |a| a.catalog().analyze_table("t").no_scan().await
        ),
        case!(
            "get_create_table_string",
            |b| b
                .catalog()
                .as_async()
                .get_create_table_string("t")
                .as_serde()
                .run(),
            |a| a.catalog().get_create_table_string("t").as_serde().await
        ),
        case!(
            "get_table_properties",
            |b| b.catalog().get_table_properties("t"),
            |a| a.catalog().get_table_properties("t").await
        ),
        case!(
            "list_partitions",
            |b| b.catalog().list_partitions("t"),
            |a| a.catalog().list_partitions("t").await
        ),
        case!(
            "list_views",
            |b| b
                .catalog()
                .as_async()
                .list_views()
                .database("db")
                .pattern("v*")
                .run(),
            |a| a.catalog().list_views().database("db").pattern("v*").await
        ),
        case!(
            "the session of the catalog",
            |b| {
                let session = b.with_tag("etl").unwrap().catalog().session();
                (session.version(), session.as_async().sql("SET x = 1").run())
            },
            |a| {
                let catalog = a.with_tag("etl").unwrap().catalog();
                (
                    catalog.session().version().await,
                    catalog.session().sql("SET x = 1").await,
                )
            }
        ),
        case!(
            "an async catalog converted",
            |b| crate_session(b).catalog().blocking().current_database(),
            |a| a.catalog().current_database().await
        ),
    ]
}

/// A write of a stream of the rate source to the memory sink, with every setting.
fn stream_writer(s: &Session) -> streaming::DataStreamWriter {
    s.read_stream()
        .format("rate")
        .option("rowsPerSecond", 5)
        .load_without_path()
        .write_stream()
        .format("memory")
        .output_mode(OutputMode::Append)
        .option("k", "v")
        .partition_by(["value"])
        .query_name("q")
        .trigger(Trigger::ProcessingTime("1 second".to_string()))
}

fn started(s: &blocking::Session) -> blocking::StreamingQuery {
    stream_writer(s.as_async())
        .blocking()
        .start_without_path()
        .unwrap()
}

async fn started_async(s: &Session) -> streaming::StreamingQuery {
    stream_writer(s).start_without_path().await.unwrap()
}

fn streaming_cases() -> Vec<Case> {
    vec![
        case!(
            "streams active",
            |b| b.with_tag("etl").unwrap().streams().active(),
            |a| a.with_tag("etl").unwrap().streams().active().await
        ),
        case!("streams get", |b| b.streams().get("query-id"), |a| a
            .streams()
            .get("query-id")
            .await),
        case!(
            "streams await_any_termination",
            |b| b.streams().await_any_termination(None),
            |a| a.streams().await_any_termination(None).await
        ),
        case!(
            "streams await_any_termination with a timeout",
            |b| b
                .streams()
                .await_any_termination(Some(Duration::from_secs(2))),
            |a| a
                .streams()
                .await_any_termination(Some(Duration::from_secs(2)))
                .await
        ),
        case!(
            "streams reset_terminated",
            |b| b.streams().reset_terminated(),
            |a| a.streams().reset_terminated().await
        ),
        case!(
            "events, dropped",
            true,
            |b| b.streams().events().map(drop),
            |a| drop(a.streams().events().await)
        ),
        case!(
            "events, read to their end",
            |b| b.streams().events().map(|events| events.count()),
            |a| if let Ok(mut events) = a.streams().events().await {
                while events.next().await.is_some() {}
            }
        ),
        case!(
            "events, converted",
            true,
            |b| crate_session(b)
                .streams()
                .blocking()
                .events()
                .map(|events| drop(events.into_async().blocking())),
            |a| drop(a.streams().events().await)
        ),
        case!(
            "start_without_path",
            |b| stream_writer(b.as_async()).blocking().start_without_path(),
            |a| stream_writer(&a).start_without_path().await
        ),
        case!(
            "start",
            |b| stream_writer(b.as_async()).blocking().start("/out"),
            |a| stream_writer(&a).start("/out").await
        ),
        case!(
            "to_table",
            |b| stream_writer(b.as_async()).blocking().to_table("t"),
            |a| stream_writer(&a).to_table("t").await
        ),
        case!("query is_active", |b| started(b).is_active(), |a| {
            started_async(&a).await.is_active().await
        }),
        case!("query status", |b| started(b).status(), |a| started_async(
            &a
        )
        .await
        .status()
        .await),
        case!("query stop", |b| started(b).stop(), |a| started_async(&a)
            .await
            .stop()
            .await),
        case!(
            "query await_termination",
            |b| started(b).await_termination(None),
            |a| started_async(&a).await.await_termination(None).await
        ),
        case!(
            "query await_termination with a timeout",
            |b| started(b).await_termination(Some(Duration::from_millis(1500))),
            |a| started_async(&a)
                .await
                .await_termination(Some(Duration::from_millis(1500)))
                .await
        ),
        case!("query last_progress", |b| started(b).last_progress(), |a| {
            started_async(&a).await.last_progress().await
        }),
        case!(
            "query recent_progress",
            |b| started(b).recent_progress(),
            |a| started_async(&a).await.recent_progress().await
        ),
        case!(
            "query process_all_available",
            |b| started(b).process_all_available(),
            |a| started_async(&a).await.process_all_available().await
        ),
        case!("query exception", |b| started(b).exception(), |a| {
            started_async(&a).await.exception().await
        }),
        case!(
            "query explain",
            |b| started(b).as_async().explain().run(),
            |a| started_async(&a).await.explain().await
        ),
        case!(
            "query explain extended",
            |b| started(b).as_async().explain().extended().run(),
            |a| started_async(&a).await.explain().extended().await
        ),
        case!(
            "the queries of active",
            |b| b
                .streams()
                .active()
                .map(|queries| queries.iter().map(|q| q.stop()).collect::<Vec<_>>()),
            |a| for query in a.streams().active().await.unwrap() {
                query.stop().await.unwrap();
            }
        ),
        case!(
            "the query of get",
            |b| b.streams().get("query-id").map(|q| q.unwrap().status()),
            |a| a
                .streams()
                .get("query-id")
                .await
                .unwrap()
                .unwrap()
                .status()
                .await
        ),
        case!(
            "a query converted both ways",
            |b| started(b).into_async().blocking().recent_progress(),
            |a| started_async(&a).await.recent_progress().await
        ),
    ]
}

/// Ten rows, a dataset of Spark ML for the recording server, which reads none of them.
fn rows_of(s: &Session) -> DataFrame {
    s.range(10)
}

/// The async model of `estimator`, fitted to `rows_of`.
async fn fitted<E: ml::Estimator>(estimator: E, s: &Session) -> E::Model {
    ml::Estimator::fit(&estimator, &rows_of(s)).await.unwrap()
}

/// A case of the attribute `$attribute` of the model of `$estimator`.
macro_rules! attribute {
    ($estimator:expr, $attribute:ident) => {
        case!(
            concat!(stringify!($estimator), " ", stringify!($attribute)),
            |b| $estimator
                .fit(&rows_of(b.as_async()))
                .and_then(|model| model.$attribute()),
            |a| fitted($estimator, &a).await.$attribute().await
        )
    };
}

fn ml_cases() -> Vec<Case> {
    vec![
        case!(
            "fit LinearRegression",
            |b| LinearRegression::new()
                .max_iter(2)
                .fit(&rows_of(b.as_async())),
            |a| fitted(LinearRegression::new().max_iter(2), &a).await
        ),
        case!(
            "fit LinearSVC",
            |b| LinearSVC::new().reg_param(0.1).fit(&rows_of(b.as_async())),
            |a| fitted(LinearSVC::new().reg_param(0.1), &a).await
        ),
        case!(
            "fit LogisticRegression",
            |b| LogisticRegression::new()
                .max_iter(3)
                .fit(&rows_of(b.as_async())),
            |a| fitted(LogisticRegression::new().max_iter(3), &a).await
        ),
        case!(
            "fit MaxAbsScaler",
            |b| MaxAbsScaler::new()
                .input_col("v")
                .fit(&rows_of(b.as_async())),
            |a| fitted(MaxAbsScaler::new().input_col("v"), &a).await
        ),
        case!(
            "fit StandardScaler",
            |b| StandardScaler::new()
                .input_col("v")
                .fit(&rows_of(b.as_async())),
            |a| fitted(StandardScaler::new().input_col("v"), &a).await
        ),
        case!(
            "fit StringIndexer",
            |b| StringIndexer::new()
                .input_col("s")
                .fit(&rows_of(b.as_async())),
            |a| fitted(StringIndexer::new().input_col("s"), &a).await
        ),
        case!(
            "fit a Pipeline",
            |b| Pipeline::new()
                .stage(VectorAssembler::new().input_cols(["id"]).output_col("v"))
                .stage(StandardScaler::new().input_col("v"))
                .fit(&rows_of(b.as_async())),
            |a| fitted(
                Pipeline::new()
                    .stage(VectorAssembler::new().input_cols(["id"]).output_col("v"))
                    .stage(StandardScaler::new().input_col("v")),
                &a
            )
            .await
        ),
        case!(
            "fit a CrossValidator",
            |b| CrossValidator::new([LinearRegression::new()], RegressionEvaluator::new())
                .num_folds(2)
                .seed(7)
                .fit(&rows_of(b.as_async())),
            |a| fitted(
                CrossValidator::new([LinearRegression::new()], RegressionEvaluator::new())
                    .num_folds(2)
                    .seed(7),
                &a
            )
            .await
        ),
        case!(
            "evaluate RegressionEvaluator",
            |b| RegressionEvaluator::new()
                .metric_name("mae")
                .evaluate(&rows_of(b.as_async())),
            |a| ml::Evaluator::evaluate(
                &RegressionEvaluator::new().metric_name("mae"),
                &rows_of(&a)
            )
            .await
        ),
        case!(
            "evaluate BinaryClassificationEvaluator",
            |b| BinaryClassificationEvaluator::new()
                .metric_name("areaUnderPR")
                .evaluate(&rows_of(b.as_async())),
            |a| ml::Evaluator::evaluate(
                &BinaryClassificationEvaluator::new().metric_name("areaUnderPR"),
                &rows_of(&a)
            )
            .await
        ),
        case!(
            "evaluate MulticlassClassificationEvaluator",
            |b| MulticlassClassificationEvaluator::new()
                .metric_name("f1")
                .evaluate(&rows_of(b.as_async())),
            |a| ml::Evaluator::evaluate(
                &MulticlassClassificationEvaluator::new().metric_name("f1"),
                &rows_of(&a)
            )
            .await
        ),
        attribute!(LogisticRegression::new(), coefficients),
        attribute!(LogisticRegression::new(), intercept),
        attribute!(LogisticRegression::new(), coefficient_matrix),
        attribute!(LogisticRegression::new(), intercept_vector),
        attribute!(LogisticRegression::new(), num_classes),
        attribute!(LogisticRegression::new(), num_features),
        attribute!(LinearSVC::new(), coefficients),
        attribute!(LinearSVC::new(), intercept),
        attribute!(LinearSVC::new(), num_classes),
        attribute!(LinearSVC::new(), num_features),
        attribute!(LinearRegression::new(), coefficients),
        attribute!(LinearRegression::new(), intercept),
        attribute!(LinearRegression::new(), scale),
        attribute!(LinearRegression::new(), num_features),
        attribute!(StandardScaler::new(), mean),
        attribute!(StandardScaler::new(), std),
        attribute!(MaxAbsScaler::new(), max_abs),
        attribute!(StringIndexer::new(), labels),
        case!(
            "a model converted both ways",
            |b| LogisticRegression::new()
                .fit(&rows_of(b.as_async()))
                .map(|model| model.into_async().blocking().intercept()),
            |a| fitted(LogisticRegression::new(), &a)
                .await
                .intercept()
                .await
        ),
        case!(
            "a transform of a blocking model",
            |b| LinearRegression::new()
                .fit(&rows_of(b.as_async()))
                .map(|model| {
                    ml::Transformer::transform(model.as_async(), &rows_of(b.as_async()))
                        .blocking()
                        .count()
                }),
            |a| ml::Transformer::transform(
                &fitted(LinearRegression::new(), &a).await,
                &rows_of(&a)
            )
            .count()
            .await
        ),
        case!(
            "a model of a fitted pipeline",
            |b| Pipeline::new()
                .stage(LinearRegression::new())
                .fit(&rows_of(b.as_async()))
                .map(|pipeline| {
                    pipeline
                        .stage::<ml::LinearRegressionModel>(0)
                        .unwrap()
                        .blocking()
                        .intercept()
                }),
            |a| fitted(Pipeline::new().stage(LinearRegression::new()), &a)
                .await
                .stage::<ml::LinearRegressionModel>(0)
                .unwrap()
                .intercept()
                .await
        ),
    ]
}

fn graph(s: &blocking::Session) -> blocking::DataflowGraph {
    s.as_async().create_dataflow_graph().run().unwrap()
}

async fn graph_async(s: &Session) -> pipelines::DataflowGraph {
    s.create_dataflow_graph().await.unwrap()
}

fn pipeline_cases() -> Vec<Case> {
    vec![
        case!(
            "create_dataflow_graph",
            |b| b
                .as_async()
                .create_dataflow_graph()
                .default_catalog("c")
                .default_database("d")
                .sql_conf("k", "v")
                .run(),
            |a| a
                .create_dataflow_graph()
                .default_catalog("c")
                .default_database("d")
                .sql_conf("k", "v")
                .await
        ),
        case!(
            "define_materialized_view",
            |b| graph(b)
                .as_async()
                .define_materialized_view("mv")
                .comment("c")
                .partition_by(["id"])
                .table_property("k", "v")
                .run(),
            |a| graph_async(&a)
                .await
                .define_materialized_view("mv")
                .comment("c")
                .partition_by(["id"])
                .table_property("k", "v")
                .await
        ),
        case!(
            "define_streaming_table",
            |b| graph(b)
                .as_async()
                .define_streaming_table("st")
                .format("parquet")
                .schema("id BIGINT")
                .run(),
            |a| graph_async(&a)
                .await
                .define_streaming_table("st")
                .format("parquet")
                .schema("id BIGINT")
                .await
        ),
        case!(
            "define_temporary_view",
            |b| graph(b)
                .as_async()
                .define_temporary_view("tv")
                .comment("c")
                .run(),
            |a| graph_async(&a)
                .await
                .define_temporary_view("tv")
                .comment("c")
                .await
        ),
        case!(
            "define_sink",
            |b| graph(b)
                .as_async()
                .define_sink("s", "console")
                .option("k", "v")
                .run(),
            |a| graph_async(&a)
                .await
                .define_sink("s", "console")
                .option("k", "v")
                .await
        ),
        case!(
            "define_flow",
            |b| graph(b)
                .as_async()
                .define_flow("f", "mv", b.as_async().range(3))
                .sql_conf("k", "v")
                .run(),
            |a| graph_async(&a)
                .await
                .define_flow("f", "mv", a.range(3))
                .sql_conf("k", "v")
                .await
        ),
        case!(
            "define_auto_cdc_flow",
            |b| graph(b)
                .as_async()
                .define_auto_cdc_flow("f", "t", "src", ["id"], col("ts"))
                .stored_as_scd_type(ScdType::Type2)
                .run(),
            |a| graph_async(&a)
                .await
                .define_auto_cdc_flow("f", "t", "src", ["id"], col("ts"))
                .stored_as_scd_type(ScdType::Type2)
                .await
        ),
        case!(
            "define_sql",
            |b| graph(b)
                .as_async()
                .define_sql("CREATE MATERIALIZED VIEW mv AS SELECT 1")
                .file_path("mv.sql")
                .run(),
            |a| graph_async(&a)
                .await
                .define_sql("CREATE MATERIALIZED VIEW mv AS SELECT 1")
                .file_path("mv.sql")
                .await
        ),
        case!(
            "start_run",
            |b| graph(b)
                .as_async()
                .start_run("file:///storage")
                .full_refresh(["a"])
                .refresh(["b"])
                .run()
                .map(|run| run.count()),
            |a| {
                let start = graph_async(&a)
                    .await
                    .start_run("file:///storage")
                    .full_refresh(["a"])
                    .refresh(["b"]);
                let mut run = start.await.unwrap();
                while run.next().await.is_some() {}
            }
        ),
        case!(
            "start_run, a dry run of every dataset",
            |b| graph(b)
                .as_async()
                .start_run("file:///storage")
                .full_refresh_all()
                .dry_run()
                .run()
                .map(|run| run.into_async().blocking().count()),
            |a| {
                let start = graph_async(&a)
                    .await
                    .start_run("file:///storage")
                    .full_refresh_all()
                    .dry_run();
                let mut run = start.await.unwrap();
                while run.next().await.is_some() {}
            }
        ),
        case!(
            "the session of a graph",
            |b| {
                let dataflow = graph(&b.with_tag("etl").unwrap());
                let session = dataflow.session();
                (session.version(), session.as_async().sql("SET x = 1").run())
            },
            |a| {
                let dataflow = graph_async(&a.with_tag("etl").unwrap()).await;
                (
                    dataflow.session().version().await,
                    dataflow.session().sql("SET x = 1").await,
                )
            }
        ),
        case!(
            "a graph converted both ways",
            |b| graph(b)
                .into_async()
                .blocking()
                .as_async()
                .define_sql("SELECT 1")
                .run(),
            |a| graph_async(&a).await.define_sql("SELECT 1").await
        ),
    ]
}

/// `text` with each UUID replaced, as the ids of the operators of Spark ML are random, but
/// `SESSION_ID`, so that a request of another session differs.
fn without_uuids(text: &str) -> String {
    let bytes = text.as_bytes();
    let is_uuid = |at: usize| {
        bytes.len() >= at + 36
            && bytes[at..at + 36].iter().enumerate().all(|(i, &byte)| {
                if matches!(i, 8 | 13 | 18 | 23) {
                    byte == b'-'
                } else {
                    byte.is_ascii_hexdigit()
                }
            })
    };
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while at < text.len() {
        if text[at..].starts_with(SESSION_ID) {
            out.push_str(SESSION_ID);
            at += SESSION_ID.len();
        } else if is_uuid(at) {
            out.push_str("<uuid>");
            at += 36;
        } else {
            let next = text[at..].chars().next().unwrap();
            out.push(next);
            at += next.len_utf8();
        }
    }
    out
}

/// Each entry as its message, with the operation ids, which the client chooses at random,
/// removed, and the UUIDs other than `SESSION_ID` replaced.
fn messages(entries: Vec<server::Entry>) -> Vec<String> {
    entries
        .into_iter()
        .map(|entry| match entry.kind.as_str() {
            "ExecutePlanRequest" => {
                let mut request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
                request.operation_id = None;
                without_uuids(&format!("{request:?}"))
            }
            "AnalyzePlanRequest" => {
                let request = proto::AnalyzePlanRequest::decode(&entry.bytes[..]).unwrap();
                format!("{request:?}")
            }
            _ => format!("{entry:?}"),
        })
        .collect()
}

fn command_of(entry: &server::Entry) -> Option<proto::command::CommandType> {
    if entry.kind != "ExecutePlanRequest" {
        return None;
    }
    match proto::ExecutePlanRequest::decode(&entry.bytes[..])
        .ok()?
        .plan?
        .op_type?
    {
        proto::plan::OpType::Command(command) => command.command_type,
        _ => None,
    }
}

/// The entries of a case, once the server has had what the handles that it dropped send in
/// the background: the deletes of its models, the drops of its graphs, the removals of its
/// checkpoints and, for a case that `subscribes`, the removal of its subscription. Those come
/// last, in the order they came: the other requests may come before or after them.
fn settled(recorder: &server::Recorder, subscribes: bool) -> Vec<server::Entry> {
    use proto::command::CommandType as C;
    use proto::pipeline_command::CommandType as P;
    use proto::streaming_query_listener_bus_command::Command as Bus;
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut entries = Vec::new();
    loop {
        entries.extend(recorder.take());
        let count = |kind: fn(&C) -> bool| {
            entries
                .iter()
                .filter(|entry| command_of(entry).as_ref().is_some_and(kind))
                .count()
        };
        let fits = count(|c| {
            matches!(
                c,
                C::MlCommand(proto::MlCommand {
                    command: Some(proto::ml_command::Command::Fit(_))
                })
            )
        });
        let deletes = count(|c| {
            matches!(
                c,
                C::MlCommand(proto::MlCommand {
                    command: Some(proto::ml_command::Command::Delete(_))
                })
            )
        });
        let graphs = count(|c| {
            matches!(
                c,
                C::PipelineCommand(proto::PipelineCommand {
                    command_type: Some(P::CreateDataflowGraph(_))
                })
            )
        });
        let dropped = count(|c| {
            matches!(
                c,
                C::PipelineCommand(proto::PipelineCommand {
                    command_type: Some(P::DropDataflowGraph(_))
                })
            )
        });
        let checkpoints = count(|c| matches!(c, C::CheckpointCommand(_)));
        let removed = count(|c| matches!(c, C::RemoveCachedRemoteRelationCommand(_)));
        let unsubscribed = count(|c| {
            matches!(
                c,
                C::StreamingQueryListenerBusCommand(proto::StreamingQueryListenerBusCommand {
                    command: Some(Bus::RemoveListenerBusListener(_))
                })
            )
        });
        let done = fits <= deletes
            && graphs <= dropped
            && checkpoints <= removed
            && (!subscribes || unsubscribed > 0);
        if done || Instant::now() > deadline {
            entries.sort_by_key(|entry| {
                command_of(entry).is_some_and(|c| {
                    matches!(
                        c,
                        C::MlCommand(proto::MlCommand {
                            command: Some(proto::ml_command::Command::Delete(_))
                        }) | C::PipelineCommand(proto::PipelineCommand {
                            command_type: Some(P::DropDataflowGraph(_))
                        }) | C::RemoveCachedRemoteRelationCommand(_)
                            | C::StreamingQueryListenerBusCommand(
                                proto::StreamingQueryListenerBusCommand {
                                    command: Some(Bus::RemoveListenerBusListener(_))
                                }
                            )
                    )
                })
            });
            return entries;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn start() -> (Arc<server::Recorder>, String) {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let url = format!("sc://{address}/;session_id={SESSION_ID};user_id=snapshots");
    (recorder, url)
}

#[test]
fn the_blocking_api_sends_the_requests_of_the_async_api() {
    let (recorder, url) = start();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let cases = [
        session_cases(),
        dataframe_cases(),
        conversion_cases(),
        read_write_cases(),
        catalog_cases(),
        streaming_cases(),
        ml_cases(),
        pipeline_cases(),
    ];
    let mut both = Vec::new();
    for case in cases.into_iter().flatten() {
        recorder.hold_listener_bus(case.subscribes);
        let session = Session::builder()
            .remote(url.clone())
            .blocking()
            .connect()
            .unwrap();
        (case.blocking)(&session);
        let blocking = messages(settled(&recorder, case.subscribes));
        runtime.block_on(async {
            let session = Session::builder()
                .remote(url.clone())
                .connect()
                .await
                .unwrap();
            (case.awaited)(session).await;
        });
        let awaited = messages(settled(&recorder, case.subscribes));
        recorder.hold_listener_bus(false);
        assert!(
            !awaited.is_empty(),
            "{}: the async API sent nothing",
            case.name
        );
        both.push((case.name, awaited, blocking));
    }
    let differences = support::differences(both);
    assert!(differences.is_empty(), "{differences}");
}

#[test]
fn the_iterators_yield_what_the_streams_yield_then_end() {
    let (_, url) = start();
    let session = Session::builder()
        .remote(url.clone())
        .blocking()
        .connect()
        .unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let streamed = runtime.block_on(async {
        let session = Session::builder().remote(url).connect().await.unwrap();
        let three = session.sql(server::THREE_BATCHES).await.unwrap();
        let batches = three.stream().await.unwrap();
        let rows: Vec<_> = batches
            .map(|batch| outcome(batch.map(|b| b.num_rows())))
            .collect()
            .await;
        let failing = session.sql(server::A_BATCH_THEN_A_FAILURE).await.unwrap();
        let failed: Vec<_> = failing
            .stream_as::<i64>()
            .await
            .unwrap()
            .map(outcome)
            .collect()
            .await;
        (rows, failed)
    });

    let mut batches = three(&session).stream().unwrap();
    let rows: Vec<_> = batches
        .by_ref()
        .map(|batch| outcome(batch.map(|b| b.num_rows())))
        .collect();
    assert!(
        batches.next().is_none(),
        "an ended iterator yields nothing more"
    );
    assert!(
        batches.as_async().schema().is_some(),
        "the schema of the stream"
    );
    let failing = session
        .as_async()
        .sql(server::A_BATCH_THEN_A_FAILURE)
        .run()
        .unwrap();
    let failed: Vec<_> = failing.stream_as::<i64>().unwrap().map(outcome).collect();
    assert_eq!((&rows, &failed), (&streamed.0, &streamed.1));
    assert_eq!(rows, ["1", "1", "1"]);

    let mut batches = three(&session).stream().unwrap();
    let first = outcome(batches.next().unwrap().map(|b| b.num_rows()));
    let rest: Vec<_> = runtime.block_on(
        batches
            .into_async()
            .map(|batch| outcome(batch.map(|b| b.num_rows())))
            .collect(),
    );
    assert_eq!(first, "1");
    assert_eq!(
        rest,
        ["1", "1"],
        "the batches into_async leaves to the stream"
    );
    assert_eq!(
        failed,
        [
            "0",
            "error: [Client specified an invalid argument] the execution failed"
        ]
    );
}

/// An item of a stream as its value, or as the message of its error.
fn outcome<T: std::fmt::Debug>(item: spark_connect_async::Result<T>) -> String {
    match item {
        Ok(value) => format!("{value:?}"),
        Err(error) => format!("error: {}", error.message()),
    }
}

#[test]
fn an_async_and_a_blocking_handle_of_one_session_share_it_and_its_tags() {
    let (recorder, url) = start();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let tagged = runtime.block_on(async {
        let session = Session::builder()
            .remote(url.clone())
            .connect()
            .await
            .unwrap();
        session.with_tag("etl").unwrap()
    });
    tagged
        .blocking()
        .as_async()
        .range(3)
        .blocking()
        .collect()
        .unwrap();
    let sent = recorder.take();
    let request = proto::ExecutePlanRequest::decode(&sent[0].bytes[..]).unwrap();
    assert_eq!(request.tags, ["etl"]);
    assert_eq!(request.session_id, tagged.session_id());

    let blocking = Session::builder().remote(url).blocking().connect().unwrap();
    let tagged = blocking.with_tag("daily").unwrap();
    let converted = Session::from(tagged.clone());
    let into_async = tagged.clone().into_async();
    for (form, awaited) in [
        ("From", &converted),
        ("into_async", &into_async),
        ("as_async", tagged.as_async()),
    ] {
        runtime.block_on(awaited.range(3).collect()).unwrap();
        let sent = recorder.take();
        let request = proto::ExecutePlanRequest::decode(&sent[0].bytes[..]).unwrap();
        assert_eq!(request.tags, ["daily"], "{form}");
        assert_eq!(request.session_id, converted.session_id(), "{form}");
    }
    assert!(
        blocking.as_async().tags().is_empty(),
        "the untagged handle keeps no tag"
    );
}
