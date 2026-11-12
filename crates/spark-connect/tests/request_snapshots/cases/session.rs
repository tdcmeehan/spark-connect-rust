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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::record_batch::RecordBatch;
use spark_connect::column::{col, lit, lit_string};
use spark_connect::datasource::{
    CommonInlineUserDefinedDataSourceExpression, PythonDataSourcePayload,
};
use spark_connect::resource::{
    ExecutorResourceRequests, ResourceProfileBuilder, TaskResourceRequests,
};
use spark_connect::row::{Row, Value};
use spark_connect::udf::eval_type;
use spark_connect::DataType;

use super::{batch_udf, range, schema};
use crate::Case;

fn extension(name: &str) -> prost_types::Any {
    prost_types::Any {
        type_url: format!("type.example/{name}"),
        value: name.as_bytes().to_vec(),
    }
}

fn arrow_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("name", ArrowType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef,
        ],
    )
    .unwrap()
}

fn rows() -> Vec<Row> {
    let names = || vec!["id".to_string(), "name".to_string()];
    vec![
        Row::new(names(), vec![Value::Long(1), Value::String("a".into())]),
        Row::new(names(), vec![Value::Long(2), Value::Null]),
    ]
}

/// A file with fixed content, named `name`, in a directory of this test.
fn artifact(name: &str) -> String {
    let dir: PathBuf = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("request-snapshots");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, format!("content of {name}")).unwrap();
    path.to_str().unwrap().to_string()
}

pub fn cases() -> Vec<Case> {
    cases![
        "SparkSession::add_tag" => |c| {
            c.spark.add_tag("one")?;
            c.spark.add_tag("two")?;
            c.df(&range(c))
        },
        "SparkSession::remove_tag" => |c| {
            c.spark.add_tag("one")?;
            c.spark.add_tag("two")?;
            c.spark.remove_tag("one");
            c.df(&range(c))
        },
        "SparkSession::clear_tags" => |c| {
            c.spark.add_tag("one")?;
            c.spark.clear_tags();
            c.df(&range(c))
        },
        "SparkSession::new_session" => |c| {
            let session = c.spark.new_session();
            c.mask(session.session_id());
            c.df(&session.range(3)?)
        },
        "SparkSession::clone_session" => |c| {
            let session = c.spark.clone_session()?;
            c.df(&session.range(3)?)
        },
        "SparkSession::clone_session_with_id" => |c| {
            let session = c
                .spark
                .clone_session_with_id(Some("00000000-0000-4000-8000-0000000000c1"))?;
            c.df(&session.range(3)?)
        },
        "SparkSession::range" => |c| c.df(&c.spark.range(10)?),
        "SparkSession::range_full" => |c| c.df(&c.spark.range_full(2, 20, 3, None)?),
        "SparkSession::range_full/partitions" => |c| c.df(&c.spark.range_full(0, 5, 1, Some(4))?),
        "SparkSession::parse_ddl" => |c| {
            let parsed = c.spark.parse_ddl("a INT, b STRING")?;
            c.data_type(&parsed);
            Ok(())
        },
        "SparkSession::sql" => |c| c.df(&c.spark.sql("SELECT 1 AS a")?),
        "SparkSession::sql_with_args" => |c| {
            let named = HashMap::from([("x".to_string(), lit(5).expression().clone())]);
            let df = c.spark.sql_with_args(
                "SELECT ? AS a, :x AS b",
                vec![lit_string("p").expression().clone()],
                named,
            )?;
            c.df(&df)
        },
        "SparkSession::sql_with_references" => |c| {
            let df = c.spark.sql_with_references(
                "SELECT * FROM t WHERE id > ?",
                vec![lit(1).expression().clone()],
                HashMap::new(),
                vec![(range(c), "t".to_string())],
            )?;
            c.df(&df)
        },
        "SparkSession::sql_without_running" => |c| {
            let df = c.spark.sql_without_running(
                "SELECT * FROM t",
                Vec::new(),
                HashMap::new(),
                vec![(range(c), "t".to_string())],
            );
            c.df(&df)
        },
        "SparkSession::create_dataframe" => |c| c.df(&c.spark.create_dataframe(rows(), schema())?),
        "SparkSession::create_dataframe/empty" => {
            |c| c.df(&c.spark.create_dataframe(Vec::new(), schema())?)
        },
        "SparkSession::create_dataframe_from_columns" => |c| {
            let batch = arrow_batch();
            c.df(&c.spark.create_dataframe_from_columns(batch.columns().to_vec(), schema())?)
        },
        "SparkSession::create_dataframe_from_inline_arrow_ipc" => |c| {
            let batch = arrow_batch();
            let mut data = Vec::new();
            let mut writer =
                arrow::ipc::writer::StreamWriter::try_new(&mut data, &batch.schema()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
            drop(writer);
            c.df(&c.spark.create_dataframe_from_inline_arrow_ipc(data, schema())?)
        },
        "SparkSession::create_dataframe_from_arrow" => {
            |c| c.df(&c.spark.create_dataframe_from_arrow(vec![arrow_batch()], None)?)
        },
        "SparkSession::create_dataframe_from_arrow/schema" => {
            |c| c.df(&c.spark.create_dataframe_from_arrow(vec![arrow_batch()], schema())?)
        },
        "SparkSession::create_dataframe_from_arrow/cached" => |c| {
            let ids: Vec<i64> = (0..2000).collect();
            let batch = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int64, true)])),
                vec![Arc::new(Int64Array::from(ids)) as ArrayRef],
            )
            .unwrap();
            c.df(&c.spark.create_dataframe_from_arrow(vec![batch], None)?)
        },
        "SparkSession::create_dataframe/cached" => |c| {
            let names = || vec!["id".to_string(), "name".to_string()];
            let rows = (0..300)
                .map(|i| Row::new(names(), vec![Value::Long(i), Value::String(format!("name {i}"))]))
                .collect();
            c.df(&c.spark.create_dataframe(rows, schema())?)
        },
        "SparkSession::arrow_input_options" => |c| c.spark.arrow_input_options().map(drop),
        "SparkSession::create_dataframe_from_arrow_with_options" => |c| {
            let options = c.spark.arrow_input_options()?;
            let df =
                c.spark
                    .create_dataframe_from_arrow_with_options(vec![arrow_batch()], schema(), &options)?;
            c.df(&df)
        },
        "SparkSession::register_java_function" => {
            |c| c.spark.register_java_function("f", "com.example.F", Some("INT"), false)
        },
        "SparkSession::register_java_function/aggregate" => {
            |c| c.spark.register_java_function("g", "com.example.G", None, true)
        },
        "SparkSession::version" => |c| c.spark.version().map(drop),
        "SparkSession::table" => |c| c.df(&c.spark.table("db.t")?),
        "SparkSession::empty_data_frame" => |c| c.df(&c.spark.empty_data_frame()?),
        "SparkSession::interrupt_all" => |c| c.spark.interrupt_all().map(drop),
        "SparkSession::interrupt_tag" => |c| c.spark.interrupt_tag("tag").map(drop),
        "SparkSession::interrupt_operation" => {
            |c| c.spark.interrupt_operation("00000000-0000-0000-0000-0000000000a1").map(drop)
        },
        "SparkSession::add_artifacts" => {
            |c| c.spark.add_artifacts(&[&artifact("lib.jar"), &artifact("other.jar")])
        },
        "SparkSession::add_artifacts_with/pyfile" => {
            |c| c.spark.add_artifacts_with(&[&artifact("module.py")], true, false, false)
        },
        "SparkSession::add_artifacts_with/archive" => {
            |c| c.spark.add_artifacts_with(&[&artifact("archive.zip")], false, true, false)
        },
        "SparkSession::add_artifacts_with/file" => {
            |c| c.spark.add_artifacts_with(&[&artifact("file.txt")], false, false, true)
        },
        "SparkSession::add_artifact" => |c| c.spark.add_artifact(&artifact("single.jar")),
        "SparkSession::copy_from_local_to_fs" => {
            |c| c.spark.copy_from_local_to_fs(&artifact("copy.txt"), "/tmp/dest.txt")
        },
        "SparkSession::register_function" => |c| c.spark.register_function(batch_udf()),
        "SparkSession::register_table_function" => |c| {
            c.spark.register_table_function(
                "t",
                Some(&schema()),
                eval_type::SQL_TABLE_UDF,
                &[4, 5],
                "3.11",
                true,
            )
        },
        "SparkSession::register_table_function/no_return_type" => |c| {
            c.spark
                .register_table_function("t", None, eval_type::SQL_ARROW_TABLE_UDF, &[4], "3.11", false)
        },
        "SparkSession::build_resource_profile" => |c| {
            let profile = ResourceProfileBuilder::new()
                .executor_resources(
                    ExecutorResourceRequests::new()
                        .memory(1024)
                        .off_heap_memory(512)
                        .cores(2)
                        .resource("gpu", 1, Some("discover.sh".to_string()), Some("nvidia".to_string())),
                )
                .task_resources(TaskResourceRequests::new().cpus(1.0).resource("gpu", 0.5))
                .build();
            c.spark.build_resource_profile(&profile).map(drop)
        },
        "SparkSession::register_data_source" => |c| {
            c.spark.register_data_source(CommonInlineUserDefinedDataSourceExpression::new(
                "source".to_string(),
                PythonDataSourcePayload::new(vec![7, 8], "3.11".to_string()),
            ))
        },
        "SparkSession::stop" => |c| c.spark.stop(),
        "SparkSession::progress_handlers" => |c| {
            let id = c.spark.register_progress_handler(|_| {});
            c.spark.remove_progress_handler(id);
            c.spark.register_progress_handler(|_| {});
            c.spark.clear_progress_handlers();
            c.df(&range(c).filter(col("id").gt(lit(1))))
        },
        "SparkSession::datatype_argument" => |c| {
            c.df(&c.spark.create_dataframe(Vec::new(), DataType::from_ddl("a ARRAY<INT>")?)?)
        },
        "RuntimeConf::set" => |c| c.spark.conf().set("k", "v").map(drop),
        "RuntimeConf::set_all" => |c| {
            c.spark.conf().set_all(&[("a".to_string(), "1".to_string()), ("b".to_string(), "2".to_string())], false).map(drop)
        },
        "RuntimeConf::set_all/silent" => {
            |c| c.spark.conf().set_all(&[("a".to_string(), "1".to_string())], true).map(drop)
        },
        "RuntimeConf::get" => |c| c.spark.conf().get("k").map(drop),
        "RuntimeConf::get_many" => |c| c.spark.conf().get_many(&["a", "b"]).map(drop),
        "RuntimeConf::get_with_default" => {
            |c| c.spark.conf().get_with_default("k", Some("d")).map(drop)
        },
        "RuntimeConf::get_with_default/none" => {
            |c| c.spark.conf().get_with_default("k", None).map(drop)
        },
        "RuntimeConf::get_all" => |c| c.spark.conf().get_all().map(drop),
        "RuntimeConf::unset" => |c| c.spark.conf().unset("k").map(drop),
        "RuntimeConf::is_modifiable" => |c| c.spark.conf().is_modifiable("k").map(drop),
        "RuntimeConf::set_pairs" => {
            |c| c.spark.conf().set_pairs(&[("a", Some("1")), ("b", None)], Some(true)).map(drop)
        },
        "RuntimeConf::set_pairs/default" => |c| c.spark.conf().set_pairs(&[("a", Some("1"))], None).map(drop),
        "RuntimeConf::get_with_defaults" => {
            |c| c.spark.conf().get_with_defaults(&[("a", Some("x")), ("b", None)]).map(drop)
        },
        "SparkSession::add_global_user_context_extension" => |c| {
            let id = c.spark.add_global_user_context_extension(extension("global"));
            c.mask(&id);
            c.df(&range(c))
        },
        "SparkSession::add_threadlocal_user_context_extension" => |c| {
            let id = c.spark.add_threadlocal_user_context_extension(extension("local"));
            c.mask(&id);
            c.df(&range(c))
        },
        "SparkSession::remove_user_context_extension" => |c| {
            let id = c.spark.add_global_user_context_extension(extension("global"));
            c.mask(&id);
            c.spark.remove_user_context_extension(&id);
            c.df(&range(c))
        },
        "SparkSession::clear_user_context_extensions" => |c| {
            c.mask(&c.spark.add_global_user_context_extension(extension("global")));
            c.mask(&c.spark.add_threadlocal_user_context_extension(extension("local")));
            c.spark.clear_user_context_extensions();
            c.df(&range(c))
        },
        "SparkSession::global_user_context_extensions" => |c| {
            c.mask(&c.spark.add_global_user_context_extension(extension("global")));
            for (_, extension) in c.spark.global_user_context_extensions() {
                c.message("Any", &extension);
            }
            Ok(())
        },
        "SparkSession::threadlocal_user_context_extensions" => |c| {
            c.mask(&c.spark.add_threadlocal_user_context_extension(extension("local")));
            for (_, extension) in c.spark.threadlocal_user_context_extensions() {
                c.message("Any", &extension);
            }
            Ok(())
        },
        "SparkSession::share_threadlocal_user_context_extensions" => |c| {
            c.mask(&c.spark.add_threadlocal_user_context_extension(extension("local")));
            let copied = c.spark.copy_threadlocal_user_context_extensions();
            c.spark.clear_user_context_extensions();
            c.spark.share_threadlocal_user_context_extensions(&copied);
            c.df(&range(c))
        },
        "SparkSession::set_server_session_id" => |c| {
            c.spark.set_server_session_id(Some("00000000-0000-4000-8000-0000000000aa".to_string()));
            c.df(&range(c))
        },
        "SparkSession::check_tag" => |c| {
            for tag in ["one", "", "a,b"] {
                let checked = spark_connect::session::SparkSession::check_tag(tag);
                c.message("Text", &format!("{tag:?}: {:?}", checked.map_err(|e| e.to_string())));
            }
            Ok(())
        },
        "SparkSession::copy_thread_tags" => |c| {
            c.spark.add_tag("one")?;
            let tags = c.spark.copy_thread_tags();
            c.spark.clear_tags();
            c.spark.share_thread_tags(&tags);
            c.df(&range(c))
        },
        "SparkSession::share_thread_tags" => |c| {
            let tags = c.spark.copy_thread_tags();
            c.spark.share_thread_tags(&tags);
            c.spark.add_tag("shared")?;
            c.df(&range(c))
        },
        "SparkSession::execute_command_proto" => |c| {
            let command = spark_connect_proto::Command {
                command_type: Some(spark_connect_proto::command::CommandType::CreateDataframeView(
                    spark_connect_proto::CreateDataFrameViewCommand {
                        input: Some(range(c).to_relation()),
                        name: "v".to_string(),
                        is_global: false,
                        replace: true,
                    },
                )),
            };
            c.spark.execute_command_proto(command).map(drop)
        },
        "SparkSession::json_to_ddl" => |c| {
            c.spark.json_to_ddl(r#"{"type":"struct","fields":[{"name":"a","type":"integer","nullable":true,"metadata":{}}]}"#).map(drop)
        },
        "SparkSession::register_java_function_with" => {
            |c| c.spark.register_java_function_with("f", "com.example.F", Some(DataType::Integer), false)
        },
        "SparkSession::register_java_function_with/aggregate" => {
            |c| c.spark.register_java_function_with("g", "com.example.G", None, true)
        },
        "SparkSession::set_execute_plan_hook" => |c| {
            let hook: spark_connect::ExecutePlanHook = Arc::new(|mut request| {
                request.tags.push("hooked".to_string());
                Ok(request)
            });
            c.spark.set_execute_plan_hook(Some(hook));
            c.df(&range(c))
        },
        "SparkSession::release_session" => |c| c.spark.release_session(),
        "SparkSession::close" => |c| {
            c.spark.close();
            c.df(&range(c))
        },
    ]
}
