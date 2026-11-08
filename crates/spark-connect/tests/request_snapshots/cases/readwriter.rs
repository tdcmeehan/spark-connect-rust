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

use spark_connect::column::{col, lit};
use spark_connect::readwriter::{SaveMode, TableSaveMethod};

use super::range;
use crate::Case;

fn options() -> HashMap<String, String> {
    HashMap::from([
        ("a".to_string(), "1".to_string()),
        ("b".to_string(), "2".to_string()),
    ])
}

// One assignment: they are sent in the order of the HashMap, which differs per process.
fn assignments() -> HashMap<String, spark_connect::Column> {
    HashMap::from([("x".to_string(), col("s.x"))])
}

pub fn cases() -> Vec<Case> {
    cases![
        "DataFrameReader::load" => |c| c.df(&c.spark.read().load(Some("/data/x"))),
        "DataFrameReader::load/no_path" => |c| c.df(&c.spark.read().format("delta").load(None)),
        "DataFrameReader::format" => {
            |c| c.df(&c.spark.read().format("parquet").schema("a INT".to_string()).option("k", "v").options(options()).load(Some("/p")))
        },
        "DataFrameReader::getters" => |c| {
            let reader = c.spark.read().format("csv").schema("a INT".to_string()).option("h", "true");
            c.message("Text", &format!("{:?} {} {:?}", reader.get_format(), reader.get_schema(), reader.get_options()));
            Ok(())
        },
        "DataFrameReader::changes" => |c| c.df(&c.spark.read().option("startingVersion", "1").changes("db.t")),
        "DataFrameReader::table" => |c| c.df(&c.spark.read().option("k", "v").table("db.t")),
        "DataFrameReader::json" => |c| c.df(&c.spark.read().json("/a.json")),
        "DataFrameReader::parquet" => |c| c.df(&c.spark.read().parquet("/a.parquet")),
        "DataFrameReader::csv" => |c| c.df(&c.spark.read().option("header", "true").csv("/a.csv")),
        "DataFrameReader::orc" => |c| c.df(&c.spark.read().orc("/a.orc")),
        "DataFrameReader::text" => |c| c.df(&c.spark.read().text("/a.txt")),
        "DataFrameReader::xml" => |c| c.df(&c.spark.read().option("rowTag", "r").xml("/a.xml")),
        "DataFrameReader::jdbc" => |c| c.df(&c.spark.read().jdbc("jdbc:x", "t", None)),
        "DataFrameReader::jdbc/predicates" => {
            |c| c.df(&c.spark.read().jdbc("jdbc:x", "t", Some(vec!["a < 1".to_string(), "a >= 1".to_string()])))
        },
        "SaveMode::to_proto" => |c| {
            for mode in ["append", "overwrite", "error", "errorifexists", "ignore", "other"] {
                c.message("Text", &format!("{mode}: {:?}", SaveMode::from_str(mode).map(|m| m.to_proto())));
            }
            Ok(())
        },
        "TableSaveMethod::to_proto" => |c| {
            c.message("Text", &format!("{} {}", TableSaveMethod::SaveAsTable.to_proto(), TableSaveMethod::InsertInto.to_proto()));
            Ok(())
        },
        "DataFrameWriter::save" => |c| range(c).write().save(None),
        "DataFrameWriter::save/path" => {
            |c| range(c).write().format("parquet").mode("overwrite").option("k", "v").options(options()).save(Some("/out"))
        },
        "DataFrameWriter::mode" => |c| {
            for mode in ["append", "overwrite", "error", "ignore"] {
                range(c).write().mode(mode).save(Some("/out"))?;
            }
            Ok(())
        },
        "DataFrameWriter::partition_by" => {
            |c| range(c).write().partition_by(["a", "b"]).bucket_by(4, ["c"]).sort_by(["d"]).save_as_table("db.t")
        },
        "DataFrameWriter::cluster_by" => |c| range(c).write().cluster_by(["a"]).save_as_table("db.t"),
        "DataFrameWriter::save_as_table" => |c| range(c).write().mode("append").save_as_table("db.t"),
        "DataFrameWriter::insert_into" => |c| range(c).write().insert_into("db.t"),
        "DataFrameWriter::insert_into/overwrite" => |c| range(c).write().mode("overwrite").insert_into("db.t"),
        "DataFrameWriter::json" => |c| range(c).write().json("/o.json"),
        "DataFrameWriter::parquet" => |c| range(c).write().parquet("/o.parquet"),
        "DataFrameWriter::csv" => |c| range(c).write().option("header", "true").csv("/o.csv"),
        "DataFrameWriter::orc" => |c| range(c).write().orc("/o.orc"),
        "DataFrameWriter::text" => |c| range(c).write().text("/o.txt"),
        "DataFrameWriter::xml" => |c| range(c).write().xml("/o.xml"),
        "DataFrameWriterV2::create" => {
            |c| range(c).write_to("db.t").using("delta").option("k", "v").options(options()).table_property("p", "q").partition_by([col("a")]).create()
        },
        "DataFrameWriterV2::replace" => |c| range(c).write_to("db.t").replace(),
        "DataFrameWriterV2::create_or_replace" => |c| range(c).write_to("db.t").cluster_by(["a"]).create_or_replace(),
        "DataFrameWriterV2::append" => |c| range(c).write_to("db.t").append(),
        "DataFrameWriterV2::overwrite" => |c| range(c).write_to("db.t").overwrite(col("a").gt(lit(1))),
        "DataFrameWriterV2::overwrite_partitions" => |c| range(c).write_to("db.t").overwrite_partitions(),
        "MergeIntoWriter::merge" => |c| {
            range(c)
                .merge_into("db.t", col("t.id").eq(col("s.id")))
                .when_matched(Some(col("s.d")))
                .delete()
                .when_matched(None)
                .update(assignments())
                .when_matched(None)
                .update_all()
                .when_not_matched(Some(col("s.n")))
                .insert(assignments())
                .when_not_matched(None)
                .insert_all()
                .when_not_matched_by_source(Some(col("t.o")))
                .delete()
                .when_not_matched_by_source(None)
                .update(assignments())
                .when_not_matched_by_source(None)
                .update_all()
                .with_schema_evolution()
                .merge()
        },
        "DataFrameWriter::clear_format" => |c| range(c).write().format("csv").clear_format().save(Some("/p")),
    ]
}
