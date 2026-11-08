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
use std::sync::Arc;

use spark_connect::streaming::{StreamingQueryListener, StreamingQueryListenerEvent, Trigger};
use spark_connect::udf::eval_type;

use super::{payload, range};
use crate::{Case, Ctx};

struct Listener;

impl StreamingQueryListener for Listener {
    fn on_event(&self, _: &StreamingQueryListenerEvent) {}
}

fn options() -> HashMap<String, String> {
    HashMap::from([
        ("a".to_string(), "1".to_string()),
        ("b".to_string(), "2".to_string()),
    ])
}

fn query(c: &Ctx) -> spark_connect::Result<spark_connect::streaming::StreamingQuery> {
    range(c)
        .write_stream()
        .format("memory")
        .query_name("q")
        .start("")
}

pub fn cases() -> Vec<Case> {
    cases![
        "DataStreamReader::load" => {
            |c| c.df(&c.spark.read_stream().format("rate").schema("a INT").option("k", "v").options(options()).name("src").load(Some("/in")))
        },
        "DataStreamReader::load/no_path" => |c| c.df(&c.spark.read_stream().format("rate").load(None)),
        "DataStreamReader::getters" => |c| {
            let reader = c.spark.read_stream().format("rate").schema("a INT").option("k", "v");
            c.message("Text", &format!("{:?} {} {:?}", reader.get_format(), reader.get_schema(), reader.get_options()));
            Ok(())
        },
        "DataStreamReader::table" => |c| c.df(&c.spark.read_stream().table("db.t")),
        "DataStreamReader::changes" => |c| c.df(&c.spark.read_stream().changes("db.t")),
        "DataStreamReader::json" => |c| c.df(&c.spark.read_stream().json("/in")),
        "DataStreamReader::parquet" => |c| c.df(&c.spark.read_stream().parquet("/in")),
        "DataStreamReader::csv" => |c| c.df(&c.spark.read_stream().csv("/in")),
        "DataStreamReader::orc" => |c| c.df(&c.spark.read_stream().orc("/in")),
        "DataStreamReader::text" => |c| c.df(&c.spark.read_stream().text("/in")),
        "DataStreamWriter::start" => |c| {
            range(c)
                .write_stream()
                .format("parquet")
                .output_mode("append")
                .option("checkpointLocation", "/cp")
                .options(options())
                .partition_by(vec!["a"])
                .cluster_by(vec!["b"])
                .query_name("q")
                .start("/out")
                .map(drop)
        },
        "DataStreamWriter::to_table" => |c| range(c).write_stream().to_table("db.t").map(drop),
        "DataStreamWriter::trigger" => |c| {
            for trigger in [
                Trigger::ProcessingTime("5 seconds".to_string()),
                Trigger::Once,
                Trigger::AvailableNow,
                Trigger::Continuous("1 second".to_string()),
                Trigger::RealTime("10 seconds".to_string()),
            ] {
                range(c).write_stream().trigger(trigger).start("/out")?;
            }
            Ok(())
        },
        "DataStreamWriter::foreach_batch" => {
            |c| range(c).write_stream().foreach_batch(payload(eval_type::SQL_BATCHED_UDF)).start("").map(drop)
        },
        "DataStreamWriter::foreach" => {
            |c| range(c).write_stream().foreach(payload(eval_type::SQL_BATCHED_UDF)).start("").map(drop)
        },
        "StreamingQuery::id" => |c| {
            let query = query(c)?;
            c.message("Text", &format!("{} {} {:?}", query.id(), query.run_id(), query.name()));
            Ok(())
        },
        "StreamingQuery::is_active" => |c| query(c)?.is_active().map(drop),
        "StreamingQuery::status" => |c| query(c)?.status().map(drop),
        "StreamingQuery::stop" => |c| query(c)?.stop(),
        "StreamingQuery::await_termination" => |c| {
            let query = query(c)?;
            query.await_termination(None)?;
            query.await_termination(Some(1.5)).map(drop)
        },
        "StreamingQuery::last_progress" => |c| query(c)?.last_progress().map(drop),
        "StreamingQuery::recent_progress" => |c| query(c)?.recent_progress().map(drop),
        "StreamingQuery::process_all_available" => |c| query(c)?.process_all_available(),
        "StreamingQuery::explain" => |c| {
            let query = query(c)?;
            query.explain(false)?;
            query.explain(true).map(drop)
        },
        "StreamingQuery::exception" => |c| query(c)?.exception().map(drop),
        "StreamingQueryManager::active" => |c| c.spark.streams().active().map(drop),
        "StreamingQueryManager::get" => |c| c.spark.streams().get("query-id").map(drop),
        "StreamingQueryManager::await_any_termination" => |c| {
            c.spark.streams().await_any_termination(None)?;
            c.spark.streams().await_any_termination(Some(2.0)).map(drop)
        },
        "StreamingQueryManager::reset_terminated" => |c| c.spark.streams().reset_terminated(),
        "StreamingQueryManager::register_python_listener" => |c| {
            let id = c.spark.streams().register_python_listener(payload(eval_type::SQL_BATCHED_UDF))?;
            c.mask(&id);
            Ok(())
        },
        "StreamingQueryManager::unregister_python_listener" => {
            |c| c.spark.streams().unregister_python_listener("listener-id")
        },
        "StreamingQueryManager::listener_event_stream" => |c| {
            for event in c.spark.streams().listener_event_stream()? {
                event?;
            }
            Ok(())
        },
        "StreamingQueryManager::add_listener" => |c| {
            c.unordered();
            let streams = c.spark.streams();
            let id = streams.add_listener(Arc::new(Listener))?;
            streams.remove_listener(&id)?;
            let id = streams.add_listener(Arc::new(Listener))?;
            c.mask(&id);
            streams.close()
        },
        "StreamingQueryManager::remove_listener_found" => |c| {
            let streams = c.spark.streams();
            let id = streams.add_listener(Arc::new(Listener))?;
            c.message("Text", &format!("{} {}", streams.remove_listener_found(&id)?, streams.remove_listener_found(&id)?));
            Ok(())
        },
        "StreamingQueryManager::listeners" => |c| {
            let streams = c.spark.streams();
            let id = streams.add_listener(Arc::new(Listener))?;
            c.message("Text", &format!("{}", streams.listeners().len()));
            streams.remove_listener(&id)
        },
        "DataStreamReader::clear_format" => {
            |c| c.df(&c.spark.read_stream().format("rate").clear_format().load(Some("/in")))
        },
    ]
}
