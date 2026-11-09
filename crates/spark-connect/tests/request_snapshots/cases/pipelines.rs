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

use proto::pipeline_command::define_flow::AutoCdcFlowDetails;
use spark_connect::column::col;
use spark_connect::pipelines::{self, OutputSpec, SinkDetailsSpec, TableDetailsSpec};
use spark_connect_proto as proto;

use super::{range, schema};
use crate::Case;

fn conf() -> HashMap<String, String> {
    HashMap::from([("k".to_string(), "v".to_string())])
}

fn location() -> proto::SourceCodeLocation {
    proto::SourceCodeLocation {
        file_name: Some("flows.py".to_string()),
        line_number: Some(4),
        definition_path: Some("flows.f".to_string()),
        ..Default::default()
    }
}

fn table(
    schema_data_type: Option<spark_connect::DataType>,
    schema_string: Option<String>,
) -> TableDetailsSpec {
    TableDetailsSpec {
        table_properties: conf(),
        partition_cols: vec!["p".to_string()],
        clustering_columns: vec!["c".to_string()],
        format: Some("delta".to_string()),
        schema_string,
        schema_data_type,
    }
}

fn output(name: &str, output_type: i32) -> OutputSpec {
    OutputSpec {
        name: name.to_string(),
        output_type,
        comment: None,
        source_code_location: None,
        table_details: None,
        sink_details: None,
    }
}

pub fn cases() -> Vec<Case> {
    cases![
        "pipelines::create_dataflow_graph" => |c| {
            pipelines::create_dataflow_graph(&c.spark, Some("cat".to_string()), Some("db".to_string()), conf())?;
            pipelines::create_dataflow_graph(&c.spark, None, None, HashMap::new()).map(drop)
        },
        "pipelines::define_output" => |c| {
            let mut mv = output("mv", 1);
            mv.comment = Some("comment".to_string());
            mv.source_code_location = Some(location());
            mv.table_details = Some(table(Some(schema()), None));
            pipelines::define_output(&c.spark, "graph-id", mv)?;
            let mut t = output("t", 2);
            t.table_details = Some(table(None, Some("a INT".to_string())));
            pipelines::define_output(&c.spark, "graph-id", t)?;
            let mut plain = output("plain", 2);
            plain.table_details = Some(table(None, None));
            pipelines::define_output(&c.spark, "graph-id", plain)?;
            pipelines::define_output(&c.spark, "graph-id", output("v", 3))?;
            let mut sink = output("s", 4);
            sink.sink_details = Some(SinkDetailsSpec { options: conf(), format: Some("kafka".to_string()) });
            pipelines::define_output(&c.spark, "graph-id", sink)
        },
        "pipelines::define_flow" => |c| {
            pipelines::define_flow(&c.spark, "graph-id", "f", "t", &range(c), conf(), Some(location()))?;
            pipelines::define_flow(&c.spark, "graph-id", "g", "t", &range(c).filter(col("id").is_not_null()), HashMap::new(), None)
        },
        "pipelines::define_auto_cdc_flow" => |c| {
            let details = AutoCdcFlowDetails {
                source: Some("src".to_string()),
                keys: vec![col("k").to_proto()],
                sequence_by: Some(col("ts").to_proto()),
                stored_as_scd_type: 2,
                ..Default::default()
            };
            pipelines::define_auto_cdc_flow(&c.spark, "graph-id", "cdc", "t", details, conf(), Some(location()))
        },
        "pipelines::define_sql_graph_elements" => {
            |c| pipelines::define_sql_graph_elements(&c.spark, "graph-id", "CREATE MATERIALIZED VIEW mv AS SELECT 1", "defs.sql")
        },
        "pipelines::start_run" => |c| {
            let mut run = pipelines::start_run(&c.spark, "graph-id", vec!["a".to_string()], false, vec!["b".to_string()], true, Some("/storage".to_string()))?;
            while run.next_response()?.is_some() {}
            let mut run = pipelines::start_run(&c.spark, "graph-id", vec![], true, vec![], false, None)?;
            while run.next_response()?.is_some() {}
            Ok(())
        },
    ]
}
