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

//! The SQL statements of the async API send the requests of `apache-spark-connect` for the same
//! statements, as the in-process server of its request snapshots records them.

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;
mod support;

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;
use prost::Message;
use spark_connect::{lit_string, SparkSession};
use spark_connect_async::{lit, DataFrame, Session};
use spark_connect_proto as proto;

const SESSION_ID: &str = "00000000-0000-4000-8000-00000000005e";
const QUERY: &str = "SELECT * FROM people WHERE age > ?";

struct Case {
    name: &'static str,
    old: fn(&SparkSession) -> spark_connect::DataFrame,
    new: fn(Session) -> BoxFuture<'static, DataFrame>,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "a statement",
            old: |s| s.sql("SHOW TABLES").unwrap(),
            new: |s| Box::pin(async move { s.sql("SHOW TABLES").await.unwrap() }),
        },
        Case {
            name: "positional arguments",
            old: |s| {
                let args = vec![
                    spark_connect::lit(18).expression().clone(),
                    lit_string("a").expression().clone(),
                ];
                s.sql_with_args(QUERY, args, HashMap::new()).unwrap()
            },
            new: |s| Box::pin(async move { s.sql(QUERY).arg(18).arg("a").await.unwrap() }),
        },
        Case {
            name: "a named argument of a column",
            old: |s| {
                let array = spark_connect::functions::array(vec![spark_connect::lit(1)]);
                let args = HashMap::from([("ages".to_string(), array.expression().clone())]);
                s.sql_with_args(QUERY, Vec::new(), args).unwrap()
            },
            new: |s| {
                Box::pin(async move {
                    let array =
                        spark_connect::functions::array(vec![spark_connect::Column::from(lit(1))]);
                    let array = spark_connect_async::Column::from(array);
                    s.sql(QUERY).named_arg("ages", array).await.unwrap()
                })
            },
        },
        Case {
            name: "references to DataFrames",
            old: |s| {
                let (people, ages) = (s.table("people").unwrap(), s.range(3).unwrap());
                let references = vec![(people, "p".to_string()), (ages, "a".to_string())];
                let query = "SELECT * FROM p JOIN a";
                s.sql_with_references(query, Vec::new(), HashMap::new(), references)
                    .unwrap()
            },
            new: |s| {
                Box::pin(async move {
                    s.sql("SELECT * FROM p JOIN a")
                        .reference("p", &s.table("people"))
                        .reference("a", &s.range(3))
                        .await
                        .unwrap()
                })
            },
        },
        Case {
            name: "a statement of a tagged session",
            old: |s| {
                s.add_tag("etl").unwrap();
                let df = s.sql("SHOW TABLES").unwrap();
                s.clear_tags();
                df
            },
            new: |s| {
                Box::pin(async move {
                    let tagged = s.with_tag("etl").unwrap();
                    tagged.sql("SHOW TABLES").await.unwrap()
                })
            },
        },
    ]
}

/// The requests, with the operation ids, which the client chooses at random, removed.
fn requests(entries: Vec<server::Entry>) -> Vec<proto::ExecutePlanRequest> {
    entries
        .into_iter()
        .map(|entry| {
            assert_eq!(entry.kind, "ExecutePlanRequest");
            let mut request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
            request.operation_id = None;
            request
        })
        .collect()
}

#[tokio::test]
async fn statements_send_the_requests_of_the_old_api_and_return_its_dataframes() {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let url = format!("sc://{address}/;session_id={SESSION_ID};user_id=snapshots");
    let mut cases_of_both = Vec::new();
    for case in cases() {
        let old_session = SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        let old = (case.old)(&old_session).to_relation();
        let old_requests = requests(recorder.take());
        let session = Session::builder()
            .remote(url.clone())
            .connect()
            .await
            .unwrap();
        let new = (case.new)(session).await.to_relation();
        let new_requests = requests(recorder.take());
        assert_eq!(old_requests.len(), 1, "{}", case.name);
        cases_of_both.push((case.name, (old_requests, old), (new_requests, new)));
    }
    let differences = support::differences(cases_of_both);
    assert!(differences.is_empty(), "{differences}");
}

#[tokio::test]
async fn a_statement_of_a_tagged_session_sends_its_tags() {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let session = Session::builder()
        .remote(format!("sc://{address}"))
        .connect()
        .await
        .unwrap();
    let tagged = session.with_tag("etl").unwrap().with_tag("daily").unwrap();
    tagged.sql("SHOW TABLES").await.unwrap();
    session.sql("SHOW TABLES").await.unwrap();
    let tags: Vec<Vec<String>> = requests(recorder.take())
        .into_iter()
        .map(|request| request.tags)
        .collect();
    assert_eq!(tags, [vec!["etl", "daily"], vec![]]);
}
