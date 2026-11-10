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

//! The actions of the async API against a Spark Connect server at `SPARK_REMOTE`; each test
//! passes without running when it is not set.

use std::time::Duration;

use serde::Deserialize;
use spark_connect::functions::count;
use spark_connect_async::actions::{Checkpoint, ExplainMode, TempView};
use spark_connect_async::{col, lit, Observation, Session};

async fn session() -> Option<Session> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    Some(Session::builder().remote(remote).connect().await.unwrap())
}

/// A query that runs for minutes.
const LONG: &str = "SELECT count(*) FROM range(1000000) a CROSS JOIN range(1000000) b \
                    WHERE a.id + b.id = -1";

#[tokio::test]
async fn results_read_into_rust_types_with_the_server_names() {
    let Some(session) = session().await else {
        return;
    };
    #[derive(Deserialize, Debug, PartialEq)]
    struct Point {
        x: i64,
    }
    #[derive(Deserialize, Debug, PartialEq)]
    struct Row {
        id: i64,
        point: Point,
        labels: Vec<(String, i64)>,
    }
    let df = session
        .sql("SELECT id, named_struct('x', id) AS point, map('k', id) AS labels FROM range(3)")
        .await
        .unwrap();
    let rows: Vec<Row> = df.collect_as().await.unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows[2],
        Row {
            id: 2,
            point: Point { x: 2 },
            labels: vec![("k".to_string(), 2)]
        }
    );
    assert_eq!(df.count().await.unwrap(), 3);
    assert_eq!(df.columns().await.unwrap(), ["id", "point", "labels"]);
    assert!(df.show().to_string().await.unwrap().contains("point"));
    assert!(df
        .explain(ExplainMode::Extended)
        .await
        .unwrap()
        .contains("Range"));
    assert_eq!(df.take(2).await.unwrap().len(), 2);
    assert!(df.filter(col("id").gt(lit(5))).is_empty().await.unwrap());
}

#[tokio::test]
async fn checkpoints_views_and_observations_work_on_the_server() {
    let Some(session) = session().await else {
        return;
    };
    let df = session.range(5);
    let checkpoint = df.checkpoint(Checkpoint::default().local()).await.unwrap();
    assert_eq!(checkpoint.count().await.unwrap(), 5);
    df.create_temp_view("b7_numbers", TempView::default().or_replace())
        .await
        .unwrap();
    let viewed = session.sql("SELECT * FROM b7_numbers").await.unwrap();
    assert_eq!(viewed.count().await.unwrap(), 5);
    let observation = Observation::new("b7_metrics");
    let observed = df.observe(&observation, [count(lit(1)).alias("rows")]);
    observed.collect().await.unwrap();
    let metrics = observation.get().unwrap();
    assert_eq!(
        metrics.get_by_name("rows"),
        Some(&spark_connect_async::Value::Long(5))
    );
}

#[tokio::test]
async fn dropping_a_running_action_stops_it_on_the_server() {
    let Some(session) = session().await else {
        return;
    };
    let long = session.sql(LONG).await.unwrap();

    // The control: the tagged query runs until it is interrupted.
    let control = session.with_tag("b7-control").unwrap();
    let running = tokio::spawn(control.sql(LONG).await.unwrap().collect());
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert_eq!(session.interrupt_tag("b7-control").await.unwrap().len(), 1);
    assert!(running.await.unwrap().is_err());

    let dropped = session.with_tag("b7-dropped").unwrap();
    let running = tokio::spawn(dropped.sql(LONG).await.unwrap().collect());
    tokio::time::sleep(Duration::from_secs(3)).await;
    running.abort();
    let _ = running.await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert!(session
        .interrupt_tag("b7-dropped")
        .await
        .unwrap()
        .is_empty());
    drop(long);
}
