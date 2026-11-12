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

//! The aggregations, statistics and storage of DataFrames of the async API against a Spark
//! Connect server at `SPARK_REMOTE`; each test passes without running when it is not set.

use std::future::{Future, IntoFuture};
use std::panic::AssertUnwindSafe;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::FutureExt;
use spark_connect_async::actions::StorageLevel;
use spark_connect_async::options::Pivot;
use spark_connect_async::{lit, DataFrame, Decimal, ErrorKind, Result, Session};

/// The longest that a call to the server may take.
const WAIT: Duration = Duration::from_secs(120);

/// Waits for the call that `call` makes until the deadline. The call's future is made and boxed
/// in a frame of its own: made in the frame of a test, the futures of a debug build overflowed
/// the stack of its thread.
async fn within<F: IntoFuture>(call: impl FnOnce() -> F) -> F::Output {
    let call = Box::pin(call().into_future());
    tokio::time::timeout(WAIT, call)
        .await
        .expect("no answer before the deadline")
}

async fn session() -> Option<Session> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    Some(
        within(|| Session::builder().remote(remote).connect())
            .await
            .unwrap(),
    )
}

fn unique(prefix: &str) -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{prefix}_{}", now.as_nanos())
}

/// Runs `body`, then unpersists `cached`, whose cache the server shares between sessions, and
/// stops `session`, also when `body` panics.
async fn cleaned_up(session: &Session, cached: Option<&DataFrame>, body: impl Future<Output = ()>) {
    let outcome = AssertUnwindSafe(body).catch_unwind().await;
    if let Some(cached) = cached {
        let _ = tokio::time::timeout(WAIT, cached.unpersist(true)).await;
    }
    let _ = tokio::time::timeout(WAIT, session.stop()).await;
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
}

/// Groups `g` 1 and 2 of numbers of every numeric type, a string, a struct and a boolean;
/// group 2's numbers are all `NULL`.
const SALES: &str = "SELECT * FROM VALUES \
    (1, 'a', CAST(1 AS TINYINT), 10L, CAST(1.5 AS FLOAT), 2.25D, CAST(1.5 AS DECIMAL(4, 1)), \
     named_struct('x', 1), true), \
    (1, 'b', CAST(3 AS TINYINT), 30L, CAST(2.5 AS FLOAT), 4.25D, CAST(2.5 AS DECIMAL(4, 1)), \
     named_struct('x', 2), false), \
    (2, 'a', NULL, NULL, NULL, NULL, NULL, named_struct('x', 3), NULL) \
    AS t(g, s, b, l, f, d, dec, st, flag)";

fn dec(unscaled: i128, precision: u8, scale: i8) -> Decimal {
    Decimal::new(unscaled, precision, scale).unwrap()
}

/// The columns of the DataFrame that `aggregated` makes, and its rows in the order of `g`.
async fn sorted<T: serde::de::DeserializeOwned + Send + 'static, F>(
    aggregated: impl FnOnce() -> F,
) -> (Vec<String>, Vec<T>)
where
    F: IntoFuture<Output = Result<DataFrame>>,
{
    let df = within(aggregated).await.unwrap().sort(["g"]);
    let columns = within(|| df.columns()).await.unwrap();
    (columns, within(|| df.collect_as()).await.unwrap())
}

fn not_numeric(aggregated: Result<DataFrame>, columns: &str) {
    let Err(error) = aggregated else {
        panic!("the columns {columns} were aggregated");
    };
    assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{error}");
    assert_eq!(error.error_class(), Some("NOT_NUMERIC_COLUMNS"));
    assert_eq!(error.sql_state().as_deref(), Some("42K09"));
    assert_eq!(error.message_parameters(), [("invalid_columns", columns)]);
}

#[tokio::test]
async fn numeric_aggregations_take_the_numeric_columns_of_the_schema_of_the_server() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, None, async {
        let sales = within(|| session.sql(SALES)).await.unwrap();
        let by_g = sales.group_by(["g"]);

        type Sums = (
            i32,
            Option<i64>,
            Option<i64>,
            Option<i64>,
            Option<f64>,
            Option<f64>,
            Option<Decimal>,
        );
        let (columns, sums) = sorted::<Sums, _>(|| by_g.sum(Vec::<String>::new())).await;
        assert_eq!(
            columns,
            ["g", "sum(g)", "sum(b)", "sum(l)", "sum(f)", "sum(d)", "sum(dec)"]
        );
        assert_eq!(
            sums,
            [
                (
                    1,
                    Some(2),
                    Some(4),
                    Some(40),
                    Some(4.0),
                    Some(6.5),
                    Some(dec(40, 14, 1))
                ),
                (2, Some(2), None, None, None, None, None)
            ]
        );

        type Averages = (i32, Option<f64>, Option<Decimal>);
        let (columns, averages) = sorted::<Averages, _>(|| by_g.avg(["l", "dec"])).await;
        assert_eq!(columns, ["g", "avg(l)", "avg(dec)"]);
        assert_eq!(
            averages,
            [(1, Some(20.0), Some(dec(200_000, 8, 5))), (2, None, None)]
        );
        type Extremes = (i32, Option<i8>, Option<f32>);
        let (columns, least) = sorted::<Extremes, _>(|| by_g.min(["b", "f"])).await;
        assert_eq!(columns, ["g", "min(b)", "min(f)"]);
        assert_eq!(least, [(1, Some(1), Some(1.5)), (2, None, None)]);
        let (columns, greatest) = sorted::<Extremes, _>(|| by_g.max(["b", "f"])).await;
        assert_eq!(columns, ["g", "max(b)", "max(f)"]);
        assert_eq!(greatest, [(1, Some(3), Some(2.5)), (2, None, None)]);

        let (_, rolled) =
            sorted::<(Option<i32>, Option<i64>), _>(|| sales.rollup(["g"]).sum(["l"])).await;
        assert_eq!(
            rolled,
            [(None, Some(40)), (Some(1), Some(40)), (Some(2), None)]
        );
        let (_, cubed) =
            sorted::<(Option<i32>, Option<i64>), _>(|| sales.cube(["g"]).max(["l"])).await;
        assert_eq!(
            cubed,
            [(None, Some(30)), (Some(1), Some(30)), (Some(2), None)]
        );
        let sets = sales.grouping_sets([vec!["g"], vec![]], ["g"]);
        let (_, set_sums) = sorted::<(Option<i32>, Option<i64>), _>(|| sets.sum(["b"])).await;
        assert_eq!(
            set_sums,
            [(None, Some(4)), (Some(1), Some(4)), (Some(2), None)]
        );

        // "z" is no value of `s`: the pivot's columns are its values.
        let pivoted = by_g.pivot(Pivot::new("s").values(["b", "z"]));
        let (columns, pivot_sums) =
            sorted::<(i32, Option<i64>, Option<i64>), _>(|| pivoted.sum(["l"])).await;
        assert_eq!(columns, ["g", "b", "z"]);
        assert_eq!(pivot_sums, [(1, Some(30), None), (2, None, None)]);
        let (columns, _) =
            sorted::<(i32, Option<i64>, Option<f64>, Option<i64>, Option<f64>), _>(|| {
                pivoted.sum(["l", "d"])
            })
            .await;
        assert_eq!(
            columns,
            ["g", "b_sum(l)", "b_sum(d)", "z_sum(l)", "z_sum(d)"]
        );

        not_numeric(within(|| by_g.sum(["s"])).await, r#"["s"]"#);
        not_numeric(
            within(|| by_g.avg(["missing", "s", "l"])).await,
            r#"["missing", "s"]"#,
        );
        not_numeric(within(|| by_g.max(["st"])).await, r#"["st"]"#);
        not_numeric(
            within(|| sales.rollup(["g"]).min(["flag"])).await,
            r#"["flag"]"#,
        );
        not_numeric(within(|| pivoted.sum(["s"])).await, r#"["s"]"#);
    })
    .await;
}

#[tokio::test]
async fn statistics_of_numeric_columns_are_the_servers_and_others_are_its_errors() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, None, async {
        let points = within(|| {
            session.sql(
                "SELECT * FROM VALUES (1, 2.0D, 'a'), (2, 4.0D, 'b'), (3, 6.0D, 'c') AS t(x, y, s)",
            )
        })
        .await
        .unwrap();
        let stat = points.stat();
        let correlation = within(|| stat.corr("x", "y")).await.unwrap();
        assert!((correlation - 1.0).abs() < 1e-12, "{correlation}");
        assert_eq!(within(|| stat.cov("x", "y")).await.unwrap(), 2.0);
        assert_eq!(
            within(|| stat.approx_quantile(["x", "y"], [0.0, 0.5, 1.0], 0.0))
                .await
                .unwrap(),
            [vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]]
        );

        let none = points.limit(0).stat();
        let undefined = within(|| none.corr("x", "y")).await.unwrap();
        assert!(undefined.is_nan(), "{undefined}");
        assert_eq!(within(|| none.cov("x", "y")).await.unwrap(), 0.0);
        assert_eq!(
            within(|| none.approx_quantile(["x"], [0.5], 0.0))
                .await
                .unwrap(),
            [Vec::<f64>::new()]
        );

        for error in [
            within(|| stat.corr("s", "x")).await.unwrap_err(),
            within(|| stat.cov("x", "s")).await.unwrap_err(),
            within(|| stat.approx_quantile(["s"], [0.5], 0.0))
                .await
                .unwrap_err(),
        ] {
            assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{error}");
            assert!(error.message().contains("not supported"), "{error}");
        }
        let missing = within(|| stat.corr("missing", "x")).await.unwrap_err();
        assert_eq!(
            missing.error_class(),
            Some("UNRESOLVED_COLUMN.WITH_SUGGESTION")
        );
    })
    .await;
}

#[tokio::test]
async fn a_cached_dataframe_has_the_storage_level_of_the_server_until_it_is_unpersisted() {
    let Some(session) = session().await else {
        return;
    };
    // The server caches by plan: a plan of this run's own.
    let numbers = session.range(5).with_column("run", lit(unique("b15")));
    cleaned_up(&session, Some(&numbers), async {
        within(|| numbers.cache()).await.unwrap();
        assert_eq!(
            within(|| numbers.storage_level()).await.unwrap(),
            StorageLevel::MEMORY_AND_DISK_DESER
        );
        assert_eq!(within(|| numbers.count()).await.unwrap(), 5);
        within(|| numbers.unpersist(true)).await.unwrap();
        assert_eq!(
            within(|| numbers.storage_level()).await.unwrap(),
            StorageLevel::NONE
        );
        within(|| numbers.persist(StorageLevel::DISK_ONLY))
            .await
            .unwrap();
        assert_eq!(
            within(|| numbers.storage_level()).await.unwrap(),
            StorageLevel::DISK_ONLY
        );
        within(|| numbers.unpersist(true)).await.unwrap();
    })
    .await;
}
