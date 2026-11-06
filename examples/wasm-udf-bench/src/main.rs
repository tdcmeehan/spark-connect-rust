// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Rust UDFs per row vs per Arrow batch, end to end through a Spark Connect server.
//!
//! Each case runs `range(n, 1 partition).select(f(id)).agg(sum)`. The cost per row is the
//! difference between two row counts divided by the difference in rows, so query, worker and
//! module start-up costs cancel out. Best of 3 per row count, after one warm-up run.
//!
//!     rustup target add wasm32-unknown-unknown
//!     SPARK_CONNECT_PYTHON_VERSION=<the executors' Python, e.g. 3.12> cargo run --release

use spark_connect_macros::spark_wasm_udf;

#[spark_wasm_udf]
mod udfs {
    pub fn add_one(x: i64) -> i64 {
        x + 1
    }

    pub fn shout(s: String) -> String {
        let mut t = s.to_uppercase();
        t.push('!');
        t
    }

    pub fn half_if_even(x: Option<i64>) -> Option<i64> {
        x.filter(|v| v % 2 == 0).map(|v| v / 2)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use spark_connect::functions::{col, expr, length, sum};
    use spark_connect::row::Value;
    use spark_connect::{Column, SparkSession, SparkSessionBuilder};
    use std::time::Instant;

    type BoxErr = Box<dyn std::error::Error>;

    let spark: SparkSession = SparkSessionBuilder::default()
        .remote("sc://127.0.0.1:15002/")
        .get_or_create()?;

    // Seconds to collect sum(f(id)) (or sum(length(f(id)))) over n ids in one partition.
    let run = |f: &dyn Fn() -> Result<Column, BoxErr>,
               sum_len: bool,
               n: i64|
     -> Result<(f64, i64), BoxErr> {
        let total = if sum_len {
            sum(length(col("x")))
        } else {
            sum(col("x"))
        };
        let df = spark
            .range_full(0, n, 1, Some(1))?
            .select([f()?.alias("x")])
            .agg(vec![total.expression().clone()]);
        let start = Instant::now();
        let rows = df.collect()?;
        let secs = start.elapsed().as_secs_f64();
        match rows[0].get(0) {
            Some(Value::Long(v)) => Ok((secs, *v)),
            other => Err(format!("unexpected result {other:?}").into()),
        }
    };

    let ns_per_row = |name: &str,
                      f: &dyn Fn() -> Result<Column, BoxErr>,
                      sum_len: bool,
                      expected: &dyn Fn(i64) -> i64,
                      n1: i64,
                      n2: i64|
     -> Result<(), BoxErr> {
        run(f, sum_len, n1)?;
        let mut best = [f64::MAX; 2];
        for (k, n) in [n1, n2].into_iter().enumerate() {
            for _ in 0..3 {
                let (secs, v) = run(f, sum_len, n)?;
                if v != expected(n) {
                    return Err(
                        format!("{name}: sum {v} for {n} rows, expected {}", expected(n)).into(),
                    );
                }
                best[k] = best[k].min(secs);
            }
        }
        let ns = (best[1] - best[0]) / (n2 - n1) as f64 * 1e9;
        println!(
            "{name:22} {ns:10.1} ns/row   ({n1:>10} rows {:7.3} s, {n2:>10} rows {:7.3} s)",
            best[0], best[1]
        );
        Ok(())
    };

    let add_one_sum = |n: i64| n * (n - 1) / 2 + n;
    let shout_len = |n: i64| (0..n).map(|i| i.to_string().len() as i64 + 1).sum::<i64>();
    let half_sum = |n: i64| (0..n).filter(|i| i % 2 == 0).map(|i| i / 2).sum::<i64>();
    let id_str = || col("id").cast_str("string");

    // Same answers on both paths, including null results.
    let n = 25_000;
    let (_, per_row) = run(
        &|| Ok(udf::half_if_even_udf().per_row().call(vec![col("id")])?),
        false,
        n,
    )?;
    let (_, batch) = run(&|| Ok(udf::half_if_even(col("id"))?), false, n)?;
    assert_eq!((per_row, batch), (half_sum(n), half_sum(n)), "half_if_even");

    let start = Instant::now();
    for _ in 0..1000 {
        udf::add_one(col("id"))?;
    }
    let us = start.elapsed().as_secs_f64() / 1000.0 * 1e6;
    println!("{:22} {us:10.1} us", "build a UDF column");

    ns_per_row(
        "builtin id + 1",
        &|| Ok(expr("id + 1")),
        false,
        &add_one_sum,
        2_000_000,
        20_000_000,
    )?;
    ns_per_row(
        "add_one per row",
        &|| Ok(udf::add_one_udf().per_row().call(vec![col("id")])?),
        false,
        &add_one_sum,
        10_000,
        50_000,
    )?;
    ns_per_row(
        "add_one per batch",
        &|| Ok(udf::add_one(col("id"))?),
        false,
        &add_one_sum,
        2_000_000,
        20_000_000,
    )?;
    ns_per_row(
        "shout per row",
        &|| Ok(udf::shout_udf().per_row().call(vec![id_str()])?),
        true,
        &shout_len,
        10_000,
        50_000,
    )?;
    ns_per_row(
        "shout per batch",
        &|| Ok(udf::shout(id_str())?),
        true,
        &shout_len,
        1_000_000,
        5_000_000,
    )?;
    Ok(())
}
