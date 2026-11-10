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

//! The DataFrames of the async API have the plans of those of `apache-spark-connect` for the
//! same operations.

mod support;

use std::collections::HashMap;

use spark_connect::dataframe::AsOfJoinOptions;
use spark_connect::plan::JoinType as OldJoinType;
use spark_connect::table_arg::TableArg as OldTableArg;
use spark_connect::{DataFrame as Old, SparkSession};
use spark_connect_async::dataframe::ColumnOrName;
use spark_connect_async::options::{
    AsOfDirection, AsOfJoin, JoinType, NearestByJoin, Partitioning, RandomSplit, RangeSpec, Sample,
    UnionByName,
};
use spark_connect_async::types::DataType;
use spark_connect_async::{col, lit, Column, DataFrame, Session};

const URL: &str = "sc://localhost:1";

struct Sessions {
    old: SparkSession,
    new: Session,
}

impl Sessions {
    async fn connect() -> Self {
        Sessions {
            old: SparkSession::builder().remote(URL).get_or_create().unwrap(),
            new: Session::builder().remote(URL).connect().await.unwrap(),
        }
    }

    fn old_df(&self) -> Old {
        self.old.range(10).unwrap()
    }

    fn new_df(&self) -> DataFrame {
        self.new.range(10)
    }

    fn old_right(&self) -> Old {
        self.old.table("t").unwrap()
    }

    fn new_right(&self) -> DataFrame {
        self.new.table("t")
    }
}

type Case = (&'static str, Old, DataFrame);

fn assert_same_plans(cases: Vec<Case>) {
    let relations = cases
        .into_iter()
        .map(|(name, old, new)| (name, old.to_relation(), new.to_relation()))
        .collect();
    let differences = support::differences(relations);
    assert!(differences.is_empty(), "{differences}");
}

fn exprs(columns: &[Column]) -> Vec<spark_connect::Expression> {
    columns.iter().map(|c| c.expression().clone()).collect()
}

#[tokio::test]
async fn projections_have_the_plans_of_the_old_api() {
    let s = Sessions::connect().await;
    let schema = DataType::from_ddl("id BIGINT, name STRING").unwrap();
    let (old, new) = (s.old_df(), s.new_df());
    let old_columns = old.select([old.col("id"), old.star(), old.col_regex("`i.*`")]);
    let old_columns = old_columns.select([old.metadata_column("_metadata")]);
    let new_columns = new.select([new.col("id"), new.star(), new.col_regex("`i.*`")]);
    let new_columns = new_columns.select([new.metadata_column("_metadata")]);
    assert_same_plans(vec![
        (
            "select of names",
            s.old_df().select(["id", "x"]),
            s.new_df().select(["id", "x"]),
        ),
        (
            "select of expressions",
            s.old_df()
                .select([col("id").alias("a"), spark_connect::lit(1)]),
            s.new_df().select([col("id").alias("a"), lit(1)]),
        ),
        (
            "select_expr",
            s.old_df().select_expr(vec!["id + 1 AS x", "2"]),
            s.new_df().select_expr(["id + 1 AS x", "2"]),
        ),
        (
            "with_column",
            s.old_df().with_column("x", col("id")),
            s.new_df().with_column("x", "id"),
        ),
        (
            "with_columns",
            s.old_df().with_columns(vec![
                ("a".into(), col("id")),
                ("b".into(), spark_connect::lit(2)),
            ]),
            s.new_df().with_columns([("a", col("id")), ("b", lit(2))]),
        ),
        (
            "with_column_renamed",
            s.old_df().with_column_renamed("id", "a"),
            s.new_df().with_column_renamed("id", "a"),
        ),
        (
            "with_columns_renamed",
            s.old_df()
                .with_columns_renamed(vec![("id".into(), "a".into()), ("a".into(), "b".into())]),
            s.new_df().with_columns_renamed([("id", "a"), ("a", "b")]),
        ),
        (
            "with_metadata",
            s.old_df().with_metadata_json("id", r#"{"unit":"m"}"#),
            s.new_df().with_metadata("id", r#"{"unit":"m"}"#),
        ),
        (
            "drop of names",
            s.old_df().drop(vec!["id", "x"]),
            s.new_df().drop(["id", "x"]),
        ),
        (
            "drop of columns and names",
            s.old_df().drop_columns(vec![col("a")], vec!["b"]),
            s.new_df().drop([ColumnOrName::from(col("a")), "b".into()]),
        ),
        (
            "to_df",
            s.old_df().to_df(vec!["a"]),
            s.new_df().to_df(["a"]),
        ),
        (
            "to_schema of names",
            s.old_df().to_schema(vec!["a"]),
            s.new_df().to_df(["a"]),
        ),
        (
            "to",
            s.old_df().to(schema.clone()),
            s.new_df().to_schema(schema),
        ),
        ("alias", s.old_df().alias("t"), s.new_df().alias("t")),
        (
            "as_table of an alias",
            s.old_df().as_table("t"),
            s.new_df().alias("t"),
        ),
        ("columns of the DataFrame", old_columns, new_columns),
        (
            "zip_with_index",
            s.old_df().zip_with_index("index"),
            s.new_df().zip_with_index("index"),
        ),
        (
            "transform",
            s.old_df().transform(|d| d.limit(1)),
            s.new_df().transform(|d| d.limit(1)),
        ),
    ]);
}

#[tokio::test]
async fn filters_orders_and_limits_have_the_plans_of_the_old_api() {
    let s = Sessions::connect().await;
    let order = [col("id").desc(), col("x")];
    assert_same_plans(vec![
        (
            "filter",
            s.old_df().filter(col("id").gt(spark_connect::lit(1))),
            s.new_df().filter(col("id").gt(lit(1))),
        ),
        (
            "where_",
            s.old_df().where_(col("ok")),
            s.new_df().filter("ok"),
        ),
        (
            "sort",
            s.old_df().sort(exprs(&order)),
            s.new_df().sort(order.clone()),
        ),
        (
            "order_by",
            s.old_df().order_by(exprs(&order)),
            s.new_df().sort(order.clone()),
        ),
        (
            "sort_within_partitions",
            s.old_df().sort_within_partitions(exprs(&order)),
            s.new_df().sort_within_partitions(order.clone()),
        ),
        ("limit", s.old_df().limit(3), s.new_df().limit(3)),
        ("offset", s.old_df().offset(3), s.new_df().offset(3)),
        ("tail", s.old_df().tail(3), s.new_df().tail(3)),
        ("distinct", s.old_df().distinct(), s.new_df().distinct()),
        (
            "drop_duplicates of every column",
            s.old_df().drop_duplicates(None),
            s.new_df().drop_duplicates(Vec::<String>::new()),
        ),
        (
            "drop_duplicates of columns",
            s.old_df().drop_duplicates(Some(vec!["a", "b"])),
            s.new_df().drop_duplicates(["a", "b"]),
        ),
        (
            "drop_duplicates_within_watermark of every column",
            s.old_df().drop_duplicates_within_watermark(None),
            s.new_df()
                .drop_duplicates_within_watermark(Vec::<String>::new()),
        ),
        (
            "drop_duplicates_within_watermark of columns",
            s.old_df().drop_duplicates_within_watermark(Some(vec!["a"])),
            s.new_df().drop_duplicates_within_watermark(["a"]),
        ),
        (
            "with_watermark",
            s.old_df().with_watermark("time", "10 minutes"),
            s.new_df().with_watermark("time", "10 minutes"),
        ),
    ]);
}

#[tokio::test]
async fn joins_have_the_plans_of_the_old_api() {
    let s = Sessions::connect().await;
    let types = [
        (OldJoinType::Inner, JoinType::Inner),
        (OldJoinType::LeftOuter, JoinType::LeftOuter),
        (OldJoinType::RightOuter, JoinType::RightOuter),
        (OldJoinType::FullOuter, JoinType::FullOuter),
        (OldJoinType::LeftSemi, JoinType::LeftSemi),
        (OldJoinType::LeftAnti, JoinType::LeftAnti),
        (OldJoinType::Cross, JoinType::Cross),
    ];
    let mut cases: Vec<Case> = types
        .into_iter()
        .map(|(old_type, new_type)| {
            let (old, old_right) = (s.old_df(), s.old_right());
            let (new, new_right) = (s.new_df(), s.new_right());
            let old_on = old.col("id").eq(old_right.col("id"));
            let new_on = new.col("id").eq(new_right.col("id"));
            (
                "join of each type",
                old.join(&old_right, Some(old_on), old_type),
                new.join(&new_right, new_on, new_type),
            )
        })
        .collect();
    let old_as_of = AsOfJoinOptions {
        on: Some(col("ok")),
        join_type: "leftouter".to_string(),
        tolerance: Some(spark_connect::lit(5)),
        allow_exact_matches: false,
        direction: "forward".to_string(),
        ..AsOfJoinOptions::new(col("t"), col("u"))
    };
    let new_as_of = AsOfJoin::new("t", "u")
        .on("ok")
        .left_outer()
        .tolerance(lit(5))
        .exclude_exact_matches()
        .direction(AsOfDirection::Forward);
    let old_using = AsOfJoinOptions {
        using_columns: vec!["k".to_string()],
        direction: "nearest".to_string(),
        ..AsOfJoinOptions::new(col("t"), col("u"))
    };
    let new_using = AsOfJoin::new("t", "u")
        .using(["k"])
        .direction(AsOfDirection::Nearest);
    let old_nearest = |d: &Old, r: &Old, mode, direction, join_type| {
        d.nearest_by_join(r, col("distance"), 3, mode, direction, join_type)
    };
    cases.extend([
        (
            "join_using",
            s.old_df()
                .join_using(&s.old_right(), ["id"], OldJoinType::LeftOuter),
            s.new_df()
                .join_using(&s.new_right(), ["id"], JoinType::LeftOuter),
        ),
        (
            "join without a condition",
            s.old_df().join(&s.old_right(), None, OldJoinType::Inner),
            s.new_df()
                .join_using(&s.new_right(), Vec::<String>::new(), JoinType::Inner),
        ),
        (
            "cross_join",
            s.old_df().cross_join(&s.old_right()),
            s.new_df().cross_join(&s.new_right()),
        ),
        (
            "lateral_join on a condition",
            s.old_df()
                .lateral_join(&s.old_right(), Some(col("ok")), OldJoinType::LeftOuter),
            s.new_df()
                .lateral_join(&s.new_right(), Some(col("ok")), JoinType::LeftOuter),
        ),
        (
            "lateral_join without a condition",
            s.old_df()
                .lateral_join(&s.old_right(), None, OldJoinType::Inner),
            s.new_df()
                .lateral_join(&s.new_right(), None, JoinType::Inner),
        ),
        (
            "as_of_join by default",
            s.old_df()
                .as_of_join(&s.old_right(), AsOfJoinOptions::new(col("t"), col("u")))
                .unwrap(),
            s.new_df()
                .as_of_join(&s.new_right(), AsOfJoin::new("t", "u")),
        ),
        (
            "as_of_join of every option",
            s.old_df().as_of_join(&s.old_right(), old_as_of).unwrap(),
            s.new_df().as_of_join(&s.new_right(), new_as_of),
        ),
        (
            "as_of_join using columns",
            s.old_df().as_of_join(&s.old_right(), old_using).unwrap(),
            s.new_df().as_of_join(&s.new_right(), new_using),
        ),
        (
            "nearest_by_join by default",
            old_nearest(&s.old_df(), &s.old_right(), "approx", "distance", "inner"),
            s.new_df()
                .nearest_by_join(&s.new_right(), NearestByJoin::new("distance", 3)),
        ),
        (
            "nearest_by_join of every option",
            old_nearest(
                &s.old_df(),
                &s.old_right(),
                "exact",
                "similarity",
                "leftouter",
            ),
            s.new_df().nearest_by_join(
                &s.new_right(),
                NearestByJoin::new("distance", 3)
                    .exact()
                    .by_similarity()
                    .left_outer(),
            ),
        ),
        (
            "zip",
            s.old_df().zip(&s.old_right()).unwrap(),
            s.new_df().zip(&s.new_right()),
        ),
    ]);
    assert_same_plans(cases);
}

#[tokio::test]
async fn set_operations_have_the_plans_of_the_old_api() {
    let s = Sessions::connect().await;
    let (old, new) = (s.old_right(), s.new_right());
    assert_same_plans(vec![
        ("union", s.old_df().union(&old), s.new_df().union(&new)),
        (
            "union_all",
            s.old_df().union_all(&old),
            s.new_df().union(&new),
        ),
        (
            "union_by_name",
            s.old_df().union_by_name(&old),
            s.new_df().union_by_name(&new, UnionByName::default()),
        ),
        (
            "union_by_name allowing missing columns",
            s.old_df().union_by_name_opt(&old, true),
            s.new_df()
                .union_by_name(&new, UnionByName::default().allow_missing_columns()),
        ),
        (
            "intersect",
            s.old_df().intersect(&old),
            s.new_df().intersect(&new),
        ),
        (
            "intersect_all",
            s.old_df().intersect_all(&old),
            s.new_df().intersect_all(&new),
        ),
        (
            "subtract",
            s.old_df().subtract(&old),
            s.new_df().subtract(&new),
        ),
        (
            "except_all",
            s.old_df().except_all(&old),
            s.new_df().except_all(&new),
        ),
    ]);
}

#[tokio::test]
async fn reshaping_and_statistics_have_the_plans_of_the_old_api() {
    let s = Sessions::connect().await;
    let metrics = [spark_connect::functions::count(col("id"))];
    assert_same_plans(vec![
        (
            "unpivot of values",
            s.old_df()
                .unpivot(["id"], Some(["a", "b"]), "variable", "value"),
            s.new_df().unpivot(["id"], ["a", "b"], "variable", "value"),
        ),
        (
            "unpivot of every other column",
            s.old_df()
                .unpivot(["id"], None::<Vec<Column>>, "variable", "value"),
            s.new_df()
                .unpivot(["id"], Vec::<Column>::new(), "variable", "value"),
        ),
        (
            "melt",
            s.old_df()
                .melt(vec!["id"], Some(vec!["a"]), "variable", "value"),
            s.new_df().unpivot(["id"], ["a"], "variable", "value"),
        ),
        (
            "transpose",
            s.old_df().transpose().unwrap(),
            s.new_df().transpose(None),
        ),
        (
            "transpose of an index",
            s.old_df().transpose_with_index(col("k")).unwrap(),
            s.new_df().transpose(Some(col("k"))),
        ),
        (
            "describe",
            s.old_df().describe(vec!["a", "b"]),
            s.new_df().describe(["a", "b"]),
        ),
        (
            "describe of every column",
            s.old_df().describe(vec![]),
            s.new_df().describe(Vec::<String>::new()),
        ),
        (
            "summary",
            s.old_df().summary(vec!["count", "25%"]),
            s.new_df().summary(["count", "25%"]),
        ),
        (
            "observe",
            s.old_df().observe("metrics", exprs(&metrics)),
            s.new_df().observe("metrics", metrics.clone()),
        ),
    ]);
}

#[tokio::test]
async fn samples_partitionings_and_hints_have_the_plans_of_the_old_api() {
    let s = Sessions::connect().await;
    let old_parts = s.old_df().random_split(vec![1.0, 3.0], Some(7)).unwrap();
    let new_parts = s
        .new_df()
        .random_split(RandomSplit::try_new([1.0, 3.0]).unwrap().seed(7));
    let keys = [col("a"), col("b").desc()];
    assert_same_plans(vec![
        (
            "sample",
            s.old_df().sample(0.1, Some(7)),
            s.new_df().sample(Sample::fraction(0.1).seed(7)),
        ),
        (
            "sample with replacement",
            s.old_df().sample_opt(1.5, true, Some(3)),
            s.new_df()
                .sample(Sample::fraction(1.5).with_replacement().seed(3)),
        ),
        (
            "random_split",
            old_parts[0].union(&old_parts[1]),
            new_parts[0].union(&new_parts[1]),
        ),
        (
            "repartition",
            s.old_df().repartition(8),
            s.new_df().repartition(8),
        ),
        (
            "repartition by hash",
            s.old_df().repartition_by_expressions(0, exprs(&keys)),
            s.new_df().repartition(Partitioning::hash(keys.clone())),
        ),
        (
            "repartition by hash into partitions",
            s.old_df().repartition_by_expressions(8, exprs(&keys)),
            s.new_df()
                .repartition(Partitioning::hash(keys.clone()).partitions(8)),
        ),
        (
            "repartition by range",
            s.old_df().repartition_by_range_opt(None, exprs(&keys)),
            s.new_df().repartition(Partitioning::range(keys.clone())),
        ),
        (
            "repartition by range into partitions",
            s.old_df().repartition_by_range(4, exprs(&keys)),
            s.new_df()
                .repartition(Partitioning::range(keys.clone()).partitions(4)),
        ),
        (
            "repartition by id",
            s.old_df().repartition_by_id(5, col("p")),
            s.new_df()
                .repartition(Partitioning::by_id("p").partitions(5)),
        ),
        ("coalesce", s.old_df().coalesce(2), s.new_df().coalesce(2)),
        (
            "broadcast",
            s.old_df().broadcast(),
            s.new_df().hint("broadcast", Vec::<i32>::new()),
        ),
        (
            "hint of an integer and a string",
            s.old_df().hint("rebalance", ["8", "a"]),
            s.new_df().hint("rebalance", [lit(8), lit("a")]),
        ),
        (
            "hint of a column",
            s.old_df()
                .hint_expressions("repartition", exprs(&[col("a")])),
            s.new_df().hint("repartition", [col("a")]),
        ),
    ]);
}

#[tokio::test]
async fn subqueries_and_table_arguments_have_the_plans_of_the_old_api() {
    let s = Sessions::connect().await;
    let (old, new) = (s.old_right(), s.new_right());
    let old_table = |table: OldTableArg| s.old.tvf().explode(&table.to_column()).unwrap();
    let new_table = |table: Column| s.new.tvf().explode(table);
    assert_same_plans(vec![
        (
            "scalar",
            s.old_df().select([old.scalar()]),
            s.new_df().select([new.scalar()]),
        ),
        (
            "exists",
            s.old_df().filter(old.exists()),
            s.new_df().filter(new.exists()),
        ),
        (
            "isin of a DataFrame",
            s.old_df().filter(col("id").isin_subquery(&old)),
            s.new_df().filter(col("id").isin_subquery(&new)),
        ),
        (
            "a table argument",
            old_table(OldTableArg::new(old.clone())),
            new_table(new.as_table().into()),
        ),
        (
            "a partitioned and ordered table argument",
            old_table(
                OldTableArg::new(old.clone())
                    .partition_by(["k"])
                    .unwrap()
                    .order_by([col("v").desc()])
                    .unwrap(),
            ),
            new_table(
                new.as_table()
                    .partition_by(["k"])
                    .order_by([col("v").desc()])
                    .into(),
            ),
        ),
        (
            "a table argument of a single partition ordered twice",
            old_table(
                OldTableArg::new(old.clone())
                    .with_single_partition()
                    .unwrap()
                    .order_by(["a"])
                    .unwrap()
                    .order_by(["b"])
                    .unwrap(),
            ),
            new_table(
                new.as_table()
                    .with_single_partition()
                    .order_by(["a"])
                    .order_by(["b"])
                    .into(),
            ),
        ),
    ]);
}

#[tokio::test]
async fn ranges_tables_and_table_functions_have_the_plans_of_the_old_api() {
    let s = Sessions::connect().await;
    let (old, new) = (&s.old, &s.new);
    let old_tvf = old.tvf();
    let new_tvf = new.tvf();
    let a = col("a");
    let fields = [
        spark_connect::lit_string("x"),
        spark_connect::lit_string("y"),
    ];
    assert_same_plans(vec![
        ("range of an end", old.range(10).unwrap(), new.range(10)),
        (
            "range of an i64 end",
            old.range(10).unwrap(),
            new.range(10i64),
        ),
        (
            "range of a range",
            old.range_full(5, 10, 1, None).unwrap(),
            new.range(5..10),
        ),
        (
            "range of a step and partitions",
            old.range_full(0, 100, 3, Some(4)).unwrap(),
            new.range(RangeSpec::new(0..100).step(3).partitions(4)),
        ),
        (
            "tvf range",
            old_tvf.range(0, Some(10), 1, None).unwrap(),
            new.range(0..10),
        ),
        ("table", old.table("db.t").unwrap(), new.table("db.t")),
        (
            "call",
            old_tvf.explode(&a).unwrap(),
            new_tvf.call("explode", [a.clone()]),
        ),
        (
            "explode",
            old_tvf.explode(&a).unwrap(),
            new_tvf.explode("a"),
        ),
        (
            "explode_outer",
            old_tvf.explode_outer(&a).unwrap(),
            new_tvf.explode_outer("a"),
        ),
        (
            "posexplode",
            old_tvf.posexplode(&a).unwrap(),
            new_tvf.posexplode("a"),
        ),
        (
            "posexplode_outer",
            old_tvf.posexplode_outer(&a).unwrap(),
            new_tvf.posexplode_outer("a"),
        ),
        ("inline", old_tvf.inline(&a).unwrap(), new_tvf.inline("a")),
        (
            "inline_outer",
            old_tvf.inline_outer(&a).unwrap(),
            new_tvf.inline_outer("a"),
        ),
        (
            "json_tuple",
            old_tvf.json_tuple(&a, fields).unwrap(),
            new_tvf.json_tuple("a", ["x", "y"]),
        ),
        (
            "stack",
            old_tvf
                .stack(&spark_connect::lit(2), ["a", "b", "c", "d"])
                .unwrap(),
            new_tvf.stack(2, ["a", "b", "c", "d"]),
        ),
        (
            "variant_explode",
            old_tvf.variant_explode(&a).unwrap(),
            new_tvf.variant_explode("a"),
        ),
        (
            "variant_explode_outer",
            old_tvf.variant_explode_outer(&a).unwrap(),
            new_tvf.variant_explode_outer("a"),
        ),
        (
            "collations",
            old_tvf.collations().unwrap(),
            new_tvf.collations(),
        ),
        (
            "sql_keywords",
            old_tvf.sql_keywords().unwrap(),
            new_tvf.sql_keywords(),
        ),
    ]);
}

#[tokio::test]
async fn a_sample_without_a_seed_chooses_one_that_each_request_sends() {
    let s = Sessions::connect().await;
    let seed = |df: &DataFrame| match df.to_relation().rel_type {
        Some(spark_connect_proto::relation::RelType::Sample(sample)) => sample.seed,
        other => panic!("not a sample: {other:?}"),
    };
    let sample = s.new_df().sample(0.1);
    let other = s.new_df().sample(0.1);
    assert!(seed(&sample).is_some());
    assert_eq!(seed(&sample), seed(&sample));
    assert_ne!(seed(&sample), seed(&other));
}

#[test]
fn weights_of_a_split_that_are_negative_or_add_up_to_zero_are_rejected() {
    let cases: [(&[f64], &str, &str); 4] = [
        (&[1.0, -1.0], "weights", "-1"),
        (&[0.0, 0.0], "sum(weights)", "0"),
        (&[], "sum(weights)", "0"),
        (&[-0.5, 2.0], "weights", "-0.5"),
    ];
    for (weights, name, value) in cases {
        let error = RandomSplit::try_new(weights.iter().copied()).unwrap_err();
        assert_eq!(
            error.kind(),
            spark_connect_async::ErrorKind::InvalidArgument,
            "{weights:?}"
        );
        assert_eq!(
            error.error_class(),
            Some("VALUE_NOT_POSITIVE"),
            "{weights:?}"
        );
        let parameters: HashMap<_, _> = error.message_parameters().into_iter().collect();
        assert_eq!(parameters["arg_name"], name, "{weights:?}");
        assert_eq!(parameters["arg_value"], value, "{weights:?}");
    }
    assert!(RandomSplit::try_new([0.0, 1.0]).is_ok());
}
