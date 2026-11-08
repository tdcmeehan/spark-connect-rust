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

use spark_connect::column::{col, lit, lit_string, Column};
use spark_connect::expression::{Expression, LiteralExpression};
use spark_connect::functions as f;
use spark_connect::observation::Observation;
use spark_connect::plan::JoinType;
use spark_connect::row::Value;
use spark_connect::types::DataType;
use spark_connect::udf::eval_type;
use spark_connect::StorageLevel;

use super::{other, range, udf};
use crate::Case;

fn e(column: Column) -> Expression {
    column.expression().clone()
}

fn disk_only() -> StorageLevel {
    StorageLevel {
        use_disk: true,
        use_memory: false,
        use_off_heap: false,
        deserialized: false,
        replication: 2,
    }
}

pub fn cases() -> Vec<Case> {
    let mut cases = cases![
        "DataFrame::select" => |c| c.df(&range(c).select(["id"])),
        "DataFrame::select/columns" => {
            |c| c.df(&range(c).select([col("id").alias("x"), lit(1) + col("id")]))
        },
        "DataFrame::filter" => |c| c.df(&range(c).filter(col("id").gt(lit(3)))),
        "DataFrame::where_" => |c| c.df(&range(c).where_(col("id").lt(lit(3)))),
        "DataFrame::with_column" => |c| c.df(&range(c).with_column("y", col("id") * lit(2))),
        "DataFrame::with_columns" => |c| {
            c.df(&range(c).with_columns(vec![
                ("a".to_string(), lit(1)),
                ("b".to_string(), col("id")),
            ]))
        },
        "DataFrame::with_column_renamed" => {
            |c| c.df(&range(c).with_column_renamed("id", "key"))
        },
        // One rename: the plan keeps renames in a HashMap, whose order differs per process.
        "DataFrame::with_columns_renamed" => {
            |c| c.df(&range(c).with_columns_renamed(vec![("id".to_string(), "key".to_string())]))
        },
        "DataFrame::drop" => |c| c.df(&range(c).drop(vec!["id", "name"])),
        "DataFrame::drop_columns" => {
            |c| c.df(&range(c).drop_columns(vec![col("a")], vec!["b", "c"]))
        },
        "DataFrame::limit" => |c| c.df(&range(c).limit(3)),
        "DataFrame::offset" => |c| c.df(&range(c).offset(2)),
        "DataFrame::tail" => |c| c.df(&range(c).tail(4)),
        "DataFrame::distinct" => |c| c.df(&range(c).distinct()),
        "DataFrame::drop_duplicates" => |c| c.df(&range(c).drop_duplicates(None)),
        "DataFrame::drop_duplicates/subset" => {
            |c| c.df(&range(c).drop_duplicates(Some(vec!["id"])))
        },
        "DataFrame::sort" => {
            |c| c.df(&range(c).sort(vec![e(col("id").desc()), e(col("name"))]))
        },
        "DataFrame::order_by" => |c| c.df(&range(c).order_by(vec![e(col("id").asc())])),
        "DataFrame::join/inner" => {
            |c| c.df(&range(c).join(&other(c), Some(col("id").eq(col("oid"))), JoinType::Inner))
        },
        "DataFrame::join/left_outer" => {
            |c| c.df(&range(c).join(&other(c), Some(col("id").eq(col("oid"))), JoinType::LeftOuter))
        },
        "DataFrame::join/right_outer" => {
            |c| c.df(&range(c).join(&other(c), None, JoinType::RightOuter))
        },
        "DataFrame::join/full_outer" => {
            |c| c.df(&range(c).join(&other(c), None, JoinType::FullOuter))
        },
        "DataFrame::join/left_semi" => {
            |c| c.df(&range(c).join(&other(c), None, JoinType::LeftSemi))
        },
        "DataFrame::join/left_anti" => {
            |c| c.df(&range(c).join(&other(c), None, JoinType::LeftAnti))
        },
        "DataFrame::join/cross" => |c| c.df(&range(c).join(&other(c), None, JoinType::Cross)),
        "DataFrame::join/self" => |c| {
            let df = range(c);
            c.df(&df.join(&df, Some(df.col("id").eq(df.col("id"))), JoinType::Inner))
        },
        "DataFrame::join_using" => {
            |c| c.df(&range(c).join_using(&other(c), ["id", "name"], JoinType::Inner))
        },
        "DataFrame::nearest_by_join" => |c| {
            c.df(&range(c).nearest_by_join(&other(c), col("d"), 3, "exact", "nearest", "inner"))
        },
        "DataFrame::cross_join" => |c| c.df(&range(c).cross_join(&other(c))),
        "DataFrame::lateral_join" => {
            |c| c.df(&range(c).lateral_join(&other(c), Some(col("a").eq(col("b"))), JoinType::Inner))
        },
        "DataFrame::lateral_join/left_outer" => {
            |c| c.df(&range(c).lateral_join(&other(c), None, JoinType::LeftOuter))
        },
        "DataFrame::union" => |c| c.df(&range(c).union(&other(c))),
        "DataFrame::union_all" => |c| c.df(&range(c).union_all(&other(c))),
        "DataFrame::union_by_name" => |c| c.df(&range(c).union_by_name(&other(c))),
        "DataFrame::union_by_name_opt" => {
            |c| c.df(&range(c).union_by_name_opt(&other(c), true))
        },
        "DataFrame::intersect" => |c| c.df(&range(c).intersect(&other(c))),
        "DataFrame::intersect_all" => |c| c.df(&range(c).intersect_all(&other(c))),
        "DataFrame::subtract" => |c| c.df(&range(c).subtract(&other(c))),
        "DataFrame::except_all" => |c| c.df(&range(c).except_all(&other(c))),
        "DataFrame::repartition" => |c| c.df(&range(c).repartition(4)),
        "DataFrame::coalesce" => |c| c.df(&range(c).coalesce(2)),
        "DataFrame::hint" => |c| c.df(&range(c).hint("repartition", ["8", "id"])),
        "DataFrame::hint_expressions" => {
            |c| c.df(&range(c).hint_expressions("repartition", vec![e(lit(8)), e(col("id"))]))
        },
        "DataFrame::broadcast" => |c| c.df(&range(c).broadcast()),
        "DataFrame::to_df" => |c| c.df(&range(c).to_df(vec!["a", "b"])),
        "DataFrame::alias" => |c| c.df(&range(c).alias("r")),
        "DataFrame::map_in_pandas" => {
            |c| c.df(&range(c).map_in_pandas(udf(eval_type::SQL_MAP_PANDAS_ITER_UDF), false))
        },
        "DataFrame::map_in_pandas_with_profile" => |c| {
            c.df(&range(c).map_in_pandas_with_profile(
                udf(eval_type::SQL_MAP_PANDAS_ITER_UDF),
                true,
                Some(3),
            ))
        },
        "DataFrame::map_in_arrow" => {
            |c| c.df(&range(c).map_in_arrow(udf(eval_type::SQL_MAP_ARROW_ITER_UDF), true))
        },
        "DataFrame::map_in_arrow_with_profile" => |c| {
            c.df(&range(c).map_in_arrow_with_profile(
                udf(eval_type::SQL_MAP_ARROW_ITER_UDF),
                false,
                None,
            ))
        },
        "DataFrame::foreach" => {
            |c| range(c).foreach(udf(eval_type::SQL_MAP_ARROW_ITER_UDF))
        },
        "DataFrame::foreach_partition" => {
            |c| range(c).foreach_partition(udf(eval_type::SQL_MAP_ARROW_ITER_UDF))
        },
        "DataFrame::sample" => |c| {
            // Without a seed the client draws one at random, as PySpark does: record that it has
            // one, and the relation without it.
            let mut relation = range(c).sample(0.5, None).to_relation();
            let Some(spark_connect_proto::relation::RelType::Sample(sample)) = relation.rel_type.as_mut() else {
                return Err(spark_connect::SparkError::value_msg("not a sample"));
            };
            c.message("Text", &format!("seed drawn: {}", sample.seed.take().is_some()));
            c.message("Relation", &relation);
            Ok(())
        },
        "DataFrame::sample/seed" => |c| c.df(&range(c).sample(0.25, Some(7))),
        "DataFrame::sample_opt" => |c| c.df(&range(c).sample_opt(0.5, true, Some(3))),
        "DataFrame::collect" => |c| range(c).collect().map(drop),
        "DataFrame::collect_with_schema" => |c| range(c).collect_with_schema().map(drop),
        "DataFrame::to_local_iterator" => |c| {
            for row in range(c).to_local_iterator(false)? {
                row?;
            }
            Ok(())
        },
        "DataFrame::to_local_iterator/prefetch" => |c| {
            for row in range(c).to_local_iterator(true)? {
                row?;
            }
            Ok(())
        },
        "DataFrame::execution_info" => |c| {
            let df = range(c);
            df.collect()?;
            df.execution_info().map(drop)
        },
        "DataFrame::collect_record_batches" => {
            |c| range(c).collect_record_batches().map(drop)
        },
        "DataFrame::collect_record_batches_with_schema" => {
            |c| range(c).collect_record_batches_with_schema().map(drop)
        },
        "DataFrame::count" => |c| range(c).count().map(drop),
        "DataFrame::show_string" => |c| range(c).show_string(5, 20, false).map(drop),
        "DataFrame::show_string/vertical" => |c| range(c).show_string(3, 0, true).map(drop),
        "DataFrame::show" => |c| range(c).show(4),
        "DataFrame::schema" => |c| range(c).schema().map(drop),
        "DataFrame::first" => |c| range(c).first().map(drop),
        "DataFrame::first_with_schema" => |c| range(c).first_with_schema().map(drop),
        "DataFrame::head" => |c| range(c).head().map(drop),
        "DataFrame::take" => |c| range(c).take(3).map(drop),
        "DataFrame::take_with_schema" => |c| range(c).take_with_schema(2).map(drop),
        "DataFrame::is_empty" => |c| range(c).is_empty().map(drop),
        "DataFrame::columns" => |c| range(c).columns().map(drop),
        "DataFrame::cache" => |c| range(c).cache().map(drop),
        "DataFrame::persist" => |c| range(c).persist(disk_only()).map(drop),
        "DataFrame::unpersist" => |c| range(c).unpersist(false).map(drop),
        "DataFrame::unpersist/blocking" => |c| range(c).unpersist(true).map(drop),
        "DataFrame::checkpoint" => |c| c.df(&range(c).checkpoint()?),
        "DataFrame::checkpoint_with" => |c| c.df(&range(c).checkpoint_with(false)?),
        "DataFrame::local_checkpoint" => |c| c.df(&range(c).local_checkpoint()?),
        "DataFrame::local_checkpoint_with" => {
            |c| c.df(&range(c).local_checkpoint_with(true, Some(disk_only()))?)
        },
        "DataFrame::local_checkpoint_with/lazy" => {
            |c| c.df(&range(c).local_checkpoint_with(false, None)?)
        },
        "DataFrame::create_temp_view" => |c| range(c).create_temp_view("v"),
        "DataFrame::create_or_replace_temp_view" => {
            |c| range(c).create_or_replace_temp_view("v")
        },
        "DataFrame::create_global_temp_view" => |c| range(c).create_global_temp_view("g"),
        "DataFrame::create_or_replace_global_temp_view" => {
            |c| range(c).create_or_replace_global_temp_view("g")
        },
        "DataFrame::register_temp_table" => |c| range(c).register_temp_table("t"),
        "DataFrame::explain" => |c| range(c).explain(),
        "DataFrame::explain_mode/extended" => |c| range(c).explain_mode("extended"),
        "DataFrame::explain_mode/codegen" => |c| range(c).explain_mode("codegen"),
        "DataFrame::explain_mode/cost" => |c| range(c).explain_mode("cost"),
        "DataFrame::explain_mode/formatted" => |c| range(c).explain_mode("formatted"),
        "DataFrame::explain_string" => |c| range(c).explain_string("simple").map(drop),
        "DataFrame::with_watermark" => {
            |c| c.df(&range(c).with_watermark("ts", "10 minutes"))
        },
        "DataFrame::repartition_by_range" => {
            |c| c.df(&range(c).repartition_by_range(3, vec![e(col("id"))]))
        },
        "DataFrame::repartition_by_range/no_partitions" => {
            |c| c.df(&range(c).repartition_by_range(0, vec![e(col("id").desc())]))
        },
        "DataFrame::repartition_by_expressions" => {
            |c| c.df(&range(c).repartition_by_expressions(5, vec![e(col("id")), e(col("name"))]))
        },
        "DataFrame::repartition_by_id" => |c| c.df(&range(c).repartition_by_id(4, col("pid"))),
        "DataFrame::to_schema" => |c| c.df(&range(c).to_schema(vec!["id", "name"])),
        "DataFrame::to" => {
            |c| c.df(&range(c).to(DataType::from_ddl("id BIGINT, name STRING")?))
        },
        "DataFrame::melt" => {
            |c| c.df(&range(c).melt(vec!["id"], Some(vec!["a", "b"]), "var", "val"))
        },
        "DataFrame::melt/all_values" => |c| c.df(&range(c).melt(vec!["id"], None, "var", "val")),
        "DataFrame::unpivot" => {
            |c| c.df(&range(c).unpivot(["id"], Some(["a", "b"]), "var", "val"))
        },
        "DataFrame::unpivot/all_values" => {
            |c| c.df(&range(c).unpivot(["id"], None::<Vec<Column>>, "var", "val"))
        },
        "DataFrame::input_files" => |c| range(c).input_files().map(drop),
        "DataFrame::observe" => {
            |c| c.df(&range(c).observe("metrics", vec![e(f::count(col("id")).alias("n"))]))
        },
        "DataFrame::observe_with" => |c| {
            let observation = Observation::new("named");
            c.df(&range(c).observe_with(&observation, vec![e(f::max(col("id")))])?)
        },
        "DataFrame::observe_with/unnamed" => |c| {
            let observation = Observation::unnamed();
            let df = range(c).observe_with(&observation, vec![e(f::min(col("id")))])?;
            c.mask(&observation.name().unwrap());
            c.df(&df)
        },
        "DataFrame::agg" => {
            |c| c.df(&range(c).agg(vec![e(f::sum(col("id"))), e(f::avg(col("id")))]))
        },
        "DataFrame::select_expr" => {
            |c| c.df(&range(c).select_expr(vec!["id + 1 AS x", "name"]))
        },
        "DataFrame::fillna" => |c| c.df(&range(c).fillna(0, None)),
        "DataFrame::fillna/subset" => |c| c.df(&range(c).fillna(1, Some(vec!["id"]))),
        "DataFrame::fillna_double" => |c| c.df(&range(c).fillna_double(0.5, None)),
        "DataFrame::fillna_string" => {
            |c| c.df(&range(c).fillna_string("none", Some(vec!["name"])))
        },
        "DataFrame::fillna_bool" => |c| c.df(&range(c).fillna_bool(true, None)),
        "DataFrame::fillna_value" => {
            |c| c.df(&range(c).fillna_value(Value::Integer(3), Some(vec!["a", "b"])))
        },
        "DataFrame::fillna_map" => |c| {
            c.df(&range(c).fillna_map(vec![
                ("a".to_string(), Value::Long(1)),
                ("b".to_string(), Value::String("x".into())),
                ("c".to_string(), Value::Double(2.5)),
                ("d".to_string(), Value::Bool(false)),
            ]))
        },
        "DataFrame::dropna" => |c| c.df(&range(c).dropna(None, None, None)),
        "DataFrame::dropna/all" => {
            |c| c.df(&range(c).dropna(Some("all"), None, Some(vec!["a"])))
        },
        "DataFrame::dropna/thresh" => |c| c.df(&range(c).dropna(Some("any"), Some(2), None)),
        "DataFrame::replace" => {
            |c| c.df(&range(c).replace(vec![("a".to_string(), "b".to_string())], None))
        },
        "DataFrame::replace/subset" => {
            |c| c.df(&range(c).replace(vec![("a".to_string(), "b".to_string())], Some(vec!["name"])))
        },
        "DataFrame::replace_literals" => |c| {
            c.df(&range(c).replace_literals(
                vec![
                    (LiteralExpression::int(1), LiteralExpression::int(2)),
                    (LiteralExpression::double(1.5), LiteralExpression::null(DataType::Double)),
                ],
                Some(vec!["id"]),
            ))
        },
        "DataFrame::describe" => |c| c.df(&range(c).describe(vec!["id"])),
        "DataFrame::describe/all" => |c| c.df(&range(c).describe(Vec::new())),
        "DataFrame::summary" => |c| c.df(&range(c).summary(vec!["count", "50%"])),
        "DataFrame::summary/default" => |c| c.df(&range(c).summary(Vec::new())),
        "DataFrame::col" => |c| {
            let df = range(c);
            c.column(&df.col("id"));
            c.df(&df.select([df.col("id")]))
        },
        "DataFrame::star" => |c| {
            let df = range(c);
            c.df(&df.select([df.star()]))
        },
        "DataFrame::col_regex" => |c| {
            let df = range(c);
            c.df(&df.select([df.col_regex("`i.*`")]))
        },
        "DataFrame::metadata_column" => |c| {
            let df = range(c);
            c.df(&df.select([df.metadata_column("_metadata")]))
        },
        "DataFrame::sort_within_partitions" => {
            |c| c.df(&range(c).sort_within_partitions(vec![e(col("id").desc_nulls_last())]))
        },
        "DataFrame::drop_duplicates_within_watermark" => {
            |c| c.df(&range(c).drop_duplicates_within_watermark(Some(vec!["id"])))
        },
        "DataFrame::drop_duplicates_within_watermark/all" => {
            |c| c.df(&range(c).drop_duplicates_within_watermark(None))
        },
        "DataFrame::transform" => |c| c.df(&range(c).transform(|df| df.limit(2))),
        "DataFrame::random_split" => |c| {
            for df in range(c).random_split(vec![1.0, 2.0, 3.0], Some(11)) {
                c.df(&df)?;
            }
            Ok(())
        },
        "DataFrame::print_schema" => |c| range(c).print_schema(),
        "DataFrame::storage_level" => |c| range(c).storage_level().map(drop),
        "DataFrame::is_cached" => |c| range(c).is_cached().map(drop),
        "DataFrame::dtypes" => |c| range(c).dtypes().map(drop),
        "DataFrame::semantic_hash" => |c| range(c).semantic_hash().map(drop),
        "DataFrame::same_semantics" => |c| range(c).same_semantics(&other(c)).map(drop),
        "DataFrame::to_json_dataframe" => |c| c.df(&range(c).to_json_dataframe()),
        "DataFrame::to_json" => |c| range(c).to_json().map(drop),
        "DataFrame::with_metadata" => {
            |c| c.df(&range(c).with_metadata("id", HashMap::from([("k".to_string(), "v".to_string())])))
        },
        "DataFrame::spark_session" => |c| c.df(&range(c).spark_session().range(2)?),
        "DataFrame::is_local" => |c| {
            let local = c.spark.create_dataframe(Vec::new(), super::schema())?;
            assert!(local.is_local());
            assert!(!range(c).is_local());
            Ok(())
        },
        "DataFrame::is_streaming" => |c| range(c).is_streaming().map(drop),
        "DataFrame::to_arrow" => |c| range(c).to_arrow().map(drop),
        "DataFrame::zip_with_index" => |c| c.df(&range(c).zip_with_index("idx")),
        "DataFrame::scalar" => |c| {
            let df = range(c);
            c.df(&df.select([other(c).agg(vec![e(f::max(col("v")))]).scalar()]))
        },
        "DataFrame::exists" => |c| c.df(&range(c).filter(other(c).exists())),
        "DataFrame::transpose" => |c| c.df(&range(c).transpose()?),
        "DataFrame::transpose_with_index" => {
            |c| c.df(&range(c).transpose_with_index(col("name"))?)
        },
        "DataFrame::zip" => |c| c.df(&range(c).zip(&other(c))?),
        "DataFrame::as_table" => |c| c.df(&range(c).as_table("t")),
        "DataFrame::group_by" => {
            |c| c.df(&range(c).group_by(["name"]).agg(vec![e(f::count(col("id")))]))
        },
        "DataFrame::rollup" => {
            |c| c.df(&range(c).rollup(["a", "b"]).agg(vec![e(f::sum(col("v")))]))
        },
        "DataFrame::cube" => |c| c.df(&range(c).cube([col("a")]).agg(vec![e(f::sum(col("v")))])),
        "DataFrame::grouping_sets" => |c| {
            c.df(&range(c)
                .grouping_sets(vec![vec![col("a")], vec![col("a"), col("b")]])
                .agg(vec![e(f::sum(col("v")))]))
        },
        "DataFrame::grouping_sets_with_columns" => |c| {
            c.df(&range(c)
                .grouping_sets_with_columns(vec![vec![col("a")], Vec::new()], vec![col("a")])
                .count())
        },
        "DataFrame::stat" => |c| c.df(&range(c).stat().crosstab("a", "b")),
        "DataFrame::na" => |c| c.df(&range(c).na().fill(0, None)),
        "DataFrame::write" => |c| range(c).write().save(Some("/tmp/out")),
        "DataFrame::write_to" => |c| range(c).write_to("db.t").append(),
        "DataFrame::merge_into" => {
            |c| range(c).merge_into("db.t", col("a").eq(col("b"))).when_matched(None).delete().merge()
        },
        "DataFrame::write_stream" => {
            |c| range(c).write_stream().format("memory").start("").map(drop)
        },
        "DataFrame::subquery_in_filter" => {
            |c| c.df(&range(c).filter(col("id").isin_subquery(&other(c).select(["oid"]))))
        },
        "DataFrame::shared_plans" => |c| {
            let base = range(c).filter(col("id").gt(lit(1)));
            let left = base.select(["id"]);
            let right = base.select([col("id").alias("other")]);
            c.df(&left.union(&right).union(&left))
        },
        "DataFrame::literals" => {
            |c| c.df(&range(c).select([lit(1), lit_string("s"), f::lit(col("id"))]))
        },
        "DataFrame::html_string" => |c| range(c).html_string(5, 20).map(drop),
        "DataFrame::repartition_by_range_opt" => {
            |c| c.df(&range(c).repartition_by_range_opt(Some(3), vec![e(col("a"))]))
        },
        "DataFrame::repartition_by_range_opt/none" => {
            |c| c.df(&range(c).repartition_by_range_opt(None, vec![e(col("a"))]))
        },
        "DataFrame::with_metadata_json" => |c| c.df(&range(c).with_metadata_json("id", r#"{"k":"v"}"#)),
        "DataFrame::to_relation" => |c| {
            c.message("Relation", &range(c).to_relation());
            Ok(())
        },
        "DataFrame::from_relation" => |c| {
            let relation = range(c).to_relation();
            c.df(&spark_connect::dataframe::DataFrame::from_relation(c.spark.clone(), relation, None))
        },
        "DataFrame::from_sent_relation" => |c| {
            let relation = range(c).to_relation();
            let df = spark_connect::dataframe::DataFrame::from_sent_relation(c.spark.clone(), relation);
            c.df(&df.select([df.col("id")]))
        },
    ];
    cases.extend(feature_cases());
    cases
}

#[cfg(any(feature = "datafusion", feature = "polars"))]
fn feature_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    #[cfg(feature = "datafusion")]
    cases.extend(cases![
        "DataFrame::to_datafusion" => |c| {
            let ctx = datafusion::prelude::SessionContext::new();
            range(c).to_datafusion(&ctx).map(drop)
        },
    ]);
    #[cfg(feature = "polars")]
    cases.extend(cases![
        "DataFrame::to_polars" => |c| range(c).to_polars().map(drop),
    ]);
    cases
}

#[cfg(not(any(feature = "datafusion", feature = "polars")))]
fn feature_cases() -> Vec<Case> {
    Vec::new()
}
