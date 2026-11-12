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

//! The columns of the async API build the expressions of the columns of `apache-spark-connect`
//! for the same operations, and convert into them and back.

mod support;

use std::collections::BTreeMap;

use spark_connect::expression::SortOrder;
use spark_connect::functions::when as old_when;
use spark_connect::window::WindowSpec as OldWindow;
use spark_connect::{col as oc, lit as ol, Column as Old};
use spark_connect_async::column::{AliasSpec, CaseWhen, IntoArgument};
use spark_connect_async::types::DataType;
use spark_connect_async::window::{FrameBound, Window};
use spark_connect_async::{col, lit, when, Column};

fn expr(name: &str) -> spark_connect::Expression {
    oc(name).expression().clone()
}

fn sent(column: Column) -> spark_connect_proto::Expression {
    Old::from(column).to_proto()
}

#[test]
fn every_column_method_builds_the_expression_of_the_old_api() {
    let metadata = BTreeMap::from([("comment".to_string(), "the key".to_string())]);
    let cases: Vec<(&str, Old, Column)> = vec![
        ("alias", oc("a").alias("b"), col("a").alias("b")),
        (
            "alias of a String",
            oc("a").alias("b"),
            col("a").alias("b".to_string()),
        ),
        (
            "name",
            oc("a").name("b"),
            col("a").alias(AliasSpec::new("b")),
        ),
        (
            "aliases",
            oc("m").aliases(["k", "v"]),
            col("m").alias(["k", "v"]),
        ),
        (
            "aliases of a Vec",
            oc("m").aliases(["k", "v"]),
            col("m").alias(vec!["k", "v"]),
        ),
        (
            "aliases of a spec",
            oc("m").aliases(["k", "v"]),
            col("m").alias(AliasSpec::names(["k", "v"])),
        ),
        (
            "alias with metadata",
            oc("a").alias_with_metadata("b", metadata),
            col("a").alias(AliasSpec::new("b").metadata(r#"{"comment":"the key"}"#)),
        ),
        (
            "cast",
            oc("a").cast(DataType::Integer),
            col("a").cast(DataType::Integer),
        ),
        (
            "astype",
            oc("a").astype(DataType::Integer),
            col("a").cast(DataType::Integer),
        ),
        ("cast_str", oc("a").cast_str("int"), col("a").cast("int")),
        (
            "cast of a String",
            oc("a").cast_str("int"),
            col("a").cast("int".to_string()),
        ),
        (
            "try_cast",
            oc("a").try_cast(DataType::Long),
            col("a").try_cast(DataType::Long),
        ),
        (
            "try_cast_str",
            oc("a").try_cast_str("bigint"),
            col("a").try_cast("bigint"),
        ),
        (
            "try_cast of a String",
            oc("a").try_cast_str("bigint"),
            col("a").try_cast("bigint".to_string()),
        ),
        ("is_null", oc("a").is_null(), col("a").is_null()),
        ("is_not_null", oc("a").is_not_null(), col("a").is_not_null()),
        ("is_nan", oc("a").is_nan(), col("a").is_nan()),
        ("eq", oc("a").eq(ol(1)), col("a").eq(lit(1))),
        ("eq of a name", oc("a").eq(oc("b")), col("a").eq("b")),
        ("ne", oc("a").ne(ol(1)), col("a").ne(lit(1))),
        ("gt", oc("a").gt(ol(1)), col("a").gt(lit(1))),
        ("lt", oc("a").lt(ol(1)), col("a").lt(lit(1))),
        ("ge", oc("a").ge(ol(1)), col("a").ge(lit(1))),
        ("le", oc("a").le(ol(1)), col("a").le(lit(1))),
        (
            "eq_null_safe",
            oc("a").eq_null_safe(oc("b")),
            col("a").eq_null_safe("b"),
        ),
        (
            "between",
            oc("a").between(ol(1), ol(9)),
            col("a").between(lit(1), lit(9)),
        ),
        (
            "isin",
            oc("a").isin([ol(1), ol(2)]),
            col("a").is_in([lit(1), lit(2)]),
        ),
        (
            "isin of names",
            oc("a").isin(["b", "c"]),
            col("a").is_in(["b", "c"]),
        ),
        ("and", oc("a").and(oc("b")), col("a").and("b")),
        ("or", oc("a").or(oc("b")), col("a").or("b")),
        ("&", oc("a").and(oc("b")), col("a") & col("b")),
        ("|", oc("a").or(oc("b")), col("a") | "b"),
        ("not", oc("a").not(), !col("a")),
        (
            "bitwise_and",
            oc("a").bitwise_and(oc("b")),
            col("a").bitwise_and("b"),
        ),
        (
            "bitwise_or",
            oc("a").bitwise_or(oc("b")),
            col("a").bitwise_or("b"),
        ),
        (
            "bitwise_xor",
            oc("a").bitwise_xor(oc("b")),
            col("a").bitwise_xor("b"),
        ),
        ("+", oc("a").add(ol(1)), col("a") + lit(1)),
        ("+ of a name", oc("a").add(oc("b")), col("a") + "b"),
        ("-", oc("a").sub(oc("b")), col("a") - col("b")),
        ("*", oc("a").mul(oc("b")), col("a") * col("b")),
        ("/", oc("a").div(oc("b")), col("a") / col("b")),
        ("%", oc("a").modulo(oc("b")), col("a") % col("b")),
        ("negation", oc("a").neg(), -col("a")),
        ("like", oc("a").like("x%"), col("a").like("x%")),
        ("ilike", oc("a").ilike("x%"), col("a").ilike("x%")),
        ("rlike", oc("a").rlike("^x"), col("a").rlike("^x")),
        (
            "contains",
            oc("a").contains(spark_connect::lit_string("x")),
            col("a").contains(lit("x")),
        ),
        (
            "startswith",
            oc("a").startswith(spark_connect::lit_string("x")),
            col("a").starts_with(lit("x")),
        ),
        (
            "endswith",
            oc("a").endswith(oc("b")),
            col("a").ends_with("b"),
        ),
        (
            "substr",
            oc("a").substr(ol(1), ol(3)),
            col("a").substr(lit(1), lit(3)),
        ),
        ("get_field", oc("s").get_field("f"), col("s").get_field("f")),
        (
            "get_item",
            oc("m").get_item(ol(0)),
            col("m").get_item(lit(0)),
        ),
        (
            "with_field",
            oc("s").with_field("f", ol(1)),
            col("s").with_field("f", lit(1)),
        ),
        (
            "drop_fields",
            oc("s").drop_fields(vec!["f", "g"]),
            col("s").drop_fields(["f", "g"]),
        ),
        ("asc", oc("a").asc(), col("a").asc()),
        (
            "asc_nulls_first",
            oc("a").asc_nulls_first(),
            col("a").asc_nulls_first(),
        ),
        (
            "asc_nulls_last",
            oc("a").asc_nulls_last(),
            col("a").asc_nulls_last(),
        ),
        ("desc", oc("a").desc(), col("a").desc()),
        (
            "desc_nulls_first",
            oc("a").desc_nulls_first(),
            col("a").desc_nulls_first(),
        ),
        (
            "desc_nulls_last",
            oc("a").desc_nulls_last(),
            col("a").desc_nulls_last(),
        ),
        (
            "when, when, otherwise",
            old_when(oc("a").lt(ol(1)), ol(10))
                .when(oc("a").lt(ol(2)), ol(20))
                .otherwise(ol(30)),
            when(col("a").lt(lit(1)), lit(10))
                .when(col("a").lt(lit(2)), lit(20))
                .otherwise(lit(30)),
        ),
        (
            "when, when",
            old_when(oc("a").lt(ol(1)), ol(10)).when(oc("a").lt(ol(2)), ol(20)),
            when(col("a").lt(lit(1)), lit(10))
                .when(col("a").lt(lit(2)), lit(20))
                .into(),
        ),
        (
            "when, otherwise",
            old_when(oc("c"), ol(1)).otherwise(ol(2)),
            when("c", lit(1)).otherwise(lit(2)),
        ),
        (
            "when as an argument",
            old_when(oc("c"), ol(1)),
            when("c", lit(1)).into_argument(),
        ),
        (
            "over a partition",
            oc("x").over(OldWindow::new().partition_by(vec![expr("a"), expr("b")])),
            col("x").over(Window::new().partition_by(["a"]).partition_by([col("b")])),
        ),
        (
            "over an order",
            oc("x").over(OldWindow::new().order_by(vec![
                SortOrder::asc_nulls_first(expr("b")),
                SortOrder::desc_nulls_last(expr("c")),
            ])),
            col("x").over(Window::new().order_by([col("b"), col("c").desc()])),
        ),
        (
            "over rows",
            oc("x").over(
                OldWindow::new()
                    .partition_by(vec![expr("a")])
                    .order_by(vec![SortOrder::asc_nulls_last(expr("b"))])
                    .rows_between(FrameBound::Preceding(1), FrameBound::Following(2)),
            ),
            col("x").over(
                Window::new()
                    .partition_by(["a"])
                    .order_by([col("b").asc_nulls_last()])
                    .rows_between(FrameBound::Preceding(1), FrameBound::Following(2)),
            ),
        ),
        (
            "over a range",
            oc("x").over(
                OldWindow::new()
                    .order_by(vec![SortOrder::asc_nulls_first(expr("b"))])
                    .range_between(FrameBound::UnboundedPreceding, FrameBound::CurrentRow),
            ),
            col("x").over(
                Window::new()
                    .order_by(["b"])
                    .range_between(FrameBound::UnboundedPreceding, FrameBound::CurrentRow),
            ),
        ),
    ];
    let cases = cases
        .into_iter()
        .map(|(name, old, new)| (name, old.to_proto(), sent(new)))
        .collect();
    let differences = support::differences(cases);
    assert!(differences.is_empty(), "{differences}");
}

#[test]
fn columns_convert_into_and_from_the_columns_of_the_old_api() {
    let old = oc("a").gt(ol(1));
    let new: Column = old.clone().into();
    assert_eq!(Old::from(new.clone()), old);
    let back: Column = Old::from(new.clone()).into();
    assert_eq!(sent(back), sent(new));
    let name = "a".to_string();
    for (case, column) in [
        ("&str", Column::from("a")),
        ("String", Column::from(name.clone())),
        ("&String", Column::from(&name)),
    ] {
        assert_eq!(sent(column), oc("a").to_proto(), "{case}");
    }
}

#[test]
fn columns_and_alias_specs_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Column>();
    assert_send_sync::<CaseWhen>();
    assert_send_sync::<AliasSpec>();
}
