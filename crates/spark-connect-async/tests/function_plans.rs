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

//! The functions of the async API build the expressions of the functions of
//! `apache-spark-connect`, one case per function of the old API.

#[path = "function_plans/cases.rs"]
mod cases;
mod support;

use spark_connect::functions as old;
use spark_connect::{Column as OldColumn, SparkSession};
use spark_connect_async::functions::*;
use spark_connect_async::{Column, Session};
use spark_connect_core::expression::{
    Expression, ExpressionNode, LambdaFunction, LiteralExpression, UnresolvedFunction,
    UnresolvedNamedLambdaVariable,
};
use spark_connect_core::types::DataType;

pub struct Case {
    pub name: &'static str,
    pub old: fn() -> OldColumn,
    pub new: fn() -> Column,
}

fn oc(name: &str) -> OldColumn {
    spark_connect::col(name)
}

fn literal(literal: LiteralExpression) -> OldColumn {
    OldColumn::new(Expression::literal(literal))
}

fn ostr(value: &str) -> OldColumn {
    literal(LiteralExpression::string(value))
}

fn oint(value: i32) -> OldColumn {
    literal(LiteralExpression::int(value))
}

fn olong(value: i64) -> OldColumn {
    literal(LiteralExpression::long(value))
}

fn odouble(value: f64) -> OldColumn {
    literal(LiteralExpression::double(value))
}

fn obool(value: bool) -> OldColumn {
    literal(LiteralExpression::boolean(value))
}

fn odecimal_zero() -> OldColumn {
    literal(LiteralExpression::Decimal {
        value: "0".to_string(),
        precision: 10,
        scale: 0,
    })
}

fn onull() -> OldColumn {
    literal(LiteralExpression::null(DataType::Null))
}

/// The expression of `column`, with plan ids and the numbers of lambda variables numbered in
/// the order of their first appearance: both come from counters of the process.
fn sent(column: impl Into<OldColumn>) -> String {
    support::numbered_after(&support::renumbered(&column.into().to_proto()), "\"x_")
}

fn differences(cases: Vec<Case>) -> String {
    support::differences(
        cases
            .into_iter()
            .map(|case| (case.name, sent((case.old)()), sent((case.new)())))
            .collect(),
    )
}

#[test]
fn every_function_builds_the_expression_of_the_old_api() {
    let differences = differences(cases::cases());
    assert!(differences.is_empty(), "{differences}");
}

#[test]
fn every_argument_of_a_function_with_optional_arguments_is_sent_in_order() {
    let differences = differences(cases::forms());
    assert!(differences.is_empty(), "{differences}");
}

#[tokio::test]
async fn broadcast_marks_the_dataframe_as_the_old_api_does() {
    const URL: &str = "sc://localhost:1";
    let old_session = SparkSession::builder().remote(URL).get_or_create().unwrap();
    let new_session = Session::builder().remote(URL).connect().await.unwrap();
    let old = old::broadcast(old_session.range(10).unwrap()).to_relation();
    let new = broadcast(&new_session.range(10)).to_relation();
    assert_eq!(support::renumbered(&old), support::renumbered(&new));
}

fn expressions(args: Vec<OldColumn>) -> Vec<Expression> {
    args.into_iter()
        .map(|arg| arg.expression().clone())
        .collect()
}

fn call(name: &str, args: Vec<OldColumn>) -> OldColumn {
    OldColumn::new(Expression::function(name, expressions(args)))
}

/// `array_sort(a, (x, y) -> array(x, y))`.
fn array_sort_with_a_comparator() -> OldColumn {
    let variables = [
        UnresolvedNamedLambdaVariable::new("x_1"),
        UnresolvedNamedLambdaVariable::new("x_2"),
    ];
    let body = call(
        "array",
        variables
            .iter()
            .map(|v| {
                OldColumn::new(ExpressionNode::UnresolvedNamedLambdaVariable(v.clone()).into())
            })
            .collect(),
    );
    let lambda = LambdaFunction::new(body.expression().clone(), variables.to_vec());
    OldColumn::new(Expression::function(
        "array_sort",
        vec![
            oc("a").expression().clone(),
            ExpressionNode::LambdaFunction(Box::new(lambda)).into(),
        ],
    ))
}

fn distinct_call(name: &str, args: Vec<OldColumn>) -> OldColumn {
    OldColumn::new(
        ExpressionNode::UnresolvedFunction(UnresolvedFunction::new_distinct(
            name,
            expressions(args),
        ))
        .into(),
    )
}

#[test]
fn a_separator_of_concat_ws_is_a_string() {
    assert_eq!(
        sent(concat_ws("-", ["a"])),
        sent(call("concat_ws", vec![ostr("-"), oc("a")])),
    );
}

#[test]
fn functions_send_the_arguments_given_and_the_defaults_of_the_others() {
    let cases: Vec<(&str, OldColumn, Column)> = vec![
        (
            "printf of its format alone",
            call("printf", vec![oc("f")]),
            printf("f", Vec::<Column>::new()),
        ),
        (
            "printf of columns",
            call("printf", vec![ostr("%s-%s"), oc("a"), oc("b")]),
            printf(lit("%s-%s"), ["a", "b"]),
        ),
        (
            "greatest of three",
            call("greatest", vec![oc("a"), oc("b"), oc("c")]),
            greatest(["a", "b", "c"]),
        ),
        (
            "least of three",
            call("least", vec![oc("a"), oc("b"), oc("c")]),
            least(["a", "b", "c"]),
        ),
        (
            "listagg with a delimiter",
            call("listagg", vec![oc("a"), ostr(",")]),
            listagg("a", Some(",")),
        ),
        (
            "string_agg with a delimiter",
            call("string_agg", vec![oc("a"), ostr(",")]),
            string_agg("a", Some(",")),
        ),
        (
            "sort_array descending",
            call("sort_array", vec![oc("a"), obool(false)]),
            sort_array("a", Some(false)),
        ),
        (
            "str_to_map with a key-value delimiter",
            call("str_to_map", vec![oc("a"), ostr(","), ostr("=")]),
            str_to_map("a", None, Some(lit("="))),
        ),
        (
            "sentences with a language",
            call("sentences", vec![oc("a"), ostr("en"), ostr("")]),
            sentences("a", Some(lit("en")), None),
        ),
        (
            "sentences with a language and a country",
            call("sentences", vec![oc("a"), ostr("en"), ostr("US")]),
            sentences("a", Some(lit("en")), Some(lit("US"))),
        ),
        (
            "randstr with a seed",
            call("randstr", vec![oint(5), olong(7)]),
            randstr(5, Some(7)),
        ),
        (
            "window with a slide",
            call(
                "window",
                vec![oc("t"), ostr("10 minutes"), ostr("5 minutes")],
            ),
            window("t", "10 minutes", Some("5 minutes"), None),
        ),
        (
            "window with a start",
            call(
                "window",
                vec![
                    oc("t"),
                    ostr("10 minutes"),
                    ostr("10 minutes"),
                    ostr("1 minute"),
                ],
            ),
            window("t", "10 minutes", None, Some("1 minute")),
        ),
        (
            "lag with an offset",
            call("lag", vec![oc("a"), oint(2)]),
            lag("a", Some(2), None),
        ),
        (
            "lead with a default",
            call("lead", vec![oc("a"), oint(1), oint(0)]),
            lead("a", None, Some(lit(0))),
        ),
        (
            "count_distinct of two columns",
            distinct_call("count", vec![oc("a"), oc("b")]),
            count_distinct(["a", "b"]),
        ),
        (
            "count_distinct of no column",
            distinct_call("count", vec![]),
            count_distinct(Vec::<Column>::new()),
        ),
        (
            "listagg without a delimiter",
            call("listagg", vec![oc("a")]),
            listagg("a", None),
        ),
        (
            "sort_array ascending by default",
            call("sort_array", vec![oc("a"), obool(true)]),
            sort_array("a", None),
        ),
        (
            "str_to_map with its default delimiters",
            call("str_to_map", vec![oc("a"), ostr(","), ostr(":")]),
            str_to_map("a", None, None),
        ),
        (
            "sentences of the default locale",
            call("sentences", vec![oc("a"), ostr(""), ostr("")]),
            sentences("a", None, None),
        ),
        (
            "randstr with the fixed seed",
            call("randstr", vec![oint(5), olong(1000462829105445681)]),
            randstr(5, None),
        ),
        (
            "window of a duration",
            call("window", vec![oc("t"), ostr("10 minutes")]),
            window("t", "10 minutes", None, None),
        ),
        (
            "window with a slide and a start",
            call(
                "window",
                vec![
                    oc("t"),
                    ostr("10 minutes"),
                    ostr("5 minutes"),
                    ostr("1 minute"),
                ],
            ),
            window("t", "10 minutes", Some("5 minutes"), Some("1 minute")),
        ),
        (
            "instr with an occurrence and the start 1",
            call("instr", vec![oc("a"), ostr("x"), oint(1), oint(2)]),
            instr("a", "x", None, Some(2)),
        ),
        (
            "make_dt_interval of seconds",
            call("make_dt_interval", vec![oint(0), oint(0), oint(0), oc("s")]),
            make_dt_interval(None, None, None, Some(col("s"))),
        ),
        (
            "make_interval of seconds",
            call(
                "make_interval",
                vec![
                    oint(0),
                    oint(0),
                    oint(0),
                    oint(0),
                    oint(0),
                    oint(0),
                    oc("s"),
                ],
            ),
            make_interval(None, None, None, None, None, None, Some(col("s"))),
        ),
        (
            "try_make_interval of weeks",
            call(
                "try_make_interval",
                vec![
                    oint(0),
                    oint(0),
                    oc("w"),
                    oint(0),
                    oint(0),
                    oint(0),
                    odecimal_zero(),
                ],
            ),
            try_make_interval(None, None, Some(col("w")), None, None, None, None),
        ),
        (
            "make_ym_interval of months",
            call("make_ym_interval", vec![oint(0), oc("m")]),
            make_ym_interval(None, Some(col("m"))),
        ),
        (
            "mask of digits",
            call(
                "mask",
                vec![oc("a"), ostr("X"), ostr("x"), oc("d"), onull()],
            ),
            mask("a", None, None, Some(col("d")), None),
        ),
        (
            "aes_encrypt with an aad",
            call(
                "aes_encrypt",
                vec![
                    oc("i"),
                    oc("k"),
                    ostr("GCM"),
                    ostr("DEFAULT"),
                    ostr(""),
                    oc("t"),
                ],
            ),
            aes_encrypt("i", "k", None, None, None, Some(col("t"))),
        ),
        (
            "aes_decrypt with an aad",
            call(
                "aes_decrypt",
                vec![oc("i"), oc("k"), ostr("GCM"), ostr("DEFAULT"), oc("t")],
            ),
            aes_decrypt("i", "k", None, None, Some(col("t"))),
        ),
        (
            "try_aes_decrypt with a padding",
            call(
                "try_aes_decrypt",
                vec![oc("i"), oc("k"), ostr("GCM"), oc("p"), ostr("")],
            ),
            try_aes_decrypt("i", "k", None, Some(col("p")), None),
        ),
        (
            "tuple_sketch_agg_double with a mode",
            call(
                "tuple_sketch_agg_double",
                vec![oc("k"), oc("s"), oint(12), ostr("max")],
            ),
            tuple_sketch_agg_double("k", "s", None, Some("max")),
        ),
        (
            "tuple_sketch_agg_integer with a mode",
            call(
                "tuple_sketch_agg_integer",
                vec![oc("k"), oc("s"), oint(12), ostr("max")],
            ),
            tuple_sketch_agg_integer("k", "s", None, Some("max")),
        ),
        (
            "tuple_union_agg_double with a mode",
            call(
                "tuple_union_agg_double",
                vec![oc("a"), oint(12), ostr("max")],
            ),
            tuple_union_agg_double("a", None, Some("max")),
        ),
        (
            "tuple_union_agg_integer with a mode",
            call(
                "tuple_union_agg_integer",
                vec![oc("a"), oint(12), ostr("max")],
            ),
            tuple_union_agg_integer("a", None, Some("max")),
        ),
        (
            "tuple_union_double with a mode",
            call(
                "tuple_union_double",
                vec![oc("a"), oc("b"), oint(12), ostr("max")],
            ),
            tuple_union_double("a", "b", None, Some("max")),
        ),
        (
            "tuple_union_integer with a mode",
            call(
                "tuple_union_integer",
                vec![oc("a"), oc("b"), oint(12), ostr("max")],
            ),
            tuple_union_integer("a", "b", None, Some("max")),
        ),
        (
            "tuple_union_theta_double with a mode",
            call(
                "tuple_union_theta_double",
                vec![oc("a"), oc("b"), oint(12), ostr("max")],
            ),
            tuple_union_theta_double("a", "b", None, Some("max")),
        ),
        (
            "tuple_union_theta_integer with a mode",
            call(
                "tuple_union_theta_integer",
                vec![oc("a"), oc("b"), oint(12), ostr("max")],
            ),
            tuple_union_theta_integer("a", "b", None, Some("max")),
        ),
        (
            "tuple_union_agg_double with entries alone",
            call("tuple_union_agg_double", vec![oc("a"), oint(10)]),
            tuple_union_agg_double("a", Some(10), None),
        ),
        (
            "unix_timestamp of a timestamp",
            call("unix_timestamp", vec![oc("t"), ostr("yyyy-MM-dd HH:mm:ss")]),
            unix_timestamp(Some(col("t")), None),
        ),
        (
            "unix_timestamp of a format alone",
            call("unix_timestamp", vec![]),
            unix_timestamp(None, Some("yyyy")),
        ),
        (
            "log in a base",
            call("log", vec![oc("b"), oc("x")]),
            log("b", Some(col("x"))),
        ),
        (
            "log without a base",
            call("ln", vec![oc("x")]),
            log("x", None),
        ),
        (
            "reduce with a finish",
            old::aggregate_with_finish(
                oc("a"),
                oc("b"),
                |v: Vec<OldColumn>| old::array(v),
                |v: Vec<OldColumn>| old::array(v),
            ),
            reduce(
                "a",
                "b",
                |x, y| array([x, y]),
                Some(Box::new(|x| array([x]))),
            ),
        ),
        (
            "array_sort with a comparator",
            array_sort_with_a_comparator(),
            array_sort("a", Some(Box::new(|x, y| array([x, y])))),
        ),
        (
            "lag with the offset 1",
            call("lag", vec![oc("a"), oint(1)]),
            lag("a", None, None),
        ),
    ];
    let differences = support::differences(
        cases
            .into_iter()
            .map(|(name, old, new)| (name, sent(old), sent(new)))
            .collect(),
    );
    assert!(differences.is_empty(), "{differences}");
}
