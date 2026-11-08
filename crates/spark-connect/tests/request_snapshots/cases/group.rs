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

use spark_connect::column::{col, lit, Column};
use spark_connect::expression::{Expression, LiteralExpression};
use spark_connect::functions as f;
use spark_connect::row::Value;
use spark_connect::udf::eval_type;

use super::{other, range, schema, udf};
use crate::Case;

fn e(column: Column) -> Expression {
    column.expression().clone()
}

pub fn cases() -> Vec<Case> {
    cases![
        "GroupedData::agg" => |c| {
            c.df(&range(c).group_by(["a"]).agg(vec![e(f::max(col("v"))), e(f::min(col("v")).alias("m"))]))
        },
        "GroupedData::count" => |c| c.df(&range(c).group_by(["a"]).count()),
        "GroupedData::sum" => |c| c.df(&range(c).group_by(["a"]).sum(vec!["v", "w"])),
        "GroupedData::sum/all" => |c| c.df(&range(c).group_by(["a"]).sum(Vec::new())),
        "GroupedData::avg" => |c| c.df(&range(c).group_by(["a"]).avg(vec!["v"])),
        "GroupedData::min" => |c| c.df(&range(c).group_by(["a"]).min(vec!["v"])),
        "GroupedData::max" => |c| c.df(&range(c).group_by(["a"]).max(vec!["v"])),
        "GroupedData::mean" => |c| c.df(&range(c).group_by(["a"]).mean(vec!["v"])),
        "GroupedData::numeric_agg" => {
            |c| c.df(&range(c).group_by(["a"]).numeric_agg("sum", vec!["value"])?)
        },
        "GroupedData::numeric_agg/all" => {
            |c| c.df(&range(c).group_by(["name"]).numeric_agg("max", Vec::new())?)
        },
        "GroupedData::pivot" => {
            |c| c.df(&range(c).group_by(["a"]).pivot(col("p"), None)?.sum(vec!["v"]))
        },
        "GroupedData::pivot/values" => |c| {
            let values = Some(vec![Value::String("x".into()), Value::Long(2)]);
            c.df(&range(c).group_by(["a"]).pivot(col("p"), values)?.count())
        },
        "GroupedData::rollup_pivot" => {
            |c| c.df(&range(c).rollup(["a"]).pivot(col("p"), None)?.count())
        },
        "GroupedData::input_columns" => |c| range(c).group_by(["a"]).input_columns().map(drop),
        "GroupedData::spark_session" => {
            |c| c.df(&range(c).group_by(["a"]).spark_session().range(1)?)
        },
        "GroupedData::apply_in_pandas" => {
            |c| c.df(&range(c).group_by(["a"]).apply_in_pandas(udf(eval_type::SQL_GROUPED_MAP_PANDAS_UDF)))
        },
        "GroupedData::apply_in_arrow" => {
            |c| c.df(&range(c).group_by(["a"]).apply_in_arrow(udf(eval_type::SQL_GROUPED_MAP_ARROW_UDF)))
        },
        "GroupedData::apply_in_pandas_with_state" => |c| {
            c.df(&range(c).group_by(["a"]).apply_in_pandas_with_state(
                udf(eval_type::SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE),
                schema(),
                "append",
                "NoTimeout",
            ))
        },
        "GroupedData::transform_with_state" => |c| {
            c.df(&range(c).group_by(["a"]).transform_with_state(
                udf(eval_type::SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF),
                "append",
                "None",
                None,
                None,
            ))
        },
        "GroupedData::transform_with_state/initial_state" => |c| {
            let initial = other(c).group_by(["a"]);
            c.df(&range(c).group_by(["a"]).transform_with_state(
                udf(eval_type::SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_INIT_STATE_UDF),
                "update",
                "EventTime",
                Some("ts"),
                Some(&initial),
            ))
        },
        "GroupedData::transform_with_state_in_pandas" => |c| {
            c.df(&range(c).group_by(["a"]).transform_with_state_in_pandas(
                udf(eval_type::SQL_TRANSFORM_WITH_STATE_PANDAS_UDF),
                schema(),
                "append",
                "ProcessingTime",
                None,
                None,
            ))
        },
        "GroupedData::transform_with_state_in_pandas/initial_state" => |c| {
            let initial = other(c).group_by(["a"]);
            c.df(&range(c).group_by(["a"]).transform_with_state_in_pandas(
                udf(eval_type::SQL_TRANSFORM_WITH_STATE_PANDAS_INIT_STATE_UDF),
                schema(),
                "complete",
                "EventTime",
                Some("ts"),
                Some(&initial),
            ))
        },
        "GroupedData::cogroup" => |c| {
            let left = range(c).group_by(["a"]);
            let right = other(c).group_by(["a"]);
            c.df(&left.cogroup(&right).apply_in_pandas(udf(eval_type::SQL_COGROUPED_MAP_PANDAS_UDF)))
        },
        "CoGroupedData::apply_in_arrow" => |c| {
            let left = range(c).group_by(["a"]);
            let right = other(c).group_by(["b"]);
            c.df(&left.cogroup(&right).apply_in_arrow(udf(eval_type::SQL_COGROUPED_MAP_ARROW_UDF)))
        },
        "CoGroupedData::input_columns" => |c| {
            let left = range(c).group_by(["a"]);
            left.cogroup(&other(c).group_by(["a"])).input_columns().map(drop)
        },
        "CoGroupedData::spark_session" => |c| {
            let left = range(c).group_by(["a"]);
            c.df(&left.cogroup(&other(c).group_by(["a"])).spark_session().range(1)?)
        },
        "StatFunctions::crosstab" => |c| c.df(&range(c).stat().crosstab("a", "b")),
        "StatFunctions::freq_items" => {
            |c| c.df(&range(c).stat().freq_items(vec!["a", "b"], 0.2))
        },
        "StatFunctions::approx_quantile" => {
            |c| c.df(&range(c).stat().approx_quantile(vec!["a"], vec![0.25, 0.5], 0.01))
        },
        "StatFunctions::corr" => |c| range(c).stat().corr("a", "b").map(drop),
        "StatFunctions::cov" => |c| range(c).stat().cov("a", "b").map(drop),
        "StatFunctions::sample_by" => |c| {
            let fractions = vec![
                (Expression::literal(LiteralExpression::int(0)), 0.1),
                (Expression::literal(LiteralExpression::string("x")), 0.5),
            ];
            c.df(&range(c).stat().sample_by("k", fractions, Some(9)))
        },
        "StatFunctions::sample_by/no_seed" => {
            |c| c.df(&range(c).stat().sample_by(col("k"), vec![(e(lit(1)), 1.0)], None))
        },
        "NaFunctions::drop" => {
            |c| c.df(&range(c).na().drop(Some("all"), Some(1), Some(vec!["a"])))
        },
        "NaFunctions::drop/default" => |c| c.df(&range(c).na().drop(None, None, None)),
        "NaFunctions::fill" => |c| c.df(&range(c).na().fill(3, Some(vec!["a"]))),
        "NaFunctions::replace" => {
            |c| c.df(&range(c).na().replace(vec![("x".to_string(), "y".to_string())], None))
        },
        "NaFunctions::replace_literals" => |c| {
            c.df(&range(c).na().replace_literals(
                vec![(LiteralExpression::boolean(true), LiteralExpression::boolean(false))],
                None,
            ))
        },
        "GroupedData::check_pivot" => |c| {
            for grouped in [range(c).group_by(["a"]), range(c).rollup(["a"]), range(c).group_by(["a"]).pivot(col("p"), None)?] {
                c.message("Text", &format!("{:?}", grouped.check_pivot().map_err(|e| e.to_string())));
            }
            Ok(())
        },
        "GroupedData::grouping_columns" => |c| {
            c.columns(range(c).group_by(["a", "b"]).grouping_columns());
            Ok(())
        },
        "GroupedData::group_type" => |c| {
            c.message("Text", &format!("{:?} {:?}", range(c).group_by(["a"]).group_type(), range(c).cube(["a"]).group_type()));
            Ok(())
        },
        "GroupedData::dataframe" => |c| c.df(range(c).group_by(["a"]).dataframe()),
    ]
}
