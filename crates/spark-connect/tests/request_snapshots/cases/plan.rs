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

use spark_connect::column::{col, lit};
use spark_connect::expression::{Expression, LiteralExpression, SortOrder};
use spark_connect::plan::{self, AggregateGroupType, JoinType, LogicalPlan, SetOpType};
use spark_connect::row::Value;

use super::schema;
use crate::Case;

fn base() -> LogicalPlan {
    plan::range(0, 10, 1)
}

fn id() -> Expression {
    Expression::column("id")
}

pub fn cases() -> Vec<Case> {
    cases![
        "plan::range" => |c| {
            c.plan(&plan::range(1, 9, 2));
            Ok(())
        },
        "plan::range_with_partitions" => |c| {
            c.plan(&plan::range_with_partitions(0, 4, 1, 3));
            Ok(())
        },
        "plan::sql" => |c| {
            c.plan(&plan::sql("SELECT 1"));
            Ok(())
        },
        "plan::project" => |c| {
            c.plan(&plan::project(base(), [col("id"), lit(1)]));
            Ok(())
        },
        "plan::filter" => |c| {
            c.plan(&plan::filter(base(), col("id").gt(lit(2))));
            Ok(())
        },
        "plan::aggregate" => |c| {
            for group_type in [AggregateGroupType::GroupBy, AggregateGroupType::Rollup, AggregateGroupType::Cube] {
                c.plan(&plan::aggregate(base(), group_type, vec![id()], vec![Expression::function("count", vec![id()])]));
            }
            Ok(())
        },
        "plan::aggregate_with_pivot" => |c| {
            let values = vec![Expression::literal(LiteralExpression::int(1)), Expression::literal(LiteralExpression::string("a"))];
            c.plan(&plan::aggregate_with_pivot(base(), AggregateGroupType::GroupBy, vec![id()], vec![Expression::function("sum", vec![id()])], Expression::column("p"), values));
            c.plan(&plan::aggregate_with_pivot(base(), AggregateGroupType::Pivot, vec![], vec![], Expression::column("q"), vec![]));
            Ok(())
        },
        "plan::aggregate_with_grouping_sets" => |c| {
            c.plan(&plan::aggregate_with_grouping_sets(base(), vec![id()], vec![Expression::function("sum", vec![id()])], vec![vec![id()], vec![]]));
            Ok(())
        },
        "plan::join" => |c| {
            for join_type in [JoinType::Inner, JoinType::LeftOuter, JoinType::RightOuter, JoinType::FullOuter, JoinType::LeftSemi, JoinType::LeftAnti, JoinType::Cross] {
                c.plan(&plan::join(base(), plan::sql("SELECT 2"), join_type, Some(col("a").eq(col("b"))), vec![]));
            }
            c.plan(&plan::join(base(), base(), JoinType::Inner, None, vec!["id".to_string()]));
            Ok(())
        },
        "plan::set_operation" => |c| {
            for op in [SetOpType::Union, SetOpType::Intersect, SetOpType::Except] {
                c.plan(&plan::set_operation(base(), base(), op, false, false, false));
            }
            c.plan(&plan::set_operation(base(), base(), SetOpType::Union, true, true, true));
            Ok(())
        },
        "plan::limit" => |c| {
            c.plan(&plan::limit(base(), 3));
            Ok(())
        },
        "plan::offset" => |c| {
            c.plan(&plan::offset(base(), 3));
            Ok(())
        },
        "plan::tail" => |c| {
            c.plan(&plan::tail(base(), 3));
            Ok(())
        },
        "plan::deduplicate" => |c| {
            c.plan(&plan::deduplicate(base(), true, vec![], false));
            c.plan(&plan::deduplicate(base(), false, vec!["id".to_string()], true));
            Ok(())
        },
        "plan::sort" => |c| {
            let order = vec![Expression::new(spark_connect::expression::ExpressionNode::SortOrder(Box::new(SortOrder::desc_nulls_last(id()))))];
            c.plan(&plan::sort(base(), order.clone(), true));
            c.plan(&plan::sort(base(), order, false));
            Ok(())
        },
        "plan::sample" => |c| {
            c.plan(&plan::sample(base(), 0.0, 0.5, false, None));
            c.plan(&plan::sample(base(), 0.1, 0.4, true, Some(5)));
            Ok(())
        },
        "plan::repartition" => |c| {
            c.plan(&plan::repartition(base(), 4, true));
            c.plan(&plan::repartition(base(), 2, false));
            Ok(())
        },
        "plan::repartition_by_expression" => |c| {
            c.plan(&plan::repartition_by_expression(base(), 3, vec![id()]));
            Ok(())
        },
        "plan::with_columns" => |c| {
            c.plan(&plan::with_columns(base(), vec!["a".to_string(), "b".to_string()], vec![lit(1), col("id")]));
            Ok(())
        },
        "plan::with_columns_renamed" => |c| {
            c.plan(&plan::with_columns_renamed(base(), HashMap::from([("id".to_string(), "k".to_string())])));
            Ok(())
        },
        "plan::drop" => |c| {
            c.plan(&plan::drop(base(), vec!["id".to_string()]));
            Ok(())
        },
        "plan::to_df" => |c| {
            c.plan(&plan::to_df(base(), vec!["x".to_string()]));
            Ok(())
        },
        "plan::to_schema" => |c| {
            c.plan(&plan::to_schema(base(), schema()));
            Ok(())
        },
        "plan::hint" => |c| {
            c.plan(&plan::hint(base(), "broadcast", vec![]));
            c.plan(&plan::hint(base(), "repartition", vec!["3".to_string()]));
            Ok(())
        },
        "plan::unpivot" => |c| {
            c.plan(&plan::unpivot(base(), vec![col("id")], Some(vec![col("a"), col("b")]), "var", "val"));
            c.plan(&plan::unpivot(base(), vec![col("id")], None, "var", "val"));
            Ok(())
        },
        "plan::na_fill" => |c| {
            for value in [Value::Long(1), Value::Double(0.5), Value::String("s".into()), Value::Bool(true), Value::Integer(2)] {
                c.plan(&plan::na_fill(base(), value, vec!["id".to_string()]));
            }
            c.plan(&plan::na_fill(base(), Value::Long(0), vec![]));
            Ok(())
        },
        "plan::na_drop" => |c| {
            c.plan(&plan::na_drop(base(), "any", None, vec![]));
            c.plan(&plan::na_drop(base(), "all", Some(2), vec!["id".to_string()]));
            Ok(())
        },
        "plan::na_replace" => |c| {
            c.plan(&plan::na_replace(base(), vec![("a".to_string(), "b".to_string())], vec!["id".to_string()]));
            Ok(())
        },
        "plan::describe" => |c| {
            c.plan(&plan::describe(base(), vec!["id".to_string()]));
            Ok(())
        },
        "plan::summary" => |c| {
            c.plan(&plan::summary(base(), vec!["min".to_string(), "max".to_string()]));
            Ok(())
        },
        "plan::col_regex" => |c| {
            c.plan(&plan::col_regex(base(), "`i.*`"));
            Ok(())
        },
        "plan::subquery_alias" => |c| {
            c.plan(&plan::subquery_alias(base(), "t"));
            Ok(())
        },
        "plan::local_relation" => |c| {
            c.plan(&plan::local_relation(schema(), None));
            c.plan(&plan::local_relation(schema(), Some(vec![1, 2, 3])));
            Ok(())
        },
        "plan::cached_remote_relation" => |c| {
            c.plan(&plan::cached_remote_relation("relation-1"));
            Ok(())
        },
        "LogicalPlan::to_proto" => |c| {
            c.message("Relation", &plan::limit(plan::filter(base(), col("id").gt(lit(1))), 2).to_proto());
            Ok(())
        },
    ]
}
