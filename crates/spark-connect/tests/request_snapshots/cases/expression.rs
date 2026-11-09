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

use spark_connect::column::col;
use spark_connect::expression::{
    Alias, CallFunctionWrapper, CaseWhen, Cast, CastEvalMode, ColumnReference, Expression,
    ExpressionNode, ExtractValue, FrameBoundary, LambdaFunction, LiteralExpression, NamedArgument,
    SortOrder, UnresolvedFunction, UnresolvedNamedLambdaVariable, UpdateFieldsExpr,
    WindowExpressionWrapper,
};
use spark_connect::types::DataType;

use super::batch_udf;
use crate::{Case, Ctx};

fn lit(value: i32) -> Expression {
    Expression::literal(LiteralExpression::int(value))
}

fn record(c: &Ctx, node: ExpressionNode) {
    let expression = Expression::new(node);
    c.expression(&expression);
    c.message("Text", &expression.render());
}

fn window() -> WindowExpressionWrapper {
    WindowExpressionWrapper::new(
        Expression::function("rank", vec![]),
        vec![Expression::column("p")],
        vec![SortOrder::desc_nulls_first(Expression::column("o"))],
        Some((1, FrameBoundary::Preceding(2), FrameBoundary::Following(3))),
    )
}

pub fn cases() -> Vec<Case> {
    cases![
        "Expression::literal" => |c| {
            for value in [
                LiteralExpression::null(DataType::Integer),
                LiteralExpression::int(1),
                LiteralExpression::long(2),
                LiteralExpression::double(0.25),
                LiteralExpression::string("s"),
                LiteralExpression::boolean(false),
                LiteralExpression::binary(vec![0, 255]),
            ] {
                c.message("Text", &value.render());
                c.message("Literal", &value.to_literal_proto());
                c.message("Expression", &value.to_proto());
                record(c, ExpressionNode::Literal(value));
            }
            Ok(())
        },
        "Expression::column" => |c| {
            c.expression(&Expression::column("a.b"));
            Ok(())
        },
        "Expression::function" => |c| {
            c.expression(&Expression::function("f", vec![lit(1), Expression::column("x")]));
            Ok(())
        },
        "Expression::node" => |c| {
            let expression = Expression::column("a");
            assert!(matches!(expression.node(), ExpressionNode::ColumnReference(_)));
            assert!(expression.origin().is_none());
            c.expression(&Expression::new(expression.node().clone()));
            Ok(())
        },
        "Expression::<From>" => |c| {
            c.expression(&Expression::from(ExpressionNode::SQLExpression("a + 1".to_string())));
            Ok(())
        },
        "ExpressionNode::ColumnReference" => |c| {
            record(c, ExpressionNode::ColumnReference(ColumnReference::new("a")));
            record(c, ExpressionNode::ColumnReference(ColumnReference::new("a").with_plan_id(4)));
            record(c, ExpressionNode::ColumnReference(ColumnReference::new("m").metadata()));
            c.message("Expression", &ColumnReference::new("z").to_proto());
            Ok(())
        },
        "ExpressionNode::UnresolvedFunction" => |c| {
            record(c, ExpressionNode::UnresolvedFunction(UnresolvedFunction::new("f", vec![lit(1)])));
            let distinct = UnresolvedFunction::new_distinct("count", vec![Expression::column("a")]);
            c.message("Text", &distinct.render());
            c.message("Expression", &distinct.to_proto());
            record(c, ExpressionNode::UnresolvedFunction(distinct));
            Ok(())
        },
        "ExpressionNode::UnresolvedStar" => |c| {
            record(c, ExpressionNode::UnresolvedStar(None, None));
            record(c, ExpressionNode::UnresolvedStar(Some("t.*".to_string()), Some(3)));
            Ok(())
        },
        "ExpressionNode::Alias" => |c| {
            let alias = Alias::new(lit(1), "one");
            c.message("Expression", &alias.to_proto());
            record(c, ExpressionNode::Alias(Box::new(alias)));
            let names = Alias::with_names(Expression::column("a"), ["x", "y"]).with_metadata("{\"k\":1}".to_string());
            record(c, ExpressionNode::Alias(Box::new(names)));
            Ok(())
        },
        "ExpressionNode::Cast" => |c| {
            let cast = Cast::new(Expression::column("a"), DataType::Long);
            c.message("Expression", &cast.to_proto());
            record(c, ExpressionNode::Cast(Box::new(cast)));
            for mode in [CastEvalMode::Legacy, CastEvalMode::Ansi, CastEvalMode::Try] {
                let cast = Cast::new_str(Expression::column("a"), "int").with_eval_mode(mode);
                record(c, ExpressionNode::Cast(Box::new(cast)));
            }
            Ok(())
        },
        "ExpressionNode::DirectShufflePartitionId" => |c| {
            record(c, ExpressionNode::DirectShufflePartitionId(Expression::column("p")));
            Ok(())
        },
        "ExpressionNode::UnresolvedRegex" => |c| {
            record(c, ExpressionNode::UnresolvedRegex("`a.*`".to_string(), None));
            record(c, ExpressionNode::UnresolvedRegex("`b.*`".to_string(), Some(2)));
            Ok(())
        },
        "ExpressionNode::SortOrder" => |c| {
            for sort in [
                SortOrder::asc_nulls_first(Expression::column("a")),
                SortOrder::asc_nulls_last(Expression::column("a")),
                SortOrder::desc_nulls_first(Expression::column("a")),
                SortOrder::desc_nulls_last(Expression::column("a")),
            ] {
                c.message("Text", &sort.render());
                c.message("SortOrder", &sort.to_proto_sort_order());
                c.message("Expression", &sort.to_proto());
                record(c, ExpressionNode::SortOrder(Box::new(sort)));
            }
            Ok(())
        },
        "ExpressionNode::CaseWhen" => |c| {
            let case = CaseWhen::new(vec![(Expression::column("a"), lit(1)), (Expression::column("b"), lit(2))]);
            c.message("Text", &case.render());
            c.message("Expression", &case.to_proto());
            record(c, ExpressionNode::CaseWhen(Box::new(case.with_else(lit(0)))));
            Ok(())
        },
        "ExpressionNode::UnresolvedExtractValue" => |c| {
            let extract = ExtractValue::new(Expression::column("m"), lit(1));
            c.message("Expression", &extract.to_proto());
            record(c, ExpressionNode::UnresolvedExtractValue(Box::new(extract)));
            Ok(())
        },
        "ExpressionNode::UpdateFields" => |c| {
            let update = UpdateFieldsExpr::new(Expression::column("s"), "f", Some(lit(1)));
            c.message("Expression", &update.to_proto());
            record(c, ExpressionNode::UpdateFields(Box::new(update)));
            record(c, ExpressionNode::UpdateFields(Box::new(UpdateFieldsExpr::new(Expression::column("s"), "g", None))));
            Ok(())
        },
        "ExpressionNode::SQLExpression" => |c| {
            record(c, ExpressionNode::SQLExpression("a + b".to_string()));
            Ok(())
        },
        "ExpressionNode::CallFunction" => |c| {
            let call = CallFunctionWrapper::new("my_fn", vec![lit(1), Expression::column("a")]);
            c.message("Text", &call.render());
            c.message("Expression", &call.to_proto());
            record(c, ExpressionNode::CallFunction(Box::new(call)));
            Ok(())
        },
        "ExpressionNode::WindowExpression" => |c| {
            let window = window();
            c.message("Text", &window.render());
            c.message("Expression", &window.to_proto());
            record(c, ExpressionNode::WindowExpression(Box::new(window)));
            let unbounded = WindowExpressionWrapper::new(
                Expression::function("sum", vec![Expression::column("v")]),
                vec![],
                vec![],
                Some((2, FrameBoundary::UnboundedPreceding, FrameBoundary::UnboundedFollowing)),
            );
            record(c, ExpressionNode::WindowExpression(Box::new(unbounded)));
            let current = WindowExpressionWrapper::new(Expression::function("rank", vec![]), vec![], vec![], Some((1, FrameBoundary::CurrentRow, FrameBoundary::CurrentRow)));
            record(c, ExpressionNode::WindowExpression(Box::new(current)));
            assert_eq!(FrameBoundary::Preceding(2).value(), -2);
            c.message("Text", &spark_connect::expression::window_spec_repr(
                &[Expression::column("p")],
                &[SortOrder::asc_nulls_first(Expression::column("o"))],
                Some(&(1, FrameBoundary::Preceding(1), FrameBoundary::CurrentRow)),
            ));
            Ok(())
        },
        "ExpressionNode::LambdaFunction" => |c| {
            let x = UnresolvedNamedLambdaVariable::new("x");
            c.message("Expression", &x.to_proto());
            let lambda = LambdaFunction::new(
                Expression::function("+", vec![Expression::new(ExpressionNode::UnresolvedNamedLambdaVariable(x.clone())), lit(1)]),
                vec![x.clone()],
            );
            c.message("Expression", &lambda.to_proto());
            record(c, ExpressionNode::LambdaFunction(Box::new(lambda)));
            record(c, ExpressionNode::UnresolvedNamedLambdaVariable(x));
            Ok(())
        },
        "ExpressionNode::CommonInlineUserDefinedFunction" => |c| {
            record(c, ExpressionNode::CommonInlineUserDefinedFunction(Box::new(batch_udf())));
            Ok(())
        },
        "ExpressionNode::NamedArgument" => |c| {
            record(c, ExpressionNode::NamedArgument(Box::new(NamedArgument::new("k", lit(1)))));
            Ok(())
        },
        "Expression::render" => |c| {
            c.message("Text", &(col("a") + col("b").alias("c")).expression().render());
            Ok(())
        },
    ]
}
