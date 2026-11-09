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

use std::collections::BTreeMap;

use spark_connect::column::{col, lit, lit_boolean, lit_double, lit_string, when, Column};
use spark_connect::expression::{Expression, SortOrder};
use spark_connect::types::DataType;
use spark_connect::window::{FrameBound, Window, WindowSpec};

use super::{other, range};
use crate::Case;

fn e(column: Column) -> Expression {
    column.expression().clone()
}

fn sort(column: &str) -> SortOrder {
    SortOrder::asc_nulls_first(Expression::column(column))
}

pub fn cases() -> Vec<Case> {
    cases![
        "column::col" => |c| {
            c.columns(&[col("a"), col("*"), col("t.*"), col("a.b"), col("`a.b`")]);
            Ok(())
        },
        "column::lit" => |c| {
            c.columns(&[lit(1), lit(i64::MAX), lit(-3)]);
            Ok(())
        },
        "column::lit_string" => |c| {
            c.column(&lit_string("text"));
            Ok(())
        },
        "column::lit_double" => |c| {
            c.columns(&[lit_double(1.5), lit_double(f64::NAN)]);
            Ok(())
        },
        "column::lit_boolean" => |c| {
            c.column(&lit_boolean(true));
            Ok(())
        },
        "column::when" => |c| {
            c.column(&when(col("a").gt(lit(1)), lit(2)).when(col("b"), lit(3)).otherwise(lit(4)));
            c.column(&when(col("a"), lit(1)));
            Ok(())
        },
        "Column::new" => |c| {
            c.column(&Column::new(Expression::column("x")));
            Ok(())
        },
        "Column::expression" => |c| {
            c.expression(col("x").expression());
            Ok(())
        },
        "Column::alias" => |c| {
            c.column(&col("a").alias("b"));
            Ok(())
        },
        "Column::alias_with_metadata" => |c| {
            let metadata = BTreeMap::from([("k".to_string(), "\"v\"".to_string())]);
            c.column(&col("a").alias_with_metadata("b", metadata));
            Ok(())
        },
        "Column::aliases" => |c| {
            c.column(&col("a").aliases(["x", "y"]));
            Ok(())
        },
        "Column::name" => |c| {
            c.column(&col("a").name("n"));
            Ok(())
        },
        "Column::cast" => |c| {
            c.column(&col("a").cast(DataType::Integer));
            c.column(&col("a").cast(DataType::from_ddl("array<string>")?));
            Ok(())
        },
        "Column::astype" => |c| {
            c.column(&col("a").astype(DataType::Double));
            Ok(())
        },
        "Column::cast_str" => |c| {
            c.column(&col("a").cast_str("decimal(10,2)"));
            Ok(())
        },
        "Column::try_cast" => |c| {
            c.column(&col("a").try_cast(DataType::Long));
            Ok(())
        },
        "Column::try_cast_str" => |c| {
            c.column(&col("a").try_cast_str("int"));
            Ok(())
        },
        "Column::is_null" => |c| {
            c.column(&col("a").is_null());
            Ok(())
        },
        "Column::is_not_null" => |c| {
            c.column(&col("a").is_not_null());
            Ok(())
        },
        "Column::substr" => |c| {
            c.column(&col("a").substr(lit(1), lit(3)));
            Ok(())
        },
        "Column::like" => |c| {
            c.column(&col("a").like("x%"));
            Ok(())
        },
        "Column::rlike" => |c| {
            c.column(&col("a").rlike("^x"));
            Ok(())
        },
        "Column::ilike" => |c| {
            c.column(&col("a").ilike("X%"));
            Ok(())
        },
        "Column::contains" => |c| {
            c.column(&col("a").contains(lit_string("x")));
            Ok(())
        },
        "Column::startswith" => |c| {
            c.column(&col("a").startswith(lit_string("x")));
            Ok(())
        },
        "Column::endswith" => |c| {
            c.column(&col("a").endswith(lit_string("x")));
            Ok(())
        },
        "Column::is_nan" => |c| {
            c.column(&col("a").is_nan());
            Ok(())
        },
        "Column::eq_null_safe" => |c| {
            c.column(&col("a").eq_null_safe(col("b")));
            Ok(())
        },
        "Column::bitwise_and" => |c| {
            c.column(&col("a").bitwise_and(col("b")));
            Ok(())
        },
        "Column::bitwise_or" => |c| {
            c.column(&col("a").bitwise_or(col("b")));
            Ok(())
        },
        "Column::bitwise_xor" => |c| {
            c.column(&col("a").bitwise_xor(col("b")));
            Ok(())
        },
        "Column::between" => |c| {
            c.column(&col("a").between(lit(1), lit(5)));
            Ok(())
        },
        "Column::isin" => |c| {
            c.column(&col("a").isin([lit(1), lit(2)]));
            c.column(&col("a").isin(["b", "c"]));
            Ok(())
        },
        "Column::isin_subquery" => |c| {
            c.df(&range(c).filter(col("id").isin_subquery(&other(c))))?;
            let pair = spark_connect::functions::r#struct([col("a"), col("b")]);
            c.df(&range(c).filter(pair.isin_subquery(&other(c).select(["x", "y"]))))
        },
        "Column::with_field" => |c| {
            c.column(&col("s").with_field("f", lit(1)));
            Ok(())
        },
        "Column::drop_fields" => |c| {
            c.column(&col("s").drop_fields(vec!["f", "g"]));
            Ok(())
        },
        "Column::asc" => |c| {
            c.column(&col("a").asc());
            Ok(())
        },
        "Column::asc_nulls_first" => |c| {
            c.column(&col("a").asc_nulls_first());
            Ok(())
        },
        "Column::asc_nulls_last" => |c| {
            c.column(&col("a").asc_nulls_last());
            Ok(())
        },
        "Column::desc" => |c| {
            c.column(&col("a").desc());
            Ok(())
        },
        "Column::desc_nulls_first" => |c| {
            c.column(&col("a").desc_nulls_first());
            Ok(())
        },
        "Column::desc_nulls_last" => |c| {
            c.column(&col("a").desc_nulls_last());
            Ok(())
        },
        "Column::when" => |c| {
            c.column(&when(col("a"), lit(1)).when(col("b"), lit(2)));
            Ok(())
        },
        "Column::otherwise" => |c| {
            c.column(&when(col("a"), lit(1)).otherwise(lit(0)));
            Ok(())
        },
        "Column::get_field" => |c| {
            c.column(&col("s").get_field("f"));
            Ok(())
        },
        "Column::get_item" => |c| {
            c.column(&col("m").get_item(lit_string("k")));
            c.column(&col("arr").get_item(lit(0)));
            Ok(())
        },
        "Column::to_proto" => |c| {
            c.message("Expression", &col("a").to_proto());
            Ok(())
        },
        "Column::over" => |c| {
            let spec = Window::partition_by(vec![e(col("p"))]).order_by(vec![sort("o")]);
            c.column(&spark_connect::functions::row_number().over(spec));
            Ok(())
        },
        "Column::eq" => |c| {
            c.column(&col("a").eq(lit(1)));
            Ok(())
        },
        "Column::ne" => |c| {
            c.column(&col("a").ne(lit(1)));
            Ok(())
        },
        "Column::gt" => |c| {
            c.column(&col("a").gt(lit(1)));
            Ok(())
        },
        "Column::lt" => |c| {
            c.column(&col("a").lt(lit(1)));
            Ok(())
        },
        "Column::ge" => |c| {
            c.column(&col("a").ge(lit(1)));
            Ok(())
        },
        "Column::le" => |c| {
            c.column(&col("a").le(lit(1)));
            Ok(())
        },
        "Column::add" => |c| {
            c.column(&col("a").add(lit(1)));
            Ok(())
        },
        "Column::sub" => |c| {
            c.column(&col("a").sub(lit(1)));
            Ok(())
        },
        "Column::mul" => |c| {
            c.column(&col("a").mul(lit(2)));
            Ok(())
        },
        "Column::div" => |c| {
            c.column(&col("a").div(lit(2)));
            Ok(())
        },
        "Column::modulo" => |c| {
            c.column(&col("a").modulo(lit(2)));
            Ok(())
        },
        "Column::and" => |c| {
            c.column(&col("a").and(col("b")));
            Ok(())
        },
        "Column::or" => |c| {
            c.column(&col("a").or(col("b")));
            Ok(())
        },
        "Column::not" => |c| {
            c.column(&col("a").not());
            Ok(())
        },
        "Column::neg" => |c| {
            c.column(&col("a").neg());
            Ok(())
        },
        "Column::<operators>" => |c| {
            c.columns(&[
                col("a") + col("b"),
                col("a") - col("b"),
                col("a") * col("b"),
                col("a") / col("b"),
                col("a") % col("b"),
                col("a") & col("b"),
                col("a") | col("b"),
                !col("a"),
            ]);
            Ok(())
        },
        "Column::<From>" => |c| {
            let name = "b".to_string();
            c.columns(&[Column::from("a"), Column::from(name.clone()), Column::from(&name)]);
            Ok(())
        },
        "Window::partition_by" => |c| {
            let spec = Window::partition_by(vec![e(col("a")), e(col("b"))]);
            c.column(&spark_connect::functions::rank().over(spec));
            Ok(())
        },
        "Window::order_by" => |c| {
            let spec = Window::order_by(vec![sort("a"), SortOrder::desc_nulls_last(Expression::column("b"))]);
            c.column(&spark_connect::functions::rank().over(spec));
            Ok(())
        },
        "Window::rows_between" => |c| {
            let spec = Window::rows_between(Window::unbounded_preceding(), Window::current_row());
            c.column(&spark_connect::functions::sum(col("v")).over(spec));
            Ok(())
        },
        "Window::range_between" => |c| {
            let spec = Window::range_between(FrameBound::Preceding(3), Window::unbounded_following());
            c.column(&spark_connect::functions::sum(col("v")).over(spec));
            Ok(())
        },
        "WindowSpec::new" => |c| {
            c.column(&spark_connect::functions::rank().over(WindowSpec::new()));
            c.column(&spark_connect::functions::rank().over(WindowSpec::default()));
            Ok(())
        },
        "WindowSpec::partition_by" => |c| {
            let spec = WindowSpec::new().partition_by(vec![e(col("a"))]);
            c.column(&spark_connect::functions::rank().over(spec));
            Ok(())
        },
        "WindowSpec::order_by" => |c| {
            let spec = WindowSpec::new().order_by(vec![sort("a")]);
            c.column(&spark_connect::functions::rank().over(spec));
            Ok(())
        },
        "WindowSpec::rows_between" => |c| {
            let spec = WindowSpec::new()
                .order_by(vec![sort("a")])
                .rows_between(FrameBound::Preceding(2), FrameBound::Following(1));
            c.column(&spark_connect::functions::sum(col("v")).over(spec));
            Ok(())
        },
        "WindowSpec::range_between" => |c| {
            let spec = WindowSpec::new()
                .order_by(vec![sort("a")])
                .range_between(FrameBound::CurrentRow, FrameBound::UnboundedFollowing);
            c.column(&spark_connect::functions::sum(col("v")).over(spec));
            Ok(())
        },
        "WindowSpec::repr" => |c| {
            let spec = WindowSpec::new()
                .partition_by(vec![e(col("p"))])
                .order_by(vec![sort("a")])
                .rows_between(FrameBound::UnboundedPreceding, FrameBound::CurrentRow);
            c.message("Text", &spec.repr());
            Ok(())
        },
    ]
}
