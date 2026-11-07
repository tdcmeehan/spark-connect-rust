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
use spark_connect::functions as f;

use super::{other, range};
use crate::Case;

pub fn cases() -> Vec<Case> {
    cases![
        "functions::broadcast" => |c| c.df(&range(c).join(&f::broadcast(other(c)), None, spark_connect::plan::JoinType::Inner)),
        "functions::col/star" => |c| {
            c.columns(&[f::col("*"), f::col("t.*"), f::count(f::col("*"))]);
            Ok(())
        },
        "functions::lit/column" => |c| {
            c.column(&f::lit(col("a")));
            Ok(())
        },
        "functions::higher_order/nested" => |c| {
            c.column(&f::transform(col("xs"), |xs| {
                let x = xs.into_iter().next().unwrap();
                f::filter(x, |ys| ys.into_iter().next().unwrap().is_not_null())
            }));
            Ok(())
        },
    ]
}
