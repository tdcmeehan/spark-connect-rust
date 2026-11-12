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

//! UDFs of the async client, built with `#[spark_udf(crate = spark_connect_async)]`.

use spark_connect_macros::spark_udf;

#[spark_udf(crate = spark_connect_async)]
mod fixtures {
    pub fn minus(x: i64, y: Option<i64>) -> Option<i64> {
        y.map(|y| x.wrapping_sub(y))
    }

    pub fn label(name: Option<String>, n: i64) -> String {
        format!("{}#{n}", name.unwrap_or_default())
    }

    pub fn total(xs: Vec<i64>) -> i64 {
        xs.iter().sum()
    }
}
