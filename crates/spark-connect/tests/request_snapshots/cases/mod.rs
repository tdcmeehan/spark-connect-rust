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

mod catalog;
mod column;
mod dataframe;
mod expression;
mod functions;
mod functions_generated;
mod group;
mod ml;
mod pipelines;
mod plan;
mod readwriter;
mod session;
mod streaming;
mod types;
mod udf;

use spark_connect::column::col;
use spark_connect::dataframe::DataFrame;
use spark_connect::types::{DataType, StructField};
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

use crate::{Case, Ctx};

/// Cases that exist only with a crate feature, whose snapshots are kept without it.
pub const FEATURE_GATED: &[&str] = &[
    "DataFrame::to_datafusion",
    "DataFrame::to_polars",
    "wasm_udf::udf",
    "UserDefinedFunction::as_nondeterministic",
    "UserDefinedFunction::with_eval_type",
    "UserDefinedFunction::with_batch_entrypoint",
    "AbiType::descriptor",
    "UdfRegistration::register",
];

pub fn all() -> Vec<Case> {
    [
        session::cases(),
        dataframe::cases(),
        group::cases(),
        column::cases(),
        expression::cases(),
        functions::cases(),
        functions_generated::cases(),
        plan::cases(),
        types::cases(),
        readwriter::cases(),
        catalog::cases(),
        streaming::cases(),
        ml::cases(),
        udf::cases(),
        pipelines::cases(),
    ]
    .into_iter()
    .flatten()
    .collect()
}

pub fn range(c: &Ctx) -> DataFrame {
    c.spark.range(10).unwrap()
}

pub fn other(c: &Ctx) -> DataFrame {
    c.spark.read().table("other")
}

pub fn field(name: &str, data_type: DataType) -> StructField {
    StructField {
        name: name.to_string(),
        data_type,
        nullable: true,
        metadata: Default::default(),
    }
}

pub fn schema() -> DataType {
    DataType::from_ddl("id BIGINT, name STRING").unwrap()
}

pub fn payload(eval: i32) -> PythonUDFPayload {
    PythonUDFPayload::new(DataType::Long, eval, vec![1, 2, 3], "3.11".to_string())
}

pub fn udf(eval: i32) -> CommonInlineUserDefinedFunctionExpression {
    CommonInlineUserDefinedFunctionExpression::new(
        "f".to_string(),
        true,
        vec![col("id").expression().clone()],
        payload(eval),
    )
}

pub fn batch_udf() -> CommonInlineUserDefinedFunctionExpression {
    udf(eval_type::SQL_BATCHED_UDF)
}
