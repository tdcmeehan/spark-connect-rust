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

//! `#[spark_udf(crate = spark_connect_async)]` builds the UDFs of the async client: its builders
//! are that client's, with the signature of each function, and its direct calls take and return
//! that client's columns.

use rust_udf_async_tests::udf;
use spark_connect_async::rust_udf::{self, AbiType, UserDefinedFunction};
use spark_connect_async::{col, lit, Column, Result};

/// The deprecated name of the macro, with the option.
#[allow(deprecated)]
mod old_name {
    use spark_connect_macros::spark_wasm_udf;

    #[spark_wasm_udf(crate = spark_connect_async)]
    pub mod fixtures {
        pub fn minus(x: i64, y: Option<i64>) -> Option<i64> {
            y.map(|y| x.wrapping_sub(y))
        }
    }
}

const MODULE: &[u8] = include_bytes!(env!("WASM_UDFS_MODULE"));

fn optional(abi: AbiType) -> AbiType {
    AbiType::Nullable(Box::new(abi))
}

#[test]
fn the_builders_are_the_udfs_of_the_functions_with_their_signatures() {
    let cases: Vec<(&str, UserDefinedFunction, UserDefinedFunction)> = vec![
        (
            "minus, per batch",
            udf::minus_udf(),
            rust_udf::udf(
                "minus",
                MODULE,
                "spark_udf_minus",
                vec![AbiType::I64, optional(AbiType::I64)],
                optional(AbiType::I64),
            )
            .with_batch_entrypoint("spark_batch_udf_minus"),
        ),
        (
            "label, per batch",
            udf::label_udf(),
            rust_udf::udf(
                "label",
                MODULE,
                "spark_udf_label",
                vec![optional(AbiType::Str), AbiType::I64],
                AbiType::Str,
            )
            .with_batch_entrypoint("spark_batch_udf_label"),
        ),
        (
            "total, per row",
            udf::total_udf(),
            rust_udf::udf(
                "total",
                MODULE,
                "spark_udf_total",
                vec![AbiType::Array(Box::new(AbiType::I64))],
                AbiType::I64,
            ),
        ),
        (
            "minus, by the deprecated name",
            old_name::udf::minus_udf(),
            udf::minus_udf(),
        ),
    ];
    for (name, generated, expected) in cases {
        assert_eq!(format!("{generated:?}"), format!("{expected:?}"), "{name}");
    }
}

#[test]
fn the_direct_calls_take_and_return_columns_of_the_async_client() {
    // Type-checked, never called: building the columns runs Python.
    let _ = || -> Result<()> {
        let by_name: Column = udf::minus("x", "y")?;
        let nested: Column = udf::minus(by_name, lit(1))?;
        let mixed: Column = udf::label(col("s"), String::from("n"))?;
        let _: [Column; 3] = [nested, mixed, udf::total(col("xs"))?];
        Ok(())
    };
}
