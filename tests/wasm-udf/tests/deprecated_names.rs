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

//! The deprecated names of the UDF API: `#[spark_wasm_udf]`, `spark_connect::wasm_udf` and
//! the `wasm-udf` feature, the only one this crate enables, so the new names here build
//! through it. `scripts/check_deprecated_udf_names.py` builds this file with the
//! deprecation lint forced on and checks which old names warn.

use spark_connect::rust_udf;
use spark_connect::session::SparkSession;

// An attribute macro's deprecation takes its lint level from the enclosing scope.
#[allow(deprecated)]
mod old_names {
    use spark_connect_macros::spark_wasm_udf;

    #[spark_wasm_udf]
    pub mod udfs {
        pub fn add_one(x: i64) -> i64 {
            x + 1
        }
    }

    // The old name ignores the arguments that the new name rejects.
    pub mod with_arguments {
        use spark_connect_macros::spark_wasm_udf;

        #[spark_wasm_udf(anything, x = "y", 1 + 1)]
        pub mod udfs {
            pub fn add_one(x: i64) -> i64 {
                x + 1
            }
        }
    }

    pub mod with_a_crate_option_and_more {
        use spark_connect_macros::spark_wasm_udf;

        #[spark_wasm_udf(crate = spark_connect_async, anything)]
        pub mod udfs {
            pub fn add_one(x: i64) -> i64 {
                x + 1
            }
        }
    }
}

mod new_names {
    use spark_connect_macros::spark_udf;

    #[spark_udf]
    pub mod udfs {
        pub fn add_one(x: i64) -> i64 {
            x + 1
        }
    }
}

#[test]
#[allow(deprecated)]
fn the_deprecated_names_build_what_the_new_names_build() {
    use spark_connect::wasm_udf;
    use spark_connect::wasm_udf::AbiType::Bool;

    let _: fn(&SparkSession) -> wasm_udf::UdfRegistration<'_> = SparkSession::udf;
    let arguments: Vec<wasm_udf::AbiType> = {
        use spark_connect::wasm_udf::AbiType::*;
        vec![Str, I64]
    };
    let by_hand: wasm_udf::UserDefinedFunction = wasm_udf::udf(
        "label",
        vec![0u8, 97, 115, 109],
        "spark_udf_label",
        arguments,
        Bool,
    )
    .with_packer(wasm_udf::PythonPacker::default());
    let by_hand_with_the_new_names = rust_udf::udf(
        "label",
        vec![0u8, 97, 115, 109],
        "spark_udf_label",
        vec![rust_udf::AbiType::Str, rust_udf::AbiType::I64],
        rust_udf::AbiType::Bool,
    )
    .with_packer(rust_udf::PythonPacker::default());
    for (name, old, new) in [
        (
            "#[spark_wasm_udf]",
            old_names::udf::add_one_udf(),
            new_names::udf::add_one_udf(),
        ),
        (
            "#[spark_wasm_udf] with arguments",
            old_names::with_arguments::udf::add_one_udf(),
            new_names::udf::add_one_udf(),
        ),
        (
            "#[spark_wasm_udf] with a crate option and more",
            old_names::with_a_crate_option_and_more::udf::add_one_udf(),
            new_names::udf::add_one_udf(),
        ),
        ("wasm_udf::udf", by_hand, by_hand_with_the_new_names),
    ] {
        assert!(
            format!("{old:?}") == format!("{new:?}"),
            "{name} builds another UDF than its new name"
        );
    }
}
