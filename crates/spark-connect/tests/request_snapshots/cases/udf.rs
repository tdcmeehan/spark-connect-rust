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

use spark_connect::column::{col, lit};
use spark_connect::datasource::{
    CommonInlineUserDefinedDataSourceExpression, PythonDataSourcePayload,
};
use spark_connect::observation::Observation;
use spark_connect::origin::{set_current, Origin};
use spark_connect::table_arg::TableArg;
use spark_connect::udf::eval_type;

use super::{batch_udf, other, payload, range, schema};
use crate::Case;

pub fn cases() -> Vec<Case> {
    let mut cases = cases![
        "PythonUDFPayload::to_proto" => |c| {
            for eval in [eval_type::SQL_BATCHED_UDF, eval_type::SQL_ARROW_BATCHED_UDF, eval_type::SQL_SCALAR_PANDAS_UDF] {
                c.message("PythonUdf", &payload(eval).to_proto());
            }
            Ok(())
        },
        "CommonInlineUserDefinedFunctionExpression::to_proto" => |c| {
            c.message("CommonInlineUserDefinedFunction", &batch_udf().to_proto());
            Ok(())
        },
        "PythonDataSourcePayload::to_proto" => |c| {
            let payload = PythonDataSourcePayload::new(vec![1, 2], "3.11".to_string());
            c.message("PythonDataSource", &payload.to_proto());
            let source = CommonInlineUserDefinedDataSourceExpression::new("src".to_string(), payload);
            c.message("CommonInlineUserDefinedDataSource", &source.to_proto());
            Ok(())
        },
        "TableValuedFunction::range" => |c| {
            c.df(&c.spark.tvf().range(0, Some(10), 2, Some(3))?)?;
            c.df(&c.spark.tvf().range(5, None, 1, None)?)
        },
        "TableValuedFunction::explode" => |c| c.df(&c.spark.tvf().explode(&col("a"))?),
        "TableValuedFunction::explode_outer" => |c| c.df(&c.spark.tvf().explode_outer(&col("a"))?),
        "TableValuedFunction::inline" => |c| c.df(&c.spark.tvf().inline(&col("a"))?),
        "TableValuedFunction::inline_outer" => |c| c.df(&c.spark.tvf().inline_outer(&col("a"))?),
        "TableValuedFunction::json_tuple" => |c| c.df(&c.spark.tvf().json_tuple(&col("j"), [lit(1), col("k")])?),
        "TableValuedFunction::posexplode" => |c| c.df(&c.spark.tvf().posexplode(&col("a"))?),
        "TableValuedFunction::posexplode_outer" => |c| c.df(&c.spark.tvf().posexplode_outer(&col("a"))?),
        "TableValuedFunction::stack" => |c| c.df(&c.spark.tvf().stack(&lit(2), [col("a"), col("b")])?),
        "TableValuedFunction::collations" => |c| c.df(&c.spark.tvf().collations()?),
        "TableValuedFunction::sql_keywords" => |c| c.df(&c.spark.tvf().sql_keywords()?),
        "TableValuedFunction::variant_explode" => |c| c.df(&c.spark.tvf().variant_explode(&col("v"))?),
        "TableValuedFunction::variant_explode_outer" => |c| c.df(&c.spark.tvf().variant_explode_outer(&col("v"))?),
        "TableValuedFunction::python_worker_logs" => |c| c.df(&c.spark.tvf().python_worker_logs()?),
        "TableValuedFunction::udtf" => |c| {
            c.df(&c.spark.tvf().udtf("t", vec![col("a")], Some(schema()), eval_type::SQL_TABLE_UDF, vec![9], "3.11".to_string(), true))?;
            c.df(&c.spark.tvf().udtf("u", vec![], None, eval_type::SQL_ARROW_TABLE_UDF, vec![8], "3.11".to_string(), false))
        },
        "TableArg::to_column" => |c| {
            let arg = TableArg::new(other(c));
            c.column(&arg.to_column());
            c.column(&spark_connect::Column::from(TableArg::new(other(c))));
            c.df(&range(c).select([arg.to_column()]))
        },
        "TableArg::partition_by" => |c| {
            let arg = TableArg::new(other(c)).partition_by(["a"])?.order_by([col("b").desc()])?;
            c.df(&range(c).select([arg.to_column()]))
        },
        "TableArg::with_single_partition" => |c| {
            let arg = TableArg::new(other(c)).with_single_partition()?.order_by(["b"])?;
            c.df(&range(c).select([arg.to_column()]))
        },
        "Observation::get" => |c| {
            let observation = Observation::new("o");
            let df = range(c).observe_with(&observation, vec![spark_connect::functions::count(col("id")).expression().clone()])?;
            c.df(&df)?;
            assert!(observation.same(&observation));
            c.message("Text", &format!("{:?}", observation.name()));
            observation.get().map(drop)
        },
        "origin::set_current" => |c| {
            let column = {
                let _origin = set_current(Origin { fragment: "col".to_string(), call_site: "main.rs:1".to_string() });
                col("a")
            };
            c.column(&column);
            c.column(&col("b"));
            Ok(())
        },
    ];
    cases.extend(wasm::cases());
    cases
}

#[cfg(all(feature = "wasm-udf", unix))]
mod wasm {
    use spark_connect::column::col;
    use spark_connect::rust_udf::{udf, AbiType, PythonPacker, UserDefinedFunction};

    use crate::{Case, Ctx};

    /// A packer that answers with the spec it is given, so the command is the spec.
    fn echo_packer() -> PythonPacker {
        let path = std::env::temp_dir().join(format!(
            "spark-connect-request-snapshots-echo-packer-{}",
            std::process::id()
        ));
        std::fs::write(&path, "#!/bin/sh\ncat\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        PythonPacker {
            python_exe: path.to_str().unwrap().to_string(),
            pythonpath: Vec::new(),
        }
    }

    fn add_one() -> UserDefinedFunction {
        udf(
            "add_one",
            vec![0u8, 97, 115, 109],
            "add_one",
            vec![AbiType::I64, AbiType::Nullable(Box::new(AbiType::Str))],
            AbiType::Array(Box::new(AbiType::F64)),
        )
        .with_packer(echo_packer())
        .with_python_ver("3.11")
    }

    fn record(c: &Ctx, f: &UserDefinedFunction) -> spark_connect::Result<()> {
        c.data_type(&f.output_type());
        c.message("PythonUdf", &f.to_payload()?.to_proto());
        c.message(
            "CommonInlineUserDefinedFunction",
            &f.to_expression(vec![col("a"), col("b")])?.to_proto(),
        );
        c.column(&f.call(["a", "b"])?);
        Ok(())
    }

    pub fn cases() -> Vec<Case> {
        cases![
            "wasm_udf::udf" => |c| record(c, &add_one()),
            "UserDefinedFunction::with_batch_entrypoint" => |c| {
                let f = udf("plus", vec![0u8, 97, 115, 109], "plus", vec![AbiType::I64, AbiType::I64], AbiType::I64)
                    .with_packer(echo_packer())
                    .with_python_ver("3.11")
                    .with_batch_entrypoint("plus_batch");
                record(c, &f)
            },
            "UserDefinedFunction::as_nondeterministic" => |c| record(c, &add_one().as_nondeterministic()),
            "UserDefinedFunction::with_eval_type" => {
                |c| record(c, &add_one().with_eval_type(spark_connect::udf::eval_type::SQL_SCALAR_ARROW_UDF))
            },
            "AbiType::descriptor" => |c| {
                for t in [AbiType::I32, AbiType::I64, AbiType::F32, AbiType::F64, AbiType::Bool, AbiType::Str, AbiType::Binary, AbiType::Array(Box::new(AbiType::I32)), AbiType::Nullable(Box::new(AbiType::Bool))] {
                    c.message("Text", &t.descriptor());
                    c.data_type(&t.to_data_type());
                }
                Ok(())
            },
            "UdfRegistration::register" => |c| c.spark.udf().register("add_one", &add_one()),
        ]
    }
}

#[cfg(not(all(feature = "wasm-udf", unix)))]
mod wasm {
    pub fn cases() -> Vec<crate::Case> {
        Vec::new()
    }
}
