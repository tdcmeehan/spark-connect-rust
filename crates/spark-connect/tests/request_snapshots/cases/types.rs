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

use spark_connect::types::{DataType, StructField};

use super::field;
use crate::{Case, Ctx};

fn every_type() -> Vec<DataType> {
    let mut types = vec![
        DataType::Null,
        DataType::Boolean,
        DataType::Byte,
        DataType::Short,
        DataType::Integer,
        DataType::Long,
        DataType::Float,
        DataType::Double,
        DataType::Decimal {
            precision: 10,
            scale: 2,
        },
        DataType::String {
            collation: String::new(),
        },
        DataType::String {
            collation: "UTF8_LCASE".to_string(),
        },
        DataType::Char { length: 3 },
        DataType::Varchar { length: 9 },
        DataType::Binary,
        DataType::Date,
        DataType::Timestamp,
        DataType::TimestampNtz,
        DataType::Time { precision: 6 },
        DataType::TimestampNtzNanos { precision: 9 },
        DataType::TimestampLtzNanos { precision: 7 },
        DataType::CalendarInterval,
        DataType::YearMonthInterval {
            start_field: 0,
            end_field: 1,
        },
        DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        },
        DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: false,
        },
        DataType::Map {
            key_type: Box::new(DataType::Long),
            value_type: Box::new(DataType::Double),
            value_contains_null: true,
        },
        DataType::Variant,
        DataType::Geometry { srid: 4326 },
        DataType::Geography { srid: -1 },
        DataType::Udt {
            type_str: "point".to_string(),
            jvm_class: Some("com.example.PointUDT".to_string()),
            python_class: Some("example.PointUDT".to_string()),
            serialized_python_class: None,
            sql_type: Some(Box::new(DataType::Binary)),
        },
        DataType::Unparsed {
            data_type_string: "some_type".to_string(),
        },
    ];
    let mut with_metadata = field("m", DataType::Integer);
    with_metadata.metadata = BTreeMap::from([("comment".to_string(), serde_json::json!("c"))]);
    types.push(DataType::Struct {
        fields: vec![
            field("a", DataType::Long),
            with_metadata,
            field(
                "n",
                DataType::Struct {
                    fields: vec![field("x", DataType::Date)],
                },
            ),
        ],
    });
    types
}

fn text(c: &Ctx, value: impl ToString) {
    c.message("Text", &value.to_string());
}

const DDL: &[&str] = &[
    "a INT, b STRING",
    "x DECIMAL(5, 1), y ARRAY<MAP<STRING, BIGINT>>, z STRUCT<p: DATE, q: TIMESTAMP_NTZ>",
    "c CHAR(4), v VARCHAR(8), t TIME(3), i INTERVAL DAY TO SECOND, j INTERVAL YEAR",
    "bigint",
    "struct<a:binary,b:boolean,c:tinyint,d:smallint,e:float,f:double,g:timestamp,h:variant>",
];

pub fn cases() -> Vec<Case> {
    cases![
        "DataType::to_proto" => |c| {
            for data_type in every_type() {
                c.data_type(&data_type);
            }
            Ok(())
        },
        "DataType::from_proto" => |c| {
            for data_type in every_type() {
                c.data_type(&DataType::from_proto(&data_type.to_proto())?);
            }
            Ok(())
        },
        "DataType::from_ddl" => |c| {
            for ddl in DDL {
                match DataType::from_ddl(ddl) {
                    Ok(data_type) => c.data_type(&data_type),
                    Err(error) => text(c, error),
                }
            }
            Ok(())
        },
        "DataType::json" => |c| {
            for data_type in every_type() {
                text(c, data_type.json());
                text(c, data_type.json_value());
            }
            Ok(())
        },
        "DataType::from_json" => |c| {
            for data_type in every_type() {
                for parsed in [DataType::from_json(&data_type.json_value()), DataType::from_json_str(&data_type.json())] {
                    match parsed {
                        Ok(parsed) => c.data_type(&parsed),
                        Err(error) => text(c, error),
                    }
                }
            }
            Ok(())
        },
        "DataType::simple_string" => |c| {
            for data_type in every_type() {
                text(c, data_type.simple_string());
                text(c, data_type.type_name());
                text(c, &data_type);
                text(c, data_type.need_conversion());
            }
            Ok(())
        },
        "DataType::to_ddl" => |c| {
            for ddl in DDL {
                let Ok(data_type) = DataType::from_ddl(ddl) else {
                    continue;
                };
                text(c, data_type.to_ddl().unwrap_or_else(|e| e.to_string()));
                text(c, data_type.tree_string().unwrap_or_else(|e| e.to_string()));
                text(c, data_type.tree_string_with_depth(1).unwrap_or_else(|e| e.to_string()));
                text(c, format!("{:?}", data_type.field_names().ok()));
                text(c, format!("{:?}", data_type.names().ok()));
            }
            Ok(())
        },
        "DataType::to_nullable" => |c| {
            for data_type in every_type() {
                c.data_type(&data_type.to_nullable());
            }
            Ok(())
        },
        "DataType::add" => |c| {
            let base = DataType::from_ddl("a INT")?;
            c.data_type(&base.add("b", DataType::String { collation: String::new() }, false, None)?);
            let metadata = BTreeMap::from([("k".to_string(), serde_json::json!(1))]);
            c.data_type(&base.add("c", DataType::Double, true, Some(metadata))?);
            Ok(())
        },
        "StructField::simple_string" => |c| {
            let field = field("f", DataType::Long);
            text(c, field.simple_string());
            text(c, field.json_value());
            let parsed = StructField::from_json_str(&field.json_value().to_string())?;
            c.data_type(&DataType::Struct { fields: vec![parsed] });
            Ok(())
        },
    ]
}
