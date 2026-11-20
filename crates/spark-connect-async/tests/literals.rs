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

//! The literals of Rust values are those of `apache-spark-connect` for the same Spark values.

use chrono::{DateTime, NaiveDate, NaiveTime, TimeDelta};
use spark_connect::{lit as old_lit, lit_boolean, lit_double, lit_string};
use spark_connect::{Column as OldColumn, Expression, LiteralExpression as L};
use spark_connect_async::{lit, Column, Decimal};
use spark_connect_proto as proto;

fn old(literal: L) -> OldColumn {
    OldColumn::new(Expression::literal(literal))
}

fn sent(column: &Column) -> proto::Expression {
    column.expression().to_proto()
}

#[test]
fn each_rust_value_is_the_literal_of_its_spark_type() {
    use spark_connect::types::DataType as T;
    let date = NaiveDate::from_ymd_opt(1969, 12, 31).unwrap();
    let instant = DateTime::from_timestamp(-1, 999_000).unwrap();
    let time = NaiveTime::from_hms_micro_opt(13, 2, 3, 4).unwrap();
    let string_type = T::String {
        collation: "UTF8_BINARY".to_string(),
    };
    let cases: Vec<(&str, Column, OldColumn)> = vec![
        ("bool", lit(true), lit_boolean(true)),
        ("i32", lit(5), old_lit(5)),
        (
            "i64 beyond i32",
            lit(5_000_000_000i64),
            old_lit(5_000_000_000),
        ),
        ("i64", lit(7i64), old(L::long(7))),
        ("i8", lit(-1i8), old(L::Byte(-1))),
        ("i16", lit(300i16), old(L::Short(300))),
        ("f32", lit(1.5f32), old(L::Float(1.5))),
        ("f64", lit(1.5), lit_double(1.5)),
        ("&str", lit("a"), lit_string("a")),
        ("String", lit("a".to_string()), lit_string("a")),
        ("bytes", lit(vec![1u8, 2]), old(L::binary(vec![1, 2]))),
        ("&[u8]", lit(&[3u8][..]), old(L::binary(vec![3]))),
        (
            "Decimal",
            lit("-1.50".parse::<Decimal>().unwrap()),
            old(L::Decimal {
                value: "-1.50".to_string(),
                precision: 3,
                scale: 2,
            }),
        ),
        ("NaiveDate", lit(date), old(L::Date(-1))),
        ("DateTime<Utc>", lit(instant), old(L::Timestamp(-999_001))),
        (
            "NaiveDateTime",
            lit(instant.naive_utc()),
            old(L::TimestampNtz(-999_001)),
        ),
        (
            "NaiveTime",
            lit(time),
            old(L::Time {
                nano: 46_923_000_004_000,
                precision: 6,
            }),
        ),
        (
            "TimeDelta",
            lit(TimeDelta::milliseconds(-1500)),
            old(L::DayTimeInterval(-1_500_000)),
        ),
        ("None::<i64>", lit(None::<i64>), old(L::null(T::Long))),
        ("Some(1)", lit(Some(1)), old_lit(1)),
        (
            "None::<&str>",
            lit(None::<&str>),
            old(L::null(string_type.clone())),
        ),
        (
            "Vec<i32>",
            lit(vec![1, 2]),
            old(L::Array {
                element_type: Box::new(T::Integer),
                elements: vec![L::int(1), L::int(2)],
            }),
        ),
        (
            "Vec<Decimal> of different scales",
            lit(vec![
                "1.5".parse::<Decimal>().unwrap(),
                "-10.25".parse().unwrap(),
            ]),
            old(L::Array {
                element_type: Box::new(T::Decimal {
                    precision: 4,
                    scale: 2,
                }),
                elements: vec![
                    L::Decimal {
                        value: "1.5".to_string(),
                        precision: 2,
                        scale: 1,
                    },
                    L::Decimal {
                        value: "-10.25".to_string(),
                        precision: 4,
                        scale: 2,
                    },
                ],
            }),
        ),
        (
            "empty Vec<String>",
            lit(Vec::<String>::new()),
            old(L::Array {
                element_type: Box::new(string_type),
                elements: Vec::new(),
            }),
        ),
    ];
    for (name, new, old) in cases {
        assert_eq!(sent(&new), old.expression().to_proto(), "{name}");
    }
}
