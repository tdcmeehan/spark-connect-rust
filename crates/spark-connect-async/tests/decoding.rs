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

//! The rows of Arrow batches of every Spark type, with nulls and nested types, read into Rust
//! types with serde and into `Value`s.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Decimal256Array,
    DurationMicrosecondArray, Float32Array, Float64Array, Int16Array, Int32Array, Int32Builder,
    Int64Array, Int8Array, IntervalYearMonthArray, ListArray, MapBuilder, NullArray, StringArray,
    StringBuilder, StructArray, Time64NanosecondArray, TimestampMicrosecondArray,
    TimestampNanosecondArray,
};
use arrow::datatypes::{i256, DataType as A, Field, Int32Type, Schema};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Utc};
use serde::Deserialize;
use spark_connect_async::de::{from_record_batch, rows};
use spark_connect_async::{Decimal, ErrorKind, Value};

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, array)| array).collect(),
    )
    .unwrap()
}

fn dec(text: &str) -> Decimal {
    text.parse().unwrap()
}

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

/// Three rows of every scalar type: a value, a null, and an edge value.
fn scalars() -> RecordBatch {
    batch(vec![
        (
            "bool",
            Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
        ),
        (
            "byte",
            Arc::new(Int8Array::from(vec![Some(1), None, Some(i8::MIN)])),
        ),
        (
            "short",
            Arc::new(Int16Array::from(vec![Some(2), None, Some(i16::MIN)])),
        ),
        (
            "int",
            Arc::new(Int32Array::from(vec![Some(3), None, Some(i32::MIN)])),
        ),
        (
            "long",
            Arc::new(Int64Array::from(vec![Some(4), None, Some(i64::MAX)])),
        ),
        (
            "float",
            Arc::new(Float32Array::from(vec![
                Some(0.5),
                None,
                Some(f32::INFINITY),
            ])),
        ),
        (
            "double",
            Arc::new(Float64Array::from(vec![Some(-0.25), None, Some(f64::NAN)])),
        ),
        (
            "dec_10_2",
            Arc::new(
                Decimal128Array::from(vec![Some(-150), None, Some(9_999_999_999)])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
        ),
        (
            "dec_38_38",
            Arc::new(
                Decimal128Array::from(vec![Some(1), None, Some(10i128.pow(38) - 1)])
                    .with_precision_and_scale(38, 38)
                    .unwrap(),
            ),
        ),
        (
            "dec_38_0",
            Arc::new(
                Decimal256Array::from(vec![
                    Some(i256::from_i128(-(10i128.pow(38) - 1))),
                    None,
                    Some(i256::from_i128(0)),
                ])
                .with_precision_and_scale(38, 0)
                .unwrap(),
            ),
        ),
        (
            "string",
            Arc::new(StringArray::from(vec![Some("héllo"), None, Some("")])),
        ),
        (
            "binary",
            Arc::new(BinaryArray::from(vec![
                Some(&b"\x00\xff"[..]),
                None,
                Some(&b""[..]),
            ])),
        ),
        (
            "date",
            Arc::new(Date32Array::from(vec![Some(19_723), None, Some(-1)])),
        ),
        (
            "timestamp",
            Arc::new(
                TimestampNanosecondArray::from(vec![
                    Some(1_700_000_000_123_456_789),
                    None,
                    Some(-1),
                ])
                .with_timezone("UTC"),
            ),
        ),
        (
            "timestamp_us",
            Arc::new(
                TimestampMicrosecondArray::from(vec![Some(1_000_001), None, Some(-1)])
                    .with_timezone("America/Los_Angeles"),
            ),
        ),
        (
            "timestamp_ntz",
            Arc::new(TimestampNanosecondArray::from(vec![
                Some(1),
                None,
                Some(-1_000_000_001),
            ])),
        ),
        (
            "time",
            Arc::new(Time64NanosecondArray::from(vec![
                Some(1),
                None,
                Some(86_399_999_999_999),
            ])),
        ),
        (
            "day_time",
            Arc::new(DurationMicrosecondArray::from(vec![
                Some(-1_500_000),
                None,
                Some(0),
            ])),
        ),
        (
            "year_month",
            Arc::new(IntervalYearMonthArray::from(vec![Some(-13), None, Some(0)])),
        ),
        ("nothing", Arc::new(NullArray::new(3))),
    ])
}

#[derive(Debug, Deserialize, PartialEq)]
struct Scalars {
    bool: Option<bool>,
    byte: Option<i8>,
    short: Option<i16>,
    int: Option<i32>,
    long: Option<i64>,
    float: Option<f32>,
    double: Option<f64>,
    dec_10_2: Option<Decimal>,
    dec_38_38: Option<Decimal>,
    dec_38_0: Option<Decimal>,
    string: Option<String>,
    binary: Option<Vec<u8>>,
    date: Option<NaiveDate>,
    timestamp: Option<DateTime<Utc>>,
    timestamp_us: Option<DateTime<Utc>>,
    timestamp_ntz: Option<NaiveDateTime>,
    time: Option<NaiveTime>,
    day_time: Option<i64>,
    year_month: Option<i32>,
    nothing: Option<i32>,
}

#[test]
fn every_scalar_type_reads_into_its_rust_type_with_nulls() {
    let rows: Vec<Scalars> = from_record_batch(&scalars()).unwrap();
    let instant = |s, n| DateTime::from_timestamp(s, n).unwrap();
    assert_eq!(
        rows[0],
        Scalars {
            bool: Some(true),
            byte: Some(1),
            short: Some(2),
            int: Some(3),
            long: Some(4),
            float: Some(0.5),
            double: Some(-0.25),
            dec_10_2: Some(Decimal::new(-150, 10, 2).unwrap()),
            dec_38_38: Some(Decimal::new(1, 38, 38).unwrap()),
            dec_38_0: Some(Decimal::new(-(10i128.pow(38) - 1), 38, 0).unwrap()),
            string: Some("héllo".to_string()),
            binary: Some(vec![0, 255]),
            date: Some(date(2024, 1, 1)),
            timestamp: Some(instant(1_700_000_000, 123_456_789)),
            timestamp_us: Some(instant(1, 1_000)),
            timestamp_ntz: Some(instant(0, 1).naive_utc()),
            time: Some(NaiveTime::from_hms_nano_opt(0, 0, 0, 1).unwrap()),
            day_time: Some(-1_500_000),
            year_month: Some(-13),
            nothing: None,
        }
    );
    assert_eq!(
        rows[1],
        Scalars {
            bool: None,
            byte: None,
            short: None,
            int: None,
            long: None,
            float: None,
            double: None,
            dec_10_2: None,
            dec_38_38: None,
            dec_38_0: None,
            string: None,
            binary: None,
            date: None,
            timestamp: None,
            timestamp_us: None,
            timestamp_ntz: None,
            time: None,
            day_time: None,
            year_month: None,
            nothing: None,
        }
    );
    let edge = &rows[2];
    assert_eq!(
        (edge.byte, edge.short, edge.int, edge.long),
        (
            Some(i8::MIN),
            Some(i16::MIN),
            Some(i32::MIN),
            Some(i64::MAX)
        )
    );
    assert!(edge.double.unwrap().is_nan() && edge.float == Some(f32::INFINITY));
    assert_eq!(
        (edge.dec_10_2, edge.dec_38_38.map(|d| d.unscaled())),
        (Some(dec("99999999.99")), Some(10i128.pow(38) - 1))
    );
    assert_eq!(edge.string.as_deref(), Some(""));
    assert_eq!(edge.date, Some(date(1969, 12, 31)));
    assert_eq!(edge.timestamp, Some(instant(-1, 999_999_999)));
    assert_eq!(
        edge.timestamp_ntz,
        Some(instant(-2, 999_999_999).naive_utc())
    );
    assert_eq!(
        edge.time,
        Some(NaiveTime::from_hms_nano_opt(23, 59, 59, 999_999_999).unwrap())
    );
}

#[test]
fn every_scalar_type_reads_into_a_value() {
    let rows = rows(&scalars()).unwrap();
    let names: Vec<&str> = rows[0].names().iter().map(String::as_str).collect();
    assert_eq!(names[..3], ["bool", "byte", "short"]);
    let instant = |s, n| DateTime::from_timestamp(s, n).unwrap();
    assert_eq!(
        rows[0].values(),
        [
            Value::Boolean(true),
            Value::Byte(1),
            Value::Short(2),
            Value::Int(3),
            Value::Long(4),
            Value::Float(0.5),
            Value::Double(-0.25),
            Value::Decimal(Decimal::new(-150, 10, 2).unwrap()),
            Value::Decimal(Decimal::new(1, 38, 38).unwrap()),
            Value::Decimal(Decimal::new(-(10i128.pow(38) - 1), 38, 0).unwrap()),
            Value::String("héllo".to_string()),
            Value::Binary(vec![0, 255]),
            Value::Date(date(2024, 1, 1)),
            Value::Timestamp(instant(1_700_000_000, 123_456_789)),
            Value::Timestamp(instant(1, 1_000)),
            Value::TimestampNtz(instant(0, 1).naive_utc()),
            Value::Time(NaiveTime::from_hms_nano_opt(0, 0, 0, 1).unwrap()),
            Value::DayTimeInterval(TimeDelta::microseconds(-1_500_000)),
            Value::YearMonthInterval(-13),
            Value::Null,
        ]
    );
    assert!(rows[1].values().iter().all(|value| *value == Value::Null));
    assert_eq!(rows[0].get_by_name("int"), Some(&Value::Int(3)));
}

#[test]
fn temporal_and_decimal_values_read_as_text_and_as_numbers() {
    #[derive(Deserialize)]
    struct AsText {
        dec_10_2: String,
        dec_38_38: String,
        date: String,
        timestamp: String,
        timestamp_ntz: String,
        time: String,
    }
    #[derive(Deserialize)]
    struct AsNumbers {
        dec_10_2: f64,
        dec_38_0: i128,
        date: i32,
        timestamp: i64,
        timestamp_us: i64,
        timestamp_ntz: i64,
        time: i64,
    }
    let batch = scalars();
    let first = batch.slice(0, 1);
    let text = &from_record_batch::<AsText>(&first).unwrap()[0];
    assert_eq!(
        [
            &text.dec_10_2,
            &text.dec_38_38,
            &text.date,
            &text.timestamp,
            &text.timestamp_ntz,
            &text.time
        ],
        [
            "-1.50",
            &format!("0.{}1", "0".repeat(37)),
            "2024-01-01",
            "2023-11-14T22:13:20.123456789Z",
            "1970-01-01T00:00:00.000000001",
            "00:00:00.000000001",
        ]
    );
    let numbers = &from_record_batch::<AsNumbers>(&first).unwrap()[0];
    assert_eq!(numbers.dec_10_2, -1.5);
    assert_eq!(numbers.dec_38_0, -(10i128.pow(38) - 1));
    assert_eq!(
        (numbers.date, numbers.timestamp, numbers.timestamp_us),
        (19_723, 1_700_000_000_123_456, 1_000_001)
    );
    assert_eq!((numbers.timestamp_ntz, numbers.time), (0, 1));
    let zero: Vec<i64> = from_record_batch(&batch.slice(2, 1).project(&[9]).unwrap()).unwrap();
    assert_eq!(zero, [0]);
    // Microseconds round down, also before the epoch.
    let edge = &from_record_batch::<AsNumbers>(&batch.slice(2, 1)).unwrap()[0];
    assert_eq!((edge.timestamp, edge.timestamp_ntz), (-1, -1_000_001));
}

/// A batch of nested values: arrays, maps and structs, with empty and null parts.
fn nested() -> RecordBatch {
    let ints = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1), None, Some(3)]),
        Some(vec![]),
        None,
    ]);
    let mut map = MapBuilder::new(None, Int32Builder::new(), StringBuilder::new());
    for (key, value) in [(3, Some("c")), (1, None), (2, Some("b"))] {
        map.keys().append_value(key);
        map.values().append_option(value);
    }
    map.append(true).unwrap();
    map.append(true).unwrap();
    map.append(false).unwrap();
    let inner = StructArray::from(vec![(
        Arc::new(Field::new("c", A::Int32, true)),
        Arc::new(Int32Array::from(vec![Some(7), None, Some(9)])) as ArrayRef,
    )]);
    let middle = StructArray::from(vec![
        (
            Arc::new(Field::new("b", inner.data_type().clone(), true)),
            Arc::new(inner) as ArrayRef,
        ),
        (
            Arc::new(Field::new("s", A::Utf8, true)),
            Arc::new(StringArray::from(vec![Some("x"), Some("y"), None])) as ArrayRef,
        ),
    ]);
    let outer = StructArray::from(vec![(
        Arc::new(Field::new("a", middle.data_type().clone(), true)),
        Arc::new(middle) as ArrayRef,
    )]);
    let pairs = StructArray::from(vec![
        (
            Arc::new(Field::new("x", A::Int32, true)),
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("y", A::Utf8, true)),
            Arc::new(StringArray::from(vec!["p", "q", "r"])) as ArrayRef,
        ),
    ]);
    batch(vec![
        ("ints", Arc::new(ints)),
        ("map", Arc::new(map.finish())),
        ("nested", Arc::new(outer)),
        ("pair", Arc::new(pairs)),
    ])
}

#[test]
fn arrays_maps_and_structs_nest_and_keep_the_order_of_map_entries() {
    #[derive(Debug, Deserialize, PartialEq)]
    struct C {
        c: Option<i32>,
    }
    #[derive(Debug, Deserialize, PartialEq)]
    struct B {
        b: C,
        s: Option<String>,
    }
    #[derive(Debug, Deserialize, PartialEq)]
    struct Outer {
        a: B,
    }
    #[derive(Deserialize)]
    struct Nested {
        ints: Option<Vec<Option<i32>>>,
        map: Option<Vec<(i32, Option<String>)>>,
        nested: Outer,
        pair: (i32, String),
    }
    let rows: Vec<Nested> = from_record_batch(&nested()).unwrap();
    assert_eq!(rows[0].ints, Some(vec![Some(1), None, Some(3)]));
    assert_eq!(rows[1].ints, Some(vec![]));
    assert_eq!(rows[2].ints, None);
    assert_eq!(
        rows[0].map,
        Some(vec![
            (3, Some("c".to_string())),
            (1, None),
            (2, Some("b".to_string()))
        ])
    );
    assert_eq!(rows[1].map, Some(vec![]));
    assert_eq!(rows[2].map, None);
    assert_eq!(
        rows[0].nested,
        Outer {
            a: B {
                b: C { c: Some(7) },
                s: Some("x".to_string()),
            }
        }
    );
    assert_eq!(rows[1].nested.a.b.c, None);
    assert_eq!(rows[2].pair, (3, "r".to_string()));

    #[derive(Deserialize)]
    struct Maps {
        map: Option<BTreeMap<i32, Option<String>>>,
    }
    let sorted: Vec<Maps> = from_record_batch(&nested()).unwrap();
    let keys: Vec<i32> = sorted[0].map.as_ref().unwrap().keys().copied().collect();
    assert_eq!(keys, [1, 2, 3]);

    let values = rows_of(&nested());
    assert_eq!(
        values[0].get_by_name("map"),
        Some(&Value::Map(vec![
            (Value::Int(3), Value::String("c".to_string())),
            (Value::Int(1), Value::Null),
            (Value::Int(2), Value::String("b".to_string())),
        ]))
    );
    assert_eq!(
        values[0].get_by_name("ints"),
        Some(&Value::Array(vec![
            Value::Int(1),
            Value::Null,
            Value::Int(3)
        ]))
    );
    let Some(Value::Struct(outer)) = values[0].get_by_name("nested") else {
        panic!("not a struct: {:?}", values[0]);
    };
    let Some(Value::Struct(middle)) = outer.get_by_name("a") else {
        panic!("not a struct: {outer:?}");
    };
    assert_eq!(middle.names(), ["b", "s"]);
}

fn rows_of(batch: &RecordBatch) -> Vec<spark_connect_async::Row> {
    rows(batch).unwrap()
}

#[test]
fn a_row_reads_as_a_tuple_a_map_a_single_value_or_with_borrowed_text() {
    let batch = batch(vec![
        ("id", Arc::new(Int64Array::from(vec![1, 2]))),
        ("name", Arc::new(StringArray::from(vec!["a", "b"]))),
        (
            "raw",
            Arc::new(BinaryArray::from(vec![&b"x"[..], &b"y"[..]])),
        ),
    ]);
    let tuples: Vec<(i64, String, Vec<u8>)> = from_record_batch(&batch).unwrap();
    assert_eq!(tuples[1], (2, "b".to_string(), b"y".to_vec()));

    #[derive(Deserialize)]
    struct Borrowed<'a> {
        name: &'a str,
        raw: &'a [u8],
    }
    let borrowed: Vec<Borrowed> = from_record_batch(&batch).unwrap();
    assert_eq!((borrowed[0].name, borrowed[0].raw), ("a", &b"x"[..]));

    let maps: Vec<HashMap<String, String>> =
        from_record_batch(&batch.project(&[1]).unwrap()).unwrap();
    assert_eq!(maps[0]["name"], "a");
    let ids: Vec<i64> = from_record_batch(&batch.project(&[0]).unwrap()).unwrap();
    assert_eq!(ids, [1, 2]);

    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(rename_all = "lowercase")]
    enum Letter {
        A,
        B,
    }
    #[derive(Deserialize)]
    struct Renamed {
        #[serde(rename = "name")]
        letter: Letter,
        #[serde(default)]
        missing: Option<i32>,
    }
    let renamed: Vec<Renamed> = from_record_batch(&batch).unwrap();
    assert_eq!(renamed[1].letter, Letter::B);
    assert_eq!(renamed[0].missing, None);
}

#[test]
fn a_value_that_does_not_fit_its_rust_type_is_an_error_naming_its_row_and_column() {
    #[derive(Debug, Deserialize)]
    #[allow(dead_code)]
    struct Int {
        int: i32,
    }
    #[derive(Debug, Deserialize)]
    #[allow(dead_code)]
    struct Text {
        int: String,
    }
    #[derive(Debug, Deserialize)]
    #[allow(dead_code)]
    struct Absent {
        absent: i32,
    }
    #[derive(Debug, Deserialize)]
    #[allow(dead_code)]
    struct Scaled {
        dec_10_2: i64,
    }
    #[derive(Debug, Deserialize)]
    #[allow(dead_code)]
    struct Nested {
        nested: Deep,
    }
    #[derive(Debug, Deserialize)]
    #[allow(dead_code)]
    struct Deep {
        a: Middle,
    }
    #[derive(Debug, Deserialize)]
    #[allow(dead_code)]
    struct Middle {
        b: Inner,
    }
    #[derive(Debug, Deserialize)]
    #[allow(dead_code)]
    struct Inner {
        c: i32,
    }
    let scalars = scalars();
    let nested = nested();
    let cases = [
        (
            "null into i32",
            from_record_batch::<Int>(&scalars).err(),
            "row 1, column `int`",
        ),
        (
            "i32 into String",
            from_record_batch::<Text>(&scalars).err(),
            "row 0, column `int`",
        ),
        (
            "missing column",
            from_record_batch::<Absent>(&scalars).err(),
            "row 0",
        ),
        (
            "decimal of scale 2 into i64",
            from_record_batch::<Scaled>(&scalars).err(),
            "row 0, column `dec_10_2`",
        ),
        (
            "nested null",
            from_record_batch::<Nested>(&nested).err(),
            "row 1, column `nested.a.b.c`",
        ),
    ];
    for (name, error, at) in cases {
        let error = error.unwrap_or_else(|| panic!("{name}: no error"));
        assert_eq!(error.kind(), ErrorKind::Decode, "{name}");
        assert!(error.to_string().contains(at), "{name}: {error}");
    }
}

#[test]
fn a_decimal256_beyond_38_digits_is_an_error() {
    let wide = Decimal256Array::from(vec![
        i256::from_i128(i128::MAX).wrapping_mul(i256::from_i128(10))
    ])
    .with_precision_and_scale(76, 0)
    .unwrap();
    let batch = batch(vec![("wide", Arc::new(wide))]);
    let error = from_record_batch::<(Decimal,)>(&batch).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Decode);
    assert!(rows(&batch).is_err());
}

#[test]
fn a_variant_reads_as_its_value_and_metadata() {
    let marked = |name: &str| {
        let metadata = HashMap::from([("variant".to_string(), "true".to_string())]);
        Arc::new(Field::new(name, A::Binary, false).with_metadata(metadata))
    };
    let variant = StructArray::from(vec![
        (
            marked("value"),
            Arc::new(BinaryArray::from(vec![&b"\x0c\x01"[..]])) as ArrayRef,
        ),
        (
            marked("metadata"),
            Arc::new(BinaryArray::from(vec![&b"\x01\x00"[..]])) as ArrayRef,
        ),
    ]);
    let batch = batch(vec![("v", Arc::new(variant))]);
    assert_eq!(
        rows(&batch).unwrap()[0].values(),
        [Value::Variant {
            value: vec![12, 1],
            metadata: vec![1, 0],
        }]
    );
    #[derive(Deserialize)]
    struct Variant {
        value: Vec<u8>,
        metadata: Vec<u8>,
    }
    let read: Vec<(Variant,)> = from_record_batch(&batch).unwrap();
    assert_eq!(
        (read[0].0.value.as_slice(), read[0].0.metadata.as_slice()),
        (&[12, 1][..], &[1, 0][..])
    );
}
