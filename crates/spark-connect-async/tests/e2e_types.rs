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

//! Results of every Spark type, from a Spark Connect server at `SPARK_REMOTE`, read into Rust
//! types and into `Value`s, and sent back as the rows of a DataFrame; each test passes without
//! running when it is not set.

use std::collections::{BTreeMap, HashMap};
use std::future::{Future, IntoFuture};
use std::panic::AssertUnwindSafe;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Utc};
use futures::FutureExt;
use serde::Deserialize;
use spark_connect_async::types::DataType;
use spark_connect_async::{col, lit, CalendarInterval, Decimal, Row, Session, Value};

/// The longest that a call to the server may take.
const WAIT: Duration = Duration::from_secs(120);

/// Waits for the call that `call` makes until the deadline. The call's future is made and boxed
/// in a frame of its own: made in the frame of a test, the futures of a debug build overflowed
/// the stack of its thread.
async fn within<F: IntoFuture>(call: impl FnOnce() -> F) -> F::Output {
    let call = Box::pin(call().into_future());
    tokio::time::timeout(WAIT, call)
        .await
        .expect("no answer before the deadline")
}

async fn session() -> Option<Session> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    Some(
        within(|| Session::builder().remote(remote).connect())
            .await
            .unwrap(),
    )
}

fn unique(prefix: &str) -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{prefix}_{}", now.as_nanos())
}

/// Runs `body`, then stops `session`, which drops its views, also when `body` panics.
async fn cleaned_up(session: &Session, body: impl Future<Output = ()>) {
    let outcome = AssertUnwindSafe(body).catch_unwind().await;
    let _ = tokio::time::timeout(WAIT, session.stop()).await;
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
}

const NESTED: &str = "ARRAY<STRUCT<k: STRING, v: MAP<STRING, ARRAY<BIGINT>>>>";

/// The columns of the table: a name, a Spark type, and the values of the first and the third
/// row; the second row is all `NULL`.
const COLUMNS: &[(&str, &str, &str, &str)] = &[
    ("boolean", "BOOLEAN", "true", "false"),
    ("byte", "TINYINT", "1", "-128"),
    ("short", "SMALLINT", "2", "-32768"),
    ("int", "INT", "3", "-2147483648"),
    ("long", "BIGINT", "4", "9223372036854775807"),
    ("float", "FLOAT", "0.5", "'Infinity'"),
    ("double", "DOUBLE", "-0.25", "'-Infinity'"),
    (
        "decimal",
        "DECIMAL(38, 9)",
        "'12345678901234567890123456789.123456789'",
        "'-99999999999999999999999999999.999999999'",
    ),
    ("string", "STRING", "'héllo'", "''"),
    ("binary", "BINARY", "X'00FF'", "X''"),
    ("date", "DATE", "'2024-02-29'", "'1969-12-31'"),
    (
        "timestamp",
        "TIMESTAMP",
        "'2024-02-29 12:34:56.123456'",
        "'1969-12-31 23:59:59.999999'",
    ),
    (
        "timestamp_ntz",
        "TIMESTAMP_NTZ",
        "'2024-02-29 12:34:56.123456'",
        "'0001-01-01 00:00:00'",
    ),
    ("time", "TIME(6)", "'12:34:56.123456'", "'23:59:59.999999'"),
    (
        "day_time",
        "INTERVAL DAY TO SECOND",
        "INTERVAL '1 02:03:04.5' DAY TO SECOND",
        "INTERVAL '-106751991 04:00:54.775808' DAY TO SECOND",
    ),
    (
        "year_month",
        "INTERVAL YEAR TO MONTH",
        "INTERVAL '1-2' YEAR TO MONTH",
        "INTERVAL '-178956970-8' YEAR TO MONTH",
    ),
    (
        "calendar",
        "INTERVAL",
        "make_interval(1, 2, 3, 4, 5, 6, 7.008009)",
        "make_interval(0, CAST(-2147483648 AS INT), 0, 2147483647, 0, 0, -9223372036.854775)",
    ),
    (
        "intervals",
        "MAP<STRING, ARRAY<INTERVAL DAY TO SECOND>>",
        "map('d', array(INTERVAL '1 02:03:04.5' DAY TO SECOND, NULL))",
        "map('max', array(INTERVAL '106751991 04:00:54.775807' DAY TO SECOND))",
    ),
    (
        "calendars",
        "STRUCT<i: INTERVAL, a: ARRAY<INTERVAL>, m: MAP<STRING, INTERVAL>>",
        "named_struct('i', make_interval(0, -14, 0, -3, -1, -2, -3.5), \
         'a', array(make_interval(0, 1, 0, -1), NULL), \
         'm', map('k', make_interval(0, 0, 0, 0, 0, 0, -0.000001)))",
        "named_struct('i', NULL, \
         'a', array(make_interval(0, 2147483647, 0, CAST(-2147483648 AS INT), \
         0, 0, 9223372036.854775)), 'm', map('k', NULL))",
    ),
    ("array", "ARRAY<INT>", "array(1, NULL, 3)", "array()"),
    ("map", "MAP<STRING, INT>", "map('b', 1, 'a', NULL)", "map()"),
    (
        "point",
        "STRUCT<x: INT, y: STRING>",
        "named_struct('x', 1, 'y', 'z')",
        "named_struct('x', NULL, 'y', NULL)",
    ),
    (
        "nested",
        NESTED,
        "array(named_struct('k', 'p', 'v', map('z', array(1L, NULL), 'y', array())))",
        "array(NULL, named_struct('k', NULL, 'v', NULL))",
    ),
    (
        "variant",
        "VARIANT",
        "parse_json('{\"a\":1}')",
        "parse_json('[true, null, \"x\"]')",
    ),
    ("void", "VOID", "NULL", "NULL"),
];

/// The query of the table; its column `ord` numbers the rows.
fn query() -> String {
    let row = |n: usize| {
        let values: Vec<String> = COLUMNS
            .iter()
            .map(|(name, spark_type, first, third)| {
                let value = [*first, "NULL", *third][n];
                format!("CAST({value} AS {spark_type}) AS {name}")
            })
            .collect();
        format!("SELECT {n} AS ord, {}", values.join(", "))
    };
    (0..3).map(row).collect::<Vec<_>>().join(" UNION ALL ")
}

/// A session whose table `name` holds a row of values of every Spark type, a row of `NULL`s,
/// and a row of extreme values.
async fn table(session: &Session, name: &str) {
    let conf = session.conf();
    within(|| conf.set("spark.sql.session.timeZone", "UTC"))
        .await
        .unwrap();
    within(|| conf.set("spark.sql.timeType.enabled", "true"))
        .await
        .unwrap();
    within(|| session.sql(format!("CREATE TEMPORARY VIEW {name} AS {}", query())))
        .await
        .unwrap();
}

#[derive(Deserialize, Debug, PartialEq)]
struct Point {
    x: Option<i32>,
    y: Option<String>,
}

/// A map of arrays of numbers, as its entries in order.
type Lists = Vec<(String, Option<Vec<Option<i64>>>)>;

#[derive(Deserialize, Debug, PartialEq)]
struct Entry {
    k: Option<String>,
    v: Option<Lists>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct Variant {
    value: Vec<u8>,
    metadata: Vec<u8>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct Calendars {
    i: Option<CalendarInterval>,
    a: Option<Vec<Option<CalendarInterval>>>,
    m: Option<Vec<(String, Option<CalendarInterval>)>>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct Types {
    ord: i32,
    boolean: Option<bool>,
    byte: Option<i8>,
    short: Option<i16>,
    int: Option<i32>,
    long: Option<i64>,
    float: Option<f32>,
    double: Option<f64>,
    decimal: Option<Decimal>,
    string: Option<String>,
    binary: Option<Vec<u8>>,
    date: Option<NaiveDate>,
    timestamp: Option<DateTime<Utc>>,
    timestamp_ntz: Option<NaiveDateTime>,
    time: Option<NaiveTime>,
    day_time: Option<i64>,
    year_month: Option<i32>,
    calendar: Option<CalendarInterval>,
    intervals: Option<Lists>,
    calendars: Option<Calendars>,
    array: Option<Vec<Option<i32>>>,
    map: Option<Vec<(String, Option<i32>)>>,
    point: Option<Point>,
    nested: Option<Vec<Option<Entry>>>,
    variant: Option<Variant>,
    void: Option<i32>,
}

/// The decimal, temporal and calendar-interval columns of the table read as text, and its map
/// by key.
#[derive(Deserialize, Debug, PartialEq)]
struct Other {
    decimal: Option<String>,
    date: Option<String>,
    timestamp: Option<String>,
    timestamp_ntz: Option<String>,
    time: Option<String>,
    calendar: Option<String>,
    map: Option<BTreeMap<String, Option<i32>>>,
}

/// The temporal columns of the table read as numbers.
#[derive(Deserialize, Debug, PartialEq)]
struct Numbers {
    date: Option<i32>,
    timestamp: Option<i64>,
    timestamp_ntz: Option<i64>,
    time: Option<i64>,
}

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn time(h: u32, m: u32, s: u32, micros: u32) -> NaiveTime {
    NaiveTime::from_hms_micro_opt(h, m, s, micros).unwrap()
}

fn decimal(unscaled: i128) -> Decimal {
    Decimal::new(unscaled, 38, 9).unwrap()
}

const FIRST_MICROS: i64 = 1_709_210_096_123_456;
const LEAP_DAY: i32 = 19_782;
/// The most microseconds of a calendar interval that the server can send: Arrow carries
/// nanoseconds in an `i64`.
const INTERVAL_MICROS: i64 = i64::MAX / 1_000;

fn expected() -> Vec<Types> {
    let instant = date(2024, 2, 29).and_time(time(12, 34, 56, 123_456));
    vec![
        Types {
            ord: 0,
            boolean: Some(true),
            byte: Some(1),
            short: Some(2),
            int: Some(3),
            long: Some(4),
            float: Some(0.5),
            double: Some(-0.25),
            decimal: Some(decimal(12_345_678_901_234_567_890_123_456_789_123_456_789)),
            string: Some("héllo".to_string()),
            binary: Some(vec![0, 255]),
            date: Some(date(2024, 2, 29)),
            timestamp: Some(instant.and_utc()),
            timestamp_ntz: Some(instant),
            time: Some(time(12, 34, 56, 123_456)),
            day_time: Some(93_784_500_000),
            year_month: Some(14),
            calendar: Some(CalendarInterval::new(14, 25, 18_367_008_009)),
            intervals: Some(vec![(
                "d".to_string(),
                Some(vec![Some(93_784_500_000), None]),
            )]),
            calendars: Some(Calendars {
                i: Some(CalendarInterval::new(-14, -3, -3_723_500_000)),
                a: Some(vec![Some(CalendarInterval::new(1, -1, 0)), None]),
                m: Some(vec![(
                    "k".to_string(),
                    Some(CalendarInterval::new(0, 0, -1)),
                )]),
            }),
            array: Some(vec![Some(1), None, Some(3)]),
            map: Some(vec![("b".to_string(), Some(1)), ("a".to_string(), None)]),
            point: Some(Point {
                x: Some(1),
                y: Some("z".to_string()),
            }),
            nested: Some(vec![Some(Entry {
                k: Some("p".to_string()),
                v: Some(vec![
                    ("z".to_string(), Some(vec![Some(1), None])),
                    ("y".to_string(), Some(vec![])),
                ]),
            })]),
            variant: Some(Variant {
                value: vec![2, 1, 0, 0, 2, 12, 1],
                metadata: vec![1, 1, 0, 1, 97],
            }),
            void: None,
        },
        Types {
            ord: 1,
            boolean: None,
            byte: None,
            short: None,
            int: None,
            long: None,
            float: None,
            double: None,
            decimal: None,
            string: None,
            binary: None,
            date: None,
            timestamp: None,
            timestamp_ntz: None,
            time: None,
            day_time: None,
            year_month: None,
            calendar: None,
            intervals: None,
            calendars: None,
            array: None,
            map: None,
            point: None,
            nested: None,
            variant: None,
            void: None,
        },
        Types {
            ord: 2,
            boolean: Some(false),
            byte: Some(i8::MIN),
            short: Some(i16::MIN),
            int: Some(i32::MIN),
            long: Some(i64::MAX),
            float: Some(f32::INFINITY),
            double: Some(f64::NEG_INFINITY),
            decimal: Some(decimal(-(10i128.pow(38) - 1))),
            string: Some(String::new()),
            binary: Some(vec![]),
            date: Some(date(1969, 12, 31)),
            timestamp: Some(DateTime::from_timestamp_micros(-1).unwrap()),
            timestamp_ntz: Some(date(1, 1, 1).and_time(time(0, 0, 0, 0))),
            time: Some(time(23, 59, 59, 999_999)),
            day_time: Some(i64::MIN),
            year_month: Some(i32::MIN),
            calendar: Some(CalendarInterval::new(i32::MIN, i32::MAX, -INTERVAL_MICROS)),
            intervals: Some(vec![("max".to_string(), Some(vec![Some(i64::MAX)]))]),
            calendars: Some(Calendars {
                i: None,
                a: Some(vec![Some(CalendarInterval::new(
                    i32::MAX,
                    i32::MIN,
                    INTERVAL_MICROS,
                ))]),
                m: Some(vec![("k".to_string(), None)]),
            }),
            array: Some(vec![]),
            map: Some(vec![]),
            point: Some(Point { x: None, y: None }),
            nested: Some(vec![None, Some(Entry { k: None, v: None })]),
            variant: Some(Variant {
                value: vec![3, 3, 0, 1, 2, 4, 4, 0, 5, 120],
                metadata: vec![1, 0, 0],
            }),
            void: None,
        },
    ]
}

/// The struct value `value`, as the names and the values of its fields.
fn fields(value: &Value) -> (Vec<&str>, &[Value]) {
    let Value::Struct(row) = value else {
        panic!("not a struct: {value:?}");
    };
    (
        row.names().iter().map(String::as_str).collect(),
        row.values(),
    )
}

fn text(value: &str) -> Value {
    Value::String(value.to_string())
}

#[tokio::test]
async fn every_spark_type_reads_into_its_rust_types_with_nulls_nesting_and_the_order_of_maps() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, async {
        let name = unique("b15_types");
        table(&session, &name).await;
        let df = session.table(&name).sort(["ord"]);

        let read: Vec<Types> = within(|| df.collect_as()).await.unwrap();
        assert_eq!(read, expected());

        let other: Vec<Other> = within(|| df.collect_as()).await.unwrap();
        let mut map = BTreeMap::new();
        map.insert("a".to_string(), None);
        map.insert("b".to_string(), Some(1));
        assert_eq!(
            other[0],
            Other {
                decimal: Some("12345678901234567890123456789.123456789".to_string()),
                date: Some("2024-02-29".to_string()),
                timestamp: Some("2024-02-29T12:34:56.123456Z".to_string()),
                timestamp_ntz: Some("2024-02-29T12:34:56.123456".to_string()),
                time: Some("12:34:56.123456".to_string()),
                calendar: Some(
                    "1 years 2 months 25 days 5 hours 6 minutes 7.008009 seconds".to_string()
                ),
                map: Some(map),
            }
        );
        assert_eq!(
            other[2].decimal.as_deref(),
            Some("-99999999999999999999999999999.999999999")
        );
        let texts: Vec<(Option<String>,)> =
            other.iter().map(|row| (row.calendar.clone(),)).collect();
        let spark_texts: Vec<(Option<String>,)> =
            within(|| df.select([col("calendar").cast("string")]).collect_as())
                .await
                .unwrap();
        assert_eq!(texts, spark_texts);
        let DataType::Struct { fields: columns } = within(|| df.schema()).await.unwrap() else {
            panic!("the schema is not a struct");
        };
        let calendar = columns.iter().find(|column| column.name == "calendar");
        assert_eq!(
            calendar.map(|column| &column.data_type),
            Some(&DataType::CalendarInterval)
        );
        let numbers: Vec<Numbers> = within(|| df.collect_as()).await.unwrap();
        assert_eq!(
            numbers[0],
            Numbers {
                date: Some(LEAP_DAY),
                timestamp: Some(FIRST_MICROS),
                timestamp_ntz: Some(FIRST_MICROS),
                time: Some(45_296_123_456_000),
            }
        );
        assert_eq!(
            numbers[2],
            Numbers {
                date: Some(-1),
                timestamp: Some(-1),
                timestamp_ntz: Some(-62_135_596_800_000_000),
                time: Some(86_399_999_999_000),
            }
        );
        type Map = Option<HashMap<String, Option<i32>>>;
        let maps: Vec<(Map,)> = within(|| df.select(["map"]).collect_as()).await.unwrap();
        let map = maps[0].0.as_ref().unwrap();
        assert_eq!((map.len(), map["b"]), (2, Some(1)));

        let rows: Vec<Row> = within(|| df.collect_rows()).await.unwrap();
        let names: Vec<&str> = std::iter::once("ord")
            .chain(COLUMNS.iter().map(|(name, ..)| *name))
            .collect();
        assert_eq!(rows[0].names(), names);
        let first = &rows[0];
        let value = |name: &str| first.get_by_name(name).unwrap();
        assert_eq!(
            value("decimal"),
            &Value::Decimal(decimal(12_345_678_901_234_567_890_123_456_789_123_456_789))
        );
        assert_eq!(
            value("timestamp"),
            &Value::Timestamp(DateTime::from_timestamp_micros(FIRST_MICROS).unwrap())
        );
        assert_eq!(
            value("day_time"),
            &Value::DayTimeInterval(TimeDelta::microseconds(93_784_500_000))
        );
        assert_eq!(value("year_month"), &Value::YearMonthInterval(14));
        assert_eq!(
            value("calendar"),
            &Value::CalendarInterval(CalendarInterval::new(14, 25, 18_367_008_009))
        );
        assert_eq!(
            value("map"),
            &Value::Map(vec![(text("b"), Value::Int(1)), (text("a"), Value::Null)])
        );
        assert_eq!(
            fields(value("point")),
            (vec!["x", "y"], &[Value::Int(1), text("z")][..])
        );
        let Value::Array(nested) = value("nested") else {
            panic!("not an array: {:?}", value("nested"));
        };
        let (names, values) = fields(&nested[0]);
        assert_eq!(names, ["k", "v"]);
        assert_eq!(
            values[1],
            Value::Map(vec![
                (text("z"), Value::Array(vec![Value::Long(1), Value::Null])),
                (text("y"), Value::Array(vec![])),
            ])
        );
        assert_eq!(
            value("variant"),
            &Value::Variant {
                value: vec![2, 1, 0, 0, 2, 12, 1],
                metadata: vec![1, 1, 0, 1, 97],
            }
        );
        assert!(rows[1].values()[1..].iter().all(|v| *v == Value::Null));
        assert_eq!(
            rows[2].get_by_name("day_time"),
            Some(&Value::DayTimeInterval(TimeDelta::microseconds(i64::MIN)))
        );
        assert_eq!(rows[2].get_by_name("map"), Some(&Value::Map(Vec::new())));
    })
    .await;
}

#[tokio::test]
async fn the_rows_of_every_spark_type_sent_back_as_a_dataframe_read_the_same() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, async {
        let name = unique("b15_types");
        table(&session, &name).await;
        // Local rows have no encoding of INTERVAL YEAR TO MONTH or of INTERVAL.
        let sendable = COLUMNS
            .iter()
            .map(|(name, ..)| *name)
            .filter(|name| !["year_month", "calendar", "calendars"].contains(name));
        let df = session
            .table(&name)
            .sort(["ord"])
            .select(std::iter::once("ord").chain(sendable));
        let schema = within(|| df.schema()).await.unwrap();
        let rows: Vec<Row> = within(|| df.collect_rows()).await.unwrap();

        let sent = within(|| {
            session.create_dataframe(rows.iter().map(|row| row.values().to_vec()), schema.clone())
        })
        .await
        .unwrap();
        assert_eq!(within(|| sent.schema()).await.unwrap(), schema);
        assert_eq!(within(|| sent.collect_rows()).await.unwrap(), rows);
    })
    .await;
}

#[tokio::test]
async fn calendar_intervals_written_as_their_text_cast_to_interval_read_back_the_same() {
    let Some(session) = session().await else {
        return;
    };
    cleaned_up(&session, async {
        let intervals = [
            CalendarInterval::new(14, 25, 18_367_008_009),
            CalendarInterval::new(-14, -3, -3_723_500_000),
            CalendarInterval::new(1, -1, 0),
            CalendarInterval::new(0, 0, -1),
            CalendarInterval::new(0, 0, 0),
            CalendarInterval::new(i32::MAX, i32::MIN, INTERVAL_MICROS),
            CalendarInterval::new(i32::MIN, i32::MAX, -INTERVAL_MICROS),
        ];
        let written = session.range(1).select(
            intervals
                .iter()
                .map(|interval| lit(interval.to_string()).cast(DataType::CalendarInterval)),
        );
        let rows: Vec<Row> = within(|| written.collect_rows()).await.unwrap();
        let read: Vec<Value> = intervals.into_iter().map(Value::CalendarInterval).collect();
        assert_eq!(rows[0].values(), read);
    })
    .await;
}
