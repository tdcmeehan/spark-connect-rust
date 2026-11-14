# Types and Values

## Reading Results into Rust Types

`collect_as` and `stream_as` read rows into any type that implements serde's `Deserialize`
(see [Typed Rows](results.md#typed-rows)). Each Spark type reads into these Rust types:

| Spark type | Rust types |
|---|---|
| `BOOLEAN`, `TINYINT` ... `BIGINT`, `FLOAT`, `DOUBLE` | `bool`, the integers, the floats |
| `DECIMAL` | `Decimal` (exact), `String` (exact text) or `f64`; an integer if the scale is not positive |
| `STRING` | `String`, `&str` |
| `BINARY` | `Vec<u8>`, `&[u8]` |
| `DATE` | `chrono::NaiveDate` or `String`; `i32` days since 1970-01-01 |
| `TIMESTAMP` | `chrono::DateTime<Utc>` or `String` (RFC 3339); `i64` microseconds since the epoch |
| `TIMESTAMP_NTZ` | `chrono::NaiveDateTime` or `String`; `i64` microseconds |
| `TIME` | `chrono::NaiveTime` or `String`; `i64` nanoseconds since midnight |
| `INTERVAL DAY TO SECOND` | `i64` microseconds |
| `INTERVAL YEAR TO MONTH` | `i32` months |
| `INTERVAL`, the calendar interval | `CalendarInterval` or `String` (Spark's text) |
| `ARRAY` | `Vec<T>`, or any sequence |
| `MAP` | any map, or `Vec<(K, V)>` in the server's order of the entries |
| `STRUCT` | a struct by field name, a tuple by position, or a map |
| the vectors of Spark ML | `ml::Vector` |
| `VARIANT` | a struct of `value` and `metadata`, both `Vec<u8>` |

A `NULL` reads into `None` of an `Option`, and is an error elsewhere.

```{.rust .async}
use chrono::{DateTime, NaiveDate, Utc};
use spark_connect_async::Decimal;

#[derive(serde::Deserialize)]
struct Order {
    id: i64,
    placed: DateTime<Utc>,
    due: Option<NaiveDate>,
    total: Decimal,
    items: Vec<String>,
    quantities: Vec<(String, i32)>,
}

let orders: Vec<Order> = session
    .sql(
        "SELECT 1L AS id, TIMESTAMP'2024-03-01 10:00:00' AS placed, DATE'2024-03-08' AS due, \
         12.50BD AS total, array('pen', 'ink') AS items, map('pen', 2, 'ink', 1) AS quantities",
    )
    .await?
    .collect_as()
    .await?;
```

A map keeps its keys' types and the order of its entries, so reading it into a `Vec` of pairs
keeps what a map type such as `HashMap` would lose.

## Values and Rows

`collect_rows`, `take` and `first` return `Row`s, and a `Row` holds a `Value` of each column,
by position and by name. `Value` has a variant for each Spark type, and `Value::Null`:

```{.rust .async}
use spark_connect_async::Value;

for row in session.table("people").collect_rows().await? {
    match row.get_by_name("age") {
        Some(Value::Int(age)) => println!("{age}"),
        Some(Value::Null) | None => println!("no age"),
        Some(other) => println!("unexpected {other:?}"),
    }
}
```

## Decimals

`Decimal` is an exact decimal of Spark's `DECIMAL(precision, scale)`: an `i128` unscaled
value, a precision and a scale, so it holds every Spark decimal, up to 38 digits. It parses
from text and prints as text:

```{.rust .async}
use spark_connect_async::{col, lit, Decimal};

let price: Decimal = "19.99".parse()?;
assert_eq!((price.unscaled(), price.precision(), price.scale()), (1999, 4, 2));
let total = col("quantity") * lit(price);
```

## Literals

`lit` makes a column of a literal of a Rust value, of the Spark type of its Rust type: `i32` is
an `INT`, `i64` a `BIGINT`, `&str` a `STRING`, `Decimal` a `DECIMAL`, `chrono::NaiveDate` a
`DATE`, `chrono::DateTime<Utc>` a `TIMESTAMP`, `chrono::TimeDelta` an
`INTERVAL DAY TO SECOND`, a `Vec` an `ARRAY`, and `None::<T>` a `NULL` of `T`'s type:

```{.rust .async}
use chrono::{NaiveDate, TimeDelta};
use spark_connect_async::{col, lit};

let since = NaiveDate::from_ymd_opt(2024, 1, 1).expect("a date");
let recent = session.table("orders").filter(col("placed").ge(lit(since)));
let later = col("placed") + lit(TimeDelta::hours(36));
let tags = lit(vec!["new", "sale"]);
let unknown = lit(None::<i64>);
```

Spark's calendar interval, `INTERVAL`, and its year-month intervals have no literal of `lit`.
To write a calendar interval, cast its text, which `CalendarInterval` prints as Spark writes
it:

```{.rust .async}
use spark_connect_async::types::DataType;
use spark_connect_async::{lit, CalendarInterval};

let interval = CalendarInterval::new(14, 25, 18_367_008_009);
assert_eq!(
    interval.to_string(),
    "1 years 2 months 25 days 5 hours 6 minutes 7.008009 seconds"
);
let written = session
    .range(1)
    .select([lit(interval.to_string()).cast(DataType::CalendarInterval)]);
let read: Vec<CalendarInterval> = written.collect_as().await?;
```

A calendar interval counts months, days and microseconds each on its own, since a month has no
fixed number of days, nor a day of microseconds.

## Schemas

`spark_connect_async::types::DataType` is a Spark type. `schema()` returns a DataFrame's, as a
struct type, and `DataType::from_ddl` parses one in Spark SQL's syntax:

```{.rust .async}
use spark_connect_async::types::DataType;

let schema = DataType::from_ddl("id BIGINT, name STRING, tags ARRAY<STRING>")?;
let people = session.read().schema(schema).json("/data/people.json");
let actual = session.table("people").schema().await?;
```
