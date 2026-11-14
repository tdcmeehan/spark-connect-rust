# Reading and Writing

## Reading

`session.read()` is a `DataFrameReader`: set the format, the schema and the options, then name
what to read. A read returns a DataFrame, which runs at its actions:

```{.rust .async}
use spark_connect_async::{col, lit};

let people = session
    .read()
    .schema("name STRING, age INT, city STRING")
    .option("header", true)
    .option("delimiter", ";")
    .csv("/data/people.csv");
let events = session.read().format("parquet").load("/data/events");
let errors = session
    .read()
    .json("/data/logs/*.json")
    .filter(col("level").eq(lit("ERROR")));
let table = session.read().table("people");
```

An option's value is a string, a `bool` or a number, and `options` sets several at once. The
reader also reads `orc`, `text`, `xml`, `jdbc`, several paths with `load_paths`, a table's
changes with `changes`, and `parse` parses a DataFrame's column of CSV or JSON text.

## Writing

`write()` is a `DataFrameWriter`. Its saves are futures that run the write:

```{.rust .async}
use spark_connect_async::readwriter::SaveMode;

let people = session.table("people");
people
    .write()
    .mode(SaveMode::Overwrite)
    .partition_by(["city"])
    .parquet("/data/people")
    .await?;
people
    .write()
    .format("csv")
    .option("header", true)
    .save("/data/people-csv")
    .await?;
people.write().mode(SaveMode::Append).save_as_table("people_copy").await?;
```

`SaveMode` says what a write does to data that is already there: `Append`, `Overwrite`,
`ErrorIfExists`, the default, or `Ignore`. `bucket_by`, `sort_by` and `cluster_by` lay out the
data, and `insert_into` adds to an existing table.

`write_to(table)` is the `DataFrameWriterV2` of a table of a catalog:

```{.rust .async}
let events = session.table("staging_events");
events.write_to("events").using("parquet").create_or_replace().await?;
events.write_to("events").append().await?;
```

## MERGE

`merge_into` merges a DataFrame's rows into a table: the rows that match by a condition update
or delete the table's, and the others insert into it.

```{.rust .async}
use spark_connect_async::col;

let updates = session.table("updates");
updates
    .merge_into("people", col("people.id").eq(col("updates.id")))
    .when_matched(None)
    .update([("name", col("updates.name"))])
    .when_not_matched(None)
    .insert_all()
    .await?;
```

## Local Data

`create_dataframe` makes a DataFrame of rows of `Value`s, of a schema, and
`create_dataframe_from_arrow` one of Arrow record batches:

```{.rust .async}
use spark_connect_async::types::DataType;
use spark_connect_async::Value;

let schema = DataType::from_ddl("id BIGINT, name STRING")?;
let rows = [
    vec![Value::Long(1), Value::String("Ada".to_string())],
    vec![Value::Long(2), Value::Null],
];
let people = session.create_dataframe(rows, schema).await?;

let batches = people.collect().await?;
let copy = session.create_dataframe_from_arrow(batches).await?;
```

Small rows travel in the plan, and large ones as artifacts of the session. A row of another
width than the schema's, or a value that its field's type can't hold, fails with an error of
kind `InvalidArgument`.
