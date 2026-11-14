# Results and Cancellation

The actions of a `DataFrame` run its plan on the server, or ask the server about it. Each
returns a future, which sends nothing until it is awaited.

## Collecting

```{.rust .async}
let people = session.table("people");

let batches = people.collect().await?;
let rows = people.collect_rows().await?;
let first = people.first().await?;
let five = people.take(5).await?;
let n: i64 = people.count().await?;
let empty: bool = people.is_empty().await?;
```

`collect` returns the Arrow `RecordBatch`es of the result, and `collect_rows` its rows as
`Row`s of `Value`s (see [Types and Values](types.md)). `first` is `None` for an empty result.

`show` prints the first rows as a table. Its builder sets the rows, the cut of long values and
a vertical layout, and `to_string` returns the table instead:

```{.rust .async}
let people = session.table("people");
people.show().await?;
people.show().rows(5).truncate(0).vertical().await?;
let table: String = people.show().to_string().await?;
```

`schema`, `columns` and `explain` ask the server about the plan without running it:

```{.rust .async}
use spark_connect_async::actions::ExplainMode;

let people = session.table("people");
let schema = people.schema().await?;
let names: Vec<String> = people.columns().await?;
println!("{}", people.explain(ExplainMode::Formatted).await?);
```

## Typed Rows

`collect_as` reads each row into a value of any type that implements serde's `Deserialize`: a
struct, by the names of the columns, which skips the columns that it doesn't name; a tuple, by
position; or, for a row of one column, that column's value. A `NULL` reads into `None` of an
`Option`:

```{.rust .async}
#[derive(serde::Deserialize)]
struct Person {
    name: String,
    age: Option<i32>,
}

let people: Vec<Person> = session.table("people").collect_as().await?;
let pairs: Vec<(String, Option<i32>)> = session
    .table("people")
    .select(["name", "age"])
    .collect_as()
    .await?;
let ids: Vec<i64> = session.range(5).collect_as().await?;
```

A value that the type can't hold, such as a `NULL` outside an `Option` or a `STRING` into an
`i64`, fails with an error of kind `Decode`, which names the row and the column. The Rust
types that each Spark type reads into are in [Types and Values](types.md).

## Result Streams

`stream` returns the record batches as the server sends them, once the server has accepted the
query, so a large result never has to fit in memory. `stream_as` streams rows as values of a
type, the way `collect_as` reads them:

```{.rust .async}
use futures::TryStreamExt;

let mut batches = session.table("events").stream().await?;
while let Some(batch) = batches.try_next().await? {
    println!("{} rows", batch.num_rows());
}

let ids = session.range(1_000_000).stream_as::<i64>().await?;
let mut ids = std::pin::pin!(ids);
while let Some(id) = ids.try_next().await? {
    if id > 10 {
        break;
    }
}
```

A stream also has the schema of its result, and what the execution reported, such as its
metrics, once it has read them.

## Cancellation

Every query or command is an operation on the server. Dropping its future before the server
has accepted the operation interrupts it, and dropping a result stream, or the future of an
action that reads one, before its end releases it, which stops it too. So the usual ways to
give up on a future, such as a timeout or `select!`, stop the work on the server:

```{.rust .async}
use std::time::Duration;

let count = session.table("events").count();
match tokio::time::timeout(Duration::from_secs(30), count).await {
    Ok(count) => println!("{} events", count?),
    Err(_) => println!("gave up after 30 seconds"),
}
```

So does dropping a stream before its end: once the loop above breaks, dropping `ids` stops
the rest of its query. Interrupts by tag are in
[Sessions and Configuration](sessions.md#tags-and-interrupts).

## Writing and Caching

`write` and `write_to` save a result (see [Reading and Writing](data-sources.md)), and
`create_temp_view` names it for SQL:

```{.rust .async}
use spark_connect_async::actions::{Checkpoint, TempView};

let people = session.table("people");
people
    .create_temp_view("people_view", TempView::default().or_replace())
    .await?;
let adults = session.sql("SELECT * FROM people_view WHERE age >= 18").await?;

people.cache().await?;
let kept = people.checkpoint(Checkpoint::default()).await?;
```

`cache` and `persist` keep a DataFrame's rows, in memory or on disk, for its later queries, and
`unpersist` removes them. A checkpoint is a
DataFrame of the rows as the server keeps them, whose plan doesn't depend on the original's;
the server removes them once the last handle of that DataFrame is dropped.

## DataFusion and Polars

With the `datafusion` or `polars` feature, `to_datafusion` and `to_polars` return the result
as a DataFrame of that library:

```{.rust .no-run}
let frame: polars::frame::DataFrame = session.table("people").to_polars().await?;

let context = datafusion::prelude::SessionContext::new();
let table = session.table("people").to_datafusion(&context).await?;
```
