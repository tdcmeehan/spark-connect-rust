# The Async API

`apache-spark-connect-async` is an async Rust client for Spark Connect. It builds the same
plans as the PySpark-shaped `apache-spark-connect` crate, through the same core, with an API
shaped for Rust programs that run on [Tokio](https://tokio.rs/):

- Every call that talks to the server is awaited: it returns a future, or a builder of its
  options that `.await` runs. It sends nothing until it is awaited, and its future is
  `Send + 'static`, so it can be spawned.
- Dropping the future of a query or a command gives up the call, and stops its operation on
  the server (see [Results and Cancellation](results.md)).
- Results are typed: Arrow record batches, rows of [values](types.md), or any type that
  implements serde's `Deserialize`, collected at once or streamed.
- Errors are one type, `Error`, with the server's error class, SQL state and message
  parameters (see [Errors](errors.md)).
- Arguments are Rust values: a column is `impl Into<Column>`, where a `&str` is the column of
  that name, `lit` makes a literal of a Rust value, and optional arguments are builder methods
  with defaults.

Code that doesn't run async can use the same client through the `blocking` feature (see
[The Blocking API](blocking.md)). The `apache-spark-connect` crate and the `pyspark` package
are unchanged by it; [Which client am I using?](../which-client.md) compares them.

## Add It to a Project

```toml
[dependencies]
apache-spark-connect-async = "4.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Building it needs what the other crate needs (see [Installation](../installation.md)): Rust
1.88 or later, or 1.91 or later with the `polars` feature, and `protobuf-compiler`. Its
features add the rest of the API, and none is on by default:

| Feature | What it adds |
|---|---|
| `blocking` | The `blocking` module: the client with calls that block, for code that isn't async |
| `datafusion` | `DataFrame::to_datafusion`, the result as a [DataFusion](https://datafusion.apache.org/) DataFrame |
| `polars` | `DataFrame::to_polars`, the result as a [Polars](https://pola.rs/) DataFrame |
| `rust-udf` | The `rust_udf` module: Rust functions that Spark runs as UDFs (see [Rust UDFs](udfs.md)) |

```toml
[dependencies]
apache-spark-connect-async = { version = "4.3", features = ["blocking", "polars"] }
```

The API reference is on [docs.rs](https://docs.rs/apache-spark-connect-async), with every
feature on.

## A First Program

With a Spark Connect server on `localhost:15002` (see
[Configuration and Connection](../configuration.md) to start one):

```{.rust .async}
use spark_connect_async::{col, lit, Session};

#[tokio::main]
async fn main() -> spark_connect_async::Result<()> {
    let session = Session::builder()
        .remote("sc://localhost:15002")
        .connect()
        .await?;

    let multiples = session
        .range(1_000_000)
        .select([(col("id") * lit(2)).alias("x")])
        .filter((col("x") % lit(3)).eq(lit(0)));

    println!("count = {}", multiples.count().await?);
    multiples.show().await?;

    session.stop().await?;
    Ok(())
}
```

## How a Call Runs

A `DataFrame` is a plan. Its transformations, such as `select` and `filter`, return a new
DataFrame and send nothing. Its actions, such as `count`, `collect` and `show`, return futures
that send the plan when they are awaited:

```{.rust .async}
use spark_connect_async::{col, lit};

let adults = session.table("people").filter(col("age").ge(lit(18)));
let count = adults.count();
let n = count.await?;
```

A `Session` is a handle: its clones share the connection and the session on the server, and a
DataFrame keeps its session. The futures own what they need, so a task can run them:

```{.rust .async}
let total = tokio::spawn(session.range(1000).count());
println!("{} rows", total.await??);
```

## Where to Next

- [Sessions and Configuration](sessions.md): connection strings, authentication, runtime
  configuration, tags and interrupts.
- [DataFrames, Columns and SQL](dataframes.md): transformations, functions, joins, grouping,
  windows and SQL with arguments.
- [Results and Cancellation](results.md): record batches, rows, typed rows, result streams,
  and cancellation by dropping.
- [Types and Values](types.md): how Spark's types read into Rust types, and literals.
- [Reading and Writing](data-sources.md), [Catalog](catalog.md),
  [Structured Streaming](streaming.md), [Machine Learning and Pipelines](ml.md) and
  [Rust UDFs](udfs.md).
- [Errors](errors.md) and [The Blocking API](blocking.md).
