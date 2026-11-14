# The Blocking API

The `blocking` feature adds `spark_connect_async::blocking`: the client with calls that block
until the server answers, for code that doesn't run async, such as a command-line tool, a
script or a synchronous callback. In async code, await the async API instead: a blocking call
holds its thread for as long as the server takes.

```toml
[dependencies]
apache-spark-connect-async = { version = "4.3", features = ["blocking"] }
```

A blocking `Session` holds an async one. Its calls that talk to the server block and return
what the async call's future returns. What sends nothing, such as the transformations of a
query, is the async API's, reached with `as_async()`, and `.blocking()` turns an async value
into its blocking form:

```{.rust .async}
use spark_connect_async::readwriter::SaveMode;
use spark_connect_async::{col, lit, Session};

let session = Session::builder()
    .remote("sc://localhost:15002")
    .blocking()
    .connect()?;
let people = session.as_async().sql("SELECT name, age FROM people").run()?;
println!("{} people", people.count()?);

let adults = people.as_async().filter(col("age").ge(lit(18))).blocking();
let names: Vec<(String,)> = adults.as_async().select(["name"]).blocking().collect_as()?;
adults
    .as_async()
    .write()
    .mode(SaveMode::Overwrite)
    .blocking()
    .parquet("/tmp/adults")?;

for table in session.catalog().as_async().list_tables().run()? {
    println!("{}", table.name);
}
session.stop()?;
```

The rules, in short:

- The blocking `Session`, `DataFrame`, `Catalog`, `RuntimeConfig`, `GroupedData`,
  `StatFunctions`, `StreamingQueryManager`, `StreamingQuery`, the fitted models of Spark ML and
  the registration of Rust UDFs have the methods of their async types that return futures,
  with the same names and arguments, and those block: `df.count()?` here is
  `df.count().await?` there.
- A builder that async code awaits, such as `sql`'s or `list_tables`'s, has a `run()` method,
  which sends it and blocks. A builder whose last step is a method, such as a session
  builder's `connect` or a writer's saves, has a `.blocking()` form with that method.
- What the server sends in parts is an iterator, each of whose items blocks: the record
  batches of a query, the events of streaming queries, and those of a pipeline's run.
- A parameter that takes a DataFrame takes an async one: pass a blocking DataFrame's
  `as_async()`. `From` converts both ways too.

## Threads and Runtimes

The blocking calls share one Tokio runtime, which starts at the first call and lasts as long
as the process; its threads drive the connections, the timers and the cleanup that a drop
sends. A call polls its future on the calling thread, so a session's progress handlers run
there.

A call doesn't panic outside a runtime, nor inside one of either flavor, in a task or in
`spawn_blocking`. The exception is the thread of a current-thread runtime that has entered the
handle of a multi-thread runtime, which Tokio can't tell apart from a worker of that runtime:
there a call panics. A call blocks the thread it runs on: on a worker of a multi-thread
runtime, the worker's other tasks move to another thread first; on the thread of a
current-thread runtime, the runtime's other tasks wait until it returns.

A session connected with the async API keeps its connection on the runtime it connected on, so
its blocking form needs that runtime to keep running. A blocking call on the thread of a
current-thread runtime that the session connected on stalls, since the connection can't run
until the call returns.

A blocking call can't be given up the way a dropped future is. Some wait for as long as the
server does, such as `await_termination` without a timeout, which ends when the query
terminates or when another thread interrupts the session's operations. The documentation of
the `blocking` module, in the API reference on
[docs.rs](https://docs.rs/apache-spark-connect-async), lists them and what ends each.
