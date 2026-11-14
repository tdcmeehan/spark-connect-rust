# Sessions and Configuration

A `Session` is a connection to a Spark Connect server and a session on it. Build one with
`Session::builder()`, and connect it with `connect().await`.

## Connecting

The connection string has the form `sc://host[:port][/;key=value;...]`, as in the other
clients (see [Configuration and Connection](../configuration.md)). Its parameters set the
token, TLS, the user id, the session id and the user agent, as the builder's methods do; any
other parameter is sent with each request as a gRPC header.

```{.rust .async}
use spark_connect_async::Session;

let session = Session::builder()
    .remote("sc://spark.example.com:443")
    .token(std::env::var("SPARK_CONNECT_TOKEN")?)
    .user_agent("nightly-report")
    .config("spark.sql.session.timeZone", "UTC")
    .connect()
    .await?;
```

A token turns on TLS, except to `localhost`; `use_ssl` sets TLS either way. `session_id`
connects to an existing session on the server, by its UUID. The configurations of `config` are
set when the session connects; if the server rejects one, `connect` releases the session and
returns the error.

`session.version().await?` is the server's Spark version, and `session.stop().await?` releases
the session on the server. A stopped session fails each later call with an error of kind
`Transport`.

## Runtime Configuration

`conf()` reads and writes the session's configuration on the server:

```{.rust .async}
let conf = session.conf();
conf.set("spark.sql.shuffle.partitions", "8").await?;
let partitions: Option<String> = conf.get("spark.sql.shuffle.partitions").await?;
let ansi = conf.get("spark.sql.ansi.enabled").await?;
conf.unset("spark.sql.shuffle.partitions").await?;
```

`get` answers the value that the session uses: the value set for the key, or else Spark's
default for it. It is `None` for a configuration that has neither, or for a key that is no
configuration. `set`, `set_all` and `unset` return the server's warnings, such as that a key
is deprecated.

## Tags and Interrupts

`with_tag` returns a handle to the same session whose operations also carry a tag. Interrupt
them from anywhere with `interrupt_tag`; `interrupt_all` interrupts every operation of the
session, and `interrupt_operation` one by its id. Each returns the ids of the operations it
interrupted.

```{.rust .async}
let report = session.with_tag("nightly-report")?;
let running = tokio::spawn(report.table("events").count());

let interrupted = session.interrupt_tag("nightly-report").await?;
println!("interrupted {interrupted:?}");
match running.await? {
    Ok(count) => println!("{count} events"),
    Err(error) => println!("stopped: {error}"),
}
```

An interrupted operation fails with an error of kind `Cancelled`. Dropping a future is the
other way to stop an operation (see [Results and Cancellation](results.md#cancellation)).

## Progress and the Requests of Executions

`on_progress` adds a handler of the progress of the session's executions. It is called with
each update of the server's progress of an execution that has tasks, and once more when the
execution ends, on the thread that polls the execution:

```{.rust .async}
use spark_connect_async::progress::ProgressUpdate;
use spark_connect_async::Session;

let session = Session::builder()
    .remote("sc://localhost:15002")
    .on_progress(|update: &ProgressUpdate| {
        let done: i64 = update.stages.iter().map(|stage| stage.num_completed_tasks).sum();
        let total: i64 = update.stages.iter().map(|stage| stage.num_tasks).sum();
        println!("{}: {done} of {total} tasks", update.operation_id);
    })
    .connect()
    .await?;
```

`execute_plan_hook` sets a function that gets each `ExecutePlanRequest` before it is sent, and
returns the request to send instead.

## Artifacts

`add_artifacts` uploads local files to the session: jars and class files to its classpath by
default, or, with `ArtifactKind`, files and archives.

```{.rust .async}
use spark_connect_async::session::ArtifactKind;

session.add_artifacts(["target/functions.jar"]).await?;
session
    .add_artifacts(["data/lookup.csv"])
    .kind(ArtifactKind::File)
    .await?;
```
