# Structured Streaming

`session.read_stream()` reads a stream as a DataFrame, and `write_stream()` starts the
`StreamingQuery` that writes a streaming DataFrame to a sink.

```{.rust .async}
use std::time::Duration;

use spark_connect_async::streaming::{OutputMode, Trigger};

let numbers = session
    .read_stream()
    .format("rate")
    .option("rowsPerSecond", 10)
    .load_without_path();
let query = numbers
    .write_stream()
    .format("memory")
    .query_name("numbers")
    .output_mode(OutputMode::Append)
    .trigger(Trigger::ProcessingTime("1 second".to_string()))
    .start_without_path()
    .await?;

query.process_all_available().await?;
if let Some(progress) = query.last_progress().await? {
    println!("batch {} read {} rows", progress.batch_id, progress.num_input_rows);
}
let ended: bool = query.await_termination(Some(Duration::from_secs(10))).await?;
query.stop().await?;
```

`start` writes to a path, `start_without_path` to a sink that has none, and `to_table` to a
table. A trigger runs micro-batches at an interval (`ProcessingTime`), once (`Once`), or on
the data that is there when the query starts (`AvailableNow`); `Continuous` and `RealTime`
process continuously, or in real time.

## Queries

A `StreamingQuery` has its `id`, `run_id` and `name`, and asks the server for the rest:

- `status` and `is_active`;
- `last_progress` and `recent_progress`, typed, with the JSON text that the server sent;
- `await_termination`, with a timeout or without one, which answers whether the query has
  terminated, and fails with the query's error if it failed;
- `process_all_available`, `explain`, `exception` and `stop`.

Dropping the future of a wait gives up the wait on the server.

## The Queries of a Session

`session.streams()` is the `StreamingQueryManager`: `active` lists the running queries, `get`
finds one by its id, `await_any_termination` waits for one to terminate, and
`reset_terminated` forgets those that have.

`events` subscribes to the events of the session's queries, and returns them as a stream:

```{.rust .async}
use futures::TryStreamExt;
use spark_connect_async::streaming::QueryEvent;

let mut events = session.streams().events().await?;
while let Some(event) = events.try_next().await? {
    match event {
        QueryEvent::Started(started) => println!("started {}", started.id),
        QueryEvent::Progress(progress) => println!("{} rows", progress.num_input_rows),
        QueryEvent::Terminated(ended) => println!("ended {}: {:?}", ended.id, ended.exception),
        _ => {}
    }
}
```

The server keeps one subscription for a session; dropping the stream removes it.

`foreach_batch`, `foreach` and the listeners of PySpark are not in this API: Spark Connect
runs those as Python or Scala functions.
