# Errors

Every call returns `spark_connect_async::Result<T>`, whose error is `Error`: one type for the
errors of the client and of the server. `kind()` says what kind of error it is, and an error
that the server raised also carries its condition, such as `TABLE_OR_VIEW_NOT_FOUND`, its SQL
state, its message parameters and, for an error in SQL text, where in the text it occurred.

```{.rust .async}
use spark_connect_async::ErrorKind;

match session.table("no_such_table").count().await {
    Ok(count) => println!("{count} rows"),
    Err(error) if error.kind() == ErrorKind::Analysis => {
        println!("{}", error.message());
        println!("condition: {:?}", error.error_class());
        println!("SQL state: {:?}", error.sql_state());
        for (name, value) in error.message_parameters() {
            println!("  {name} = {value}");
        }
    }
    Err(error) => return Err(error.into()),
}
```

The kinds are:

| `ErrorKind` | When |
|---|---|
| `InvalidArgument` | An argument that the client or the server rejected, such as a malformed connection string |
| `Transport` | The server couldn't be reached, refused the connection or its credentials, or the session no longer exists |
| `Analysis` | The server couldn't resolve the plan, such as a missing table or column |
| `Parse` | The server couldn't parse SQL text |
| `Execution` | The plan failed while it ran |
| `Cancelled` | The operation was interrupted |
| `Timeout` | The operation didn't finish within its deadline |
| `Decode` | A result couldn't be read into the requested Rust type |
| `Unsupported` | The server or the client doesn't support the operation |
| `Internal` | An error inside the client or the server that no other kind describes |

`ErrorKind` is non-exhaustive, so a `match` on it needs a `_` arm.

## Where in SQL Text

`query_contexts()` lists where in SQL text an error occurred, as the server reports it:

```{.rust .async}
use spark_connect_async::ErrorKind;

if let Err(error) = session.sql("SELECT nme FROM VALUES (1) AS t(name)").await {
    for context in error.query_contexts() {
        println!("at {}..{}: {}", context.start_index, context.stop_index, context.fragment);
    }
}
```

## The Rest of an Error

- `grpc_code()` is the gRPC status code of the response that carried the error;
- `operation_id()` is the id of the operation that failed;
- `server_stack_trace()` is the server's stack trace, when the server sends one;
- `breaking_change()` is the change of the server's behavior that the error comes from, with
  how to migrate and a configuration that restores the previous behavior, when the server
  reports one.

An error of kind `Decode` names the row, and the field of the column down to the value, that
failed. `Error` implements `std::error::Error`, so `?` converts it into a
`Box<dyn std::error::Error>`.
