# Rust UDFs

With the `rust-udf` feature, the async API runs Rust functions as Spark UDFs on the executors,
compiled to WebAssembly, as [Rust UDFs via WebAssembly](../udfs.md) describes for the other
crate: the supported types, the preconditions and how it works are the same.

## Writing Them

Annotate a module of plain functions with `#[spark_udf(crate = spark_connect_async)]`: the
`crate` option makes the macro generate the async API's columns. A build script compiles the
module to WebAssembly, which needs the `wasm32-unknown-unknown` target
(`rustup target add wasm32-unknown-unknown`):

```toml
# Cargo.toml
[dependencies]
spark-connect-macros = { package = "apache-spark-connect-macros", version = "4.3" }

# host-only: the client is not pulled into the wasm build
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
apache-spark-connect-async = { version = "4.3", features = ["rust-udf"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }

[build-dependencies]
spark-connect-build  = { package = "apache-spark-connect-build",  version = "4.3" }
spark-connect-macros = { package = "apache-spark-connect-macros", version = "4.3" }
```

```{.rust .no-run}
// build.rs
fn main() {
    spark_connect_build::embed_wasm_udf("src/main.rs");
}
```

```{.rust .no-run}
// src/main.rs
use spark_connect_macros::spark_udf;

#[spark_udf(crate = spark_connect_async)]
mod udfs {
    pub fn add_one(x: i64) -> i64 {
        x + 1
    }

    pub fn label(name: Option<String>, n: i64) -> String {
        format!("{}#{n}", name.unwrap_or_default())
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() -> spark_connect_async::Result<()> {
    use spark_connect_async::Session;

    let session = Session::builder()
        .remote("sc://localhost:15002")
        .connect()
        .await?;

    let numbers = session.range(5);
    numbers
        .select([udf::add_one("id")?.alias("plus_one")])
        .show()
        .await?;

    session.udf().register("add_one", &udf::add_one_udf()).await?;
    session.sql("SELECT add_one(id) FROM range(5)").await?.show().await?;
    Ok(())
}
```

For each function, the macro generates `udf::<name>(...)`, the column of the function applied
to one column per argument, whose arity the compiler checks, and `udf::<name>_udf()`, its
`UserDefinedFunction`.

## Using Them

A `UserDefinedFunction` applies to columns with `call`, and `session.udf().register` registers
it for the session's SQL, as a temporary function that other sessions don't see. `rust_udf::udf`
makes one of a function of a WebAssembly module built some other way, with its types spelled
out:

```{.rust .async}
use spark_connect_async::rust_udf::{udf, AbiType};

let wasm = std::fs::read("shout.wasm")?;
let shout = udf("shout", wasm, "shout", vec![AbiType::Str], AbiType::Str);

let shouted = session.table("people").select([shout.call(["name"])?]);
session.udf().register("shout", &shout).await?;
let in_sql = session.sql("SELECT shout(name) FROM people").await?;
```

`as_nondeterministic` marks a UDF whose result may change between calls with the same
arguments. Spark calls a function once per Arrow batch through the batch export that
`#[spark_udf]` generates when its arguments and result are each an `i32`, `i64`, `f32`, `f64`,
`bool`, `String`, `Vec<u8>` or an `Option` of one, and once per row otherwise;
`with_batch_entrypoint` names the batch export of a module built some other way.
