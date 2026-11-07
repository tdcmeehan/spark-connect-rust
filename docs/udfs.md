# Rust UDFs via WebAssembly

Run **Rust** user-defined functions on Spark, distributed on the executors, with
no server-side plugin. You write plain Rust functions; the toolchain compiles
them to WebAssembly, embeds the module, and ships it inside a standard Spark
`PythonUDF`. The executors need only the `wasmtime` Python package - no Rust
toolchain server-side.

!!! note "Opt-in"
    This is behind the `wasm-udf` feature. Users who don't run Rust UDFs pull
    none of the extra dependencies and need none of the preconditions below.

## Write plain Rust functions

Annotate a module of functions with `#[spark_wasm_udf]`. For each function the
macro infers the Spark signature from the Rust types, exports it to WASM, and
generates a constructor under `udf::` - a direct call `udf::<name>(col0, col1, ...)`
(one column per argument, **arity checked at compile time**) that returns the
result `Column`. A one-line `build.rs` compiles the module.

```{.rust .no-run}
// src/main.rs
use spark_connect_macros::spark_wasm_udf;

#[spark_wasm_udf]
mod udfs {
    pub fn add_one(x: i64) -> i64 { x + 1 }                          // (Long) -> Long
    pub fn shout(s: String) -> String { format!("{}!", s.to_uppercase()) }
    pub fn sum(xs: Vec<i64>) -> i64 { xs.iter().sum() }              // ArrayType arg
    pub fn double_or_null(x: Option<i64>) -> Option<i64> { x.map(|v| v * 2) }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use spark_connect::functions::col;
    use spark_connect::SparkSessionBuilder;

    let spark = SparkSessionBuilder::default()
        .remote("sc://localhost:15002")
        .get_or_create()?;

    // Pass columns straight in - one per argument. Signatures are inferred and
    // the compiled module is embedded; there is no `.wasm` file to load.
    spark
        .range(5)?
        .select([
            col("id"),
            udf::add_one(col("id"))?.alias("plus_one"),
        ])
        .show(20)?;
    Ok(())
}
```

```{.rust .no-run}
// build.rs
fn main() {
    spark_connect_build::embed_wasm_udf("src/main.rs");
}
```

```toml
# Cargo.toml
[dependencies]
spark-connect-macros = { package = "apache-spark-connect-macros", version = "4.3" }

# host-only: the client is not pulled into the wasm build
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
spark-connect = { package = "apache-spark-connect", version = "4.3", features = ["wasm-udf"] }

[build-dependencies]
spark-connect-build  = { package = "apache-spark-connect-build",  version = "4.3" }
spark-connect-macros = { package = "apache-spark-connect-macros", version = "4.3" }
```

Run it:

```bash
rustup target add wasm32-unknown-unknown            # once
cargo run
```

!!! tip "Runnable examples"
    See `examples/wasm-udf-inline/` (the UDFs and client in one file) and
    `examples/wasm-udfs/` + `examples/src/wasm_udf_macro.rs` (UDFs as a reusable
    crate, called as `wasm_udfs::udf::add_one(col("id"))?`).

## Supported types

Arguments and results cross the WASM boundary with a length-prefixed binary ABI
(`spark_connect::wasm_udf::AbiType`), inferred from the Rust signature:

| Rust        | Spark SQL type       |
|-------------|----------------------|
| `i32`       | `IntegerType`        |
| `i64`       | `LongType`           |
| `f32`       | `FloatType`          |
| `f64`       | `DoubleType`         |
| `bool`      | `BooleanType`        |
| `String`    | `StringType`         |
| `Vec<u8>`   | `BinaryType`         |
| `Vec<T>`    | `ArrayType` (of `T`) |
| `Option<T>` | nullable `T`         |

These nest arbitrarily - e.g. `Vec<Option<String>>` -> `ArrayType(StringType, nullable)`.

## How it works

```text
build time   build.rs -> embed_wasm_udf() recompiles the source for wasm32
             (host-only code is cfg'd out) and embeds the .wasm module.

client       udf::add_one(col) builds a standard PythonUDF in Rust: its pickled
             command carries the module and the source of a small Python runner,
             so the client needs no Python and executors need nothing pre-deployed.

executors    the Python worker compiles the runner, instantiates the module with
             wasmtime, and calls the exported entrypoint once per Arrow batch (or
             once per row for signatures the batch path doesn't cover).
```

The user never writes WASM, never touches the ABI, and never loads a `.wasm`
file - they write `add_one(x: i64) -> i64` and call `udf::add_one(col("id"))`.

## Preconditions (only when using Rust UDFs)

Nothing here is needed unless the `wasm-udf` feature is enabled:

- **Build machine** - the `wasm32-unknown-unknown` target (`rustup target add
  wasm32-unknown-unknown`), plus `apache-spark-connect-macros` and
  `apache-spark-connect-build` as (build-)dependencies.
- **Client** - nothing beyond the Rust program. Set `SPARK_CONNECT_PYTHON_VERSION`
  (or call `.with_python_ver()`) to the executors' Python version, such as `3.12`,
  if it isn't `3.11`: the worker rejects a UDF built for another version.
- **Executors** - the `wasmtime` Python package, and `pyarrow` for batch UDFs.
- **Spark** - 4.3.0+.

## Advanced

For non-deterministic UDFs, the per-row path, or another Python version, the
macro also generates a builder `udf::<name>_udf()`:

```{.rust .no-run}
udf::add_one_udf()
    .as_nondeterministic()
    .call(vec![col("id")])?;
```

If you'd rather load a prebuilt module and spell out the types yourself, the
lower-level factory mirrors `pyspark.sql.functions.udf`:

```{.rust .no-run}
use spark_connect::wasm_udf::{udf, AbiType};

let wasm = std::fs::read("shout.wasm")?;
let shout = udf("shout", wasm, "shout", vec![AbiType::Str], AbiType::Str);
let c = shout.call(vec![col("name")])?;
```
