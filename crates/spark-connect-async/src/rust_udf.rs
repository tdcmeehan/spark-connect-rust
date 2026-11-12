//! Rust UDFs: functions written in Rust that Spark runs on its executors, compiled to
//! WebAssembly. Requires the `rust-udf` feature.
//!
//! # Defining them
//!
//! Annotate a module of plain functions with `#[spark_udf(crate = spark_connect_async)]` of the
//! `apache-spark-connect-macros` crate, and compile the module to WebAssembly in the build script
//! with `embed_wasm_udf` of the `apache-spark-connect-build` crate, which needs the
//! `wasm32-unknown-unknown` target:
//!
//! ```ignore
//! // build.rs
//! fn main() {
//!     spark_connect_build::embed_wasm_udf("src/lib.rs");
//! }
//!
//! // src/lib.rs
//! use spark_connect_macros::spark_udf;
//!
//! #[spark_udf(crate = spark_connect_async)]
//! mod functions {
//!     pub fn add_one(x: i64) -> i64 {
//!         x + 1
//!     }
//! }
//! ```
//!
//! For each function, the macro generates `udf::<name>(...)`, the column of the function applied
//! to one column per argument, and `udf::<name>_udf()`, its [`UserDefinedFunction`]. [`udf`]
//! builds one for a function of a module built otherwise.
//!
//! # Using them
//!
//! ```no_run
//! # async fn run(
//! #     session: spark_connect_async::Session,
//! #     add_one: spark_connect_async::rust_udf::UserDefinedFunction,
//! # ) -> spark_connect_async::Result<()> {
//! let plus_one = session.range(10).select([add_one.call(["id"])?]);
//!
//! session.udf().register("add_one", &add_one).await?;
//! let in_sql = session.sql("SELECT add_one(id) FROM range(10)").await?;
//! # Ok(())
//! # }
//! ```
//!
//! # How they run
//!
//! Spark runs the module in its Python workers, which need the `wasmtime` package: once per
//! Arrow batch through the batch export that `#[spark_udf]` generates for a function whose
//! arguments and result are each an `i32`, `i64`, `f32`, `f64`, `bool`, `String`, `Vec<u8>` or
//! an `Option` of one, and once per row through its other export otherwise. The client sends a
//! UDF as a Python UDF that carries the module, and builds it with Python: the interpreter of
//! `SPARK_CONNECT_PYTHON`, else of `PYSPARK_PYTHON`, else `python3`, which needs `pyspark`, with
//! the directory of `SPARK_CONNECT_WASM_PACKER_PATH` added to its `PYTHONPATH` if it is set.
//!
//! [`udf`] asks that Python for its version when it creates a UDF value. The value builds its
//! payload with it once: at its first call on columns, on the calling thread, or at its first
//! registration, on a blocking thread of the runtime. Its clones share that build, and later
//! calls and registrations reuse it. A build that fails is not kept, so the next call or
//! registration tries again. Each `udf::<name>(...)` function that `#[spark_udf]` generates
//! keeps one value for all its calls, so it builds the payload once per process.

use std::fmt;
use std::future::Future;
use std::sync::{Arc, OnceLock};

use spark_connect_core::error::SparkError;
use spark_connect_core::rust_udf::{BatchProblem, UserDefinedFunction as CoreUdf};
use spark_connect_core::types::DataType;
use spark_connect_core::udf::PythonUDFPayload;
use spark_connect_proto as proto;

use crate::column::{to_core, Column};
use crate::error::{Error, Result};
use crate::session::Session;
use crate::stream;

pub use spark_connect_core::rust_udf::AbiType;

/// The UDF `name` of the function `entrypoint` of the WebAssembly module `wasm_module`, whose
/// arguments and result have the types `arg_types` and `ret_type`. Spark calls `entrypoint` once
/// per row, or calls a batch export once per Arrow batch after
/// [`with_batch_entrypoint`](UserDefinedFunction::with_batch_entrypoint).
pub fn udf(
    name: impl Into<String>,
    wasm_module: impl Into<Vec<u8>>,
    entrypoint: impl Into<String>,
    arg_types: Vec<AbiType>,
    ret_type: AbiType,
) -> UserDefinedFunction {
    UserDefinedFunction::new(spark_connect_core::rust_udf::udf(
        name,
        wasm_module,
        entrypoint,
        arg_types,
        ret_type,
    ))
}

/// A Rust UDF: a function of a WebAssembly module, with the types of its arguments and result.
///
/// Apply it to columns with [`call`](UserDefinedFunction::call), or register it on a session for
/// its SQL with [`UdfRegistration::register`]. The value builds its payload once (see the
/// [module](crate::rust_udf)), and its clones share that build.
#[derive(Clone)]
pub struct UserDefinedFunction {
    udf: CoreUdf,
    built: Arc<OnceLock<PythonUDFPayload>>,
}

impl fmt::Debug for UserDefinedFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("UserDefinedFunction")
            .field(&self.udf)
            .finish()
    }
}

impl UserDefinedFunction {
    fn new(udf: CoreUdf) -> UserDefinedFunction {
        UserDefinedFunction {
            udf,
            built: Arc::default(),
        }
    }

    /// This UDF, marked nondeterministic, so that Spark's optimizer doesn't assume that the same
    /// arguments give the same result. It shares the build of this value, since its payload is
    /// the same.
    pub fn as_nondeterministic(self) -> UserDefinedFunction {
        UserDefinedFunction {
            udf: self.udf.as_nondeterministic(),
            built: self.built,
        }
    }

    /// This UDF, called once per Arrow batch through `entrypoint`, the batch export that
    /// `#[spark_udf]` generates for a function, instead of once per row. Its payload differs,
    /// so it has a build of its own.
    pub fn with_batch_entrypoint(self, entrypoint: impl Into<String>) -> UserDefinedFunction {
        UserDefinedFunction::new(self.udf.with_batch_entrypoint(entrypoint))
    }

    /// The Spark type of the result.
    pub fn output_type(&self) -> DataType {
        self.udf.output_type()
    }

    /// The column of this UDF applied to `args`, one column per argument of the function.
    ///
    /// The first call or registration of the value builds its payload (see the
    /// [module](crate::rust_udf)). That fails if Python fails, or if the UDF has a batch
    /// entrypoint but no arguments, or an argument or a result that a batch export can't take.
    pub fn call(&self, args: impl IntoIterator<Item = impl Into<Column>>) -> Result<Column> {
        let payload = self.payload()?;
        let column = self
            .udf
            .call_with_payload(args.into_iter().map(to_core), payload);
        Ok(Column::from(column))
    }

    /// The payload of this UDF: the one built before, else a new one, which it keeps.
    fn payload(&self) -> Result<PythonUDFPayload> {
        if let Some(payload) = self.built.get() {
            return Ok(payload.clone());
        }
        if let Some(problem) = self.udf.batch_problem() {
            let name = self.udf.name();
            return Err(Error::invalid_argument(match problem {
                // Not reachable through this API: only with_batch_entrypoint calls a UDF per batch.
                BatchProblem::NoEntrypoint => {
                    format!(
                        "Rust UDF '{name}' is called once per batch but has no batch entrypoint"
                    )
                }
                BatchProblem::NoArguments => format!(
                    "Rust UDF '{name}' has no arguments, so it can't be called once per batch; \
                     build it without with_batch_entrypoint to call it once per row"
                ),
                BatchProblem::Unsupported(ty) => format!(
                    "Rust UDF '{name}' has an argument or a result of type '{}', so it can't be \
                     called once per batch; build it without with_batch_entrypoint to call it \
                     once per row",
                    ty.descriptor()
                ),
            }));
        }
        let payload = self.udf.to_payload()?;
        Ok(self.built.get_or_init(|| payload).clone())
    }
}

/// The registration of UDFs on a session, for its SQL: [`Session::udf`].
#[derive(Clone)]
pub struct UdfRegistration {
    session: Session,
}

impl UdfRegistration {
    /// Registers `udf` under `name`, so that the session's SQL and
    /// [`call_udf`](crate::functions::call_udf) can call it.
    ///
    /// The server keeps it as a temporary function of the session, which other sessions don't
    /// see, until the session ends: registering another UDF under the same name replaces it, and
    /// SQL's `DROP TEMPORARY FUNCTION` removes it.
    pub fn register(
        &self,
        name: impl Into<String>,
        udf: &UserDefinedFunction,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let session = self.session.clone();
        let name = name.into();
        let udf = udf.clone();
        async move {
            let payload = match udf.built.get() {
                Some(payload) => payload.clone(),
                None => {
                    let building = udf.clone();
                    off_the_runtime(move || building.payload()).await?
                }
            };
            let function = udf.udf.registration_with_payload(&name, payload);
            let command = proto::command::CommandType::RegisterFunction(function.to_proto());
            stream::command(session, command).await
        }
    }
}

/// Runs `build`, which runs Python, on a blocking thread of the runtime.
async fn off_the_runtime<T: Send + 'static>(
    build: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    match tokio::task::spawn_blocking(build).await {
        Ok(built) => built,
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => Err(SparkError::connect_msg(error.to_string()).into()),
    }
}

impl Session {
    /// The registration of UDFs on this session, for its SQL.
    pub fn udf(&self) -> UdfRegistration {
        UdfRegistration {
            session: self.clone(),
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;

    use spark_connect_core::rust_udf::PythonPacker;

    use super::*;
    use crate::ErrorKind;

    /// A packer that runs the shell script `script`, as `dir/name`, with the spec on its
    /// standard input.
    fn packer(dir: &Path, name: &str, script: &str) -> PythonPacker {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        PythonPacker {
            python_exe: path.to_str().unwrap().to_string(),
            pythonpath: Vec::new(),
        }
    }

    fn with_packer(udf: UserDefinedFunction, packer: PythonPacker) -> UserDefinedFunction {
        UserDefinedFunction::new(udf.udf.with_packer(packer))
    }

    fn add(arg_types: Vec<AbiType>) -> UserDefinedFunction {
        udf(
            "add",
            vec![0u8, 97, 115, 109],
            "add",
            arg_types,
            AbiType::I64,
        )
    }

    #[test]
    fn a_udf_that_cannot_be_built_is_an_error_of_its_kind() {
        let dir = tempfile::tempdir().unwrap();
        // It reads the spec first: exiting before core writes it fails the write with EPIPE.
        let failing = packer(
            dir.path(),
            "failing",
            "cat > /dev/null; echo 'no pyspark here' >&2; exit 3",
        );
        let echo = packer(dir.path(), "echo", "cat");
        let array = AbiType::Array(Box::new(AbiType::I64));
        for (name, udf, kind, message) in [
            (
                "Python fails",
                with_packer(add(vec![AbiType::I64]), failing.clone()),
                ErrorKind::Internal,
                "no pyspark here",
            ),
            (
                "an array per batch",
                with_packer(add(vec![array]), echo.clone()).with_batch_entrypoint("add_batch"),
                ErrorKind::InvalidArgument,
                "Rust UDF 'add' has an argument or a result of type 'array:i64', so it can't be \
                 called once per batch; build it without with_batch_entrypoint to call it once \
                 per row",
            ),
            (
                "no argument per batch",
                with_packer(add(vec![]), echo.clone()).with_batch_entrypoint("add_batch"),
                ErrorKind::InvalidArgument,
                "Rust UDF 'add' has no arguments, so it can't be called once per batch; build it \
                 without with_batch_entrypoint to call it once per row",
            ),
        ] {
            let error = udf.call(["x"]).unwrap_err();
            assert_eq!(error.kind(), kind, "{name}");
            let text = error.message();
            match kind {
                ErrorKind::InvalidArgument => assert_eq!(text, message, "{name}"),
                _ => assert!(text.contains(message), "{name}: {text}"),
            }
        }
    }

    #[test]
    fn a_registration_builds_its_udf_off_the_threads_of_the_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let go = dir.path().join("go");
        let waiting = packer(
            dir.path(),
            "waiting",
            &format!(
                "for _ in $(seq 1 1000); do [ -e '{go}' ] && exec cat; sleep 0.01; done; \
                 echo 'the runtime is blocked' >&2; exit 1",
                go = go.display()
            ),
        );
        let udf = with_packer(add(vec![AbiType::I64]), waiting);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let built = runtime.block_on(async move {
            let building = off_the_runtime(move || udf.payload());
            let unblocking = async move {
                tokio::task::yield_now().await;
                std::fs::write(go, b"").unwrap();
            };
            futures::join!(building, unblocking).0
        });
        assert!(built.is_ok(), "{:?}", built.err());
    }
}
