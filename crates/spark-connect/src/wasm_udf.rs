//! Rust user-defined functions executed on Spark via WebAssembly.
//!
//! # Overview
//!
//! Spark Connect has no native "run this Rust code" UDF type. What it *does*
//! have is the [`PythonUDF`](crate::udf) path: a pickled `(callable,
//! return_type)` tuple shipped inside a `CommonInlineUserDefinedFunction` and
//! executed on the Spark executors by a Python worker.
//!
//! This module leverages that path to run **Rust** UDFs, distributed on the
//! cluster, without touching the Spark server:
//!
//! 1. The user compiles their Rust function to a WebAssembly module.
//! 2. [`udf`] writes the `command` pickle itself, so the client needs no Python. It
//!    carries the `.wasm` bytes, the signature, and the source of the Python runner
//!    embedded in this crate (`wasm_runner.py`), which the worker compiles when it
//!    loads the command.
//! 3. On each executor the runner instantiates the module with `wasmtime` and
//!    calls the exported entrypoint once per Arrow batch, or once per row for
//!    signatures the batch export doesn't cover.
//!
//! # ABI
//!
//! Arguments and the result cross the WASM boundary as a little-endian,
//! length-prefixed byte buffer (see [`AbiType`]): the runner encodes the row's
//! arguments into linear memory (via the module's exported `spark_udf_alloc`),
//! calls the entrypoint `fn(ptr, len) -> packed_ptr_len`, and decodes the
//! result. This supports scalars, `bool`, strings, binary, arrays, and
//! nullability — not just numbers.
//!
//! # API
//!
//! Prefer the [`spark_wasm_udf`](https://docs.rs/apache-spark-connect-macros)
//! macro, which writes the [`AbiType`]s and the constructor for you. This
//! lower-level [`udf`] factory (mirroring `pyspark.sql.functions.udf`) is for
//! loading a prebuilt module by hand:
//!
//! ```no_run
//! use spark_connect::functions::col;
//! use spark_connect::wasm_udf::{udf, AbiType};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wasm = std::fs::read("shout.wasm")?;
//! // fn shout(s: String) -> String
//! let shout = udf("shout", wasm, "shout", vec![AbiType::Str], AbiType::Str);
//! let c = shout.call(vec![col("name")])?;   // -> Column
//! # Ok(()) }
//! ```
//!
//! # Requirements
//!
//! Nothing on the client. The Spark executors need the `wasmtime` Python package,
//! and `pyarrow` for batch UDFs. The command names the executors' Python version,
//! which the worker checks: set it with [`UserDefinedFunction::with_python_ver`] or
//! `$SPARK_CONNECT_PYTHON_VERSION` (default `3.11`).

use crate::column::Column;
use crate::expression::ExpressionNode;
use crate::types::DataType;
use crate::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};
use spark_connect_core::error::Result;

/// The type of one WASM UDF argument or its result, describing how the value is
/// encoded across the WASM boundary and which Spark SQL type it maps to.
///
/// Built for you by the `#[spark_wasm_udf]` macro from the Rust signature:
///
/// | Rust                | `AbiType`                     | Spark SQL type |
/// |---------------------|-------------------------------|----------------|
/// | `i32`               | `I32`                         | `IntegerType`  |
/// | `i64`               | `I64`                         | `LongType`     |
/// | `f32`               | `F32`                         | `FloatType`    |
/// | `f64`               | `F64`                         | `DoubleType`   |
/// | `bool`              | `Bool`                        | `BooleanType`  |
/// | `String` / `&str`   | `Str`                         | `StringType`   |
/// | `Vec<u8>`           | `Binary`                      | `BinaryType`   |
/// | `Vec<T>`            | `Array(T)`                    | `ArrayType`    |
/// | `Option<T>`         | `Nullable(T)`                 | (nullable `T`) |
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiType {
    I32,
    I64,
    F32,
    F64,
    Bool,
    /// UTF-8 string (`String` / `&str`).
    Str,
    /// Byte string (`Vec<u8>`).
    Binary,
    /// Homogeneous list (`Vec<T>`).
    Array(Box<AbiType>),
    /// Nullable value (`Option<T>`).
    Nullable(Box<AbiType>),
}

impl AbiType {
    /// The descriptor string the Python runner uses to drive its codec, e.g.
    /// `"i64"`, `"string"`, `"array:i64"`, `"option:string"`.
    pub fn descriptor(&self) -> String {
        match self {
            AbiType::I32 => "i32".to_string(),
            AbiType::I64 => "i64".to_string(),
            AbiType::F32 => "f32".to_string(),
            AbiType::F64 => "f64".to_string(),
            AbiType::Bool => "bool".to_string(),
            AbiType::Str => "string".to_string(),
            AbiType::Binary => "binary".to_string(),
            AbiType::Array(inner) => format!("array:{}", inner.descriptor()),
            AbiType::Nullable(inner) => format!("option:{}", inner.descriptor()),
        }
    }

    /// The Spark [`DataType`] this maps to (used for the proto `output_type`).
    pub fn to_data_type(&self) -> DataType {
        match self {
            AbiType::I32 => DataType::Integer,
            AbiType::I64 => DataType::Long,
            AbiType::F32 => DataType::Float,
            AbiType::F64 => DataType::Double,
            AbiType::Bool => DataType::Boolean,
            AbiType::Str => DataType::String {
                collation: "UTF8_BINARY".to_string(),
            },
            AbiType::Binary => DataType::Binary,
            AbiType::Array(inner) => DataType::Array {
                element_type: Box::new(inner.to_data_type()),
                contains_null: matches!(**inner, AbiType::Nullable(_)),
            },
            // Nullability is tracked at the column/field level; the element type
            // itself is what matters here.
            AbiType::Nullable(inner) => inner.to_data_type(),
        }
    }

    /// Spark's canonical type JSON (`DataType.jsonValue()` form), which the
    /// worker parses with `_parse_datatype_json_string` — so any type maps
    /// generically without a hand-maintained token table.
    fn to_spark_json(&self) -> serde_json::Value {
        use serde_json::json;
        match self {
            AbiType::I32 => json!("integer"),
            AbiType::I64 => json!("long"),
            AbiType::F32 => json!("float"),
            AbiType::F64 => json!("double"),
            AbiType::Bool => json!("boolean"),
            AbiType::Str => json!("string"),
            AbiType::Binary => json!("binary"),
            AbiType::Array(inner) => json!({
                "type": "array",
                "elementType": inner.to_spark_json(),
                "containsNull": matches!(**inner, AbiType::Nullable(_)),
            }),
            AbiType::Nullable(inner) => inner.to_spark_json(),
        }
    }
}

/// The Python runner the executors use, carried inside every command.
const RUNNER_SRC: &str = include_str!("wasm_runner.py");

/// The UDF's `command`: a protocol-4 pickle that, when the Python worker loads it,
/// compiles [`RUNNER_SRC`] into a fresh namespace and returns
/// `_spark_connect_wasm_udf_command(wasm, entrypoint, arg_types, ret_type,
/// output_type, batch)`. Spark loads commands with `pickle`, so writing these few
/// opcodes here is what keeps Python off the client.
fn command_pickle(
    wasm: &[u8],
    entrypoint: &str,
    arg_types: &[String],
    ret_type: &str,
    output_type: &str,
    batch: bool,
) -> Vec<u8> {
    fn text(out: &mut Vec<u8>, s: &str) {
        match u8::try_from(s.len()) {
            Ok(n) => out.extend_from_slice(&[0x8c, n]), // SHORT_BINUNICODE
            Err(_) => {
                out.push(b'X'); // BINUNICODE
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            }
        }
        out.extend_from_slice(s.as_bytes());
    }
    fn global(out: &mut Vec<u8>, module: &str, name: &str) {
        text(out, module);
        text(out, name);
        out.push(0x93); // STACK_GLOBAL
    }

    let mut out = vec![0x80, 4]; // PROTO 4
    out.extend_from_slice(b"}\x94"); // EMPTY_DICT, MEMOIZE: the namespace is memo 0
    text(&mut out, "__name__");
    text(&mut out, "spark_connect_wasm_udf");
    out.extend_from_slice(b"s0"); // SETITEM, POP
    global(&mut out, "builtins", "exec");
    text(&mut out, RUNNER_SRC);
    out.extend_from_slice(b"h\x00\x86R0"); // BINGET 0, TUPLE2, REDUCE, POP
    global(&mut out, "operator", "getitem");
    out.extend_from_slice(b"h\x00");
    text(&mut out, "_spark_connect_wasm_udf_command");
    out.extend_from_slice(b"\x86R("); // TUPLE2, REDUCE, then MARK the call's arguments
    match u32::try_from(wasm.len()) {
        Ok(n) => {
            out.push(b'B'); // BINBYTES
            out.extend_from_slice(&n.to_le_bytes());
        }
        Err(_) => {
            out.push(0x8e); // BINBYTES8
            out.extend_from_slice(&(wasm.len() as u64).to_le_bytes());
        }
    }
    out.extend_from_slice(wasm);
    text(&mut out, entrypoint);
    out.extend_from_slice(b"]("); // EMPTY_LIST, MARK
    for t in arg_types {
        text(&mut out, t);
    }
    out.push(b'e'); // APPENDS
    text(&mut out, ret_type);
    text(&mut out, output_type);
    out.push(if batch { 0x88 } else { 0x89 }); // NEWTRUE / NEWFALSE
    out.extend_from_slice(b"tR."); // TUPLE, REDUCE, STOP
    out
}

/// Create a Rust/WASM user-defined function, mirroring
/// `pyspark.sql.functions.udf`.
///
/// * `name` — the UDF name reported to Spark.
/// * `wasm_module` — the raw bytes of the compiled `.wasm` module.
/// * `entrypoint` — the exported function to invoke per row.
/// * `arg_types` — the [`AbiType`] of each argument, in order.
/// * `ret_type` — the [`AbiType`] returned by `entrypoint`; also determines the
///   Spark output type.
///
/// The returned [`UserDefinedFunction`] is callable on columns via
/// [`UserDefinedFunction::call`].
pub fn udf(
    name: impl Into<String>,
    wasm_module: impl Into<Vec<u8>>,
    entrypoint: impl Into<String>,
    arg_types: Vec<AbiType>,
    ret_type: AbiType,
) -> UserDefinedFunction {
    UserDefinedFunction {
        name: name.into(),
        wasm_module: wasm_module.into(),
        entrypoint: entrypoint.into(),
        arg_types,
        ret_type,
        deterministic: true,
        // Arrow-optimized Python UDF: func is still called once per row
        // (see pyspark worker `wrap_arrow_batch_udf`), but IO is Arrow.
        eval_type: eval_type::SQL_ARROW_BATCHED_UDF,
        batch_entrypoint: None,
        python_ver: default_python_ver(),
    }
}

/// A Rust/WASM user-defined function, mirroring
/// `pyspark.sql.udf.UserDefinedFunction`.
#[derive(Debug, Clone)]
pub struct UserDefinedFunction {
    name: String,
    wasm_module: Vec<u8>,
    entrypoint: String,
    arg_types: Vec<AbiType>,
    ret_type: AbiType,
    deterministic: bool,
    eval_type: i32,
    batch_entrypoint: Option<String>,
    python_ver: String,
}

impl UserDefinedFunction {
    /// Mark the UDF non-deterministic, mirroring
    /// `pyspark.sql.udf.UserDefinedFunction.asNondeterministic`.
    pub fn as_nondeterministic(mut self) -> Self {
        self.deterministic = false;
        self
    }

    /// Override the Python evaluation type (default
    /// [`SQL_ARROW_BATCHED_UDF`](eval_type::SQL_ARROW_BATCHED_UDF)).
    pub fn with_eval_type(mut self, value: i32) -> Self {
        self.eval_type = value;
        self
    }

    /// Call `entrypoint`, a batch export generated by `#[spark_wasm_udf]`, once per Arrow
    /// batch: the UDF is sent as a scalar Arrow UDF
    /// ([`SQL_SCALAR_ARROW_UDF`](eval_type::SQL_SCALAR_ARROW_UDF)), Spark 4.1 or later.
    pub fn with_batch_entrypoint(mut self, entrypoint: impl Into<String>) -> Self {
        self.batch_entrypoint = Some(entrypoint.into());
        self.eval_type = eval_type::SQL_SCALAR_ARROW_UDF;
        self
    }

    /// Call the per-row export once per row
    /// ([`SQL_ARROW_BATCHED_UDF`](eval_type::SQL_ARROW_BATCHED_UDF)), also on Spark before 4.1.
    pub fn per_row(mut self) -> Self {
        self.batch_entrypoint = None;
        self.eval_type = eval_type::SQL_ARROW_BATCHED_UDF;
        self
    }

    /// Override the Python version sent with the command (`"major.minor"`). It must
    /// match the executors' Python, which the worker checks.
    pub fn with_python_ver(mut self, value: impl Into<String>) -> Self {
        self.python_ver = value.into();
        self
    }

    /// The Spark output [`DataType`], derived from the return [`AbiType`].
    pub fn output_type(&self) -> DataType {
        self.ret_type.to_data_type()
    }

    fn build_command(&self) -> Vec<u8> {
        let (entrypoint, batch) = match &self.batch_entrypoint {
            Some(e) => (e.as_str(), true),
            None => (self.entrypoint.as_str(), false),
        };
        let arg_types: Vec<String> = self.arg_types.iter().map(AbiType::descriptor).collect();
        command_pickle(
            &self.wasm_module,
            entrypoint,
            &arg_types,
            &self.ret_type.descriptor(),
            &self.ret_type.to_spark_json().to_string(),
            batch,
        )
    }

    /// Build the [`PythonUDFPayload`] carrying the pickled command.
    pub fn to_payload(&self) -> Result<PythonUDFPayload> {
        Ok(PythonUDFPayload::new(
            self.output_type(),
            self.eval_type,
            self.build_command(),
            self.python_ver.clone(),
        ))
    }

    /// Build the [`CommonInlineUserDefinedFunctionExpression`] for the given
    /// argument columns.
    pub fn to_expression(
        &self,
        args: Vec<Column>,
    ) -> Result<CommonInlineUserDefinedFunctionExpression> {
        Ok(CommonInlineUserDefinedFunctionExpression::new(
            self.name.clone(),
            self.deterministic,
            args.iter().map(|c| c.expression().clone()).collect(),
            self.to_payload()?,
        ))
    }

    /// Apply the UDF to `args`, producing a [`Column`] — mirroring calling a
    /// PySpark `UserDefinedFunction` on columns (`my_udf(col("x"))`).
    pub fn call<C: Into<Column>>(&self, args: impl IntoIterator<Item = C>) -> Result<Column> {
        let args: Vec<Column> = args.into_iter().map(Into::into).collect();
        let expr = self.to_expression(args)?;
        Ok(Column::new(
            ExpressionNode::CommonInlineUserDefinedFunction(Box::new(expr)).into(),
        ))
    }

    /// Build the registration expression for this UDF under `name` (no bound
    /// arguments — arguments are supplied at each SQL/DataFrame call site).
    fn registration_expression(
        &self,
        name: &str,
    ) -> Result<CommonInlineUserDefinedFunctionExpression> {
        Ok(CommonInlineUserDefinedFunctionExpression::new(
            name.to_string(),
            self.deterministic,
            vec![],
            self.to_payload()?,
        ))
    }
}

/// UDF registration accessor, returned by [`crate::session::SparkSession::udf`] and
/// mirroring `pyspark.sql.SparkSession.udf`.
pub struct UdfRegistration<'a> {
    session: &'a crate::session::SparkSession,
}

impl<'a> UdfRegistration<'a> {
    /// Register `udf` under `name` so it can be called from SQL
    /// (`spark.sql("SELECT name(col) ...")`) and the DataFrame API.
    ///
    /// Mirrors `spark.udf.register(name, udf)`: the runner travels inside the pickled
    /// command of a `RegisterFunction` command, so the Spark executors need only the
    /// `wasmtime` Python package (and `pyarrow` for batch UDFs).
    pub fn register(&self, name: &str, udf: &UserDefinedFunction) -> Result<()> {
        self.session
            .register_function(udf.registration_expression(name)?)
    }
}

impl crate::session::SparkSession {
    /// UDF registration accessor. Mirrors `pyspark.sql.SparkSession.udf`, so a WASM
    /// UDF is registered with `spark.udf().register("name", &udf)`.
    pub fn udf(&self) -> UdfRegistration<'_> {
        UdfRegistration { session: self }
    }
}

/// The executors' Python version (`"major.minor"`, as pyspark reports
/// `sys.version_info[:2]`): `$SPARK_CONNECT_PYTHON_VERSION`, else `3.11`. Callers can
/// override with [`UserDefinedFunction::with_python_ver`].
fn default_python_ver() -> String {
    std::env::var("SPARK_CONNECT_PYTHON_VERSION")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "3.11".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> UserDefinedFunction {
        // fn add_one(x: i64) -> i64
        udf(
            "add_one",
            vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00],
            "add_one",
            vec![AbiType::I64],
            AbiType::I64,
        )
    }

    #[test]
    fn defaults_to_arrow_batched_udf() {
        let f = sample();
        assert_eq!(f.eval_type, eval_type::SQL_ARROW_BATCHED_UDF);
        assert!(f.deterministic);
    }

    #[test]
    fn as_nondeterministic_flips_flag() {
        assert!(!sample().as_nondeterministic().deterministic);
    }

    #[test]
    fn abitype_descriptors() {
        assert_eq!(AbiType::I64.descriptor(), "i64");
        assert_eq!(AbiType::Str.descriptor(), "string");
        assert_eq!(AbiType::Binary.descriptor(), "binary");
        assert_eq!(
            AbiType::Array(Box::new(AbiType::I64)).descriptor(),
            "array:i64"
        );
        assert_eq!(
            AbiType::Nullable(Box::new(AbiType::Str)).descriptor(),
            "option:string"
        );
        assert_eq!(
            AbiType::Array(Box::new(AbiType::Nullable(Box::new(AbiType::I32)))).descriptor(),
            "array:option:i32"
        );
    }

    #[test]
    fn abitype_to_data_type() {
        assert_eq!(AbiType::I64.to_data_type(), DataType::Long);
        assert_eq!(AbiType::Bool.to_data_type(), DataType::Boolean);
        assert_eq!(AbiType::Binary.to_data_type(), DataType::Binary);
        match AbiType::Array(Box::new(AbiType::Nullable(Box::new(AbiType::I64)))).to_data_type() {
            DataType::Array {
                element_type,
                contains_null,
            } => {
                assert_eq!(*element_type, DataType::Long);
                assert!(contains_null);
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn abitype_spark_json() {
        assert_eq!(AbiType::I64.to_spark_json(), serde_json::json!("long"));
        assert_eq!(
            AbiType::Array(Box::new(AbiType::Str)).to_spark_json(),
            serde_json::json!({"type": "array", "elementType": "string", "containsNull": false})
        );
    }

    /// Loads the command as a Python worker does. Needs a Python with `pyspark`
    /// (`$SPARK_CONNECT_TEST_PYTHON`, default `python3`); run with
    /// `cargo test --features wasm-udf -- --ignored`.
    #[test]
    #[ignore]
    fn command_loads_as_runner_and_return_type() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let python =
            std::env::var("SPARK_CONNECT_TEST_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let load = |f: &UserDefinedFunction| -> String {
            let mut child = Command::new(&python)
                .args([
                    "-c",
                    "import pickle, sys; r, t = pickle.loads(sys.stdin.buffer.read()); \
                     print(type(r).__name__, r.entrypoint, r.arg_types, r.ret_type, \
                     len(r.wasm), t.simpleString())",
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("python should start");
            let command = f.build_command();
            child.stdin.take().unwrap().write_all(&command).unwrap();
            let out = child.wait_with_output().unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8(out.stdout).unwrap().trim().to_string()
        };

        assert_eq!(
            load(&sample()),
            "WasmScalarUDF add_one ['i64'] i64 6 bigint"
        );
        assert_eq!(
            load(&sample().with_batch_entrypoint("spark_udf_batch_add_one")),
            "WasmArrowUDF spark_udf_batch_add_one ['i64'] i64 6 bigint"
        );
        let nested = udf(
            "f",
            vec![0u8; 300],
            "f",
            vec![AbiType::Str, AbiType::Nullable(Box::new(AbiType::I32))],
            AbiType::Array(Box::new(AbiType::Nullable(Box::new(AbiType::I64)))),
        );
        assert_eq!(
            load(&nested),
            "WasmScalarUDF f ['string', 'option:i32'] array:option:i64 300 array<bigint>"
        );
    }
}
