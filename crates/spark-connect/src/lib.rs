//! Pure-Rust Spark Connect DataFrame client mirroring the PySpark API surface.
//!
//! This crate provides the core data types and operations for interacting with
//! Apache Spark via the Spark Connect protocol.
//!
//! # Synchronous API
//!
//! Every method is synchronous: it returns once the server has answered. The API may be called
//! from any thread, including inside a Tokio runtime of either flavor. The call blocks the
//! calling thread until it completes, as any blocking call does, so avoid it in a task that
//! must keep other tasks of the same runtime running. The work of a call, its callbacks
//! included, runs on the calling thread, while a runtime of the client drives its I/O, so it uses
//! the tags and user context extensions of the calling thread. The exceptions run on a thread of
//! the client's runtime: the read of a prefetching [`DataFrame::to_local_iterator`], with its
//! progress and accumulator handlers, and the background releases of a dropped ML model or
//! checkpoint, with their execute-plan hooks. In a multi-thread runtime the
//! outermost call first hands the thread's other tasks to another thread
//! ([`tokio::task::block_in_place`]). In a `current_thread` runtime (the default of
//! `#[tokio::test]` and of `#[tokio::main(flavor = "current_thread")]`) that runtime's other
//! tasks wait until the call returns.

#![allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API

use spark_connect_core::arrow_input;
#[doc(hidden)]
pub mod blocking;
mod callbacks;
pub mod catalog;
pub mod conf;
pub mod dataframe;
pub mod datasource;
pub mod group;
pub mod merge;
pub mod ml;
pub mod pipelines;
pub mod profiler;
pub mod progress;
pub mod readwriter;
pub mod resource;
pub mod session;
pub mod storage;
pub mod streaming;
pub mod table_arg;
pub mod tvf;
/// Run Rust UDFs on Spark via WebAssembly. Requires the `wasm-udf` feature.
///
/// See [`spark_connect_core::wasm_udf`] for how they are built and run.
#[cfg(feature = "wasm-udf")]
pub mod wasm_udf {
    pub use spark_connect_core::wasm_udf::{udf, AbiType, PythonPacker, UserDefinedFunction};

    use spark_connect_core::error::Result;

    /// UDF registration accessor, returned by [`crate::session::SparkSession::udf`] and
    /// mirroring `pyspark.sql.SparkSession.udf`.
    pub struct UdfRegistration<'a> {
        session: &'a crate::session::SparkSession,
    }

    impl<'a> UdfRegistration<'a> {
        /// Register `udf` under `name` so it can be called from SQL
        /// (`spark.sql("SELECT name(col) ...")`) and the DataFrame API.
        ///
        /// Mirrors `spark.udf.register(name, udf)`: the cloudpickled WASM runner is shipped
        /// by value inside a `RegisterFunction` command, so the Spark executors only need
        /// the `wasmtime` Python package (not `pyspark_wasm_udf`).
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
}

pub use spark_connect_core::{
    column, expression, observation, origin, plan, row, types, udf, window,
};

/// SQL functions that build [`Column`]s.
pub mod functions {
    pub use spark_connect_core::functions::*;

    use crate::dataframe::DataFrame;

    /// Mirrors `pyspark.sql.functions.broadcast`.
    /// Marks a DataFrame as eligible for broadcast join (smaller table in a join).
    pub fn broadcast(df: DataFrame) -> DataFrame {
        df.broadcast()
    }
}

// Re-export commonly used types
pub use column::{col, lit, lit_boolean, lit_double, lit_string, Column};
pub use dataframe::{DataFrame, LocalRowIterator};
pub use datasource::{CommonInlineUserDefinedDataSourceExpression, PythonDataSourcePayload};
pub use expression::{
    Alias, CaseWhen, Cast, ColumnReference, Expression, ExpressionNode, LiteralExpression,
    SortOrder, UnresolvedFunction,
};
pub use group::{CoGroupedData, GroupedData};
pub use merge::MergeIntoWriter;
pub use observation::Observation;
pub use progress::{ProgressUpdate, StageInfo};
pub use readwriter::{
    DataFrameReader, DataFrameWriter, DataFrameWriterV2, ReadType, SaveMode, TableSaveMethod,
};
pub use resource::{
    ExecutorResourceRequests, ResourceProfile, ResourceProfileBuilder, TaskResourceRequests,
};
pub use session::{SparkSession, SparkSessionBuilder};
#[doc(hidden)]
pub use spark_connect_core::arrow_input::ArrowInputOptions;
pub use spark_connect_core::client::ExecutePlanHook;
/// Re-exported so fallible APIs (e.g. the `#[spark_wasm_udf]`-generated UDF
/// constructors) can name the client's `Result`/error types.
pub use spark_connect_core::error::{
    BreakingChangeInfo, QueryContext, QueryContextType, Result, ServerErrorInfo, SparkError,
    SparkErrorKind,
};
/// Re-exported so `persist(...)` / `storage_level()` have a public storage-level type.
pub use spark_connect_proto::StorageLevel;
/// Named `StorageLevel` presets (`MEMORY_AND_DISK`, `DISK_ONLY`, ...) mirroring `pyspark.StorageLevel`.
pub use storage::StorageLevelExt;
pub use streaming::{
    DataStreamReader, DataStreamWriter, StreamingQuery, StreamingQueryManager, Trigger,
};
pub use types::{
    DataType, DateType, DayTimeIntervalType, GeographyType, GeometryType, SpatialType, StructField,
    YearMonthIntervalType,
};
pub use window::{FrameBound, FrameType, Window, WindowSpec};
