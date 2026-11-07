//! Pure-Rust Spark Connect DataFrame client mirroring the PySpark API surface.
//!
//! This crate provides the core data types and operations for interacting with
//! Apache Spark via the Spark Connect protocol.

#![allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API

mod arrow_input;
pub mod catalog;
pub mod conf;
pub mod dataframe;
pub mod datasource;
pub mod group;
pub mod merge;
pub mod ml;
pub mod pipelines;
pub mod profiler;
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
#[doc(hidden)]
pub use arrow_input::ArrowInputOptions;
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
pub use profiler::ProfilerCollector;
pub use readwriter::{
    DataFrameReader, DataFrameWriter, DataFrameWriterV2, ReadType, SaveMode, TableSaveMethod,
};
pub use resource::{
    ExecutorResourceRequests, ResourceProfile, ResourceProfileBuilder, TaskResourceRequests,
};
pub use session::{SparkSession, SparkSessionBuilder};
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
