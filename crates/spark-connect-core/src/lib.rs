//! The shared core of the Spark Connect client: the gRPC transport, the plans and requests
//! that it sends, and the decoding of their results.

pub mod actions;
#[doc(hidden)]
pub mod arrow_chunks;
pub mod arrow_input;
pub mod artifact;
pub mod bytes_codec;
pub mod channel;
pub mod client;
pub mod column;
pub mod decode;
pub mod error;
pub mod execution;
pub mod expression;
#[doc(hidden)]
pub mod function_forms;
pub mod functions;
#[doc(hidden)]
pub mod group;
#[doc(hidden)]
pub mod interop;
#[doc(hidden)]
pub mod local_rows;
pub mod merge;
pub mod ml;
pub mod observation;
pub mod origin;
pub mod pipelines;
pub mod plan;
mod proxy;
pub mod readwriter;
pub mod reattach;
pub mod retries;
pub mod row;
pub mod runtime;
mod srs;
pub mod streaming;
// Shared with `apache-spark-connect`, whose session keeps per-thread tags the same way.
#[doc(hidden)]
pub mod thread_entries;
pub mod types;
pub mod udf;
/// Run Rust UDFs on Spark via WebAssembly. Requires the `wasm-udf` feature.
#[cfg(feature = "wasm-udf")]
pub mod wasm_udf;
pub mod window;

pub use artifact::{
    build_artifact_request_stream, compute_crc32, ArtifactData, FileArtifact, InMemoryArtifact,
    NamedArtifact, CHUNK_SIZE,
};
pub use channel::ChannelBuilder;
pub use client::{ExecutePlanHook, RpcDeadlines, SparkConnectClient};
pub use error::{QueryContext, QueryContextType, Result, SparkError, SparkErrorKind};
pub use reattach::ExecutePlanResponseReattachableIterator;
pub use retries::{RetryPolicy, RetryPolicyState, DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME};
