//! The shared core of the Spark Connect client: the gRPC transport, the plans and requests
//! that it sends, and the decoding of their results.

#![allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API

pub mod artifact;
pub mod bytes_codec;
pub mod channel;
pub mod client;
pub mod column;
pub mod decode;
pub mod error;
pub mod execution;
pub mod expression;
pub mod functions;
pub mod observation;
pub mod origin;
pub mod plan;
pub mod readwriter;
pub mod reattach;
pub mod retries;
pub mod row;
pub mod runtime;
// Shared with `apache-spark-connect`, whose session keeps per-thread tags the same way.
#[doc(hidden)]
pub mod thread_entries;
pub mod types;
pub mod udf;
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
pub use runtime::{block_on, get_runtime};
