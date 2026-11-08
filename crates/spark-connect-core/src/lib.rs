//! The shared core of the Spark Connect client: the transport, and the building of plans.
//!
//! The transport mirrors `pyspark.sql.connect.client.*`: connection-string parsing and
//! channel configuration ([`channel`]), the gRPC client, retries, reattach, artifacts,
//! and structured errors ([`error`]). Plans are built from the [`types`], columns,
//! [`expression`]s, [`functions`] and [`window`]s of the [`plan`] module.

#![allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API

pub mod artifact;
pub mod bytes_codec;
pub mod channel;
pub mod client;
pub mod column;
pub mod error;
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
