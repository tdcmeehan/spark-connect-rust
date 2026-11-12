//! An async Rust client for Spark Connect.
//!
//! Connect a [`Session`] to a Spark Connect server, then run SQL, build DataFrames and read
//! their results. Every call that talks to the server is a future: it does nothing until it is
//! awaited, and it can run inside any Tokio runtime. Errors are [`Error`]s, with the condition,
//! SQL state and message parameters of the server's error.
//!
//! For code that doesn't run async, the `blocking` feature adds the `blocking` module: the
//! calls that talk to the server, in forms that block until the server answers.
//!
//! ```no_run
//! # async fn run() -> spark_connect_async::Result<()> {
//! use spark_connect_async::Session;
//!
//! let session = Session::builder()
//!     .remote("sc://localhost:15002")
//!     .connect()
//!     .await?;
//! session.conf().set("spark.sql.shuffle.partitions", "8").await?;
//! println!("Spark {}", session.version().await?);
//! session.stop().await?;
//! # Ok(())
//! # }
//! ```

pub mod actions;
#[cfg(feature = "blocking")]
pub mod blocking;
pub mod catalog;
pub mod column;
pub mod dataframe;
pub mod de;
pub mod error;
pub mod functions;
pub mod group;
pub mod merge;
pub mod ml;
pub mod observation;
pub mod options;
pub mod pipelines;
pub mod progress;
pub mod readwriter;
#[cfg(feature = "rust-udf")]
pub mod rust_udf;
pub mod session;
pub mod sql;
pub mod stat;
pub mod stream;
pub mod streaming;
pub mod tvf;
pub mod value;
pub mod window;

/// Spark data types.
pub mod types {
    #[doc(no_inline)]
    pub use spark_connect_core::types::{DataType, StructField};
}

pub use column::{col, lit, when, Column, IntoArgument, IntoLiteral};
pub use dataframe::DataFrame;
pub use error::{Error, ErrorKind, Result};
pub use observation::Observation;
pub use session::{Session, SessionBuilder};
pub use stream::RecordBatchStream;
pub use value::{Decimal, Row, Value};

pub use value::CalendarInterval;
