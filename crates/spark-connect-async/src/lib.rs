//! An async Rust client for Spark Connect.
//!
//! Connect a [`Session`] to a Spark Connect server, then run SQL, build DataFrames and read
//! their results. Every call that talks to the server is a future: it does nothing until it is
//! awaited, and it can run inside any Tokio runtime. Errors are [`Error`]s, with the condition,
//! SQL state and message parameters of the server's error.
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

pub mod column;
pub mod de;
pub mod error;
pub mod session;
pub mod value;

pub use column::{lit, Column, IntoLiteral};
pub use error::{Error, ErrorKind, Result};
pub use session::{Session, SessionBuilder};
pub use value::{Decimal, Row, Value};
