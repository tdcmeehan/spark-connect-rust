//! The blocking form of the registration of Rust UDFs.

use super::{wait, Session};
use crate::error::Result;
use crate::rust_udf::{self, UserDefinedFunction};

/// The registration of UDFs on a session, for its SQL, whose registrations block until the
/// server has them, from [`Session::udf`].
///
/// A UDF and its calls on columns are the async API's: [`crate::rust_udf::udf`] builds a
/// [`UserDefinedFunction`], whose [`call`](UserDefinedFunction::call) returns a column. A UDF
/// value builds its payload once, whichever API calls or registers it first, and its clones share
/// that build.
///
/// ```no_run
/// # fn run(
/// #     session: spark_connect_async::blocking::Session,
/// #     add_one: spark_connect_async::rust_udf::UserDefinedFunction,
/// # ) -> spark_connect_async::Result<()> {
/// session.udf().register("add_one", &add_one)?;
/// let plus_one = session.as_async().range(10).select([add_one.call(["id"])?]).blocking();
/// println!("{} rows", plus_one.count()?);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct UdfRegistration(rust_udf::UdfRegistration);

impl UdfRegistration {
    /// This registration as the async [`crate::rust_udf::UdfRegistration`] it holds, borrowed.
    pub fn as_async(&self) -> &rust_udf::UdfRegistration {
        &self.0
    }

    /// This registration as the async [`crate::rust_udf::UdfRegistration`] it holds, as
    /// [`From`] converts it.
    pub fn into_async(self) -> rust_udf::UdfRegistration {
        self.0
    }

    /// Registers `udf` under `name`, as [`crate::rust_udf::UdfRegistration::register`] does,
    /// and blocks until the server has it. A first build of its payload runs on a blocking
    /// thread of the runtime of this module.
    pub fn register(&self, name: impl Into<String>, udf: &UserDefinedFunction) -> Result<()> {
        wait(self.0.register(name, udf))
    }
}

impl From<rust_udf::UdfRegistration> for UdfRegistration {
    fn from(registration: rust_udf::UdfRegistration) -> Self {
        UdfRegistration(registration)
    }
}

impl From<UdfRegistration> for rust_udf::UdfRegistration {
    fn from(registration: UdfRegistration) -> Self {
        registration.0
    }
}

impl rust_udf::UdfRegistration {
    /// The blocking form of this registration, whose registrations block.
    pub fn blocking(&self) -> UdfRegistration {
        UdfRegistration(self.clone())
    }
}

impl Session {
    /// The registration of UDFs on this session, for its SQL.
    pub fn udf(&self) -> UdfRegistration {
        UdfRegistration(self.0.udf())
    }
}
