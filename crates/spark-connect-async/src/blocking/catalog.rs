//! The blocking form of the catalog of a session.

use std::collections::BTreeMap;
use std::future::IntoFuture;

use super::{wait, DataFrame, Session};
use crate::catalog::{
    self as catalog, CatalogColumn, CatalogMetadata, Database, Function, Table, TablePartition,
};
use crate::error::Result;

/// The catalog of a session, whose operations block until the server answers.
///
/// Its operations with optional arguments are the async [`crate::catalog::Catalog`]'s, which
/// [`as_async`](Catalog::as_async) borrows. They return builders:
/// [`run`](crate::catalog::ListTables::run) one to send it and block.
///
/// ```no_run
/// # fn run(session: spark_connect_async::blocking::Session) -> spark_connect_async::Result<()> {
/// let catalog = session.catalog();
/// println!("in {}", catalog.current_database()?);
/// for table in catalog.as_async().list_tables().database("sales").run()? {
///     println!("{} ({})", table.name, table.table_type);
/// }
/// catalog.as_async().drop_table("sales.staging").if_exists().run()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Catalog(catalog::Catalog);

impl Catalog {
    /// This catalog as the async [`crate::catalog::Catalog`] it holds, borrowed: its operations
    /// with optional arguments are called on it.
    pub fn as_async(&self) -> &catalog::Catalog {
        &self.0
    }

    /// This catalog as the async [`crate::catalog::Catalog`] it holds, as [`From`] converts it.
    pub fn into_async(self) -> catalog::Catalog {
        self.0
    }

    /// The session of this catalog.
    pub fn session(&self) -> Session {
        Session::from(self.0.session().clone())
    }

    /// The name of the current catalog.
    pub fn current_catalog(&self) -> Result<String> {
        wait(self.0.current_catalog())
    }

    /// Makes `name` the current catalog.
    pub fn set_current_catalog(&self, name: impl Into<String>) -> Result<()> {
        wait(self.0.set_current_catalog(name))
    }

    /// The name of the current database.
    pub fn current_database(&self) -> Result<String> {
        wait(self.0.current_database())
    }

    /// Makes `name` the current database.
    pub fn set_current_database(&self, name: impl Into<String>) -> Result<()> {
        wait(self.0.set_current_database(name))
    }

    /// The database `name`, which may be qualified by its catalog. Fails if it does not exist.
    pub fn get_database(&self, name: impl Into<String>) -> Result<Database> {
        wait(self.0.get_database(name))
    }

    /// Whether the database `name` exists.
    pub fn database_exists(&self, name: impl Into<String>) -> Result<bool> {
        wait(self.0.database_exists(name))
    }

    /// Drops the temporary view `name` of the session, and returns whether it existed.
    pub fn drop_temp_view(&self, name: impl Into<String>) -> Result<bool> {
        wait(self.0.drop_temp_view(name))
    }

    /// Drops the global temporary view `name`, a view of every session of the server, and
    /// returns whether it existed.
    pub fn drop_global_temp_view(&self, name: impl Into<String>) -> Result<bool> {
        wait(self.0.drop_global_temp_view(name))
    }

    /// Removes the table or view `name` from the cache.
    pub fn uncache_table(&self, name: impl Into<String>) -> Result<()> {
        wait(self.0.uncache_table(name))
    }

    /// Whether the table or view `name` is cached.
    pub fn is_cached(&self, name: impl Into<String>) -> Result<bool> {
        wait(self.0.is_cached(name))
    }

    /// Removes every table and view from the cache.
    pub fn clear_cache(&self) -> Result<()> {
        wait(self.0.clear_cache())
    }

    /// Drops what the server caches of the data and metadata of the table `name`, as
    /// [`crate::catalog::Catalog::refresh_table`] does.
    pub fn refresh_table(&self, name: impl Into<String>) -> Result<()> {
        wait(self.0.refresh_table(name))
    }

    /// Drops what the server caches of every DataFrame that reads files under `path`.
    pub fn refresh_by_path(&self, path: impl Into<String>) -> Result<()> {
        wait(self.0.refresh_by_path(path))
    }

    /// Adds to the catalog the partitions of the table `name` that its directory has and the
    /// catalog lacks.
    pub fn recover_partitions(&self, name: impl Into<String>) -> Result<()> {
        wait(self.0.recover_partitions(name))
    }

    /// Removes every row of the table `name`.
    pub fn truncate_table(&self, name: impl Into<String>) -> Result<()> {
        wait(self.0.truncate_table(name))
    }

    /// The properties of the table `name`.
    pub fn get_table_properties(
        &self,
        name: impl Into<String>,
    ) -> Result<BTreeMap<String, String>> {
        wait(self.0.get_table_properties(name))
    }

    /// The partitions of the table `name`.
    pub fn list_partitions(&self, name: impl Into<String>) -> Result<Vec<TablePartition>> {
        wait(self.0.list_partitions(name))
    }
}

impl From<catalog::Catalog> for Catalog {
    fn from(catalog: catalog::Catalog) -> Self {
        Catalog(catalog)
    }
}

impl From<Catalog> for catalog::Catalog {
    fn from(catalog: Catalog) -> Self {
        catalog.0
    }
}

impl catalog::Catalog {
    /// The blocking form of this catalog, whose operations block.
    pub fn blocking(&self) -> Catalog {
        Catalog(self.clone())
    }
}

impl Session {
    /// The catalog of the session: its catalogs, databases, tables, views and functions.
    pub fn catalog(&self) -> Catalog {
        Catalog(self.0.catalog())
    }
}

/// `run()` for builders of the catalog that async code awaits.
macro_rules! run {
    ($($(#[$doc:meta])* $builder:ident -> $output:ty;)*) => {$(
        impl catalog::$builder {
            $(#[$doc])*
            pub fn run(self) -> Result<$output> {
                wait(self.into_future())
            }
        }
    )*};
}

run! {
    /// Lists the catalogs, as awaiting the listing does, and blocks until the server answers.
    ListCatalogs -> Vec<CatalogMetadata>;
    /// Lists the databases, as awaiting the listing does, and blocks until the server answers.
    ListDatabases -> Vec<Database>;
    /// Lists the tables, as awaiting the listing does, and blocks until the server answers.
    ListTables -> Vec<Table>;
    /// Gets the table, as awaiting it does, and blocks until the server answers.
    GetTable -> Table;
    /// Checks whether the table exists, as awaiting the check does, and blocks until the server
    /// answers.
    TableExists -> bool;
    /// Lists the columns, as awaiting the listing does, and blocks until the server answers.
    ListColumns -> Vec<CatalogColumn>;
    /// Lists the functions, as awaiting the listing does, and blocks until the server answers.
    ListFunctions -> Vec<Function>;
    /// Gets the function, as awaiting it does, and blocks until the server answers.
    GetFunction -> Function;
    /// Checks whether the function exists, as awaiting the check does, and blocks until the
    /// server answers.
    FunctionExists -> bool;
    /// Caches the table, as awaiting the caching does, and blocks until the server answers.
    CacheTable -> ();
    /// Creates the database, as awaiting the creation does, and blocks until the server
    /// answers.
    CreateDatabase -> ();
    /// Drops the database, as awaiting the drop does, and blocks until the server answers.
    DropDatabase -> ();
    /// Drops the table, as awaiting the drop does, and blocks until the server answers.
    DropTable -> ();
    /// Drops the view, as awaiting the drop does, and blocks until the server answers.
    DropView -> ();
    /// Computes the statistics of the table, as awaiting the analysis does, and blocks until
    /// the server answers.
    AnalyzeTable -> ();
    /// Gets the statement that creates the table, as awaiting it does, and blocks until the
    /// server answers.
    GetCreateTableString -> String;
    /// Lists the views, as awaiting the listing does, and blocks until the server answers.
    ListViews -> Vec<Table>;
}

impl catalog::CreateTable {
    /// Creates the table, as awaiting the creation does, and blocks until the server answers;
    /// returns the DataFrame that reads it.
    pub fn run(self) -> Result<DataFrame> {
        wait(self.into_future()).map(DataFrame::from)
    }
}
