//! The catalog of a session: its catalogs, databases, tables, views and functions, and the
//! tables it caches.

use std::collections::BTreeMap;
use std::future::{Future, IntoFuture};
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::FutureExt;
use spark_connect_core::plan::{relation_with_ids, LogicalPlan};
use spark_connect_core::row::{Row as CoreRow, Value as CoreValue};
use spark_connect_core::types::DataType;
use spark_connect_proto as proto;

use crate::actions::StorageLevel;
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::readwriter::{insert_options, OptionValue};
use crate::session::Session;
use crate::stream;

use proto::catalog::CatType;

/// A catalog of the server, such as `spark_catalog`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CatalogMetadata {
    /// The name of the catalog.
    pub name: String,
    /// The description of the catalog, if it has one.
    pub description: Option<String>,
}

/// A database, which the server also calls a namespace or a schema.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Database {
    /// The name of the database.
    pub name: String,
    /// The catalog of the database, if it has one.
    pub catalog: Option<String>,
    /// The description of the database, if it has one.
    pub description: Option<String>,
    /// Where the database keeps its tables, if its catalog says.
    pub location_uri: Option<String>,
}

/// A table or a view.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Table {
    /// The name of the table.
    pub name: String,
    /// The catalog of the table; `None` for a temporary view.
    pub catalog: Option<String>,
    /// The database of the table, as the parts of its name, if the server knows it; empty for
    /// a temporary view.
    pub namespace: Option<Vec<String>>,
    /// The description of the table, if it has one.
    pub description: Option<String>,
    /// The type of the table, such as `MANAGED`, `EXTERNAL`, `VIEW` or `TEMPORARY`.
    pub table_type: String,
    /// Whether the table is a temporary view.
    pub is_temporary: bool,
}

/// A column of a table.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CatalogColumn {
    /// The name of the column.
    pub name: String,
    /// The comment of the column, if it has one.
    pub description: Option<String>,
    /// The type of the column in Spark SQL's syntax, such as `int` or `array<string>`. The
    /// server shortens the types of structs with many fields.
    pub data_type: String,
    /// Whether the column can hold NULLs.
    pub nullable: bool,
    /// Whether the table is partitioned by the column.
    pub is_partition: bool,
    /// Whether the table is bucketed by the column.
    pub is_bucket: bool,
    /// Whether the table is clustered by the column.
    pub is_cluster: bool,
}

/// A function.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Function {
    /// The name of the function.
    pub name: String,
    /// The catalog of the function; `None` for a built-in or temporary function.
    pub catalog: Option<String>,
    /// The database of the function, as the parts of its name; `None` for a built-in or
    /// temporary function.
    pub namespace: Option<Vec<String>>,
    /// The description of the function, if it has one.
    pub description: Option<String>,
    /// The class that implements the function, if it has one.
    pub class_name: Option<String>,
    /// Whether the function is built in or temporary.
    pub is_temporary: bool,
}

/// A partition of a table.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TablePartition {
    /// The values of the partition columns, as in `year=2024/month=1`.
    pub partition: String,
}

/// The catalog of a session, which [`Session::catalog`] returns.
///
/// A table, view or function name may be qualified by its catalog and database, as in
/// `spark_catalog.sales.orders`; otherwise it is looked up in the current database. Each
/// method that talks to the server returns a future, or a builder of the operation's optional
/// arguments that runs the operation when it is awaited.
///
/// ```no_run
/// # async fn run(session: spark_connect_async::Session) -> spark_connect_async::Result<()> {
/// let catalog = session.catalog();
/// for table in catalog.list_tables().database("sales").pattern("order*").await? {
///     println!("{} ({})", table.name, table.table_type);
/// }
/// catalog.drop_table("sales.staging").if_exists().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Catalog {
    session: Session,
}

impl Catalog {
    /// The session of this catalog.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// The name of the current catalog.
    pub fn current_catalog(&self) -> impl Future<Output = Result<String>> + Send + 'static {
        let current = proto::CurrentCatalog::default();
        run(&self.session, CatType::CurrentCatalog(current), answer_text)
    }

    /// Makes `name` the current catalog.
    pub fn set_current_catalog(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let set = proto::SetCurrentCatalog {
            catalog_name: name.into(),
        };
        run(&self.session, CatType::SetCurrentCatalog(set), nothing)
    }

    /// The catalogs of the server.
    pub fn list_catalogs(&self) -> ListCatalogs {
        ListCatalogs {
            session: self.session.clone(),
            message: proto::ListCatalogs::default(),
        }
    }

    /// The name of the current database.
    pub fn current_database(&self) -> impl Future<Output = Result<String>> + Send + 'static {
        let current = proto::CurrentDatabase::default();
        run(
            &self.session,
            CatType::CurrentDatabase(current),
            answer_text,
        )
    }

    /// Makes `name` the current database.
    pub fn set_current_database(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let set = proto::SetCurrentDatabase {
            db_name: name.into(),
        };
        run(&self.session, CatType::SetCurrentDatabase(set), nothing)
    }

    /// The databases of the current catalog.
    pub fn list_databases(&self) -> ListDatabases {
        ListDatabases {
            session: self.session.clone(),
            message: proto::ListDatabases::default(),
        }
    }

    /// The database `name`, which may be qualified by its catalog. Fails if it does not exist.
    pub fn get_database(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<Database>> + Send + 'static {
        let get = proto::GetDatabase {
            db_name: name.into(),
        };
        run(&self.session, CatType::GetDatabase(get), |rows| {
            first(rows, database, "database")
        })
    }

    /// Whether the database `name` exists.
    pub fn database_exists(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<bool>> + Send + 'static {
        let exists = proto::DatabaseExists {
            db_name: name.into(),
        };
        run(&self.session, CatType::DatabaseExists(exists), answer_bool)
    }

    /// The tables and views of the current database, temporary views included.
    pub fn list_tables(&self) -> ListTables {
        ListTables {
            session: self.session.clone(),
            message: proto::ListTables::default(),
        }
    }

    /// The table or view `name`. Fails if it does not exist.
    pub fn get_table(&self, name: impl Into<String>) -> GetTable {
        GetTable {
            session: self.session.clone(),
            message: proto::GetTable {
                table_name: name.into(),
                db_name: None,
            },
        }
    }

    /// Whether the table or view `name` exists.
    pub fn table_exists(&self, name: impl Into<String>) -> TableExists {
        TableExists {
            session: self.session.clone(),
            message: proto::TableExists {
                table_name: name.into(),
                db_name: None,
            },
        }
    }

    /// The columns of the table or view `table`.
    pub fn list_columns(&self, table: impl Into<String>) -> ListColumns {
        ListColumns {
            session: self.session.clone(),
            message: proto::ListColumns {
                table_name: table.into(),
                db_name: None,
            },
        }
    }

    /// The functions of the current database, built-in and temporary functions included.
    pub fn list_functions(&self) -> ListFunctions {
        ListFunctions {
            session: self.session.clone(),
            message: proto::ListFunctions::default(),
        }
    }

    /// The function `name`. Fails if it does not exist.
    pub fn get_function(&self, name: impl Into<String>) -> GetFunction {
        GetFunction {
            session: self.session.clone(),
            message: proto::GetFunction {
                function_name: name.into(),
                db_name: None,
            },
        }
    }

    /// Whether the function `name` exists.
    pub fn function_exists(&self, name: impl Into<String>) -> FunctionExists {
        FunctionExists {
            session: self.session.clone(),
            message: proto::FunctionExists {
                function_name: name.into(),
                db_name: None,
            },
        }
    }

    /// Creates the table `name`: awaiting [`CreateTable`] creates it, and returns a DataFrame
    /// of it.
    ///
    /// ```no_run
    /// # async fn run(session: spark_connect_async::Session) -> spark_connect_async::Result<()> {
    /// let events = session
    ///     .catalog()
    ///     .create_table("events")
    ///     .source("json")
    ///     .path("/data/events")
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_table(&self, name: impl Into<String>) -> CreateTable {
        CreateTable {
            session: self.session.clone(),
            message: proto::CreateTable {
                table_name: name.into(),
                ..Default::default()
            },
        }
    }

    /// Drops the temporary view `name` of the session, and returns whether it existed.
    pub fn drop_temp_view(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<bool>> + Send + 'static {
        let drop = proto::DropTempView {
            view_name: name.into(),
        };
        run(&self.session, CatType::DropTempView(drop), answer_bool)
    }

    /// Drops the global temporary view `name`, a view of every session of the server, and
    /// returns whether it existed.
    pub fn drop_global_temp_view(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<bool>> + Send + 'static {
        let drop = proto::DropGlobalTempView {
            view_name: name.into(),
        };
        run(
            &self.session,
            CatType::DropGlobalTempView(drop),
            answer_bool,
        )
    }

    /// Caches the rows of the table or view `name` on the server.
    pub fn cache_table(&self, name: impl Into<String>) -> CacheTable {
        CacheTable {
            session: self.session.clone(),
            message: proto::CacheTable {
                table_name: name.into(),
                storage_level: None,
            },
        }
    }

    /// Removes the table or view `name` from the cache.
    pub fn uncache_table(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let uncache = proto::UncacheTable {
            table_name: name.into(),
        };
        run(&self.session, CatType::UncacheTable(uncache), nothing)
    }

    /// Whether the table or view `name` is cached.
    pub fn is_cached(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<bool>> + Send + 'static {
        let cached = proto::IsCached {
            table_name: name.into(),
        };
        run(&self.session, CatType::IsCached(cached), answer_bool)
    }

    /// Removes every table and view from the cache.
    pub fn clear_cache(&self) -> impl Future<Output = Result<()>> + Send + 'static {
        let clear = proto::ClearCache::default();
        run(&self.session, CatType::ClearCache(clear), nothing)
    }

    /// Drops what the server caches of the data and metadata of the table `name`, such as after
    /// another program changed its files. A cached table is cached again when it is next used.
    pub fn refresh_table(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let refresh = proto::RefreshTable {
            table_name: name.into(),
        };
        run(&self.session, CatType::RefreshTable(refresh), nothing)
    }

    /// Drops, as [`Catalog::refresh_table`] does, what the server caches of every DataFrame
    /// that reads files under `path`.
    pub fn refresh_by_path(
        &self,
        path: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let refresh = proto::RefreshByPath { path: path.into() };
        run(&self.session, CatType::RefreshByPath(refresh), nothing)
    }

    /// Adds to the catalog the partitions of the table `name` that its directory has and the
    /// catalog lacks.
    pub fn recover_partitions(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let recover = proto::RecoverPartitions {
            table_name: name.into(),
        };
        run(&self.session, CatType::RecoverPartitions(recover), nothing)
    }

    /// Creates the database `name`, which must not exist yet unless
    /// [`CreateDatabase::if_not_exists`] says otherwise.
    pub fn create_database(&self, name: impl Into<String>) -> CreateDatabase {
        CreateDatabase {
            session: self.session.clone(),
            message: proto::CreateDatabase {
                db_name: name.into(),
                ..Default::default()
            },
        }
    }

    /// Drops the database `name`, which must exist and be empty unless [`DropDatabase`] says
    /// otherwise.
    pub fn drop_database(&self, name: impl Into<String>) -> DropDatabase {
        DropDatabase {
            session: self.session.clone(),
            message: proto::DropDatabase {
                db_name: name.into(),
                ..Default::default()
            },
        }
    }

    /// Drops the table `name`, which must exist unless [`DropTable::if_exists`] says otherwise.
    pub fn drop_table(&self, name: impl Into<String>) -> DropTable {
        DropTable {
            session: self.session.clone(),
            message: proto::DropTable {
                table_name: name.into(),
                ..Default::default()
            },
        }
    }

    /// Drops the view `name`, which must exist unless [`DropView::if_exists`] says otherwise.
    pub fn drop_view(&self, name: impl Into<String>) -> DropView {
        DropView {
            session: self.session.clone(),
            message: proto::DropView {
                view_name: name.into(),
                if_exists: false,
            },
        }
    }

    /// Removes every row of the table `name`.
    pub fn truncate_table(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let truncate = proto::TruncateTable {
            table_name: name.into(),
        };
        run(&self.session, CatType::TruncateTable(truncate), nothing)
    }

    /// Computes the statistics of the table `name`, which the optimizer uses.
    pub fn analyze_table(&self, name: impl Into<String>) -> AnalyzeTable {
        AnalyzeTable {
            session: self.session.clone(),
            message: proto::AnalyzeTable {
                table_name: name.into(),
                no_scan: false,
            },
        }
    }

    /// The `CREATE TABLE` statement of the table `name`.
    pub fn get_create_table_string(&self, name: impl Into<String>) -> GetCreateTableString {
        GetCreateTableString {
            session: self.session.clone(),
            message: proto::GetCreateTableString {
                table_name: name.into(),
                as_serde: false,
            },
        }
    }

    /// The properties of the table `name`.
    pub fn get_table_properties(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<BTreeMap<String, String>>> + Send + 'static {
        let get = proto::GetTableProperties {
            table_name: name.into(),
        };
        run(&self.session, CatType::GetTableProperties(get), properties)
    }

    /// The partitions of the table `name`.
    pub fn list_partitions(
        &self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<Vec<TablePartition>>> + Send + 'static {
        let list = proto::ListPartitions {
            table_name: name.into(),
        };
        run(&self.session, CatType::ListPartitions(list), |rows| {
            each(rows, partition)
        })
    }

    /// The views of the current database, temporary views included.
    pub fn list_views(&self) -> ListViews {
        ListViews {
            session: self.session.clone(),
            message: proto::ListViews::default(),
        }
    }
}

impl Session {
    /// The catalog of the session: its databases, tables, views and functions.
    pub fn catalog(&self) -> Catalog {
        Catalog {
            session: self.clone(),
        }
    }
}

/// The catalogs that [`Catalog::list_catalogs`] lists: await it for them.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct ListCatalogs {
    session: Session,
    message: proto::ListCatalogs,
}

impl ListCatalogs {
    /// Only the catalogs whose names match `pattern`, as in `SHOW CATALOGS LIKE`.
    pub fn pattern(mut self, pattern: impl Into<String>) -> Self {
        self.message.pattern = Some(pattern.into());
        self
    }
}

impl IntoFuture for ListCatalogs {
    type Output = Result<Vec<CatalogMetadata>>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let list = CatType::ListCatalogs(self.message);
        run(&self.session, list, |rows| each(rows, catalog_metadata))
    }
}

/// The databases that [`Catalog::list_databases`] lists: await it for them.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct ListDatabases {
    session: Session,
    message: proto::ListDatabases,
}

impl ListDatabases {
    /// Only the databases whose names match `pattern`, as in `SHOW DATABASES LIKE`.
    pub fn pattern(mut self, pattern: impl Into<String>) -> Self {
        self.message.pattern = Some(pattern.into());
        self
    }
}

impl IntoFuture for ListDatabases {
    type Output = Result<Vec<Database>>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let list = CatType::ListDatabases(self.message);
        run(&self.session, list, |rows| each(rows, database))
    }
}

/// The tables that [`Catalog::list_tables`] lists: await it for them.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct ListTables {
    session: Session,
    message: proto::ListTables,
}

impl ListTables {
    /// The tables of the database `database`, which may be qualified by its catalog, rather
    /// than of the current database.
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.message.db_name = Some(database.into());
        self
    }

    /// Only the tables whose names match `pattern`, as in `SHOW TABLES LIKE`.
    pub fn pattern(mut self, pattern: impl Into<String>) -> Self {
        self.message.pattern = Some(pattern.into());
        self
    }
}

impl IntoFuture for ListTables {
    type Output = Result<Vec<Table>>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(&self.session, CatType::ListTables(self.message), |rows| {
            each(rows, table)
        })
    }
}

/// The table that [`Catalog::get_table`] gets: await it for it.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct GetTable {
    session: Session,
    message: proto::GetTable,
}

impl GetTable {
    /// Look the table up in the database `database` rather than in the current database.
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.message.db_name = Some(database.into());
        self
    }
}

impl IntoFuture for GetTable {
    type Output = Result<Table>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(&self.session, CatType::GetTable(self.message), |rows| {
            first(rows, table, "table")
        })
    }
}

/// The check of [`Catalog::table_exists`]: await it for its answer.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct TableExists {
    session: Session,
    message: proto::TableExists,
}

impl TableExists {
    /// Look the table up in the database `database` rather than in the current database.
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.message.db_name = Some(database.into());
        self
    }
}

impl IntoFuture for TableExists {
    type Output = Result<bool>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let exists = CatType::TableExists(self.message);
        run(&self.session, exists, answer_bool)
    }
}

/// The columns that [`Catalog::list_columns`] lists: await it for them.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct ListColumns {
    session: Session,
    message: proto::ListColumns,
}

impl ListColumns {
    /// Look the table up in the database `database` rather than in the current database.
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.message.db_name = Some(database.into());
        self
    }
}

impl IntoFuture for ListColumns {
    type Output = Result<Vec<CatalogColumn>>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(&self.session, CatType::ListColumns(self.message), |rows| {
            each(rows, column)
        })
    }
}

/// The functions that [`Catalog::list_functions`] lists: await it for them.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct ListFunctions {
    session: Session,
    message: proto::ListFunctions,
}

impl ListFunctions {
    /// The functions of the database `database`, which may be qualified by its catalog, rather
    /// than of the current database.
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.message.db_name = Some(database.into());
        self
    }

    /// Only the functions whose names match `pattern`, as in `SHOW FUNCTIONS LIKE`.
    pub fn pattern(mut self, pattern: impl Into<String>) -> Self {
        self.message.pattern = Some(pattern.into());
        self
    }
}

impl IntoFuture for ListFunctions {
    type Output = Result<Vec<Function>>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let list = CatType::ListFunctions(self.message);
        run(&self.session, list, |rows| each(rows, function))
    }
}

/// The function that [`Catalog::get_function`] gets: await it for it.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct GetFunction {
    session: Session,
    message: proto::GetFunction,
}

impl GetFunction {
    /// Look the function up in the database `database` rather than in the current database.
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.message.db_name = Some(database.into());
        self
    }
}

impl IntoFuture for GetFunction {
    type Output = Result<Function>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(&self.session, CatType::GetFunction(self.message), |rows| {
            first(rows, function, "function")
        })
    }
}

/// The check of [`Catalog::function_exists`]: await it for its answer.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct FunctionExists {
    session: Session,
    message: proto::FunctionExists,
}

impl FunctionExists {
    /// Look the function up in the database `database` rather than in the current database.
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.message.db_name = Some(database.into());
        self
    }
}

impl IntoFuture for FunctionExists {
    type Output = Result<bool>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let exists = CatType::FunctionExists(self.message);
        run(&self.session, exists, answer_bool)
    }
}

/// The creation of the table of [`Catalog::create_table`]: await it to create the table. By
/// default, a managed table of the data source of `spark.sql.sources.default`.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct CreateTable {
    session: Session,
    message: proto::CreateTable,
}

impl CreateTable {
    /// The files of the table: a table with a path is external, and dropping it keeps them.
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.message.path = Some(path.into());
        self
    }

    /// The data source of the table, such as `parquet`.
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.message.source = Some(source.into());
        self
    }

    /// The schema of the table, a struct type; without one, the data source infers it from the
    /// files.
    pub fn schema(mut self, schema: DataType) -> Self {
        self.message.schema = Some(schema.to_proto());
        self
    }

    /// The description of the table.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.message.description = Some(description.into());
        self
    }

    /// Sets an option of the data source.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<OptionValue>) -> Self {
        insert_options(&mut self.message.options, [(key, value)]);
        self
    }

    /// Sets options of the data source.
    pub fn options<K: Into<String>, V: Into<OptionValue>>(
        mut self,
        options: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        insert_options(&mut self.message.options, options);
        self
    }
}

impl IntoFuture for CreateTable {
    type Output = Result<DataFrame>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let table = self.session.table(self.message.table_name.clone());
        run(&self.session, CatType::CreateTable(self.message), |_| {
            Ok(table)
        })
    }
}

/// The caching of the table of [`Catalog::cache_table`]: await it to cache the table.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct CacheTable {
    session: Session,
    message: proto::CacheTable,
}

impl CacheTable {
    /// Where the server keeps the rows, instead of its default storage level.
    pub fn storage_level(mut self, level: StorageLevel) -> Self {
        self.message.storage_level = Some(level.proto());
        self
    }
}

impl IntoFuture for CacheTable {
    type Output = Result<()>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(&self.session, CatType::CacheTable(self.message), nothing)
    }
}

/// The creation of the database of [`Catalog::create_database`]: await it to create the
/// database.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct CreateDatabase {
    session: Session,
    message: proto::CreateDatabase,
}

impl CreateDatabase {
    /// Do nothing if the database exists.
    pub fn if_not_exists(mut self) -> Self {
        self.message.if_not_exists = true;
        self
    }

    /// Sets a property of the database.
    pub fn property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.message.properties.insert(key.into(), value.into());
        self
    }
}

impl IntoFuture for CreateDatabase {
    type Output = Result<()>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(
            &self.session,
            CatType::CreateDatabase(self.message),
            nothing,
        )
    }
}

/// The drop of the database of [`Catalog::drop_database`]: await it to drop the database.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct DropDatabase {
    session: Session,
    message: proto::DropDatabase,
}

impl DropDatabase {
    /// Do nothing if the database does not exist.
    pub fn if_exists(mut self) -> Self {
        self.message.if_exists = true;
        self
    }

    /// Drop the tables and functions of the database too.
    pub fn cascade(mut self) -> Self {
        self.message.cascade = true;
        self
    }
}

impl IntoFuture for DropDatabase {
    type Output = Result<()>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(&self.session, CatType::DropDatabase(self.message), nothing)
    }
}

/// The drop of the table of [`Catalog::drop_table`]: await it to drop the table.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct DropTable {
    session: Session,
    message: proto::DropTable,
}

impl DropTable {
    /// Do nothing if the table does not exist.
    pub fn if_exists(mut self) -> Self {
        self.message.if_exists = true;
        self
    }

    /// Delete the files of a managed table at once, rather than moving them to the trash of
    /// the file system.
    pub fn purge(mut self) -> Self {
        self.message.purge = true;
        self
    }
}

impl IntoFuture for DropTable {
    type Output = Result<()>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(&self.session, CatType::DropTable(self.message), nothing)
    }
}

/// The drop of the view of [`Catalog::drop_view`]: await it to drop the view.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct DropView {
    session: Session,
    message: proto::DropView,
}

impl DropView {
    /// Do nothing if the view does not exist.
    pub fn if_exists(mut self) -> Self {
        self.message.if_exists = true;
        self
    }
}

impl IntoFuture for DropView {
    type Output = Result<()>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(&self.session, CatType::DropView(self.message), nothing)
    }
}

/// The analysis of the table of [`Catalog::analyze_table`]: await it to compute the
/// statistics.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct AnalyzeTable {
    session: Session,
    message: proto::AnalyzeTable,
}

impl AnalyzeTable {
    /// Compute only the statistics that need no scan of the rows, such as the size of the
    /// files.
    pub fn no_scan(mut self) -> Self {
        self.message.no_scan = true;
        self
    }
}

impl IntoFuture for AnalyzeTable {
    type Output = Result<()>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(&self.session, CatType::AnalyzeTable(self.message), nothing)
    }
}

/// The statement that [`Catalog::get_create_table_string`] gets: await it for it.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct GetCreateTableString {
    session: Session,
    message: proto::GetCreateTableString,
}

impl GetCreateTableString {
    /// The statement of a Hive table in Hive's syntax, with its SerDe.
    pub fn as_serde(mut self) -> Self {
        self.message.as_serde = true;
        self
    }
}

impl IntoFuture for GetCreateTableString {
    type Output = Result<String>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let get = CatType::GetCreateTableString(self.message);
        run(&self.session, get, statement)
    }
}

/// The views that [`Catalog::list_views`] lists: await it for them.
#[must_use = "a catalog operation does nothing until it is awaited"]
pub struct ListViews {
    session: Session,
    message: proto::ListViews,
}

impl ListViews {
    /// The views of the database `database`, which may be qualified by its catalog, rather
    /// than of the current database.
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.message.db_name = Some(database.into());
        self
    }

    /// Only the views whose names match `pattern`, as in `SHOW VIEWS LIKE`.
    pub fn pattern(mut self, pattern: impl Into<String>) -> Self {
        self.message.pattern = Some(pattern.into());
        self
    }
}

impl IntoFuture for ListViews {
    type Output = Result<Vec<Table>>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        run(&self.session, CatType::ListViews(self.message), |rows| {
            each(rows, table)
        })
    }
}

/// Executes the catalog operation `operation` of `session`, and reads the rows of its result
/// with `read`.
fn run<T: Send + 'static>(
    session: &Session,
    operation: CatType,
    read: impl FnOnce(Vec<CoreRow>) -> Result<T> + Send + 'static,
) -> BoxFuture<'static, Result<T>> {
    let plan = Arc::new(LogicalPlan::Catalog {
        catalog: proto::Catalog {
            cat_type: Some(operation),
        },
    });
    let relation = relation_with_ids(&plan, session.plan_ids());
    let rows = stream::catalog_rows(session.clone(), relation);
    async move { read(rows.await?) }.boxed()
}

fn nothing(_: Vec<CoreRow>) -> Result<()> {
    Ok(())
}

fn answer_text(rows: Vec<CoreRow>) -> Result<String> {
    first(rows, |row| text(row, 0, "value"), "answer")
}

fn answer_bool(rows: Vec<CoreRow>) -> Result<bool> {
    first(rows, |row| boolean(row, 0, "value"), "answer")
}

/// The statement of the result of `GetCreateTableString`: empty if it has no row.
fn statement(rows: Vec<CoreRow>) -> Result<String> {
    if rows.is_empty() {
        return Ok(String::new());
    }
    answer_text(rows)
}

/// Each row of `rows` as `read` reads it.
fn each<T>(rows: Vec<CoreRow>, read: impl Fn(&CoreRow) -> Result<T>) -> Result<Vec<T>> {
    rows.iter()
        .enumerate()
        .map(|(index, row)| read(row).map_err(|error| error.in_row(index)))
        .collect()
}

/// The first row of `rows`, a result of one `what`, as `read` reads it.
fn first<T>(rows: Vec<CoreRow>, read: impl Fn(&CoreRow) -> Result<T>, what: &str) -> Result<T> {
    let row = rows
        .first()
        .ok_or_else(|| Error::decode(format!("the server returned no {what}")))?;
    read(row).map_err(|error| error.in_row(0))
}

// The columns of a result are read by position; their names only label errors.

fn catalog_metadata(row: &CoreRow) -> Result<CatalogMetadata> {
    Ok(CatalogMetadata {
        name: text(row, 0, "name")?,
        description: optional_text(row, 1, "description")?,
    })
}

fn database(row: &CoreRow) -> Result<Database> {
    Ok(Database {
        name: text(row, 0, "name")?,
        catalog: optional_text(row, 1, "catalog")?,
        description: optional_text(row, 2, "description")?,
        location_uri: optional_text(row, 3, "locationUri")?,
    })
}

fn table(row: &CoreRow) -> Result<Table> {
    Ok(Table {
        name: text(row, 0, "name")?,
        catalog: optional_text(row, 1, "catalog")?,
        namespace: optional_texts(row, 2, "namespace")?,
        description: optional_text(row, 3, "description")?,
        table_type: text(row, 4, "tableType")?,
        is_temporary: boolean(row, 5, "isTemporary")?,
    })
}

fn column(row: &CoreRow) -> Result<CatalogColumn> {
    Ok(CatalogColumn {
        name: text(row, 0, "name")?,
        description: optional_text(row, 1, "description")?,
        data_type: text(row, 2, "dataType")?,
        nullable: boolean(row, 3, "nullable")?,
        is_partition: boolean(row, 4, "isPartition")?,
        is_bucket: boolean(row, 5, "isBucket")?,
        is_cluster: boolean(row, 6, "isCluster")?,
    })
}

fn function(row: &CoreRow) -> Result<Function> {
    Ok(Function {
        name: text(row, 0, "name")?,
        catalog: optional_text(row, 1, "catalog")?,
        namespace: optional_texts(row, 2, "namespace")?,
        description: optional_text(row, 3, "description")?,
        class_name: optional_text(row, 4, "className")?,
        is_temporary: boolean(row, 5, "isTemporary")?,
    })
}

fn partition(row: &CoreRow) -> Result<TablePartition> {
    Ok(TablePartition {
        partition: text(row, 0, "partition")?,
    })
}

fn properties(rows: Vec<CoreRow>) -> Result<BTreeMap<String, String>> {
    let property = |row: &CoreRow| Ok((text(row, 0, "key")?, text(row, 1, "value")?));
    each(rows, property).map(BTreeMap::from_iter)
}

/// The value of the column `name`, the `index`th, of `row` as `read` reads it, or `None` for a
/// NULL. `expected` says what `read` reads.
fn cell<T>(
    row: &CoreRow,
    index: usize,
    name: &str,
    read: impl FnOnce(&CoreValue) -> Option<T>,
    expected: &str,
) -> Result<Option<T>> {
    let value = row
        .get(index)
        .ok_or_else(|| Error::decode("a result without this column").in_field(name))?;
    if value.is_null() {
        return Ok(None);
    }
    read(value)
        .map(Some)
        .ok_or_else(|| Error::decode(format!("a value that is not {expected}")).in_field(name))
}

fn required<T>(value: Option<T>, name: &str) -> Result<T> {
    value.ok_or_else(|| Error::decode("a NULL in a column that always has a value").in_field(name))
}

fn optional_text(row: &CoreRow, index: usize, name: &str) -> Result<Option<String>> {
    cell(
        row,
        index,
        name,
        |value| value.as_str().map(str::to_owned),
        "text",
    )
}

fn text(row: &CoreRow, index: usize, name: &str) -> Result<String> {
    required(optional_text(row, index, name)?, name)
}

fn boolean(row: &CoreRow, index: usize, name: &str) -> Result<bool> {
    required(
        cell(row, index, name, CoreValue::as_bool, "a boolean")?,
        name,
    )
}

fn optional_texts(row: &CoreRow, index: usize, name: &str) -> Result<Option<Vec<String>>> {
    let texts = |value: &CoreValue| match value {
        CoreValue::List(items) => items
            .iter()
            .map(|item| item.as_str().map(str::to_owned))
            .collect(),
        _ => None,
    };
    cell(row, index, name, texts, "a list of texts")
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use arrow::array::{ArrayRef, BooleanArray, ListBuilder, StringArray, StringBuilder};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;
    use spark_connect_core::decode::decode_arrow_batch;

    use super::*;
    use crate::ErrorKind;

    /// The rows that the core decodes from a result of `columns`, sent as the server sends it.
    fn rows(columns: Vec<(&str, ArrayRef)>) -> Vec<CoreRow> {
        let batch = RecordBatch::try_from_iter(columns).unwrap();
        let mut data = Vec::new();
        let mut writer = StreamWriter::try_new(&mut data, &batch.schema()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);
        decode_arrow_batch(&proto::execute_plan_response::ArrowBatch {
            row_count: batch.num_rows() as i64,
            data: data.into(),
            ..Default::default()
        })
        .unwrap()
    }

    fn texts(values: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(values.to_vec()))
    }

    fn booleans(values: &[bool]) -> ArrayRef {
        Arc::new(BooleanArray::from(values.to_vec()))
    }

    fn lists(values: &[Option<&[Option<&str>]>]) -> ArrayRef {
        let mut builder = ListBuilder::new(StringBuilder::new());
        for value in values {
            if let Some(items) = value {
                for item in *items {
                    builder.values().append_option(*item);
                }
            }
            builder.append(value.is_some());
        }
        Arc::new(builder.finish())
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn some(value: &str) -> Option<String> {
        Some(value.to_string())
    }

    fn tables(
        names: &[Option<&str>],
        namespaces: &[Option<&[Option<&str>]>],
        temporary: ArrayRef,
    ) -> Vec<CoreRow> {
        let catalogs: Vec<Option<&str>> = vec![Some("spark_catalog"); names.len()];
        let types: Vec<Option<&str>> = vec![Some("MANAGED"); names.len()];
        rows(vec![
            ("name", texts(names)),
            ("catalog", texts(&catalogs)),
            ("namespace", lists(namespaces)),
            ("description", texts(&vec![None; names.len()])),
            ("tableType", texts(&types)),
            ("isTemporary", temporary),
        ])
    }

    #[test]
    fn the_results_are_read_with_their_nulls() {
        fn case<T: Debug>(name: &str, read: Result<T>, expected: T) -> (String, String, String) {
            let expected: Result<T> = Ok(expected);
            (
                name.to_string(),
                format!("{read:?}"),
                format!("{expected:?}"),
            )
        }
        let catalogs = rows(vec![
            ("name", texts(&[Some("spark_catalog"), Some("lake")])),
            ("description", texts(&[None, Some("the lake")])),
        ]);
        let databases = rows(vec![
            ("name", texts(&[Some("default"), Some("staging")])),
            ("catalog", texts(&[Some("spark_catalog"), None])),
            ("description", texts(&[Some("the default database"), None])),
            ("locationUri", texts(&[Some("file:/warehouse"), None])),
        ]);
        let empty: &[Option<&str>] = &[];
        let tables = rows(vec![
            (
                "name",
                texts(&[Some("orders"), Some("recent"), Some("raw")]),
            ),
            (
                "catalog",
                texts(&[Some("spark_catalog"), None, Some("lake")]),
            ),
            (
                "namespace",
                lists(&[Some(&[Some("sales")]), Some(empty), None]),
            ),
            ("description", texts(&[Some("every order"), None, None])),
            (
                "tableType",
                texts(&[Some("MANAGED"), Some("TEMPORARY"), Some("EXTERNAL")]),
            ),
            ("isTemporary", booleans(&[false, true, false])),
        ]);
        let columns = rows(vec![
            ("name", texts(&[Some("id"), Some("day")])),
            ("description", texts(&[Some("the id"), None])),
            ("dataType", texts(&[Some("int"), Some("array<string>")])),
            ("nullable", booleans(&[false, true])),
            ("isPartition", booleans(&[false, true])),
            ("isBucket", booleans(&[true, false])),
            ("isCluster", booleans(&[false, true])),
        ]);
        let functions = rows(vec![
            ("name", texts(&[Some("abs"), Some("tax")])),
            ("catalog", texts(&[None, Some("spark_catalog")])),
            ("namespace", lists(&[None, Some(&[Some("sales")])])),
            ("description", texts(&[Some("the absolute value"), None])),
            ("className", texts(&[Some("org.apache.spark.Abs"), None])),
            ("isTemporary", booleans(&[true, false])),
        ]);
        let partitions = rows(vec![("partition", texts(&[Some("day=2024-01-01")]))]);
        let table_properties = rows(vec![
            ("key", texts(&[Some("owner"), Some("kind")])),
            ("value", texts(&[Some("sales"), Some("fact")])),
        ]);
        let text = rows(vec![("value", texts(&[Some("spark_catalog")]))]);
        let answer = rows(vec![("value", booleans(&[true]))]);
        let cases = vec![
            case(
                "catalogs",
                each(catalogs, catalog_metadata),
                vec![
                    CatalogMetadata {
                        name: "spark_catalog".to_string(),
                        description: None,
                    },
                    CatalogMetadata {
                        name: "lake".to_string(),
                        description: some("the lake"),
                    },
                ],
            ),
            case(
                "databases",
                each(databases, database),
                vec![
                    Database {
                        name: "default".to_string(),
                        catalog: some("spark_catalog"),
                        description: some("the default database"),
                        location_uri: some("file:/warehouse"),
                    },
                    Database {
                        name: "staging".to_string(),
                        catalog: None,
                        description: None,
                        location_uri: None,
                    },
                ],
            ),
            case(
                "tables",
                each(tables, table),
                vec![
                    Table {
                        name: "orders".to_string(),
                        catalog: some("spark_catalog"),
                        namespace: Some(strings(&["sales"])),
                        description: some("every order"),
                        table_type: "MANAGED".to_string(),
                        is_temporary: false,
                    },
                    Table {
                        name: "recent".to_string(),
                        catalog: None,
                        namespace: Some(Vec::new()),
                        description: None,
                        table_type: "TEMPORARY".to_string(),
                        is_temporary: true,
                    },
                    Table {
                        name: "raw".to_string(),
                        catalog: some("lake"),
                        namespace: None,
                        description: None,
                        table_type: "EXTERNAL".to_string(),
                        is_temporary: false,
                    },
                ],
            ),
            case(
                "columns",
                each(columns, column),
                vec![
                    CatalogColumn {
                        name: "id".to_string(),
                        description: some("the id"),
                        data_type: "int".to_string(),
                        nullable: false,
                        is_partition: false,
                        is_bucket: true,
                        is_cluster: false,
                    },
                    CatalogColumn {
                        name: "day".to_string(),
                        description: None,
                        data_type: "array<string>".to_string(),
                        nullable: true,
                        is_partition: true,
                        is_bucket: false,
                        is_cluster: true,
                    },
                ],
            ),
            case(
                "functions",
                each(functions, function),
                vec![
                    Function {
                        name: "abs".to_string(),
                        catalog: None,
                        namespace: None,
                        description: some("the absolute value"),
                        class_name: some("org.apache.spark.Abs"),
                        is_temporary: true,
                    },
                    Function {
                        name: "tax".to_string(),
                        catalog: some("spark_catalog"),
                        namespace: Some(strings(&["sales"])),
                        description: None,
                        class_name: None,
                        is_temporary: false,
                    },
                ],
            ),
            case(
                "partitions",
                each(partitions, partition),
                vec![TablePartition {
                    partition: "day=2024-01-01".to_string(),
                }],
            ),
            case(
                "properties",
                properties(table_properties),
                BTreeMap::from([
                    ("kind".to_string(), "fact".to_string()),
                    ("owner".to_string(), "sales".to_string()),
                ]),
            ),
            case(
                "a text",
                answer_text(text.clone()),
                "spark_catalog".to_string(),
            ),
            case("an answer", answer_bool(answer), true),
            case("a statement", statement(text), "spark_catalog".to_string()),
            case("no statement", statement(Vec::new()), String::new()),
        ];
        let failed: Vec<String> = cases
            .iter()
            .filter(|(_, read, expected)| read != expected)
            .map(|(name, read, expected)| {
                format!("{name}:\n  read     {read}\n  expected {expected}")
            })
            .collect();
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }

    #[test]
    fn a_result_that_does_not_fit_its_type_is_a_decode_error() {
        let one = |temporary: bool| booleans(&[temporary]);
        let cases: Vec<(&str, Result<Vec<Table>>, &str)> = vec![
            (
                "a NULL name",
                each(
                    tables(&[Some("a"), None], &[None, None], booleans(&[false, true])),
                    table,
                ),
                "a NULL in a column that always has a value (row 1, column `name`)",
            ),
            (
                "a text for a boolean",
                each(tables(&[Some("a")], &[None], texts(&[Some("true")])), table),
                "a value that is not a boolean (row 0, column `isTemporary`)",
            ),
            (
                "a NULL in a namespace",
                each(
                    tables(&[Some("a")], &[Some(&[Some("db"), None])], one(false)),
                    table,
                ),
                "a value that is not a list of texts (row 0, column `namespace`)",
            ),
            (
                "a missing column",
                each(rows(vec![("name", texts(&[Some("a")]))]), table),
                "a result without this column (row 0, column `catalog`)",
            ),
            (
                "a NULL name of the one table",
                first(tables(&[None], &[None], one(true)), table, "table").map(|t| vec![t]),
                "a NULL in a column that always has a value (row 0, column `name`)",
            ),
            (
                "no table",
                first(Vec::new(), table, "table").map(|table| vec![table]),
                "the server returned no table",
            ),
            (
                "no answer",
                answer_bool(Vec::new()).map(|_| Vec::new()),
                "the server returned no answer",
            ),
        ];
        let failed: Vec<String> = cases
            .into_iter()
            .filter_map(|(name, read, expected)| match read {
                Err(error) if error.kind() == ErrorKind::Decode && error.message() == expected => {
                    None
                }
                other => Some(format!(
                    "{name}: {other:?}, expected the error {expected:?}"
                )),
            })
            .collect();
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
