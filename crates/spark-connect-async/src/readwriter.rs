//! DataFrames read from data sources, tables and local data, and their writes to data sources
//! and tables.
//!
//! ```no_run
//! # async fn run(session: spark_connect_async::Session) -> spark_connect_async::Result<()> {
//! use spark_connect_async::readwriter::SaveMode;
//!
//! let people = session
//!     .read()
//!     .schema("name STRING, age INT")
//!     .option("header", true)
//!     .csv("/data/people.csv");
//! people
//!     .write()
//!     .mode(SaveMode::Overwrite)
//!     .partition_by(["age"])
//!     .parquet("/data/people")
//!     .await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use futures::FutureExt;
use spark_connect_core::actions::session as core_session;
use spark_connect_core::arrow_input::create_relation;
use spark_connect_core::local_rows::{rows_to_arrow_batch, small_local_relation};
use spark_connect_core::plan::{self, LogicalPlan, ParseFormat as CoreParseFormat};
use spark_connect_core::readwriter::{
    table_destination, ReadType, SaveMode as CoreSaveMode, TableSaveMethod, WriteSpec, WriteV2Spec,
};
use spark_connect_core::row::Row as CoreRow;
use spark_connect_core::types::DataType;
use spark_connect_proto as proto;
use spark_connect_proto::write_operation::SaveType;
use spark_connect_proto::write_operation_v2::Mode;

use crate::column::{to_core, Column, IntoDataType, TypeOrDdl};
use crate::dataframe::{strings, DataFrame};
use crate::error::{Error, Result};
use crate::session::Session;
use crate::stream;
use crate::value::Value;

/// The value of an option of a read or a write, which is sent as text: a string as it is, a
/// `bool` as `true` or `false`, and a number as Rust displays it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionValue(String);

impl From<&str> for OptionValue {
    fn from(value: &str) -> Self {
        OptionValue(value.to_string())
    }
}

impl From<String> for OptionValue {
    fn from(value: String) -> Self {
        OptionValue(value)
    }
}

impl From<&String> for OptionValue {
    fn from(value: &String) -> Self {
        OptionValue(value.clone())
    }
}

macro_rules! option_value_from_display {
    ($($t:ty),*) => {
        $(impl From<$t> for OptionValue {
            fn from(value: $t) -> Self {
                OptionValue(value.to_string())
            }
        })*
    };
}

option_value_from_display!(bool, i32, i64, f64);

pub(crate) fn insert_options<K: Into<String>, V: Into<OptionValue>>(
    options: &mut HashMap<String, String>,
    pairs: impl IntoIterator<Item = (K, V)>,
) {
    options.extend(
        pairs
            .into_iter()
            .map(|(key, value)| (key.into(), value.into().0)),
    );
}

/// How a write treats data that its destination already has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaveMode {
    /// Add the rows to the existing data.
    Append,
    /// Replace the existing data.
    Overwrite,
    /// Fail when there is data.
    #[default]
    ErrorIfExists,
    /// Write nothing when there is data.
    Ignore,
}

impl SaveMode {
    fn core(self) -> CoreSaveMode {
        match self {
            SaveMode::Append => CoreSaveMode::Append,
            SaveMode::Overwrite => CoreSaveMode::Overwrite,
            SaveMode::ErrorIfExists => CoreSaveMode::ErrorIfExists,
            SaveMode::Ignore => CoreSaveMode::Ignore,
        }
    }
}

/// Parses `append`, `overwrite`, `error` or `errorifexists`, and `ignore`, in any case.
impl FromStr for SaveMode {
    type Err = Error;

    fn from_str(mode: &str) -> Result<SaveMode> {
        match CoreSaveMode::from_str(mode) {
            Some(CoreSaveMode::Append) => Ok(SaveMode::Append),
            Some(CoreSaveMode::Overwrite) => Ok(SaveMode::Overwrite),
            Some(CoreSaveMode::ErrorIfExists) => Ok(SaveMode::ErrorIfExists),
            Some(CoreSaveMode::Ignore) => Ok(SaveMode::Ignore),
            None => Err(Error::invalid_argument(format!(
                "{mode:?} is not a save mode: append, overwrite, error, errorifexists or ignore"
            ))),
        }
    }
}

/// The format of the strings that [`DataFrameReader::parse`] parses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseFormat {
    /// CSV.
    Csv,
    /// JSON.
    Json,
    /// XML.
    Xml,
}

impl ParseFormat {
    fn core(self) -> CoreParseFormat {
        match self {
            ParseFormat::Csv => CoreParseFormat::Csv,
            ParseFormat::Json => CoreParseFormat::Json,
            ParseFormat::Xml => CoreParseFormat::Xml,
        }
    }
}

/// A read of a DataFrame from a data source, from [`Session::read`]: its format, schema and
/// options, then the method that names what to read and returns the DataFrame.
///
/// The read happens at the DataFrame's actions. A later option of the same key replaces an
/// earlier one.
#[derive(Clone)]
#[must_use = "a reader reads nothing until a method names what to read"]
pub struct DataFrameReader {
    session: Session,
    format: Option<String>,
    schema: Option<TypeOrDdl>,
    options: HashMap<String, String>,
    streaming: bool,
    source_name: Option<String>,
}

impl DataFrameReader {
    /// A read of a stream, as [`crate::streaming::DataStreamReader`] makes it.
    pub(crate) fn of_stream(session: Session) -> Self {
        DataFrameReader {
            streaming: true,
            ..session.read()
        }
    }

    /// Read the source under the name `name`, which a stream's checkpoint keeps its progress
    /// under.
    pub(crate) fn source_name(mut self, name: String) -> Self {
        self.source_name = Some(name);
        self
    }

    /// Read the format `source`, such as `parquet`; by default, the server's default source.
    pub fn format(mut self, source: impl Into<String>) -> Self {
        self.format = Some(source.into());
        self
    }

    /// Read with this schema, rather than the one that the source infers: a [`DataType`], or a
    /// type in Spark SQL's syntax or as JSON text.
    pub fn schema(mut self, schema: impl IntoDataType) -> Self {
        self.schema = Some(schema.into_type_or_ddl());
        self
    }

    /// Set the option `key` of the source.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<OptionValue>) -> Self {
        self.options.insert(key.into(), value.into().0);
        self
    }

    /// Set the options of the source.
    pub fn options<K: Into<String>, V: Into<OptionValue>>(
        mut self,
        options: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        insert_options(&mut self.options, options);
        self
    }

    fn data_source(self, paths: Vec<String>, predicates: Vec<String>) -> DataFrame {
        let schema = self.schema.map(|schema| match schema {
            TypeOrDdl::Type(data_type) => data_type.json(),
            TypeOrDdl::Text(ddl) => ddl,
        });
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format,
                schema,
                options: self.options,
                paths,
                predicates,
                source_name: self.source_name,
            },
            is_streaming: self.streaming,
        };
        DataFrame::new(self.session, plan)
    }

    /// The DataFrame of the data at `path`.
    pub fn load(self, path: impl Into<String>) -> DataFrame {
        self.data_source(vec![path.into()], Vec::new())
    }

    /// The DataFrame of the data at `paths`.
    pub fn load_paths(self, paths: impl IntoIterator<Item = impl Into<String>>) -> DataFrame {
        self.data_source(strings(paths), Vec::new())
    }

    /// The DataFrame of the data that the source's options name, for a source that takes no
    /// path.
    pub fn load_without_path(self) -> DataFrame {
        self.data_source(Vec::new(), Vec::new())
    }

    /// The DataFrame of the CSV files at `path`.
    pub fn csv(self, path: impl Into<String>) -> DataFrame {
        self.format("csv").load(path)
    }

    /// The DataFrame of the JSON Lines files at `path`.
    pub fn json(self, path: impl Into<String>) -> DataFrame {
        self.format("json").load(path)
    }

    /// The DataFrame of the ORC files at `path`.
    pub fn orc(self, path: impl Into<String>) -> DataFrame {
        self.format("orc").load(path)
    }

    /// The DataFrame of the Parquet files at `path`.
    pub fn parquet(self, path: impl Into<String>) -> DataFrame {
        self.format("parquet").load(path)
    }

    /// The DataFrame of the lines of the text files at `path`, in a column `value`.
    pub fn text(self, path: impl Into<String>) -> DataFrame {
        self.format("text").load(path)
    }

    /// The DataFrame of the XML files at `path`.
    pub fn xml(self, path: impl Into<String>) -> DataFrame {
        self.format("xml").load(path)
    }

    /// The DataFrame of the table `table` of the database at the JDBC `url`, read in one
    /// partition per predicate of `predicates` (a `WHERE` condition), else in one.
    pub fn jdbc(
        mut self,
        url: impl Into<String>,
        table: impl Into<String>,
        predicates: Option<&[&str]>,
    ) -> DataFrame {
        self.format = Some("jdbc".to_string());
        self.options.insert("url".to_string(), url.into());
        self.options.insert("dbtable".to_string(), table.into());
        self.data_source(
            Vec::new(),
            strings(predicates.unwrap_or(&[]).iter().copied()),
        )
    }

    /// The DataFrame of the table or view `name`, read with the options of this reader.
    pub fn table(self, name: impl Into<String>) -> DataFrame {
        let plan = LogicalPlan::Read {
            read_type: ReadType::NamedTable {
                table_name: name.into(),
                options: self.options,
            },
            is_streaming: self.streaming,
        };
        DataFrame::new(self.session, plan)
    }

    /// The DataFrame of the changes of the table `name`, which the options of this reader
    /// select, such as `startingVersion`.
    pub fn changes(self, name: impl Into<String>) -> DataFrame {
        let plan = LogicalPlan::RelationChanges {
            table_name: name.into(),
            options: self.options,
            is_streaming: self.streaming.then_some(true),
        };
        DataFrame::new(self.session, plan)
    }

    /// The DataFrame of the strings of `input`, a DataFrame of one string column, parsed as
    /// `format` with the options of this reader, and with its schema, else one that the server
    /// infers. A schema in Spark SQL's syntax is parsed by the server first, and a schema as
    /// JSON text by the client.
    pub fn parse(
        self,
        format: ParseFormat,
        input: &DataFrame,
    ) -> impl Future<Output = Result<DataFrame>> + Send + 'static {
        let input = Arc::clone(input.plan());
        async move {
            let schema = match self.schema {
                None => None,
                Some(TypeOrDdl::Type(data_type)) => Some(data_type),
                Some(TypeOrDdl::Text(json)) if json.starts_with('{') => {
                    Some(DataType::from_json_str(&json)?)
                }
                Some(TypeOrDdl::Text(ddl)) => {
                    Some(core_session::parse_ddl(self.session.client(), ddl).await?)
                }
            };
            let plan = LogicalPlan::Parse {
                input,
                format: format.core(),
                schema,
                options: self.options,
            };
            Ok(DataFrame::new(self.session, plan))
        }
    }
}

/// A write of a DataFrame to a data source or a table, from [`DataFrame::write`]: its format,
/// mode, options and layout, then the method that names the destination and returns the
/// future of the write.
#[derive(Clone)]
#[must_use = "a write needs a method that names its destination"]
pub struct DataFrameWriter {
    input: DataFrame,
    spec: WriteSpec,
}

impl DataFrameWriter {
    /// Write the format `source`, such as `parquet`; by default, the server's default source.
    pub fn format(mut self, source: impl Into<String>) -> Self {
        self.spec.format = Some(source.into());
        self
    }

    /// Treat the data of the destination as `mode` says; by default
    /// [`SaveMode::ErrorIfExists`].
    pub fn mode(mut self, mode: SaveMode) -> Self {
        self.spec.mode = mode.core();
        self
    }

    /// Set the option `key` of the source.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<OptionValue>) -> Self {
        self.spec.options.insert(key.into(), value.into().0);
        self
    }

    /// Set the options of the source.
    pub fn options<K: Into<String>, V: Into<OptionValue>>(
        mut self,
        options: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        insert_options(&mut self.spec.options, options);
        self
    }

    /// Partition the output by the values of the columns `names`, one directory per value.
    pub fn partition_by(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.spec.partition_cols = strings(names);
        self
    }

    /// Hash the rows into `buckets` buckets by the columns `names`, for a table.
    pub fn bucket_by(
        mut self,
        buckets: i32,
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.spec.num_buckets = Some(buckets);
        self.spec.bucket_cols = strings(names);
        self
    }

    /// Sort the rows of each bucket by the columns `names`.
    pub fn sort_by(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.spec.sort_cols = strings(names);
        self
    }

    /// Cluster the output by the columns `names`.
    pub fn cluster_by(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.spec.cluster_cols = strings(names);
        self
    }

    fn write(
        self,
        save_type: Option<SaveType>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let operation = self.spec.to_proto(self.input.to_relation(), save_type);
        self.input.holding(stream::command(
            self.input.session().clone(),
            proto::command::CommandType::WriteOperation(operation),
        ))
    }

    /// Writes the DataFrame to `path`.
    pub fn save(
        self,
        path: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        self.write(Some(SaveType::Path(path.into())))
    }

    /// Writes the DataFrame where the source's options say, for a source that takes no path.
    pub fn save_without_path(self) -> impl Future<Output = Result<()>> + Send + 'static {
        self.write(None)
    }

    /// Writes the DataFrame as the table `name`, which the mode says what to do with if it
    /// exists.
    pub fn save_as_table(
        self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        self.write(Some(table_destination(
            &name.into(),
            TableSaveMethod::SaveAsTable,
        )))
    }

    /// Inserts the rows of the DataFrame into the existing table `name`, by the position of
    /// their columns; [`SaveMode::Overwrite`] replaces the table's rows.
    pub fn insert_into(
        self,
        name: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        self.write(Some(table_destination(
            &name.into(),
            TableSaveMethod::InsertInto,
        )))
    }

    /// Writes the DataFrame to `path` as CSV.
    pub fn csv(self, path: impl Into<String>) -> impl Future<Output = Result<()>> + Send + 'static {
        self.format("csv").save(path)
    }

    /// Writes the DataFrame to `path` as JSON Lines.
    pub fn json(
        self,
        path: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        self.format("json").save(path)
    }

    /// Writes the DataFrame to `path` as ORC.
    pub fn orc(self, path: impl Into<String>) -> impl Future<Output = Result<()>> + Send + 'static {
        self.format("orc").save(path)
    }

    /// Writes the DataFrame to `path` as Parquet.
    pub fn parquet(
        self,
        path: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        self.format("parquet").save(path)
    }

    /// Writes the DataFrame, of one string column, to `path` as lines of text.
    pub fn text(
        self,
        path: impl Into<String>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        self.format("text").save(path)
    }

    /// Writes the DataFrame to `path` as XML.
    pub fn xml(self, path: impl Into<String>) -> impl Future<Output = Result<()>> + Send + 'static {
        self.format("xml").save(path)
    }
}

/// A write of a DataFrame to a table of a catalog, from [`DataFrame::write_to`]: its provider,
/// options, table properties and layout, then the method that says how to write the table and
/// returns the future of the write.
#[derive(Clone)]
#[must_use = "a write needs a method that says how to write the table"]
pub struct DataFrameWriterV2 {
    input: DataFrame,
    spec: WriteV2Spec,
}

impl DataFrameWriterV2 {
    /// Create the table with the provider `provider`, such as `parquet`.
    pub fn using(mut self, provider: impl Into<String>) -> Self {
        self.spec.provider = Some(provider.into());
        self
    }

    /// Set the write option `key`.
    pub fn option(mut self, key: impl Into<String>, value: impl Into<OptionValue>) -> Self {
        self.spec.options.insert(key.into(), value.into().0);
        self
    }

    /// Set the write options.
    pub fn options<K: Into<String>, V: Into<OptionValue>>(
        mut self,
        options: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        insert_options(&mut self.spec.options, options);
        self
    }

    /// Create the table with the property `property` set to `value`.
    pub fn table_property(mut self, property: impl Into<String>, value: impl Into<String>) -> Self {
        self.spec
            .table_properties
            .insert(property.into(), value.into());
        self
    }

    /// Create the table partitioned by `columns`, which may be columns or transforms of them,
    /// such as [`years`](crate::functions::years).
    pub fn partitioned_by(mut self, columns: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        self.spec.partition_cols = columns.into_iter().map(to_core).collect();
        self
    }

    /// Create the table clustered by the columns `names`.
    pub fn cluster_by(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.spec.cluster_cols = strings(names);
        self
    }

    fn write(
        self,
        mode: Mode,
        condition: Option<proto::Expression>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let operation = self
            .spec
            .to_proto(self.input.to_relation(), mode, condition);
        self.input.holding(stream::command(
            self.input.session().clone(),
            proto::command::CommandType::WriteOperationV2(operation),
        ))
    }

    /// Creates the table from the DataFrame; fails if it exists.
    pub fn create(self) -> impl Future<Output = Result<()>> + Send + 'static {
        self.write(Mode::Create, None)
    }

    /// Replaces the existing table with one of the DataFrame.
    pub fn replace(self) -> impl Future<Output = Result<()>> + Send + 'static {
        self.write(Mode::Replace, None)
    }

    /// Creates the table from the DataFrame, or replaces it if it exists.
    pub fn create_or_replace(self) -> impl Future<Output = Result<()>> + Send + 'static {
        self.write(Mode::CreateOrReplace, None)
    }

    /// Appends the rows of the DataFrame to the existing table.
    pub fn append(self) -> impl Future<Output = Result<()>> + Send + 'static {
        self.write(Mode::Append, None)
    }

    /// Replaces the rows of the existing table for which `condition` holds with the rows of
    /// the DataFrame.
    pub fn overwrite(
        self,
        condition: impl Into<Column>,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let condition = to_core(condition).to_proto();
        self.write(Mode::Overwrite, Some(condition))
    }

    /// Replaces the partitions of the existing table that the rows of the DataFrame fall in
    /// with those rows.
    pub fn overwrite_partitions(self) -> impl Future<Output = Result<()>> + Send + 'static {
        self.write(Mode::OverwritePartitions, None)
    }
}

impl DataFrame {
    /// A write of this DataFrame to a data source or a table.
    pub fn write(&self) -> DataFrameWriter {
        DataFrameWriter {
            input: self.clone(),
            spec: WriteSpec::default(),
        }
    }

    /// A write of this DataFrame to the table `table` of a catalog.
    pub fn write_to(&self, table: impl Into<String>) -> DataFrameWriterV2 {
        DataFrameWriterV2 {
            input: self.clone(),
            spec: WriteV2Spec::new(&table.into()),
        }
    }
}

/// The DataFrame of local Arrow data, from [`Session::create_dataframe_from_arrow`]. Awaiting
/// it sends the data in the plan, or, when it is larger than the session's
/// `spark.sql.session.localRelationCacheThreshold`, uploads it to the session first.
#[must_use = "a DataFrame of local data is created when it is awaited"]
pub struct FromArrow {
    session: Session,
    batches: Vec<RecordBatch>,
    schema: Option<DataType>,
}

impl FromArrow {
    /// The schema of the DataFrame, which the server applies to the data, rather than the one
    /// of the Arrow data.
    pub fn schema(mut self, schema: DataType) -> Self {
        self.schema = Some(schema);
        self
    }
}

impl IntoFuture for FromArrow {
    type Output = Result<DataFrame>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<DataFrame>> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        async move {
            let plan =
                create_relation(self.session.client(), self.batches, self.schema, None).await?;
            Ok(DataFrame::new(self.session, plan))
        }
        .boxed()
    }
}

impl Session {
    /// A read of a DataFrame from a data source.
    pub fn read(&self) -> DataFrameReader {
        DataFrameReader {
            session: self.clone(),
            format: None,
            schema: None,
            options: HashMap::new(),
            streaming: false,
            source_name: None,
        }
    }

    /// The DataFrame of the local `rows` of `schema`, a struct type: each row has the values of
    /// its fields, in order.
    ///
    /// The rows are sent in the plan, or, when they are large, as
    /// [`create_dataframe_from_arrow`](Session::create_dataframe_from_arrow) sends them. A row
    /// of another width than the schema's, or a value that its field's type cannot hold, fails
    /// with [`ErrorKind::InvalidArgument`](crate::ErrorKind::InvalidArgument).
    ///
    /// ```no_run
    /// # async fn run(session: spark_connect_async::Session) -> spark_connect_async::Result<()> {
    /// use spark_connect_async::types::DataType;
    /// use spark_connect_async::Value;
    ///
    /// let schema = DataType::from_ddl("id BIGINT, name STRING")?;
    /// let rows = [
    ///     vec![Value::Long(1), Value::String("a".to_string())],
    ///     vec![Value::Long(2), Value::Null],
    /// ];
    /// let people = session.create_dataframe(rows, schema).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_dataframe<R: IntoIterator<Item = Value>>(
        &self,
        rows: impl IntoIterator<Item = R>,
        schema: DataType,
    ) -> impl Future<Output = Result<DataFrame>> + Send + 'static {
        let session = self.clone();
        let rows = core_rows(rows, &schema);
        async move {
            let rows = rows?;
            let plan = if rows.is_empty() {
                plan::local_relation(schema, None)
            } else {
                let batch = rows_to_arrow_batch(&rows, &schema)
                    .map_err(|error| Error::invalid_argument(error.to_string()))?;
                match small_local_relation(&batch, &schema) {
                    Some(relation) => relation?,
                    None => {
                        create_relation(session.client(), vec![batch], Some(schema), None).await?
                    }
                }
            };
            Ok(DataFrame::new(session, plan))
        }
    }

    /// The DataFrame of the local Arrow `batches`, which must have one schema. Its schema is
    /// that of the batches, unless [`FromArrow::schema`] gives one.
    pub fn create_dataframe_from_arrow(
        &self,
        batches: impl IntoIterator<Item = RecordBatch>,
    ) -> FromArrow {
        FromArrow {
            session: self.clone(),
            batches: batches.into_iter().collect(),
            schema: None,
        }
    }
}

/// `rows` as the rows of the shared encoder, which takes the values of a row by position.
fn core_rows<R: IntoIterator<Item = Value>>(
    rows: impl IntoIterator<Item = R>,
    schema: &DataType,
) -> Result<Vec<CoreRow>> {
    let names: Arc<[String]> = match schema {
        DataType::Struct { fields } => fields.iter().map(|field| field.name.clone()).collect(),
        _ => Arc::from([]),
    };
    rows.into_iter()
        .map(|row| {
            let values = row
                .into_iter()
                .map(Value::into_core)
                .collect::<Result<Vec<_>>>()?;
            // The encoder rejects a row of another width, whose names are then placeholders.
            let names = if values.len() == names.len() {
                Arc::clone(&names)
            } else {
                vec![String::new(); values.len()].into()
            };
            Ok(CoreRow::with_shared_fields(names, values))
        })
        .collect()
}
