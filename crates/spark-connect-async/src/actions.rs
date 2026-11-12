//! The actions of a DataFrame: they run its plan on the server, or ask the server about it.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::Int64Type;
use arrow::record_batch::RecordBatch;
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt};
use serde::de::DeserializeOwned;
use spark_connect_core::actions::dataframe as core_actions;
use spark_connect_core::plan::{self, AggregateGroupType, LogicalPlan, RelationOwner};
use spark_connect_core::types::DataType;
use spark_connect_proto as proto;

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::stream::{self, RecordBatchStream};
use crate::value::Row;

/// The detail of [`DataFrame::explain`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExplainMode {
    /// The physical plan.
    #[default]
    Simple,
    /// The logical plans, parsed, analyzed and optimized, and the physical plan.
    Extended,
    /// The physical plan and the code generated for it.
    Codegen,
    /// The optimized logical plan with its statistics, and the physical plan.
    Cost,
    /// The physical plan as an outline, and the details of its nodes.
    Formatted,
}

impl ExplainMode {
    fn proto(self) -> proto::analyze_plan_request::explain::ExplainMode {
        use proto::analyze_plan_request::explain::ExplainMode as Mode;
        match self {
            ExplainMode::Simple => Mode::Simple,
            ExplainMode::Extended => Mode::Extended,
            ExplainMode::Codegen => Mode::Codegen,
            ExplainMode::Cost => Mode::Cost,
            ExplainMode::Formatted => Mode::Formatted,
        }
    }
}

/// Where the server keeps the rows of a persisted DataFrame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageLevel {
    /// On disk.
    pub use_disk: bool,
    /// In memory.
    pub use_memory: bool,
    /// In off-heap memory.
    pub use_off_heap: bool,
    /// As objects, rather than serialized.
    pub deserialized: bool,
    /// On how many executors.
    pub replication: i32,
}

impl StorageLevel {
    const fn of(
        disk: bool,
        memory: bool,
        off_heap: bool,
        deserialized: bool,
        replication: i32,
    ) -> Self {
        StorageLevel {
            use_disk: disk,
            use_memory: memory,
            use_off_heap: off_heap,
            deserialized,
            replication,
        }
    }

    /// Not persisted.
    pub const NONE: StorageLevel = StorageLevel::of(false, false, false, false, 1);
    /// Serialized on disk.
    pub const DISK_ONLY: StorageLevel = StorageLevel::of(true, false, false, false, 1);
    /// Serialized on disk, on two executors.
    pub const DISK_ONLY_2: StorageLevel = StorageLevel::of(true, false, false, false, 2);
    /// Serialized on disk, on three executors.
    pub const DISK_ONLY_3: StorageLevel = StorageLevel::of(true, false, false, false, 3);
    /// Serialized in memory.
    pub const MEMORY_ONLY: StorageLevel = StorageLevel::of(false, true, false, false, 1);
    /// Serialized in memory, on two executors.
    pub const MEMORY_ONLY_2: StorageLevel = StorageLevel::of(false, true, false, false, 2);
    /// Serialized in memory, and on disk what does not fit.
    pub const MEMORY_AND_DISK: StorageLevel = StorageLevel::of(true, true, false, false, 1);
    /// Serialized in memory and on disk, on two executors.
    pub const MEMORY_AND_DISK_2: StorageLevel = StorageLevel::of(true, true, false, false, 2);
    /// As objects in memory, and serialized on disk what does not fit.
    pub const MEMORY_AND_DISK_DESER: StorageLevel = StorageLevel::of(true, true, false, true, 1);
    /// Serialized in off-heap memory.
    pub const OFF_HEAP: StorageLevel = StorageLevel::of(true, true, true, false, 1);

    pub(crate) fn proto(self) -> proto::StorageLevel {
        proto::StorageLevel {
            use_disk: self.use_disk,
            use_memory: self.use_memory,
            use_off_heap: self.use_off_heap,
            deserialized: self.deserialized,
            replication: self.replication,
        }
    }

    fn of_proto(level: proto::StorageLevel) -> Self {
        StorageLevel::of(
            level.use_disk,
            level.use_memory,
            level.use_off_heap,
            level.deserialized,
            level.replication,
        )
    }
}

/// The temporary view of [`DataFrame::create_temp_view`]. By default, a view of the session,
/// which must not exist yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TempView {
    replace: bool,
    global: bool,
}

impl TempView {
    /// Replace a view of the same name.
    pub fn or_replace(mut self) -> Self {
        self.replace = true;
        self
    }

    /// A view of every session of the server, in the schema `global_temp`.
    pub fn global(mut self) -> Self {
        self.global = true;
        self
    }
}

/// The checkpoint of [`DataFrame::checkpoint`]. By default, the rows are computed now and
/// kept in the checkpoint directory of the server.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Checkpoint {
    local: bool,
    lazy: bool,
    storage_level: Option<StorageLevel>,
}

impl Checkpoint {
    /// Keep the rows on the executors instead, which is faster and not fault tolerant.
    pub fn local(mut self) -> Self {
        self.local = true;
        self
    }

    /// Compute the rows when the DataFrame first runs, rather than now.
    pub fn lazy(mut self) -> Self {
        self.lazy = true;
        self
    }

    /// Where a local checkpoint keeps its rows.
    pub fn storage_level(mut self, level: StorageLevel) -> Self {
        self.storage_level = Some(level);
        self
    }
}

/// The table of [`DataFrame::show`]: await it to print it, or call [`Show::to_string`].
pub struct Show {
    dataframe: DataFrame,
    rows: i32,
    truncate: i32,
    vertical: bool,
}

impl Show {
    /// Show the first `rows` rows; by default 20.
    pub fn rows(mut self, rows: usize) -> Self {
        self.rows = i32::try_from(rows).unwrap_or(i32::MAX);
        self
    }

    /// Cut values to `characters` characters; by default 20, and 0 does not cut them.
    pub fn truncate(mut self, characters: usize) -> Self {
        self.truncate = i32::try_from(characters).unwrap_or(i32::MAX);
        self
    }

    /// Show each row as a block of a line per column.
    pub fn vertical(mut self) -> Self {
        self.vertical = true;
        self
    }

    /// The table, rather than printing it.
    pub fn to_string(self) -> impl Future<Output = Result<String>> + Send + 'static {
        let shown = self.dataframe.with_plan(LogicalPlan::ShowString {
            input: Arc::clone(self.dataframe.plan()),
            num_rows: self.rows,
            truncate: self.truncate,
            vertical: self.vertical,
        });
        async move {
            let table = shown.collect_as::<(String,)>().await?;
            table
                .into_iter()
                .next()
                .map(|(table,)| table)
                .ok_or_else(|| Error::decode("the server returned no table"))
        }
    }
}

impl IntoFuture for Show {
    type Output = Result<()>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<()>> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        let table = self.to_string();
        Box::pin(async move {
            println!("{}", table.await?);
            Ok(())
        })
    }
}

impl DataFrame {
    /// Runs the query, and returns the stream of its record batches once the server has
    /// accepted it.
    ///
    /// Dropping the future before then interrupts the operation on the server; dropping the
    /// stream before its end releases it, which stops it too.
    pub fn stream(&self) -> impl Future<Output = Result<RecordBatchStream>> + Send + 'static {
        self.holding(stream::query(self.session().clone(), self.to_relation()))
    }

    /// Runs the query, and returns the stream of its rows as values of `T`, which
    /// [`de`](crate::de) describes.
    pub fn stream_as<T: DeserializeOwned + Send + 'static>(
        &self,
    ) -> impl Future<Output = Result<impl Stream<Item = Result<T>> + Send + 'static>> + Send + 'static
    {
        self.stream().map_ok(RecordBatchStream::rows_as::<T>)
    }

    /// The record batches of the result.
    pub fn collect(&self) -> impl Future<Output = Result<Vec<RecordBatch>>> + Send + 'static {
        let stream = self.stream();
        async move { stream.await?.try_collect().await }
    }

    /// The rows of the result as [`Row`]s.
    pub fn collect_rows(&self) -> impl Future<Output = Result<Vec<Row>>> + Send + 'static {
        self.collect_with(crate::de::rows)
    }

    /// The rows of the result as values of `T`, which [`de`](crate::de) describes.
    ///
    /// ```no_run
    /// # async fn run(session: spark_connect_async::Session) -> spark_connect_async::Result<()> {
    /// #[derive(serde::Deserialize)]
    /// struct Person {
    ///     name: String,
    ///     age: Option<i32>,
    /// }
    ///
    /// let people: Vec<Person> = session.table("people").collect_as().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn collect_as<T: DeserializeOwned + Send + 'static>(
        &self,
    ) -> impl Future<Output = Result<Vec<T>>> + Send + 'static {
        self.collect_with(|batch| crate::de::from_record_batch::<T>(batch))
    }

    /// The rows of the result, each batch read by `read` as it arrives.
    fn collect_with<T: Send + 'static>(
        &self,
        read: impl Fn(&RecordBatch) -> Result<Vec<T>> + Send + 'static,
    ) -> impl Future<Output = Result<Vec<T>>> + Send + 'static {
        let stream = self.stream();
        async move {
            let mut batches = stream.await?;
            let mut rows = Vec::new();
            while let Some(batch) = batches.next().await {
                rows.extend(read(&batch?)?);
            }
            Ok(rows)
        }
    }

    /// The number of rows.
    pub fn count(&self) -> impl Future<Output = Result<i64>> + Send + 'static {
        let count = spark_connect_core::functions::count(spark_connect_core::column::lit(1))
            .expression()
            .clone();
        let counted = self.with_plan(plan::aggregate(
            Arc::clone(self.plan()),
            AggregateGroupType::GroupBy,
            vec![],
            vec![count],
        ));
        let batches = counted.collect();
        async move {
            let batches = batches.await?;
            let counts = batches
                .iter()
                .find(|batch| batch.num_rows() > 0)
                .and_then(|batch| batch.column(0).as_primitive_opt::<Int64Type>());
            Ok(counts.map_or(0, |counts| counts.value(0)))
        }
    }

    /// The table of the first rows, which awaiting [`Show`] prints.
    pub fn show(&self) -> Show {
        Show {
            dataframe: self.clone(),
            rows: 20,
            truncate: 20,
            vertical: false,
        }
    }

    /// The first `n` rows.
    pub fn take(&self, n: usize) -> impl Future<Output = Result<Vec<Row>>> + Send + 'static {
        self.limit(i32::try_from(n).unwrap_or(i32::MAX))
            .collect_rows()
    }

    /// The first row, if there is one.
    pub fn first(&self) -> impl Future<Output = Result<Option<Row>>> + Send + 'static {
        self.take(1).map_ok(|rows| rows.into_iter().next())
    }

    /// Whether the result has no rows.
    pub fn is_empty(&self) -> impl Future<Output = Result<bool>> + Send + 'static {
        self.limit(1).count().map_ok(|count| count == 0)
    }

    /// The schema of the result, a struct type. Only the first successful call asks the
    /// server; its clones share the answer.
    pub fn schema(&self) -> impl Future<Output = Result<DataType>> + Send + 'static {
        let dataframe = self.clone();
        async move {
            let cache = dataframe.schema_cache();
            let schema = cache
                .get_or_try_init(|| {
                    core_actions::schema(dataframe.session().client(), dataframe.to_relation())
                })
                .await?;
            Ok(schema.clone())
        }
    }

    /// The names of the columns of the result.
    pub fn columns(&self) -> impl Future<Output = Result<Vec<String>>> + Send + 'static {
        let schema = self.schema();
        async move {
            let DataType::Struct { fields } = schema.await? else {
                return Ok(Vec::new());
            };
            Ok(fields.into_iter().map(|field| field.name).collect())
        }
    }

    /// The plan of the query as the server would run it.
    pub fn explain(
        &self,
        mode: ExplainMode,
    ) -> impl Future<Output = Result<String>> + Send + 'static {
        self.holding(
            core_actions::explain(self.session().client(), self.to_relation(), mode.proto())
                .err_into(),
        )
    }

    /// Persists the rows in memory and on disk, as [`StorageLevel::MEMORY_AND_DISK_DESER`].
    pub fn cache(&self) -> impl Future<Output = Result<()>> + Send + 'static {
        self.persist(StorageLevel::MEMORY_AND_DISK_DESER)
    }

    /// Keeps the rows at `level` once they are computed, for the later queries of this
    /// DataFrame.
    pub fn persist(
        &self,
        level: StorageLevel,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        self.holding(
            core_actions::persist(self.session().client(), self.to_relation(), level.proto())
                .err_into(),
        )
    }

    /// Stops keeping the rows; with `blocking`, it returns once they are removed.
    pub fn unpersist(&self, blocking: bool) -> impl Future<Output = Result<()>> + Send + 'static {
        self.holding(
            core_actions::unpersist(self.session().client(), self.to_relation(), blocking)
                .err_into(),
        )
    }

    /// Where the server keeps the rows, or [`StorageLevel::NONE`].
    pub fn storage_level(&self) -> impl Future<Output = Result<StorageLevel>> + Send + 'static {
        self.holding(
            core_actions::storage_level(self.session().client(), self.to_relation())
                .map_ok(StorageLevel::of_proto)
                .err_into(),
        )
    }

    /// Creates the temporary view `name` of this DataFrame.
    pub fn create_temp_view(
        &self,
        name: impl Into<String>,
        view: TempView,
    ) -> impl Future<Output = Result<()>> + Send + 'static {
        let command =
            proto::command::CommandType::CreateDataframeView(proto::CreateDataFrameViewCommand {
                input: Some(self.to_relation()),
                name: name.into(),
                is_global: view.global,
                replace: view.replace,
            });
        self.holding(stream::command(self.session().clone(), command))
    }

    /// A DataFrame of the rows of this one as the server keeps them, whose plan does not
    /// depend on this one's. The server removes them when the last DataFrame of the checkpoint
    /// is dropped.
    pub fn checkpoint(
        &self,
        checkpoint: Checkpoint,
    ) -> impl Future<Output = Result<DataFrame>> + Send + 'static {
        let command = proto::command::CommandType::CheckpointCommand(proto::CheckpointCommand {
            relation: Some(self.to_relation()),
            local: checkpoint.local,
            eager: !checkpoint.lazy,
            storage_level: checkpoint.storage_level.map(StorageLevel::proto),
        });
        let session = self.session().clone();
        self.holding(async move {
            let relation_id =
                stream::command_result(session.clone(), command, |response| match response {
                    proto::execute_plan_response::ResponseType::CheckpointCommandResult(result) => {
                        result.relation.map(|relation| relation.relation_id)
                    }
                    _ => None,
                })
                .await?
                .ok_or_else(|| Error::decode("the server returned no checkpoint"))?;
            let remove = proto::command::CommandType::RemoveCachedRemoteRelationCommand(
                proto::RemoveCachedRemoteRelationCommand {
                    relation: Some(proto::CachedRemoteRelation {
                        relation_id: relation_id.clone(),
                    }),
                },
            );
            let owner = stream::RemoveOnDrop::new(session.clone(), remove);
            Ok(DataFrame::new(
                session,
                LogicalPlan::CachedRemoteRelation {
                    relation_id,
                    owner: Some(RelationOwner(Arc::new(owner))),
                },
            ))
        })
    }

    /// The files that the query reads, as far as the server can tell.
    pub fn input_files(&self) -> impl Future<Output = Result<Vec<String>>> + Send + 'static {
        self.holding(
            core_actions::input_files(self.session().client(), self.to_relation()).err_into(),
        )
    }

    /// Whether this DataFrame and `other` return the same rows by their plans.
    pub fn same_semantics(
        &self,
        other: &DataFrame,
    ) -> impl Future<Output = Result<bool>> + Send + 'static {
        let same = core_actions::same_semantics(
            self.session().client(),
            self.to_relation(),
            other.to_relation(),
        );
        self.holding(other.holding(same.err_into()))
    }

    /// A hash of the plan, the same for plans of [`DataFrame::same_semantics`].
    pub fn semantic_hash(&self) -> impl Future<Output = Result<i32>> + Send + 'static {
        self.holding(
            core_actions::semantic_hash(self.session().client(), self.to_relation()).err_into(),
        )
    }

    /// Whether the query reads a stream, so that its rows never end.
    pub fn is_streaming(&self) -> impl Future<Output = Result<bool>> + Send + 'static {
        self.holding(
            core_actions::is_streaming(self.session().client(), self.to_relation()).err_into(),
        )
    }

    /// Each row as the text of a JSON object.
    pub fn to_json(&self) -> impl Future<Output = Result<Vec<String>>> + Send + 'static {
        use spark_connect_core::functions::{col, r#struct, to_json};
        let rows = self
            .select([to_json(r#struct([col("*")])).alias("value")])
            .collect_as::<(String,)>();
        async move { Ok(rows.await?.into_iter().map(|(row,)| row).collect()) }
    }

    /// The result as a polars DataFrame.
    #[cfg(feature = "polars")]
    pub fn to_polars(
        &self,
    ) -> impl Future<Output = Result<polars::frame::DataFrame>> + Send + 'static {
        let batches = self.collect();
        async move {
            Ok(spark_connect_core::interop::record_batches_to_polars(
                &batches.await?,
            )?)
        }
    }

    /// The result as a DataFusion DataFrame of `context`; an empty result is an error.
    #[cfg(feature = "datafusion")]
    pub fn to_datafusion(
        &self,
        context: &datafusion::prelude::SessionContext,
    ) -> impl Future<Output = Result<datafusion::dataframe::DataFrame>> + Send + 'static {
        let batches = self.collect();
        let context = context.clone();
        async move {
            Ok(spark_connect_core::interop::record_batches_to_datafusion(
                &context,
                batches.await?,
            )?)
        }
    }
}
