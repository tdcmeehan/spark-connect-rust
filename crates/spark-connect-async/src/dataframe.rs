//! DataFrames and their transformations.

use std::future::Future;
use std::sync::Arc;

use spark_connect_core::column::Column as CoreColumn;
use spark_connect_core::expression::{
    AsSubquery, ColumnReference, Expression, ExpressionNode, SubqueryExpression, SubqueryType,
};
use spark_connect_core::plan::{relation_with_ids, LogicalPlan, SetOpType};
use spark_connect_core::types::DataType;
use spark_connect_proto as proto;
use tokio::sync::OnceCell;

use crate::column::{expressions, subquery_column, to_core, Column};
use crate::group::{ByColumns, GroupedData, Grouping};
use crate::observation::Observation;
use crate::options::{
    AsOfJoin, JoinType, NearestByJoin, Partitioning, PartitioningKind, RandomSplit, Sample,
    UnionByName,
};
use crate::session::Session;
use crate::stat::{NaFunctions, StatFunctions};
use crate::tvf::TableArg;

/// A query of a session, which runs on the server at its actions.
///
/// A DataFrame is immutable: each transformation returns a new DataFrame, which shares the
/// plan of its input, so building one is cheap and sends nothing. The server checks the plan
/// when it runs it, so a transformation of a column that does not exist fails at the action.
#[derive(Clone)]
pub struct DataFrame {
    session: Session,
    plan: Arc<LogicalPlan>,
    schema: Arc<OnceCell<DataType>>,
}

/// A column given by its name, or as an expression, of [`DataFrame::drop`].
#[derive(Debug, Clone)]
pub enum ColumnOrName {
    /// The column of this name, if there is one.
    Name(String),
    /// The column of this expression, which must resolve.
    Column(Column),
}

impl From<&str> for ColumnOrName {
    fn from(name: &str) -> Self {
        ColumnOrName::Name(name.to_string())
    }
}

impl From<String> for ColumnOrName {
    fn from(name: String) -> Self {
        ColumnOrName::Name(name)
    }
}

impl From<Column> for ColumnOrName {
    fn from(column: Column) -> Self {
        ColumnOrName::Column(column)
    }
}

pub(crate) fn strings(names: impl IntoIterator<Item = impl Into<String>>) -> Vec<String> {
    names.into_iter().map(Into::into).collect()
}

fn columns(columns: impl IntoIterator<Item = impl Into<Column>>) -> Vec<CoreColumn> {
    columns.into_iter().map(to_core).collect()
}

impl DataFrame {
    pub(crate) fn new(session: Session, plan: impl Into<Arc<LogicalPlan>>) -> Self {
        DataFrame {
            session,
            plan: plan.into(),
            schema: Arc::default(),
        }
    }

    pub(crate) fn plan(&self) -> &Arc<LogicalPlan> {
        &self.plan
    }

    fn input(&self) -> Arc<LogicalPlan> {
        Arc::clone(&self.plan)
    }

    pub(crate) fn schema_cache(&self) -> Arc<OnceCell<DataType>> {
        Arc::clone(&self.schema)
    }

    pub(crate) fn with_plan(&self, plan: LogicalPlan) -> DataFrame {
        DataFrame::new(self.session.clone(), plan)
    }

    /// `future`, which keeps the plan of this DataFrame until it is done, and so what the server
    /// keeps for the plan: the checkpoints and the fitted models that it refers to.
    pub(crate) fn holding<T>(
        &self,
        future: impl Future<Output = T> + Send + 'static,
    ) -> impl Future<Output = T> + Send + 'static {
        let plan = self.input();
        async move {
            let _plan = plan;
            future.await
        }
    }

    /// The session of this DataFrame.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// The relation of this DataFrame as it is sent, with the plan ids of its session.
    #[doc(hidden)]
    pub fn to_relation(&self) -> proto::Relation {
        relation_with_ids(&self.plan, self.session.plan_ids())
    }

    fn plan_id(&self) -> i64 {
        self.session.plan_ids().id_of(&self.plan)
    }

    /// The column `name` of this DataFrame. Unlike [`col`](crate::column::col), it refers to
    /// this DataFrame, so it tells apart the columns of the same name of two joined
    /// DataFrames.
    pub fn col(&self, name: &str) -> Column {
        let reference = ColumnReference::new(name).with_plan_id(self.plan_id());
        Column::new(ExpressionNode::ColumnReference(reference).into())
    }

    /// Every column of this DataFrame, as `*` refers to them.
    pub fn star(&self) -> Column {
        Column::new(ExpressionNode::UnresolvedStar(None, Some(self.plan_id())).into())
    }

    /// The columns of this DataFrame whose names match the regular expression `regex`, which
    /// is in backticks, such as `` "`(id|name)`" ``.
    pub fn col_regex(&self, regex: &str) -> Column {
        Column::new(ExpressionNode::UnresolvedRegex(regex.to_string(), Some(self.plan_id())).into())
    }

    /// The metadata column `name` of this DataFrame, such as `_metadata` of a file source.
    pub fn metadata_column(&self, name: &str) -> Column {
        let reference = ColumnReference::new(name)
            .with_plan_id(self.plan_id())
            .metadata();
        Column::new(ExpressionNode::ColumnReference(reference).into())
    }

    fn subquery(&self, subquery_type: SubqueryType) -> SubqueryExpression {
        SubqueryExpression::new(self.input(), self.plan_id(), subquery_type)
    }

    /// The value of this DataFrame, which has one row of one column, as a column of another
    /// query. A column of the other query is `col(name).outer()` in this one.
    pub fn scalar(&self) -> Column {
        subquery_column(self.subquery(SubqueryType::Scalar))
    }

    /// Whether this DataFrame has a row, as a column of another query.
    pub fn exists(&self) -> Column {
        subquery_column(self.subquery(SubqueryType::Exists))
    }

    /// The rows of this DataFrame as an argument of a table-valued function.
    pub fn as_table(&self) -> TableArg {
        TableArg::new(self.subquery(SubqueryType::TableArg))
    }

    /// The columns `columns`, such as names, expressions and aliases of expressions.
    pub fn select(&self, columns: impl IntoIterator<Item = impl Into<Column>>) -> DataFrame {
        self.with_plan(LogicalPlan::Project {
            input: self.input(),
            columns: self::columns(columns),
        })
    }

    /// The columns of the SQL expressions `expressions`, such as `"id + 1 AS next"`.
    pub fn select_expr(&self, expressions: impl IntoIterator<Item = impl AsRef<str>>) -> DataFrame {
        let columns = expressions
            .into_iter()
            .map(|expression| spark_connect_core::functions::expr(expression.as_ref()));
        self.select(columns)
    }

    /// The rows for which `condition` is true.
    pub fn filter(&self, condition: impl Into<Column>) -> DataFrame {
        self.with_plan(LogicalPlan::Filter {
            input: self.input(),
            condition: to_core(condition),
        })
    }

    /// Every column, with the column `name` added, or replaced, by `column`.
    pub fn with_column(&self, name: impl Into<String>, column: impl Into<Column>) -> DataFrame {
        self.with_columns([(name, column)])
    }

    /// Every column, with each column of `columns` added, or replaced, in their order.
    pub fn with_columns(
        &self,
        columns: impl IntoIterator<Item = (impl Into<String>, impl Into<Column>)>,
    ) -> DataFrame {
        let (column_names, columns) = columns
            .into_iter()
            .map(|(name, column)| (name.into(), to_core(column)))
            .unzip();
        self.with_plan(LogicalPlan::WithColumns {
            input: self.input(),
            column_names,
            columns,
        })
    }

    /// Every column, with the column `existing` renamed to `new`. Nothing changes if there is
    /// no column `existing`.
    pub fn with_column_renamed(
        &self,
        existing: impl Into<String>,
        new: impl Into<String>,
    ) -> DataFrame {
        self.with_columns_renamed([(existing, new)])
    }

    /// Every column, with each `(existing, new)` of `renames` renamed in their order: so
    /// `[("id", "a"), ("a", "b")]` renames `id` to `b`.
    pub fn with_columns_renamed(
        &self,
        renames: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> DataFrame {
        let renames = renames
            .into_iter()
            .map(|(existing, new)| (existing.into(), new.into()))
            .collect();
        self.with_plan(LogicalPlan::WithColumnsRenamed {
            input: self.input(),
            renames,
        })
    }

    /// Every column, with the metadata of the column `name` set to the JSON object
    /// `metadata_json`.
    pub fn with_metadata(&self, name: &str, metadata_json: impl Into<String>) -> DataFrame {
        self.with_plan(LogicalPlan::WithColumnMetadata {
            input: self.input(),
            column_name: name.to_string(),
            column: self.col(name).expression().clone(),
            metadata_json: metadata_json.into(),
        })
    }

    /// Every column but `columns`. A column given by name is dropped if there is one; a column
    /// given as an expression must resolve.
    pub fn drop(&self, columns: impl IntoIterator<Item = impl Into<ColumnOrName>>) -> DataFrame {
        let (mut names, mut expressions) = (Vec::new(), Vec::new());
        for column in columns {
            match column.into() {
                ColumnOrName::Name(name) => names.push(name),
                ColumnOrName::Column(column) => expressions.push(column.expression().clone()),
            }
        }
        self.with_plan(LogicalPlan::Drop {
            input: self.input(),
            columns: names,
            expressions,
        })
    }

    /// The columns, renamed to `names` in their order.
    pub fn to_df(&self, names: impl IntoIterator<Item = impl Into<String>>) -> DataFrame {
        self.with_plan(LogicalPlan::ToDF {
            input: self.input(),
            column_names: strings(names),
        })
    }

    /// The columns of the struct type `schema`, by name, each cast to the type of its field.
    pub fn to_schema(&self, schema: DataType) -> DataFrame {
        self.with_plan(LogicalPlan::ToSchema {
            input: self.input(),
            schema,
        })
    }

    /// This DataFrame under the name `alias`, by which its columns can be qualified, as
    /// `alias.id`.
    pub fn alias(&self, alias: impl Into<String>) -> DataFrame {
        self.with_plan(LogicalPlan::SubqueryAlias {
            input: self.input(),
            alias: alias.into(),
        })
    }

    /// The first `n` rows.
    pub fn limit(&self, n: i32) -> DataFrame {
        self.with_plan(LogicalPlan::Limit {
            input: self.input(),
            limit: n,
        })
    }

    /// The rows after the first `n`.
    pub fn offset(&self, n: i32) -> DataFrame {
        self.with_plan(LogicalPlan::Offset {
            input: self.input(),
            offset: n,
        })
    }

    /// The last `n` rows.
    pub fn tail(&self, n: i32) -> DataFrame {
        self.with_plan(LogicalPlan::Tail {
            input: self.input(),
            limit: n,
        })
    }

    /// The distinct rows.
    pub fn distinct(&self) -> DataFrame {
        self.deduplicate(Vec::new(), false)
    }

    /// One row of each distinct value of the columns `names`, or of every column when `names`
    /// is empty.
    pub fn drop_duplicates(&self, names: impl IntoIterator<Item = impl Into<String>>) -> DataFrame {
        self.deduplicate(strings(names), false)
    }

    /// As [`DataFrame::drop_duplicates`] of a stream, keeping only the state of the rows
    /// within the watermark, so a duplicate that comes later than the watermark is kept.
    pub fn drop_duplicates_within_watermark(
        &self,
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> DataFrame {
        self.deduplicate(strings(names), true)
    }

    fn deduplicate(&self, column_names: Vec<String>, within_watermark: bool) -> DataFrame {
        self.with_plan(LogicalPlan::Deduplicate {
            input: self.input(),
            all_columns_as_keys: column_names.is_empty(),
            column_names,
            within_watermark,
        })
    }

    /// The rows in the order of `columns`, which sort ascending unless they are sort orders,
    /// such as `col.desc()`.
    pub fn sort(&self, columns: impl IntoIterator<Item = impl Into<Column>>) -> DataFrame {
        self.sorted(columns, true)
    }

    /// The rows of each partition in the order of `columns`, as [`DataFrame::sort`] orders
    /// all of them.
    pub fn sort_within_partitions(
        &self,
        columns: impl IntoIterator<Item = impl Into<Column>>,
    ) -> DataFrame {
        self.sorted(columns, false)
    }

    fn sorted(
        &self,
        columns: impl IntoIterator<Item = impl Into<Column>>,
        is_global: bool,
    ) -> DataFrame {
        self.with_plan(LogicalPlan::Sort {
            input: self.input(),
            order: expressions(columns),
            is_global,
        })
    }

    /// The join of this DataFrame and `right` on `condition`.
    pub fn join(
        &self,
        right: &DataFrame,
        condition: impl Into<Column>,
        join_type: JoinType,
    ) -> DataFrame {
        self.joined(right, Some(condition.into()), Vec::new(), join_type)
    }

    /// The join of this DataFrame and `right` on equal values of the columns `names`, which
    /// both have; the result has one column of each name. With no names, every pair of rows
    /// matches.
    pub fn join_using(
        &self,
        right: &DataFrame,
        names: impl IntoIterator<Item = impl Into<String>>,
        join_type: JoinType,
    ) -> DataFrame {
        self.joined(right, None, strings(names), join_type)
    }

    /// Every pair of a row of this DataFrame and a row of `right`.
    pub fn cross_join(&self, right: &DataFrame) -> DataFrame {
        self.joined(right, None, Vec::new(), JoinType::Cross)
    }

    fn joined(
        &self,
        right: &DataFrame,
        on: Option<Column>,
        using_columns: Vec<String>,
        join_type: JoinType,
    ) -> DataFrame {
        self.with_plan(LogicalPlan::Join {
            left: self.input(),
            right: right.input(),
            join_type: join_type.plan(),
            on: on.map(to_core),
            using_columns,
        })
    }

    /// The lateral join of this DataFrame and `right`, a query that may refer to the columns
    /// of this one as `col(name).outer()`, on `condition`, if any.
    pub fn lateral_join(
        &self,
        right: &DataFrame,
        condition: Option<Column>,
        join_type: JoinType,
    ) -> DataFrame {
        self.with_plan(LogicalPlan::LateralJoin {
            left: self.input(),
            right: right.input(),
            join_type: join_type.plan(),
            on: condition.map(to_core),
        })
    }

    /// The as-of join of this DataFrame and `right`: each row is joined with the row of `right`
    /// whose as-of value is the nearest.
    pub fn as_of_join(&self, right: &DataFrame, join: AsOfJoin) -> DataFrame {
        self.with_plan(LogicalPlan::AsOfJoin {
            left: self.input(),
            right: right.input(),
            left_as_of: to_core(join.left_as_of),
            right_as_of: to_core(join.right_as_of),
            on: join.on.map(to_core),
            using_columns: join.using_columns,
            join_type: join_type_name(join.left_outer),
            tolerance: join.tolerance.map(to_core),
            allow_exact_matches: join.allow_exact_matches,
            direction: join.direction.name().to_string(),
        })
    }

    /// The nearest-by join of this DataFrame and `right`: each row is joined with the rows of
    /// `right` that rank first.
    pub fn nearest_by_join(&self, right: &DataFrame, join: NearestByJoin) -> DataFrame {
        self.with_plan(LogicalPlan::NearestByJoin {
            left: self.input(),
            right: right.input(),
            ranking_expression: join.ranking.expression().clone(),
            num_results: join.num_results,
            join_type: join_type_name(join.left_outer),
            mode: if join.exact { "exact" } else { "approx" }.to_string(),
            direction: if join.similarity {
                "similarity"
            } else {
                "distance"
            }
            .to_string(),
        })
    }

    /// The columns of this DataFrame and of `other` side by side, row by row: both must have
    /// the same number of rows in each partition.
    pub fn zip(&self, other: &DataFrame) -> DataFrame {
        self.with_plan(LogicalPlan::Zip {
            left: self.input(),
            right: other.input(),
        })
    }

    fn set_operation(
        &self,
        other: &DataFrame,
        set_op_type: SetOpType,
        is_all: bool,
        by_name: Option<UnionByName>,
    ) -> DataFrame {
        self.with_plan(LogicalPlan::SetOperation {
            left: self.input(),
            right: other.input(),
            set_op_type,
            is_all,
            by_name: by_name.is_some(),
            allow_missing_columns: by_name.is_some_and(|by_name| by_name.allow_missing_columns),
        })
    }

    /// The rows of this DataFrame and of `other`, duplicates included, matching the columns
    /// by position.
    pub fn union(&self, other: &DataFrame) -> DataFrame {
        self.set_operation(other, SetOpType::Union, true, None)
    }

    /// The rows of this DataFrame and of `other`, duplicates included, matching the columns
    /// by name.
    pub fn union_by_name(&self, other: &DataFrame, options: UnionByName) -> DataFrame {
        self.set_operation(other, SetOpType::Union, true, Some(options))
    }

    /// The distinct rows that are in this DataFrame and in `other`.
    pub fn intersect(&self, other: &DataFrame) -> DataFrame {
        self.set_operation(other, SetOpType::Intersect, false, None)
    }

    /// The rows that are in this DataFrame and in `other`, as many times as they are in both.
    pub fn intersect_all(&self, other: &DataFrame) -> DataFrame {
        self.set_operation(other, SetOpType::Intersect, true, None)
    }

    /// The distinct rows of this DataFrame that are not in `other`.
    pub fn subtract(&self, other: &DataFrame) -> DataFrame {
        self.set_operation(other, SetOpType::Except, false, None)
    }

    /// The rows of this DataFrame that are not in `other`, as many times as they are more in
    /// this one.
    pub fn except_all(&self, other: &DataFrame) -> DataFrame {
        self.set_operation(other, SetOpType::Except, true, None)
    }

    /// A sample of the rows: `df.sample(0.1)`, or a [`Sample`].
    pub fn sample(&self, sample: impl Into<Sample>) -> DataFrame {
        let sample = sample.into();
        self.sampled(
            0.0..sample.fraction,
            sample.with_replacement,
            sample.seed,
            false,
        )
    }

    fn sampled(
        &self,
        bounds: std::ops::Range<f64>,
        with_replacement: bool,
        seed: i64,
        deterministic_order: bool,
    ) -> DataFrame {
        self.with_plan(LogicalPlan::Sample {
            input: self.input(),
            lower_bound: bounds.start,
            upper_bound: bounds.end,
            with_replacement,
            seed: Some(seed),
            deterministic_order,
        })
    }

    /// The parts of a random split of the rows, one of each weight. Every row is in one part.
    pub fn random_split(&self, split: RandomSplit) -> Vec<DataFrame> {
        let mut lower = 0.0;
        split
            .shares
            .iter()
            .map(|share| {
                let upper = lower + share;
                let part = self.sampled(lower..upper, false, split.seed, true);
                lower = upper;
                part
            })
            .collect()
    }

    /// The rows in new partitions: `df.repartition(8)`, or a [`Partitioning`].
    pub fn repartition(&self, partitioning: impl Into<Partitioning>) -> DataFrame {
        let Partitioning { kind, partitions } = partitioning.into();
        let input = self.input();
        self.with_plan(match kind {
            PartitioningKind::RoundRobin => LogicalPlan::Repartition {
                input,
                num_partitions: partitions.unwrap_or_default(),
                shuffle: true,
            },
            PartitioningKind::Hash(expressions) => LogicalPlan::RepartitionByExpression {
                input,
                num_partitions: partitions.unwrap_or_default(),
                expressions,
            },
            PartitioningKind::Range(partition_exprs) => LogicalPlan::RepartitionByRange {
                input,
                num_partitions: partitions,
                partition_exprs,
            },
        })
    }

    /// The rows in `partitions` partitions, merging partitions without a shuffle; it cannot
    /// add partitions.
    pub fn coalesce(&self, partitions: i32) -> DataFrame {
        self.with_plan(LogicalPlan::Repartition {
            input: self.input(),
            num_partitions: partitions,
            shuffle: false,
        })
    }

    /// This DataFrame with the hint `name` of `parameters` for the server's planner, such as
    /// `hint("rebalance", [8])` or `hint("broadcast", Vec::<i32>::new())`.
    pub fn hint(
        &self,
        name: impl Into<String>,
        parameters: impl IntoIterator<Item = impl crate::column::IntoArgument>,
    ) -> DataFrame {
        let parameters = parameters
            .into_iter()
            .map(|parameter| parameter.into_argument().expression().clone())
            .collect();
        self.with_plan(LogicalPlan::Hint {
            input: self.input(),
            name: name.into(),
            parameters,
        })
    }

    /// This DataFrame with the event-time watermark of the column `event_time`: a stream
    /// drops the state of rows older than the latest event time less `delay`, such as
    /// `"10 minutes"`.
    pub fn with_watermark(
        &self,
        event_time: impl Into<String>,
        delay: impl Into<String>,
    ) -> DataFrame {
        self.with_plan(LogicalPlan::WithWatermark {
            input: self.input(),
            time_column: event_time.into(),
            delay_threshold: delay.into(),
        })
    }

    /// The rows turned into a row of each value of the columns `values`, with the columns
    /// `ids`, a column `variable_column` of the name of the value's column, and a column
    /// `value_column` of the value. With no `values`, every column but `ids` is one.
    pub fn unpivot(
        &self,
        ids: impl IntoIterator<Item = impl Into<Column>>,
        values: impl IntoIterator<Item = impl Into<Column>>,
        variable_column: impl Into<String>,
        value_column: impl Into<String>,
    ) -> DataFrame {
        let values = self::columns(values);
        self.with_plan(LogicalPlan::Unpivot {
            input: self.input(),
            ids: self::columns(ids),
            values: (!values.is_empty()).then_some(values),
            variable_column_name: variable_column.into(),
            value_column_name: value_column.into(),
        })
    }

    /// The rows turned into columns: the values of `index`, or else of the first column, name
    /// the new columns, and each other column becomes a row.
    pub fn transpose(&self, index: Option<Column>) -> DataFrame {
        self.with_plan(LogicalPlan::Transpose {
            input: self.input(),
            index_columns: index
                .into_iter()
                .map(|index| index.expression().clone())
                .collect(),
        })
    }

    /// The rows in groups of the values of `columns`.
    pub fn group_by(
        &self,
        columns: impl IntoIterator<Item = impl Into<Column>>,
    ) -> GroupedData<ByColumns> {
        GroupedData::new(self.clone(), expressions(columns), Grouping::By)
    }

    /// The rows in the groups of a rollup of `columns`: of the values of every column, of
    /// every column but the last, and so on, down to one group of every row. A column that a
    /// group is not of is `NULL` in its row.
    pub fn rollup(&self, columns: impl IntoIterator<Item = impl Into<Column>>) -> GroupedData {
        GroupedData::new(self.clone(), expressions(columns), Grouping::Rollup)
    }

    /// The rows in the groups of a cube of `columns`: of the values of each combination of the
    /// columns. A column that a group is not of is `NULL` in its row.
    pub fn cube(&self, columns: impl IntoIterator<Item = impl Into<Column>>) -> GroupedData {
        GroupedData::new(self.clone(), expressions(columns), Grouping::Cube)
    }

    /// The rows in the groups of each set of `sets`, with the columns `columns` in each row of
    /// the result besides the aggregations. A column that a set does not hold is `NULL` in the
    /// rows of that set.
    pub fn grouping_sets(
        &self,
        sets: impl IntoIterator<Item = impl IntoIterator<Item = impl Into<Column>>>,
        columns: impl IntoIterator<Item = impl Into<Column>>,
    ) -> GroupedData {
        let sets = sets.into_iter().map(expressions).collect();
        GroupedData::new(self.clone(), expressions(columns), Grouping::Sets(sets))
    }

    /// One row of the aggregations `aggregates` of every row.
    pub fn agg(&self, aggregates: impl IntoIterator<Item = impl Into<Column>>) -> DataFrame {
        GroupedData::<ByColumns>::new(self.clone(), vec![], Grouping::By).agg(aggregates)
    }

    /// The statistics of the columns.
    pub fn stat(&self) -> StatFunctions {
        StatFunctions::new(self.clone())
    }

    /// The handling of the missing values of the columns.
    pub fn na(&self) -> NaFunctions {
        NaFunctions::new(self.clone())
    }

    /// The count, mean, standard deviation, minimum and maximum of the columns `names`, or of
    /// every numeric and string column when `names` is empty.
    pub fn describe(&self, names: impl IntoIterator<Item = impl Into<String>>) -> DataFrame {
        self.with_plan(LogicalPlan::Describe {
            input: self.input(),
            columns: strings(names),
        })
    }

    /// The statistics `statistics` of every column, such as `"count"`, `"mean"`, `"min"` and
    /// percentiles as `"25%"`; with none, the count, mean, standard deviation, minimum,
    /// quartiles and maximum.
    pub fn summary(&self, statistics: impl IntoIterator<Item = impl Into<String>>) -> DataFrame {
        self.with_plan(LogicalPlan::Summary {
            input: self.input(),
            percentiles: strings(statistics),
        })
    }

    /// Every column, and the column `name` of the index of each row, from 0, in the order of
    /// the rows.
    pub fn zip_with_index(&self, name: &str) -> DataFrame {
        let index =
            Column::new(Expression::function("distributed_sequence_id", vec![])).alias(name);
        let every = Column::new(ExpressionNode::UnresolvedStar(None, None).into());
        self.select([every, index])
    }

    /// This DataFrame, which collects the metrics `metrics`, aggregate expressions, of its
    /// rows when an action runs it, into `observation`: an [`Observation`], or a name, whose
    /// metrics a [`RecordBatchStream`](crate::stream::RecordBatchStream) reports in its
    /// execution info. An observation has the metrics of the latest action that ran it.
    pub fn observe(
        &self,
        observation: impl Into<Observation>,
        metrics: impl IntoIterator<Item = impl Into<Column>>,
    ) -> DataFrame {
        let observation = observation.into().into_core();
        let name = observation
            .name_to_observe()
            .unwrap_or_else(|_| observation.name().unwrap_or_default());
        observation.start();
        self.with_plan(LogicalPlan::Observe {
            input: self.input(),
            name,
            exprs: expressions(metrics),
            observation: Some(observation),
        })
    }

    /// Whether this DataFrame is of rows that the client holds, such as the result of a
    /// command.
    pub fn is_local(&self) -> bool {
        use proto::relation::RelType;
        match self.plan.as_ref() {
            LogicalPlan::LocalRelation { .. } | LogicalPlan::ChunkedCachedLocalRelation { .. } => {
                true
            }
            LogicalPlan::CachedRelation { relation } => matches!(
                relation.rel_type,
                Some(RelType::LocalRelation(_))
                    | Some(RelType::CachedLocalRelation(_))
                    | Some(RelType::ChunkedCachedLocalRelation(_))
            ),
            _ => false,
        }
    }

    /// `f` of this DataFrame, so that a chain of transformations can call a function of its
    /// own.
    pub fn transform(&self, f: impl FnOnce(&DataFrame) -> DataFrame) -> DataFrame {
        f(self)
    }
}

/// The name by which the server parses the join type of an as-of or a nearest-by join.
fn join_type_name(left_outer: bool) -> String {
    if left_outer { "leftouter" } else { "inner" }.to_string()
}

impl AsSubquery for DataFrame {
    fn as_subquery(&self, subquery_type: SubqueryType) -> SubqueryExpression {
        self.subquery(subquery_type)
    }
}
