//! `MERGE INTO` a table: the rows of a DataFrame update, insert into or delete from the rows of
//! the table that they match, or do not match, by a condition.
//!
//! ```no_run
//! # async fn run(updates: spark_connect_async::DataFrame) -> spark_connect_async::Result<()> {
//! use spark_connect_async::{col, Column};
//!
//! updates
//!     .merge_into("people", col("people.id").eq(col("updates.id")))
//!     .when_matched(None)
//!     .update([("name", col("updates.name"))])
//!     .when_not_matched(None)
//!     .insert_all()
//!     .await?;
//! # Ok(())
//! # }
//! ```

use std::future::{Future, IntoFuture};
use std::pin::Pin;

use spark_connect_core::column::Column as CoreColumn;
use spark_connect_core::merge::merge_action;
use spark_connect_proto as proto;
use spark_connect_proto::merge_action::ActionType;

use crate::column::{to_core, Column};
use crate::dataframe::DataFrame;
use crate::error::Result;
use crate::stream;

/// A `MERGE INTO` of the rows of a DataFrame into a table, from [`DataFrame::merge_into`]: its
/// clauses, in order, then awaited to run it. The server applies the first clause of each kind
/// whose condition holds for a row.
#[derive(Clone)]
#[must_use = "a merge does nothing until it is awaited"]
pub struct MergeIntoWriter {
    source: DataFrame,
    table: String,
    condition: CoreColumn,
    schema_evolution: bool,
    matched: Vec<proto::Expression>,
    not_matched: Vec<proto::Expression>,
    not_matched_by_source: Vec<proto::Expression>,
}

/// The clauses of a merge, by when they apply.
#[derive(Clone, Copy)]
enum Clause {
    Matched,
    NotMatched,
    NotMatchedBySource,
}

impl MergeIntoWriter {
    /// A clause for the rows of the table that a source row matches, and for which `condition`
    /// holds, if one is given.
    pub fn when_matched(self, condition: Option<Column>) -> WhenMatched {
        WhenMatched(self.clause(Clause::Matched, condition))
    }

    /// A clause for the source rows that match no row of the table, and for which `condition`
    /// holds, if one is given.
    pub fn when_not_matched(self, condition: Option<Column>) -> WhenNotMatched {
        WhenNotMatched(self.clause(Clause::NotMatched, condition))
    }

    /// A clause for the rows of the table that no source row matches, and for which `condition`
    /// holds, if one is given.
    pub fn when_not_matched_by_source(self, condition: Option<Column>) -> WhenNotMatchedBySource {
        WhenNotMatchedBySource(self.clause(Clause::NotMatchedBySource, condition))
    }

    /// Add the columns of the source that the table lacks to the table, as the clauses set
    /// them.
    pub fn with_schema_evolution(mut self) -> Self {
        self.schema_evolution = true;
        self
    }

    fn clause(self, clause: Clause, condition: Option<Column>) -> Pending {
        Pending {
            writer: self,
            clause,
            condition: condition.map(to_core),
        }
    }
}

impl IntoFuture for MergeIntoWriter {
    type Output = Result<()>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<()>> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        let command = proto::MergeIntoTableCommand {
            target_table_name: self.table,
            source_table_plan: Some(self.source.to_relation()),
            merge_condition: Some(self.condition.to_proto()),
            match_actions: self.matched,
            not_matched_actions: self.not_matched,
            not_matched_by_source_actions: self.not_matched_by_source,
            with_schema_evolution: self.schema_evolution,
        };
        stream::command(
            self.source.session().clone(),
            proto::command::CommandType::MergeIntoTableCommand(command),
        )
    }
}

/// A clause whose action is not chosen yet.
struct Pending {
    writer: MergeIntoWriter,
    clause: Clause,
    condition: Option<CoreColumn>,
}

impl Pending {
    fn action(
        self,
        action_type: ActionType,
        assignments: impl IntoIterator<Item = (String, CoreColumn)>,
    ) -> MergeIntoWriter {
        let mut writer = self.writer;
        let action = merge_action(action_type, self.condition.as_ref(), assignments);
        match self.clause {
            Clause::Matched => writer.matched.push(action),
            Clause::NotMatched => writer.not_matched.push(action),
            Clause::NotMatchedBySource => writer.not_matched_by_source.push(action),
        }
        writer
    }

    fn assigning(
        self,
        action_type: ActionType,
        assignments: impl IntoIterator<Item = (impl Into<String>, impl Into<Column>)>,
    ) -> MergeIntoWriter {
        let assignments = assignments
            .into_iter()
            .map(|(name, value)| (name.into(), to_core(value)));
        self.action(action_type, assignments)
    }
}

/// The action of a [`MergeIntoWriter::when_matched`] clause.
#[must_use = "a clause needs its action"]
pub struct WhenMatched(Pending);

impl WhenMatched {
    /// Set the columns of the matched row of the table, sent in the given order, to their
    /// values, which may refer to the source row.
    pub fn update(
        self,
        assignments: impl IntoIterator<Item = (impl Into<String>, impl Into<Column>)>,
    ) -> MergeIntoWriter {
        self.0.assigning(ActionType::Update, assignments)
    }

    /// Set every column of the matched row of the table to the source row's column of the same
    /// name.
    pub fn update_all(self) -> MergeIntoWriter {
        self.0.action(ActionType::UpdateStar, [])
    }

    /// Delete the matched row of the table.
    pub fn delete(self) -> MergeIntoWriter {
        self.0.action(ActionType::Delete, [])
    }
}

/// The action of a [`MergeIntoWriter::when_not_matched`] clause.
#[must_use = "a clause needs its action"]
pub struct WhenNotMatched(Pending);

impl WhenNotMatched {
    /// Insert a row whose columns are set to their values, sent in the given order, which may
    /// refer to the source row.
    pub fn insert(
        self,
        assignments: impl IntoIterator<Item = (impl Into<String>, impl Into<Column>)>,
    ) -> MergeIntoWriter {
        self.0.assigning(ActionType::Insert, assignments)
    }

    /// Insert the source row, its columns into those of the same names.
    pub fn insert_all(self) -> MergeIntoWriter {
        self.0.action(ActionType::InsertStar, [])
    }
}

/// The action of a [`MergeIntoWriter::when_not_matched_by_source`] clause.
#[must_use = "a clause needs its action"]
pub struct WhenNotMatchedBySource(Pending);

impl WhenNotMatchedBySource {
    /// Set the columns of the row of the table to their values, sent in the given order.
    pub fn update(
        self,
        assignments: impl IntoIterator<Item = (impl Into<String>, impl Into<Column>)>,
    ) -> MergeIntoWriter {
        self.0.assigning(ActionType::Update, assignments)
    }

    /// Update every column of the row of the table.
    pub fn update_all(self) -> MergeIntoWriter {
        self.0.action(ActionType::UpdateStar, [])
    }

    /// Delete the row of the table.
    pub fn delete(self) -> MergeIntoWriter {
        self.0.action(ActionType::Delete, [])
    }
}

impl DataFrame {
    /// A `MERGE INTO` of the rows of this DataFrame into the table `table`, whose rows a row of
    /// this DataFrame matches when `condition` holds.
    pub fn merge_into(
        &self,
        table: impl Into<String>,
        condition: impl Into<Column>,
    ) -> MergeIntoWriter {
        MergeIntoWriter {
            source: self.clone(),
            table: table.into(),
            condition: to_core(condition),
            schema_evolution: false,
            matched: Vec::new(),
            not_matched: Vec::new(),
            not_matched_by_source: Vec::new(),
        }
    }
}
