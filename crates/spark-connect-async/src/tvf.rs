//! Table-valued functions, and the table arguments that they take.

use spark_connect_core::expression::SubqueryExpression;
use spark_connect_core::plan::LogicalPlan;

use crate::column::{expressions, lit, subquery_column, Column, IntoArgument};
use crate::dataframe::DataFrame;
use crate::session::Session;

/// The table-valued functions of a session, which [`Session::tvf`] returns: each returns a
/// DataFrame of the rows of a function call.
#[derive(Clone)]
pub struct TableFunctions {
    session: Session,
}

impl TableFunctions {
    pub(crate) fn new(session: Session) -> Self {
        TableFunctions { session }
    }

    /// The rows of the table-valued function `name`, such as a registered user-defined table
    /// function, of `arguments`: Rust values, as [`lit`] makes them, columns of literals, or
    /// [`TableArg`]s.
    pub fn call(
        &self,
        name: impl Into<String>,
        arguments: impl IntoIterator<Item = impl IntoArgument>,
    ) -> DataFrame {
        let arguments = arguments
            .into_iter()
            .map(|argument| argument.into_argument().expression().clone())
            .collect();
        DataFrame::new(
            self.session.clone(),
            LogicalPlan::UnresolvedTableValuedFunction {
                name: name.into(),
                arguments,
            },
        )
    }

    /// A row of each element of the array or map `collection`: a column `col` of the
    /// elements, or `key` and `value` of the entries.
    pub fn explode(&self, collection: impl Into<Column>) -> DataFrame {
        self.call("explode", [collection.into()])
    }

    /// As [`TableFunctions::explode`], with one row of `NULL`s for a `NULL` or empty
    /// collection.
    pub fn explode_outer(&self, collection: impl Into<Column>) -> DataFrame {
        self.call("explode_outer", [collection.into()])
    }

    /// As [`TableFunctions::explode`], with the position of each element in a column `pos`.
    pub fn posexplode(&self, collection: impl Into<Column>) -> DataFrame {
        self.call("posexplode", [collection.into()])
    }

    /// As [`TableFunctions::posexplode`], with one row of `NULL`s for a `NULL` or empty
    /// collection.
    pub fn posexplode_outer(&self, collection: impl Into<Column>) -> DataFrame {
        self.call("posexplode_outer", [collection.into()])
    }

    /// A row of each struct of the array of structs `input`, with a column of each field.
    pub fn inline(&self, input: impl Into<Column>) -> DataFrame {
        self.call("inline", [input.into()])
    }

    /// As [`TableFunctions::inline`], with one row of `NULL`s for a `NULL` or empty array.
    pub fn inline_outer(&self, input: impl Into<Column>) -> DataFrame {
        self.call("inline_outer", [input.into()])
    }

    /// A row of the values of the fields `fields` of the JSON objects of `json`, with a column
    /// `c0`, `c1`, ... of each field.
    pub fn json_tuple(
        &self,
        json: impl Into<Column>,
        fields: impl IntoIterator<Item = impl Into<String>>,
    ) -> DataFrame {
        let arguments =
            std::iter::once(json.into()).chain(fields.into_iter().map(|field| lit(field.into())));
        self.call("json_tuple", arguments)
    }

    /// `rows` rows of the values of `columns`, in the order of the columns: each row takes the
    /// next values, into as many columns as it takes to hold them.
    pub fn stack(
        &self,
        rows: i32,
        columns: impl IntoIterator<Item = impl Into<Column>>,
    ) -> DataFrame {
        let arguments = std::iter::once(lit(rows)).chain(columns.into_iter().map(Into::into));
        self.call("stack", arguments)
    }

    /// A row of each element of the variant `input`, an array or an object, with its position,
    /// its key and its value.
    pub fn variant_explode(&self, input: impl Into<Column>) -> DataFrame {
        self.call("variant_explode", [input.into()])
    }

    /// As [`TableFunctions::variant_explode`], with one row of `NULL`s for a `NULL` variant or
    /// one that is not an array or an object.
    pub fn variant_explode_outer(&self, input: impl Into<Column>) -> DataFrame {
        self.call("variant_explode_outer", [input.into()])
    }

    /// A row of each collation that the server supports.
    pub fn collations(&self) -> DataFrame {
        self.call("collations", Vec::<Column>::new())
    }

    /// A row of each keyword of the server's SQL, with whether it is reserved.
    pub fn sql_keywords(&self) -> DataFrame {
        self.call("sql_keywords", Vec::<Column>::new())
    }
}

/// The rows of a DataFrame as an argument of a table-valued function, which
/// [`DataFrame::as_table`] returns. Partition them with [`TableArg::partition_by`] or
/// [`TableArg::with_single_partition`], and then order each partition with
/// [`PartitionedTableArg::order_by`].
#[derive(Debug, Clone)]
pub struct TableArg {
    subquery: SubqueryExpression,
}

impl TableArg {
    pub(crate) fn new(subquery: SubqueryExpression) -> Self {
        TableArg { subquery }
    }

    /// The rows, in partitions by the values of `columns`, so that the function sees the rows
    /// of each value together.
    pub fn partition_by(
        mut self,
        columns: impl IntoIterator<Item = impl Into<Column>>,
    ) -> PartitionedTableArg {
        self.subquery.partition_spec = expressions(columns);
        PartitionedTableArg {
            subquery: self.subquery,
        }
    }

    /// The rows, in one partition.
    pub fn with_single_partition(mut self) -> PartitionedTableArg {
        self.subquery.with_single_partition = Some(true);
        PartitionedTableArg {
            subquery: self.subquery,
        }
    }
}

/// A [`TableArg`] whose rows are partitioned.
#[derive(Debug, Clone)]
pub struct PartitionedTableArg {
    subquery: SubqueryExpression,
}

impl PartitionedTableArg {
    /// The rows of each partition, in the order of `columns`, which sort ascending unless they
    /// are sort orders, such as `col.desc()`. A later call adds to the order.
    pub fn order_by(mut self, columns: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        self.subquery.order_spec.extend(expressions(columns));
        self
    }
}

impl From<TableArg> for Column {
    fn from(table: TableArg) -> Column {
        subquery_column(table.subquery)
    }
}

impl From<PartitionedTableArg> for Column {
    fn from(table: PartitionedTableArg) -> Column {
        subquery_column(table.subquery)
    }
}

impl IntoArgument for TableArg {
    fn into_argument(self) -> Column {
        self.into()
    }
}

impl IntoArgument for PartitionedTableArg {
    fn into_argument(self) -> Column {
        self.into()
    }
}
