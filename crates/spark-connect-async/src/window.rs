//! The windows of window functions: the rows that a function computes each row's value over.

use spark_connect_core::expression::{ExpressionNode, SortOrder};
use spark_connect_core::window::WindowSpec;

pub use spark_connect_core::window::FrameBound;

use crate::column::Column;

/// The window of [`Column::over`]: the rows of a row's partition, in an order, within a frame.
/// Without a partition, every row is in one partition; without a frame, the frame is the
/// server's default for the order.
///
/// ```
/// use spark_connect_async::col;
/// use spark_connect_async::window::{FrameBound, Window};
///
/// let window = Window::new()
///     .partition_by(["department"])
///     .order_by([col("salary").desc()])
///     .rows_between(FrameBound::UnboundedPreceding, FrameBound::CurrentRow);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Window(WindowSpec);

impl Window {
    /// The window of every row, in no order.
    pub fn new() -> Self {
        Window::default()
    }

    /// Partitions the rows by the values of `columns`, after any partitioning already set.
    pub fn partition_by(self, columns: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        Window(self.0.partition_by(crate::column::expressions(columns)))
    }

    /// Orders the rows of a partition by `columns`, after any order already set. A column is
    /// ascending with `NULL`s first, unless it is a sort order such as `col("a").desc()`.
    pub fn order_by(self, columns: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        let orders = columns
            .into_iter()
            .map(|column| {
                let expression = column.into().expression().clone();
                match expression.node() {
                    ExpressionNode::SortOrder(order) => (**order).clone(),
                    _ => SortOrder::asc_nulls_first(expression),
                }
            })
            .collect();
        Window(self.0.order_by(orders))
    }

    /// The frame of the rows from `start` to `end`, both included, counted in rows from the
    /// current one.
    pub fn rows_between(self, start: FrameBound, end: FrameBound) -> Self {
        Window(self.0.rows_between(start, end))
    }

    /// The frame of the rows whose order value is from the current row's plus `start` to the
    /// current row's plus `end`, of a window ordered by one numeric column.
    pub fn range_between(self, start: FrameBound, end: FrameBound) -> Self {
        Window(self.0.range_between(start, end))
    }

    pub(crate) fn into_core(self) -> WindowSpec {
        self.0
    }
}
