//! The actions of a `MergeIntoTableCommand`.

use spark_connect_proto as proto;

use crate::column::Column;
use crate::error::Result;

/// The merge action of `action_type`, when `condition` holds, with `assignments` of the
/// target columns to the values to set them to, sent in their order.
pub fn merge_action(
    action_type: proto::merge_action::ActionType,
    condition: Option<&Column>,
    assignments: impl IntoIterator<Item = (String, Column)>,
) -> MergeClause {
    MergeClause {
        action_type,
        condition: condition.cloned(),
        assignments: assignments.into_iter().collect(),
    }
}

/// A clause of a merge: its action, and its condition and assignments, which are serialized when
/// the merge runs, as a request serializes them (see `MergeIntoWriter::merge`).
#[derive(Clone)]
pub struct MergeClause {
    action_type: proto::merge_action::ActionType,
    condition: Option<Column>,
    assignments: Vec<(String, Column)>,
}

impl MergeClause {
    /// The `MergeAction` expression of the clause, as PySpark's `_build_merge_action` builds it.
    pub fn to_proto(&self) -> Result<proto::Expression> {
        let assignments = self.assignments.iter().map(|(k, v)| {
            Ok(proto::merge_action::Assignment {
                // Mirrors the reference `expr(k)` for the assignment target.
                key: Some(crate::functions::expr(k).try_to_proto()?),
                value: Some(v.try_to_proto()?),
            })
        });
        let action = proto::MergeAction {
            action_type: self.action_type as i32,
            assignments: assignments.collect::<Result<_>>()?,
            condition: self
                .condition
                .as_ref()
                .map(|c| c.try_to_proto().map(Box::new))
                .transpose()?,
        };
        Ok(proto::Expression {
            expr_type: Some(proto::expression::ExprType::MergeAction(Box::new(action))),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{col, lit};
    use crate::functions::expr;

    #[test]
    fn the_assignments_of_a_merge_action_are_sent_in_their_order() {
        let assignments = [("c", 3), ("a", 1), ("b", 2)];
        let action = merge_action(
            proto::merge_action::ActionType::Update,
            Some(&col("matched")),
            assignments.map(|(name, value)| (name.to_string(), lit(value))),
        )
        .to_proto()
        .unwrap();
        let Some(proto::expression::ExprType::MergeAction(action)) = action.expr_type else {
            panic!("not a merge action: {action:?}");
        };
        let sent: Vec<_> = action
            .assignments
            .into_iter()
            .map(|assignment| (assignment.key.unwrap(), assignment.value.unwrap()))
            .collect();
        let expected: Vec<_> = assignments
            .iter()
            .map(|&(name, value)| (expr(name).to_proto(), lit(value).to_proto()))
            .collect();
        assert_eq!(sent, expected);
    }
}
