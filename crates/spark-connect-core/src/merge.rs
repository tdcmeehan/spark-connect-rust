//! The actions of a `MergeIntoTableCommand`.

use std::collections::HashMap;

use spark_connect_proto as proto;

use crate::column::Column;
use crate::error::Result;

/// The merge action of `action_type`, when `condition` holds, with `assignments` of the
/// target columns to the values to set them to.
pub fn merge_action(
    action_type: proto::merge_action::ActionType,
    condition: Option<&Column>,
    assignments: Option<HashMap<String, Column>>,
) -> MergeClause {
    MergeClause {
        action_type,
        condition: condition.cloned(),
        assignments,
    }
}

/// A clause of a merge: its action, and its condition and assignments, which are serialized when
/// the merge runs, as a request serializes them (see `MergeIntoWriter::merge`).
#[derive(Clone)]
pub struct MergeClause {
    action_type: proto::merge_action::ActionType,
    condition: Option<Column>,
    assignments: Option<HashMap<String, Column>>,
}

impl MergeClause {
    /// The `MergeAction` expression of the clause, as PySpark's `_build_merge_action` builds it.
    pub fn to_proto(&self) -> Result<proto::Expression> {
        let assignments = self.assignments.iter().flatten().map(|(k, v)| {
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
