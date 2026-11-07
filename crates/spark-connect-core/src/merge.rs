//! The actions of a `MergeIntoTableCommand`.

use std::collections::HashMap;

use spark_connect_proto as proto;

use crate::column::Column;

/// The merge action of `action_type`, when `condition` holds, with `assignments` of the
/// target columns to the values to set them to.
pub fn merge_action(
    action_type: proto::merge_action::ActionType,
    condition: Option<&Column>,
    assignments: Option<HashMap<String, Column>>,
) -> proto::Expression {
    let mut action = proto::MergeAction::default();
    action.action_type = action_type as i32;
    action.condition = condition.map(|c| Box::new(c.to_proto()));
    if let Some(assignments) = assignments {
        action.assignments = assignments
            .into_iter()
            .map(|(k, v)| proto::merge_action::Assignment {
                // Mirrors the reference `expr(k)` for the assignment target.
                key: Some(crate::functions::expr(&k).to_proto()),
                value: Some(v.to_proto()),
            })
            .collect();
    }
    let mut expr = proto::Expression::default();
    expr.expr_type = Some(proto::expression::ExprType::MergeAction(Box::new(action)));
    expr
}
