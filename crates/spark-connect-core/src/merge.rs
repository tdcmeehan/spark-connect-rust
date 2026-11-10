//! The actions of a `MergeIntoTableCommand`.

use spark_connect_proto as proto;

use crate::column::Column;

/// The merge action of `action_type`, when `condition` holds, with `assignments` of the
/// target columns to the values to set them to, sent in their order.
pub fn merge_action(
    action_type: proto::merge_action::ActionType,
    condition: Option<&Column>,
    assignments: impl IntoIterator<Item = (String, Column)>,
) -> proto::Expression {
    let mut action = proto::MergeAction::default();
    action.action_type = action_type as i32;
    action.condition = condition.map(|c| Box::new(c.to_proto()));
    action.assignments = assignments
        .into_iter()
        .map(|(k, v)| proto::merge_action::Assignment {
            // Mirrors the reference `expr(k)` for the assignment target.
            key: Some(crate::functions::expr(&k).to_proto()),
            value: Some(v.to_proto()),
        })
        .collect();
    let mut expr = proto::Expression::default();
    expr.expr_type = Some(proto::expression::ExprType::MergeAction(Box::new(action)));
    expr
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
        );
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
