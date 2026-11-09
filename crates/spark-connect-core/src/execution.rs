//! What an execution reports, and the requests that execute and analyze plans.

use spark_connect_proto as proto;

use crate::error::SparkError;
use crate::plan::{LogicalPlan, PlanIds};

/// Metrics captured from the last executed query. Mirrors the data behind
/// `pyspark.sql.DataFrame.executionInfo` / `client.core.ExecutionInfo`.
#[derive(Debug, Clone, Default)]
pub struct ExecutionInfo {
    /// Plan metrics (typically present on the final response of an execution).
    pub metrics: Option<proto::execute_plan_response::Metrics>,
    /// Observed metrics collected during execution (also where `observe(...)`
    /// and UDF-profiler results surface).
    pub observed_metrics: Vec<proto::execute_plan_response::ObservedMetrics>,
    /// The ExecutePlan operation id: assigned by the client before the request is sent
    /// (the request's id, or a new UUID) and echoed by the server in
    /// `ExecutePlanResponse.operation_id`. Mirrors `ExecutionInfo.operation_id` (new in
    /// pyspark 4.3.0).
    pub operation_id: Option<String>,
}

impl ExecutionInfo {
    /// Takes the operation id, the metrics and the observed metrics off `response`, and sets
    /// the observed metrics in the observations of the plans of `plan_ids`. Returns the
    /// observed metrics that it took. They must be taken before the `response_type` of the
    /// response is consumed, as they are fields of their own.
    #[doc(hidden)]
    pub fn capture(
        &mut self,
        response: &mut proto::ExecutePlanResponse,
        plan_ids: &PlanIds,
    ) -> &[proto::execute_plan_response::ObservedMetrics] {
        if self.operation_id.is_none() && !response.operation_id.is_empty() {
            self.operation_id = Some(response.operation_id.clone());
        }
        if let Some(metrics) = response.metrics.take() {
            self.metrics = Some(metrics);
        }
        let taken = self.observed_metrics.len();
        let metrics = std::mem::take(&mut response.observed_metrics);
        set_observations(plan_ids, &metrics);
        self.observed_metrics.extend(metrics);
        &self.observed_metrics[taken..]
    }
}

/// Set the metrics that the server observed in the Observations of their relations, as
/// PySpark's client sets them in the Observations of the plan: those of each relation, which
/// the server reports by its plan ID, or the error of collecting them.
fn set_observations(plan_ids: &PlanIds, metrics: &[proto::execute_plan_response::ObservedMetrics]) {
    for observed in metrics {
        let Some(node) = plan_ids.node_of(observed.plan_id) else {
            continue;
        };
        let LogicalPlan::Observe {
            name,
            observation: Some(observation),
            ..
        } = node.as_ref()
        else {
            continue;
        };
        if *name != observed.name {
            continue;
        }
        match observed.root_error_idx {
            Some(root_error_idx) => observation.set_error(SparkError::from_observation_errors(
                root_error_idx,
                observed.errors.clone(),
            )),
            None => observation.update(&observed.keys, &observed.values),
        }
    }
}

/// The plan that runs `relation` and returns its rows.
pub fn root(relation: proto::Relation) -> proto::Plan {
    proto::Plan {
        op_type: Some(proto::plan::OpType::Root(relation)),
    }
}

/// The plan that runs `command`.
pub fn command(command_type: proto::command::CommandType) -> proto::Plan {
    proto::Plan {
        op_type: Some(proto::plan::OpType::Command(proto::Command {
            command_type: Some(command_type),
        })),
    }
}

/// The request that executes `plan` in the session `session_id`, tagged with `tags`.
pub fn execute_plan_request(
    session_id: &str,
    tags: Vec<String>,
    plan: proto::Plan,
) -> proto::ExecutePlanRequest {
    proto::ExecutePlanRequest {
        session_id: session_id.to_string(),
        user_context: Some(proto::UserContext::default()),
        tags,
        plan: Some(plan),
        ..Default::default()
    }
}

/// The request that analyzes a plan in the session `session_id`, as `analyze` says.
pub fn analyze_plan_request(
    session_id: &str,
    analyze: proto::analyze_plan_request::Analyze,
) -> proto::AnalyzePlanRequest {
    proto::AnalyzePlanRequest {
        session_id: session_id.to_string(),
        user_context: Some(proto::UserContext::default()),
        analyze: Some(analyze),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observed(name: &str) -> proto::execute_plan_response::ObservedMetrics {
        proto::execute_plan_response::ObservedMetrics {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn metrics(plan_id: i64) -> proto::execute_plan_response::Metrics {
        proto::execute_plan_response::Metrics {
            metrics: vec![proto::execute_plan_response::metrics::MetricObject {
                plan_id,
                ..Default::default()
            }],
        }
    }

    fn names(metrics: &[proto::execute_plan_response::ObservedMetrics]) -> Vec<&str> {
        metrics.iter().map(|m| m.name.as_str()).collect()
    }

    #[test]
    fn capture_returns_only_the_observed_metrics_of_its_response() {
        let plan_ids = PlanIds::new();
        let mut info = ExecutionInfo::default();
        let mut first = proto::ExecutePlanResponse {
            operation_id: "op-1".to_string(),
            metrics: Some(metrics(1)),
            observed_metrics: vec![observed("a")],
            ..Default::default()
        };
        assert_eq!(names(info.capture(&mut first, &plan_ids)), ["a"]);
        let mut second = proto::ExecutePlanResponse {
            operation_id: "op-2".to_string(),
            metrics: Some(metrics(2)),
            observed_metrics: vec![observed("b"), observed("c")],
            ..Default::default()
        };
        assert_eq!(names(info.capture(&mut second, &plan_ids)), ["b", "c"]);
        assert_eq!(info.operation_id.as_deref(), Some("op-1"));
        assert_eq!(info.metrics, Some(metrics(2)));
        assert_eq!(names(&info.observed_metrics), ["a", "b", "c"]);
        assert!(second.metrics.is_none() && second.observed_metrics.is_empty());
    }
}
