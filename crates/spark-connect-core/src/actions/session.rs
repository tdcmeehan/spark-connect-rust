//! The actions of a session: the analyses it asks the server for, and its SQL commands.

use std::sync::Arc;

use spark_connect_proto as proto;

use super::{execute_command, ExecutionObserver};
use crate::client::SparkConnectClient;
use crate::error::{Result, SparkError};
use crate::execution;
use crate::plan::PlanIds;
use crate::types::DataType;

/// Asks the server for `analyze`, and returns what `take` reads from the result, or else an
/// error that the response did not contain `what`.
async fn analyze<T>(
    client: &SparkConnectClient,
    analyze: proto::analyze_plan_request::Analyze,
    what: &str,
    take: impl FnOnce(proto::analyze_plan_response::Result) -> Option<Result<T>>,
) -> Result<T> {
    let request = execution::analyze_plan_request(client.session_id(), analyze);
    let result = client.analyze_plan(request).await?.result;
    result.and_then(take).unwrap_or_else(|| {
        Err(SparkError::connect_msg(format!(
            "AnalyzePlan response did not contain {what}"
        )))
    })
}

/// The Spark version of the server.
pub async fn spark_version(client: Arc<SparkConnectClient>) -> Result<String> {
    use proto::analyze_plan_response::Result as R;
    let request = proto::analyze_plan_request::Analyze::SparkVersion(Default::default());
    analyze(&client, request, "a spark version", |result| match result {
        R::SparkVersion(v) => Some(Ok(v.version)),
        _ => None,
    })
    .await
}

/// The type of the DDL string `ddl`, as the server parses it.
pub async fn parse_ddl(client: Arc<SparkConnectClient>, ddl: String) -> Result<DataType> {
    use proto::analyze_plan_response::Result as R;
    let request =
        proto::analyze_plan_request::Analyze::DdlParse(proto::analyze_plan_request::DdlParse {
            ddl_string: ddl,
        });
    analyze(
        &client,
        request,
        "a DDL parse result",
        |result| match result {
            R::DdlParse(r) => Some(DataType::from_proto(&r.parsed.unwrap_or_default())),
            _ => None,
        },
    )
    .await
}

/// The DDL string of the schema `json`, as the server converts it.
pub async fn json_to_ddl(client: Arc<SparkConnectClient>, json: String) -> Result<String> {
    use proto::analyze_plan_response::Result as R;
    let request =
        proto::analyze_plan_request::Analyze::JsonToDdl(proto::analyze_plan_request::JsonToDdl {
            json_string: json,
        });
    analyze(&client, request, "a DDL string", |result| match result {
        R::JsonToDdl(r) => Some(Ok(r.ddl_string)),
        _ => None,
    })
    .await
}

/// Runs the SQL statement of `input` as a command, as PySpark's `sql` does, and returns the
/// relation of its result, if the server returns one: the result of a command, or the relation
/// of a query, which runs at an action.
pub async fn sql(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    input: proto::Relation,
    observer: impl ExecutionObserver,
) -> Result<(Option<proto::Relation>, execution::ExecutionInfo)> {
    let command = proto::command::CommandType::SqlCommand(proto::SqlCommand {
        input: Some(input),
        ..Default::default()
    });
    let (responses, info) = execute_command(client, plan_ids, tags, command, observer).await?;
    let relation = responses
        .into_iter()
        .find_map(|response| match response.response_type {
            Some(proto::execute_plan_response::ResponseType::SqlCommandResult(result)) => {
                result.relation
            }
            _ => None,
        });
    Ok((relation, info))
}

/// The function that registers the Java UDF of the class `class_name` as `name`: of
/// `return_type`, or else an aggregate function when `aggregate`.
pub fn java_function(
    name: &str,
    class_name: &str,
    return_type: Option<&DataType>,
    aggregate: bool,
) -> proto::CommonInlineUserDefinedFunction {
    let mut java_udf = proto::JavaUdf::default();
    java_udf.class_name = class_name.to_string();
    if let Some(return_type) = return_type {
        java_udf.output_type = Some(return_type.to_proto());
    } else {
        java_udf.aggregate = aggregate;
    }
    proto::CommonInlineUserDefinedFunction {
        function_name: name.to_string(),
        // As PySpark's CommonInlineUserDefinedFunction, which is not deterministic by default.
        deterministic: false,
        function: Some(proto::common_inline_user_defined_function::Function::JavaUdf(java_udf)),
        ..Default::default()
    }
}

/// Creates the resource profile `profile` on the server and returns the id it gave it.
pub async fn create_resource_profile(
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    tags: Vec<String>,
    profile: proto::ResourceProfile,
    observer: impl ExecutionObserver,
) -> Result<i32> {
    let command = proto::command::CommandType::CreateResourceProfileCommand(
        proto::CreateResourceProfileCommand {
            profile: Some(profile),
        },
    );
    let (responses, _) = execute_command(client, plan_ids, tags, command, observer).await?;
    responses
        .into_iter()
        .find_map(|response| match response.response_type {
            Some(
                proto::execute_plan_response::ResponseType::CreateResourceProfileCommandResult(
                    result,
                ),
            ) => Some(result.profile_id),
            _ => None,
        })
        .ok_or_else(|| {
            SparkError::connect_msg(
                "build_resource_profile: server returned no CreateResourceProfileCommandResult",
            )
        })
}
