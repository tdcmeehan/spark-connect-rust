// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

//! The requests of the Spark ML of the async API, recorded by the old API's recording server
//! and compared with the old API's.

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;
mod support;

use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use prost::Message;
use spark_connect::ml as old;
use spark_connect::ml::{Estimator as _, Evaluator as _, Transformer as _};
use spark_connect::SparkSession;
use spark_connect_async::ml::{
    BinaryClassificationEvaluator, Estimator, Evaluator, LinearRegression, LinearSVC,
    LogisticRegression, MaxAbsScaler, MulticlassClassificationEvaluator, RegressionEvaluator,
    StandardScaler, StringIndexer, Transformer, VectorAssembler,
};
use spark_connect_async::Session;
use spark_connect_proto as proto;

const SESSION_ID: &str = "00000000-0000-4000-8000-0000000000bb";

struct Case {
    name: &'static str,
    /// Whether the operations drop a fitted model, whose deletion is sent in the background.
    deletes: bool,
    old: fn(&SparkSession) -> Option<proto::Relation>,
    new: fn(Session) -> BoxFuture<'static, Option<proto::Relation>>,
}

/// A case of a fit of `$old` and `$new`, and of the transform of their models, which are then
/// dropped.
macro_rules! fit {
    ($name:expr, $old:expr, $new:expr) => {
        Case {
            name: $name,
            deletes: true,
            old: |s| {
                let data = s.range(3).unwrap();
                let mut estimator = $old;
                let mut model = estimator.fit(&data).unwrap();
                Some(model.transform(&data).unwrap().to_relation())
            },
            new: |s| {
                Box::pin(async move {
                    let data = s.range(3);
                    let model = $new.fit(&data).await.unwrap();
                    Some(model.transform(&data).to_relation())
                })
            },
        }
    };
}

/// A case of an evaluation by `$old` and `$new`.
macro_rules! evaluate {
    ($name:expr, $old:expr, $new:expr) => {
        Case {
            name: $name,
            deletes: false,
            old: |s| {
                $old.evaluate(&s.range(3).unwrap()).unwrap();
                None
            },
            new: |s| {
                Box::pin(async move {
                    $new.evaluate(&s.range(3)).await.unwrap();
                    None
                })
            },
        }
    };
}

/// A case of a transform by `$old` and `$new`, compared by the relations of the DataFrames.
macro_rules! transform {
    ($name:expr, $old:expr, $new:expr) => {
        Case {
            name: $name,
            deletes: false,
            old: |s| {
                let mut transformer = $old;
                Some(
                    transformer
                        .transform(&s.range(3).unwrap())
                        .unwrap()
                        .to_relation(),
                )
            },
            new: |s| Box::pin(async move { Some($new.transform(&s.range(3)).to_relation()) }),
        }
    };
}

fn cases() -> Vec<Case> {
    vec![
        fit!(
            "StandardScaler of columns",
            old::StandardScaler::new()
                .set_input_col("features")
                .set_output_col("scaled"),
            StandardScaler::new()
                .input_col("features")
                .output_col("scaled")
        ),
        fit!(
            "StandardScaler::default",
            old::StandardScaler::default(),
            StandardScaler::default()
        ),
        fit!(
            "MaxAbsScaler of columns",
            old::MaxAbsScaler::new()
                .set_input_col("f")
                .set_output_col("o"),
            MaxAbsScaler::new().input_col("f").output_col("o")
        ),
        fit!(
            "MaxAbsScaler::default",
            old::MaxAbsScaler::default(),
            MaxAbsScaler::default()
        ),
        fit!(
            "StringIndexer of columns",
            old::StringIndexer::new()
                .set_input_col("s")
                .set_output_col("i"),
            StringIndexer::new().input_col("s").output_col("i")
        ),
        fit!(
            "StringIndexer::default",
            old::StringIndexer::default(),
            StringIndexer::default()
        ),
        fit!(
            "LogisticRegression of every setting",
            old::LogisticRegression::new()
                .set_feature_col("f")
                .set_label_col("l")
                .set_prediction_col("p")
                .set_max_iter(7),
            LogisticRegression::new()
                .features_col("f")
                .label_col("l")
                .prediction_col("p")
                .max_iter(7)
        ),
        fit!(
            "LogisticRegression::default",
            old::LogisticRegression::default(),
            LogisticRegression::default()
        ),
        fit!(
            "LinearSVC of every setting",
            old::LinearSVC::new()
                .set_feature_col("f")
                .set_label_col("l")
                .set_prediction_col("p")
                .set_raw_prediction_col("r")
                .set_weight_col("w")
                .set_max_iter(7)
                .set_reg_param(0.1)
                .set_tol(0.001)
                .set_fit_intercept(false)
                .set_standardization(false)
                .set_threshold(0.3)
                .set_aggregation_depth(3)
                .set_max_block_size_in_mb(1.5),
            LinearSVC::new()
                .features_col("f")
                .label_col("l")
                .prediction_col("p")
                .raw_prediction_col("r")
                .weight_col("w")
                .max_iter(7)
                .reg_param(0.1)
                .tol(0.001)
                .fit_intercept(false)
                .standardization(false)
                .threshold(0.3)
                .aggregation_depth(3)
                .max_block_size_in_mb(1.5)
        ),
        fit!(
            "LinearSVC::default",
            old::LinearSVC::default(),
            LinearSVC::default()
        ),
        fit!(
            "LinearRegression of every setting",
            old::LinearRegression::new()
                .set_feature_col("f")
                .set_label_col("l")
                .set_prediction_col("p")
                .set_weight_col("w")
                .set_max_iter(7)
                .set_reg_param(0.1)
                .set_elastic_net_param(0.5)
                .set_tol(0.001)
                .set_fit_intercept(false)
                .set_standardization(false)
                .set_solver("normal")
                .set_aggregation_depth(3)
                .set_loss("huber")
                .set_epsilon(1.5)
                .set_max_block_size_in_mb(1.5),
            LinearRegression::new()
                .features_col("f")
                .label_col("l")
                .prediction_col("p")
                .weight_col("w")
                .max_iter(7)
                .reg_param(0.1)
                .elastic_net_param(0.5)
                .tol(0.001)
                .fit_intercept(false)
                .standardization(false)
                .solver("normal")
                .aggregation_depth(3)
                .loss("huber")
                .epsilon(1.5)
                .max_block_size_in_mb(1.5)
        ),
        fit!(
            "LinearRegression::default",
            old::LinearRegression::default(),
            LinearRegression::default()
        ),
        transform!(
            "VectorAssembler of an output column",
            old::VectorAssembler::new().set_output_col("v"),
            VectorAssembler::new().output_col("v")
        ),
        transform!(
            "VectorAssembler::default",
            old::VectorAssembler::default(),
            VectorAssembler::default()
        ),
        evaluate!(
            "RegressionEvaluator of every setting",
            old::RegressionEvaluator::new()
                .set_label_col("l")
                .set_prediction_col("p")
                .set_metric_name("mae"),
            RegressionEvaluator::new()
                .label_col("l")
                .prediction_col("p")
                .metric_name("mae")
        ),
        evaluate!(
            "RegressionEvaluator::default",
            old::RegressionEvaluator::default(),
            RegressionEvaluator::default()
        ),
        evaluate!(
            "BinaryClassificationEvaluator of a label column and a metric",
            old::BinaryClassificationEvaluator::new()
                .set_label_col("l")
                .set_metric_name("areaUnderPR"),
            BinaryClassificationEvaluator::new()
                .label_col("l")
                .metric_name("areaUnderPR")
        ),
        evaluate!(
            "BinaryClassificationEvaluator::default",
            old::BinaryClassificationEvaluator::default(),
            BinaryClassificationEvaluator::default()
        ),
        evaluate!(
            "MulticlassClassificationEvaluator of every setting",
            old::MulticlassClassificationEvaluator::new()
                .set_label_col("l")
                .set_prediction_col("p")
                .set_metric_name("accuracy"),
            MulticlassClassificationEvaluator::new()
                .label_col("l")
                .prediction_col("p")
                .metric_name("accuracy")
        ),
        evaluate!(
            "MulticlassClassificationEvaluator::default",
            old::MulticlassClassificationEvaluator::default(),
            MulticlassClassificationEvaluator::default()
        ),
    ]
}

/// `text` with each UUID replaced: the ids of the operators are random.
fn without_uuids(text: &str) -> String {
    let bytes = text.as_bytes();
    let is_uuid = |at: usize| {
        bytes.len() >= at + 36
            && bytes[at..at + 36].iter().enumerate().all(|(i, &byte)| {
                if matches!(i, 8 | 13 | 18 | 23) {
                    byte == b'-'
                } else {
                    byte.is_ascii_hexdigit()
                }
            })
    };
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while at < text.len() {
        if is_uuid(at) {
            out.push_str("<uuid>");
            at += 36;
        } else {
            let next = text[at..].chars().next().unwrap();
            out.push(next);
            at += next.len_utf8();
        }
    }
    out
}

fn is_delete(entry: &server::Entry) -> bool {
    entry.kind == "ExecutePlanRequest"
        && matches!(
            command_of(entry),
            Some(proto::command::CommandType::MlCommand(proto::MlCommand {
                command: Some(proto::ml_command::Command::Delete(_)),
            }))
        )
}

fn command_of(entry: &server::Entry) -> Option<proto::command::CommandType> {
    match proto::ExecutePlanRequest::decode(&entry.bytes[..])
        .ok()?
        .plan?
        .op_type?
    {
        proto::plan::OpType::Command(command) => command.command_type,
        _ => None,
    }
}

/// Each entry as its message, without the operation id, which the client chooses at random,
/// and without the ids of the operators.
fn messages(entries: Vec<server::Entry>) -> Vec<String> {
    entries
        .into_iter()
        .map(|entry| {
            let text = match entry.kind.as_str() {
                "ExecutePlanRequest" => {
                    let mut request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
                    request.operation_id = None;
                    format!("{request:?}")
                }
                "AnalyzePlanRequest" => format!(
                    "{:?}",
                    proto::AnalyzePlanRequest::decode(&entry.bytes[..]).unwrap()
                ),
                _ => format!("{entry:?}"),
            };
            without_uuids(&text)
        })
        .collect()
}

fn start() -> (Arc<server::Recorder>, String) {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let url = format!("sc://{address}/;session_id={SESSION_ID};user_id=snapshots");
    (recorder, url)
}

/// Waits for the deletion of a model that the operations of a case dropped.
async fn deleted(recorder: &Arc<server::Recorder>) -> bool {
    let recorder = Arc::clone(recorder);
    tokio::task::spawn_blocking(move || recorder.wait_for(Duration::from_secs(5), is_delete))
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spark_ml_sends_the_requests_of_the_old_api() {
    let (recorder, url) = start();
    let mut both = Vec::new();
    for case in cases() {
        let old_session = SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        let old_relation = (case.old)(&old_session);
        if case.deletes {
            assert!(deleted(&recorder).await, "{}: the old model", case.name);
        }
        let old = messages(recorder.take());
        let session = Session::builder()
            .remote(url.clone())
            .connect()
            .await
            .unwrap();
        let new_relation = (case.new)(session).await;
        if case.deletes {
            assert!(deleted(&recorder).await, "{}: the new model", case.name);
        }
        let new = messages(recorder.take());
        assert!(
            !old.is_empty() || old_relation.is_some(),
            "{}: the old API sent and built nothing",
            case.name
        );
        let text = |relation: Option<proto::Relation>| {
            relation.map(|relation| without_uuids(&format!("{relation:?}")))
        };
        both.push((
            case.name,
            (old, text(old_relation)),
            (new, text(new_relation)),
        ));
    }
    let differences = support::differences(both);
    assert!(differences.is_empty(), "{differences}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_operation_sends_the_tags_of_its_handle() {
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let nightly = session.with_tag("nightly").unwrap();
    for case in cases() {
        (case.new)(nightly.clone()).await;
        if case.deletes {
            assert!(deleted(&recorder).await, "{}", case.name);
        }
    }
    let model = LogisticRegression::new()
        .fit(&nightly.range(3))
        .await
        .unwrap();
    let _ = model.coefficients().await;
    let mut failures = Vec::new();
    let mut sent = 0;
    for entry in recorder.take() {
        if entry.kind != "ExecutePlanRequest" {
            continue;
        }
        sent += 1;
        let request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
        if request.tags != ["nightly"] {
            failures.push(format!("{:?}: tags {:?}", command_of(&entry), request.tags));
        }
    }
    assert!(sent > 30, "only {sent} requests");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The value of the param `name` of `params`.
fn param(params: &Option<proto::MlParams>, name: &str) -> Option<proto::expression::Literal> {
    params.as_ref()?.params.get(name).cloned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_params_that_the_old_api_sends_in_a_form_that_the_server_rejects_are_sent_as_it_takes_them(
) {
    use proto::expression::literal::LiteralType;
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let assembled = VectorAssembler::new()
        .input_cols(["a", "b"])
        .transform(&session.range(3))
        .to_relation();
    let Some(proto::relation::RelType::MlRelation(relation)) = assembled.rel_type else {
        panic!("not an ML relation: {assembled:?}");
    };
    let Some(proto::ml_relation::MlType::Transform(transform)) = relation.ml_type else {
        panic!("not a transform: {relation:?}");
    };
    let Some(LiteralType::Array(columns)) =
        param(&transform.params, "inputCols").and_then(|columns| columns.literal_type)
    else {
        panic!("inputCols is not an array: {:?}", transform.params);
    };
    let names: Vec<_> = columns
        .elements
        .iter()
        .map(|name| name.literal_type.clone())
        .collect();
    assert_eq!(
        names,
        [
            Some(LiteralType::String("a".to_string())),
            Some(LiteralType::String("b".to_string()))
        ]
    );
    BinaryClassificationEvaluator::new()
        .raw_prediction_col("margin")
        .evaluate(&session.range(3))
        .await
        .unwrap();
    let evaluations: Vec<_> = recorder
        .take()
        .iter()
        .filter_map(command_of)
        .filter_map(|command| match command {
            proto::command::CommandType::MlCommand(proto::MlCommand {
                command: Some(proto::ml_command::Command::Evaluate(evaluate)),
            }) => Some(evaluate.params),
            _ => None,
        })
        .collect();
    assert_eq!(evaluations.len(), 1);
    assert_eq!(
        param(&evaluations[0], "rawPredictionCol").and_then(|column| column.literal_type),
        Some(LiteralType::String("margin".to_string()))
    );
    assert_eq!(param(&evaluations[0], "scoreCol"), None);
}
