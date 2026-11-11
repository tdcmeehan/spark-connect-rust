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

//! The models of Spark ML on a server that answers its commands as Spark does: what their
//! attributes read, and when the server is told to delete them.

mod gated;

use std::fmt::Debug;
use std::future::Future;

use futures::future::BoxFuture;
use gated::{Received, Server, Wait};
use spark_connect_async::ml::{
    Estimator, Evaluator, LinearRegression, LinearSVC, LogisticRegression, MaxAbsScaler,
    MulticlassClassificationEvaluator, StandardScaler, StandardScalerModel, StringIndexer,
    Transformer,
};
use spark_connect_async::{ErrorKind, Result, Session};

async fn connect(server: &Server) -> Session {
    Session::builder()
        .remote(server.url())
        .connect()
        .await
        .unwrap()
}

fn text<T: Debug>(
    value: impl Future<Output = Result<T>> + Send + 'static,
) -> BoxFuture<'static, String> {
    Box::pin(async move { format!("{:?}", value.await.unwrap()) })
}

fn deletes(server: &Server) -> Vec<Vec<String>> {
    server
        .received()
        .into_iter()
        .filter_map(|received| match received {
            Received::Delete(ids) => Some(ids),
            _ => None,
        })
        .collect()
}

fn executions(server: &Server) -> usize {
    server
        .received()
        .iter()
        .filter(|received| matches!(received, Received::Execute(_)))
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_attribute_reads_the_value_of_its_type_that_the_server_sends() {
    let server = Server::start();
    let session = connect(&server).await;
    let data = session.range(3);
    let scaler = StandardScaler::new().fit(&data).await.unwrap();
    let max_abs = MaxAbsScaler::new().fit(&data).await.unwrap();
    let indexer = StringIndexer::new().fit(&data).await.unwrap();
    let logistic = LogisticRegression::new().fit(&data).await.unwrap();
    let svc = LinearSVC::new().fit(&data).await.unwrap();
    let linear = LinearRegression::new().fit(&data).await.unwrap();
    let vector = "Dense([1.0, 2.0])";
    let matrix = "Dense { num_rows: 1, num_cols: 2, values: [1.0, 2.0], is_transposed: true }";
    let attributes: Vec<(&str, &str, BoxFuture<'static, String>, &str)> = vec![
        (
            "StandardScalerModel::mean",
            "mean",
            text(scaler.mean()),
            vector,
        ),
        (
            "StandardScalerModel::std",
            "std",
            text(scaler.std()),
            vector,
        ),
        (
            "MaxAbsScalerModel::max_abs",
            "maxAbs",
            text(max_abs.max_abs()),
            vector,
        ),
        (
            "StringIndexerModel::labels",
            "labelsArray",
            text(indexer.labels()),
            r#"["a", "b"]"#,
        ),
        (
            "LogisticRegressionModel::coefficients",
            "coefficients",
            text(logistic.coefficients()),
            vector,
        ),
        (
            "LogisticRegressionModel::intercept",
            "intercept",
            text(logistic.intercept()),
            "0.5",
        ),
        (
            "LogisticRegressionModel::coefficient_matrix",
            "coefficientMatrix",
            text(logistic.coefficient_matrix()),
            matrix,
        ),
        (
            "LogisticRegressionModel::intercept_vector",
            "interceptVector",
            text(logistic.intercept_vector()),
            vector,
        ),
        (
            "LogisticRegressionModel::num_classes",
            "numClasses",
            text(logistic.num_classes()),
            "2",
        ),
        (
            "LogisticRegressionModel::num_features",
            "numFeatures",
            text(logistic.num_features()),
            "2",
        ),
        (
            "LinearSVCModel::coefficients",
            "coefficients",
            text(svc.coefficients()),
            vector,
        ),
        (
            "LinearSVCModel::intercept",
            "intercept",
            text(svc.intercept()),
            "0.5",
        ),
        (
            "LinearSVCModel::num_classes",
            "numClasses",
            text(svc.num_classes()),
            "2",
        ),
        (
            "LinearSVCModel::num_features",
            "numFeatures",
            text(svc.num_features()),
            "2",
        ),
        (
            "LinearRegressionModel::coefficients",
            "coefficients",
            text(linear.coefficients()),
            vector,
        ),
        (
            "LinearRegressionModel::intercept",
            "intercept",
            text(linear.intercept()),
            "0.5",
        ),
        (
            "LinearRegressionModel::scale",
            "scale",
            text(linear.scale()),
            "0.5",
        ),
        (
            "LinearRegressionModel::num_features",
            "numFeatures",
            text(linear.num_features()),
            "2",
        ),
    ];
    for (attribute, method, value, expected) in attributes {
        let before = server.received().len();
        assert_eq!(value.await, expected, "{attribute}");
        let fetched: Vec<_> = server.received()[before..]
            .iter()
            .filter_map(|received| match received {
                Received::Fetch(method) => Some(method.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(fetched, [method], "{attribute}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fit_an_evaluation_or_an_attribute_without_its_answer_is_a_decode_error() {
    let server = Server::start();
    let session = connect(&server).await;
    let data = session.range(3);
    let model = LogisticRegression::new().fit(&data).await.unwrap();
    server.wait_at(Wait::BeforeReattaching);
    let calls: [(&str, BoxFuture<'static, Result<()>>); 3] = [
        (
            "a fit",
            Box::pin(async move { LogisticRegression::new().fit(&data).await.map(drop) }),
        ),
        ("an evaluation", {
            let data = session.range(3);
            Box::pin(async move {
                MulticlassClassificationEvaluator::new()
                    .evaluate(&data)
                    .await
                    .map(drop)
            })
        }),
        (
            "an attribute",
            Box::pin({
                let coefficients = model.coefficients();
                async move { coefficients.await.map(drop) }
            }),
        ),
    ];
    for (call, run) in calls {
        server.open_gate();
        let error = run.await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Decode, "{call}: {error}");
        if call == "an attribute" {
            assert!(error.to_string().contains("coefficients"), "{error}");
        }
    }
}

/// A runtime of `flavor`, a current-thread or a multi-thread one.
fn runtime(flavor: &str) -> tokio::runtime::Runtime {
    match flavor {
        "current-thread" => tokio::runtime::Builder::new_current_thread(),
        _ => tokio::runtime::Builder::new_multi_thread(),
    }
    .enable_all()
    .build()
    .unwrap()
}

#[test]
fn a_dropped_model_is_deleted_from_the_runtime_of_its_fit() {
    for flavor in ["current-thread", "multi-thread"] {
        for place in ["in its runtime", "outside a runtime", "in another runtime"] {
            let runtime = runtime(flavor);
            let server = Server::start();
            let (session, model) = runtime.block_on(async {
                let session = connect(&server).await;
                let model = StandardScaler::new().fit(&session.range(3)).await.unwrap();
                (session, model)
            });
            match place {
                "in its runtime" => runtime.block_on(async move { drop(model) }),
                "outside a runtime" => drop(model),
                _ => self::runtime("current-thread").block_on(async move { drop(model) }),
            }
            assert!(
                runtime.block_on(server.wait_for(|r| matches!(r, Received::Delete(_)))),
                "{flavor}, {place}: not deleted"
            );
            assert_eq!(deletes(&server), [["model-1"]], "{flavor}, {place}");
            runtime.block_on(async move { drop(session) });
        }
    }
}

/// What holds a model, and lets it go when it is called.
type Holder<'a> = Box<dyn FnOnce() + 'a>;

#[test]
fn a_model_is_deleted_once_when_the_last_of_what_holds_it_lets_it_go() {
    let holders = [
        "a clone",
        "a DataFrame that it transformed",
        "an attribute in flight",
        "an action in flight of a DataFrame that it transformed",
        "an evaluation in flight of a DataFrame that it transformed",
        "a fit in flight to a DataFrame that it transformed",
    ];
    for holder in holders {
        // On a current-thread runtime that the test does not drive, a spawned deletion stays a
        // live task of the runtime until the test drives it.
        let runtime = runtime("current-thread");
        let tasks = || runtime.metrics().num_alive_tasks();
        let server = Server::start();
        let (session, model): (Session, StandardScalerModel) = runtime.block_on(async {
            let session = connect(&server).await;
            let model = StandardScaler::new().fit(&session.range(3)).await.unwrap();
            (session, model)
        });
        let data = session.range(3);
        let in_flight = |call: BoxFuture<'static, ()>| -> Holder<'_> {
            server.wait_at(Wait::BeforeAccepting);
            let before = executions(&server);
            let call = runtime.spawn(call);
            assert!(
                runtime.block_on(async {
                    for _ in 0..500 {
                        if executions(&server) > before {
                            return true;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                    false
                }),
                "{holder}: not sent"
            );
            Box::new(|| {
                server.wait_at(Wait::Never);
                server.open_gate();
                let _ = runtime.block_on(call);
            })
        };
        let release: Holder<'_> = match holder {
            "a clone" => {
                let clone = model.clone();
                Box::new(move || drop(clone))
            }
            "a DataFrame that it transformed" => {
                let transformed = model.transform(&data);
                Box::new(move || drop(transformed))
            }
            "an attribute in flight" => {
                let mean = model.mean();
                in_flight(Box::pin(async move { drop(mean.await) }))
            }
            "an action in flight of a DataFrame that it transformed" => {
                let count = model.transform(&data).count();
                in_flight(Box::pin(async move { drop(count.await) }))
            }
            "an evaluation in flight of a DataFrame that it transformed" => {
                let evaluation =
                    MulticlassClassificationEvaluator::new().evaluate(&model.transform(&data));
                in_flight(Box::pin(async move { drop(evaluation.await) }))
            }
            _ => {
                let fit = StandardScaler::new().fit(&model.transform(&data));
                in_flight(Box::pin(async move { drop(fit.await) }))
            }
        };
        let before = tasks();
        drop(model);
        assert_eq!(tasks(), before, "{holder}: deleted while held");
        release();
        let deleted = Received::Delete(vec!["model-1".to_string()]);
        assert!(
            runtime.block_on(server.wait_for(|r| *r == deleted)),
            "{holder}: not deleted after"
        );
        // The fit in flight deletes its own model too.
        let once = deletes(&server)
            .iter()
            .filter(|ids| *ids == &["model-1"])
            .count();
        assert_eq!(once, 1, "{holder}");
        runtime.block_on(async move { drop(session) });
    }
}
