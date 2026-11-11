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

//! The Spark ML of the async API against a Spark Connect server at `SPARK_REMOTE`; each test
//! passes without running when it is not set.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::FutureExt;
use spark_connect_async::ml::{
    BinaryClassificationEvaluator, CrossValidator, Estimator, Evaluator, LogisticRegression,
    LogisticRegressionModel, MulticlassClassificationEvaluator, Pipeline, StandardScaler,
    StandardScalerModel, Transformer, Vector, VectorAssembler,
};
use spark_connect_async::types::DataType;
use spark_connect_async::{DataFrame, Session, Value};
use spark_connect_proto as proto;
use tokio::time::timeout;

/// The longest that a step waits for the server.
const WAIT: Duration = Duration::from_secs(120);

/// The connection string of the server, and that of a session of the server of its own.
fn remote() -> Option<(String, String)> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let id = format!(
        "00000000-0000-4000-8000-{:012x}",
        now.as_nanos() & 0xffff_ffff_ffff
    );
    let url = format!("{remote}/;session_id={id};user_id=e2e_ml");
    Some((remote, url))
}

/// Runs `body`, then stops `session`, which removes what the server keeps for it, also when
/// `body` panics.
async fn cleaned_up(session: &Session, body: impl Future<Output = ()>) {
    let outcome = AssertUnwindSafe(body).catch_unwind().await;
    let _ = timeout(WAIT, session.stop()).await;
    if let Err(panic) = outcome {
        std::panic::resume_unwind(panic);
    }
}

/// Rows of two features, `a` and `b`, whose `label` is 1 when `a` is 10 or more.
async fn training(session: &Session) -> DataFrame {
    let schema = DataType::from_ddl("a DOUBLE, b DOUBLE, label DOUBLE").unwrap();
    let rows = (0..20).map(|i| {
        vec![
            Value::Double(f64::from(i)),
            Value::Double(f64::from(i % 3)),
            Value::Double(if i >= 10 { 1.0 } else { 0.0 }),
        ]
    });
    timeout(WAIT, session.create_dataframe(rows, schema))
        .await
        .unwrap()
        .unwrap()
}

/// The ids of the models that the plan of `dataset` refers to, as the server keeps them.
fn models_of(dataset: &DataFrame) -> Vec<String> {
    let text = format!("{:?}", dataset.to_relation());
    text.split("ObjRef(ObjectRef { id: \"")
        .skip(1)
        .map(|rest| rest.split('"').next().unwrap().to_string())
        .collect()
}

/// Whether the server of the session of `url` keeps the model `id`, asked through the old API,
/// which sends the commands of Spark ML as they are.
async fn kept(url: String, id: String) -> bool {
    tokio::task::spawn_blocking(move || {
        let session = spark_connect::SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        let fetch = proto::Command {
            command_type: Some(spark_connect_core::ml::fetch(
                proto::ObjectRef { id },
                "toString",
            )),
        };
        match session.execute_command_proto(fetch) {
            Ok(_) => true,
            Err(error) if error.to_string().contains("CONNECT_ML.CACHE_INVALID") => false,
            Err(error) => panic!("{error}"),
        }
    })
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fitted_pipeline_predicts_and_evaluates_and_its_dropped_models_are_deleted() {
    let Some((_, url)) = remote() else {
        return;
    };
    let session = Session::builder()
        .remote(url.clone())
        .connect()
        .await
        .unwrap();
    cleaned_up(&session, async {
        let data = training(&session).await;
        let pipeline = Pipeline::new()
            .stage(
                VectorAssembler::new()
                    .input_cols(["a", "b"])
                    .output_col("raw"),
            )
            .stage(
                StandardScaler::new()
                    .input_col("raw")
                    .output_col("features"),
            )
            .stage(LogisticRegression::new().max_iter(50));
        let model = timeout(WAIT, pipeline.fit(&data)).await.unwrap().unwrap();
        let predictions = model.transform(&data);
        let rows: Vec<(f64, f64, Vector)> = timeout(
            WAIT,
            predictions
                .select(["label", "prediction", "probability"])
                .collect_as(),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(rows.len(), 20);
        for (label, prediction, probability) in &rows {
            assert_eq!(prediction, label);
            let probability = probability.to_dense();
            assert_eq!(probability.len(), 2);
            assert!((probability[0] + probability[1] - 1.0).abs() < 1e-9);
            assert!(probability[*label as usize] > 0.5, "{probability:?}");
        }
        let area = BinaryClassificationEvaluator::new().evaluate(&predictions);
        assert_eq!(timeout(WAIT, area).await.unwrap().unwrap(), 1.0);
        let accuracy = MulticlassClassificationEvaluator::new()
            .metric_name("accuracy")
            .evaluate(&predictions);
        assert_eq!(timeout(WAIT, accuracy).await.unwrap().unwrap(), 1.0);

        let scaler = model.stage::<StandardScalerModel>(1).unwrap();
        let mean = timeout(WAIT, scaler.mean()).await.unwrap().unwrap();
        assert_eq!(mean.to_dense(), [9.5, 0.95]);
        let regression = model.stage::<LogisticRegressionModel>(2).unwrap();
        let coefficients = timeout(WAIT, regression.coefficients())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(coefficients.size(), 2);
        assert!(coefficients.to_dense()[0] > 0.0, "{coefficients:?}");
        let intercept = timeout(WAIT, regression.intercept())
            .await
            .unwrap()
            .unwrap();
        assert!(intercept.is_finite());
        assert_eq!(
            timeout(WAIT, regression.num_classes())
                .await
                .unwrap()
                .unwrap(),
            2
        );
        assert_eq!(
            timeout(WAIT, regression.num_features())
                .await
                .unwrap()
                .unwrap(),
            2
        );
        let matrix = timeout(WAIT, regression.coefficient_matrix())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(
                matrix,
                spark_connect_async::ml::Matrix::Dense {
                    num_rows: 1,
                    num_cols: 2,
                    ..
                }
            ),
            "{matrix:?}"
        );

        let ids = models_of(&predictions);
        assert_eq!(ids.len(), 2, "{ids:?}");
        for id in &ids {
            assert!(kept(url.clone(), id.clone()).await, "{id} is not kept");
        }
        drop(predictions);
        drop(model);
        for id in ids {
            let deleted = async {
                while kept(url.clone(), id.clone()).await {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            };
            timeout(WAIT, deleted).await.expect("not deleted");
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cross_validation_of_pipelines_refits_the_candidate_of_the_best_metric() {
    let Some((_, url)) = remote() else {
        return;
    };
    let session = Session::builder().remote(url).connect().await.unwrap();
    cleaned_up(&session, async {
        let data = training(&session).await;
        let candidate = |iterations| {
            Pipeline::new()
                .stage(
                    VectorAssembler::new()
                        .input_cols(["a", "b"])
                        .output_col("features"),
                )
                .stage(LogisticRegression::new().max_iter(iterations))
        };
        let loss = MulticlassClassificationEvaluator::new().metric_name("logLoss");
        let validation = CrossValidator::new([candidate(1), candidate(50)], loss)
            .num_folds(2)
            .seed(3);
        let model = timeout(WAIT, validation.fit(&data)).await.unwrap().unwrap();
        let metrics = [model.avg_metrics(), model.std_metrics()];
        assert!(
            metrics
                .iter()
                .all(|metrics| metrics.len() == 2 && metrics.iter().all(|m| m.is_finite())),
            "{metrics:?}"
        );
        let best = model
            .best_model()
            .stage::<LogisticRegressionModel>(1)
            .unwrap();
        assert_eq!(
            timeout(WAIT, best.num_features()).await.unwrap().unwrap(),
            2
        );
        let predictions: Vec<(f64,)> = timeout(
            WAIT,
            model.transform(&data).select(["prediction"]).collect_as(),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(predictions.len(), 20);
        assert!(predictions.iter().all(|(p,)| *p == 0.0 || *p == 1.0));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_fit_is_an_error_with_the_condition_of_the_server() {
    let Some((_, url)) = remote() else {
        return;
    };
    let session = Session::builder().remote(url).connect().await.unwrap();
    cleaned_up(&session, async {
        let data = training(&session).await;
        let fit = LogisticRegression::new().features_col("missing").fit(&data);
        let error = timeout(WAIT, fit).await.unwrap().unwrap_err();
        assert_eq!(error.error_class(), Some("FIELD_NOT_FOUND"), "{error}");
        assert_eq!(error.sql_state().as_deref(), Some("42704"), "{error}");
        assert!(!error.message().contains("Exception"), "{error}");
    })
    .await;
}
