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

use spark_connect::ml::{
    BinaryClassificationEvaluator, CrossValidator, Estimator, Evaluator, LogisticRegression,
    MaxAbsScaler, MlOperator, MulticlassClassificationEvaluator, OperatorType, Params, Pipeline,
    RegressionEvaluator, StandardScaler, StringIndexer, Transformer, VectorAssembler,
};

use super::range;
use crate::{Case, Ctx};

fn fit(c: &Ctx, estimator: &mut dyn Estimator) -> spark_connect::Result<()> {
    let mut model = estimator.fit(&range(c))?;
    c.mask(&estimator.operator().uid);
    c.message("MlOperator", &model.operator().to_proto());
    c.message("MlParams", &model.params().to_proto());
    c.df(&model.transform(&range(c))?)
}

fn evaluate(c: &Ctx, evaluator: &dyn Evaluator) -> spark_connect::Result<()> {
    c.mask(&evaluator.operator().uid);
    c.message("MlParams", &evaluator.params().to_proto());
    evaluator.evaluate(&range(c)).map(drop)
}

pub fn cases() -> Vec<Case> {
    cases![
        "Params::to_proto" => |c| {
            let params = Params::new()
                .set_param_int("i", 3)
                .set_param_double("d", 0.5)
                .set_param_string("s", "x")
                .set_param_bool("b", true);
            assert!(params.get_param("i").is_some());
            c.message("MlParams", &params.to_proto());
            c.message("MlParams", &Params::from_proto(&params.to_proto()).to_proto());
            Ok(())
        },
        "MlOperator::to_proto" => |c| {
            for op_type in [OperatorType::Estimator, OperatorType::Transformer, OperatorType::Evaluator, OperatorType::Model] {
                let operator = MlOperator::with_uid("op", "uid-1", op_type);
                c.message("MlOperator", &operator.to_proto());
                c.message("MlOperator", &MlOperator::from_proto(&operator.to_proto()).to_proto());
                c.message("Text", &format!("{:?}", OperatorType::from_proto(op_type.to_proto() as i32)));
            }
            let generated = MlOperator::new("op", OperatorType::Model);
            c.mask(&generated.uid);
            c.message("MlOperator", &generated.to_proto());
            Ok(())
        },
        "StandardScaler::fit" => |c| {
            let mut scaler = StandardScaler::new().set_input_col("features").set_output_col("scaled");
            c.message("Text", &format!("{} {}", scaler.input_col(), scaler.output_col()));
            fit(c, &mut scaler)
        },
        "StandardScaler::default" => |c| fit(c, &mut StandardScaler::default()),
        "MaxAbsScaler::fit" => |c| {
            let mut scaler = MaxAbsScaler::new().set_input_col("f").set_output_col("o");
            c.message("Text", &format!("{} {}", scaler.input_col(), scaler.output_col()));
            fit(c, &mut scaler)
        },
        "MaxAbsScaler::default" => |c| fit(c, &mut MaxAbsScaler::default()),
        "StringIndexer::fit" => |c| {
            let mut indexer = StringIndexer::new().set_input_col("s").set_output_col("i");
            c.message("Text", &format!("{} {}", indexer.input_col(), indexer.output_col()));
            fit(c, &mut indexer)
        },
        "StringIndexer::default" => |c| fit(c, &mut StringIndexer::default()),
        "LogisticRegression::fit" => |c| {
            let mut lr = LogisticRegression::new()
                .set_feature_col("f")
                .set_label_col("l")
                .set_prediction_col("p")
                .set_max_iter(7);
            c.message("Text", &format!("{} {} {} {}", lr.feature_col(), lr.label_col(), lr.prediction_col(), lr.max_iter()));
            fit(c, &mut lr)
        },
        "LogisticRegression::default" => |c| fit(c, &mut LogisticRegression::default()),
        "Pipeline::fit" => |c| {
            let mut pipeline = Pipeline::new().set_stages(vec!["a", "b"]);
            c.message("Text", &format!("{:?}", pipeline.stages()));
            fit(c, &mut pipeline)
        },
        "Pipeline::default" => |c| fit(c, &mut Pipeline::default()),
        "CrossValidator::fit" => |c| {
            let mut cv = CrossValidator::new().set_num_folds(4).set_parallelism(2).set_seed(9);
            c.message("Text", &format!("{} {}", cv.num_folds(), cv.parallelism()));
            fit(c, &mut cv)
        },
        "CrossValidator::default" => |c| fit(c, &mut CrossValidator::default()),
        "VectorAssembler::transform" => |c| {
            let mut assembler = VectorAssembler::new().set_input_cols(vec!["a", "b"]).set_output_col("v");
            c.message("Text", &format!("{:?} {}", assembler.input_cols(), assembler.output_col()));
            let df = assembler.transform(&range(c))?;
            c.mask(&assembler.operator().uid);
            c.message("MlParams", &assembler.params().to_proto());
            c.message("MlRelation", &assembler.build_ml_relation(&range(c)));
            c.df(&df)
        },
        "VectorAssembler::default" => |c| {
            let mut assembler = VectorAssembler::default();
            let df = assembler.transform(&range(c))?;
            c.mask(&assembler.operator().uid);
            c.df(&df)
        },
        "RegressionEvaluator::evaluate" => |c| {
            let evaluator = RegressionEvaluator::new().set_label_col("l").set_prediction_col("p").set_metric_name("mae");
            c.message("Text", &format!("{} {} {}", evaluator.label_col(), evaluator.prediction_col(), evaluator.metric_name()));
            evaluate(c, &evaluator)
        },
        "RegressionEvaluator::default" => |c| evaluate(c, &RegressionEvaluator::default()),
        "BinaryClassificationEvaluator::evaluate" => |c| {
            let evaluator = BinaryClassificationEvaluator::new().set_label_col("l").set_score_col("s").set_metric_name("areaUnderPR");
            c.message("Text", &format!("{} {} {}", evaluator.label_col(), evaluator.score_col(), evaluator.metric_name()));
            evaluate(c, &evaluator)
        },
        "BinaryClassificationEvaluator::default" => |c| evaluate(c, &BinaryClassificationEvaluator::default()),
        "MulticlassClassificationEvaluator::evaluate" => |c| {
            let evaluator = MulticlassClassificationEvaluator::new().set_label_col("l").set_prediction_col("p").set_metric_name("f1");
            c.message("Text", &format!("{} {} {}", evaluator.label_col(), evaluator.prediction_col(), evaluator.metric_name()));
            evaluate(c, &evaluator)
        },
        "MulticlassClassificationEvaluator::default" => |c| evaluate(c, &MulticlassClassificationEvaluator::default()),
    ]
}
