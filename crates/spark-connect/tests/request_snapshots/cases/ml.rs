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
    BinaryClassificationEvaluator, CrossValidator, Estimator, Evaluator, LinearRegression,
    LinearSVC, LogisticRegression, MaxAbsScaler, MlOperator, Model,
    MulticlassClassificationEvaluator, OperatorType, Params, Pipeline, PipelineStage,
    RegressionEvaluator, StandardScaler, StringIndexer, Transformer, VectorAssembler,
};

use prost::Message;

use super::range;
use crate::{Case, Ctx};

const PATH: &str = "/tmp/spark-connect-snapshots/ml";

fn fit(c: &Ctx, estimator: &mut dyn Estimator) -> spark_connect::Result<()> {
    let model = estimator.fit(&range(c))?;
    record_fit(c, estimator, model)
}

/// As `fit`, for an estimator whose model the client builds, of a new uid.
fn fit_client_model(c: &Ctx, estimator: &mut dyn Estimator) -> spark_connect::Result<()> {
    let model = estimator.fit(&range(c))?;
    c.mask(&model.operator().uid);
    record_fit(c, estimator, model)
}

fn record_fit(
    c: &Ctx,
    estimator: &dyn Estimator,
    mut model: Box<dyn Model>,
) -> spark_connect::Result<()> {
    c.mask(&estimator.operator().uid);
    c.message("MlOperator", &model.operator().to_proto());
    c.message("MlParams", &model.params().to_proto());
    c.df(&model.transform(&range(c))?)
}

/// As `fit`, for an estimator whose model's server copy is deleted once its last owner is dropped:
/// that request is sent in the background, so wait for it.
fn fit_native(c: &Ctx, estimator: &mut dyn Estimator) -> spark_connect::Result<()> {
    fit(c, estimator)?;
    c.wait_for(|entry| {
        entry.kind == "ExecutePlanRequest"
            && matches!(
                spark_connect_proto::ExecutePlanRequest::decode(&entry.bytes[..])
                    .ok()
                    .and_then(|request| request.plan?.op_type),
                Some(spark_connect_proto::plan::OpType::Command(
                    spark_connect_proto::Command {
                        command_type: Some(spark_connect_proto::command::CommandType::MlCommand(
                            spark_connect_proto::MlCommand {
                                command: Some(spark_connect_proto::ml_command::Command::Delete(_)),
                            }
                        )),
                    }
                ))
            )
    });
    Ok(())
}

/// Records whether `name`, a call whose result the case does not use, succeeded.
pub fn outcome<T>(c: &Ctx, name: &str, result: spark_connect::Result<T>) {
    let text = match result {
        Ok(_) => format!("{name}: ok"),
        Err(error) => format!("{name}: error: {error}"),
    };
    c.message("Text", &text);
}

/// The estimator and the evaluator of a validator, whose uids are masked. One fit at a time, so
/// that the requests of the folds come in one order. A validator's fit drops the model of each
/// fold, whose delete the client sends in the background, so its case records them unordered.
fn validated_stages(c: &Ctx) -> (LogisticRegression, BinaryClassificationEvaluator) {
    let estimator = LogisticRegression::new();
    c.mask(&Estimator::operator(&estimator).uid);
    let evaluator = BinaryClassificationEvaluator::new();
    c.mask(&evaluator.operator().uid);
    (estimator, evaluator)
}

/// A validator of the stages of `validated_stages` and one param map, whose uid is masked.
fn cross_validator(c: &Ctx) -> CrossValidator {
    let (estimator, evaluator) = validated_stages(c);
    let validator = CrossValidator::new()
        .set_estimator(estimator)
        .set_evaluator(evaluator)
        .set_estimator_param_maps(vec![spark_connect::ml::ParamMap::new()])
        .set_parallelism(1)
        .set_seed(9);
    c.mask(validator.uid());
    validator
}

/// As `cross_validator`, of a train-validation split.
fn train_validation_split(c: &Ctx) -> spark_connect::ml::TrainValidationSplit {
    let (estimator, evaluator) = validated_stages(c);
    let validator = spark_connect::ml::TrainValidationSplit::new()
        .set_estimator(estimator)
        .set_evaluator(evaluator)
        .set_estimator_param_maps(vec![spark_connect::ml::ParamMap::new()])
        .set_parallelism(1)
        .set_seed(9);
    c.mask(validator.uid());
    validator
}

/// The pipeline of the `Pipeline::fit` case, of an assembler and an estimator, with masked uids.
fn assembled_pipeline(c: &Ctx) -> Pipeline {
    let pipeline = Pipeline::new().set_stages([
        PipelineStage::transformer(
            VectorAssembler::new()
                .set_input_cols(["id", "value"])
                .set_output_col("features"),
        ),
        PipelineStage::estimator(LogisticRegression::new()),
    ]);
    for stage in pipeline.stages() {
        c.mask(stage.uid());
    }
    c.mask(pipeline.uid());
    pipeline
}

fn literal(
    literal_type: spark_connect_proto::expression::literal::LiteralType,
) -> spark_connect_proto::expression::Literal {
    spark_connect_proto::expression::Literal {
        literal_type: Some(literal_type),
        ..Default::default()
    }
}

fn evaluate(c: &Ctx, evaluator: &dyn Evaluator) -> spark_connect::Result<()> {
    c.mask(&evaluator.operator().uid);
    c.message("MlParams", &evaluator.params().to_proto());
    evaluator.evaluate(&range(c)).map(drop)
}

pub fn cases() -> Vec<Case> {
    cases![
        "Summarizer::mean" => |c| {
            use spark_connect::column::col;
            use spark_connect::ml::{Summarizer, SummaryBuilder};
            let weight = || Some(col("w"));
            c.columns(&[
                Summarizer::mean(col("f"), weight()),
                Summarizer::sum(col("f"), None),
                Summarizer::variance(col("f"), weight()),
                Summarizer::std(col("f"), None),
                Summarizer::count(col("f"), weight()),
                Summarizer::num_non_zeros(col("f"), None),
                Summarizer::max(col("f"), weight()),
                Summarizer::min(col("f"), None),
                Summarizer::norm_l1(col("f"), weight()),
                Summarizer::norm_l2(col("f"), None),
                Summarizer::metrics(["mean", "max"]).summary(col("f"), weight()),
                SummaryBuilder::from_metric_columns([spark_connect::column::lit_string("min")]).summary(col("f"), None),
            ]);
            Ok(())
        },
        "ChiSquareTest::test" => |c| c.df(&spark_connect::ml::ChiSquareTest::test(&range(c), "f", "l", true)?),
        "Correlation::corr" => |c| c.df(&spark_connect::ml::Correlation::corr(&range(c), "f", "spearman")?),
        "KolmogorovSmirnovTest::test" => {
            |c| c.df(&spark_connect::ml::KolmogorovSmirnovTest::test(&range(c), "x", "norm", [0.5, 2.0])?)
        },
        "OperatorEstimator::new" => |c| {
            let operator = MlOperator::new("org.apache.spark.ml.feature.StandardScaler", OperatorType::Estimator);
            fit(c, &mut spark_connect::ml::OperatorEstimator::new(operator, Params::new().set_param_bool("withMean", true)))
        },
        "OperatorTransformer::new" => |c| {
            let operator = MlOperator::new("org.apache.spark.ml.feature.Binarizer", OperatorType::Transformer);
            let mut transformer = spark_connect::ml::OperatorTransformer::new(operator, Params::new().set_param_double("threshold", 0.5));
            c.mask(&transformer.operator().uid);
            c.df(&transformer.transform(&range(c))?)
        },
        "MlWriter::save" => |c| {
            let operator = MlOperator::new("org.apache.spark.ml.feature.Binarizer", OperatorType::Transformer);
            c.mask(&operator.uid);
            spark_connect::ml::MlWriter::operator(&operator, &Params::new().set_param_double("threshold", 0.5))
                .option("Mode", "x")
                .overwrite()
                .session(&c.spark)
                .save(PATH)
        },
        "meta::load_stage" => |c| spark_connect::ml::load_stage(&c.spark, PATH).map(drop),
        "ml::ml_helper_attribute" => |c| spark_connect::ml::ml_helper_attribute(&c.spark, "x", Vec::new()).map(drop),
        "tuning::k_fold" => |c| {
            for (training, validation) in spark_connect::ml::k_fold(&range(c), 2, "", Some(3), "CrossValidator_000000000000")? {
                c.df(&training)?;
                c.df(&validation)?;
            }
            Ok(())
        },
        "tuning::train_validation_split" => |c| {
            let (training, validation) = spark_connect::ml::train_validation_split(&range(c), 0.75, Some(3), "TrainValidationSplit_000000000000");
            c.df(&training)?;
            c.df(&validation)
        },
        "object::attribute_relation" => |c| {
            let target = spark_connect::ml::MlTarget::helper(&c.spark);
            c.df(&spark_connect::ml::attribute_relation(&target, "x", Vec::new())?)
        },
        "object::evaluated_summary" => |c| {
            let target = spark_connect::ml::MlTarget::helper(&c.spark);
            let output = spark_connect::ml::fetch(&target, "evaluate", Vec::new())?;
            spark_connect::ml::evaluated_summary(&target, output).map(drop)
        },
        "object::write_object" => |c| {
            let mut estimator = spark_connect::ml::KMeans::new();
            let model = estimator.fit(&range(c))?;
            c.mask(&Estimator::operator(&estimator).uid);
            let Some(object) = Model::ml_ref(&model) else {
                return Err(spark_connect::SparkError::value_msg("no object"));
            };
            spark_connect::ml::write_object(object, &Params::new(), PATH, true, &std::collections::HashMap::new())
        },
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
            c.message("Text", &format!("{} {}", scaler.input_col().unwrap_or_default(), scaler.output_col()));
            fit_native(c, &mut scaler)
        },
        "StandardScaler::default" => |c| fit_native(c, &mut StandardScaler::default()),
        "MaxAbsScaler::fit" => |c| {
            let mut scaler = MaxAbsScaler::new().set_input_col("f").set_output_col("o");
            c.message("Text", &format!("{} {}", scaler.input_col().unwrap_or_default(), scaler.output_col()));
            fit_native(c, &mut scaler)
        },
        "MaxAbsScaler::default" => |c| fit_native(c, &mut MaxAbsScaler::default()),
        "StringIndexer::fit" => |c| {
            let mut indexer = StringIndexer::new().set_input_col("s").set_output_col("i");
            c.message("Text", &format!("{} {}", indexer.input_col().unwrap_or_default(), indexer.output_col()));
            fit_native(c, &mut indexer)
        },
        "StringIndexer::default" => |c| fit_native(c, &mut StringIndexer::default()),
        "LogisticRegression::fit" => |c| {
            let mut lr = LogisticRegression::new()
                .set_features_col("f")
                .set_label_col("l")
                .set_prediction_col("p")
                .set_max_iter(7);
            c.message("Text", &format!("{} {} {} {}", lr.features_col(), lr.label_col(), lr.prediction_col(), lr.max_iter()));
            fit_native(c, &mut lr)
        },
        "LogisticRegression::default" => |c| fit_native(c, &mut LogisticRegression::default()),
        "Pipeline::fit" => |c| {
            let mut pipeline = Pipeline::new().set_stages([
                PipelineStage::transformer(VectorAssembler::new().set_input_cols(["id", "value"]).set_output_col("features")),
                PipelineStage::estimator(LogisticRegression::new()),
            ]);
            for stage in pipeline.stages() {
                c.mask(stage.uid());
            }
            c.message("Text", &format!("{:?}", pipeline.stages().iter().map(PipelineStage::is_estimator).collect::<Vec<_>>()));
            fit_client_model(c, &mut pipeline)
        },
        "Pipeline::default" => |c| fit_client_model(c, &mut Pipeline::default()),
        "CrossValidator::fit" => |c| {
            c.unordered();
            let (estimator, evaluator) = validated_stages(c);
            let mut cv = CrossValidator::new()
                .set_estimator(estimator)
                .set_evaluator(evaluator)
                .set_estimator_param_maps(vec![spark_connect::ml::ParamMap::new()])
                .set_num_folds(4)
                .set_parallelism(1)
                .set_seed(9);
            c.message("Text", &format!("{} {}", cv.num_folds(), cv.parallelism()));
            fit_client_model(c, &mut cv)
        },
        "TrainValidationSplit::fit" => |c| {
            c.unordered();
            let (estimator, evaluator) = validated_stages(c);
            let mut tvs = spark_connect::ml::TrainValidationSplit::new()
                .set_estimator(estimator)
                .set_evaluator(evaluator)
                .set_estimator_param_maps(vec![spark_connect::ml::ParamMap::new()])
                .set_train_ratio(0.75)
                .set_parallelism(1)
                .set_seed(9);
            fit_client_model(c, &mut tvs)
        },
        "CrossValidator::default" => |c| fit(c, &mut CrossValidator::default()),
        "VectorAssembler::transform" => |c| {
            let mut assembler = VectorAssembler::new().set_input_cols(vec!["a", "b"]).set_output_col("v");
            c.message("Text", &format!("{:?} {}", assembler.input_cols(), assembler.output_col()));
            let df = assembler.transform(&range(c))?;
            c.mask(&assembler.operator().uid);
            c.message("MlParams", &assembler.params().to_proto());
            c.message("MlRelation", &assembler.build_ml_relation(&range(c))?);
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
            let evaluator = BinaryClassificationEvaluator::new().set_label_col("l").set_raw_prediction_col("s").set_metric_name("areaUnderPR");
            c.message("Text", &format!("{} {} {}", evaluator.label_col(), evaluator.raw_prediction_col(), evaluator.metric_name()));
            evaluate(c, &evaluator)
        },
        "BinaryClassificationEvaluator::default" => |c| evaluate(c, &BinaryClassificationEvaluator::default()),
        "MulticlassClassificationEvaluator::evaluate" => |c| {
            let evaluator = MulticlassClassificationEvaluator::new().set_label_col("l").set_prediction_col("p").set_metric_name("f1");
            c.message("Text", &format!("{} {} {}", evaluator.label_col(), evaluator.prediction_col(), evaluator.metric_name()));
            evaluate(c, &evaluator)
        },
        "MulticlassClassificationEvaluator::default" => |c| evaluate(c, &MulticlassClassificationEvaluator::default()),
        "LinearSVC::fit" => |c| {
            let mut svc = LinearSVC::new()
                .set_features_col("f")
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
                .set_max_block_size_in_mb(1.5);
            fit_native(c, &mut svc)
        },
        "LinearSVC::default" => |c| fit_native(c, &mut LinearSVC::default()),
        "LinearRegression::fit" => |c| {
            let mut regression = LinearRegression::new()
                .set_features_col("f")
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
                .set_max_block_size_in_mb(1.5);
            fit_native(c, &mut regression)
        },
        "LinearRegression::default" => |c| fit_native(c, &mut LinearRegression::default()),
        "LogisticRegression::fit_multiple" => |c| {
            let estimator = LogisticRegression::new();
            c.mask(&Estimator::operator(&estimator).uid);
            let models = estimator.fit_multiple(&range(c), &[Params::new().set_param_int("maxIter", 3), Params::new()])?;
            c.message("Text", &models.len().to_string());
            Ok(())
        },
        "LinearSVC::fit_multiple" => |c| {
            let estimator = LinearSVC::new();
            c.mask(&Estimator::operator(&estimator).uid);
            let models = estimator.fit_multiple(&range(c), &[Params::new().set_param_int("maxIter", 3), Params::new()])?;
            c.message("Text", &models.len().to_string());
            Ok(())
        },
        "CrossValidator::<Estimator>::write" => |c| {
            Estimator::write(&cross_validator(c)).session(&c.spark).overwrite().save(PATH)
        },
        "CrossValidatorModel::<Model>::write" => |c| {
            c.unordered();
            let model = cross_validator(c).fit_validator(&range(c))?;
            c.mask(&Transformer::operator(&model).uid);
            Model::write(&model)?.session(&c.spark).overwrite().save(PATH)
        },
        "TrainValidationSplit::<Estimator>::write" => |c| {
            Estimator::write(&train_validation_split(c)).session(&c.spark).overwrite().save(PATH)
        },
        "TrainValidationSplitModel::<Model>::write" => |c| {
            c.unordered();
            let model = train_validation_split(c).fit_validator(&range(c))?;
            c.mask(&Transformer::operator(&model).uid);
            Model::write(&model)?.session(&c.spark).overwrite().save(PATH)
        },
        "Pipeline::<Estimator>::write" => |c| {
            Estimator::write(&assembled_pipeline(c)).session(&c.spark).overwrite().save(PATH)
        },
        "PipelineModel::<Model>::write" => |c| {
            let model = assembled_pipeline(c).fit_pipeline(&range(c))?;
            c.mask(model.uid());
            Model::write(&model)?.session(&c.spark).overwrite().save(PATH)
        },
        "PipelineModel::of_stages" => |c| {
            let mut estimator = LogisticRegression::new();
            c.mask(&Estimator::operator(&estimator).uid);
            let fitted = estimator.fit(&range(c))?;
            let mut assembler = PipelineStage::transformer(VectorAssembler::new().set_input_cols(["id", "value"]).set_output_col("features"));
            c.mask(assembler.uid());
            c.df(&assembler.transform(&range(c))?)?;
            let stages = vec![assembler, PipelineStage::model(Box::new(fitted))];
            for stage in &stages {
                outcome(c, "write", stage.write().and_then(|writer| writer.session(&c.spark).overwrite().save(PATH)));
            }
            let mut model = spark_connect::ml::PipelineModel::of_stages(stages);
            c.mask(model.uid());
            c.df(&Transformer::transform(&mut model, &range(c))?)
        },
        "OneVsRestModel::<Model>::write" => |c| {
            let classifier = LogisticRegression::new();
            c.mask(&Estimator::operator(&classifier).uid);
            let mut estimator = spark_connect::ml::OneVsRest::new().set_classifier(classifier).set_parallelism(1);
            c.mask(&Estimator::operator(&estimator).uid);
            let model = estimator.fit(&range(c))?;
            c.mask(&Transformer::operator(&model).uid);
            Model::write(&model)?.session(&c.spark).overwrite().save(PATH)
        },
        "MlWriter::object" => |c| {
            let mut estimator = spark_connect::ml::KMeans::new();
            c.mask(&Estimator::operator(&estimator).uid);
            let model = estimator.fit(&range(c))?;
            let Some(object) = Model::ml_ref(&model) else {
                return Err(spark_connect::SparkError::value_msg("no object"));
            };
            spark_connect::ml::MlWriter::object(object, &Params::new().set_param_int("k", 3)).overwrite().save(PATH)
        },
        "CountVectorizerModel::from_vocabulary" => |c| {
            let mut model = spark_connect::ml::CountVectorizerModel::from_vocabulary(&c.spark, ["a", "b"], "x", Some("y"), Some(1.0), Some(true))?;
            c.mask(&Transformer::operator(&model).uid);
            c.df(&Transformer::transform(&mut model, &range(c))?)
        },
        "StringIndexerModel::from_labels" => |c| {
            let mut model = spark_connect::ml::StringIndexerModel::from_labels(&c.spark, ["a", "b"], "x", Some("y"), Some("keep"))?;
            c.mask(&Transformer::operator(&model).uid);
            c.df(&Transformer::transform(&mut model, &range(c))?)
        },
        "StringIndexerModel::from_arrays_of_labels" => |c| {
            let outputs = Some(vec!["i".to_string(), "j".to_string()]);
            let mut model = spark_connect::ml::StringIndexerModel::from_arrays_of_labels(&c.spark, [["a", "b"], ["c", "d"]], ["x", "y"], outputs, Some("keep"))?;
            c.mask(&Transformer::operator(&model).uid);
            c.df(&Transformer::transform(&mut model, &range(c))?)
        },
        "StopWordsRemover::load_default_stop_words" => |c| {
            spark_connect::ml::StopWordsRemover::load_default_stop_words(&c.spark, "english").map(drop)
        },
        "QuantileDiscretizer::fit_bucketizer" => |c| {
            let mut estimator = spark_connect::ml::QuantileDiscretizer::new().set_input_col("x").set_output_col("y");
            c.mask(&Estimator::operator(&estimator).uid);
            let mut bucketizer = estimator.fit_bucketizer(&range(c))?;
            c.mask(&Transformer::operator(&bucketizer).uid);
            c.df(&Transformer::transform(&mut bucketizer, &range(c))?)
        },
        "LDAModel::into_local" => |c| {
            let mut estimator = spark_connect::ml::LDA::new().set_optimizer("online");
            c.mask(&Estimator::operator(&estimator).uid);
            let Some(mut model) = estimator.fit(&range(c))?.into_local() else {
                return Err(spark_connect::SparkError::value_msg("no local model"));
            };
            c.df(&Transformer::transform(&mut model, &range(c))?)
        },
        "LDAModel::into_distributed" => |c| {
            let mut estimator = spark_connect::ml::LDA::new().set_optimizer("em");
            c.mask(&Estimator::operator(&estimator).uid);
            let Some(mut model) = estimator.fit(&range(c))?.into_distributed() else {
                return Err(spark_connect::SparkError::value_msg("no distributed model"));
            };
            c.df(&Transformer::transform(&mut model, &range(c))?)
        },
        "ALSModel::recommend_with_literal" => |c| {
            use spark_connect_proto::expression::literal::LiteralType;
            let mut estimator = spark_connect::ml::ALS::new();
            c.mask(&Estimator::operator(&estimator).uid);
            let model = estimator.fit(&range(c))?;
            c.df(&model.recommend_with_literal(true, None, literal(LiteralType::Integer(3)))?)?;
            c.df(&model.recommend_with_literal(false, Some(&range(c)), literal(LiteralType::Long(2)))?)
        },
        "Params::to_proto/typed_setters" => |c| {
            use spark_connect_proto::expression::literal::LiteralType;
            let matrix = spark_connect::ml::MlMatrix::Dense { num_rows: 1, num_cols: 2, values: vec![0.5, 0.25], is_transposed: false };
            let params = Params::new()
                .set_param_int32("int32", 3)
                .set_param_int_narrow("narrow", 3)
                .set_param_int32_array("int32s", [1, 2])
                .set_param_int_array("longs", [1, 2])
                .set_param_double_array("doubles", [0.5, 0.25])
                .set_param_double_arrays("splits", [vec![0.5, 0.25], vec![1.0]])
                .set_param_string_array("strings", ["a", "b"])
                .set_param_vector("vector", vec![0.5, 0.25])
                .set_param_matrix("matrix", matrix)
                .set_param_literal("literal", literal(LiteralType::Short(3)))
                .set_param_null("null")
                .set_param_string("cleared", "x")
                .clear_param("cleared");
            c.message("MlParams", &params.to_proto());
            c.message("MlParams", &params.retain_names(|name| name.starts_with("int")).to_proto());
            Ok(())
        },
    ]
}
