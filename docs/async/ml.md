# Machine Learning and Pipelines

## Spark ML

`spark_connect_async::ml` has the estimators, transformers and evaluators of Spark ML that run
without Python, and the pipelines and cross validations that combine them. A fit and an
evaluation run on the server when they are awaited; a transform only builds the plan of a
DataFrame, which runs at its actions.

```{.rust .async}
use spark_connect_async::ml::{
    BinaryClassificationEvaluator, Estimator, Evaluator, LogisticRegression, Pipeline,
    Transformer, Vector, VectorAssembler,
};

let training = session.table("training");
let pipeline = Pipeline::new()
    .stage(VectorAssembler::new().input_cols(["age", "income"]).output_col("features"))
    .stage(LogisticRegression::new().max_iter(20));
let model = pipeline.fit(&training).await?;

let predictions = model.transform(&session.table("test"));
let probabilities: Vec<(f64, Vector)> = predictions
    .select(["prediction", "probability"])
    .collect_as()
    .await?;
let area = BinaryClassificationEvaluator::new().evaluate(&predictions).await?;
```

`fit`, `transform` and `evaluate` are the methods of the traits `Estimator`, `Transformer` and
`Evaluator`. The operators of the module convert into the stages of a `Pipeline`, and
`PipelineStage::estimator` and `PipelineStage::transformer` make one of any other estimator or
transformer. A `Pipeline` fits its stages in order, on the client, and a `CrossValidator`
chooses the best of several settings of an estimator by a k-fold cross validation, measured by
an evaluator. The server keeps a fitted model for the session until its last handle is
dropped: the model, its clones, and the DataFrames and calls that use it.

## Declarative Pipelines

A dataflow graph of Spark Declarative Pipelines holds datasets and the flows that fill them,
defined from SQL text or from DataFrames. `start_run` starts a run, whose events stream back as
the server runs the flows:

```{.rust .async}
use futures::StreamExt;

let graph = session.create_dataflow_graph().await?;
graph
    .define_sql("CREATE MATERIALIZED VIEW numbers AS SELECT id FROM range(10)")
    .await?;
graph.define_materialized_view("first_numbers").await?;
let first = session.table("numbers").limit(3);
graph.define_flow("first_numbers", "first_numbers", &first).await?;

let mut run = graph.start_run("file:///tmp/pipeline-storage").await?;
while let Some(event) = run.next().await {
    let event = event?;
    println!("{} {}", event.timestamp, event.message);
}
```

The server removes the graph once its last handle is dropped.
