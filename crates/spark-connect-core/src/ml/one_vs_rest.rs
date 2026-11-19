// SPDX-License-Identifier: Apache-2.0
//! The logic of PySpark's `OneVsRest` and `OneVsRestModel` that needs no session: the
//! expressions of their predictions, and the params of their metadata of PySpark's Python
//! format.

use spark_connect_proto as proto;

use super::meta::{json_to_ml_value, metadata_item, metadata_str, ml_value_json};
use super::persistence::json_object;
use super::{MlValue, Params};
use crate::error::{Result, SparkError};

#[doc(hidden)]
pub fn ml_exact_column(name: &str) -> crate::column::Column {
    crate::column::col(&format!("`{}`", name.replace('`', "``")))
}

/// Whether a storage level is PySpark's `StorageLevel(False, False, False, False)` (of replication
/// 1, which its `==` compares too): the storage of a DataFrame that is not persisted.
#[doc(hidden)]
pub fn is_storage_level_none(level: &proto::StorageLevel) -> bool {
    !level.use_disk
        && !level.use_memory
        && !level.use_off_heap
        && !level.deserialized
        && level.replication == 1
}

/// PySpark's `InternalFunction.vector_get`: the element `idx` of the vector `vec`, which is dense
/// or sparse, and an error for an index out of range.
#[doc(hidden)]
pub fn vector_get(vec: crate::column::Column, idx: crate::column::Column) -> crate::column::Column {
    use crate::column::{lit, lit_double, lit_string, when, Column};
    use crate::expression::Expression;
    use crate::functions::{array_size, get, raise_error, unwrap_udt};
    let unwrapped = unwrap_udt(vec);
    let is_dense = unwrapped.clone().get_field("type").eq(lit(1));
    let values = unwrapped.clone().get_field("values");
    let size = when(is_dense.clone(), array_size(values.clone()))
        .otherwise(unwrapped.clone().get_field("size"));
    // The internal function of the server, sent without a flag, as PySpark's connect client does.
    let sparse_idx = Column::new(Expression::function(
        "array_binary_search",
        vec![
            unwrapped.clone().get_field("indices").expression().clone(),
            idx.expression().clone(),
        ],
    ));
    let value = when(is_dense, get(values.clone(), idx.clone()))
        .when(
            sparse_idx.clone().ge(lit(0)),
            get(values.clone(), sparse_idx),
        )
        .otherwise(lit_double(0.0));
    let message = Column::new(Expression::function(
        "printf",
        vec![
            lit_string("Vector index must be in [0, %s), but got %s")
                .expression()
                .clone(),
            size.clone().expression().clone(),
            idx.clone().expression().clone(),
        ],
    ));
    when(
        idx.clone().ge(lit(0)).and(idx.clone().lt(size.clone())),
        value,
    )
    .otherwise(raise_error(message))
}

/// PySpark's `InternalFunction.array_argmax`: the index of the first greatest of `scores`, which
/// ignores NaN and null, and is -1 when no score is above negative infinity.
#[doc(hidden)]
pub fn array_argmax(scores: crate::column::Column) -> crate::column::Column {
    use crate::column::{lit, lit_double, when};
    use crate::functions::{aggregate_with_finish, r#struct};
    aggregate_with_finish(
        scores,
        r#struct([
            lit_double(f64::NEG_INFINITY).alias("v"),
            lit(-1).alias("i"),
            lit(0).alias("j"),
        ]),
        |args| {
            let acc = args[0].clone();
            let value = args[1].clone();
            let best = acc.clone().get_field("v");
            let index = acc.clone().get_field("i");
            let current = acc.get_field("j");
            when(
                value
                    .clone()
                    .is_nan()
                    .not()
                    .and(value.clone().is_null().not())
                    .and(value.clone().gt(best.clone())),
                r#struct([
                    value.alias("v"),
                    current.clone().alias("i"),
                    current.clone() + lit(1),
                ]),
            )
            .otherwise(r#struct([
                best.alias("v"),
                index.alias("i"),
                current + lit(1),
            ]))
        },
        |args| args[0].clone().get_field("i"),
    )
}

// ---------------------------------------------------------------------------------------------
// Persistence, as `OneVsRestWriter`, `OneVsRestModelWriter` and their readers of PySpark.
// ---------------------------------------------------------------------------------------------

/// The defaults of the params of a OneVsRest, as JSON texts, in the order of PySpark's
/// `_defaultParamMap` of the estimator.
#[doc(hidden)]
pub const ESTIMATOR_DEFAULTS: &[(&str, &str)] = &[
    ("parallelism", "1"),
    ("predictionCol", "\"prediction\""),
    ("featuresCol", "\"features\""),
    ("labelCol", "\"label\""),
    ("rawPredictionCol", "\"rawPrediction\""),
];

/// The defaults of the params of a OneVsRestModel, as JSON texts, in the order of PySpark's
/// `_defaultParamMap` of the model: the estimator's, but for `parallelism`, which a model has not.
#[doc(hidden)]
pub const MODEL_DEFAULTS: &[(&str, &str)] = &[
    ("predictionCol", "\"prediction\""),
    ("featuresCol", "\"features\""),
    ("labelCol", "\"label\""),
    ("rawPredictionCol", "\"rawPrediction\""),
];

/// The JSON text of the literal of a param, as `json.dumps` writes its value.
fn expression_json(expr: &spark_connect_proto::Expression) -> Result<String> {
    use spark_connect_proto::expression::ExprType;
    match &expr.expr_type {
        Some(ExprType::Literal(literal)) => ml_value_json(&MlValue::from_literal(literal)?),
        _ => Err(SparkError::value_msg(
            "a param of a OneVsRest is no literal value",
        )),
    }
}

/// The JSON text of the `paramMap` of a OneVsRest or its model, as `extractJsonParams` writes it
/// (without the classifier): of each of `defaults`, in its order, the set value or the default,
/// and then the set `weightCol`, which has no default.
#[doc(hidden)]
pub fn param_map_json(params: &Params, defaults: &[(&str, &str)]) -> Result<String> {
    let mut fields = Vec::with_capacity(defaults.len() + 1);
    for (name, default) in defaults {
        let value = match params.get_param(name) {
            Some(expr) => expression_json(expr)?,
            None => default.to_string(),
        };
        fields.push((*name, value));
    }
    if let Some(expr) = params.get_param("weightCol") {
        fields.push(("weightCol", expression_json(expr)?));
    }
    Ok(json_object(&fields))
}

/// The JSON text of the `defaultParamMap` of a OneVsRest or its model: of `defaults`.
#[doc(hidden)]
pub fn default_param_map_json(defaults: &[(&str, &str)]) -> String {
    let fields: Vec<(&str, String)> = defaults
        .iter()
        .map(|(name, value)| (*name, value.to_string()))
        .collect();
    json_object(&fields)
}

/// Whether the metadata `metadata` is of the class `expected`, as `_parseMetaData` checks it.
#[doc(hidden)]
pub fn check_class(metadata: &serde_json::Value, expected: &str) -> Result<()> {
    let class = metadata_str(metadata, "class")?;
    if class != expected {
        return Err(SparkError::value_msg(format!(
            "Error loading metadata: Expected class name {expected} but found class name {class}"
        )));
    }
    Ok(())
}

/// The params of the `paramMap` of the metadata `metadata`, set as `getAndSetParams` sets them,
/// without the classifier.
#[doc(hidden)]
pub fn one_vs_rest_params(metadata: &serde_json::Value) -> Result<Params> {
    let map = metadata_item(metadata, "paramMap")?
        .as_object()
        .ok_or_else(|| SparkError::value_msg("the paramMap of the metadata is no object"))?;
    let mut params = Params::new();
    for (name, value) in map {
        if name != "classifier" {
            params = params.set_param_literal(name, json_to_ml_value(value).to_literal()?);
        }
    }
    Ok(params)
}
