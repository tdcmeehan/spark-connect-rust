// SPDX-License-Identifier: Apache-2.0
//! The logic of PySpark's validators of `pyspark.ml.tuning` that needs no session: their
//! default seeds, the choice of the best param map, and the JSON texts of their metadata.

use std::sync::OnceLock;

use super::meta::{
    converted, ml_value_json, param_doc, python_float, python_float_json, python_str_repr, ParamMap,
};
use super::persistence::{json_object, json_string};
use super::{MlOperator, MlValue};
use crate::error::{Result, SparkError};

/// A random seed, drawn for the default seed of a class of validators.
fn random_seed() -> i64 {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    i64::from_str_radix(&hex[..15], 16).unwrap_or_default()
}

// The default seed of each class of validators and of its model, drawn once per process: PySpark's
// default seed is `hash(type(self).__name__)`, which is the same for every instance of a class
// within a process (and differs between processes, as Python randomizes the hash of a string).
#[doc(hidden)]
pub static CROSS_VALIDATOR_SEED: OnceLock<i64> = OnceLock::new();
#[doc(hidden)]
pub static CROSS_VALIDATOR_MODEL_SEED: OnceLock<i64> = OnceLock::new();
#[doc(hidden)]
pub static TRAIN_VALIDATION_SPLIT_SEED: OnceLock<i64> = OnceLock::new();
#[doc(hidden)]
pub static TRAIN_VALIDATION_SPLIT_MODEL_SEED: OnceLock<i64> = OnceLock::new();

/// The default seed of the class whose cell is `cell`, the same in each call of the process.
#[doc(hidden)]
pub fn process_seed(cell: &'static OnceLock<i64>) -> i64 {
    *cell.get_or_init(random_seed)
}

/// The value of the param `name` of the validator of the JVM class `class` and the uid `uid` in
/// `map`, converted as its param's type converter converts it, if the map sets it.
#[doc(hidden)]
pub fn own_value(class: &str, uid: &str, map: &ParamMap, name: &str) -> Result<Option<MlValue>> {
    map.get(uid, name)
        .map(|value| converted(class, name, value))
        .transpose()
}

/// PySpark's check of its pool of `min(parallelism, number of param maps)` threads.
#[doc(hidden)]
pub fn check_pool(parallelism: i64, maps: usize) -> Result<()> {
    if parallelism.min(maps as i64) < 1 {
        return Err(SparkError::value_msg(
            "Number of processes must be at least 1",
        ));
    }
    Ok(())
}

/// The index of the best of `metrics`, as numpy's `argmax` for an evaluator whose larger metrics
/// are better, else `argmin`: the first of the best, and the first NaN of any.
#[doc(hidden)]
pub fn best_index(metrics: &[f64], larger_is_better: bool) -> usize {
    if let Some(nan) = metrics.iter().position(|m| m.is_nan()) {
        return nan;
    }
    let mut best = 0;
    for (i, &metric) in metrics.iter().enumerate().skip(1) {
        let better = if larger_is_better {
            metric > metrics[best]
        } else {
            metric < metrics[best]
        };
        if better {
            best = i;
        }
    }
    best
}

/// The mean and the standard deviation of each metric over the folds, as numpy's `mean` and `std`
/// of axis 0 (the deviation of the population).
#[doc(hidden)]
pub fn mean_and_std(metrics: &[Vec<f64>], num_models: usize) -> (Vec<f64>, Vec<f64>) {
    let n = metrics.len() as f64;
    let mut means = vec![0.0; num_models];
    for fold in metrics {
        for (j, m) in fold.iter().enumerate() {
            means[j] += m;
        }
    }
    for mean in &mut means {
        *mean /= n;
    }
    let mut stds = vec![0.0; num_models];
    for fold in metrics {
        for (j, m) in fold.iter().enumerate() {
            stds[j] += (m - means[j]).powi(2);
        }
    }
    for std in &mut stds {
        *std = (*std / n).sqrt();
    }
    (means, stds)
}

/// The JSON of a list of floats, as Python's `json.dumps` writes it (of `NaN`).
#[doc(hidden)]
pub fn floats_json(values: &[f64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|v| python_float_json(*v))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// The floats of a list of a metadata.
#[doc(hidden)]
pub fn floats(value: &serde_json::Value) -> Result<Vec<f64>> {
    value
        .as_array()
        .ok_or_else(|| SparkError::value_msg("the metrics of the metadata are no list"))?
        .iter()
        .map(|v| {
            python_float(v)
                .ok_or_else(|| SparkError::value_msg("a metric of the metadata is no float"))
        })
        .collect()
}

/// Whether the sub-models were saved, of the `persistSubModels` of a model's metadata.
#[doc(hidden)]
pub fn persisted_sub_models(metadata: &serde_json::Value) -> bool {
    metadata
        .get("persistSubModels")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// PySpark's error of a save of the sub-models of a model of none.
#[doc(hidden)]
pub const NO_SUB_MODELS: &str =
    "When persisting tuning models, you can only set persistSubModels to \
                             true if the tuning was done with collectSubModels set to true. To \
                             save the sub-models, try rerunning fitting with collectSubModels set \
                             to true.";

/// The uid and the JVM class of a stage's operator.
#[doc(hidden)]
pub fn stage_of(operator: &MlOperator) -> (String, String) {
    (operator.uid.clone(), operator.name.clone())
}

/// PySpark's `repr` of the param `name` of the stage of the uid `parent`, `Param(parent=..,
/// name=.., doc=..)`. A uid is its class's name and a suffix, as PySpark's `Identifiable` makes it,
/// so the doc is the one of that class; a param that the tables do not have has an empty doc.
#[doc(hidden)]
pub fn param_repr(parent: &str, name: &str) -> String {
    let class = parent.rsplit_once('_').map_or(parent, |(class, _)| class);
    let doc = param_doc(class, name).unwrap_or("");
    format!(
        "Param(parent={}, name={}, doc={})",
        python_str_repr(parent),
        python_str_repr(name),
        python_str_repr(doc)
    )
}

/// The param maps of a validator, as `_ValidatorSharedReadWrite.saveImpl` writes them: each value
/// converted by the type converter of its param (of its stage, in `stages`), as PySpark converts
/// the values of its grid when it builds them, and written as the JSON of its `json.dumps`, a
/// string.
#[doc(hidden)]
pub fn param_maps_json(maps: &[ParamMap], stages: &[(String, String)]) -> Result<String> {
    let mut out = Vec::with_capacity(maps.len());
    for map in maps {
        let mut pairs = Vec::with_capacity(map.len());
        for pair in map.pairs() {
            let class = stages
                .iter()
                .find(|(uid, _)| *uid == pair.parent)
                .map_or("", |(_, class)| class.as_str());
            let value = converted(class, &pair.name, &pair.value)?;
            pairs.push(json_object(&[
                ("parent", json_string(&pair.parent)),
                ("name", json_string(&pair.name)),
                ("value", json_string(&ml_value_json(&value)?)),
                ("isJson", "true".to_string()),
            ]));
        }
        out.push(format!("[{}]", pairs.join(",")));
    }
    Ok(format!("[{}]", out.join(",")))
}
