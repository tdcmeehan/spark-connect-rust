// SPDX-License-Identifier: Apache-2.0
//! The logic of PySpark's meta-algorithms of `pyspark.ml` that needs no session: param maps
//! and their grids, the values of params as the type converters of their classes convert
//! them, and the Python representations and JSON texts of values and of the metadata of
//! PySpark's Python format.

use super::param_specs::{ParamConverter, MODEL_CLASSES, PARAM_CONVERTERS, PARAM_DOCS};
use super::persistence::json_string;
use super::stage_kinds::STAGE_KINDS;
use super::{MlValue, OperatorType, Params};
use crate::error::{Result, SparkError};

// ---------------------------------------------------------------------------------------------
// Param maps.
// ---------------------------------------------------------------------------------------------

/// The value of a param of a stage in a [`ParamMap`]: the uid of the stage (PySpark's
/// `Param.parent`) and the name of the param of the server's class.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamPair {
    pub parent: String,
    pub name: String,
    pub value: MlValue,
}

/// A param map, as PySpark's (a dict of Params): the values of the params of stages, in the order
/// of their insertion, each param of each stage once.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParamMap {
    pairs: Vec<ParamPair>,
}

impl ParamMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// The map with the param `name` of the stage of the uid `parent` of `value`, in place of a
    /// value that it has, as a dict sets a key.
    pub fn put(mut self, parent: &str, name: &str, value: MlValue) -> Self {
        self.insert(parent, name, value);
        self
    }

    /// Set the param `name` of the stage of the uid `parent` to `value` (see [`Self::put`]).
    pub fn insert(&mut self, parent: &str, name: &str, value: MlValue) {
        match self
            .pairs
            .iter_mut()
            .find(|p| p.parent == parent && p.name == name)
        {
            Some(pair) => pair.value = value,
            None => self.pairs.push(ParamPair {
                parent: parent.to_string(),
                name: name.to_string(),
                value,
            }),
        }
    }

    /// The value of the param `name` of the stage of the uid `parent`.
    pub fn get(&self, parent: &str, name: &str) -> Option<&MlValue> {
        self.pairs
            .iter()
            .find(|p| p.parent == parent && p.name == name)
            .map(|p| &p.value)
    }

    /// The values, in the order of their insertion.
    pub fn pairs(&self) -> &[ParamPair] {
        &self.pairs
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The values of the params of the stage of the uid `uid`.
    pub fn of_stage<'a>(&'a self, uid: &'a str) -> impl Iterator<Item = &'a ParamPair> + 'a {
        self.pairs.iter().filter(move |p| p.parent == uid)
    }

    /// Whether the map has a param of the stage of the uid `uid`.
    pub fn has_stage(&self, uid: &str) -> bool {
        self.of_stage(uid).next().is_some()
    }
}

/// PySpark's `ParamGridBuilder`: a grid of the values of params, whose param maps
/// ([`Self::build`]) are each combination of them, as `itertools.product` of the values of the
/// params in their order (the last param varies fastest).
#[derive(Debug, Clone, Default)]
pub struct ParamGridBuilder {
    grid: Vec<(String, String, Vec<MlValue>)>,
}

impl ParamGridBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The values of the param `name` of the stage of the uid `parent`, as `addGrid`: in place of
    /// the values that it has, as a dict sets a key.
    pub fn add_grid(
        mut self,
        parent: &str,
        name: &str,
        values: impl IntoIterator<Item = MlValue>,
    ) -> Self {
        let values: Vec<MlValue> = values.into_iter().collect();
        match self
            .grid
            .iter_mut()
            .find(|(p, n, _)| p == parent && n == name)
        {
            Some((_, _, old)) => *old = values,
            None => self
                .grid
                .push((parent.to_string(), name.to_string(), values)),
        }
        self
    }

    /// Each value of `map` as the only value of its param, as `baseOn`.
    pub fn base_on(mut self, map: &ParamMap) -> Self {
        for pair in map.pairs() {
            self = self.add_grid(&pair.parent, &pair.name, [pair.value.clone()]);
        }
        self
    }

    /// Every combination of the values of the params, as `build`: one empty map of no params.
    pub fn build(&self) -> Vec<ParamMap> {
        let mut maps = vec![ParamMap::new()];
        for (parent, name, values) in &self.grid {
            maps = maps
                .iter()
                .flat_map(|map| {
                    values
                        .iter()
                        .map(move |value| map.clone().put(parent, name, value.clone()))
                })
                .collect();
        }
        maps
    }
}

/// Set the values of `map` of the params of the stage of the uid `uid` and the JVM class `class`
/// in `params`, as PySpark's `copy(extra)` does: each value converted as its param's type
/// converter converts it, and none of the params that the class does not have (PySpark's
/// `_copyValues` sets only the params of the copied object).
#[doc(hidden)]
pub fn apply_pairs(params: &mut Params, class: &str, uid: &str, map: &ParamMap) -> Result<()> {
    for pair in map.of_stage(uid) {
        if class_known(class) && param_converter(class, &pair.name).is_none() {
            continue;
        }
        let literal = converted(class, &pair.name, &pair.value)?.to_literal()?;
        let taken = std::mem::replace(params, Params::new());
        *params = taken.set_param_literal(&pair.name, literal);
    }
    Ok(())
}

/// The simple name of a JVM class (of its package stripped), which the generated table names.
fn simple_name(class: &str) -> &str {
    class.rsplit('.').next().unwrap_or(class)
}

/// Whether the generated table of the params of pyspark.ml has the class `class`.
fn class_known(class: &str) -> bool {
    PARAM_CONVERTERS
        .iter()
        .any(|(name, _)| *name == simple_name(class))
}

/// The doc of the param `name` of the class of the simple name `class`, as PySpark's `Param.doc`
/// of the class, if the table has the param.
pub(super) fn param_doc(class: &str, name: &str) -> Option<&'static str> {
    let (_, params) = PARAM_DOCS.iter().find(|(simple, _)| *simple == class)?;
    params
        .iter()
        .find(|(param, _)| *param == name)
        .map(|(_, doc)| *doc)
}

/// The type converter of the param `name` of the class `class`, if the table has the param.
pub(super) fn param_converter(class: &str, name: &str) -> Option<ParamConverter> {
    let (_, params) = PARAM_CONVERTERS
        .iter()
        .find(|(simple, _)| *simple == simple_name(class))?;
    params
        .iter()
        .find(|(param, _)| *param == name)
        .map(|(_, converter)| *converter)
}

/// The value `value` of the param `name` of the class `class`, converted as PySpark's
/// `Param.typeConverter` converts it, or as it is for a param that the table does not have.
#[doc(hidden)]
pub fn converted(class: &str, name: &str, value: &MlValue) -> Result<MlValue> {
    match param_converter(class, name) {
        Some(converter) => convert_param(converter, value),
        None => Ok(value.clone()),
    }
}

/// The JVM class of the model that an estimator of the JVM class `estimator` fits (PySpark's
/// `_create_model`): the class of the estimator's name with `Model` appended, unless the table
/// names another (the models of the tree, ensemble, FM and perceptron classifiers and regressors,
/// and QuantileDiscretizer's `Bucketizer`).
#[doc(hidden)]
pub fn model_class(estimator: &str) -> String {
    match MODEL_CLASSES
        .iter()
        .find(|(simple, _)| *simple == simple_name(estimator))
    {
        Some((_, model)) => model.to_string(),
        None => format!("{estimator}Model"),
    }
}

/// The params of `params` that the JVM class `class` has, as PySpark's `_copyValues` copies them
/// to an object of the class. A class that the table does not have keeps all of them.
#[doc(hidden)]
pub fn params_of_class(params: &Params, class: &str) -> Params {
    if !class_known(class) {
        return params.clone();
    }
    params.retain_names(|name| param_converter(class, name).is_some())
}

/// The value of `value` as PySpark's type converter of `converter` converts it, or PySpark's
/// `TypeError` (of its message) when it cannot.
pub(super) fn convert_param(converter: ParamConverter, value: &MlValue) -> Result<MlValue> {
    let could_not = |target: &str| {
        SparkError::value_msg(format!("Could not convert {} to {target}", py_str(value)))
    };
    match converter {
        ParamConverter::Identity => Ok(value.clone()),
        ParamConverter::Boolean => match value {
            MlValue::Bool(b) => Ok(MlValue::Bool(*b)),
            other => Err(SparkError::value_msg(format!(
                "Boolean Param requires value of type bool. Found {}.",
                py_type(other)
            ))),
        },
        ParamConverter::Float => to_float(value)
            .map(MlValue::Double)
            .ok_or_else(|| could_not("float")),
        ParamConverter::Int => to_int(value).ok_or_else(|| could_not("int")),
        ParamConverter::String => match value {
            MlValue::String(s) => Ok(MlValue::String(s.clone())),
            other => Err(SparkError::value_msg(format!(
                "Could not convert {} to string type",
                py_type(other)
            ))),
        },
        ParamConverter::ListFloat => list_of(value, "list of floats", |v| {
            to_float(v).map(MlValue::Double)
        }),
        ParamConverter::ListInt => list_of(value, "list of ints", to_int),
        ParamConverter::ListString => list_of(value, "list of strings", |v| match v {
            MlValue::String(s) => Some(MlValue::String(s.clone())),
            _ => None,
        }),
        ParamConverter::ListListFloat => match value {
            MlValue::Array(items) => Ok(MlValue::Array(
                items
                    .iter()
                    .map(|item| convert_param(ParamConverter::ListFloat, item))
                    .collect::<Result<Vec<_>>>()?,
            )),
            _ => Err(could_not("list of list of floats")),
        },
    }
}

/// The list of the converted items of `value`, as PySpark's list converters convert it, or
/// PySpark's `TypeError` of a value that is no list of such items.
fn list_of(
    value: &MlValue,
    target: &str,
    item: impl Fn(&MlValue) -> Option<MlValue>,
) -> Result<MlValue> {
    let fail = || SparkError::value_msg(format!("Could not convert {} to {target}", py_str(value)));
    match value {
        MlValue::Array(items) => items
            .iter()
            .map(|v| item(v).ok_or_else(fail))
            .collect::<Result<Vec<_>>>()
            .map(MlValue::Array),
        _ => Err(fail()),
    }
}

/// The float of a numeric value (a Python `int` or `float`, not a `bool`), as `TypeConverters`.
fn to_float(value: &MlValue) -> Option<f64> {
    match value {
        MlValue::Byte(v) | MlValue::Short(v) | MlValue::Int(v) => Some(f64::from(*v)),
        MlValue::Long(v) => Some(*v as f64),
        MlValue::Float(v) => Some(f64::from(*v)),
        MlValue::Double(v) => Some(*v),
        _ => None,
    }
}

/// The int of a numeric value that is integral, as `TypeConverters.toInt`: an `Int` when it fits,
/// else a `Long`.
fn to_int(value: &MlValue) -> Option<MlValue> {
    let whole = match value {
        MlValue::Byte(v) | MlValue::Short(v) | MlValue::Int(v) => i64::from(*v),
        MlValue::Long(v) => *v,
        MlValue::Float(v) if f64::from(*v).fract() == 0.0 => f64::from(*v) as i64,
        MlValue::Double(v) if v.is_finite() && v.fract() == 0.0 => *v as i64,
        _ => return None,
    };
    Some(match i32::try_from(whole) {
        Ok(v) => MlValue::Int(v),
        Err(_) => MlValue::Long(whole),
    })
}

/// The Python type of a value, as PySpark's messages name `type(value)`.
fn py_type(value: &MlValue) -> &'static str {
    match value {
        MlValue::Null => "<class 'NoneType'>",
        MlValue::Bool(_) => "<class 'bool'>",
        MlValue::Byte(_) | MlValue::Short(_) | MlValue::Int(_) | MlValue::Long(_) => {
            "<class 'int'>"
        }
        MlValue::Float(_) | MlValue::Double(_) => "<class 'float'>",
        MlValue::String(_) => "<class 'str'>",
        MlValue::Array(_) => "<class 'list'>",
        _ => "<class 'object'>",
    }
}

/// The Python `repr` of the str `text`: in single quotes, or in double quotes when it has a single
/// quote and no double one, with the backslashes, the quote and the control characters escaped.
#[doc(hidden)]
pub fn python_str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// The Python `str` of a value, as PySpark's messages show it (a list as its `repr`).
#[doc(hidden)]
pub fn py_str(value: &MlValue) -> String {
    match value {
        MlValue::Null => "None".to_string(),
        MlValue::Bool(true) => "True".to_string(),
        MlValue::Bool(false) => "False".to_string(),
        MlValue::Byte(v) | MlValue::Short(v) | MlValue::Int(v) => v.to_string(),
        MlValue::Long(v) => v.to_string(),
        MlValue::Float(v) => python_float_repr(f64::from(*v)),
        MlValue::Double(v) => python_float_repr(*v),
        MlValue::String(s) => s.clone(),
        MlValue::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|item| match item {
                    MlValue::String(s) => python_str_repr(s),
                    other => py_str(other),
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => format!("{other:?}"),
    }
}

/// The JVM class kind of the class `class` of a stage of pyspark.ml (see `stage_kinds`).
#[doc(hidden)]
pub fn stage_kind(class: &str) -> Option<OperatorType> {
    STAGE_KINDS
        .iter()
        .find(|(name, _)| *name == class)
        .map(|(_, kind)| *kind)
}

/// The JSON text `text` of a metadata of Python's `json.dumps`, parsed: of its floats `NaN`,
/// `Infinity` and `-Infinity` (which JSON has not), as the strings of [`python_float`].
#[doc(hidden)]
pub fn parse_python_json(text: &str) -> Result<serde_json::Value> {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.char_indices().peekable();
    let mut in_string = false;
    while let Some((i, c)) = chars.next() {
        if in_string {
            out.push(c);
            if c == '\\' {
                if let Some((_, escaped)) = chars.next() {
                    out.push(escaped);
                }
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            continue;
        }
        let rest = &text[i..];
        let token = ["-Infinity", "Infinity", "NaN"]
            .into_iter()
            .find(|t| rest.starts_with(t));
        if let Some(token) = token {
            // The marker, of the JSON escape of its control character.
            out.push_str(&format!("\"\\u0000float:{token}\""));
            for _ in 1..token.len() {
                chars.next();
            }
            continue;
        }
        out.push(c);
    }
    serde_json::from_str(&out).map_err(|e| SparkError::value_msg(format!("the metadata: {e}")))
}

/// The prefix of the strings of the non-finite floats of [`parse_python_json`].
const NON_FINITE: &str = "\u{0}float:";

/// The float of a JSON value of a metadata of [`parse_python_json`]: a number, or a non-finite
/// float.
#[doc(hidden)]
pub fn python_float(value: &serde_json::Value) -> Option<f64> {
    if let Some(number) = value.as_f64() {
        return Some(number);
    }
    match value.as_str()?.strip_prefix(NON_FINITE)? {
        "NaN" => Some(f64::NAN),
        "Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

/// The JSON text of the float `value`, as Python's `json.dumps` writes it (`NaN`, `Infinity`).
#[doc(hidden)]
pub fn python_float_json(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value == f64::INFINITY {
        "Infinity".to_string()
    } else if value == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        python_float_repr(value)
    }
}

/// The text of the finite or non-finite float `value` as Python's `repr` (and `str`) writes it:
/// the shortest digits that read back the same, in the decimal form for a decimal exponent from -4
/// to 15, else in the exponent form with a sign and at least two digits (`1e-06`, `1e+16`).
#[doc(hidden)]
pub fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    // Rust's `Debug` of a float has the same shortest digits and the same thresholds of the
    // exponent form, but writes the exponent without its sign and its padding.
    let text = format!("{value:?}");
    match text.split_once('e') {
        None => text,
        Some((mantissa, exponent)) => {
            let (sign, digits) = match exponent.strip_prefix('-') {
                Some(digits) => ('-', digits),
                None => ('+', exponent),
            };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
    }
}

/// `metadata[key]`, an error of a metadata without it, as Python's dict raises a KeyError.
#[doc(hidden)]
pub fn metadata_item<'a>(
    metadata: &'a serde_json::Value,
    key: &str,
) -> Result<&'a serde_json::Value> {
    metadata
        .get(key)
        .ok_or_else(|| SparkError::value_msg(format!("the metadata has no '{key}'")))
}

/// The string `metadata[key]`.
#[doc(hidden)]
pub fn metadata_str<'a>(metadata: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    metadata_item(metadata, key)?
        .as_str()
        .ok_or_else(|| SparkError::value_msg(format!("the '{key}' of the metadata is no string")))
}

/// The MlValue of a JSON value of a metadata: a bool, an int (of a long), a float, a string, a
/// null, or a list of them.
#[doc(hidden)]
pub fn json_to_ml_value(value: &serde_json::Value) -> MlValue {
    match value {
        serde_json::Value::Null => MlValue::Null,
        serde_json::Value::Bool(b) => MlValue::Bool(*b),
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) if i32::try_from(i).is_ok() => MlValue::Int(i as i32),
            Some(i) => MlValue::Long(i),
            None => MlValue::Double(n.as_f64().unwrap_or(f64::NAN)),
        },
        serde_json::Value::String(s) => match python_float(value) {
            Some(f) if s.starts_with(NON_FINITE) => MlValue::Double(f),
            _ => MlValue::String(s.clone()),
        },
        serde_json::Value::Array(items) => {
            MlValue::Array(items.iter().map(json_to_ml_value).collect())
        }
        serde_json::Value::Object(_) => MlValue::String(value.to_string()),
    }
}

/// The JSON text of an MlValue, as Python's `json.dumps` writes the value of a param.
#[doc(hidden)]
pub fn ml_value_json(value: &MlValue) -> Result<String> {
    Ok(match value {
        MlValue::Null => "null".to_string(),
        MlValue::Bool(b) => b.to_string(),
        MlValue::Byte(i) | MlValue::Short(i) | MlValue::Int(i) => i.to_string(),
        MlValue::Long(i) => i.to_string(),
        MlValue::Float(f) => python_float_json(f64::from(*f)),
        MlValue::Double(f) => python_float_json(*f),
        MlValue::String(s) => json_string(s),
        MlValue::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(ml_value_json)
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        ),
        other => {
            return Err(SparkError::value_msg(format!(
                "a value of a param of no JSON: {other:?}"
            )))
        }
    })
}

/// The Python class of the class `class` of a stage's metadata, as `DefaultParamsReader` names it:
/// a class of PySpark's Python format as it is, else the JVM class with `pyspark` for
/// `org.apache.spark`. The pipelines of the JVM writer are `pyspark.ml.Pipeline` and
/// `pyspark.ml.PipelineModel` there, which are the classes of `pyspark.ml.pipeline`.
#[doc(hidden)]
pub fn python_class_of(class: &str) -> String {
    if class.starts_with("pyspark.ml.") {
        return class.to_string();
    }
    match class.replace("org.apache.spark", "pyspark").as_str() {
        "pyspark.ml.Pipeline" => PIPELINE_CLASS.to_string(),
        "pyspark.ml.PipelineModel" => PIPELINE_MODEL_CLASS.to_string(),
        python => python.to_string(),
    }
}

#[doc(hidden)]
pub const PIPELINE_CLASS: &str = "pyspark.ml.pipeline.Pipeline";

#[doc(hidden)]
pub const PIPELINE_MODEL_CLASS: &str = "pyspark.ml.pipeline.PipelineModel";

#[doc(hidden)]
pub const CROSS_VALIDATOR_CLASS: &str = "pyspark.ml.tuning.CrossValidator";

#[doc(hidden)]
pub const CROSS_VALIDATOR_MODEL_CLASS: &str = "pyspark.ml.tuning.CrossValidatorModel";

#[doc(hidden)]
pub const TRAIN_VALIDATION_SPLIT_CLASS: &str = "pyspark.ml.tuning.TrainValidationSplit";

#[doc(hidden)]
pub const ONE_VS_REST_CLASS: &str = "pyspark.ml.classification.OneVsRest";

#[doc(hidden)]
pub const ONE_VS_REST_MODEL_CLASS: &str = "pyspark.ml.classification.OneVsRestModel";

#[doc(hidden)]
pub const TRAIN_VALIDATION_SPLIT_MODEL_CLASS: &str = "pyspark.ml.tuning.TrainValidationSplitModel";

#[doc(hidden)]
pub const QUANTILE_DISCRETIZER_CLASS: &str = "pyspark.ml.feature.QuantileDiscretizer";
