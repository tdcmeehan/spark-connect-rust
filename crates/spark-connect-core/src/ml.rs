//! The operators of the ML API, the values of their params and methods, the commands and
//! relations that fit, evaluate and apply them, that fetch, save and load server objects, and that
//! delete the objects that the server caches, and the logic of PySpark's meta-algorithms and of its
//! Python format of ML objects that needs no session.

use std::collections::HashMap;

use spark_connect_proto as proto;
use uuid::Uuid;

use crate::error::{SparkError, SparkErrorKind};
use crate::expression::LiteralExpression;
use crate::types::DataType;

mod meta;
pub use meta::*;
mod object;
pub use object::*;
mod one_vs_rest;
pub use one_vs_rest::*;
mod param_specs;
mod persistence;
pub use persistence::*;
mod stage_kinds;
mod validators;
pub use validators::*;

/// A dense or sparse ML vector used as an estimator parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum MlVector {
    Dense(Vec<f64>),
    Sparse {
        size: i32,
        indices: Vec<i32>,
        values: Vec<f64>,
    },
}

impl From<Vec<f64>> for MlVector {
    fn from(values: Vec<f64>) -> Self {
        Self::Dense(values)
    }
}

impl<const N: usize> From<[f64; N]> for MlVector {
    fn from(values: [f64; N]) -> Self {
        Self::Dense(values.into())
    }
}

impl MlVector {
    #[allow(deprecated)] // PySpark's ML serializer uses struct_type for vector UDTs.
    fn to_literal(&self) -> proto::expression::Literal {
        use proto::expression::literal::{specialized_array::ValueType as A, LiteralType as L};
        let lit = |value| proto::expression::Literal {
            literal_type: Some(value),
            ..Default::default()
        };
        let null = || lit(L::Null(DataType::Null.to_proto()));
        let doubles = |values: Vec<f64>| {
            lit(L::SpecializedArray(
                proto::expression::literal::SpecializedArray {
                    value_type: Some(A::Doubles(proto::Doubles { values })),
                },
            ))
        };
        let elements = match self {
            Self::Dense(values) => vec![lit(L::Byte(1)), null(), null(), doubles(values.clone())],
            Self::Sparse {
                size,
                indices,
                values,
            } => vec![
                lit(L::Byte(0)),
                lit(L::Integer(*size)),
                lit(L::SpecializedArray(
                    proto::expression::literal::SpecializedArray {
                        value_type: Some(A::Ints(proto::Ints {
                            values: indices.clone(),
                        })),
                    },
                )),
                doubles(values.clone()),
            ],
        };
        lit(L::Struct(proto::expression::literal::Struct {
            struct_type: Some(proto::DataType {
                kind: Some(proto::data_type::Kind::Udt(Box::new(
                    proto::data_type::Udt {
                        r#type: "udt".into(),
                        jvm_class: Some("org.apache.spark.ml.linalg.VectorUDT".into()),
                        ..Default::default()
                    },
                ))),
            }),
            elements,
        }))
    }

    #[doc(hidden)]
    pub fn from_literal(value: &proto::expression::literal::LiteralType) -> Option<Self> {
        use proto::expression::literal::{specialized_array::ValueType as A, LiteralType as L};
        let L::Struct(value) = value else {
            return None;
        };
        let fields = &value.elements;
        let L::SpecializedArray(array) = fields.get(3)?.literal_type.as_ref()? else {
            return None;
        };
        let A::Doubles(values) = array.value_type.as_ref()? else {
            return None;
        };
        match fields.first()?.literal_type.as_ref()? {
            L::Byte(1) => Some(Self::Dense(values.values.clone())),
            L::Byte(0) => {
                let L::Integer(size) = fields.get(1)?.literal_type.as_ref()? else {
                    return None;
                };
                let L::SpecializedArray(array) = fields.get(2)?.literal_type.as_ref()? else {
                    return None;
                };
                let A::Ints(indices) = array.value_type.as_ref()? else {
                    return None;
                };
                Some(Self::Sparse {
                    size: *size,
                    indices: indices.values.clone(),
                    values: values.values.clone(),
                })
            }
            _ => None,
        }
    }
}

/// A dense or sparse ML matrix used as an estimator parameter, such as the
/// `lowerBoundsOnCoefficients` of `LogisticRegression`. As Spark's `Matrix`, its values are in
/// column-major order, or in row-major order when it `is_transposed`; a sparse matrix is of the
/// compressed sparse column format (of rows, when it `is_transposed`).
#[derive(Debug, Clone, PartialEq)]
pub enum MlMatrix {
    Dense {
        num_rows: i32,
        num_cols: i32,
        values: Vec<f64>,
        is_transposed: bool,
    },
    Sparse {
        num_rows: i32,
        num_cols: i32,
        col_ptrs: Vec<i32>,
        row_indices: Vec<i32>,
        values: Vec<f64>,
        is_transposed: bool,
    },
}

impl MlMatrix {
    /// The literal of the matrix, as PySpark's ML serializer (`serialize_param` of
    /// `pyspark.ml.connect.serialize`) sends a `DenseMatrix` or a `SparseMatrix`: a struct of the
    /// MatrixUDT, of its type (1 dense, 0 sparse), its shape, its column pointers and row indices
    /// (nulls of a dense matrix), its values and whether it is transposed.
    #[allow(deprecated)] // PySpark's ML serializer uses struct_type for matrix UDTs.
    fn to_literal(&self) -> proto::expression::Literal {
        use proto::expression::literal::{specialized_array::ValueType as A, LiteralType as L};
        let lit = |value| proto::expression::Literal {
            literal_type: Some(value),
            ..Default::default()
        };
        let null = || lit(L::Null(DataType::Null.to_proto()));
        let array = |value| {
            lit(L::SpecializedArray(
                proto::expression::literal::SpecializedArray {
                    value_type: Some(value),
                },
            ))
        };
        let ints = |values: &[i32]| {
            array(A::Ints(proto::Ints {
                values: values.to_vec(),
            }))
        };
        let doubles = |values: &[f64]| {
            array(A::Doubles(proto::Doubles {
                values: values.to_vec(),
            }))
        };
        let elements = match self {
            Self::Dense {
                num_rows,
                num_cols,
                values,
                is_transposed,
            } => vec![
                lit(L::Byte(1)),
                lit(L::Integer(*num_rows)),
                lit(L::Integer(*num_cols)),
                null(),
                null(),
                doubles(values),
                lit(L::Boolean(*is_transposed)),
            ],
            Self::Sparse {
                num_rows,
                num_cols,
                col_ptrs,
                row_indices,
                values,
                is_transposed,
            } => vec![
                lit(L::Byte(0)),
                lit(L::Integer(*num_rows)),
                lit(L::Integer(*num_cols)),
                ints(col_ptrs),
                ints(row_indices),
                doubles(values),
                lit(L::Boolean(*is_transposed)),
            ],
        };
        lit(L::Struct(proto::expression::literal::Struct {
            struct_type: Some(proto::DataType {
                kind: Some(proto::data_type::Kind::Udt(Box::new(
                    proto::data_type::Udt {
                        r#type: "udt".into(),
                        jvm_class: Some("org.apache.spark.ml.linalg.MatrixUDT".into()),
                        ..Default::default()
                    },
                ))),
            }),
            elements,
        }))
    }
}

/// ML parameter handling: stores name -> literal value mappings.
///
/// Mirrors the structure of parameter handling in `pyspark.ml.param.Params`.
#[derive(Debug, Clone, Default)]
pub struct Params {
    /// User-supplied parameters as name -> Literal mappings
    #[doc(hidden)]
    pub params: HashMap<String, proto::Expression>,
}

impl Params {
    /// Create a new empty Params collection.
    pub fn new() -> Self {
        Params {
            params: HashMap::new(),
        }
    }

    /// Set a parameter to an integer value.
    pub fn set_param_int(mut self, name: &str, value: i64) -> Self {
        let mut literal = proto::Expression::default();
        let mut lit = proto::expression::Literal::default();
        lit.literal_type = Some(proto::expression::literal::LiteralType::Long(value));
        literal.expr_type = Some(proto::expression::ExprType::Literal(lit));
        self.params.insert(name.to_string(), literal);
        self
    }

    /// Set a parameter to a 32-bit integer (`Integer`) literal, as PySpark sends a Python int
    /// that fits 32 bits; for a parameter whose JVM type is `Int`, such as `maxIter`.
    pub fn set_param_int32(mut self, name: &str, value: i32) -> Self {
        let literal = proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Integer(value)),
            ..Default::default()
        };
        self.params.insert(
            name.to_string(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Set a parameter to the narrowest integer literal that holds `value`, as PySpark sends a
    /// Python int: an `Integer` literal when it fits 32 bits, else a `Long`. The server converts
    /// it to the JVM type of the parameter (`Int` or `Long`, such as `seed`), as it converts the
    /// ints of PySpark.
    pub fn set_param_int_narrow(self, name: &str, value: i64) -> Self {
        match i32::try_from(value) {
            Ok(value) => self.set_param_int32(name, value),
            Err(_) => self.set_param_int(name, value),
        }
    }

    /// Set a parameter whose JVM type is `Array[String]` (such as `inputCols`), as PySpark sends
    /// a list of strings: an array literal of `string` elements.
    pub fn set_param_string_array<I, S>(mut self, name: &str, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let elements = values
            .into_iter()
            .map(|v| LiteralExpression::String(v.into()))
            .collect();
        let literal = LiteralExpression::Array {
            element_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            elements,
        }
        .to_literal_proto();
        self.params.insert(
            name.to_string(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Set a parameter to an array of 32-bit integers, as PySpark sends a list of ints.
    pub fn set_param_int32_array(self, name: &str, values: impl IntoIterator<Item = i32>) -> Self {
        self.set_param_int_array(name, values.into_iter().map(i64::from))
    }

    /// Set a parameter whose JVM type is `Array[Double]` (such as `thresholds`), as PySpark
    /// sends a list of floats: an array literal of `double` elements.
    pub fn set_param_double_array(
        mut self,
        name: &str,
        values: impl IntoIterator<Item = f64>,
    ) -> Self {
        let literal = LiteralExpression::Array {
            element_type: Box::new(DataType::Double),
            elements: values.into_iter().map(LiteralExpression::Double).collect(),
        }
        .to_literal_proto();
        self.params.insert(
            name.to_string(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Set a parameter to an integer array, preserving PySpark's literal width.
    #[allow(deprecated)] // Matches the array representation of PySpark's ML parameters.
    pub fn set_param_int_array(
        mut self,
        name: &str,
        values: impl IntoIterator<Item = i64>,
    ) -> Self {
        use proto::expression::literal::LiteralType as L;
        let values: Vec<i64> = values.into_iter().collect();
        let narrow = values.iter().all(|v| i32::try_from(*v).is_ok());
        let elements = values
            .into_iter()
            .map(|v| proto::expression::Literal {
                literal_type: Some(if narrow {
                    L::Integer(v as i32)
                } else {
                    L::Long(v)
                }),
                ..Default::default()
            })
            .collect();
        let literal = proto::expression::Literal {
            literal_type: Some(L::Array(proto::expression::literal::Array {
                element_type: Some(
                    if narrow {
                        DataType::Integer
                    } else {
                        DataType::Long
                    }
                    .to_proto(),
                ),
                elements,
            })),
            ..Default::default()
        };
        self.params.insert(
            name.into(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Set a vector parameter using Spark's VectorUDT representation.
    pub fn set_param_vector(mut self, name: &str, value: impl Into<MlVector>) -> Self {
        self.params.insert(
            name.into(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(
                    value.into().to_literal(),
                )),
                ..Default::default()
            },
        );
        self
    }

    /// Set a matrix parameter using Spark's MatrixUDT representation, as PySpark sends a
    /// `DenseMatrix` or a `SparseMatrix` (see [`MlMatrix`]).
    pub fn set_param_matrix(mut self, name: &str, value: impl Into<MlMatrix>) -> Self {
        self.params.insert(
            name.into(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(
                    value.into().to_literal(),
                )),
                ..Default::default()
            },
        );
        self
    }

    /// Set a parameter to a double value.
    pub fn set_param_double(mut self, name: &str, value: f64) -> Self {
        let mut literal = proto::Expression::default();
        let mut lit = proto::expression::Literal::default();
        lit.literal_type = Some(proto::expression::literal::LiteralType::Double(value));
        literal.expr_type = Some(proto::expression::ExprType::Literal(lit));
        self.params.insert(name.to_string(), literal);
        self
    }

    /// Set a parameter to a string value.
    pub fn set_param_string(mut self, name: &str, value: &str) -> Self {
        let mut literal = proto::Expression::default();
        let mut lit = proto::expression::Literal::default();
        lit.literal_type = Some(proto::expression::literal::LiteralType::String(
            value.to_string(),
        ));
        literal.expr_type = Some(proto::expression::ExprType::Literal(lit));
        self.params.insert(name.to_string(), literal);
        self
    }

    /// Set a parameter to a boolean value.
    pub fn set_param_bool(mut self, name: &str, value: bool) -> Self {
        let mut literal = proto::Expression::default();
        let mut lit = proto::expression::Literal::default();
        lit.literal_type = Some(proto::expression::literal::LiteralType::Boolean(value));
        literal.expr_type = Some(proto::expression::ExprType::Literal(lit));
        self.params.insert(name.to_string(), literal);
        self
    }

    /// A copy of the params whose names `keep` keeps: the params that a stage's class has, of a
    /// fitted model (PySpark's `_copyValues` copies only those).
    pub fn retain_names(&self, keep: impl Fn(&str) -> bool) -> Self {
        Params {
            params: self
                .params
                .iter()
                .filter(|(name, _)| keep(name))
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        }
    }

    /// Remove a parameter, so that it is not sent and the server uses its default, as
    /// PySpark's `Params.clear` does.
    pub fn clear_param(mut self, name: &str) -> Self {
        self.params.remove(name);
        self
    }

    /// Set a parameter to a null, as PySpark sends a None value of a param it was given.
    pub fn set_param_null(mut self, name: &str) -> Self {
        let literal = proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Null(
                DataType::Null.to_proto(),
            )),
            ..Default::default()
        };
        self.params.insert(
            name.to_string(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Whether a parameter is an array without elements. PySpark cannot infer the element type
    /// of such an array, so it cannot send the params that hold one.
    pub fn has_empty_array(&self) -> bool {
        fn cannot_infer(lit: &proto::expression::Literal) -> bool {
            match &lit.literal_type {
                Some(proto::expression::literal::LiteralType::Array(array)) => {
                    array.elements.first().is_none_or(cannot_infer)
                }
                _ => false,
            }
        }
        self.params.values().any(|expr| match &expr.expr_type {
            Some(proto::expression::ExprType::Literal(lit)) => cannot_infer(lit),
            _ => false,
        })
    }

    /// Set a parameter to a literal, preserving its wire type.
    pub fn set_param_literal(mut self, name: &str, literal: proto::expression::Literal) -> Self {
        self.params.insert(
            name.into(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Set nested double arrays, such as Bucketizer's per-column splits.
    pub fn set_param_double_arrays(
        self,
        name: &str,
        values: impl IntoIterator<Item = impl IntoIterator<Item = f64>>,
    ) -> Self {
        let literal = LiteralExpression::Array {
            element_type: Box::new(DataType::Array {
                element_type: Box::new(DataType::Double),
                contains_null: true,
            }),
            elements: values
                .into_iter()
                .map(|values| LiteralExpression::Array {
                    element_type: Box::new(DataType::Double),
                    elements: values.into_iter().map(LiteralExpression::Double).collect(),
                })
                .collect(),
        }
        .to_literal_proto();
        self.set_param_literal(name, literal)
    }

    /// Get a parameter value.
    pub fn get_param(&self, name: &str) -> Option<&proto::Expression> {
        self.params.get(name)
    }

    /// The literal value of the parameter `name`, if it is set.
    #[doc(hidden)]
    pub fn literal(&self, name: &str) -> Option<&proto::expression::literal::LiteralType> {
        match &self.params.get(name)?.expr_type {
            Some(proto::expression::ExprType::Literal(literal)) => literal.literal_type.as_ref(),
            _ => None,
        }
    }

    /// The string parameter `name`, or `default` when it is not set.
    #[doc(hidden)]
    pub fn string_or<'a>(&'a self, name: &str, default: &'a str) -> &'a str {
        self.string_opt(name).unwrap_or(default)
    }

    /// The `outputCol` parameter, or Spark's default of `HasOutputCol`, `<uid>__output`, of the
    /// object of `uid`, the column that the server writes when it is not set.
    #[doc(hidden)]
    pub fn output_col_or_default(&self, uid: &str) -> String {
        self.string_opt("outputCol")
            .map_or_else(|| format!("{uid}__output"), str::to_string)
    }

    /// The string parameter `name`, if it is set.
    #[doc(hidden)]
    pub fn string_opt(&self, name: &str) -> Option<&str> {
        match self.literal(name)? {
            proto::expression::literal::LiteralType::String(value) => Some(value),
            _ => None,
        }
    }

    /// The integer parameter `name`, if it is set, as the server reads it into a JVM `Long`.
    #[doc(hidden)]
    pub fn long_opt(&self, name: &str) -> Option<i64> {
        use proto::expression::literal::LiteralType as L;
        match self.literal(name)? {
            L::Integer(value) => Some(i64::from(*value)),
            L::Long(value) => Some(*value),
            // As the JVM's `Double.longValue`, which saturates.
            L::Double(value) => Some(*value as i64),
            _ => None,
        }
    }

    /// The integer parameter `name`, or `default` when it is not set, as the server reads it
    /// into a JVM `Int`: a `Long` wraps to 32 bits, as the JVM's `Long.intValue` does.
    #[doc(hidden)]
    pub fn int_or(&self, name: &str, default: i32) -> i32 {
        use proto::expression::literal::LiteralType as L;
        match self.literal(name) {
            Some(L::Integer(value)) => *value,
            Some(L::Long(value)) => *value as i32,
            Some(L::Double(value)) => *value as i32,
            _ => default,
        }
    }

    /// The floating-point parameter `name`, or `default` when it is not set.
    #[doc(hidden)]
    pub fn double_or(&self, name: &str, default: f64) -> f64 {
        use proto::expression::literal::LiteralType as L;
        match self.literal(name) {
            Some(L::Double(value)) => *value,
            Some(L::Integer(value)) => f64::from(*value),
            Some(L::Long(value)) => *value as f64,
            _ => default,
        }
    }

    /// The boolean parameter `name`, or `default` when it is not set.
    #[doc(hidden)]
    pub fn bool_or(&self, name: &str, default: bool) -> bool {
        match self.literal(name) {
            Some(proto::expression::literal::LiteralType::Boolean(value)) => *value,
            _ => default,
        }
    }

    /// The elements of the array parameter `name`, if it is set.
    #[doc(hidden)]
    pub fn array_opt(&self, name: &str) -> Option<&[proto::expression::Literal]> {
        match self.literal(name)? {
            proto::expression::literal::LiteralType::Array(array) => Some(&array.elements),
            _ => None,
        }
    }

    /// The string array parameter `name`, empty when it is not set.
    #[doc(hidden)]
    pub fn string_array(&self, name: &str) -> Vec<String> {
        self.array_opt(name)
            .unwrap_or_default()
            .iter()
            .filter_map(|element| match &element.literal_type {
                Some(proto::expression::literal::LiteralType::String(value)) => Some(value.clone()),
                _ => None,
            })
            .collect()
    }

    /// The floating-point array parameter `name`, if it is set.
    #[doc(hidden)]
    pub fn double_array_opt(&self, name: &str) -> Option<Vec<f64>> {
        let elements = self.array_opt(name)?;
        Some(
            elements
                .iter()
                .filter_map(|element| match element.literal_type {
                    Some(proto::expression::literal::LiteralType::Double(value)) => Some(value),
                    _ => None,
                })
                .collect(),
        )
    }
}

// The optional getters below are read by the rows that scripts/gen_rust_params_table.py prints
// for each converted class. Some have no row of the classes converted so far, so the allow stays.
#[allow(dead_code)]
impl Params {
    /// The integer parameter `name`, if it is set, as the server reads it into a JVM `Int`.
    #[doc(hidden)]
    pub fn int_opt(&self, name: &str) -> Option<i32> {
        use proto::expression::literal::LiteralType as L;
        match self.literal(name)? {
            L::Integer(value) => Some(*value),
            L::Long(value) => Some(*value as i32),
            L::Double(value) => Some(*value as i32),
            _ => None,
        }
    }

    /// The floating-point parameter `name`, if it is set.
    #[doc(hidden)]
    pub fn double_opt(&self, name: &str) -> Option<f64> {
        use proto::expression::literal::LiteralType as L;
        match self.literal(name)? {
            L::Double(value) => Some(*value),
            L::Integer(value) => Some(f64::from(*value)),
            L::Long(value) => Some(*value as f64),
            _ => None,
        }
    }

    /// The boolean parameter `name`, if it is set.
    fn bool_opt(&self, name: &str) -> Option<bool> {
        match self.literal(name)? {
            proto::expression::literal::LiteralType::Boolean(value) => Some(*value),
            _ => None,
        }
    }

    /// The string array parameter `name`, if it is set.
    #[doc(hidden)]
    pub fn string_array_opt(&self, name: &str) -> Option<Vec<String>> {
        let elements = self.array_opt(name)?;
        Some(
            elements
                .iter()
                .filter_map(|element| match &element.literal_type {
                    Some(proto::expression::literal::LiteralType::String(value)) => {
                        Some(value.clone())
                    }
                    _ => None,
                })
                .collect(),
        )
    }

    /// The integer array parameter `name`, if it is set, as the server reads it into `Int`s.
    #[doc(hidden)]
    pub fn int_array_opt(&self, name: &str) -> Option<Vec<i32>> {
        let elements = self.array_opt(name)?;
        Some(
            elements
                .iter()
                .filter_map(|element| match element.literal_type {
                    Some(proto::expression::literal::LiteralType::Integer(value)) => Some(value),
                    Some(proto::expression::literal::LiteralType::Long(value)) => {
                        Some(value as i32)
                    }
                    _ => None,
                })
                .collect(),
        )
    }

    /// The nested floating-point array parameter `name`, if it is set.
    fn double_arrays_opt(&self, name: &str) -> Option<Vec<Vec<f64>>> {
        use proto::expression::literal::LiteralType as L;
        let outer = self.array_opt(name)?;
        Some(
            outer
                .iter()
                .map(|element| match &element.literal_type {
                    Some(L::Array(inner)) => inner
                        .elements
                        .iter()
                        .filter_map(|value| match value.literal_type {
                            Some(L::Double(value)) => Some(value),
                            _ => None,
                        })
                        .collect(),
                    _ => Vec::new(),
                })
                .collect(),
        )
    }
}

impl Params {
    /// Convert to proto MlParams for transmission. MlParams stores `Literal`
    /// values, so we unwrap the literal from each stored Expression.
    pub fn to_proto(&self) -> proto::MlParams {
        let mut ml_params = proto::MlParams::default();
        for (name, expr) in &self.params {
            if let Some(proto::expression::ExprType::Literal(lit)) = &expr.expr_type {
                ml_params.params.insert(name.clone(), lit.clone());
            }
        }
        ml_params
    }

    /// Convert from proto MlParams (wrapping each Literal back into an Expression).
    pub fn from_proto(proto_params: &proto::MlParams) -> Self {
        let mut params = HashMap::new();
        for (name, lit) in &proto_params.params {
            let mut e = proto::Expression::default();
            e.expr_type = Some(proto::expression::ExprType::Literal(lit.clone()));
            params.insert(name.clone(), e);
        }
        Params { params }
    }
}

/// ML Operator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorType {
    /// An estimator that learns from data via fit().
    Estimator,
    /// A transformer that applies transformations (possibly stateless).
    Transformer,
    /// An evaluator that measures model quality.
    Evaluator,
    /// A fitted model (result of fit).
    Model,
}

impl OperatorType {
    /// Convert to proto OperatorType.
    pub fn to_proto(&self) -> proto::ml_operator::OperatorType {
        match self {
            OperatorType::Estimator => proto::ml_operator::OperatorType::Estimator,
            OperatorType::Transformer => proto::ml_operator::OperatorType::Transformer,
            OperatorType::Evaluator => proto::ml_operator::OperatorType::Evaluator,
            OperatorType::Model => proto::ml_operator::OperatorType::Model,
        }
    }

    /// Convert from proto OperatorType.
    pub fn from_proto(proto_type: i32) -> Self {
        match proto::ml_operator::OperatorType::try_from(proto_type) {
            Ok(proto::ml_operator::OperatorType::Estimator) => OperatorType::Estimator,
            Ok(proto::ml_operator::OperatorType::Transformer) => OperatorType::Transformer,
            Ok(proto::ml_operator::OperatorType::Evaluator) => OperatorType::Evaluator,
            Ok(proto::ml_operator::OperatorType::Model) => OperatorType::Model,
            _ => OperatorType::Transformer,
        }
    }
}

/// ML Operator represents an ML class (Estimator, Transformer, Model, or Evaluator).
///
/// Mirrors `spark.connect.MlOperator` protobuf.
#[derive(Debug, Clone)]
pub struct MlOperator {
    /// Qualified class name (e.g., "org.apache.spark.ml.feature.StandardScaler")
    pub name: String,
    /// Unique identifier for this operator instance
    pub uid: String,
    /// Type of operator
    pub op_type: OperatorType,
}

impl MlOperator {
    /// Create a new MlOperator with an auto-generated UID: the simple name of the class and
    /// twelve random hex digits (`KMeans_1b2c3d4e5f60`), as PySpark's `Identifiable._randomUID`
    /// and Spark's `Identifiable.randomUID` make it, so that server messages name the class.
    pub fn new(name: &str, op_type: OperatorType) -> Self {
        let class = name.rsplit('.').next().unwrap_or(name);
        let hex = Uuid::new_v4().simple().to_string();
        MlOperator {
            name: name.to_string(),
            uid: format!("{class}_{}", &hex[hex.len() - 12..]),
            op_type,
        }
    }

    /// Create with a specific UID.
    pub fn with_uid(name: &str, uid: &str, op_type: OperatorType) -> Self {
        MlOperator {
            name: name.to_string(),
            uid: uid.to_string(),
            op_type,
        }
    }

    /// Convert to proto MlOperator.
    pub fn to_proto(&self) -> proto::MlOperator {
        proto::MlOperator {
            name: self.name.clone(),
            uid: self.uid.clone(),
            r#type: self.op_type.to_proto() as i32,
        }
    }

    /// Convert from proto MlOperator.
    pub fn from_proto(proto_op: &proto::MlOperator) -> Self {
        MlOperator {
            name: proto_op.name.clone(),
            uid: proto_op.uid.clone(),
            op_type: OperatorType::from_proto(proto_op.r#type),
        }
    }
}

/// The command that fits `estimator`, with `params`, to `dataset`.
pub fn fit(
    estimator: &MlOperator,
    params: &Params,
    dataset: proto::Relation,
) -> proto::command::CommandType {
    proto::command::CommandType::MlCommand(proto::MlCommand {
        command: Some(proto::ml_command::Command::Fit(proto::ml_command::Fit {
            estimator: Some(estimator.to_proto()),
            params: Some(params.to_proto()),
            dataset: Some(dataset),
        })),
    })
}

/// The command that evaluates `dataset` with `evaluator` and `params`.
pub fn evaluate(
    evaluator: &MlOperator,
    params: &Params,
    dataset: proto::Relation,
) -> proto::command::CommandType {
    proto::command::CommandType::MlCommand(proto::MlCommand {
        command: Some(proto::ml_command::Command::Evaluate(
            proto::ml_command::Evaluate {
                evaluator: Some(evaluator.to_proto()),
                params: Some(params.to_proto()),
                dataset: Some(dataset),
            },
        )),
    })
}

/// The relation that applies `operator`, with `params`, to `input`.
pub fn transform(
    operator: proto::ml_relation::transform::Operator,
    params: &Params,
    input: proto::Relation,
) -> proto::MlRelation {
    proto::MlRelation {
        ml_type: Some(proto::ml_relation::MlType::Transform(Box::new(
            proto::ml_relation::Transform {
                operator: Some(operator),
                input: Some(Box::new(input)),
                params: Some(params.to_proto()),
            },
        ))),
        ..Default::default()
    }
}

/// The strings of `literal`, an array literal of strings, as the ML helper returns a `String[]`.
pub fn ml_string_array(literal: proto::expression::Literal) -> crate::error::Result<Vec<String>> {
    use proto::expression::literal::LiteralType as L;
    let Some(L::Array(array)) = literal.literal_type else {
        return Err(SparkError::connect_msg("expected an array of strings"));
    };
    array
        .elements
        .into_iter()
        .map(|element| match element.literal_type {
            Some(L::String(value)) => Ok(value),
            _ => Err(SparkError::connect_msg("expected an array of strings")),
        })
        .collect()
}
