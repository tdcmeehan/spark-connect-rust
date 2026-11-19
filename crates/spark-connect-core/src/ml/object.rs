// SPDX-License-Identifier: Apache-2.0
//! The values that the methods of Spark Connect ML objects take and return, and the commands
//! and relations that fetch those methods and save and load the objects.

use super::*;

/// The id of Spark Connect ML's helper object, whose methods (`invoke_helper_attr`,
/// `invoke_helper_relation` of `pyspark.ml.util`) the server runs without a cached object.
pub const ML_CONNECT_HELPER_ID: &str = "______ML_CONNECT_HELPER______";

impl MlMatrix {
    /// The matrix of the struct of its UDT, as `deserialize_param` reads a `MatrixUDT`: its type
    /// (0 sparse, 1 dense), its shape, its column pointers and row indices (of a sparse matrix),
    /// its values and whether it is transposed.
    #[doc(hidden)]
    pub fn from_literal(value: &proto::expression::literal::LiteralType) -> Option<Self> {
        use proto::expression::literal::{specialized_array::ValueType as A, LiteralType as L};
        let L::Struct(value) = value else {
            return None;
        };
        let fields = &value.elements;
        let int = |i: usize| match fields.get(i)?.literal_type.as_ref()? {
            L::Integer(v) => Some(*v),
            _ => None,
        };
        let ints = |i: usize| match fields.get(i)?.literal_type.as_ref()? {
            L::SpecializedArray(array) => match array.value_type.as_ref()? {
                A::Ints(values) => Some(values.values.clone()),
                _ => None,
            },
            _ => None,
        };
        let values = match fields.get(5)?.literal_type.as_ref()? {
            L::SpecializedArray(array) => match array.value_type.as_ref()? {
                A::Doubles(values) => values.values.clone(),
                _ => return None,
            },
            _ => return None,
        };
        let is_transposed = match fields.get(6)?.literal_type.as_ref()? {
            L::Boolean(v) => *v,
            _ => return None,
        };
        match fields.first()?.literal_type.as_ref()? {
            L::Byte(1) => Some(Self::Dense {
                num_rows: int(1)?,
                num_cols: int(2)?,
                values,
                is_transposed,
            }),
            L::Byte(0) => Some(Self::Sparse {
                num_rows: int(1)?,
                num_cols: int(2)?,
                col_ptrs: ints(3)?,
                row_indices: ints(4)?,
                values,
                is_transposed,
            }),
            _ => None,
        }
    }
}

/// A value that the server returns of a method, or that a method takes as an argument: the types
/// of `pyspark.ml.connect.serialize.deserialize_param`, which reads an ML vector or matrix of its
/// UDT struct and another literal as `LiteralExpression._to_value` does.
#[derive(Debug, Clone, PartialEq)]
pub enum MlValue {
    Null,
    Bool(bool),
    Byte(i32),
    Short(i32),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    /// The decimal of its string value.
    Decimal(String),
    String(String),
    Binary(Vec<u8>),
    /// Days since the epoch.
    Date(i32),
    /// Microseconds since the epoch, in UTC.
    Timestamp(i64),
    /// Microseconds since the epoch, without a time zone.
    TimestampNtz(i64),
    Vector(MlVector),
    Matrix(MlMatrix),
    Array(Vec<MlValue>),
    Map(Vec<(MlValue, MlValue)>),
    /// A struct of its field names and values: a struct literal that is neither a vector nor a
    /// matrix, for which PySpark's `deserialize_param` raises `ValueError("Unknown UDT ...")`.
    Struct(Vec<(String, MlValue)>),
}

impl MlValue {
    /// The value of `literal`, as `deserialize_param` reads it: the struct of a `VectorUDT` or a
    /// `MatrixUDT` is a vector or a matrix, a struct of another UDT is an error (PySpark's
    /// `ValueError` of an unknown UDT), and another literal is its value.
    pub fn from_literal(literal: &proto::expression::Literal) -> crate::error::Result<Self> {
        use proto::expression::literal::{specialized_array::ValueType as A, LiteralType as L};
        let unsupported =
            || SparkError::value_msg(format!("Unsupported literal of an ML result: {literal:?}"));
        let Some(value) = literal.literal_type.as_ref() else {
            return Err(unsupported());
        };
        Ok(match value {
            L::Null(_) => Self::Null,
            L::Boolean(v) => Self::Bool(*v),
            L::Byte(v) => Self::Byte(*v),
            L::Short(v) => Self::Short(*v),
            L::Integer(v) => Self::Int(*v),
            L::Long(v) => Self::Long(*v),
            L::Float(v) => Self::Float(*v),
            L::Double(v) => Self::Double(*v),
            L::Decimal(v) => Self::Decimal(v.value.clone()),
            L::String(v) => Self::String(v.clone()),
            L::Binary(v) => Self::Binary(v.to_vec()),
            L::Date(v) => Self::Date(*v),
            L::Timestamp(v) => Self::Timestamp(*v),
            L::TimestampNtz(v) => Self::TimestampNtz(*v),
            L::Array(array) => Self::Array(
                array
                    .elements
                    .iter()
                    .map(Self::from_literal)
                    .collect::<crate::error::Result<_>>()?,
            ),
            L::Map(map) => Self::Map(
                map.keys
                    .iter()
                    .zip(&map.values)
                    .map(|(k, v)| Ok((Self::from_literal(k)?, Self::from_literal(v)?)))
                    .collect::<crate::error::Result<_>>()?,
            ),
            L::SpecializedArray(array) => match array.value_type.as_ref() {
                Some(A::Bools(v)) => Self::Array(v.values.iter().map(|x| Self::Bool(*x)).collect()),
                Some(A::Ints(v)) => Self::Array(v.values.iter().map(|x| Self::Int(*x)).collect()),
                Some(A::Longs(v)) => Self::Array(v.values.iter().map(|x| Self::Long(*x)).collect()),
                Some(A::Floats(v)) => {
                    Self::Array(v.values.iter().map(|x| Self::Float(*x)).collect())
                }
                Some(A::Doubles(v)) => {
                    Self::Array(v.values.iter().map(|x| Self::Double(*x)).collect())
                }
                Some(A::Strings(v)) => {
                    Self::Array(v.values.iter().map(|x| Self::String(x.clone())).collect())
                }
                None => return Err(unsupported()),
            },
            L::Struct(s) => {
                #[allow(deprecated)] // PySpark's ML serializer types a UDT struct by struct_type.
                let udt = match s.struct_type.as_ref().and_then(|t| t.kind.as_ref()) {
                    Some(proto::data_type::Kind::Udt(udt)) => udt.jvm_class.clone(),
                    _ => None,
                };
                match udt.as_deref() {
                    Some("org.apache.spark.ml.linalg.VectorUDT") => {
                        Self::Vector(MlVector::from_literal(value).ok_or_else(unsupported)?)
                    }
                    Some("org.apache.spark.ml.linalg.MatrixUDT") => {
                        Self::Matrix(MlMatrix::from_literal(value).ok_or_else(unsupported)?)
                    }
                    Some(other) => {
                        return Err(SparkError::value_msg(format!("Unknown UDT {other}")))
                    }
                    None => {
                        #[allow(deprecated)]
                        let names: Vec<String> =
                            match s.struct_type.as_ref().and_then(|t| t.kind.as_ref()) {
                                Some(proto::data_type::Kind::Struct(st)) => {
                                    st.fields.iter().map(|f| f.name.clone()).collect()
                                }
                                _ => Vec::new(),
                            };
                        Self::Struct(
                            s.elements
                                .iter()
                                .enumerate()
                                .map(|(i, v)| {
                                    let name = names.get(i).cloned().unwrap_or_default();
                                    Ok((name, Self::from_literal(v)?))
                                })
                                .collect::<crate::error::Result<_>>()?,
                        )
                    }
                }
            }
            _ => return Err(unsupported()),
        })
    }

    /// The literal of the value, as PySpark's `serialize_param` sends an argument: a vector or a
    /// matrix as its UDT struct, a scalar of its type, and an array of the type of its first
    /// element, as `LiteralExpression._infer_type` types a list. An array that is empty, or whose
    /// first element is null, cannot be typed (`CANNOT_INFER_ARRAY_ELEMENT_TYPE`): send it as a
    /// `FetchArg::Typed` instead.
    pub fn to_literal(&self) -> crate::error::Result<proto::expression::Literal> {
        match self {
            Self::Vector(v) => Ok(v.to_literal()),
            Self::Matrix(m) => Ok(m.to_literal()),
            other => Ok(other.to_literal_expression(None)?.to_literal_proto()),
        }
    }

    /// The literal of the value as the type `data_type`, as PySpark sends a `(value, DataType)`
    /// pair of `serialize`, such as an empty list of strings.
    pub fn to_typed_literal(
        &self,
        data_type: &DataType,
    ) -> crate::error::Result<proto::expression::Literal> {
        Ok(self
            .to_literal_expression(Some(data_type))?
            .to_literal_proto())
    }

    fn to_literal_expression(
        &self,
        data_type: Option<&DataType>,
    ) -> crate::error::Result<LiteralExpression> {
        let cannot_infer = || {
            SparkError::classed(
                SparkErrorKind::TypeError,
                "CANNOT_INFER_ARRAY_ELEMENT_TYPE",
                &[],
            )
        };
        Ok(match self {
            Self::Null => LiteralExpression::Null(data_type.cloned().unwrap_or(DataType::Null)),
            Self::Bool(v) => LiteralExpression::Boolean(*v),
            Self::Byte(v) => LiteralExpression::Byte(*v),
            Self::Short(v) => LiteralExpression::Short(*v),
            Self::Int(v) => LiteralExpression::Integer(*v),
            Self::Long(v) => LiteralExpression::Long(*v),
            Self::Float(v) => LiteralExpression::Float(*v),
            Self::Double(v) => LiteralExpression::Double(*v),
            Self::Decimal(v) => {
                let (precision, scale) = match data_type {
                    Some(DataType::Decimal { precision, scale }) => (*precision, *scale),
                    _ => DECIMAL_TYPE_PRECISION_SCALE,
                };
                LiteralExpression::Decimal {
                    value: v.clone(),
                    precision,
                    scale,
                }
            }
            Self::String(v) => LiteralExpression::String(v.clone()),
            Self::Binary(v) => LiteralExpression::Binary(v.clone()),
            Self::Date(v) => LiteralExpression::Date(*v),
            Self::Timestamp(v) => LiteralExpression::Timestamp(*v),
            Self::TimestampNtz(v) => LiteralExpression::TimestampNtz(*v),
            Self::Array(elements) => {
                let element_type = match data_type {
                    Some(DataType::Array { element_type, .. }) => (**element_type).clone(),
                    Some(_) | None => match elements.first() {
                        None | Some(Self::Null) => return Err(cannot_infer()),
                        Some(first) => first.data_type()?,
                    },
                };
                LiteralExpression::Array {
                    elements: elements
                        .iter()
                        .map(|e| e.to_literal_expression(Some(&element_type)))
                        .collect::<crate::error::Result<_>>()?,
                    element_type: Box::new(element_type),
                }
            }
            Self::Vector(_) | Self::Matrix(_) | Self::Map(_) | Self::Struct(_) => {
                return Err(SparkError::value_msg(
                    "An ML vector, matrix, map or struct cannot be an element of an array \
                     argument",
                ))
            }
        })
    }

    /// The Spark type of the value, as `LiteralExpression._infer_type` types a value.
    fn data_type(&self) -> crate::error::Result<DataType> {
        Ok(match self {
            Self::Null => DataType::Null,
            Self::Bool(_) => DataType::Boolean,
            Self::Byte(_) => DataType::Byte,
            Self::Short(_) => DataType::Short,
            Self::Int(_) => DataType::Integer,
            Self::Long(_) => DataType::Long,
            Self::Float(_) => DataType::Float,
            Self::Double(_) => DataType::Double,
            Self::Decimal(_) => {
                let (precision, scale) = DECIMAL_TYPE_PRECISION_SCALE;
                DataType::Decimal { precision, scale }
            }
            Self::String(_) => DataType::String {
                collation: "UTF8_BINARY".to_string(),
            },
            Self::Binary(_) => DataType::Binary,
            Self::Date(_) => DataType::Date,
            Self::Timestamp(_) => DataType::Timestamp,
            Self::TimestampNtz(_) => DataType::TimestampNtz,
            Self::Array(elements) => match elements.first() {
                None | Some(Self::Null) => {
                    return Err(SparkError::classed(
                        SparkErrorKind::TypeError,
                        "CANNOT_INFER_ARRAY_ELEMENT_TYPE",
                        &[],
                    ))
                }
                Some(first) => DataType::Array {
                    element_type: Box::new(first.data_type()?),
                    contains_null: true,
                },
            },
            Self::Vector(_) | Self::Matrix(_) | Self::Map(_) | Self::Struct(_) => {
                return Err(SparkError::value_msg(
                    "An ML vector, matrix, map or struct cannot be an element of an array \
                     argument",
                ))
            }
        })
    }

    /// The value as a double, of a numeric value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Double(v) => Some(*v),
            Self::Float(v) => Some(f64::from(*v)),
            Self::Int(v) | Self::Byte(v) | Self::Short(v) => Some(f64::from(*v)),
            Self::Long(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// The value as a long, of an integral value.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Long(v) => Some(*v),
            Self::Int(v) | Self::Byte(v) | Self::Short(v) => Some(i64::from(*v)),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_vector(&self) -> Option<&MlVector> {
        match self {
            Self::Vector(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_matrix(&self) -> Option<&MlMatrix> {
        match self {
            Self::Matrix(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MlValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// The precision and scale of a decimal value of no type, as `LiteralExpression._infer_type` types
/// a `decimal.Decimal`: `DecimalType()`, of precision 10 and scale 0 whatever the digits of the
/// value, which the server widens to them. PySpark's `serialize_param` sends every decimal so,
/// including one that a server returned and a loaded param or a fetch argument sends again.
const DECIMAL_TYPE_PRECISION_SCALE: (i32, i32) = (10, 0);

/// The params that a `Read` returns, as PySpark sends them again: its load turns each param into a
/// Python value (`deserialize_param`), and its next save, fit or transform serializes that value
/// with the inference of `serialize_param`, of Python's one int and one float. So an integral
/// literal is an `Integer` when it fits in 32 bits, and a `Long` when it does not, and a `Float` is
/// a `Double`: a `seed` that the server reads back as a `Long` is saved again as an `Integer`, as
/// PySpark saves it. The other literals are serialized again as the value they read as, as
/// `serialize_param` infers it (a string array of UTF8_BINARY elements, a vector without its
/// Python class), and one that reads as no value is kept as it is.
#[doc(hidden)]
pub fn read_params(params: Option<&proto::MlParams>) -> Params {
    use proto::expression::literal::LiteralType as L;
    let Some(params) = params else {
        return Params::default();
    };
    let mut params = params.clone();
    for literal in params.params.values_mut() {
        let retyped = match literal.literal_type {
            Some(L::Byte(value)) | Some(L::Short(value)) => Some(L::Integer(value)),
            Some(L::Long(value)) => i32::try_from(value).ok().map(L::Integer),
            Some(L::Float(value)) => Some(L::Double(f64::from(value))),
            _ => None,
        };
        // A literal of the new type, as PySpark builds it of the Python value. A Long that does not
        // fit in 32 bits is a Python int of the Long range, and so are the other literals, inferred
        // by their values.
        let inferred = match retyped {
            Some(literal_type) => proto::expression::Literal {
                literal_type: Some(literal_type),
                ..Default::default()
            },
            None => match MlValue::from_literal(literal).and_then(|value| value.to_literal()) {
                Ok(inferred) => inferred,
                Err(_) => continue,
            },
        };
        *literal = inferred;
    }
    Params::from_proto(&params)
}

/// The object reference of Spark Connect ML's helper object ([`ML_CONNECT_HELPER_ID`]).
pub fn helper() -> proto::ObjectRef {
    proto::ObjectRef {
        id: ML_CONNECT_HELPER_ID.to_string(),
    }
}

/// An argument of a method of a fetch: a literal, or the relation of a DataFrame.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum MethodArg {
    Param(proto::expression::Literal),
    Input(proto::Relation),
}

/// The `Fetch` of the method `method`, with `args`, of the object that the methods `chain`
/// reach from `object`.
#[doc(hidden)]
pub fn fetch(
    object: proto::ObjectRef,
    chain: &[String],
    method: &str,
    args: Vec<MethodArg>,
) -> proto::Fetch {
    use proto::fetch::method::{args::ArgsType, Args};
    let mut methods: Vec<proto::fetch::Method> = chain
        .iter()
        .map(|m| proto::fetch::Method {
            method: m.clone(),
            args: Vec::new(),
        })
        .collect();
    let args = args
        .into_iter()
        .map(|arg| Args {
            args_type: Some(match arg {
                MethodArg::Param(literal) => ArgsType::Param(literal),
                MethodArg::Input(relation) => ArgsType::Input(relation),
            }),
        })
        .collect();
    methods.push(proto::fetch::Method {
        method: method.to_string(),
        args,
    });
    proto::Fetch {
        obj_ref: Some(object),
        methods,
    }
}

/// The command that runs `fetch`, whose result is the value of the method.
pub fn fetch_command(fetch: proto::Fetch) -> proto::command::CommandType {
    ml_command(proto::ml_command::Command::Fetch(fetch))
}

/// The command that recreates the summary of `model` from `dataset`, when the server has lost it
/// with the model it offloaded.
pub fn create_summary(
    model: proto::ObjectRef,
    dataset: proto::Relation,
) -> proto::command::CommandType {
    ml_command(proto::ml_command::Command::CreateSummary(
        proto::ml_command::CreateSummary {
            model_ref: Some(model),
            dataset: Some(dataset),
        },
    ))
}

/// The command that deletes the objects of `refs` that the server caches.
pub fn delete(refs: Vec<proto::ObjectRef>) -> proto::command::CommandType {
    ml_command(proto::ml_command::Command::Delete(
        proto::ml_command::Delete {
            obj_refs: refs,
            // As PySpark's `_delete_ml_cache`, whose `evict_only` is False.
            evict_only: Some(false),
        },
    ))
}

fn ml_command(command: proto::ml_command::Command) -> proto::command::CommandType {
    proto::command::CommandType::MlCommand(proto::MlCommand {
        command: Some(command),
    })
}

/// The relation of the DataFrame that `fetch` returns. `summary_dataset` is the dataset from
/// which the server recreates a summary that `fetch` reaches, when it has lost it.
pub fn fetch_relation(
    fetch: proto::Fetch,
    summary_dataset: Option<proto::Relation>,
) -> proto::MlRelation {
    proto::MlRelation {
        ml_type: Some(proto::ml_relation::MlType::Fetch(fetch)),
        model_summary_dataset: summary_dataset.map(Box::new),
    }
}

/// The command that saves `operator` (not a fitted model), with `params`, to `path`.
pub fn write_operator(
    operator: &MlOperator,
    params: &Params,
    path: &str,
    overwrite: bool,
    options: &HashMap<String, String>,
) -> proto::command::CommandType {
    write_command(
        proto::ml_command::write::Type::Operator(operator.to_proto()),
        params,
        path,
        overwrite,
        options,
    )
}

/// The command that saves the server-side object `object` (a fitted model), with `params`, to
/// `path`.
pub fn write_object(
    object: proto::ObjectRef,
    params: &Params,
    path: &str,
    overwrite: bool,
    options: &HashMap<String, String>,
) -> proto::command::CommandType {
    write_command(
        proto::ml_command::write::Type::ObjRef(object),
        params,
        path,
        overwrite,
        options,
    )
}

fn write_command(
    r#type: proto::ml_command::write::Type,
    params: &Params,
    path: &str,
    overwrite: bool,
    options: &HashMap<String, String>,
) -> proto::command::CommandType {
    ml_command(proto::ml_command::Command::Write(
        proto::ml_command::Write {
            r#type: Some(r#type),
            params: Some(params.to_proto()),
            path: path.to_string(),
            should_overwrite: Some(overwrite),
            options: options.clone(),
        },
    ))
}

/// The command that loads the operator or model of the JVM class `jvm_class`, of the type
/// `op_type`, from `path`.
pub fn read(jvm_class: &str, op_type: OperatorType, path: &str) -> proto::command::CommandType {
    ml_command(proto::ml_command::Command::Read(proto::ml_command::Read {
        operator: Some(proto::MlOperator {
            name: jvm_class.to_string(),
            uid: String::new(),
            r#type: op_type.to_proto() as i32,
        }),
        path: path.to_string(),
    }))
}
