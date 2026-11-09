//! The operators of the ML API, and the commands and relations that fit, evaluate and
//! apply them, and delete the models that the server caches.

use std::collections::HashMap;

use spark_connect_proto as proto;
use uuid::Uuid;

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

    /// Encode an ML parameter whose JVM type is a 32-bit integer.
    #[doc(hidden)]
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

    /// Get a parameter value.
    pub fn get_param(&self, name: &str) -> Option<&proto::Expression> {
        self.params.get(name)
    }

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
    /// Create a new MlOperator with auto-generated UID.
    pub fn new(name: &str, op_type: OperatorType) -> Self {
        MlOperator {
            name: name.to_string(),
            uid: Uuid::new_v4().to_string(),
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

/// The command that deletes the objects of `refs` that the server caches.
pub fn delete(refs: Vec<proto::ObjectRef>) -> proto::command::CommandType {
    proto::command::CommandType::MlCommand(proto::MlCommand {
        command: Some(proto::ml_command::Command::Delete(
            proto::ml_command::Delete {
                obj_refs: refs,
                evict_only: None,
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
