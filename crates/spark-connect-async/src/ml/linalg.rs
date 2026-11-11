//! The vectors and matrices of Spark ML, as results and columns hold them.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use spark_connect_proto as proto;

use proto::expression::literal::specialized_array::ValueType;
use proto::expression::literal::LiteralType;
use proto::expression::Literal;

use crate::error::{Error, Result};
use crate::value::Value;

/// A vector of Spark ML, such as a column of features or of probabilities holds.
///
/// A column of vectors reads into a `Vector` with [`collect_as`](crate::DataFrame::collect_as),
/// or into an `Option<Vector>` when it has `NULL`s; a [`Value`] of one converts with
/// `Vector::try_from`.
#[derive(Debug, Clone, PartialEq)]
pub enum Vector {
    /// Every value.
    Dense(Vec<f64>),
    /// A vector of `size` values that are 0 except at `indices`, in increasing order, where
    /// they are `values`.
    Sparse {
        /// The number of values.
        size: usize,
        /// The positions of the values that are not 0.
        indices: Vec<usize>,
        /// The values at `indices`.
        values: Vec<f64>,
    },
}

impl Vector {
    /// The number of values, the zeros of a sparse vector included.
    pub fn size(&self) -> usize {
        match self {
            Vector::Dense(values) => values.len(),
            Vector::Sparse { size, .. } => *size,
        }
    }

    /// Every value, the zeros of a sparse vector included.
    pub fn to_dense(&self) -> Vec<f64> {
        match self {
            Vector::Dense(values) => values.clone(),
            Vector::Sparse {
                size,
                indices,
                values,
            } => {
                let mut dense = vec![0.0; *size];
                for (&index, &value) in indices.iter().zip(values) {
                    dense[index] = value;
                }
                dense
            }
        }
    }

    /// The vector of the fields of Spark's `VectorUDT`: its type (0 sparse, 1 dense), size,
    /// indices and values.
    fn of_fields(
        kind: i8,
        size: Option<i32>,
        indices: Option<Vec<i32>>,
        values: Option<Vec<f64>>,
    ) -> std::result::Result<Vector, String> {
        let values = values.ok_or("a vector without values")?;
        match kind {
            1 => Ok(Vector::Dense(values)),
            0 => {
                let size = count(size.ok_or("a sparse vector without a size")?)?;
                let indices = indices
                    .ok_or("a sparse vector without indices")?
                    .into_iter()
                    .map(count)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                if indices.len() != values.len() || indices.iter().any(|&index| index >= size) {
                    return Err(format!(
                        "a sparse vector of size {size} with {} indices and {} values",
                        indices.len(),
                        values.len()
                    ));
                }
                Ok(Vector::Sparse {
                    size,
                    indices,
                    values,
                })
            }
            other => Err(format!("a vector of the unknown type {other}")),
        }
    }
}

/// A matrix of Spark ML, such as the coefficients of a logistic regression of several classes,
/// one row per class.
#[derive(Debug, Clone, PartialEq)]
pub enum Matrix {
    /// Every value, column after column, or row after row if `is_transposed`.
    Dense {
        /// The number of rows.
        num_rows: usize,
        /// The number of columns.
        num_cols: usize,
        /// The values.
        values: Vec<f64>,
        /// Whether `values` are row after row.
        is_transposed: bool,
    },
    /// The values that are not 0, by column (compressed sparse column), or by row if
    /// `is_transposed`: those of column `j` are `values[col_ptrs[j]..col_ptrs[j + 1]]`, in the
    /// rows `row_indices` at the same positions.
    Sparse {
        /// The number of rows.
        num_rows: usize,
        /// The number of columns.
        num_cols: usize,
        /// Where the values of each column start in `values`, and the end of the last.
        col_ptrs: Vec<usize>,
        /// The row of each value.
        row_indices: Vec<usize>,
        /// The values.
        values: Vec<f64>,
        /// Whether the roles of rows and columns are swapped.
        is_transposed: bool,
    },
}

/// `value`, a size or an index that Spark sends as an `INT`.
pub(crate) fn count(value: i32) -> std::result::Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("a negative size or index {value}"))
}

#[derive(Deserialize)]
struct VectorFields {
    #[serde(rename = "type")]
    kind: i8,
    size: Option<i32>,
    indices: Option<Vec<i32>>,
    values: Option<Vec<f64>>,
}

impl<'de> Deserialize<'de> for Vector {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let fields = VectorFields::deserialize(deserializer)?;
        Vector::of_fields(fields.kind, fields.size, fields.indices, fields.values)
            .map_err(D::Error::custom)
    }
}

impl TryFrom<&Value> for Vector {
    type Error = Error;

    /// The vector of a value of a column of vectors, which is a struct of Spark's `VectorUDT`
    /// fields. A `NULL` is not a vector.
    fn try_from(value: &Value) -> Result<Vector> {
        let Value::Struct(row) = value else {
            return Err(Error::decode(format!("{value:?} is not a vector")));
        };
        let field = |name: &str| {
            row.get_by_name(name)
                .ok_or_else(|| Error::decode("a vector without this field").in_field(name))
        };
        let Value::Byte(kind) = field("type")? else {
            return Err(Error::decode("the type of a vector is not a TINYINT").in_field("type"));
        };
        let size = match field("size")? {
            Value::Null => None,
            Value::Int(size) => Some(*size),
            _ => return Err(Error::decode("the size of a vector is not an INT").in_field("size")),
        };
        let indices = numbers(field("indices")?, |value| match value {
            Value::Int(index) => Some(*index),
            _ => None,
        })
        .map_err(|error| error.in_field("indices"))?;
        let values = numbers(field("values")?, |value| match value {
            Value::Double(value) => Some(*value),
            _ => None,
        })
        .map_err(|error| error.in_field("values"))?;
        Vector::of_fields(*kind, size, indices, values).map_err(Error::decode)
    }
}

/// The elements of `value`, an array of numbers that `number` reads, or `None` for a `NULL`.
fn numbers<T>(value: &Value, number: fn(&Value) -> Option<T>) -> Result<Option<Vec<T>>> {
    match value {
        Value::Null => Ok(None),
        Value::Array(elements) => elements
            .iter()
            .enumerate()
            .map(|(index, element)| {
                number(element).ok_or_else(|| {
                    Error::decode(format!("{element:?} is not a number of a vector"))
                        .in_element(index)
                })
            })
            .collect::<Result<Vec<T>>>()
            .map(Some),
        other => Err(Error::decode(format!("{other:?} is not an array"))),
    }
}

/// The vector of a result of the server, a struct of Spark's `VectorUDT` fields.
pub(crate) fn vector_of(literal: &Literal) -> Result<Vector> {
    let [kind, size, indices, values] = fields::<4>(literal, "vector")?;
    Vector::of_fields(
        byte_of(kind)?,
        optional(size, int_of)?,
        optional(indices, ints_of)?,
        optional(values, doubles_of)?,
    )
    .map_err(Error::decode)
}

/// The matrix of a result of the server, a struct of Spark's `MatrixUDT` fields: its type
/// (0 sparse, 1 dense), numbers of rows and columns, column pointers, row indices, values, and
/// whether it is transposed.
pub(crate) fn matrix_of(literal: &Literal) -> Result<Matrix> {
    let [kind, num_rows, num_cols, col_ptrs, row_indices, values, transposed] =
        fields::<7>(literal, "matrix")?;
    let num_rows = count(int_of(num_rows)?).map_err(Error::decode)?;
    let num_cols = count(int_of(num_cols)?).map_err(Error::decode)?;
    let values = doubles_of(values)?;
    let is_transposed = match &transposed.literal_type {
        Some(LiteralType::Boolean(transposed)) => *transposed,
        other => return Err(Error::decode(format!("{other:?} is not a BOOLEAN"))),
    };
    match byte_of(kind)? {
        1 if values.len() == num_rows * num_cols => Ok(Matrix::Dense {
            num_rows,
            num_cols,
            values,
            is_transposed,
        }),
        0 => {
            let counts = |values: Vec<i32>| {
                values
                    .into_iter()
                    .map(count)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(Error::decode)
            };
            let col_ptrs = counts(ints_of(col_ptrs)?)?;
            let row_indices = counts(ints_of(row_indices)?)?;
            if row_indices.len() != values.len() {
                return Err(Error::decode(format!(
                    "a sparse matrix with {} row indices and {} values",
                    row_indices.len(),
                    values.len()
                )));
            }
            Ok(Matrix::Sparse {
                num_rows,
                num_cols,
                col_ptrs,
                row_indices,
                values,
                is_transposed,
            })
        }
        kind => Err(Error::decode(format!(
            "a matrix of the type {kind} of {num_rows} rows and {num_cols} columns with {} values",
            values.len()
        ))),
    }
}

/// The `N` fields of `literal`, a struct of the user-defined type `what`.
fn fields<'a, const N: usize>(literal: &'a Literal, what: &str) -> Result<&'a [Literal; N]> {
    match &literal.literal_type {
        Some(LiteralType::Struct(fields)) => fields
            .elements
            .as_slice()
            .try_into()
            .map_err(|_| Error::decode(format!("a {what} of {} fields", fields.elements.len()))),
        other => Err(Error::decode(format!("{other:?} is not a {what}"))),
    }
}

fn optional<T>(literal: &Literal, read: fn(&Literal) -> Result<T>) -> Result<Option<T>> {
    match literal.literal_type {
        Some(LiteralType::Null(_)) => Ok(None),
        _ => read(literal).map(Some),
    }
}

fn byte_of(literal: &Literal) -> Result<i8> {
    match literal.literal_type {
        Some(LiteralType::Byte(byte)) => {
            i8::try_from(byte).map_err(|_| Error::decode(format!("{byte} is not a TINYINT")))
        }
        ref other => Err(Error::decode(format!("{other:?} is not a TINYINT"))),
    }
}

pub(crate) fn int_of(literal: &Literal) -> Result<i32> {
    match literal.literal_type {
        Some(LiteralType::Integer(value)) => Ok(value),
        ref other => Err(Error::decode(format!("{other:?} is not an INT"))),
    }
}

pub(crate) fn double_of(literal: &Literal) -> Result<f64> {
    match literal.literal_type {
        Some(LiteralType::Double(value)) => Ok(value),
        ref other => Err(Error::decode(format!("{other:?} is not a DOUBLE"))),
    }
}

/// The elements of `literal`, an array that the server sends as a specialized array or as an
/// array of literals.
fn elements_of<T>(
    literal: &Literal,
    specialized: fn(ValueType) -> Option<Vec<T>>,
    element: fn(&Literal) -> Result<T>,
) -> Result<Vec<T>> {
    match &literal.literal_type {
        Some(LiteralType::SpecializedArray(array)) => array
            .value_type
            .clone()
            .and_then(specialized)
            .ok_or_else(|| Error::decode(format!("{array:?} is not an array of this type"))),
        Some(LiteralType::Array(array)) => array
            .elements
            .iter()
            .enumerate()
            .map(|(index, value)| element(value).map_err(|error| error.in_element(index)))
            .collect(),
        other => Err(Error::decode(format!("{other:?} is not an array"))),
    }
}

/// The elements of `literal`, an array of literals that `element` reads, such as of arrays.
pub(crate) fn arrays_of<T>(
    literal: &Literal,
    element: fn(&Literal) -> Result<T>,
) -> Result<Vec<T>> {
    elements_of(literal, |_| None, element)
}

fn ints_of(literal: &Literal) -> Result<Vec<i32>> {
    elements_of(
        literal,
        |values| match values {
            ValueType::Ints(ints) => Some(ints.values),
            _ => None,
        },
        int_of,
    )
}

fn doubles_of(literal: &Literal) -> Result<Vec<f64>> {
    elements_of(
        literal,
        |values| match values {
            ValueType::Doubles(doubles) => Some(doubles.values),
            _ => None,
        },
        double_of,
    )
}

pub(crate) fn strings_of(literal: &Literal) -> Result<Vec<String>> {
    elements_of(
        literal,
        |values| match values {
            ValueType::Strings(strings) => Some(strings.values),
            _ => None,
        },
        |value| match &value.literal_type {
            Some(LiteralType::String(text)) => Ok(text.clone()),
            other => Err(Error::decode(format!("{other:?} is not a STRING"))),
        },
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int32Array, Int8Array, ListArray, RecordBatch, StructArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field, Float64Type, Int32Type};
    use proto::expression::literal::{Array, SpecializedArray, Struct};

    use super::*;
    use crate::ErrorKind;

    fn literal(literal_type: LiteralType) -> Literal {
        Literal {
            literal_type: Some(literal_type),
            ..Default::default()
        }
    }

    fn null() -> Literal {
        literal(LiteralType::Null(Default::default()))
    }

    fn fields(elements: Vec<Literal>) -> Literal {
        literal(LiteralType::Struct(Struct {
            elements,
            ..Default::default()
        }))
    }

    fn specialized(values: ValueType) -> Literal {
        literal(LiteralType::SpecializedArray(SpecializedArray {
            value_type: Some(values),
        }))
    }

    fn ints(values: &[i32]) -> Literal {
        specialized(ValueType::Ints(proto::Ints {
            values: values.to_vec(),
        }))
    }

    fn doubles(values: &[f64]) -> Literal {
        specialized(ValueType::Doubles(proto::Doubles {
            values: values.to_vec(),
        }))
    }

    fn array(elements: Vec<Literal>) -> Literal {
        literal(LiteralType::Array(Array {
            elements,
            ..Default::default()
        }))
    }

    fn byte(value: i32) -> Literal {
        literal(LiteralType::Byte(value))
    }

    fn int(value: i32) -> Literal {
        literal(LiteralType::Integer(value))
    }

    /// A column of vectors as the server sends it: dense `[1, 2]`, sparse `(4, [1, 3], [5, 6])`,
    /// then `NULL`.
    fn vector_column() -> RecordBatch {
        let fields = vec![
            Field::new("type", DataType::Int8, false),
            Field::new("size", DataType::Int32, true),
            Field::new_list(
                "indices",
                Field::new_list_field(DataType::Int32, true),
                true,
            ),
            Field::new_list(
                "values",
                Field::new_list_field(DataType::Float64, true),
                true,
            ),
        ];
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int8Array::from(vec![1, 0, 1])),
            Arc::new(Int32Array::from(vec![None, Some(4), None])),
            Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                None,
                Some(vec![Some(1), Some(3)]),
                None,
            ])),
            Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(vec![
                Some(vec![Some(1.0), Some(2.0)]),
                Some(vec![Some(5.0), Some(6.0)]),
                None,
            ])),
        ];
        let nulls = NullBuffer::from(vec![true, true, false]);
        let vectors = StructArray::new(fields.into(), columns, Some(nulls));
        RecordBatch::try_from_iter([("probability", Arc::new(vectors) as ArrayRef)]).unwrap()
    }

    fn sparse() -> Vector {
        Vector::Sparse {
            size: 4,
            indices: vec![1, 3],
            values: vec![5.0, 6.0],
        }
    }

    #[test]
    fn vectors_and_matrices_are_read_from_results_and_columns_with_their_nulls() {
        let dense = Vector::Dense(vec![1.0, 2.0]);
        let from_results = [
            (
                "a dense vector",
                fields(vec![byte(1), null(), null(), doubles(&[1.0, 2.0])]),
                dense.clone(),
            ),
            (
                "a dense vector of an array of literals",
                fields(vec![
                    byte(1),
                    null(),
                    null(),
                    array(vec![
                        literal(LiteralType::Double(1.0)),
                        literal(LiteralType::Double(2.0)),
                    ]),
                ]),
                dense.clone(),
            ),
            (
                "a sparse vector",
                fields(vec![byte(0), int(4), ints(&[1, 3]), doubles(&[5.0, 6.0])]),
                sparse(),
            ),
        ];
        for (case, literal, expected) in from_results {
            assert_eq!(vector_of(&literal).unwrap(), expected, "{case}");
        }
        let column = vector_column();
        let read: Vec<(Option<Vector>,)> = crate::de::from_record_batch(&column).unwrap();
        assert_eq!(
            read,
            [(Some(dense.clone()),), (Some(sparse()),), (None,)],
            "collect_as"
        );
        let rows = crate::de::rows(&column).unwrap();
        let values: Vec<Option<Vector>> = rows
            .iter()
            .map(|row| match row.get(0).unwrap() {
                Value::Null => None,
                value => Some(Vector::try_from(value).unwrap()),
            })
            .collect();
        assert_eq!(values, [Some(dense.clone()), Some(sparse()), None], "rows");
        assert_eq!((dense.size(), dense.to_dense()), (2, vec![1.0, 2.0]));
        assert_eq!(
            (sparse().size(), sparse().to_dense()),
            (4, vec![0.0, 5.0, 0.0, 6.0])
        );

        let matrices = [
            (
                "a dense matrix",
                fields(vec![
                    byte(1),
                    int(1),
                    int(2),
                    null(),
                    null(),
                    doubles(&[1.0, 2.0]),
                    literal(LiteralType::Boolean(true)),
                ]),
                Matrix::Dense {
                    num_rows: 1,
                    num_cols: 2,
                    values: vec![1.0, 2.0],
                    is_transposed: true,
                },
            ),
            (
                "a sparse matrix",
                fields(vec![
                    byte(0),
                    int(2),
                    int(2),
                    ints(&[0, 1, 2]),
                    ints(&[1, 0]),
                    doubles(&[3.0, 4.0]),
                    literal(LiteralType::Boolean(false)),
                ]),
                Matrix::Sparse {
                    num_rows: 2,
                    num_cols: 2,
                    col_ptrs: vec![0, 1, 2],
                    row_indices: vec![1, 0],
                    values: vec![3.0, 4.0],
                    is_transposed: false,
                },
            ),
        ];
        for (case, literal, expected) in matrices {
            assert_eq!(matrix_of(&literal).unwrap(), expected, "{case}");
        }
    }

    #[test]
    fn what_is_not_a_vector_or_a_matrix_is_a_decode_error() {
        let vectors = [
            (
                "an unknown type",
                fields(vec![byte(2), null(), null(), doubles(&[1.0])]),
            ),
            ("no values", fields(vec![byte(1), null(), null(), null()])),
            (
                "an index out of the size",
                fields(vec![byte(0), int(2), ints(&[2]), doubles(&[1.0])]),
            ),
            (
                "fewer indices than values",
                fields(vec![byte(0), int(4), ints(&[1]), doubles(&[1.0, 2.0])]),
            ),
            (
                "a negative size",
                fields(vec![byte(0), int(-1), ints(&[]), doubles(&[])]),
            ),
            ("three fields", fields(vec![byte(1), null(), null()])),
            ("a number", literal(LiteralType::Double(1.0))),
            (
                "values of strings",
                fields(vec![
                    byte(1),
                    null(),
                    null(),
                    array(vec![literal(LiteralType::String("1".into()))]),
                ]),
            ),
        ];
        for (case, literal) in vectors {
            let error = vector_of(&literal).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Decode, "{case}: {error}");
        }
        let matrices = [
            (
                "a dense matrix of fewer values than cells",
                fields(vec![
                    byte(1),
                    int(2),
                    int(2),
                    null(),
                    null(),
                    doubles(&[1.0]),
                    literal(LiteralType::Boolean(false)),
                ]),
            ),
            (
                "a sparse matrix of more values than row indices",
                fields(vec![
                    byte(0),
                    int(2),
                    int(2),
                    ints(&[0, 1, 1]),
                    ints(&[0]),
                    doubles(&[1.0, 2.0]),
                    literal(LiteralType::Boolean(false)),
                ]),
            ),
        ];
        for (case, literal) in matrices {
            let error = matrix_of(&literal).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Decode, "{case}: {error}");
        }
        let column = vector_column();
        let row = crate::de::rows(&column).unwrap().remove(0);
        let Value::Struct(fields) = row.get(0).unwrap().clone() else {
            panic!("not a struct");
        };
        let without_values = Value::Struct(crate::value::Row::new(
            fields.names()[..3].into(),
            fields.values()[..3].to_vec(),
        ));
        for (case, value) in [
            ("a NULL", Value::Null),
            ("a struct without values", without_values),
        ] {
            let error = Vector::try_from(&value).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Decode, "{case}: {error}");
        }
    }
}
