//! The aggregations of grouped data that both front ends build alike: those of every numeric
//! column, which read the schema of the grouped DataFrame.

use crate::error::{Result, SparkError, SparkErrorKind};
use crate::expression::Expression;
use crate::types::DataType;

/// The aggregations of [`numeric_aggregates`], or why the given columns have none.
#[derive(Debug, Clone)]
pub enum NumericAggregates {
    /// The aggregation of each column, in order.
    Aggregates(Vec<Expression>),
    /// The given columns that are not numeric columns of the schema, in their order.
    NotNumeric(Vec<String>),
    /// The first given column that names a struct, and the `IndexError` of
    /// [`verify_numeric_col_name`] for it.
    Struct {
        column: String,
        error: Box<SparkError>,
    },
}

/// The aggregations by `function` ("min", "max", "avg" or "sum") of the numeric `columns` of
/// `schema`: of every numeric column of `schema`, in its order, without `columns`.
pub fn numeric_aggregates(
    function: &str,
    schema: &DataType,
    columns: &[impl AsRef<str>],
) -> NumericAggregates {
    let aggregate = |name: &str| Expression::function(function, vec![Expression::column(name)]);
    if columns.is_empty() {
        let aggregates = match schema {
            DataType::Struct { fields } => fields
                .iter()
                .filter(|field| is_numeric(&field.data_type))
                .map(|field| aggregate(&field.name))
                .collect(),
            _ => Vec::new(),
        };
        return NumericAggregates::Aggregates(aggregates);
    }
    let mut invalid = Vec::new();
    for column in columns.iter().map(AsRef::as_ref) {
        match verify_numeric_col_name(column, schema) {
            Ok(true) => {}
            Ok(false) => invalid.push(column.to_string()),
            Err(error) => {
                return NumericAggregates::Struct {
                    column: column.to_string(),
                    error: Box::new(error),
                }
            }
        }
    }
    if !invalid.is_empty() {
        return NumericAggregates::NotNumeric(invalid);
    }
    NumericAggregates::Aggregates(
        columns
            .iter()
            .map(|column| aggregate(column.as_ref()))
            .collect(),
    )
}

/// Whether `data_type` is a `NumericType` of PySpark.
fn is_numeric(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Decimal { .. }
    )
}

/// The parts of an attribute name, as PySpark's `parse_attr_name`: split by dots, of
/// backtick-quoted parts too; None for a name that it does not parse.
pub fn parse_attr_name(name: &str) -> Option<Vec<String>> {
    let chars: Vec<char> = name.chars().collect();
    let n = chars.len();
    let (mut parts, mut part, mut in_backtick, mut i) = (Vec::new(), String::new(), false, 0);
    while i < n {
        let c = chars[i];
        if in_backtick {
            if c == '`' {
                if i + 1 < n && chars[i + 1] == '`' {
                    part.push('`');
                    i += 1;
                } else {
                    in_backtick = false;
                    if i + 1 < n && chars[i + 1] != '.' {
                        return None;
                    }
                }
            } else {
                part.push(c);
            }
        } else if c == '`' {
            if !part.is_empty() {
                return None;
            }
            in_backtick = true;
        } else if c == '.' {
            // As Python's `name[i - 1]`, the last character for the first.
            let previous = chars[(i + n - 1) % n];
            if previous == '.' || i == n - 1 {
                return None;
            }
            parts.push(std::mem::take(&mut part));
        } else {
            part.push(c);
        }
        i += 1;
    }
    if in_backtick {
        return None;
    }
    parts.push(part);
    Some(parts)
}

/// Whether `name` is of a numeric column of `schema`, as PySpark's
/// `verify_numeric_col_name`: by its parts, of the fields of structs and of the structs of
/// arrays. As PySpark's, the name of a struct is an `IndexError`.
pub fn verify_numeric_col_name(name: &str, schema: &DataType) -> Result<bool> {
    fn verify(parts: &[String], data_type: &DataType) -> Result<bool> {
        if parts.is_empty() && is_numeric(data_type) {
            return Ok(true);
        }
        let fields = match data_type {
            DataType::Struct { fields } => fields,
            DataType::Array { element_type, .. } => match element_type.as_ref() {
                DataType::Struct { fields } => fields,
                _ => return Ok(false),
            },
            _ => return Ok(false),
        };
        let Some((part, rest)) = parts.split_first() else {
            let mut err = SparkError::value_msg("list index out of range");
            err.kind = SparkErrorKind::IndexError;
            return Err(err);
        };
        match fields.iter().find(|field| field.name == *part) {
            Some(field) => verify(rest, &field.data_type),
            None => Ok(false),
        }
    }
    match parse_attr_name(name) {
        Some(parts) if !parts.is_empty() => verify(&parts, schema),
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StructField;

    #[test]
    fn numeric_aggregates_are_of_the_numeric_columns_of_the_schema() {
        let field = |name: &str, data_type| StructField {
            name: name.to_string(),
            data_type,
            nullable: true,
            metadata: Default::default(),
        };
        let schema = DataType::Struct {
            fields: vec![
                field("i", DataType::Integer),
                field(
                    "s",
                    DataType::String {
                        collation: "UTF8_BINARY".to_string(),
                    },
                ),
                field("d", DataType::Double),
                field(
                    "st",
                    DataType::Struct {
                        fields: vec![field("a", DataType::Long)],
                    },
                ),
            ],
        };
        let sums = |names: &[&str]| -> Vec<String> {
            names
                .iter()
                .map(|name| {
                    format!(
                        "{:?}",
                        Expression::function("sum", vec![Expression::column(*name)])
                    )
                })
                .collect()
        };
        let rendered = |aggregates: NumericAggregates| match aggregates {
            NumericAggregates::Aggregates(aggregates) => Ok(aggregates
                .iter()
                .map(|a| format!("{a:?}"))
                .collect::<Vec<_>>()),
            NumericAggregates::NotNumeric(invalid) => Err(invalid),
            NumericAggregates::Struct { column, .. } => Err(vec![format!("struct {column}")]),
        };
        let none: [&str; 0] = [];
        assert_eq!(
            rendered(numeric_aggregates("sum", &schema, &none)),
            Ok(sums(&["i", "d"]))
        );
        assert_eq!(
            rendered(numeric_aggregates("sum", &schema, &["d", "st.a"])),
            Ok(sums(&["d", "st.a"]))
        );
        assert_eq!(
            rendered(numeric_aggregates("sum", &schema, &["s", "i", "x"])),
            Err(vec!["s".to_string(), "x".to_string()])
        );
        assert_eq!(
            rendered(numeric_aggregates("sum", &schema, &["s", "st", "x"])),
            Err(vec!["struct st".to_string()])
        );
    }
}
