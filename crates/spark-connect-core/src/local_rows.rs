//! Local rows as Arrow data: the encoding of rows of [`Value`]s against a schema, and the
//! local relation that sends them. Shared by the client crates; not a stable API.

use std::sync::Arc;

use arrow::record_batch::RecordBatch;

use crate::arrow_input::batch_data_size;
use crate::error::{Result, SparkError};
use crate::plan::LogicalPlan;
use crate::row::{Row, Value};
use crate::types::DataType;

/// The error for a row of `create_dataframe` whose width differs from the schema's. The
/// Python client checks widths itself, raising pyspark's `AXIS_LENGTH_MISMATCH`.
fn row_width_mismatch(row_index: usize, actual: usize, expected: usize) -> SparkError {
    SparkError::value_msg(format!(
        "Row {row_index} has {actual} values, but the schema has {expected} fields"
    ))
}

/// Convert Arrow RecordBatch data to IPC bytes.
/// Used by createDataFrame to package local data.
/// The size of a local relation, in Arrow data, from which `createDataFrame` reads the
/// server's local-relation settings and sends the relation as the Arrow importer does: inline
/// below `spark.sql.session.localRelationCacheThreshold`, uploaded at it, and failing past
/// `spark.sql.session.localRelationSizeLimit`, as PySpark does. Smaller relations are sent
/// inline without that RPC: 64 KiB is far below the server's gRPC message size limit (128 MiB
/// by default), so a relation this small cannot be too large to send, and the RPC would cost
/// more than the transfer.
pub const LOCAL_RELATION_SETTINGS_BYTES: usize = 64 << 10;

/// Internal binding bridge for encoding already converted local rows. The Python
/// binding also uses it for native columns beside imported pandas Arrow columns.
/// Not a stable public API.
pub fn rows_to_arrow_batch(
    rows: &[Row],
    schema: &DataType,
) -> Result<arrow::record_batch::RecordBatch> {
    use arrow::datatypes::Schema as ArrowSchema;

    // Convert our schema to Arrow schema
    let arrow_schema_vec = schema_to_arrow_fields(schema)?;
    let arrow_schema = ArrowSchema::new(arrow_schema_vec);

    // Build column arrays from rows
    let mut columns = vec![];
    let fields = match schema {
        DataType::Struct { fields } => fields,
        _ => return Err(SparkError::connect_msg("Schema is not a struct type")),
    };

    // Conversion is positional, so every row must match the schema width.
    for (index, row) in rows.iter().enumerate() {
        if row.len() != fields.len() {
            return Err(row_width_mismatch(index, row.len(), fields.len()));
        }
    }

    for (field_idx, field) in fields.iter().enumerate() {
        let column: Vec<&Value> = rows.iter().map(|row| &row.values()[field_idx]).collect();
        columns.push(values_to_arrow(&column, &field.data_type)?);
    }

    columns_to_arrow_batch(columns, arrow_schema, rows.len())
}
pub fn columns_to_arrow_batch(
    columns: Vec<arrow::array::ArrayRef>,
    arrow_schema: arrow::datatypes::Schema,
    row_count: usize,
) -> Result<arrow::record_batch::RecordBatch> {
    use arrow::record_batch::RecordBatch;

    // Preserve the row count even for a batch without columns.
    let options = arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(row_count));
    RecordBatch::try_new_with_options(Arc::new(arrow_schema), columns, &options)
        .map_err(|e| SparkError::connect_msg(format!("Failed to create Arrow batch: {}", e)))
}

pub fn batch_to_arrow_ipc(batch: &arrow::record_batch::RecordBatch) -> Result<Vec<u8>> {
    use arrow::ipc::writer::StreamWriter;

    // Write to IPC stream
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())
            .map_err(|e| SparkError::connect_msg(format!("Failed to create IPC writer: {}", e)))?;
        writer
            .write(&batch)
            .map_err(|e| SparkError::connect_msg(format!("Failed to write batch to IPC: {}", e)))?;
        writer
            .finish()
            .map_err(|e| SparkError::connect_msg(format!("Failed to finish IPC writer: {}", e)))?;
    }

    Ok(buffer)
}

/// Convert DataType fields to Arrow fields.
pub fn schema_to_arrow_fields(schema: &DataType) -> Result<Vec<arrow::datatypes::Field>> {
    use arrow::datatypes::Field;

    match schema {
        DataType::Struct { fields } => {
            let mut arrow_fields = vec![];
            for field in fields {
                arrow_fields.push(Field::new(
                    &field.name,
                    datatype_to_arrow(&field.data_type)?,
                    field.nullable,
                ));
            }
            Ok(arrow_fields)
        }
        _ => Err(SparkError::connect_msg("Schema is not a struct")),
    }
}

/// The names of nested struct fields as PySpark's `_dedup_names` gives them: a name that occurs
/// more than once is numbered per occurrence, as `name_0`, `name_1`, and others are kept.
pub fn dedup_field_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let names: Vec<&str> = names.into_iter().collect();
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for name in &names {
        *counts.entry(*name).or_insert(0) += 1;
    }
    let mut next: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    names
        .iter()
        .map(|&name| {
            if counts[name] > 1 {
                let i = next.entry(name).or_insert(0);
                let deduped = format!("{name}_{i}");
                *i += 1;
                deduped
            } else {
                name.to_string()
            }
        })
        .collect()
}

/// Map a Spark `DataType` to its Arrow `DataType`, recursively (array/map/struct included).
pub fn datatype_to_arrow(dt: &DataType) -> Result<arrow::datatypes::DataType> {
    use arrow::datatypes::{DataType as A, Field, Fields, TimeUnit};
    use std::collections::HashMap;
    Ok(match dt {
        DataType::Null => A::Null,
        DataType::Boolean => A::Boolean,
        DataType::Byte => A::Int8,
        DataType::Short => A::Int16,
        DataType::Integer => A::Int32,
        DataType::Long => A::Int64,
        DataType::Float => A::Float32,
        DataType::Double => A::Float64,
        // CHAR/VARCHAR are string-backed on the wire.
        DataType::String { .. } | DataType::Char { .. } | DataType::Varchar { .. } => A::Utf8,
        DataType::Binary => A::Binary,
        DataType::Date => A::Date32,
        // TIMESTAMP (LTZ) and TIMESTAMP_NTZ are both micros; NTZ carries no zone.
        DataType::Timestamp => A::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        DataType::TimestampNtz => A::Timestamp(TimeUnit::Microsecond, None),
        DataType::TimestampLtzNanos { .. } => {
            A::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))
        }
        DataType::TimestampNtzNanos { .. } => A::Timestamp(TimeUnit::Nanosecond, None),
        // As PySpark's `to_arrow_type` of TimeType.
        DataType::Time { .. } => A::Time64(TimeUnit::Nanosecond),
        // As Spark's Arrow conversion of DayTimeIntervalType.
        DataType::DayTimeInterval { .. } => A::Duration(TimeUnit::Microsecond),
        DataType::Decimal { precision, scale } => A::Decimal128(*precision as u8, *scale as i8),
        DataType::Array {
            element_type,
            contains_null,
        } => A::List(Arc::new(Field::new(
            "element",
            datatype_to_arrow(element_type)?,
            *contains_null,
        ))),
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let entry_fields: Fields = vec![
                Field::new("key", datatype_to_arrow(key_type)?, false),
                Field::new(
                    "value",
                    datatype_to_arrow(value_type)?,
                    *value_contains_null,
                ),
            ]
            .into();
            A::Map(
                Arc::new(Field::new("entries", A::Struct(entry_fields), false)),
                false,
            )
        }
        // The names of nested fields are deduplicated, as PySpark's `_deduplicate_field_names`
        // does for the Arrow schema it sends. The server cannot read duplicate names, while the
        // schema of the relation keeps the names of the fields.
        DataType::Struct { fields } => {
            let names = dedup_field_names(fields.iter().map(|field| field.name.as_str()));
            let f: Fields = fields
                .iter()
                .zip(names)
                .map(|(field, name)| {
                    Ok(Field::new(
                        name,
                        datatype_to_arrow(&field.data_type)?,
                        field.nullable,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into();
            A::Struct(f)
        }
        // As PySpark's `to_arrow_type`: the struct of the binary value and metadata of a
        // VARIANT, and of the srid and WKB of a GEOMETRY or a GEOGRAPHY, whose metadata tags
        // the struct as one of these types.
        DataType::Variant => A::Struct(
            vec![
                Field::new("value", A::Binary, false),
                Field::new("metadata", A::Binary, false)
                    .with_metadata(HashMap::from([("variant".to_string(), "true".to_string())])),
            ]
            .into(),
        ),
        DataType::Geometry { srid } | DataType::Geography { srid } => {
            let kind = if matches!(dt, DataType::Geometry { .. }) {
                "geometry"
            } else {
                "geography"
            };
            A::Struct(
                vec![
                    Field::new("srid", A::Int32, false),
                    Field::new("wkb", A::Binary, false).with_metadata(HashMap::from([
                        (kind.to_string(), "true".to_string()),
                        ("srid".to_string(), srid.to_string()),
                    ])),
                ]
                .into(),
            )
        }
        // As PySpark's, a user-defined type is sent as its SQL type.
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => datatype_to_arrow(sql_type)?,
        other => {
            return Err(SparkError::connect_msg(format!(
                "Unsupported type for Arrow createDataFrame conversion: {other:?}"
            )))
        }
    })
}

/// Recursively build an Arrow array from a column of `Value`s against a declared
/// `DataType`, covering scalars and the nested array/map/struct types. Temporary
/// columns borrow their values; building the Arrow output still copies payloads.
pub fn values_to_arrow(values: &[&Value], dt: &DataType) -> Result<Arc<dyn arrow::array::Array>> {
    use arrow::array::*;
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{Field, Fields};

    macro_rules! prim {
        ($arr:ty, $pat:path) => {{
            let vals: Result<Vec<_>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    $pat(x) => Ok(Some(*x)),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(<$arr>::from(vals?)) as Arc<dyn Array>)
        }};
    }

    // Numeric encoders coerce across integer/float widths, so a value the schema-unaware
    // Python->Value conversion produced as a wider/narrower numeric (e.g. a nested-struct
    // int arriving as Long) still matches its declared field type. An integer that does
    // not fit the declared type is an error, as PySpark's Arrow conversion raises for it.
    macro_rules! int_prim {
        ($arr:ty, $t:ty, $sql:expr) => {{
            let narrow = |x: i64| {
                <$t>::try_from(x).map(Some).map_err(|_| {
                    SparkError::value_msg(format!(
                        "Value {x} does not fit the {} type of its column, whose range is {}..={}",
                        $sql,
                        <$t>::MIN,
                        <$t>::MAX
                    ))
                })
            };
            let vals: Result<Vec<_>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Byte(x) => narrow(*x as i64),
                    Value::Short(x) => narrow(*x as i64),
                    Value::Integer(x) => narrow(*x as i64),
                    Value::Long(x) => narrow(*x),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(<$arr>::from(vals?)) as Arc<dyn Array>)
        }};
    }
    macro_rules! float_prim {
        ($arr:ty, $t:ty) => {{
            let vals: Result<Vec<_>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Byte(x) => Ok(Some(*x as $t)),
                    Value::Short(x) => Ok(Some(*x as $t)),
                    Value::Float(x) => Ok(Some(*x as $t)),
                    Value::Double(x) => Ok(Some(*x as $t)),
                    Value::Integer(x) => Ok(Some(*x as $t)),
                    Value::Long(x) => Ok(Some(*x as $t)),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(<$arr>::from(vals?)) as Arc<dyn Array>)
        }};
    }

    match dt {
        DataType::Null => Ok(Arc::new(NullArray::new(values.len()))),
        DataType::Boolean => prim!(BooleanArray, Value::Bool),
        DataType::Byte => int_prim!(Int8Array, i8, "TINYINT"),
        DataType::Short => int_prim!(Int16Array, i16, "SMALLINT"),
        DataType::Integer => int_prim!(Int32Array, i32, "INT"),
        DataType::Long => int_prim!(Int64Array, i64, "BIGINT"),
        DataType::Float => float_prim!(Float32Array, f32),
        DataType::Double => float_prim!(Float64Array, f64),
        DataType::String { .. } | DataType::Char { .. } | DataType::Varchar { .. } => {
            let vals: Result<Vec<Option<&str>>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::String(s) => Ok(Some(s.as_str())),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(StringArray::from(vals?)))
        }
        DataType::Binary => {
            let vals: Result<Vec<Option<&[u8]>>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Binary(b) => Ok(Some(b.as_slice())),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(BinaryArray::from(vals?)))
        }
        DataType::Date => prim!(Date32Array, Value::Date),
        // As PySpark's `to_arrow_type`: nanoseconds since midnight, as `TimeType.toInternal`.
        DataType::Time { .. } => prim!(Time64NanosecondArray, Value::Time),
        DataType::DayTimeInterval { .. } => prim!(DurationMicrosecondArray, Value::DayTimeInterval),
        DataType::Timestamp | DataType::TimestampNtz => {
            let vals: Result<Vec<Option<i64>>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Timestamp(t) | Value::TimestampNtz(t) => Ok(Some(*t)),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            let arr = TimestampMicrosecondArray::from(vals?);
            let arr = if matches!(dt, DataType::TimestampNtz) {
                arr
            } else {
                arr.with_timezone("UTC")
            };
            Ok(Arc::new(arr))
        }
        // Row values are microseconds (collect truncates nanosecond columns to micros), so
        // scale them up to the nanosecond unit `to_arrow_type` declares for these types.
        DataType::TimestampNtzNanos { .. } | DataType::TimestampLtzNanos { .. } => {
            let vals: Result<Vec<Option<i64>>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Timestamp(t) | Value::TimestampNtz(t) => {
                        t.checked_mul(1_000).map(Some).ok_or_else(|| {
                            SparkError::connect_msg(
                                "Timestamp out of range for nanosecond precision",
                            )
                        })
                    }
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            let arr = TimestampNanosecondArray::from(vals?);
            let arr = if matches!(dt, DataType::TimestampNtzNanos { .. }) {
                arr
            } else {
                arr.with_timezone("UTC")
            };
            Ok(Arc::new(arr))
        }
        DataType::Decimal { precision, scale } => {
            let col_scale = *scale;
            let vals: Result<Vec<Option<i128>>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Decimal { value, .. } => {
                        decimal_str_to_unscaled(value, *precision, col_scale).map(Some)
                    }
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            // Use the DECLARED precision/scale so the array matches the Arrow schema field
            // (Decimal128(precision, scale)); hardcoding 38 mismatched a DecimalType(10, 2).
            let arr = Decimal128Array::from(vals?)
                .with_precision_and_scale(*precision as u8, col_scale as i8)
                .map_err(|e| SparkError::connect_msg(format!("decimal build: {e}")))?;
            Ok(Arc::new(arr))
        }
        DataType::Array {
            element_type,
            contains_null,
        } => {
            let mut child: Vec<&Value> = Vec::new();
            let mut offsets: Vec<i32> = vec![0];
            let mut valid = arrow::array::builder::BooleanBufferBuilder::new(values.len());
            for v in values {
                match *v {
                    Value::Null => {
                        valid.append(false);
                        offsets.push(*offsets.last().unwrap());
                    }
                    Value::List(items) => {
                        valid.append(true);
                        child.extend(items.iter());
                        offsets.push(child.len() as i32);
                    }
                    _ => return Err(SparkError::connect_msg("Type mismatch: expected array")),
                }
            }
            let child_arr = values_to_arrow(&child, element_type)?;
            let field = Arc::new(Field::new(
                "element",
                datatype_to_arrow(element_type)?,
                *contains_null,
            ));
            // A non-nullable element that has nulls is an error, not a panic, as for a struct.
            ListArray::try_new(
                field,
                OffsetBuffer::new(offsets.into()),
                child_arr,
                Some(NullBuffer::new(valid.finish())),
            )
            .map(|array| Arc::new(array) as Arc<dyn Array>)
            .map_err(|e| SparkError::connect_msg(format!("list build: {e}")))
        }
        // The struct of a VARIANT's value and metadata, and of a GEOMETRY's or a GEOGRAPHY's
        // srid and WKB, as `datatype_to_arrow` declares; their fields are not nullable, so a
        // null value has empty ones.
        DataType::Variant | DataType::Geometry { .. } | DataType::Geography { .. } => {
            let mut first = Vec::with_capacity(values.len());
            let mut second: Vec<&[u8]> = Vec::with_capacity(values.len());
            let mut valid = arrow::array::builder::BooleanBufferBuilder::new(values.len());
            for v in values {
                match (*v, dt) {
                    (Value::Null, _) => {
                        valid.append(false);
                        first.push(None);
                        second.push(&[]);
                    }
                    (Value::Variant { value, metadata }, DataType::Variant) => {
                        valid.append(true);
                        first.push(Some(value.as_slice()));
                        second.push(metadata.as_slice());
                    }
                    (Value::Struct(sf), DataType::Geometry { .. } | DataType::Geography { .. }) => {
                        let (Some(Value::Integer(_)), Some(Value::Binary(wkb))) =
                            (sf.get_by_name("srid"), sf.get_by_name("wkb"))
                        else {
                            return Err(SparkError::connect_msg("Type mismatch in row data"));
                        };
                        valid.append(true);
                        first.push(None);
                        second.push(wkb.as_slice());
                    }
                    _ => return Err(SparkError::connect_msg("Type mismatch in row data")),
                }
            }
            let arrow::datatypes::DataType::Struct(afields) = datatype_to_arrow(dt)? else {
                unreachable!("these types are structs")
            };
            let nulls = Some(NullBuffer::new(valid.finish()));
            let arrays: Vec<Arc<dyn Array>> = match dt {
                DataType::Variant => vec![
                    Arc::new(BinaryArray::from(
                        first
                            .iter()
                            .map(|v| v.unwrap_or(&[]))
                            .collect::<Vec<&[u8]>>(),
                    )),
                    Arc::new(BinaryArray::from(second)),
                ],
                _ => {
                    let srids: Vec<i32> = values
                        .iter()
                        .map(|v| match *v {
                            Value::Struct(sf) => match sf.get_by_name("srid") {
                                Some(Value::Integer(srid)) => *srid,
                                _ => 0,
                            },
                            _ => 0,
                        })
                        .collect();
                    vec![
                        Arc::new(Int32Array::from(srids)),
                        Arc::new(BinaryArray::from(second)),
                    ]
                }
            };
            Ok(Arc::new(
                StructArray::try_new(afields, arrays, nulls).map_err(|e| {
                    SparkError::connect_msg(format!("Arrow struct build error: {e}"))
                })?,
            ))
        }
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => values_to_arrow(values, sql_type),
        DataType::Struct { fields } => {
            let mut cols: Vec<Vec<&Value>> = vec![Vec::with_capacity(values.len()); fields.len()];
            let mut valid = arrow::array::builder::BooleanBufferBuilder::new(values.len());
            for v in values {
                match *v {
                    Value::Null => {
                        valid.append(false);
                        for c in cols.iter_mut() {
                            c.push(&Value::Null);
                        }
                    }
                    Value::Struct(sf) => {
                        valid.append(true);
                        // A struct of the declared fields in order, as the Python client
                        // converts one, is taken by position, so that fields of the same
                        // name keep their own values. Otherwise match by field name, else
                        // by position.
                        let in_order = sf.len() == fields.len()
                            && sf.fields().iter().zip(fields).all(|(n, f)| *n == f.name);
                        for (i, f) in fields.iter().enumerate() {
                            let val = if in_order {
                                sf.get(i)
                            } else {
                                sf.get_by_name(&f.name).or_else(|| sf.get(i))
                            };
                            cols[i].push(val.unwrap_or(&Value::Null));
                        }
                    }
                    // A Python dict supplied for a struct field (e.g. array<struct> data)
                    // arrives as a Map; match its entries to the struct fields by name.
                    Value::Map(m) => {
                        valid.append(true);
                        for (i, f) in fields.iter().enumerate() {
                            let value = m.iter().find_map(|(key, value)| match key {
                                Value::String(key) if *key == f.name => Some(value),
                                _ => None,
                            });
                            cols[i].push(value.unwrap_or(&Value::Null));
                        }
                    }
                    _ => return Err(SparkError::connect_msg("Type mismatch: expected struct")),
                }
            }
            let arrays: Vec<Arc<dyn Array>> = fields
                .iter()
                .enumerate()
                .map(|(i, f)| values_to_arrow(&cols[i], &f.data_type))
                .collect::<Result<_>>()?;
            // The Arrow fields of the struct, with the deduplicated names of `datatype_to_arrow`,
            // which the column's type has.
            let arrow::datatypes::DataType::Struct(afields) = datatype_to_arrow(dt)? else {
                unreachable!("a struct is a struct")
            };
            // A struct without fields takes its length from the values.
            StructArray::try_new_with_length(
                afields,
                arrays,
                Some(NullBuffer::new(valid.finish())),
                values.len(),
            )
            .map(|array| Arc::new(array) as Arc<dyn Array>)
            .map_err(|e| SparkError::connect_msg(format!("struct build: {e}")))
        }
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let mut keys: Vec<&Value> = Vec::new();
            let mut vals: Vec<&Value> = Vec::new();
            let mut offsets: Vec<i32> = vec![0];
            let mut valid = arrow::array::builder::BooleanBufferBuilder::new(values.len());
            for v in values {
                match *v {
                    Value::Null => {
                        valid.append(false);
                        offsets.push(*offsets.last().unwrap());
                    }
                    Value::Map(m) => {
                        valid.append(true);
                        for (k, val) in m {
                            if matches!(k, Value::Null) {
                                return Err(SparkError::connect_msg("A map key cannot be null"));
                            }
                            keys.push(k);
                            vals.push(val);
                        }
                        offsets.push(keys.len() as i32);
                    }
                    _ => return Err(SparkError::connect_msg("Type mismatch: expected map")),
                }
            }
            let key_arr = values_to_arrow(&keys, key_type)?;
            let val_arr = values_to_arrow(&vals, value_type)?;
            let entry_fields: Fields = vec![
                Field::new("key", datatype_to_arrow(key_type)?, false),
                Field::new(
                    "value",
                    datatype_to_arrow(value_type)?,
                    *value_contains_null,
                ),
            ]
            .into();
            let entries = StructArray::try_new(entry_fields.clone(), vec![key_arr, val_arr], None)
                .map_err(|e| SparkError::connect_msg(format!("map build: {e}")))?;
            let entries_field = Arc::new(Field::new(
                "entries",
                arrow::datatypes::DataType::Struct(entry_fields),
                false,
            ));
            MapArray::try_new(
                entries_field,
                OffsetBuffer::new(offsets.into()),
                entries,
                Some(NullBuffer::new(valid.finish())),
                false,
            )
            .map(|array| Arc::new(array) as Arc<dyn Array>)
            .map_err(|e| SparkError::connect_msg(format!("map build: {e}")))
        }
        other => Err(SparkError::connect_msg(format!(
            "Unsupported type for Arrow createDataFrame conversion: {other:?}"
        ))),
    }
}

/// Converts a decimal string, in fixed-point (`-1.50`) or scientific (`1.5E-7`) notation, to
/// its unscaled `i128` at `scale` (`-1.50` at scale 2 is -150). The value must fit
/// `decimal(precision, scale)` exactly, as Arrow requires of the Decimals of PySpark's pandas
/// conversion: a nonzero digit beyond the scale, or a value beyond the precision, is an error
/// rather than truncated. The work is bounded by the length of the string, never by its
/// exponent, so `1E+1000000000` fails at once.
pub fn decimal_str_to_unscaled(s: &str, precision: i32, scale: i32) -> Result<i128> {
    if !(1..=38).contains(&precision) || !(0..=precision).contains(&scale) {
        return Err(SparkError::connect_msg(format!(
            "unsupported decimal type decimal({precision},{scale})"
        )));
    }
    let invalid = || SparkError::connect_msg(format!("invalid decimal '{s}'"));
    let inexact = || {
        SparkError::connect_msg(format!(
            "decimal '{s}' does not fit decimal({precision},{scale}) exactly"
        ))
    };
    let text = s.trim();
    let (negative, unsigned) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let (coefficient, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((coefficient, exponent)) => {
            (coefficient, exponent.parse::<i64>().map_err(|_| invalid())?)
        }
        None => (unsigned, 0),
    };
    let mut point = false;
    let mut fractional = 0_i128;
    let mut digits = 0_usize;
    for byte in coefficient.bytes() {
        match byte {
            b'.' if !point => point = true,
            b'0'..=b'9' => {
                digits += 1;
                if point {
                    fractional += 1;
                }
            }
            _ => return Err(invalid()),
        }
    }
    if digits == 0 {
        return Err(invalid());
    }
    // The digits from the first nonzero one: the value is them times 10^(exponent -
    // fractional), and the unscaled value them times 10^shift.
    let significant: Vec<u8> = coefficient
        .bytes()
        .filter(|b| *b != b'.')
        .skip_while(|b| *b == b'0')
        .collect();
    // i128 arithmetic, so that i64::MIN and i64::MAX exponents cannot overflow.
    let shift = i128::from(scale) + i128::from(exponent) - fractional;
    let dropped = if shift < 0 {
        (-shift).min(significant.len() as i128) as usize
    } else {
        0
    };
    let (kept, lost) = significant.split_at(significant.len() - dropped);
    if lost.iter().any(|b| *b != b'0') {
        return Err(inexact());
    }
    if kept.is_empty() {
        return Ok(0);
    }
    if kept.len() as i128 + shift.max(0) > i128::from(precision) {
        return Err(inexact());
    }
    // At most 38 digits: no overflow.
    let mut value = kept
        .iter()
        .fold(0_i128, |value, digit| value * 10 + i128::from(digit - b'0'));
    for _ in 0..shift.max(0) {
        value *= 10;
    }
    Ok(if negative { -value } else { value })
}

/// The local relation of `batch`, inline, when it is smaller than
/// [`LOCAL_RELATION_SETTINGS_BYTES`]; `None` when it is not, and the server's local-relation
/// settings decide how it is sent, as [`create_relation`](crate::arrow_input::create_relation)
/// reads them.
pub fn small_local_relation(batch: &RecordBatch, schema: &DataType) -> Option<Result<LogicalPlan>> {
    if batch_data_size(batch) >= LOCAL_RELATION_SETTINGS_BYTES {
        return None;
    }
    Some(
        batch_to_arrow_ipc(batch).map(|data| LogicalPlan::LocalRelation {
            schema: Some(schema.clone()),
            data: Some(data),
        }),
    )
}
