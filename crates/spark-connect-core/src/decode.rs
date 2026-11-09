//! The decoding of the Arrow batches of a result into record batches and rows.

use std::sync::Arc;

use spark_connect_proto as proto;

use crate::error::{Result, SparkError};
use crate::row::{Row, StructValue, Value};
use crate::types::DataType;

/// Decode an Arrow batch into rows.
pub fn decode_arrow_batch(batch: &proto::execute_plan_response::ArrowBatch) -> Result<Vec<Row>> {
    decode_arrow_batch_of(batch, None)
}

/// The fields of `schema`, a struct of `columns` fields, for the columns of a batch: none of
/// another schema, whose names the batch keeps.
fn spark_fields(schema: Option<&DataType>, columns: usize) -> Option<&[crate::types::StructField]> {
    match schema {
        Some(DataType::Struct { fields }) if fields.len() == columns => Some(fields),
        _ => None,
    }
}

/// The rows of `batch`, of the names of `schema`, the schema of the result that the server sends
/// before the rows, if any: as PySpark's `ArrowTableToRowsConversion` builds them by the schema,
/// the fields of nested structs have the names of the schema, which keep the duplicate names
/// that the Arrow data of the server numbers (`a_0`, `a_1`).
pub fn decode_arrow_batch_of(
    batch: &proto::execute_plan_response::ArrowBatch,
    spark_schema: Option<&DataType>,
) -> Result<Vec<Row>> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    if batch.data.is_empty() {
        return Ok(vec![]);
    }

    let cursor = Cursor::new(&batch.data);
    let mut reader = StreamReader::try_new(cursor, None).map_err(|e| {
        SparkError::connect_msg(format!("Failed to create Arrow stream reader: {}", e))
    })?;

    let mut rows = vec![];

    while let Some(record_batch) = reader
        .next()
        .transpose()
        .map_err(|e| SparkError::connect_msg(format!("Failed to decode Arrow batch: {}", e)))?
    {
        let schema = record_batch.schema();
        let num_rows = record_batch.num_rows();
        if num_rows == 0 {
            continue;
        }
        let fields = spark_fields(spark_schema, record_batch.num_columns());
        let field_names = match fields {
            Some(fields) => fields.iter().map(|field| field.name.clone()).collect(),
            None => schema
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect::<Arc<[String]>>(),
        };
        let layouts = record_batch
            .columns()
            .iter()
            .enumerate()
            .map(|(i, column)| {
                ArrowValueLayout::of(column.as_ref(), fields.map(|fields| &fields[i].data_type))
            })
            .collect::<Vec<_>>();

        for row_idx in 0..num_rows {
            let mut values = Vec::with_capacity(record_batch.num_columns());

            for layout in &layouts {
                values.push(layout.value_at(row_idx)?);
            }

            rows.push(Row::with_shared_fields(Arc::clone(&field_names), values));
        }
    }

    Ok(rows)
}

/// Decode Arrow IPC stream into RecordBatches without converting to Rows.
///
/// Used by `collect_record_batches()` to provide raw Arrow data for conversions
/// to DataFusion and Polars.
pub fn decode_arrow_record_batches(
    batch: &proto::execute_plan_response::ArrowBatch,
) -> Result<Vec<arrow::record_batch::RecordBatch>> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    if batch.data.is_empty() {
        return Ok(vec![]);
    }

    let cursor = Cursor::new(&batch.data);
    let mut reader = StreamReader::try_new(cursor, None).map_err(|e| {
        SparkError::connect_msg(format!("Failed to create Arrow stream reader: {}", e))
    })?;

    let mut batches = vec![];

    while let Some(record_batch) = reader
        .next()
        .transpose()
        .map_err(|e| SparkError::connect_msg(format!("Failed to decode Arrow batch: {}", e)))?
    {
        batches.push(record_batch);
    }

    Ok(batches)
}

/// Extract a value at a specific index from an Arrow array.
/// Format an unscaled `i128` and scale as a decimal string (e.g. 150, scale 2 -> "1.50").
#[doc(hidden)]
pub fn i128_to_decimal_string(unscaled: i128, scale: i32) -> String {
    if scale < 0 {
        // A negative scale, of spark.sql.legacy.allowNegativeScaleOfDecimal: the unscaled value
        // times ten to the negated scale, in scientific notation, as PyArrow's Decimal of it,
        // which PySpark collects: Decimal("1E+1") of 1 at the scale -1.
        return format!("{unscaled}E+{}", -i64::from(scale));
    }
    if scale == 0 {
        return unscaled.to_string();
    }
    let scale = scale as usize;
    let neg = unscaled < 0;
    let mut digits = unscaled.unsigned_abs().to_string();
    if digits.len() <= scale {
        digits = format!("{}{}", "0".repeat(scale - digits.len() + 1), digits);
    }
    let point = digits.len() - scale;
    let s = format!("{}.{}", &digits[..point], &digits[point..]);
    if neg {
        format!("-{s}")
    } else {
        s
    }
}

/// A borrowed column decoder, resolved once per record batch. Returned Values
/// own their contents; neither rows nor nested values retain the Arrow arrays.
#[doc(hidden)]
pub struct ArrowValueLayout<'a> {
    array: &'a dyn arrow::array::Array,
    decoder: ArrowValueDecoder<'a>,
}

/// The Value of a timestamp of `micros` since the epoch. Spark's Arrow types declare
/// TIMESTAMP_NTZ without a time zone and TIMESTAMP with one (`ArrowUtils.toArrowType`), so the
/// time zone of the Arrow type tells the two apart.
fn timestamp_value(micros: i64, has_time_zone: bool) -> Value {
    if has_time_zone {
        Value::Timestamp(micros)
    } else {
        Value::TimestampNtz(micros)
    }
}

enum ArrowValueDecoder<'a> {
    Null,
    Boolean(&'a arrow::array::BooleanArray),
    Int8(&'a arrow::array::Int8Array),
    Int16(&'a arrow::array::Int16Array),
    Int32(&'a arrow::array::Int32Array),
    Int64(&'a arrow::array::Int64Array),
    Float32(&'a arrow::array::Float32Array),
    Float64(&'a arrow::array::Float64Array),
    String(&'a arrow::array::StringArray),
    Binary(&'a arrow::array::BinaryArray),
    Date32(&'a arrow::array::Date32Array),
    TimestampMicrosecond(&'a arrow::array::TimestampMicrosecondArray),
    UInt8(&'a arrow::array::UInt8Array),
    UInt16(&'a arrow::array::UInt16Array),
    UInt32(&'a arrow::array::UInt32Array),
    UInt64(&'a arrow::array::UInt64Array),
    Decimal128(&'a arrow::array::Decimal128Array),
    LargeString(&'a arrow::array::LargeStringArray),
    LargeBinary(&'a arrow::array::LargeBinaryArray),
    StringView(&'a arrow::array::StringViewArray),
    BinaryView(&'a arrow::array::BinaryViewArray),
    TimestampSecond(&'a arrow::array::TimestampSecondArray),
    TimestampMillisecond(&'a arrow::array::TimestampMillisecondArray),
    TimestampNanosecond(&'a arrow::array::TimestampNanosecondArray),
    Date64(&'a arrow::array::Date64Array),
    DurationMicrosecond(&'a arrow::array::DurationMicrosecondArray),
    Decimal256(&'a arrow::array::Decimal256Array),
    FixedSizeBinary(&'a arrow::array::FixedSizeBinaryArray),
    Time64Microsecond(&'a arrow::array::Time64MicrosecondArray),
    Time64Nanosecond(&'a arrow::array::Time64NanosecondArray),
    Time32Millisecond(&'a arrow::array::Time32MillisecondArray),
    Time32Second(&'a arrow::array::Time32SecondArray),
    IntervalYearMonth(&'a arrow::array::IntervalYearMonthArray),
    IntervalDayTime(&'a arrow::array::IntervalDayTimeArray),
    IntervalMonthDayNano(&'a arrow::array::IntervalMonthDayNanoArray),
    List {
        array: &'a arrow::array::ListArray,
        child: Box<ArrowValueLayout<'a>>,
    },
    Struct {
        names: Arc<[String]>,
        children: Vec<ArrowValueLayout<'a>>,
        variant_fields: Option<(Option<usize>, Option<usize>)>,
    },
    Map {
        array: &'a arrow::array::MapArray,
        children: Box<(ArrowValueLayout<'a>, ArrowValueLayout<'a>)>,
    },
    Unsupported,
}

impl<'a> ArrowValueLayout<'a> {
    #[doc(hidden)]
    pub fn new(array: &'a dyn arrow::array::Array) -> Self {
        Self::of(array, None)
    }

    /// The layout of `array`, of the Spark type `data_type` if known: the fields of its structs,
    /// also in arrays, maps and user-defined types, have the names of the type.
    fn of(array: &'a dyn arrow::array::Array, data_type: Option<&DataType>) -> Self {
        use arrow::array::*;

        // A user-defined type is in the Arrow data of its SQL type.
        let mut data_type = data_type;
        while let Some(DataType::Udt {
            sql_type: Some(sql_type),
            ..
        }) = data_type
        {
            data_type = Some(sql_type);
        }

        // Resolution never fails: null parents and empty containers must not
        // reject unsupported children that are never read.
        let any = array.as_any();
        let decoder = if any.is::<NullArray>() {
            ArrowValueDecoder::Null
        } else if let Some(array) = any.downcast_ref::<BooleanArray>() {
            ArrowValueDecoder::Boolean(array)
        } else if let Some(array) = any.downcast_ref::<Int8Array>() {
            ArrowValueDecoder::Int8(array)
        } else if let Some(array) = any.downcast_ref::<Int16Array>() {
            ArrowValueDecoder::Int16(array)
        } else if let Some(array) = any.downcast_ref::<Int32Array>() {
            ArrowValueDecoder::Int32(array)
        } else if let Some(array) = any.downcast_ref::<Int64Array>() {
            ArrowValueDecoder::Int64(array)
        } else if let Some(array) = any.downcast_ref::<Float32Array>() {
            ArrowValueDecoder::Float32(array)
        } else if let Some(array) = any.downcast_ref::<Float64Array>() {
            ArrowValueDecoder::Float64(array)
        } else if let Some(array) = any.downcast_ref::<StringArray>() {
            ArrowValueDecoder::String(array)
        } else if let Some(array) = any.downcast_ref::<BinaryArray>() {
            ArrowValueDecoder::Binary(array)
        } else if let Some(array) = any.downcast_ref::<Date32Array>() {
            ArrowValueDecoder::Date32(array)
        } else if let Some(array) = any.downcast_ref::<TimestampMicrosecondArray>() {
            ArrowValueDecoder::TimestampMicrosecond(array)
        } else if let Some(array) = any.downcast_ref::<UInt8Array>() {
            ArrowValueDecoder::UInt8(array)
        } else if let Some(array) = any.downcast_ref::<UInt16Array>() {
            ArrowValueDecoder::UInt16(array)
        } else if let Some(array) = any.downcast_ref::<UInt32Array>() {
            ArrowValueDecoder::UInt32(array)
        } else if let Some(array) = any.downcast_ref::<UInt64Array>() {
            ArrowValueDecoder::UInt64(array)
        } else if let Some(array) = any.downcast_ref::<Decimal128Array>() {
            ArrowValueDecoder::Decimal128(array)
        } else if let Some(array) = any.downcast_ref::<LargeStringArray>() {
            ArrowValueDecoder::LargeString(array)
        } else if let Some(array) = any.downcast_ref::<LargeBinaryArray>() {
            ArrowValueDecoder::LargeBinary(array)
        } else if let Some(array) = any.downcast_ref::<StringViewArray>() {
            ArrowValueDecoder::StringView(array)
        } else if let Some(array) = any.downcast_ref::<BinaryViewArray>() {
            ArrowValueDecoder::BinaryView(array)
        } else if let Some(array) = any.downcast_ref::<TimestampSecondArray>() {
            ArrowValueDecoder::TimestampSecond(array)
        } else if let Some(array) = any.downcast_ref::<TimestampMillisecondArray>() {
            ArrowValueDecoder::TimestampMillisecond(array)
        } else if let Some(array) = any.downcast_ref::<TimestampNanosecondArray>() {
            ArrowValueDecoder::TimestampNanosecond(array)
        } else if let Some(array) = any.downcast_ref::<Date64Array>() {
            ArrowValueDecoder::Date64(array)
        } else if let Some(array) = any.downcast_ref::<DurationMicrosecondArray>() {
            ArrowValueDecoder::DurationMicrosecond(array)
        } else if let Some(array) = any.downcast_ref::<Decimal256Array>() {
            ArrowValueDecoder::Decimal256(array)
        } else if let Some(array) = any.downcast_ref::<FixedSizeBinaryArray>() {
            ArrowValueDecoder::FixedSizeBinary(array)
        } else if let Some(array) = any.downcast_ref::<Time64MicrosecondArray>() {
            ArrowValueDecoder::Time64Microsecond(array)
        } else if let Some(array) = any.downcast_ref::<Time64NanosecondArray>() {
            ArrowValueDecoder::Time64Nanosecond(array)
        } else if let Some(array) = any.downcast_ref::<Time32MillisecondArray>() {
            ArrowValueDecoder::Time32Millisecond(array)
        } else if let Some(array) = any.downcast_ref::<Time32SecondArray>() {
            ArrowValueDecoder::Time32Second(array)
        } else if let Some(array) = any.downcast_ref::<IntervalYearMonthArray>() {
            ArrowValueDecoder::IntervalYearMonth(array)
        } else if let Some(array) = any.downcast_ref::<IntervalDayTimeArray>() {
            ArrowValueDecoder::IntervalDayTime(array)
        } else if let Some(array) = any.downcast_ref::<IntervalMonthDayNanoArray>() {
            ArrowValueDecoder::IntervalMonthDayNano(array)
        } else if let Some(array) = any.downcast_ref::<ListArray>() {
            let element_type = match data_type {
                Some(DataType::Array { element_type, .. }) => Some(element_type.as_ref()),
                _ => None,
            };
            ArrowValueDecoder::List {
                array,
                child: Box::new(Self::of(array.values().as_ref(), element_type)),
            }
        } else if let Some(array) = any.downcast_ref::<StructArray>() {
            let fields = array.fields();
            let is_variant = fields.iter().any(|field| {
                field
                    .metadata()
                    .get("variant")
                    .is_some_and(|value| value == "true")
            });
            let spark_fields = match data_type {
                Some(DataType::Struct { fields: spark }) if spark.len() == fields.len() => {
                    Some(spark)
                }
                _ => None,
            };
            ArrowValueDecoder::Struct {
                names: match spark_fields {
                    Some(spark) => spark.iter().map(|field| field.name.clone()).collect(),
                    None => fields.iter().map(|field| field.name().clone()).collect(),
                },
                children: array
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(i, child)| {
                        Self::of(
                            child.as_ref(),
                            spark_fields.map(|spark| &spark[i].data_type),
                        )
                    })
                    .collect(),
                variant_fields: is_variant.then(|| {
                    (
                        fields.iter().position(|field| field.name() == "value"),
                        fields.iter().position(|field| field.name() == "metadata"),
                    )
                }),
            }
        } else if let Some(array) = any.downcast_ref::<MapArray>() {
            let (key_type, value_type) = match data_type {
                Some(DataType::Map {
                    key_type,
                    value_type,
                    ..
                }) => (Some(key_type.as_ref()), Some(value_type.as_ref())),
                _ => (None, None),
            };
            ArrowValueDecoder::Map {
                array,
                children: Box::new((
                    Self::of(array.keys().as_ref(), key_type),
                    Self::of(array.values().as_ref(), value_type),
                )),
            }
        } else {
            ArrowValueDecoder::Unsupported
        };
        Self { array, decoder }
    }

    pub fn value_at(&self, index: usize) -> Result<Value> {
        if self.array.is_null(index) {
            return Ok(Value::Null);
        }
        match &self.decoder {
            // NullType may have no null buffer; every element is still Null.
            ArrowValueDecoder::Null => Ok(Value::Null),
            ArrowValueDecoder::Boolean(arr) => Ok(Value::Bool(arr.value(index))),
            ArrowValueDecoder::Int8(arr) => Ok(Value::Byte(arr.value(index))),
            ArrowValueDecoder::Int16(arr) => Ok(Value::Short(arr.value(index))),
            ArrowValueDecoder::Int32(arr) => Ok(Value::Integer(arr.value(index))),
            ArrowValueDecoder::Int64(arr) => Ok(Value::Long(arr.value(index))),
            ArrowValueDecoder::Float32(arr) => Ok(Value::Float(arr.value(index))),
            ArrowValueDecoder::Float64(arr) => Ok(Value::Double(arr.value(index))),
            ArrowValueDecoder::String(arr) => Ok(Value::String(arr.value(index).to_string())),
            ArrowValueDecoder::Binary(arr) => Ok(Value::Binary(arr.value(index).to_vec())),
            ArrowValueDecoder::Date32(arr) => Ok(Value::Date(arr.value(index))),
            ArrowValueDecoder::TimestampMicrosecond(arr) => {
                Ok(timestamp_value(arr.value(index), arr.timezone().is_some()))
            }
            ArrowValueDecoder::UInt8(arr) => Ok(Value::Short(arr.value(index) as i16)),
            ArrowValueDecoder::UInt16(arr) => Ok(Value::Integer(arr.value(index) as i32)),
            ArrowValueDecoder::UInt32(arr) => Ok(Value::Long(arr.value(index) as i64)),
            ArrowValueDecoder::UInt64(arr) => {
                let val = arr.value(index);
                let i64_val = i64::try_from(val).map_err(|_| {
                    SparkError::connect_msg(format!("UInt64 value {} exceeds i64 range", val))
                })?;
                Ok(Value::Long(i64_val))
            }
            // Decimal128 stays exact, preserving precision/scale rather than using lossy f64.
            ArrowValueDecoder::Decimal128(arr) => {
                let scale = arr.scale() as i32;
                Ok(Value::Decimal {
                    value: i128_to_decimal_string(arr.value(index), scale),
                    precision: Some(arr.precision() as i32),
                    scale: Some(scale),
                })
            }
            ArrowValueDecoder::LargeString(arr) => Ok(Value::String(arr.value(index).to_string())),
            ArrowValueDecoder::LargeBinary(arr) => Ok(Value::Binary(arr.value(index).to_vec())),
            ArrowValueDecoder::StringView(arr) => Ok(Value::String(arr.value(index).to_string())),
            ArrowValueDecoder::BinaryView(arr) => Ok(Value::Binary(arr.value(index).to_vec())),
            ArrowValueDecoder::TimestampSecond(arr) => Ok(timestamp_value(
                arr.value(index) * 1_000_000,
                arr.timezone().is_some(),
            )),
            ArrowValueDecoder::TimestampMillisecond(arr) => Ok(timestamp_value(
                arr.value(index) * 1_000,
                arr.timezone().is_some(),
            )),
            // Collect gives microseconds: floor, so a pre-1970 value keeps its sign and its
            // sub-microsecond digits round down as Spark's micros do (`to_arrow` keeps nanos).
            ArrowValueDecoder::TimestampNanosecond(arr) => Ok(timestamp_value(
                arr.value(index).div_euclid(1_000),
                arr.timezone().is_some(),
            )),
            ArrowValueDecoder::Date64(arr) => {
                Ok(Value::Date((arr.value(index) / 86_400_000) as i32))
            }
            ArrowValueDecoder::DurationMicrosecond(arr) => {
                Ok(Value::DayTimeInterval(arr.value(index)))
            }
            // Decimal256 also preserves exact digits; Arrow formats with the scale.
            ArrowValueDecoder::Decimal256(arr) => Ok(Value::Decimal {
                value: arr.value_as_string(index),
                precision: Some(arr.precision() as i32),
                scale: Some(arr.scale() as i32),
            }),
            ArrowValueDecoder::FixedSizeBinary(arr) => Ok(Value::Binary(arr.value(index).to_vec())),
            // A TIME, in nanoseconds since midnight, as PySpark's `TimeType.fromInternal` reads.
            ArrowValueDecoder::Time64Microsecond(arr) => Ok(Value::Time(arr.value(index) * 1_000)),
            ArrowValueDecoder::Time64Nanosecond(arr) => Ok(Value::Time(arr.value(index))),
            ArrowValueDecoder::Time32Millisecond(arr) => {
                Ok(Value::Time(arr.value(index) as i64 * 1_000_000))
            }
            ArrowValueDecoder::Time32Second(arr) => {
                Ok(Value::Time(arr.value(index) as i64 * 1_000_000_000))
            }
            // Intervals likewise have no dedicated Value variant, so render as strings.
            ArrowValueDecoder::IntervalYearMonth(arr) => {
                // The sign goes on the years, as Spark's `toYearMonthIntervalString`
                // does, so -1 month is "-0-1" and not "0-1".
                let months = arr.value(index) as i64;
                let sign = if months < 0 { "-" } else { "" };
                let abs = months.abs();
                Ok(Value::String(format!("{sign}{}-{}", abs / 12, abs % 12)))
            }
            ArrowValueDecoder::IntervalDayTime(arr) => {
                let v = arr.value(index);
                Ok(Value::String(format!(
                    "{} days {} ms",
                    v.days, v.milliseconds
                )))
            }
            ArrowValueDecoder::IntervalMonthDayNano(arr) => {
                let v = arr.value(index);
                Ok(Value::String(format!(
                    "{} months {} days {} ns",
                    v.months, v.days, v.nanoseconds
                )))
            }
            ArrowValueDecoder::List { array, child } => {
                // Sliced list/map arrays retain their original children. Their
                // offsets already identify the correct absolute child range.
                let offsets = array.value_offsets();
                let start = offsets[index] as usize;
                let end = offsets[index + 1] as usize;
                let mut items = Vec::with_capacity(end - start);
                for child_index in start..end {
                    items.push(child.value_at(child_index)?);
                }
                Ok(Value::List(items))
            }
            ArrowValueDecoder::Struct {
                names,
                children,
                variant_fields,
            } => {
                // VARIANT arrives as struct<value: binary, metadata: binary>, with
                // {"variant": "true"} on the metadata field. Return raw-byte
                // Value::Variant so it materializes as pyspark's VariantVal rather
                // than a plain {value, metadata} struct/dict.
                if let Some((value, metadata)) = variant_fields {
                    let binary_field = |column: Option<usize>| -> Result<Vec<u8>> {
                        match column {
                            Some(column) => match children[column].value_at(index)? {
                                Value::Binary(bytes) => Ok(bytes),
                                Value::Null => Ok(vec![]),
                                _ => Err(SparkError::connect_msg("variant field is not binary")),
                            },
                            None => Ok(vec![]),
                        }
                    };
                    return Ok(Value::Variant {
                        value: binary_field(*value)?,
                        metadata: binary_field(*metadata)?,
                    });
                }
                let mut values = Vec::with_capacity(children.len());
                for child in children {
                    values.push(child.value_at(index)?);
                }
                Ok(Value::Struct(StructValue::new(Arc::clone(names), values)))
            }
            ArrowValueDecoder::Map { array, children } => {
                let offsets = array.value_offsets();
                let start = offsets[index] as usize;
                let end = offsets[index + 1] as usize;
                let mut map = Vec::with_capacity(end - start);
                for child_index in start..end {
                    map.push((
                        children.0.value_at(child_index)?,
                        children.1.value_at(child_index)?,
                    ));
                }
                Ok(Value::Map(map))
            }
            ArrowValueDecoder::Unsupported => Err(SparkError::connect_msg(format!(
                "Unsupported Arrow type {:?} - cannot convert to Value",
                self.array.data_type()
            ))),
        }
    }
}
