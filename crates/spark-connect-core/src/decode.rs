//! The decoding of the Arrow batches of a result into record batches and rows.

use std::sync::Arc;

use spark_connect_proto as proto;

use crate::error::{Result, SparkError};
use crate::row::{Row, StructValue, Value};

/// Decode an Arrow batch into rows.
pub fn decode_arrow_batch(batch: &proto::execute_plan_response::ArrowBatch) -> Result<Vec<Row>> {
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
        let field_names = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect::<Arc<[String]>>();
        let layouts = record_batch
            .columns()
            .iter()
            .map(|column| ArrowValueLayout::new(column.as_ref()))
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
    if scale <= 0 {
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
    pub fn new(array: &'a dyn arrow::array::Array) -> Self {
        use arrow::array::*;

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
            ArrowValueDecoder::List {
                array,
                child: Box::new(Self::new(array.values().as_ref())),
            }
        } else if let Some(array) = any.downcast_ref::<StructArray>() {
            let fields = array.fields();
            let is_variant = fields.iter().any(|field| {
                field
                    .metadata()
                    .get("variant")
                    .is_some_and(|value| value == "true")
            });
            ArrowValueDecoder::Struct {
                names: fields.iter().map(|field| field.name().clone()).collect(),
                children: array
                    .columns()
                    .iter()
                    .map(|child| Self::new(child.as_ref()))
                    .collect(),
                variant_fields: is_variant.then(|| {
                    (
                        fields.iter().position(|field| field.name() == "value"),
                        fields.iter().position(|field| field.name() == "metadata"),
                    )
                }),
            }
        } else if let Some(array) = any.downcast_ref::<MapArray>() {
            ArrowValueDecoder::Map {
                array,
                children: Box::new((
                    Self::new(array.keys().as_ref()),
                    Self::new(array.values().as_ref()),
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
            ArrowValueDecoder::TimestampMicrosecond(arr) => Ok(Value::Timestamp(arr.value(index))),
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
            ArrowValueDecoder::TimestampSecond(arr) => {
                Ok(Value::Timestamp(arr.value(index) * 1_000_000))
            }
            ArrowValueDecoder::TimestampMillisecond(arr) => {
                Ok(Value::Timestamp(arr.value(index) * 1_000))
            }
            ArrowValueDecoder::TimestampNanosecond(arr) => {
                Ok(Value::Timestamp(arr.value(index) / 1_000))
            }
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
                let months = arr.value(index);
                Ok(Value::String(format!(
                    "{}-{}",
                    months / 12,
                    (months % 12).abs()
                )))
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
