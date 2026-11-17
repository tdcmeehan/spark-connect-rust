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
    let (record_batches, _) = read_record_batches(&batch.data, decode_error)?;

    let mut rows = vec![];

    for record_batch in record_batches {
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

/// The record batches of the Arrow IPC stream `data`, and the schema of them when they are read
/// with the fields nullable, as [`read_stream_relaxed`] reads them.
///
/// arrow-rs refuses a batch with nulls in a non-nullable field, at any depth, but the server
/// declares such a field for values that are nulls, as `arr.getItem(None)` does, and PyArrow
/// reads them: such a stream is read again with its fields nullable, and the batches of it have
/// that schema. Any other error of a batch is `batch_error` of it.
fn read_record_batches(
    data: &[u8],
    batch_error: impl Fn(&[u8], arrow::error::ArrowError) -> SparkError,
) -> Result<(
    Vec<arrow::record_batch::RecordBatch>,
    Option<arrow::datatypes::SchemaRef>,
)> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    if data.is_empty() {
        return Ok((vec![], None));
    }

    let reader = StreamReader::try_new(Cursor::new(data), None).map_err(|e| {
        SparkError::connect_msg(format!("Failed to create Arrow stream reader: {}", e))
    })?;

    let mut batches = vec![];
    for item in reader {
        match item {
            Ok(batch) => batches.push(batch),
            Err(error) => {
                let relaxed = is_non_nullable_error(&error)
                    .then(|| read_stream_relaxed(data, false))
                    .flatten();
                return match relaxed {
                    Some((_, relaxed)) => {
                        let schema = relaxed.first().map(|batch| batch.schema());
                        Ok((relaxed, schema))
                    }
                    // The strings of a stream that does not read with its fields nullable are
                    // checked as those of any other, so that an invalid one is still named.
                    None => Err(match first_invalid_utf8(data) {
                        Some((column, value)) => SparkError::invalid_utf8(column, &value),
                        None => batch_error(data, error),
                    }),
                };
            }
        }
    }
    Ok((batches, None))
}

/// The batches of a result, which share one schema: when some were read with their fields
/// nullable, as `relaxed` is, each batch takes `relaxed`. A batch read without that relaxation
/// only has its fields widened to nullable, which `with_schema` allows, so no data changes.
#[doc(hidden)]
pub fn share_schema(
    batches: Vec<arrow::record_batch::RecordBatch>,
    relaxed: Option<arrow::datatypes::SchemaRef>,
) -> Result<Vec<arrow::record_batch::RecordBatch>> {
    let Some(relaxed) = relaxed else {
        return Ok(batches);
    };
    batches
        .into_iter()
        .map(|batch| batch.with_schema(relaxed.clone()))
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| SparkError::connect_msg(format!("Failed to decode Arrow batch: {e}")))
}

/// Whether `error` is arrow-rs refusing a non-nullable field, at any depth, for its nulls.
fn is_non_nullable_error(error: &arrow::error::ArrowError) -> bool {
    let message = error.to_string();
    message.contains("non-nullable") || message.contains("Non-nullable")
}

/// The error of an Arrow IPC stream `data` that arrow-rs fails with `error`. A string that is not
/// valid UTF-8 is an error of its own, with its bytes, which the Python client raises as the
/// reference client does.
fn decode_error(data: &[u8], error: arrow::error::ArrowError) -> SparkError {
    let message = error.to_string();
    if message.contains("UTF8 sequence") || message.contains("utf-8 byte sequence") {
        if let Some((column, value)) = first_invalid_utf8(data) {
            return SparkError::invalid_utf8(column, &value);
        }
    }
    SparkError::connect_msg(format!("Failed to decode Arrow batch: {}", error))
}

/// The first string of the Arrow IPC stream `data` that is not valid UTF-8, with its column and
/// its bytes, in the order in which PySpark converts a result: column by column, each batch by
/// batch, in the order of [`first_invalid_utf8_in`].
#[doc(hidden)]
pub fn first_invalid_utf8(data: &[u8]) -> Option<(usize, Vec<u8>)> {
    let (strings, batches) = read_stream_relaxed(data, true)?;
    let columns = batches.first()?.num_columns();
    (0..columns).find_map(|column| {
        let data_type = strings.field(column).data_type();
        batches
            .iter()
            .find_map(|batch| first_invalid_utf8_in(batch.column(column).as_ref(), data_type))
            .map(|value| (column, value))
    })
}

/// The schema of the Arrow IPC stream `data`, as the server sends it, and its record batches, read
/// with each field nullable (see [`read_schema`]) and, if `strings_as_binary`, with its string
/// types read as binary types of the same layout. arrow-rs still checks the offsets and lengths of
/// these batches, but not their UTF-8 when they are read as binary, nor their nulls. The messages
/// are split here because arrow-rs's stream reader cannot take a schema of its own. None for a
/// stream with a message that is not a schema or a record batch, such as a dictionary, or one that
/// does not read.
fn read_stream_relaxed(
    data: &[u8],
    strings_as_binary: bool,
) -> Option<(
    arrow::datatypes::Schema,
    Vec<arrow::record_batch::RecordBatch>,
)> {
    use arrow::buffer::Buffer;
    use arrow::ipc::convert::fb_to_schema;
    use arrow::ipc::reader::read_record_batch;
    use arrow::ipc::{root_as_message, MessageHeader};
    use std::collections::HashMap;

    fn take<'a>(rest: &mut &'a [u8], len: usize) -> Option<&'a [u8]> {
        let (head, tail) = rest.split_at_checked(len)?;
        *rest = tail;
        Some(head)
    }

    let mut rest = data;
    let mut schema = None;
    let mut strings = None;
    let mut batches = vec![];
    loop {
        let mut prefix = take(&mut rest, 4)?;
        if prefix == [0xff; 4] {
            prefix = take(&mut rest, 4)?;
        }
        let meta_len = u32::from_le_bytes(prefix.try_into().ok()?) as usize;
        if meta_len == 0 {
            break;
        }
        let message = root_as_message(take(&mut rest, meta_len)?).ok()?;
        let body = take(&mut rest, usize::try_from(message.bodyLength()).ok()?)?;
        match message.header_type() {
            // A stream has one schema: with another, its batches would not have the columns
            // of the types read, so it does not read.
            MessageHeader::Schema if strings.is_some() => return None,
            MessageHeader::Schema => {
                let fields = fb_to_schema(message.header_as_schema()?);
                schema = Some(Arc::new(read_schema(&fields, strings_as_binary)));
                strings = Some(fields);
            }
            MessageHeader::RecordBatch => {
                let batch = read_record_batch(
                    &Buffer::from(body.to_vec()),
                    message.header_as_record_batch()?,
                    schema.clone()?,
                    &HashMap::new(),
                    None,
                    &message.version(),
                )
                .ok()?;
                batches.push(batch);
            }
            _ => return None,
        }
    }
    Some((strings?, batches))
}

/// The schema `schema` as its batches are read by [`read_stream_relaxed`]: each field nullable,
/// at any depth (see [`read_field`]), and, if `strings_as_binary`, each string type of it as the
/// binary type of the same layout (see [`read_type`]).
fn read_schema(
    schema: &arrow::datatypes::Schema,
    strings_as_binary: bool,
) -> arrow::datatypes::Schema {
    let fields: Vec<arrow::datatypes::Field> = schema
        .fields()
        .iter()
        .map(|field| read_field(field, strings_as_binary))
        .collect();
    arrow::datatypes::Schema::new_with_metadata(fields, schema.metadata().clone())
}

/// The field `field` as it is read: nullable, so that arrow-rs does not refuse its nulls, with
/// its type as [`read_type`] reads it.
fn read_field(field: &arrow::datatypes::Field, strings_as_binary: bool) -> arrow::datatypes::Field {
    field
        .clone()
        .with_data_type(read_type(field.data_type(), strings_as_binary))
        .with_nullable(true)
}

/// The type `data_type` as it is read: its string types, at any depth, as the binary types of the
/// same layout if `strings_as_binary` (Utf8 as Binary, LargeUtf8 as LargeBinary and Utf8View as
/// BinaryView), and its fields nullable (see [`read_field`]), except the entries of a map, whose
/// keys arrow-rs requires to be non-nullable (see [`read_map_entries`]).
fn read_type(
    data_type: &arrow::datatypes::DataType,
    strings_as_binary: bool,
) -> arrow::datatypes::DataType {
    use arrow::datatypes::DataType as ArrowDataType;

    let read = |field: &arrow::datatypes::Field| read_field(field, strings_as_binary);
    match data_type {
        ArrowDataType::Utf8 if strings_as_binary => ArrowDataType::Binary,
        ArrowDataType::LargeUtf8 if strings_as_binary => ArrowDataType::LargeBinary,
        ArrowDataType::Utf8View if strings_as_binary => ArrowDataType::BinaryView,
        ArrowDataType::List(field) => ArrowDataType::List(Arc::new(read(field))),
        ArrowDataType::LargeList(field) => ArrowDataType::LargeList(Arc::new(read(field))),
        ArrowDataType::FixedSizeList(field, size) => {
            ArrowDataType::FixedSizeList(Arc::new(read(field)), *size)
        }
        ArrowDataType::Map(entries, sorted) => ArrowDataType::Map(
            Arc::new(read_map_entries(entries, strings_as_binary)),
            *sorted,
        ),
        ArrowDataType::Struct(fields) => {
            ArrowDataType::Struct(fields.iter().map(|field| Arc::new(read(field))).collect())
        }
        other => other.clone(),
    }
}

/// The entries field of a map as it is read (see [`read_type`]): arrow-rs requires the entries
/// to be non-nullable and their key too, as a map has no null entry or key, so only the value is
/// read as any other field.
fn read_map_entries(
    entries: &arrow::datatypes::Field,
    strings_as_binary: bool,
) -> arrow::datatypes::Field {
    use arrow::datatypes::DataType as ArrowDataType;

    let ArrowDataType::Struct(children) = entries.data_type() else {
        return entries.clone();
    };
    let children = children
        .iter()
        .enumerate()
        .map(|(i, child)| {
            let child = if i == 0 {
                child
                    .as_ref()
                    .clone()
                    .with_data_type(read_type(child.data_type(), strings_as_binary))
            } else {
                read_field(child, strings_as_binary)
            };
            Arc::new(child)
        })
        .collect();
    entries
        .clone()
        .with_data_type(ArrowDataType::Struct(children))
}

/// The bytes of the first string of `array` that is not valid UTF-8, in the order in which
/// PySpark's `ArrowTableToRowsConversion._to_pylist` converts an array, in bulk: a list by the
/// values of all its rows, a struct field by field, each over all its rows, and a map by the keys
/// of all its rows, then their values. A null is valid. `data_type` is the type of `array` as the
/// server sends it, so that only its strings, at any depth, are checked: a binary value is not.
#[doc(hidden)]
pub fn first_invalid_utf8_in(
    array: &dyn arrow::array::Array,
    data_type: &arrow::datatypes::DataType,
) -> Option<Vec<u8>> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::DataType as ArrowDataType;

    // The values of the rows of a list or a map, which its offsets delimit in its child, of the
    // type `child_type`.
    let rows_of = |child: &dyn Array, child_type: &ArrowDataType, start: usize, end: usize| {
        first_invalid_utf8_in(child.slice(start, end - start).as_ref(), child_type)
    };
    match data_type {
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            invalid_string_in(array)
        }
        ArrowDataType::List(field) => {
            let list = array.as_list::<i32>();
            let offsets = list.value_offsets();
            let (start, end) = (offsets[0] as usize, offsets[list.len()] as usize);
            rows_of(list.values().as_ref(), field.data_type(), start, end)
        }
        ArrowDataType::LargeList(field) => {
            let list = array.as_list::<i64>();
            let offsets = list.value_offsets();
            let (start, end) = (offsets[0] as usize, offsets[list.len()] as usize);
            rows_of(list.values().as_ref(), field.data_type(), start, end)
        }
        // A slice of a fixed-size list slices its values.
        ArrowDataType::FixedSizeList(field, _) => first_invalid_utf8_in(
            array.as_fixed_size_list().values().as_ref(),
            field.data_type(),
        ),
        ArrowDataType::Map(entries, _) => {
            let ArrowDataType::Struct(fields) = entries.data_type() else {
                return None;
            };
            let map = array.as_map();
            let offsets = map.value_offsets();
            let (start, end) = (offsets[0] as usize, offsets[map.len()] as usize);
            rows_of(map.keys().as_ref(), fields[0].data_type(), start, end)
                .or_else(|| rows_of(map.values().as_ref(), fields[1].data_type(), start, end))
        }
        // A slice of a struct slices its fields.
        ArrowDataType::Struct(fields) => array
            .as_struct()
            .columns()
            .iter()
            .zip(fields.iter())
            .find_map(|(column, field)| first_invalid_utf8_in(column.as_ref(), field.data_type())),
        _ => None,
    }
}

/// The bytes of the first value of `array`, a string read as binary, that is not valid UTF-8.
fn invalid_string_in(array: &dyn arrow::array::Array) -> Option<Vec<u8>> {
    use arrow::array::AsArray;
    use arrow::datatypes::DataType as ArrowDataType;

    match array.data_type() {
        ArrowDataType::Binary => array
            .as_binary::<i32>()
            .iter()
            .flatten()
            .find_map(invalid_bytes),
        ArrowDataType::LargeBinary => array
            .as_binary::<i64>()
            .iter()
            .flatten()
            .find_map(invalid_bytes),
        ArrowDataType::BinaryView => array
            .as_binary_view()
            .iter()
            .flatten()
            .find_map(invalid_bytes),
        _ => None,
    }
}

/// The bytes, if they are not valid UTF-8.
fn invalid_bytes(bytes: &[u8]) -> Option<Vec<u8>> {
    std::str::from_utf8(bytes).is_err().then(|| bytes.to_vec())
}

/// Decode Arrow IPC stream into RecordBatches without converting to Rows.
///
/// Used by `collect_record_batches()` to provide raw Arrow data for conversions
/// to DataFusion and Polars.
pub fn decode_arrow_record_batches(
    batch: &proto::execute_plan_response::ArrowBatch,
) -> Result<(
    Vec<arrow::record_batch::RecordBatch>,
    Option<arrow::datatypes::SchemaRef>,
)> {
    read_record_batches(&batch.data, |_, error| {
        SparkError::connect_msg(format!("Failed to decode Arrow batch: {}", error))
    })
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
            // A year-month interval, the months that the server sends of it.
            ArrowValueDecoder::IntervalYearMonth(arr) => {
                Ok(Value::YearMonthInterval(arr.value(index)))
            }
            // These intervals have no dedicated Value variant, so render as strings.
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
