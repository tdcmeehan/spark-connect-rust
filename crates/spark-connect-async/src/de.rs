//! Reading the rows of a result into Rust values: into any type that implements serde's
//! `Deserialize` with [`from_record_batch`], or into [`Row`]s of [`Value`]s with [`rows`].
//!
//! A row deserializes as a struct, by the names of its columns (columns that the struct does not
//! name are skipped), as a tuple or a sequence, by position, or as a map. A row of one column also
//! deserializes as that column's value. A value deserializes into what its Spark type means:
//!
//! | Spark type | Rust types |
//! |---|---|
//! | `BOOLEAN`, `TINYINT` … `BIGINT`, `FLOAT`, `DOUBLE` | `bool`, the integers, the floats |
//! | `DECIMAL` | [`Decimal`] (exact), `String` (exact text), `f64`; an integer if the scale is not positive |
//! | `STRING` | `String`, `&str` |
//! | `BINARY` | `Vec<u8>`, `&[u8]` |
//! | `DATE` | `chrono::NaiveDate` or `String` (`2024-01-31`); `i32` days since 1970-01-01 |
//! | `TIMESTAMP` | `chrono::DateTime<Utc>` or `String` (RFC 3339, with nanoseconds); `i64` microseconds since the epoch |
//! | `TIMESTAMP_NTZ` | `chrono::NaiveDateTime` or `String`; `i64` microseconds |
//! | `TIME` | `chrono::NaiveTime` or `String`; `i64` nanoseconds since midnight |
//! | `INTERVAL DAY TO SECOND` | `i64` microseconds |
//! | `INTERVAL YEAR TO MONTH` | `i32` months |
//! | `ARRAY` | `Vec<T>`, any sequence |
//! | `MAP` | any map, or `Vec<(K, V)>`, in the server's order of the entries |
//! | `STRUCT` | a struct by field name, a tuple by position, or a map |
//! | `VARIANT` | a struct of `value` and `metadata`, both `Vec<u8>` |
//!
//! A `NULL` deserializes into `None` of an `Option`, and is an error elsewhere. An error names its
//! row and column.

use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, GenericListArray, MapArray, OffsetSizeTrait, StructArray,
};
use arrow::datatypes::{
    DataType as ArrowType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
    DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType, DurationSecondType,
    Field, Fields as ArrowFields, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
    Int8Type, IntervalUnit, IntervalYearMonthType, Schema, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, SecondsFormat, TimeDelta, Utc};
use serde::de::{
    self, DeserializeSeed, Deserializer, IntoDeserializer, MapAccess, SeqAccess, Visitor,
};
use serde::Deserialize;
use spark_connect_core::types::{DataType as SparkType, StructField};

use crate::error::{Error, Result};
use crate::value::{Decimal, Row, Value, DECIMAL_PARTS};

/// The rows of `batch`, each deserialized into a `T`.
pub fn from_record_batch<'de, T: Deserialize<'de>>(batch: &'de RecordBatch) -> Result<Vec<T>> {
    (0..batch.num_rows())
        .map(|row| T::deserialize(RowDeserializer { batch, row }).map_err(|e| e.in_row(row)))
        .collect()
}

/// The rows of `batch`.
pub fn rows(batch: &RecordBatch) -> Result<Vec<Row>> {
    let names: Arc<[String]> = batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    (0..batch.num_rows())
        .map(|row| {
            let values = batch
                .columns()
                .iter()
                .zip(names.iter())
                .map(|(column, name)| value_at(column, row).map_err(|e| e.in_field(name)))
                .collect::<Result<_>>();
            Ok(Row::new(
                Arc::clone(&names),
                values.map_err(|e| e.in_row(row))?,
            ))
        })
        .collect()
}

/// `batch` with the column names of `schema`, the Spark schema of the result, nested names
/// included; the Arrow names stand where `schema` is `None` or does not match.
pub(crate) fn with_names_of(batch: RecordBatch, schema: Option<&SparkType>) -> Result<RecordBatch> {
    let Some(SparkType::Struct { fields }) = schema else {
        return Ok(batch);
    };
    if fields.len() != batch.num_columns() || have_names_of(batch.schema().fields(), fields) {
        return Ok(batch);
    }
    let columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .zip(fields)
        .map(|(column, field)| renamed(column, &field.data_type))
        .collect();
    let names = named(batch.schema().fields(), fields, &columns);
    let schema = Arc::new(Schema::new_with_metadata(
        names,
        batch.schema().metadata().clone(),
    ));
    RecordBatch::try_new(schema, columns).map_err(|error| Error::decode(error.to_string()))
}

/// Whether the Arrow fields `arrow` have the names of the Spark fields `spark` at every depth.
fn have_names_of(arrow: &ArrowFields, spark: &[StructField]) -> bool {
    arrow.len() == spark.len()
        && arrow.iter().zip(spark).all(|(arrow, spark)| {
            arrow.name() == &spark.name && has_names_of(arrow.data_type(), &spark.data_type)
        })
}

fn has_names_of(arrow: &ArrowType, spark: &SparkType) -> bool {
    match (arrow, spark) {
        (
            _,
            SparkType::Udt {
                sql_type: Some(sql_type),
                ..
            },
        ) => has_names_of(arrow, sql_type),
        (ArrowType::Struct(arrow), SparkType::Struct { fields }) => have_names_of(arrow, fields),
        (
            ArrowType::List(field) | ArrowType::LargeList(field),
            SparkType::Array { element_type, .. },
        ) => has_names_of(field.data_type(), element_type),
        (
            ArrowType::Map(field, _),
            SparkType::Map {
                key_type,
                value_type,
                ..
            },
        ) => match field.data_type() {
            ArrowType::Struct(entries) if entries.len() == 2 => {
                has_names_of(entries[0].data_type(), key_type)
                    && has_names_of(entries[1].data_type(), value_type)
            }
            _ => true,
        },
        _ => true,
    }
}

/// The Arrow fields `arrow` with the names of the Spark fields `spark` and the types of
/// `columns`.
fn named(arrow: &ArrowFields, spark: &[StructField], columns: &[ArrayRef]) -> Vec<Field> {
    arrow
        .iter()
        .zip(spark)
        .zip(columns)
        .map(|((arrow, spark), column)| retyped(arrow, column).with_name(&spark.name))
        .collect()
}

/// `field`, with the type of `column`.
fn retyped(field: &Field, column: &ArrayRef) -> Field {
    field.clone().with_data_type(column.data_type().clone())
}

/// `array` with the field names of the Spark type `data_type`.
fn renamed(array: &ArrayRef, data_type: &SparkType) -> ArrayRef {
    match (array.data_type(), data_type) {
        (
            _,
            SparkType::Udt {
                sql_type: Some(sql_type),
                ..
            },
        ) => renamed(array, sql_type),
        (ArrowType::Struct(arrow), SparkType::Struct { fields }) if arrow.len() == fields.len() => {
            let structs = array.as_struct();
            let columns: Vec<ArrayRef> = structs
                .columns()
                .iter()
                .zip(fields)
                .map(|(column, field)| renamed(column, &field.data_type))
                .collect();
            let names = named(arrow, fields, &columns);
            Arc::new(StructArray::new(
                names.into(),
                columns,
                structs.nulls().cloned(),
            ))
        }
        (ArrowType::List(field), SparkType::Array { element_type, .. }) => {
            renamed_list(array.as_list::<i32>(), field, element_type)
        }
        (ArrowType::LargeList(field), SparkType::Array { element_type, .. }) => {
            renamed_list(array.as_list::<i64>(), field, element_type)
        }
        (
            ArrowType::Map(field, sorted),
            SparkType::Map {
                key_type,
                value_type,
                ..
            },
        ) => {
            let map = array.as_map();
            let entries = map.entries();
            let ArrowType::Struct(entry_fields) = entries.data_type() else {
                return Arc::clone(array);
            };
            let keys = renamed(entries.column(0), key_type);
            let values = renamed(entries.column(1), value_type);
            let entry_fields: ArrowFields = vec![
                retyped(&entry_fields[0], &keys),
                retyped(&entry_fields[1], &values),
            ]
            .into();
            let entries =
                StructArray::new(entry_fields, vec![keys, values], entries.nulls().cloned());
            let field = Arc::new(Field::new(
                field.name(),
                entries.data_type().clone(),
                field.is_nullable(),
            ));
            Arc::new(MapArray::new(
                field,
                map.offsets().clone(),
                entries,
                map.nulls().cloned(),
                *sorted,
            ))
        }
        _ => Arc::clone(array),
    }
}

fn renamed_list<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    field: &Field,
    element_type: &SparkType,
) -> ArrayRef {
    let values = renamed(list.values(), element_type);
    let field = Arc::new(retyped(field, &values));
    Arc::new(GenericListArray::new(
        field,
        list.offsets().clone(),
        values,
        list.nulls().cloned(),
    ))
}

/// Whether the value at `row` is `NULL`: also every value of an array of the null type, which
/// has no validity buffer.
fn null_at(array: &ArrayRef, row: usize) -> bool {
    array.is_null(row) || *array.data_type() == ArrowType::Null
}

fn unsupported(data_type: &ArrowType) -> Error {
    Error::decode(format!(
        "a result of the Arrow type {data_type} cannot be read"
    ))
}

fn out_of_range(what: &str) -> Error {
    Error::decode(format!("{what} is out of the range of its Rust type"))
}

/// The value at `row` of `array`.
fn value_at(array: &ArrayRef, row: usize) -> Result<Value> {
    if null_at(array, row) {
        return Ok(Value::Null);
    }
    Ok(match array.data_type() {
        ArrowType::Boolean => Value::Boolean(array.as_boolean().value(row)),
        ArrowType::Int8 => Value::Byte(array.as_primitive::<Int8Type>().value(row)),
        ArrowType::Int16 => Value::Short(array.as_primitive::<Int16Type>().value(row)),
        ArrowType::Int32 => Value::Int(array.as_primitive::<Int32Type>().value(row)),
        ArrowType::Int64 => Value::Long(array.as_primitive::<Int64Type>().value(row)),
        ArrowType::Float32 => Value::Float(array.as_primitive::<Float32Type>().value(row)),
        ArrowType::Float64 => Value::Double(array.as_primitive::<Float64Type>().value(row)),
        ArrowType::Decimal128(..) | ArrowType::Decimal256(..) => {
            Value::Decimal(decimal_at(array, row)?)
        }
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => {
            Value::String(str_at(array, row).to_string())
        }
        ArrowType::Binary
        | ArrowType::LargeBinary
        | ArrowType::BinaryView
        | ArrowType::FixedSizeBinary(_) => Value::Binary(bytes_at(array, row).to_vec()),
        ArrowType::Date32 | ArrowType::Date64 => Value::Date(date_at(array, row)?),
        ArrowType::Timestamp(_, zone) => {
            let instant = timestamp_at(array, row)?;
            match zone {
                Some(_) => Value::Timestamp(instant),
                None => Value::TimestampNtz(instant.naive_utc()),
            }
        }
        ArrowType::Time32(_) | ArrowType::Time64(_) => Value::Time(time_at(array, row)?),
        ArrowType::Duration(_) => Value::DayTimeInterval(duration_at(array, row)?),
        ArrowType::Interval(IntervalUnit::YearMonth) => {
            Value::YearMonthInterval(array.as_primitive::<IntervalYearMonthType>().value(row))
        }
        ArrowType::List(_) | ArrowType::LargeList(_) | ArrowType::FixedSizeList(..) => {
            let (elements, range) = elements_of(array, row);
            let values = range
                .enumerate()
                .map(|(i, at)| value_at(elements, at).map_err(|e| e.in_element(i)))
                .collect::<Result<_>>()?;
            Value::Array(values)
        }
        ArrowType::Map(..) => {
            let (keys, values, range) = entries_of(array, row);
            let entries = range
                .enumerate()
                .map(|(i, at)| {
                    let key = value_at(keys, at).map_err(|e| e.in_field("key").in_element(i))?;
                    let value =
                        value_at(values, at).map_err(|e| e.in_field("value").in_element(i))?;
                    Ok((key, value))
                })
                .collect::<Result<_>>()?;
            Value::Map(entries)
        }
        ArrowType::Struct(fields) => {
            let array = array.as_struct();
            if is_variant(array) {
                let part = |name: &str| -> Result<Vec<u8>> {
                    let column = array
                        .column_by_name(name)
                        .ok_or_else(|| Error::decode(format!("a variant without its {name}")))?;
                    Ok(bytes_at(column, row).to_vec())
                };
                return Ok(Value::Variant {
                    value: part("value")?,
                    metadata: part("metadata")?,
                });
            }
            let names: Arc<[String]> = fields.iter().map(|field| field.name().clone()).collect();
            let values = array
                .columns()
                .iter()
                .zip(names.iter())
                .map(|(column, name)| value_at(column, row).map_err(|e| e.in_field(name)))
                .collect::<Result<_>>()?;
            Value::Struct(Row::new(names, values))
        }
        other => return Err(unsupported(other)),
    })
}

/// Whether `array` is Spark's encoding of a `VARIANT`, which marks a field `variant`.
fn is_variant(array: &arrow::array::StructArray) -> bool {
    array.fields().iter().any(|field| {
        field
            .metadata()
            .get("variant")
            .is_some_and(|value| value == "true")
    })
}

fn decimal_at(array: &ArrayRef, row: usize) -> Result<Decimal> {
    let (unscaled, precision, scale) = match array.data_type() {
        ArrowType::Decimal128(precision, scale) => (
            array.as_primitive::<Decimal128Type>().value(row),
            *precision,
            *scale,
        ),
        ArrowType::Decimal256(precision, scale) => {
            let value = array.as_primitive::<Decimal256Type>().value(row);
            let unscaled = value.to_i128().ok_or_else(|| {
                Error::decode(format!(
                    "the decimal {} has more than 38 digits",
                    array.as_primitive::<Decimal256Type>().value_as_string(row)
                ))
            })?;
            (unscaled, (*precision).min(Decimal::MAX_PRECISION), *scale)
        }
        other => return Err(unsupported(other)),
    };
    Decimal::new(unscaled, precision, scale).map_err(|e| Error::decode(e.message()))
}

fn str_at(array: &ArrayRef, row: usize) -> &str {
    match array.data_type() {
        ArrowType::LargeUtf8 => array.as_string::<i64>().value(row),
        ArrowType::Utf8View => array.as_string_view().value(row),
        _ => array.as_string::<i32>().value(row),
    }
}

fn bytes_at(array: &ArrayRef, row: usize) -> &[u8] {
    match array.data_type() {
        ArrowType::LargeBinary => array.as_binary::<i64>().value(row),
        ArrowType::BinaryView => array.as_binary_view().value(row),
        ArrowType::FixedSizeBinary(_) => array.as_fixed_size_binary().value(row),
        _ => array.as_binary::<i32>().value(row),
    }
}

fn date_at(array: &ArrayRef, row: usize) -> Result<NaiveDate> {
    let days = match array.data_type() {
        ArrowType::Date64 => i32::try_from(
            array
                .as_primitive::<Date64Type>()
                .value(row)
                .div_euclid(86_400_000),
        )
        .ok(),
        _ => Some(array.as_primitive::<Date32Type>().value(row)),
    };
    days.and_then(NaiveDate::from_epoch_days)
        .ok_or_else(|| out_of_range("the date"))
}

fn timestamp_at(array: &ArrayRef, row: usize) -> Result<DateTime<Utc>> {
    let instant = match array.data_type() {
        ArrowType::Timestamp(TimeUnit::Second, _) => {
            DateTime::from_timestamp(array.as_primitive::<TimestampSecondType>().value(row), 0)
        }
        ArrowType::Timestamp(TimeUnit::Millisecond, _) => DateTime::from_timestamp_millis(
            array.as_primitive::<TimestampMillisecondType>().value(row),
        ),
        ArrowType::Timestamp(TimeUnit::Microsecond, _) => DateTime::from_timestamp_micros(
            array.as_primitive::<TimestampMicrosecondType>().value(row),
        ),
        ArrowType::Timestamp(TimeUnit::Nanosecond, _) => Some(DateTime::from_timestamp_nanos(
            array.as_primitive::<TimestampNanosecondType>().value(row),
        )),
        other => return Err(unsupported(other)),
    };
    instant.ok_or_else(|| out_of_range("the timestamp"))
}

fn time_at(array: &ArrayRef, row: usize) -> Result<NaiveTime> {
    let nanos: i64 = match array.data_type() {
        ArrowType::Time32(TimeUnit::Second) => {
            i64::from(array.as_primitive::<Time32SecondType>().value(row)) * 1_000_000_000
        }
        ArrowType::Time32(_) => {
            i64::from(array.as_primitive::<Time32MillisecondType>().value(row)) * 1_000_000
        }
        ArrowType::Time64(TimeUnit::Microsecond) => {
            array.as_primitive::<Time64MicrosecondType>().value(row) * 1_000
        }
        _ => array.as_primitive::<Time64NanosecondType>().value(row),
    };
    let seconds = u32::try_from(nanos.div_euclid(1_000_000_000)).ok();
    seconds
        .and_then(|s| {
            NaiveTime::from_num_seconds_from_midnight_opt(s, nanos.rem_euclid(1_000_000_000) as u32)
        })
        .ok_or_else(|| out_of_range("the time"))
}

fn duration_at(array: &ArrayRef, row: usize) -> Result<TimeDelta> {
    let delta = match array.data_type() {
        ArrowType::Duration(TimeUnit::Second) => {
            TimeDelta::try_seconds(array.as_primitive::<DurationSecondType>().value(row))
        }
        ArrowType::Duration(TimeUnit::Millisecond) => {
            TimeDelta::try_milliseconds(array.as_primitive::<DurationMillisecondType>().value(row))
        }
        ArrowType::Duration(TimeUnit::Microsecond) => Some(TimeDelta::microseconds(
            array.as_primitive::<DurationMicrosecondType>().value(row),
        )),
        _ => Some(TimeDelta::nanoseconds(
            array.as_primitive::<DurationNanosecondType>().value(row),
        )),
    };
    delta.ok_or_else(|| out_of_range("the interval"))
}

/// The elements of the list at `row` of `array`: the array that holds them, and their indexes
/// there.
fn elements_of(array: &ArrayRef, row: usize) -> (&ArrayRef, Range<usize>) {
    match array.data_type() {
        ArrowType::LargeList(_) => {
            let list = array.as_list::<i64>();
            let offsets = list.value_offsets();
            (
                list.values(),
                offsets[row] as usize..offsets[row + 1] as usize,
            )
        }
        ArrowType::FixedSizeList(_, size) => {
            let list = array.as_fixed_size_list();
            let start = list.value_offset(row) as usize;
            (list.values(), start..start + *size as usize)
        }
        _ => {
            let list = array.as_list::<i32>();
            let offsets = list.value_offsets();
            (
                list.values(),
                offsets[row] as usize..offsets[row + 1] as usize,
            )
        }
    }
}

/// The entries of the map at `row` of `array`: the arrays that hold their keys and values, and
/// their indexes there.
fn entries_of(array: &ArrayRef, row: usize) -> (&ArrayRef, &ArrayRef, Range<usize>) {
    let map = array.as_map();
    let offsets = map.value_offsets();
    (
        map.keys(),
        map.values(),
        offsets[row] as usize..offsets[row + 1] as usize,
    )
}

/// A row, as a struct, tuple, sequence or map of its columns.
struct RowDeserializer<'de> {
    batch: &'de RecordBatch,
    row: usize,
}

impl<'de> RowDeserializer<'de> {
    fn only_column(&self) -> Result<ValueDeserializer<'de>> {
        match self.batch.columns() {
            [column] => Ok(ValueDeserializer {
                array: column,
                index: self.row,
            }),
            columns => Err(Error::decode(format!(
                "a row of {} columns is not a single value",
                columns.len()
            ))),
        }
    }

    fn fields(&self) -> Fields<'de> {
        Fields {
            names: self.batch.schema_ref().fields(),
            columns: self.batch.columns(),
            index: self.row,
            next: 0,
        }
    }
}

macro_rules! forward_to_the_only_column {
    ($($method:ident)*) => {
        $(fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
            self.only_column()?.$method(visitor)
        })*
    };
}

impl<'de> Deserializer<'de> for RowDeserializer<'de> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_map(self.fields())
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_map(self.fields())
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_map(self.fields())
    }

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_seq(self.fields())
    }

    fn deserialize_tuple<V: Visitor<'de>>(self, _len: usize, visitor: V) -> Result<V::Value> {
        visitor.visit_seq(self.fields())
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_seq(self.fields())
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value> {
        if name == DECIMAL_PARTS {
            return self
                .only_column()?
                .deserialize_newtype_struct(name, visitor);
        }
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        match self.batch.num_columns() {
            1 => self.only_column()?.deserialize_option(visitor),
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        self.only_column()?
            .deserialize_enum(name, variants, visitor)
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_unit()
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_unit()
    }

    forward_to_the_only_column! {
        deserialize_bool deserialize_i8 deserialize_i16 deserialize_i32 deserialize_i64
        deserialize_i128 deserialize_u8 deserialize_u16 deserialize_u32 deserialize_u64
        deserialize_u128 deserialize_f32 deserialize_f64 deserialize_char deserialize_str
        deserialize_string deserialize_bytes deserialize_byte_buf deserialize_unit
        deserialize_identifier
    }
}

/// The columns of a row, or the fields of a struct: by name as a map, by position as a sequence.
struct Fields<'de> {
    names: &'de arrow::datatypes::Fields,
    columns: &'de [ArrayRef],
    index: usize,
    next: usize,
}

impl<'de> Fields<'de> {
    /// The next field's value and name.
    fn next_field(&mut self) -> Option<(ValueDeserializer<'de>, &'de str)> {
        let column = self.columns.get(self.next)?;
        let name = self.names[self.next].name().as_str();
        self.next += 1;
        Some((
            ValueDeserializer {
                array: column,
                index: self.index,
            },
            name,
        ))
    }
}

impl<'de> MapAccess<'de> for Fields<'de> {
    type Error = Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        match self.names.get(self.next) {
            Some(field) => seed
                .deserialize(de::value::BorrowedStrDeserializer::new(field.name()))
                .map(Some),
            None => Ok(None),
        }
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value> {
        let (value, name) = self
            .next_field()
            .ok_or_else(|| Error::decode("a value without its field"))?;
        seed.deserialize(value).map_err(|e| e.in_field(name))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.columns.len() - self.next)
    }
}

impl<'de> SeqAccess<'de> for Fields<'de> {
    type Error = Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>> {
        let Some((value, name)) = self.next_field() else {
            return Ok(None);
        };
        seed.deserialize(value)
            .map(Some)
            .map_err(|e| e.in_field(name))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.columns.len() - self.next)
    }
}

/// The elements of an array.
struct Elements<'de> {
    values: &'de ArrayRef,
    range: Range<usize>,
    next: usize,
}

impl<'de> SeqAccess<'de> for Elements<'de> {
    type Error = Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>> {
        let Some(index) = self.range.next() else {
            return Ok(None);
        };
        let element = self.next;
        self.next += 1;
        seed.deserialize(ValueDeserializer {
            array: self.values,
            index,
        })
        .map(Some)
        .map_err(|e| e.in_element(element))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.range.len())
    }
}

/// The entries of a map: as a map, or as a sequence of pairs, for `Vec<(K, V)>`.
struct Entries<'de> {
    keys: &'de ArrayRef,
    values: &'de ArrayRef,
    range: Range<usize>,
    /// The index of the entry whose key was read last.
    current: usize,
    next: usize,
}

impl<'de> Entries<'de> {
    fn of(array: &'de ArrayRef, row: usize) -> Self {
        let (keys, values, range) = entries_of(array, row);
        Entries {
            keys,
            values,
            current: range.start,
            range,
            next: 0,
        }
    }

    fn key(&self, index: usize) -> ValueDeserializer<'de> {
        ValueDeserializer {
            array: self.keys,
            index,
        }
    }

    fn value(&self, index: usize) -> ValueDeserializer<'de> {
        ValueDeserializer {
            array: self.values,
            index,
        }
    }
}

impl<'de> MapAccess<'de> for Entries<'de> {
    type Error = Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        let Some(index) = self.range.next() else {
            return Ok(None);
        };
        self.current = index;
        let entry = self.next;
        seed.deserialize(self.key(index))
            .map(Some)
            .map_err(|e| e.in_field("key").in_element(entry))
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value> {
        let entry = self.next;
        self.next += 1;
        seed.deserialize(self.value(self.current))
            .map_err(|e| e.in_field("value").in_element(entry))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.range.len())
    }
}

impl<'de> SeqAccess<'de> for Entries<'de> {
    type Error = Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>> {
        let Some(index) = self.range.next() else {
            return Ok(None);
        };
        let entry = self.next;
        self.next += 1;
        let pair = Pair {
            key: Some(self.key(index)),
            value: Some(self.value(index)),
        };
        seed.deserialize(pair)
            .map(Some)
            .map_err(|e| e.in_element(entry))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.range.len())
    }
}

/// An entry of a map as a pair of its key and value.
struct Pair<'de> {
    key: Option<ValueDeserializer<'de>>,
    value: Option<ValueDeserializer<'de>>,
}

impl<'de> Deserializer<'de> for Pair<'de> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_seq(self)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf
        option unit unit_struct newtype_struct seq tuple tuple_struct map struct enum identifier
        ignored_any
    }
}

impl<'de> SeqAccess<'de> for Pair<'de> {
    type Error = Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>> {
        if let Some(key) = self.key.take() {
            return seed
                .deserialize(key)
                .map(Some)
                .map_err(|e| e.in_field("key"));
        }
        match self.value.take() {
            Some(value) => seed
                .deserialize(value)
                .map(Some)
                .map_err(|e| e.in_field("value")),
            None => Ok(None),
        }
    }
}

/// The exact parts of a decimal, for [`Decimal`]'s `Deserialize`.
struct DecimalParts {
    decimal: Decimal,
    next: usize,
}

impl<'de> SeqAccess<'de> for DecimalParts {
    type Error = Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>> {
        self.next += 1;
        match self.next {
            1 => seed
                .deserialize(self.decimal.unscaled().into_deserializer())
                .map(Some),
            2 => seed
                .deserialize(self.decimal.precision().into_deserializer())
                .map(Some),
            3 => seed
                .deserialize(self.decimal.scale().into_deserializer())
                .map(Some),
            _ => Ok(None),
        }
    }
}

#[derive(Clone, Copy)]
enum Number {
    Integer,
    Float,
}

/// The value at `index` of an array that the batch holds, whose strings and bytes it lends.
struct ValueDeserializer<'de> {
    array: &'de ArrayRef,
    index: usize,
}

impl<'de> ValueDeserializer<'de> {
    fn is_null(&self) -> bool {
        null_at(self.array, self.index)
    }

    /// An error for a `NULL`, for the hints that only an `Option` answers with `None`.
    fn non_null(self) -> Result<Self> {
        if self.is_null() {
            Err(Error::decode("a NULL can only be read into an Option"))
        } else {
            Ok(self)
        }
    }

    fn fields(&self) -> Fields<'de> {
        let array = self.array.as_struct();
        Fields {
            names: array.fields(),
            columns: array.columns(),
            index: self.index,
            next: 0,
        }
    }

    fn number<V: Visitor<'de>>(self, visitor: V, number: Number) -> Result<V::Value> {
        let this = self.non_null()?;
        let (array, row) = (this.array, this.index);
        match (array.data_type(), number) {
            (ArrowType::Decimal128(..) | ArrowType::Decimal256(..), Number::Float) => {
                visitor.visit_f64(decimal_at(array, row)?.to_f64())
            }
            (ArrowType::Decimal128(..) | ArrowType::Decimal256(..), Number::Integer) => {
                let decimal = decimal_at(array, row)?;
                if decimal.scale() > 0 {
                    return Err(Error::decode(format!(
                        "the decimal {decimal} of scale {} is not an integer",
                        decimal.scale()
                    )));
                }
                let value = 10i128
                    .checked_pow(u32::from(decimal.scale().unsigned_abs()))
                    .and_then(|factor| decimal.unscaled().checked_mul(factor))
                    .ok_or_else(|| out_of_range("the decimal"))?;
                match i64::try_from(value) {
                    Ok(value) => visitor.visit_i64(value),
                    Err(_) => visitor.visit_i128(value),
                }
            }
            (ArrowType::Date32 | ArrowType::Date64, Number::Integer) => {
                visitor.visit_i32(date_at(array, row)?.to_epoch_days())
            }
            (ArrowType::Timestamp(..), Number::Integer) => {
                visitor.visit_i64(timestamp_at(array, row)?.timestamp_micros())
            }
            (ArrowType::Time32(_) | ArrowType::Time64(_), Number::Integer) => {
                let time = time_at(array, row)?;
                let nanos = (time - NaiveTime::MIN)
                    .num_nanoseconds()
                    .unwrap_or(i64::MAX);
                visitor.visit_i64(nanos)
            }
            _ => this.deserialize_any(visitor),
        }
    }
}

macro_rules! numbers {
    ($($method:ident => $number:ident)*) => {
        $(fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
            self.number(visitor, Number::$number)
        })*
    };
}

macro_rules! non_null_any {
    ($($method:ident)*) => {
        $(fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
            self.non_null()?.deserialize_any(visitor)
        })*
    };
}

impl<'de> Deserializer<'de> for ValueDeserializer<'de> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        if self.is_null() {
            return visitor.visit_none();
        }
        let (array, row) = (self.array, self.index);
        match array.data_type() {
            ArrowType::Boolean => visitor.visit_bool(array.as_boolean().value(row)),
            ArrowType::Int8 => visitor.visit_i8(array.as_primitive::<Int8Type>().value(row)),
            ArrowType::Int16 => visitor.visit_i16(array.as_primitive::<Int16Type>().value(row)),
            ArrowType::Int32 => visitor.visit_i32(array.as_primitive::<Int32Type>().value(row)),
            ArrowType::Int64 => visitor.visit_i64(array.as_primitive::<Int64Type>().value(row)),
            ArrowType::UInt8 => visitor.visit_u8(array.as_primitive::<UInt8Type>().value(row)),
            ArrowType::UInt16 => visitor.visit_u16(array.as_primitive::<UInt16Type>().value(row)),
            ArrowType::UInt32 => visitor.visit_u32(array.as_primitive::<UInt32Type>().value(row)),
            ArrowType::UInt64 => visitor.visit_u64(array.as_primitive::<UInt64Type>().value(row)),
            ArrowType::Float32 => visitor.visit_f32(array.as_primitive::<Float32Type>().value(row)),
            ArrowType::Float64 => visitor.visit_f64(array.as_primitive::<Float64Type>().value(row)),
            ArrowType::Decimal128(..) | ArrowType::Decimal256(..) => {
                visitor.visit_string(decimal_at(array, row)?.to_string())
            }
            ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => {
                visitor.visit_borrowed_str(str_at(array, row))
            }
            ArrowType::Binary
            | ArrowType::LargeBinary
            | ArrowType::BinaryView
            | ArrowType::FixedSizeBinary(_) => visitor.visit_borrowed_bytes(bytes_at(array, row)),
            ArrowType::Date32 | ArrowType::Date64 => {
                visitor.visit_string(date_at(array, row)?.to_string())
            }
            ArrowType::Timestamp(_, Some(_)) => visitor.visit_string(
                timestamp_at(array, row)?.to_rfc3339_opts(SecondsFormat::AutoSi, true),
            ),
            ArrowType::Timestamp(_, None) => {
                visitor.visit_string(ntz_text(timestamp_at(array, row)?.naive_utc()))
            }
            ArrowType::Time32(_) | ArrowType::Time64(_) => {
                visitor.visit_string(time_at(array, row)?.to_string())
            }
            ArrowType::Duration(_) => {
                let delta = duration_at(array, row)?;
                visitor.visit_i64(
                    delta
                        .num_microseconds()
                        .ok_or_else(|| out_of_range("the interval"))?,
                )
            }
            ArrowType::Interval(IntervalUnit::YearMonth) => {
                visitor.visit_i32(array.as_primitive::<IntervalYearMonthType>().value(row))
            }
            ArrowType::List(_) | ArrowType::LargeList(_) | ArrowType::FixedSizeList(..) => {
                let (values, range) = elements_of(array, row);
                visitor.visit_seq(Elements {
                    values,
                    range,
                    next: 0,
                })
            }
            ArrowType::Map(..) => visitor.visit_map(Entries::of(array, row)),
            ArrowType::Struct(_) => visitor.visit_map(self.fields()),
            other => Err(unsupported(other)),
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        if self.is_null() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        let this = self.non_null()?;
        let (array, row) = (this.array, this.index);
        match array.data_type() {
            ArrowType::Map(..) => visitor.visit_seq(Entries::of(array, row)),
            ArrowType::Binary
            | ArrowType::LargeBinary
            | ArrowType::BinaryView
            | ArrowType::FixedSizeBinary(_) => {
                visitor.visit_seq(de::value::SeqDeserializer::<_, Error>::new(
                    bytes_at(array, row).iter().copied(),
                ))
            }
            ArrowType::Struct(_) => visitor.visit_seq(this.fields()),
            _ => this.deserialize_any(visitor),
        }
    }

    fn deserialize_tuple<V: Visitor<'de>>(self, _len: usize, visitor: V) -> Result<V::Value> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        self.non_null()?.deserialize_any(visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value> {
        let decimal = matches!(
            self.array.data_type(),
            ArrowType::Decimal128(..) | ArrowType::Decimal256(..)
        );
        if name == DECIMAL_PARTS && decimal {
            let this = self.non_null()?;
            return visitor.visit_seq(DecimalParts {
                decimal: decimal_at(this.array, this.index)?,
                next: 0,
            });
        }
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value> {
        let this = self.non_null()?;
        match this.array.data_type() {
            ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => {
                let variant: &'de str = str_at(this.array, this.index);
                visitor.visit_enum(variant.into_deserializer())
            }
            _ => Err(Error::decode(
                "only a STRING can be read into an enum, as the name of its variant",
            )),
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_unit()
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_unit()
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value> {
        visitor.visit_unit()
    }

    non_null_any! {
        deserialize_map deserialize_str deserialize_string deserialize_bytes deserialize_byte_buf
        deserialize_identifier deserialize_bool deserialize_char
    }

    numbers! {
        deserialize_i8 => Integer deserialize_i16 => Integer deserialize_i32 => Integer
        deserialize_i64 => Integer deserialize_i128 => Integer deserialize_u8 => Integer
        deserialize_u16 => Integer deserialize_u32 => Integer deserialize_u64 => Integer
        deserialize_u128 => Integer deserialize_f32 => Float deserialize_f64 => Float
    }
}

/// A `TIMESTAMP_NTZ` as `chrono::NaiveDateTime` parses it, with a `T` between date and time.
fn ntz_text(value: NaiveDateTime) -> String {
    value.format("%Y-%m-%dT%H:%M:%S%.f").to_string()
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int32Array, ListArray, MapBuilder, StringBuilder};
    use spark_connect_core::types::StructField;

    use super::*;

    fn spark_field(name: &str, data_type: SparkType) -> StructField {
        StructField {
            name: name.to_string(),
            data_type,
            nullable: true,
            metadata: Default::default(),
        }
    }

    #[test]
    fn a_batch_takes_the_names_of_the_spark_schema_at_every_depth() {
        let inner = StructArray::from(vec![(
            Arc::new(Field::new("a", ArrowType::Int32, true)),
            Arc::new(Int32Array::from(vec![7])) as ArrayRef,
        )]);
        let list = ListArray::new(
            Arc::new(Field::new("item", inner.data_type().clone(), true)),
            arrow::buffer::OffsetBuffer::from_lengths([1]),
            Arc::new(inner.clone()),
            None,
        );
        let mut map = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
        map.keys().append_value("k");
        map.values().append_value("v");
        map.append(true).unwrap();
        let columns: Vec<ArrayRef> = vec![Arc::new(inner), Arc::new(list), Arc::new(map.finish())];
        let arrow_schema = Schema::new(
            columns
                .iter()
                .enumerate()
                .map(|(i, column)| Field::new(format!("c{i}"), column.data_type().clone(), true))
                .collect::<Vec<_>>(),
        );
        let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns).unwrap();
        let struct_of = |name: &str| SparkType::Struct {
            fields: vec![spark_field(name, SparkType::Integer)],
        };
        let string = || SparkType::String {
            collation: "UTF8_BINARY".to_string(),
        };
        let schema = SparkType::Struct {
            fields: vec![
                spark_field("point", struct_of("x")),
                spark_field(
                    "points",
                    SparkType::Array {
                        element_type: Box::new(struct_of("y")),
                        contains_null: true,
                    },
                ),
                spark_field(
                    "labels",
                    SparkType::Map {
                        key_type: Box::new(string()),
                        value_type: Box::new(string()),
                        value_contains_null: true,
                    },
                ),
            ],
        };

        #[derive(Deserialize, Debug, PartialEq)]
        struct X {
            x: i32,
        }
        #[derive(Deserialize, Debug, PartialEq)]
        struct Y {
            y: i32,
        }
        #[derive(Deserialize, Debug, PartialEq)]
        struct Named {
            point: X,
            points: Vec<Y>,
            labels: Vec<(String, String)>,
        }
        let renamed = with_names_of(batch.clone(), Some(&schema)).unwrap();
        let rows: Vec<Named> = from_record_batch(&renamed).unwrap();
        assert_eq!(
            rows,
            [Named {
                point: X { x: 7 },
                points: vec![Y { y: 7 }],
                labels: vec![("k".to_string(), "v".to_string())],
            }]
        );
        let unnamed = with_names_of(batch, None).unwrap();
        assert!(from_record_batch::<Named>(&unnamed).is_err());
    }
}
