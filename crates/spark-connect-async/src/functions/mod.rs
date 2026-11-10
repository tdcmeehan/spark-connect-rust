// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

//! Spark's SQL functions, as builders of columns.
//!
//! Each parameter takes what its SQL argument is:
//! - a column (`impl Into<Column>`), where a `&str` is the column of that name;
//! - a Rust value (`&str`, `i32`, `f64`, `bool`, ...) where the argument is always a literal;
//! - a value or a column ([`IntoArgument`]), where a `&str` is a string;
//! - an `Option` for a trailing optional argument, where `None` leaves it out.
//!
//! ```
//! use spark_connect_async::functions::{concat_ws, lit, round, split, upper, when};
//!
//! let label = concat_ws("-", ["first", "last"]);
//! let words = split("text", " ");
//! let price = round("price", Some(2));
//! let size = when(lit(true), upper("name"));
//! ```
//!
//! The documentation of a function is the description of its SQL function in Apache Spark.

use std::iter;

use spark_connect_core::column::Column as CoreColumn;
use spark_connect_core::{function_forms, functions as sql};

pub use crate::column::{col, lit, when, IntoArgument};
use crate::column::{to_core, Column, IntoLiteral};
use crate::dataframe::DataFrame;

mod generated;
pub use generated::*;

fn columns(cols: impl IntoIterator<Item = impl Into<Column>>) -> impl Iterator<Item = CoreColumn> {
    cols.into_iter().map(to_core)
}

fn literal(value: impl IntoLiteral) -> CoreColumn {
    to_core(lit(value))
}

fn variables<const N: usize>(variables: Vec<CoreColumn>) -> [Column; N] {
    let variables: Vec<Column> = variables.into_iter().map(Column::from).collect();
    match variables.try_into() {
        Ok(variables) => variables,
        Err(variables) => unreachable!("{N} lambda variables expected, got {}", variables.len()),
    }
}

fn lambda1(f: impl FnOnce(Column) -> Column) -> impl FnOnce(Vec<CoreColumn>) -> CoreColumn {
    move |vars| {
        let [x] = variables(vars);
        to_core(f(x))
    }
}

fn lambda2(f: impl FnOnce(Column, Column) -> Column) -> impl FnOnce(Vec<CoreColumn>) -> CoreColumn {
    move |vars| {
        let [x, y] = variables(vars);
        to_core(f(x, y))
    }
}

fn lambda3(
    f: impl FnOnce(Column, Column, Column) -> Column,
) -> impl FnOnce(Vec<CoreColumn>) -> CoreColumn {
    move |vars| {
        let [x, y, z] = variables(vars);
        to_core(f(x, y, z))
    }
}

/// The DataFrame `df`, marked to be broadcast to every executor in a join.
pub fn broadcast(df: &DataFrame) -> DataFrame {
    df.hint("broadcast", Vec::<i32>::new())
}

/// The strings of `cols` joined by `sep`, skipping NULLs: the SQL function `concat_ws`.
pub fn concat_ws(sep: &str, cols: impl IntoIterator<Item = impl Into<Column>>) -> Column {
    Column::from(sql::concat_ws(
        iter::once(literal(sep)).chain(columns(cols)),
    ))
}

/// The printf-style `format` filled with the values of `cols`: the SQL function
/// `format_string`.
pub fn format_string(format: &str, cols: impl IntoIterator<Item = impl Into<Column>>) -> Column {
    Column::from(sql::format_string(
        iter::once(literal(format)).chain(columns(cols)),
    ))
}

/// The printf-style format of the column `format` filled with the values of `cols`: the SQL
/// function `printf`.
pub fn printf(
    format: impl Into<Column>,
    cols: impl IntoIterator<Item = impl Into<Column>>,
) -> Column {
    Column::from(function_forms::printf(to_core(format), columns(cols)))
}

/// One column per name of `fields`, with the value of that field of the JSON text `col`: the
/// SQL function `json_tuple`.
pub fn json_tuple(
    col: impl Into<Column>,
    fields: impl IntoIterator<Item = impl AsRef<str>>,
) -> Column {
    let fields = fields.into_iter().map(|field| literal(field.as_ref()));
    Column::from(sql::json_tuple(iter::once(to_core(col)).chain(fields)))
}

/// A struct with a field of each name and value of `fields`: the SQL function `named_struct`.
pub fn named_struct<N: AsRef<str>, C: Into<Column>>(
    fields: impl IntoIterator<Item = (N, C)>,
) -> Column {
    let fields = fields
        .into_iter()
        .flat_map(|(name, value)| [literal(name.as_ref()), to_core(value)]);
    Column::from(sql::named_struct(fields))
}

/// The values of `cols` split into `rows` rows: the SQL function `stack`.
pub fn stack(rows: i32, cols: impl IntoIterator<Item = impl Into<Column>>) -> Column {
    Column::from(sql::stack(iter::once(literal(rows)).chain(columns(cols))))
}

fn method_call(
    class: &str,
    method: &str,
    args: impl IntoIterator<Item = impl Into<Column>>,
) -> impl Iterator<Item = CoreColumn> {
    [literal(class), literal(method)]
        .into_iter()
        .chain(columns(args))
}

/// The result of the Java method `method` of the class `class`, called with `args`: the SQL
/// function `java_method`.
pub fn java_method(
    class: &str,
    method: &str,
    args: impl IntoIterator<Item = impl Into<Column>>,
) -> Column {
    Column::from(sql::java_method(method_call(class, method, args)))
}

/// The result of the Java method `method` of the class `class`, called with `args`: the SQL
/// function `reflect`.
pub fn reflect(
    class: &str,
    method: &str,
    args: impl IntoIterator<Item = impl Into<Column>>,
) -> Column {
    Column::from(sql::reflect(method_call(class, method, args)))
}

/// The result of the Java method `method` of the class `class`, called with `args`, or NULL
/// when the call fails: the SQL function `try_reflect`.
pub fn try_reflect(
    class: &str,
    method: &str,
    args: impl IntoIterator<Item = impl Into<Column>>,
) -> Column {
    Column::from(sql::try_reflect(method_call(class, method, args)))
}

/// The value of `inputs` at the 1-based position `index`: the SQL function `elt`.
pub fn elt(
    index: impl Into<Column>,
    inputs: impl IntoIterator<Item = impl Into<Column>>,
) -> Column {
    Column::from(sql::elt(iter::once(to_core(index)).chain(columns(inputs))))
}

/// A map of the keys and values of `entries`: the SQL function `map`.
pub fn create_map<K: Into<Column>, V: Into<Column>>(
    entries: impl IntoIterator<Item = (K, V)>,
) -> Column {
    let entries = entries
        .into_iter()
        .flat_map(|(key, value)| [to_core(key), to_core(value)]);
    Column::from(sql::create_map(entries))
}

/// The 1-based position of the `occurrence`-th match of `substr` in `str`, searching from the
/// position `start`, or 0 when there is none: the SQL function `instr`. `start` and
/// `occurrence` are optional: `None` leaves an argument out of the call, and `start` is 1 when
/// only `occurrence` is given.
pub fn instr(
    str: impl Into<Column>,
    substr: impl IntoArgument,
    start: Option<i32>,
    occurrence: Option<i32>,
) -> Column {
    Column::from(function_forms::instr(
        to_core(str),
        to_core(substr.into_argument()),
        start.map(literal),
        occurrence.map(literal),
    ))
}

/// The greatest value of `cols`, skipping NULLs: the SQL function `greatest`.
pub fn greatest(cols: impl IntoIterator<Item = impl Into<Column>>) -> Column {
    Column::from(function_forms::greatest(columns(cols)))
}

/// The least value of `cols`, skipping NULLs: the SQL function `least`.
pub fn least(cols: impl IntoIterator<Item = impl Into<Column>>) -> Column {
    Column::from(function_forms::least(columns(cols)))
}

/// The number of distinct combinations of the values of `cols` that are not NULL: the SQL
/// aggregate `count(DISTINCT ...)`.
pub fn count_distinct(cols: impl IntoIterator<Item = impl Into<Column>>) -> Column {
    Column::from(function_forms::count_distinct(columns(cols)))
}

/// The values of `col` that are not NULL, concatenated with `delimiter` between them, or with
/// nothing between them when it is `None`: the SQL aggregate `listagg`.
pub fn listagg(col: impl Into<Column>, delimiter: Option<&str>) -> Column {
    Column::from(function_forms::listagg(
        to_core(col),
        delimiter.map(literal),
    ))
}

/// The values of `col` that are not NULL, concatenated with `delimiter` between them, or with
/// nothing between them when it is `None`: the SQL aggregate `string_agg`.
pub fn string_agg(col: impl Into<Column>, delimiter: Option<&str>) -> Column {
    Column::from(function_forms::string_agg(
        to_core(col),
        delimiter.map(literal),
    ))
}

/// The array `col` sorted in ascending order, or in descending order when `asc` is
/// `Some(false)`: the SQL function `sort_array`.
pub fn sort_array(col: impl Into<Column>, asc: Option<bool>) -> Column {
    Column::from(function_forms::sort_array(to_core(col), asc.map(literal)))
}

/// The map of the text `text`, split into entries by `pair_delim` and into keys and values
/// by `key_value_delim`, `,` and `:` when they are `None`: the SQL function `str_to_map`.
pub fn str_to_map(
    text: impl Into<Column>,
    pair_delim: Option<Column>,
    key_value_delim: Option<Column>,
) -> Column {
    Column::from(function_forms::str_to_map(
        to_core(text),
        pair_delim.map(to_core),
        key_value_delim.map(to_core),
    ))
}

/// The sentences of `string`, each an array of words, split by the rules of the locale of
/// `language` and `country`, or of the default locale when they are `None`: the SQL function
/// `sentences`.
pub fn sentences(
    string: impl Into<Column>,
    language: Option<Column>,
    country: Option<Column>,
) -> Column {
    Column::from(function_forms::sentences(
        to_core(string),
        language.map(to_core),
        country.map(to_core),
    ))
}

/// A random string of `length` characters, from `seed`, or from a fixed seed when it is
/// `None`: the SQL function `randstr`.
pub fn randstr(length: impl IntoArgument, seed: Option<i64>) -> Column {
    Column::from(function_forms::randstr(
        to_core(length.into_argument()),
        seed.map(literal),
    ))
}

/// The time window of `window_duration` that holds the timestamp `time_column`: windows
/// start every `slide_duration`, or every `window_duration` when it is `None`, shifted by
/// `start_time`. Durations are interval strings, such as `"10 minutes"`. The SQL function
/// `window`.
pub fn window(
    time_column: impl Into<Column>,
    window_duration: &str,
    slide_duration: Option<&str>,
    start_time: Option<&str>,
) -> Column {
    Column::from(function_forms::window(
        to_core(time_column),
        window_duration,
        slide_duration,
        start_time,
    ))
}

/// The value of `col` `offset` rows (1 when it is `None`) before the current row of the
/// window, or `default` (NULL when it is `None`) when there is no such row: the SQL function
/// `lag`.
pub fn lag(col: impl Into<Column>, offset: Option<i32>, default: Option<Column>) -> Column {
    Column::from(function_forms::lag(
        to_core(col),
        offset.map(literal),
        default.map(to_core),
    ))
}

/// The value of `col` `offset` rows (1 when it is `None`) after the current row of the
/// window, or `default` (NULL when it is `None`) when there is no such row: the SQL function
/// `lead`.
pub fn lead(col: impl Into<Column>, offset: Option<i32>, default: Option<Column>) -> Column {
    Column::from(function_forms::lead(
        to_core(col),
        offset.map(literal),
        default.map(to_core),
    ))
}
