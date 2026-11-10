//! Forms of SQL function calls with the arguments that the builders of [`crate::functions`]
//! leave out, fix or take only in part. Those builders call these.

use std::iter;

use crate::column::Column;
use crate::expression::Expression;
use crate::functions::{func, func_distinct, lit_bool, lit_int, lit_long, lit_str};

fn expressions(columns: impl IntoIterator<Item = Column>) -> Vec<Expression> {
    columns
        .into_iter()
        .map(|column| column.expression().clone())
        .collect()
}

fn or_literal(column: Option<Column>, literal: Expression) -> Expression {
    column.map_or(literal, |column| column.expression().clone())
}

/// `window(time, duration[, slide[, start]])`: a slide is the duration when only a start is
/// given.
pub fn window(
    col: Column,
    window_duration: &str,
    slide_duration: Option<&str>,
    start_time: Option<&str>,
) -> Column {
    let mut args = vec![col.expression().clone(), lit_str(window_duration)];
    match (slide_duration, start_time) {
        (None, None) => {}
        (Some(slide), None) => args.push(lit_str(slide)),
        (slide, Some(start)) => {
            args.push(lit_str(slide.unwrap_or(window_duration)));
            args.push(lit_str(start));
        }
    }
    func("window", args)
}

/// `instr(str, substr[, start[, occurrence]])`: a start is 1 when only an occurrence is given.
pub fn instr(
    str: Column,
    substr: Column,
    start: Option<Column>,
    occurrence: Option<Column>,
) -> Column {
    let mut args = vec![str.expression().clone(), substr.expression().clone()];
    match (start, occurrence) {
        (None, None) => {}
        (Some(start), None) => args.push(start.expression().clone()),
        (start, Some(occurrence)) => {
            args.push(or_literal(start, lit_int(1)));
            args.push(occurrence.expression().clone());
        }
    }
    func("instr", args)
}

/// `count(DISTINCT cols...)`.
pub fn count_distinct(cols: impl IntoIterator<Item = Column>) -> Column {
    func_distinct("count", expressions(cols))
}

/// `greatest(cols...)`.
pub fn greatest(cols: impl IntoIterator<Item = Column>) -> Column {
    func("greatest", expressions(cols))
}

/// `least(cols...)`.
pub fn least(cols: impl IntoIterator<Item = Column>) -> Column {
    func("least", expressions(cols))
}

/// `printf(format, args...)`.
pub fn printf(format: Column, args: impl IntoIterator<Item = Column>) -> Column {
    func("printf", expressions(iter::once(format).chain(args)))
}

/// `lag(col, offset, default)`, with an offset of 1 when none is given.
pub fn lag(col: Column, offset: Option<Column>, default: Option<Column>) -> Column {
    offset_function("lag", col, offset, default)
}

/// `lead(col, offset, default)`, with an offset of 1 when none is given.
pub fn lead(col: Column, offset: Option<Column>, default: Option<Column>) -> Column {
    offset_function("lead", col, offset, default)
}

fn offset_function(
    name: &str,
    col: Column,
    offset: Option<Column>,
    default: Option<Column>,
) -> Column {
    let mut args = vec![col.expression().clone(), or_literal(offset, lit_int(1))];
    args.extend(default.map(|default| default.expression().clone()));
    func(name, args)
}

/// `listagg(col[, delimiter])`.
pub fn listagg(col: Column, delimiter: Option<Column>) -> Column {
    func("listagg", expressions(iter::once(col).chain(delimiter)))
}

/// `string_agg(col[, delimiter])`.
pub fn string_agg(col: Column, delimiter: Option<Column>) -> Column {
    func("string_agg", expressions(iter::once(col).chain(delimiter)))
}

/// `sort_array(col, asc)`, ascending when `asc` is not given.
pub fn sort_array(col: Column, asc: Option<Column>) -> Column {
    func(
        "sort_array",
        vec![col.expression().clone(), or_literal(asc, lit_bool(true))],
    )
}

/// `str_to_map(text, pair_delim, key_value_delim)`, with `,` and `:` when not given.
pub fn str_to_map(
    text: Column,
    pair_delim: Option<Column>,
    key_value_delim: Option<Column>,
) -> Column {
    func(
        "str_to_map",
        vec![
            text.expression().clone(),
            or_literal(pair_delim, lit_str(",")),
            or_literal(key_value_delim, lit_str(":")),
        ],
    )
}

/// `sentences(string, language, country)`, with empty strings when not given.
pub fn sentences(string: Column, language: Option<Column>, country: Option<Column>) -> Column {
    func(
        "sentences",
        vec![
            string.expression().clone(),
            or_literal(language, lit_str("")),
            or_literal(country, lit_str("")),
        ],
    )
}

/// `randstr(length, seed)`, with a fixed seed when none is given.
pub fn randstr(length: Column, seed: Option<Column>) -> Column {
    func(
        "randstr",
        vec![
            length.expression().clone(),
            or_literal(seed, lit_long(1000462829105445681)),
        ],
    )
}
