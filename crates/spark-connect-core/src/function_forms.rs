//! Forms of SQL function calls with the arguments that the builders of [`crate::functions`]
//! leave out, fix or take only in part. Those builders call these.

use std::iter;

use crate::column::Column;
use crate::expression::{Expression, ExpressionNode, LiteralExpression};
use crate::functions::{
    self, func, func_distinct, lit_bool, lit_int, lit_long, lit_str, make_lambda,
};
use crate::types::DataType;

fn expressions(columns: impl IntoIterator<Item = Column>) -> Vec<Expression> {
    columns
        .into_iter()
        .map(|column| column.expression().clone())
        .collect()
}

fn string(value: &str) -> Column {
    Column::new(lit_str(value))
}

fn long(value: i64) -> Column {
    Column::new(lit_long(value))
}

const TIMESTAMP_FORMAT: &str = "yyyy-MM-dd HH:mm:ss";

/// What a call sends for an optional argument that is not given.
enum Absent {
    /// This literal, always.
    Sent(Expression),
    /// Nothing, or this literal when a later argument is given.
    Filled(Expression),
    /// Nothing; only the last arguments are omitted this way.
    Omitted,
}

/// `name(args..., options...)`, sending each option up to the last one that is given or
/// always sent.
fn with_options<const N: usize>(
    name: &str,
    args: impl IntoIterator<Item = Column>,
    options: [(Option<Column>, Absent); N],
) -> Column {
    let mut args = expressions(args);
    let sent = options
        .iter()
        .rposition(|(given, absent)| given.is_some() || matches!(absent, Absent::Sent(_)))
        .map_or(0, |last| last + 1);
    args.reserve(sent);
    for (given, absent) in options.into_iter().take(sent) {
        args.push(match (given, absent) {
            (Some(given), _) => given.expression().clone(),
            (None, Absent::Sent(literal) | Absent::Filled(literal)) => literal,
            (None, Absent::Omitted) => {
                unreachable!("an omitted argument of {name} precedes a given one")
            }
        });
    }
    func(name, args)
}

fn decimal_zero() -> Expression {
    Expression::literal(LiteralExpression::Decimal {
        value: "0".to_string(),
        precision: 10,
        scale: 0,
    })
}

fn null() -> Expression {
    Expression::literal(LiteralExpression::null(DataType::Null))
}

/// `ln(arg1)`, or `log(arg1, arg2)`: the logarithm of `arg2` in the base `arg1`.
pub fn log(arg1: Column, arg2: Option<Column>) -> Column {
    match arg2 {
        None => func("ln", expressions([arg1])),
        Some(arg2) => func("log", expressions([arg1, arg2])),
    }
}

/// `unix_timestamp()`, the current time, or `unix_timestamp(timestamp, format)`, with the format
/// `yyyy-MM-dd HH:mm:ss` when none is given. A format without a timestamp is not sent.
pub fn unix_timestamp(timestamp: Option<Column>, format: Option<Column>) -> Column {
    match timestamp {
        None => func("unix_timestamp", vec![]),
        Some(timestamp) => with_options(
            "unix_timestamp",
            [timestamp],
            [(format, Absent::Sent(lit_str(TIMESTAMP_FORMAT)))],
        ),
    }
}

/// `array_sort(col[, comparator])`, with a comparator of two elements.
pub fn array_sort<F>(col: Column, comparator: Option<F>) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let mut args = vec![col.expression().clone()];
    args.extend(comparator.map(|comparator| {
        ExpressionNode::LambdaFunction(Box::new(make_lambda(2, comparator))).into()
    }));
    func("array_sort", args)
}

/// `window(time, duration[, slide[, start]])`: a slide is the duration when only a start is
/// given.
pub fn window(
    col: Column,
    window_duration: &str,
    slide_duration: Option<&str>,
    start_time: Option<&str>,
) -> Column {
    with_options(
        "window",
        [col, string(window_duration)],
        [
            (
                slide_duration.map(string),
                Absent::Filled(lit_str(window_duration)),
            ),
            (start_time.map(string), Absent::Omitted),
        ],
    )
}

/// `instr(str, substr[, start[, occurrence]])`: a start is 1 when only an occurrence is given.
pub fn instr(
    str: Column,
    substr: Column,
    start: Option<Column>,
    occurrence: Option<Column>,
) -> Column {
    with_options(
        "instr",
        [str, substr],
        [
            (start, Absent::Filled(lit_int(1))),
            (occurrence, Absent::Omitted),
        ],
    )
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
    with_options(
        "lag",
        [col],
        [
            (offset, Absent::Sent(lit_int(1))),
            (default, Absent::Omitted),
        ],
    )
}

/// `lead(col, offset, default)`, with an offset of 1 when none is given.
pub fn lead(col: Column, offset: Option<Column>, default: Option<Column>) -> Column {
    with_options(
        "lead",
        [col],
        [
            (offset, Absent::Sent(lit_int(1))),
            (default, Absent::Omitted),
        ],
    )
}

/// `listagg(col[, delimiter])`.
pub fn listagg(col: Column, delimiter: Option<Column>) -> Column {
    with_options("listagg", [col], [(delimiter, Absent::Omitted)])
}

/// `string_agg(col[, delimiter])`.
pub fn string_agg(col: Column, delimiter: Option<Column>) -> Column {
    with_options("string_agg", [col], [(delimiter, Absent::Omitted)])
}

/// `sort_array(col, asc)`, ascending when `asc` is not given.
pub fn sort_array(col: Column, asc: Option<Column>) -> Column {
    with_options("sort_array", [col], [(asc, Absent::Sent(lit_bool(true)))])
}

/// `str_to_map(text, pair_delim, key_value_delim)`, with `,` and `:` when not given.
pub fn str_to_map(
    text: Column,
    pair_delim: Option<Column>,
    key_value_delim: Option<Column>,
) -> Column {
    with_options(
        "str_to_map",
        [text],
        [
            (pair_delim, Absent::Sent(lit_str(","))),
            (key_value_delim, Absent::Sent(lit_str(":"))),
        ],
    )
}

/// `sentences(string, language, country)`, with empty strings when not given.
pub fn sentences(string: Column, language: Option<Column>, country: Option<Column>) -> Column {
    with_options(
        "sentences",
        [string],
        [
            (language, Absent::Sent(lit_str(""))),
            (country, Absent::Sent(lit_str(""))),
        ],
    )
}

/// `randstr(length, seed)`, with a fixed seed when none is given.
pub fn randstr(length: Column, seed: Option<i64>) -> Column {
    with_options(
        "randstr",
        [length],
        [(seed.map(long), Absent::Sent(lit_long(1000462829105445681)))],
    )
}

/// `aggregate(col, initial_value, merge[, finish])`, which `reduce` is.
pub fn reduce<F, G>(col: Column, initial_value: Column, merge: F, finish: Option<G>) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
    G: FnOnce(Vec<Column>) -> Column,
{
    match finish {
        None => functions::aggregate(col, initial_value, merge),
        Some(finish) => functions::aggregate_with_finish(col, initial_value, merge, finish),
    }
}

/// `make_dt_interval(days, hours, mins, secs)`. Arguments not given are `days = 0`, `hours = 0`,
/// `mins = 0`, `secs = 0`.
pub fn make_dt_interval(
    days: Option<Column>,
    hours: Option<Column>,
    mins: Option<Column>,
    secs: Option<Column>,
) -> Column {
    with_options(
        "make_dt_interval",
        [],
        [
            (days, Absent::Sent(lit_int(0))),
            (hours, Absent::Sent(lit_int(0))),
            (mins, Absent::Sent(lit_int(0))),
            (secs, Absent::Sent(decimal_zero())),
        ],
    )
}

/// `make_interval(years, months, weeks, days, hours, mins, secs)`. Arguments not given are
/// `years = 0`, `months = 0`, `weeks = 0`, `days = 0`, `hours = 0`, `mins = 0`, `secs = 0`.
pub fn make_interval(
    years: Option<Column>,
    months: Option<Column>,
    weeks: Option<Column>,
    days: Option<Column>,
    hours: Option<Column>,
    mins: Option<Column>,
    secs: Option<Column>,
) -> Column {
    with_options(
        "make_interval",
        [],
        [
            (years, Absent::Sent(lit_int(0))),
            (months, Absent::Sent(lit_int(0))),
            (weeks, Absent::Sent(lit_int(0))),
            (days, Absent::Sent(lit_int(0))),
            (hours, Absent::Sent(lit_int(0))),
            (mins, Absent::Sent(lit_int(0))),
            (secs, Absent::Sent(decimal_zero())),
        ],
    )
}

/// `try_make_interval(years, months, weeks, days, hours, mins, secs)`. Arguments not given are
/// `years = 0`, `months = 0`, `weeks = 0`, `days = 0`, `hours = 0`, `mins = 0`, `secs = 0`.
pub fn try_make_interval(
    years: Option<Column>,
    months: Option<Column>,
    weeks: Option<Column>,
    days: Option<Column>,
    hours: Option<Column>,
    mins: Option<Column>,
    secs: Option<Column>,
) -> Column {
    with_options(
        "try_make_interval",
        [],
        [
            (years, Absent::Sent(lit_int(0))),
            (months, Absent::Sent(lit_int(0))),
            (weeks, Absent::Sent(lit_int(0))),
            (days, Absent::Sent(lit_int(0))),
            (hours, Absent::Sent(lit_int(0))),
            (mins, Absent::Sent(lit_int(0))),
            (secs, Absent::Sent(decimal_zero())),
        ],
    )
}

/// `make_ym_interval(years, months)`. Arguments not given are `years = 0`, `months = 0`.
pub fn make_ym_interval(years: Option<Column>, months: Option<Column>) -> Column {
    with_options(
        "make_ym_interval",
        [],
        [
            (years, Absent::Sent(lit_int(0))),
            (months, Absent::Sent(lit_int(0))),
        ],
    )
}

/// `rand(seed)`. Arguments not given are `seed = 2371029763967735218`.
pub fn rand(seed: Option<i64>) -> Column {
    with_options(
        "rand",
        [],
        [(seed.map(long), Absent::Sent(lit_long(2371029763967735218)))],
    )
}

/// `randn(seed)`. Arguments not given are `seed = 2561831801369089374`.
pub fn randn(seed: Option<i64>) -> Column {
    with_options(
        "randn",
        [],
        [(seed.map(long), Absent::Sent(lit_long(2561831801369089374)))],
    )
}

/// `uuid(seed)`. Arguments not given are `seed = 1214072022411175128`.
pub fn uuid(seed: Option<i64>) -> Column {
    with_options(
        "uuid",
        [],
        [(seed.map(long), Absent::Sent(lit_long(1214072022411175128)))],
    )
}

/// `shuffle(col, seed)`. Arguments not given are `seed = 1810822539051918711`.
pub fn shuffle(col: Column, seed: Option<i64>) -> Column {
    with_options(
        "shuffle",
        [col],
        [(seed.map(long), Absent::Sent(lit_long(1810822539051918711)))],
    )
}

/// `uniform(min, max, seed)`. Arguments not given are `seed = 1779860837174053365`.
pub fn uniform(min: Column, max: Column, seed: Option<i64>) -> Column {
    with_options(
        "uniform",
        [min, max],
        [(seed.map(long), Absent::Sent(lit_long(1779860837174053365)))],
    )
}

/// `count_min_sketch(col, eps, confidence, seed)`. Arguments not given are
/// `seed = 9151685696388386553`.
pub fn count_min_sketch(col: Column, eps: Column, confidence: Column, seed: Option<i64>) -> Column {
    with_options(
        "count_min_sketch",
        [col, eps, confidence],
        [(seed.map(long), Absent::Sent(lit_long(9151685696388386553)))],
    )
}

/// `first(col, ignorenulls)`. Arguments not given are `ignorenulls = false`.
pub fn first(col: Column, ignorenulls: Option<Column>) -> Column {
    with_options(
        "first",
        [col],
        [(ignorenulls, Absent::Sent(lit_bool(false)))],
    )
}

/// `last(col, ignorenulls)`. Arguments not given are `ignorenulls = false`.
pub fn last(col: Column, ignorenulls: Option<Column>) -> Column {
    with_options(
        "last",
        [col],
        [(ignorenulls, Absent::Sent(lit_bool(false)))],
    )
}

/// `mode(col, deterministic)`. Arguments not given are `deterministic = false`.
pub fn mode(col: Column, deterministic: Option<Column>) -> Column {
    with_options(
        "mode",
        [col],
        [(deterministic, Absent::Sent(lit_bool(false)))],
    )
}

/// `nth_value(col, offset, ignore_nulls)`. Arguments not given are `ignore_nulls = false`.
pub fn nth_value(col: Column, offset: Column, ignore_nulls: Option<Column>) -> Column {
    with_options(
        "nth_value",
        [col, offset],
        [(ignore_nulls, Absent::Sent(lit_bool(false)))],
    )
}

/// `months_between(date1, date2, round_off)`. Arguments not given are `round_off = true`.
pub fn months_between(date1: Column, date2: Column, round_off: Option<Column>) -> Column {
    with_options(
        "months_between",
        [date1, date2],
        [(round_off, Absent::Sent(lit_bool(true)))],
    )
}

/// `from_unixtime(timestamp, format)`. Arguments not given are `format = "yyyy-MM-dd HH:mm:ss"`.
pub fn from_unixtime(timestamp: Column, format: Option<Column>) -> Column {
    with_options(
        "from_unixtime",
        [timestamp],
        [(format, Absent::Sent(lit_str(TIMESTAMP_FORMAT)))],
    )
}

/// `mask(col, upper_char, lower_char, digit_char, other_char)`. Arguments not given are
/// `upper_char = "X"`, `lower_char = "x"`, `digit_char = "n"`, `other_char = NULL`.
pub fn mask(
    col: Column,
    upper_char: Option<Column>,
    lower_char: Option<Column>,
    digit_char: Option<Column>,
    other_char: Option<Column>,
) -> Column {
    with_options(
        "mask",
        [col],
        [
            (upper_char, Absent::Sent(lit_str("X"))),
            (lower_char, Absent::Sent(lit_str("x"))),
            (digit_char, Absent::Sent(lit_str("n"))),
            (other_char, Absent::Sent(null())),
        ],
    )
}

/// `aes_encrypt(input, key, mode, padding, iv, aad)`. Arguments not given are `mode = "GCM"`,
/// `padding = "DEFAULT"`, `iv = ""`, `aad = ""`.
pub fn aes_encrypt(
    input: Column,
    key: Column,
    mode: Option<Column>,
    padding: Option<Column>,
    iv: Option<Column>,
    aad: Option<Column>,
) -> Column {
    with_options(
        "aes_encrypt",
        [input, key],
        [
            (mode, Absent::Sent(lit_str("GCM"))),
            (padding, Absent::Sent(lit_str("DEFAULT"))),
            (iv, Absent::Sent(lit_str(""))),
            (aad, Absent::Sent(lit_str(""))),
        ],
    )
}

/// `aes_decrypt(input, key, mode, padding, aad)`. Arguments not given are `mode = "GCM"`,
/// `padding = "DEFAULT"`, `aad = ""`.
pub fn aes_decrypt(
    input: Column,
    key: Column,
    mode: Option<Column>,
    padding: Option<Column>,
    aad: Option<Column>,
) -> Column {
    with_options(
        "aes_decrypt",
        [input, key],
        [
            (mode, Absent::Sent(lit_str("GCM"))),
            (padding, Absent::Sent(lit_str("DEFAULT"))),
            (aad, Absent::Sent(lit_str(""))),
        ],
    )
}

/// `try_aes_decrypt(input, key, mode, padding, aad)`. Arguments not given are `mode = "GCM"`,
/// `padding = "DEFAULT"`, `aad = ""`.
pub fn try_aes_decrypt(
    input: Column,
    key: Column,
    mode: Option<Column>,
    padding: Option<Column>,
    aad: Option<Column>,
) -> Column {
    with_options(
        "try_aes_decrypt",
        [input, key],
        [
            (mode, Absent::Sent(lit_str("GCM"))),
            (padding, Absent::Sent(lit_str("DEFAULT"))),
            (aad, Absent::Sent(lit_str(""))),
        ],
    )
}

/// `approx_percentile(col, percentage, accuracy)`. Arguments not given are `accuracy = 10000`.
pub fn approx_percentile(col: Column, percentage: Column, accuracy: Option<Column>) -> Column {
    with_options(
        "approx_percentile",
        [col, percentage],
        [(accuracy, Absent::Sent(lit_int(10000)))],
    )
}

/// `percentile_approx(col, percentage, accuracy)`. Arguments not given are `accuracy = 10000`.
pub fn percentile_approx(col: Column, percentage: Column, accuracy: Option<Column>) -> Column {
    with_options(
        "percentile_approx",
        [col, percentage],
        [(accuracy, Absent::Sent(lit_int(10000)))],
    )
}

/// `percentile(col, percentage, frequency)`. Arguments not given are `frequency = 1`.
pub fn percentile(col: Column, percentage: Column, frequency: Option<Column>) -> Column {
    with_options(
        "percentile",
        [col, percentage],
        [(frequency, Absent::Sent(lit_int(1)))],
    )
}

/// `locate(substr, str, pos)`. Arguments not given are `pos = 1`.
pub fn locate(substr: Column, str: Column, pos: Option<Column>) -> Column {
    with_options("locate", [substr, str], [(pos, Absent::Sent(lit_int(1)))])
}

/// `split(str, pattern, limit)`. Arguments not given are `limit = -1`.
pub fn split(str: Column, pattern: Column, limit: Option<Column>) -> Column {
    with_options(
        "split",
        [str, pattern],
        [(limit, Absent::Sent(lit_int(-1)))],
    )
}

/// `overlay(src, replace, pos, len)`. Arguments not given are `len = -1`.
pub fn overlay(src: Column, replace: Column, pos: Column, len: Option<Column>) -> Column {
    with_options(
        "overlay",
        [src, replace, pos],
        [(len, Absent::Sent(lit_int(-1)))],
    )
}

/// `tuple_sketch_agg_double(key, summary, lg_nom_entries, mode)`. Arguments not given are
/// `lg_nom_entries = 12`, `mode = "sum"`.
pub fn tuple_sketch_agg_double(
    key: Column,
    summary: Column,
    lg_nom_entries: Option<Column>,
    mode: Option<Column>,
) -> Column {
    with_options(
        "tuple_sketch_agg_double",
        [key, summary],
        [
            (lg_nom_entries, Absent::Sent(lit_int(12))),
            (mode, Absent::Sent(lit_str("sum"))),
        ],
    )
}

/// `tuple_union_agg_double(col, lg_nom_entries, mode)`. `lg_nom_entries` is `12` when only a later
/// argument is given.
pub fn tuple_union_agg_double(
    col: Column,
    lg_nom_entries: Option<Column>,
    mode: Option<Column>,
) -> Column {
    with_options(
        "tuple_union_agg_double",
        [col],
        [
            (lg_nom_entries, Absent::Filled(lit_int(12))),
            (mode, Absent::Omitted),
        ],
    )
}

/// `tuple_union_double(col1, col2, lg_nom_entries, mode)`. `lg_nom_entries` is `12` when only a
/// later argument is given.
pub fn tuple_union_double(
    col1: Column,
    col2: Column,
    lg_nom_entries: Option<Column>,
    mode: Option<Column>,
) -> Column {
    with_options(
        "tuple_union_double",
        [col1, col2],
        [
            (lg_nom_entries, Absent::Filled(lit_int(12))),
            (mode, Absent::Omitted),
        ],
    )
}

/// `tuple_union_theta_double(col1, col2, lg_nom_entries, mode)`. `lg_nom_entries` is `12` when only
/// a later argument is given.
pub fn tuple_union_theta_double(
    col1: Column,
    col2: Column,
    lg_nom_entries: Option<Column>,
    mode: Option<Column>,
) -> Column {
    with_options(
        "tuple_union_theta_double",
        [col1, col2],
        [
            (lg_nom_entries, Absent::Filled(lit_int(12))),
            (mode, Absent::Omitted),
        ],
    )
}

/// `tuple_intersection_agg_double(col, mode)`.
pub fn tuple_intersection_agg_double(col: Column, mode: Option<Column>) -> Column {
    with_options(
        "tuple_intersection_agg_double",
        [col],
        [(mode, Absent::Omitted)],
    )
}

/// `tuple_intersection_double(col1, col2, mode)`.
pub fn tuple_intersection_double(col1: Column, col2: Column, mode: Option<Column>) -> Column {
    with_options(
        "tuple_intersection_double",
        [col1, col2],
        [(mode, Absent::Omitted)],
    )
}

/// `tuple_intersection_theta_double(col1, col2, mode)`.
pub fn tuple_intersection_theta_double(col1: Column, col2: Column, mode: Option<Column>) -> Column {
    with_options(
        "tuple_intersection_theta_double",
        [col1, col2],
        [(mode, Absent::Omitted)],
    )
}

/// `tuple_sketch_summary_double(col, mode)`.
pub fn tuple_sketch_summary_double(col: Column, mode: Option<Column>) -> Column {
    with_options(
        "tuple_sketch_summary_double",
        [col],
        [(mode, Absent::Omitted)],
    )
}

/// `tuple_sketch_agg_integer(key, summary, lg_nom_entries, mode)`. Arguments not given are
/// `lg_nom_entries = 12`, `mode = "sum"`.
pub fn tuple_sketch_agg_integer(
    key: Column,
    summary: Column,
    lg_nom_entries: Option<Column>,
    mode: Option<Column>,
) -> Column {
    with_options(
        "tuple_sketch_agg_integer",
        [key, summary],
        [
            (lg_nom_entries, Absent::Sent(lit_int(12))),
            (mode, Absent::Sent(lit_str("sum"))),
        ],
    )
}

/// `tuple_union_agg_integer(col, lg_nom_entries, mode)`. `lg_nom_entries` is `12` when only a later
/// argument is given.
pub fn tuple_union_agg_integer(
    col: Column,
    lg_nom_entries: Option<Column>,
    mode: Option<Column>,
) -> Column {
    with_options(
        "tuple_union_agg_integer",
        [col],
        [
            (lg_nom_entries, Absent::Filled(lit_int(12))),
            (mode, Absent::Omitted),
        ],
    )
}

/// `tuple_union_integer(col1, col2, lg_nom_entries, mode)`. `lg_nom_entries` is `12` when only a
/// later argument is given.
pub fn tuple_union_integer(
    col1: Column,
    col2: Column,
    lg_nom_entries: Option<Column>,
    mode: Option<Column>,
) -> Column {
    with_options(
        "tuple_union_integer",
        [col1, col2],
        [
            (lg_nom_entries, Absent::Filled(lit_int(12))),
            (mode, Absent::Omitted),
        ],
    )
}

/// `tuple_union_theta_integer(col1, col2, lg_nom_entries, mode)`. `lg_nom_entries` is `12` when
/// only a later argument is given.
pub fn tuple_union_theta_integer(
    col1: Column,
    col2: Column,
    lg_nom_entries: Option<Column>,
    mode: Option<Column>,
) -> Column {
    with_options(
        "tuple_union_theta_integer",
        [col1, col2],
        [
            (lg_nom_entries, Absent::Filled(lit_int(12))),
            (mode, Absent::Omitted),
        ],
    )
}

/// `tuple_intersection_agg_integer(col, mode)`.
pub fn tuple_intersection_agg_integer(col: Column, mode: Option<Column>) -> Column {
    with_options(
        "tuple_intersection_agg_integer",
        [col],
        [(mode, Absent::Omitted)],
    )
}

/// `tuple_intersection_integer(col1, col2, mode)`.
pub fn tuple_intersection_integer(col1: Column, col2: Column, mode: Option<Column>) -> Column {
    with_options(
        "tuple_intersection_integer",
        [col1, col2],
        [(mode, Absent::Omitted)],
    )
}

/// `tuple_intersection_theta_integer(col1, col2, mode)`.
pub fn tuple_intersection_theta_integer(
    col1: Column,
    col2: Column,
    mode: Option<Column>,
) -> Column {
    with_options(
        "tuple_intersection_theta_integer",
        [col1, col2],
        [(mode, Absent::Omitted)],
    )
}

/// `tuple_sketch_summary_integer(col, mode)`.
pub fn tuple_sketch_summary_integer(col: Column, mode: Option<Column>) -> Column {
    with_options(
        "tuple_sketch_summary_integer",
        [col],
        [(mode, Absent::Omitted)],
    )
}

/// `any_value(col, ignore_nulls)`.
pub fn any_value(col: Column, ignore_nulls: Option<Column>) -> Column {
    with_options("any_value", [col], [(ignore_nulls, Absent::Omitted)])
}

/// `first_value(col, ignore_nulls)`.
pub fn first_value(col: Column, ignore_nulls: Option<Column>) -> Column {
    with_options("first_value", [col], [(ignore_nulls, Absent::Omitted)])
}

/// `last_value(col, ignore_nulls)`.
pub fn last_value(col: Column, ignore_nulls: Option<Column>) -> Column {
    with_options("last_value", [col], [(ignore_nulls, Absent::Omitted)])
}

/// `assert_true(col, err_msg)`.
pub fn assert_true(col: Column, err_msg: Option<Column>) -> Column {
    with_options("assert_true", [col], [(err_msg, Absent::Omitted)])
}

/// `btrim(str, trim)`.
pub fn btrim(str: Column, trim: Option<Column>) -> Column {
    with_options("btrim", [str], [(trim, Absent::Omitted)])
}

/// `ceiling(col, scale)`.
pub fn ceiling(col: Column, scale: Option<Column>) -> Column {
    with_options("ceiling", [col], [(scale, Absent::Omitted)])
}

/// `hll_sketch_agg(col, lg_config_k)`.
pub fn hll_sketch_agg(col: Column, lg_config_k: Option<Column>) -> Column {
    with_options("hll_sketch_agg", [col], [(lg_config_k, Absent::Omitted)])
}

/// `hll_union_agg(col, allow_different_lg_config_k)`.
pub fn hll_union_agg(col: Column, allow_different_lg_config_k: Option<Column>) -> Column {
    with_options(
        "hll_union_agg",
        [col],
        [(allow_different_lg_config_k, Absent::Omitted)],
    )
}

/// `hll_union(col1, col2, allow_different_lg_config_k)`.
pub fn hll_union(
    col1: Column,
    col2: Column,
    allow_different_lg_config_k: Option<Column>,
) -> Column {
    with_options(
        "hll_union",
        [col1, col2],
        [(allow_different_lg_config_k, Absent::Omitted)],
    )
}

/// `schema_of_csv(csv, options)`.
pub fn schema_of_csv(csv: Column, options: Option<Column>) -> Column {
    with_options("schema_of_csv", [csv], [(options, Absent::Omitted)])
}

/// `schema_of_json(json, options)`.
pub fn schema_of_json(json: Column, options: Option<Column>) -> Column {
    with_options("schema_of_json", [json], [(options, Absent::Omitted)])
}

/// `schema_of_xml(xml, options)`.
pub fn schema_of_xml(xml: Column, options: Option<Column>) -> Column {
    with_options("schema_of_xml", [xml], [(options, Absent::Omitted)])
}

/// `to_csv(col, options)`.
pub fn to_csv(col: Column, options: Option<Column>) -> Column {
    with_options("to_csv", [col], [(options, Absent::Omitted)])
}

/// `to_json(col, options)`.
pub fn to_json(col: Column, options: Option<Column>) -> Column {
    with_options("to_json", [col], [(options, Absent::Omitted)])
}

/// `to_xml(col, options)`.
pub fn to_xml(col: Column, options: Option<Column>) -> Column {
    with_options("to_xml", [col], [(options, Absent::Omitted)])
}

/// `from_csv(col, schema, options)`.
pub fn from_csv(col: Column, schema: Column, options: Option<Column>) -> Column {
    with_options("from_csv", [col, schema], [(options, Absent::Omitted)])
}

/// `from_json(col, schema, options)`.
pub fn from_json(col: Column, schema: Column, options: Option<Column>) -> Column {
    with_options("from_json", [col, schema], [(options, Absent::Omitted)])
}

/// `from_xml(col, schema, options)`.
pub fn from_xml(col: Column, schema: Column, options: Option<Column>) -> Column {
    with_options("from_xml", [col, schema], [(options, Absent::Omitted)])
}

/// `to_date(col, format)`.
pub fn to_date(col: Column, format: Option<Column>) -> Column {
    with_options("to_date", [col], [(format, Absent::Omitted)])
}

/// `to_timestamp(col, format)`.
pub fn to_timestamp(col: Column, format: Option<Column>) -> Column {
    with_options("to_timestamp", [col], [(format, Absent::Omitted)])
}

/// `try_to_date(col, format)`.
pub fn try_to_date(col: Column, format: Option<Column>) -> Column {
    with_options("try_to_date", [col], [(format, Absent::Omitted)])
}

/// `to_timestamp_ltz(timestamp, format)`.
pub fn to_timestamp_ltz(timestamp: Column, format: Option<Column>) -> Column {
    with_options("to_timestamp_ltz", [timestamp], [(format, Absent::Omitted)])
}

/// `to_timestamp_ntz(timestamp, format)`.
pub fn to_timestamp_ntz(timestamp: Column, format: Option<Column>) -> Column {
    with_options("to_timestamp_ntz", [timestamp], [(format, Absent::Omitted)])
}

/// `to_unix_timestamp(timestamp, format)`.
pub fn to_unix_timestamp(timestamp: Column, format: Option<Column>) -> Column {
    with_options(
        "to_unix_timestamp",
        [timestamp],
        [(format, Absent::Omitted)],
    )
}

/// `try_to_binary(col, format)`.
pub fn try_to_binary(col: Column, format: Option<Column>) -> Column {
    with_options("try_to_binary", [col], [(format, Absent::Omitted)])
}

/// `try_to_timestamp(col, format)`.
pub fn try_to_timestamp(col: Column, format: Option<Column>) -> Column {
    with_options("try_to_timestamp", [col], [(format, Absent::Omitted)])
}

/// `to_time(str, format)`.
pub fn to_time(str: Column, format: Option<Column>) -> Column {
    with_options("to_time", [str], [(format, Absent::Omitted)])
}

/// `try_to_time(str, format)`.
pub fn try_to_time(str: Column, format: Option<Column>) -> Column {
    with_options("try_to_time", [str], [(format, Absent::Omitted)])
}

/// `array_join(col, delimiter, null_replacement)`.
pub fn array_join(col: Column, delimiter: Column, null_replacement: Option<Column>) -> Column {
    with_options(
        "array_join",
        [col, delimiter],
        [(null_replacement, Absent::Omitted)],
    )
}

/// `ilike(str, pattern, escape_char)`.
pub fn ilike(str: Column, pattern: Column, escape_char: Option<Column>) -> Column {
    with_options("ilike", [str, pattern], [(escape_char, Absent::Omitted)])
}

/// `like(str, pattern, escape_char)`.
pub fn like(str: Column, pattern: Column, escape_char: Option<Column>) -> Column {
    with_options("like", [str, pattern], [(escape_char, Absent::Omitted)])
}

/// `levenshtein(left, right, threshold)`.
pub fn levenshtein(left: Column, right: Column, threshold: Option<Column>) -> Column {
    with_options("levenshtein", [left, right], [(threshold, Absent::Omitted)])
}

/// `max_by(col, ord, k)`.
pub fn max_by(col: Column, ord: Column, k: Option<Column>) -> Column {
    with_options("max_by", [col, ord], [(k, Absent::Omitted)])
}

/// `min_by(col, ord, k)`.
pub fn min_by(col: Column, ord: Column, k: Option<Column>) -> Column {
    with_options("min_by", [col, ord], [(k, Absent::Omitted)])
}

/// `parse_url(url, part_to_extract, key)`.
pub fn parse_url(url: Column, part_to_extract: Column, key: Option<Column>) -> Column {
    with_options(
        "parse_url",
        [url, part_to_extract],
        [(key, Absent::Omitted)],
    )
}

/// `try_parse_url(url, part_to_extract, key)`.
pub fn try_parse_url(url: Column, part_to_extract: Column, key: Option<Column>) -> Column {
    with_options(
        "try_parse_url",
        [url, part_to_extract],
        [(key, Absent::Omitted)],
    )
}

/// `position(substr, str, start)`.
pub fn position(substr: Column, str: Column, start: Option<Column>) -> Column {
    with_options("position", [substr, str], [(start, Absent::Omitted)])
}

/// `regexp_extract_all(str, regexp, idx)`.
pub fn regexp_extract_all(str: Column, regexp: Column, idx: Option<Column>) -> Column {
    with_options(
        "regexp_extract_all",
        [str, regexp],
        [(idx, Absent::Omitted)],
    )
}

/// `regexp_instr(str, regexp, idx)`.
pub fn regexp_instr(str: Column, regexp: Column, idx: Option<Column>) -> Column {
    with_options("regexp_instr", [str, regexp], [(idx, Absent::Omitted)])
}

/// `replace(src, search, replace)`.
pub fn replace(src: Column, search: Column, replace: Option<Column>) -> Column {
    with_options("replace", [src, search], [(replace, Absent::Omitted)])
}

/// `sequence(start, stop, step)`.
pub fn sequence(start: Column, stop: Column, step: Option<Column>) -> Column {
    with_options("sequence", [start, stop], [(step, Absent::Omitted)])
}

/// `substr(str, pos, len)`.
pub fn substr(str: Column, pos: Column, len: Option<Column>) -> Column {
    with_options("substr", [str, pos], [(len, Absent::Omitted)])
}

/// `current_time(precision)`.
pub fn current_time(precision: Option<Column>) -> Column {
    with_options("current_time", [], [(precision, Absent::Omitted)])
}

/// `kll_merge_agg_bigint(col, k)`.
pub fn kll_merge_agg_bigint(col: Column, k: Option<Column>) -> Column {
    with_options("kll_merge_agg_bigint", [col], [(k, Absent::Omitted)])
}

/// `kll_merge_agg_double(col, k)`.
pub fn kll_merge_agg_double(col: Column, k: Option<Column>) -> Column {
    with_options("kll_merge_agg_double", [col], [(k, Absent::Omitted)])
}

/// `kll_merge_agg_float(col, k)`.
pub fn kll_merge_agg_float(col: Column, k: Option<Column>) -> Column {
    with_options("kll_merge_agg_float", [col], [(k, Absent::Omitted)])
}

/// `kll_sketch_agg_bigint(col, k)`.
pub fn kll_sketch_agg_bigint(col: Column, k: Option<Column>) -> Column {
    with_options("kll_sketch_agg_bigint", [col], [(k, Absent::Omitted)])
}

/// `kll_sketch_agg_double(col, k)`.
pub fn kll_sketch_agg_double(col: Column, k: Option<Column>) -> Column {
    with_options("kll_sketch_agg_double", [col], [(k, Absent::Omitted)])
}

/// `kll_sketch_agg_float(col, k)`.
pub fn kll_sketch_agg_float(col: Column, k: Option<Column>) -> Column {
    with_options("kll_sketch_agg_float", [col], [(k, Absent::Omitted)])
}

/// `theta_sketch_agg(col, lg_nom_entries)`.
pub fn theta_sketch_agg(col: Column, lg_nom_entries: Option<Column>) -> Column {
    with_options(
        "theta_sketch_agg",
        [col],
        [(lg_nom_entries, Absent::Omitted)],
    )
}

/// `theta_union(col1, col2, lg_nom_entries)`.
pub fn theta_union(col1: Column, col2: Column, lg_nom_entries: Option<Column>) -> Column {
    with_options(
        "theta_union",
        [col1, col2],
        [(lg_nom_entries, Absent::Omitted)],
    )
}

/// `theta_union_agg(col, lg_nom_entries)`.
pub fn theta_union_agg(col: Column, lg_nom_entries: Option<Column>) -> Column {
    with_options(
        "theta_union_agg",
        [col],
        [(lg_nom_entries, Absent::Omitted)],
    )
}

/// `make_timestamp(years, months, days, hours, mins, secs, timezone)`.
pub fn make_timestamp(
    years: Column,
    months: Column,
    days: Column,
    hours: Column,
    mins: Column,
    secs: Column,
    timezone: Option<Column>,
) -> Column {
    with_options(
        "make_timestamp",
        [years, months, days, hours, mins, secs],
        [(timezone, Absent::Omitted)],
    )
}

/// `make_timestamp_ltz(years, months, days, hours, mins, secs, timezone)`.
pub fn make_timestamp_ltz(
    years: Column,
    months: Column,
    days: Column,
    hours: Column,
    mins: Column,
    secs: Column,
    timezone: Option<Column>,
) -> Column {
    with_options(
        "make_timestamp_ltz",
        [years, months, days, hours, mins, secs],
        [(timezone, Absent::Omitted)],
    )
}

/// `try_make_timestamp(years, months, days, hours, mins, secs, timezone)`.
pub fn try_make_timestamp(
    years: Column,
    months: Column,
    days: Column,
    hours: Column,
    mins: Column,
    secs: Column,
    timezone: Option<Column>,
) -> Column {
    with_options(
        "try_make_timestamp",
        [years, months, days, hours, mins, secs],
        [(timezone, Absent::Omitted)],
    )
}

/// `try_make_timestamp_ltz(years, months, days, hours, mins, secs, timezone)`.
pub fn try_make_timestamp_ltz(
    years: Column,
    months: Column,
    days: Column,
    hours: Column,
    mins: Column,
    secs: Column,
    timezone: Option<Column>,
) -> Column {
    with_options(
        "try_make_timestamp_ltz",
        [years, months, days, hours, mins, secs],
        [(timezone, Absent::Omitted)],
    )
}

/// `st_asbinary(geo, endianness)`.
pub fn st_asbinary(geo: Column, endianness: Option<Column>) -> Column {
    with_options("st_asbinary", [geo], [(endianness, Absent::Omitted)])
}

/// `st_geomfromwkb(wkb, srid)`.
pub fn st_geomfromwkb(wkb: Column, srid: Option<Column>) -> Column {
    with_options("st_geomfromwkb", [wkb], [(srid, Absent::Omitted)])
}

/// `time_bucket(bucket_size, ts, origin)`.
pub fn time_bucket(bucket_size: Column, ts: Column, origin: Option<Column>) -> Column {
    with_options(
        "time_bucket",
        [bucket_size, ts],
        [(origin, Absent::Omitted)],
    )
}
