#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
"""Generate the SQL functions of the async crate from the core crate's function builders.

Inputs:
- the `pub fn` builders of `crates/spark-connect-core/src/functions.rs`;
- `scripts/async_functions/functions.tsv`, which says what each builder becomes: the async
  function, the Rust type of each parameter, and, for hand-written functions, the calls of the
  plan-equality test;
- `scripts/async_functions/descriptions.json`, the descriptions of Spark's SQL functions
  (`extract_spark_function_descriptions.py`), which become the rustdoc.

Outputs: `crates/spark-connect-async/src/functions/generated.rs` and the case list
`crates/spark-connect-async/tests/function_plans/cases.rs`. `--check` fails when an output or
an input is stale: a builder without a row, a row without a builder, or a SQL function without
its description.

An optional `seed:?i64` that is `None` is drawn at each call (`options::random_seed`), as
PySpark draws one; its plan-equality case gives the seed of the core form, and the case list's
`seeded()` names each seeded function for the test of the seeds drawn.

`--derive` rewrites the parameter types of the table's derived rows from the annotations of an
installed pyspark 4.3.0 (`inspect.signature`), keeping the rows marked `!`. Only it needs
pyspark; the rest uses the standard library and `rustfmt`.
"""

import argparse
import inspect
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORE = REPO / "crates/spark-connect-core/src/functions.rs"
FORMS = REPO / "crates/spark-connect-core/src/function_forms.rs"
DATA = REPO / "scripts/async_functions"
TABLE = DATA / "functions.tsv"
DESCRIPTIONS = DATA / "descriptions.json"
HAND_WRITTEN = REPO / "crates/spark-connect-async/src/functions/mod.rs"
GENERATED = REPO / "crates/spark-connect-async/src/functions/generated.rs"
CASES = REPO / "crates/spark-connect-async/tests/function_plans/cases.rs"
PYSPARK_VERSION = "4.3.0"
COLUMNS = ("core", "function", "arguments", "doc", "old", "new")
HEADER = f"""\
# What each builder of crates/spark-connect-core/src/functions.rs becomes in the async crate.
# Read by scripts/gen_async_functions.py; one row per builder, tab-separated.
#   core       the core builder
#   function   the async function: generated from this row, or hand-written in
#              crates/spark-connect-async/src/functions/mod.rs, or one that another row makes
#   arguments  for a generated function, `name:type` per core parameter, in order. Types:
#                col (impl Into<Column>), cols (columns), str, i32, i64, f64, bool,
#                bytes (Vec<u8>), arg (impl IntoArgument; arg-str and arg-float test with
#                a string or a double), args (arguments), fn1..fn3 (closures over 1..3
#                columns);
#                ?type is an Option; a `seed:?i64` that is None is drawn at each call.
#              Builders of one function with different options are one function that calls
#              the builder of the options given. A row with more arguments than its builder
#              has parameters calls the form of that name in function_forms.rs, which takes
#              the further ones as options. Derived from the parameter types of
#              pyspark {PYSPARK_VERSION} (--derive); `!` marks a row set by hand.
#              `hand`: hand-written; `same`: the function of another row.
#   doc        the rustdoc of a generated function whose SQL function has no description
#   old, new   for hand and same rows, the calls of the plan-equality test
"""

KEYWORDS = {
    "as", "async", "await", "box", "break", "const", "continue", "crate", "do", "dyn", "else",
    "enum", "extern", "false", "final", "fn", "for", "gen", "if", "impl", "in", "let", "loop",
    "macro", "match", "mod", "move", "mut", "override", "priv", "pub", "ref", "return", "self",
    "static", "struct", "super", "trait", "true", "try", "type", "typeof", "unsafe", "unsized",
    "use", "virtual", "where", "while", "yield",
}  # fmt: skip
VALUE_TYPES = {
    "str": "&str",
    "i32": "i32",
    "i64": "i64",
    "f64": "f64",
    "bool": "bool",
    "bytes": "Vec<u8>",
}
SAMPLE_VALUES = {
    "str": ['"x"', '"y"', '"z"', '"w"'],
    "i32": ["3", "4", "5", "6"],
    "i64": ["3_000_000_000", "3_000_000_001"],
    "f64": ["0.5", "0.25", "0.125"],
    "bool": ["false", "true"],
    "bytes": ["vec![1, 2]", "vec![3]"],
}
OLD_LITERALS = {"str": "ostr", "i32": "oint", "i64": "olong", "f64": "odouble", "bool": "obool"}
NATIVE = {"&str": "str", "i32": "i32", "i64": "i64", "bool": "bool", "Vec<u8>": "bytes"}


# --- core -------------------------------------------------------------------------------------


@dataclass
class CoreFunction:
    name: str
    raw: bool
    params: list[tuple[str, str]]
    sql: list[str] = field(default_factory=list)
    lambdas: list[int] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    default_seed: str | None = None

    @property
    def path(self) -> str:
        return ("r#" if self.raw else "") + self.name


def split_params(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for c in text:
        if c in "<(":
            depth += 1
        elif c in ">)":
            depth -= 1
        if c == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += c
    if current.strip():
        parts.append(current.strip())
    return parts


SIGNATURE = re.compile(
    r"^pub fn (r#)?(\w+)(?:<[^(]*>)?\((.*?)\)\s*->\s*Column\s*(?:where[^{]*)?\{", re.M | re.S
)


def form_sql_names(text: str) -> dict[str, str]:
    """The SQL function of each builder of core's function_forms module."""
    out = {}
    for match in re.finditer(r"^pub fn (\w+)\b", text, re.M):
        body = text[match.end() : text.index("\n}\n", match.end())]
        literal = re.search(r'"(\w+)"', body)
        if literal:
            out[match.group(1)] = literal.group(1)
    return out


def form_functions(text: str | None = None) -> dict[str, CoreFunction]:
    """The forms of core's function_forms module that take optional arguments, by name."""
    text = FORMS.read_text() if text is None else text
    return {f.name: f for f in core_functions(text, text)}


def core_functions(text: str | None = None, forms: str | None = None) -> list[CoreFunction]:
    """The public builders of core's functions module, in the order of the file."""
    text = CORE.read_text() if text is None else text
    forms_sql = form_sql_names(FORMS.read_text() if forms is None else forms)
    out = []
    for match in SIGNATURE.finditer(text):
        raw, name, params = match.groups()
        body = text[match.end() : text.index("\n}\n", match.end())]
        seed = re.search(r"seed\.map\(long\),\s*Absent::Sent\(lit_long\((\d+)\)\)", body)
        typed = []
        for param in split_params(params):
            pname, _, ptype = param.partition(":")
            typed.append((pname.strip(), " ".join(ptype.split())))
        out.append(
            CoreFunction(
                name=name,
                raw=bool(raw),
                params=typed,
                sql=re.findall(r'(?:func(?:_distinct)?|with_options)\(\s*"([^"]+)"', body)
                + [
                    forms_sql[n]
                    for n in re.findall(r"function_forms::(\w+)\(", body)
                    if n in forms_sql
                ],
                lambdas=[int(n) for n in re.findall(r"make_lambda\((\d+),", body)],
                calls=re.findall(r"\b(\w+)\(", body),
                default_seed=seed.group(1) if seed else None,
            )
        )
    return out


def sql_name(function: CoreFunction, by_name: dict[str, CoreFunction]) -> str | None:
    """The SQL function a builder calls, also through another builder it delegates to."""
    if function.sql:
        return function.sql[0]
    for called in function.calls:
        if called != function.name and called in by_name and by_name[called].sql:
            return by_name[called].sql[0]
    return None


def sql_names(functions: list[CoreFunction]) -> set[str]:
    """The SQL functions that the builders call."""
    by_name = {f.name: f for f in functions}
    return {n for f in functions if (n := sql_name(f, by_name))}


# --- the table --------------------------------------------------------------------------------


@dataclass
class Argument:
    name: str
    kind: str

    @property
    def optional(self) -> bool:
        return self.kind.startswith("?")

    @property
    def base(self) -> str:
        return self.kind.lstrip("?")

    def render(self) -> str:
        return f"{self.name}:{self.kind}"


@dataclass
class Row:
    core: str
    function: str
    form: str  # "derived", "pinned", "hand" or "same"
    arguments: list[Argument]
    doc: str = ""
    old: str = ""
    new: str = ""

    def render(self) -> str:
        if self.form in ("hand", "same"):
            arguments = self.form
        else:
            arguments = ("!" if self.form == "pinned" else "") + ", ".join(
                a.render() for a in self.arguments
            )
        return "\t".join([self.core, self.function, arguments, self.doc, self.old, self.new])


def is_seed(argument: Argument) -> bool:
    return argument.name == "seed" and argument.kind == "?i64"


def parse_arguments(text: str) -> list[Argument]:
    out = []
    for item in [t.strip() for t in text.split(",") if t.strip()]:
        name, _, kind = item.partition(":")
        out.append(Argument(name.strip(), kind.strip()))
    return out


def read_table(text: str | None = None) -> list[Row]:
    text = TABLE.read_text() if text is None else text
    rows = []
    lines = [line for line in text.split("\n") if line and not line.startswith("#")]
    if not lines or tuple(lines[0].split("\t")) != COLUMNS:
        raise ValueError(f"the table's first row must be {COLUMNS}")
    for line in lines[1:]:
        cells = line.split("\t")
        cells += [""] * (len(COLUMNS) - len(cells))
        core, function, arguments, doc, old, new = cells
        if arguments in ("hand", "same"):
            rows.append(Row(core, function, arguments, [], doc, old, new))
        else:
            pinned = arguments.startswith("!")
            rows.append(
                Row(
                    core,
                    function,
                    "pinned" if pinned else "derived",
                    parse_arguments(arguments.lstrip("!")),
                    doc,
                    old,
                    new,
                )
            )
    return rows


def render_table(rows: list[Row]) -> str:
    return HEADER + "\t".join(COLUMNS) + "\n" + "".join(r.render() + "\n" for r in rows)


# --- deriving the parameter types from pyspark ---------------------------------------------


def snake(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()
    return name + "_" if name in KEYWORDS else name


LITERALS = {"str": "str", "int": "i32", "float": "f64", "bool": "bool", "bytes": "bytes"}


def union_members(annotation: str) -> list[str]:
    """The types of a pyspark annotation, with `Optional` and `NoneType` dropped."""
    text = re.sub(r"ForwardRef\('([^']+)'\)", r"\1", annotation)
    text = text.replace("pyspark.sql.column.Column", "Column").replace("typing.", "")
    text = re.sub(r"^Optional\[(.*)\]$", r"\1", text)
    match = re.fullmatch(r"Union\[(.*)\]", text)
    members = split_params(match.group(1)) if match else [text]
    return [m.strip("'\" ") for m in members if m.strip() not in ("NoneType", "None")]


def derived_kind(annotation: str, core_type: str, optional: bool, arity: int = 0) -> str:
    """The parameter type of a core parameter that pyspark annotates `annotation`."""
    if core_type in NATIVE:
        kind = NATIVE[core_type]
    elif core_type.startswith("impl IntoIterator"):
        members = union_members(annotation)
        literal = "ColumnOrName" not in members and any(m in LITERALS for m in members)
        kind = "args" if literal else "cols"
    elif core_type in ("F", "G"):
        kind = f"fn{arity or callable_arity(annotation)}"
    else:
        members = union_members(annotation)
        if any(m.startswith(("Mapping[", "Dict[")) for m in members):
            members = ["Column"]
        literals = [LITERALS[m] for m in members if m in LITERALS]
        if "ColumnOrName" in members or not literals and "Column" in members:
            kind = "col"
        elif members == ["Any"]:
            kind = "arg"
        elif len(members) == 1 and literals:
            kind = literals[0]
        elif "Column" in members and literals:
            kind = (
                literals[0]
                if optional
                else "arg" + {"str": "-str", "f64": "-float"}.get(literals[0], "")
            )
        else:
            raise ValueError(f"no parameter type for the annotation {annotation!r}")
    return ("?" if optional else "") + kind


def callable_arity(annotation: str) -> int:
    """The number of columns that a pyspark `Callable[[Column, ...], Column]` takes."""
    match = re.search(r"Callable\[\[([^\]]*)\]", annotation)
    return len(split_params(match.group(1))) if match else 0


def option_type(form_type: str) -> str:
    """The core type inside a form's `Option<..>` parameter type."""
    match = re.fullmatch(r"Option<(.*)>", form_type)
    return match.group(1) if match else form_type


def annotation_text(annotation) -> str:
    return annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)


def derive(
    rows: list[Row], functions: list[CoreFunction], forms: dict[str, CoreFunction]
) -> list[Row]:
    """The table with the derived rows' parameter types recomputed from pyspark: one per core
    parameter, then, where a core form takes more, one optional argument per further parameter
    of the form."""
    import pyspark
    import pyspark.sql.functions as reference

    if pyspark.__version__ != PYSPARK_VERSION:
        raise SystemExit(f"--derive needs pyspark {PYSPARK_VERSION}, not {pyspark.__version__}")
    by_name = {f.name: f for f in functions}
    base_of: dict[str, Row] = {}
    for row in rows:
        known = base_of.get(row.function)
        if known is None or len(by_name[row.core].params) < len(by_name[known.core].params):
            base_of[row.function] = row
    out = []
    for row in rows:
        if row.form != "derived":
            out.append(row)
            continue
        core = by_name[row.core]
        base = base_of[row.function]
        reference_name = row.core if hasattr(reference, row.core) else base.core
        parameters = list(inspect.signature(getattr(reference, reference_name)).parameters.values())
        base_count = len(by_name[base.core].params)
        arguments = []
        lambdas = iter(core.lambdas)
        for i, (core_param, core_type) in enumerate(core.params):
            parameter = parameters[i] if i < len(parameters) else None
            optional = row is not base and i >= base_count
            arity = next(lambdas, 0) if core_type in ("F", "G") else 0
            kind = derived_kind(
                annotation_text(parameter.annotation) if parameter else "",
                core_type,
                optional,
                arity,
            )
            arguments.append(Argument(snake(parameter.name if parameter else core_param), kind))
        further = forms[row.core].params[len(core.params) :] if row.core in forms else []
        for parameter, (_, form_type) in zip(parameters[len(core.params) :], further):
            kind = derived_kind(annotation_text(parameter.annotation), option_type(form_type), True)
            arguments.append(Argument(snake(parameter.name), kind))
        out.append(Row(row.core, row.function, "derived", arguments, row.doc))
    return out


# --- generating -------------------------------------------------------------------------------


@dataclass
class Function:
    name: str
    rows: list[Row]
    arguments: list[Argument]
    form: CoreFunction | None

    @property
    def options(self) -> list[Argument]:
        return [a for a in self.arguments if a.optional]


def functions_of(
    rows: list[Row],
    functions: list[CoreFunction] | None = None,
    forms: dict[str, CoreFunction] | None = None,
) -> list[Function]:
    """The generated functions. One whose row has more arguments than its builder has
    parameters calls the core form of that name, which takes them all."""
    by_name = {f.name: f for f in functions or []}
    forms = forms or {}
    groups: dict[str, list[Row]] = {}
    for row in rows:
        if row.form in ("derived", "pinned"):
            groups.setdefault(row.function, []).append(row)
    out = []
    for name, group in groups.items():
        arguments: list[Argument] = []
        for row in group:
            for argument in row.arguments:
                known = next((a for a in arguments if a.name == argument.name), None)
                if known is None:
                    arguments.append(argument)
                elif known.kind != argument.kind:
                    raise ValueError(f"{name}: {argument.name} has two types")
        core = by_name.get(group[0].core)
        takes_more = len(group) == 1 and core and len(arguments) > len(core.params)
        out.append(
            Function(name, group, arguments, forms.get(group[0].core) if takes_more else None)
        )
    return out


def rust_type(kind: str) -> str:
    optional = kind.startswith("?")
    base = kind.lstrip("?")
    if base == "col":
        inner = "Column" if optional else "impl Into<Column>"
    elif base == "cols":
        inner = "impl IntoIterator<Item = impl Into<Column>>"
    elif base == "args":
        inner = "impl IntoIterator<Item = impl IntoArgument>"
    elif base.startswith("arg"):
        inner = "impl IntoArgument"
    elif base.startswith("fn"):
        columns = ", ".join(["Column"] * int(base[2:]))
        inner = (
            f"Box<dyn FnOnce({columns}) -> Column>"
            if optional
            else f"impl FnOnce({columns}) -> Column"
        )
    else:
        inner = VALUE_TYPES[base]
    return f"Option<{inner}>" if optional else inner


def core_argument(value: str, kind: str, core_type: str) -> str:
    """The expression that passes `value`, of the parameter type `kind`, to a core parameter
    of the type `core_type`."""
    base = kind.lstrip("?")
    if base == "col":
        return f"to_core({value})"
    if base == "cols":
        return f"{value}.into_iter().map(to_core)"
    if base == "args":
        return f"{value}.into_iter().map(|argument| to_core(argument.into_argument()))"
    if base.startswith("arg"):
        return f"to_core({value}.into_argument())"
    if base.startswith("fn"):
        return f"lambda{base[2:]}({value})"
    if core_type in NATIVE:
        return value
    return f"to_core(lit({value}))"


def variant_for(function: Function, given: set[str]) -> Row:
    """The row of the builder that takes the options `given`."""
    for row in function.rows:
        if {a.name for a in row.arguments if a.optional} == given:
            return row
    raise ValueError(f"{function.name}: no builder takes the options {sorted(given)}")


def call(row: Row, by_name: dict[str, CoreFunction]) -> str:
    core = by_name[row.core]
    arguments = [
        core_argument(a.name, a.kind, core_type)
        for a, (_, core_type) in zip(row.arguments, core.params)
    ]
    return f"sql::{core.path}({', '.join(arguments)})"


def form_argument(argument: Argument, form_type: str) -> str:
    """The expression that passes `argument` to a core form's parameter of type `form_type`."""
    if not argument.optional:
        return core_argument(argument.name, argument.kind, form_type)
    if is_seed(argument):
        return f"Some({argument.name}.unwrap_or_else(random_seed))"
    if form_type != "Option<Column>" and not argument.base.startswith("fn"):
        return argument.name
    if argument.base == "col":
        return f"{argument.name}.map(to_core)"
    if argument.base.startswith("fn"):
        return f"{argument.name}.map(lambda{argument.base[2:]})"
    return f"{argument.name}.map(literal)"


def body(function: Function, by_name: dict[str, CoreFunction]) -> str:
    if function.form is not None:
        arguments = ", ".join(
            form_argument(a, form_type)
            for a, (_, form_type) in zip(function.arguments, function.form.params)
        )
        return f"Column::from(function_forms::{function.form.name}({arguments}))"
    options = function.options
    if any(is_seed(o) for o in options):
        raise ValueError(f"{function.name}: a seed needs a core form")
    if not options:
        return f"Column::from({call(function.rows[0], by_name)})"
    arms = []
    for combination in product([False, True], repeat=len(options)):
        given = {o.name for o, set_ in zip(options, combination) if set_}
        patterns = [f"Some({o.name})" if o.name in given else "None" for o in options]
        arms.append(f"{tuple_of(patterns)} => {call(variant_for(function, given), by_name)},")
    scrutinee = tuple_of([o.name for o in options])
    return f"Column::from(match {scrutinee} {{ {' '.join(arms)} }})"


def tuple_of(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"({', '.join(items)})"


def escape(text: str) -> str:
    return re.sub(r"([\\\[\]<>])", r"\\\1", text)


def doc_lines(function: Function, sql: str | None, descriptions: dict) -> list[str]:
    row = function.rows[0]
    if row.doc:
        lines = [row.doc]
    else:
        entry = descriptions[sql]
        usage = entry["usage"]
        summary = usage.split(" - ", 1)[1] if " - " in usage else usage
        summary = " ".join(summary.split("\n")[0].split())
        lines = [escape(summary), "", f"The SQL function `{sql}`:", "", "```text", usage]
        for part in ("arguments", "note"):
            if part in entry:
                lines += ["", entry[part]]
        if "since" in entry:
            lines += ["", f"Since: {entry['since']}"]
        lines += ["```"]
        if "examples" in entry:
            lines += ["", "```sql", entry["examples"], "```"]
    options = [o for o in function.options if not is_seed(o)]
    if options:
        names = " and ".join(f"`{o.name}`" for o in options)
        verb = "is" if len(options) == 1 else "are"
        lines += ["", f"{names} {verb} optional: `None` takes the default of the SQL function."]
    if any(is_seed(o) for o in function.options):
        lines += ["", "`seed` is optional: `None` draws a new random seed for each call."]
    out = []
    for line in "\n".join(lines).expandtabs(4).split("\n"):
        if "```" in line and not line.startswith("```"):
            raise ValueError(f"{function.name}: a fence inside the description")
        out.append(("/// " + line).rstrip())
    return out


GENERATED_HEADER = """\
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

// @generated by scripts/gen_async_functions.py from scripts/async_functions/; do not edit.
"""


def generate(
    rows: list[Row],
    functions: list[CoreFunction],
    forms: dict[str, CoreFunction],
    descriptions: dict,
) -> tuple[str, str]:
    by_name = {f.name: f for f in functions}
    out = [
        GENERATED_HEADER,
        "use spark_connect_core::{function_forms, functions as sql};\n",
        "use super::{lambda1, lambda2, lambda3, literal};",
        "use crate::column::{lit, to_core, Column, IntoArgument};",
        "use crate::options::random_seed;\n",
    ]
    for function in functions_of(rows, functions, forms):
        params = ", ".join(f"{a.name}: {rust_type(a.kind)}" for a in function.arguments)
        sql = sql_name(by_name[function.rows[0].core], by_name)
        out += doc_lines(function, sql, descriptions)
        out.append(
            f"pub fn {rust_name(function.name)}({params}) -> Column {{ {body(function, by_name)} }}\n"
        )
    return rustfmt("\n".join(out)), rustfmt(cases(rows, functions, forms))


def rust_name(name: str) -> str:
    return ("r#" + name) if name in KEYWORDS else name


def old_literal(base: str, value: str) -> str:
    """The old API's column of the literal `value` of the value type `base`."""
    return f"{OLD_LITERALS[base]}({value})" if base in OLD_LITERALS else value


def value_sample(base: str, used: dict[str, int]) -> tuple[str, str]:
    """The next test value of the value type `base`, and the old API's column of it."""
    index = used.get(base, 0)
    used[base] = index + 1
    value = SAMPLE_VALUES[base][index]
    return value, old_literal(base, value)


def sample(kind: str, names, used: dict[str, int]) -> tuple[str, str]:
    """A test value of the parameter type `kind`, and the old API's column of it."""
    base = kind.lstrip("?")
    if base == "col":
        name = next(names)
        return (f'col("{name}")' if kind.startswith("?") else f'"{name}"'), f'oc("{name}")'
    if base == "cols":
        a, b = next(names), next(names)
        return f'["{a}", "{b}"]', f'[oc("{a}"), oc("{b}")]'
    if base == "args":
        return '["x", "y"]', '[ostr("x"), ostr("y")]'
    if base.startswith("arg"):
        return value_sample({"arg": "i32", "arg-str": "str", "arg-float": "f64"}[base], used)
    if base.startswith("fn"):
        variables = ", ".join(["x", "y", "z"][: int(base[2:])])
        closure = f"|{variables}| array([{variables}])"
        return (f"Box::new({closure})" if kind.startswith("?") else closure), (
            "|v: Vec<OldColumn>| old::array(v)"
        )
    return value_sample(base, used)


def case(function: Function, row: Row, core: CoreFunction) -> str:
    """The test case that calls `function` with the options of `row`'s builder."""
    names = iter("abcdefghijklmnopqrstuvwxyz")
    used: dict[str, int] = {}
    given = {a.name for a in row.arguments[: len(core.params)] if a.optional}
    new, old, raw = {}, {}, {}
    for argument in function.arguments:
        if argument.optional and argument.name not in given:
            new[argument.name] = (
                f"Some({function.form.default_seed})" if is_seed(argument) else "None"
            )
            continue
        value, old[argument.name] = sample(argument.kind, names, used)
        new[argument.name] = f"Some({value})" if argument.optional else value
        raw[argument.name] = value
    old_arguments = [
        raw[a.name] if core_type in NATIVE and a.base in SAMPLE_VALUES else old[a.name]
        for a, (_, core_type) in zip(row.arguments, core.params)
    ]
    new_call = ", ".join(new[a.name] for a in function.arguments)
    return (
        f'Case {{ name: "{row.core}", old: || old::{core.path}({", ".join(old_arguments)}), '
        f"new: || {rust_name(function.name)}({new_call}) }},"
    )


def form_case(function: Function, sql: str) -> str:
    """The test case that gives every argument of a function whose core form sends them in
    order through `with_options`, with the expected call written out."""
    names = iter("abcdefghijklmnopqrstuvwxyz")
    used: dict[str, int] = {}
    new, old = [], []
    for argument in function.arguments:
        value, previous = sample(argument.kind, names, used)
        new.append(f"Some({value})" if argument.optional else value)
        old.append(previous)
    return (
        f'Case {{ name: "{function.name} with every argument", '
        f'old: || call("{sql}", vec![{", ".join(old)}]), '
        f"new: || {rust_name(function.name)}({', '.join(new)}) }},"
    )


def cases(rows: list[Row], functions: list[CoreFunction], forms: dict[str, CoreFunction]) -> str:
    by_name = {f.name: f for f in functions}
    out = [GENERATED_HEADER, "use super::*;\n", "pub fn cases() -> Vec<Case> {", "vec!["]
    generated = {f.name: f for f in functions_of(rows, functions, forms)}
    for row in rows:
        if row.form in ("hand", "same"):
            out.append(f'Case {{ name: "{row.core}", old: || {row.old}, new: || {row.new} }},')
        else:
            out.append(case(generated[row.function], row, by_name[row.core]))
    out += ["]", "}", "", "pub fn forms() -> Vec<Case> {", "vec!["]
    for function in generated.values():
        if function.form is not None and "with_options" in function.form.calls:
            out.append(form_case(function, function.form.sql[0]))
    out += ["]", "}", "", "pub fn seeded() -> Vec<(&'static str, fn() -> Column)> {", "vec!["]
    for function in generated.values():
        if any(is_seed(a) for a in function.arguments):
            out.append(seeded_case(function))
    out += ["]", "}"]
    return "\n".join(out) + "\n"


def seeded_case(function: Function) -> str:
    """The call of a seeded function with no option given."""
    names = iter("abcdefghijklmnopqrstuvwxyz")
    used: dict[str, int] = {}
    arguments = [
        "None" if a.optional else sample(a.kind, names, used)[0] for a in function.arguments
    ]
    return f'("{function.name}", || {rust_name(function.name)}({", ".join(arguments)})),'


def rustfmt(text: str) -> str:
    return subprocess.run(
        ["rustfmt", "--edition", "2021", "--config-path", str(REPO / "rustfmt.toml")],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


# --- checking ---------------------------------------------------------------------------------


def hand_written_names(text: str) -> set[str]:
    """The functions that the hand-written module defines or re-exports."""
    names = set(re.findall(r"^pub fn (\w+)", text, re.M))
    for group in re.findall(r"^pub use crate::column::\{([^}]*)\};", text, re.M):
        names |= {name.strip() for name in group.split(",")}
    return names


def problems(
    rows: list[Row],
    functions: list[CoreFunction],
    forms: dict[str, CoreFunction],
    descriptions: dict,
    hand_written: set[str],
) -> list[str]:
    """What makes the table or the descriptions stale for the core builders."""
    out = []
    by_name = {f.name: f for f in functions}
    names = [r.core for r in rows]
    for name in sorted(set(by_name) - set(names)):
        out.append(f"the table has no row for the builder {name}")
    for name in sorted(set(names) - set(by_name)):
        out.append(f"the table's row {name} has no builder")
    for name in sorted({n for n in names if names.count(n) > 1}):
        out.append(f"the table has two rows for {name}")
    generated = functions_of(rows, functions, forms)
    made = {f.name for f in generated}
    takes_form = {f.rows[0].core for f in generated if f.form is not None}
    for name in sorted(made & hand_written):
        out.append(f"{name} is both generated and hand-written")
    for row in rows:
        if row.core not in by_name:
            continue
        core = by_name[row.core]
        expected = forms[row.core].params if row.core in takes_form else core.params
        if row.form in ("derived", "pinned") and len(row.arguments) != len(expected):
            out.append(f"{row.core}: {len(row.arguments)} arguments for {len(expected)} parameters")
        if row.core in takes_form and not all(
            a.optional for a in row.arguments[len(core.params) :]
        ):
            out.append(f"{row.core}: the arguments the builder does not take must be optional")
        if row.form in ("hand", "same") and not (row.old and row.new):
            out.append(f"{row.core}: a {row.form} row needs its old and new calls")
        if row.form == "hand" and row.function not in hand_written:
            out.append(f"{row.core}: {row.function} is not in functions/mod.rs")
        if row.form == "same" and row.function not in made | hand_written:
            out.append(f"{row.core}: no row makes {row.function}")
    wanted = set()
    for function in generated:
        if function.rows[0].core in by_name and not function.rows[0].doc:
            sql = sql_name(by_name[function.rows[0].core], by_name)
            if sql is None:
                out.append(f"{function.name}: no SQL function and no doc")
            else:
                wanted.add(sql)
    present = set(descriptions.get("functions", {}))
    for sql in sorted(wanted - present):
        out.append(f"descriptions.json has no description of {sql}")
    expected = sql_names(functions)
    for sql in sorted(present - expected):
        out.append(f"descriptions.json describes {sql}, which no builder calls")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if an output or input is stale")
    parser.add_argument("--derive", action="store_true", help="rederive the table from pyspark")
    args = parser.parse_args()
    functions = core_functions()
    forms = form_functions()
    rows = read_table()
    if args.derive:
        TABLE.write_text(render_table(derive(rows, functions, forms)))
        rows = read_table()
    descriptions = json.loads(DESCRIPTIONS.read_text())
    found = problems(
        rows, functions, forms, descriptions, hand_written_names(HAND_WRITTEN.read_text())
    )
    if render_table(rows) != TABLE.read_text():
        found.append(f"{TABLE.relative_to(REPO)} is not in the form the generator writes")
    if found:
        print("\n".join(found), file=sys.stderr)
        return 1
    generated, test_cases = generate(rows, functions, forms, descriptions["functions"])
    if args.check:
        stale = [
            p for p, text in ((GENERATED, generated), (CASES, test_cases)) if p.read_text() != text
        ]
        for path in stale:
            print(
                f"{path.relative_to(REPO)} is stale: rerun {Path(__file__).name}", file=sys.stderr
            )
        return 1 if stale else 0
    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    CASES.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(generated)
    CASES.write_text(test_cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
