#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "gen_async_functions", SCRIPTS / "gen_async_functions.py"
)
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)

CORE = """
fn func(name: &str, args: Vec<Expression>) -> Column {
    Column::new(Expression::function(name, args))
}

/// Docs.
pub fn abs(col1: Column) -> Column {
    func("abs", vec![col1.expression().clone()])
}

pub fn concat<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    func("concat", vec![])
}

pub fn r#struct<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    func("struct", vec![])
}

pub fn greatest(col1: Column, col2: Column) -> Column {
    function_forms::greatest([col1, col2])
}

pub fn transform<F>(col: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(1, f);
    func("transform", vec![])
}

pub fn alias_of_abs(col1: Column) -> Column {
    abs(col1)
}

pub fn round(col1: Column) -> Column {
    func("round", vec![])
}

pub fn round_scale(col1: Column, scale: Column) -> Column {
    func("round", vec![])
}
"""
FORMS = """
pub fn greatest(cols: impl IntoIterator<Item = Column>) -> Column {
    func("greatest", expressions(cols))
}

pub fn abs(col: Column, scale: Option<Column>, mode: Option<Column>) -> Column {
    with_options("abs", [col], [(scale, Absent::Omitted), (mode, Absent::Omitted)])
}

pub fn alias_of_abs<F>(col: Column, f: Option<F>) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    func("abs", vec![])
}
"""


def row(core, function, arguments, form="derived"):
    return gen.Row(core, function, form, gen.parse_arguments(arguments))


class CoreParsingTests(unittest.TestCase):
    def test_builders_are_read_with_their_parameters_and_sql_functions(self):
        functions = {f.name: f for f in gen.core_functions(CORE, FORMS)}
        self.assertNotIn("func", functions)
        expected = {
            "abs": ([("col1", "Column")], "abs"),
            "concat": ([("cols", "impl IntoIterator<Item = C>")], "concat"),
            "struct": ([("cols", "impl IntoIterator<Item = C>")], "struct"),
            "greatest": ([("col1", "Column"), ("col2", "Column")], "greatest"),
            "transform": ([("col", "Column"), ("f", "F")], "transform"),
            "alias_of_abs": ([("col1", "Column")], "abs"),
        }
        for name, (params, sql) in expected.items():
            with self.subTest(name):
                self.assertEqual(functions[name].params, params)
                self.assertEqual(gen.sql_name(functions[name], functions), sql)
        self.assertTrue(functions["struct"].raw)
        self.assertEqual(functions["struct"].path, "r#struct")
        self.assertEqual(functions["transform"].lambdas, [1])


class DerivedKindTests(unittest.TestCase):
    def test_pyspark_annotations_give_the_parameter_types(self):
        cases = [
            ("ColumnOrName", "Column", False, "col"),
            ("pyspark.sql.column.Column", "Column", False, "col"),
            ("Union[ForwardRef('ColumnOrName'), int]", "Column", False, "col"),
            ("Union[pyspark.sql.column.Column, int]", "Column", False, "arg"),
            ("Union[pyspark.sql.column.Column, str]", "Column", False, "arg-str"),
            (
                "Union[pyspark.sql.column.Column, float, Sequence[float]]",
                "Column",
                False,
                "arg-float",
            ),
            ("Any", "Column", False, "arg"),
            ("str", "Column", False, "str"),
            ("int", "Column", False, "i32"),
            ("bool", "Column", False, "bool"),
            ("Union[pyspark.sql.column.Column, int, NoneType]", "Column", True, "?i32"),
            ("Optional[ForwardRef('ColumnOrName')]", "Column", True, "?col"),
            ("str", "&str", False, "str"),
            ("int", "i32", False, "i32"),
            ("ColumnOrName", "impl IntoIterator<Item = C>", False, "cols"),
            ("Union[pyspark.sql.column.Column, str]", "impl IntoIterator<Item = C>", False, "args"),
        ]
        for annotation, core_type, optional, kind in cases:
            with self.subTest(annotation=annotation, core_type=core_type):
                self.assertEqual(gen.derived_kind(annotation, core_type, optional), kind)
        self.assertEqual(gen.derived_kind("Callable", "F", False, 2), "fn2")
        self.assertEqual(gen.derived_kind("Optional[Mapping[str, str]]", "Column", True), "?col")
        comparator = (
            "Optional[Callable[[pyspark.sql.column.Column, pyspark.sql.column.Column], Any]]"
        )
        self.assertEqual(gen.derived_kind(comparator, "F", True), "?fn2")
        self.assertEqual(gen.option_type("Option<Column>"), "Column")
        self.assertEqual(gen.option_type("Option<&str>"), "&str")

    def test_an_annotation_without_a_rule_is_an_error(self):
        with self.assertRaises(ValueError):
            gen.derived_kind("Iterable[int]", "Column", False)

    def test_names_are_snake_case_and_not_keywords(self):
        self.assertEqual(gen.snake("ignoreNulls"), "ignore_nulls")
        self.assertEqual(gen.snake("type"), "type_")


class TableTests(unittest.TestCase):
    def test_a_table_reads_back_as_it_was_written(self):
        rows = [
            row("abs", "abs", "col:col"),
            row("instr_with_start", "instr", "str:col, substr:arg-str, start:?i32", "pinned"),
            gen.Row("concat_ws", "concat_ws", "hand", [], "", "old()", "new()"),
        ]
        self.assertEqual(gen.read_table(gen.render_table(rows)), rows)


class GeneratingTests(unittest.TestCase):
    def setUp(self):
        self.by_name = {f.name: f for f in gen.core_functions(CORE, FORMS)}

    def test_options_call_the_builder_that_takes_them(self):
        (function,) = gen.functions_of(
            [row("round", "round", "col:col"), row("round_scale", "round", "col:col, scale:?i32")]
        )
        self.assertEqual(
            gen.body(function, self.by_name),
            "Column::from(match scale { "
            "None => sql::round(to_core(col)), "
            "Some(scale) => sql::round_scale(to_core(col), to_core(lit(scale))), })",
        )

    def test_the_options_given_choose_the_builder_that_takes_exactly_them(self):
        rows = [
            row("a", "f", "x:col"),
            row("b", "f", "x:col, start:?i32"),
            row("c", "f", "x:col, start:?i32, count:?i32"),
        ]
        (function,) = gen.functions_of(rows)
        for given, core in [(set(), "a"), ({"start"}, "b"), ({"start", "count"}, "c")]:
            with self.subTest(given=sorted(given)):
                self.assertEqual(gen.variant_for(function, given).core, core)
        with self.assertRaises(ValueError):
            gen.variant_for(function, {"count"})

    def test_arguments_the_builder_does_not_take_go_to_the_core_form(self):
        functions = gen.core_functions(CORE, FORMS)
        forms = gen.form_functions(FORMS)
        (function,) = gen.functions_of(
            [row("abs", "abs", "col:col, scale:?i32, mode:?col")], functions, forms
        )
        self.assertEqual(
            gen.body(function, self.by_name),
            "Column::from(function_forms::abs("
            "to_core(col), scale.map(literal), mode.map(to_core)))",
        )
        (closure,) = gen.functions_of(
            [row("alias_of_abs", "alias_of_abs", "col:col, f:?fn1")], functions, forms
        )
        self.assertIn("f.map(lambda1)", gen.body(closure, self.by_name))
        self.assertIn("with_options", forms["abs"].calls)
        self.assertNotIn("with_options", forms["alias_of_abs"].calls)
        self.assertEqual(forms["abs"].sql, ["abs"])
        self.assertEqual(
            gen.form_case(function, "abs"),
            'Case { name: "abs with every argument", old: || call("abs", vec![oc("a"), oint(3), '
            'oc("b")]), new: || abs("a", Some(3), Some(col("b"))) },',
        )

    def test_parameter_types_become_rust_types(self):
        cases = {
            "col": "impl Into<Column>",
            "?col": "Option<Column>",
            "cols": "impl IntoIterator<Item = impl Into<Column>>",
            "args": "impl IntoIterator<Item = impl IntoArgument>",
            "arg-str": "impl IntoArgument",
            "str": "&str",
            "?i32": "Option<i32>",
            "bytes": "Vec<u8>",
            "fn2": "impl FnOnce(Column, Column) -> Column",
            "?fn1": "Option<Box<dyn FnOnce(Column) -> Column>>",
        }
        for kind, rust in cases.items():
            with self.subTest(kind):
                self.assertEqual(gen.rust_type(kind), rust)

    def test_a_literal_parameter_is_passed_as_a_literal_unless_core_takes_the_value(self):
        self.assertEqual(gen.core_argument("p", "str", "Column"), "to_core(lit(p))")
        self.assertEqual(gen.core_argument("p", "str", "&str"), "p")
        self.assertEqual(gen.core_argument("p", "arg", "Column"), "to_core(p.into_argument())")

    def test_descriptions_become_docs_with_brackets_escaped(self):
        (function,) = gen.functions_of([row("abs", "abs", "col:col")])
        descriptions = {
            "abs": {
                "usage": "abs(expr) - Returns the [absolute] value of <expr>.",
                "examples": "Examples:\n  > SELECT abs(-1);\n   1",
            }
        }
        lines = gen.doc_lines(function, "abs", descriptions)
        self.assertEqual(lines[0], "/// Returns the \\[absolute\\] value of \\<expr\\>.")
        self.assertIn("/// ```sql", lines)

    def test_a_fence_inside_a_description_is_an_error(self):
        (function,) = gen.functions_of([row("abs", "abs", "col:col")])
        with self.assertRaises(ValueError):
            gen.doc_lines(function, "abs", {"abs": {"usage": "abs(x) - see ```code```"}})


class StalenessTests(unittest.TestCase):
    def test_a_table_or_descriptions_out_of_step_with_the_builders_are_reported(self):
        functions = gen.core_functions(CORE, FORMS)
        complete = [
            row("abs", "abs", "col:col"),
            row("concat", "concat", "cols:cols"),
            row("struct", "struct", "cols:cols"),
            gen.Row("greatest", "greatest", "hand", [], "", "o()", "n()"),
            row("transform", "transform", "col:col, f:fn1"),
            gen.Row("alias_of_abs", "abs", "same", [], "", "o()", "n()"),
            row("round", "round", "col:col"),
            row("round_scale", "round", "col:col, scale:?i32"),
        ]
        names = ["abs", "concat", "struct", "greatest", "transform", "round"]
        descriptions = {"functions": {name: {"usage": name} for name in names}}
        hand_written = {"greatest", "col", "lit", "when"}
        forms = gen.form_functions(FORMS)
        self.assertEqual(gen.problems(complete, functions, forms, descriptions, hand_written), [])
        missing_description = {"functions": {n: {"usage": n} for n in names if n != "concat"}}
        cases = [
            ("a builder without a row", complete[1:], descriptions, "no row for the builder abs"),
            (
                "a row without a builder",
                complete + [row("gone", "gone", "col:col")],
                descriptions,
                "gone has no builder",
            ),
            (
                "an argument count that differs",
                [complete[0], row("concat", "concat", "cols:cols, x:col")] + complete[2:],
                descriptions,
                "2 arguments for 1 parameters",
            ),
            (
                "a form argument that is not optional",
                [row("abs", "abs", "col:col, scale:i32, mode:?col")] + complete[1:],
                descriptions,
                "the arguments the builder does not take must be optional",
            ),
            (
                "a form with other arguments",
                [row("abs", "abs", "col:col, scale:?i32")] + complete[1:],
                descriptions,
                "2 arguments for 3 parameters",
            ),
            (
                "a missing description",
                complete,
                missing_description,
                "no description of concat",
            ),
            (
                "a description of a removed builder",
                complete,
                {"functions": {**descriptions["functions"], "gone": {"usage": "gone"}}},
                "describes gone",
            ),
        ]
        for name, rows, data, problem in cases:
            with self.subTest(name):
                found = gen.problems(rows, functions, forms, data, hand_written)
                self.assertTrue(any(problem in p for p in found), found)
        names_cases = [
            (
                "a hand row without its function",
                {"col", "lit", "when"},
                "is not in functions/mod.rs",
            ),
            (
                "a generated function also written by hand",
                hand_written | {"abs"},
                "abs is both generated and hand-written",
            ),
        ]
        for name, names, problem in names_cases:
            with self.subTest(name):
                found = gen.problems(complete, functions, forms, descriptions, names)
                self.assertTrue(any(problem in p for p in found), found)

    def test_the_hand_written_names_are_its_functions_and_column_re_exports(self):
        text = "pub use crate::column::{col, lit, when};\npub fn concat_ws() {}\nfn private() {}\n"
        self.assertEqual(gen.hand_written_names(text), {"col", "lit", "when", "concat_ws"})


if __name__ == "__main__":
    unittest.main()
