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
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "extract_spark_function_descriptions", SCRIPTS / "extract_spark_function_descriptions.py"
)
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)

EXPRESSIONS = '''
@ExpressionDescription(
  usage = "_FUNC_(expr) - Returns the absolute value, " +
    "as \\"abs\\" does (in SQL).",
  examples = """
    Examples:
      > SELECT _FUNC_(-1);
       1
  """,
  note = """Raw text, """ + """joined.""",
  since = "1.2.0",
  group = "math_funcs")
case class Abs(child: Expression) extends UnaryExpression

@ExpressionDescription(
  usage = "_FUNC_(expr) - Rounds up.",
  since = "3.3.0")
object CeilExpressionBuilder extends ExpressionBuilder
'''
REGISTRY = """
    expression[Abs]("abs"),
    expressionBuilder("ceil", CeilExpressionBuilder),
    expressionBuilder("ceiling", CeilExpressionBuilder, true),
"""
AVRO = """
@ExpressionDescription(usage = "_FUNC_(data) - Reads Avro.")
case class AvroDataToCatalyst(child: Expression) {
  override def prettyName: String = "from_avro"
}
"""


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        main = root / "sql/catalyst/src/main/scala"
        main.mkdir(parents=True)
        (main / "expressions.scala").write_text(EXPRESSIONS)
        registry = root / extract.REGISTRY
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(REGISTRY)
        avro = root / "connector/avro/src/main/scala"
        avro.mkdir(parents=True)
        (avro / "avro.scala").write_text(AVRO)
        test = root / "sql/core/src/test/scala"
        test.mkdir(parents=True)
        (test / "test.scala").write_text(
            '@ExpressionDescription(usage = "test only")\ncase class Abs(x: Int)\n'
        )
        self.root = root

    def tearDown(self):
        self.directory.cleanup()

    def test_annotation_strings_are_joined_and_unescaped(self):
        args, _ = extract.scala_strings(EXPRESSIONS, EXPRESSIONS.index("("))
        self.assertEqual(
            args["usage"], '_FUNC_(expr) - Returns the absolute value, as "abs" does (in SQL).'
        )
        self.assertEqual(args["note"], "Raw text, joined.")
        self.assertEqual(args["since"], "1.2.0")
        self.assertIn("> SELECT _FUNC_(-1);", args["examples"])

    def test_each_wanted_function_gets_the_description_of_its_expression(self):
        data = extract.extract(self.root, {"abs", "ceiling", "from_avro", "unknown"})
        functions = data["functions"]
        self.assertEqual(sorted(functions), ["abs", "ceiling", "from_avro"])
        self.assertEqual(
            functions["abs"]["usage"],
            'abs(expr) - Returns the absolute value, as "abs" does (in SQL).',
        )
        self.assertEqual(functions["abs"]["examples"], "Examples:\n  > SELECT abs(-1);\n   1")
        self.assertEqual(functions["ceiling"]["usage"], "ceiling(expr) - Rounds up.")
        self.assertEqual(functions["from_avro"]["usage"], "from_avro(data) - Reads Avro.")
        self.assertEqual(data["commit"], extract.SPARK_COMMIT)


if __name__ == "__main__":
    unittest.main()
