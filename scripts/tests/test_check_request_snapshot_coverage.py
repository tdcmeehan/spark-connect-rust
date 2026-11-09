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

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_request_snapshot_coverage.py"
SPEC = importlib.util.spec_from_file_location("check_request_snapshot_coverage", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

INVENTORY = [
    ("spark_connect::dataframe::DataFrame", "struct"),
    ("spark_connect::dataframe::DataFrame::select", "method"),
    ("spark_connect::dataframe::DataFrame::filter", "method"),
    ("spark_connect::functions::col", "fn"),
    ("spark_connect::row::Row::<Display>::fmt", "trait-method"),
]


class CheckRequestSnapshotCoverageTests(unittest.TestCase):
    def check(self, cases, excluded):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            inventory = tmp / "inventory.tsv"
            inventory.write_text("".join(f"{p}\t{k}\n" for p, k in INVENTORY))
            snapshots = tmp / "snapshots.jsonl"
            snapshots.write_text("".join(json.dumps({"case": c}) + "\n" for c in cases))
            exclusions = tmp / "excluded.tsv"
            exclusions.write_text("".join(f"{p}\t{r}\n" for p, r in excluded))
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = checker.main(
                    [str(inventory), "--snapshots", str(snapshots), "--excluded", str(exclusions)]
                )
            return code, out.getvalue()

    def test_every_function_covered_by_a_case_or_an_exclusion_passes(self):
        cases = ["DataFrame::select/columns", "DataFrame::filter", "functions::col"]
        code, out = self.check(cases, [("*::<Display>::fmt", "formatting")])
        self.assertEqual((code, out), (0, ""))

    def test_a_function_without_a_case_fails_and_is_named(self):
        code, out = self.check(["DataFrame::select"], [("*::<Display>::fmt", "formatting")])
        self.assertEqual(code, 1)
        self.assertIn("no case: DataFrame::filter", out)
        self.assertIn("no case: functions::col", out)

    def test_an_exclusion_naming_a_missing_case_fails(self):
        cases = ["DataFrame::select", "functions::col"]
        excluded = [("DataFrame::filter", "case: DataFrame::where"), ("*::<Display>::fmt", "x")]
        code, out = self.check(cases, excluded)
        self.assertEqual(code, 1)
        self.assertIn("no such case: case: DataFrame::where", out)

    def test_an_exclusion_that_matches_nothing_fails(self):
        cases = ["DataFrame::select", "DataFrame::filter", "functions::col"]
        excluded = [("*::<Display>::fmt", "x"), ("Gone::*", "x")]
        code, out = self.check(cases, excluded)
        self.assertEqual(code, 1)
        self.assertIn("unused exclusion: Gone::*", out)


if __name__ == "__main__":
    unittest.main()
