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
"""Extract the descriptions of Spark's SQL functions for the async crate's function docs.

Reads the `@ExpressionDescription` annotations of a Spark source checkout and the names
`FunctionRegistry` gives their expressions, keeps the SQL functions that the core crate's
function builders call, and writes them to `scripts/async_functions/descriptions.json`, which
`gen_async_functions.py` turns into rustdoc.

The committed file was extracted from Apache Spark v4.3.0, commit
98bc0212c94705434471627068f41e374ccab143:

    python3 scripts/extract_spark_function_descriptions.py --spark-src <checkout>

`--check` compares the committed file with a fresh extraction instead of writing it.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_async_functions  # noqa: E402

SPARK_TAG = "v4.3.0"
SPARK_COMMIT = "98bc0212c94705434471627068f41e374ccab143"
REGISTRY = (
    "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/analysis/FunctionRegistry.scala"
)
SOURCE_DIRS = ("sql", "connector")
FIELDS = ("usage", "arguments", "examples", "note", "since")


def scala_strings(text: str, start: int) -> tuple[dict[str, str], int]:
    """The `name = <string literals joined by +>` arguments of the annotation whose `(` is
    at `start`, and the index after its `)`."""
    args: dict[str, str] = {}
    depth = 0
    i = start
    arg_start = start + 1
    name = None
    value: list[str] = []
    while i < len(text):
        c = text[i]
        if text.startswith('"""', i):
            end = text.index('"""', i + 3)
            while text.startswith('"', end + 3):
                end += 1
            value.append(text[i + 3 : end])
            i = end + 3
            continue
        if c == '"':
            j = i + 1
            chars = []
            while text[j] != '"':
                if text[j] == "\\":
                    escaped = text[j + 1]
                    chars.append({"n": "\n", "t": "\t"}.get(escaped, escaped))
                    j += 2
                else:
                    chars.append(text[j])
                    j += 1
            value.append("".join(chars))
            i = j + 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                if name is not None:
                    args[name] = "".join(value)
                return args, i + 1
        elif c == "," and depth == 1:
            if name is not None:
                args[name] = "".join(value)
            name, value, arg_start = None, [], i + 1
        elif c == "=" and depth == 1 and name is None:
            name = text[arg_start:i].strip()
            value = []
        i += 1
    raise ValueError(f"unclosed annotation at {start}")


def scala_sources(root: Path):
    """The text of each main (not test) Scala source of the checkout, in path order."""
    for top in SOURCE_DIRS:
        for path in sorted((root / top).rglob("*.scala")):
            if "/src/main/" in path.as_posix():
                yield path.read_text()


def annotated_definitions(root: Path) -> dict[str, dict[str, str]]:
    """The annotation of each annotated class or object, by its name."""
    out: dict[str, dict[str, str]] = {}
    for text in scala_sources(root):
        for match in re.finditer(r"@ExpressionDescription\s*\(", text):
            args, end = scala_strings(text, match.end() - 1)
            definition = re.search(r"\b(?:class|object)\s+(\w+)", text[end:])
            if definition is not None:
                out.setdefault(definition.group(1), args)
    return out


def registered_names(root: Path) -> dict[str, str]:
    """The class or builder object behind each SQL function name of the registry."""
    text = (root / REGISTRY).read_text()
    names: dict[str, str] = {}
    for match in re.finditer(r'\b(?:expression|generator)\[(\w+)\]\(\s*"([^"]+)"', text):
        names.setdefault(match.group(2), match.group(1))
    for match in re.finditer(
        r'\b(?:expressionBuilder|generatorBuilder|expressionGeneratorBuilderOuter)\(\s*"([^"]+)"\s*,\s*(\w+)',
        text,
    ):
        names.setdefault(match.group(1), match.group(2))
    return names


def pretty_names(root: Path) -> dict[str, str]:
    """The class of each `prettyName` outside the registry, such as the Avro and Protobuf
    expressions."""
    out: dict[str, str] = {}
    for text in scala_sources(root):
        for match in re.finditer(r'def prettyName: String = "([^"]+)"', text):
            owner = re.findall(r"\b(?:class|object)\s+(\w+)", text[: match.start()])
            if owner:
                out.setdefault(match.group(1), owner[-1])
    return out


def clean(text: str) -> str:
    lines = [line.rstrip() for line in text.strip("\n").split("\n")]
    indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
    return "\n".join(line[indent:] for line in lines).strip()


def extract(root: Path, wanted: set[str]) -> dict:
    definitions = annotated_definitions(root)
    registry = registered_names(root)
    pretty = pretty_names(root)
    functions = {}
    for name in sorted(wanted):
        owner = registry.get(name) or pretty.get(name)
        args = definitions.get(owner or "")
        if not args:
            continue
        entry = {}
        for field in FIELDS:
            value = clean(args.get(field, "")).replace("_FUNC_", name)
            if value:
                entry[field] = value
        functions[name] = entry
    return {
        "source": f"Apache Spark {SPARK_TAG}",
        "commit": SPARK_COMMIT,
        "functions": functions,
    }


def render(data: dict) -> str:
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spark-src", required=True, type=Path, help="a Spark checkout")
    parser.add_argument("--check", action="store_true", help="fail if the committed file differs")
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "-C", str(args.spark_src), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != SPARK_COMMIT:
        print(f"{args.spark_src} is at {head}, not {SPARK_TAG} ({SPARK_COMMIT})", file=sys.stderr)
        return 1
    wanted = gen_async_functions.sql_names(gen_async_functions.core_functions())
    text = render(extract(args.spark_src, wanted))
    target = gen_async_functions.DESCRIPTIONS
    if args.check:
        if target.read_text() != text:
            print(f"{target} is stale: rerun {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    target.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
