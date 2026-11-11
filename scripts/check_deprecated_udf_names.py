#!/usr/bin/env python3
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

"""Check which deprecated names of the Rust UDF API warn where they are used.

Builds tests/wasm-udf/tests/deprecated_names.rs, which uses the old names under a narrow
``#[allow(deprecated)]``, with the deprecation lint forced on, and fails unless each item of
OLD_NAMES gets a deprecation warning and no other item does. ``AbiType`` through the old path
gets none: it is a re-export, which can't warn, so that its variants can still be imported
through that path. Cargo can't deprecate the ``wasm-udf`` feature; that crate enables only it,
so its build checks the feature instead.

    python3 scripts/check_deprecated_udf_names.py
"""

import json
import re
import subprocess
import sys

FIXTURE = "tests/wasm-udf/tests/deprecated_names.rs"
BUILD = [
    "cargo",
    "rustc",
    "--manifest-path",
    "tests/wasm-udf/Cargo.toml",
    "--test",
    "deprecated_names",
    "--message-format=json",
    "--",
    "--force-warn",
    "deprecated",
]
OLD_NAMES = {
    "macro `spark_wasm_udf`",
    "module `spark_connect::wasm_udf`",
    "function `spark_connect::wasm_udf::udf`",
    "type alias `spark_connect::wasm_udf::PythonPacker`",
    "type alias `spark_connect::wasm_udf::UdfRegistration`",
    "type alias `spark_connect::wasm_udf::UserDefinedFunction`",
}
SILENT = "spark_connect::wasm_udf::AbiType"
DEPRECATED_ITEM = re.compile(r"use of deprecated (.+? `[^`]+`)")


def deprecated_items(lines):
    """The items that the deprecation warnings among cargo's JSON messages name."""
    items = set()
    for line in lines:
        message = json.loads(line)
        if message.get("reason") != "compiler-message":
            continue
        diagnostic = message["message"]
        if (diagnostic.get("code") or {}).get("code") == "deprecated":
            item = DEPRECATED_ITEM.match(diagnostic["message"])
            items.add(item.group(1) if item else diagnostic["message"])
    return items


def main():
    with open(FIXTURE) as f:
        if f"{SILENT}::" not in f.read():
            print(f"{FIXTURE} no longer imports a variant through {SILENT}")
            return 1
    build = subprocess.run(BUILD, capture_output=True, text=True)
    if build.returncode != 0:
        print(build.stderr, file=sys.stderr)
        return 1
    found = deprecated_items(line for line in build.stdout.splitlines() if line.strip())
    silent = {item for item in found if SILENT in item}
    for item in sorted(OLD_NAMES - found):
        print(f"no deprecation warning: {item}")
    for item in sorted(silent):
        print(
            f"deprecation warning for {SILENT}, which is re-exported to keep its variants: {item}"
        )
    for item in sorted(found - OLD_NAMES - silent):
        print(f"unexpected deprecation warning: {item}")
    print(
        f"{len(found & OLD_NAMES)} of {len(OLD_NAMES)} old names warn; "
        f"{SILENT} {'warns' if silent else 'does not'}",
        file=sys.stderr,
    )
    return 0 if found == OLD_NAMES else 1


if __name__ == "__main__":
    sys.exit(main())
