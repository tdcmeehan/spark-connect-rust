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

"""Check that every public function of the Rust API has a request-snapshot case.

Takes the inventory printed by scripts/gen_rust_api_inventory.py, by default
tests/golden/rust_api_inventory.tsv, the API that the snapshots were captured
from (see that script for how to regenerate it). Run it from the repository root:

    python3 scripts/check_request_snapshot_coverage.py

An item is
covered by a case named after it (``DataFrame::select`` or ``DataFrame::select/variant``
for ``spark_connect::dataframe::DataFrame::select``, ``functions::col`` for the free
function ``spark_connect::functions::col``), or by a pattern in
tests/golden/rust_requests_excluded.tsv, whose lines are ``<pattern>\t<reason>``
with ``*`` matching any text. A reason of the form ``case: <name>`` says the item is
exercised by that case, which must exist. Prints the items that are neither, and
exits nonzero if there are any, if an exclusion matches nothing, or if it names a
case that does not exist.
"""

import argparse
import fnmatch
import json
import sys

INVENTORY = "tests/golden/rust_api_inventory.tsv"
SNAPSHOTS = "tests/golden/rust_requests.jsonl"
EXCLUDED = "tests/golden/rust_requests_excluded.tsv"
KINDS = {"fn", "method", "trait-method"}


def case_key(path, kind):
    parts = path.split("::")[1:]
    # A free function is named by its module, a method by its type.
    return "::".join(parts[-2:] if kind == "fn" else parts[1:])


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", nargs="?", default=INVENTORY)
    parser.add_argument("--snapshots", default=SNAPSHOTS)
    parser.add_argument("--excluded", default=EXCLUDED)
    args = parser.parse_args(argv)

    with open(args.snapshots) as f:
        names = {json.loads(line)["case"] for line in f if line.strip()}
    cases = {name.split("/")[0] for name in names}
    exclusions = []
    with open(args.excluded) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                pattern, reason = line.rstrip("\n").split("\t", 1)
                exclusions.append([pattern, reason, 0])

    uncovered = []
    total = covered = 0
    with open(args.inventory) as f:
        for line in f:
            path, kind = line.rstrip("\n").split("\t")
            if kind not in KINDS:
                continue
            total += 1
            key = case_key(path, kind)
            if key in cases:
                covered += 1
                continue
            matched = [e for e in exclusions if fnmatch.fnmatchcase(key, e[0])]
            for exclusion in matched:
                exclusion[2] += 1
            if not matched:
                uncovered.append(key)

    unused = [e[0] for e in exclusions if e[2] == 0]
    missing = [
        e[1] for e in exclusions if e[1].startswith("case: ") and e[1][len("case: ") :] not in names
    ]
    for key in uncovered:
        print(f"no case: {key}")
    for pattern in unused:
        print(f"unused exclusion: {pattern}")
    for reason in missing:
        print(f"no such case: {reason}")
    print(
        f"{total} functions: {covered} with a case, "
        f"{total - covered - len(uncovered)} excluded, {len(uncovered)} uncovered",
        file=sys.stderr,
    )
    return 1 if uncovered or unused or missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
