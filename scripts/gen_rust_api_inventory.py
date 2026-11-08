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

"""List the public API of a crate from its rustdoc JSON.

Prints one line per public free function, inherent method, and method of a
hand-written trait impl, as ``<path>\t<kind>``, where ``<path>`` is the
item's defining path (for example ``spark_connect::dataframe::DataFrame::select``).
Items re-exported from other crates are listed as ``<path>\treexport``.

Generate the JSON with a nightly toolchain:

    cargo +nightly rustdoc -p apache-spark-connect --lib --all-features -- \
        -Z unstable-options --output-format json --document-hidden-items
    python3 scripts/gen_rust_api_inventory.py target/doc/spark_connect.json

tests/golden/rust_api_inventory.tsv is this inventory of the API of commit
6196e648f54fd15fd346700af0e4422e85f079c6, before its modules moved into
apache-spark-connect-core. Regenerate it from a copy of that commit:

    git archive 6196e648f54fd15fd346700af0e4422e85f079c6 | tar -x -C <dir>
    (cd <dir> && cargo +nightly rustdoc -p apache-spark-connect --lib --all-features -- \
        -Z unstable-options --output-format json --document-hidden-items)
    python3 scripts/gen_rust_api_inventory.py <dir>/target/doc/spark_connect.json \
        > tests/golden/rust_api_inventory.tsv
"""

import argparse
import json
import sys


def _inner_kind(item):
    return next(iter(item["inner"]))


class Inventory:
    def __init__(self, doc):
        self.index = doc["index"]
        self.paths = doc["paths"]
        self.crate_name = self.index[str(doc["root"])]["name"]
        self.lines = {}
        self.seen = set()

    def item(self, item_id):
        return self.index.get(str(item_id))

    def add(self, path, kind):
        self.lines[path] = kind

    def defining_path(self, item, fallback):
        known = self.paths.get(str(item["id"]))
        return "::".join(known["path"]) if known else fallback

    def walk_module(self, module, prefix):
        for child_id in module["inner"]["module"]["items"]:
            child = self.item(child_id)
            if child is None or child["visibility"] != "public":
                continue
            self.walk_item(child, prefix)

    def walk_item(self, item, prefix):
        kind = _inner_kind(item)
        if kind == "use":
            use = item["inner"]["use"]
            target = self.item(use["id"]) if use["id"] is not None else None
            if use["is_glob"]:
                if target is not None and _inner_kind(target) == "module":
                    self.walk_module(target, prefix)
                return
            name = use["name"]
            if target is None or target["crate_id"] != 0:
                self.add(f"{prefix}::{name}", "reexport")
                return
            self.walk_named(target, f"{prefix}::{name}")
            return
        if item["name"] is None:
            return
        self.walk_named(item, f"{prefix}::{item['name']}")

    def walk_named(self, item, path):
        kind = _inner_kind(item)
        if kind != "module":
            path = self.defining_path(item, path)
        if item["id"] in self.seen:
            return
        self.seen.add(item["id"])
        if kind == "module":
            self.walk_module(item, path)
        elif kind == "function":
            self.add(path, "fn")
        elif kind in ("struct", "enum", "union", "trait"):
            self.add(path, kind)
            if kind != "trait":
                self.walk_impls(item["inner"][kind]["impls"], path)
        elif kind in ("constant", "static", "type_alias", "macro"):
            self.add(path, kind)

    def walk_impls(self, impl_ids, type_path):
        for impl_id in impl_ids:
            imp = self.item(impl_id)
            if imp is None:
                continue
            inner = imp["inner"]["impl"]
            if inner["is_synthetic"] or inner["blanket_impl"] is not None:
                continue
            if any("automatically_derived" in str(attr) for attr in imp["attrs"]):
                continue
            trait = inner["trait"]
            for method_id in inner["items"]:
                method = self.item(method_id)
                if method is None or _inner_kind(method) != "function":
                    continue
                if trait is None:
                    if method["visibility"] == "public":
                        self.add(f"{type_path}::{method['name']}", "method")
                else:
                    trait_path = trait["path"]
                    self.add(f"{type_path}::<{trait_path}>::{method['name']}", "trait-method")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", help="rustdoc JSON of the crate")
    args = parser.parse_args(argv)
    with open(args.json) as f:
        doc = json.load(f)
    inventory = Inventory(doc)
    root = inventory.item(doc["root"])
    inventory.walk_module(root, inventory.crate_name)
    for path in sorted(inventory.lines):
        print(f"{path}\t{inventory.lines[path]}")


if __name__ == "__main__":
    main(sys.argv[1:])
