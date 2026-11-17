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

"""Generate the request-snapshot cases of the ML classes of the Rust API.

Reads the rustdoc JSON of apache-spark-connect (made as scripts/gen_rust_api_inventory.py
says) and writes crates/spark-connect/tests/request_snapshots/cases/ml_generated.rs:

- ``<Estimator>::fit/every_setter``: the estimator with each of its setters called, its fit,
  then its model with each of its setters called, each member that the model reads from the
  server, and its transform;
- ``<Transformer>::transform/every_setter`` and ``<Evaluator>::evaluate/every_setter``: the
  stage with each of its setters called, and its transform or evaluation;
- ``<Class>::new/every_setter``: another class of a constructor and of members that return a
  ``Result``, with each of its setters called and each of those members;
- ``<Class>::load``: the load of the class from a path;
- ``<Model>::load/every_setter``: a model that no fit case makes, loaded, then with each of its
  setters called, each of its members and its transform.

A member that returns a summary has each member of the summary read too. A stage that a setter
takes is a new one whose uid is masked, and parallelism is 1, so that the requests of a validator
come in one order. A stage of a uid setter has its uid set, which its requests carry, in place of
the masked one. A validator's fit drops the model of each fold, whose delete the client sends in
the background, so the case of a validator records its requests unordered.

A setter or member with a parameter type the generator cannot fill is listed on stderr.
"""

import argparse
import json
import sys
from pathlib import Path

OUTPUT = "crates/spark-connect/tests/request_snapshots/cases/ml_generated.rs"
PATH = "/tmp/spark-connect-snapshots/ml"

LICENSE = """\
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
"""

VALUES = {
    "&str": '"x"',
    "f64": "0.5",
    "i32": "3",
    "i64": "3",
    "usize": "3",
    "bool": "true",
    "impl IntoIterator<Item=impl Into<String>>": '["a", "b"]',
    "impl IntoIterator<Item=f64>": "[0.5, 0.25]",
    "impl IntoIterator<Item=i64>": "[1, 2]",
    "impl IntoIterator<Item=i32>": "[1, 2]",
    "impl IntoIterator<Item=impl IntoIterator<Item=f64>>": "[[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]",
    "impl Into<MlVector>": "vec![0.5, 0.25]",
    "&MlVector": "&ml::MlVector::from(vec![0.5, 0.25])",
    "impl Into<MlMatrix>": (
        "ml::MlMatrix::Dense { num_rows: 1, num_cols: 2, values: vec![0.5, 0.25], is_transposed: false }"
    ),
    "impl Into<Classifier>": "classifier",
    "impl EstimatorStage": "nested_estimator",
    "impl EvaluatorStage": "nested_evaluator",
    "Vec<ParamMap>": "vec![ml::ParamMap::new()]",
    "&DataFrame": "&range(c)",
    "Option<&DataFrame>": "Some(&range(c))",
}


# The values that a case builds first, as a stage whose new uid it masks.
PREAMBLES = {
    "impl Into<Classifier>": [
        "let classifier = ml::LogisticRegression::new();",
        "c.mask(&Estimator::operator(&classifier).uid);",
    ],
    "impl EstimatorStage": [
        "let nested_estimator = ml::LogisticRegression::new();",
        "c.mask(&Estimator::operator(&nested_estimator).uid);",
    ],
    "impl EvaluatorStage": [
        "let nested_evaluator = ml::BinaryClassificationEvaluator::new();",
        "c.mask(&Evaluator::operator(&nested_evaluator).uid);",
    ],
}
SETTER_VALUES = {"set_parallelism": "1"}


def type_name(t):
    if t is None:
        return "()"
    kind, value = next(iter(t.items()))
    if kind in ("primitive", "generic"):
        return value
    if kind == "borrowed_ref":
        return "&" + ("mut " if value.get("is_mutable") else "") + type_name(value["type"])
    if kind == "resolved_path":
        name = (value.get("path") or value.get("name")).split("::")[-1]
        args = (value.get("args") or {}).get("angle_bracketed", {}).get("args", [])
        parts = [type_name(a["type"]) for a in args if "type" in a]
        return name + ("<" + ", ".join(parts) + ">" if parts else "")
    if kind == "slice":
        return "[" + type_name(value) + "]"
    if kind == "impl_trait":
        bounds = []
        for bound in value:
            trait = (bound.get("trait_bound") or {}).get("trait")
            if not trait:
                continue
            name = (trait.get("path") or trait.get("name")).split("::")[-1]
            args = (trait.get("args") or {}).get("angle_bracketed", {})
            parts = [type_name(a["type"]) for a in args.get("args", []) if "type" in a]
            parts += [
                c["name"] + "=" + type_name(c["binding"]["equality"]["type"])
                for c in args.get("constraints", [])
                if "equality" in c.get("binding", {})
            ]
            bounds.append(name + ("<" + ", ".join(parts) + ">" if parts else ""))
        return "impl " + "+".join(bounds)
    return kind


class Api:
    def __init__(self, doc):
        self.index = doc["index"]
        self.paths = doc["paths"]
        self.types = {}  # name -> {"methods": {name: (self, [arg types], output)}, "traits": set()}
        for item in self.index.values():
            impl = item.get("inner", {}).get("impl")
            if not impl or impl.get("blanket_impl") is not None:
                continue
            target = impl["for"].get("resolved_path")
            if not target or str(target["id"]) not in self.paths:
                continue
            path = self.paths[str(target["id"])]["path"]
            if path[:2] != ["spark_connect", "ml"]:
                continue
            entry = self.types.setdefault(path[-1], {"methods": {}, "traits": set()})
            if impl.get("trait") is not None:
                entry["traits"].add(
                    (impl["trait"].get("path") or impl["trait"]["name"]).split("::")[-1]
                )
                continue
            for method_id in impl["items"]:
                method = self.index.get(str(method_id))
                if not method or "function" not in method.get("inner", {}):
                    continue
                if method.get("visibility") != "public" or method.get("deprecation"):
                    continue
                sig = method["inner"]["function"]["sig"]
                inputs = sig["inputs"]
                receiver = "static"
                if inputs and inputs[0][0] == "self":
                    receiver = type_name(inputs[0][1])
                    inputs = inputs[1:]
                args = [type_name(t) for _, t in inputs]
                entry["methods"][method["name"]] = (receiver, args, type_name(sig.get("output")))

    def methods(self, name):
        return self.types.get(name, {"methods": {}})["methods"]

    def has_trait(self, name, trait):
        return trait in self.types.get(name, {"traits": set()})["traits"]


def call_args(args):
    values = [VALUES.get(a) for a in args]
    return None if any(v is None for v in values) else ", ".join(values)


def preambles(api, name):
    lines = []
    for method, (receiver, args, output) in sorted(api.methods(name).items()):
        if method.startswith("set_") and receiver == "Self" and output == "Self" and len(args) == 1:
            lines += [line for line in PREAMBLES.get(args[0], []) if line not in lines]
    return ["            " + line for line in lines]


def setters(api, name, skipped):
    calls = []
    for method, (receiver, args, output) in sorted(api.methods(name).items()):
        if not method.startswith("set_") or receiver != "Self" or output != "Self":
            continue
        values = SETTER_VALUES.get(method) or (call_args(args) if len(args) == 1 else None)
        if values is None:
            skipped.append(f"{name}::{method}({', '.join(args)})")
            continue
        calls.append(f".{method}({values})")
    return calls


def members(api, name, skipped, var="model", receivers=("&Self",)):
    """The members that read from the server: the methods that return a `Result`, other than a
    writer, and for each that returns a summary with members of its own, those members."""
    calls = []
    for method, (receiver, args, output) in sorted(api.methods(name).items()):
        if receiver not in receivers or not output.startswith("Result") or method in ("write",):
            continue
        values = call_args(args)
        if values is None:
            skipped.append(f"{name}::{method}({', '.join(args)})")
            continue
        calls.append(f'outcome(c, "{method}", {var}.{method}({values}));')
        returned = output[len("Result<") : -1]
        nested = members(api, returned, skipped, "summary") if returned != name else []
        if nested:
            calls.append(f"if let Ok(summary) = {var}.{method}({values}) {{")
            calls += [
                "    " + call.replace('outcome(c, "', f'outcome(c, "{method}.') for call in nested
            ]
            calls.append("}")
    return calls


def chain(head, calls, indent):
    if not calls:
        return head
    pad = " " * indent
    return head + "".join(f"\n{pad}{call}" for call in calls)


def model_lines(calls):
    if not calls:
        return ["            let mut model = estimator.fit(&range(c))?;"]
    return [
        "            let model = estimator.fit(&range(c))?;",
        "            let mut model = " + chain("model", calls, 16) + ";",
    ]


def fit_case(api, name, model, skipped):
    lines = [
        f'        "{name}::fit/every_setter" => |c| {{',
        *preambles(api, name),
        "            let mut estimator = "
        + chain(f"ml::{name}::new()", setters(api, name, skipped), 16)
        + ";",
        "            c.mask(&Estimator::operator(&estimator).uid);",
        '            outcome(c, "write", Estimator::write(&estimator).session(&c.spark).overwrite().save(PATH));',
        *model_lines(setters(api, model, skipped)),
        "            c.mask(&Transformer::operator(&model).uid);",
    ]
    lines += ["            " + call for call in members(api, model, skipped)]
    lines += [
        "            c.df(&Transformer::transform(&mut model, &range(c))?)",
        "        },",
    ]
    return lines


def has_uid_setter(api, name):
    return api.methods(name).get("set_uid") == ("&mut Self", ["&str"], "()")


def stage_case(name, action, calls, uid_setter):
    trait = "Transformer" if action == "transform" else "Evaluator"
    mutable = "mut " if action == "transform" or uid_setter else ""
    body = [
        "            let " + mutable + "stage = " + chain(f"ml::{name}::new()", calls, 16) + ";"
    ]
    if uid_setter:
        body.append(f"            stage.set_uid({VALUES['&str']});")
    else:
        body.append(f"            c.mask(&{trait}::operator(&stage).uid);")
    if action == "transform":
        body.append("            c.df(&Transformer::transform(&mut stage, &range(c))?)")
    else:
        body.append("            Evaluator::evaluate(&stage, &range(c)).map(drop)")
    return [f'        "{name}::{action}/every_setter" => |c| {{', *body, "        },"]


def object_case(api, name, skipped):
    calls = members(api, name, skipped, "object", ("&Self", "&mut Self"))
    if not calls:
        return []
    lines = [f'        "{name}::new/every_setter" => |c| {{', *preambles(api, name)]
    if "fit_validator" in api.methods(name):
        lines.append("            c.unordered();")
    lines.append(
        "            let mut object = "
        + chain(f"ml::{name}::new()", setters(api, name, skipped), 16)
        + ";"
    )
    if api.methods(name).get("uid", (None, None, None))[:2] == ("&Self", []):
        lines.append("            c.mask(object.uid());")
    lines += ["            " + call for call in calls]
    lines += ["            let _ = &mut object;", "            Ok(())", "        },"]
    return lines


def generate(api):
    skipped, cases = [], []
    for name in sorted(api.types):
        methods = api.methods(name)
        new = methods.get("new")
        if new is None or new[0] != "static" or new[1]:
            continue
        fit = methods.get("fit")
        if (
            fit
            and fit[0] == "&mut Self"
            and fit[1] == ["&DataFrame"]
            and fit[2].startswith("Result<")
        ):
            model = fit[2][len("Result<") : -1]
            if api.has_trait(name, "Estimator") and api.has_trait(model, "Transformer"):
                cases += fit_case(api, name, model, skipped)
                continue
        uid_setter = has_uid_setter(api, name)
        if api.has_trait(name, "Transformer") and not api.has_trait(name, "Estimator"):
            cases += stage_case(name, "transform", setters(api, name, skipped), uid_setter)
        elif api.has_trait(name, "Evaluator"):
            cases += stage_case(name, "evaluate", setters(api, name, skipped), uid_setter)
        else:
            cases += object_case(api, name, skipped)
    made = {
        api.methods(name)["fit"][2][len("Result<") : -1]
        for name in api.types
        if "fit" in api.methods(name) and api.methods(name)["fit"][2].startswith("Result<")
    }
    for name in sorted(api.types):
        load = api.methods(name).get("load")
        if load and load[0] == "static" and api.has_trait(name, "Model") and name not in made:
            cases += [
                f'        "{name}::load/every_setter" => |c| {{',
                f"            let model = ml::{name}::load(&c.spark, PATH)?;",
                "            let mut model = "
                + chain("model", setters(api, name, skipped), 16)
                + ";",
                "            c.mask(&Transformer::operator(&model).uid);",
                *["            " + call for call in members(api, name, skipped)],
                "            c.df(&Transformer::transform(&mut model, &range(c))?)",
                "        },",
            ]
    for name in sorted(api.types):
        load = api.methods(name).get("load")
        if load and load[0] == "static" and load[1] == ["&SparkSession", "&str"]:
            cases.append(
                f'        "{name}::load" => |c| ml::{name}::load(&c.spark, PATH).map(drop),'
            )
    text = (
        f"{LICENSE}\n"
        "// Generated by scripts/gen_request_snapshot_ml.py. Do not edit.\n"
        "\n"
        "use spark_connect::ml::{self, Estimator, Evaluator, Transformer};\n"
        "\n"
        "use super::ml::outcome;\n"
        "use super::range;\n"
        "use crate::Case;\n"
        "\n"
        f'const PATH: &str = "{PATH}";\n'
        "\n"
        "pub fn cases() -> Vec<Case> {\n"
        "    cases![\n" + "\n".join(cases) + "\n    ]\n}\n"
    )
    return text, skipped


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json", help="rustdoc JSON of apache-spark-connect (target/doc/spark_connect.json)"
    )
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args(argv)
    text, skipped = generate(Api(json.loads(Path(args.json).read_text())))
    Path(args.output).write_text(text)
    if skipped:
        print("not generated: " + ", ".join(skipped), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
