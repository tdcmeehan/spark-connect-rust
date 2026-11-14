#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Compile the ``rust`` code snippets in the docs against the crate.

The published guides drifted out of sync with the API (examples that used `lit`
without importing it, or `SparkSessionBuilder` without a `use`), and nothing in CI
caught it because the docs workflow only *deploys*. This type-checks every
```rust``` block so a doc example can't reference a symbol the crate doesn't export.

Each block is wrapped in a small harness that binds ``spark`` and ``df`` (so the
common "continue from an earlier block" fragments type-check) and is run through
``rustc`` against the already-built ``apache-spark-connect`` rlib. Only compilation
(type-checking) is performed - snippets are never executed, so no server is needed.
Emit metadata only so checking each snippet does not link and retain a separate
native executable containing the crate and its dependencies.

A block that must be skipped (pseudo-code, a deliberate compile-error example, or
a snippet that needs a feature/macro the snippet harness doesn't build - e.g. the
`rust-udf` `#[spark_udf]` examples) should be fenced as ```{.rust .no-run}```.
That brace form still renders as a Rust code block in mkdocs/superfences (unlike
```rust,ignore```, whose comma makes superfences drop the fence entirely and print
the backticks literally), and it is not matched by the checker below, so it is not
compiled.

A block of the async API is fenced as ```{.rust .async}```, which renders as Rust
too. It is compiled against ``apache-spark-connect-async`` instead, which the script
builds itself with ``ASYNC_FEATURES``, in a harness that is an ``async fn`` binding
``session``, ``df``, ``df1`` and ``df2`` the same way. It may name the crates in
``ASYNC_EXTERNS``. The script takes their rlibs, and the directories of their
dependencies, from what ``cargo build --message-format=json`` reports, so a block
compiles against the build it describes. An async block that must be skipped is
fenced ```{.rust .no-run}``` like any other.

Usage:
    cargo build -p apache-spark-connect
    python scripts/check_doc_snippets.py
"""

import glob
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# cargo places the primary lib without a hash in target/debug; its dependencies
# (arrow, prost, ...) live in target/debug/deps and are found via -L dependency.
RLIB = REPO / "target" / "debug" / "libspark_connect.rlib"
DEPS = REPO / "target" / "debug" / "deps"

# ```rust``` (checked) vs ```{.rust .no-run}``` (rendered but skipped). We only check
# plain ```rust``` fences; any info string with a comma/space/brace is left alone.
_BLOCK = re.compile(r"^```rust\s*$\n(.*?)^```\s*$", re.DOTALL | re.MULTILINE)

# Harness: bind the identifiers the fragments assume from earlier blocks. `use`
# statements are legal inside a fn body, so a block can carry its own imports; a
# block that re-binds `spark`/`df` just shadows these (harmless).
HARNESS = """#![allow(unused, unused_imports, dead_code, clippy::all)]
fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let spark = spark_connect::SparkSession::builder()
        .remote("sc://localhost:15002")
        .get_or_create()?;
    // The ambient names guide fragments assume from a preceding block.
    let df = spark.range(10)?;
    let df1 = spark.range(10)?;
    let df2 = spark.range(10)?;
    {{
{body}
    }}
    Ok(())
}}
"""


def check_block(code: str, tmpdir: Path, idx: int) -> tuple[bool, str]:
    indented = "\n".join("        " + ln if ln.strip() else ln for ln in code.splitlines())
    src = tmpdir / f"snippet_{idx}.rs"
    src.write_text(HARNESS.format(body=indented))
    out = tmpdir / f"snippet_{idx}.rmeta"
    r = subprocess.run(
        [
            "rustc",
            "--edition",
            "2021",
            "--crate-type",
            "bin",
            "--emit=metadata",
            "-L",
            f"dependency={DEPS}",
            "--extern",
            f"spark_connect={RLIB}",
            "-o",
            str(out),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0, r.stderr


def main() -> int:
    if not RLIB.exists():
        print(f"!! {RLIB} not found - run `cargo build -p apache-spark-connect` first")
        return 2

    failures = []
    checked = 0
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        idx = 0
        for md in sorted(glob.glob(str(REPO / "docs" / "**" / "*.md"), recursive=True)):
            text = Path(md).read_text()
            for m in _BLOCK.finditer(text):
                idx += 1
                checked += 1
                ok, err = check_block(m.group(1), tmpdir, idx)
                if not ok:
                    rel = Path(md).relative_to(REPO)
                    # First rustc error line is the most informative.
                    first = next(
                        (ln for ln in err.splitlines() if ln.startswith("error")), err[:200]
                    )
                    failures.append((str(rel), first))

    print(f"Checked {checked} rust snippet(s); {len(failures)} failed to compile.")
    for rel, err in failures:
        print(f"  {rel}: {err}")
    return max(1 if failures else 0, check_async_blocks())


_ASYNC_BLOCK = re.compile(r"^```\{\.rust \.async\}\s*$\n(.*?)^```\s*$", re.DOTALL | re.MULTILINE)
ASYNC_FEATURES = "blocking,rust-udf"
# Besides the client: tokio for a runtime, futures for streams of results, serde for
# typed rows, chrono for dates and timestamps.
ASYNC_EXTERNS = ("spark_connect_async", "chrono", "futures", "serde", "tokio")
ASYNC_HARNESS = """#![allow(unused, unused_imports, dead_code, clippy::all)]
async fn snippet() -> Result<(), Box<dyn std::error::Error>> {{
    let session = spark_connect_async::Session::builder()
        .remote("sc://localhost:15002")
        .connect()
        .await?;
    // The ambient names guide fragments assume from a preceding block.
    let df = session.range(10);
    let df1 = session.range(10);
    let df2 = session.range(10);
    {{
{body}
    }}
    Ok(())
}}

fn main() {{}}
"""


def async_artifacts() -> tuple[dict[str, Path], list[Path]] | None:
    """The rlibs of ``ASYNC_EXTERNS`` and the directories of all the artifacts, as cargo
    reports them for ``cargo build -p apache-spark-connect-async``, fresh units too."""
    r = subprocess.run(
        [
            "cargo",
            "build",
            "-p",
            "apache-spark-connect-async",
            "--features",
            ASYNC_FEATURES,
            "--message-format=json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr[-4000:])
        return None
    rlibs: dict[str, set[Path]] = {}
    dirs: list[Path] = []
    for line in r.stdout.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("reason") != "compiler-artifact":
            continue
        for name in msg.get("filenames", []):
            path = Path(name)
            if (
                path.suffix in (".rlib", ".rmeta", ".so", ".dylib", ".dll")
                and path.parent not in dirs
            ):
                dirs.append(path.parent)
            if path.suffix == ".rlib":
                rlibs.setdefault(msg["target"]["name"], set()).add(path)
    externs = {}
    for crate in ASYNC_EXTERNS:
        found = rlibs.get(crate, set())
        if len(found) != 1:
            print(f"!! {len(found)} rlibs of {crate} in the build of apache-spark-connect-async")
            return None
        externs[crate] = found.pop()
    return externs, dirs


def check_async_block(
    code: str, tmpdir: Path, idx: int, externs: dict[str, Path], dirs: list[Path]
) -> tuple[bool, str]:
    indented = "\n".join("        " + ln if ln.strip() else ln for ln in code.splitlines())
    src = tmpdir / f"async_snippet_{idx}.rs"
    src.write_text(ASYNC_HARNESS.format(body=indented))
    r = subprocess.run(
        [
            "rustc",
            "--edition",
            "2021",
            "--crate-type",
            "bin",
            "--emit=metadata",
            *(arg for d in dirs for arg in ("-L", f"dependency={d}")),
            *(arg for crate, rlib in externs.items() for arg in ("--extern", f"{crate}={rlib}")),
            "-o",
            str(tmpdir / f"async_snippet_{idx}.rmeta"),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0, r.stderr


def check_async_blocks() -> int:
    blocks = [
        (Path(md).relative_to(REPO), m.group(1))
        for md in sorted(glob.glob(str(REPO / "docs" / "**" / "*.md"), recursive=True))
        for m in _ASYNC_BLOCK.finditer(Path(md).read_text())
    ]
    if not blocks:
        return 0
    artifacts = async_artifacts()
    if artifacts is None:
        print("!! could not build apache-spark-connect-async for its snippets")
        return 2
    failures = []
    with tempfile.TemporaryDirectory() as td:
        for idx, (rel, code) in enumerate(blocks, 1):
            ok, err = check_async_block(code, Path(td), idx, *artifacts)
            if not ok:
                first = next((ln for ln in err.splitlines() if ln.startswith("error")), err[:200])
                failures.append((rel, first))
    print(
        f"Checked {len(blocks)} async rust snippet(s) against apache-spark-connect-async; "
        f"{len(failures)} failed to compile."
    )
    for rel, err in failures:
        print(f"  {rel}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
