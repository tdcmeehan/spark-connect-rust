// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The direct calls that `#[spark_udf(crate = spark_connect_async)]` generates share one UDF
//! value per function, so they build its payload once per process, and again after a build
//! that failed.
//!
//! Builds run the Python of `SPARK_CONNECT_PYTHON`, which the test sets first to a script that
//! answers with the spec it is given and logs the entrypoint of each build.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use rust_udf_async_tests::udf;
use spark_connect_async::ErrorKind;

fn scratch() -> &'static Path {
    Path::new(env!("CARGO_TARGET_TMPDIR"))
}

/// The script that answers with the spec it is given, after it logs the spec's entrypoint to
/// `direct-call-builds`. It fails for an entrypoint `e` while the file `e.fail` is beside it. A
/// run without a spec is the version check, which it answers with nothing.
const ECHO: &str = r#"#!/bin/sh
spec=$(cat)
[ -n "$spec" ] || exit 0
dir=$(dirname "$0")
entrypoint=$(printf '%s' "$spec" | sed -n 's/.*"entrypoint":"\([^"]*\)".*/\1/p')
echo "$entrypoint" >> "$dir/direct-call-builds"
if [ -e "$dir/$entrypoint.fail" ]; then
  echo 'the packer failed' >&2
  exit 1
fi
printf '%s' "$spec"
"#;

/// Points `SPARK_CONNECT_PYTHON` at `ECHO`, with an empty log, before anything else runs.
fn echo_python() {
    let path = scratch().join("direct-call-python");
    let staged = scratch().join(format!("direct-call-python.{}", std::process::id()));
    std::fs::write(&staged, ECHO).unwrap();
    std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::rename(&staged, &path).unwrap();
    std::fs::write(scratch().join("direct-call-builds"), "").unwrap();
    std::env::set_var("SPARK_CONNECT_PYTHON", &path);
}

/// The builds of UDFs whose entrypoint is `entrypoint`, as `ECHO` logged them.
fn builds(entrypoint: &str) -> usize {
    let log = std::fs::read_to_string(scratch().join("direct-call-builds")).unwrap();
    log.lines().filter(|line| *line == entrypoint).count()
}

#[test]
fn the_direct_calls_of_a_function_build_its_payload_once_unless_the_build_fails() {
    echo_python();
    let failing = scratch().join("spark_udf_minus.fail");
    std::fs::write(&failing, "").unwrap();
    let error = udf::minus("x", "y").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Internal, "{error:?}");
    assert_eq!(builds("spark_udf_minus"), 1, "the failed call");

    std::fs::remove_file(&failing).unwrap();
    for _ in 0..3 {
        std::thread::spawn(|| {
            udf::minus("x", "y").unwrap();
        })
        .join()
        .unwrap();
    }
    assert_eq!(
        builds("spark_udf_minus"),
        2,
        "three calls after it, on threads of their own"
    );
}
