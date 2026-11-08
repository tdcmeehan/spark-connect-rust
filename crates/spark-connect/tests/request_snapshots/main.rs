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

//! Request snapshots of the Rust API: every case builds a request, or a plan, column or type,
//! with the public API, and its bytes must equal those in `tests/golden/rust_requests.jsonl`.
//!
//! Plans, columns and types are recorded as the API serializes them. Requests are recorded by
//! an in-process server as the client sends them. Each case runs on a fresh session. Plan ids
//! come from one counter of the process, which each case starts at a value of its own (see
//! `PLAN_IDS_PER_CASE`). Random ids (the operation id of an execution, and those a
//! case masks) are replaced by a placeholder of the same length, which keeps the encoding.
//!
//! Rewrite the snapshots with `SPARK_CONNECT_UPDATE_REQUEST_SNAPSHOTS=1`.

#![allow(clippy::result_large_err)] // SparkError is large; the API returns it

macro_rules! cases {
    ($($name:literal => $run:expr),* $(,)?) => {
        vec![$(crate::Case { name: $name, run: $run }),*]
    };
}

mod cases;
mod server;

use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use prost::Message;
use serde_json::{json, Value as Json};
use spark_connect::column::Column;
use spark_connect::dataframe::DataFrame;
use spark_connect::expression::Expression;
use spark_connect::plan::LogicalPlan;
use spark_connect::session::SparkSession;
use spark_connect::types::DataType;
use spark_connect::Result;
use spark_connect_proto as proto;

use server::{Entry, Recorder};

const SNAPSHOTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/golden/rust_requests.jsonl"
);
const SESSION_ID: &str = "00000000-0000-4000-8000-00000000005e";
/// Without one in the connection string, the user id is that of the environment.
const USER_ID: &str = "snapshots";
/// The plan ids of the case of rank `r`, among the names of all cases (of every feature), start
/// at `(r + 1) * PLAN_IDS_PER_CASE`. The counter only rises, so the cases run in that order.
const PLAN_IDS_PER_CASE: i64 = 1_000_000;

pub struct Case {
    pub name: &'static str,
    pub run: fn(&Ctx) -> Result<()>,
}

/// What a case runs with: a session on the recording server, and ways to record values.
pub struct Ctx {
    pub spark: SparkSession,
    recorder: Arc<Recorder>,
    masks: RefCell<Vec<String>>,
    unordered: std::cell::Cell<bool>,
}

impl Ctx {
    pub fn column(&self, column: &Column) {
        self.message("Expression", &column.to_proto());
    }

    pub fn columns<'a>(&self, columns: impl IntoIterator<Item = &'a Column>) {
        for column in columns {
            self.column(column);
        }
    }

    pub fn expression(&self, expression: &Expression) {
        self.message("Expression", &expression.to_proto());
    }

    pub fn plan(&self, plan: &LogicalPlan) {
        self.message("Relation", &plan.to_proto());
    }

    pub fn data_type(&self, data_type: &DataType) {
        self.message("DataType", &data_type.to_proto());
    }

    pub fn message(&self, kind: &str, message: &impl Message) {
        self.recorder.push(kind, message.encode_to_vec());
    }

    /// Sends the plan of `df` to the server.
    pub fn df(&self, df: &DataFrame) -> Result<()> {
        df.collect().map(drop)
    }

    /// Records the entries of the case in a sorted order, for a case whose requests are sent
    /// from a background thread.
    pub fn unordered(&self) {
        self.unordered.set(true);
    }

    /// Waits for a request that the client sends in the background, so it is recorded in this
    /// case: one that `found` accepts.
    pub fn wait_for(&self, found: impl Fn(&Entry) -> bool) {
        let timeout = std::time::Duration::from_secs(10);
        assert!(
            self.recorder.wait_for(timeout, found),
            "no such request in {timeout:?}"
        );
    }

    /// Masks `value`, a random id, wherever it occurs in what the case records.
    pub fn mask(&self, value: &str) {
        self.masks.borrow_mut().push(value.to_string());
    }
}

/// The recorded entries of a case, and whether it succeeded.
#[derive(Debug, PartialEq)]
struct Snapshot {
    outcome: String,
    entries: Vec<Entry>,
}

fn placeholder(value: &str) -> Vec<u8> {
    value
        .bytes()
        .map(|b| if b == b'-' { b'-' } else { b'0' })
        .collect()
}

fn replace_all(bytes: &mut [u8], needle: &[u8], with: &[u8]) {
    if needle.is_empty() {
        return;
    }
    let mut i = 0;
    while i + needle.len() <= bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            bytes[i..i + needle.len()].copy_from_slice(with);
            i += needle.len();
        } else {
            i += 1;
        }
    }
}

fn mask(mut entries: Vec<Entry>, masks: &[String]) -> Vec<Entry> {
    for entry in &mut entries {
        let mut values = masks.to_vec();
        if entry.kind == "ExecutePlanRequest" {
            let request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
            values.extend(request.operation_id);
        }
        for value in values {
            replace_all(&mut entry.bytes, value.as_bytes(), &placeholder(&value));
        }
    }
    entries
}

fn run(
    case: &Case,
    first_plan_id: i64,
    address: std::net::SocketAddr,
    recorder: &Arc<Recorder>,
) -> Snapshot {
    let spark = SparkSession::builder()
        .remote(&format!(
            "sc://{address}/;session_id={SESSION_ID};user_id={USER_ID}"
        ))
        .get_or_create()
        .unwrap();
    // A relation the session sent raises the counter past its plan id. It stays alive during the
    // case, so no node of the case can take its place in the session's plan ids.
    let mut relation = proto::Relation::default();
    relation.common = Some(proto::RelationCommon {
        plan_id: Some(first_plan_id - 1),
        ..Default::default()
    });
    let _plan_ids_start = DataFrame::from_sent_relation(spark.clone(), relation);
    recorder.take();
    let ctx = Ctx {
        spark,
        recorder: recorder.clone(),
        masks: RefCell::new(Vec::new()),
        unordered: std::cell::Cell::new(false),
    };
    let outcome = match std::panic::catch_unwind(AssertUnwindSafe(|| (case.run)(&ctx))) {
        Ok(Ok(())) => "ok".to_string(),
        Ok(Err(error)) => format!("error: {error}"),
        Err(panic) => format!(
            "panic: {}",
            panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        ),
    };
    let mut entries = mask(recorder.take(), &ctx.masks.borrow());
    if ctx.unordered.get() {
        entries.sort_by(|a, b| (&a.kind, &a.bytes).cmp(&(&b.kind, &b.bytes)));
    }
    Snapshot { outcome, entries }
}

// Hex, not base64: the base64 of request bytes can spell what secret scanners take for keys.
fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn from_hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

fn to_json(name: &str, snapshot: &Snapshot) -> String {
    json!({
        "case": name,
        "outcome": snapshot.outcome,
        "entries": snapshot.entries.iter().map(|e| json!({
            "kind": e.kind,
            "hex": to_hex(&e.bytes),
        })).collect::<Vec<_>>(),
    })
    .to_string()
}

fn from_json(line: &str) -> (String, Snapshot) {
    let value: Json = serde_json::from_str(line).unwrap();
    let entries = value["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| Entry {
            kind: e["kind"].as_str().unwrap().to_string(),
            bytes: from_hex(e["hex"].as_str().unwrap()),
        })
        .collect();
    (
        value["case"].as_str().unwrap().to_string(),
        Snapshot {
            outcome: value["outcome"].as_str().unwrap().to_string(),
            entries,
        },
    )
}

fn decoded<M: Message + Default + PartialEq + std::fmt::Debug>(
    expected: &[u8],
    actual: &[u8],
) -> (bool, String, String) {
    let expected = M::decode(expected).unwrap();
    let actual = M::decode(actual).unwrap();
    (
        expected == actual,
        format!("{expected:?}"),
        format!("{actual:?}"),
    )
}

/// Whether two encodings of `kind` are the same message, and their debug forms.
fn same_message(kind: &str, expected: &[u8], actual: &[u8]) -> (bool, String, String) {
    match kind {
        "Text" => decoded::<String>(expected, actual),
        "Expression" => decoded::<proto::Expression>(expected, actual),
        "Literal" => decoded::<proto::expression::Literal>(expected, actual),
        "SortOrder" => decoded::<proto::expression::SortOrder>(expected, actual),
        "PythonUdf" => decoded::<proto::PythonUdf>(expected, actual),
        "PythonDataSource" => decoded::<proto::PythonDataSource>(expected, actual),
        "MlRelation" => decoded::<proto::MlRelation>(expected, actual),
        "Relation" => decoded::<proto::Relation>(expected, actual),
        "DataType" => decoded::<proto::DataType>(expected, actual),
        "Command" => decoded::<proto::Command>(expected, actual),
        "MlOperator" => decoded::<proto::MlOperator>(expected, actual),
        "MlParams" => decoded::<proto::MlParams>(expected, actual),
        "CommonInlineUserDefinedFunction" => {
            decoded::<proto::CommonInlineUserDefinedFunction>(expected, actual)
        }
        "CommonInlineUserDefinedDataSource" => {
            decoded::<proto::CommonInlineUserDefinedDataSource>(expected, actual)
        }
        "ExecutePlanRequest" => decoded::<proto::ExecutePlanRequest>(expected, actual),
        "AnalyzePlanRequest" => decoded::<proto::AnalyzePlanRequest>(expected, actual),
        "ConfigRequest" => decoded::<proto::ConfigRequest>(expected, actual),
        "AddArtifactsRequest" => decoded::<proto::AddArtifactsRequest>(expected, actual),
        "ArtifactStatusesRequest" => decoded::<proto::ArtifactStatusesRequest>(expected, actual),
        "InterruptRequest" => decoded::<proto::InterruptRequest>(expected, actual),
        "ReattachExecuteRequest" => decoded::<proto::ReattachExecuteRequest>(expected, actual),
        "ReleaseSessionRequest" => decoded::<proto::ReleaseSessionRequest>(expected, actual),
        "CloneSessionRequest" => decoded::<proto::CloneSessionRequest>(expected, actual),
        "FetchErrorDetailsRequest" => decoded::<proto::FetchErrorDetailsRequest>(expected, actual),
        other => panic!("no message type for snapshot entries of kind {other}"),
    }
}

/// Why `actual` differs from `expected`, if it does.
fn difference(expected: &Snapshot, actual: &Snapshot) -> Option<String> {
    if expected.outcome != actual.outcome {
        return Some(format!(
            "outcome: expected {:?}, got {:?}",
            expected.outcome, actual.outcome
        ));
    }
    if expected.entries.len() != actual.entries.len() {
        let kinds = |s: &Snapshot| s.entries.iter().map(|e| e.kind.clone()).collect::<Vec<_>>();
        return Some(format!(
            "entries: expected {:?}, got {:?}",
            kinds(expected),
            kinds(actual)
        ));
    }
    for (index, (e, a)) in expected.entries.iter().zip(&actual.entries).enumerate() {
        if e.kind != a.kind {
            return Some(format!(
                "entry {index}: expected a {}, got a {}",
                e.kind, a.kind
            ));
        }
        if e.bytes == a.bytes {
            continue;
        }
        // prost encodes the entries of a map field in the HashMap's iteration order, which
        // differs between processes, so encodings that differ may still be the same message.
        let (same, expected, actual) = same_message(&e.kind, &e.bytes, &a.bytes);
        if !same {
            return Some(format!(
                "entry {index} ({}):\n  expected {expected}\n  got      {actual}",
                e.kind
            ));
        }
    }
    None
}

#[test]
fn requests_of_the_rust_api_are_unchanged() {
    let recorder = Arc::new(Recorder::default());
    let address = server::start(recorder.clone());
    let mut all = cases::all();
    let mut names = HashSet::new();
    for case in &all {
        assert!(names.insert(case.name), "duplicate case {}", case.name);
    }
    all.sort_by_key(|case| case.name);
    let mut ranked: Vec<&str> = names
        .into_iter()
        .chain(cases::FEATURE_GATED.iter().copied())
        .collect();
    ranked.sort_unstable();
    ranked.dedup();
    let actual: Vec<(&str, Snapshot)> = all
        .iter()
        .map(|case| {
            let rank = ranked.binary_search(&case.name).unwrap() as i64;
            let first_plan_id = (rank + 1) * PLAN_IDS_PER_CASE;
            (case.name, run(case, first_plan_id, address, &recorder))
        })
        .collect();

    if std::env::var_os("SPARK_CONNECT_UPDATE_REQUEST_SNAPSHOTS").is_some() {
        let mut text = String::new();
        for (name, snapshot) in &actual {
            text.push_str(&to_json(name, snapshot));
            text.push('\n');
        }
        std::fs::write(SNAPSHOTS, text).unwrap();
        return;
    }

    let mut expected: BTreeMap<String, Snapshot> = std::fs::read_to_string(SNAPSHOTS)
        .expect("tests/golden/rust_requests.jsonl")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(from_json)
        .collect();
    let mut failures = Vec::new();
    for (name, snapshot) in &actual {
        match expected.remove(*name) {
            None => failures.push(format!("{name}: no snapshot")),
            Some(expected) => {
                if let Some(difference) = difference(&expected, snapshot) {
                    failures.push(format!("{name}: {difference}"));
                }
            }
        }
    }
    failures.extend(
        expected
            .keys()
            .filter(|name| !cases::FEATURE_GATED.contains(&name.as_str()))
            .map(|name| format!("{name}: no such case")),
    );
    assert!(
        failures.is_empty(),
        "{} of {} request snapshots differ:\n{}\nIf the change is intended, rerun with \
         SPARK_CONNECT_UPDATE_REQUEST_SNAPSHOTS=1 to rewrite them.",
        failures.len(),
        actual.len(),
        failures.join("\n")
    );
}
