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

//! Comparison of the messages of both APIs.

use std::collections::HashMap;
use std::fmt::Debug;

/// The `Debug` text of `message` with its plan ids numbered 1, 2, ... in the order in which
/// they first appear. Plan ids come from one counter of the process, so two plans built apart
/// never share them; what is compared is which nodes and references share an id.
pub fn renumbered(message: &impl Debug) -> String {
    numbered_after(&format!("{message:?}"), "plan_id: ")
}

/// `text` with the number after each `marker`, bare or in `Some(..)`, numbered 1, 2, ... in the
/// order in which the numbers first appear.
pub fn numbered_after(text: &str, marker: &str) -> String {
    let mut ids: HashMap<String, usize> = HashMap::new();
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(marker) {
        let (before, after) = rest.split_at(at + marker.len());
        out.push_str(before);
        let prefix = if after.starts_with("Some(") {
            "Some("
        } else {
            ""
        };
        out.push_str(prefix);
        let after = &after[prefix.len()..];
        let digits = after.len() - after.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if digits > 0 {
            let next = ids.len() + 1;
            let id = *ids.entry(after[..digits].to_string()).or_insert(next);
            out.push_str(&format!("#{id}"));
        }
        rest = &after[digits..];
    }
    out.push_str(rest);
    out
}

/// The cases whose old and new messages differ, as a report.
pub fn differences<T: Debug, U: Debug>(cases: Vec<(&str, T, U)>) -> String {
    cases
        .into_iter()
        .filter_map(|(name, old, new)| {
            let (old, new) = (renumbered(&old), renumbered(&new));
            (old != new).then(|| format!("{name}:\n  old {old}\n  new {new}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}
