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
    numbered_after(&message_text(message), "plan_id: ")
}

/// The `Debug` text of `message` with the entries of each map in key order: `Debug` prints those
/// of a `HashMap`, such as the options of a message, in no fixed order.
pub fn message_text(message: &impl Debug) -> String {
    sorted_maps(&format!("{message:?}"))
}

/// `text` with the entries of each map with quoted keys in key order, whatever their values, and
/// those of the texts it quotes too: a message's text compared as a `String` is escaped.
fn sorted_maps(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(['"', '{']) {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let (sorted, len) = sorted_item(rest).unwrap_or_else(|| (rest[..1].to_string(), 1));
        out.push_str(&sorted);
        rest = &rest[len..];
    }
    out.push_str(rest);
    out
}

/// The quoted string or the map that `text` starts with, with its maps sorted, and the length of
/// its text.
fn sorted_item(text: &str) -> Option<(String, usize)> {
    if text.starts_with('"') {
        let string = quoted(text)?;
        return Some((sorted_quoted(string), string.len()));
    }
    let (entries, len) = map_entries(text)?;
    let mut entries: Vec<String> = entries
        .into_iter()
        .map(|(key, value)| format!("{key}: {}", sorted_maps(value)))
        .collect();
    entries.sort();
    Some((format!("{{{}}}", entries.join(", ")), len))
}

/// `string`, a string as `Debug` prints it, with the maps of its text sorted.
fn sorted_quoted(string: &str) -> String {
    let Some(text) = unquoted(string) else {
        return string.to_string();
    };
    let sorted = sorted_maps(&text);
    if sorted == text {
        string.to_string()
    } else {
        format!("{sorted:?}")
    }
}

/// The entries of the map with quoted keys that `text` starts with, as the texts of their keys
/// and values, and the length of the map's text.
fn map_entries(text: &str) -> Option<(Vec<(&str, &str)>, usize)> {
    let mut entries = Vec::new();
    let mut at = 1;
    loop {
        let key = quoted(&text[at..])?;
        at += key.len();
        if !text[at..].starts_with(": ") {
            return None;
        }
        at += 2;
        let value = entry_value(&text[at..])?;
        at += value.len();
        entries.push((key, value));
        match text.as_bytes()[at..].first()? {
            b'}' => return Some((entries, at + 1)),
            b',' if text[at..].starts_with(", ") => at += 2,
            _ => return None,
        }
    }
}

/// The value of a map entry that `text` starts with: the text up to the first `, ` or `}` outside
/// its quoted strings and brackets.
fn entry_value(text: &str) -> Option<&str> {
    let mut depth = 0;
    let mut at = 0;
    while let Some(&byte) = text.as_bytes().get(at) {
        match byte {
            b'"' => {
                at += quoted(&text[at..])?.len();
                continue;
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' if depth > 0 => depth -= 1,
            b'}' => return Some(&text[..at]),
            b',' if depth == 0 && text[at..].starts_with(", ") => return Some(&text[..at]),
            b')' | b']' => return None,
            _ => {}
        }
        at += 1;
    }
    None
}

/// The quoted string, with its quotes, that `text` starts with.
fn quoted(text: &str) -> Option<&str> {
    let body = text.strip_prefix('"')?;
    let mut escaped = false;
    for (i, c) in body.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' => escaped = true,
            '"' => return Some(&text[..i + 2]),
            _ => {}
        }
    }
    None
}

/// The text of `string`, a string as `Debug` prints it, if its only escapes are `\\` and `\"`, as
/// in the quoted text of a message.
fn unquoted(string: &str) -> Option<String> {
    let mut text = String::with_capacity(string.len());
    let mut rest = &string[1..string.len() - 1];
    while let Some((before, after)) = rest.split_once('\\') {
        text.push_str(before);
        text.push(after.chars().next().filter(|&c| matches!(c, '\\' | '"'))?);
        rest = &after[1..];
    }
    text.push_str(rest);
    Some(text)
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
