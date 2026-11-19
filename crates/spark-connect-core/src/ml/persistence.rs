// SPDX-License-Identifier: Apache-2.0
//! The metadata of PySpark's Python format of ML objects, as `DefaultParamsWriter` of
//! `pyspark.ml.util` writes it, with the JSON texts of its values and the paths of its files.

/// The metadata of an object of PySpark's Python format, as `DefaultParamsWriter` makes it: the
/// fully qualified name of its Python class (PySpark's own, `pyspark.ml.pipeline.Pipeline`), the
/// time of the save, the Spark version of the server, the uid, and the params and default params,
/// as the JSON texts of their objects (of [`json_object`], in PySpark's order of their keys), with
/// any `extra` fields after them (each a key and the JSON text of its value).
#[derive(Debug, Clone, PartialEq)]
pub struct Metadata {
    pub class: String,
    pub timestamp: i64,
    pub spark_version: String,
    pub uid: String,
    pub param_map: String,
    pub default_param_map: String,
    pub extra: Vec<(String, String)>,
}

impl Metadata {
    /// The metadata of an object of the class `class` and the uid `uid`, of `param_map` and
    /// `default_param_map` (the JSON texts of objects), of now and of the Spark version
    /// `spark_version`.
    pub fn new(
        class: &str,
        uid: &str,
        spark_version: &str,
        param_map: String,
        default_param_map: String,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or_default();
        Self {
            class: class.to_string(),
            timestamp,
            spark_version: spark_version.to_string(),
            uid: uid.to_string(),
            param_map,
            default_param_map,
            extra: Vec::new(),
        }
    }

    /// The JSON of the metadata, of its fields in the order of `DefaultParamsWriter`, compact as
    /// its `json.dumps(..., separators=(",", ":"))`.
    pub fn to_json(&self) -> String {
        let mut fields = vec![
            ("class", json_string(&self.class)),
            ("timestamp", self.timestamp.to_string()),
            ("sparkVersion", json_string(&self.spark_version)),
            ("uid", json_string(&self.uid)),
            ("paramMap", self.param_map.clone()),
            ("defaultParamMap", self.default_param_map.clone()),
        ];
        fields.extend(
            self.extra
                .iter()
                .map(|(key, value)| (key.as_str(), value.clone())),
        );
        json_object(&fields)
    }
}

/// The JSON text of the string `text`, as Python's `json.dumps` writes a str (of its default
/// `ensure_ascii`): the escapes `\"`, `\\`, `\n`, `\r`, `\t`, `\b` and `\f`, and `\uXXXX` in
/// lowercase hex for each other character that is not printable ASCII (a surrogate pair above
/// U+FFFF), as `json.dumps` escapes `\x7f` and the non-ASCII characters.
pub fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

/// The compact JSON text of an object of the fields `fields` (each a key and the JSON text of its
/// value), in their order, as `json.dumps` writes a Python dict, of the order of its insertion.
pub fn json_object(fields: &[(&str, String)]) -> String {
    let body = fields
        .iter()
        .map(|(key, value)| format!("{}:{value}", json_string(key)))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{body}}}")
}

/// `os.path.join(path, name)` of a path of the server's file system, with `/`.
pub fn join_path(path: &str, name: &str) -> String {
    if path.ends_with('/') {
        format!("{path}{name}")
    } else {
        format!("{path}/{name}")
    }
}

/// The path of the stage of the uid `stage_uid` of a pipeline of `num_stages` stages, at the
/// index `stage_idx`, under `stages_dir`, as `getStagePath`: the index, zero-padded to the digits
/// of the number of stages, and the uid.
pub fn stage_path(
    stage_uid: &str,
    stage_idx: usize,
    num_stages: usize,
    stages_dir: &str,
) -> String {
    let digits = num_stages.to_string().len();
    join_path(stages_dir, &format!("{stage_idx:0digits$}_{stage_uid}"))
}
