// SPDX-License-Identifier: Apache-2.0
//! The persistence of PySpark's Python-format ML objects over Spark Connect, of the meta-algorithms
//! (`Pipeline`, `PipelineModel`, the validators), as `pyspark.ml.util` saves and loads them:
//! - [`Metadata`], [`save_metadata`] and [`load_metadata`]: the metadata of an object, which
//!   `DefaultParamsWriter.saveMetadata` writes as a text file of one line at `<path>/metadata` (a
//!   DataFrame of one row) and `DefaultParamsReader.loadMetadata` reads;
//! - [`stage_path`]: the path of a stage of a pipeline (`PipelineSharedReadWrite.getStagePath`);
//! - [`handle_overwrite`]: the deletion of a path by the ML helper before an overwrite, as
//!   `RemoteMLWriter.handleOverwrite` asks it.
//!
//! What these write is what PySpark writes, so that each loads what the other saves.
use super::*;
use crate::column::lit_string;
use spark_connect_core::error::Result;
pub use spark_connect_core::ml::{join_path, json_object, json_string, stage_path, Metadata};

/// Write the metadata `metadata` (its JSON) to `<path>/metadata`, as `saveMetadata` does: a text
/// file of one line, of a DataFrame of one row in one partition, of the default save mode, which
/// fails if the path exists.
pub fn save_metadata(session: &SparkSession, path: &str, metadata: &str) -> Result<()> {
    session
        .range(1)?
        .select([lit_string(metadata).alias("value")])
        .coalesce(1)
        .write()
        .text(&join_path(path, "metadata"))
}

/// The text of the metadata at `<path>/metadata`, as `loadMetadata` reads it: the first line of
/// the text files there, the JSON of the metadata as Python writes it (of `NaN` and `Infinity`
/// for those floats, which [`load_metadata`] does not parse).
pub fn load_metadata_text(session: &SparkSession, path: &str) -> Result<String> {
    let metadata_path = join_path(path, "metadata");
    let row = session
        .read()
        .text(&metadata_path)
        .first()?
        .ok_or_else(|| {
            SparkError::value_msg(format!("no metadata was found at {metadata_path}"))
        })?;
    row.get(0)
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            SparkError::value_msg(format!("the metadata at {metadata_path} is not a string"))
        })
}

/// The metadata at `<path>/metadata`, as `loadMetadata` reads it: the JSON of the first line of
/// the text files there (see [`load_metadata_text`]). When `expected_class` is given, a metadata
/// of another class is an error, as `_parseMetaData` raises it.
pub fn load_metadata(
    session: &SparkSession,
    path: &str,
    expected_class: Option<&str>,
) -> Result<serde_json::Value> {
    let metadata_path = join_path(path, "metadata");
    let text = load_metadata_text(session, path)?;
    let metadata: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| SparkError::value_msg(format!("the metadata at {metadata_path}: {e}")))?;
    if let Some(expected) = expected_class {
        let class = metadata["class"].as_str().unwrap_or_default();
        if class != expected {
            return Err(SparkError::value_msg(format!(
                "Error loading metadata: Expected class name {expected} but found class name \
                 {class}"
            )));
        }
    }
    Ok(metadata)
}

/// Delete `path` on the server, as `RemoteMLWriter.handleOverwrite` asks the ML helper to before
/// it saves a meta-algorithm of overwrite (`handleOverwrite(path, true)`).
pub fn handle_overwrite(session: &SparkSession, path: &str) -> Result<()> {
    fetch(
        &MlTarget::helper(session),
        "handleOverwrite",
        vec![
            FetchArg::Value(MlValue::String(path.to_string())),
            FetchArg::Value(MlValue::Bool(true)),
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_the_json_of_default_params_writer_in_its_order() {
        let param_map = json_object(&[
            ("stageUids", "[\"a\",\"b\"]".to_string()),
            ("language", json_string("Python")),
        ]);
        let mut metadata = Metadata::new(
            "pyspark.ml.pipeline.Pipeline",
            "Pipeline_4866ebac8670",
            "4.3.0",
            param_map,
            "{}".to_string(),
        );
        metadata.timestamp = 1794647224405;
        assert_eq!(
            metadata.to_json(),
            "{\"class\":\"pyspark.ml.pipeline.Pipeline\",\"timestamp\":1794647224405,\
             \"sparkVersion\":\"4.3.0\",\"uid\":\"Pipeline_4866ebac8670\",\
             \"paramMap\":{\"stageUids\":[\"a\",\"b\"],\"language\":\"Python\"},\
             \"defaultParamMap\":{}}"
        );
        metadata.extra.push(("avgMetrics".into(), "[0.5]".into()));
        assert!(metadata.to_json().ends_with(",\"avgMetrics\":[0.5]}"));
    }

    #[test]
    fn strings_are_json_as_python_dumps_them_with_ascii_escapes() {
        // The expected text is Python 3.11's `json.dumps` of the same string (`ensure_ascii`).
        assert_eq!(
            json_string("\u{e9}\u{1f600}\x7f\n\t\"\\/\x01\u{8}\u{c}\r"),
            "\"\\u00e9\\ud83d\\ude00\\u007f\\n\\t\\\"\\\\/\\u0001\\b\\f\\r\""
        );
        assert_eq!(json_string("plain 'q' ~"), "\"plain 'q' ~\"");
        assert_eq!(json_string(""), "\"\"");
    }

    #[test]
    fn stage_paths_pad_the_index_to_the_digits_of_the_number_of_stages() {
        assert_eq!(stage_path("A_1", 0, 3, "p/stages"), "p/stages/0_A_1");
        assert_eq!(stage_path("A_1", 7, 12, "p/stages/"), "p/stages/07_A_1");
        assert_eq!(stage_path("A_1", 10, 100, "s"), "s/010_A_1");
        assert_eq!(join_path("hdfs://h/p", "metadata"), "hdfs://h/p/metadata");
    }
}
