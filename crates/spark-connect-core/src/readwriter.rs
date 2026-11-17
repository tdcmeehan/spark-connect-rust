//! The data sources of a read, which a `Read` relation names.

use std::collections::HashMap;

/// ReadType variant for Read relation.
#[derive(Debug, Clone)]
pub enum ReadType {
    /// DataSource read (file-based or other formats).
    DataSource {
        format: Option<String>,
        schema: Option<String>,
        options: HashMap<String, String>,
        paths: Vec<String>,
        predicates: Vec<String>,
        source_name: Option<String>,
    },
    /// Named table read.
    NamedTable {
        table_name: String,
        options: HashMap<String, String>,
    },
}
