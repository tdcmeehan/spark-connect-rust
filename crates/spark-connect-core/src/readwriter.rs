//! The data sources of a read, which a `Read` relation names, and the operations that
//! write a DataFrame.

use std::collections::HashMap;

use spark_connect_proto as proto;

use crate::column::Column;

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

/// SaveMode for write operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveMode {
    Append,
    Overwrite,
    ErrorIfExists,
    Ignore,
}

impl SaveMode {
    /// Convert SaveMode to proto i32 value.
    pub fn to_proto(&self) -> i32 {
        match self {
            SaveMode::Append => 1i32,
            SaveMode::Overwrite => 2i32,
            SaveMode::ErrorIfExists => 3i32,
            SaveMode::Ignore => 4i32,
        }
    }

    /// Parse SaveMode from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "append" => Some(SaveMode::Append),
            "overwrite" => Some(SaveMode::Overwrite),
            "error" | "errorifexists" => Some(SaveMode::ErrorIfExists),
            "ignore" => Some(SaveMode::Ignore),
            _ => None,
        }
    }
}

/// TableSaveMethod for WriteOperation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableSaveMethod {
    SaveAsTable,
    InsertInto,
}

impl TableSaveMethod {
    /// Convert to proto i32 value.
    pub fn to_proto(&self) -> i32 {
        match self {
            TableSaveMethod::SaveAsTable => 1i32,
            TableSaveMethod::InsertInto => 2i32,
        }
    }
}

/// What a write of a DataFrame to a data source sends besides its input and destination.
#[derive(Debug, Clone)]
pub struct WriteSpec {
    pub format: Option<String>,
    pub mode: SaveMode,
    pub options: HashMap<String, String>,
    pub partition_cols: Vec<String>,
    pub cluster_cols: Vec<String>,
    pub bucket_cols: Vec<String>,
    pub sort_cols: Vec<String>,
    pub num_buckets: Option<i32>,
}

impl Default for WriteSpec {
    fn default() -> Self {
        WriteSpec {
            format: None,
            mode: SaveMode::ErrorIfExists,
            options: HashMap::new(),
            partition_cols: vec![],
            cluster_cols: vec![],
            bucket_cols: vec![],
            sort_cols: vec![],
            num_buckets: None,
        }
    }
}

impl WriteSpec {
    /// The `WriteOperation` that writes `input` to `save_type`, a path or a table.
    pub fn to_proto(
        &self,
        input: proto::Relation,
        save_type: Option<proto::write_operation::SaveType>,
    ) -> proto::WriteOperation {
        let mut op = proto::WriteOperation::default();
        op.input = Some(input);
        op.source = self.format.clone();
        op.mode = self.mode.to_proto();
        op.sort_column_names = self.sort_cols.clone();
        op.partitioning_columns = self.partition_cols.clone();
        op.clustering_columns = self.cluster_cols.clone();
        op.options = self.options.clone();
        op.save_type = save_type;
        if let Some(num_buckets) = self.num_buckets {
            let mut bucket_by = proto::write_operation::BucketBy::default();
            bucket_by.num_buckets = num_buckets;
            bucket_by.bucket_column_names = self.bucket_cols.clone();
            op.bucket_by = Some(bucket_by);
        }
        op
    }
}

/// The destination of a write to the table `table_name`, by `method`.
pub fn table_destination(
    table_name: &str,
    method: TableSaveMethod,
) -> proto::write_operation::SaveType {
    let mut table = proto::write_operation::SaveTable::default();
    table.table_name = table_name.to_string();
    table.save_method = method.to_proto();
    proto::write_operation::SaveType::Table(table)
}

/// What a write of a DataFrame to the table `table_name` of a catalog sends besides its
/// input and mode.
#[derive(Debug, Clone)]
pub struct WriteV2Spec {
    pub table_name: String,
    pub provider: Option<String>,
    pub options: HashMap<String, String>,
    pub table_properties: HashMap<String, String>,
    pub partition_cols: Vec<Column>,
    pub cluster_cols: Vec<String>,
}

impl WriteV2Spec {
    /// The write to `table_name`, with nothing else set.
    pub fn new(table_name: &str) -> Self {
        WriteV2Spec {
            table_name: table_name.to_string(),
            provider: None,
            options: HashMap::new(),
            table_properties: HashMap::new(),
            partition_cols: vec![],
            cluster_cols: vec![],
        }
    }

    /// The `WriteOperationV2` that writes `input` in `mode`.
    pub fn to_proto(
        &self,
        input: proto::Relation,
        mode: proto::write_operation_v2::Mode,
        overwrite_condition: Option<proto::Expression>,
    ) -> proto::WriteOperationV2 {
        let mut op = proto::WriteOperationV2::default();
        op.input = Some(input);
        op.table_name = self.table_name.clone();
        op.provider = self.provider.clone();
        op.partitioning_columns = self.partition_cols.iter().map(|c| c.to_proto()).collect();
        op.clustering_columns = self.cluster_cols.clone();
        op.options = self.options.clone();
        op.table_properties = self.table_properties.clone();
        op.mode = mode as i32;
        op.overwrite_condition = overwrite_condition;
        op
    }
}
