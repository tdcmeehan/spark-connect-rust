//! Conversions of collected Arrow record batches into the data frames of other libraries, which
//! both front ends offer behind their `datafusion` and `polars` features.

use arrow::record_batch::RecordBatch;

use crate::error::{Result, SparkError};

/// Serialize record batches to Arrow IPC (file format) bytes; an empty input yields a valid
/// empty-schema IPC file.
pub fn record_batches_to_ipc(batches: &[RecordBatch]) -> Result<Vec<u8>> {
    use arrow::ipc::writer::FileWriter;
    let schema = match batches.first() {
        Some(b) => b.schema(),
        None => std::sync::Arc::new(arrow::datatypes::Schema::empty()),
    };
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut buf, schema.as_ref())
            .map_err(|e| SparkError::connect_msg(format!("Arrow IPC writer init failed: {e}")))?;
        for batch in batches {
            writer
                .write(batch)
                .map_err(|e| SparkError::connect_msg(format!("Arrow IPC write failed: {e}")))?;
        }
        writer
            .finish()
            .map_err(|e| SparkError::connect_msg(format!("Arrow IPC finish failed: {e}")))?;
    }
    Ok(buf)
}

/// Build a DataFusion DataFrame from record batches; an empty input is an error.
#[cfg(feature = "datafusion")]
pub fn record_batches_to_datafusion(
    ctx: &datafusion::prelude::SessionContext,
    batches: Vec<RecordBatch>,
) -> Result<datafusion::dataframe::DataFrame> {
    if batches.is_empty() {
        return Err(SparkError::connect_msg(
            "Cannot create DataFusion DataFrame from empty result",
        ));
    }
    ctx.read_batches(batches)
        .map_err(|e| SparkError::connect_msg(format!("Failed to create DataFusion DataFrame: {e}")))
}

/// Build a Polars DataFrame from record batches by bridging through Arrow IPC bytes, so polars'
/// vendored arrow need not match this crate's arrow-rs.
#[cfg(feature = "polars")]
pub fn record_batches_to_polars(batches: &[RecordBatch]) -> Result<polars::frame::DataFrame> {
    use polars::prelude::{IpcReader, SerReader};
    use std::io::Cursor;
    if batches.is_empty() {
        return Ok(polars::frame::DataFrame::empty());
    }
    let buf = record_batches_to_ipc(batches)?;
    IpcReader::new(Cursor::new(buf))
        .finish()
        .map_err(|e| SparkError::connect_msg(format!("Failed to create Polars DataFrame: {e}")))
}
