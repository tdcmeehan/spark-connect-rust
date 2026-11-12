//! The blocking forms of the reads, writes and MERGE of a DataFrame, and of its creation from
//! local data.

use std::future::IntoFuture;

use spark_connect_core::types::DataType;

use super::{wait, DataFrame, Session};
use crate::column::Column;
use crate::error::Result;
use crate::merge::MergeIntoWriter;
use crate::readwriter::{FromArrow, ParseFormat};
use crate::value::Value;

/// The parse of a [`crate::readwriter::DataFrameReader`], from
/// [`crate::readwriter::DataFrameReader::blocking`]. A reader's loads send nothing: they stay
/// the async reader's.
#[derive(Clone)]
#[must_use = "a reader parses when its parse is called"]
pub struct DataFrameReader(crate::readwriter::DataFrameReader);

impl DataFrameReader {
    /// The DataFrame of the strings of `input`, a DataFrame of one string column, async or
    /// blocking, parsed as `format` with the options of this reader, as
    /// [`crate::readwriter::DataFrameReader::parse`] parses them.
    pub fn parse(self, format: ParseFormat, input: &crate::DataFrame) -> Result<DataFrame> {
        wait(self.0.parse(format, input)).map(DataFrame::from)
    }
}

impl From<crate::readwriter::DataFrameReader> for DataFrameReader {
    fn from(reader: crate::readwriter::DataFrameReader) -> Self {
        DataFrameReader(reader)
    }
}

impl From<DataFrameReader> for crate::readwriter::DataFrameReader {
    fn from(reader: DataFrameReader) -> Self {
        reader.0
    }
}

impl crate::readwriter::DataFrameReader {
    /// The blocking form of this reader, whose [`parse`](DataFrameReader::parse) blocks.
    pub fn blocking(self) -> DataFrameReader {
        DataFrameReader(self)
    }
}

/// The destinations of a [`crate::readwriter::DataFrameWriter`], from
/// [`crate::readwriter::DataFrameWriter::blocking`]: each writes, and blocks until the server
/// has written.
#[derive(Clone)]
#[must_use = "a write needs a method that names its destination"]
pub struct DataFrameWriter(crate::readwriter::DataFrameWriter);

impl DataFrameWriter {
    /// Writes the DataFrame to `path`.
    pub fn save(self, path: impl Into<String>) -> Result<()> {
        wait(self.0.save(path))
    }

    /// Writes the DataFrame where the source's options say, for a source that takes no path.
    pub fn save_without_path(self) -> Result<()> {
        wait(self.0.save_without_path())
    }

    /// Writes the DataFrame as the table `name`, which the mode says what to do with if it
    /// exists.
    pub fn save_as_table(self, name: impl Into<String>) -> Result<()> {
        wait(self.0.save_as_table(name))
    }

    /// Inserts the rows of the DataFrame into the existing table `name`, by the position of
    /// their columns; [`SaveMode::Overwrite`](crate::readwriter::SaveMode::Overwrite) replaces
    /// the table's rows.
    pub fn insert_into(self, name: impl Into<String>) -> Result<()> {
        wait(self.0.insert_into(name))
    }

    /// Writes the DataFrame to `path` as CSV.
    pub fn csv(self, path: impl Into<String>) -> Result<()> {
        wait(self.0.csv(path))
    }

    /// Writes the DataFrame to `path` as JSON Lines.
    pub fn json(self, path: impl Into<String>) -> Result<()> {
        wait(self.0.json(path))
    }

    /// Writes the DataFrame to `path` as ORC.
    pub fn orc(self, path: impl Into<String>) -> Result<()> {
        wait(self.0.orc(path))
    }

    /// Writes the DataFrame to `path` as Parquet.
    pub fn parquet(self, path: impl Into<String>) -> Result<()> {
        wait(self.0.parquet(path))
    }

    /// Writes the DataFrame, of one string column, to `path` as lines of text.
    pub fn text(self, path: impl Into<String>) -> Result<()> {
        wait(self.0.text(path))
    }

    /// Writes the DataFrame to `path` as XML.
    pub fn xml(self, path: impl Into<String>) -> Result<()> {
        wait(self.0.xml(path))
    }
}

impl From<crate::readwriter::DataFrameWriter> for DataFrameWriter {
    fn from(writer: crate::readwriter::DataFrameWriter) -> Self {
        DataFrameWriter(writer)
    }
}

impl From<DataFrameWriter> for crate::readwriter::DataFrameWriter {
    fn from(writer: DataFrameWriter) -> Self {
        writer.0
    }
}

impl crate::readwriter::DataFrameWriter {
    /// The blocking form of this write, whose destinations block: set its format, mode,
    /// options and layout first.
    pub fn blocking(self) -> DataFrameWriter {
        DataFrameWriter(self)
    }
}

/// The ways to write a table of a [`crate::readwriter::DataFrameWriterV2`], from
/// [`crate::readwriter::DataFrameWriterV2::blocking`]: each writes, and blocks until the server
/// has written.
#[derive(Clone)]
#[must_use = "a write needs a method that says how to write the table"]
pub struct DataFrameWriterV2(crate::readwriter::DataFrameWriterV2);

impl DataFrameWriterV2 {
    /// Creates the table from the DataFrame; fails if it exists.
    pub fn create(self) -> Result<()> {
        wait(self.0.create())
    }

    /// Replaces the existing table with one of the DataFrame.
    pub fn replace(self) -> Result<()> {
        wait(self.0.replace())
    }

    /// Creates the table from the DataFrame, or replaces it if it exists.
    pub fn create_or_replace(self) -> Result<()> {
        wait(self.0.create_or_replace())
    }

    /// Appends the rows of the DataFrame to the existing table.
    pub fn append(self) -> Result<()> {
        wait(self.0.append())
    }

    /// Replaces the rows of the existing table for which `condition` holds with the rows of the
    /// DataFrame.
    pub fn overwrite(self, condition: impl Into<Column>) -> Result<()> {
        wait(self.0.overwrite(condition))
    }

    /// Replaces the partitions of the existing table that the rows of the DataFrame fall in with
    /// those rows.
    pub fn overwrite_partitions(self) -> Result<()> {
        wait(self.0.overwrite_partitions())
    }
}

impl From<crate::readwriter::DataFrameWriterV2> for DataFrameWriterV2 {
    fn from(writer: crate::readwriter::DataFrameWriterV2) -> Self {
        DataFrameWriterV2(writer)
    }
}

impl From<DataFrameWriterV2> for crate::readwriter::DataFrameWriterV2 {
    fn from(writer: DataFrameWriterV2) -> Self {
        writer.0
    }
}

impl crate::readwriter::DataFrameWriterV2 {
    /// The blocking form of this write, whose ways to write the table block: set its provider,
    /// options, table properties and layout first.
    pub fn blocking(self) -> DataFrameWriterV2 {
        DataFrameWriterV2(self)
    }
}

impl MergeIntoWriter {
    /// Runs the MERGE, as awaiting it does, and blocks until the server has merged the rows.
    pub fn run(self) -> Result<()> {
        wait(self.into_future())
    }
}

impl FromArrow {
    /// Creates the DataFrame, as awaiting it does, and blocks until the data is sent or
    /// uploaded.
    pub fn run(self) -> Result<DataFrame> {
        wait(self.into_future()).map(DataFrame::from)
    }
}

impl Session {
    /// The DataFrame of the local `rows` of `schema`, a struct type, as
    /// [`crate::Session::create_dataframe`] creates it: each row has the values of its fields, in
    /// order.
    pub fn create_dataframe<R: IntoIterator<Item = Value>>(
        &self,
        rows: impl IntoIterator<Item = R>,
        schema: DataType,
    ) -> Result<DataFrame> {
        wait(self.0.create_dataframe(rows, schema)).map(DataFrame::from)
    }
}
