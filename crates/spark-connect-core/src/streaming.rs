//! The operation that starts a streaming query.

use std::collections::HashMap;

use spark_connect_proto as proto;

use crate::udf::PythonUDFPayload;

/// Trigger for streaming queries.
#[derive(Debug, Clone)]
pub enum Trigger {
    /// Process every `interval` milliseconds or duration string (e.g., "10 seconds").
    ProcessingTime(String),
    /// Process only once.
    Once,
    /// Process as soon as available data arrives.
    AvailableNow,
    /// Continuous processing with checkpoint interval.
    Continuous(String),
    /// Real-time processing, of batches of the duration (e.g., "5 seconds"), as PySpark's
    /// `trigger(realTime=...)`.
    RealTime(String),
}

impl Trigger {
    /// The trigger of a `WriteStreamOperationStart`.
    pub fn to_proto(&self) -> proto::write_stream_operation_start::Trigger {
        use proto::write_stream_operation_start::Trigger as Proto;
        match self {
            Trigger::ProcessingTime(interval) => Proto::ProcessingTimeInterval(interval.clone()),
            Trigger::Once => Proto::Once(true),
            Trigger::AvailableNow => Proto::AvailableNow(true),
            Trigger::Continuous(interval) => Proto::ContinuousCheckpointInterval(interval.clone()),
            Trigger::RealTime(duration) => Proto::RealTimeBatchDuration(duration.clone()),
        }
    }
}

/// What the start of a streaming query of a DataFrame sends besides its input.
pub struct WriteStreamSpec {
    pub format: Option<String>,
    pub output_mode: Option<String>,
    pub options: HashMap<String, String>,
    pub partitioning_columns: Vec<String>,
    pub clustering_columns: Vec<String>,
    pub query_name: Option<String>,
    pub trigger: Option<Trigger>,
    pub path: Option<String>,
    pub table_name: Option<String>,
    pub foreach_batch_payload: Option<PythonUDFPayload>,
    pub foreach_payload: Option<PythonUDFPayload>,
}

impl WriteStreamSpec {
    /// The `WriteStreamOperationStart` that starts the query of `input`.
    pub fn to_proto(self, input: proto::Relation) -> proto::WriteStreamOperationStart {
        let mut write_op = proto::WriteStreamOperationStart::default();
        write_op.input = Some(input);

        if let Some(fmt) = &self.format {
            write_op.format = fmt.clone();
        }

        if let Some(mode) = &self.output_mode {
            write_op.output_mode = mode.clone();
        }

        write_op.options = self.options;
        write_op.partitioning_column_names = self.partitioning_columns;
        write_op.clustering_column_names = self.clustering_columns;

        if let Some(name) = &self.query_name {
            write_op.query_name = name.clone();
        }

        write_op.trigger = self.trigger.as_ref().map(Trigger::to_proto);

        if let Some(path) = &self.path {
            // A memory/console/foreach sink has no path; only set the destination for a
            // real (non-empty) path so those sinks send an unset sink_destination.
            if !path.is_empty() {
                write_op.sink_destination = Some(
                    proto::write_stream_operation_start::SinkDestination::Path(path.clone()),
                );
            }
        }

        if let Some(table) = &self.table_name {
            write_op.sink_destination = Some(
                proto::write_stream_operation_start::SinkDestination::TableName(table.clone()),
            );
        }

        if let Some(foreach_batch) = &self.foreach_batch_payload {
            let mut foreach_func = proto::StreamingForeachFunction::default();
            foreach_func.function =
                Some(proto::streaming_foreach_function::Function::PythonFunction(
                    foreach_batch.to_proto(),
                ));
            write_op.foreach_batch = Some(foreach_func);
        }

        if let Some(foreach) = &self.foreach_payload {
            let mut foreach_func = proto::StreamingForeachFunction::default();
            foreach_func.function = Some(
                proto::streaming_foreach_function::Function::PythonFunction(foreach.to_proto()),
            );
            write_op.foreach_writer = Some(foreach_func);
        }

        write_op
    }
}
