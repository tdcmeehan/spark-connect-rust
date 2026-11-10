//! The error type of the client.

use std::fmt;

use spark_connect_core::error::{QueryContextType, SparkError, SparkErrorKind};

/// A `Result` whose error is an [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// An error of the client or of the server.
///
/// An error that the server raised carries its condition ([`Error::error_class`]), its SQL state,
/// its message parameters and, for an error in SQL text, where in the text it occurred. An error
/// of reading a result into Rust values ([`ErrorKind::Decode`]) names the row and the column.
#[derive(Clone)]
pub struct Error(Box<Repr>);

#[derive(Clone)]
enum Repr {
    Spark(SparkError),
    Decode {
        message: String,
        /// The field names from the column of the row down to the value that failed.
        path: Vec<String>,
        row: Option<usize>,
    },
}

/// What kind of error an [`Error`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An argument the client or the server rejected, such as a malformed connection string.
    InvalidArgument,
    /// The server could not be reached, or refused the connection or its credentials.
    Transport,
    /// The server could not resolve the plan, such as a missing table or column.
    Analysis,
    /// The server could not parse SQL text.
    Parse,
    /// The plan failed while it ran.
    Execution,
    /// The operation was cancelled.
    Cancelled,
    /// The operation did not finish within its deadline.
    Timeout,
    /// A result could not be read into the requested Rust type.
    Decode,
    /// The server or the client does not support the operation.
    Unsupported,
    /// An error inside the client or the server that none of the other kinds describe.
    Internal,
}

/// Where in SQL text an error occurred, as the server reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct QueryContext {
    /// The kind of object that holds the text, such as `VIEW`; empty for a query.
    pub object_type: String,
    /// The name of that object; empty for a query.
    pub object_name: String,
    /// The index of the first character of [`QueryContext::fragment`] in the text.
    pub start_index: i32,
    /// The index of its last character.
    pub stop_index: i32,
    /// The part of the text that the error is about.
    pub fragment: String,
    /// A summary of the text, with the fragment marked.
    pub summary: String,
}

/// A change of the server's behavior that an error comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BreakingChange {
    /// How to migrate.
    pub migration_message: Vec<String>,
    /// A configuration, as a key and a value, that restores the previous behavior.
    pub mitigation_config: Option<(String, String)>,
    /// Whether the change needs an audit.
    pub needs_audit: bool,
}

const CANCELLED: i32 = tonic::Code::Cancelled as i32;
const DEADLINE_EXCEEDED: i32 = tonic::Code::DeadlineExceeded as i32;
const PERMISSION_DENIED: i32 = tonic::Code::PermissionDenied as i32;
const UNIMPLEMENTED: i32 = tonic::Code::Unimplemented as i32;
const UNAVAILABLE: i32 = tonic::Code::Unavailable as i32;
const UNAUTHENTICATED: i32 = tonic::Code::Unauthenticated as i32;

impl Error {
    pub(crate) fn invalid_argument(message: impl Into<String>) -> Error {
        SparkError::value_msg(message).into()
    }

    pub(crate) fn decode(message: impl Into<String>) -> Error {
        Error(Box::new(Repr::Decode {
            message: message.into(),
            path: Vec::new(),
            row: None,
        }))
    }

    /// This error of decoding, as of the value of the field `name` of what it decoded.
    pub(crate) fn in_field(mut self, name: &str) -> Error {
        if let Repr::Decode { path, .. } = &mut *self.0 {
            path.insert(0, name.to_string());
        }
        self
    }

    /// This error of decoding, as of the element `index` of what it decoded.
    pub(crate) fn in_element(self, index: usize) -> Error {
        match &*self.0 {
            Repr::Decode { .. } => self.in_field(&format!("[{index}]")),
            Repr::Spark(_) => self,
        }
    }

    /// This error of decoding, as of the row `index`.
    pub(crate) fn in_row(mut self, index: usize) -> Error {
        if let Repr::Decode { row, .. } = &mut *self.0 {
            row.get_or_insert(index);
        }
        self
    }

    fn spark(&self) -> Option<&SparkError> {
        match &*self.0 {
            Repr::Spark(error) => Some(error),
            Repr::Decode { .. } => None,
        }
    }

    /// The kind of the error.
    pub fn kind(&self) -> ErrorKind {
        let Some(error) = self.spark() else {
            return ErrorKind::Decode;
        };
        match error.grpc_code {
            Some(CANCELLED) => return ErrorKind::Cancelled,
            Some(DEADLINE_EXCEEDED) => return ErrorKind::Timeout,
            Some(UNAVAILABLE | UNAUTHENTICATED | PERMISSION_DENIED) => return ErrorKind::Transport,
            Some(UNIMPLEMENTED) => return ErrorKind::Unsupported,
            _ => {}
        }
        use SparkErrorKind as K;
        match error.kind {
            K::ValueError
            | K::TypeError
            | K::IndexError
            | K::KeyError
            | K::AttributeError
            | K::IllegalArgument
            | K::InvalidPlanInput
            | K::SessionNotSame => ErrorKind::InvalidArgument,
            K::Analysis | K::TempTableAlreadyExists => ErrorKind::Analysis,
            K::Parse => ErrorKind::Parse,
            K::ConnectGrpc
            | K::Arithmetic
            | K::ArrayIndexOutOfBounds
            | K::DateTime
            | K::NumberFormat
            | K::StreamingQuery
            | K::StreamingPythonRunnerInitialization
            | K::QueryExecution
            | K::Python
            | K::SparkRuntime
            | K::SparkUpgrade
            | K::SparkNoSuchElement
            | K::PickleException
            | K::Spark => ErrorKind::Execution,
            K::NotImplementedError | K::UnsupportedOperation => ErrorKind::Unsupported,
            K::Connect
            | K::RuntimeError
            | K::AssertionError
            | K::PicklingError
            | K::ImportError
            | K::RecursionError
            | K::Unknown => ErrorKind::Internal,
        }
    }

    /// The error's condition, such as `TABLE_OR_VIEW_NOT_FOUND`, if it has one.
    pub fn error_class(&self) -> Option<&str> {
        Some(self.spark()?.error_class.as_str()).filter(|class| !class.is_empty())
    }

    /// The SQL state of the error, such as `42P01`, if it has one.
    pub fn sql_state(&self) -> Option<String> {
        self.spark()?.sql_state()
    }

    /// The message parameters of the condition, as names and values, in the order the server
    /// gave them.
    pub fn message_parameters(&self) -> Vec<(&str, &str)> {
        let Some(error) = self.spark() else {
            return Vec::new();
        };
        let params = &error.params;
        let order = error
            .server
            .as_ref()
            .map(|server| server.parameter_order.as_slice())
            .unwrap_or_default();
        let ordered = order
            .iter()
            .filter_map(|name| params.get_key_value(name.as_str()));
        let rest = params.iter().filter(|(name, _)| !order.contains(name));
        ordered
            .chain(rest)
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect()
    }

    /// Where in SQL text the error occurred, as the server reports it.
    pub fn query_contexts(&self) -> Vec<QueryContext> {
        let Some(error) = self.spark() else {
            return Vec::new();
        };
        error
            .contexts
            .iter()
            .filter(|context| context.context_type == QueryContextType::SQL)
            .map(|context| QueryContext {
                object_type: context.object_type.clone(),
                object_name: context.object_name.clone(),
                start_index: context.start_index,
                stop_index: context.stop_index,
                fragment: context.fragment.clone(),
                summary: context.summary.clone(),
            })
            .collect()
    }

    /// The message of the error.
    pub fn message(&self) -> String {
        match &*self.0 {
            Repr::Spark(error) => error.message(),
            Repr::Decode { message, path, row } => {
                let mut column = String::new();
                for segment in path {
                    if !column.is_empty() && !segment.starts_with('[') {
                        column.push('.');
                    }
                    column.push_str(segment);
                }
                let at = match (row, column.is_empty()) {
                    (Some(row), false) => format!(" (row {row}, column `{column}`)"),
                    (Some(row), true) => format!(" (row {row})"),
                    (None, false) => format!(" (column `{column}`)"),
                    (None, true) => String::new(),
                };
                format!("{message}{at}")
            }
        }
    }

    /// The gRPC status code of the response that carried the error, if one did.
    pub fn grpc_code(&self) -> Option<i32> {
        self.spark()?.grpc_code
    }

    /// The id of the operation that failed, if the error is of one.
    pub fn operation_id(&self) -> Option<&str> {
        self.spark()?.operation_id.as_deref()
    }

    /// The server's stack trace of the error, if the server sent one.
    pub fn server_stack_trace(&self) -> Option<&str> {
        self.spark()?.server_stacktrace.as_deref()
    }

    /// The change of the server's behavior that the error comes from, if the server reports
    /// one.
    pub fn breaking_change(&self) -> Option<BreakingChange> {
        let change = self.spark()?.server.as_ref()?.breaking_change.as_ref()?;
        Some(BreakingChange {
            migration_message: change.migration_message.clone(),
            mitigation_config: change.mitigation_config.clone(),
            needs_audit: change.needs_audit,
        })
    }
}

impl From<SparkError> for Error {
    fn from(error: SparkError) -> Self {
        Error(Box::new(Repr::Spark(error)))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.0 {
            Repr::Spark(error) => fmt::Display::fmt(error, f),
            Repr::Decode { .. } => f.write_str(&self.message()),
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            .field("kind", &self.kind())
            .field("error_class", &self.error_class())
            .field("sql_state", &self.sql_state())
            .field("message", &self.message())
            .finish()
    }
}

impl std::error::Error for Error {}

impl serde::de::Error for Error {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Error::decode(message.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_connect_core::error::{QueryContext as CoreContext, ServerErrorInfo};
    use std::collections::BTreeMap;

    fn error(kind: SparkErrorKind, grpc_code: Option<i32>) -> Error {
        let mut error = SparkError::classed(kind, "", &[]);
        error.grpc_code = grpc_code;
        error.into()
    }

    #[test]
    fn each_kind_of_error_has_a_rust_kind() {
        use SparkErrorKind as K;
        let cases = [
            (K::ValueError, None, ErrorKind::InvalidArgument),
            (K::TypeError, None, ErrorKind::InvalidArgument),
            (K::IllegalArgument, None, ErrorKind::InvalidArgument),
            (K::InvalidPlanInput, None, ErrorKind::InvalidArgument),
            (K::Analysis, None, ErrorKind::Analysis),
            (K::TempTableAlreadyExists, None, ErrorKind::Analysis),
            (K::Parse, None, ErrorKind::Parse),
            (K::ConnectGrpc, Some(13), ErrorKind::Execution),
            (K::Arithmetic, None, ErrorKind::Execution),
            (K::StreamingQuery, None, ErrorKind::Execution),
            (K::Spark, None, ErrorKind::Execution),
            (K::UnsupportedOperation, None, ErrorKind::Unsupported),
            (K::NotImplementedError, None, ErrorKind::Unsupported),
            (K::RecursionError, None, ErrorKind::Internal),
            (K::Connect, None, ErrorKind::Internal),
            (K::Unknown, None, ErrorKind::Internal),
            (K::ConnectGrpc, Some(CANCELLED), ErrorKind::Cancelled),
            (K::ConnectGrpc, Some(DEADLINE_EXCEEDED), ErrorKind::Timeout),
            (K::ConnectGrpc, Some(UNAVAILABLE), ErrorKind::Transport),
            (K::ConnectGrpc, Some(UNAUTHENTICATED), ErrorKind::Transport),
            (
                K::ConnectGrpc,
                Some(PERMISSION_DENIED),
                ErrorKind::Transport,
            ),
            (K::Analysis, Some(UNIMPLEMENTED), ErrorKind::Unsupported),
        ];
        for (kind, grpc_code, expected) in cases {
            assert_eq!(
                error(kind, grpc_code).kind(),
                expected,
                "{kind:?} with gRPC code {grpc_code:?}"
            );
        }
    }

    #[test]
    fn message_parameters_keep_the_order_of_the_server() {
        let mut spark_error = SparkError::classed(SparkErrorKind::Analysis, "COND", &[]);
        spark_error.params = BTreeMap::from([
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string()),
            ("c".to_string(), "3".to_string()),
        ]);
        spark_error.server = Some(Box::new(ServerErrorInfo {
            parameter_order: vec!["c".to_string(), "a".to_string()],
            ..Default::default()
        }));
        let error = Error::from(spark_error);
        assert_eq!(
            error.message_parameters(),
            [("c", "3"), ("a", "1"), ("b", "2")]
        );
        assert_eq!(error.error_class(), Some("COND"));
    }

    #[test]
    fn only_the_contexts_of_sql_text_are_reported() {
        let context = |context_type, fragment: &str| {
            CoreContext::new(
                context_type,
                String::new(),
                String::new(),
                7,
                9,
                fragment.to_string(),
                String::new(),
                String::new(),
            )
        };
        let mut spark_error = SparkError::classed(SparkErrorKind::Analysis, "COND", &[]);
        spark_error.contexts = vec![
            context(QueryContextType::DataFrame, "col"),
            context(QueryContextType::SQL, "foo"),
        ];
        let contexts = Error::from(spark_error).query_contexts();
        assert_eq!(contexts.len(), 1);
        assert_eq!(
            (contexts[0].fragment.as_str(), contexts[0].start_index),
            ("foo", 7)
        );
    }
}
