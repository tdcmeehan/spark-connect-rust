//! The progress of the executions of a session, as PySpark Connect's
//! `pyspark.sql.connect.shell.progress`: the progress handlers of the session are called with
//! each progress of the server that has tasks, and once more, as done, when the execution ends.

use std::sync::Arc;

use spark_connect_core::actions::ExecutionObserver;
use spark_connect_core::client::ReattachableResponseStream;
use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::{check_callback_depth, CallbackGuard};
use spark_connect_proto as proto;

use crate::dataframe::{DataFrame, ExecutionSlot};
use crate::profiler::PythonAccumulators;
use crate::session::{ExecutionInfo, ProgressHandler, ProgressHandlers, SparkSession};

/// The progress of a stage of an execution, as PySpark's `StageInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageInfo {
    pub stage_id: i64,
    pub num_tasks: i64,
    pub num_completed_tasks: i64,
    pub num_bytes_read: i64,
    pub done: bool,
}

impl StageInfo {
    /// The stage of a progress of the server, as PySpark's `from_proto`.
    fn from_proto(stage: &proto::execute_plan_response::execution_progress::StageInfo) -> Self {
        StageInfo {
            stage_id: stage.stage_id,
            num_tasks: stage.num_tasks,
            num_completed_tasks: stage.num_completed_tasks,
            num_bytes_read: stage.input_bytes_read,
            done: stage.done,
        }
    }
}

/// A progress of an execution, of which PySpark calls each `ProgressHandler` with the keyword
/// arguments: the stages and the running tasks of the last progress of the server, the
/// operation, and whether the execution ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressUpdate {
    pub stages: Vec<StageInfo>,
    pub inflight_tasks: i64,
    pub operation_id: Option<String>,
    pub done: bool,
}

/// The progress of one execution, as PySpark's `Progress` of the execution of a request.
pub(crate) struct ProgressTracker {
    handlers: ProgressHandlers,
    operation_id: Option<String>,
    stages: Vec<StageInfo>,
    inflight_tasks: i64,
}

impl ProgressTracker {
    pub(crate) fn new(session: &SparkSession, operation_id: Option<String>) -> Self {
        Self::with_handlers(session.progress_handlers(), operation_id)
    }

    fn with_handlers(handlers: ProgressHandlers, operation_id: Option<String>) -> Self {
        ProgressTracker {
            handlers,
            operation_id,
            stages: Vec::new(),
            inflight_tasks: 0,
        }
    }

    /// A progress of the server, as PySpark's `Progress.update_ticks`: the handlers are called
    /// only for one that has tasks, whose stages and running tasks the last call, of `done`,
    /// has too.
    pub(crate) fn update(
        &mut self,
        progress: &proto::execute_plan_response::ExecutionProgress,
        operation_id: &str,
    ) {
        if self.operation_id.as_deref().is_none_or(str::is_empty) {
            self.operation_id = Some(operation_id.to_string());
        }
        let stages: Vec<StageInfo> = progress.stages.iter().map(StageInfo::from_proto).collect();
        if stages.iter().map(|stage| stage.num_tasks).sum::<i64>() > 0 {
            self.stages = stages;
            self.inflight_tasks = progress.num_inflight_tasks;
            self.notify(false);
        }
    }

    /// The progress of `response`, if it is one.
    fn observe(&mut self, response: &proto::ExecutePlanResponse) {
        if let Some(proto::execute_plan_response::ResponseType::ExecutionProgress(progress)) =
            &response.response_type
        {
            self.update(progress, &response.operation_id);
        }
    }

    fn notify(&self, done: bool) {
        // Clone handlers out of the registry lock so callbacks can add or remove handlers.
        let handlers: Vec<ProgressHandler> = self
            .handlers
            .lock()
            .unwrap()
            .iter()
            .map(|(_, handler)| Arc::clone(handler))
            .collect();
        if handlers.is_empty() {
            return;
        }
        let update = ProgressUpdate {
            stages: self.stages.clone(),
            inflight_tasks: self.inflight_tasks,
            operation_id: self.operation_id.clone(),
            done,
        };
        // A handler may run a query whose progress calls the handlers again. Preserve the
        // execution callback-depth guard without retaining the session or its listener bus.
        let _callback = CallbackGuard::enter();
        for handler in handlers {
            handler(&update);
        }
    }
}

impl Drop for ProgressTracker {
    /// As PySpark's `Progress.finish` when its execution ends, also of an error. While a panic
    /// unwinds, a handler is not called: one that panics again would abort the process.
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.notify(true);
        }
    }
}

/// The observer of an execution of a session that the core runs: the progress of the execution
/// for the handlers of the session, the updates of its Python accumulators, and the record of the
/// execution as the session's most recent one.
pub(crate) struct SessionObserver {
    handlers: ProgressHandlers,
    accumulators: PythonAccumulators,
    last_execution: ExecutionSlot,
    progress: Option<ProgressTracker>,
    /// The slot of the DataFrame whose execution this is, which records it as its own too.
    dataframe_execution: Option<ExecutionSlot>,
    /// Whether an execution that fails is recorded, as the local iterator records it.
    record_failure: bool,
}

impl SessionObserver {
    /// The observer of an execution about to start, which fails with a `RecursionError` when
    /// the callbacks that run queries are nested too deep on this thread.
    pub(crate) fn start(session: &SparkSession) -> Result<Self> {
        check_callback_depth()?;
        Ok(SessionObserver {
            handlers: session.progress_handlers(),
            accumulators: session.python_accumulators().clone(),
            last_execution: session.last_execution_slot(),
            progress: None,
            dataframe_execution: None,
            record_failure: false,
        })
    }

    /// As [`SessionObserver::start`], for an execution of `dataframe`.
    pub(crate) fn for_dataframe(dataframe: &DataFrame) -> Result<Self> {
        Ok(SessionObserver {
            dataframe_execution: Some(dataframe.execution_slot()),
            ..SessionObserver::start(&dataframe.session)?
        })
    }

    /// As [`SessionObserver::start`], for an execution that the local iterator reads, which
    /// records the execution also when it fails.
    pub(crate) fn for_iterator(session: &SparkSession) -> Result<Self> {
        Ok(SessionObserver {
            record_failure: true,
            ..SessionObserver::start(session)?
        })
    }
}

impl ExecutionObserver for SessionObserver {
    fn on_start(&mut self, operation_id: &str) {
        self.progress = Some(ProgressTracker::with_handlers(
            Arc::clone(&self.handlers),
            Some(operation_id.to_string()),
        ));
    }

    fn on_response(&mut self, response: &proto::ExecutePlanResponse) {
        if let Some(progress) = &mut self.progress {
            progress.observe(response);
        }
    }

    fn on_observed_metrics(&mut self, metrics: &[proto::execute_plan_response::ObservedMetrics]) {
        // The updates of the Python accumulators, such as the results of the UDF profilers.
        self.accumulators.accumulate(metrics);
    }

    fn on_complete(&mut self, info: &ExecutionInfo) {
        if let Some(dataframe_execution) = &self.dataframe_execution {
            *dataframe_execution.lock().unwrap() = Some(info.clone());
        }
        *self.last_execution.lock().unwrap() = Some(info.clone());
    }

    fn on_failure(&mut self, info: &ExecutionInfo, _error: &SparkError) {
        if self.record_failure {
            *self.last_execution.lock().unwrap() = Some(info.clone());
        }
    }
}

/// The responses of an execution, with its progress for the handlers of the session: each
/// `ExecutionProgress` updates it, and it ends when the stream is dropped.
pub(crate) struct ExecutionStream {
    stream: ReattachableResponseStream,
    progress: ProgressTracker,
}

impl ExecutionStream {
    pub(crate) fn new(stream: ReattachableResponseStream, progress: ProgressTracker) -> Self {
        ExecutionStream { stream, progress }
    }

    /// The next response of the execution, as `ReattachableResponseStream::message`.
    pub(crate) async fn message(&mut self) -> Result<Option<proto::ExecutePlanResponse>> {
        let response = self.stream.message().await?;
        if let Some(response) = &response {
            self.progress.observe(response);
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;

    fn progress(
        stages: &[(i64, i64)],
        inflight: i64,
    ) -> proto::execute_plan_response::ExecutionProgress {
        proto::execute_plan_response::ExecutionProgress {
            stages: stages
                .iter()
                .map(|&(stage_id, num_tasks)| {
                    proto::execute_plan_response::execution_progress::StageInfo {
                        stage_id,
                        num_tasks,
                        num_completed_tasks: 1,
                        input_bytes_read: 8,
                        done: false,
                    }
                })
                .collect(),
            num_inflight_tasks: inflight,
        }
    }

    /// As PySpark's `Progress`: the handlers are called with each progress that has tasks, of the
    /// operation of the first response, and once more, as done, with the last one, when the
    /// execution ends.
    #[test]
    fn handlers_are_called_as_the_progress_of_pyspark_calls_them() {
        let session = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let updates = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&updates);
        let id = session.register_progress_handler(move |update| {
            seen.lock().unwrap().push(update.clone());
        });
        {
            let mut tracker = ProgressTracker::new(&session, None);
            tracker.update(&progress(&[(1, 0)], 0), "op");
            tracker.update(&progress(&[(1, 4), (2, 2)], 3), "other");
        }
        let updates = updates.lock().unwrap().clone();
        assert_eq!(updates.len(), 2);
        assert!(!updates[0].done && updates[1].done);
        assert_eq!(updates[0].operation_id.as_deref(), Some("op"));
        assert_eq!(updates[1].stages, updates[0].stages);
        assert_eq!(updates[1].inflight_tasks, 3);
        assert_eq!(
            updates[0].stages[1],
            StageInfo {
                stage_id: 2,
                num_tasks: 2,
                num_completed_tasks: 1,
                num_bytes_read: 8,
                done: false,
            }
        );
        session.remove_progress_handler(id);
    }

    /// A handler that panics on every call, as a user's `unwrap` may: the panic of the update
    /// unwinds out of the tracker, and its drop must not call the handler again while
    /// unwinding, which would abort the process.
    #[test]
    fn a_handler_that_panics_does_not_abort_when_the_tracker_unwinds() {
        let session = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let id = session.register_progress_handler(move |_| {
            seen.fetch_add(1, Ordering::SeqCst);
            panic!("handler panic");
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut tracker = ProgressTracker::new(&session, None);
            tracker.update(&progress(&[(1, 1)], 0), "op");
        }));
        assert!(result.is_err());
        // Only the update called the handler; the done call during unwinding is skipped.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        session.remove_progress_handler(id);
    }

    #[test]
    fn detached_progress_registry_preserves_the_execution_recursion_bound() {
        use spark_connect_core::error::SparkErrorKind;
        use spark_connect_core::runtime::{check_callback_depth, MAX_CALLBACK_DEPTH};

        let session = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let handlers = Arc::downgrade(&session.progress_handlers());
        let calls = Arc::new(AtomicUsize::new(0));
        let failures = Arc::new(AtomicUsize::new(0));
        let seen_calls = Arc::clone(&calls);
        let seen_failures = Arc::clone(&failures);
        session.register_progress_handler(move |update| {
            if update.done {
                return;
            }
            let call = seen_calls.fetch_add(1, Ordering::SeqCst);
            match check_callback_depth() {
                Err(error) => {
                    assert_eq!(error.kind, SparkErrorKind::RecursionError);
                    seen_failures.fetch_add(1, Ordering::SeqCst);
                }
                Ok(()) => {
                    // Bound this test even if the guard regresses; no network execution or
                    // unbounded native recursion is needed to observe the missing guard.
                    assert!(call < MAX_CALLBACK_DEPTH);
                    let nested = ProgressTracker {
                        handlers: handlers.upgrade().expect("registry is alive"),
                        operation_id: Some("nested".to_string()),
                        stages: Vec::new(),
                        inflight_tasks: 0,
                    };
                    nested.notify(false);
                }
            }
        });
        let tracker = ProgressTracker::new(&session, None);
        tracker.notify(false);
        assert_eq!(calls.load(Ordering::SeqCst), MAX_CALLBACK_DEPTH);
        assert_eq!(failures.load(Ordering::SeqCst), 1);
        assert!(check_callback_depth().is_ok());
        session.clear_progress_handlers();
    }
}
