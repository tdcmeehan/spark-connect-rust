//! The progress of executions, which [`SessionBuilder::on_progress`] reports, and the bound on
//! how deeply callbacks nest.
//!
//! [`SessionBuilder::on_progress`]: crate::SessionBuilder::on_progress

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use spark_connect_core::actions::ExecutionObserver;
use spark_connect_core::error::SparkError;
use spark_connect_core::execution::ExecutionInfo;
use spark_connect_proto as proto;

/// The progress of one stage of an execution.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StageInfo {
    /// The id of the stage.
    pub stage_id: i64,
    /// The number of tasks of the stage.
    pub num_tasks: i64,
    /// The number of its tasks that have completed.
    pub num_completed_tasks: i64,
    /// The number of bytes that its tasks have read.
    pub num_bytes_read: i64,
    /// Whether the stage has completed.
    pub done: bool,
}

/// The progress of an execution, as a progress handler gets it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProgressUpdate {
    /// The stages of the execution so far.
    pub stages: Vec<StageInfo>,
    /// The number of tasks that are running.
    pub inflight_tasks: i64,
    /// The id of the operation.
    pub operation_id: String,
    /// Whether the execution has ended; the last update of an execution is done.
    pub done: bool,
}

pub(crate) type ProgressHandler = Arc<dyn Fn(&ProgressUpdate) + Send + Sync>;

/// How deeply a progress handler or a hook of an execution may run executions of its own on
/// the thread that runs it.
const MAX_CALLBACK_DEPTH: usize = 2;

thread_local! {
    static CALLBACK_DEPTH: Cell<usize> = const { Cell::new(0) };
    /// Whether an execution of the outermost callback running on this thread failed for the
    /// depth: its other nested executions fail too, so that a callback that runs several
    /// executions does not try each of them at every depth.
    static CALLBACK_DEPTH_EXCEEDED: Cell<bool> = const { Cell::new(false) };
}

/// A callback that runs on this thread, while it is alive.
pub(crate) struct Callback;

impl Callback {
    pub(crate) fn enter() -> Callback {
        CALLBACK_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Callback
    }
}

impl Drop for Callback {
    fn drop(&mut self) {
        let depth = CALLBACK_DEPTH.with(|depth| {
            depth.set(depth.get() - 1);
            depth.get()
        });
        if depth == 0 {
            CALLBACK_DEPTH_EXCEEDED.with(|exceeded| exceeded.set(false));
        }
    }
}

/// Fails when the callbacks that run on this thread are nested as deeply as they may be.
pub(crate) fn check_callback_depth() -> crate::Result<()> {
    let too_deep = CALLBACK_DEPTH.with(Cell::get) >= MAX_CALLBACK_DEPTH;
    if too_deep || CALLBACK_DEPTH_EXCEEDED.with(Cell::get) {
        CALLBACK_DEPTH_EXCEEDED.with(|exceeded| exceeded.set(true));
        return Err(SparkError::recursion_msg(
            "callbacks nest too deeply: a progress handler or a hook of an execution runs an \
             execution whose callbacks run another",
        )
        .into());
    }
    Ok(())
}

/// What a session does while an execution runs: it calls its progress handlers with each
/// progress of the server that has tasks, and once more, as done, when the execution ends.
pub(crate) struct Observer {
    handlers: Arc<[ProgressHandler]>,
    operation_id: String,
    stages: Vec<StageInfo>,
    inflight_tasks: i64,
    done: bool,
    abandoned: Arc<AtomicBool>,
}

impl Observer {
    /// An observer of the operation `operation_id`, which reports nothing once `abandoned` is
    /// set: the caller stopped waiting for the execution before the server accepted it.
    pub(crate) fn new(
        handlers: Arc<[ProgressHandler]>,
        operation_id: String,
        abandoned: Arc<AtomicBool>,
    ) -> Self {
        Observer {
            handlers,
            operation_id,
            stages: Vec::new(),
            inflight_tasks: 0,
            done: false,
            abandoned,
        }
    }

    fn notify(&mut self, done: bool) {
        if self.done || self.handlers.is_empty() || self.abandoned.load(Ordering::Acquire) {
            return;
        }
        self.done = done;
        let update = ProgressUpdate {
            stages: self.stages.clone(),
            inflight_tasks: self.inflight_tasks,
            operation_id: self.operation_id.clone(),
            done,
        };
        let _callback = Callback::enter();
        for handler in self.handlers.iter() {
            handler(&update);
        }
    }
}

impl ExecutionObserver for Observer {
    fn on_response(&mut self, response: &proto::ExecutePlanResponse) {
        let Some(proto::execute_plan_response::ResponseType::ExecutionProgress(progress)) =
            &response.response_type
        else {
            return;
        };
        if self.handlers.is_empty()
            || progress
                .stages
                .iter()
                .map(|stage| stage.num_tasks)
                .sum::<i64>()
                <= 0
        {
            return;
        }
        self.stages = progress
            .stages
            .iter()
            .map(|stage| StageInfo {
                stage_id: stage.stage_id,
                num_tasks: stage.num_tasks,
                num_completed_tasks: stage.num_completed_tasks,
                num_bytes_read: stage.input_bytes_read,
                done: stage.done,
            })
            .collect();
        self.inflight_tasks = progress.num_inflight_tasks;
        self.notify(false);
    }

    fn on_complete(&mut self, _info: &ExecutionInfo) {
        self.notify(true);
    }

    fn on_failure(&mut self, _info: &ExecutionInfo, _error: &SparkError) {
        self.notify(true);
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.notify(true);
        }
    }
}
