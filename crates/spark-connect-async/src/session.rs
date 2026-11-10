//! A session with a Spark Connect server: its connection, configuration, tags and artifacts.

use std::collections::BTreeMap;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use spark_connect_core::actions;
use spark_connect_core::channel::{
    ChannelBuilder, PARAM_SESSION_ID, PARAM_TOKEN, PARAM_USER_AGENT, PARAM_USER_ID, PARAM_USE_SSL,
};
use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::error::SparkError;
use spark_connect_core::execution::check_tag;
use spark_connect_core::plan::{LogicalPlan, PlanIds};
use spark_connect_core::readwriter::ReadType;
use spark_connect_proto::ExecutePlanRequest;
use tokio::sync::OnceCell;

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::options::RangeSpec;
use crate::progress::{Callback, ProgressHandler, ProgressUpdate};
use crate::sql::Sql;
use crate::tvf::TableFunctions;

/// A session with a Spark Connect server.
///
/// A `Session` is a handle: its clones share the connection and the session on the server. The
/// futures that its methods return own what they need, so they are `Send + 'static` and can be
/// spawned.
#[derive(Clone)]
pub struct Session {
    shared: Arc<Shared>,
    tags: Arc<[String]>,
}

struct Shared {
    client: Arc<SparkConnectClient>,
    plan_ids: Arc<PlanIds>,
    progress_handlers: Arc<[ProgressHandler]>,
    hook: Option<ExecutePlanHook>,
    version: OnceCell<String>,
    stopped: AtomicBool,
}

impl Session {
    /// A builder of a session.
    pub fn builder() -> SessionBuilder {
        SessionBuilder::default()
    }

    fn new(
        client: SparkConnectClient,
        progress_handlers: Vec<ProgressHandler>,
        hook: Option<ExecutePlanHook>,
    ) -> Self {
        Session {
            shared: Arc::new(Shared {
                client: Arc::new(client),
                plan_ids: Arc::new(PlanIds::new()),
                progress_handlers: progress_handlers.into(),
                hook,
                version: OnceCell::new(),
                stopped: AtomicBool::new(false),
            }),
            tags: Arc::from([]),
        }
    }

    pub(crate) fn client(&self) -> Arc<SparkConnectClient> {
        Arc::clone(&self.shared.client)
    }

    pub(crate) fn plan_ids(&self) -> &Arc<PlanIds> {
        &self.shared.plan_ids
    }

    pub(crate) fn progress_handlers(&self) -> Arc<[ProgressHandler]> {
        Arc::clone(&self.shared.progress_handlers)
    }

    /// `request` as the hook of the session's executions returns it, with the operation id of
    /// `request`.
    pub(crate) fn hooked(&self, request: ExecutePlanRequest) -> Result<ExecutePlanRequest> {
        let Some(hook) = &self.shared.hook else {
            return Ok(request);
        };
        let operation_id = request.operation_id.clone();
        let _callback = Callback::enter();
        let mut request = hook(request)?;
        request.operation_id = operation_id;
        Ok(request)
    }

    /// The id of the session on the server.
    pub fn session_id(&self) -> &str {
        self.shared.client.session_id()
    }

    /// The runtime configuration of the session.
    pub fn conf(&self) -> RuntimeConfig {
        RuntimeConfig {
            client: self.client(),
        }
    }

    /// A handle to the same session whose operations also carry `tag`, so that
    /// [`Session::interrupt_tag`] can interrupt them. This handle's tags are unchanged.
    ///
    /// Fails with [`ErrorKind::InvalidArgument`](crate::ErrorKind::InvalidArgument) for an empty
    /// tag or one with a comma.
    pub fn with_tag(&self, tag: impl Into<String>) -> Result<Session> {
        let tag = tag.into();
        check_tag(&tag)?;
        let mut tags = self.tags.to_vec();
        if !tags.contains(&tag) {
            tags.push(tag);
        }
        Ok(Session {
            shared: Arc::clone(&self.shared),
            tags: tags.into(),
        })
    }

    /// The tags of this handle's operations.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// A DataFrame of the integers of `range`, in a column `id` of type `BIGINT`: `range(10)`
    /// is 0 to 9, `range(5..10)` is 5 to 9, and [`RangeSpec`] sets a step and the number of
    /// partitions.
    ///
    /// ```no_run
    /// # fn run(session: spark_connect_async::Session) {
    /// use spark_connect_async::options::RangeSpec;
    ///
    /// let first_ten = session.range(10);
    /// let even = session.range(RangeSpec::new(0..100).step(2).partitions(4));
    /// # }
    /// ```
    pub fn range(&self, range: impl Into<RangeSpec>) -> DataFrame {
        let range = range.into();
        DataFrame::new(
            self.clone(),
            LogicalPlan::Range {
                start: range.start,
                end: range.end,
                step: range.step,
                num_partitions: range.partitions,
            },
        )
    }

    /// A DataFrame of the table or view `name`, which may be qualified by its catalog and
    /// schema.
    pub fn table(&self, name: impl Into<String>) -> DataFrame {
        DataFrame::new(
            self.clone(),
            LogicalPlan::Read {
                read_type: ReadType::NamedTable {
                    table_name: name.into(),
                    options: Default::default(),
                },
                is_streaming: false,
            },
        )
    }

    /// A SQL statement, with the arguments and DataFrames that [`Sql`] adds. Awaiting it sends
    /// the statement: a command, such as `CREATE VIEW`, runs then, and a query returns a
    /// DataFrame that runs at its actions.
    ///
    /// ```no_run
    /// # async fn run(session: spark_connect_async::Session) -> spark_connect_async::Result<()> {
    /// let adults = session
    ///     .sql("SELECT * FROM people WHERE age >= :age")
    ///     .named_arg("age", 18)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn sql(&self, query: impl Into<String>) -> Sql {
        Sql::new(self.clone(), query.into())
    }

    /// The table-valued functions of the session, such as `explode`.
    pub fn tvf(&self) -> TableFunctions {
        TableFunctions::new(self.clone())
    }

    /// Interrupts every running operation of the session, and returns the ids of those it
    /// interrupted.
    pub fn interrupt_all(&self) -> impl Future<Output = Result<Vec<String>>> + Send + 'static {
        let client = self.client();
        async move { Ok(client.interrupt_all().await?) }
    }

    /// Interrupts the running operations of the session that carry `tag`, and returns the ids of
    /// those it interrupted.
    pub fn interrupt_tag(
        &self,
        tag: impl Into<String>,
    ) -> impl Future<Output = Result<Vec<String>>> + Send + 'static {
        let client = self.client();
        let tag = tag.into();
        async move { Ok(client.interrupt_tag(&tag).await?) }
    }

    /// Interrupts the operation `operation_id` if it is running, and returns the ids of those it
    /// interrupted.
    pub fn interrupt_operation(
        &self,
        operation_id: impl Into<String>,
    ) -> impl Future<Output = Result<Vec<String>>> + Send + 'static {
        let client = self.client();
        let operation_id = operation_id.into();
        async move { Ok(client.interrupt_operation(&operation_id).await?) }
    }

    /// Uploads local files to the session, as [`ArtifactKind::Classpath`] artifacts unless
    /// [`AddArtifacts::kind`] says otherwise. A path may be a `file:` URI.
    pub fn add_artifacts(
        &self,
        paths: impl IntoIterator<Item = impl Into<String>>,
    ) -> AddArtifacts {
        AddArtifacts {
            client: self.client(),
            paths: paths.into_iter().map(Into::into).collect(),
            kind: ArtifactKind::default(),
        }
    }

    /// The Spark version of the server. Only the first successful call asks the server.
    pub fn version(&self) -> impl Future<Output = Result<String>> + Send + 'static {
        let shared = Arc::clone(&self.shared);
        async move {
            let version = shared
                .version
                .get_or_try_init(|| actions::session::spark_version(Arc::clone(&shared.client)))
                .await?;
            Ok(version.clone())
        }
    }

    /// Releases the session on the server and closes it: later operations of every handle to it
    /// fail. Stopping a stopped session does nothing.
    pub fn stop(&self) -> impl Future<Output = Result<()>> + Send + 'static {
        let shared = Arc::clone(&self.shared);
        async move {
            if shared.stopped.swap(true, Ordering::SeqCst) {
                return Ok(());
            }
            // The client closes also when this future is dropped before the server answers.
            let _close = CloseOnDrop(&shared.client);
            if shared.client.is_closed() {
                return Ok(());
            }
            Ok(shared.client.release_session().await?)
        }
    }

    /// Whether the session is stopped, by [`Session::stop`] or by the server.
    pub fn is_stopped(&self) -> bool {
        self.shared.stopped.load(Ordering::SeqCst) || self.shared.client.is_closed()
    }
}

struct CloseOnDrop<'a>(&'a SparkConnectClient);

impl Drop for CloseOnDrop<'_> {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// A builder of a [`Session`].
///
/// The connection string of [`SessionBuilder::remote`] has the form
/// `sc://host[:port][/;key=value;...]`. Its parameters set the token, TLS, the user id, the
/// session id and the user agent, as the methods of the builder do; any other parameter is sent
/// with each request as a gRPC header.
#[derive(Clone, Default)]
pub struct SessionBuilder {
    remote: Option<String>,
    params: Vec<(&'static str, String)>,
    configs: Vec<(String, String)>,
    progress_handlers: Vec<ProgressHandler>,
    hook: Option<ExecutePlanHook>,
}

/// A hook of the executions of a session, which [`SessionBuilder::execute_plan_hook`] sets.
pub type ExecutePlanHook =
    Arc<dyn Fn(ExecutePlanRequest) -> Result<ExecutePlanRequest> + Send + Sync>;

impl SessionBuilder {
    /// The connection string of the server.
    pub fn remote(mut self, url: impl Into<String>) -> Self {
        self.remote = Some(url.into());
        self
    }

    /// The bearer token to authenticate with. A token turns on TLS, except to `localhost`.
    pub fn token(self, token: impl Into<String>) -> Self {
        self.param(PARAM_TOKEN, token.into())
    }

    /// Whether to connect with TLS.
    pub fn use_ssl(self, use_ssl: bool) -> Self {
        self.param(PARAM_USE_SSL, use_ssl.to_string())
    }

    /// The user id of the session.
    pub fn user_id(self, user_id: impl Into<String>) -> Self {
        self.param(PARAM_USER_ID, user_id.into())
    }

    /// The id of the session on the server, a UUID, to connect to an existing session.
    pub fn session_id(self, session_id: impl Into<String>) -> Self {
        self.param(PARAM_SESSION_ID, session_id.into())
    }

    /// The user agent that the client reports to the server.
    pub fn user_agent(self, user_agent: impl Into<String>) -> Self {
        self.param(PARAM_USER_AGENT, user_agent.into())
    }

    /// A runtime configuration that [`SessionBuilder::connect`] sets on the session.
    pub fn config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.configs.push((key.into(), value.into()));
        self
    }

    /// A handler of the progress of the session's executions. It is called with each update
    /// of the server's progress of an execution that has tasks, and once more, done, when the
    /// execution ends; it runs on the thread that polls the execution's future or stream.
    ///
    /// A handler that runs an execution of its own, which calls the handlers again, may nest
    /// that way twice on a thread; a deeper execution fails with
    /// [`ErrorKind::Internal`](crate::ErrorKind::Internal).
    pub fn on_progress(
        mut self,
        handler: impl Fn(&ProgressUpdate) + Send + Sync + 'static,
    ) -> Self {
        self.progress_handlers.push(Arc::new(handler));
        self
    }

    /// A hook that gets each `ExecutePlanRequest` of the session before it is sent, and returns
    /// the request to send instead; the request keeps its operation id. An error of the hook is
    /// the error of the execution. The hook runs on the thread that polls the execution, and
    /// nests as a progress handler does. A later call replaces the hook.
    pub fn execute_plan_hook(
        mut self,
        hook: impl Fn(ExecutePlanRequest) -> Result<ExecutePlanRequest> + Send + Sync + 'static,
    ) -> Self {
        self.hook = Some(Arc::new(hook));
        self
    }

    fn param(mut self, key: &'static str, value: String) -> Self {
        self.params.push((key, value));
        self
    }

    /// Connects to the server, and sets the configurations of [`SessionBuilder::config`]. If the
    /// server rejects one, the session is released on the server, and the error returned.
    pub async fn connect(self) -> Result<Session> {
        let url = self.remote.ok_or_else(|| {
            Error::from(SparkError::value_msg(
                "the connection string is missing: call remote(url) before connect()",
            ))
        })?;
        let mut channel = ChannelBuilder::parse(&url)?;
        for (key, value) in &self.params {
            channel.set(key, value);
        }
        let client = SparkConnectClient::connect(&channel).await?;
        let session = Session::new(client, self.progress_handlers, self.hook);
        if !self.configs.is_empty() {
            if let Err(error) = session.client().set_configs(&self.configs, false).await {
                let _ = session.stop().await;
                return Err(error.into());
            }
        }
        Ok(session)
    }
}

/// The runtime configuration of a session, which the server keeps.
#[derive(Clone)]
pub struct RuntimeConfig {
    client: Arc<SparkConnectClient>,
}

impl RuntimeConfig {
    /// Sets `key` to `value`, and returns the server's warnings, such as of a deprecated key.
    pub fn set(
        &self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> impl Future<Output = Result<Vec<String>>> + Send + 'static {
        let client = Arc::clone(&self.client);
        let (key, value) = (key.into(), value.into());
        async move { Ok(client.set_config(&key, &value).await?) }
    }

    /// Sets each key to its value, and returns the server's warnings.
    pub fn set_all(
        &self,
        pairs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> impl Future<Output = Result<Vec<String>>> + Send + 'static {
        let client = Arc::clone(&self.client);
        let pairs: Vec<(String, String)> = pairs
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        async move { Ok(client.set_configs(&pairs, false).await?) }
    }

    /// The value of `key`, or `None` if it has none.
    pub fn get(
        &self,
        key: impl Into<String>,
    ) -> impl Future<Output = Result<Option<String>>> + Send + 'static {
        let client = Arc::clone(&self.client);
        let key = key.into();
        async move {
            let values = client.get_config_with_defaults(&[(&key, None)]).await?;
            Ok(values.into_iter().next().flatten())
        }
    }

    /// Every key that has a value, with its value.
    pub fn get_all(
        &self,
    ) -> impl Future<Output = Result<BTreeMap<String, String>>> + Send + 'static {
        let client = Arc::clone(&self.client);
        async move { Ok(client.get_configs_all().await?.into_iter().collect()) }
    }

    /// Removes the value of `key`, and returns the server's warnings.
    pub fn unset(
        &self,
        key: impl Into<String>,
    ) -> impl Future<Output = Result<Vec<String>>> + Send + 'static {
        let client = Arc::clone(&self.client);
        let key = key.into();
        async move { Ok(client.unset_config(&key).await?) }
    }

    /// Whether `key` can be set on a running session.
    pub fn is_modifiable(
        &self,
        key: impl Into<String>,
    ) -> impl Future<Output = Result<bool>> + Send + 'static {
        let client = Arc::clone(&self.client);
        let key = key.into();
        async move { Ok(client.is_config_modifiable(&key).await?) }
    }
}

/// How the server uses an uploaded artifact.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArtifactKind {
    /// A `.jar`, added to the classpath, or a `.class` file, added to the classes.
    #[default]
    Classpath,
    /// Any file, added to the files of the session.
    File,
    /// A `.zip`, `.jar`, `.tar.gz`, `.tgz` or `.tar`, which the server extracts into the
    /// directory that a `#name` fragment of its path names.
    Archive,
}

/// The upload of artifacts that [`Session::add_artifacts`] returns: await it to upload them.
pub struct AddArtifacts {
    client: Arc<SparkConnectClient>,
    paths: Vec<String>,
    kind: ArtifactKind,
}

impl AddArtifacts {
    /// How the server uses the artifacts.
    pub fn kind(mut self, kind: ArtifactKind) -> Self {
        self.kind = kind;
        self
    }
}

impl IntoFuture for AddArtifacts {
    type Output = Result<()>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<()>> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        let AddArtifacts {
            client,
            paths,
            kind,
        } = self;
        Box::pin(async move {
            let paths: Vec<&str> = paths.iter().map(String::as_str).collect();
            let archive = kind == ArtifactKind::Archive;
            let file = kind == ArtifactKind::File;
            Ok(client.add_artifacts(&paths, false, archive, file).await?)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send_static<T: Send + 'static>(_: T) {}

    #[test]
    fn the_session_types_and_their_futures_are_send() {
        assert_send_sync::<Session>();
        assert_send_sync::<SessionBuilder>();
        assert_send_sync::<RuntimeConfig>();
        assert_send_sync::<AddArtifacts>();
        assert_send_sync::<Error>();
        assert_send_sync::<crate::DataFrame>();
        assert_send_sync::<crate::sql::Sql>();
        assert_send_sync::<TableFunctions>();
        assert_send_sync::<crate::tvf::TableArg>();
        assert_send_sync::<crate::tvf::PartitionedTableArg>();
        assert_send_sync::<crate::actions::Show>();
        assert_send_sync::<crate::Observation>();
        assert_send_sync::<crate::stream::ExecutionInfo>();
        assert_send_sync::<crate::group::GroupedData<crate::group::ByColumns>>();
        assert_send_sync::<crate::group::GroupedData>();
        assert_send_sync::<crate::stat::StatFunctions>();
        assert_send_sync::<crate::stat::NaFunctions>();
        assert_send_sync::<crate::window::Window>();
        assert_send_sync::<crate::options::Pivot>();
        assert_send_sync::<crate::options::FillValue>();
        assert_send_sync::<crate::options::Replace>();
        fn assert_send<T: Send>() {}
        assert_send::<crate::RecordBatchStream>();
        // Type-checked, never called: it would need a server.
        let _ = |session: Session| {
            let conf = session.conf();
            assert_send_static(Session::builder().connect());
            assert_send_static(session.interrupt_all());
            assert_send_static(session.interrupt_tag("t"));
            assert_send_static(session.interrupt_operation("op"));
            assert_send_static(session.add_artifacts(["a.jar"]).into_future());
            assert_send_static(session.version());
            assert_send_static(session.stop());
            assert_send_static(session.sql("SELECT 1").arg(1).into_future());
            let df = session.range(3);
            assert_send_static(df.stream());
            assert_send_static(df.stream_as::<i64>());
            assert_send_static(df.collect());
            assert_send_static(df.collect_as::<(i64,)>());
            assert_send_static(df.count());
            assert_send_static(df.show().into_future());
            assert_send_static(df.schema());
            assert_send_static(df.explain(Default::default()));
            assert_send_static(df.checkpoint(Default::default()));
            assert_send_static(df.create_temp_view("v", Default::default()));
            assert_send_static(df.group_by(["id"]).sum(["id"]));
            assert_send_static(df.stat().corr("a", "b"));
            assert_send_static(df.stat().approx_quantile(["a"], [0.5], 0.1));
            assert_send_static(conf.set("k", "v"));
            assert_send_static(conf.get("k"));
            assert_send_static(conf.get_all());
            assert_send_static(conf.unset("k"));
            assert_send_static(conf.is_modifiable("k"));
        };
    }

    #[tokio::test]
    async fn a_tag_extends_a_new_handle_and_an_invalid_one_is_rejected() {
        let session = Session::builder()
            .remote("sc://localhost:1")
            .connect()
            .await
            .unwrap();
        let etl = session.with_tag("etl").unwrap();
        let both = etl.with_tag("daily").unwrap().with_tag("etl").unwrap();
        assert!(session.tags().is_empty());
        assert_eq!(etl.tags(), ["etl"]);
        assert_eq!(both.tags(), ["etl", "daily"]);
        for invalid in ["", "a,b"] {
            let error = session.with_tag(invalid).err().unwrap();
            assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{invalid:?}");
        }
    }
}
