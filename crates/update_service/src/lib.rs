//! Update Service Temporal worker settings and runner.
//!
//! [`UpdateServiceSettings`] reads two groups of settings:
//!
//! - The Temporal connection, through the Temporal SDK's environment configuration:
//!   `TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE`, `TEMPORAL_API_KEY`, the `TEMPORAL_TLS*`
//!   variables, or a profile in a `temporal.toml` file. An API key or a TLS setting turns
//!   TLS on, and an address without a scheme gets `https://` when TLS is on, `http://`
//!   otherwise.
//! - Worker Versioning, from `WORKER_VERSIONING_ENABLED`, `WORKER_VERSIONING_BEHAVIOR`,
//!   `WORKER_DEPLOYMENT_NAME` and `WORKER_BUILD_ID`.
//!
//! [`run_workers`] connects one client, runs one worker per task queue the Update Service
//! owns ([`list_worker_queue_specs`]), and stops every worker when the shutdown future
//! resolves or any worker stops. Each worker drains its in-flight tasks before
//! [`run_workers`] returns.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::pin;

use anyhow::{Context as _, anyhow, bail, ensure};
use futures::future::join_all;
use temporal_workflows::{
    DEFAULT_TEMPORAL_NAMESPACE, Provider, REFRESH_TRIGGER_SEARCH_ATTRIBUTE,
    UPDATE_SERVICE_TASK_QUEUE, WorkerQueueSpec, build_spike_worker_options,
};
use temporalio_client::tonic::{Code, Request};
use temporalio_client::{Client, ClientOptions, ConnectionOptions};
use temporalio_common::envconfig::{LoadClientConfigProfileOptions, load_client_config_profile};
use temporalio_common::protos::temporal::api::enums::v1::IndexedValueType;
use temporalio_common::protos::temporal::api::operatorservice::v1::{
    AddSearchAttributesRequest, ListSearchAttributesRequest,
};
use temporalio_common::protos::temporal::api::workflowservice::v1::DescribeNamespaceRequest;
use temporalio_common::worker::{VersioningBehavior, WorkerDeploymentVersion};
use temporalio_sdk::runtime::RuntimeOptions;
use temporalio_sdk::{Runtime, Worker, WorkerOptions};
use tokio::sync::watch;
use tracing::{Level, event};

/// Settings for connecting the Update Service workers to Temporal.
///
/// `Debug` output redacts the API key.
#[derive(Clone)]
pub struct UpdateServiceSettings {
    /// Temporal server connection, including TLS and API key when configured.
    pub connection: ConnectionOptions,
    /// Temporal namespace the workers poll.
    pub namespace: String,
    /// Worker Deployment name and build ID the workers report.
    pub deployment_version: WorkerDeploymentVersion,
    /// Default versioning behavior, or `None` when Worker Versioning is off.
    pub versioning_behavior: Option<VersioningBehavior>,
}

impl UpdateServiceSettings {
    /// Loads settings from process environment variables and `temporal.toml`.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Self::from_vars`], or if
    /// the `temporal.toml` file cannot be read.
    pub fn from_env() -> anyhow::Result<Self> {
        Self::load(LoadClientConfigProfileOptions::default(), None, |key| {
            std::env::var(key).ok()
        })
    }

    /// Loads settings from `vars` only, ignoring the process environment.
    ///
    /// No `temporal.toml` file is read, so the result depends only on `vars`.
    /// Blank values count as unset, except `TEMPORAL_ADDRESS`, which the
    /// Temporal SDK rejects when blank.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// - `TEMPORAL_ADDRESS` is blank or not a `host:port` address or URL without
    ///   a path, or the TLS variables are
    ///   inconsistent (a client certificate without its key);
    /// - `WORKER_VERSIONING_ENABLED` is not one of `true`, `false`, `1`, `0`,
    ///   `yes` or `no`;
    /// - `WORKER_VERSIONING_BEHAVIOR` is not `pinned` or `auto_upgrade`, or is
    ///   set while Worker Versioning is off;
    /// - Worker Versioning is on and `WORKER_BUILD_ID` is unset, since every
    ///   deploy needs its own build ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use update_service::UpdateServiceSettings;
    ///
    /// let vars = HashMap::from([("TEMPORAL_ADDRESS".to_owned(), "temporal:7233".to_owned())]);
    /// let settings = UpdateServiceSettings::from_vars(&vars)?;
    /// assert_eq!(settings.connection.target.as_str(), "http://temporal:7233/");
    /// assert_eq!(settings.namespace, "echora-dev");
    /// assert_eq!(settings.versioning_behavior, None);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn from_vars(vars: &HashMap<String, String>) -> anyhow::Result<Self> {
        let profile_options = LoadClientConfigProfileOptions::builder()
            .disable_file(true)
            .build();
        Self::load(profile_options, Some(vars), |key| vars.get(key).cloned())
    }

    fn load(
        profile_options: LoadClientConfigProfileOptions,
        env_vars: Option<&HashMap<String, String>>,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> anyhow::Result<Self> {
        let mut profile =
            load_client_config_profile(profile_options, env_vars).map_err(|config_error| {
                anyhow!("failed to load Temporal client configuration: {config_error}")
            })?;
        let namespace = profile
            .namespace
            .take()
            .filter(|namespace| !namespace.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_TEMPORAL_NAMESPACE.to_owned());
        let connection = ConnectionOptions::try_from(profile).map_err(|config_error| {
            anyhow!("invalid Temporal connection settings: {config_error}")
        })?;
        // The SDK retries an address it cannot parse with `http://` in front, which turns
        // input such as `http://` into a target with a path. A gRPC target has no path.
        ensure!(
            connection.target.path() == "/",
            "invalid Temporal connection settings: TEMPORAL_ADDRESS must be host:port or a URL \
             without a path, got {}",
            connection.target
        );

        let versioning_enabled =
            parse_versioning_enabled(find_non_blank(&lookup, "WORKER_VERSIONING_ENABLED"))?;
        let versioning_behavior = resolve_versioning_behavior(
            versioning_enabled,
            find_non_blank(&lookup, "WORKER_VERSIONING_BEHAVIOR"),
        )?;
        let build_id = match find_non_blank(&lookup, "WORKER_BUILD_ID") {
            Some(build_id) => build_id,
            None if versioning_enabled => {
                bail!("WORKER_BUILD_ID must be set when WORKER_VERSIONING_ENABLED is true")
            }
            None => env!("CARGO_PKG_VERSION").to_owned(),
        };
        let deployment_name = find_non_blank(&lookup, "WORKER_DEPLOYMENT_NAME")
            .unwrap_or_else(|| UPDATE_SERVICE_TASK_QUEUE.to_owned());

        Ok(Self {
            connection,
            namespace,
            deployment_version: WorkerDeploymentVersion::builder()
                .deployment_name(deployment_name)
                .build_id(build_id)
                .build(),
            versioning_behavior,
        })
    }
}

impl fmt::Debug for UpdateServiceSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UpdateServiceSettings")
            .field("target", &self.connection.target.as_str())
            .field("tls", &self.connection.tls_options.is_some())
            .field(
                "api_key",
                &self.connection.api_key.as_ref().map(|_| "<redacted>"),
            )
            .field("namespace", &self.namespace)
            .field("deployment_version", &self.deployment_version)
            .field("versioning_behavior", &self.versioning_behavior)
            .finish_non_exhaustive()
    }
}

/// Lists every task queue the Update Service polls, `update-service` first.
///
/// `agent-service` is left out, since `agent_service` polls it.
///
/// # Examples
///
/// ```
/// use temporal_workflows::{Provider, WorkerQueueSpec};
/// use update_service::list_worker_queue_specs;
///
/// let specs = list_worker_queue_specs();
/// assert_eq!(specs.len(), 1 + Provider::ALL.len());
/// assert_eq!(specs[0], WorkerQueueSpec::UpdateService);
/// ```
#[must_use]
pub fn list_worker_queue_specs() -> Vec<WorkerQueueSpec> {
    let mut specs = Vec::with_capacity(1 + Provider::ALL.len());
    specs.push(WorkerQueueSpec::UpdateService);
    specs.extend(Provider::ALL.into_iter().map(WorkerQueueSpec::Provider));
    specs
}

/// Builds worker options for every task queue the Update Service polls.
///
/// The options come back in the order of [`list_worker_queue_specs`].
///
/// # Errors
///
/// Returns an error if Temporal workflow registration fails for any queue.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use temporal_workflows::Provider;
/// use update_service::{UpdateServiceSettings, build_all_worker_options};
///
/// let settings = UpdateServiceSettings::from_vars(&HashMap::new())?;
/// let options = build_all_worker_options(&settings)?;
/// assert_eq!(options.len(), 1 + Provider::ALL.len());
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn build_all_worker_options(
    settings: &UpdateServiceSettings,
) -> anyhow::Result<Vec<WorkerOptions>> {
    list_worker_queue_specs()
        .into_iter()
        .map(|spec| {
            build_spike_worker_options(
                spec,
                settings.deployment_version.clone(),
                settings.versioning_behavior,
            )
            .with_context(|| format!("failed to build worker options for {spec}"))
        })
        .collect()
}

/// Runs one worker per Update Service task queue until shutdown.
///
/// Every worker shares one Temporal client. When `shutdown` resolves, or any
/// worker stops on its own, every worker stops polling, finishes its
/// in-flight tasks, and releases its slots on the server before this
/// function returns. A `shutdown` that resolves while the workers are still
/// starting ends startup at once and returns `Ok`.
///
/// # Errors
///
/// Returns an error if the Temporal connection fails, the namespace does not
/// exist, the `RefreshTrigger` search attribute has the wrong type or cannot be
/// created, a worker cannot be created, or any worker fails while running.
///
/// # Examples
///
/// ```no_run
/// use update_service::{UpdateServiceSettings, run_workers};
///
/// # async fn example() -> anyhow::Result<()> {
/// let settings = UpdateServiceSettings::from_env()?;
/// run_workers(&settings, async {
///     let _ = tokio::signal::ctrl_c().await;
/// })
/// .await?;
/// # Ok(())
/// # }
/// ```
pub async fn run_workers<F>(settings: &UpdateServiceSettings, shutdown: F) -> anyhow::Result<()>
where
    F: Future<Output = ()>,
{
    let specs = list_worker_queue_specs();
    let mut shutdown = pin!(shutdown);
    let runtime = Runtime::from_current_tokio(RuntimeOptions::default())?;
    let workers = tokio::select! {
        started_workers = start_workers(settings, &runtime) => started_workers?,
        () = &mut shutdown => {
            event!(
                name: "update_service.workers.startup_cancelled",
                Level::INFO,
                "shutdown signal received before Temporal workers started",
            );
            return Ok(());
        }
    };

    event!(
        name: "update_service.workers.starting",
        Level::INFO,
        temporal.address = %settings.connection.target,
        temporal.namespace = %settings.namespace,
        worker.deployment_name = %settings.deployment_version.deployment_name,
        worker.build_id = %settings.deployment_version.build_id,
        worker.count = workers.len(),
        "starting {{worker.count}} Temporal workers on {{temporal.address}}",
    );

    let (stop_sender, stop_receiver) = watch::channel(false);
    let worker_runs = workers.into_iter().map(|worker| {
        let mut worker_stop_receiver = stop_receiver.clone();
        let stop_sender = &stop_sender;
        async move {
            let run_outcome = worker
                .run_until(async move {
                    let _ = worker_stop_receiver
                        .wait_for(|is_stopping| *is_stopping)
                        .await;
                })
                .await;
            stop_sender.send_replace(true);
            run_outcome
        }
    });
    let mut shutdown_stop_receiver = stop_receiver.clone();
    let stop_on_shutdown = async {
        tokio::select! {
            () = &mut shutdown => {
                event!(
                    name: "update_service.workers.draining",
                    Level::INFO,
                    "shutdown signal received; draining Temporal workers",
                );
                stop_sender.send_replace(true);
            }
            _ = shutdown_stop_receiver.wait_for(|is_stopping| *is_stopping) => {}
        }
    };

    let (run_outcomes, ()) = tokio::join!(join_all(worker_runs), stop_on_shutdown);
    for (spec, run_outcome) in specs.into_iter().zip(run_outcomes) {
        run_outcome.with_context(|| format!("Temporal worker for {spec} failed"))?;
    }
    Ok(())
}

/// Connects to Temporal, checks the namespace and search attribute, and creates the workers.
async fn start_workers(
    settings: &UpdateServiceSettings,
    runtime: &Runtime,
) -> anyhow::Result<Vec<Worker>> {
    let worker_options = build_all_worker_options(settings)?;
    let client = Client::connect(
        settings.connection.clone(),
        ClientOptions::new(settings.namespace.clone()).build(),
    )
    .await
    .with_context(|| {
        format!(
            "failed to connect to Temporal at {}",
            settings.connection.target
        )
    })?;
    // A Temporal worker keeps polling a namespace that does not exist without
    // ever failing, so a misspelled namespace is caught here instead.
    client
        .connection()
        .workflow_service()
        .describe_namespace(Request::new(DescribeNamespaceRequest {
            namespace: settings.namespace.clone(),
            ..DescribeNamespaceRequest::default()
        }))
        .await
        .with_context(|| format!("Temporal namespace {} is not available", settings.namespace))?;
    register_refresh_trigger_attribute(&client, &settings.namespace).await?;
    worker_options
        .into_iter()
        .map(|options| Worker::new(runtime, client.clone(), options))
        .collect::<Result<Vec<_>, _>>()
        .context("failed to create Temporal workers")
}

/// Creates the `RefreshTrigger` keyword search attribute when the namespace lacks it.
///
/// Temporal's databases can be lost, so the Update Service recreates what its
/// workflows need instead of relying on server setup. A second replica creating
/// the attribute at the same moment gets `ALREADY_EXISTS`, which counts as done.
async fn register_refresh_trigger_attribute(
    client: &Client,
    namespace: &str,
) -> anyhow::Result<()> {
    let mut operator_service = client.connection().operator_service();
    let listed_attributes = operator_service
        .list_search_attributes(Request::new(ListSearchAttributesRequest {
            namespace: namespace.to_owned(),
        }))
        .await
        .with_context(|| format!("failed to list search attributes of {namespace}"))?
        .into_inner();
    let keyword_type = i32::from(IndexedValueType::Keyword);
    match listed_attributes
        .custom_attributes
        .get(REFRESH_TRIGGER_SEARCH_ATTRIBUTE)
    {
        Some(&attribute_type) if attribute_type == keyword_type => return Ok(()),
        Some(&attribute_type) => bail!(
            "search attribute {REFRESH_TRIGGER_SEARCH_ATTRIBUTE} in {namespace} has type \
             {:?}, expected Keyword",
            IndexedValueType::try_from(attribute_type).unwrap_or(IndexedValueType::Unspecified)
        ),
        None => {}
    }
    let add_outcome = operator_service
        .add_search_attributes(Request::new(AddSearchAttributesRequest {
            search_attributes: HashMap::from([(
                REFRESH_TRIGGER_SEARCH_ATTRIBUTE.to_owned(),
                keyword_type,
            )]),
            namespace: namespace.to_owned(),
        }))
        .await;
    match add_outcome {
        Ok(_) => {
            event!(
                name: "update_service.search_attribute.created",
                Level::INFO,
                temporal.namespace = namespace,
                search_attribute.name = REFRESH_TRIGGER_SEARCH_ATTRIBUTE,
                "created search attribute {{search_attribute.name}} in {{temporal.namespace}}",
            );
            Ok(())
        }
        Err(status) if status.code() == Code::AlreadyExists => Ok(()),
        Err(status) => Err(anyhow::Error::from(status).context(format!(
            "failed to create search attribute {REFRESH_TRIGGER_SEARCH_ATTRIBUTE} in {namespace}"
        ))),
    }
}

fn find_non_blank(lookup: impl Fn(&str) -> Option<String>, key: &str) -> Option<String> {
    lookup(key)
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn parse_versioning_enabled(raw_value: Option<String>) -> anyhow::Result<bool> {
    let Some(value) = raw_value else {
        return Ok(false);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Ok(true),
        "0" | "false" | "no" => Ok(false),
        _ => bail!(
            "invalid WORKER_VERSIONING_ENABLED: {value:?} (expected true, false, 1, 0, yes or no)"
        ),
    }
}

fn resolve_versioning_behavior(
    versioning_enabled: bool,
    raw_value: Option<String>,
) -> anyhow::Result<Option<VersioningBehavior>> {
    let behavior = match raw_value.as_deref().map(str::to_ascii_lowercase).as_deref() {
        None | Some("pinned") => VersioningBehavior::Pinned,
        Some("auto_upgrade" | "auto-upgrade") => VersioningBehavior::AutoUpgrade,
        Some(_) => bail!(
            "invalid WORKER_VERSIONING_BEHAVIOR: {:?} (expected pinned or auto_upgrade)",
            raw_value.unwrap_or_default()
        ),
    };
    match (versioning_enabled, raw_value) {
        (true, _) => Ok(Some(behavior)),
        (false, None) => Ok(None),
        (false, Some(_)) => {
            bail!("WORKER_VERSIONING_BEHAVIOR is set but WORKER_VERSIONING_ENABLED is not true")
        }
    }
}
