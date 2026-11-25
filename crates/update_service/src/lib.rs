//! Configuration and worker-set helpers for `update_service`.

use std::env;
use std::future::Future;

use anyhow::Context as _;
use futures::future::try_join_all;
use temporal_workflows::{PROVIDER_QUEUE_NAMES, WorkerQueueSpec, build_worker_options};
use temporalio_client::{Client, ClientOptions, Connection, ConnectionOptions};
use temporalio_common::worker::WorkerDeploymentVersion;
use temporalio_sdk::runtime::{Runtime, RuntimeOptions};
use temporalio_sdk::{Worker, WorkerOptions};
use tracing::Level;
use url::Url;

/// Default Temporal gRPC address for local development.
const DEFAULT_TEMPORAL_ADDRESS: &str = "http://localhost:7233";

/// Default Temporal namespace for Echora workflows.
const DEFAULT_TEMPORAL_NAMESPACE: &str = "echora";

/// Default Worker Deployment name for Update Service workers.
const DEFAULT_DEPLOYMENT_NAME: &str = "update-service";

/// Temporal client and Worker Versioning settings for `update_service`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateServiceSettings {
    /// Temporal frontend gRPC URL (for example `http://localhost:7233`).
    pub temporal_address: String,
    /// Temporal namespace name (default `echora`).
    pub temporal_namespace: String,
    /// Worker Deployment name for deployment-based Worker Versioning.
    pub deployment_name: String,
    /// Build identifier for deployment-based Worker Versioning.
    pub build_id: String,
    /// Opts the worker into Temporal Worker Deployment Versioning.
    pub use_worker_versioning: bool,
}

impl UpdateServiceSettings {
    /// Loads Temporal settings from process environment variables.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use update_service::UpdateServiceSettings;
    ///
    /// let settings = UpdateServiceSettings::from_env();
    /// assert_eq!(settings.temporal_namespace, "echora");
    /// ```
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_lookup(|name| env::var(name).ok())
    }

    /// Loads Temporal settings using a custom `lookup` function.
    ///
    /// # Examples
    ///
    /// ```
    /// use update_service::UpdateServiceSettings;
    ///
    /// let settings = UpdateServiceSettings::from_lookup(|_| None);
    /// assert_eq!(settings.temporal_address, "http://localhost:7233");
    /// ```
    #[must_use]
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let raw_address = lookup("TEMPORAL_ADDRESS")
            .filter(|val| !val.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_TEMPORAL_ADDRESS.to_owned());
        Self {
            temporal_address: normalize_temporal_address(&raw_address),
            temporal_namespace: lookup("TEMPORAL_NAMESPACE")
                .filter(|val| !val.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_TEMPORAL_NAMESPACE.to_owned()),
            deployment_name: lookup("WORKER_DEPLOYMENT_NAME")
                .filter(|val| !val.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_DEPLOYMENT_NAME.to_owned()),
            build_id: lookup("WORKER_BUILD_ID")
                .filter(|val| !val.trim().is_empty())
                .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_owned()),
            use_worker_versioning: lookup("WORKER_VERSIONING_ENABLED")
                .is_some_and(|val| val.eq_ignore_ascii_case("true") || val == "1"),
        }
    }
}

/// Prepends `http://` when `address` omits a URL scheme (such as `temporal:7233`).
fn normalize_temporal_address(address: &str) -> String {
    let trimmed = address.trim();
    if trimmed.contains("://") {
        trimmed.to_owned()
    } else {
        format!("http://{trimmed}")
    }
}

/// Returns all task queue specifications polled by the `update_service` process.
///
/// # Examples
///
/// ```
/// use update_service::list_worker_queue_specs;
///
/// assert_eq!(list_worker_queue_specs().len(), 10);
/// ```
#[must_use]
pub fn list_worker_queue_specs() -> Vec<WorkerQueueSpec> {
    let mut specs = Vec::with_capacity(2 + PROVIDER_QUEUE_NAMES.len());
    specs.push(WorkerQueueSpec::UpdateService);
    for provider in PROVIDER_QUEUE_NAMES {
        specs.push(WorkerQueueSpec::Provider(provider.to_owned()));
    }
    specs.push(WorkerQueueSpec::AgentService);
    specs
}

/// Builds `WorkerOptions` for every task queue polled by `update_service`.
///
/// # Errors
///
/// Returns an error if any worker option fails Temporal SDK registration.
///
/// # Examples
///
/// ```
/// use update_service::{UpdateServiceSettings, build_all_worker_options};
///
/// let settings = UpdateServiceSettings::from_lookup(|_| None);
/// let workers = build_all_worker_options(&settings)?;
/// assert_eq!(workers.len(), 10);
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
pub fn build_all_worker_options(
    settings: &UpdateServiceSettings,
) -> Result<Vec<WorkerOptions>, Box<dyn std::error::Error + Send + Sync>> {
    let deployment = WorkerDeploymentVersion::builder()
        .deployment_name(settings.deployment_name.clone())
        .build_id(settings.build_id.clone())
        .build();
    list_worker_queue_specs()
        .iter()
        .map(|spec| build_worker_options(spec, deployment.clone(), settings.use_worker_versioning))
        .collect()
}

/// Connects to Temporal, polls every configured task queue, and runs until `shutdown` resolves.
///
/// # Errors
///
/// Returns an error when `TEMPORAL_ADDRESS` is invalid, Temporal connection fails,
/// worker initialization fails, or any worker exits with an error.
///
/// # Examples
///
/// ```no_run
/// use update_service::{UpdateServiceSettings, run_workers};
///
/// # async fn example() -> anyhow::Result<()> {
/// let settings = UpdateServiceSettings::from_env();
/// run_workers(&settings, std::future::pending()).await?;
/// # Ok(())
/// # }
/// ```
pub async fn run_workers(
    settings: &UpdateServiceSettings,
    shutdown: impl Future<Output = ()>,
) -> anyhow::Result<()> {
    let temporal_url =
        Url::parse(&settings.temporal_address).context("TEMPORAL_ADDRESS must be a valid URL")?;
    let connection_options = ConnectionOptions::new(temporal_url).build();
    let connection = Connection::connect(connection_options)
        .await
        .context("failed to connect to Temporal server")?;
    let client_options = ClientOptions::new(settings.temporal_namespace.clone()).build();
    let client = Client::new(connection, client_options).map_err(anyhow::Error::msg)?;

    let runtime =
        Runtime::from_current_tokio(RuntimeOptions::default()).map_err(anyhow::Error::msg)?;
    let worker_options =
        build_all_worker_options(settings).map_err(|err| anyhow::anyhow!("{err}"))?;

    tracing::event!(
        name: "update_service.workers.starting",
        Level::INFO,
        temporal.address = %settings.temporal_address,
        temporal.namespace = %settings.temporal_namespace,
        worker.count = worker_options.len(),
        "starting update_service Temporal workers",
    );

    let mut workers = Vec::with_capacity(worker_options.len());
    for opts in worker_options {
        let worker = Worker::new(&runtime, client.clone(), opts).map_err(anyhow::Error::msg)?;
        workers.push(worker);
    }

    tokio::select! {
        res = try_join_all(workers.iter_mut().map(Worker::run)) => {
            res.map_err(anyhow::Error::msg)?;
        }
        () = shutdown => {}
    }

    Ok(())
}
