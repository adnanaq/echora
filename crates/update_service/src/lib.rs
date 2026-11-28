//! Update Service Temporal worker settings, gRPC adapters, observability, and runner.
//!
//! [`UpdateServiceSettings`] reads three groups of settings:
//!
//! - The Temporal connection, through the Temporal SDK's environment configuration:
//!   `TEMPORAL_ADDRESS`, `TEMPORAL_NAMESPACE`, `TEMPORAL_API_KEY`, the `TEMPORAL_TLS*`
//!   variables, or a profile in a `temporal.toml` file. An API key or a TLS setting turns
//!   TLS on, and an address without a scheme gets `https://` when TLS is on, `http://`
//!   otherwise.
//! - Worker Versioning and shutdown drain: `WORKER_VERSIONING_ENABLED`,
//!   `WORKER_VERSIONING_BEHAVIOR`, `WORKER_DEPLOYMENT_NAME`, `WORKER_BUILD_ID`, and
//!   `WORKER_GRACEFUL_SHUTDOWN_SECONDS`.
//! - Downstream gRPC addresses and refresh defaults:
//!   `POSTGRES_SERVICE_ADDRESS`, `ENRICHMENT_SERVICE_ADDRESS`, `REFRESH_AFTER_DAYS`,
//!   `MERGE_VERSION`, and `MERGE_FIRST_WAIT_MINUTES`.
//!
//! Temporal SDK Core OTLP metrics reuse [`common::CommonSettings`] (`OTEL_ENABLED`,
//! `OTEL_ENABLE_METRICS`, `OTEL_EXPORTER_OTLP_ENDPOINT`, and `ENVIRONMENT`) rather than
//! reparsing OpenTelemetry settings.
//!
//! [`run_workers`] connects one Temporal client and lazy gRPC channels to `postgres_service`
//! and `enrichment_service`, runs one worker per task queue the Update Service owns
//! ([`list_worker_queue_specs`]), and drains every worker cleanly when the shutdown future
//! resolves or any worker stops.

mod grpc_clients;

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, anyhow, bail, ensure};
use common::CommonSettings;
use futures::future::join_all;
#[doc(inline)]
pub use grpc_clients::{
    FETCH_ERROR_METADATA_KEY, FETCH_ERROR_PROVIDER_BLOCKED, GrpcCatalogueAdapter,
    MAX_GRPC_MESSAGE_BYTES, TemporalWorkflowStarter, build_enrichment_client,
    build_postgres_client, enrichment_service_proto, map_grpc_status, postgres_service_proto,
    shared_proto,
};
use temporal_workflows::{
    DEFAULT_MERGE_COALESCE_WAIT_SECONDS, DEFAULT_MERGE_FIRST_WAIT_MINUTES, DEFAULT_MERGE_VERSION,
    DEFAULT_REFRESH_AFTER_DAYS, DEFAULT_TEMPORAL_NAMESPACE, PAGE_FETCH_PROVIDERS,
    REFRESH_TRIGGER_SEARCH_ATTRIBUTE, RefreshSettings, UPDATE_SERVICE_TASK_QUEUE,
    UpdateServicePorts, UpdateServiceWorkerConfig, WorkerQueueSpec,
    build_update_service_worker_options,
};
use temporalio_client::tonic::{Code, Request};
use temporalio_client::{Client, ClientOptions, ConnectionOptions};
use temporalio_common::envconfig::{LoadClientConfigProfileOptions, load_client_config_profile};
use temporalio_common::protos::temporal::api::enums::v1::IndexedValueType;
use temporalio_common::protos::temporal::api::operatorservice::v1::{
    AddSearchAttributesRequest, ListSearchAttributesRequest,
};
use temporalio_common::protos::temporal::api::workflowservice::v1::DescribeNamespaceRequest;
use temporalio_common::telemetry::metrics::CoreMeter;
use temporalio_common::telemetry::{
    OtelCollectorOptions, TelemetryOptions, build_otlp_metric_exporter,
};
use temporalio_common::worker::{VersioningBehavior, WorkerDeploymentVersion};
use temporalio_sdk::runtime::RuntimeOptions;
use temporalio_sdk::{Runtime, Worker, WorkerOptions};
use tokio::sync::watch;
use tonic::transport::Endpoint;
use tracing::{Level, event};
use url::Url;

/// Default PostgreSQL Service gRPC URL for local development.
const DEFAULT_POSTGRES_SERVICE_ADDRESS: &str = "http://localhost:8004";

/// Default Enrichment Service gRPC URL for local development.
const DEFAULT_ENRICHMENT_SERVICE_ADDRESS: &str = "http://localhost:8002";

/// Default graceful shutdown window before Temporal cancels in-flight activities.
///
/// Leaves 10 seconds of headroom inside Docker/Kubernetes' standard 30-second stop timeout
/// so cancelled `refresh_page` activities unwind cleanly before `SIGKILL`.
const DEFAULT_GRACEFUL_SHUTDOWN_SECONDS: u64 = 20;

/// Interval at which Temporal Core pushes OTLP metrics to the collector.
///
/// Matches Prometheus's 15-second scrape interval in `docker/observability/prometheus.yaml`.
const TEMPORAL_OTEL_METRIC_PERIODICITY: Duration = Duration::from_secs(15);

/// Settings for connecting Update Service workers to Temporal and downstream gRPC services.
///
/// `Debug` output redacts the Temporal API key.
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
    /// PostgreSQL Service gRPC URL (default `http://localhost:8004`).
    pub postgres_service_address: Url,
    /// Enrichment Service gRPC URL (default `http://localhost:8002`).
    pub enrichment_service_address: Url,
    /// Age in days before a work or page counts as due (`REFRESH_AFTER_DAYS`, default `7`).
    pub refresh_after_days: i64,
    /// Current mapping/merge code version (`MERGE_VERSION`, default `1`).
    pub merge_version: i32,
    /// Wait after the first stored page before merging (`MERGE_FIRST_WAIT_MINUTES`, default `10`).
    pub merge_first_wait_minutes: u64,
    /// Grace period before in-flight activities receive cancellation on worker shutdown.
    pub graceful_shutdown_period: Duration,
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
    ///   a path, or the TLS variables are inconsistent (a client certificate without its key);
    /// - `WORKER_VERSIONING_ENABLED` is not one of `true`, `false`, `1`, `0`, `yes` or `no`;
    /// - `WORKER_VERSIONING_BEHAVIOR` is not `pinned` or `auto_upgrade`, or is
    ///   set while Worker Versioning is off;
    /// - Worker Versioning is on and `WORKER_BUILD_ID` is unset;
    /// - `POSTGRES_SERVICE_ADDRESS` or `ENRICHMENT_SERVICE_ADDRESS` is not a valid URL
    ///   without a path;
    /// - `REFRESH_AFTER_DAYS`, `MERGE_VERSION`, `MERGE_FIRST_WAIT_MINUTES`, or
    ///   `WORKER_GRACEFUL_SHUTDOWN_SECONDS` is not a valid positive/non-negative integer.
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
    /// assert_eq!(settings.postgres_service_address.as_str(), "http://localhost:8004/");
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

        let versioning_enabled = parse_bool_flag(
            "WORKER_VERSIONING_ENABLED",
            find_non_blank(&lookup, "WORKER_VERSIONING_ENABLED"),
            false,
        )?;
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

        let postgres_service_address = parse_grpc_endpoint_url(
            "POSTGRES_SERVICE_ADDRESS",
            find_non_blank(&lookup, "POSTGRES_SERVICE_ADDRESS")
                .as_deref()
                .unwrap_or(DEFAULT_POSTGRES_SERVICE_ADDRESS),
        )?;
        let enrichment_service_address = parse_grpc_endpoint_url(
            "ENRICHMENT_SERVICE_ADDRESS",
            find_non_blank(&lookup, "ENRICHMENT_SERVICE_ADDRESS")
                .as_deref()
                .unwrap_or(DEFAULT_ENRICHMENT_SERVICE_ADDRESS),
        )?;

        let refresh_after_days = parse_min_int(
            "REFRESH_AFTER_DAYS",
            find_non_blank(&lookup, "REFRESH_AFTER_DAYS"),
            DEFAULT_REFRESH_AFTER_DAYS,
            1,
        )?;
        let merge_version = parse_min_int(
            "MERGE_VERSION",
            find_non_blank(&lookup, "MERGE_VERSION"),
            DEFAULT_MERGE_VERSION,
            1,
        )?;
        let merge_first_wait_minutes = parse_min_int(
            "MERGE_FIRST_WAIT_MINUTES",
            find_non_blank(&lookup, "MERGE_FIRST_WAIT_MINUTES"),
            DEFAULT_MERGE_FIRST_WAIT_MINUTES,
            0,
        )?;
        let graceful_shutdown_seconds = parse_min_int(
            "WORKER_GRACEFUL_SHUTDOWN_SECONDS",
            find_non_blank(&lookup, "WORKER_GRACEFUL_SHUTDOWN_SECONDS"),
            DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
            0,
        )?;

        Ok(Self {
            connection,
            namespace,
            deployment_version: WorkerDeploymentVersion::builder()
                .deployment_name(deployment_name)
                .build_id(build_id)
                .build(),
            versioning_behavior,
            postgres_service_address,
            enrichment_service_address,
            refresh_after_days,
            merge_version,
            merge_first_wait_minutes,
            graceful_shutdown_period: Duration::from_secs(graceful_shutdown_seconds),
        })
    }

    /// Returns the worker refresh policy settings passed into Update Service activities.
    #[must_use]
    pub const fn refresh_settings(&self) -> RefreshSettings {
        RefreshSettings::new(
            self.refresh_after_days,
            self.merge_version,
            self.merge_first_wait_minutes.saturating_mul(60),
            DEFAULT_MERGE_COALESCE_WAIT_SECONDS,
        )
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
            .field(
                "postgres_service_address",
                &self.postgres_service_address.as_str(),
            )
            .field(
                "enrichment_service_address",
                &self.enrichment_service_address.as_str(),
            )
            .field("refresh_after_days", &self.refresh_after_days)
            .field("merge_version", &self.merge_version)
            .field("merge_first_wait_minutes", &self.merge_first_wait_minutes)
            .field("graceful_shutdown_period", &self.graceful_shutdown_period)
            .finish_non_exhaustive()
    }
}

/// Lists every task queue the Update Service polls, `update-service` first.
///
/// One provider queue follows for each provider in [`PAGE_FETCH_PROVIDERS`], in that order.
///
/// `agent-service` is left out, since `agent_service` polls it.
///
/// # Examples
///
/// ```
/// use temporal_workflows::{PAGE_FETCH_PROVIDERS, WorkerQueueSpec};
/// use update_service::list_worker_queue_specs;
///
/// let specs = list_worker_queue_specs();
/// assert_eq!(specs.len(), 1 + PAGE_FETCH_PROVIDERS.len());
/// assert_eq!(specs[0], WorkerQueueSpec::UpdateService);
/// ```
#[must_use]
pub fn list_worker_queue_specs() -> Vec<WorkerQueueSpec> {
    let mut specs = Vec::with_capacity(1 + PAGE_FETCH_PROVIDERS.len());
    specs.push(WorkerQueueSpec::UpdateService);
    specs.extend(
        PAGE_FETCH_PROVIDERS
            .into_iter()
            .map(WorkerQueueSpec::Provider),
    );
    specs
}

/// Builds Temporal SDK `RuntimeOptions`, wiring OTLP metrics from [`CommonSettings`].
///
/// Leaves Temporal Core logging disabled so logs flow through the global `tracing` subscriber
/// installed by `common::init_telemetry`.
///
/// # Errors
///
/// Returns an error if the OTLP endpoint URL is invalid or the OTLP metric exporter cannot be
/// constructed.
pub fn build_runtime_options(common_settings: &CommonSettings) -> anyhow::Result<RuntimeOptions> {
    if !(common_settings.observability.otel_enabled
        && common_settings.observability.otel_enable_metrics)
    {
        return Ok(RuntimeOptions::default());
    }
    let endpoint = parse_grpc_endpoint_url(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        &common_settings.observability.otel_exporter_otlp_endpoint,
    )?;
    let global_tags = HashMap::from([
        ("service.name".to_owned(), "update-service".to_owned()),
        (
            "deployment.environment.name".to_owned(),
            common_settings.environment.clone(),
        ),
    ]);
    let collector_options = OtelCollectorOptions::builder()
        .url(endpoint)
        .global_tags(global_tags)
        .metric_periodicity(TEMPORAL_OTEL_METRIC_PERIODICITY)
        .use_seconds_for_durations(true)
        .build();
    let core_meter: Arc<dyn CoreMeter> = Arc::new(
        build_otlp_metric_exporter(collector_options)
            .context("failed to build Temporal SDK OTLP metric exporter")?,
    );
    let telemetry_options = TelemetryOptions::builder().metrics(core_meter).build();
    RuntimeOptions::builder()
        .telemetry_options(telemetry_options)
        .build()
        .map_err(|err| anyhow!("invalid Temporal SDK runtime options: {err}"))
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
/// # #[cfg(feature = "test-util")]
/// # fn example() -> anyhow::Result<()> {
/// use std::collections::HashMap;
///
/// use temporal_workflows::{PAGE_FETCH_PROVIDERS, UpdateServicePorts};
/// use update_service::{UpdateServiceSettings, build_all_worker_options};
///
/// let settings = UpdateServiceSettings::from_vars(&HashMap::new())?;
/// let options = build_all_worker_options(&settings, &UpdateServicePorts::noop())?;
/// assert_eq!(options.len(), 1 + PAGE_FETCH_PROVIDERS.len());
/// # Ok(())
/// # }
/// ```
pub fn build_all_worker_options(
    settings: &UpdateServiceSettings,
    ports: &UpdateServicePorts,
) -> anyhow::Result<Vec<WorkerOptions>> {
    let config = UpdateServiceWorkerConfig {
        deployment_version: settings.deployment_version.clone(),
        versioning_behavior: settings.versioning_behavior,
        graceful_shutdown_period: Some(settings.graceful_shutdown_period),
        ports: ports.clone(),
        refresh_settings: settings.refresh_settings(),
    };
    list_worker_queue_specs()
        .into_iter()
        .map(|spec| {
            build_update_service_worker_options(spec, &config)
                .with_context(|| format!("failed to build worker options for {spec}"))
        })
        .collect()
}

/// Runs one worker per Update Service task queue until shutdown.
///
/// Every worker shares one Temporal client and lazy gRPC channels to `postgres_service`
/// and `enrichment_service`. When `shutdown` resolves, or any worker stops on its own,
/// every worker stops polling, drains in-flight tasks (cancelling activities after
/// `graceful_shutdown_period`), and releases its slots on the server before returning.
/// A `shutdown` that resolves while the workers are still starting ends startup at once
/// and returns `Ok`.
///
/// # Errors
///
/// Returns an error if the Temporal runtime or connection fails, the namespace does not
/// exist, the `RefreshTrigger` search attribute has the wrong type or cannot be created,
/// a worker cannot be created, or any worker fails while running.
///
/// # Examples
///
/// ```no_run
/// use common::CommonSettings;
/// use update_service::{UpdateServiceSettings, run_workers};
///
/// # async fn example() -> anyhow::Result<()> {
/// let common_settings = CommonSettings::from_env()?;
/// let settings = UpdateServiceSettings::from_env()?;
/// run_workers(&settings, &common_settings, async {
///     let _ = tokio::signal::ctrl_c().await;
/// })
/// .await?;
/// # Ok(())
/// # }
/// ```
pub async fn run_workers<F>(
    settings: &UpdateServiceSettings,
    common_settings: &CommonSettings,
    shutdown: F,
) -> anyhow::Result<()>
where
    F: Future<Output = ()>,
{
    let specs = list_worker_queue_specs();
    let mut shutdown = pin!(shutdown);
    let runtime_options = build_runtime_options(common_settings)?;
    let runtime = Runtime::from_current_tokio(runtime_options)?;
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

/// Connects to Temporal, verifies the namespace and search attribute, and builds all workers.
async fn start_workers(
    settings: &UpdateServiceSettings,
    runtime: &Runtime,
) -> anyhow::Result<Vec<Worker>> {
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
    // Verify the namespace up front before listing search attributes or creating 9 workers,
    // so a misspelled namespace fails fast with a clear message.
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

    let postgres_channel = Endpoint::from_shared(settings.postgres_service_address.to_string())
        .context("invalid POSTGRES_SERVICE_ADDRESS endpoint")?
        .connect_lazy();
    let enrichment_channel = Endpoint::from_shared(settings.enrichment_service_address.to_string())
        .context("invalid ENRICHMENT_SERVICE_ADDRESS endpoint")?
        .connect_lazy();
    let ports = UpdateServicePorts::new(
        GrpcCatalogueAdapter::new(
            build_postgres_client(postgres_channel),
            build_enrichment_client(enrichment_channel),
        ),
        TemporalWorkflowStarter::new(client.clone()),
    );
    let worker_options = build_all_worker_options(settings, &ports)?;

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

fn parse_bool_flag(
    name: &str,
    raw_value: Option<String>,
    default_value: bool,
) -> anyhow::Result<bool> {
    let Some(value) = raw_value else {
        return Ok(default_value);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Ok(true),
        "0" | "false" | "no" => Ok(false),
        _ => bail!("invalid {name}: {value:?} (expected true, false, 1, 0, yes or no)"),
    }
}

fn parse_grpc_endpoint_url(name: &str, raw_value: &str) -> anyhow::Result<Url> {
    let trimmed = raw_value.trim();
    let candidate = if trimmed.contains("://") {
        trimmed.to_owned()
    } else {
        format!("http://{trimmed}")
    };
    let parsed = Url::parse(&candidate)
        .with_context(|| format!("invalid {name}: {raw_value:?} is not a valid URL"))?;
    ensure!(
        matches!(parsed.scheme(), "http" | "https") && parsed.has_host() && parsed.path() == "/",
        "invalid {name}: must be host:port or an http(s) URL without a path, got {raw_value:?}"
    );
    Ok(parsed)
}

fn parse_min_int<T>(
    name: &str,
    raw_value: Option<String>,
    default_value: T,
    min_value: T,
) -> anyhow::Result<T>
where
    T: std::str::FromStr + PartialOrd + Copy + fmt::Display,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let Some(value) = raw_value else {
        return Ok(default_value);
    };
    let parsed = value
        .parse::<T>()
        .with_context(|| format!("invalid {name}: {value:?} must be an integer"))?;
    ensure!(
        parsed >= min_value,
        "invalid {name}: must be at least {min_value}"
    );
    Ok(parsed)
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
