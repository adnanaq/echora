//! Temporal workflow, activity, and task-queue definitions for Echora.
//!
//! Centralizes the names every Temporal worker and client must agree on:
//!
//! - the development namespace, `echora-dev` ([`DEFAULT_TEMPORAL_NAMESPACE`]); production
//!   runs in `echora`, set through `TEMPORAL_NAMESPACE`,
//! - the task queues ([`UPDATE_SERVICE_TASK_QUEUE`], [`AGENT_SERVICE_TASK_QUEUE`] and one
//!   `provider-<slug>` queue per [`Provider`]),
//! - the `RefreshTrigger` search attribute ([`REFRESH_TRIGGER_SEARCH_ATTRIBUTE`]).
//!
//! [`build_worker_options`] starts a Temporal `WorkerOptionsBuilder` for one queue with
//! deployment-based Worker Versioning applied, and [`build_update_service_worker_options`]
//! registers the Update Service workflows and activities on the appropriate queue.
//!
//! # Features
//!
//! - `spike`: the ECHO-95 SDK spike workflow and activities, plus
//!   `build_spike_worker_options`, which registers them.
//! - `test-util`: `testing::start_local_environment`, which starts a Temporal CLI dev server
//!   configured like the development server in `docker/docker-compose.dev.yml`, plus
//!   `UpdateServicePorts::noop` and `with_heartbeat_interval`.

mod activities;
mod ports;
mod refresh_anime_types;
mod refresh_anime_workflow;

use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use temporalio_common::worker::{
    VersioningBehavior, WorkerDeploymentOptions, WorkerDeploymentVersion,
};
use temporalio_sdk::workflows::WorkflowRegistrationError;
use temporalio_sdk::{WorkerOptions, WorkerOptionsBuilder, worker_options_builder};

#[doc(inline)]
pub use activities::{
    MAX_ABORTED_RETRIES_PER_ATTEMPT, ProviderActivities, REFRESH_PAGE_HEARTBEAT_INTERVAL,
    UpdateServiceActivities,
};
#[doc(inline)]
pub use ports::{
    CatalogueDataPort, MergeAttemptOutcome, PortError, UpdateServicePorts, WorkflowStarterPort,
};
#[doc(inline)]
pub use refresh_anime_types::{
    DEFAULT_MERGE_COALESCE_WAIT_SECONDS, DEFAULT_MERGE_FIRST_WAIT_MINUTES, DEFAULT_MERGE_VERSION,
    DEFAULT_REFRESH_AFTER_DAYS, EntityPageSummary, EntityPages, EntityType, FetchRequestedSignal,
    ListEntityPagesOutput, MAX_PAGE_RETRY_BACKOFF_HOURS, MergeAnimeOutput, ParseSlugError,
    RecordRefreshInput, RefreshAnimeInput, RefreshAnimeOutcome, RefreshOutcome, RefreshPageInput,
    RefreshPageKey, RefreshPageOutput, RefreshSettings, RefreshTrigger, SECONDS_PER_DAY,
    StartEntityWorkflowsInput, build_refresh_anime_workflow_id, build_refresh_episode_workflow_id,
    build_write_synopsis_workflow_id,
};
#[doc(inline)]
pub use refresh_anime_workflow::{
    RefreshAnimeWorkflow, build_priority, compute_page_backoff_seconds, is_page_due,
    matches_fetch_request, merge_priority, select_pages_to_refresh,
};

/// Development namespace, used when `TEMPORAL_NAMESPACE` is unset.
///
/// Production sets `TEMPORAL_NAMESPACE` to `echora`, so a worker started without it
/// polls the development namespace and never takes production work.
pub const DEFAULT_TEMPORAL_NAMESPACE: &str = "echora-dev";

/// Task queue for Update Service workflows, merges, and workflow starters.
pub const UPDATE_SERVICE_TASK_QUEUE: &str = "update-service";

/// Task queue for AI agent activities handled by `agent_service`.
pub const AGENT_SERVICE_TASK_QUEUE: &str = "agent-service";

/// Temporal Keyword search attribute recording the trigger of a refresh.
pub const REFRESH_TRIGGER_SEARCH_ATTRIBUTE: &str = "RefreshTrigger";

/// Metadata provider that receives a dedicated `provider-<slug>` task queue.
///
/// Each provider has its own queue so its request rate can be limited on the
/// Temporal server, independent of the other providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(try_from = "String", into = "&'static str")]
pub enum Provider {
    /// `AniDB` (`provider-anidb`).
    AniDb,
    /// `AniList` (`provider-anilist`).
    AniList,
    /// Anime-Planet (`provider-anime_planet`).
    AnimePlanet,
    /// `AnimeSchedule` (`provider-animeschedule`).
    AnimeSchedule,
    /// aniSearch (`provider-anisearch`).
    AniSearch,
    /// Kitsu (`provider-kitsu`).
    Kitsu,
    /// `MyAnimeList` (`provider-mal`).
    Mal,
    /// Syoboi Calendar (`provider-syoboi`).
    Syoboi,
}

impl Provider {
    /// Single source of truth pairing each provider with its wire slug and task queue name.
    const METADATA: [(Self, &'static str, &'static str); 8] = [
        (Self::AniDb, "anidb", "provider-anidb"),
        (Self::AniList, "anilist", "provider-anilist"),
        (Self::AnimePlanet, "anime_planet", "provider-anime_planet"),
        (
            Self::AnimeSchedule,
            "animeschedule",
            "provider-animeschedule",
        ),
        (Self::AniSearch, "anisearch", "provider-anisearch"),
        (Self::Kitsu, "kitsu", "provider-kitsu"),
        (Self::Mal, "mal", "provider-mal"),
        (Self::Syoboi, "syoboi", "provider-syoboi"),
    ];

    /// Every supported metadata provider, each listed once.
    pub const ALL: [Self; 8] = [
        Self::AniDb,
        Self::AniList,
        Self::AnimePlanet,
        Self::AnimeSchedule,
        Self::AniSearch,
        Self::Kitsu,
        Self::Mal,
        Self::Syoboi,
    ];

    const fn index(self) -> usize {
        match self {
            Self::AniDb => 0,
            Self::AniList => 1,
            Self::AnimePlanet => 2,
            Self::AnimeSchedule => 3,
            Self::AniSearch => 4,
            Self::Kitsu => 5,
            Self::Mal => 6,
            Self::Syoboi => 7,
        }
    }

    /// Returns the provider slug used in queue names and stored page keys.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::Provider;
    ///
    /// assert_eq!(Provider::AnimePlanet.slug(), "anime_planet");
    /// ```
    #[must_use]
    pub const fn slug(self) -> &'static str {
        Self::METADATA[self.index()].1
    }

    /// Returns the `provider-<slug>` task queue name for this provider.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::Provider;
    ///
    /// assert_eq!(Provider::AniDb.task_queue(), "provider-anidb");
    /// ```
    #[must_use]
    pub const fn task_queue(self) -> &'static str {
        Self::METADATA[self.index()].2
    }

    /// Parses a provider slug, returning `None` for unknown values.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::Provider;
    ///
    /// assert_eq!(Provider::from_slug("mal"), Some(Provider::Mal));
    /// assert_eq!(Provider::from_slug("unknown"), None);
    /// ```
    #[must_use]
    pub fn from_slug(slug: &str) -> Option<Self> {
        Self::METADATA
            .into_iter()
            .find_map(|(provider, known_slug, _)| (known_slug == slug).then_some(provider))
    }
}

impl From<Provider> for &'static str {
    fn from(value: Provider) -> Self {
        value.slug()
    }
}

impl TryFrom<String> for Provider {
    type Error = ParseSlugError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

impl FromStr for Provider {
    type Err = ParseSlugError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::from_slug(input).ok_or_else(|| ParseSlugError::new("provider", input))
    }
}

/// Task queue a worker polls, which decides what the worker registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerQueueSpec {
    /// Registers workflows and core orchestration activities on `update-service`.
    UpdateService,
    /// Registers provider fetch activities on `provider-<slug>`.
    Provider(Provider),
    /// Registers AI activities on `agent-service`.
    AgentService,
}

impl WorkerQueueSpec {
    /// Returns the Temporal task queue name for this worker specification.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::{Provider, WorkerQueueSpec};
    ///
    /// assert_eq!(WorkerQueueSpec::UpdateService.task_queue(), "update-service");
    /// assert_eq!(
    ///     WorkerQueueSpec::Provider(Provider::Mal).task_queue(),
    ///     "provider-mal"
    /// );
    /// ```
    #[must_use]
    pub const fn task_queue(self) -> &'static str {
        match self {
            Self::UpdateService => UPDATE_SERVICE_TASK_QUEUE,
            Self::Provider(provider) => provider.task_queue(),
            Self::AgentService => AGENT_SERVICE_TASK_QUEUE,
        }
    }
}

impl fmt::Display for WorkerQueueSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.task_queue())
    }
}

/// Configuration bundle passed to [`build_update_service_worker_options`].
#[derive(Debug, Clone)]
pub struct UpdateServiceWorkerConfig {
    /// Worker Deployment name and build ID reported to Temporal.
    pub deployment_version: WorkerDeploymentVersion,
    /// Default versioning behavior, or `None` when Worker Versioning is disabled.
    pub versioning_behavior: Option<VersioningBehavior>,
    /// Grace period before Temporal cancels in-flight activities on worker shutdown.
    pub graceful_shutdown_period: Option<Duration>,
    /// External service and workflow-starter ports injected into activities.
    pub ports: UpdateServicePorts,
    /// Worker-level refresh policy settings supplied to `UpdateServiceActivities`.
    pub refresh_settings: RefreshSettings,
}

/// Starts worker options for `queue_spec` with Worker Versioning applied.
///
/// `Some(behavior)` turns on deployment-based Worker Versioning with that
/// default behavior for workflows started on this worker. `None` leaves the
/// worker unversioned. Taking one `Option` keeps the two Temporal settings
/// consistent: the SDK rejects a default behavior on an unversioned worker.
///
/// The returned builder has no workflows or activities registered, so the
/// caller registers what its queue runs and then calls `build`.
///
/// # Examples
///
/// ```
/// use temporal_workflows::{WorkerQueueSpec, build_worker_options};
/// use temporalio_common::worker::{VersioningBehavior, WorkerDeploymentVersion};
///
/// let version = WorkerDeploymentVersion::builder()
///     .deployment_name("update-service".to_owned())
///     .build_id("0.1.0".to_owned())
///     .build();
/// let options = build_worker_options(
///     WorkerQueueSpec::UpdateService,
///     version,
///     Some(VersioningBehavior::Pinned),
/// )
/// .build();
/// assert_eq!(options.task_queue, "update-service");
/// assert!(options.deployment_options.use_worker_versioning);
/// ```
pub fn build_worker_options(
    queue_spec: WorkerQueueSpec,
    deployment_version: WorkerDeploymentVersion,
    versioning_behavior: Option<VersioningBehavior>,
) -> WorkerOptionsBuilder<worker_options_builder::SetDeploymentOptions> {
    let deployment_options = WorkerDeploymentOptions::new(deployment_version)
        .use_worker_versioning(versioning_behavior.is_some())
        .maybe_default_versioning_behavior(versioning_behavior)
        .build();

    WorkerOptions::new(queue_spec.task_queue()).deployment_options(deployment_options)
}

/// Builds `WorkerOptions` registering Update Service workflows and activities for `queue_spec`.
///
/// `UpdateService` registers [`RefreshAnimeWorkflow`] and [`UpdateServiceActivities`];
/// `Provider(_)` registers [`ProviderActivities`]; `AgentService` leaves registration empty
/// because `agent_service` owns that queue. Setting `graceful_shutdown_period` causes the SDK
/// to cancel in-flight activities after that duration during worker drain.
///
/// # Errors
///
/// Returns [`WorkflowRegistrationError`] if workflow registration fails inside the Temporal SDK.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "test-util")]
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use std::time::Duration;
/// use temporal_workflows::{
///     RefreshSettings, UpdateServicePorts, UpdateServiceWorkerConfig, WorkerQueueSpec,
///     build_update_service_worker_options,
/// };
/// use temporalio_common::worker::WorkerDeploymentVersion;
///
/// let config = UpdateServiceWorkerConfig {
///     deployment_version: WorkerDeploymentVersion::builder()
///         .deployment_name("update-service".to_owned())
///         .build_id("0.1.0".to_owned())
///         .build(),
///     versioning_behavior: None,
///     graceful_shutdown_period: Some(Duration::from_secs(20)),
///     ports: UpdateServicePorts::noop(),
///     refresh_settings: RefreshSettings::default(),
/// };
/// let options = build_update_service_worker_options(WorkerQueueSpec::UpdateService, &config)?;
/// assert_eq!(options.task_queue, "update-service");
/// assert_eq!(options.graceful_shutdown_period, Some(Duration::from_secs(20)));
/// # Ok(())
/// # }
/// ```
pub fn build_update_service_worker_options(
    queue_spec: WorkerQueueSpec,
    config: &UpdateServiceWorkerConfig,
) -> Result<WorkerOptions, WorkflowRegistrationError> {
    let builder = build_worker_options(
        queue_spec,
        config.deployment_version.clone(),
        config.versioning_behavior,
    )
    .maybe_graceful_shutdown_period(config.graceful_shutdown_period);

    let builder = match queue_spec {
        WorkerQueueSpec::UpdateService => builder
            .register_activities(UpdateServiceActivities::with_settings(
                config.ports.clone(),
                config.refresh_settings,
            ))
            .register_workflow::<RefreshAnimeWorkflow>()?,
        WorkerQueueSpec::Provider(_) => {
            builder.register_activities(ProviderActivities::new(config.ports.clone()))
        }
        WorkerQueueSpec::AgentService => builder,
    };

    Ok(builder.build())
}

#[cfg(feature = "spike")]
#[allow(
    missing_docs,
    reason = "temporalio_macros generates undocumented associated constants"
)]
#[allow(
    clippy::unused_async_trait_impl,
    reason = "temporalio_macros copies each async #[activity] into a generated trait impl"
)]
mod spike {
    use std::time::Duration;

    use serde::{Deserialize, Serialize};
    use temporalio_common::worker::{VersioningBehavior, WorkerDeploymentVersion};
    use temporalio_macros::{activities, workflow, workflow_methods};
    use temporalio_sdk::activities::{ActivityContext, ActivityError};
    use temporalio_sdk::workflows::WorkflowRegistrationError;
    use temporalio_sdk::{
        ActivityOptions, SyncWorkflowContext, WorkerOptions, WorkflowContext, WorkflowContextView,
        WorkflowResult,
    };

    use crate::{WorkerQueueSpec, build_worker_options};

    /// How long one `format_spike_tag` attempt may run before Temporal retries it.
    ///
    /// The activity only echoes its input, so any attempt that takes this long
    /// is stuck. Lowering the value below a worker's poll latency would make
    /// healthy attempts time out.
    const SPIKE_ACTIVITY_TIMEOUT: Duration = Duration::from_secs(30);

    /// Builds worker options registering the ECHO-95 spike workflow and activities.
    ///
    /// Every queue registers [`SpikeActivities`]; only `update-service` also
    /// registers [`SpikeWorkflow`].
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowRegistrationError`] if workflow registration fails
    /// inside the Temporal SDK.
    pub fn build_spike_worker_options(
        queue_spec: WorkerQueueSpec,
        deployment_version: WorkerDeploymentVersion,
        versioning_behavior: Option<VersioningBehavior>,
    ) -> Result<WorkerOptions, WorkflowRegistrationError> {
        let builder = build_worker_options(queue_spec, deployment_version, versioning_behavior)
            .register_activities(SpikeActivities);

        let builder = match queue_spec {
            WorkerQueueSpec::UpdateService => builder.register_workflow::<SpikeWorkflow>()?,
            WorkerQueueSpec::Provider(_) | WorkerQueueSpec::AgentService => builder,
        };

        Ok(builder.build())
    }

    /// Input payload for `SpikeWorkflow`.
    #[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
    pub struct SpikeInput {
        /// Human-readable label echoed in the spike workflow output.
        pub label: String,
    }

    /// Minimal activity set verifying worker polling and activity dispatch.
    #[derive(Debug)]
    pub struct SpikeActivities;

    #[activities]
    impl SpikeActivities {
        /// Echoes the completion tag to verify activity dispatch.
        ///
        /// # Errors
        ///
        /// Never fails; the `Result` is the signature Temporal activities require.
        #[activity]
        #[expect(
            clippy::unused_async,
            reason = "temporalio_macros #[activity] requires an async fn"
        )]
        pub async fn format_spike_tag(
            _ctx: ActivityContext,
            tag: String,
        ) -> Result<String, ActivityError> {
            Ok(tag)
        }
    }

    /// Spike workflow proving signals, conflict policy, search attributes, and versioning.
    #[workflow]
    #[derive(Debug)]
    pub struct SpikeWorkflow {
        label: String,
        completion_tag: Option<String>,
    }

    #[workflow_methods]
    impl SpikeWorkflow {
        /// Initializes `SpikeWorkflow` state from `input`.
        #[init]
        fn new(_ctx: &WorkflowContextView, input: SpikeInput) -> Self {
            Self {
                label: input.label,
                completion_tag: None,
            }
        }

        /// Waits for `finish`, runs one activity, and returns `<label>:<tag>`.
        ///
        /// # Errors
        ///
        /// Returns a workflow failure if the activity execution fails.
        #[run]
        pub async fn run(ctx: &mut WorkflowContext<Self>) -> WorkflowResult<String> {
            ctx.wait_condition(|state| state.completion_tag.is_some())
                .await?;
            let tag = ctx.state(|state| state.completion_tag.clone().unwrap_or_default());
            let formatted = ctx
                .execute_activity(
                    SpikeActivities::format_spike_tag,
                    tag,
                    ActivityOptions::start_to_close_timeout(SPIKE_ACTIVITY_TIMEOUT),
                )
                .await?;
            let label = ctx.state(|state| state.label.clone());
            Ok(format!("{label}:{formatted}"))
        }

        /// Delivers the completion tag and unblocks `SpikeWorkflow::run`.
        ///
        /// The first `finish` signal wins, so later duplicates are ignored.
        #[signal]
        pub fn finish(&mut self, _ctx: &mut SyncWorkflowContext<Self>, tag: String) {
            self.completion_tag.get_or_insert(tag);
        }
    }
}

#[cfg(feature = "spike")]
#[doc(inline)]
pub use spike::{SpikeActivities, SpikeInput, SpikeWorkflow, build_spike_worker_options};

#[cfg(feature = "test-util")]
pub mod testing {
    //! Starts a local Temporal dev server for tests.
    //!
    //! The server is the Temporal CLI dev server, downloaded and cached by the
    //! Temporal SDK on first use. It runs with the settings the development
    //! server in `docker/docker-compose.dev.yml` uses: the `echora-dev` namespace,
    //! the `RefreshTrigger` keyword search attribute, fairness turned on, and one
    //! partition per task queue. Each call starts its own server with in-memory
    //! storage, so tests never share workflows or pollers.

    use temporalio_client::ClientOptions;
    use temporalio_sdk::testing::{
        LocalServer, LocalWorkflowEnvironmentOptions, WorkflowEnvironment, WorkflowEnvironmentError,
    };

    use crate::{DEFAULT_TEMPORAL_NAMESPACE, REFRESH_TRIGGER_SEARCH_ATTRIBUTE};

    /// Starts a dev server configured like Echora's development server.
    ///
    /// `port` fixes the frontend port, for tests that hand the address to
    /// another process; `None` lets the operating system choose one.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowEnvironmentError`] if the Temporal CLI cannot be
    /// downloaded or started, or if the client cannot connect to it.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use temporal_workflows::testing::start_local_environment;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let environment = start_local_environment(None).await?;
    /// let client = environment.client().clone();
    /// // Start workers and workflows with `client`.
    /// # drop(client);
    /// environment.shutdown().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn start_local_environment(
        port: Option<u16>,
    ) -> Result<WorkflowEnvironment<LocalServer>, WorkflowEnvironmentError> {
        let options = LocalWorkflowEnvironmentOptions::builder()
            .client_options(ClientOptions::new(DEFAULT_TEMPORAL_NAMESPACE).build())
            .maybe_port(port)
            .extra_args(vec![
                "--search-attribute".to_owned(),
                format!("{REFRESH_TRIGGER_SEARCH_ATTRIBUTE}=Keyword"),
                "--dynamic-config-value".to_owned(),
                "matching.enableFairness=true".to_owned(),
                "--dynamic-config-value".to_owned(),
                "matching.numTaskqueueReadPartitions=1".to_owned(),
                "--dynamic-config-value".to_owned(),
                "matching.numTaskqueueWritePartitions=1".to_owned(),
            ])
            .build();
        WorkflowEnvironment::start_local(options).await
    }
}
