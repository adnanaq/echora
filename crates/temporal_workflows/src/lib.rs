//! Temporal workflow, activity, and task-queue definitions for Echora.
//!
//! Centralizes shared Temporal task queue names, search attribute keys,
//! deployment-based worker versioning helpers, and SDK spike primitives used
//! by `update_service`.

mod activities;
mod ports;
mod refresh_anime_types;
mod refresh_anime_workflow;
mod refresh_entity_workflows;

use temporalio_common::worker::{WorkerDeploymentOptions, WorkerDeploymentVersion};
use temporalio_sdk::WorkerOptions;

#[doc(inline)]
pub use activities::{
    MAX_ABORTED_RETRIES_PER_ATTEMPT, ProviderActivities, REFRESH_PAGE_HEARTBEAT_INTERVAL,
    UpdateServiceActivities,
};
#[doc(inline)]
pub use ports::{
    CatalogueDataPort, MergeAttemptOutcome, PortError, PortErrorKind, UpdateServicePorts,
    WorkflowStarterPort, is_non_retryable_grpc_code,
};
#[doc(inline)]
pub use refresh_anime_types::{
    CharacterFetchTarget, DEFAULT_MERGE_FIRST_WAIT_MINUTES, DEFAULT_REFRESH_AFTER_DAYS,
    EntityPageSummary, FetchRequestedSignal, ListEntityPagesInput, ListEntityPagesOutput,
    MAX_PAGE_RETRY_BACKOFF_HOURS, MergeEntityInput, MergeEntityOutput, RecordRefreshInput,
    RefreshAnimeInput, RefreshAnimeOutcome, RefreshEntityInput, RefreshEntityKind,
    RefreshEntityOutcome, RefreshOutcomeKind, RefreshPageInput, RefreshPageKey, RefreshPageOutput,
    StartEntityWorkflowsInput, build_refresh_anime_workflow_id,
    build_refresh_character_workflow_id, build_refresh_episode_workflow_id,
    build_write_synopsis_workflow_id,
};
#[doc(inline)]
pub use refresh_anime_workflow::{
    RefreshAnimeWorkflow, build_priority, compute_page_backoff_seconds, is_page_due,
    matches_fetch_request, merge_priority, select_pages_to_refresh,
};
#[doc(inline)]
pub use refresh_entity_workflows::{RefreshCharacterWorkflow, RefreshEpisodeWorkflow};

/// Task queue for Update Service workflows, merges, and workflow starters.
pub const UPDATE_SERVICE_TASK_QUEUE: &str = "update-service";

/// Task queue for AI agent activities handled by `agent_service`.
pub const AGENT_SERVICE_TASK_QUEUE: &str = "agent-service";

/// Temporal Keyword search attribute recording the trigger of a refresh.
pub const REFRESH_TRIGGER_SEARCH_ATTRIBUTE: &str = "RefreshTrigger";

/// Supported provider slugs that each receive a dedicated `provider-<name>` queue.
pub const PROVIDER_QUEUE_NAMES: [&str; 8] = [
    "anidb",
    "anilist",
    "anime_planet",
    "animeschedule",
    "anisearch",
    "kitsu",
    "mal",
    "syoboi",
];

/// Formats the Temporal task queue name for a provider slug.
///
/// # Examples
///
/// ```
/// use temporal_workflows::provider_task_queue;
///
/// assert_eq!(provider_task_queue("anidb"), "provider-anidb");
/// ```
#[must_use]
pub fn provider_task_queue(provider: &str) -> String {
    format!("provider-{provider}")
}

/// Identifies which worker queue registration profile to construct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerQueueSpec {
    /// Registers workflows and core orchestration activities on `update-service`.
    UpdateService,
    /// Registers provider fetch activities on `provider-<name>`.
    Provider(String),
    /// Registers AI activities on `agent-service`.
    AgentService,
}

impl WorkerQueueSpec {
    /// Returns the Temporal task queue name for this worker specification.
    ///
    /// # Examples
    ///
    /// ```
    /// use temporal_workflows::WorkerQueueSpec;
    ///
    /// assert_eq!(WorkerQueueSpec::UpdateService.task_queue(), "update-service");
    /// assert_eq!(
    ///     WorkerQueueSpec::Provider("mal".to_owned()).task_queue(),
    ///     "provider-mal"
    /// );
    /// ```
    #[must_use]
    pub fn task_queue(&self) -> String {
        match self {
            Self::UpdateService => UPDATE_SERVICE_TASK_QUEUE.to_owned(),
            Self::Provider(provider) => provider_task_queue(provider),
            Self::AgentService => AGENT_SERVICE_TASK_QUEUE.to_owned(),
        }
    }
}

/// Builds `WorkerOptions` with deployment-based Worker Versioning for `queue_spec`.
///
/// # Errors
///
/// Returns an error if workflow registration fails inside the Temporal SDK.
///
/// # Examples
///
/// ```
/// use temporal_workflows::{UpdateServicePorts, WorkerQueueSpec, build_worker_options};
/// use temporalio_common::worker::WorkerDeploymentVersion;
///
/// let version = WorkerDeploymentVersion::builder()
///     .deployment_name("update-service".to_owned())
///     .build_id("0.1.0".to_owned())
///     .build();
/// let opts = build_worker_options(
///     &WorkerQueueSpec::UpdateService,
///     version,
///     true,
///     UpdateServicePorts::noop(),
/// )?;
/// assert_eq!(opts.task_queue, "update-service");
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
pub fn build_worker_options(
    queue_spec: &WorkerQueueSpec,
    deployment_version: WorkerDeploymentVersion,
    use_worker_versioning: bool,
    ports: UpdateServicePorts,
) -> Result<WorkerOptions, Box<dyn std::error::Error + Send + Sync>> {
    let task_queue = queue_spec.task_queue();
    let deployment_options = WorkerDeploymentOptions::new(deployment_version)
        .use_worker_versioning(use_worker_versioning)
        .build();

    let builder = WorkerOptions::new(task_queue).deployment_options(deployment_options);

    let builder = match queue_spec {
        WorkerQueueSpec::UpdateService => builder
            .register_activities(UpdateServiceActivities::new(ports))
            .register_workflow::<RefreshAnimeWorkflow>()?
            .register_workflow::<RefreshEpisodeWorkflow>()?
            .register_workflow::<RefreshCharacterWorkflow>()?
            .register_workflow::<SpikeWorkflow>()?,
        WorkerQueueSpec::Provider(_) => builder.register_activities(ProviderActivities::new(ports)),
        WorkerQueueSpec::AgentService => builder.register_activities(SpikeActivities),
    };

    Ok(builder.build())
}

#[allow(
    missing_docs,
    reason = "temporalio_macros generates undocumented associated constants"
)]
mod spike {
    use std::time::Duration;

    use serde::{Deserialize, Serialize};
    use temporalio_macros::{activities, workflow, workflow_methods};
    use temporalio_sdk::activities::{ActivityContext, ActivityError};
    use temporalio_sdk::{
        ActivityOptions, SyncWorkflowContext, WorkflowContext, WorkflowContextView, WorkflowResult,
    };

    /// Input payload for `SpikeWorkflow`.
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct SpikeInput {
        /// Human-readable label echoed in the spike workflow output.
        pub label: String,
    }

    /// Minimal activity set used to verify worker polling and activity dispatch.
    pub struct SpikeActivities;

    #[activities]
    impl SpikeActivities {
        /// Formats a completion tag for the ECHO-95 SDK spike.
        ///
        /// # Errors
        ///
        /// Never fails in normal execution; returns `ActivityError` on signature match.
        #[activity]
        pub async fn format_spike_tag(
            _ctx: ActivityContext,
            tag: String,
        ) -> Result<String, ActivityError> {
            tokio::task::yield_now().await;
            Ok(tag)
        }
    }

    /// Spike workflow proving signals, conflict policy, search attributes, and versioning.
    #[workflow]
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

        /// Waits for `finish` signal, runs a verification activity, and returns `<label>:<tag>`.
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
                    ActivityOptions::start_to_close_timeout(Duration::from_secs(30)),
                )
                .await?;
            let label = ctx.state(|state| state.label.clone());
            Ok(format!("{label}:{formatted}"))
        }

        /// Delivers the completion tag and unblocks `SpikeWorkflow::run`.
        #[signal]
        pub fn finish(&mut self, _ctx: &mut SyncWorkflowContext<Self>, tag: String) {
            self.completion_tag = Some(tag);
        }
    }
}

#[doc(inline)]
pub use spike::{SpikeActivities, SpikeInput, SpikeWorkflow};
