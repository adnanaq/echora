//! Temporal activity implementations for `update-service` and `provider-<name>` queues.

#[allow(
    missing_docs,
    reason = "temporalio_macros generates undocumented associated constants"
)]
mod definitions {
    use std::sync::Arc;
    use std::time::Duration;

    use temporalio_macros::activities;
    use temporalio_sdk::activities::{ActivityContext, ActivityError};
    use temporalio_sdk::error::ApplicationFailure;

    use crate::ports::{MergeAttemptOutcome, PortError, PortErrorKind, UpdateServicePorts};
    use crate::refresh_anime_types::{
        ListEntityPagesOutput, MergeAnimeInput, MergeAnimeOutput, RecordRefreshInput,
        RefreshPageInput, RefreshPageOutput, StartEntityWorkflowsInput,
    };

    /// Maximum `LoadAnime` -> `MergeAnime` -> `SaveAnime` retries inside one `merge_anime` attempt
    /// when `SaveAnime` returns `ABORTED` due to concurrent updates.
    pub const MAX_ABORTED_RETRIES_PER_ATTEMPT: usize = 10;

    /// Interval between `refresh_page` heartbeats while `FetchPage` + `StorePage` run.
    pub const REFRESH_PAGE_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(20);

    /// Core Update Service activities running on the `update-service` task queue.
    pub struct UpdateServiceActivities {
        ports: UpdateServicePorts,
    }

    impl UpdateServiceActivities {
        /// Creates core Update Service activities backed by `ports`.
        #[must_use]
        pub fn new(ports: UpdateServicePorts) -> Self {
            Self { ports }
        }
    }

    /// Provider page-fetch activities running on each `provider-<name>` task queue.
    pub struct ProviderActivities {
        ports: UpdateServicePorts,
        heartbeat_interval: Duration,
    }

    impl ProviderActivities {
        /// Creates provider activities backed by `ports` with the standard 20-second heartbeat.
        #[must_use]
        pub fn new(ports: UpdateServicePorts) -> Self {
            Self {
                ports,
                heartbeat_interval: REFRESH_PAGE_HEARTBEAT_INTERVAL,
            }
        }

        /// Overrides the heartbeat interval (useful in fast unit/integration tests).
        #[must_use]
        pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
            self.heartbeat_interval = interval;
            self
        }
    }

    /// Converts a [`PortError`] into an [`ActivityError`], marking permanent failures non-retryable.
    fn to_activity_error(err: &PortError) -> ActivityError {
        let is_non_retryable = err.kind == PortErrorKind::NonRetryable;
        ActivityError::from(
            ApplicationFailure::builder(anyhow::anyhow!("{err}"))
                .non_retryable(is_non_retryable)
                .build(),
        )
    }

    #[activities]
    impl UpdateServiceActivities {
        /// Lists an entity's linked pages and fetch state via `PostgresService.ListEntityPages`.
        ///
        /// # Errors
        ///
        /// Returns a non-retryable [`ActivityError`] on invalid arguments or missing entities,
        /// or a retryable [`ActivityError`] on transient transport/server failures.
        #[activity]
        pub async fn list_pages(
            self: Arc<Self>,
            _ctx: ActivityContext,
            anime_id: String,
        ) -> Result<ListEntityPagesOutput, ActivityError> {
            self.ports
                .catalogue
                .list_entity_pages(anime_id)
                .await
                .map_err(|err| to_activity_error(&err))
        }

        /// Loads, merges, and saves an anime, retrying `ABORTED` version conflicts up to 10 times.
        ///
        /// # Errors
        ///
        /// Returns a retryable [`ActivityError`] if more than 10 consecutive version conflicts
        /// occur or a transient gRPC failure happens, or non-retryable on permanent gRPC errors.
        #[activity]
        pub async fn merge_anime(
            self: Arc<Self>,
            _ctx: ActivityContext,
            input: MergeAnimeInput,
        ) -> Result<MergeAnimeOutput, ActivityError> {
            for _attempt in 0..MAX_ABORTED_RETRIES_PER_ATTEMPT {
                match self
                    .ports
                    .catalogue
                    .merge_anime_attempt(input.anime_id.clone(), input.merge_version)
                    .await
                    .map_err(|err| to_activity_error(&err))?
                {
                    MergeAttemptOutcome::Saved(output) => return Ok(output),
                    MergeAttemptOutcome::VersionConflict(_) => {}
                }
            }
            Err(to_activity_error(&PortError::retryable(format!(
                "merge_anime exceeded {MAX_ABORTED_RETRIES_PER_ATTEMPT} ABORTED retries for {}",
                input.anime_id
            ))))
        }

        /// Starts top-level downstream workflows (`RefreshEpisodeWorkflow` / `WriteSynopsisWorkflow`)
        /// with `USE_EXISTING` conflict policy and inherited `RefreshTrigger` / priority.
        ///
        /// # Errors
        ///
        /// Returns an [`ActivityError`] if the Temporal client fails to start any workflow.
        #[activity]
        pub async fn start_entity_workflows(
            self: Arc<Self>,
            _ctx: ActivityContext,
            input: StartEntityWorkflowsInput,
        ) -> Result<usize, ActivityError> {
            self.ports
                .starter
                .start_entity_workflows(input)
                .await
                .map_err(|err| to_activity_error(&err))
        }

        /// Records the final refresh outcome via `PostgresService.RecordRefresh`.
        ///
        /// # Errors
        ///
        /// Returns a non-retryable [`ActivityError`] when the entity is missing or arguments
        /// are invalid, or a retryable [`ActivityError`] on transient failures.
        #[activity]
        pub async fn record_refresh(
            self: Arc<Self>,
            _ctx: ActivityContext,
            input: RecordRefreshInput,
        ) -> Result<(), ActivityError> {
            self.ports
                .catalogue
                .record_refresh(input)
                .await
                .map_err(|err| to_activity_error(&err))
        }
    }

    #[activities]
    impl ProviderActivities {
        /// Fetches and stores one provider page while recording heartbeats every 20 seconds
        /// and watching for activity cancellation.
        ///
        /// # Errors
        ///
        /// Returns [`ActivityError::Cancelled`] when Temporal cancels the activity, or a
        /// retryable/non-retryable [`ActivityError`] on unrecoverable fetch/store errors.
        #[activity]
        pub async fn refresh_page(
            self: Arc<Self>,
            ctx: ActivityContext,
            input: RefreshPageInput,
        ) -> Result<RefreshPageOutput, ActivityError> {
            let mut heartbeat_ticker = tokio::time::interval(self.heartbeat_interval);
            heartbeat_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // Consume the immediate first tick so heartbeats fire at each interval.
            heartbeat_ticker.tick().await;

            let refresh_fut = self
                .ports
                .catalogue
                .refresh_page(input.page_key.clone(), input.max_age_seconds);
            tokio::pin!(refresh_fut);

            loop {
                tokio::select! {
                    res = &mut refresh_fut => {
                        return res.map_err(|err| to_activity_error(&err));
                    }
                    _ = heartbeat_ticker.tick() => {
                        let _ = ctx.record_heartbeat(input.page_key.address.clone()).await;
                    }
                    () = ctx.cancelled() => {
                        return Err(ActivityError::Cancelled { details: None });
                    }
                }
            }
        }
    }
}

#[doc(inline)]
pub use definitions::{
    MAX_ABORTED_RETRIES_PER_ATTEMPT, ProviderActivities, REFRESH_PAGE_HEARTBEAT_INTERVAL,
    UpdateServiceActivities,
};
