//! Temporal activity implementations for `update-service` and `provider-<slug>` queues.

#[allow(
    missing_docs,
    reason = "temporalio_macros generates undocumented associated constants"
)]
#[allow(
    clippy::unused_async_trait_impl,
    reason = "temporalio_macros copies each async #[activity] into a generated trait impl"
)]
mod definitions {
    use std::future::Future;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime};

    use temporalio_macros::activities;
    use temporalio_sdk::activities::{ActivityContext, ActivityError};
    use tracing::{Level, event};
    use uuid::Uuid;

    use crate::ports::{MergeAttemptOutcome, PortError, UpdateServicePorts};
    use crate::refresh_anime_types::{
        ListEntityPagesOutput, MergeAnimeOutput, RecordRefreshInput, RefreshPageInput,
        RefreshPageKey, RefreshPageOutput, RefreshSettings, StartEntityWorkflowsInput,
    };

    /// Maximum `LoadAnime` -> `MergeAnime` -> `SaveAnime` retries inside one `merge_anime` attempt.
    ///
    /// Matches `docs/architecture/update_service.md`: optimistic concurrency conflicts resolve
    /// quickly and should not consume the Temporal retry policy's 5-attempt budget.
    pub const MAX_ABORTED_RETRIES_PER_ATTEMPT: usize = 10;

    /// Interval between activity heartbeats while long-running RPCs execute.
    ///
    /// Set to one third of the 60-second heartbeat timeout so a healthy fetch always delivers
    /// multiple heartbeats while a crashed worker is detected within a minute.
    pub const REFRESH_PAGE_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(20);

    /// Shared dependencies wrapped by both core and provider activity bundles.
    #[derive(Debug, Clone)]
    struct ActivityDependencies {
        ports: UpdateServicePorts,
        settings: RefreshSettings,
        heartbeat_interval: Duration,
    }

    impl ActivityDependencies {
        fn new(ports: UpdateServicePorts, settings: RefreshSettings) -> Self {
            Self {
                ports,
                settings,
                heartbeat_interval: REFRESH_PAGE_HEARTBEAT_INTERVAL,
            }
        }
    }

    /// Core Update Service activities running on the `update-service` task queue.
    #[derive(Debug)]
    pub struct UpdateServiceActivities {
        deps: ActivityDependencies,
    }

    impl UpdateServiceActivities {
        /// Creates core Update Service activities backed by `ports` and default settings.
        #[must_use]
        pub fn new(ports: UpdateServicePorts) -> Self {
            Self::with_settings(ports, RefreshSettings::default())
        }

        /// Creates core Update Service activities with explicit worker `settings`.
        #[must_use]
        pub fn with_settings(ports: UpdateServicePorts, settings: RefreshSettings) -> Self {
            Self {
                deps: ActivityDependencies::new(ports, settings),
            }
        }

        /// Overrides the heartbeat interval for fast tests.
        #[cfg(feature = "test-util")]
        #[must_use]
        pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
            self.deps.heartbeat_interval = interval;
            self
        }
    }

    /// Provider page-fetch activities running on each `provider-<slug>` task queue.
    #[derive(Debug)]
    pub struct ProviderActivities {
        deps: ActivityDependencies,
    }

    impl ProviderActivities {
        /// Creates provider activities backed by `ports` with the 20-second heartbeat.
        #[must_use]
        pub fn new(ports: UpdateServicePorts) -> Self {
            Self {
                deps: ActivityDependencies::new(ports, RefreshSettings::default()),
            }
        }

        /// Overrides the heartbeat interval for fast tests.
        #[cfg(feature = "test-util")]
        #[must_use]
        pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
            self.deps.heartbeat_interval = interval;
            self
        }
    }

    /// Drives the future returned by `start_operation` while heartbeating and watching cancellation.
    ///
    /// Returns [`ActivityError::Cancelled`] before calling `start_operation` if the context is
    /// already cancelled, so cancelled retries never start a new RPC.
    async fn run_with_heartbeats<T, F>(
        ctx: &ActivityContext,
        heartbeat_interval: Duration,
        heartbeat_detail: &str,
        start_operation: impl FnOnce() -> F,
    ) -> Result<T, ActivityError>
    where
        F: Future<Output = Result<T, PortError>>,
    {
        if ctx.is_cancelled() {
            return Err(ActivityError::Cancelled { details: None });
        }

        let mut heartbeat_ticker = tokio::time::interval(heartbeat_interval);
        heartbeat_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Consume the immediate first tick so heartbeats fire at each interval.
        heartbeat_ticker.tick().await;

        let operation = start_operation();
        tokio::pin!(operation);

        loop {
            tokio::select! {
                biased;
                () = ctx.cancelled() => {
                    return Err(ActivityError::Cancelled { details: None });
                }
                outcome = &mut operation => {
                    return outcome.map_err(PortError::into_activity_error);
                }
                _ = heartbeat_ticker.tick() => {
                    if let Err(err) = ctx.record_heartbeat(heartbeat_detail.to_owned()).await {
                        event!(
                            name: "update_service.activity.heartbeat_failed",
                            Level::DEBUG,
                            heartbeat.detail = heartbeat_detail,
                            error.message = %err,
                            "failed to encode activity heartbeat {{heartbeat.detail}}: {{error.message}}",
                        );
                    }
                }
            }
        }
    }

    #[activities]
    impl UpdateServiceActivities {
        /// Lists an entity's linked pages, stored `merge_version`, and worker refresh settings.
        ///
        /// # Errors
        ///
        /// Returns a non-retryable [`ActivityError`] on invalid arguments or missing entities,
        /// or a retryable [`ActivityError`] on transient transport/server failures.
        #[activity]
        pub async fn list_pages(
            self: Arc<Self>,
            _ctx: ActivityContext,
            anime_id: Uuid,
        ) -> Result<ListEntityPagesOutput, ActivityError> {
            let entity_pages = self
                .deps
                .ports
                .catalogue()
                .list_entity_pages(anime_id)
                .await
                .map_err(PortError::into_activity_error)?;
            Ok(ListEntityPagesOutput {
                pages: entity_pages.pages,
                entity_merge_version: entity_pages.entity_merge_version,
                refresh_settings: self.deps.settings,
            })
        }

        /// Records a failed page fetch via `PostgresService.StorePage` with empty responses.
        ///
        /// Called by `RefreshAnimeWorkflow` after `refresh_page` fails for any non-cancellation
        /// reason (provider block, permanent error, exhausted retries, or activity timeout).
        ///
        /// # Errors
        ///
        /// Returns a non-retryable [`ActivityError`] on invalid arguments, or a retryable
        /// [`ActivityError`] on transient `PostgresService` failures.
        #[activity]
        pub async fn record_page_failure(
            self: Arc<Self>,
            _ctx: ActivityContext,
            page_key: RefreshPageKey,
        ) -> Result<(), ActivityError> {
            self.deps
                .ports
                .catalogue()
                .record_page_failure(page_key, SystemTime::now())
                .await
                .map_err(PortError::into_activity_error)
        }

        /// Loads, merges, and saves an anime, retrying `ABORTED` conflicts up to 10 times.
        ///
        /// Emits heartbeats every 20 seconds and watches `ctx.cancelled()` across all retry
        /// attempts so workflow cancellations and graceful worker shutdowns stop promptly.
        ///
        /// # Errors
        ///
        /// Returns [`ActivityError::Cancelled`] when Temporal cancels the activity, a retryable
        /// [`ActivityError`] after 10 consecutive version conflicts or on transient gRPC failures,
        /// or a non-retryable error on permanent gRPC statuses.
        #[activity]
        pub async fn merge_anime(
            self: Arc<Self>,
            ctx: ActivityContext,
            anime_id: Uuid,
        ) -> Result<MergeAnimeOutput, ActivityError> {
            let merge_version = self.deps.settings.merge_version;
            for attempt in 1..=MAX_ABORTED_RETRIES_PER_ATTEMPT {
                let heartbeat_detail = format!("merge:{anime_id}:{attempt}");
                match run_with_heartbeats(
                    &ctx,
                    self.deps.heartbeat_interval,
                    &heartbeat_detail,
                    || {
                        self.deps
                            .ports
                            .catalogue()
                            .merge_anime_attempt(anime_id, merge_version)
                    },
                )
                .await?
                {
                    MergeAttemptOutcome::Saved(output) => return Ok(output),
                    MergeAttemptOutcome::VersionConflict(reason) => {
                        event!(
                            name: "update_service.merge_anime.version_conflict",
                            Level::DEBUG,
                            anime.id = %anime_id,
                            merge.attempt = attempt,
                            conflict.reason = %reason,
                            "retrying merge_anime for {{anime.id}} after ABORTED attempt {{merge.attempt}}: {{conflict.reason}}",
                        );
                    }
                }
            }
            Err(PortError::retryable(format!(
                "merge_anime exceeded {MAX_ABORTED_RETRIES_PER_ATTEMPT} ABORTED retries for {anime_id}"
            ))
            .into_activity_error())
        }

        /// Starts top-level downstream workflows (`RefreshEpisodeWorkflow` / `WriteSynopsisWorkflow`).
        ///
        /// Emits heartbeats every 20 seconds and watches `ctx.cancelled()` while starting large
        /// episode batches.
        ///
        /// # Errors
        ///
        /// Returns [`ActivityError::Cancelled`] on cancellation, or an [`ActivityError`] if the
        /// Temporal client fails to start any workflow.
        #[activity]
        pub async fn start_entity_workflows(
            self: Arc<Self>,
            ctx: ActivityContext,
            input: StartEntityWorkflowsInput,
        ) -> Result<usize, ActivityError> {
            let heartbeat_detail = format!("start:{}:{}", input.anime_id, input.episode_ids.len());
            run_with_heartbeats(
                &ctx,
                self.deps.heartbeat_interval,
                &heartbeat_detail,
                || self.deps.ports.starter().start_entity_workflows(input),
            )
            .await
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
            self.deps
                .ports
                .catalogue()
                .record_refresh(
                    input.anime_id,
                    input.outcome,
                    self.deps.settings.refresh_after_days,
                )
                .await
                .map_err(PortError::into_activity_error)
        }
    }

    #[activities]
    impl ProviderActivities {
        /// Fetches and stores one provider page while heartbeating every 20 seconds.
        ///
        /// Transient fetch failures are returned to Temporal for exponential retry up to the
        /// policy's maximum attempts. Provider blocks and permanent gRPC errors return
        /// non-retryable failures immediately; `RefreshAnimeWorkflow` records every terminal
        /// page failure uniformly through `UpdateServiceActivities::record_page_failure`.
        ///
        /// # Errors
        ///
        /// Returns [`ActivityError::Cancelled`] when Temporal cancels the activity, a
        /// non-retryable [`ActivityError`] on provider blocks or permanent gRPC statuses, or a
        /// retryable [`ActivityError`] on transient upstream failures.
        #[activity]
        pub async fn refresh_page(
            self: Arc<Self>,
            ctx: ActivityContext,
            input: RefreshPageInput,
        ) -> Result<RefreshPageOutput, ActivityError> {
            let page_key = input.page_key;
            let heartbeat_detail = format!("{}:{}", page_key.provider, page_key.address);
            run_with_heartbeats(
                &ctx,
                self.deps.heartbeat_interval,
                &heartbeat_detail,
                || {
                    self.deps
                        .ports
                        .catalogue()
                        .fetch_and_store_page(page_key, input.max_age)
                },
            )
            .await
        }
    }
}

#[doc(inline)]
pub use definitions::{
    MAX_ABORTED_RETRIES_PER_ATTEMPT, ProviderActivities, REFRESH_PAGE_HEARTBEAT_INTERVAL,
    UpdateServiceActivities,
};
